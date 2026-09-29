"""The distributed write lifecycle: idempotent commits, aborts, merged fragments.

What a Ray datasink leans on: a commit that can be repeated after a lost
response, files deleted when a job will never commit them, a driver that does
not hold one schema per task, and batches that leave out a defaulted column.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

from deltalake import write_deltalake  # noqa: E402
from deltaswamp.distributed import merge_fragments  # noqa: E402
from deltaswamp.errors import (  # noqa: E402
    CommitConflictError,
    InvalidArgumentError,
    MetadataChangedError,
    TransientCommitError,
    UnreachableTableError,
)

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")


def _rows(n: int = 1, start: int = 0) -> Any:
    return pa.table({"id": pa.array(range(start, start + n), pa.int64())})


def _table(conn: Any, tmp_path: Any, name: str = "t") -> str:
    path = str(tmp_path / name)
    conn.create_table(path, pa.schema([("id", pa.int64())]))
    return path


def _parquet(path: str) -> set[str]:
    return {
        os.path.relpath(p, path)
        for p in glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        if "_delta_log" not in p
    }


def _actions(path: str, version: int) -> list[dict[str, Any]]:
    with open(os.path.join(path, "_delta_log", f"{version:020}.json")) as f:
        return [json.loads(line) for line in f if line.strip()]


def _ids(conn: Any, path: str) -> list[int]:
    return sorted(conn.open_table(path).to_arrow().column("id").to_pylist())


# ------------------------------------------------------------------ idempotency


class TestLandedCommits:
    def test_an_overwrite_that_landed_is_not_committed_again(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """Re-committed after a lost response, an overwrite added and removed its own files.

        The kernel then read the rows, delta-rs listed the file once, and
        Spark's replay (add, then remove) dropped it.
        """
        path = _table(conn, tmp_path)
        conn.open_table(path).append(_rows(1, 100))
        plan = conn.open_table(path).plan_write(mode="overwrite")
        fragment = plan.write(_rows(3))
        real = plan.engine.commit_files

        def lands_then_fails(*args: Any, **kwargs: Any) -> int:
            real(*args, **kwargs)
            raise TransientCommitError("timed out after the put")

        monkeypatch.setattr(plan.engine, "commit_files", lands_then_fails)
        with pytest.raises(TransientCommitError, match="commit the same fragments again"):
            plan.commit([fragment], allow_concurrent_overwrite=True)
        monkeypatch.undo()
        # The same fragments, committed again: found, not re-committed.
        assert plan.commit([fragment], allow_concurrent_overwrite=True) == 2
        assert conn.open_table(path).version == 2
        adds = {a["add"]["path"] for a in _actions(path, 2) if "add" in a}
        removes = {a["remove"]["path"] for a in _actions(path, 2) if "remove" in a}
        assert adds and not adds & removes
        assert _ids(conn, path) == [0, 1, 2]

    def test_the_commit_refuses_to_add_and_remove_one_file(self, conn: Any, tmp_path: Any) -> None:
        """Below the landed check: the native commit refuses the overlap itself."""
        path = _table(conn, tmp_path)
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_rows(2))
        plan.commit([fragment])
        table = conn.open_table(path)
        with pytest.raises(ds.DeltaSwampError, match="add and remove it in one commit"):
            plan.engine.commit_files(table._resolved, [fragment], overwrite=True)
        assert _ids(conn, path) == [0, 1]

    def test_the_check_reads_only_the_commits_since_planning(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """It listed every live file (with stats) on every attempt: O(table), not O(job)."""
        path = _table(conn, tmp_path)
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_rows(2))

        def no_listing(*_: Any, **__: Any) -> Any:
            raise AssertionError("the commit listed the table's files")

        monkeypatch.setattr(plan.engine, "files", no_listing)
        assert plan.commit([fragment]) == 1

    def test_a_check_that_cannot_run_does_not_commit(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """ "Cannot tell" was taken as "not landed", and the files could be committed twice."""
        path = _table(conn, tmp_path)
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_rows(2))

        def unreadable(*_: Any, **__: Any) -> Any:
            raise OSError("commit 1 is gone")

        monkeypatch.setattr(plan.engine, "commits_adding", unreadable)
        with pytest.raises(UnreachableTableError, match="cannot be told"):
            plan.commit([fragment])
        monkeypatch.undo()
        assert conn.open_table(path).version == 0
        assert plan.commit([fragment]) == 1

    def test_fragments_committed_in_part_are_refused(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path)
        plan = conn.open_table(path).plan_write()
        first, second = plan.write(_rows(1)), plan.write(_rows(1, 1))
        plan.commit([first])
        with pytest.raises(InvalidArgumentError, match="only some"):
            plan.commit([first, second])
        assert _ids(conn, path) == [0]


# ------------------------------------------------------------------ abort


class TestAbort:
    def test_abort_deletes_the_files(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path)
        before = _parquet(path)
        plan = conn.open_table(path).plan_write()
        fragments = [plan.write(_rows(2)), plan.write(_rows(2, 2))]
        assert len(_parquet(path) - before) == 2
        assert plan.abort(fragments) == 2
        assert _parquet(path) == before
        assert plan.abort(fragments) == 2  # already gone counts as deleted

    def test_abort_refuses_committed_files(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path)
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_rows(2))
        plan.commit([fragment])
        with pytest.raises(UnreachableTableError, match="committed"):
            plan.abort([fragment])
        assert _ids(conn, path) == [0, 1]

    def test_a_schema_change_aborts_the_write(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        write_deltalake(path, _rows(2))
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_rows(1, 5))
        written = _parquet(path)
        write_deltalake(path, pa.table({"id": ["x"]}), mode="overwrite", schema_mode="overwrite")
        with pytest.raises(MetadataChangedError):
            plan.commit([fragment])
        left = _parquet(path)
        ours = {p for p in written if p not in left}
        assert len(ours) == 1, "the fragment's file is deleted, the table's are not"

    def test_a_moved_table_aborts_a_guarded_overwrite(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path)
        plan = conn.open_table(path).plan_write(mode="overwrite")
        fragment = plan.write(_rows(2))
        mine = _parquet(path)
        conn.open_table(path).append(_rows(1, 9))
        with pytest.raises(UnreachableTableError, match="version"):
            plan.commit([fragment])
        assert not mine & _parquet(path)
        assert _ids(conn, path) == [9]

    def test_a_lost_race_aborts_the_write(self, conn: Any, tmp_path: Any, monkeypatch: Any) -> None:
        path = _table(conn, tmp_path)
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_rows(2))
        mine = _parquet(path)

        def lost(*_: Any, **__: Any) -> int:
            raise CommitConflictError(1, "another writer committed version 1 first")

        monkeypatch.setattr(plan.engine, "commit_files", lost)
        with pytest.raises(CommitConflictError):
            plan.commit([fragment], retries=1)
        assert not mine & _parquet(path)

    def test_abort_on_failure_false_keeps_the_files(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        path = _table(conn, tmp_path)
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_rows(2))

        def lost(*_: Any, **__: Any) -> int:
            raise CommitConflictError(1, "another writer committed version 1 first")

        monkeypatch.setattr(plan.engine, "commit_files", lost)
        with pytest.raises(CommitConflictError):
            plan.commit([fragment], retries=0, abort_on_failure=False)
        monkeypatch.undo()
        assert plan.commit([fragment]) == 1
        assert _ids(conn, path) == [0, 1]

    def test_an_unknown_outcome_keeps_the_files(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        path = _table(conn, tmp_path)
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_rows(2))
        mine = _parquet(path)

        def timed_out(*_: Any, **__: Any) -> int:
            raise TransientCommitError("timed out")

        monkeypatch.setattr(plan.engine, "commit_files", timed_out)
        with pytest.raises(TransientCommitError, match="abort them only if"):
            plan.commit([fragment], retries=1)
        assert mine <= _parquet(path)

    def test_paths_outside_the_table_are_never_deleted(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path)
        victim = tmp_path / "victim.parquet"
        victim.write_bytes(b"x")
        engine = conn.open_table(path).plan_write().engine
        resolved = conn.open_table(path)._resolved
        for bad in (
            "../victim.parquet",
            "/etc/passwd",
            "_delta_log/00000000000000000000.json",
            f"file://{victim}",
        ):
            with pytest.raises(ds.DeltaSwampError, match="not a data file path"):
                engine.delete_uncommitted(resolved, [bad])
        assert victim.exists()
        assert conn.open_table(path).version == 0


# ------------------------------------------------------------------ fragments


class TestMergedFragments:
    def test_merged_fragments_commit_as_the_parts_would(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path)
        plan = conn.open_table(path).plan_write()
        parts = [plan.write(_rows(1, i)) for i in range(40)]
        merged = merge_fragments(parts)
        assert len(merged) < sum(len(p) for p in parts) / 10
        assert plan.commit([merged]) == 1
        assert _ids(conn, path) == list(range(40))
        assert sum(1 for a in _actions(path, 1) if "add" in a) == 40

    def test_commit_takes_a_generator(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path)
        plan = conn.open_table(path).plan_write()
        parts = [plan.write(_rows(1, i)) for i in range(5)]
        assert plan.commit(p for p in parts) == 1
        assert _ids(conn, path) == list(range(5))

    def test_fragments_of_two_plans_are_not_merged(self, conn: Any, tmp_path: Any) -> None:
        a, b = _table(conn, tmp_path, "a"), _table(conn, tmp_path, "b")
        first = conn.open_table(a).plan_write().write(_rows(1))
        second = conn.open_table(b).plan_write().write(_rows(1))
        with pytest.raises(InvalidArgumentError, match="different table"):
            merge_fragments([first, second])
        with pytest.raises(ds.DeltaSwampError, match="different table"):
            conn.open_table(a).plan_write().commit([first, second])

    def test_a_duplicate_is_still_refused_after_merging(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path)
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_rows(1))
        with pytest.raises(ds.DeltaSwampError, match="more than one fragment"):
            plan.commit([fragment, fragment])

    def test_an_overwrite_writes_its_removes_without_stats(self, conn: Any, tmp_path: Any) -> None:
        """The removes are cut to what a remove needs as the scan streams in."""
        path = _table(conn, tmp_path)
        for i in range(3):
            conn.open_table(path).append(_rows(1, i))
        plan = conn.open_table(path).plan_write(mode="overwrite")
        version = plan.commit([plan.write(_rows(1, 7))])
        removes = [a["remove"] for a in _actions(path, version) if "remove" in a]
        assert len(removes) == 3
        assert all("stats" not in r or r["stats"] is None for r in removes)
        assert all(r.get("size") for r in removes)
        assert _ids(conn, path) == [7]


# ------------------------------------------------------------------ column defaults


def _defaults_table(tmp_path: Any, default: str) -> str:
    from tests.integration.test_audit_dialect import _write_log

    path = str(tmp_path / "defaults")
    schema = {
        "type": "struct",
        "fields": [
            {"name": "id", "type": "long", "nullable": True, "metadata": {}},
            {
                "name": "city",
                "type": "string",
                "nullable": True,
                "metadata": {"CURRENT_DEFAULT": default},
            },
            {
                "name": "n",
                "type": "integer",
                "nullable": True,
                "metadata": {"CURRENT_DEFAULT": "42"},
            },
        ],
    }
    protocol = {
        "minReaderVersion": 1,
        "minWriterVersion": 7,
        "writerFeatures": ["allowColumnDefaults"],
    }
    data = pa.table({"id": [1], "city": ["x"], "n": pa.array([1], pa.int32())})
    _write_log(path, schema, protocol, data)
    return path


class TestColumnDefaults:
    def test_a_worker_fills_a_literal_default(self, conn: Any, tmp_path: Any) -> None:
        """Every worker failed on a batch that left out a defaulted column."""
        path = _defaults_table(tmp_path, "'new'")
        plan = conn.open_table(path).plan_write()
        plan.commit(
            [
                plan.write(pa.table({"id": [2]})),
                plan.write(pa.RecordBatch.from_pydict({"id": [3], "city": ["given"]})),
            ]
        )
        rows = sorted(conn.open_table(path).to_arrow().to_pylist(), key=lambda r: r["id"])
        assert rows[1:] == [{"id": 2, "city": "new", "n": 42}, {"id": 3, "city": "given", "n": 42}]

    def test_an_expression_default_is_refused_at_planning(self, conn: Any, tmp_path: Any) -> None:
        path = _defaults_table(tmp_path, "concat('a', 'b')")
        with pytest.raises(UnreachableTableError, match="supplies_defaults"):
            conn.open_table(path).plan_write()
        plan = conn.open_table(path).plan_write(supplies_defaults=True)
        plan.commit([plan.write(pa.table({"id": [2], "city": ["given"]}))])
        assert sorted(conn.open_table(path).to_arrow().column("id").to_pylist()) == [1, 2]
