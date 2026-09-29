"""VACUUM and RESTORE on the tables delta-rs cannot write, served by the kernel.

Round-7 gap report, #1 and #2: VACUUM was refused on every table carrying
clustering, row tracking, in-commit timestamps, type widening or
`vacuumProtocolCheck` (most tables Databricks creates), and RESTORE on every
deletion-vector table (delta-rs#4613) and every kernel-only one. Each test
here failed before: the call raised UnreachableTableError, or (for the DV
vacuum) kept vector files nothing referenced.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
deltalake = pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

from deltaswamp.capability import Engine  # noqa: E402
from deltaswamp.errors import (  # noqa: E402
    InvalidArgumentError,
    MissingDataFileError,
    UnreachableTableError,
)

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

SCHEMA = pa.schema([("id", pa.int64()), ("v", pa.string()), ("g", pa.int32())])


def _data(n: int = 10, off: int = 0) -> Any:
    return pa.table(
        {
            "id": pa.array(range(off, off + n), pa.int64()),
            "v": [f"s{i}" for i in range(off, off + n)],
            "g": pa.array([i % 3 for i in range(off, off + n)], pa.int32()),
        }
    )


def _age(path: str, hours: float) -> None:
    """Make every data file look `hours` old, as a VACUUM's modification-time rule reads it."""
    old = time.time() - hours * 3600
    for f in pathlib.Path(path).rglob("*"):
        if "_delta_log" not in f.parts and f.is_file():
            os.utime(f, (old, old))


def _files(path: str) -> set[str]:
    root = pathlib.Path(path)
    return {
        str(f.relative_to(root))
        for f in root.rglob("*")
        if f.is_file() and "_delta_log" not in f.parts
    }


def _ids(conn: Any, path: str, **kw: Any) -> list[int]:
    return sorted(conn.open_table(path).to_arrow(**kw).column("id").to_pylist())


def _deltars_version(path: str, version: int) -> list[int]:
    dt = deltalake.DeltaTable(path, version=version)
    return sorted(pa.table(dt.to_pyarrow_table()).column("id").to_pylist())


#: Tables only the kernel writes, as the gap report's demos built them.
KERNEL_ONLY = [
    pytest.param({"properties": {"delta.enableInCommitTimestamps": "true"}}, id="ict"),
    pytest.param(
        {
            "properties": {
                "delta.enableRowTracking": "true",
                "delta.enableDeletionVectors": "true",
            }
        },
        id="rowtracking",
    ),
    pytest.param({"properties": {"delta.enableTypeWidening": "true"}}, id="typewidening"),
    pytest.param({"properties": {"delta.feature.vacuumProtocolCheck": "supported"}}, id="vpc"),
    pytest.param({"cluster_by": ["g"]}, id="clustered"),
    pytest.param(
        {
            "properties": {
                "delta.enableDeletionVectors": "true",
                "delta.enableRowTracking": "true",
                "delta.enableInCommitTimestamps": "true",
            }
        },
        id="dv-rt-ict",
    ),
]


#: What `_history_table` holds at its latest version.
LATEST = [*range(100, 105), *range(200, 203)]


def _history_table(conn: Any, path: str, kw: dict[str, Any]) -> Any:
    """v1 append, v2 append, v3 delete every v1 row (its file removed), v4 append.

    A DELETE rather than an overwrite: the kernel overwrites no row-tracked
    table.
    """
    conn.create_table(path, SCHEMA, **kw)
    conn.open_table(path).append(_data(10))
    conn.open_table(path).append(_data(5, 100))
    conn.open_table(path).delete("id < 100")
    conn.open_table(path).append(_data(3, 200))
    return conn.open_table(path)


class TestVacuumOnKernelOnlyTables:
    @pytest.mark.parametrize("kw", KERNEL_ONLY)
    def test_vacuum_deletes_only_what_nothing_references(
        self, conn: Any, tmp_path: Any, kw: dict[str, Any]
    ) -> None:
        path = str(tmp_path / "t")
        t = _history_table(conn, path, kw)
        assert t.can("vacuum", dry_run=False).engine is Engine.KERNEL
        live = {a["path"] for a in _add_actions(conn, path)}
        orphan = pathlib.Path(path) / "part-00000-orphan-c000.snappy.parquet"
        orphan.write_bytes(b"left by a crashed write")
        _age(path, 2)
        planned = t.vacuum(retention_hours=0, enforce_retention_duration=False)
        # The orphan and the files the delete removed (none on a row-tracked
        # table, where it marks every row deleted instead); no live file.
        assert orphan.name in planned
        assert not set(planned) & live
        assert set(planned) - {orphan.name} == _removed_at(path, 3) - live
        before = conn.open_table(path).version
        deleted = t.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False)
        assert sorted(deleted) == sorted(planned)
        assert not orphan.exists()
        assert _ids(conn, path) == LATEST
        history = conn.open_table(path).history(limit=2)
        assert [h["operation"] for h in history] == ["VACUUM END", "VACUUM START"]
        assert conn.open_table(path).version == before + 2
        # Nothing is left to delete, and the second run commits nothing.
        assert t.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False) == []
        assert conn.open_table(path).version == before + 2

    @pytest.mark.parametrize("kw", KERNEL_ONLY)
    def test_lite_vacuum(self, conn: Any, tmp_path: Any, kw: dict[str, Any]) -> None:
        path = str(tmp_path / "t")
        t = _history_table(conn, path, kw)
        orphan = pathlib.Path(path) / "part-00000-orphan-c000.snappy.parquet"
        orphan.write_bytes(b"x")
        assert t.can("vacuum", lite=True, dry_run=False).engine is Engine.KERNEL
        planned = t.vacuum(retention_hours=0, enforce_retention_duration=False, lite=True)
        assert set(planned) == _removed_at(path, 3) - {a["path"] for a in _add_actions(conn, path)}
        t.vacuum(retention_hours=0, enforce_retention_duration=False, lite=True, dry_run=False)
        assert orphan.exists()
        assert _ids(conn, path) == LATEST

    def test_the_retention_check_is_enforced(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = _history_table(conn, path, {"properties": {"delta.enableInCommitTimestamps": "true"}})
        with pytest.raises(InvalidArgumentError, match="deletedFileRetentionDuration"):
            t.vacuum(retention_hours=1, dry_run=False)
        assert t.vacuum(retention_hours=168) == []

    def test_a_table_retention_below_a_week_is_honoured(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        kw = {
            "properties": {
                "delta.enableInCommitTimestamps": "true",
                "delta.deletedFileRetentionDuration": "interval 1 hours",
            }
        }
        t = _history_table(conn, path, kw)
        _age(path, 3)
        # The removes are younger than an hour, so their files stay.
        assert t.vacuum() == []
        assert t.vacuum(retention_hours=1) == []


def _add_actions(conn: Any, path: str, version: int | None = None) -> list[dict[str, Any]]:
    from deltaswamp import _native

    snapshot = _native.Snapshot.resolve(path, version=version)
    return [json.loads(a) for a in snapshot.add_actions()]


def _removed_at(path: str, version: int) -> set[str]:
    commit = pathlib.Path(path) / "_delta_log" / f"{version:020}.json"
    actions = [json.loads(line) for line in commit.read_text().splitlines() if line]
    return {a["remove"]["path"] for a in actions if "remove" in a}


def _rewrite_log(path: str, version: int, edit: Any) -> None:
    commit = pathlib.Path(path) / "_delta_log" / f"{version:020}.json"
    lines = [json.loads(line) for line in commit.read_text().splitlines() if line]
    commit.write_text("".join(json.dumps(edit(a)) + "\n" for a in lines))


class TestVacuumMatchesSpark:
    """Compared with delta-rs's full VACUUM on tables both engines serve."""

    @pytest.mark.parametrize("partitioned", [False, True])
    def test_dry_run_lists_what_delta_rs_lists(
        self, conn: Any, tmp_path: Any, partitioned: bool
    ) -> None:
        path = str(tmp_path / "t")
        kw = {"partition_by": ["g"]} if partitioned else {}
        deltalake.write_deltalake(path, _data(9), **kw)
        deltalake.write_deltalake(path, _data(6, 100), mode="overwrite", **kw)
        deltalake.write_deltalake(path, _data(3, 200), mode="append", **kw)
        dt = deltalake.DeltaTable(path)
        dt.delete("id = 201")
        (pathlib.Path(path) / "part-00000-orphan-c000.snappy.parquet").write_bytes(b"x")
        (pathlib.Path(path) / "_tmp").mkdir()
        (pathlib.Path(path) / "_tmp" / "part-hidden.parquet").write_bytes(b"x")
        (pathlib.Path(path) / ".part-00000-x.parquet.crc").write_bytes(b"x")
        _age(path, 2)
        kernel = conn.router.engines[Engine.KERNEL]
        t = conn.open_table(path)
        ours = kernel.vacuum(t._enrich(), retention_hours=0, enforce_retention_duration=False)
        theirs = deltalake.DeltaTable(path).vacuum(
            retention_hours=0, dry_run=True, enforce_retention_duration=False, full=True
        )
        assert sorted(ours) == sorted(theirs)
        assert "_tmp/part-hidden.parquet" not in ours
        ours_lite = kernel.vacuum(
            t._enrich(), retention_hours=0, enforce_retention_duration=False, lite=True
        )
        theirs_lite = deltalake.DeltaTable(path).vacuum(
            retention_hours=0, dry_run=True, enforce_retention_duration=False
        )
        assert sorted(ours_lite) == sorted(theirs_lite)

    def test_removes_within_the_retention_protect_their_files(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "t")
        t = _history_table(conn, path, {"properties": {"delta.enableInCommitTimestamps": "true"}})
        conn.open_table(path).delete("id >= 100")  # v5 removes the v2 and v4 files
        conn.open_table(path).append(_data(2, 300))
        old_ms = int((time.time() - 5 * 3600) * 1000)

        def age_removes(action: dict[str, Any]) -> dict[str, Any]:
            if "remove" in action:
                action["remove"]["deletionTimestamp"] = old_ms
            return action

        _rewrite_log(path, 3, age_removes)  # the v1 file was removed five hours ago
        _age(path, 6)
        t = conn.open_table(path)
        planned = t.vacuum(retention_hours=2, enforce_retention_duration=False)
        assert set(planned) == _removed_at(path, 3)
        assert {a["path"] for a in _add_actions(conn, path, 1)} <= set(planned)
        t.vacuum(retention_hours=2, enforce_retention_duration=False, dry_run=False)
        # Every version the retention keeps still reads.
        assert _ids(conn, path, version=4) == LATEST
        assert _ids(conn, path) == [300, 301]
        with pytest.raises(MissingDataFileError):
            conn.open_table(path).to_arrow(version=1)

    def test_tombstones_in_a_checkpoint_still_protect(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = _history_table(conn, path, {"properties": {"delta.enableInCommitTimestamps": "true"}})
        t.checkpoint()
        conn.open_table(path).append(_data(1, 400))
        _age(path, 6)
        t = conn.open_table(path)
        assert t.vacuum(retention_hours=2, enforce_retention_duration=False) == []
        t.vacuum(retention_hours=2, enforce_retention_duration=False, dry_run=False)
        assert _ids(conn, path, version=1) == list(range(10))

    def test_deletion_vector_files(self, conn: Any, tmp_path: Any) -> None:
        """A live vector stays; one only an expired remove names goes (delta-rs kept all)."""
        path = str(tmp_path / "t")
        props = {"delta.enableDeletionVectors": "true", "delta.enableRowTracking": "true"}
        conn.create_table(path, SCHEMA, properties=props)
        conn.open_table(path).append(_data(10))
        conn.open_table(path).delete("id < 2")  # vector 1
        conn.open_table(path).delete("id = 5")  # vector 2 replaces it
        vectors = sorted(pathlib.Path(path).rglob("deletion_vector_*.bin"))
        assert len(vectors) == 2
        live = {
            a["deletionVector"]["pathOrInlineDv"]
            for a in _add_actions(conn, path)
            if a.get("deletionVector")
        }
        assert len(live) == 1
        _age(path, 2)
        t = conn.open_table(path)
        assert t.can("vacuum").engine is Engine.KERNEL
        planned = t.vacuum(retention_hours=0, enforce_retention_duration=False)
        assert len(planned) == 1 and planned[0].rsplit("/", 1)[-1].startswith("deletion_vector_")
        t.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False)
        assert len(list(pathlib.Path(path).rglob("deletion_vector_*.bin"))) == 1
        assert _ids(conn, path) == [2, 3, 4, 6, 7, 8, 9]

    def test_change_data_files(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        deltalake.write_deltalake(
            path, _data(6), configuration={"delta.enableChangeDataFeed": "true"}
        )
        deltalake.DeltaTable(path).delete("id = 1")
        cdc = [f for f in _files(path) if f.startswith("_change_data/")]
        assert cdc
        t = conn.open_table(path)
        kernel = conn.router.engines[Engine.KERNEL]
        # Written since the cutoff: referenced by their commit.
        recent = kernel.vacuum(t._enrich(), retention_hours=1, enforce_retention_duration=False)
        assert not set(cdc) & set(recent)
        _age(path, 2)
        planned = kernel.vacuum(t._enrich(), retention_hours=0, enforce_retention_duration=False)
        assert set(cdc) <= set(planned)

    def test_files_not_named_as_delta_names_them_are_kept(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = _history_table(conn, path, {"properties": {"delta.enableInCommitTimestamps": "true"}})
        notes = pathlib.Path(path) / "notes.txt"
        notes.write_text("mine")
        _age(path, 2)
        with pytest.warns(ds.DeltaSwampWarning, match="not deleted"):
            t.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False)
        assert notes.exists()


def _row_ids(path: str, version: int | None = None) -> list[tuple[int, str, int]]:
    from deltaswamp import _native

    snapshot = _native.Snapshot.resolve(path, version=version)
    read = pa.RecordBatchReader.from_stream(snapshot.scan(row_positions=True, row_ids=True))
    rows = read.read_all()
    return sorted(
        zip(
            rows.column("id").to_pylist(),
            rows.column("v").to_pylist(),
            rows.column("__deltaswamp_row_id").to_pylist(),
            strict=True,
        )
    )


class TestRestore:
    def _dv_table(self, conn: Any, path: str, **props: str) -> Any:
        conn.create_table(path, SCHEMA, properties={"delta.enableDeletionVectors": "true", **props})
        conn.open_table(path).append(_data(10))  # v1
        conn.open_table(path).delete("id < 3")  # v2: a deletion vector
        conn.open_table(path).update(predicate="id = 5", updates={"v": "'X'"})  # v3
        conn.open_table(path).append(_data(3, 100))  # v4
        return conn.open_table(path)

    @pytest.mark.parametrize("target", [1, 2, 3])
    def test_restore_undoes_deletion_vectors(self, conn: Any, tmp_path: Any, target: int) -> None:
        path = str(tmp_path / "t")
        t = self._dv_table(conn, path)
        assert t.can("restore").engine is Engine.KERNEL
        expected = conn.open_table(path).to_arrow(version=target).sort_by("id")
        result = t.restore(target)
        assert result["num_restored_files"] + result["num_removed_files"] >= 1
        restored = conn.open_table(path)
        assert restored.to_arrow().sort_by("id").equals(expected)
        assert _ids(conn, path) == _ids(conn, path, version=target)
        assert restored.history(limit=1)[0]["operation"] == "RESTORE"
        assert restored.history(limit=1)[0]["operationParameters"] == {"version": str(target)}

    def test_restored_rows_keep_their_row_ids(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = self._dv_table(conn, path, **{"delta.enableRowTracking": "true"})
        at_one = _row_ids(path, 1)
        t.restore(1)
        assert _row_ids(path) == at_one
        # New rows still get fresh ids above every restored one.
        conn.open_table(path).append(_data(1, 500))
        ids = [r[2] for r in _row_ids(path)]
        assert len(set(ids)) == len(ids)

    @pytest.mark.parametrize("kw", KERNEL_ONLY)
    def test_restore_on_kernel_only_tables(
        self, conn: Any, tmp_path: Any, kw: dict[str, Any]
    ) -> None:
        path = str(tmp_path / "t")
        t = _history_table(conn, path, kw)
        assert t.can("restore").engine is Engine.KERNEL
        t.restore(1)
        assert _ids(conn, path) == list(range(10))
        if not t.resolved.effective_reader_features:
            # delta-rs reads the restored table as it read version 1.
            latest = conn.open_table(path).version
            assert _deltars_version(path, latest) == _deltars_version(path, 1)
        t = conn.open_table(path)
        t.append(_data(1, 900))
        assert _ids(conn, path) == [*range(10), 900]

    def test_restore_by_timestamp(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = _history_table(conn, path, {"properties": {"delta.enableInCommitTimestamps": "true"}})
        import datetime

        at = conn.open_table(path).history()[-2]["timestamp"]  # version 1
        t.restore(datetime.datetime.fromtimestamp(at / 1000, datetime.UTC))
        assert _ids(conn, path) == list(range(10))

    def test_restore_brings_back_the_schema(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = _history_table(conn, path, {"properties": {"delta.enableInCommitTimestamps": "true"}})
        t.add_column(pa.field("extra", pa.int64()))
        conn.open_table(path).append(
            pa.table(
                {
                    "id": pa.array([7], pa.int64()),
                    "v": ["e"],
                    "g": pa.array([1], pa.int32()),
                    "extra": pa.array([1], pa.int64()),
                }
            )
        )
        conn.open_table(path).restore(4)
        restored = conn.open_table(path)
        assert restored.schema().names == ["id", "v", "g"]
        assert restored.properties()["delta.enableInCommitTimestamps"] == "true"
        assert _ids(conn, path) == LATEST

    def test_restore_brings_back_the_clustering_columns(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        _history_table(conn, path, {"cluster_by": ["g"]})
        conn.open_table(path).cluster_by(["id"])
        conn.open_table(path).restore(2)
        kernel = conn.router.engines[Engine.KERNEL]
        snapshot = kernel.snapshot(conn.open_table(path)._enrich())
        assert json.loads(snapshot.domain_metadata("delta.clustering")) == {
            "clusteringColumns": [["g"]]
        }

    def test_a_vacuumed_target_is_refused(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = self._dv_table(conn, path)
        conn.open_table(path).delete("id >= 0")
        _age(path, 2)
        conn.open_table(path).vacuum(
            retention_hours=0, enforce_retention_duration=False, dry_run=False
        )
        before = conn.open_table(path).version
        with pytest.raises(MissingDataFileError, match="gone from storage"):
            conn.open_table(path).restore(2)
        assert conn.open_table(path).version == before
        t = conn.open_table(path)
        t.restore(2, ignore_missing_files=True)
        assert _ids(conn, path) == []

    def test_the_change_feed_sees_restored_rows(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        props = {"delta.enableChangeDataFeed": "true", "delta.enableDeletionVectors": "true"}
        conn.create_table(path, SCHEMA, properties=props)
        conn.open_table(path).append(_data(5))
        conn.open_table(path).delete("id < 2")
        t = conn.open_table(path)
        t.restore(1)
        t = conn.open_table(path)
        feed = t.cdf(starting_version=t.version)
        feed = feed if hasattr(feed, "num_rows") else feed.read_all()
        changes = zip(
            feed.column("id").to_pylist(), feed.column("_change_type").to_pylist(), strict=True
        )
        assert sorted(changes) == [(0, "insert"), (1, "insert")]

    def test_restore_across_a_column_mapping_change_is_refused_up_front(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "t")
        _history_table(conn, path, {"properties": {"delta.enableInCommitTimestamps": "true"}})
        conn.open_table(path).set_properties({"delta.columnMapping.mode": "name"})
        t = conn.open_table(path)
        verdict = t.can("restore", target=1)
        assert not verdict and "column-mapping" in verdict.reason
        with pytest.raises(UnreachableTableError, match="column-mapping"):
            t.restore(1)
