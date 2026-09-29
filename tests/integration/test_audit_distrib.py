"""Regressions for distributed-write and change-feed defects found by the audit.

Every test here failed before its fix. The worst of them left the table
unreadable while the commit reported success.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

from deltalake import DeltaTable, write_deltalake  # noqa: E402
from deltaswamp.errors import (  # noqa: E402
    BackfillRequiredError,
    ChangeFeedSchemaChangeError,
    CommitConflictError,
    InvalidArgumentError,
    MetadataChangedError,
    MissingDataFileError,
    UnreachableTableError,
)


@pytest.fixture
def conn() -> Any:
    return ds.connect("file://")


def _ids(data: Any, schema: Any = None) -> Any:
    return pa.table(data, schema=schema)


class TestCommitAfterAConcurrentMetadataChange:
    """A planned commit rebased onto a snapshot whose schema had changed."""

    def test_type_change_is_refused_and_the_table_stays_readable(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "t")
        write_deltalake(path, _ids({"id": pa.array([1, 2], pa.int64()), "s": ["a", "b"]}))
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_ids({"id": pa.array([3], pa.int64()), "s": ["c"]}))
        write_deltalake(
            path, _ids({"id": ["x"], "s": [1.5]}), mode="overwrite", schema_mode="overwrite"
        )
        with pytest.raises(MetadataChangedError) as caught:
            plan.commit([fragment])
        # A CommitConflictError, so existing handlers still see a conflict.
        assert isinstance(caught.value, CommitConflictError)
        assert conn.open_table(path).to_arrow().to_pylist() == [{"id": "x", "s": 1.5}]
        assert DeltaTable(path).to_pyarrow_table().num_rows == 1

    def test_is_not_retried(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        write_deltalake(path, _ids({"id": pa.array([1], pa.int64())}))
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_ids({"id": pa.array([3], pa.int64())}))
        write_deltalake(path, _ids({"id": ["x"]}), mode="overwrite", schema_mode="overwrite")
        calls = []
        commit_files = plan.engine.commit_files

        def counted(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return commit_files(*args, **kwargs)

        plan.engine.commit_files = counted
        try:
            with pytest.raises(MetadataChangedError):
                plan.commit([fragment], retries=5)
        finally:
            del plan.engine.commit_files
        assert len(calls) == 1

    def test_column_mapping_drop_and_re_add_is_refused(self, conn: Any, tmp_path: Any) -> None:
        """The value landed under the old physical name and read back as null."""
        path = str(tmp_path / "t")
        t = conn.create_table(
            path,
            pa.schema([("id", pa.int64()), ("b", pa.string())]),
            properties={
                "delta.columnMapping.mode": "name",
                "delta.feature.columnMapping": "supported",
            },
        )
        t.append(_ids({"id": pa.array([1], pa.int64()), "b": ["a"]}))
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_ids({"id": pa.array([3], pa.int64()), "b": ["c"]}))
        conn.open_table(path).drop_column("b")
        conn.open_table(path).add_column(pa.field("b", pa.string()))
        with pytest.raises(MetadataChangedError):
            plan.commit([fragment])
        assert conn.open_table(path).count() == 1

    def test_adding_a_nullable_column_still_commits(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        conn.create_table(
            path,
            pa.schema([("id", pa.int64())]),
            properties={
                "delta.columnMapping.mode": "name",
                "delta.feature.columnMapping": "supported",
            },
        )
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_ids({"id": pa.array([3], pa.int64())}))
        conn.open_table(path).add_column(pa.field("x", pa.string()))
        plan.commit([fragment])
        assert conn.open_table(path).to_arrow().to_pylist() == [{"id": 3, "x": None}]

    def test_partitioning_change_is_refused(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        write_deltalake(
            path, _ids({"id": pa.array([1], pa.int64()), "p": ["a"]}), partition_by=["p"]
        )
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_ids({"id": pa.array([3], pa.int64()), "p": ["c"]}))
        write_deltalake(
            path,
            _ids({"id": pa.array([9], pa.int64()), "p": ["z"]}),
            mode="overwrite",
            schema_mode="overwrite",
            partition_by=["id"],
        )
        with pytest.raises(MetadataChangedError):
            plan.commit([fragment])

    def test_concurrent_overwrite_allowed_is_still_refused(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        write_deltalake(path, _ids({"id": pa.array([1], pa.int64()), "s": ["a"]}))
        plan = conn.open_table(path).plan_write(mode="overwrite")
        fragment = plan.write(_ids({"id": pa.array([3], pa.int64()), "s": ["c"]}))
        write_deltalake(
            path, _ids({"id": ["x"], "s": [1.5]}), mode="overwrite", schema_mode="overwrite"
        )
        with pytest.raises(MetadataChangedError):
            plan.commit([fragment], allow_concurrent_overwrite=True)
        assert conn.open_table(path).to_arrow().to_pylist() == [{"id": "x", "s": 1.5}]

    def test_unrelated_metadata_changes_still_commit(self, conn: Any, tmp_path: Any) -> None:
        """A property or table comment does not change what a file holds."""
        path = str(tmp_path / "t")
        write_deltalake(path, _ids({"id": pa.array([1], pa.int64())}))
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(_ids({"id": pa.array([3], pa.int64())}))
        conn.open_table(path).set_properties({"team": "x"})
        conn.open_table(path).append(_ids({"id": pa.array([2], pa.int64())}))
        plan.commit([fragment])
        assert sorted(conn.open_table(path).to_arrow().column("id").to_pylist()) == [1, 2, 3]


class TestPlannedCommitPublishesOnBackfillDemand:
    """Nothing on the distributed path published, so it wedged at the catalog's cap."""

    def test_commits_keep_landing_past_the_cap(self, tmp_path: Any) -> None:
        from deltaswamp.catalog.ossuc import OSSUnityCatalog
        from deltaswamp.connection import Connection

        from tests.fake_uc_strict import StrictUnityCatalog
        from tests.helpers import direct_router

        with StrictUnityCatalog(staging_root=str(tmp_path / "m"), max_unbackfilled=2) as uc:
            conn = Connection(catalog=OSSUnityCatalog(uc.url), router=direct_router())
            conn.create_catalog("main")
            conn.create_schema("main.s")
            conn.create_table("main.s.cm", pa.schema([("id", pa.int64())]))
            for i in range(5):
                plan = conn.table("main.s.cm").plan_write()
                try:
                    plan.commit([plan.write(_ids({"id": pa.array([i], pa.int64())}))])
                except BackfillRequiredError as exc:  # pragma: no cover - the regression
                    pytest.fail(f"commit {i} hit the backfill cap: {exc}")
            assert conn.table("main.s.cm").count() == 5


class TestChangeFeedErrorsAreTyped:
    """The change feed raised raw ArrowInvalid where scans raise typed errors."""

    def test_vacuumed_file(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        cdf = {"delta.enableChangeDataFeed": "true"}
        write_deltalake(path, _ids({"id": pa.array([0], pa.int64())}), configuration=cdf)
        write_deltalake(path, _ids({"id": pa.array([1], pa.int64())}), mode="append")
        write_deltalake(path, _ids({"id": pa.array([2], pa.int64())}), mode="overwrite")
        DeltaTable(path).vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False)
        with pytest.raises(MissingDataFileError):
            list(conn.open_table(path).changes(1))
        with pytest.raises(MissingDataFileError):
            conn.open_table(path).cdf(starting_version=1).read_all()

    def test_incompatible_schema_change_names_the_version(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        cdf = {"delta.enableChangeDataFeed": "true"}
        write_deltalake(
            path, _ids({"id": pa.array([1], pa.int64()), "s": ["a"]}), configuration=cdf
        )
        write_deltalake(path, _ids({"id": pa.array([2], pa.int64()), "s": ["b"]}), mode="append")
        write_deltalake(
            path, _ids({"id": ["x"], "s": [1.0]}), mode="overwrite", schema_mode="overwrite"
        )
        with pytest.raises(ChangeFeedSchemaChangeError) as caught:
            list(conn.open_table(path).changes(0))
        assert caught.value.version == 2
        assert "Traceback" not in str(caught.value)
        # The range before the change still reads.
        assert conn.open_table(path).cdf(starting_version=0, ending_version=1).read_all().num_rows


class TestCommittingNoFragments:
    """commit([]) added an empty version every call; an overwrite truncated silently."""

    def test_empty_append_adds_no_version(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        write_deltalake(path, _ids({"id": pa.array([1], pa.int64())}))
        plan = conn.open_table(path).plan_write()
        assert plan.commit([]) == 0
        assert plan.commit([plan.write(_ids({"id": pa.array([], pa.int64())}))]) == 0
        assert conn.open_table(path).version == 0

    def test_empty_append_with_txn_records_it(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        write_deltalake(path, _ids({"id": pa.array([1], pa.int64())}))
        conn.open_table(path).plan_write(txn=("job", 1)).commit([])
        assert conn.open_table(path).txn_version("job") == 1

    def test_empty_overwrite_needs_opt_in(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        write_deltalake(path, _ids({"id": pa.array([1], pa.int64())}))
        plan = conn.open_table(path).plan_write(mode="overwrite")
        with pytest.raises(UnreachableTableError, match="allow_empty_overwrite"):
            plan.commit([])
        assert conn.open_table(path).count() == 1
        plan.commit([], allow_empty_overwrite=True)
        assert conn.open_table(path).count() == 0


@pytest.mark.parametrize("n", [0, -5, True, None, 2.5])
def test_partitions_refuses_a_non_positive_count(conn: Any, tmp_path: Any, n: Any) -> None:
    path = str(tmp_path / "t")
    write_deltalake(path, _ids({"id": pa.array([1], pa.int64())}))
    plan = conn.open_table(path).plan_scan()
    with pytest.raises(InvalidArgumentError):
        plan.partitions(n)


class TestBlindAppendsUnderContention:
    """Blind appends failed with CommitConflictError that they should have retried."""

    def _race(self, tables: list[Any], work: Any) -> dict[int, str]:
        barrier = threading.Barrier(len(tables))
        out: dict[int, str] = {}

        def run(i: int) -> None:
            barrier.wait()
            try:
                work(i, tables[i])
                out[i] = "ok"
            except Exception as exc:
                out[i] = f"{type(exc).__name__}: {exc}"

        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(tables))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return out

    def test_delta_rs_append_racing_a_delete(self, conn: Any, tmp_path: Any) -> None:
        """delta-rs's checker failed the append: "a concurrent transaction deleted data"."""
        schema = pa.schema([("id", pa.int64()), ("p", pa.string())])
        for rep in range(3):
            path = str(tmp_path / f"t{rep}")
            conn.create_table(path, schema, partition_by=["p"])
            conn.open_table(path).append(_ids({"id": [0], "p": ["base"]}, schema))
            tables = [conn.open_table(path) for _ in range(8)]

            def work(i: int, t: Any) -> None:
                if i % 2:
                    t.append(_ids({"id": [100 + i], "p": ["other"]}, schema))
                else:
                    t.delete("p = 'base'")

            out = self._race(tables, work)
            appends = {i: v for i, v in out.items() if i % 2}
            assert all(v == "ok" for v in appends.values()), appends
            ids = conn.open_table(path).to_arrow().column("id").to_pylist()
            # Every append landed exactly once.
            assert sorted(i for i in ids if i >= 100) == [101, 103, 105, 107]

    def test_kernel_appends_from_many_threads(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        conn.create_table(path, pa.schema([("id", pa.int64())]), cluster_by=["id"])
        assert conn.open_table(path).can("append").engine is not None
        tables = [conn.open_table(path) for _ in range(8)]

        def work(i: int, t: Any) -> None:
            for j in range(5):
                t.append(_ids({"id": pa.array([i * 100 + j], pa.int64())}))

        out = self._race(tables, work)
        assert all(v == "ok" for v in out.values()), out
        ids = conn.open_table(path).to_arrow().column("id").to_pylist()
        assert len(ids) == 40 and len(set(ids)) == 40
