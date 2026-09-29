"""Planned reads: a worker reads its splits without touching the log.

A split carries its file's scan row, and the plan the protocol and metadata it
was made from, so a read task builds its snapshot from those and hands the rows
to the kernel: no listing, no replay. Every read task used to replay the whole
log (with a catalog-managed table, the staged commits too), which cost each of
N tasks the table's full file list, and failed once the catalog had published
and removed the staged commits a long job was planned from.
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Any

import deltaswamp as ds
import pytest

pa = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")


def _planned_scan_built() -> bool:
    from deltaswamp.engine.kernel import _native_has

    return _native_has("planned_scan")


needs_planned = pytest.mark.skipif(
    not _planned_scan_built(), reason="native extension predates planned scans"
)


def _sorted(t: Any) -> Any:
    return t.sort_by([(t.column_names[0], "ascending")])


@pytest.fixture
def mapped(tmp_path: Any) -> Any:
    """Partitioned, column-mapped, with a deletion vector and a later append."""
    conn = ds.connect()
    path = str(tmp_path / "t")
    data = pa.table(
        {
            "id": pa.array(range(40), pa.int64()),
            "part": [f"p{i % 3}" for i in range(40)],
            "name": [f"n{i}" for i in range(40)],
        }
    )
    conn.write_table(
        path,
        data,
        partition_by=["part"],
        properties={"delta.enableDeletionVectors": "true", "delta.columnMapping.mode": "name"},
    )
    table = conn.table(path)
    table.delete("id % 7 = 0")
    table.append(pa.table({"id": [100, 101], "part": ["p0", "p9"], "name": ["x", "y"]}))
    return table


@needs_planned
class TestSplitsCarryWhatAWorkerNeeds:
    def test_every_split_has_its_scan_row_and_one_shared_planned_snapshot(
        self, mapped: Any
    ) -> None:
        plan = mapped.plan_scan()
        assert all(s.scan_row for s in plan.splits)
        states = {id(s.planned) for s in plan.splits}
        assert len(states) == 1  # one object, so a pickle carries it once
        state = plan.splits[0].planned
        assert state.version == plan.version
        one = len(pickle.dumps(plan.splits[:1]))
        two = len(pickle.dumps(plan.splits[:2]))
        assert two - one < len(state.metadata_json)

    def test_scan_rows_leave_statistics_out(self, mapped: Any) -> None:
        plan = mapped.plan_scan()
        assert all('"stats"' not in s.scan_row for s in plan.splits)


@needs_planned
class TestAPlannedReadIsTheTable:
    @pytest.mark.parametrize(
        ("columns", "predicate"),
        [
            (None, None),
            (["id"], None),
            (["part"], None),  # partition columns only
            (["name", "id"], "id > 10"),
            (None, "part = 'p1'"),
            (["id"], "part = 'p9' OR name = 'n5'"),
        ],
    )
    def test_matches_the_driver_read(self, mapped: Any, columns: Any, predicate: Any) -> None:
        plan = pickle.loads(pickle.dumps(mapped.plan_scan(columns=columns, predicate=predicate)))
        got = pa.concat_tables([plan.read(g) for g in plan.partitions(3)])
        want = mapped.to_arrow(columns=columns, predicate=predicate)
        assert got.schema == want.schema
        assert _sorted(got).equals(_sorted(want))

    def test_row_tracked_table(self, tmp_path: Any) -> None:
        conn = ds.connect()
        path = str(tmp_path / "rt")
        conn.write_table(
            path,
            pa.table({"id": pa.array(range(10), pa.int64())}),
            properties={"delta.enableRowTracking": "true"},
        )
        table = conn.table(path)
        table.append(pa.table({"id": pa.array([10, 11], pa.int64())}))
        plan = pickle.loads(pickle.dumps(table.plan_scan()))
        assert sorted(plan.read().column("id").to_pylist()) == list(range(12))


@needs_planned
class TestNoLogOnTheWorker:
    def test_reads_after_the_log_is_gone(self, mapped: Any) -> None:
        """The strongest form of "no replay": there is no log left to replay."""
        plan = pickle.dumps(mapped.plan_scan())
        want = mapped.to_arrow()
        root = Path(mapped.location.removeprefix("file://"))
        shutil.rmtree(root / "_delta_log")
        from deltaswamp.engine.kernel import KernelEngine

        KernelEngine._snapshots.clear()  # a fresh worker holds no snapshot
        got = pickle.loads(plan).read()
        assert _sorted(got).equals(_sorted(want))

    def test_a_planned_snapshot_refuses_what_needs_the_log(self, mapped: Any) -> None:
        from deltaswamp._native import Snapshot

        state = mapped.plan_scan().splits[0].planned
        snapshot = Snapshot.planned(
            mapped.location, state.version, state.protocol_json, state.metadata_json
        )
        with pytest.raises(ValueError, match="built from a scan plan"):
            snapshot.scan()
        with pytest.raises(ValueError, match="built from a scan plan"):
            snapshot.files()

    def test_hand_built_splits_still_read_through_the_log(self, mapped: Any) -> None:
        from dataclasses import replace

        plan = mapped.plan_scan()
        bare = tuple(replace(s, scan_row=None, planned=None) for s in plan.splits)
        got = replace(plan, splits=bare).read()
        assert _sorted(got).equals(_sorted(mapped.to_arrow()))


@needs_planned
class TestCatalogManaged:
    def test_reads_after_the_staged_commits_are_removed(self, tmp_path: Any) -> None:
        """A long job's reads outlive the catalog publishing its staged commits."""
        pytest.importorskip("deltalake")
        from deltaswamp.catalog.ossuc import OSSUnityCatalog
        from deltaswamp.engine.kernel import KernelEngine

        from tests.fake_uc import FakeUnityCatalog

        with FakeUnityCatalog(staging_root=tmp_path / "m") as uc:
            catalog = OSSUnityCatalog(uc.url)
            catalog.create_catalog("main")
            catalog.create_schema("main", "s")
            conn = ds.connect(catalog=catalog)
            conn.create_table("main.s.t", pa.schema([("id", pa.int64())]))
            for i in range(3):
                conn.table("main.s.t").append(pa.table({"id": [i]}))
            table = conn.table("main.s.t")
            plan = pickle.dumps(table.plan_scan())
            staged = Path(table.location.removeprefix("file://")) / "_delta_log" / "_staged_commits"
            assert any(staged.iterdir())  # the tail really is staged
            shutil.rmtree(staged)
            KernelEngine._snapshots.clear()
            worker = pickle.loads(plan)
            # The log alone can no longer produce the planned version...
            with pytest.raises(Exception):  # noqa: B017 -- any resolve failure
                worker.engine.snapshot(worker.table, version=worker.version)
            # ...and the planned read never asks it to.
            assert sorted(worker.read().column("id").to_pylist()) == [0, 1, 2]


class _CountingSnapshot:
    """`_native.Snapshot`, counting resolves (the class itself cannot be patched)."""

    resolves = 0

    def __init__(self, real: Any) -> None:
        self._real = real

    def resolve(self, *args: Any, **kwargs: Any) -> Any:
        type(self).resolves += 1
        return self._real.resolve(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@pytest.fixture
def catalog_table(tmp_path: Any) -> Any:
    pytest.importorskip("deltalake")
    from deltaswamp.catalog.ossuc import OSSUnityCatalog

    from tests.fake_uc import FakeUnityCatalog

    with FakeUnityCatalog(staging_root=tmp_path / "m") as uc:
        catalog = OSSUnityCatalog(uc.url)
        catalog.create_catalog("main")
        catalog.create_schema("main", "s")
        conn = ds.connect(catalog=catalog)
        conn.create_table("main.s.t", pa.schema([("id", pa.int64())]))
        conn.table("main.s.t").append(pa.table({"id": [0]}))
        yield conn


class TestCatalogManagedSnapshotsAreReused:
    def test_repeated_writes_of_one_plan_resolve_once(
        self, catalog_table: Any, monkeypatch: Any
    ) -> None:
        import deltaswamp._native as native

        counting = _CountingSnapshot(native.Snapshot)
        monkeypatch.setattr(native, "Snapshot", counting)
        plan = catalog_table.table("main.s.t").plan_write()
        before = type(counting).resolves
        fragments = [plan.write(pa.table({"id": [i]})) for i in range(1, 5)]
        assert type(counting).resolves - before <= 1
        plan.commit(fragments)
        got = catalog_table.table("main.s.t").to_arrow()
        assert sorted(got.column("id").to_pylist()) == [0, 1, 2, 3, 4]

    def test_a_new_commit_is_a_new_snapshot(self, catalog_table: Any) -> None:
        table = catalog_table.table("main.s.t")
        assert table.to_arrow().num_rows == 1
        table.append(pa.table({"id": [1]}))
        assert catalog_table.table("main.s.t").to_arrow().num_rows == 2


class TestRayNeverReadsOnTheDriverUnasked:
    """A table no engine can plan used to be read whole on the driver, silently."""

    @pytest.fixture
    def unplannable(self, mapped: Any, monkeypatch: Any) -> Any:
        from deltaswamp.errors import UnreachableTableError

        def refuse(**_: Any) -> Any:
            raise UnreachableTableError(
                "scan", "sql: does not support distributed_scan (the table has a row filter)"
            )

        monkeypatch.setattr(mapped, "plan_scan", refuse)
        return mapped

    def test_is_refused_by_default_with_the_reason(self, unplannable: Any) -> None:
        pytest.importorskip("ray.data")
        from deltaswamp.errors import UnreachableTableError

        with pytest.raises(UnreachableTableError, match="row filter") as info:
            unplannable.to_ray_dataset()
        assert "allow_driver_read=True" in str(info.value)

    def test_reads_on_the_driver_when_asked(self, unplannable: Any) -> None:
        pytest.importorskip("ray.data")
        dataset = unplannable.to_ray_dataset(allow_driver_read=True)
        assert dataset.count() == unplannable.to_arrow().num_rows

    def test_a_driver_read_past_the_bound_is_refused(self, unplannable: Any) -> None:
        pytest.importorskip("ray.data")
        from deltaswamp.errors import EngineLimitError

        with pytest.raises(EngineLimitError, match="driver_read_max_bytes"):
            unplannable.to_ray_dataset(allow_driver_read=True, driver_read_max_bytes=16)
