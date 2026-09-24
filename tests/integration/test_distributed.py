"""Distributed reads: plan on the driver, read splits elsewhere.

The property that matters is that the union of the splits is the table --
no row lost to a split boundary, none read twice -- and that the plan survives
pickling, since that is how it reaches a worker.
"""

from __future__ import annotations

import pickle
from typing import Any

import deltaswamp as ds
import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")


@pytest.fixture
def table(tmp_path: Any) -> Any:
    from deltalake import write_deltalake
    from deltaswamp.catalog.filesystem import FilesystemCatalog

    path = str(tmp_path / "t")
    for i in range(5):
        write_deltalake(
            path,
            pa.table({"id": list(range(i * 10, i * 10 + 10)), "part": [f"p{i % 2}"] * 10}),
            mode="append",
            partition_by=["part"],
        )
    return ds.connect(catalog=FilesystemCatalog()).open_table(path)


def ids(t: Any) -> list[int]:
    return sorted(t.column("id").to_pylist())


class TestPlan:
    def test_one_split_per_file_pinned_to_a_version(self, table: Any) -> None:
        plan = table.plan_scan()
        assert len(plan.splits) == 5
        assert {s.commit_version for s in plan.splits} == {4}

    def test_splits_union_to_the_table(self, table: Any) -> None:
        plan = pickle.loads(pickle.dumps(table.plan_scan()))
        parts = [plan.read(group) for group in plan.partitions(3)]
        assert sum(p.num_rows for p in parts) == 50
        assert ids(pa.concat_tables(parts)) == list(range(50))

    def test_plan_ignores_later_writes(self, table: Any) -> None:
        plan = table.plan_scan()
        table.append(pa.table({"id": [999], "part": ["p0"]}))
        assert 999 not in ids(plan.read())

    def test_predicate_prunes_and_filters(self, table: Any) -> None:
        plan = table.plan_scan(predicate="id >= 35", columns=["id"])
        got = plan.read()
        assert got.column_names == ["id"]
        assert ids(got) == list(range(35, 50))
        assert len(plan.splits) < 5  # statistics ruled some files out

    def test_balance_is_deterministic_and_complete(self, table: Any) -> None:
        from deltaswamp.distributed import balance

        splits = table.plan_scan().splits
        groups = balance(splits, 2)
        assert groups == balance(splits, 2)
        assert sorted(s.path for g in groups for s in g) == sorted(s.path for s in splits)


class TestCatalogManaged:
    def test_plan_and_read_through_the_catalog(self, tmp_path: Any) -> None:
        from deltaswamp.catalog.ossuc import OSSUnityCatalog

        from tests.fake_uc import FakeUnityCatalog

        with FakeUnityCatalog(staging_root=tmp_path / "m") as uc:
            catalog = OSSUnityCatalog(uc.url)
            catalog.create_catalog("main")
            catalog.create_schema("main", "s")
            conn = ds.connect(catalog=catalog)
            conn.create_table("main.s.t", pa.schema([("id", pa.int64())]))
            for i in range(3):
                conn.table("main.s.t").append(pa.table({"id": [i]}))
            plan = pickle.loads(pickle.dumps(conn.table("main.s.t").plan_scan()))
            assert ids(pa.concat_tables([plan.read([s]) for s in plan.splits])) == [0, 1, 2]


class TestRay:
    def test_to_ray_dataset_reads_in_parallel(self, table: Any) -> None:
        ray = pytest.importorskip("ray")
        ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
        try:
            dataset = table.to_ray_dataset(override_num_blocks=3)
            assert sorted(r["id"] for r in dataset.take_all()) == list(range(50))
        finally:
            ray.shutdown()
