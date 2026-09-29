"""Distributed change-feed reads: runs of whole commits, read on workers.

The property that matters is the one `plan_scan` has: the union of the splits
is exactly what `cdf()` returns for the range, however the range is cut, and
anything that would fail on a worker fails at planning instead.
"""

from __future__ import annotations

import pickle
from typing import Any

import deltaswamp as ds
import pytest

pa = pytest.importorskip("pyarrow")


def _planned_scan_built() -> bool:
    from deltaswamp.engine.kernel import _native_has

    return ds.has_native() and _native_has("planned_scan")


pytestmark = pytest.mark.skipif(
    not _planned_scan_built(), reason="native extension not built, or predates this"
)

KEY = ["_commit_version", "_change_type", "id"]


def _rows(t: Any) -> list[dict[str, Any]]:
    t = t.select(sorted(t.column_names))
    rows: list[dict[str, Any]] = t.sort_by([(k, "ascending") for k in KEY]).to_pylist()
    return rows


@pytest.fixture
def feed(tmp_path: Any) -> Any:
    """Inserts, a deletion-vector delete and update, and a copy-on-write delete."""
    conn = ds.connect()
    path = str(tmp_path / "t")
    conn.write_table(
        path,
        pa.table({"id": pa.array(range(10), pa.int64()), "v": [f"v{i}" for i in range(10)]}),
        properties={
            "delta.enableChangeDataFeed": "true",
            "delta.enableDeletionVectors": "true",
        },
    )
    table = conn.table(path)
    for i in range(10, 16):
        table.append(pa.table({"id": pa.array([i], pa.int64()), "v": [f"v{i}"]}))
    table.delete("id = 3")
    table.update({"v": "'changed'"}, predicate="id = 12")
    table.append(pa.table({"id": pa.array([99], pa.int64()), "v": ["late"]}))
    return conn.table(path)


class TestTheUnionIsTheFeed:
    @pytest.mark.parametrize("split_bytes", [0, 1, 1 << 40])
    def test_any_cut_reads_what_cdf_reads(self, feed: Any, split_bytes: int) -> None:
        plan = pickle.loads(pickle.dumps(feed.plan_changes(0, split_bytes=split_bytes)))
        got = pa.concat_tables([plan.read(g) for g in plan.partitions(3)])
        want = pa.table(feed.cdf(starting_version=0, ending_version=plan.ending_version))
        assert _rows(got) == _rows(want)
        if split_bytes == 0:
            # Every commit is its own split; a split never cuts a commit.
            assert [(s.start, s.end) for s in plan.splits] == [
                (v, v) for v in range(plan.ending_version + 1)
            ]

    def test_projection_and_predicate(self, feed: Any) -> None:
        plan = feed.plan_changes(2, columns=["id"], predicate="id >= 12", split_bytes=1)
        got = plan.read()
        want = pa.table(feed.cdf(starting_version=2, columns=["id"], predicate="id >= 12"))
        assert sorted(got.column_names) == sorted(want.column_names)
        assert _rows(got) == _rows(want)

    def test_the_end_is_pinned_at_planning(self, feed: Any) -> None:
        plan = feed.plan_changes(1)
        end = plan.ending_version
        feed.append(pa.table({"id": pa.array([1000], pa.int64()), "v": ["after"]}))
        assert max(plan.read().column("_commit_version").to_pylist()) == end

    def test_splits_are_weighed_by_the_bytes_they_read(self, feed: Any) -> None:
        plan = feed.plan_changes(1, split_bytes=1)
        assert all(s.size > 0 for s in plan.splits)
        assert plan.total_bytes == sum(s.size for s in plan.splits)


class TestRefusedAtPlanning:
    def test_a_catalog_managed_table(self, tmp_path: Any) -> None:
        pytest.importorskip("deltalake")
        from deltaswamp.catalog.ossuc import OSSUnityCatalog
        from deltaswamp.errors import UnreachableTableError

        from tests.fake_uc import FakeUnityCatalog

        with FakeUnityCatalog(staging_root=tmp_path / "m") as uc:
            catalog = OSSUnityCatalog(uc.url)
            catalog.create_catalog("main")
            catalog.create_schema("main", "s")
            conn = ds.connect(catalog=catalog)
            conn.create_table(
                "main.s.t",
                pa.schema([("id", pa.int64())]),
                properties={"delta.enableChangeDataFeed": "true"},
            )
            with pytest.raises(UnreachableTableError, match="catalog commit tail"):
                conn.table("main.s.t").plan_changes(0)

    def test_the_feed_off_within_the_range(self, feed: Any) -> None:
        from deltaswamp.errors import UnreachableTableError

        feed.set_properties({"delta.enableChangeDataFeed": "false"})
        feed.append(pa.table({"id": pa.array([500], pa.int64()), "v": ["off"]}))
        feed.set_properties({"delta.enableChangeDataFeed": "true"})
        with pytest.raises(UnreachableTableError, match="not enabled at version"):
            feed.plan_changes(1)

    def test_a_schema_change_within_the_range(self, feed: Any) -> None:
        from deltaswamp.errors import UnreachableTableError

        feed.add_column([pa.field("extra", pa.string())])
        feed.append(pa.table({"id": pa.array([7], pa.int64()), "v": ["x"], "extra": ["e"]}))
        with pytest.raises(UnreachableTableError, match="schema changed at version"):
            feed.plan_changes(0)

    def test_an_unknown_column(self, feed: Any) -> None:
        from deltaswamp.errors import DeltaSwampError

        with pytest.raises((DeltaSwampError, ValueError)):
            feed.plan_changes(0, columns=["nope"])

    def test_a_table_without_the_feed(self, tmp_path: Any) -> None:
        from deltaswamp.errors import UnreachableTableError

        conn = ds.connect()
        path = str(tmp_path / "plain")
        conn.write_table(path, pa.table({"id": pa.array([1], pa.int64())}))
        with pytest.raises(UnreachableTableError):
            conn.table(path).plan_changes(0)
