"""The change data feed read from the log (`engine/log_changes.py`).

The kernel's TableChanges cannot open a catalog-managed table, so its feed is
derived from the commits of a snapshot resolved with the catalog's tail. On
path tables the same reader must return exactly what TableChanges returns,
shape by shape; on a catalog-managed table it must see the commits the
catalog ratified and has not published.
"""

from __future__ import annotations

import os
import pathlib
import pickle
from collections.abc import Iterator
from typing import Any

import deltaswamp as ds
import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

from tests.fake_uc_strict import StrictUnityCatalog  # noqa: E402

CDF = {"delta.enableChangeDataFeed": "true"}
DV = {"delta.enableDeletionVectors": "true"}


def _kernel(conn: Any) -> Any:
    from deltaswamp.capability import Engine

    return conn.router.engines[Engine.KERNEL]


def _key(table: Any) -> list[tuple[Any, ...]]:
    names = [n for n in table.schema.names if n != "_commit_timestamp"]
    rows = [(*(row[n] for n in names), row["_commit_timestamp"]) for row in table.to_pylist()]
    return sorted(rows, key=repr)


def _log_feed(conn: Any, path: str, start: int = 0) -> Any:
    from deltaswamp.engine.log_changes import LogChangeFeed

    kernel = _kernel(conn)
    resolved = conn.open_table(path)._enrich()
    return (
        LogChangeFeed(
            kernel.snapshot(resolved),
            kernel.snapshot(resolved, version=start),
            start,
            None,
            kernel.file_commit_times(resolved),
        )
        .reader()
        .read_all()
    )


def _kernel_feed(conn: Any, path: str, start: int = 0) -> Any:
    resolved = conn.open_table(path)._enrich()
    return _kernel(conn).cdf(resolved, starting_version=start).read_all()


def _ids(n: int, base: int = 0) -> Any:
    return pa.table(
        {
            "id": pa.array(range(base, base + n), pa.int64()),
            "g": pa.array(["a", "b"] * (n // 2) + ["a"] * (n % 2)),
        }
    )


SHAPES: dict[str, dict[str, Any]] = {
    "plain": {},
    "partitioned": {"partition_by": ["g"]},
    "deletion_vectors": {"properties": DV},
    "partitioned_deletion_vectors": {"properties": DV, "partition_by": ["g"]},
    "column_mapping": {"properties": {**DV, "delta.columnMapping.mode": "name"}},
    "row_tracking": {"properties": {**DV, "delta.enableRowTracking": "true"}},
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_log_reader_is_what_table_changes_reads(conn: Any, tmp_path: Any, shape: str) -> None:
    spec = SHAPES[shape]
    path = str(tmp_path / "t")
    properties = {**CDF, **spec.get("properties", {})}
    kwargs = {"partition_by": spec["partition_by"]} if "partition_by" in spec else {}
    conn.write_table(path, _ids(10), properties=properties, **kwargs)
    conn.open_table(path).append(_ids(4, base=10))
    conn.open_table(path).delete("id % 3 = 0")
    conn.open_table(path).delete("id = 4")
    conn.open_table(path).overwrite(_ids(3, base=100))
    expected = _kernel_feed(conn, path)
    got = _log_feed(conn, path)
    assert got.schema.equals(expected.schema)
    assert _key(got) == _key(expected)
    later = _kernel_feed(conn, path, start=2)
    assert _key(_log_feed(conn, path, start=2)) == _key(later)


def test_a_restored_deletion_vector_reads_back_as_inserts(conn: Any, tmp_path: Any) -> None:
    """RESTORE puts the old vector back: the rows it no longer deletes are inserts."""
    path = str(tmp_path / "t")
    conn.write_table(path, _ids(6), properties={**CDF, **DV})
    conn.open_table(path).delete("id < 3")
    conn.open_table(path).restore(1)
    assert _key(_log_feed(conn, path, start=2)) == _key(_kernel_feed(conn, path, start=2))
    rows = _log_feed(conn, path, start=3).to_pylist()
    assert sorted((r["_change_type"], r["id"]) for r in rows) == [
        ("insert", 0),
        ("insert", 1),
        ("insert", 2),
    ]


# ------------------------------------------------------------ catalog-managed

NAME = "main.sales.cm"
SCHEMA = pa.schema([("id", pa.int64()), ("c", pa.string())])


def _router() -> Any:
    from deltaswamp.capability import Engine
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router

    return Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()})


@pytest.fixture
def uc(tmp_path: pathlib.Path) -> Iterator[StrictUnityCatalog]:
    server = StrictUnityCatalog(staging_root=tmp_path / "managed")
    with server:
        yield server


def _oss(uc: StrictUnityCatalog) -> Any:
    from deltaswamp.catalog.ossuc import OSSUnityCatalog
    from deltaswamp.connection import Connection

    return Connection(catalog=OSSUnityCatalog(uc.url), router=_router())


def _dbx(uc: StrictUnityCatalog, monkeypatch: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("databricks.sdk")
    from deltaswamp.catalog.databricks import DatabricksUnityCatalog
    from deltaswamp.connection import Connection

    for key in list(os.environ):
        if key.startswith("DATABRICKS"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "/nonexistent/databrickscfg")
    return Connection(
        catalog=DatabricksUnityCatalog(host=uc.url, token="dapi-fake"), router=_router()
    )


@pytest.fixture(params=["oss", "databricks"])
def managed(request: Any, uc: StrictUnityCatalog, monkeypatch: pytest.MonkeyPatch) -> Any:
    setup = _oss(uc)
    setup.create_catalog("main")
    setup.create_schema("main.sales")
    return setup if request.param == "oss" else _dbx(uc, monkeypatch)


def _history(conn: Any, properties: dict[str, str]) -> Any:
    conn.create_table(NAME, SCHEMA, properties={**CDF, **properties})
    for i in range(3):
        conn.table(NAME).append(pa.table({"id": [i * 10 + 1, i * 10 + 2], "c": ["a", "b"]}))
    conn.table(NAME).delete("id = 11")
    return conn.table(NAME)


def _changes(feed: Any) -> list[tuple[Any, ...]]:
    table = pa.table(feed) if not isinstance(feed, pa.Table) else feed
    return sorted(
        (r["_commit_version"], r["_change_type"], r["id"], r["c"]) for r in table.to_pylist()
    )


@pytest.mark.parametrize("properties", [DV, {"delta.enableDeletionVectors": "false"}])
def test_a_catalog_managed_feed_reads_its_unpublished_commits(
    managed: Any, uc: StrictUnityCatalog, properties: dict[str, str]
) -> None:
    table = _history(managed, properties)
    assert table.is_catalog_managed
    assert table.can("cdf").ok, table.can("cdf").reason
    # The catalog holds the tail unpublished: TableChanges would miss it.
    assert table.resolved.log_tail
    got = _changes(table.cdf(starting_version=1))
    inserts = [(v, i, c) for v, kind, i, c in got if kind == "insert"]
    assert (1, 1, "a") in inserts and (3, 22, "b") in inserts
    assert [(v, i) for v, kind, i, _ in got if kind == "delete"] == [(4, 11)]


def test_a_catalog_managed_feed_takes_timestamp_bounds(managed: Any) -> None:
    table = _history(managed, DV)
    # Every catalog-managed commit carries its in-commit timestamp.
    feed = pa.table(table.cdf(starting_version=1))
    stamps = {row["_commit_version"]: row["_commit_timestamp"] for row in feed.to_pylist()}
    at = stamps[4]
    assert _changes(table.cdf(starting_timestamp=at, ending_timestamp=at)) == [
        (4, "delete", 11, "a")
    ]
    assert {v for v, *_ in _changes(table.cdf(starting_timestamp=stamps[2]))} == {2, 3, 4}


def test_a_catalog_managed_feed_plans_for_workers(managed: Any) -> None:
    table = _history(managed, DV)
    plan = table.plan_changes(1, split_bytes=1)
    assert len(plan.splits) > 1
    worker = pickle.loads(pickle.dumps(plan))
    rows = pa.concat_tables(
        [pa.table(worker.read(group)) for group in worker.partitions(3) if group]
    )
    assert _changes(rows) == _changes(table.cdf(starting_version=1))


def test_a_catalog_managed_feed_off_at_a_version_is_named(managed: Any) -> None:
    managed.create_table(NAME, SCHEMA, properties=DV)
    managed.table(NAME).append(pa.table({"id": [1], "c": ["a"]}))
    with pytest.raises(ds.UnreachableTableError):
        managed.table(NAME).cdf(starting_version=0).read_all()
