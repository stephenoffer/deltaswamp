"""CDC files written by the kernel's DML on change-data-feed tables.

An UPDATE, a MERGE, a replaceWhere or a copy-on-write DELETE on a table with
the change data feed must say what changed: kernel 0.28 writes no CDC files
and refuses such a commit, so they are written here (`change_files.rs`) and
committed as `cdc` actions. The change feed -- the kernel's TableChanges on a
path table, and the log reader on any table -- must then return exactly the
rows the DML changed, with Spark's kinds of change.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Iterator
from typing import Any

import deltaswamp as ds
import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
pytest.importorskip("duckdb")
pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

from tests.fake_uc_strict import StrictUnityCatalog  # noqa: E402

# In-commit timestamps keep delta-rs from writing the table, so every DML
# below is the kernel's.
BASE = {"delta.enableChangeDataFeed": "true", "delta.enableInCommitTimestamps": "true"}
SHAPES = {
    "deletion_vectors": {"delta.enableDeletionVectors": "true"},
    "copy_on_write": {"delta.enableDeletionVectors": "false"},
    "row_tracking": {"delta.enableDeletionVectors": "true", "delta.enableRowTracking": "true"},
    "row_tracking_copy_on_write": {
        "delta.enableDeletionVectors": "false",
        "delta.enableRowTracking": "true",
    },
}
SOURCE = pa.table({"id": pa.array([1, 2, 9], pa.int64()), "v": ["m", "d", "n"]})


def _dml(table: Any, op: str) -> None:
    if op == "update":
        table.update({"v": "'y'"}, predicate="id < 2")
    elif op == "delete":
        table.delete("id < 2")
    elif op == "merge":
        (
            table.merge(SOURCE, "t.id = s.id", source_alias="s", target_alias="t")
            .when_matched_update({"v": "s.v"}, predicate="s.v = 'm'")
            .when_matched_delete(predicate="s.v = 'd'")
            .when_not_matched_insert_all()
            .execute()
        )
    else:
        table.overwrite(
            pa.table({"id": pa.array([0, 1], pa.int64()), "v": ["r", "r"]}), predicate="id < 2"
        )


EXPECTED = {
    "update": [
        ("update_postimage", 0, "y"),
        ("update_postimage", 1, "y"),
        ("update_preimage", 0, "x"),
        ("update_preimage", 1, "x"),
    ],
    "delete": [("delete", 0, "x"), ("delete", 1, "x")],
    "merge": [
        ("delete", 2, "x"),
        ("insert", 9, "n"),
        ("update_postimage", 1, "m"),
        ("update_preimage", 1, "x"),
    ],
    "replace_where": [
        ("delete", 0, "x"),
        ("delete", 1, "x"),
        ("insert", 0, "r"),
        ("insert", 1, "r"),
    ],
}
AFTER = {
    "update": [(0, "y"), (1, "y"), (2, "x"), (3, "x"), (4, "x"), (5, "x")],
    "delete": [(2, "x"), (3, "x"), (4, "x"), (5, "x")],
    "merge": [(0, "x"), (1, "m"), (3, "x"), (4, "x"), (5, "x"), (9, "n")],
    "replace_where": [(0, "r"), (1, "r"), (2, "x"), (3, "x"), (4, "x"), (5, "x")],
}


def _changes(feed: Any, version: int) -> list[tuple[Any, ...]]:
    table = feed if isinstance(feed, pa.Table) else pa.table(feed)
    return sorted(
        (r["_change_type"], r["id"], r["v"])
        for r in table.to_pylist()
        if r["_commit_version"] == version
    )


def _log_feed(conn: Any, path: str, start: int) -> Any:
    from deltaswamp.capability import Engine
    from deltaswamp.engine.log_changes import LogChangeFeed

    kernel = conn.router.engines[Engine.KERNEL]
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


def _commit(path: str, version: int) -> list[dict[str, Any]]:
    text = (pathlib.Path(path) / "_delta_log" / f"{version:020d}.json").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.mark.parametrize("op", sorted(EXPECTED))
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_feed_says_what_the_dml_changed(conn: Any, tmp_path: Any, shape: str, op: str) -> None:
    path = str(tmp_path / "t")
    rows = pa.table({"id": pa.array(range(6), pa.int64()), "v": ["x"] * 6})
    conn.write_table(path, rows, properties={**BASE, **SHAPES[shape]})
    table = conn.open_table(path)
    assert table.can("delete" if op == "delete" else "update").ok
    version = table.version + 1
    _dml(table, op)
    after = conn.open_table(path)
    assert after.version == version
    data = after.to_arrow()
    ids, values = data.column("id").to_pylist(), data.column("v").to_pylist()
    assert sorted(zip(ids, values, strict=True)) == AFTER[op]

    kernel_feed = after.cdf(starting_version=version)
    assert _changes(kernel_feed, version) == EXPECTED[op]
    assert _changes(_log_feed(conn, path, version), version) == EXPECTED[op]

    actions = _commit(path, version)
    cdc = [a["cdc"] for a in actions if "cdc" in a]
    dv_delete = op == "delete" and SHAPES[shape]["delta.enableDeletionVectors"] == "true"
    if dv_delete:
        # Readers derive a deletion-vector DELETE's rows from the vectors.
        assert cdc == []
    else:
        assert cdc and all(c["path"].startswith("_change_data/") for c in cdc)
        assert all(not c["dataChange"] for c in cdc)
        for c in cdc:
            assert (pathlib.Path(path) / c["path"]).stat().st_size == c["size"]


def test_partitioned_change_files_follow_the_partitions(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    rows = pa.table({"id": pa.array(range(6), pa.int64()), "v": ["x", "y"] * 3})
    conn.write_table(path, rows, properties={**BASE, **SHAPES["copy_on_write"]}, partition_by=["v"])
    version = conn.open_table(path).version + 1
    conn.open_table(path).update({"id": "id + 100"}, predicate="id < 2")
    cdc = [a["cdc"] for a in _commit(path, version) if "cdc" in a]
    assert {tuple(sorted(c["partitionValues"].items())) for c in cdc} == {
        (("v", "x"),),
        (("v", "y"),),
    }
    assert all(c["path"].startswith("_change_data/v=") for c in cdc)
    got = sorted(
        (r["_change_type"], r["id"], r["v"])
        for r in pa.table(conn.open_table(path).cdf(starting_version=version)).to_pylist()
    )
    assert got == [
        ("update_postimage", 100, "x"),
        ("update_postimage", 101, "y"),
        ("update_preimage", 0, "x"),
        ("update_preimage", 1, "y"),
    ]


def test_column_mapped_change_files_use_physical_names(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    rows = pa.table({"id": pa.array(range(4), pa.int64()), "v": ["x"] * 4})
    conn.write_table(
        path,
        rows,
        properties={
            **BASE,
            **SHAPES["deletion_vectors"],
            "delta.columnMapping.mode": "name",
        },
    )
    conn.open_table(path).rename_column("v", "w")
    conn.open_table(path).update({"w": "'z'"}, predicate="id = 3")
    version = conn.open_table(path).version
    feed = pa.table(conn.open_table(path).cdf(starting_version=version))
    assert sorted((r["_change_type"], r["id"], r["w"]) for r in feed.to_pylist()) == [
        ("update_postimage", 3, "z"),
        ("update_preimage", 3, "x"),
    ]


# ------------------------------------------------------------ catalog-managed

NAME = "main.sales.cm"


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


@pytest.fixture(params=["oss", "databricks"])
def managed(request: Any, uc: StrictUnityCatalog, monkeypatch: pytest.MonkeyPatch) -> Any:
    from deltaswamp.catalog.ossuc import OSSUnityCatalog
    from deltaswamp.connection import Connection

    setup = Connection(catalog=OSSUnityCatalog(uc.url), router=_router())
    setup.create_catalog("main")
    setup.create_schema("main.sales")
    if request.param == "oss":
        return setup
    pytest.importorskip("databricks.sdk")
    from deltaswamp.catalog.databricks import DatabricksUnityCatalog

    for key in list(os.environ):
        if key.startswith("DATABRICKS"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "/nonexistent/databrickscfg")
    return Connection(
        catalog=DatabricksUnityCatalog(host=uc.url, token="dapi-fake"), router=_router()
    )


@pytest.mark.parametrize("op", ["update", "merge", "delete"])
@pytest.mark.parametrize("shape", ["deletion_vectors", "copy_on_write"])
def test_catalog_managed_dml_writes_its_changes(managed: Any, shape: str, op: str) -> None:
    schema = pa.schema([("id", pa.int64()), ("v", pa.string())])
    managed.create_table(NAME, schema, properties={**BASE, **SHAPES[shape]})
    managed.table(NAME).append(pa.table({"id": pa.array(range(6), pa.int64()), "v": ["x"] * 6}))
    table = managed.table(NAME)
    _dml(table, op)
    after = managed.table(NAME)
    feed = pa.table(after.cdf(starting_version=after.version))
    assert _changes(feed, after.version) == EXPECTED[op]
