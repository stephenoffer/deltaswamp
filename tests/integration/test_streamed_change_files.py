"""CDC files from DML that does not hold its rows.

A MERGE evaluated in buckets over spill files, and an UPDATE or DELETE whose
rows are streamed into their files, must say exactly what the same DML run in
one piece says on a change-data-feed table: the change rows are classified
per bucket and streamed into their files as the data rows are. The feed is
read both through the kernel's TableChanges (path tables) and the log reader,
and on catalog-managed tables through the catalog.
"""

from __future__ import annotations

import os
import pathlib
import shutil
from collections.abc import Iterator
from typing import Any

import pytest
from deltaswamp.capability import Engine

pa = pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")
native = pytest.importorskip("deltaswamp._native")
pytestmark = pytest.mark.skipif(
    not {"dml_stream", "change_files"} <= set(native.FEATURES),
    reason="native build predates streamed DML or change files",
)

from tests.fake_uc_strict import StrictUnityCatalog  # noqa: E402

# In-commit timestamps keep delta-rs from writing the table: every DML below
# is the kernel's.
BASE = {"delta.enableChangeDataFeed": "true", "delta.enableInCommitTimestamps": "true"}
SHAPES = {
    "deletion_vectors": {"delta.enableDeletionVectors": "true"},
    "copy_on_write": {"delta.enableDeletionVectors": "false"},
}
SCHEMA = pa.schema([("id", pa.int64()), ("name", pa.string()), ("qty", pa.int64())])


def _batches() -> Iterator[Any]:
    # Several files, so buckets and streamed rewrites span more than one.
    for start in range(0, 400, 100):
        ids = list(range(start, start + 100))
        yield pa.table(
            {
                "id": pa.array(ids, pa.int64()),
                "name": [f"n{i}" for i in ids],
                "qty": pa.array([i % 7 for i in ids], pa.int64()),
            }
        )


def _source() -> Any:
    ids = [*range(150, 250), *range(1000, 1060)]
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "name": [f"s{i}" for i in ids],
            "qty": pa.array([i % 5 for i in ids], pa.int64()),
        }
    )


def _merge(table: Any) -> Any:
    return (
        table.merge(_source(), "t.id = s.id", source_alias="s", target_alias="t")
        .when_matched_delete(predicate="s.qty = 0")
        .when_matched_update({"name": "s.name", "qty": "t.qty + s.qty"})
        .when_not_matched_insert_all(predicate="s.qty <> 3")
        .when_not_matched_by_source_update({"qty": "t.qty * 10"}, predicate="t.id < 20")
        .execute()
    )


def _dml(table: Any, op: str) -> None:
    if op == "merge":
        _merge(table)
    elif op == "update":
        table.update({"qty": "qty + 100"}, predicate="id % 3 = 0")
    else:
        table.delete("id % 4 = 1")


def _kernel(conn: Any) -> Any:
    return conn.router.engines[Engine.KERNEL]


def _streamed(monkeypatch: pytest.MonkeyPatch, kernel: Any, spill: Any) -> None:
    """Every bucket, spill and chunk as small as it goes."""
    monkeypatch.setattr(kernel, "dml_bucket_bytes", 1)
    monkeypatch.setattr(kernel, "dml_max_buckets", 7)
    monkeypatch.setattr(kernel, "dml_spill_directory", str(spill))
    monkeypatch.setattr(kernel, "dml_rewrite_files", 1)
    monkeypatch.setattr(kernel, "dml_chunk_bytes", 1)


def _feed(rows: Any, version: int) -> list[tuple[Any, ...]]:
    table = rows if isinstance(rows, pa.Table) else pa.table(rows)
    return sorted(
        (r["_change_type"], r["id"], r["name"], r["qty"])
        for r in table.to_pylist()
        if r["_commit_version"] == version
    )


def _log_feed(conn: Any, path: str, start: int) -> Any:
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


def _path_table(conn: Any, path: str, shape: str) -> str:
    conn.create_table(path, SCHEMA, properties={**BASE, **SHAPES[shape]})
    for batch in _batches():
        conn.open_table(path).append(batch)
    return path


@pytest.mark.parametrize("op", ["merge", "update", "delete"])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_streamed_dml_says_what_the_dml_in_one_piece_says(
    conn: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, shape: str, op: str
) -> None:
    whole = _path_table(conn, str(tmp_path / "whole"), shape)
    streamed = str(tmp_path / "streamed")
    shutil.copytree(whole, streamed)
    version = conn.open_table(whole).version + 1

    _dml(conn.open_table(whole), op)
    expected = _feed(conn.open_table(whole).cdf(starting_version=version), version)
    assert expected, "the DML changed nothing"

    _streamed(monkeypatch, _kernel(conn), tmp_path / "spill")
    _dml(conn.open_table(streamed), op)
    after = conn.open_table(streamed)
    assert after.version == version
    # The kernel's TableChanges and the log reader both read the CDC files.
    assert _feed(after.cdf(starting_version=version), version) == expected
    assert _feed(_log_feed(conn, streamed, version), version) == expected
    assert sorted(tuple(r.values()) for r in after.to_arrow().to_pylist()) == sorted(
        tuple(r.values()) for r in conn.open_table(whole).to_arrow().to_pylist()
    )


def test_a_bucketed_merge_spills_its_changes_and_cleans_up(
    conn: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The change rows of every bucket go through a spill file into the CDC
    files, and the spill is removed once the MERGE is done."""
    import json

    from deltaswamp.engine import kernel_merge

    spilled: list[str] = []
    real = kernel_merge.KernelMerger._run_bucketed

    def run_bucketed(self: Any, *args: Any, **kwargs: Any) -> Any:
        out = real(self, *args, **kwargs)
        spilled.append(type(out[3]).__name__)
        return out

    monkeypatch.setattr(kernel_merge.KernelMerger, "_run_bucketed", run_bucketed)
    path = _path_table(conn, str(tmp_path / "t"), "copy_on_write")
    version = conn.open_table(path).version + 1
    _streamed(monkeypatch, _kernel(conn), tmp_path / "spill")
    _merge(conn.open_table(path))
    assert spilled == ["Spill"]
    log = pathlib.Path(path) / "_delta_log" / f"{version:020d}.json"
    cdc = [json.loads(line)["cdc"] for line in log.read_text().splitlines() if '"cdc"' in line]
    assert cdc and all(c["path"].startswith("_change_data/") for c in cdc)
    assert not list((tmp_path / "spill").iterdir()), "spill files left behind"


# ------------------------------------------------------------ catalog-managed

NAME = "main.sales.cm"


def _router() -> Any:
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router

    return Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()})


@pytest.fixture
def managed(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    from deltaswamp.catalog.ossuc import OSSUnityCatalog
    from deltaswamp.connection import Connection

    for key in list(os.environ):
        if key.startswith("DATABRICKS"):
            monkeypatch.delenv(key, raising=False)
    server = StrictUnityCatalog(staging_root=tmp_path / "managed")
    with server:
        conn = Connection(catalog=OSSUnityCatalog(server.url), router=_router())
        conn.create_catalog("main")
        conn.create_schema("main.sales")
        yield conn


@pytest.mark.parametrize("op", ["merge", "update"])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_streamed_dml_on_a_catalog_managed_feed_table(
    conn: Any,
    managed: Any,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
    op: str,
) -> None:
    # The same DML in one piece on a path table says what to expect.
    control = _path_table(conn, str(tmp_path / "control"), shape)
    version = conn.open_table(control).version + 1
    _dml(conn.open_table(control), op)
    expected = _feed(conn.open_table(control).cdf(starting_version=version), version)

    managed.create_table(NAME, SCHEMA, properties={**BASE, **SHAPES[shape]})
    for batch in _batches():
        managed.table(NAME).append(batch)
    _streamed(monkeypatch, _kernel(managed), tmp_path / "spill")
    _dml(managed.table(NAME), op)
    after = managed.table(NAME)
    assert after.is_catalog_managed
    assert _feed(after.cdf(starting_version=after.version), after.version) == expected
