"""Small feature gaps against delta-rs and delta-kernel-rs, filled.

#8 version checksums after every commit, #10 incremental reads without the
change feed, #12 shallow and deep clones of path tables, #13 DML from a
handle pinned to a version, #15 a result from append and overwrite.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

from deltaswamp.capability import Engine  # noqa: E402


def _only(kind: Engine) -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router

    engine = KernelEngine() if kind is Engine.KERNEL else DeltaRsEngine()
    return Connection(catalog=FilesystemCatalog(), router=Router(engines={kind: engine}))


def _log(path: str) -> list[str]:
    return sorted(os.listdir(os.path.join(path, "_delta_log")))


def _crc(path: str, version: int) -> dict[str, Any]:
    with open(os.path.join(path, "_delta_log", f"{version:020}.crc")) as f:
        crc: dict[str, Any] = json.load(f)
    return crc


def _live_totals(t: Any) -> tuple[int, int]:
    from deltaswamp._native import Snapshot

    files = pa.table(Snapshot.resolve(t.location).files())
    return files.num_rows, sum(files.column("size").to_pylist())


# ------------------------------------------------------------ #8 checksums


def test_every_commit_writes_a_checksum_on_either_engine(conn: Any, tmp_path: Any) -> None:
    """delta-rs#4190: only the kernel's CREATE wrote one; Databricks keeps one per version."""
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"a": [1, 2]}))
    t = conn.table(path)
    t.append(pa.table({"a": [3]}))  # delta-rs
    t.delete("a = 1")
    t.merge(pa.table({"a": [3, 9]}), "target.a = source.a").when_not_matched_insert_all().execute()
    t.set_properties({"x.y": "1"})  # a raw kernel commit, no files changed
    t.add_column(pa.field("b", pa.string()))
    t.append(pa.table({"a": [5], "b": ["x"]}))
    kernel = _only(Engine.KERNEL).table(path)
    kernel.append(pa.table({"a": [6], "b": ["y"]}))  # a kernel commit
    latest = conn.table(path).version
    for v in range(latest + 1):
        assert f"{v:020}.crc" in _log(path), v
    crc = _crc(path, latest)
    files, size = _live_totals(conn.table(path))
    assert (crc["numFiles"], crc["tableSizeBytes"]) == (files, size)
    assert crc["metadata"]["configuration"] == {"x.y": "1"}
    assert '"b"' in crc["metadata"]["schemaString"]


def test_a_failing_checksum_never_fails_the_write(
    conn: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    from deltaswamp import _native

    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"a": [1]}))

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("storage is down")

    monkeypatch.setattr(_native.Snapshot, "write_checksum", broken, raising=False)
    conn.table(path).append(pa.table({"a": [2]}))
    t = conn.table(path)
    assert t.count() == 2
    assert f"{t.version:020}.crc" not in _log(path)


def test_a_checksum_carried_over_a_metadata_commit_keeps_the_file_stats(
    conn: Any, tmp_path: Any
) -> None:
    """The kernel gives up on SET TBLPROPERTIES; the chain went on unbroken."""
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"a": [1, 2]}))
    t = conn.table(path)
    t.set_properties({"k.v": "1"})
    v = conn.table(path).version
    before, after = _crc(path, v - 1), _crc(path, v)
    assert after["numFiles"] == before["numFiles"]
    assert after["tableSizeBytes"] == before["tableSizeBytes"]
    assert after["metadata"]["configuration"] == {"k.v": "1"}
    t.append(pa.table({"a": [3]}))
    files, size = _live_totals(conn.table(path))
    assert (_crc(path, v + 1)["numFiles"], _crc(path, v + 1)["tableSizeBytes"]) == (files, size)


# ------------------------------------------------------- #15 write results


@pytest.mark.parametrize("kind", [Engine.DELTARS, Engine.KERNEL])
def test_append_and_overwrite_return_what_they_committed(kind: Engine, tmp_path: Any) -> None:
    """delta-rs#3952: both returned None, so a pipeline could not log what it wrote."""
    import deltaswamp as ds

    conn = _only(kind)
    path = str(tmp_path / "t")
    ds.connect().write_table(path, pa.table({"a": [1, 2]}))
    t = conn.table(path)
    appended = t.append(pa.table({"a": [3, 4, 5]}))
    assert isinstance(appended, ds.OperationResult)
    assert appended.engine == kind.value
    assert appended["version"] == conn.table(path).version
    assert (appended["num_files"], appended["num_rows"]) == (1, 3)
    assert appended["num_bytes"] > 0
    replaced = t.overwrite(pa.table({"a": [7]}))
    assert replaced["version"] == appended["version"] + 1
    assert (replaced["num_files"], replaced["num_rows"], replaced["num_removed_files"]) == (1, 1, 2)


def test_a_skipped_write_says_so(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"a": [1]}))
    t = conn.table(path)
    first = t.append(pa.table({"a": [2]}), txn=("job", 1))
    again = t.append(pa.table({"a": [2]}), txn=("job", 1))
    assert first["num_rows"] == 1 and "skipped" not in first
    assert again["skipped"] and again["num_rows"] == 0 and "version" not in again
