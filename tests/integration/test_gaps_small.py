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


def test_a_row_tracked_tables_checksum_is_counted_from_storage(tmp_path: Any) -> None:
    """Removes on a row-tracked table are written past the kernel's commit, so
    the post-commit snapshot's in-memory checksum would not count them."""
    conn = _only(Engine.KERNEL)
    path = str(tmp_path / "t")
    props = {"delta.enableRowTracking": "true", "delta.enableDeletionVectors": "true"}
    conn.create_table(path, pa.schema([("id", pa.int64())]), properties=props)
    t = conn.table(path)
    t.append(pa.table({"id": [1, 2, 3]}))
    t.append(pa.table({"id": [4]}))
    t.delete("id = 4")
    t.update(new_values={"id": 9}, predicate="id = 1")
    latest = conn.table(path).version
    files, size = _live_totals(conn.table(path))
    crc = _crc(path, latest)
    assert (crc["numFiles"], crc["tableSizeBytes"]) == (files, size)


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


# --------------------------------------------------- #10 incremental reads


def _appended(conn: Any, path: str) -> tuple[Any, int]:
    conn.write_table(path, pa.table({"id": [1, 2]}))
    t = conn.table(path)
    base = t.version
    t.append(pa.table({"id": [3, 4]}))
    t.append(pa.table({"id": [5]}))
    return conn.table(path), base


def test_added_since_reads_the_appended_rows_without_the_feed(conn: Any, tmp_path: Any) -> None:
    """delta-rs#4554, delta-kernel-rs#1177: an incremental read needed CDF on."""
    t, base = _appended(conn, str(tmp_path / "t"))
    cap = t.can("added_since", version=base)
    assert cap.ok and cap.engine is Engine.KERNEL
    assert pa.table(t.added_since(base)).column("id").to_pylist() == [3, 4, 5]
    assert pa.table(t.added_since(base, until=base + 1)).column("id").to_pylist() == [3, 4]
    got = pa.table(t.added_since(base, columns=["id"], predicate="id > 3"))
    assert got.column("id").to_pylist() == [4, 5]
    assert pa.table(t.added_since(t.version)).num_rows == 0
    pinned = conn.table(t.location, version=base + 1)
    assert pa.table(pinned.added_since(base)).column("id").to_pylist() == [3, 4]


def test_added_since_refuses_a_range_that_rewrote_files(conn: Any, tmp_path: Any) -> None:
    from deltaswamp.errors import InvalidArgumentError, UnreachableTableError

    t, base = _appended(conn, str(tmp_path / "t"))
    t.delete("id = 3")
    with pytest.raises(UnreachableTableError, match="removed or rewritten"):
        t.added_since(base)
    assert sorted(pa.table(t.added_since(base, only_appends=True)).column("id").to_pylist()) == [
        4,
        5,
    ]
    with pytest.raises(InvalidArgumentError, match="after the table's version"):
        t.added_since(t.version + 5)
    with pytest.raises(InvalidArgumentError, match="before version"):
        t.added_since(3, until=2)


def test_added_since_needs_the_kernel(tmp_path: Any) -> None:
    conn = _only(Engine.DELTARS)
    t, base = _appended(conn, str(tmp_path / "t"))
    cap = t.can("added_since", version=base)
    assert not cap.ok and "incremental scan" in (cap.reason or "")


def test_changes_can_start_from_a_snapshot(conn: Any, tmp_path: Any) -> None:
    """The feed was on only from version N: a consumer bootstraps from the table at N."""
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": [1, 2]}))
    t = conn.table(path)
    t.set_properties({"delta.enableChangeDataFeed": "true"})
    start = conn.table(path).version
    t.append(pa.table({"id": [3]}))
    t.delete("id = 1")
    got = list(conn.table(path).changes(start, include_snapshot=True))
    assert [v for v, _ in got] == [start, start + 1, start + 2]
    snap = got[0][1]
    assert sorted(snap.column("id").to_pylist()) == [1, 2]
    assert set(snap.column("_change_type").to_pylist()) == {"insert"}
    assert set(snap.column("_commit_version").to_pylist()) == {start}
    assert snap.schema.names == got[1][1].schema.names
    assert (
        snap.schema.field("_commit_timestamp").type
        == got[1][1].schema.field("_commit_timestamp").type
    )
    assert got[2][1].column("_change_type").to_pylist() == ["delete"]
    projected = next(iter(conn.table(path).changes(start, include_snapshot=True, columns=["id"])))
    assert projected[1].column_names == ["id"]


# ------------------------------------------------ #13 DML from a pinned read

_DV = {"delta.enableDeletionVectors": "true"}


def _one_file_per_row(conn: Any, path: str, ids: list[int]) -> None:
    conn.write_table(path, pa.table({"id": [ids[0]], "v": [0]}), properties=_DV)
    for i in ids[1:]:
        conn.table(path).append(pa.table({"id": [i], "v": [0]}))


def _rows(conn: Any, path: str) -> list[tuple[int, int]]:
    return sorted((r["id"], r["v"]) for r in conn.table(path).to_arrow().to_pylist())


def test_dml_from_a_pinned_handle_commits_over_blind_appends(conn: Any, tmp_path: Any) -> None:
    """delta-rs#4417: a pinned handle refused every write."""
    path = str(tmp_path / "t")
    _one_file_per_row(conn, path, [1, 2, 3])
    pinned = conn.table(path, version=conn.table(path).version)
    conn.table(path).append(pa.table({"id": [4], "v": [0]}))  # a blind append since
    for op, args in (("delete", {"predicate": "id = 1"}), ("merge", {})):
        cap = pinned.can(op, **args)
        assert cap.ok and cap.engine is Engine.KERNEL, (op, cap)
    assert pinned.delete("id = 1")["num_deleted_rows"] == 1
    assert pinned.update(new_values={"v": 9}, predicate="id = 2")["num_updated_rows"] == 1
    merged = (
        pinned.merge(pa.table({"id": [3, 7], "v": [5, 5]}), "target.id = source.id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute()
    )
    assert (merged["num_updated_rows"], merged["num_inserted_rows"]) == (1, 1)
    # The row appended since was not read, so it is left alone.
    assert _rows(conn, path) == [(2, 9), (3, 5), (4, 0), (7, 5)]


def test_dml_from_a_pinned_handle_conflicts_with_what_changed_since(
    conn: Any, tmp_path: Any
) -> None:
    from deltaswamp.errors import CommitConflictError

    path = str(tmp_path / "t")
    _one_file_per_row(conn, path, [1, 2])
    pinned = conn.table(path, version=conn.table(path).version)
    conn.table(path).delete("id = 2")  # changes the file the pinned read matches
    with pytest.raises(CommitConflictError, match="pinned to version"):
        pinned.update(new_values={"v": 1}, predicate="id = 2")
    assert _rows(conn, path) == [(1, 0)]
    # A later non-blind write that added a row the read would have matched.
    other = str(tmp_path / "u")
    _one_file_per_row(conn, other, [1, 2])
    pinned = conn.table(other, version=conn.table(other).version)
    conn.table(other).merge(
        pa.table({"id": [5], "v": [0]}), "target.id = source.id"
    ).when_not_matched_insert_all().execute()
    with pytest.raises(CommitConflictError, match="could have matched"):
        pinned.delete("id >= 1")
    assert _rows(conn, other) == [(1, 0), (2, 0), (5, 0)]


def test_a_pinned_append_appends_at_the_latest_version(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": [1]}))
    conn.table(path).add_column(pa.field("s", pa.string()))
    pinned = conn.table(path, version=0)
    assert pinned.can("append").ok
    pinned.append(pa.table({"id": [2]}))
    assert conn.table(path).to_arrow().sort_by("id").to_pylist() == [
        {"id": 1, "s": None},
        {"id": 2, "s": None},
    ]


def test_pinned_dml_is_still_refused_where_it_cannot_be_checked(conn: Any, tmp_path: Any) -> None:
    from deltaswamp.errors import InvalidArgumentError

    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": [1, 2]}))  # no deletion vectors
    conn.table(path).append(pa.table({"id": [3]}))
    pinned = conn.table(path, version=0)
    cap = pinned.can("delete", predicate="id = 1")
    assert not cap.ok and "deletion vectors" in (cap.reason or "")
    with pytest.raises(InvalidArgumentError, match="deletion vectors"):
        pinned.delete("id = 1")
    for op in ("overwrite", "optimize"):
        assert not pinned.can(op).ok
    assert conn.table(path).count() == 3
