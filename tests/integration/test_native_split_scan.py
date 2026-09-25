"""File-restricted scans: `Snapshot.scan(files=[...])`.

A distributed reader plans splits from `Snapshot.files()` and hands each worker
a subset of paths. That is only correct if each restricted scan has exactly the
full scan's semantics -- deletion vectors applied, column mapping and partition
values resolved, row order within a file preserved -- so the invariant tested
throughout is: the union of per-file scans equals the full scan.

Deletion vectors: deltalake cannot write them, so the DV table here is built
by committing a hand-encoded *inline* DV (Z85 over a portable RoaringTreemap)
against a kernel-created table with `delta.enableDeletionVectors`. The expected
survivors are known independently, so this also checks the DV is really
applied rather than just matching the full scan.
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from typing import Any

import deltaswamp as ds
import pytest

pytestmark = pytest.mark.skipif(
    not ds.has_native(), reason="native extension not built (run `maturin develop`)"
)

pa = pytest.importorskip("pyarrow")
deltalake = pytest.importorskip("deltalake")

from tests import helpers  # noqa: E402


def _read(snapshot: Any, **kwargs: Any) -> Any:
    return pa.RecordBatchReader.from_stream(snapshot.scan(**kwargs)).read_all()


def _rows(table: Any) -> Counter[tuple[Any, ...]]:
    """Rows as a multiset, so duplicates and order are both accounted for."""
    names = table.schema.names
    return Counter(tuple(row[n] for n in names) for row in table.to_pylist())


def _files(snapshot: Any, **kwargs: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = pa.table(snapshot.files(**kwargs)).to_pylist()
    return rows


def _append(path: str, table: Any) -> None:
    snapshot = helpers.snapshot(path)
    snapshot.append(pa.RecordBatchReader.from_batches(table.schema, table.to_batches()))


def _assert_split_equals_full(path: str, **scan_kwargs: Any) -> list[dict[str, Any]]:
    """Per-file scans union to the full scan; each keeps the full scan's schema."""
    snapshot = helpers.snapshot(path)
    full = _read(snapshot, **scan_kwargs)
    files = _files(snapshot)
    assert len(files) > 1, "test table should span several files"

    union: Counter[tuple[Any, ...]] = Counter()
    for f in files:
        part = _read(snapshot, files=[f["path"]], **scan_kwargs)
        assert part.schema == full.schema
        union += _rows(part)
    assert union == _rows(full)
    return files


# --------------------------------------------------------------------------
# Inline deletion vectors, encoded by hand.
# --------------------------------------------------------------------------

_Z85 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"


def _z85(data: bytes) -> str:
    assert len(data) % 4 == 0
    out = []
    for i in range(0, len(data), 4):
        value = int.from_bytes(data[i : i + 4], "big")
        chunk = []
        for _ in range(5):
            chunk.append(_Z85[value % 85])
            value //= 85
        out.append("".join(reversed(chunk)))
    return "".join(out)


def _inline_dv(deleted: list[int]) -> dict[str, Any]:
    """A portable RoaringTreemap with one array container, Z85-encoded.

    Layout: magic u32 | bitmap count u64 | high key u32 | cookie 12346 u32 |
    container count u32 | (key u16, cardinality-1 u16) | offset u32 | u16 values.
    """
    rows = sorted(set(deleted))
    assert rows and rows[-1] < 65536 and len(rows) <= 4096
    if len(rows) % 2:  # Z85 needs a multiple of 4 bytes; pad with a real deletion.
        raise ValueError("pass an even number of deleted rows")
    bitmap = (
        struct.pack("<II", 12346, 1)
        + struct.pack("<HH", 0, len(rows) - 1)
        + struct.pack("<I", 16)
        + b"".join(struct.pack("<H", r) for r in rows)
    )
    payload = struct.pack("<IQI", 1681511377, 1, 0) + bitmap
    return {
        "storageType": "i",
        "pathOrInlineDv": _z85(payload),
        "sizeInBytes": len(payload),
        "cardinality": len(rows),
    }


def _attach_dv(path: str, file: dict[str, Any], deleted: list[int]) -> None:
    snapshot = helpers.snapshot(path)
    common = {"path": file["path"], "partitionValues": {}, "size": file["size"]}
    remove = {**common, "deletionTimestamp": 1, "dataChange": True, "extendedFileMetadata": True}
    add = {
        **common,
        "modificationTime": file["modification_time"],
        "dataChange": True,
        "stats": file["stats"],
        "deletionVector": _inline_dv(deleted),
    }
    helpers.native().commit_raw(
        path, snapshot.version + 1, [json.dumps({"remove": remove}), json.dumps({"add": add})]
    )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def multi_file(tmp_path: Any) -> str:
    from deltalake import write_deltalake

    path = str(tmp_path / "multi")
    write_deltalake(path, pa.table({"id": [1, 2, 3], "g": ["a", "b", "a"]}))
    write_deltalake(path, pa.table({"id": [10, 11], "g": ["c", "c"]}), mode="append")
    write_deltalake(
        path, pa.table({"id": [20, 21, 22, 23], "g": ["d", "d", "e", None]}), mode="append"
    )
    # A duplicate of an earlier row: the multiset comparison must count it twice.
    write_deltalake(path, pa.table({"id": [1], "g": ["a"]}), mode="append")
    return path


@pytest.fixture
def partitioned(tmp_path: Any) -> str:
    from deltalake import write_deltalake

    path = str(tmp_path / "part")
    # Values that need URL-encoding in the stored path ("a b", "x/y", "%").
    data = pa.table(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "region": ["eu", "us", "a b", "x/y", "100%", "eu"],
        }
    )
    write_deltalake(path, data, partition_by=["region"])
    write_deltalake(
        path,
        pa.table({"id": [7, 8], "region": ["eu", "a b"]}),
        partition_by=["region"],
        mode="append",
    )
    return path


@pytest.fixture
def column_mapped(tmp_path: Any) -> str:
    path = str(tmp_path / "cm")
    schema = pa.schema([("id", pa.int64()), ("Odd Name", pa.string()), ("p", pa.string())])
    helpers.native().create_table(
        path,
        schema,
        properties={"delta.columnMapping.mode": "name"},
        partition_by=["p"],
    )
    for k in range(3):
        ids = list(range(k * 10, k * 10 + 5))
        _append(
            path,
            pa.table(
                {"id": ids, "Odd Name": [f"v{i}" for i in ids], "p": [f"p{k}"] * len(ids)},
                schema=schema,
            ),
        )
    return path


@pytest.fixture
def with_dvs(tmp_path: Any) -> tuple[str, set[int]]:
    """Three files; the 5000-row one gets a DV spanning several read batches."""
    path = str(tmp_path / "dv")
    schema = pa.schema([("id", pa.int64()), ("s", pa.string())])
    helpers.native().create_table(path, schema, properties={"delta.enableDeletionVectors": "true"})
    for start, count in ((0, 5000), (10_000, 10), (20_000, 10)):
        ids = list(range(start, start + count))
        _append(path, pa.table({"id": ids, "s": [f"r{i}" for i in ids]}, schema=schema))

    files = _files(helpers.snapshot(path))
    big = next(f for f in files if f["num_records"] == 5000)
    small = next(f for f in files if f["num_records"] == 10)
    # Row positions == ids for the big file, whose rows were written 0..4999.
    big_deleted = [0, 1023, 1024, 2047, 2048, 3000, 4998, 4999]
    _attach_dv(path, big, big_deleted)
    small_ids = json.loads(small["stats"])["minValues"]["id"]
    _attach_dv(path, small, [2, 7])
    deleted = set(big_deleted) | {small_ids + 2, small_ids + 7}
    return path, deleted


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_feature_is_advertised() -> None:
    assert "file_restricted_scan" in helpers.native().FEATURES


def test_split_union_equals_full_scan(multi_file: str) -> None:
    files = _assert_split_equals_full(multi_file)
    snapshot = helpers.snapshot(multi_file)
    for f in files:
        assert _read(snapshot, files=[f["path"]]).num_rows == f["num_records"]


def test_split_union_with_projection(multi_file: str) -> None:
    _assert_split_equals_full(multi_file, columns=["g"])


def test_restricting_to_every_file_is_the_full_scan(multi_file: str) -> None:
    snapshot = helpers.snapshot(multi_file)
    paths = [f["path"] for f in _files(snapshot)]
    assert _read(snapshot, files=paths) == _read(snapshot)
    # Duplicates in the request do not read a file twice.
    assert _rows(_read(snapshot, files=paths + paths)) == _rows(_read(snapshot))


def test_partitioned_values_are_materialised(partitioned: str) -> None:
    files = _assert_split_equals_full(partitioned)
    snapshot = helpers.snapshot(partitioned)
    assert any("%" in f["path"] for f in files), "expected URL-encoded partition paths"
    for f in files:
        part = _read(snapshot, files=[f["path"]])
        assert part.num_rows == f["num_records"]
        # Every row of a single file carries that file's partition value.
        assert set(part.column("region").to_pylist()) == set(dict(f["partition_values"]).values())


def test_column_mapping_resolves_logical_names(column_mapped: str) -> None:
    snapshot = helpers.snapshot(column_mapped)
    assert snapshot.table_properties()["delta.columnMapping.mode"] == "name"
    files = _assert_split_equals_full(column_mapped)
    for f in files:
        part = _read(snapshot, files=[f["path"]])
        assert part.schema.names == ["id", "Odd Name", "p"]
        assert part.num_rows == f["num_records"]
        ids = part.column("id").to_pylist()
        assert part.column("Odd Name").to_pylist() == [f"v{i}" for i in ids]
        assert len(set(part.column("p").to_pylist())) == 1


def test_deletion_vectors_applied_per_file(with_dvs: tuple[str, set[int]]) -> None:
    path, deleted = with_dvs
    snapshot = helpers.snapshot(path)
    files = _assert_split_equals_full(path)
    assert sum(f["deletion_vector"] is not None for f in files) == 2

    everything = set(range(5000)) | set(range(10_000, 10_010)) | set(range(20_000, 20_010))
    full_ids = _read(snapshot).column("id").to_pylist()
    assert sorted(full_ids) == sorted(everything - deleted)

    for f in files:
        part = _read(snapshot, files=[f["path"]])
        ids = part.column("id").to_pylist()
        # Written ascending, so surviving rows must still be ascending: the DV
        # mask was applied in physical order and nothing reordered the rows.
        assert ids == sorted(ids)
        dv = f["deletion_vector"]
        cardinality = json.loads(dv)["cardinality"] if dv else 0
        assert len(ids) == f["num_records"] - cardinality
        assert not set(ids) & deleted
        assert part.column("s").to_pylist() == [f"r{i}" for i in ids]


def test_predicate_composes_with_files(multi_file: str) -> None:
    snapshot = helpers.snapshot(multi_file)
    predicate = json.dumps(
        {"op": "ge", "args": [{"column": ["id"]}, {"literal": 10, "type": "long"}]}
    )
    kept = {f["path"] for f in _files(snapshot, predicate=predicate)}
    all_paths = [f["path"] for f in _files(snapshot)]
    skipped = [p for p in all_paths if p not in kept]
    assert kept and skipped

    # A file the predicate skips stays skipped even when requested.
    for p in skipped:
        assert _read(snapshot, files=[p], predicate=predicate).num_rows == 0
    # Requested and not skipped: read in full (predicate only prunes files).
    union: Counter[tuple[Any, ...]] = Counter()
    for p in kept:
        union += _rows(_read(snapshot, files=[p], predicate=predicate))
    assert union == _rows(_read(snapshot, predicate=predicate))
    assert _rows(_read(snapshot, files=all_paths, predicate=predicate)) == _rows(
        _read(snapshot, predicate=predicate)
    )


def test_empty_list_gives_empty_stream_with_schema(column_mapped: str) -> None:
    snapshot = helpers.snapshot(column_mapped)
    for columns in (None, ["Odd Name"]):
        full = _read(snapshot, columns=columns)
        empty = _read(snapshot, files=[], columns=columns)
        assert empty.num_rows == 0
        assert empty.schema == full.schema


def test_unknown_paths_are_ignored(multi_file: str) -> None:
    snapshot = helpers.snapshot(multi_file)
    first = _files(snapshot)[0]
    assert _read(snapshot, files=["does-not-exist.parquet"]).num_rows == 0
    got = _read(snapshot, files=["does-not-exist.parquet", first["path"]])
    assert got.num_rows == first["num_records"]
    # Paths are matched exactly as stored, not resolved: an absolute URL for
    # the same file is a different string and selects nothing.
    absolute = snapshot.table_root + first["path"]
    assert _read(snapshot, files=[absolute]).num_rows == 0
