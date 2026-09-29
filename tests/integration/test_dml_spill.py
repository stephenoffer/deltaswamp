"""Kernel DML that does not hold its rows: a MERGE evaluated in buckets over
local spill files, a streamed source, UPDATEs transformed a chunk at a time,
and copy-on-write rewrites read back a few files at a time.

Each bucketed result is checked against the same MERGE run in one piece,
which is what the kernel did before and still does for a small MERGE.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import pytest
from deltaswamp.capability import Engine

pa = pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")
native = pytest.importorskip("deltaswamp._native")
pytestmark = pytest.mark.skipif(
    "dml_stream" not in native.FEATURES, reason="native build predates streamed DML"
)

SHAPES = {
    "deletion-vectors": {"delta.enableDeletionVectors": "true", "delta.enableRowTracking": "true"},
    "copy-on-write-row-tracked": {
        "delta.enableDeletionVectors": "false",
        "delta.enableRowTracking": "true",
    },
    "copy-on-write": {
        "delta.enableDeletionVectors": "false",
        "delta.enableInCommitTimestamps": "true",
    },
}


def _kernel(conn: Any) -> Any:
    return conn.router.engines[Engine.KERNEL]


def _table(conn: Any, path: str, properties: dict[str, str]) -> str:
    conn.create_table(
        path,
        pa.schema([("id", pa.int64()), ("name", pa.string()), ("qty", pa.int64())]),
        properties=properties,
    )
    # Several files, so buckets and rewrites span more than one.
    for start in range(0, 400, 100):
        ids = list(range(start, start + 100))
        conn.open_table(path).append(
            pa.table(
                {
                    "id": pa.array(ids, pa.int64()),
                    "name": [f"n{i}" for i in ids],
                    "qty": pa.array([i % 7 for i in ids], pa.int64()),
                }
            )
        )
    return path


def _rows(conn: Any, path: str) -> list[tuple[Any, ...]]:
    return sorted(tuple(r.values()) for r in conn.open_table(path).to_arrow().to_pylist())


def _row_ids(conn: Any, path: str) -> dict[int, int]:
    snapshot = _kernel(conn).snapshot(conn.open_table(path).resolved)
    table = pa.table(snapshot.scan(row_positions=True, row_ids=True))
    return dict(
        zip(
            table.column("id").to_pylist(),
            table.column("__deltaswamp_row_id").to_pylist(),
            strict=True,
        )
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


def _merge(table: Any, source: Any, by_source: bool = True) -> Any:
    merger = (
        table.merge(source, "t.id = s.id", source_alias="s", target_alias="t")
        .when_matched_delete(predicate="s.qty = 0")
        .when_matched_update({"name": "s.name", "qty": "t.qty + s.qty"})
        .when_not_matched_insert_all(predicate="s.qty <> 3")
    )
    if by_source:
        merger = merger.when_not_matched_by_source_update(
            {"qty": "t.qty * 10"}, predicate="t.id < 20"
        )
    return merger.execute()


def _bucketed(monkeypatch: pytest.MonkeyPatch, conn: Any, spill: Any, buckets: int = 7) -> None:
    kernel = _kernel(conn)
    monkeypatch.setattr(kernel, "dml_bucket_bytes", 1)
    monkeypatch.setattr(kernel, "dml_max_buckets", buckets)
    monkeypatch.setattr(kernel, "dml_spill_directory", str(spill))


@pytest.mark.parametrize("shape", list(SHAPES))
def test_a_bucketed_merge_is_the_merge_in_one_piece(
    conn: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    whole = _table(conn, str(tmp_path / "whole"), SHAPES[shape])
    bucketed = str(tmp_path / "bucketed")
    shutil.copytree(whole, bucketed)
    assert conn.open_table(whole).can("merge").engine is Engine.KERNEL
    tracked = "delta.enableRowTracking" in SHAPES[shape]
    before = _row_ids(conn, bucketed) if tracked else {}

    expected = _merge(conn.open_table(whole), _source())
    _bucketed(monkeypatch, conn, tmp_path / "spill")
    got = _merge(conn.open_table(bucketed), _source())

    assert _rows(conn, bucketed) == _rows(conn, whole)
    for key in (
        "num_target_rows_updated",
        "num_target_rows_deleted",
        "num_target_rows_inserted",
        "num_source_rows",
    ):
        assert got[key] == expected[key], key
    if tracked:
        # Updated rows keep their ids; inserted ones get new, distinct ones.
        after = _row_ids(conn, bucketed)
        for row in range(150, 250):
            if row in after:
                assert after[row] == before[row]
        assert len(set(after.values())) == len(after)
    # The spill is gone once the MERGE is.
    assert os.listdir(tmp_path / "spill") == []


def test_a_streamed_source_is_spilled_not_held(
    conn: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = _table(conn, str(tmp_path / "whole"), SHAPES["deletion-vectors"])
    streamed = str(tmp_path / "streamed")
    shutil.copytree(whole, streamed)
    _merge(conn.open_table(whole), _source())
    _bucketed(monkeypatch, conn, tmp_path / "spill", buckets=3)
    reader = pa.RecordBatchReader.from_batches(
        _source().schema, _source().to_batches(max_chunksize=17)
    )
    result = _merge(conn.open_table(streamed), reader)
    assert _rows(conn, streamed) == _rows(conn, whole)
    assert result["num_source_rows"] == _source().num_rows


def test_a_streamed_source_in_one_bucket(conn: Any, tmp_path: Any) -> None:
    path = _table(conn, str(tmp_path / "t"), SHAPES["copy-on-write"])
    copy = str(tmp_path / "copy")
    shutil.copytree(path, copy)
    _merge(conn.open_table(copy), _source())
    reader = pa.RecordBatchReader.from_batches(
        _source().schema, _source().to_batches(max_chunksize=9)
    )
    _merge(conn.open_table(path), reader)
    assert _rows(conn, path) == _rows(conn, copy)


def test_duplicate_matches_are_caught_in_a_bucket(
    conn: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _table(conn, str(tmp_path / "t"), SHAPES["deletion-vectors"])
    _bucketed(monkeypatch, conn, tmp_path / "spill")
    source = pa.concat_tables([_source(), _source().slice(0, 1)])
    with pytest.raises(Exception, match="more than one source row"):
        _merge(conn.open_table(path), source)
    assert conn.open_table(path).version == 4


def test_string_and_composite_keys_bucket_alike(
    conn: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = _table(conn, str(tmp_path / "whole"), SHAPES["deletion-vectors"])
    bucketed = str(tmp_path / "bucketed")
    shutil.copytree(whole, bucketed)
    # int32 against the int64 column, and a string key.
    source = _source().set_column(0, "id", _source().column("id").cast(pa.int32()))

    def merge(path: str) -> None:
        (
            conn.open_table(path)
            .merge(source, "t.id = s.id AND t.name = s.name", source_alias="s", target_alias="t")
            .when_matched_update({"qty": "s.qty"})
            .when_not_matched_insert_all()
            .execute()
        )

    merge(whole)
    _bucketed(monkeypatch, conn, tmp_path / "spill")
    merge(bucketed)
    assert _rows(conn, bucketed) == _rows(conn, whole)


@pytest.mark.parametrize("shape", ["copy-on-write", "copy-on-write-row-tracked"])
def test_a_rewrite_reads_back_a_few_files_at_a_time(
    conn: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    path = _table(conn, str(tmp_path / "t"), SHAPES[shape])
    kernel = _kernel(conn)
    # One file read back at a time, and every matched row its own chunk.
    monkeypatch.setattr(kernel, "dml_rewrite_files", 1)
    monkeypatch.setattr(kernel, "dml_chunk_bytes", 1)
    table = conn.open_table(path)
    table.update({"qty": "qty + 100"}, predicate="id % 50 = 0")
    table = conn.open_table(path)
    table.delete("id % 40 = 1")
    rows = {r[0]: r for r in _rows(conn, path)}
    assert rows[50][2] == 50 % 7 + 100 and 41 not in rows and len(rows) == 390


def test_no_cap_by_default(conn: Any) -> None:
    assert _kernel(conn).dml_max_bytes is None
