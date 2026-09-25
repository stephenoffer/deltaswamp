"""Regressions for defects found auditing the native (Rust) extension."""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

n = pytest.importorskip("deltaswamp._native")


def _read(snapshot: Any, **kwargs: Any) -> pa.Table:
    return pa.RecordBatchReader.from_stream(snapshot.scan(**kwargs)).read_all()


def _rows(path: str | Path) -> list[dict[str, Any]]:
    got = _read(n.Snapshot.resolve(str(path)))
    return sorted(got.to_pylist(), key=lambda r: json.dumps(r, default=str, sort_keys=True))


def _table(tmp_path: Path, schema: pa.Schema, **kwargs: Any) -> str:
    path = str(tmp_path / f"t{uuid.uuid4().hex[:6]}")
    n.create_table(path, schema, **kwargs)
    return path


def _append(path: str, data: pa.Table, **kwargs: Any) -> int:
    return int(n.Snapshot.resolve(path).append(data, **kwargs))


def _newest_parquet(path: str) -> str:
    return max(
        glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True), key=os.path.getmtime
    )


AB = pa.schema([("a", pa.int64()), ("b", pa.int64())])


# ------------------------------------------------------------ column binding


class TestColumnsBindByName:
    def test_reordered_columns_are_not_swapped(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        _append(path, pa.table({"b": [1], "a": [2]}))
        assert _rows(path) == [{"a": 2, "b": 1}]

    def test_missing_leading_column_does_not_shift_the_rest(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        _append(path, pa.table({"b": [9]}))
        assert _rows(path) == [{"a": None, "b": 9}]

    def test_case_insensitive_names(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        _append(path, pa.table({"B": [1], "A": [2]}))
        assert _rows(path) == [{"a": 2, "b": 1}]

    def test_extra_column_is_a_clear_error(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        with pytest.raises(ValueError, match="not in the table schema"):
            _append(path, pa.table({"a": [1], "b": [2], "zz": [3]}))

    def test_missing_not_null_column_is_refused(self, tmp_path: Path) -> None:
        path = _table(tmp_path, pa.schema([pa.field("a", pa.int64(), False), ("b", pa.int64())]))
        with pytest.raises(ValueError, match="NOT NULL"):
            _append(path, pa.table({"b": [2]}))

    def test_struct_fields_are_not_swapped(self, tmp_path: Path) -> None:
        inner = pa.struct([("a", pa.int64()), ("b", pa.int64())])
        path = _table(tmp_path, pa.schema([("s", inner)]))
        swapped = pa.struct([("b", pa.int64()), ("a", pa.int64())])
        _append(path, pa.table({"s": pa.array([{"b": 1, "a": 2}], swapped)}))
        assert _rows(path) == [{"s": {"a": 2, "b": 1}}]

    def test_missing_partition_column_is_refused_not_nulled(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB, partition_by=["b"])
        with pytest.raises(ValueError, match="no such column"):
            _append(path, pa.table({"a": [1]}))


class TestPhysicalTypes:
    def test_int32_data_is_written_as_the_tables_long(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        _append(path, pa.table({"a": pa.array([1], pa.int32()), "b": pa.array([2], pa.int32())}))
        assert pq.read_schema(_newest_parquet(path)).field("a").type == pa.int64()

    def test_ms_and_naive_timestamps_are_written_as_utc_micros(self, tmp_path: Path) -> None:
        path = _table(tmp_path, pa.schema([("t", pa.timestamp("us", "UTC"))]))
        for t in (pa.timestamp("ms", "UTC"), pa.timestamp("us")):
            _append(path, pa.table({"t": pa.array([1_000], t)}))
            assert pq.read_schema(_newest_parquet(path)).field("t").type == pa.timestamp(
                "us", "UTC"
            )

    def test_sub_microsecond_nanos_are_refused_not_truncated(self, tmp_path: Path) -> None:
        path = _table(tmp_path, pa.schema([("t", pa.timestamp("us", "UTC"))]))
        with pytest.raises(ValueError, match="change some values"):
            _append(path, pa.table({"t": pa.array([1_000_000_001], pa.timestamp("ns", "UTC"))}))

    def test_dictionary_strings_are_accepted(self, tmp_path: Path) -> None:
        path = _table(tmp_path, pa.schema([("s", pa.string())]))
        _append(path, pa.table({"s": pa.array(["x", "y"]).dictionary_encode()}))
        assert sorted(r["s"] for r in _rows(path)) == ["x", "y"]

    def test_unsigned_columns_are_widened_at_create(self, tmp_path: Path) -> None:
        path = _table(tmp_path, pa.schema([("u8", pa.uint8()), ("u32", pa.uint32())]))
        schema = pa.schema(n.Snapshot.resolve(path).schema())
        assert schema.field("u8").type == pa.int16()
        assert schema.field("u32").type == pa.int64()
        _append(
            path,
            pa.table(
                {"u8": pa.array([255], pa.uint8()), "u32": pa.array([2**32 - 1], pa.uint32())}
            ),
        )
        assert _rows(path) == [{"u8": 255, "u32": 2**32 - 1}]

    def test_uint64_above_long_is_refused(self, tmp_path: Path) -> None:
        path = _table(tmp_path, pa.schema([("u", pa.uint64())]))
        with pytest.raises(ValueError):
            _append(path, pa.table({"u": pa.array([2**64 - 1], pa.uint64())}))


class TestSmallFiles:
    def test_many_small_batches_become_one_file(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        data = pa.table({"a": list(range(50)), "b": list(range(50))})
        reader = pa.RecordBatchReader.from_batches(data.schema, data.to_batches(max_chunksize=1))
        _append(path, reader)
        files = pa.table(n.Snapshot.resolve(path).files())
        assert files.num_rows == 1
        assert [r["a"] for r in _read(n.Snapshot.resolve(path)).to_pylist()] == list(range(50))

    def test_empty_batches_write_no_files(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        _append(path, pa.table({"a": pa.array([], pa.int64()), "b": pa.array([], pa.int64())}))
        assert pa.table(n.Snapshot.resolve(path).files()).num_rows == 0


# -------------------------------------------------------- partition values


class TestPartitionValues:
    def _int_part(self, tmp_path: Path) -> str:
        return _table(
            tmp_path, pa.schema([("id", pa.int64()), ("p", pa.int32())]), partition_by=["p"]
        )

    def test_overflowing_value_is_refused_not_written_as_null(self, tmp_path: Path) -> None:
        path = self._int_part(tmp_path)
        with pytest.raises(ValueError, match="Int32"):
            _append(path, pa.table({"id": [1], "p": [3_000_000_000]}))
        assert _rows(path) == []

    def test_unparsable_string_is_refused_not_nulled(self, tmp_path: Path) -> None:
        path = self._int_part(tmp_path)
        with pytest.raises(ValueError):
            _append(path, pa.table({"id": [1], "p": ["abc"]}))

    def test_fractional_float_is_refused_not_truncated(self, tmp_path: Path) -> None:
        path = self._int_part(tmp_path)
        with pytest.raises(ValueError, match="change some values"):
            _append(path, pa.table({"id": [1], "p": [1.7]}))
        _append(path, pa.table({"id": [2], "p": [3.0]}))
        assert _rows(path) == [{"id": 2, "p": 3}]


# ------------------------------------------------------------------- scans


class TestScanColumns:
    def test_empty_column_list_raises_instead_of_aborting(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        _append(path, pa.table({"a": [1], "b": [2]}))
        with pytest.raises(ValueError, match=r"columns=\[\]"):
            n.Snapshot.resolve(path).scan(columns=[])

    def test_columns_resolve_case_insensitively(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        _append(path, pa.table({"a": [1], "b": [2]}))
        got = _read(n.Snapshot.resolve(path), columns=["B"])
        assert got.column_names == ["b"]

    def test_unknown_column_names_the_available_ones(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        with pytest.raises(ValueError, match="not in the table schema"):
            n.Snapshot.resolve(path).scan(columns=["nope"])


class TestPredicateSkipping:
    def test_negative_zero_partition_is_not_skipped(self, tmp_path: Path) -> None:
        path = _table(
            tmp_path, pa.schema([("id", pa.int64()), ("f", pa.float64())]), partition_by=["f"]
        )
        _append(path, pa.table({"id": [1], "f": [-0.0]}))
        pred = {"op": "eq", "args": [{"column": ["f"]}, {"literal": 0.0, "type": "double"}]}
        files = pa.table(n.Snapshot.resolve(path).files(predicate=json.dumps(pred)))
        assert files.num_rows == 1

    def test_decimal_literal_against_a_string_column_does_not_skip(self, tmp_path: Path) -> None:
        path = _table(
            tmp_path, pa.schema([("id", pa.int64()), ("s", pa.string())]), partition_by=["s"]
        )
        _append(path, pa.table({"id": [1], "s": ["1.5"]}))
        pred = {"op": "eq", "args": [{"column": ["s"]}, {"literal": "1.50", "type": "decimal"}]}
        files = pa.table(n.Snapshot.resolve(path).files(predicate=json.dumps(pred)))
        assert files.num_rows == 1


# ------------------------------------------------------------ table roots


class TestTableRoots:
    def test_relative_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert n.create_table("rel/t", AB) == 0
        assert n.Snapshot.resolve(str(tmp_path / "rel" / "t")).version == 0

    def test_percent_encoded_file_url_creates_the_decoded_directory(self, tmp_path: Path) -> None:
        url = "file://" + str(tmp_path / "my table").replace(" ", "%20")
        assert n.create_table(url, AB) == 0
        assert (tmp_path / "my table" / "_delta_log").is_dir()
        assert not (tmp_path / "my%20table").exists()

    def test_url_fragment_is_refused_rather_than_writing_elsewhere(self, tmp_path: Path) -> None:
        url = "file://" + str(tmp_path / "a#b")
        with pytest.raises(ValueError, match="fragment"):
            n.commit_raw(url, 0, ['{"commitInfo":{}}'])
        assert not (tmp_path / "a").exists()

    def test_failed_create_leaves_no_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            n.create_table(str(tmp_path / "x"), AB, partition_by=["a"], cluster_by=["a"])
        assert not (tmp_path / "x").exists()

    def test_empty_schema_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one column"):
            n.create_table(str(tmp_path / "e"), pa.schema([]))

    def test_duplicate_partition_column(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="listed twice"):
            n.create_table(str(tmp_path / "d"), AB, partition_by=["a", "A"])


class TestLogTail:
    def test_path_like_staged_commit_name_is_refused(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        name = "../../other/_delta_log/00000000000000000001.json"
        with pytest.raises(ValueError, match="bare staged-commit file name"):
            n.Snapshot.resolve(path, log_tail=[(1, name, 0, 1)], max_catalog_version=1)

    def test_version_disagreeing_with_file_name_is_refused(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        name = f"{2:020}.{uuid.uuid4()}.json"
        with pytest.raises(ValueError, match="different version"):
            n.Snapshot.resolve(path, log_tail=[(1, name, 0, 1)], max_catalog_version=1)


# ----------------------------------------------------------------- commits


class TestCommits:
    def test_reserved_commit_metadata_key_is_refused(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        with pytest.raises(ValueError, match="reserved"):
            _append(path, pa.table({"a": [1], "b": [1]}), commit_metadata={"operation": "X"})

    def test_raw_commit_that_leaves_a_gap_is_refused(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        with pytest.raises(ValueError, match="gap"):
            n.commit_raw(path, 5, ['{"commitInfo":{}}'])
        assert not os.path.exists(os.path.join(path, "_delta_log", f"{5:020}.json"))
        assert n.commit_raw(path, 1, ['{"commitInfo":{}}']) == 1

    def test_app_id_version(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        assert n.Snapshot.resolve(path).app_id_version("job") is None
        _append(path, pa.table({"a": [1], "b": [1]}), txn=("job", 7))
        assert n.Snapshot.resolve(path).app_id_version("job") == 7
        assert "app_id_version" in n.FEATURES


# ------------------------------------------------------------------ errors


class TestErrorClasses:
    def test_missing_data_file_mid_stream_is_an_os_error(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        _append(path, pa.table({"a": [1], "b": [1]}))
        os.remove(_newest_parquet(path))
        with pytest.raises(OSError):
            _read(n.Snapshot.resolve(path))

    def test_nul_byte_in_a_stream_error_does_not_abort(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB)
        add = {
            "add": {
                "path": "a%00b.parquet",
                "partitionValues": {},
                "size": 10,
                "modificationTime": 1,
                "dataChange": True,
            }
        }
        n.commit_raw(path, 1, ['{"commitInfo":{}}', json.dumps(add)])
        with pytest.raises((OSError, pa.ArrowException)):
            _read(n.Snapshot.resolve(path))


@pytest.mark.skipif(
    not hasattr(os, "fork") or sys.platform in ("win32", "darwin"),
    # macOS forbids libdispatch (which Rust thread parking uses) in a forked
    # child of a threaded process, so fork-after-threads cannot work there.
    reason="needs a fork that tolerates a threaded parent (Linux)",
)
def test_forked_child_gets_a_working_runtime(tmp_path: Path) -> None:
    path = _table(tmp_path, AB)
    _append(path, pa.table({"a": [1], "b": [2]}))
    n.Snapshot.resolve(path)  # the parent's runtime now exists
    pid = os.fork()
    if pid == 0:  # child
        try:
            rows = _read(n.Snapshot.resolve(path)).num_rows
            os._exit(0 if rows == 1 else 2)
        except BaseException:
            os._exit(3)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            assert os.waitstatus_to_exitcode(status) == 0
            return
        time.sleep(0.05)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    pytest.fail("the forked child hung on the parent's runtime")


def test_date_partition_round_trip(tmp_path: Path) -> None:
    path = _table(tmp_path, pa.schema([("id", pa.int64()), ("d", pa.date32())]), partition_by=["d"])
    _append(path, pa.table({"id": [1], "d": [dt.date(2024, 1, 2)]}))
    assert _rows(path) == [{"id": 1, "d": dt.date(2024, 1, 2)}]


# ------------------------------------------------- adversarial-review follow-ups


def _partition_only_table(tmp_path: Path) -> str:
    """A table whose every column is a partition column, as Spark writes one.

    delta-rs refuses to write such a table, so the log is hand-written. Each
    data file holds one column the table does not declare, so the Parquet read
    projection is empty while its row groups still carry the row count.
    """
    root = tmp_path / "ponly"
    (root / "_delta_log").mkdir(parents=True)
    schema = {
        "type": "struct",
        "fields": [
            {"name": "p", "type": "long", "nullable": True, "metadata": {}},
            {"name": "q", "type": "string", "nullable": True, "metadata": {}},
        ],
    }
    actions: list[dict[str, Any]] = [
        {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
        {
            "metaData": {
                "id": "ponly",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps(schema),
                "partitionColumns": ["p", "q"],
                "configuration": {},
                "createdTime": 0,
            }
        },
    ]
    for p, q, count in [(1, "x", 3), (2, None, 2)]:
        rel = f"p={p}/q={'__HIVE_DEFAULT_PARTITION__' if q is None else q}/part-0.parquet"
        (root / rel).parent.mkdir(parents=True)
        pq.write_table(pa.table({"_unused": list(range(count))}), root / rel)
        add = {
            "path": rel,
            "partitionValues": {"p": str(p), "q": q},
            "size": (root / rel).stat().st_size,
            "modificationTime": 0,
            "dataChange": True,
            "stats": json.dumps({"numRecords": count}),
        }
        actions.append({"add": add})
    (root / "_delta_log" / f"{0:020}.json").write_text(
        "\n".join(json.dumps(a) for a in actions) + "\n"
    )
    return str(root)


class TestPartitionOnlyReads:
    """Reading only partition columns used to panic in kernel's Parquet reader
    (an empty read schema), which also killed the shared I/O executor."""

    def test_table_with_only_partition_columns(self, tmp_path: Path) -> None:
        snapshot = n.Snapshot.resolve(_partition_only_table(tmp_path))
        got = _read(snapshot)
        assert got.column_names == ["p", "q"]
        assert (
            sorted(got.to_pylist(), key=lambda r: r["p"])
            == [{"p": 1, "q": "x"}] * 3 + [{"p": 2, "q": None}] * 2
        )
        assert _read(snapshot, columns=["q"]).num_rows == 5

    def test_file_restricted_read_of_a_partition_only_table(self, tmp_path: Path) -> None:
        snapshot = n.Snapshot.resolve(_partition_only_table(tmp_path))
        first = snapshot.files().column("path")[0].as_py()
        assert _read(snapshot, files=[first]).num_rows in (2, 3)

    def test_projection_of_a_partition_column(self, tmp_path: Path) -> None:
        path = _table(tmp_path, AB, partition_by=["a"])
        _append(path, pa.table({"a": [1, 1, 2], "b": [1, 2, 3]}))
        got = _read(n.Snapshot.resolve(path), columns=["a"])
        assert got.column_names == ["a"] and sorted(got.column("a").to_pylist()) == [1, 1, 2]
        # The engine is still usable afterwards (the panic used to kill it).
        assert _read(n.Snapshot.resolve(path)).num_rows == 3

    def test_change_feed_projection_of_partition_columns(self, tmp_path: Path) -> None:
        schema = pa.schema([("p", pa.int64()), ("v", pa.int64())])
        path = _table(
            tmp_path,
            schema,
            partition_by=["p"],
            properties={"delta.enableChangeDataFeed": "true"},
        )
        _append(path, pa.table({"p": [1, 2], "v": [1, 2]}, schema=schema))
        for columns in (["p"], []):
            got = pa.RecordBatchReader.from_stream(
                n.table_changes(path, start_version=0, columns=columns)
            ).read_all()
            assert "__deltaswamp_row_index" not in got.column_names
            assert got.num_rows == 2, columns


class TestWriteConformReview:
    def test_float64_data_into_a_float_column(self, tmp_path: Path) -> None:
        path = _table(tmp_path, pa.schema([("f", pa.float32())]))
        _append(path, pa.table({"f": pa.array([0.1, 2.5], pa.float64())}))
        assert pq.read_schema(_newest_parquet(path)).field("f").type == pa.float32()
        assert _rows(path) == [{"f": pytest.approx(0.1)}, {"f": 2.5}]

    def test_float64_overflowing_float_is_refused(self, tmp_path: Path) -> None:
        path = _table(tmp_path, pa.schema([("f", pa.float32())]))
        with pytest.raises(ValueError, match="infinity"):
            _append(path, pa.table({"f": pa.array([1e300], pa.float64())}))

    def test_null_partition_strings_become_null(self, tmp_path: Path) -> None:
        schema = pa.schema([("id", pa.int64()), ("p", pa.int64())])
        path = _table(tmp_path, schema, partition_by=["p"])
        _append(
            path,
            pa.table({"id": [1, 2, 3], "p": ["", "__HIVE_DEFAULT_PARTITION__", "7"]}),
        )
        got = sorted(_read(n.Snapshot.resolve(path)).to_pylist(), key=lambda r: r["id"])
        assert [r["p"] for r in got] == [None, None, 7]
        with pytest.raises(ValueError):
            _append(path, pa.table({"id": [4], "p": ["abc"]}))
