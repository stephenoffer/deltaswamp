"""The kernel binding's extended surface, called directly on `deltaswamp._native`.

Covers predicate-based file skipping, file listing, timestamp time travel, the
change data feed, raw metadata access, raw put-if-absent commits, partitioned
appends, the Unity Catalog creation helpers and checkpointing. Every table is a
local temp table, and wherever it matters the result is cross-checked against
delta-rs so the two implementations cannot drift apart unnoticed.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from typing import Any

import deltaswamp as ds
import pytest

pytestmark = pytest.mark.skipif(
    not ds.has_native(), reason="native extension not built (run `maturin develop`)"
)

pa = pytest.importorskip("pyarrow")
deltalake = pytest.importorskip("deltalake")


def _native() -> Any:
    from deltaswamp import _native

    return _native


def _snapshot(path: str, **kwargs: Any) -> Any:
    return _native().Snapshot.resolve(path, **kwargs)


def _col(*path: str) -> dict[str, Any]:
    return {"column": list(path)}


def _lit(value: Any, type_: str) -> dict[str, Any]:
    return {"literal": value, "type": type_}


def _op(op: str, *args: Any) -> dict[str, Any]:
    return {"op": op, "args": list(args)}


def _pred(node: dict[str, Any]) -> str:
    return json.dumps(node)


@pytest.fixture
def two_file_table(tmp_path: Any) -> str:
    """Two commits, two files: ids 1-3 and ids 10-11."""
    from deltalake import write_deltalake

    path = str(tmp_path / "t")
    write_deltalake(path, pa.table({"id": [1, 2, 3], "g": ["a", "b", "a"]}))
    write_deltalake(path, pa.table({"id": [10, 11], "g": ["c", "c"]}), mode="append")
    return path


@pytest.fixture
def partitioned(tmp_path: Any) -> str:
    from deltalake import write_deltalake

    path = str(tmp_path / "part")
    write_deltalake(path, pa.table({"id": [1], "region": ["eu"]}), partition_by=["region"])
    return path


def test_features_lists_every_shipped_capability() -> None:
    assert set(_native().FEATURES) >= {
        "predicate_skipping",
        "timestamp_travel",
        "table_changes",
        "files",
        "metadata_json",
        "commit_raw",
        "partitioned_append",
        "uc_create_table_request",
        "checkpoint",
    }


class TestFiles:
    def test_one_row_per_live_file(self, two_file_table: str) -> None:
        files = pa.table(_snapshot(two_file_table).files())
        assert files.column_names == [
            "path",
            "size",
            "modification_time",
            "partition_values",
            "stats",
            "deletion_vector",
            "num_records",
        ]
        assert files.num_rows == 2
        assert sorted(files.column("num_records").to_pylist()) == [2, 3]
        assert files.column("deletion_vector").null_count == 2
        for row in files.to_pylist():
            assert not row["path"].startswith("/"), "paths are reported as stored (relative)"
            assert row["size"] > 0
            assert json.loads(row["stats"])["numRecords"] == row["num_records"]

    def test_paths_match_delta_rs(self, two_file_table: str) -> None:
        from deltalake import DeltaTable

        ours = set(pa.table(_snapshot(two_file_table).files()).column("path").to_pylist())
        theirs = {os.path.basename(u) for u in DeltaTable(two_file_table).file_uris()}
        assert ours == theirs

    def test_partition_values_are_reported(self, partitioned: str) -> None:
        rows = pa.table(_snapshot(partitioned).files()).to_pylist()
        assert rows[0]["partition_values"] == [("region", "eu")]


class TestPredicateSkipping:
    def test_stats_skip_a_file(self, two_file_table: str) -> None:
        pred = _pred(_op("gt", _col("id"), _lit(5, "long")))
        assert pa.table(_snapshot(two_file_table).files(pred)).num_rows == 1

    def test_scan_skips_files_but_does_not_filter_rows(self, two_file_table: str) -> None:
        # id > 10 skips the 1-3 file but keeps the whole 10-11 file: skipping
        # is file-level only, the exact row filter is the caller's job.
        pred = _pred(_op("gt", _col("id"), _lit(10, "long")))
        got = pa.table(_snapshot(two_file_table).scan(predicate=pred))
        assert sorted(got.column("id").to_pylist()) == [10, 11]

    def test_scan_with_projection_and_predicate_on_another_column(
        self, two_file_table: str
    ) -> None:
        pred = _pred(_op("eq", _col("g"), _lit("c", "string")))
        got = pa.table(_snapshot(two_file_table).scan(columns=["id"], predicate=pred))
        assert got.column_names == ["id"]
        assert sorted(got.column("id").to_pylist()) == [10, 11]

    def test_unconvertible_conjunct_is_dropped(self, two_file_table: str) -> None:
        pred = _pred(
            _op(
                "and",
                _op("gt", _col("id"), _lit(5, "long")),
                _op("eq", _col("no_such_column"), _lit(1, "long")),
            )
        )
        assert pa.table(_snapshot(two_file_table).files(pred)).num_rows == 1

    def test_or_with_unconvertible_child_means_no_skipping(self, two_file_table: str) -> None:
        pred = _pred(
            _op(
                "or",
                _op("gt", _col("id"), _lit(5, "long")),
                _op("eq", _col("no_such_column"), _lit(1, "long")),
            )
        )
        assert pa.table(_snapshot(two_file_table).files(pred)).num_rows == 2

    def test_not_over_partially_convertible_and_means_no_skipping(
        self, two_file_table: str
    ) -> None:
        # NOT(id > 5 AND ?) must not become NOT(id > 5), which would wrongly
        # skip the 10-11 file even though some of its rows could match.
        pred = _pred(
            _op(
                "not",
                _op(
                    "and",
                    _op("gt", _col("id"), _lit(5, "long")),
                    _op("eq", _col("no_such_column"), _lit(1, "long")),
                ),
            )
        )
        assert pa.table(_snapshot(two_file_table).files(pred)).num_rows == 2

    def test_unsupported_op_is_not_an_error(self, two_file_table: str) -> None:
        pred = _pred(_op("like", _col("g"), _lit("a%", "string")))
        assert pa.table(_snapshot(two_file_table).files(pred)).num_rows == 2

    def test_malformed_json_is_an_error(self, two_file_table: str) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            _snapshot(two_file_table).files("{nope")

    def test_partition_pruning(self, partitioned: str) -> None:
        from deltalake import write_deltalake

        write_deltalake(
            partitioned,
            pa.table({"id": [2], "region": ["us"]}),
            mode="append",
            partition_by=["region"],
        )
        pred = _pred(_op("eq", _col("region"), _lit("us", "string")))
        rows = pa.table(_snapshot(partitioned).files(pred)).to_pylist()
        assert [r["partition_values"] for r in rows] == [[("region", "us")]]


class TestTimeTravel:
    def test_resolves_the_version_as_of_a_timestamp(self, tmp_path: Any) -> None:
        from deltalake import write_deltalake

        path = str(tmp_path / "tt")
        write_deltalake(path, pa.table({"id": [1]}))
        time.sleep(0.05)
        between = int(time.time() * 1000)
        time.sleep(0.05)
        write_deltalake(path, pa.table({"id": [2]}), mode="append")

        assert _snapshot(path, timestamp_ms=between).version == 0
        assert _snapshot(path, timestamp_ms=int(time.time() * 1000) + 10_000).version == 1

    def test_timestamp_before_history_is_a_clear_error(self, two_file_table: str) -> None:
        with pytest.raises(ValueError, match="earliest recreatable commit"):
            _snapshot(two_file_table, timestamp_ms=1000)

    def test_version_and_timestamp_are_exclusive(self, two_file_table: str) -> None:
        with pytest.raises(ValueError, match="not both"):
            _snapshot(two_file_table, version=0, timestamp_ms=1000)


@pytest.fixture
def cdf_table(tmp_path: Any) -> tuple[str, int]:
    """A CDF-enabled table: create (v0), insert 1,2 (v1), insert 3 (v2), delete 1 (v3).

    Returns the path and a timestamp strictly between v1 and v2.
    """
    from deltalake import DeltaTable

    native = _native()
    path = str(tmp_path / "cdf")
    native.create_table(
        path,
        pa.schema([("id", pa.int64()), ("g", pa.string())]),
        properties={"delta.enableChangeDataFeed": "true"},
    )
    _snapshot(path).append(pa.table({"id": [1, 2], "g": ["a", "b"]}).to_reader())
    time.sleep(0.05)
    between = int(time.time() * 1000)
    time.sleep(0.05)
    _snapshot(path).append(pa.table({"id": [3], "g": ["c"]}).to_reader())
    DeltaTable(path).delete("id = 1")
    return path, between


class TestTableChanges:
    def test_full_feed(self, cdf_table: tuple[str, int]) -> None:
        path, _ = cdf_table
        rows = pa.table(_native().table_changes(path)).to_pylist()
        got = sorted((r["_commit_version"], r["_change_type"], r["id"]) for r in rows)
        assert got == [(1, "insert", 1), (1, "insert", 2), (2, "insert", 3), (3, "delete", 1)]
        assert all(r["_commit_timestamp"] is not None for r in rows)

    def test_version_range_and_projection_keep_change_columns(
        self, cdf_table: tuple[str, int]
    ) -> None:
        path, _ = cdf_table
        table = pa.table(
            _native().table_changes(path, start_version=2, end_version=2, columns=["id"])
        )
        assert table.column_names == ["id", "_change_type", "_commit_version", "_commit_timestamp"]
        assert table.column("id").to_pylist() == [3]

    def test_timestamp_bounds(self, cdf_table: tuple[str, int]) -> None:
        path, between = cdf_table
        after = pa.table(_native().table_changes(path, start_timestamp_ms=between))
        assert set(after.column("_commit_version").to_pylist()) == {2, 3}
        before = pa.table(_native().table_changes(path, end_timestamp_ms=between))
        assert set(before.column("_commit_version").to_pylist()) == {1}

    def test_start_version_and_timestamp_are_exclusive(self, cdf_table: tuple[str, int]) -> None:
        path, between = cdf_table
        with pytest.raises(ValueError, match="not both"):
            _native().table_changes(path, start_version=1, start_timestamp_ms=between)

    def test_refuses_a_table_without_cdf(self, two_file_table: str) -> None:
        with pytest.raises(ValueError):
            pa.table(_native().table_changes(two_file_table))


class TestRawMetadata:
    def test_metadata_json_is_a_protocol_metadata_action(self, two_file_table: str) -> None:
        from deltalake import DeltaTable

        md = json.loads(_snapshot(two_file_table).metadata_json())
        assert md["id"] == DeltaTable(two_file_table).metadata().id
        assert md["format"]["provider"] == "parquet"
        assert json.loads(md["schemaString"])["type"] == "struct"
        assert md["partitionColumns"] == []
        assert isinstance(md["configuration"], dict)
        assert "createdTime" in md

    def test_protocol_json_omits_feature_lists_below_table_features(
        self, two_file_table: str
    ) -> None:
        proto = json.loads(_snapshot(two_file_table).protocol_json())
        assert proto["minReaderVersion"] < 3 and proto["minWriterVersion"] < 7
        assert "readerFeatures" not in proto and "writerFeatures" not in proto

    def test_protocol_json_includes_features_at_v3_v7(self, cdf_table: tuple[str, int]) -> None:
        proto = json.loads(_snapshot(cdf_table[0]).protocol_json())
        assert proto["minWriterVersion"] == 7
        assert "changeDataFeed" in proto["writerFeatures"]

    def test_timestamp_is_the_commit_time_of_this_version(self, two_file_table: str) -> None:
        # Without in-commit timestamps this is the commit file's mtime.
        latest = _snapshot(two_file_table).timestamp()
        first = _snapshot(two_file_table, version=0).timestamp()
        log = os.path.join(two_file_table, "_delta_log", "00000000000000000001.json")
        assert abs(latest - int(os.path.getmtime(log) * 1000)) <= 1
        assert first <= latest

    def test_absent_domain_is_none_and_system_domains_are_readable(
        self, two_file_table: str
    ) -> None:
        snap = _snapshot(two_file_table)
        assert snap.domain_metadata("my.domain") is None
        assert snap.domain_metadata("delta.clustering") is None

    def test_clustering_domain_is_returned(self, tmp_path: Any) -> None:
        path = str(tmp_path / "clustered")
        _native().create_table(path, pa.schema([("id", pa.int64())]), cluster_by=["id"])
        config = _snapshot(path).domain_metadata("delta.clustering")
        assert config is not None
        assert json.loads(config)["clusteringColumns"] == [["id"]]


class TestCommitRaw:
    def test_metadata_only_commit_is_visible_to_both_engines(self, two_file_table: str) -> None:
        from deltalake import DeltaTable

        snap = _snapshot(two_file_table)
        md = json.loads(snap.metadata_json())
        md["configuration"]["custom.key"] = "v"
        actions = [
            json.dumps({"commitInfo": {"operation": "SET TBLPROPERTIES", "timestamp": 1}}),
            json.dumps({"metaData": md}),
        ]
        assert _native().commit_raw(two_file_table, snap.version + 1, actions) == snap.version + 1
        assert _snapshot(two_file_table).table_properties()["custom.key"] == "v"
        assert DeltaTable(two_file_table).metadata().configuration["custom.key"] == "v"

    def test_existing_version_is_a_commit_conflict(self, two_file_table: str) -> None:
        native = _native()
        with pytest.raises(native.CommitConflictError, match="already exists"):
            native.commit_raw(two_file_table, 1, [json.dumps({"commitInfo": {}})])

    def test_malformed_actions_are_refused_before_writing(self, two_file_table: str) -> None:
        native = _native()
        for bad in ([], ["not json"], ['{"a": 1, "b": 2}']):
            with pytest.raises(ValueError):
                native.commit_raw(two_file_table, 2, bad)
        assert not os.path.exists(
            os.path.join(two_file_table, "_delta_log", "00000000000000000002.json")
        )


class TestPartitionedAppend:
    def test_rows_land_in_their_partitions(self, partitioned: str) -> None:
        from deltalake import DeltaTable

        version = _snapshot(partitioned).append(
            pa.table({"id": [2, 3, 4], "region": ["us", None, "eu"]}).to_reader()
        )
        assert version == 1

        dirs = sorted(d for d in os.listdir(partitioned) if d.startswith("region="))
        assert dirs == ["region=__HIVE_DEFAULT_PARTITION__", "region=eu", "region=us"]

        ours = pa.table(_snapshot(partitioned).scan()).sort_by("id").to_pydict()
        theirs = DeltaTable(partitioned).to_pyarrow_table().sort_by("id").to_pydict()
        expected = {"id": [1, 2, 3, 4], "region": ["eu", "us", None, "eu"]}
        assert ours == expected
        assert theirs == expected

        by_value = {
            dict(r["partition_values"])["region"]: r["num_records"]
            for r in pa.table(_snapshot(partitioned).files()).to_pylist()
            if r["path"].startswith(("region=us", "region=__HIVE"))
        }
        assert by_value == {"us": 1, None: 1}

    def test_typed_partition_values_are_serialised_by_the_protocol(self, tmp_path: Any) -> None:
        from deltalake import DeltaTable

        path = str(tmp_path / "typed")
        schema = pa.schema([("v", pa.string()), ("day", pa.date32()), ("n", pa.int32())])
        _native().create_table(path, schema, partition_by=["day", "n"])
        data = pa.table(
            {
                "v": ["a", "b", "c"],
                "day": [dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2025, 6, 30)],
                "n": pa.array([7, 7, 8], pa.int32()),
            }
        )
        _snapshot(path).append(data.to_reader())

        parts = sorted(
            tuple(sorted(r["partition_values"]))
            for r in pa.table(_snapshot(path).files()).to_pylist()
        )
        assert parts == [
            (("day", "2024-01-02"), ("n", "7")),
            (("day", "2025-06-30"), ("n", "8")),
        ]
        theirs = DeltaTable(path).to_pyarrow_table().sort_by("v").to_pydict()
        assert theirs["day"] == [dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2025, 6, 30)]
        assert theirs["n"] == [7, 7, 8]

    def test_missing_partition_column_is_refused(self, partitioned: str) -> None:
        with pytest.raises(ValueError, match="no such column"):
            _snapshot(partitioned).append(pa.table({"id": [9]}).to_reader())


class TestUnityCatalogHelpers:
    def test_required_properties(self) -> None:
        props = _native().uc_required_properties("tbl-123")
        assert props["io.unitycatalog.tableId"] == "tbl-123"
        assert props["delta.feature.catalogManaged"] == "supported"

    def test_create_table_request_from_v0(self, tmp_path: Any) -> None:
        path = str(tmp_path / "uc")
        _native().create_table(path, pa.schema([("id", pa.int64())]), properties={"k": "v"})
        body = json.loads(_native().uc_create_table_request(path, "main.s.t"))
        assert body["name"] == "main.s.t"
        assert body["table-type"] == "MANAGED"
        assert body["properties"]["k"] == "v"
        assert body["properties"]["delta.checkpointPolicy"] == "v2"


def _with_timeout(fn: Any, seconds: float = 5.0) -> Any:
    """Run `fn` on a daemon thread so a deadlock fails the test, not the run."""
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        pytest.fail(f"call did not finish within {seconds}s (deadlock?)")
    if "error" in result:
        raise result["error"]
    return result["value"]


class TestCheckpoint:
    def test_writes_a_checkpoint_readable_by_both_engines(self, two_file_table: str) -> None:
        from deltalake import DeltaTable

        assert _with_timeout(lambda: _snapshot(two_file_table).checkpoint()) is True
        log = os.listdir(os.path.join(two_file_table, "_delta_log"))
        assert "00000000000000000001.checkpoint.parquet" in log
        assert "_last_checkpoint" in log

        assert pa.table(_snapshot(two_file_table).scan()).num_rows == 5
        assert DeltaTable(two_file_table).to_pyarrow_table().num_rows == 5

    def test_second_checkpoint_reports_already_exists(self, two_file_table: str) -> None:
        _with_timeout(lambda: _snapshot(two_file_table).checkpoint())
        assert _with_timeout(lambda: _snapshot(two_file_table).checkpoint()) is False

    def test_checkpoint_on_partitioned_table_after_kernel_append(self, partitioned: str) -> None:
        _snapshot(partitioned).append(pa.table({"id": [2], "region": ["us"]}).to_reader())
        assert _with_timeout(lambda: _snapshot(partitioned).checkpoint()) is True
        assert pa.table(_snapshot(partitioned).scan()).num_rows == 2
