"""Regressions for the write-path audit, each proven on a real local table.

Every test here failed before its fix.
"""

from __future__ import annotations

import datetime as dt
import decimal
import glob
import json
import os
from typing import Any

import pytest
from deltaswamp.errors import InvalidArgumentError, UnreachableTableError

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")


@pytest.fixture
def conn() -> Any:
    return ds.connect("file://")


def _actions(path: str) -> list[dict[str, Any]]:
    out = []
    for f in sorted(glob.glob(os.path.join(path, "_delta_log", "*.json"))):
        with open(f) as fh:
            out += [json.loads(line) for line in fh if line.strip()]
    return out


def _protocol(path: str) -> dict[str, Any]:
    found: dict[str, Any] = [a["protocol"] for a in _actions(path) if "protocol" in a][-1]
    return found


def _adds(path: str) -> list[dict[str, Any]]:
    return [a["add"] for a in _actions(path) if "add" in a]


class TestColumnMappingWithTableFeatures:
    """LW-10: delta-rs's create dropped columnMapping next to timestamp_ntz."""

    def test_the_protocol_carries_column_mapping(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        schema = pa.schema([("id", pa.int64()), ("ntz", pa.timestamp("us"))])
        t = conn.create_table(path, schema, properties={"delta.columnMapping.mode": "name"})
        assert "columnMapping" in _protocol(path)["writerFeatures"]
        t.append(
            pa.table({"id": [1], "ntz": pa.array([dt.datetime(2024, 1, 1)], pa.timestamp("us"))})
        )
        t.rename_column("id", "id2")
        assert conn.open_table(path).to_arrow().column("id2").to_pylist() == [1]

    def test_rename_needs_the_feature_not_just_the_mode(self, conn: Any, tmp_path: Any) -> None:
        from deltalake import DeltaTable

        # Exactly the table the old create wrote: the mode, but no feature.
        path = str(tmp_path / "t")
        schema = pa.schema([("id", pa.int64()), ("ntz", pa.timestamp("us"))])
        DeltaTable.create(path, schema, configuration={"delta.columnMapping.mode": "name"})
        t = conn.open_table(path)
        assert not t.can("rename_column")
        with pytest.raises(UnreachableTableError, match="columnMapping feature"):
            t.rename_column("id", "id2")


class TestGeneratedAndIdentityColumnsAtCreate:
    """LW-1/LW-2: column metadata committed without the feature that gives it meaning."""

    def test_kernel_does_not_create_generated_columns(self, conn: Any, tmp_path: Any) -> None:
        schema = pa.schema(
            [
                ("id", pa.int64()),
                pa.field("g", pa.int64(), metadata={"delta.generationExpression": "id * 2"}),
            ]
        )
        with pytest.raises(UnreachableTableError, match="generatedColumns"):
            conn.create_table(
                str(tmp_path / "t"), schema, properties={"delta.enableDeletionVectors": "true"}
            )
        assert not os.path.exists(tmp_path / "t" / "_delta_log")

    def test_generated_columns_are_enforced(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        schema = pa.schema(
            [
                ("id", pa.int64()),
                pa.field("g", pa.int64(), metadata={"delta.generationExpression": "id * 2"}),
            ]
        )
        t = conn.create_table(path, schema)
        with pytest.raises(InvalidArgumentError):
            t.append(pa.table({"id": [4], "g": [99]}))

    def test_identity_columns_are_refused_locally(self, conn: Any, tmp_path: Any) -> None:
        md = {
            "delta.identity.start": "1",
            "delta.identity.step": "1",
            "delta.identity.allowExplicitInsert": "false",
        }
        schema = pa.schema([pa.field("id", pa.int64(), metadata=md), ("v", pa.string())])
        for props in (None, {"delta.enableDeletionVectors": "true"}):
            with pytest.raises(UnreachableTableError, match="identity"):
                conn.create_table(str(tmp_path / "t"), schema, properties=props)
        assert not os.path.exists(tmp_path / "t" / "_delta_log")


class TestDecimalStats:
    """RP-2: delta-rs logged decimal min/max as doubles, and Databricks skipped files."""

    SCHEMA = pa.schema(
        [
            ("id", pa.int64()),
            ("dec", pa.decimal128(38, 18)),
            ("small", pa.decimal128(5, 2)),
            ("s", pa.struct([("d", pa.decimal128(20, 2)), ("x", pa.int64())])),
        ]
    )

    def _data(self) -> Any:
        return pa.table(
            {
                "id": [1],
                "dec": [decimal.Decimal("9999999999999999.99")],
                "small": [decimal.Decimal("1.10")],
                "s": [{"d": decimal.Decimal("123456789012345678.91"), "x": 1}],
            },
            schema=self.SCHEMA,
        )

    @pytest.mark.parametrize("props", [None, {"delta.columnMapping.mode": "name"}])
    def test_imprecise_bounds_are_left_out(self, conn: Any, tmp_path: Any, props: Any) -> None:
        path = str(tmp_path / "t")
        t = conn.create_table(path, self.SCHEMA, properties=props)
        t.append(self._data())
        t.update(new_values={"small": decimal.Decimal("2.20")}, predicate="id = 1")
        schema = json.loads(
            [a["metaData"] for a in _actions(path) if "metaData" in a][-1]["schemaString"]
        )
        phys = {
            f["name"]: f["metadata"].get("delta.columnMapping.physicalName", f["name"])
            for f in schema["fields"]
        }
        for add in _adds(path):
            stats = json.loads(add["stats"])
            assert phys["dec"] not in stats.get("maxValues", {})
            assert phys["s"] not in stats.get("maxValues", {})
            # Exact through a double, so still kept.
            assert phys["small"] in stats["maxValues"]

    def test_kernel_writes_them_exactly(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = conn.create_table(path, self.SCHEMA, properties={"delta.enableRowTracking": "true"})
        t.append(self._data())
        assert '"dec":9999999999999999.990000000000000000' in _adds(path)[-1]["stats"]


class TestPartitionValues:
    def test_binary_partitions_round_trip(self, conn: Any, tmp_path: Any) -> None:
        """LW-11: delta-rs wrote b"ab" as the text \\u0061\\u0062."""
        path = str(tmp_path / "t")
        t = conn.create_table(
            path, pa.schema([("id", pa.int64()), ("bin", pa.binary())]), partition_by=["bin"]
        )
        assert conn.open_table(path).can("append").engine != "deltars"
        t.append(pa.table({"id": [1], "bin": pa.array([b"ab"], pa.binary())}))
        assert conn.open_table(path).to_arrow().column("bin").to_pylist() == [b"ab"]
        with pytest.raises(InvalidArgumentError, match="UTF-8"):
            t.append(pa.table({"id": [2], "bin": pa.array([b"\x00\xff"], pa.binary())}))

    def test_dictionary_partition_column(self, conn: Any, tmp_path: Any) -> None:
        """LW-3: a pyarrow dictionary / polars Categorical partition column crashed delta-rs."""
        import pyarrow.compute as pc

        path = str(tmp_path / "t")
        t = conn.create_table(
            path, pa.schema([("id", pa.int64()), ("p", pa.string())]), partition_by=["p"]
        )
        t.append(pa.table({"id": [1], "p": pc.dictionary_encode(pa.array(["a"]))}))
        assert conn.open_table(path).to_arrow().to_pylist() == [{"id": 1, "p": "a"}]

    def test_empty_string_partition_is_null(self, conn: Any, tmp_path: Any) -> None:
        """LW-7: Spark writes an empty-string partition value as null."""
        path = str(tmp_path / "t")
        t = conn.create_table(
            path, pa.schema([("id", pa.int64()), ("p", pa.string())]), partition_by=["p"]
        )
        t.append(pa.table({"id": [1], "p": [""]}))
        assert [a["partitionValues"]["p"] for a in _adds(path)] == [None]


class TestWriteShapes:
    def test_merge_schema_on_column_mapped_table(self, conn: Any, tmp_path: Any) -> None:
        """LW-14: can() said delta-rs would serve it, and delta-rs then failed."""
        path = str(tmp_path / "t")
        t = conn.create_table(
            path, pa.schema([("id", pa.int64())]), properties={"delta.columnMapping.mode": "name"}
        )
        assert not t.can("append", schema_mode="merge")
        with pytest.raises(UnreachableTableError, match="column mapping"):
            t.append(pa.table({"id": [1], "w": [2]}), schema_mode="merge")

    def test_zoned_timestamps_create(self, conn: Any, tmp_path: Any) -> None:
        """LW-12: a non-UTC zone raised a bare Exception from delta-rs."""
        path = str(tmp_path / "t")
        ny = pa.timestamp("us", "America/New_York")
        conn.write_table(path, pa.table({"ts": pa.array([0], ny)}))
        assert conn.open_table(path).schema().field("ts").type == pa.timestamp("us", "UTC")

    @pytest.mark.parametrize("props", [None, {"delta.enableRowTracking": "true"}])
    def test_null_into_not_null(self, conn: Any, tmp_path: Any, props: Any) -> None:
        """LW-5: a raw DeltaError / ValueError, not a library error."""
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False), ("v", pa.string())])
        t = conn.create_table(str(tmp_path / "t"), schema, properties=props)
        with pytest.raises(InvalidArgumentError):
            t.append(pa.table({"id": pa.array([None], pa.int64()), "v": ["x"]}))

    def test_reserved_commit_metadata_on_delta_rs(self, conn: Any, tmp_path: Any) -> None:
        t = conn.create_table(str(tmp_path / "t"), pa.schema([("id", pa.int64())]))
        with pytest.raises(InvalidArgumentError, match="reserved"):
            t.append(pa.table({"id": [1]}), commit_metadata={"operation": "x"})

    def test_stream_missing_a_nullable_column(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = conn.create_table(path, pa.schema([("id", pa.int64()), ("v", pa.string())]))
        batch = pa.record_batch({"id": [1]})
        t.append(pa.RecordBatchReader.from_batches(batch.schema, [batch]))
        assert conn.open_table(path).to_arrow().to_pylist() == [{"id": 1, "v": None}]

    @pytest.mark.parametrize("sql_type", ["bigint", "int", "array<int>", "varchar(10)"])
    def test_sql_type_names_on_delta_rs(self, conn: Any, tmp_path: Any, sql_type: str) -> None:
        """OC-8: only Delta JSON names worked when delta-rs served add_column."""
        t = conn.create_table(str(tmp_path / "t"), pa.schema([("id", pa.int64())]))
        t.add_column({"c": sql_type})
        assert "c" in t.schema().names


class TestLocalPaths:
    """OC-4: paths delta-rs cannot write through, or no engine can read back."""

    @pytest.mark.parametrize("name", ["t[1]", "t|x", "t^x"])
    def test_url_hostile_characters_write_through_the_kernel(
        self, conn: Any, tmp_path: Any, name: str
    ) -> None:
        path = str(tmp_path / name)
        t = conn.create_table(path, pa.schema([("id", pa.int64())]))
        t.append(pa.table({"id": [1, 2]}))
        t.delete("id = 1")
        assert conn.open_table(path).count() == 1

    @pytest.mark.parametrize("name", ["p%20q/t", "b\\x/t"])
    def test_escapes_are_refused_before_anything_is_written(
        self, conn: Any, tmp_path: Any, name: str
    ) -> None:
        path = str(tmp_path / name)
        with pytest.raises(InvalidArgumentError, match="different path"):
            conn.create_table(path, pa.schema([("id", pa.int64())]))
        assert not os.path.exists(os.path.join(path, "_delta_log"))


class TestCreateArguments:
    """LC-12: argument checks that only path creates made."""

    def test_zero_columns(self, conn: Any, tmp_path: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="at least one column"):
            conn.create_table(str(tmp_path / "t"), pa.schema([]))

    def test_bogus_mode_on_a_catalog_name(self, conn: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="create mode"):
            conn.create_table("c.s.t", pa.schema([("id", pa.int64())]), mode="bogus")
