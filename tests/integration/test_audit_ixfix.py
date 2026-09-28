"""Regressions for the round-7 live interop findings, reproduced on local tables."""

from __future__ import annotations

import json
import math
import os
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("deltaswamp")


@pytest.fixture
def conn() -> Any:
    return ds.connect("file://")


# ----------------------------------------------------------------- helpers


def _kernel_only() -> Any:
    from deltaswamp.capability import Engine
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router

    return Connection(
        catalog=FilesystemCatalog(), router=Router(engines={Engine.KERNEL: KernelEngine()})
    )


def _deltars_only() -> Any:
    from deltaswamp.capability import Engine
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.router import Router

    return Connection(
        catalog=FilesystemCatalog(), router=Router(engines={Engine.DELTARS: DeltaRsEngine()})
    )


def _adds(path: str) -> list[dict[str, Any]]:
    """The add actions of the newest commit."""
    log = os.path.join(path, "_delta_log")
    last = sorted(n for n in os.listdir(log) if n.endswith(".json"))[-1]
    with open(os.path.join(log, last)) as f:
        return [json.loads(line)["add"] for line in f if '"add"' in line]


# --------------------------------------------------- 1: DV stats precision


class TestDeletionVectorStatsKeepDecimals:
    """The kernel's DV update re-serialised each add's stats through f64."""

    MAX = "12345678901234567890.123456789012345678"
    D0 = "9" * 38

    def _table(self, tmp_path: Any) -> str:
        import decimal

        d = decimal.Decimal
        path = str(tmp_path / "dv")
        conn = _kernel_only()
        conn.create_table(
            path,
            pa.schema(
                [("id", pa.int64()), ("dec", pa.decimal128(38, 18)), ("d0", pa.decimal128(38, 0))]
            ),
            properties={"delta.enableDeletionVectors": "true"},
        )
        conn.open_table(path).append(
            pa.table(
                {
                    "id": pa.array([1, 2, 3], pa.int64()),
                    "dec": pa.array(
                        [d(self.MAX), d("0.1"), d("-" + self.MAX)], pa.decimal128(38, 18)
                    ),
                    "d0": pa.array([d(self.D0), d(1), d("-" + self.D0)], pa.decimal128(38, 0)),
                }
            )
        )
        return path

    @pytest.mark.parametrize("op", ["delete", "update", "merge"])
    def test_the_rewritten_add_keeps_exact_bounds(self, tmp_path: Any, op: str) -> None:
        path = self._table(tmp_path)
        t = ds.connect("file://").open_table(path)
        if op == "delete":
            t.delete("id = 2")
        elif op == "update":
            t.update({"id": "id"}, predicate="id = 2")
        else:
            (
                t.merge(pa.table({"id": pa.array([2], pa.int64())}), "target.id = source.id")
                .when_matched_delete()
                .execute()
            )
        dv_adds = [a for a in _adds(path) if a.get("deletionVector")]
        assert dv_adds, _adds(path)
        stats = dv_adds[0]["stats"]
        assert f'"dec":{self.MAX}' in stats and f'"dec":-{self.MAX}' in stats, stats
        assert f'"d0":{self.D0}' in stats and f'"d0":-{self.D0}' in stats, stats
        assert '"tightBounds":false' in stats
        assert "e+" not in stats


# ------------------------------------------------------- 2: NaN statistics


def _footer_bounds(path: str, add: dict[str, Any]) -> dict[str, bool]:
    import pyarrow.parquet as pq

    group = pq.ParquetFile(os.path.join(path, add["path"])).metadata.row_group(0)
    return {
        group.column(i).path_in_schema: bool(
            group.column(i).statistics is not None and group.column(i).statistics.has_min_max
        )
        for i in range(group.num_columns)
    }


NAN_DATA = {
    "id": pa.array([1, 2, 3, 4], pa.int64()),
    "f": pa.array([1.0, float("nan"), 3.0, None], pa.float64()),
    "g": pa.array([1.0, 2.0, 3.0, 4.0], pa.float64()),
    "st": pa.array(
        [{"x": float("nan")}, {"x": 1.0}, None, {"x": 2.0}], pa.struct([("x", pa.float32())])
    ),
}


class TestNaNStatistics:
    """Files with NaN carried a max below it, and Databricks skipped the NaN rows."""

    def test_kernel_footer_has_no_bounds_for_nan_columns(self, tmp_path: Any) -> None:
        path = str(tmp_path / "k")
        conn = _kernel_only()
        data = pa.table(NAN_DATA)
        conn.create_table(path, data.schema)
        conn.open_table(path).append(data)
        (add,) = _adds(path)
        assert _footer_bounds(path, add) == {"id": True, "f": False, "g": True, "st.x": False}
        stats = json.loads(add["stats"])
        # NaN is the maximum: the kernel's collector writes no max for it.
        assert stats["maxValues"].get("f") is None and stats["maxValues"]["g"] == 4.0

    def test_kernel_keeps_bounds_without_nan(self, tmp_path: Any) -> None:
        path = str(tmp_path / "k2")
        conn = _kernel_only()
        data = pa.table({"id": pa.array([1, 2], pa.int64()), "f": pa.array([1.0, 2.0])})
        conn.create_table(path, data.schema)
        conn.open_table(path).append(data)
        (add,) = _adds(path)
        assert _footer_bounds(path, add) == {"id": True, "f": True}

    def test_deltars_writes_no_bounds_below_a_nan(self, tmp_path: Any) -> None:
        path = str(tmp_path / "d")
        conn = _deltars_only()
        data = pa.table(NAN_DATA)
        conn.create_table(path, data.schema)
        conn.open_table(path).append(data)
        (add,) = _adds(path)
        stats = json.loads(add["stats"])
        assert "f" not in stats.get("maxValues", {}) and "st" not in stats.get("maxValues", {})
        assert stats["maxValues"]["g"] == 4.0  # NaN-free: still skippable
        assert not _footer_bounds(path, add)["f"]

    def test_deltars_rewrite_turns_float_bounds_off(self, tmp_path: Any) -> None:
        path = str(tmp_path / "d2")
        conn = _deltars_only()
        data = pa.table(NAN_DATA)
        conn.create_table(path, data.schema)
        conn.open_table(path).append(data)
        conn.open_table(path).update({"id": "id + 10"}, predicate="id = 1")
        for add in _adds(path):
            stats = json.loads(add["stats"])
            assert "f" not in stats.get("maxValues", {}) and "g" not in stats.get("maxValues", {})
        got = pa.table(conn.open_table(path).to_arrow()).sort_by("id").column("f").to_pylist()
        assert math.isnan(got[0]) and got[1:] == [3.0, None, 1.0]


# ----------------------------------------- 3: INT96 timestamps inside lists


def test_int96_timestamps_in_lists_read(tmp_path: Any) -> None:
    """Databricks writes INT96; a nested ARRAY<TIMESTAMP> failed to cast."""
    import datetime as dt

    import pyarrow.parquet as pq

    utc = dt.UTC
    ts = dt.datetime(2020, 1, 1, 12, 30, tzinfo=utc)
    old = dt.datetime(1850, 3, 1, 0, 0, tzinfo=utc)
    ns = pa.timestamp("ns")
    inner = pa.struct([("x", pa.int64()), ("y", pa.list_(ns))])
    data = pa.table(
        {
            "id": pa.array([1, 2], pa.int64()),
            "a": pa.array([[ts.replace(tzinfo=None), None], None], pa.list_(ns)),
            "s": pa.array(
                [{"y": [ts.replace(tzinfo=None)]}, {"y": None}], pa.struct([("y", pa.list_(ns))])
            ),
            "aos": pa.array([[{"x": 1, "y": [old.replace(tzinfo=None)]}], []], pa.list_(inner)),
        }
    )
    path = str(tmp_path / "int96")
    os.makedirs(os.path.join(path, "_delta_log"))
    data = data.replace_schema_metadata({"org.apache.spark.version": "3.5.0"})
    pq.write_table(data, os.path.join(path, "part-0.parquet"), use_deprecated_int96_timestamps=True)
    ts_list = {"type": "array", "elementType": "timestamp", "containsNull": True}

    def field(name: str, kind: Any) -> dict[str, Any]:
        return {"name": name, "type": kind, "nullable": True, "metadata": {}}

    schema = {
        "type": "struct",
        "fields": [
            field("id", "long"),
            field("a", ts_list),
            field("s", {"type": "struct", "fields": [field("y", ts_list)]}),
            field(
                "aos",
                {
                    "type": "array",
                    "elementType": {
                        "type": "struct",
                        "fields": [field("x", "long"), field("y", ts_list)],
                    },
                    "containsNull": True,
                },
            ),
        ],
    }
    actions = [
        {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
        {
            "metaData": {
                "id": "00000000-0000-0000-0000-000000000096",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps(schema),
                "partitionColumns": [],
                "configuration": {},
                "createdTime": 0,
            }
        },
        {
            "add": {
                "path": "part-0.parquet",
                "partitionValues": {},
                "size": os.path.getsize(os.path.join(path, "part-0.parquet")),
                "modificationTime": 0,
                "dataChange": True,
            }
        },
    ]
    with open(os.path.join(path, "_delta_log", f"{0:020}.json"), "w") as f:
        f.write("\n".join(json.dumps(a) for a in actions))
    rows = sorted(
        pa.table(ds.connect("file://").open_table(path).to_arrow()).to_pylist(),
        key=lambda r: r["id"],
    )
    assert rows == [
        {"id": 1, "a": [ts, None], "s": {"y": [ts]}, "aos": [{"x": 1, "y": [old]}]},
        {"id": 2, "a": None, "s": {"y": None}, "aos": []},
    ]
