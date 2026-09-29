"""Regressions for Spark legacy-calendar defects found by the audit.

Every test here failed before its fix. The tables are golden files written as
Spark writes them in LEGACY rebase mode (see `test_audit_read.py`).
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import time
from collections.abc import Sequence
from typing import Any, ClassVar

import pytest
from deltaswamp.capability import Operation

from tests.integration.test_audit_read import (
    JULIAN_TS,
    SCHEMA_STRING,
    _micros,
    _spark_file,
    _spark_table,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

UTC = dt.UTC
EXPECTED_DATES = [dt.date(1, 1, 1), dt.date(1500, 6, 15), dt.date(2024, 1, 1)]


def _rows(conn: Any, path: str) -> list[tuple[int, dt.date]]:
    table = conn.open_table(path).to_arrow().sort_by("rid")
    return list(zip(table.column("rid").to_pylist(), table.column("dt").to_pylist(), strict=True))


def _raw_days(path: str) -> list[int]:
    """What a reader that does not rebase sees in the live files, as days since the epoch."""
    from deltalake import DeltaTable

    days = DeltaTable(path).to_pyarrow_table(columns=["dt"]).column("dt").cast(pa.int32())
    return sorted(days.to_pylist())


def _days(*dates: dt.date) -> list[int]:
    return sorted((d - dt.date(1970, 1, 1)).days for d in dates)


def _ts_table(
    root: pathlib.Path,
    micros: Sequence[int | None],
    meta: dict[str, str],
    *,
    int96: bool = False,
    nested: bool = False,
) -> str:
    """A one-file table with a zoned timestamp column `ts`, stored as given."""
    (root / "_delta_log").mkdir(parents=True)
    ts = pa.array(micros, pa.int64()).cast(pa.timestamp("us", tz="UTC"))
    columns: dict[str, Any] = {"rid": pa.array(range(len(micros)), pa.int64()), "ts": ts}
    fields: list[dict[str, Any]] = [
        {"name": "rid", "type": "long", "nullable": True, "metadata": {}},
        {"name": "ts", "type": "timestamp", "nullable": True, "metadata": {}},
    ]
    if nested:
        columns["st"] = pa.StructArray.from_arrays([ts], ["t"])
        fields.append(
            {
                "name": "st",
                "type": {
                    "type": "struct",
                    "fields": [
                        {"name": "t", "type": "timestamp", "nullable": True, "metadata": {}}
                    ],
                },
                "nullable": True,
                "metadata": {},
            }
        )
    table = pa.table(columns).replace_schema_metadata(meta)
    kwargs: dict[str, Any] = {"use_deprecated_int96_timestamps": True} if int96 else {}
    pq.write_table(table, root / "f.parquet", **kwargs)
    actions = [
        {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
        {
            "metaData": {
                "id": "6a6e6f3e-0000-4000-8000-000000000002",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps({"type": "struct", "fields": fields}),
                "partitionColumns": [],
                "configuration": {},
                "createdTime": 0,
            }
        },
        {
            "add": {
                "path": "f.parquet",
                "partitionValues": {},
                "size": (root / "f.parquet").stat().st_size,
                "modificationTime": int(time.time() * 1000),
                "dataChange": True,
            }
        },
    ]
    (root / "_delta_log" / f"{0:020d}.json").write_text(
        "\n".join(json.dumps(a) for a in actions) + "\n"
    )
    return str(root)


def _ts(conn: Any, path: str, column: str = "ts") -> list[int | None]:
    table = conn.open_table(path).to_arrow().sort_by("rid")
    values: list[int | None] = table.column(column).cast(pa.int64()).to_pylist()
    return values


class TestRewritesKeepTheCalendar:
    """delta-rs rewrote legacy files from their raw Julian values: 0001-01-01 became 0000-12-30."""

    def test_delta_rs_is_refused_and_the_kernel_computes(self, conn: Any, tmp_path: Any) -> None:
        # delta-rs is refused the rewrite; the kernel evaluates `rid + 10`
        # itself (DuckDB, Spark's dialect) and writes the dates it read.
        path = _spark_table(tmp_path / "t")
        t = conn.open_table(path)
        verdict = t.can("update", updates={"rid": "rid + 10"})
        assert verdict.engine is ds.Engine.KERNEL, verdict
        assert t.update({"rid": "rid + 10"}, predicate="rid = 2").engine == "kernel"
        assert _rows(conn, path) == [
            (1, dt.date(1, 1, 1)),
            (3, dt.date(2024, 1, 1)),
            (12, dt.date(1500, 6, 15)),
        ]
        assert sorted(_raw_days(path)) == sorted(_days(*EXPECTED_DATES))

    def test_kernel_merge_keeps_the_values(self, conn: Any, tmp_path: Any) -> None:
        # delta-rs is refused this MERGE; the kernel's copy-on-write one
        # reads the legacy file rebased and writes what it read.
        path = _spark_table(tmp_path / "t")
        t = conn.open_table(path)
        assert t.can("merge").engine is ds.Engine.KERNEL
        source = pa.table({"rid": pa.array([2], pa.int64())})
        t.merge(source, "target.rid = source.rid").when_matched_update(
            {"rid": "target.rid + 100"}
        ).execute()
        assert _rows(conn, path) == [
            (1, dt.date(1, 1, 1)),
            (3, dt.date(2024, 1, 1)),
            (102, dt.date(1500, 6, 15)),
        ]
        assert sorted(_raw_days(path)) == sorted(_days(*EXPECTED_DATES))

    def test_kernel_rewrites_keep_the_values(self, conn: Any, tmp_path: Any) -> None:
        path = _spark_table(tmp_path / "t")
        t = conn.open_table(path)
        t.update(new_values={"rid": 12}, predicate="rid = 2")
        t.delete("rid = 3")
        assert _rows(conn, path) == [(1, dt.date(1, 1, 1)), (12, dt.date(1500, 6, 15))]
        # The kernel wrote what it read, rebased: a plain reader agrees now.
        assert _raw_days(path) == _days(dt.date(1, 1, 1), dt.date(1500, 6, 15))

    def test_deletion_vector_dml_keeps_the_values(self, conn: Any, tmp_path: Any) -> None:
        path = _spark_table(tmp_path / "t")
        conn.open_table(path).set_properties({"delta.enableDeletionVectors": "true"})
        t = conn.open_table(path)
        t.update(new_values={"rid": 12}, predicate="rid = 2")
        source = pa.table({"rid": pa.array([1], pa.int64())})
        t = conn.open_table(path)
        t.merge(source, "target.rid = source.rid").when_matched_update(
            {"rid": "target.rid + 100"}
        ).execute()
        assert _rows(conn, path) == [
            (3, dt.date(2024, 1, 1)),
            (12, dt.date(1500, 6, 15)),
            (101, dt.date(1, 1, 1)),
        ]

    @pytest.mark.parametrize("operation", ["optimize", "zorder"])
    def test_the_kernel_compacts_legacy_files_keeping_the_values(
        self, conn: Any, tmp_path: Any, operation: str
    ) -> None:
        import pyarrow.parquet as pq

        path = _spark_table(tmp_path / "t")
        t = conn.open_table(path)
        assert t.can(operation).engine is ds.Engine.KERNEL
        if operation == "optimize":
            t.optimize()
        else:
            t.z_order(["rid"])
        assert sorted(d for _, d in _rows(conn, path)) == sorted(EXPECTED_DATES)
        # Written proleptic, and marked so Databricks does not rebase them.
        assert sorted(_raw_days(path)) == sorted(_days(*EXPECTED_DATES))
        for f in conn.open_table(path).files().column("path").to_pylist():
            meta = pq.read_metadata(f"{path}/{f}").metadata or {}
            assert meta.get(b"org.apache.spark.version", b"").startswith(b"3.")
            assert b"org.apache.spark.legacyDateTime" not in meta

    def test_proleptic_files_are_still_compacted(self, conn: Any, tmp_path: Any) -> None:
        path = _spark_table(tmp_path / "t", legacy=False)
        t = conn.open_table(path)
        # By the kernel, which commits every compaction now.
        assert t.can("optimize").engine is ds.Engine.KERNEL
        t.optimize()
        assert [d for _, d in _rows(conn, path)] == EXPECTED_DATES

    def test_legacy_files_with_only_modern_values_are_not_read(
        self, conn: Any, tmp_path: Any
    ) -> None:
        # Their statistics rule a rebase out, so no footer is even read.
        root = tmp_path / "t"
        (root / "_delta_log").mkdir(parents=True)
        add = _spark_file(
            root / "part-00000.parquet",
            [1],
            [dt.date(2024, 1, 1)],
            [dt.datetime(2024, 1, 1, tzinfo=UTC)],
            legacy=True,
        )
        actions = [
            {
                "protocol": {
                    "minReaderVersion": 3,
                    "minWriterVersion": 7,
                    "readerFeatures": ["timestampNtz"],
                    "writerFeatures": ["timestampNtz"],
                }
            },
            {
                "metaData": {
                    "id": "6a6e6f3e-0000-4000-8000-000000000003",
                    "format": {"provider": "parquet", "options": {}},
                    "schemaString": SCHEMA_STRING,
                    "partitionColumns": [],
                    "configuration": {},
                    "createdTime": 0,
                }
            },
            add,
        ]
        (root / "_delta_log" / f"{0:020d}.json").write_text(
            "\n".join(json.dumps(a) for a in actions) + "\n"
        )
        t = conn.open_table(str(root))
        assert t.can("optimize").engine is ds.Engine.KERNEL

    def test_old_int96_files_are_compacted_by_the_kernel(self, conn: Any, tmp_path: Any) -> None:
        # Not legacy at all, but delta-rs decodes INT96 as nanoseconds; the
        # kernel reads them at microseconds and rewrites them as INT64.
        meta = {"org.apache.spark.version": "3.5.0"}
        path = _ts_table(tmp_path / "t", [JULIAN_TS[min(JULIAN_TS)]], meta, int96=True)
        t = conn.open_table(path)
        before = _ts(conn, path)
        assert t.can(Operation.OPTIMIZE).engine is ds.Engine.KERNEL
        t.optimize()
        assert _ts(conn, path) == before


class TestInt96:
    """INT96 values before 1677 overflowed as nanoseconds: year 1500 came back as 2085."""

    def test_ancient_int96_is_read_exactly(self, conn: Any, tmp_path: Any) -> None:
        values = [-14816604303211000, -62135596800000000 + 5, None, 1704067200000000]
        path = _ts_table(
            tmp_path / "t", values, {"org.apache.spark.version": "3.5.0"}, int96=True, nested=True
        )
        assert _ts(conn, path) == values
        rows = conn.open_table(path).to_arrow().sort_by("rid").column("st").to_pylist()
        epoch = dt.datetime(1970, 1, 1, tzinfo=UTC)
        assert [r["t"] for r in rows] == [
            None if v is None else epoch + dt.timedelta(microseconds=v) for v in values
        ]

    def test_legacy_int96_is_rebased(self, conn: Any, tmp_path: Any) -> None:
        meta = {
            "org.apache.spark.version": "3.5.0",
            "org.apache.spark.legacyINT96": "",
            "org.apache.spark.timeZone": "UTC",
        }
        stored = [JULIAN_TS[t] for t in sorted(JULIAN_TS)]
        path = _ts_table(tmp_path / "t", stored, meta, int96=True)
        assert _ts(conn, path) == [_micros(t) for t in sorted(JULIAN_TS)]


class TestWriterZone:
    """Other zones need Spark's per-zone rebase until 1900; 1850 in Los Angeles read 422 s late."""

    LA: ClassVar[dict[str, str]] = {
        "org.apache.spark.version": "3.5.0",
        "org.apache.spark.legacyDateTime": "",
        "org.apache.spark.timeZone": "America/Los_Angeles",
    }

    def test_before_1900_is_refused(self, conn: Any, tmp_path: Any) -> None:
        # Spark LEGACY's stored value for 1850-06-01T19:00Z (computed in Java).
        path = _ts_table(tmp_path / "t", [-3773710378000000], self.LA)
        with pytest.raises(Exception, match=r"1900-01-01T00:00:00Z.*America/Los_Angeles"):
            _ts(conn, path)

    def test_from_1900_is_read_as_written(self, conn: Any, tmp_path: Any) -> None:
        values = [-2208988800000000, 1704067200000000]
        path = _ts_table(tmp_path / "t", values, self.LA)
        assert _ts(conn, path) == values

    def test_an_unrecorded_zone_is_refused_rather_than_taken_for_utc(
        self, conn: Any, tmp_path: Any
    ) -> None:
        # Spark 2.x records no zone; Spark reads such files in its session zone.
        meta = {"org.apache.spark.version": "2.4.8"}
        path = _ts_table(tmp_path / "t", [-14816604303211000], meta)
        with pytest.raises(Exception, match="does not record the writer's time zone"):
            _ts(conn, path)
        modern = _ts_table(tmp_path / "m", [1704067200000000], meta)
        assert _ts(conn, modern) == [1704067200000000]
