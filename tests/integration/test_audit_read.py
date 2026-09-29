"""Regressions for read-path defects found by the audit, each proven on a real table.

Every test here failed before its fix.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import time
import warnings
from typing import Any

import pytest
from deltaswamp.errors import EngineFallbackWarning, InvalidArgumentError, UnreachableTableError
from deltaswamp.predicate import PredicateError

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

UTC = dt.UTC
EPOCH = dt.date(1970, 1, 1)


def _days(d: dt.date) -> int:
    return (d - EPOCH).days


def _micros(t: dt.datetime) -> int:
    return (t - dt.datetime(1970, 1, 1, tzinfo=UTC)) // dt.timedelta(microseconds=1)


# What Photon writes for these values in LEGACY rebase mode (Julian day
# numbers), and what the warehouse reads back -- taken from a live table.
JULIAN = {dt.date(1, 1, 1): -719164, dt.date(1500, 6, 15): -171489, dt.date(2024, 1, 1): 19723}
JULIAN_TS = {
    dt.datetime(1, 1, 1, tzinfo=UTC): -62135769600000000,
    dt.datetime(1500, 6, 15, 12, 34, 56, 789000, tzinfo=UTC): -14816604303211000,
    dt.datetime(2024, 1, 1, tzinfo=UTC): 1704067200000000,
}

SCHEMA_STRING = json.dumps(
    {
        "type": "struct",
        "fields": [
            {"name": "rid", "type": "long", "nullable": True, "metadata": {}},
            {"name": "dt", "type": "date", "nullable": True, "metadata": {}},
            {"name": "ts", "type": "timestamp", "nullable": True, "metadata": {}},
            {"name": "ntz", "type": "timestamp_ntz", "nullable": True, "metadata": {}},
            {
                "name": "st",
                "type": {
                    "type": "struct",
                    "fields": [{"name": "d", "type": "date", "nullable": True, "metadata": {}}],
                },
                "nullable": True,
                "metadata": {},
            },
        ],
    }
)


def _spark_file(
    path: pathlib.Path,
    rids: list[int],
    dates: list[dt.date],
    stamps: list[dt.datetime],
    *,
    legacy: bool,
    zone: str = "Etc/UTC",
) -> dict[str, Any]:
    """A data file as Spark writes it, and its `add` action (stats are proleptic)."""
    if legacy:
        days = [JULIAN[d] for d in dates]
        micros = [JULIAN_TS[t] for t in stamps]
        meta = {
            "org.apache.spark.version": "4.2.0",
            "org.apache.spark.legacyDateTime": "",
            "org.apache.spark.legacyINT96": "",
            "org.apache.spark.timeZone": zone,
        }
    else:
        days = [_days(d) for d in dates]
        micros = [_micros(t) for t in stamps]
        meta = {"org.apache.spark.version": "4.2.0", "org.apache.spark.timeZone": zone}
    date_array = pa.array(days, pa.int32()).cast(pa.date32())
    # TIMESTAMP_NTZ is never rebased: it holds the proleptic value either way.
    ntz = pa.array([_micros(t) for t in stamps], pa.int64()).cast(pa.timestamp("us"))
    table = pa.table(
        {
            "rid": pa.array(rids, pa.int64()),
            "dt": date_array,
            "ts": pa.array(micros, pa.int64()).cast(pa.timestamp("us", tz="UTC")),
            "ntz": ntz,
            "st": pa.StructArray.from_arrays([date_array], ["d"]),
        }
    )
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, path)

    def ts_text(t: dt.datetime) -> str:
        return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"

    stats = {
        "numRecords": len(rids),
        "minValues": {
            "rid": min(rids),
            "dt": min(dates).isoformat(),
            "ts": ts_text(min(stamps)),
            "st": {"d": min(dates).isoformat()},
        },
        "maxValues": {
            "rid": max(rids),
            "dt": max(dates).isoformat(),
            "ts": ts_text(max(stamps)),
            "st": {"d": max(dates).isoformat()},
        },
        "nullCount": {"rid": 0, "dt": 0, "ts": 0, "ntz": 0, "st": {"d": 0}},
    }
    return {
        "add": {
            "path": path.name,
            "partitionValues": {},
            "size": path.stat().st_size,
            "modificationTime": int(time.time() * 1000),
            "dataChange": True,
            "stats": json.dumps(stats),
        }
    }


def _spark_table(root: pathlib.Path, *, legacy: bool = True, zone: str = "Etc/UTC") -> str:
    """A Delta table of two files written by Spark in the hybrid calendar."""
    root.mkdir()
    (root / "_delta_log").mkdir()
    ancient = _spark_file(
        root / "part-00000.parquet",
        [1, 2],
        [dt.date(1, 1, 1), dt.date(1500, 6, 15)],
        [
            dt.datetime(1, 1, 1, tzinfo=UTC),
            dt.datetime(1500, 6, 15, 12, 34, 56, 789000, tzinfo=UTC),
        ],
        legacy=legacy,
        zone=zone,
    )
    modern = _spark_file(
        root / "part-00001.parquet",
        [3],
        [dt.date(2024, 1, 1)],
        [dt.datetime(2024, 1, 1, tzinfo=UTC)],
        legacy=legacy,
        zone=zone,
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
                "id": "6a6e6f3e-0000-4000-8000-000000000001",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": SCHEMA_STRING,
                "partitionColumns": [],
                "configuration": {},
                "createdTime": 0,
            }
        },
        ancient,
        modern,
    ]
    (root / "_delta_log" / f"{0:020d}.json").write_text(
        "\n".join(json.dumps(a) for a in actions) + "\n"
    )
    return str(root)


class TestSparkLegacyCalendar:
    """Databricks writes pre-1582 dates in the hybrid calendar; the kernel read them days off."""

    def _read(self, conn: Any, path: str, **kwargs: Any) -> Any:
        with warnings.catch_warnings():
            # The kernel must serve it: delta-rs does not rebase.
            warnings.simplefilter("error", EngineFallbackWarning)
            return conn.open_table(path).to_arrow(**kwargs)

    def test_values_are_rebased_to_the_proleptic_calendar(self, conn: Any, tmp_path: Any) -> None:
        path = _spark_table(tmp_path / "t")
        rows = self._read(conn, path).sort_by("rid").to_pylist()
        assert [r["dt"] for r in rows] == [
            dt.date(1, 1, 1),
            dt.date(1500, 6, 15),
            dt.date(2024, 1, 1),
        ]
        assert [r["st"]["d"] for r in rows] == [r["dt"] for r in rows]
        assert [r["ts"] for r in rows] == [
            dt.datetime(1, 1, 1, tzinfo=UTC),
            dt.datetime(1500, 6, 15, 12, 34, 56, 789000, tzinfo=UTC),
            dt.datetime(2024, 1, 1, tzinfo=UTC),
        ]
        # TIMESTAMP_NTZ is written without a rebase and read without one.
        assert [r["ntz"] for r in rows] == [r["ts"].replace(tzinfo=None) for r in rows]

    @pytest.mark.parametrize(
        ("predicate", "expected"),
        [
            ("dt = DATE '0001-01-01'", [1]),
            ("dt < DATE '1582-10-15'", [1, 2]),
            ("st.d = DATE '1500-06-15'", [2]),
            ("ts = TIMESTAMP '1500-06-15 12:34:56.789Z'", [2]),
            ("dt >= DATE '1582-10-15'", [3]),
        ],
    )
    def test_predicates_and_skipping_see_the_rebased_values(
        self, conn: Any, tmp_path: Any, predicate: str, expected: list[int]
    ) -> None:
        path = _spark_table(tmp_path / "t")
        got = self._read(conn, path, predicate=predicate).column("rid").to_pylist()
        assert sorted(got) == expected
        assert conn.open_table(path).count(predicate=predicate) == len(expected)

    def test_files_without_the_legacy_marker_are_read_as_written(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = _spark_table(tmp_path / "t", legacy=False)
        rows = self._read(conn, path).sort_by("rid").to_pylist()
        assert [r["dt"] for r in rows] == [
            dt.date(1, 1, 1),
            dt.date(1500, 6, 15),
            dt.date(2024, 1, 1),
        ]

    def test_another_zone_is_refused_rather_than_guessed(self, conn: Any, tmp_path: Any) -> None:
        path = _spark_table(tmp_path / "t", zone="America/Los_Angeles")
        # Dates do not depend on the zone, and modern timestamps need no rebase.
        assert self._read(conn, path, columns=["rid", "dt"]).num_rows == 3
        assert self._read(conn, path, predicate="rid = 3").num_rows == 1
        with pytest.raises(Exception, match="America/Los_Angeles"):
            conn.open_table(path).to_arrow(columns=["rid", "ts"])


class TestProjectionErrors:
    """A misspelt column warned about an engine failure and surfaced delta-rs's DeltaError."""

    def test_unknown_column_is_an_invalid_argument(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = conn.create_table(path, pa.schema([("id", pa.int64())]))
        t.append(pa.table({"id": [1]}))
        with warnings.catch_warnings():
            warnings.simplefilter("error", EngineFallbackWarning)
            with pytest.raises(InvalidArgumentError, match="nope"):
                t.to_arrow(columns=["nope"])
            with pytest.raises(InvalidArgumentError, match="nope"):
                pa.table(t.scan(columns=["nope"], predicate="id = 1"))

    def test_delta_rs_refuses_it_too(self, conn: Any, tmp_path: Any) -> None:
        from deltaswamp.engine.deltars import DeltaRsEngine

        path = str(tmp_path / "t")
        t = conn.create_table(path, pa.schema([("id", pa.int64())]))
        t.append(pa.table({"id": [1]}))
        with pytest.raises(InvalidArgumentError, match="nope"):
            DeltaRsEngine().scan(t._resolved, columns=["nope"])

    def test_one_column_named_twice_is_refused(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = conn.create_table(path, pa.schema([("id", pa.int64())]))
        t.append(pa.table({"id": [1]}))
        with pytest.raises(InvalidArgumentError, match="same column"):
            t.to_arrow(columns=["id", "ID"])


class TestArgumentErrors:
    """Plausible mistakes escaped as bare ValueError/OverflowError/TypeError/AttributeError."""

    @pytest.fixture
    def table(self, conn: Any, tmp_path: Any) -> Any:
        t = conn.create_table(str(tmp_path / "t"), pa.schema([("id", pa.int64())]))
        t.append(pa.table({"id": [1]}))
        return t

    def test_restore_to_text_that_is_not_a_timestamp(self, table: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="ISO-8601"):
            table.restore("x")

    @pytest.mark.parametrize("option", ["target_file_size", "max_commit_retries"])
    def test_negative_write_sizes(self, table: Any, option: str) -> None:
        with pytest.raises(InvalidArgumentError, match=option):
            table.append(pa.table({"id": [2]}), **{option: -1})
        with pytest.raises(InvalidArgumentError, match=option):
            table.overwrite(pa.table({"id": [2]}), **{option: -1})

    def test_txn_version_needs_an_app_id(self, table: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="app_id"):
            table.txn_version(None)

    def test_sql_tables_must_be_a_mapping(self, conn: Any, table: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="tables="):
            conn.sql("SELECT 1", tables=[table])

    def test_set_not_null_on_a_missing_column(self, table: Any) -> None:
        with pytest.raises(ds.DeltaSwampError):
            table.set_not_null("nope")

    def test_sql_statement_error_is_exported(self) -> None:
        assert issubclass(ds.SqlStatementError, ds.DeltaSwampError)


class TestDeltaRsTimeTravel:
    """delta-rs ordered commits by file time, ignoring in-commit timestamps."""

    def _table(self, conn: Any, tmp_path: Any) -> tuple[Any, dict[int, int]]:
        path = tmp_path / "t"
        schema = pa.schema([("id", pa.int64())])
        t = conn.create_table(
            str(path), schema, properties={"delta.enableInCommitTimestamps": "true"}
        )
        for k in range(3):
            t.append(pa.table({"id": [k]}, schema=schema))
        # A copied log: every file's mtime is "now", the ICTs keep the real times.
        later = time.time() + 3600
        icts: dict[int, int] = {}
        for f in sorted((path / "_delta_log").glob("*.json")):
            import os

            os.utime(f, (later, later))
            for line in f.read_text().splitlines():
                action = json.loads(line)
                if "commitInfo" in action:
                    icts[int(f.name[:20])] = action["commitInfo"]["inCommitTimestamp"]
        return conn.open_table(str(path)), icts

    def test_timestamp_resolves_by_in_commit_timestamp(self, conn: Any, tmp_path: Any) -> None:
        from deltaswamp.engine.deltars import DeltaRsEngine

        t, icts = self._table(conn, tmp_path)
        at = dt.datetime.fromtimestamp((icts[2] + 1) / 1000, UTC)
        expected = t.to_arrow(timestamp=at).num_rows
        assert expected == 2
        assert pa.table(DeltaRsEngine().scan(t._resolved, timestamp=at.isoformat())).num_rows == 2

    def test_timestamp_before_the_table_is_refused(self, conn: Any, tmp_path: Any) -> None:
        from deltaswamp.engine.deltars import DeltaRsEngine

        t, icts = self._table(conn, tmp_path)
        before = dt.datetime.fromtimestamp((icts[0] - 1000) / 1000, UTC).isoformat()
        with pytest.raises(UnreachableTableError):
            t.to_arrow(timestamp=before)
        with pytest.raises(UnreachableTableError):
            DeltaRsEngine().scan(t._resolved, timestamp=before)


class TestPredicateMeaningMatchesSpark:
    """One predicate string must select the same rows on every engine."""

    @pytest.fixture
    def table(self, conn: Any, tmp_path: Any) -> Any:
        t = conn.create_table(
            str(tmp_path / "t"), pa.schema([("id", pa.int64()), ("city", pa.string())])
        )
        t.append(pa.table({"id": [1, 2, 3], "city": ["it's", "its", "a\\b"]}))
        return t

    def test_doubled_quote_is_two_adjacent_literals(self, table: Any) -> None:
        assert table.to_arrow(predicate="city = 'it''s'").column("id").to_pylist() == [2]
        assert table.to_arrow(predicate=r"city = 'it\'s'").column("id").to_pylist() == [1]

    def test_delta_rs_deletes_the_row_spark_would(self, table: Any, conn: Any) -> None:
        from deltaswamp.engine.deltars import DeltaRsEngine

        DeltaRsEngine().delete(table._resolved, "city = 'it''s'")
        assert sorted(conn.open_table(table.location).to_arrow().column("city").to_pylist()) == [
            "a\\b",
            "it's",
        ]

    def test_delta_rs_passthrough_reads_literals_as_spark(self, table: Any, conn: Any) -> None:
        from deltaswamp.engine.deltars import DeltaRsEngine

        # A function call is not parsed here, so the text reaches DataFusion.
        DeltaRsEngine().delete(table._resolved, "lower(city) = 'it''s' OR upper(city) = 'A\\\\B'")
        assert conn.open_table(table.location).to_arrow().column("city").to_pylist() == ["it's"]

    def test_string_column_against_a_number_is_refused(self, table: Any) -> None:
        with pytest.raises(PredicateError, match="Quote the literal"):
            table.to_arrow(predicate="city = 1")

    def test_integer_column_against_a_boolean_is_refused(self, table: Any) -> None:
        with pytest.raises(PredicateError, match="type mismatch"):
            table.to_arrow(predicate="id = true")
