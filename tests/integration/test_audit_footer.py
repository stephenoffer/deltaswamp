"""Regressions for the Parquet footer Databricks needs to read old dates right (r6 CP-5).

A Databricks SQL warehouse reads a Parquet file whose footer names no Spark
version with Spark's legacy calendar rebase: ``0001-01-01`` written here read
there as ``0001-01-03``, ``1500-06-15`` as ``1500-06-05``, timestamps alike,
while deltaswamp read the values written. Every data file the kernel writes now
names a Spark 3 version; delta-rs cannot, so values it would write shifted are
kept off it.

Every test here failed before the fix.
"""

from __future__ import annotations

import datetime as dt
import pathlib
from typing import Any

import pytest
from deltaswamp.errors import UnreachableTableError

from tests.integration.test_audit_read import _spark_table

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
deltalake = pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

UTC = dt.UTC
SPARK_VERSION = b"org.apache.spark.version"
EARLY = [dt.date(1, 1, 1), dt.date(1500, 6, 15), dt.date(1582, 10, 4), dt.date(1899, 12, 31)]
LATE = [dt.date(1900, 1, 1), dt.date(2024, 1, 1)]


def _rows(days: list[dt.date], base: int = 0) -> Any:
    ts = [dt.datetime(d.year, d.month, d.day, 12, 34, 56, 123456) for d in days]
    return pa.table(
        {
            "rid": pa.array(range(base, base + len(days)), pa.int64()),
            "d": pa.array(days, pa.date32()),
            "ts": pa.array([t.replace(tzinfo=UTC) for t in ts], pa.timestamp("us", "UTC")),
            "ntz": pa.array(ts, pa.timestamp("us")),
            "st": pa.array([{"d": d} for d in days], pa.struct([("d", pa.date32())])),
            "s": pa.array([f"v{i}" for i in range(base, base + len(days))]),
        }
    )


def _live_files(path: str) -> list[pathlib.Path]:
    return [pathlib.Path(u.removeprefix("file://")) for u in deltalake.DeltaTable(path).file_uris()]


def _assert_footers(path: str) -> None:
    """Every live file names a Spark 3 version, and no legacy calendar."""
    files = _live_files(path)
    assert files
    for f in files:
        kv = pq.ParquetFile(f).metadata.metadata or {}
        assert kv.get(SPARK_VERSION) == b"3.5.0", (f.name, kv)
        assert b"org.apache.spark.legacyDateTime" not in kv
        assert b"org.apache.spark.legacyINT96" not in kv


def _values(path: str) -> list[tuple[Any, ...]]:
    t = ds.connect().open_table(path).to_arrow(columns=["rid", "d", "ts", "ntz", "st"])
    return sorted(tuple(r.values()) for r in t.to_pylist())


def _raw_values(path: str) -> list[tuple[Any, ...]]:
    """The values as stored, by a reader that never rebases."""
    out: list[tuple[Any, ...]] = []
    for f in _live_files(path):
        t = pq.read_table(f, columns=["rid", "d", "ts", "ntz", "st"])
        out += [tuple(r.values()) for r in t.to_pylist()]
    return sorted(out)


@pytest.fixture
def early_table(conn: Any, tmp_path: pathlib.Path) -> Any:
    def make(name: str = "t", **kwargs: Any) -> Any:
        path = str(tmp_path / name)
        return conn.write_table(path, _rows(EARLY + LATE), **kwargs)

    return make


class TestKernelFilesNameASparkVersion:
    def test_an_append_of_early_values(self, early_table: Any) -> None:
        t = early_table()
        t.append(_rows(EARLY, base=100))
        _assert_footers(t.location)
        assert _values(t.location) == _raw_values(t.location)
        assert [r[1] for r in _values(t.location)][:4] == EARLY

    def test_an_overwrite_and_a_replace_where(self, early_table: Any) -> None:
        t = early_table()
        t.overwrite(_rows(EARLY, base=10))
        _assert_footers(t.location)
        t.overwrite(_rows(EARLY[:2], base=10), predicate="rid < 12")
        _assert_footers(t.location)
        assert _values(t.location) == _raw_values(t.location)

    def test_deletion_vector_dml_rewrites(self, early_table: Any) -> None:
        t = early_table(properties={"delta.enableDeletionVectors": "true"})
        t.delete("rid = 0")
        t.update({"s": "'upd'"}, predicate="rid = 1")
        (
            t.merge(_rows(EARLY[:2], base=2), "target.rid = source.rid")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute()
        )
        _assert_footers(t.location)

    def test_copy_on_write_dml_on_a_table_holding_early_values(self, early_table: Any) -> None:
        t = early_table()
        assert t.can("delete", predicate="rid = 0").engine is ds.Engine.KERNEL
        t.delete("rid = 0")
        t.update({"s": "'upd'"}, predicate="rid = 1")
        _assert_footers(t.location)
        assert _values(t.location) == _raw_values(t.location)

    def test_compaction_of_spark_written_files(self, conn: Any, tmp_path: pathlib.Path) -> None:
        # CP-5: Spark 4 files Databricks read right were compacted into files
        # it read two to ten days off.
        path = _spark_table(tmp_path / "t", legacy=False)
        t = conn.open_table(path)
        before = t.to_arrow().sort_by("rid").to_pylist()
        t.append(t.to_arrow())
        t.optimize()
        _assert_footers(path)
        after = conn.open_table(path).to_arrow().sort_by("rid").to_pylist()
        assert after[::2] == before

    def test_a_distributed_write(self, early_table: Any) -> None:
        t = early_table()
        plan = t.plan_write()
        plan.commit([plan.write(_rows(EARLY, base=50))])
        _assert_footers(t.location)


class TestDeltaRsIsNotHandedEarlyValues:
    def test_an_append_of_early_values_goes_to_the_kernel(self, early_table: Any) -> None:
        t = early_table()
        assert t.can("append", data=_rows(EARLY)).engine is ds.Engine.KERNEL
        # Values no rebase moves are delta-rs's still.
        assert t.can("append", data=_rows(LATE)).engine is ds.Engine.DELTARS

    def test_a_table_delta_rs_wrote_is_repaired_by_compaction(
        self, conn: Any, tmp_path: pathlib.Path
    ) -> None:
        path = str(tmp_path / "t")
        deltalake.write_deltalake(path, _rows(EARLY))
        deltalake.write_deltalake(path, _rows(LATE, base=10), mode="append")
        before = _values(path)
        conn.open_table(path).optimize()
        _assert_footers(path)
        assert _values(path) == before

    def test_a_merge_that_delta_rs_would_write_is_refused(self, early_table: Any) -> None:
        t = early_table()
        verdict = t.can("merge", data=_rows(EARLY))
        assert not verdict.ok
        assert "legacy calendar rebase" in verdict.reason
        with pytest.raises(UnreachableTableError, match="legacy calendar"):
            (
                t.merge(_rows(EARLY), "target.rid = source.rid")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute()
            )

    def test_a_delta_rs_rewrite_of_early_files_is_refused(self, early_table: Any) -> None:
        # Even with late source rows: a MERGE copies the target rows it does
        # not change into delta-rs files.
        t = early_table()
        verdict = t.can("merge", data=_rows(LATE, base=100))
        assert not verdict.ok
        assert "may hold dates before 1582-10-15" in verdict.reason

    def test_an_optimize_only_delta_rs_runs_is_refused(self, early_table: Any) -> None:
        t = early_table()
        t.append(_rows(EARLY, base=100))
        props = deltalake.WriterProperties(compression="ZSTD")
        with pytest.raises(UnreachableTableError, match="legacy calendar"):
            t.optimize(writer_properties=props)

    def test_a_late_only_table_is_still_rewritten_by_delta_rs(
        self, conn: Any, tmp_path: pathlib.Path
    ) -> None:
        t = conn.write_table(str(tmp_path / "t"), _rows(LATE))
        assert t.can("delete", predicate="rid = 0").engine is ds.Engine.DELTARS


class TestEarlyValueDetection:
    @pytest.mark.parametrize(
        ("array", "early"),
        [
            (pa.array([dt.date(1582, 10, 14)], pa.date32()), True),
            (pa.array([dt.date(1582, 10, 15), None], pa.date32()), False),
            (pa.array([dt.date(1500, 1, 1)], pa.date64()), True),
            (pa.array([dt.datetime(1899, 12, 31, 23, 59, 59)], pa.timestamp("s")), True),
            (pa.array([dt.datetime(1900, 1, 1)], pa.timestamp("ns", "UTC")), False),
            (pa.array([dt.datetime(1850, 1, 1)], pa.timestamp("ms", "UTC")), True),
            (pa.array([None, None], pa.date32()), False),
            (pa.array([[None, dt.date(1, 1, 1)]], pa.list_(pa.date32())), True),
            (pa.array([{"x": {"d": dt.date(1, 1, 1)}}]), True),
            (
                pa.array(
                    [[(dt.date(2000, 1, 1), dt.date(1000, 1, 1))]],
                    pa.map_(pa.date32(), pa.date32()),
                ),
                True,
            ),
            (pa.array([1, 2]), False),
        ],
    )
    def test_arrays(self, array: Any, early: bool) -> None:
        from deltaswamp.engine.calendar import holds_early_datetimes

        assert holds_early_datetimes(pa.table({"c": array})) is early

    def test_a_stream_is_not_consumed(self) -> None:
        from deltaswamp.engine.calendar import holds_early_datetimes

        reader = pa.RecordBatchReader.from_batches(_rows(EARLY).schema, _rows(EARLY).to_batches())
        assert holds_early_datetimes(reader) is False
        assert reader.read_all().num_rows == len(EARLY)

    def test_pandas(self) -> None:
        pd = pytest.importorskip("pandas")
        from deltaswamp.engine.calendar import holds_early_datetimes

        assert holds_early_datetimes(pd.DataFrame({"t": [pd.Timestamp("1850-01-01")]}))
        assert not holds_early_datetimes(pd.DataFrame({"t": [pd.Timestamp("2000-01-01")]}))


def test_a_footer_without_spark_keys_is_read_as_written(tmp_path: pathlib.Path) -> None:
    # Spark reads such a file in the session's rebase mode, and Databricks
    # warehouses choose LEGACY; the writer (pyarrow, arrow-rs, delta-rs) meant
    # proleptic Gregorian, which is what deltaswamp returns.
    path = str(tmp_path / "t")
    deltalake.write_deltalake(path, _rows(EARLY))
    assert _values(path) == _raw_values(path)
