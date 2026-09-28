"""Regressions for the round-7 audit of the code rounds 5 and 6 added (nc7).

Every test here failed before its fix.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import pathlib
from typing import Any

import pytest
from deltaswamp.errors import UnreachableTableError

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
deltalake = pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

UTC = dt.UTC
SPARK_VERSION = b"org.apache.spark.version"
DV = {"delta.enableDeletionVectors": "true"}


def _late(n: int = 2) -> Any:
    return pa.table(
        {
            "id": pa.array(range(1, n + 1), pa.int64()),
            "d": pa.array([dt.date(2020 + i, 1, 1) for i in range(n)]),
            "ts": pa.array([dt.datetime(2020, 1, 1, tzinfo=UTC)] * n, pa.timestamp("us", "UTC")),
        }
    )


def _table(tmp_path: pathlib.Path, props: dict[str, str] | None = None) -> Any:
    path = str(tmp_path / "t")
    kw = {"properties": props} if props else {}
    return ds.connect().write_table(path, _late(), **kw)


def _footers(path: str, since: int | None = None) -> list[bytes | None]:
    """The Spark version each live file added after version `since` names (the table's
    first version when None)."""
    dtb = deltalake.DeltaTable(path)
    if since is None:
        since = min(h["version"] for h in dtb.history())
    older = set(deltalake.DeltaTable(path, version=since).file_uris())
    out = []
    for uri in dtb.file_uris():
        if uri in older:
            continue
        kv = pq.ParquetFile(uri.removeprefix("file://")).metadata.metadata or {}
        out.append(kv.get(SPARK_VERSION))
    return out


class TestEarlyValuesSetByDml:
    """A1: SET/INSERT values that may be early dates stay off delta-rs."""

    @pytest.mark.parametrize(
        "updates",
        [
            {"d": "DATE '1000-01-01'"},
            {"d": "'1000-01-01'"},
            {"ts": "TIMESTAMP '1850-01-01 00:00:00'"},
            {"d": "date_sub(d, 400000)"},
            {"d": "CAST(ts AS DATE) - 1"},
            {"ts": "d"},  # a DATE after 1582 may be a TIMESTAMP before 1900
        ],
    )
    def test_update_set_value_needs_the_footer(self, tmp_path: pathlib.Path, updates: Any) -> None:
        t = _table(tmp_path)
        cap = t.can("update", updates=updates, predicate="id = 1")
        assert cap.engine is not ds.Engine.DELTARS, cap
        try:
            t.update(updates, predicate="id = 1")
        except UnreachableTableError:
            assert not cap.ok
            return
        assert cap.ok
        assert all(v == b"3.5.0" for v in _footers(t.location)), _footers(t.location)

    @pytest.mark.parametrize(
        "updates",
        [
            {"d": "DATE '2000-01-01'"},
            {"d": "NULL"},
            {"d": "d"},
            {"d": "ts"},
            {"d": "current_date()"},
            {"ts": "TIMESTAMP '1950-01-01 00:00:00'"},
            {"id": "id + 1"},
        ],
    )
    def test_bounded_set_value_still_goes_to_deltars(
        self, tmp_path: pathlib.Path, updates: Any
    ) -> None:
        t = _table(tmp_path)
        assert t.can("update", updates=updates, predicate="id = 1").engine is ds.Engine.DELTARS

    def test_update_new_values(self, tmp_path: pathlib.Path) -> None:
        t = _table(tmp_path)
        early = {"d": dt.date(1000, 1, 1)}
        cap = t.can("update", new_values=early, predicate="id = 1")
        assert cap.engine is ds.Engine.KERNEL
        late = {"d": dt.date(2000, 1, 1)}
        assert t.can("update", new_values=late, predicate="id = 1").engine is ds.Engine.DELTARS
        t.update(new_values=early, predicate="id = 1")
        assert _footers(t.location) == [b"3.5.0"]

    def test_merge_set_and_insert_refused_without_a_footer_writer(
        self, tmp_path: pathlib.Path
    ) -> None:
        t = _table(tmp_path)
        before = deltalake.DeltaTable(t.location).version()
        src = pa.table({"id": pa.array([1, 9], pa.int64())})
        for clause in (
            ("when_matched_update", None, {"d": "DATE '1000-01-01'"}),
            ("when_not_matched_insert", None, {"id": "source.id", "d": "DATE '1000-01-01'"}),
        ):
            cap = t.can("merge", source=src, predicate="target.id = source.id", clauses=[clause])
            assert not cap.ok and "1582" in cap.reason, cap
        with pytest.raises(UnreachableTableError, match="1582"):
            t.merge(src, "target.id = source.id").when_matched_update(
                {"d": "DATE '1000-01-01'"}
            ).execute()
        with pytest.raises(UnreachableTableError, match="1582"):
            t.merge(src, "target.id = source.id").when_not_matched_insert(
                {"id": "source.id", "d": "DATE '1000-01-01'"}
            ).execute()
        assert deltalake.DeltaTable(t.location).version() == before

    def test_merge_copy_of_a_bounded_source_column_goes_to_deltars(
        self, tmp_path: pathlib.Path
    ) -> None:
        t = _table(tmp_path)
        src = pa.table({"id": pa.array([1], pa.int64()), "d": pa.array([dt.date(2022, 1, 1)])})
        clause = ("when_matched_update", None, {"d": "source.d"})
        cap = t.can("merge", source=src, predicate="target.id = source.id", clauses=[clause])
        assert cap.engine is ds.Engine.DELTARS

    def test_merge_moves_to_the_engine_its_clauses_need(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The builder is made before the clauses are known; one made on
        # delta-rs moves to the kernel once a clause sets an early date.
        from deltaswamp import router

        monkeypatch.setattr(router, "_preference", lambda op, table, engines: engines)
        t = _table(tmp_path, DV)
        before = deltalake.DeltaTable(t.location).version()
        src = pa.table({"id": pa.array([1], pa.int64())})
        assert t.can("merge", source=src, predicate="target.id = source.id").engine is (
            ds.Engine.DELTARS
        )
        t.merge(src, "target.id = source.id").when_matched_update(
            {"d": "DATE '1000-01-01'"}
        ).execute()
        rows = ds.connect().open_table(t.location).to_arrow().sort_by("id")
        assert rows.column("d").to_pylist() == [dt.date(1000, 1, 1), dt.date(2021, 1, 1)]
        assert set(_footers(t.location, before)) == {b"3.5.0"}

    def test_dv_merge_of_early_set_values_runs_on_the_kernel(self, tmp_path: pathlib.Path) -> None:
        t = _table(tmp_path, DV)
        before = deltalake.DeltaTable(t.location).version()
        src = pa.table(
            {"id": pa.array([1, 9], pa.int64()), "d": pa.array([dt.date(2022, 1, 1)] * 2)}
        )
        t.merge(src, "target.id = source.id").when_matched_update(
            {"d": "DATE '1000-01-01'"}
        ).when_not_matched_insert({"id": "source.id", "d": "source.d"}).execute()
        rows = ds.connect().open_table(t.location).to_arrow().sort_by("id")
        assert rows.column("d").to_pylist() == [
            dt.date(1000, 1, 1),
            dt.date(2021, 1, 1),
            dt.date(2022, 1, 1),
        ]
        assert set(_footers(t.location, before)) == {b"3.5.0"}

    def test_a_failed_file_check_refuses_the_rewrite(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # router.py _footerless_rewrite failed open: an error listing the
        # files let delta-rs rewrite them.
        from deltaswamp.engine import calendar

        t = _table(tmp_path)

        def broken(snapshot: Any) -> Any:
            raise OSError("listing failed")

        monkeypatch.setattr(calendar, "early_datetime_files", broken)
        cap = t.can("delete", predicate="abs(id) = 1")
        assert cap.engine is not ds.Engine.DELTARS
        assert not cap.ok or cap.engine is ds.Engine.KERNEL


class TestStreamedEarlyValues:
    """D11: a stream cannot be looked into; one with DATE/TIMESTAMP columns goes to the kernel."""

    EARLY = pa.table({"id": pa.array([7], pa.int64()), "d": pa.array([dt.date(1, 1, 1)])})

    @pytest.mark.parametrize(
        "make",
        [lambda t: t.to_reader(), lambda t: t.to_batches()],
        ids=["reader", "batches"],
    )
    def test_streamed_append_writes_the_footer(self, tmp_path: pathlib.Path, make: Any) -> None:
        path = str(tmp_path / "t")
        first = pa.table({"id": pa.array([0], pa.int64()), "d": pa.array([dt.date(2020, 1, 1)])})
        ds.connect().write_table(path, first)
        t = ds.connect().open_table(path)
        before = deltalake.DeltaTable(path).version()
        assert t.can("append", data=make(self.EARLY)).engine is ds.Engine.KERNEL
        t.append(make(self.EARLY))
        assert _footers(path, before) == [b"3.5.0"]
        got = ds.connect().open_table(path).to_arrow().sort_by("id").column("d").to_pylist()
        assert got == [dt.date(2020, 1, 1), dt.date(1, 1, 1)]

    def test_list_of_late_batches_still_goes_to_deltars(self, tmp_path: pathlib.Path) -> None:
        t = _table(tmp_path)
        assert t.can("append", data=_late().to_batches()).engine is ds.Engine.DELTARS

    def test_stream_without_datetime_columns_still_goes_to_deltars(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = str(tmp_path / "t")
        data = pa.table({"id": pa.array([1], pa.int64())})
        t = ds.connect().write_table(path, data)
        assert t.can("append", data=data.to_reader()).engine is ds.Engine.DELTARS


class TestCalendarCacheIdentity:
    """D1: the calendar verdicts were cached by (root, version) and outlived the table."""

    def test_recreated_table_is_checked_again(self, tmp_path: pathlib.Path) -> None:
        import json
        import shutil

        from tests.integration.test_audit_read import _spark_table

        root = tmp_path / "t"
        first = ds.connect().write_table(
            str(root),
            pa.table({"rid": pa.array([9], pa.int64()), "dt": pa.array([dt.date(2024, 1, 1)])}),
        )
        version = first.version
        assert first.can("delete", predicate="abs(rid) > 100").engine is ds.Engine.DELTARS
        shutil.rmtree(root)
        _spark_table(root)
        for v in range(1, version + 1):
            (root / "_delta_log" / f"{v:020d}.json").write_text(
                json.dumps({"commitInfo": {"operation": "SET TBLPROPERTIES", "timestamp": 0}})
                + "\n"
            )
        second = ds.connect().open_table(str(root))
        assert second.version == version
        assert second.can("delete", predicate="abs(rid) = 2").engine is not ds.Engine.DELTARS
        with contextlib.suppress(UnreachableTableError):
            second.delete("abs(rid) = 2")
        got = ds.connect().open_table(str(root)).to_arrow().sort_by("rid").column("dt")
        assert dt.date(1, 1, 1) in got.to_pylist()
