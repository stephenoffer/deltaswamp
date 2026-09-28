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


class TestKernelCompression:
    """A2: the kernel's writer wrote UNCOMPRESSED Parquet (OPTIMIZE grew tables 2-4x)."""

    @staticmethod
    def _codecs(path: str, since: int) -> set[str]:
        dtb = deltalake.DeltaTable(path)
        older = set(deltalake.DeltaTable(path, version=since).file_uris())
        out = set()
        for uri in dtb.file_uris():
            if uri in older:
                continue
            meta = pq.ParquetFile(uri.removeprefix("file://")).metadata
            out |= {
                meta.row_group(g).column(c).compression
                for g in range(meta.num_row_groups)
                for c in range(meta.num_columns)
            }
        return out

    def _early_rows(self, n: int, base: int = 0) -> Any:
        # Early dates send the write to the kernel.
        return pa.table(
            {
                "id": pa.array(range(base, base + n), pa.int64()),
                "d": pa.array([dt.date(1000, 1, 1)] * n),
                "s": pa.array([f"row-{i % 97}-payload" for i in range(n)]),
            }
        )

    def test_kernel_writes_snappy_and_optimize_does_not_grow_the_table(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = str(tmp_path / "t")
        t = ds.connect().write_table(path, self._early_rows(1000))
        for i in range(1, 5):
            t.append(self._early_rows(1000, i * 1000))
        before = sum(
            pathlib.Path(u.removeprefix("file://")).stat().st_size
            for u in deltalake.DeltaTable(path).file_uris()
        )
        assert self._codecs(path, 0) == {"SNAPPY"}
        version = deltalake.DeltaTable(path).version()
        t.optimize()
        assert self._codecs(path, version) == {"SNAPPY"}
        after = sum(
            pathlib.Path(u.removeprefix("file://")).stat().st_size
            for u in deltalake.DeltaTable(path).file_uris()
        )
        assert after <= before

    def test_table_codec_property_is_honored(self, tmp_path: pathlib.Path) -> None:
        path = str(tmp_path / "t")
        t = ds.connect().write_table(path, self._early_rows(10))
        t.set_properties({"delta.parquet.compression.codec": "zstd"})
        version = deltalake.DeltaTable(path).version()
        t.append(self._early_rows(10, 10))
        assert self._codecs(path, version) == {"ZSTD"}


class TestIntervalColumns:
    """D5-D7: SQL on interval columns, and year-month text validation."""

    @pytest.fixture
    def table(self, tmp_path: pathlib.Path) -> Any:
        from tests.integration.test_audit_live5 import _interval_table

        return _interval_table(ds.connect("file://"), tmp_path)

    @pytest.mark.parametrize("predicate", ["ym = -14", "i > 0", "y = 36", "`ym` IS NULL"])
    def test_dml_on_an_interval_column_is_refused_up_front(
        self, table: Any, predicate: str
    ) -> None:
        # can() named delta-rs, which then failed "No such field: ym".
        for op, kw in (("delete", {}), ("update", {"updates": {"id": "id + 1"}})):
            cap = table.can(op, predicate=predicate, **kw)
            assert not cap.ok and "interval" in cap.reason, cap
        with pytest.raises(UnreachableTableError, match="interval"):
            table.delete(predicate)
        assert table.count() == 2

    def test_setting_an_interval_column_is_refused(self, table: Any) -> None:
        # ym + 1 added a month to the stored integer; Spark refuses the types.
        assert not table.can("update", updates={"ym": "ym + 1"}, predicate="id = 1").ok
        with pytest.raises(UnreachableTableError, match="interval"):
            table.update({"ym": "ym + 1"}, predicate="id = 1")

    def test_other_columns_still_go_direct(self, table: Any) -> None:
        assert table.can("delete", predicate="id = 1").ok
        assert table.can("scan", predicate="id = 1").ok
        # A string literal that happens to spell a column name is no reference.
        assert table.can("scan", predicate="'ym' = 'ym'").ok

    def test_reads_never_compare_the_stored_integers(self, table: Any) -> None:
        # ym = -14 matched INTERVAL '-1-2' YEAR TO MONTH.
        assert not table.can("scan", predicate="ym = -14").ok
        with pytest.raises(UnreachableTableError, match="interval"):
            table.to_arrow(predicate="ym = -14")

    def test_lazy_hand_offs_filter_the_shown_values(self, table: Any) -> None:
        text = "INTERVAL '-1-2' YEAR TO MONTH"
        rel = table.to_duckdb().filter(f"ym = '{text.replace(chr(39), chr(39) * 2)}'")
        assert rel.project("id").fetchall() == [(1,)]
        assert table.to_duckdb().filter("i > INTERVAL 1 DAY").project("id").fetchall() == [(1,)]
        got = table.to_pyarrow_dataset().to_table(filter=pa.compute.field("ym") == text)
        assert got.column("id").to_pylist() == [1]
        pl = pytest.importorskip("polars")
        frame = table.to_polars(lazy=True).filter(pl.col("i") > dt.timedelta(days=1))
        assert frame.select("id").collect().to_series().to_list() == [1]

    @pytest.mark.parametrize(
        "column, text, message",
        [
            ("ym", "INTERVAL '1-13' YEAR TO MONTH", "0 to 11"),
            ("ym", "INTERVAL '999999999' YEAR", "out of range"),
        ],
    )
    def test_year_month_text_is_validated(
        self, table: Any, column: str, text: str, message: str
    ) -> None:
        from deltaswamp.errors import InvalidArgumentError

        with pytest.raises(InvalidArgumentError, match=message):
            table.append(pa.table({"id": pa.array([9], pa.int64()), column: pa.array([text])}))

    def test_year_to_month_text_into_a_year_column_drops_the_months(self, table: Any) -> None:
        table.append(
            pa.table(
                {"id": pa.array([9], pa.int64()), "y": pa.array(["INTERVAL '1-2' YEAR TO MONTH"])}
            )
        )
        path = table.location
        raw = deltalake.DeltaTable(path).to_pyarrow_table(filters=[("id", "=", 9)])
        assert raw.column("y").to_pylist() == [12]


def _legacy_float_table(tmp_path: pathlib.Path) -> Any:
    """A table delta-rs misreads (a Spark legacy-calendar file), so SQL is RowFilter's."""
    import json
    import time

    root = tmp_path / "t"
    (root / "_delta_log").mkdir(parents=True)
    rows = pa.table(
        {
            "rid": pa.array([1, 2], pa.int64()),
            "dt": pa.array([-719164, 19723], pa.int32()).cast(pa.date32()),
            "f": pa.array([0.1, 0.5], pa.float32()),
        }
    ).replace_schema_metadata(
        {
            "org.apache.spark.version": "3.5.0",
            "org.apache.spark.legacyDateTime": "",
            "org.apache.spark.timeZone": "UTC",
        }
    )
    pq.write_table(rows, root / "a.parquet")
    fields = [
        {"name": n, "type": k, "nullable": True, "metadata": {}}
        for n, k in (("rid", "long"), ("dt", "date"), ("f", "float"))
    ]
    actions = [
        {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
        {
            "metaData": {
                "id": "6a6e6f3e-0000-4000-8000-00000000000f",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps({"type": "struct", "fields": fields}),
                "partitionColumns": [],
                "configuration": {},
                "createdTime": 0,
            }
        },
        {
            "add": {
                "path": "a.parquet",
                "partitionValues": {},
                "size": (root / "a.parquet").stat().st_size,
                "modificationTime": int(time.time() * 1000),
                "dataChange": True,
            }
        },
    ]
    (root / "_delta_log" / f"{0:020d}.json").write_text("\n".join(map(json.dumps, actions)) + "\n")
    return ds.connect().open_table(str(root))


class TestRowFilterSemantics:
    """D2-D4: the DuckDB evaluation of Spark predicates outside the kernel's grammar."""

    @pytest.mark.parametrize(
        "predicate, expected",
        [("abs(f) = 0.1", []), ("abs(f) > 0.1", [1, 2]), ("abs(rid) > 0", [1, 2])],
    )
    def test_float_compares_as_double(
        self, tmp_path: pathlib.Path, predicate: str, expected: list[int]
    ) -> None:
        t = _legacy_float_table(tmp_path)
        assert t.can("scan", predicate=predicate).engine is ds.Engine.KERNEL
        assert sorted(t.to_arrow(predicate=predicate).column("rid").to_pylist()) == expected

    @pytest.mark.parametrize(
        "predicate", ["rid + 9223372036854775807 > 0", "rid * 9223372036854775807 > 0"]
    )
    def test_integer_overflow_raises(self, tmp_path: pathlib.Path, predicate: str) -> None:
        from deltaswamp.predicate import PredicateError

        t = _legacy_float_table(tmp_path)
        with pytest.raises(PredicateError, match="Overflow"):
            t.to_arrow(predicate=predicate)

    @pytest.mark.parametrize(
        "text",
        [
            "$$'$$ = $$x$$ OR rid IN (SELECT 1) OR $$'$$ = $$y$$",
            "$t$'$t$ = 'x' OR rid IN (SELECT 1) OR 'a' = $t$'$t$",
            "e'a\\'' OR rid IN (SELECT 1) OR 'x' = 'x'",
            "rid IN (VALUES (1))",
            "rid IN ( PIVOT t ON a)",
            "EXISTS (DESCRIBE t)",
            "rid IN (SUMMARIZE t)",
        ],
    )
    def test_screen_reads_literals_as_duckdb_does(self, text: str) -> None:
        pytest.importorskip("duckdb")
        from deltaswamp.engine.sharing import _screen_expression
        from deltaswamp.predicate import PredicateError

        with pytest.raises(PredicateError):
            _screen_expression(text)

    @pytest.mark.parametrize(
        "text", ["(rid = 1)", "(s = 'select')", '("values" > 1)', "(s = 'it''s')"]
    )
    def test_screen_passes_plain_expressions(self, text: str) -> None:
        pytest.importorskip("duckdb")
        from deltaswamp.engine.sharing import _screen_expression

        _screen_expression(text)


class TestRetryOptions:
    """D8/D9: retry storage options are read as delta-rs reads them, for every engine."""

    @pytest.mark.parametrize(
        "options",
        [
            {"retry_timeout": "2"},
            {"backoff_config.init_backoff": "1e19"},
            {"retry_timeout": "1e30"},
            {"retry_timeout": "20000000000000000000s"},
            {"backoff_config.base": "nan"},
            {"backoff_config.base": "-5"},
            {"max_retries": " 3 "},
        ],
    )
    def test_refused_at_connect(self, options: dict[str, str]) -> None:
        from deltaswamp.errors import InvalidArgumentError

        with pytest.raises(InvalidArgumentError, match="storage_options"):
            ds.connect(storage_options=options)

    @pytest.mark.parametrize(
        "options",
        [{"retry_timeout": "30 s"}, {"retry_timeout": "2 minutes"}, {"retry_timeout": "1d"}],
    )
    def test_humantime_durations_serve_both_engines(
        self, tmp_path: pathlib.Path, options: dict[str, str]
    ) -> None:
        path = str(tmp_path / "t")
        ds.connect().write_table(path, pa.table({"id": pa.array([1, 2], pa.int64())}))
        t = ds.connect(storage_options=options).open_table(path)
        assert t.to_arrow().num_rows == 2
        assert t.can("delete", predicate="abs(id) = 1").engine is ds.Engine.DELTARS
        t.delete("abs(id) = 1")
        assert t.count() == 1


class TestKernelStreamFailures:
    """B1, B2, B4: what a kernel write fed by a Python stream, or losing a race, raises."""

    @staticmethod
    def _table(tmp_path: pathlib.Path, props: dict[str, str] | None = None) -> Any:
        path = str(tmp_path / "t")
        ds.connect().create_table(
            path, schema=pa.schema([("id", pa.int64())]), properties=props or {}
        )
        t = ds.connect().open_table(path)
        for i in range(3):
            t.append(pa.table({"id": pa.array([i], pa.int64())}))
        return t

    def test_interrupt_in_an_appended_stream_is_not_an_argument_error(
        self, tmp_path: pathlib.Path
    ) -> None:
        from deltaswamp.engine.kernel import KernelEngine

        t = self._table(tmp_path)

        def gen() -> Any:
            yield pa.record_batch({"id": pa.array([1], pa.int64())})
            raise KeyboardInterrupt

        reader = pa.RecordBatchReader.from_batches(pa.schema([("id", pa.int64())]), gen())
        with pytest.raises(KeyboardInterrupt):
            KernelEngine().append(t.resolved, reader)
        assert t.count() == 3

    def test_interrupt_during_optimize_propagates(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deltaswamp.engine.kernel import KernelEngine

        t = self._table(tmp_path)
        real = KernelEngine._compacted_batches

        def interrupted(self: Any, *args: Any, **kwargs: Any) -> Any:
            stream = real(self, *args, **kwargs)

            def batches() -> Any:
                yield from stream
                raise KeyboardInterrupt

            return pa.RecordBatchReader.from_batches(stream.schema, batches())

        monkeypatch.setattr(KernelEngine, "_compacted_batches", interrupted)
        with pytest.raises(KeyboardInterrupt):
            t.optimize()
        assert sorted(ds.connect().open_table(t.location).to_arrow().column("id").to_pylist()) == [
            0,
            1,
            2,
        ]

    def test_a_vacuumed_input_file_replans_the_compaction(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deltaswamp.engine.kernel import KernelEngine

        t = self._table(tmp_path)
        t.append(pa.table({"id": pa.array([3], pa.int64())}))
        path = t.location
        real = KernelEngine._compacted_batches
        fired: list[int] = []

        def raced(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not fired:
                fired.append(1)
                # Another writer rewrites a file this step reads, and VACUUMs it.
                deltalake.DeltaTable(path).delete("id = 0")
                deltalake.DeltaTable(path).vacuum(
                    retention_hours=0, enforce_retention_duration=False, dry_run=False
                )
            return real(self, *args, **kwargs)

        monkeypatch.setattr(KernelEngine, "_compacted_batches", raced)
        t.optimize()
        got = sorted(ds.connect().open_table(path).to_arrow().column("id").to_pylist())
        assert got == [1, 2, 3]

    def test_dv_delete_losing_to_a_metadata_change(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deltaswamp.engine.kernel import KernelEngine
        from deltaswamp.errors import MetadataChangedError

        t = self._table(tmp_path, DV)
        path = t.location
        real = KernelEngine.snapshot
        fired: list[int] = []

        def racing(self: Any, table: Any, **kw: Any) -> Any:
            snapshot = real(self, table, **kw)
            if kw.get("write") and not fired:
                fired.append(1)
                deltalake.DeltaTable(path).alter.set_table_properties({"delta.appendOnly": "false"})
            return snapshot

        monkeypatch.setattr(KernelEngine, "snapshot", racing)
        with pytest.raises(MetadataChangedError):
            t.delete("id = 1")


class TestCatalogRecheck:
    """D10: the read-time catalog re-check."""

    def _handle(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Any:
        import deltaswamp.table as table_module

        from tests.integration.test_audit_live5 import _catalog_table

        monkeypatch.setattr(table_module, "_NAME_RECHECK_SECONDS", 0.0)
        conn = ds.connect("file://")
        path = str(tmp_path / "t")
        conn.create_table(path, pa.schema([("id", pa.int64())])).append(
            pa.table({"id": pa.array([1], pa.int64())})
        )
        return _catalog_table(conn, path)

    def test_a_revoked_privilege_surfaces_on_read(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deltaswamp.errors import PreflightError

        t, catalog = self._handle(tmp_path, monkeypatch)
        assert pa.table(t.to_arrow()).num_rows == 1

        def denied(ref: Any) -> Any:
            raise PreflightError(f"access to {ref} was denied", denied=True)

        catalog.resolve = denied
        with pytest.raises(PreflightError, match="denied"):
            t.to_arrow()

    def test_a_catalog_that_does_not_answer_is_still_tolerated(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deltaswamp.errors import PreflightError

        t, catalog = self._handle(tmp_path, monkeypatch)

        def throttled(ref: Any) -> Any:
            raise PreflightError("the workspace is throttling")

        catalog.resolve = throttled
        assert pa.table(t.to_arrow()).num_rows == 1

    def test_the_recheck_clock_does_not_travel(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        t, _ = self._handle(tmp_path, monkeypatch)
        t.to_arrow()
        assert "_named_checked_at" in t.__dict__
        assert "_named_checked_at" not in t.__getstate__()
