"""Regression tests for defects fixed in the SQL warehouse fallback.

One small test per bug, against fakes only: a recording statement backend, a
fake files API, and a scripted `statement_execution` with Arrow IPC bodies
served by a stub opener.
"""

from __future__ import annotations

import decimal
import io
import urllib.error
import urllib.request
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

from deltaswamp.catalog import ResolvedTable  # noqa: E402
from deltaswamp.engine import sql as sqlmod  # noqa: E402
from deltaswamp.engine.sql import SqlEngine  # noqa: E402
from deltaswamp.engine.sql_backend import (  # noqa: E402
    ParameterBinder,
    SdkStatementBackend,
    SqlParameter,
    SqlStatementError,
    _parameter,
)
from deltaswamp.errors import UnreachableTableError  # noqa: E402
from deltaswamp.identity import RefKind, TableRef  # noqa: E402

NAME = "`main`.`sales`.`orders`"


# ------------------------------------------------------------------ fakes


class Rec:
    def __init__(self, results: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, list[SqlParameter], bool]] = []
        self.results = results or {}

    def execute(self, statement: str, parameters: Any = (), *, fetch: bool = True) -> Any:
        self.calls.append((statement, list(parameters), fetch))
        if not fetch:
            return None
        for prefix, result in self.results.items():
            if statement.startswith(prefix):
                return result
        return pa.table({"x": pa.array([], pa.int64())})

    @property
    def last(self) -> str:
        return self.calls[-1][0]

    @property
    def params(self) -> dict[str, tuple[str | None, str]]:
        return {p.name: (p.value, p.type) for p in self.calls[-1][1]}


class Files:
    def __init__(self, fail_upload: bool = False) -> None:
        self.uploaded: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.fail_upload = fail_upload

    def upload(self, path: str, contents: Any, *, overwrite: bool | None = None) -> None:
        if self.fail_upload:
            raise OSError("connection reset mid-upload")
        self.uploaded[path] = contents.read()

    def delete(self, path: str) -> None:
        self.deleted.append(path)


def tbl(**kwargs: Any) -> ResolvedTable:
    ref = TableRef(kind=RefKind.CATALOG, catalog="main", schema="sales", table="orders")
    kwargs.setdefault("location", "s3://bucket/t")
    return ResolvedTable(ref=ref, **kwargs)


def eng(rec: Rec | None = None, files: Files | None = None, **kwargs: Any) -> tuple[Any, Rec]:
    rec = rec or Rec()
    client = SimpleNamespace(files=files or Files())
    kwargs.setdefault("staging_volume", "cat.sch.vol")
    kwargs.setdefault("warn_on_use", False)
    return SqlEngine(backend=rec, client=client, **kwargs), rec


def ipc(table: Any) -> bytes:
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    return sink.getvalue()


def status(state: str, message: str | None = None) -> Any:
    error = SimpleNamespace(message=message, error_code=None) if message else None
    return SimpleNamespace(state=SimpleNamespace(value=state), error=error, sql_state=None)


class Statements:
    def __init__(
        self,
        responses: list[Any],
        polls: list[Any] | None = None,
        chunks: dict[int, Any] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.polls = list(polls or [])
        self.chunks = chunks or {}
        self.cancelled: list[str] = []
        self.chunk_requests: list[int] = []
        self.poll_error: BaseException | None = None

    def execute_statement(self, **kwargs: Any) -> Any:
        return self.responses.pop(0)

    def get_statement(self, statement_id: str) -> Any:
        if self.poll_error is not None:
            raise self.poll_error
        return self.polls.pop(0)

    def get_statement_result_chunk_n(self, statement_id: str, chunk_index: int) -> Any:
        self.chunk_requests.append(chunk_index)
        value = self.chunks[chunk_index]
        return value.pop(0) if isinstance(value, list) else value

    def cancel_execution(self, statement_id: str) -> None:
        self.cancelled.append(statement_id)


class Opener:
    def __init__(self, bodies: dict[str, bytes], fail: set[str] | None = None) -> None:
        self.bodies = bodies
        self.fail = fail or set()
        self.urls: list[str] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> Any:
        self.urls.append(request.full_url)
        if request.full_url in self.fail:
            raise urllib.error.HTTPError(request.full_url, 403, "expired", {}, None)  # type: ignore[arg-type]
        return io.BytesIO(self.bodies[request.full_url])


def link(url: str, *, chunk_index: int | None = None, next_chunk_index: int | None = None) -> Any:
    return SimpleNamespace(
        external_link=url,
        http_headers={},
        chunk_index=chunk_index,
        next_chunk_index=next_chunk_index,
    )


def sdk(statements: Statements, **kwargs: Any) -> SdkStatementBackend:
    kwargs.setdefault("sleep", lambda _s: None)
    return SdkStatementBackend(SimpleNamespace(statement_execution=statements), "wh", **kwargs)


# ------------------------------------------------------------ parameters


def test_decimal_with_positive_exponent_binds_its_real_precision() -> None:
    p = _parameter("p", decimal.Decimal("1E+5"))
    assert (p.value, p.type) == ("100000", "DECIMAL(6,0)")


def test_decimal_nan_is_refused_clearly() -> None:
    with pytest.raises(ValueError, match="finite"):
        _parameter("p", decimal.Decimal("NaN"))


def test_decimal_beyond_38_digits_is_refused_not_mis_typed() -> None:
    with pytest.raises(ValueError, match="38"):
        _parameter("p", decimal.Decimal("1." + "1" * 40))


def test_int_beyond_bigint_binds_as_decimal() -> None:
    p = _parameter("p", 2**70)
    assert p.type == "DECIMAL(22,0)"
    assert p.value == str(2**70)


def test_numpy_scalars_bind_as_their_python_values() -> None:
    np = pytest.importorskip("numpy")
    assert _parameter("p", np.int64(7)).type == "BIGINT"
    assert _parameter("p", np.bool_(True)).value == "true"
    ts = _parameter("p", np.datetime64("2024-01-02T03:04:05.123456789", "ns"))
    assert ts.type == "TIMESTAMP_NTZ"
    assert ts.value is not None
    assert ts.value.startswith("2024-01-02T03:04:05")


def test_float_specials_use_spark_spelling() -> None:
    assert _parameter("p", float("nan")).value == "NaN"
    assert _parameter("p", float("-inf")).value == "-Infinity"


# ------------------------------------------------------------- backend


def test_link_without_next_index_does_not_erase_the_chunks() -> None:
    a, b = pa.table({"x": [1]}), pa.table({"x": [2]})
    first = SimpleNamespace(chunk_index=0, next_chunk_index=1, external_links=[link("https://s/a")])
    second = SimpleNamespace(
        chunk_index=1, next_chunk_index=None, external_links=[link("https://s/b")]
    )
    resp = SimpleNamespace(
        statement_id="s1", status=status("SUCCEEDED"), manifest=None, result=first
    )
    opener = Opener({"https://s/a": ipc(a), "https://s/b": ipc(b)})
    got = sdk(Statements([resp], chunks={1: second}), opener=opener).execute("SELECT")
    assert got.column("x").to_pylist() == [1, 2]


def test_truncated_result_raises() -> None:
    manifest = SimpleNamespace(truncated=True, schema=None)
    resp = SimpleNamespace(
        statement_id="s1", status=status("SUCCEEDED"), manifest=manifest, result=None
    )
    with pytest.raises(SqlStatementError, match="truncated"):
        sdk(Statements([resp])).execute("SELECT")


def test_missing_first_chunk_is_fetched_from_the_manifest() -> None:
    manifest = SimpleNamespace(total_chunk_count=1, total_row_count=1, schema=None)
    chunk0 = SimpleNamespace(
        chunk_index=0, next_chunk_index=None, external_links=[link("https://s/a")]
    )
    resp = SimpleNamespace(
        statement_id="s1", status=status("SUCCEEDED"), manifest=manifest, result=None
    )
    statements = Statements([resp], chunks={0: chunk0})
    got = sdk(statements, opener=Opener({"https://s/a": ipc(pa.table({"x": [5]}))})).execute("S")
    assert got.num_rows == 1 and statements.chunk_requests == [0]


def test_row_count_mismatch_raises() -> None:
    manifest = SimpleNamespace(total_row_count=3, schema=None)
    first = SimpleNamespace(
        chunk_index=0, next_chunk_index=None, external_links=[link("https://s/a")]
    )
    resp = SimpleNamespace(
        statement_id="s1", status=status("SUCCEEDED"), manifest=manifest, result=first
    )
    with pytest.raises(SqlStatementError, match="lost"):
        sdk(Statements([resp]), opener=Opener({"https://s/a": ipc(pa.table({"x": [1]}))})).execute(
            "S"
        )


def test_repeated_chunk_raises_instead_of_silently_stopping() -> None:
    first = SimpleNamespace(chunk_index=0, next_chunk_index=0, external_links=[link("https://s/a")])
    resp = SimpleNamespace(
        statement_id="s1", status=status("SUCCEEDED"), manifest=None, result=first
    )
    opener = Opener({"https://s/a": ipc(pa.table({"x": [1]}))})
    with pytest.raises(SqlStatementError, match="twice"):
        sdk(Statements([resp], chunks={0: first}), opener=opener).execute("S")


def test_expired_link_is_refreshed_once() -> None:
    first = SimpleNamespace(
        chunk_index=0, next_chunk_index=None, external_links=[link("https://s/old")]
    )
    fresh = SimpleNamespace(
        chunk_index=0, next_chunk_index=None, external_links=[link("https://s/new")]
    )
    resp = SimpleNamespace(
        statement_id="s1", status=status("SUCCEEDED"), manifest=None, result=first
    )
    opener = Opener({"https://s/new": ipc(pa.table({"x": [1]}))}, fail={"https://s/old"})
    statements = Statements([resp], chunks={0: fresh})
    got = sdk(statements, opener=opener).execute("S")
    assert got.num_rows == 1
    assert opener.urls == ["https://s/old", "https://s/new"]


def test_poll_failure_cancels_the_running_statement() -> None:
    pending = SimpleNamespace(statement_id="s1", status=status("RUNNING"))
    statements = Statements([pending])
    statements.poll_error = ConnectionError("network down")
    with pytest.raises(ConnectionError):
        sdk(statements).execute("SELECT")
    assert statements.cancelled == ["s1"]


def test_timeout_cancels_exactly_once() -> None:
    pending = SimpleNamespace(statement_id="s1", status=status("PENDING"))
    clock = iter([0.0, 0.0, 100.0, 100.0, 100.0])
    statements = Statements([pending], polls=[pending] * 5)
    with pytest.raises(SqlStatementError, match="did not finish"):
        sdk(statements, timeout=10.0, clock=lambda: next(clock)).execute("S")
    assert statements.cancelled == ["s1"]


def test_no_statement_id_does_not_poll_none() -> None:
    pending = SimpleNamespace(statement_id=None, status=status("PENDING"))
    with pytest.raises(SqlStatementError, match="statement_id"):
        sdk(Statements([pending])).execute("S")


def test_sleep_never_overshoots_the_deadline() -> None:
    pending = SimpleNamespace(statement_id="s1", status=status("PENDING"))
    now = [0.0]
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    statements = Statements([pending], polls=[pending] * 50)
    with pytest.raises(SqlStatementError):
        sdk(
            statements,
            timeout=5.0,
            poll_interval=4.0,
            max_poll_interval=10.0,
            sleep=sleep,
            clock=lambda: now[0],
        ).execute("S")
    assert sum(sleeps) == pytest.approx(5.0)


def test_zero_poll_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        sdk(Statements([]), poll_interval=0)


def test_wait_timeout_accepts_int_seconds() -> None:
    b = sdk(Statements([]), wait_timeout=10)
    assert b._wait_timeout == "10s"


def test_empty_result_keeps_decimal_and_ntz_types() -> None:
    cols = [
        SimpleNamespace(
            name="d",
            position=0,
            type_name=SimpleNamespace(value="DECIMAL"),
            type_precision=10,
            type_scale=2,
            type_text="DECIMAL(10,2)",
        ),
        SimpleNamespace(name="t", position=1, type_name=None, type_text="TIMESTAMP_NTZ"),
        SimpleNamespace(
            name="n", position=2, type_name=SimpleNamespace(value="NULL"), type_text="VOID"
        ),
    ]
    manifest = SimpleNamespace(schema=SimpleNamespace(columns=cols))
    resp = SimpleNamespace(
        statement_id="s1", status=status("SUCCEEDED"), manifest=manifest, result=None
    )
    got = sdk(Statements([resp])).execute("S")
    assert got.schema.field("d").type == pa.decimal128(10, 2)
    assert got.schema.field("t").type == pa.timestamp("us")
    assert got.schema.field("n").type == pa.null()


# --------------------------------------------------------------- engine


def test_zorder_with_a_single_string_column() -> None:
    e, rec = eng()
    e.zorder(tbl(), "city")
    assert rec.last.endswith("ZORDER BY (`city`)")


def test_scan_columns_as_a_string() -> None:
    e, rec = eng()
    e.scan(tbl(), columns="city")
    assert rec.last == f"SELECT `city` FROM {NAME}"


def test_scan_version_and_timestamp_together_is_refused() -> None:
    e, _ = eng()
    with pytest.raises(ValueError, match="not both"):
        e.scan(tbl(), version=1, timestamp="2024-01-01")


def test_add_columns_from_an_arrow_struct_field() -> None:
    e, rec = eng()
    field = pa.field("s", pa.struct([("a b", pa.int32())]))
    e.add_columns(tbl(), [field])
    assert rec.last.endswith("ADD COLUMNS (`s` STRUCT<`a b`: INT>)")


def test_sql_type_still_rejects_injection_outside_backticks() -> None:
    with pytest.raises(ValueError):
        sqlmod._sql_type("STRUCT<`a`: INT>; DROP TABLE x")


def test_arrow_unsigned_and_dictionary_types_map() -> None:
    assert sqlmod._arrow_to_sql(pa.uint32()) == "BIGINT"
    assert sqlmod._arrow_to_sql(pa.uint64()) == "DECIMAL(20,0)"
    assert sqlmod._arrow_to_sql(pa.dictionary(pa.int32(), pa.string())) == "STRING"
    assert sqlmod._arrow_to_sql(pa.list_(pa.int8(), 3)) == "ARRAY<TINYINT>"


def test_arrow_decimal256_beyond_38_is_refused() -> None:
    with pytest.raises(UnreachableTableError, match="38"):
        sqlmod._arrow_to_sql(pa.decimal256(50, 2))


def test_merge_clauses_are_emitted_in_grammar_order() -> None:
    e, _ = eng()
    m = (
        e.merge(tbl(), pa.table({"id": [1]}), "target.id = source.id")
        .when_not_matched_insert_all()
        .when_matched_delete("source.gone")
        .when_matched_update_all()
    )
    sql = m.statement("rel", ["id"])
    i_matched_delete = sql.index("WHEN MATCHED AND source.gone THEN DELETE")
    i_matched_update = sql.index("WHEN MATCHED THEN UPDATE SET *")
    i_insert = sql.index("WHEN NOT MATCHED THEN INSERT *")
    assert i_matched_delete < i_matched_update < i_insert


def test_merge_empty_update_is_refused() -> None:
    e, _ = eng()
    m = e.merge(tbl(), pa.table({"id": [1]}), "target.id = source.id")
    # Refused when the clause is added (w2: before the source is staged).
    with pytest.raises(ValueError, match="at least one column"):
        m.when_matched_update({})


def test_merge_except_cols_is_case_insensitive() -> None:
    e, _ = eng()
    m = e.merge(tbl(), pa.table({"id": [1]}), "t").when_matched_update_all(except_cols=["ID"])
    sql = m.statement("rel", ["id", "v"])
    assert "`target`.`id`" not in sql and "`target`.`v` = `source`.`v`" in sql


def test_partition_filter_in_with_a_bare_string_is_refused() -> None:
    e, _ = eng()
    with pytest.raises(ValueError, match="list of values"):
        e.optimize(tbl(), partition_filters=[("city", "in", "abc")])


def test_overwrite_schema_merge_is_applied() -> None:
    e, rec = eng()
    e.overwrite(tbl(), pa.table({"id": [1]}), schema_mode="merge")
    assert rec.last.startswith(f"INSERT WITH SCHEMA EVOLUTION OVERWRITE {NAME} BY NAME")


def test_replace_where_matches_columns_case_insensitively() -> None:
    schema = pa.table({"ID": pa.array([], pa.int64()), "City": pa.array([], pa.string())})
    e, rec = eng(Rec({"SELECT * FROM": schema}))
    e.overwrite(tbl(), pa.table({"city": ["x"], "id": [1]}), predicate="id > 0")
    assert "SELECT `id`, `city` FROM read_files(" in rec.last


def test_replace_where_with_schema_merge_keeps_new_columns() -> None:
    schema = pa.table({"id": pa.array([], pa.int64())})
    e, rec = eng(Rec({"SELECT * FROM": schema}))
    e.overwrite(tbl(), pa.table({"id": [1], "new": [2]}), predicate="id > 0", schema_mode="merge")
    assert rec.last.startswith(f"INSERT WITH SCHEMA EVOLUTION INTO {NAME} REPLACE WHERE id > 0")
    assert "SELECT `id`, `new` FROM" in rec.last


def test_dynamic_overwrite_partition_column_case_insensitive() -> None:
    schema = pa.table({"region": pa.array([], pa.string()), "v": pa.array([], pa.int64())})
    e, rec = eng(Rec({"SELECT * FROM": schema}))
    e.overwrite(
        tbl(partition_columns=("Region",)),
        pa.table({"region": ["eu"], "v": [1]}),
        partition_overwrite="dynamic",
    )
    assert "REPLACE WHERE (`Region` = :p0)" in rec.last
    assert rec.params["p0"] == ("eu", "STRING")


def test_vacuum_for_real_does_not_report_the_table_path_as_deleted() -> None:
    e, _ = eng(Rec({"VACUUM": pa.table({"path": ["s3://bucket/t"]})}))
    assert e.vacuum(tbl(), dry_run=False) == []


def test_vacuum_dry_run_still_lists_files() -> None:
    e, _ = eng(Rec({"VACUUM": pa.table({"path": ["s3://bucket/t/a.parquet"]})}))
    assert e.vacuum(tbl(), dry_run=True) == ["s3://bucket/t/a.parquet"]


def test_describe_extended_partition_rows_do_not_leak_into_info() -> None:
    rows = pa.table(
        {
            "col_name": [
                "id",
                "region",
                "# Partition Information",
                "# col_name",
                "region",
                "",
                "# Detailed Table Information",
                "Name",
            ],
            "data_type": ["bigint", "string", "", "data_type", "string", "", "", "main.s.t"],
            "comment": [None] * 8,
        }
    )
    e, _ = eng(Rec({"DESCRIBE TABLE EXTENDED": rows}))
    out = e.describe_extended(tbl())
    assert "region" not in out and out["Name"] == "main.s.t"
    assert [c["name"] for c in out["columns"]] == ["id", "region"]


def test_failed_upload_cleans_up_the_partial_file() -> None:
    files = Files(fail_upload=True)
    e, _ = eng(files=files)
    with pytest.raises(OSError):
        e.append(tbl(), pa.table({"id": [1]}))
    assert len(files.deleted) == 1


def test_time_columns_are_refused_before_upload() -> None:
    import datetime as dt

    files = Files()
    e, _ = eng(files=files)
    with pytest.raises(UnreachableTableError, match="t"):
        e.append(tbl(), pa.table({"t": pa.array([dt.time(1, 2)])}))
    assert files.uploaded == {}


def test_http_path_with_query_string() -> None:
    e = SqlEngine(http_path="/sql/1.0/warehouses/abc123?o=42", backend=Rec(), warn_on_use=False)
    assert e.warehouse_id == "abc123"


def test_bad_wait_timeout_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="wait_timeout"):
        SqlEngine(warehouse_id="w", wait_timeout="99s")


def test_restore_numpy_integer_is_a_version() -> None:
    np = pytest.importorskip("numpy")
    e, rec = eng()
    e.restore(tbl(), np.int64(3))
    assert rec.last.endswith("TO VERSION AS OF 3")


def test_set_properties_none_value_is_refused() -> None:
    e, _ = eng()
    with pytest.raises(ValueError, match="None"):
        e.set_properties(tbl(), {"delta.appendOnly": None})


def test_variant_preview_feature_wire_name() -> None:
    feature = SimpleNamespace(value="TableFeatures.VariantTypePreview")
    assert sqlmod._feature_name(feature) == "variantType-preview"


def test_update_new_values_none_is_the_null_literal() -> None:
    e, rec = eng()
    e.update(tbl(), new_values={"n": None, "c": "x"})
    assert rec.last == f"UPDATE {NAME} SET `n` = NULL, `c` = :p0"
    assert rec.params == {"p0": ("x", "STRING")}


def test_update_expression_none_is_null() -> None:
    e, rec = eng()
    e.update(tbl(), updates={"n": None})
    assert rec.last == f"UPDATE {NAME} SET `n` = NULL"


def test_append_rows_as_dicts() -> None:
    files = Files()
    e, _ = eng(files=files)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e.append(tbl(), [{"id": 1}, {"id": 2}])
    assert len(files.uploaded) == 1


def test_binder_still_types_parameters() -> None:
    b = ParameterBinder()
    assert b.bind(decimal.Decimal("12.345")) == ":p0"
    assert b.parameters[0].type == "DECIMAL(5,3)"


def test_delete_accepts_tuning_and_refuses_unsupported_kwargs_clearly() -> None:
    e, rec = eng()
    e.delete(tbl(), "id = 1", max_concurrent_tasks=4)
    assert rec.last == f"DELETE FROM {NAME} WHERE id = 1"
    with pytest.raises(UnreachableTableError, match="writer_properties"):
        e.delete(tbl(), "id = 1", writer_properties=object())


def test_update_accepts_tuning_kwargs() -> None:
    e, rec = eng()
    e.update(tbl(), updates={"n": "n + 1"}, max_concurrent_tasks=4)
    assert rec.last == f"UPDATE {NAME} SET `n` = n + 1"


def test_merge_clause_none_value_is_null() -> None:
    e, _ = eng()
    m = e.merge(tbl(), pa.table({"id": [1]}), "t").when_not_matched_insert(
        {"id": "source.id", "v": None}
    )
    assert "INSERT (`id`, `v`) VALUES (source.id, NULL)" in m.statement("rel", ["id"])


def test_add_columns_mapping_with_arrow_types() -> None:
    e, rec = eng()
    e.add_columns(tbl(), {"n": pa.int32(), "s": "STRING"})
    assert rec.last.endswith("ADD COLUMNS (`n` INT, `s` STRING)")


def test_set_tags_none_value_is_empty_not_the_text_none() -> None:
    e, rec = eng()
    e.set_tags(tbl(), {"pii": None})
    assert rec.last.endswith("SET TAGS ('pii' = '')")


def test_staging_volume_with_a_blank_part_is_refused() -> None:
    with pytest.raises(ValueError):
        SqlEngine(staging_volume="cat.` `.vol", backend=Rec())


def test_http_path_without_an_id_is_refused() -> None:
    with pytest.raises(ValueError, match="HTTP path"):
        SqlEngine(http_path="/sql/1.0/warehouses/", backend=Rec())


def test_transient_warehouse_listing_failure_is_not_cached() -> None:
    class Warehouses:
        calls = 0

        def list(self) -> list[Any]:
            Warehouses.calls += 1
            if Warehouses.calls == 1:
                raise ConnectionError("blip")
            return [SimpleNamespace(id="w1", name="w", state=SimpleNamespace(value="RUNNING"))]

    e = SqlEngine(client=SimpleNamespace(warehouses=Warehouses()), warn_on_use=False)
    assert e.warehouse_id is None
    assert e.warehouse_id == "w1"


def test_rows_reported_but_no_data_raises() -> None:
    manifest = SimpleNamespace(total_row_count=2, schema=None)
    resp = SimpleNamespace(
        statement_id="s1", status=status("SUCCEEDED"), manifest=manifest, result=None
    )
    with pytest.raises(SqlStatementError, match="lost"):
        sdk(Statements([resp])).execute("S")


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_predicate_never_means_every_row(blank: str) -> None:
    e, rec = eng()
    with pytest.raises(ValueError, match="empty"):
        e.delete(tbl(), blank)
    with pytest.raises(ValueError, match="empty"):
        e.update(tbl(), updates={"n": "1"}, predicate=blank)
    with pytest.raises(ValueError, match="empty"):
        e.scan(tbl(), predicate=blank)
    with pytest.raises(ValueError, match="empty"):
        e.overwrite(tbl(), pa.table({"id": [1]}), predicate=blank)
    assert rec.calls == []


def test_delete_with_none_still_deletes_every_row() -> None:
    e, rec = eng()
    e.delete(tbl(), None)
    assert rec.last == f"DELETE FROM {NAME}"


def test_history_limit_zero_is_empty_not_unlimited() -> None:
    e, rec = eng()
    assert e.history(tbl(), limit=0) == []
    assert rec.calls == []
    e.history(tbl(), limit=2)
    assert rec.last == f"DESCRIBE HISTORY {NAME} LIMIT 2"
    with pytest.raises(ValueError):
        e.history(tbl(), limit=-1)
