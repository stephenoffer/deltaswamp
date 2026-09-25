"""Second-pass regression tests for the SQL warehouse fallback (w2_sql).

Fakes only: the recording backend, fake files API and scripted
`statement_execution` from the first pass's test module.
"""

from __future__ import annotations

import datetime as dt
import io
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from deltaswamp.engine import sql as sqlmod  # noqa: E402
from deltaswamp.engine.sql import SqlEngine  # noqa: E402
from deltaswamp.engine.sql_backend import (  # noqa: E402
    SdkStatementBackend,
    _parameter,
    parameters_from_mapping,
)
from deltaswamp.errors import UnreachableTableError  # noqa: E402

from tests.unit.test_bughunt_sql import (  # noqa: E402
    NAME,
    Files,
    Rec,
    Statements,
    eng,
    sdk,
    status,
    tbl,
)


def staged_path(files: Files) -> str:
    (path,) = files.uploaded
    return path


def staged_schema(files: Files) -> Any:
    return pq.read_schema(io.BytesIO(files.uploaded[staged_path(files)]))


# ------------------------------------------------ _rescued_data / staging


def test_append_projects_the_staged_columns_not_star() -> None:
    e, rec = eng()
    e.append(tbl(), pa.table({"id": [1], "city": ["x"]}))
    assert "SELECT * FROM read_files" not in rec.last
    assert rec.last.startswith(f"INSERT INTO {NAME} BY NAME SELECT `id`, `city` FROM read_files(")


def test_static_overwrite_projects_the_staged_columns() -> None:
    e, rec = eng()
    e.overwrite(tbl(), pa.table({"id": [1]}), schema_mode="merge")
    assert rec.last.startswith(
        f"INSERT WITH SCHEMA EVOLUTION OVERWRITE {NAME} BY NAME SELECT `id` FROM read_files("
    )


def test_merge_source_projects_the_staged_columns() -> None:
    e, rec = eng()
    e.merge(
        tbl(), pa.table({"id": [1], "v": [2]}), "target.id = source.id", merge_schema=True
    ).when_matched_update_all().when_not_matched_insert_all().execute()
    assert "USING (SELECT `id`, `v` FROM read_files(" in rec.last
    assert "SELECT *" not in rec.last


def test_float16_is_staged_as_float32() -> None:
    files = Files()
    e, _ = eng(files=files)
    data = pa.table(
        {
            "h": pa.array([1.5], pa.float16()),
            "s": pa.array([{"x": 1.5}], pa.struct([("x", pa.float16())])),
        }
    )
    e.append(tbl(), data)
    schema = staged_schema(files)
    assert schema.field("h").type == pa.float32()
    assert schema.field("s").type == pa.struct([("x", pa.float32())])


def test_extension_types_are_staged_as_their_storage() -> None:
    if not hasattr(pa, "uuid"):
        pytest.skip("pyarrow without the uuid extension type")
    files = Files()
    e, _ = eng(files=files)
    data = pa.table({"u": pa.array([uuid.uuid4().bytes], pa.binary(16)).cast(pa.uuid())})
    e.append(tbl(), data)
    assert staged_schema(files).field("u").type == pa.binary(16)
    assert sqlmod._arrow_to_sql(pa.uuid()) == "BINARY"


def test_decimal_beyond_38_is_refused_before_upload() -> None:
    files = Files()
    e, _ = eng(files=files)
    data = pa.table({"d": pa.array([1], pa.int64()).cast(pa.decimal256(50, 0))})
    with pytest.raises(UnreachableTableError, match="38 digits"):
        e.append(tbl(), data)
    assert not files.uploaded


def test_case_insensitive_duplicate_columns_are_refused_before_upload() -> None:
    files = Files()
    e, _ = eng(files=files)
    with pytest.raises(UnreachableTableError, match="same column"):
        e.append(tbl(), pa.table({"id": [1], "ID": [2]}))
    assert not files.uploaded


def test_data_without_columns_is_refused() -> None:
    files = Files()
    e, _ = eng(files=files)
    with pytest.raises(UnreachableTableError, match="no columns"):
        e.append(tbl(), pa.table({}))
    assert not files.uploaded


def test_pandas_named_index_is_kept_as_a_column() -> None:
    pd = pytest.importorskip("pandas")
    files = Files()
    e, rec = eng(files=files)
    e.append(tbl(), pd.DataFrame({"id": [1, 2], "v": [3, 4]}).set_index("id"))
    assert "id" in staged_schema(files).names
    assert "`id`" in rec.last


def test_pandas_unnamed_index_is_still_dropped() -> None:
    pd = pytest.importorskip("pandas")
    files = Files()
    e, _ = eng(files=files)
    e.append(tbl(), pd.DataFrame({"v": [3, 4]}, index=[7, 9]))
    assert staged_schema(files).names == ["v"]


# ------------------------------------------------------ empty-result types


def col(name: str, pos: int, type_text: str, type_name: str | None = None) -> Any:
    return SimpleNamespace(
        name=name,
        position=pos,
        type_name=SimpleNamespace(value=type_name) if type_name else None,
        type_text=type_text,
    )


def empty_result(*cols: Any) -> Any:
    manifest = SimpleNamespace(schema=SimpleNamespace(columns=list(cols)))
    resp = SimpleNamespace(
        statement_id="s1", status=status("SUCCEEDED"), manifest=manifest, result=None
    )
    return sdk(Statements([resp])).execute("S")


def test_empty_result_parses_nested_types() -> None:
    got = empty_result(
        col("a", 0, "ARRAY<STRUCT<a: INT, `B c`: STRING NOT NULL>>", "ARRAY"),
        col("m", 1, "map<string,decimal(10,2)>", "MAP"),
        col("s", 2, "struct<x:array<timestamp_ntz>,y:bigint COMMENT 'hi'>", "STRUCT"),
    )
    assert got.schema.field("a").type == pa.list_(
        pa.struct([pa.field("a", pa.int32()), pa.field("B c", pa.string(), nullable=False)])
    )
    assert got.schema.field("m").type == pa.map_(pa.string(), pa.decimal128(10, 2))
    assert got.schema.field("s").type == pa.struct(
        [("x", pa.list_(pa.timestamp("us"))), ("y", pa.int64())]
    )


def test_empty_result_names_from_type_text_only() -> None:
    got = empty_result(
        col("t", 0, "TINYINT"),
        col("sm", 1, "SMALLINT"),
        col("b", 2, "BIGINT"),
        col("f", 3, "FLOAT"),
        col("bin", 4, "BINARY"),
        col("v", 5, "VARIANT"),
        col("c", 6, "VARCHAR(10)"),
    )
    assert [f.type for f in got.schema] == [
        pa.int8(),
        pa.int16(),
        pa.int64(),
        pa.float32(),
        pa.binary(),
        pa.string(),
        pa.string(),
    ]


def test_empty_result_day_time_interval_is_a_duration() -> None:
    got = empty_result(col("i", 0, "INTERVAL DAY TO SECOND", "INTERVAL"))
    assert got.schema.field("i").type == pa.duration("us")


def test_empty_result_unparseable_type_text_is_still_a_column() -> None:
    got = empty_result(col("x", 0, "WEIRD<<", "STRING"))
    assert got.schema.field("x").type == pa.string()


# ------------------------------------------------------ backend timing


class Recording(Statements):
    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.executed: list[dict[str, Any]] = []

    def execute_statement(self, **kwargs: Any) -> Any:
        self.executed.append(kwargs)
        return super().execute_statement(**kwargs)


def test_wait_timeout_never_outlasts_the_overall_timeout() -> None:
    done = SimpleNamespace(
        statement_id="s1", status=status("SUCCEEDED"), manifest=None, result=None
    )
    st = Recording([done, done])
    sdk(st, timeout=10.0, wait_timeout="30s").execute("S", fetch=False)
    assert st.executed[0]["wait_timeout"] == "10s"
    st2 = Recording([done])
    sdk(st2, timeout=2.0, wait_timeout="30s").execute("S", fetch=False)
    assert st2.executed[0]["wait_timeout"] == "0s"


def test_deadline_includes_the_blocking_first_call() -> None:
    now = [0.0]
    pending = SimpleNamespace(statement_id="s1", status=status("RUNNING"))

    class Slow(Statements):
        def execute_statement(self, **kwargs: Any) -> Any:
            now[0] += 9.0  # the server held the call for 9s
            return super().execute_statement(**kwargs)

    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    st = Slow([pending], polls=[pending] * 20)
    with pytest.raises(Exception, match="did not finish"):
        sdk(st, timeout=10.0, sleep=sleep, clock=lambda: now[0]).execute("S")
    assert sum(sleeps) == pytest.approx(1.0)
    assert st.cancelled == ["s1"]


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_nonsense_timeouts_are_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        SdkStatementBackend(SimpleNamespace(), "wh", timeout=bad)
    with pytest.raises(ValueError, match="timeout"):
        SqlEngine(warehouse_id="wh", timeout=bad)


# ------------------------------------------------------------ parameters


def test_parameter_names_may_carry_the_colon() -> None:
    (p,) = parameters_from_mapping({":id": 5})
    assert (p.name, p.value, p.type) == ("id", "5", "BIGINT")


def test_pandas_missing_values_bind_as_null() -> None:
    pd = pytest.importorskip("pandas")
    assert _parameter("p", pd.NaT).value is None
    assert _parameter("p", pd.NA).value is None


# --------------------------------------------------------- time travel


def test_naive_timestamps_travel_in_utc_not_the_session_zone() -> None:
    e, rec = eng()
    e.scan(tbl(), timestamp=dt.datetime(2024, 1, 2, 3, 4, 5))
    assert rec.params["p0"] == ("2024-01-02T03:04:05+00:00", "TIMESTAMP")
    e.scan(tbl(), timestamp="2024-01-02")
    assert rec.params["p0"] == ("2024-01-02T00:00:00+00:00", "TIMESTAMP")
    e.clone(tbl(), "a.b.c", timestamp=dt.date(2024, 1, 2))
    assert rec.last.endswith("TIMESTAMP AS OF '2024-01-02T00:00:00+00:00'")


def test_aware_timestamps_keep_their_offset() -> None:
    e, rec = eng()
    tz = dt.timezone(dt.timedelta(hours=5, minutes=30))
    e.scan(tbl(), timestamp=dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=tz))
    assert rec.params["p0"] == ("2024-01-02T03:04:05+05:30", "TIMESTAMP")


def test_epoch_millisecond_timestamps_are_utc() -> None:
    e, rec = eng()
    e.cdf(tbl(), starting_timestamp=86_400_000)
    assert rec.params["p0"] == ("1970-01-02T00:00:00+00:00", "TIMESTAMP")


def test_clone_with_version_and_timestamp_is_refused() -> None:
    e, rec = eng()
    with pytest.raises(ValueError, match="not both"):
        e.clone(tbl(), "a.b.c", version=1, timestamp="2024-01-01")
    assert not rec.calls


# ------------------------------------------------------------ properties


def test_boolean_property_values_are_lowercase() -> None:
    e, rec = eng()
    e.set_properties(tbl(), {"delta.appendOnly": True, "delta.enableChangeDataFeed": False})
    assert "'delta.appendOnly' = 'true'" in rec.last
    assert "'delta.enableChangeDataFeed' = 'false'" in rec.last


def test_create_table_as_properties_none_and_bool() -> None:
    e, rec = eng()
    with pytest.raises(ValueError, match="None"):
        e.create_table_as("a.b.c", "SELECT 1", properties={"k": None})
    assert not rec.calls
    e.create_table_as("a.b.c", "SELECT 1", properties={"delta.appendOnly": True})
    assert "TBLPROPERTIES ('delta.appendOnly' = 'true')" in rec.last


def test_type_widening_preview_feature_wire_name() -> None:
    feature = SimpleNamespace(value="TableFeatures.TypeWideningPreview")
    assert sqlmod._feature_name(feature) == "typeWidening-preview"


def test_add_feature_accepts_a_set_of_features() -> None:
    e, rec = eng()
    e.add_feature(tbl(), {"deletionVectors"})
    assert "'delta.feature.deletionVectors' = 'supported'" in rec.last


# ------------------------------------------------------------------ ddl


def test_alter_column_type_accepts_arrow_types() -> None:
    e, rec = eng()
    e.alter_column_type(tbl(), "n", pa.int64())
    assert rec.last.endswith("ALTER COLUMN `n` TYPE BIGINT")
    e.alter_column_type(tbl(), "n", pa.field("n", pa.decimal128(12, 2)))
    assert rec.last.endswith("ALTER COLUMN `n` TYPE DECIMAL(12,2)")


def test_add_columns_mapping_with_arrow_fields() -> None:
    e, rec = eng()
    e.add_columns(tbl(), {"n": pa.field("n", pa.int16())})
    assert rec.last.endswith("ADD COLUMNS (`n` SMALLINT)")


def test_add_constraint_empty_or_blank_is_refused() -> None:
    e, rec = eng()
    with pytest.raises(ValueError, match="no constraints"):
        e.add_constraint(tbl(), {})
    with pytest.raises(ValueError):
        e.add_constraint(tbl(), {"ok": "x > 0", "bad": "  "})
    with pytest.raises(ValueError):
        e.add_constraint(tbl(), {"bad": None})
    assert not rec.calls  # nothing half-applied


def test_repair_does_not_report_the_file_none() -> None:
    rows = pa.table({"dataFilePath": ["f1", None], "dataFileMissing": [True, False]})
    e, _ = eng(Rec({"FSCK": rows}))
    assert e.repair(tbl(), dry_run=True)["files_removed"] == ["f1"]


# ---------------------------------------------------------------- merge


def test_merge_without_a_predicate_is_refused_before_staging() -> None:
    files = Files()
    e, _ = eng(files=files)
    with pytest.raises(ValueError, match="join predicate"):
        e.merge(tbl(), pa.table({"id": [1]}), "  ")
    assert not files.uploaded


def test_merge_blank_clause_predicate_is_refused_when_added() -> None:
    files = Files()
    e, _ = eng(files=files)
    m = e.merge(tbl(), pa.table({"id": [1]}), "target.id = source.id")
    with pytest.raises(ValueError, match="empty"):
        m.when_matched_delete("")
    assert not files.uploaded


def test_merge_same_aliases_are_refused() -> None:
    e, _ = eng()
    with pytest.raises(ValueError, match="aliases"):
        e.merge(tbl(), pa.table({"id": [1]}), "a.id = a.id", source_alias="a", target_alias="A")


def test_merge_keys_qualified_with_the_target_alias() -> None:
    e, _ = eng()
    m = (
        e.merge(
            tbl(),
            pa.table({"id": [1], "v": [2]}),
            "t.id = s.id",
            source_alias="s",
            target_alias="t",
        )
        .when_matched_update({"t.v": "s.v"})
        .when_not_matched_insert({"t.id": "s.id", "v": "s.v"})
    )
    sql = m.statement("rel", ["id", "v"])
    assert "UPDATE SET `t`.`v` = s.v" in sql
    assert "INSERT (`id`, `v`) VALUES (s.id, s.v)" in sql


# ------------------------------------------------------------ misc args


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_vacuum_nonsense_retention_is_refused(bad: float) -> None:
    e, rec = eng()
    with pytest.raises(ValueError, match="retention_hours"):
        e.vacuum(tbl(), retention_hours=bad)
    assert not rec.calls


def test_merge_except_cols_as_a_single_string() -> None:
    e, _ = eng()
    m = e.merge(tbl(), pa.table({"id": [1], "ts": [2], "t": [3]}), "target.id = source.id")
    sql = m.when_matched_update_all(except_cols="ts").statement("rel", ["id", "ts", "t"])
    # list("ts") used to exclude the columns "t" and "s" instead.
    assert "`target`.`t` = `source`.`t`" in sql
    assert "`ts`" not in sql.split("THEN", 1)[1]


def test_update_same_column_in_both_mappings_is_refused() -> None:
    e, rec = eng()
    with pytest.raises(ValueError, match="both"):
        e.update(tbl(), updates={"n": "n + 1"}, new_values={"N": 3})
    assert not rec.calls
