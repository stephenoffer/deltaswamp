"""Regressions for the fifth live Databricks verification, reproduced without a warehouse.

Each reproduces a read, a write or a can() verdict a real workspace
disagreed with; the warehouse's side is a fake where one is needed.
"""

from __future__ import annotations

import datetime as dt
import io
import os
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
ds = pytest.importorskip("deltaswamp")

from deltaswamp.capability import Capability, Operation  # noqa: E402
from deltaswamp.capability import Engine as EngineKind  # noqa: E402
from deltaswamp.catalog.base import ResolvedTable  # noqa: E402
from deltaswamp.errors import (  # noqa: E402
    ChangeFeedSchemaChangeError,
    DeltaSwampError,
    InvalidArgumentError,
    InvalidReferenceError,
    PreflightError,
    TableNotFoundError,
    UnreachableTableError,
)
from deltaswamp.identity import parse_ref  # noqa: E402
from deltaswamp.table import Table  # noqa: E402

from tests.helpers import resolved_table  # noqa: E402
from tests.integration.test_audit_live import _write_log  # noqa: E402
from tests.unit.sql_fakes import RecordingBackend, engine  # noqa: E402


@pytest.fixture
def conn() -> Any:
    return ds.connect("file://")


def _field(name: str, kind: Any, **metadata: Any) -> dict[str, Any]:
    return {"name": name, "type": kind, "nullable": True, "metadata": metadata}


_PLAIN = {"minReaderVersion": 1, "minWriterVersion": 2}


def _describe(**types: str) -> Any:
    rows = [{"col_name": k, "data_type": v, "comment": None} for k, v in types.items()]
    return pa.Table.from_pylist(rows)


def _staged(client: Any) -> Any:
    (payload,) = client.files.uploaded.values()
    return pq.read_table(io.BytesIO(payload))


# ------------------------------------------------------------ #1 intervals


def _interval_table(conn: Any, tmp_path: Any) -> Any:
    schema = {
        "type": "struct",
        "fields": [
            _field("id", "long"),
            _field("i", "interval day to second"),
            _field("ym", "interval year to month"),
            _field("y", "interval year"),
            _field("s", {"type": "struct", "fields": [_field("x", "interval hour to minute")]}),
            _field("a", {"type": "array", "elementType": "interval month", "containsNull": True}),
        ],
    }
    # What Databricks writes: microseconds as INT64, months as INT32.
    data = pa.table(
        {
            "id": pa.array([1, 2], pa.int64()),
            "i": pa.array([93_784_500_000, None], pa.int64()),
            "ym": pa.array([-14, None], pa.int32()),
            "y": pa.array([36, None], pa.int32()),
            "s": pa.array([{"x": 11_040_000_000}, None], pa.struct([("x", pa.int64())])),
            "a": pa.array([[14, -1], None], pa.list_(pa.int32())),
        }
    )
    path = str(tmp_path / "intervals")
    _write_log(path, schema, _PLAIN, data)
    return conn.open_table(path)


class TestIntervals:
    """#1: an INTERVAL DAY TO SECOND read as bare microseconds, and written back as seconds."""

    def test_reads_give_durations_and_year_month_text(self, conn: Any, tmp_path: Any) -> None:
        t = _interval_table(conn, tmp_path)
        schema = t.schema()
        assert schema.field("i").type == pa.duration("us")
        assert schema.field("s").type == pa.struct([("x", pa.duration("us"))])
        assert schema.field("ym").type == pa.string()
        got = pa.table(t.to_arrow()).sort_by("id").to_pylist()
        assert got[0]["i"] == dt.timedelta(days=1, hours=2, minutes=3, seconds=4.5)
        assert got[0]["s"] == {"x": dt.timedelta(hours=3, minutes=4)}
        # Spark's own text for each qualifier, as the warehouse's cast prints it.
        assert got[0]["ym"] == "INTERVAL '-1-2' YEAR TO MONTH"
        assert got[0]["y"] == "INTERVAL '3' YEAR"
        assert got[0]["a"] == ["INTERVAL '14' MONTH", "INTERVAL '-1' MONTH"]
        assert got[1]["i"] is None and got[1]["ym"] is None

    def test_a_read_appends_back_unchanged(self, conn: Any, tmp_path: Any) -> None:
        t = _interval_table(conn, tmp_path)
        before = pa.table(t.to_arrow()).sort_by("id")
        t.append(before.set_column(0, "id", pa.array([3, 4], pa.int64())))
        after = pa.table(t.to_arrow()).sort_by("id").to_pylist()
        for old, new in zip(before.to_pylist(), after[2:], strict=True):
            assert {k: v for k, v in new.items() if k != "id"} == {
                k: v for k, v in old.items() if k != "id"
            }

    def test_bare_integers_are_refused_for_a_day_time_column(
        self, conn: Any, tmp_path: Any
    ) -> None:
        t = _interval_table(conn, tmp_path)
        with pytest.raises(InvalidArgumentError, match="duration"):
            t.append(pa.table({"id": pa.array([5], pa.int64()), "i": pa.array([5], pa.int64())}))

    def test_a_staged_duration_is_multiplied_back_into_an_interval(self) -> None:
        # A BIGINT cast to INTERVAL DAY TO SECOND counts seconds: staged bare,
        # 1 day came back as 1085462 days.
        eng, rec, client = engine()
        data = pa.table(
            {"id": [1], "i": pa.array([dt.timedelta(days=1, microseconds=5)], pa.duration("us"))}
        )
        eng.append(resolved_table(), data)
        staged = _staged(client)
        assert staged.column("i").type == pa.int64()
        assert staged.column("i").to_pylist() == [86_400_000_005]
        insert = next(s for s in rec.sql if s.startswith("INSERT"))
        assert "`i` * INTERVAL '0.000001' SECOND AS `i`" in insert

    def test_a_nanosecond_duration_is_staged_as_microseconds(self) -> None:
        eng, _, client = engine()
        data = pa.table({"i": pa.array([1_000], pa.duration("ns"))})
        eng.append(resolved_table(), data)
        assert _staged(client).column("i").to_pylist() == [1]

    def test_year_month_text_parses_back_to_months(self) -> None:
        from deltaswamp.engine.intervals import storage_columns, year_month_text

        months = pa.array([-14, 0, 25, None], pa.int32())
        for qualifier in ("YEAR TO MONTH", "MONTH"):
            text = year_month_text(pa, months, qualifier)
            back = storage_columns(
                pa, pa.table({"m": text}), {qualifier: frozenset({("m",)})}
            ).column("m")
            assert back.to_pylist() == months.to_pylist()


# ------------------------------------------------- #3 / #10 change feed


class _Warehouse:
    """A SQL engine that serves the change feed, recording that it did."""

    kind = EngineKind.SQL

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def available(self) -> bool:
        return True

    def supports(self, op: Any, table: Any, **_: Any) -> Capability:
        return Capability(op, ok=True, engine=EngineKind.SQL)

    def cdf(self, table: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return pa.table(
            {
                "id": pa.array([2], pa.int64()),
                "_change_type": ["insert"],
                "_commit_version": pa.array([3], pa.int64()),
                "_commit_timestamp": pa.array([0], pa.timestamp("us", tz="UTC")),
            }
        )


class _Catalog:
    name = "fake"

    def __init__(self, path: str) -> None:
        self.path = path
        self.gone = False

    def resolve(self, ref: Any) -> ResolvedTable:
        if self.gone:
            raise TableNotFoundError(f"{ref} does not exist")
        return ResolvedTable(ref=ref, location=self.path)


def _catalog_table(conn: Any, path: str, *, warehouse: Any = None) -> Any:
    catalog = _Catalog(path)
    conn.catalog = catalog
    if warehouse is not None:
        conn.router.engines[EngineKind.SQL] = warehouse
        conn.router.allow_sql_fallback = True
        conn.router.warehouse_catalog = True
    return Table(conn, ResolvedTable(ref=parse_ref("main.s.t"), location=path)), catalog


def _renamed(conn: Any, path: str, *, cdf: bool) -> None:
    props = {"delta.columnMapping.mode": "name"}
    if cdf:
        props["delta.enableChangeDataFeed"] = "true"
    t = conn.create_table(
        path, pa.schema([("id", pa.int64()), ("city", pa.string())]), properties=props
    )
    t.append(pa.table({"id": [1], "city": ["a"]}))
    t.rename_column("city", "town")
    t.append(pa.table({"id": [2], "town": ["b"]}))


class TestChangeFeedAcrossSchemaChanges:
    def test_a_rename_the_kernel_stops_at_goes_to_the_warehouse(
        self, conn: Any, tmp_path: Any
    ) -> None:
        # #3: can(CDF) and the warehouse's table_changes() both serve a range
        # across a column-mapping RENAME; the call raised the kernel's limit.
        path = str(tmp_path / "cm")
        _renamed(conn, path, cdf=True)
        warehouse = _Warehouse()
        t, _ = _catalog_table(conn, path, warehouse=warehouse)
        got = pa.table(t.cdf(starting_version=1))
        assert warehouse.calls and got.column("id").to_pylist() == [2]

    def test_without_a_warehouse_the_schema_change_is_still_refused(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "cm")
        _renamed(conn, path, cdf=True)
        with pytest.raises(ChangeFeedSchemaChangeError):
            conn.open_table(path).cdf(starting_version=1)

    def test_a_table_without_a_feed_is_refused_as_can_refuses_it(
        self, conn: Any, tmp_path: Any
    ) -> None:
        # #3 (direct): can() named the disabled feed, the call a schema change.
        path = str(tmp_path / "nocdf")
        _renamed(conn, path, cdf=False)
        t = conn.open_table(path)
        verdict = t.can(Operation.CDF, starting_version=1)
        assert not verdict.ok and "enableChangeDataFeed" in verdict.reason
        with pytest.raises(UnreachableTableError, match="enableChangeDataFeed") as info:
            t.cdf(starting_version=1)
        assert not isinstance(info.value, ChangeFeedSchemaChangeError)

    def test_a_change_undone_by_restore_is_typed(self, conn: Any, tmp_path: Any) -> None:
        # #10 / LV-04: ADD COLUMN, then RESTORE to before it: the ends of the
        # range agree, and pa.table(t.cdf()) raised a bare ArrowInvalid.
        path = str(tmp_path / "aba")
        schema = pa.schema([("id", pa.int64()), ("s", pa.string())])
        t = conn.create_table(path, schema, properties={"delta.enableChangeDataFeed": "true"})
        t.append(pa.table({"id": [1], "s": ["a"]}))
        t.add_column([pa.field("n", pa.int32())])
        t.append(pa.table({"id": [2], "s": ["b"], "n": pa.array([2], pa.int32())}))
        t.restore(1)
        t.append(pa.table({"id": [3], "s": ["c"]}))
        with pytest.raises(ChangeFeedSchemaChangeError) as info:
            pa.table(conn.open_table(path).cdf(starting_version=0))
        assert info.value.version == 4


# ----------------------------------------------------------- #9 dropped


def test_a_read_through_a_handle_whose_table_was_dropped_fails(
    conn: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    import deltaswamp.table as table_module

    monkeypatch.setattr(table_module, "_NAME_RECHECK_SECONDS", 0.0)
    path = str(tmp_path / "dropped")
    conn.create_table(path, pa.schema([("id", pa.int64())])).append(pa.table({"id": [1]}))
    t, catalog = _catalog_table(conn, path)
    assert pa.table(t.to_arrow()).num_rows == 1
    catalog.gone = True
    with pytest.raises(TableNotFoundError):
        t.to_arrow()
    with pytest.raises(TableNotFoundError):
        t.count()


def test_reads_recheck_the_name_at_most_once_a_window(
    conn: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    """Each read cost a catalog round trip; reads within a window share one."""
    import deltaswamp.table as table_module

    clock = [1000.0]
    monkeypatch.setattr("deltaswamp.table.time.monotonic", lambda: clock[0])
    path = str(tmp_path / "window")
    conn.create_table(path, pa.schema([("id", pa.int64())])).append(pa.table({"id": [1]}))
    t, catalog = _catalog_table(conn, path)
    t.to_arrow()
    catalog.gone = True
    clock[0] += table_module._NAME_RECHECK_SECONDS / 2
    assert pa.table(t.to_arrow()).num_rows == 1  # trusted within the window
    clock[0] += table_module._NAME_RECHECK_SECONDS
    with pytest.raises(TableNotFoundError):
        t.to_arrow()


# ------------------------------------------------------ #4 / #7 Iceberg


def test_iceberg_refuses_a_table_with_a_variant() -> None:
    from deltaswamp.engine.iceberg import IcebergEngine

    if not IcebergEngine.available():
        pytest.skip("pyiceberg not installed")
    features = frozenset({"variantType", "icebergCompatV3", "columnMapping"})
    table = resolved_table(
        iceberg_rest_uri="https://h/api/2.1/unity-catalog/iceberg-rest",
        reader_features=features,
        writer_features=features,
    )
    verdict = IcebergEngine().supports(Operation.SCAN, table)
    assert not verdict.ok and "VARIANT" in verdict.reason
    plain = resolved_table(
        iceberg_rest_uri="https://h/api/2.1/unity-catalog/iceberg-rest",
        writer_features=frozenset({"icebergCompatV2"}),
    )
    assert IcebergEngine().supports(Operation.SCAN, plain).ok


def test_a_shallow_clone_of_managed_iceberg_is_refused() -> None:
    eng, _, _ = engine()
    managed = resolved_table(writer_features=frozenset({"icebergWriterCompatV1"}))
    verdict = eng.supports(Operation.CLONE, managed, target="main.s.c")
    assert not verdict.ok and "MANAGED_ICEBERG_OPERATION_NOT_SUPPORTED" in verdict.reason
    assert eng.supports(Operation.CLONE, managed, target="main.s.c", shallow=False).ok


# ----------------------------------------------------------- #5 geospatial


def test_geospatial_values_are_staged_as_the_column_type() -> None:
    # Databricks casts neither text nor even a NULL string to GEOMETRY(4326),
    # so every append failed, the geo column left out or not.
    geo: dict[str, Any] = {"writer_features": frozenset({"geospatial"})}
    describe = _describe(id="int", g="geometry(4326)", h="geography(4326)")
    eng, rec, _ = engine(RecordingBackend({"DESCRIBE TABLE": describe}))
    data = pa.table(
        {
            "id": pa.array([1], pa.int32()),
            "g": ["SRID=4326;POINT(1 2)"],
            "h": pa.array([None], pa.string()),
        }
    )
    eng.append(resolved_table(**geo), data)
    insert = next(s for s in rec.sql if s.startswith("INSERT"))
    assert "st_geomfromtext(regexp_replace(CAST(`g` AS STRING), '^SRID=4326;', ''), 4326)" in insert
    assert "st_geogfromtext(regexp_replace(CAST(`h` AS STRING), '^SRID=4326;', ''))" in insert


def test_a_geospatial_refusal_names_the_feature() -> None:
    from deltaswamp.router import Router

    from tests.helpers import FakeEngine

    router = Router(engines={EngineKind.KERNEL: FakeEngine(EngineKind.KERNEL)})
    table = resolved_table(
        open_error="InvalidArgumentError: Unsupported Delta table type: 'geometry(OGC:CRS84)'"
    )
    verdict = router.capability(Operation.SCAN, table)
    assert not verdict.ok and "GEOMETRY" in verdict.reason


# ------------------------------------------------------------- #6 identity


def _identity_table(conn: Any, tmp_path: Any) -> Any:
    schema = {
        "type": "struct",
        "fields": [
            _field(
                "id",
                "long",
                **{
                    "delta.identity.start": 1,
                    "delta.identity.step": 1,
                    "delta.identity.allowExplicitInsert": False,
                },
            ),
            _field("v", "string"),
        ],
    }
    protocol = {"minReaderVersion": 1, "minWriterVersion": 7, "writerFeatures": ["identityColumns"]}
    path = str(tmp_path / "identity")
    _write_log(path, schema, protocol, pa.table({"id": [1], "v": ["a"]}))
    return conn.open_table(path)


def test_can_refuses_values_for_a_generated_always_identity(conn: Any, tmp_path: Any) -> None:
    t = _identity_table(conn, tmp_path)
    given = pa.table({"id": [10], "v": ["x"]})
    verdict = t.can(Operation.APPEND, data=given)
    assert not verdict.ok and "IDENTITY_COLUMNS_EXPLICIT_INSERT" in verdict.reason
    with pytest.raises(UnreachableTableError, match="IDENTITY_COLUMNS_EXPLICIT_INSERT"):
        t.append(given)
    update = t.can(
        Operation.MERGE,
        source=given,
        predicate="target.id = source.id",
        clauses=["when_matched_update_all"],
    )
    assert not update.ok and "IDENTITY_COLUMNS_UPDATE" in update.reason
    left_out = t.can(Operation.APPEND, data=pa.table({"v": ["y"]}))
    assert "IDENTITY" not in left_out.reason


# ------------------------------------------------------------- #8 CHAR(n)


def test_predicates_on_char_columns_leave_the_direct_engines(conn: Any, tmp_path: Any) -> None:
    schema = {
        "type": "struct",
        "fields": [
            _field("id", "long"),
            _field("c", "string", __CHAR_VARCHAR_TYPE_STRING="char(3)"),
        ],
    }
    path = str(tmp_path / "chars")
    _write_log(path, schema, _PLAIN, pa.table({"id": [1], "c": ["a"]}))
    t = conn.open_table(path)
    # Spark pads to 3 before comparing: c = 'a  ' is true for 'a'. The direct
    # engines compared the bytes and returned no row.
    verdict = t.can(Operation.SCAN, predicate="c = 'a  '")
    assert not verdict.ok and "char_padding" in verdict.reason
    assert not t.can(Operation.DELETE, predicate="c = 'a  '").ok
    assert t.can(Operation.SCAN, predicate="id = 1").ok


# ------------------------------------------------------- #11 warehouse id


def test_a_malformed_warehouse_id_is_named_before_any_statement() -> None:
    from tests.unit.sql_fakes import FakeClient

    class InvalidParameterValue(Exception):
        error_code = "INVALID_PARAMETER_VALUE"

    client = FakeClient()

    def get(warehouse_id: str) -> Any:
        raise InvalidParameterValue(f"{warehouse_id} is not a valid endpoint id.")

    client.warehouses.get = get  # type: ignore[attr-defined]
    from deltaswamp.engine.sql import SqlEngine

    eng = SqlEngine(warehouse_id="bogus", client=client, warn_on_use=False)
    assert eng.warehouse_id is None
    verdict = eng.supports(Operation.DELETE, resolved_table())
    assert not verdict.ok and "not a SQL warehouse id" in verdict.reason


# --------------------------------------------------- #12 warehouse refusals


def test_can_knows_what_the_warehouse_refuses_up_front() -> None:
    eng, _, _ = engine()
    constrained = resolved_table(
        writer_features=frozenset({"checkConstraints"}),
        properties={"delta.constraints.pos": "id > 0"},
    )
    verdict = eng.supports(Operation.DROP_FEATURE, constrained, feature="checkConstraints")
    assert not verdict.ok and "pos" in verdict.reason
    plain = resolved_table()
    spaced = eng.supports(Operation.ADD_COLUMN, plain, fields={"a b": "INT"})
    assert not spaced.ok and "INVALID_CHARACTERS" in spaced.reason
    mapped = resolved_table(properties={"delta.columnMapping.mode": "name"})
    assert eng.supports(Operation.ADD_COLUMN, mapped, fields={"a b": "INT"}).ok


# ---------------------------------------------------- #13 exact narrowing


def test_python_ints_append_to_an_int_column_when_they_fit(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "narrow")
    schema = pa.schema([("id", pa.int32()), ("f", pa.float32())])
    t = conn.create_table(path, schema)
    t.append(pa.table({"id": [1, 2], "f": [0.5, 2.0]}))  # int64 and double, as Python gives
    got = pa.table(t.to_arrow())
    assert got.schema.field("id").type == pa.int32()
    assert sorted(got.column("id").to_pylist()) == [1, 2]
    with pytest.raises(InvalidArgumentError, match="a value does not fit"):
        t.append(pa.table({"id": [2**40]}))
    with pytest.raises(InvalidArgumentError, match="a value does not fit"):
        t.append(pa.table({"id": pa.array([1], pa.int32()), "f": [0.1]}))


# ---------------------------------------------------- #14 / #15 can() spellings


def test_can_takes_the_z_order_columns_spelling(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "z")
    conn.create_table(path, pa.schema([("id", pa.int64())])).append(pa.table({"id": [1]}))
    t = conn.open_table(path)
    assert t.can(Operation.ZORDER, columns=["id"]).ok == t.can("z_order", columns=["id"]).ok


def test_can_count_skips_a_shredded_variant(conn: Any, tmp_path: Any) -> None:
    from deltaswamp._variant import encode, variant_type

    m, v = encode('{"a":1}')
    data = pa.table(
        {
            "v": pa.array([{"metadata": m, "value": v}], variant_type(pa)),
            "id": pa.array([1], pa.int64()),
        }
    )
    schema = {"type": "struct", "fields": [_field("v", "variant"), _field("id", "long")]}
    protocol = {
        "minReaderVersion": 3,
        "minWriterVersion": 7,
        "readerFeatures": ["variantType"],
        "writerFeatures": ["variantType"],
    }
    path = str(tmp_path / "shredding")
    _write_log(path, schema, protocol, data)
    log = os.path.join(path, "_delta_log", f"{0:020}.json")
    with open(log) as f:
        text = f.read().replace(
            '"configuration": {}', '"configuration": {"delta.enableVariantShredding": "true"}'
        )
    with open(log, "w") as f:
        f.write(text)
    t = conn.open_table(path)
    assert not t.can(Operation.SCAN).ok
    assert t.can("count").ok
    assert t.count() == 1


# ------------------------------------------------------ #16 permissions


def test_a_permission_denial_is_a_typed_preflight_error() -> None:
    from deltaswamp.engine.sql_backend import SqlStatementError

    denial = SqlStatementError(
        "PERMISSION_DENIED: User does not have MODIFY on Table 'samples.nyctaxi.trips'.",
        error_code="BAD_REQUEST",
        sql_state="42501",
        state="FAILED",
    )

    class Denying(RecordingBackend):
        def execute(self, statement: str, parameters: Any = (), *, fetch: bool = True) -> Any:
            raise denial

    eng, _, _ = engine(Denying())
    with pytest.raises(PreflightError) as info:
        eng.set_comment(resolved_table("samples.nyctaxi.trips"), "x")
    error: Any = info.value
    assert isinstance(info.value, SqlStatementError)
    assert (error.privilege, error.securable) == ("MODIFY", "samples.nyctaxi.trips")
    assert "GRANT MODIFY ON TABLE samples.nyctaxi.trips" in error.remedy


# ------------------------------------------------------- #18 / #19 messages


def test_open_table_with_a_catalog_name_says_so(conn: Any) -> None:
    with pytest.raises(InvalidReferenceError, match="catalog table name"):
        conn.open_table("main.sales.orders")


def test_the_sdk_config_dump_is_left_out() -> None:
    from deltaswamp.credentials.databricks import sdk_message

    raw = (
        "Invalid access token. [ReqId: 1]. Config: host=https://h, account_id=a, "
        "token=***, auth_type=pat. Env: DATABRICKS_HOST, DATABRICKS_TOKEN"
    )
    assert sdk_message(Exception(raw)) == "Invalid access token. [ReqId: 1]"


# ----------------------------------------------------------------- #20


def test_update_takes_a_tuple_path(conn: Any, tmp_path: Any) -> None:
    # LM-14: ("s", "a") was refused as "not a string" where "s.a" is a path.
    path = str(tmp_path / "nested")
    schema = pa.schema([("id", pa.int64()), ("s", pa.struct([("a", pa.int64())]))])
    t = conn.create_table(path, schema)
    t.append(pa.table({"id": [1], "s": [{"a": 1}]}, schema=schema))
    outcomes = []
    for key in ("s.a", ("s", "a")):
        try:
            t.update(new_values={key: 6}, predicate="id = 1")
            outcomes.append(pa.table(t.to_arrow()).column("s").to_pylist())
        except DeltaSwampError as exc:
            outcomes.append(str(exc))
    assert outcomes[0] == outcomes[1]
    assert "must be strings" not in str(outcomes[1])
    t.update(new_values={("id",): 2}, predicate="id = 1")
    assert pa.table(t.to_arrow()).column("id").to_pylist() == [2]


def test_time_travel_past_the_latest_commit_is_refused(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "tt")
    conn.create_table(path, pa.schema([("id", pa.int64())])).append(pa.table({"id": [1]}))
    t = conn.open_table(path)
    with pytest.raises(InvalidArgumentError, match="after the latest commit"):
        t.to_arrow(timestamp="2999-01-01T00:00:00Z")
    latest = max(h["timestamp"] for h in t.history())
    assert pa.table(t.to_arrow(timestamp=latest)).num_rows == 1
