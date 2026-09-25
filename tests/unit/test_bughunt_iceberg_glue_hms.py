"""Regression tests for defects in the Iceberg engine and the Glue / HMS catalogs.

No network: Iceberg runs against real PyIceberg tables on local disk (the
`DirectoryCatalog` from test_iceberg), and boto3 / pymetastore are replaced by
fakes shaped like the real APIs.
"""

from __future__ import annotations

import pickle
import sys
import threading
import types
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("pyiceberg")

import deltaswamp as ds  # noqa: E402
from deltaswamp.catalog.base import TableType  # noqa: E402
from deltaswamp.catalog.glue import GlueCatalog  # noqa: E402
from deltaswamp.catalog.hms import HiveMetastoreCatalog  # noqa: E402
from deltaswamp.engine.iceberg import IcebergEngine, _timestamp_ms  # noqa: E402
from deltaswamp.errors import (  # noqa: E402
    InvalidReferenceError,
    PreflightError,
    UnreachableTableError,
)
from deltaswamp.identity import RefKind, TableRef, parse_ref  # noqa: E402
from pyiceberg.partitioning import PartitionField, PartitionSpec  # noqa: E402
from pyiceberg.transforms import BucketTransform, IdentityTransform  # noqa: E402

from tests.integration.test_iceberg import (  # noqa: E402
    ARROW,
    SCHEMA,
    DirectoryCatalog,
    FakeProvider,
    _resolved,
    _rows,
    _uniform,
)

# ---------------------------------------------------------------------------
# Iceberg fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog(tmp_path: Path) -> DirectoryCatalog:
    cat = DirectoryCatalog("test", warehouse=tmp_path.as_uri())
    spec = PartitionSpec(
        PartitionField(source_id=3, field_id=1000, transform=IdentityTransform(), name="region")
    )
    cat.create_table(
        ("sales", "orders"), SCHEMA, location=(tmp_path / "orders").as_uri(), partition_spec=spec
    )
    return cat


@pytest.fixture
def engine(catalog: DirectoryCatalog) -> IcebergEngine:
    return IcebergEngine(catalog_factory=lambda name, props: catalog)


def _nulls() -> Any:
    return pa.Table.from_pylist(
        [
            {"id": None, "name": None, "region": None, "day": None},
            {"id": 4, "name": "x", "region": None, "day": None},
        ],
        schema=ARROW,
    )


@pytest.fixture
def loaded(engine: IcebergEngine) -> IcebergEngine:
    table = _resolved()
    engine.append(table, _rows([1, 2], "eu"))
    engine.append(table, _rows([3], "us"))
    engine.append(table, _nulls())
    return engine


def _ids(reader: Any) -> list[Any]:
    return sorted(pa.table(reader).column("id").to_pylist(), key=lambda v: (v is None, v))


# ---------------------------------------------------------------------------
# Iceberg: reads
# ---------------------------------------------------------------------------


class TestIcebergPredicates:
    def test_fractional_literal_on_long_column_is_not_rounded(self, loaded: IcebergEngine) -> None:
        # PyIceberg rounds 1.5 to 2 for a long column: `id > 1.5` read as `id > 2`.
        assert _ids(loaded.scan(_resolved(), predicate="id > 1.5")) == [2, 3, 4]
        assert _ids(loaded.scan(_resolved(), predicate="id = 1.5")) == []

    def test_not_in_excludes_nulls_like_sql(self, loaded: IcebergEngine) -> None:
        assert _ids(loaded.scan(_resolved(), predicate="id NOT IN (1, 2)")) == [3, 4]

    def test_not_in_on_a_partition_column_excludes_null_partitions(
        self, loaded: IcebergEngine
    ) -> None:
        got = _ids(loaded.scan(_resolved(), predicate="region NOT IN ('eu', 'xx')"))
        assert got == [3]

    def test_type_mismatched_literal_does_not_crash(self, loaded: IcebergEngine) -> None:
        # PyIceberg raised TypeError binding `name = 1`; SQL coerces.
        assert _ids(loaded.scan(_resolved(), predicate="name = 1")) == []

    def test_column_case_is_resolved_like_sql(self, loaded: IcebergEngine) -> None:
        assert _ids(loaded.scan(_resolved(), predicate="ID = 1")) == [1]

    def test_limit_is_honoured(self, loaded: IcebergEngine) -> None:
        assert pa.table(loaded.scan(_resolved(), limit=2)).num_rows == 2

    def test_limit_applies_after_a_residual_filter(self, loaded: IcebergEngine) -> None:
        got = pa.table(loaded.scan(_resolved(), predicate="id > 1.5", limit=2))
        assert got.num_rows == 2 and all(i > 1.5 for i in got.column("id").to_pylist())

    def test_limit_zero(self, loaded: IcebergEngine) -> None:
        assert pa.table(loaded.scan(_resolved(), predicate="id > 1.5", limit=0)).num_rows == 0

    def test_negative_limit_is_refused(self, loaded: IcebergEngine) -> None:
        with pytest.raises(ValueError, match="limit"):
            loaded.scan(_resolved(), limit=-1)


class TestIcebergHistoryDetail:
    def test_history_limit_zero_is_empty(self, loaded: IcebergEngine) -> None:
        assert loaded.history(_resolved(), limit=0) == []

    def test_uniform_history_version_is_the_delta_version(self, engine: IcebergEngine) -> None:
        native = _resolved()
        engine.append(native, _rows([1], "eu"), commit_metadata={"delta-version": 7})
        uniform = _uniform()
        [entry] = engine.history(uniform)
        assert entry["version"] == 7
        # ...which is what scan(version=...) takes back for a UniForm table.
        assert _ids(engine.scan(uniform, version=entry["version"])) == [1]
        assert engine.detail(uniform)["version"] == 7

    def test_partition_columns_are_identity_partitions_only(self, tmp_path: Path) -> None:
        cat = DirectoryCatalog("t", warehouse=tmp_path.as_uri())
        spec = PartitionSpec(
            PartitionField(source_id=1, field_id=1000, transform=BucketTransform(4), name="b")
        )
        cat.create_table(
            ("sales", "orders"), SCHEMA, location=(tmp_path / "o").as_uri(), partition_spec=spec
        )
        eng = IcebergEngine(catalog_factory=lambda n, p: cat)
        assert eng.detail(_resolved())["partition_columns"] == []


class TestIcebergTimestamps:
    def test_datetime_and_date_are_accepted(self) -> None:
        moment = datetime(2024, 1, 1, 0, 0, 0, 123000, tzinfo=UTC)
        assert _timestamp_ms(moment) == 1704067200123
        assert _timestamp_ms(date(2024, 1, 1)) == 1704067200000
        assert _timestamp_ms(1704067200123) == 1704067200123

    def test_pre_epoch_rounds_down(self) -> None:
        # int(ts * 1000) truncated toward zero: -0.5 ms became 0.
        assert _timestamp_ms("1969-12-31T23:59:59.9995+00:00") == -1

    def test_bool_is_refused(self) -> None:
        with pytest.raises(UnreachableTableError):
            _timestamp_ms(True)


class TestIcebergToken:
    def test_none_token_is_not_sent_as_the_string_none(self) -> None:
        eng = IcebergEngine(catalog_factory=lambda n, p: None)
        props = eng.rest_properties(_resolved(FakeProvider(token=None)))  # type: ignore[arg-type]
        assert "token" not in props


# ---------------------------------------------------------------------------
# Iceberg: writes
# ---------------------------------------------------------------------------


class TestIcebergOverwrite:
    def test_not_in_overwrite_keeps_null_rows(self, loaded: IcebergEngine) -> None:
        # Iceberg counts NULL NOT IN (...) as a match and dropped the null-only
        # files; SQL (replaceWhere) keeps those rows.
        loaded.overwrite(_resolved(), _rows([9], "us"), predicate="region NOT IN ('eu', 'xx')")
        assert _ids(loaded.scan(_resolved())) == [1, 2, 4, 9, None]

    def test_lossy_literal_overwrite_is_refused(self, loaded: IcebergEngine) -> None:
        with pytest.raises(UnreachableTableError, match="lossily"):
            loaded.overwrite(_resolved(), _rows([8], "us"), predicate="id > 1.5")
        assert _ids(loaded.scan(_resolved())) == [1, 2, 3, 4, None]

    def test_rows_outside_the_predicate_are_refused(self, loaded: IcebergEngine) -> None:
        with pytest.raises(UnreachableTableError, match="do not match"):
            loaded.overwrite(_resolved(), _rows([9], "us"), predicate="region = 'eu'")
        assert _ids(loaded.scan(_resolved())) == [1, 2, 3, 4, None]

    def test_unknown_partition_overwrite_mode_is_refused(self, loaded: IcebergEngine) -> None:
        # A typo used to fall through to a whole-table overwrite.
        with pytest.raises(ValueError, match="partition_overwrite"):
            loaded.overwrite(_resolved(), _rows([9], "us"), partition_overwrite="dynmaic")
        assert _ids(loaded.scan(_resolved())) == [1, 2, 3, 4, None]

    def test_missing_namespace_is_a_clear_error(self, engine: IcebergEngine) -> None:
        with pytest.raises(UnreachableTableError, match="no table"):
            engine.scan(_resolved(ref=parse_ref("main.nope.orders")))


# ---------------------------------------------------------------------------
# Glue
# ---------------------------------------------------------------------------


class _ClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(f"An error occurred ({code})")
        self.response = {"Error": {"Code": code, "Message": code}}


class FakeGlue:
    def __init__(self, tables: dict[tuple[str | None, str, str], dict[str, Any]]) -> None:
        self.tables = tables
        self.calls: list[dict[str, Any]] = []

    def get_table(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        key = (kwargs.get("CatalogId"), kwargs["DatabaseName"], kwargs["Name"])
        if key not in self.tables:
            raise _ClientError("EntityNotFoundException")
        return {"Table": self.tables[key]}

    def get_paginator(self, name: str) -> Any:
        assert name == "get_tables"
        glue = self

        class Paginator:
            def paginate(self, **kwargs: Any) -> Any:
                glue.calls.append({"paginate": kwargs})
                rows = [
                    t
                    for (cid, db, _), t in glue.tables.items()
                    if db == kwargs["DatabaseName"] and cid == kwargs.get("CatalogId")
                ]
                yield {"TableList": rows[:1]}
                yield {"TableList": rows[1:]}

        return Paginator()


def _glue_table(name: str, **overrides: Any) -> dict[str, Any]:
    table: dict[str, Any] = {
        "Name": name,
        "TableType": "EXTERNAL_TABLE",
        "Parameters": {"spark.sql.sources.provider": "delta"},
        "StorageDescriptor": {"Location": f"s3://bucket/{name}"},
    }
    table.update(overrides)
    return table


@pytest.fixture
def fake_boto3(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"regions": [], "glue": FakeGlue({})}
    module = types.ModuleType("boto3")

    def client(service: str, region_name: str | None = None) -> Any:
        state["regions"].append(region_name)
        return state["glue"]

    module.client = client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", module)
    return state


def _ref(db: str = "db", table: str = "t", catalog: str = "glue") -> TableRef:
    return TableRef(kind=RefKind.CATALOG, catalog=catalog, schema=db, table=table, scheme="glue")


class TestGlue:
    def test_connect_does_not_crash_on_the_token_kwarg(self) -> None:
        conn = ds.connect("glue://123456789012")
        assert isinstance(conn.catalog, GlueCatalog)

    def test_uri_trailing_slash_and_region(self) -> None:
        cat = GlueCatalog.from_uri("glue://123456789012/?region=eu-west-1", token=None)
        assert cat._catalog_id == "123456789012"
        assert cat._region_name == "eu-west-1"

    def test_non_delta_table_without_provider_is_refused(self, fake_boto3: Any) -> None:
        # Athena/crawler Parquet tables carry no Spark provider; they were read as Delta.
        fake_boto3["glue"].tables[(None, "db", "t")] = _glue_table(
            "t", Parameters={"classification": "parquet"}
        )
        with pytest.raises(InvalidReferenceError, match="parquet"):
            GlueCatalog().resolve(_ref())

    def test_iceberg_table_is_refused(self, fake_boto3: Any) -> None:
        fake_boto3["glue"].tables[(None, "db", "t")] = _glue_table(
            "t", Parameters={"table_type": "ICEBERG"}
        )
        with pytest.raises(InvalidReferenceError, match="ICEBERG"):
            GlueCatalog().resolve(_ref())

    def test_view_is_refused(self, fake_boto3: Any) -> None:
        fake_boto3["glue"].tables[(None, "db", "t")] = _glue_table("t", TableType="VIRTUAL_VIEW")
        with pytest.raises(InvalidReferenceError, match="view"):
            GlueCatalog().resolve(_ref())

    def test_athena_and_crawler_markers_are_delta(self, fake_boto3: Any) -> None:
        glue = fake_boto3["glue"]
        glue.tables[(None, "db", "a")] = _glue_table("a", Parameters={"table_type": "DELTA"})
        glue.tables[(None, "db", "c")] = _glue_table("c", Parameters={"classification": "delta"})
        assert GlueCatalog().resolve(_ref(table="a")).location == "s3://bucket/a"
        assert GlueCatalog().resolve(_ref(table="c")).location == "s3://bucket/c"

    def test_spark_placeholder_location_uses_the_serde_path(self, fake_boto3: Any) -> None:
        fake_boto3["glue"].tables[(None, "db", "t")] = _glue_table(
            "t",
            StorageDescriptor={
                "Location": "s3://warehouse/db.db/t-__PLACEHOLDER__",
                "SerdeInfo": {"Parameters": {"path": "s3a://bucket/real/t"}},
            },
        )
        assert GlueCatalog().resolve(_ref()).location == "s3://bucket/real/t"

    def test_missing_location_is_a_clear_error(self, fake_boto3: Any) -> None:
        fake_boto3["glue"].tables[(None, "db", "t")] = _glue_table("t", StorageDescriptor={})
        with pytest.raises(InvalidReferenceError, match="no storage location"):
            GlueCatalog().resolve(_ref())

    def test_missing_table_is_invalid_reference(self, fake_boto3: Any) -> None:
        with pytest.raises(InvalidReferenceError, match="does not exist"):
            GlueCatalog().resolve(_ref())

    def test_unknown_table_type_is_not_declared_managed(self, fake_boto3: Any) -> None:
        fake_boto3["glue"].tables[(None, "db", "t")] = _glue_table("t", TableType=None)
        assert GlueCatalog().resolve(_ref()).table_type is None
        fake_boto3["glue"].tables[(None, "db", "t")] = _glue_table("t")
        assert GlueCatalog().resolve(_ref()).table_type is TableType.EXTERNAL

    def test_resource_link_is_followed(self, fake_boto3: Any) -> None:
        glue = fake_boto3["glue"]
        glue.tables[(None, "db", "link")] = {
            "Name": "link",
            "TargetTable": {"CatalogId": "999999999999", "DatabaseName": "src", "Name": "t"},
        }
        glue.tables[("999999999999", "src", "t")] = _glue_table("t")
        assert GlueCatalog().resolve(_ref(table="link")).location == "s3://bucket/t"

    def test_account_id_catalog_name_becomes_catalog_id(self, fake_boto3: Any) -> None:
        glue = fake_boto3["glue"]
        glue.tables[("123456789012", "db", "t")] = _glue_table("t")
        GlueCatalog().resolve(_ref(catalog="123456789012"))
        assert glue.calls[-1]["CatalogId"] == "123456789012"

    def test_list_tables_does_not_refetch_each_table(self, fake_boto3: Any) -> None:
        glue = fake_boto3["glue"]
        glue.tables[(None, "db", "a")] = _glue_table("a")
        glue.tables[(None, "db", "b")] = _glue_table("b")
        glue.tables[(None, "db", "p")] = _glue_table("p", Parameters={"classification": "csv"})
        out = GlueCatalog().list_tables("glue", "db")
        assert sorted(cast(str, t.ref.table) for t in out) == ["a", "b"]
        assert not [c for c in glue.calls if "Name" in c]

    def test_list_tables_on_missing_database(self, fake_boto3: Any) -> None:
        class Missing(FakeGlue):
            def get_paginator(self, name: str) -> Any:
                class P:
                    def paginate(self, **kwargs: Any) -> Any:
                        raise _ClientError("EntityNotFoundException")
                        yield  # type: ignore[unreachable]  # pragma: no cover

                return P()

        fake_boto3["glue"] = Missing({})
        with pytest.raises(InvalidReferenceError, match="does not exist"):
            GlueCatalog().list_tables("glue", "nope")

    def test_no_region_is_a_preflight_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = types.ModuleType("boto3")

        class NoRegionError(Exception):
            pass

        def client(service: str, region_name: str | None = None) -> Any:
            raise NoRegionError("You must specify a region.")

        module.client = client  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "boto3", module)
        with pytest.raises(PreflightError, match="AWS_REGION"):
            GlueCatalog().resolve(_ref())

    def test_catalog_pickles_without_its_client(self, fake_boto3: Any) -> None:
        fake_boto3["glue"].tables[(None, "db", "t")] = _glue_table("t")
        cat = GlueCatalog(region_name="us-east-1")
        cat.resolve(_ref())
        clone = pickle.loads(pickle.dumps(cat))
        assert clone._region_name == "us-east-1" and clone._clients == {}


# ---------------------------------------------------------------------------
# Hive Metastore
# ---------------------------------------------------------------------------


class NoSuchObjectException(Exception):
    pass


class _Thrift:
    """Shaped like pymetastore's raw Thrift `Table`."""

    def __init__(
        self,
        parameters: dict[str, str] | None,
        location: str | None,
        table_type: str = "EXTERNAL_TABLE",
        serde: dict[str, str] | None = None,
        view: str | None = None,
    ) -> None:
        self.parameters = parameters
        self.tableType = table_type
        self.viewOriginalText = view
        self.sd = types.SimpleNamespace(
            location=location, serdeInfo=types.SimpleNamespace(parameters=serde)
        )


class FakeMetastore:
    def __init__(self) -> None:
        self.tables: dict[tuple[str, str], _Thrift] = {}
        self.opened: list[tuple[str, int]] = []
        self.closed = 0
        self.lock = threading.Lock()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = self

        class Client:
            def get_table(self, db: str, name: str) -> _Thrift:
                if (db, name) not in store.tables:
                    raise NoSuchObjectException(f"{db}.{name} table not found")
                return store.tables[(db, name)]

        class HMS:
            def __init__(self) -> None:
                self.client = Client()

            def list_tables(self, db: str) -> list[str]:
                return [n for (d, n) in store.tables if d == db]

            @staticmethod
            def create(host: str = "localhost", port: int = 9083) -> Any:
                return Connection(host, port)

        class Connection:
            def __init__(self, host: str, port: int) -> None:
                self.host, self.port = host, port

            def __enter__(self) -> HMS:
                with store.lock:
                    store.opened.append((self.host, self.port))
                return HMS()

            def __exit__(self, *exc: Any) -> None:
                with store.lock:
                    store.closed += 1

        package = types.ModuleType("pymetastore")
        module = types.ModuleType("pymetastore.metastore")
        module.HMS = HMS  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pymetastore", package)
        monkeypatch.setitem(sys.modules, "pymetastore.metastore", module)
        monkeypatch.delitem(sys.modules, "pymetastore.hms", raising=False)


@pytest.fixture
def metastore(monkeypatch: pytest.MonkeyPatch) -> FakeMetastore:
    store = FakeMetastore()
    store.install(monkeypatch)
    return store


def _href(db: str = "db", table: str = "t", endpoint: str | None = None) -> TableRef:
    return TableRef(
        kind=RefKind.CATALOG,
        catalog="hive_metastore",
        schema=db,
        table=table,
        scheme="hms",
        endpoint=endpoint,
    )


DELTA = {"spark.sql.sources.provider": "delta"}


class TestHms:
    def test_connect_does_not_crash_on_the_token_kwarg(self) -> None:
        conn = ds.connect("hms://metastore:9083")
        assert isinstance(conn.catalog, HiveMetastoreCatalog)

    def test_pymetastore_metastore_module_is_used(self, metastore: FakeMetastore) -> None:
        # The import used to be `pymetastore.hms`, which does not exist.
        metastore.tables[("db", "t")] = _Thrift(DELTA, "s3://b/t")
        assert HiveMetastoreCatalog("hms://h:9083").resolve(_href()).location == "s3://b/t"

    def test_second_resolve_opens_a_fresh_connection(self, metastore: FakeMetastore) -> None:
        metastore.tables[("db", "t")] = _Thrift(DELTA, "s3://b/t")
        cat = HiveMetastoreCatalog("hms://h:9083")
        cat.resolve(_href())
        cat.resolve(_href())
        assert len(metastore.opened) == 2 and metastore.closed == 2

    def test_connection_closed_when_lookup_fails(self, metastore: FakeMetastore) -> None:
        with pytest.raises(InvalidReferenceError, match="does not exist"):
            HiveMetastoreCatalog("hms://h:9083").resolve(_href())
        assert metastore.closed == 1

    def test_s3a_location_is_normalised(self, metastore: FakeMetastore) -> None:
        metastore.tables[("db", "t")] = _Thrift(DELTA, "s3a://bucket/t")
        assert HiveMetastoreCatalog("hms://h:9083").resolve(_href()).location == "s3://bucket/t"

    def test_spark_placeholder_location_uses_the_serde_path(self, metastore: FakeMetastore) -> None:
        metastore.tables[("db", "t")] = _Thrift(
            DELTA, "file:/warehouse/db.db/t-__PLACEHOLDER__", serde={"path": "s3a://b/real"}
        )
        assert HiveMetastoreCatalog("hms://h:9083").resolve(_href()).location == "s3://b/real"

    def test_missing_location_is_a_clear_error(self, metastore: FakeMetastore) -> None:
        metastore.tables[("db", "t")] = _Thrift(DELTA, None)
        with pytest.raises(InvalidReferenceError, match="no storage location"):
            HiveMetastoreCatalog("hms://h:9083").resolve(_href())

    def test_view_is_refused(self, metastore: FakeMetastore) -> None:
        metastore.tables[("db", "v")] = _Thrift(DELTA, None, "VIRTUAL_VIEW", view="select 1")
        with pytest.raises(InvalidReferenceError, match="view"):
            HiveMetastoreCatalog("hms://h:9083").resolve(_href(table="v"))

    def test_null_parameters_are_tolerated(self, metastore: FakeMetastore) -> None:
        metastore.tables[("db", "t")] = _Thrift(None, "s3://b/t")
        with pytest.raises(InvalidReferenceError, match="does not advertise"):
            HiveMetastoreCatalog("hms://h:9083").resolve(_href())

    def test_external_parameter_marks_external(self, metastore: FakeMetastore) -> None:
        metastore.tables[("db", "t")] = _Thrift(
            {**DELTA, "EXTERNAL": "TRUE"}, "s3://b/t", "MANAGED_TABLE"
        )
        assert HiveMetastoreCatalog("hms://h:9083").resolve(_href()).table_type is (
            TableType.EXTERNAL
        )

    def test_ref_for_another_metastore_is_refused(self, metastore: FakeMetastore) -> None:
        # It used to be read silently from this connection's metastore.
        metastore.tables[("db", "t")] = _Thrift(DELTA, "s3://b/t")
        cat = HiveMetastoreCatalog("hms://h:9083")
        with pytest.raises(InvalidReferenceError, match="other-host:9083"):
            cat.resolve(_href(endpoint="other-host:9083"))
        with pytest.raises(InvalidReferenceError, match="h:9999"):
            cat.resolve(_href(endpoint="h:9999"))
        assert metastore.opened == []

    @pytest.mark.parametrize("endpoint", ["h", "h:9083", "H:9083", "thrift://h:9083"])
    def test_ref_for_the_same_metastore_is_accepted(
        self, metastore: FakeMetastore, endpoint: str
    ) -> None:
        metastore.tables[("db", "t")] = _Thrift(DELTA, "s3://b/t")
        cat = HiveMetastoreCatalog("hms://thrift://h")
        assert cat.resolve(_href(endpoint=endpoint)).location == "s3://b/t"
        assert metastore.opened == [("h", 9083)]

    def test_parsed_hms_uri_resolves_on_its_own_connection(self, metastore: FakeMetastore) -> None:
        metastore.tables[("db", "t")] = _Thrift(DELTA, "s3://b/t")
        ref = parse_ref("hms://h:9083/db/t")
        assert HiveMetastoreCatalog("hms://h:9083").resolve(ref).location == "s3://b/t"

    def test_list_tables_uses_one_connection(self, metastore: FakeMetastore) -> None:
        metastore.tables[("db", "a")] = _Thrift(DELTA, "s3://b/a")
        metastore.tables[("db", "b")] = _Thrift({"table_type": "DELTA"}, "s3://b/b")
        metastore.tables[("db", "p")] = _Thrift({"k": "v"}, "s3://b/p")
        out = HiveMetastoreCatalog("hms://h:9083").list_tables("hive_metastore", "db")
        assert sorted(cast(str, t.ref.table) for t in out) == ["a", "b"]
        assert len(metastore.opened) == 1 and metastore.closed == 1

    def test_connection_failure_is_a_preflight_error(
        self, metastore: FakeMetastore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = sys.modules["pymetastore.metastore"]

        def refuse(*_: Any, **__: Any) -> None:
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(module.HMS, "create", staticmethod(refuse))
        with pytest.raises(PreflightError, match="h:9083"):
            HiveMetastoreCatalog("hms://h:9083").resolve(_href())

    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("hms://[::1]:9084", ("::1", 9084)),
            ("hms://thrift://h:9085/", ("h", 9085)),
            ("hive://h", ("h", 9083)),
        ],
    )
    def test_uri_parsing(self, uri: str, expected: tuple[str, int]) -> None:
        cat = HiveMetastoreCatalog(uri)
        assert (cat._host, cat._port) == expected

    def test_bad_port_is_an_invalid_reference(self) -> None:
        with pytest.raises(InvalidReferenceError, match="port"):
            HiveMetastoreCatalog("hms://h:abc")

    def test_bare_scheme_is_not_taken_as_a_host(self) -> None:
        with pytest.raises(InvalidReferenceError, match="host is required"):
            HiveMetastoreCatalog.from_uri("hive", token=None)

    def test_catalog_pickles(self) -> None:
        cat = HiveMetastoreCatalog("hms://h:9083")
        clone = pickle.loads(pickle.dumps(cat))
        assert (clone._host, clone._port) == ("h", 9083)


class TestIcebergCommitMetadata:
    def test_reserved_summary_key_is_refused_before_writing(self, engine: IcebergEngine) -> None:
        # PyIceberg raised "Summary() got multiple values for keyword argument"
        # at commit time, after writing the data files.
        with pytest.raises(ValueError, match="added-records"):
            engine.append(_resolved(), _rows([1], "eu"), commit_metadata={"added-records": "5"})
        with pytest.raises(ValueError, match="operation"):
            engine.overwrite(_resolved(), _rows([1], "eu"), commit_metadata={"operation": "x"})
        assert engine.history(_resolved()) == []


# ---------------------------------------------------------------------------
# Iceberg: second pass
# ---------------------------------------------------------------------------


class TestIcebergSecondPass:
    def test_version_and_timestamp_together_are_refused(self, loaded: IcebergEngine) -> None:
        with pytest.raises(ValueError, match="not both"):
            loaded.scan(_resolved(), version=1, timestamp="2024-01-01")

    def test_dynamic_overwrite_of_an_unpartitioned_table_replaces_it(self, tmp_path: Path) -> None:
        cat = DirectoryCatalog("t", warehouse=tmp_path.as_uri())
        cat.create_table(("sales", "orders"), SCHEMA, location=(tmp_path / "o").as_uri())
        eng = IcebergEngine(catalog_factory=lambda n, p: cat)
        eng.append(_resolved(), _rows([1, 2], "eu"))
        eng.overwrite(_resolved(), _rows([7], "us"), partition_overwrite="dynamic")
        assert _ids(eng.scan(_resolved())) == [7]

    def test_empty_column_list_keeps_the_row_count(self, loaded: IcebergEngine) -> None:
        out = pa.table(loaded.scan(_resolved(), columns=[]))
        assert out.num_columns == 0 and out.num_rows == 5
        out = pa.table(loaded.scan(_resolved(), columns=[], predicate="id > 1.5"))
        assert out.num_columns == 0 and out.num_rows == 3

    def test_zero_row_append_commits_nothing(self, loaded: IcebergEngine) -> None:
        before = len(loaded.history(_resolved()))
        loaded.append(_resolved(), _rows([], "eu"))
        loaded.overwrite(_resolved(), _rows([], "eu"), partition_overwrite="dynamic")
        assert len(loaded.history(_resolved())) == before

    def test_nanosecond_and_zoned_timestamps_are_written(self, tmp_path: Path) -> None:
        from pyiceberg.schema import Schema
        from pyiceberg.types import LongType, NestedField, TimestamptzType

        cat = DirectoryCatalog("t", warehouse=tmp_path.as_uri())
        cat.create_table(
            ("sales", "orders"),
            Schema(
                NestedField(1, "id", LongType(), required=True),
                NestedField(2, "ts", TimestamptzType(), required=False),
            ),
            location=(tmp_path / "o").as_uri(),
        )
        eng = IcebergEngine(catalog_factory=lambda n, p: cat)
        moment = datetime(2024, 1, 1, 12, tzinfo=UTC)
        schema = pa.schema(
            [pa.field("id", pa.int64(), nullable=False), ("ts", pa.timestamp("ns", tz="UTC"))]
        )
        # pandas' default unit; PyIceberg refused it. The required `id` must stay non-nullable.
        eng.append(_resolved(), pa.table({"id": [1], "ts": [moment]}, schema=schema))
        ny = pa.table(
            {"id": [2], "ts": pa.array([moment], pa.timestamp("us", tz="America/New_York"))},
            schema=pa.schema(
                [
                    pa.field("id", pa.int64(), nullable=False),
                    ("ts", pa.timestamp("us", tz="America/New_York")),
                ]
            ),
        )
        eng.append(_resolved(), ny)
        got = pa.table(eng.scan(_resolved())).column("ts").to_pylist()
        assert got == [moment, moment]

    def test_uuid_strings_and_wide_decimals_are_written(self, tmp_path: Path) -> None:
        import uuid
        from decimal import Decimal

        from pyiceberg.schema import Schema
        from pyiceberg.types import DecimalType, NestedField, UUIDType

        cat = DirectoryCatalog("t", warehouse=tmp_path.as_uri())
        cat.create_table(
            ("sales", "orders"),
            Schema(
                NestedField(1, "u", UUIDType(), required=False),
                NestedField(2, "amt", DecimalType(10, 2), required=False),
            ),
            location=(tmp_path / "o").as_uri(),
        )
        eng = IcebergEngine(catalog_factory=lambda n, p: cat)
        key = uuid.uuid4()
        eng.append(
            _resolved(),
            pa.table({"u": [str(key)], "amt": pa.array([Decimal("1.23")], pa.decimal128(38, 2))}),
        )
        row = pa.table(eng.scan(_resolved())).to_pylist()[0]
        assert row["amt"] == Decimal("1.23")
        assert row["u"] == key
        with pytest.raises(ValueError, match="does not fit"):
            eng.append(
                _resolved(),
                pa.table({"amt": pa.array([Decimal("123456789012.00")], pa.decimal128(38, 2))}),
            )

    def test_column_case_is_matched_on_write(self, engine: IcebergEngine) -> None:
        engine.append(_resolved(), pa.table({"ID": [5], "Region": ["eu"]}))
        assert _ids(engine.scan(_resolved())) == [5]

    def test_extra_column_is_a_clear_error(self, engine: IcebergEngine) -> None:
        with pytest.raises(UnreachableTableError, match="does not evolve schemas"):
            engine.append(_resolved(), pa.table({"id": [1], "zzz": [1]}))


class _Snap:
    def __init__(self, sid: int, ts: int, parent: int | None) -> None:
        self.snapshot_id, self.timestamp_ms, self.parent_snapshot_id = sid, ts, parent


class _Entry:
    def __init__(self, sid: int, ts: int) -> None:
        self.snapshot_id, self.timestamp_ms = sid, ts


class _Meta:
    def __init__(self, snaps: list[_Snap], log: list[_Entry], current: int | None) -> None:
        self.snaps = {s.snapshot_id: s for s in snaps}
        self.log, self.current = log, current

    def history(self) -> list[_Entry]:
        return self.log

    def snapshot_by_id(self, sid: int) -> _Snap | None:
        return self.snaps.get(sid)

    def current_snapshot(self) -> _Snap | None:
        return self.snaps.get(self.current) if self.current is not None else None


class TestSnapshotAsOf:
    def test_empty_snapshot_log_falls_back_to_ancestry(self) -> None:
        meta = _Meta([_Snap(1, 1000, None), _Snap(2, 2000, 1)], [], current=2)
        assert IcebergEngine._snapshot_at(meta, 1500) == 1
        assert IcebergEngine._snapshot_at(meta, 2000) == 2

    def test_expired_snapshot_is_named(self) -> None:
        meta = _Meta([_Snap(2, 2000, None)], [_Entry(1, 1000), _Entry(2, 2000)], current=2)
        with pytest.raises(UnreachableTableError, match="expired"):
            IcebergEngine._snapshot_at(meta, 1500)

    def test_boundary_is_inclusive_in_milliseconds(self) -> None:
        meta = _Meta([_Snap(1, 1704067200123, None)], [_Entry(1, 1704067200123)], current=1)
        assert IcebergEngine._snapshot_at(meta, "2024-01-01T00:00:00.123999+00:00") == 1
        with pytest.raises(UnreachableTableError):
            IcebergEngine._snapshot_at(meta, "2024-01-01T00:00:00.122999+00:00")


class TestTokenRace:
    def test_an_older_token_does_not_overwrite_a_newer_catalog(self) -> None:
        tokens = iter(["old", "new"])
        first_building = threading.Event()
        release_first = threading.Event()
        built: list[str] = []

        class Provider:
            table_id = None

            def workspace_auth(self) -> tuple[str, str]:
                return "https://ws", next(tokens)

        def factory(name: str, props: dict[str, str]) -> Any:
            if props["token"] == "old":
                first_building.set()
                release_first.wait(5)
            built.append(props["token"])
            return props["token"]

        eng = IcebergEngine(catalog_factory=factory)
        table = _resolved(Provider())
        slow = threading.Thread(target=eng._catalog, args=(table,))
        slow.start()
        assert first_building.wait(5)
        assert eng._catalog(table) == "new"  # the later fetch finishes first
        release_first.set()
        slow.join(5)
        assert eng._catalogs[(cast(str, table.iceberg_rest_uri), "main")][0] == "new"


class TestReviewCatalogMarkers:
    """Adversarial-review follow-ups for the Glue/HMS resolution fixes."""

    def test_hive_metastore_named_hms_table_routes_to_a_direct_engine(
        self, metastore: FakeMetastore
    ) -> None:
        from deltaswamp.capability import Operation
        from deltaswamp.router import Router

        metastore.tables[("db", "t")] = _Thrift(DELTA, "s3://b/t")
        ref = parse_ref("hive_metastore.db.t")
        assert ref.scheme != "hms"  # the default scheme, as conn.table() parses it
        resolved = HiveMetastoreCatalog("hms://h:9083").resolve(ref)
        assert resolved.ref.scheme == "hms"
        assert Router._direct_refusal(Operation.SCAN, resolved) is None

    def test_hms_markers_are_case_insensitive(self, metastore: FakeMetastore) -> None:
        metastore.tables[("db", "a")] = _Thrift({"spark.sql.sources.provider": "DELTA"}, "s3://b/a")
        metastore.tables[("db", "h")] = _Thrift(
            {"storage_handler": "io.delta.hive.DeltaStorageHandler"}, "s3://b/h"
        )
        cat = HiveMetastoreCatalog("hms://h:9083")
        assert cat.resolve(_href(table="a")).location == "s3://b/a"
        assert cat.resolve(_href(table="h")).location == "s3://b/h"

    def test_glue_delta_markers_every_writer_uses(self, fake_boto3: Any) -> None:
        glue = fake_boto3["glue"]
        # Trino/EMR uppercase value, a lowercase Athena value, the crawler's
        # classification on the storage descriptor, a key in another case, and
        # the Delta Hive connector's input format.
        glue.tables[(None, "db", "u")] = _glue_table(
            "u", Parameters={"spark.sql.sources.provider": "DELTA"}
        )
        glue.tables[(None, "db", "l")] = _glue_table("l", Parameters={"table_type": "delta"})
        glue.tables[(None, "db", "k")] = _glue_table("k", Parameters={"TABLE_TYPE": "DELTA"})
        glue.tables[(None, "db", "s")] = _glue_table(
            "s",
            Parameters={},
            StorageDescriptor={
                "Location": "s3://bucket/s",
                "Parameters": {"classification": "delta"},
            },
        )
        glue.tables[(None, "db", "i")] = _glue_table(
            "i",
            Parameters={},
            StorageDescriptor={
                "Location": "s3://bucket/i",
                "InputFormat": "io.delta.hive.DeltaInputFormat",
            },
        )
        for name in "ulksi":
            assert GlueCatalog().resolve(_ref(table=name)).location == f"s3://bucket/{name}"

    def test_glue_non_delta_still_refused(self, fake_boto3: Any) -> None:
        fake_boto3["glue"].tables[(None, "db", "p")] = _glue_table(
            "p",
            Parameters={"classification": "parquet"},
            StorageDescriptor={
                "Location": "s3://bucket/p",
                "Parameters": {"table_type": "ICEBERG"},
            },
        )
        with pytest.raises(InvalidReferenceError, match="not Delta"):
            GlueCatalog().resolve(_ref(table="p"))


class TestReviewFollowUps:
    def test_numpy_snapshot_id_and_limit_are_accepted(self, loaded: IcebergEngine) -> None:
        # [regression] isinstance(int) refused numpy integers, e.g. a snapshot
        # id read back from a history DataFrame, which PyIceberg accepts.
        np = pytest.importorskip("numpy")
        snapshot = loaded.history(_resolved())[0]["version"]
        got = pa.table(loaded.scan(_resolved(), version=np.int64(snapshot), limit=np.int64(2)))
        assert got.num_rows == 2
        with pytest.raises(TypeError):
            loaded.scan(_resolved(), version=1.5)  # type: ignore[arg-type]
