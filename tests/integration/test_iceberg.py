"""The Iceberg engine against real PyIceberg tables on local disk.

No REST server is involved: `catalog_factory` hands the engine a small
file-backed PyIceberg catalog (PyIceberg's own `InMemoryCatalog` needs
SQLAlchemy, which is not a dependency here). Metadata, manifests and Parquet
data are real, so scans, time travel, appends and overwrites run for real. The
REST-specific part -- which properties the engine hands `load_catalog` -- is
tested separately by stubbing `pyiceberg.catalog.load_catalog`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("pyiceberg")

from deltaswamp.capability import Engine, Operation  # noqa: E402
from deltaswamp.catalog.base import ResolvedTable  # noqa: E402
from deltaswamp.engine.iceberg import (  # noqa: E402
    ACCESS_DELEGATION_HEADER,
    IcebergEngine,
    iceberg_rest_uri,
)
from deltaswamp.errors import UnreachableTableError  # noqa: E402
from deltaswamp.identity import parse_ref  # noqa: E402
from pyiceberg.catalog import Catalog, MetastoreCatalog  # noqa: E402
from pyiceberg.exceptions import NoSuchTableError  # noqa: E402
from pyiceberg.io import load_file_io  # noqa: E402
from pyiceberg.partitioning import PartitionField, PartitionSpec  # noqa: E402
from pyiceberg.schema import Schema  # noqa: E402
from pyiceberg.serializers import FromInputFile  # noqa: E402
from pyiceberg.table import CommitTableResponse, Table  # noqa: E402
from pyiceberg.table.locations import load_location_provider  # noqa: E402
from pyiceberg.table.metadata import new_table_metadata  # noqa: E402
from pyiceberg.transforms import IdentityTransform  # noqa: E402
from pyiceberg.types import DateType, LongType, NestedField, StringType  # noqa: E402

REST_URI = "https://ws.example.com/api/2.1/unity-catalog/iceberg-rest"


class DirectoryCatalog(MetastoreCatalog):  # type: ignore[misc,unused-ignore]
    """The smallest PyIceberg catalog that commits: identifier -> metadata file."""

    def __init__(self, name: str, **properties: str) -> None:
        super().__init__(name, **properties)
        self.pointers: dict[tuple[str, ...], str] = {}

    def create_table(  # type: ignore[override,unused-ignore]
        self,
        identifier: str | tuple[str, ...],
        schema: Schema,
        location: str | None = None,
        partition_spec: PartitionSpec | None = None,
        properties: dict[str, str] | None = None,
        **_: Any,
    ) -> Table:
        ident = Catalog.identifier_to_tuple(identifier)
        assert location is not None
        props = properties or {}
        metadata_location = load_location_provider(
            table_location=location, table_properties=props
        ).new_table_metadata_file_location()
        metadata = new_table_metadata(
            location=location,
            schema=schema,
            partition_spec=partition_spec or PartitionSpec(),
            sort_order=_unsorted(),
            properties=props,
        )
        self._write_metadata(metadata, load_file_io(self.properties, location), metadata_location)
        self.pointers[ident] = metadata_location
        return self.load_table(ident)

    def load_table(self, identifier: str | tuple[str, ...]) -> Table:
        ident = Catalog.identifier_to_tuple(identifier)
        if ident not in self.pointers:
            raise NoSuchTableError(f"Table does not exist: {'.'.join(ident)}")
        location = self.pointers[ident]
        io = load_file_io(self.properties, location)
        metadata = FromInputFile.table_metadata(io.new_input(location))
        return Table(ident, metadata, location, io, self)

    def commit_table(self, table: Table, requirements: Any, updates: Any) -> CommitTableResponse:
        ident = Catalog.identifier_to_tuple(table.name())
        current = self.load_table(ident) if ident in self.pointers else None
        staged = self._update_and_stage_table(current, ident, requirements, updates)
        self._write_metadata(staged.metadata, staged.io, staged.metadata_location)
        self.pointers[ident] = staged.metadata_location
        return CommitTableResponse(  # type: ignore[call-arg,unused-ignore]  # pydantic alias
            metadata=staged.metadata, metadata_location=staged.metadata_location
        )

    def _nope(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    create_namespace = drop_namespace = list_namespaces = load_namespace_properties = _nope
    update_namespace_properties = list_tables = drop_table = rename_table = _nope
    register_table = list_views = drop_view = view_exists = load_view = register_view = _nope


def _unsorted() -> Any:
    from pyiceberg.table.sorting import UNSORTED_SORT_ORDER

    return UNSORTED_SORT_ORDER


SCHEMA = Schema(
    NestedField(1, "id", LongType(), required=False),
    NestedField(2, "name", StringType(), required=False),
    NestedField(3, "region", StringType(), required=False),
    NestedField(4, "day", DateType(), required=False),
)
ARROW = pa.schema(
    [("id", pa.int64()), ("name", pa.string()), ("region", pa.string()), ("day", pa.date32())]
)


def _rows(ids: list[int], region: str) -> Any:
    return pa.Table.from_pylist(
        [{"id": i, "name": f"n{i}", "region": region, "day": date(2024, 1, i)} for i in ids],
        schema=ARROW,
    )


class FakeProvider:
    """A credential provider exposing `workspace_auth`, like the UC ones do."""

    def __init__(self, token: str = "tok-1") -> None:
        self.token = token
        self.calls = 0

    table_id = None

    def workspace_auth(self) -> tuple[str, str]:
        self.calls += 1
        return "https://ws.example.com", self.token


@pytest.fixture
def catalog(tmp_path: Path) -> DirectoryCatalog:
    cat = DirectoryCatalog("test", warehouse=tmp_path.as_uri())
    spec = PartitionSpec(
        PartitionField(source_id=3, field_id=1000, transform=IdentityTransform(), name="region")
    )
    cat.create_table(
        ("sales", "orders"),
        SCHEMA,
        location=(tmp_path / "orders").as_uri(),
        partition_spec=spec,
        properties={"owner": "ops"},
    )
    return cat


@pytest.fixture
def engine(catalog: DirectoryCatalog) -> IcebergEngine:
    return IcebergEngine(catalog_factory=lambda name, props: catalog)


def _resolved(provider: Any = None, **overrides: Any) -> ResolvedTable:
    fields: dict[str, Any] = {
        "ref": parse_ref("main.sales.orders"),
        "location": None,
        "data_source_format": "ICEBERG",
        "iceberg_rest_uri": REST_URI,
        "credential_provider": provider,
    }
    fields.update(overrides)
    return ResolvedTable(**fields)


def _uniform(**overrides: Any) -> ResolvedTable:
    return _resolved(
        data_source_format="DELTA",
        location="s3://bucket/orders",
        properties={"delta.universalFormat.enabledFormats": "iceberg"},
        writer_features=frozenset({"icebergCompatV2"}),
        **overrides,
    )


def _ids(reader: Any) -> list[int]:
    return sorted(pa.table(reader).column("id").to_pylist())


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestSupports:
    @pytest.mark.parametrize(
        "op",
        [
            Operation.SCAN,
            Operation.TIME_TRAVEL,
            Operation.HISTORY,
            Operation.DETAIL,
            Operation.APPEND,
            Operation.OVERWRITE,
            Operation.REPLACE_WHERE,
        ],
    )
    def test_native_iceberg(self, engine: IcebergEngine, op: Operation) -> None:
        cap = engine.supports(op, _resolved())
        assert cap.ok and cap.engine is Engine.ICEBERG

    @pytest.mark.parametrize("op", [Operation.CDF, Operation.DELETE, Operation.MERGE])
    def test_unimplemented_operations(self, engine: IcebergEngine, op: Operation) -> None:
        cap = engine.supports(op, _resolved())
        assert not cap.ok and "does not implement" in cap.reason

    def test_needs_a_rest_endpoint(self, engine: IcebergEngine) -> None:
        cap = engine.supports(Operation.SCAN, _resolved(iceberg_rest_uri=None))
        assert not cap.ok and "no Iceberg REST endpoint" in cap.reason and cap.remedy

    def test_plain_delta_is_not_ours(self, engine: IcebergEngine) -> None:
        cap = engine.supports(Operation.SCAN, _resolved(data_source_format="DELTA"))
        assert not cap.ok and "without UniForm" in cap.reason and "Delta engines" in cap.remedy

    @pytest.mark.parametrize(
        "op", [Operation.SCAN, Operation.TIME_TRAVEL, Operation.HISTORY, Operation.DETAIL]
    )
    def test_uniform_reads(self, engine: IcebergEngine, op: Operation) -> None:
        assert engine.supports(op, _uniform()).ok

    @pytest.mark.parametrize("op", [Operation.APPEND, Operation.OVERWRITE])
    def test_uniform_writes_are_refused(self, engine: IcebergEngine, op: Operation) -> None:
        cap = engine.supports(op, _uniform())
        assert not cap.ok and "UniForm" in cap.reason and "Delta engines" in cap.remedy

    def test_manifest_refusals(self, engine: IcebergEngine) -> None:
        assert not engine.supports(Operation.SCAN, _resolved(external_read_supported=False)).ok
        no_write = _resolved(external_write_supported=False)
        assert engine.supports(Operation.SCAN, no_write).ok
        assert not engine.supports(Operation.APPEND, no_write).ok

    def test_flags(self, engine: IcebergEngine) -> None:
        assert engine.supports_predicates and engine.supports_timestamp_travel
        assert engine.supports_commit_metadata and engine.supports_dynamic_overwrite
        assert not engine.supports_distributed_scan and not engine.supports_schema_merge
        assert not engine.supports_idempotent_txn
        with pytest.raises(NotImplementedError):
            engine.plan_scan(_resolved())

    def test_router_lets_iceberg_tables_through(self, engine: IcebergEngine) -> None:
        from deltaswamp.router import Router

        router = Router(engines={Engine.ICEBERG: engine})
        assert router.capability(Operation.SCAN, _resolved()).engine is Engine.ICEBERG
        assert router.engine_for(Operation.APPEND, _resolved()) is engine


# ---------------------------------------------------------------------------
# Reads and writes, for real
# ---------------------------------------------------------------------------


class TestReadWrite:
    def test_append_then_scan(self, engine: IcebergEngine) -> None:
        table = _resolved()
        engine.append(table, _rows([1, 2], "eu"))
        engine.append(table, _rows([3], "us").to_batches()[0])  # a RecordBatch: PyCapsule
        reader = engine.scan(table)
        assert isinstance(reader, pa.RecordBatchReader)
        out = pa.table(reader)
        assert out.schema.field("day").type == pa.date32()
        assert sorted(out.column("id").to_pylist()) == [1, 2, 3]

    def test_columns_and_pushed_down_predicate(self, engine: IcebergEngine) -> None:
        table = _resolved()
        engine.append(table, _rows([1, 2, 3], "eu"))
        out = pa.table(engine.scan(table, columns=["name"], predicate="id >= 2 AND region = 'eu'"))
        assert out.column_names == ["name"]
        assert sorted(out.column("name").to_pylist()) == ["n2", "n3"]

    def test_unparseable_predicate_falls_back_to_an_exact_filter(
        self, engine: IcebergEngine
    ) -> None:
        table = _resolved()
        engine.append(table, _rows([1, 2, 3], "eu"))
        # PyIceberg's parser has no typed DATE literal; the residual filter does.
        out = pa.table(engine.scan(table, columns=["id"], predicate="day > DATE '2024-01-01'"))
        assert out.column_names == ["id"]
        assert sorted(out.column("id").to_pylist()) == [2, 3]

    def test_time_travel(self, engine: IcebergEngine, catalog: DirectoryCatalog) -> None:
        table = _resolved()
        engine.append(table, _rows([1], "eu"))
        first = catalog.load_table(("sales", "orders")).current_snapshot()
        assert first is not None
        engine.append(table, _rows([2], "eu"))
        assert _ids(engine.scan(table, version=first.snapshot_id)) == [1]
        assert _ids(engine.scan(table, timestamp=str(first.timestamp_ms))) == [1]
        assert _ids(engine.scan(table)) == [1, 2]
        with pytest.raises(UnreachableTableError, match="no snapshot with that id"):
            engine.scan(table, version=12345)
        with pytest.raises(UnreachableTableError, match="no snapshot at or before"):
            engine.scan(table, timestamp="2001-01-01T00:00:00Z")

    def test_history_and_detail(self, engine: IcebergEngine) -> None:
        table = _resolved()
        engine.append(table, _rows([1], "eu"), commit_metadata={"job": "nightly", "n": 1})
        engine.overwrite(table, _rows([5], "us"))
        history = engine.history(table)
        # PyIceberg's overwrite commits a delete snapshot, then an append one.
        assert [h["operation"] for h in history] == ["append", "delete", "append"]
        assert history[0]["parent_snapshot_id"] == history[1]["snapshot_id"]
        assert history[-1]["summary"]["job"] == "nightly"
        assert history[-1]["summary"]["n"] == "1"
        assert len(engine.history(table, limit=1)) == 1

        detail = engine.detail(table)
        assert detail["version"] == history[0]["snapshot_id"]
        assert detail["format"] == "iceberg" and detail["format_version"] == 2
        assert detail["partition_columns"] == ["region"]
        assert detail["properties"]["owner"] == "ops"
        assert detail["location"].endswith("/orders")
        assert [f["name"] for f in detail["schema"]["fields"]] == ["id", "name", "region", "day"]
        old = engine.detail(table, version=history[-1]["snapshot_id"])
        assert old["version"] == history[-1]["snapshot_id"]

    def test_empty_table_detail(self, engine: IcebergEngine) -> None:
        assert engine.detail(_resolved())["version"] is None

    def test_overwrite_variants(self, engine: IcebergEngine) -> None:
        table = _resolved()
        engine.append(table, _rows([1, 2], "eu"))
        engine.append(table, _rows([3, 4], "us"))

        engine.overwrite(table, _rows([9], "eu"), predicate="region = 'eu'")
        assert _ids(engine.scan(table)) == [3, 4, 9]

        engine.overwrite(table, _rows([7], "us"), partition_overwrite="dynamic")
        assert _ids(engine.scan(table)) == [7, 9]

        engine.overwrite(table, _rows([8], "ap"))
        assert _ids(engine.scan(table)) == [8]

    def test_overwrite_refuses_what_it_cannot_do_exactly(self, engine: IcebergEngine) -> None:
        table = _resolved()
        with pytest.raises(UnreachableTableError, match="cannot parse"):
            engine.overwrite(table, _rows([1], "eu"), predicate="day > DATE '2024-01-01'")
        with pytest.raises(UnreachableTableError, match="schema_mode"):
            engine.append(table, _rows([1], "eu"), schema_mode="merge")

    def test_missing_table(self, engine: IcebergEngine) -> None:
        with pytest.raises(UnreachableTableError, match=r"no table sales\.nope"):
            engine.scan(_resolved(ref=parse_ref("main.sales.nope")))

    def test_uniform_version_maps_through_the_snapshot_summary(self, engine: IcebergEngine) -> None:
        native = _resolved()
        engine.append(native, _rows([1], "eu"), commit_metadata={"delta-version": "5"})
        engine.append(native, _rows([2], "eu"), commit_metadata={"delta-version": "6"})
        uniform = _uniform()
        assert _ids(engine.scan(uniform, version=5)) == [1]
        with pytest.raises(UnreachableTableError, match="no Iceberg snapshot records"):
            engine.scan(uniform, version=4)


# ---------------------------------------------------------------------------
# REST catalog properties
# ---------------------------------------------------------------------------


class TestRestProperties:
    @pytest.fixture
    def loads(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, str]]]:
        import pyiceberg.catalog

        calls: list[tuple[str, dict[str, str]]] = []

        def fake_load_catalog(name: str, **properties: str) -> object:
            calls.append((name, properties))
            return object()

        monkeypatch.setattr(pyiceberg.catalog, "load_catalog", fake_load_catalog)
        return calls

    def test_properties_passed_to_load_catalog(
        self, loads: list[tuple[str, dict[str, str]]]
    ) -> None:
        engine = IcebergEngine()
        engine._catalog(_resolved(FakeProvider("tok-1")))
        name, props = loads[-1]
        assert name == "deltaswamp-main"
        assert props == {
            "type": "rest",
            "uri": REST_URI,
            "warehouse": "main",
            "token": "tok-1",
            ACCESS_DELEGATION_HEADER: "vended-credentials",
        }
        assert ACCESS_DELEGATION_HEADER == "header.X-Iceberg-Access-Delegation"

    def test_catalog_is_cached_until_the_token_changes(
        self, loads: list[tuple[str, dict[str, str]]]
    ) -> None:
        engine = IcebergEngine()
        provider = FakeProvider("tok-1")
        first = engine._catalog(_resolved(provider))
        assert engine._catalog(_resolved(provider)) is first
        assert len(loads) == 1
        assert provider.calls == 2  # asked every time, never trusted from cache

        provider.token = "tok-2"  # the SDK refreshed it
        second = engine._catalog(_resolved(provider))
        assert second is not first and loads[-1][1]["token"] == "tok-2"

        # A different warehouse is a different catalog.
        engine._catalog(_resolved(provider, ref=parse_ref("other.sales.orders")))
        assert loads[-1][1]["warehouse"] == "other"

    def test_static_token_and_overrides(self, loads: list[tuple[str, dict[str, str]]]) -> None:
        engine = IcebergEngine(token="static", properties={"ssl.cabundle": "/ca.pem"})
        engine._catalog(_resolved())
        props = loads[-1][1]
        assert props["token"] == "static" and props["ssl.cabundle"] == "/ca.pem"

    def test_no_token_at_all(self, loads: list[tuple[str, dict[str, str]]]) -> None:
        IcebergEngine()._catalog(_resolved(FakeProvider("")))
        assert "token" not in loads[-1][1]

    def test_engine_pickles_without_catalogs(self) -> None:
        import pickle

        engine = IcebergEngine(token="t")
        engine._catalogs[("u", "w")] = ("t", object())
        clone = pickle.loads(pickle.dumps(engine))
        assert clone._catalogs == {}

    def test_endpoint_helper(self) -> None:
        assert (
            iceberg_rest_uri("https://ws.example.com/", databricks=True)
            == "https://ws.example.com/api/2.1/unity-catalog/iceberg-rest"
        )
        assert (
            iceberg_rest_uri("http://localhost:8080", databricks=False)
            == "http://localhost:8080/api/2.1/unity-catalog/iceberg"
        )
