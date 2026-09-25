"""Delta Sharing, end to end against an in-process protocol server.

The server below speaks the open Delta Sharing REST protocol (PROTOCOL.md):
list shares/schemas/tables, the table-version header, metadata, ``query`` with
NDJSON protocol/metaData/file lines, and ``changes``. File URLs point back at
the same server, which serves real Parquet bytes -- so the real `delta_sharing`
client and the real presigned-URL download path are exercised, not mocks.
"""

from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlparse

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("delta_sharing")

from deltaswamp.capability import Engine, Operation  # noqa: E402
from deltaswamp.catalog.base import ResolvedTable  # noqa: E402
from deltaswamp.catalog.registry import catalog_for_uri, scheme_to_catalog  # noqa: E402
from deltaswamp.catalog.sharing import SharingCatalog  # noqa: E402
from deltaswamp.engine.sharing import (  # noqa: E402
    SharingEngine,
    delta_schema_to_arrow,
    filter_arrow_exact,
    json_predicate_hints,
)
from deltaswamp.errors import InvalidReferenceError, UnreachableTableError  # noqa: E402
from deltaswamp.identity import RefKind, TableRef, parse_ref  # noqa: E402

TOKEN = "s3cr3t-recipient-token"

# ---------------------------------------------------------------------------
# The fake server
# ---------------------------------------------------------------------------


def _parquet(table: Any) -> bytes:
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


def _ms(text: str) -> int:
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)


@dataclass
class FakeFile:
    id: str
    data: bytes
    partition_values: dict[str, str | None] = field(default_factory=dict)


@dataclass
class FakeVersion:
    version: int
    timestamp_ms: int
    files: list[FakeFile]


@dataclass
class FakeChange:
    kind: str  # "add" | "cdf" | "remove"
    version: int
    timestamp_ms: int
    file: FakeFile


@dataclass
class FakeSharedTable:
    share: str
    schema: str
    name: str
    schema_string: str
    versions: list[FakeVersion]
    partition_columns: list[str] = field(default_factory=list)
    configuration: dict[str, str] = field(default_factory=dict)
    changes: list[FakeChange] = field(default_factory=list)
    history_shared: bool = True
    #: Reader features that make the server answer only in delta format.
    delta_reader_features: list[str] = field(default_factory=list)
    table_id: str = "11111111-2222-3333-4444-555555555555"

    @property
    def latest(self) -> FakeVersion:
        return self.versions[-1]


class FakeSharingServer:
    """Serves tables over the Delta Sharing protocol. Start, use `.endpoint`, stop."""

    def __init__(self) -> None:
        self.tables: list[FakeSharedTable] = []
        self.files: dict[str, bytes] = {}
        self.queries: list[dict[str, Any]] = []
        self.request_headers: list[dict[str, str]] = []
        self._server: ThreadingHTTPServer | None = None

    @property
    def endpoint(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}/delta-sharing"

    def add(self, table: FakeSharedTable) -> FakeSharedTable:
        self.tables.append(table)
        for v in table.versions:
            for f in v.files:
                self.files[f.id] = f.data
        for c in table.changes:
            self.files[c.file.id] = c.file.data
        return table

    def _table(self, share: str, schema: str, name: str) -> FakeSharedTable | None:
        for t in self.tables:
            if (t.share, t.schema, t.name) == (share, schema, name):
                return t
        return None

    def start(self) -> FakeSharingServer:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _reply(
                self,
                code: int,
                lines: list[dict[str, Any]],
                headers: dict[str, str] | None = None,
            ) -> None:
                body = "".join(json.dumps(line) + "\n" for line in lines).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _error(self, code: int, error_code: str, message: str) -> None:
                body = json.dumps({"errorCode": error_code, "message": message}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authorised(self) -> bool:
                server.request_headers.append(dict(self.headers.items()))
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    self._error(401, "UNAUTHENTICATED", "bad bearer token")
                    return False
                return True

            def _wants_delta(self, t: FakeSharedTable) -> bool | None:
                """True: answer in delta format. None: refuse (the table needs it)."""
                asked = "responseformat=delta" in (
                    self.headers.get("delta-sharing-capabilities") or ""
                )
                if t.delta_reader_features:
                    return True if asked else None
                return False

            def _delta_lines(self, t: FakeSharedTable) -> list[dict[str, Any]]:
                features = t.delta_reader_features
                return [
                    {
                        "protocol": {
                            "deltaProtocol": {
                                "minReaderVersion": 3,
                                "minWriterVersion": 7,
                                "readerFeatures": features,
                                "writerFeatures": features,
                            }
                        }
                    },
                    {
                        "metaData": {
                            "version": t.latest.version,
                            "deltaMetadata": {
                                "id": t.table_id,
                                "name": t.name,
                                "format": {"provider": "parquet", "options": {}},
                                "schemaString": t.schema_string,
                                "partitionColumns": t.partition_columns,
                                "configuration": t.configuration,
                            },
                        }
                    },
                ]

            def _delta_file(self, f: FakeFile) -> dict[str, Any]:
                return {
                    "file": {
                        "id": f.id,
                        "deltaSingleAction": {
                            "add": {
                                "path": f"{server.endpoint}/files/{f.id}",
                                "partitionValues": f.partition_values,
                                "size": len(f.data),
                                "modificationTime": 0,
                                "dataChange": True,
                            }
                        },
                    }
                }

            def _needs_delta_error(self) -> None:
                self._error(
                    400,
                    "INVALID_PARAMETER_VALUE",
                    "table requires delta format: responseformat=delta and readerfeatures",
                )

            @staticmethod
            def _protocol_and_metadata(t: FakeSharedTable) -> list[dict[str, Any]]:
                return [
                    {"protocol": {"minReaderVersion": 1}},
                    {
                        "metaData": {
                            "id": t.table_id,
                            "name": t.name,
                            "format": {"provider": "parquet"},
                            "schemaString": t.schema_string,
                            "partitionColumns": t.partition_columns,
                            "configuration": t.configuration,
                        }
                    },
                ]

            def _file_json(self, f: FakeFile) -> dict[str, Any]:
                return {
                    "url": f"{server.endpoint}/files/{f.id}",
                    "id": f.id,
                    "partitionValues": f.partition_values,
                    "size": len(f.data),
                }

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                parts = [p for p in parsed.path.split("/") if p][1:]  # drop "delta-sharing"
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}

                if parts[:1] == ["files"]:
                    data = server.files.get(parts[1])
                    if data is None:
                        return self._error(403, "EXPIRED", "presigned URL expired")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return None

                if not self._authorised():
                    return None

                if parts == ["shares"]:
                    shares = sorted({t.share for t in server.tables})
                    return self._reply(200, [{"items": [{"name": s} for s in shares]}])
                if len(parts) == 3 and parts[0] == "shares" and parts[2] == "schemas":
                    names = sorted({t.schema for t in server.tables if t.share == parts[1]})
                    items = [{"name": n, "share": parts[1]} for n in names]
                    return self._reply(200, [{"items": items}])
                if len(parts) == 5 and parts[4] == "tables":
                    items = [
                        {"name": t.name, "share": t.share, "schema": t.schema}
                        for t in server.tables
                        if (t.share, t.schema) == (parts[1], parts[3])
                    ]
                    return self._reply(200, [{"items": items}])
                if len(parts) == 7 and parts[4] == "tables":
                    t = server._table(parts[1], parts[3], parts[5])
                    if t is None:
                        return self._error(404, "RESOURCE_DOES_NOT_EXIST", "no such table")
                    version_header = {"Delta-Table-Version": str(t.latest.version)}
                    if parts[6] == "version":
                        if "startingTimestamp" in query:
                            ts = _ms(query["startingTimestamp"])
                            later = [v for v in t.versions if v.timestamp_ms >= ts]
                            version_header = {"Delta-Table-Version": str(later[0].version)}
                        return self._reply(200, [], version_header)
                    if parts[6] == "metadata":
                        delta = self._wants_delta(t)
                        if delta:
                            version_header["delta-sharing-capabilities"] = "responseformat=delta"
                            return self._reply(200, self._delta_lines(t), version_header)
                        return self._reply(200, self._protocol_and_metadata(t), version_header)
                    if parts[6] == "changes":
                        return self._changes(t, query)
                return self._error(404, "NOT_FOUND", self.path)

            def _changes(self, t: FakeSharedTable, query: dict[str, str]) -> None:
                if not t.history_shared:
                    return self._error(
                        400,
                        "INVALID_PARAMETER_VALUE",
                        "cdf is not enabled on table " + f"{t.share}.{t.schema}.{t.name}",
                    )
                start = int(query.get("startingVersion", 0))
                end = int(query.get("endingVersion", t.latest.version))
                lines = self._protocol_and_metadata(t)
                for c in t.changes:
                    if start <= c.version <= end:
                        body = self._file_json(c.file)
                        body.update(version=c.version, timestamp=c.timestamp_ms)
                        lines.append({c.kind: body})
                return self._reply(200, lines, {"Delta-Table-Version": str(end)})

            def do_POST(self) -> None:
                if not self._authorised():
                    return None
                parts = [p for p in urlparse(self.path).path.split("/") if p][1:]
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                server.queries.append(body)
                t = server._table(parts[1], parts[3], parts[5])
                if t is None or parts[6] != "query":
                    return self._error(404, "RESOURCE_DOES_NOT_EXIST", "no such table")

                snapshot = t.latest
                if "version" in body or "timestamp" in body:
                    if not t.history_shared:
                        return self._error(
                            400,
                            "INVALID_PARAMETER_VALUE",
                            "Reading table by version or timestamp is not supported because "
                            "history sharing is not enabled on table",
                        )
                    if "version" in body:
                        matches = [v for v in t.versions if v.version == int(body["version"])]
                    else:
                        ts = _ms(body["timestamp"])
                        matches = [v for v in t.versions if v.timestamp_ms <= ts][-1:]
                    if not matches:
                        return self._error(400, "INVALID_PARAMETER_VALUE", "no such version")
                    snapshot = matches[0]
                headers = {"Delta-Table-Version": str(snapshot.version)}
                delta = self._wants_delta(t)
                if delta is None:
                    return self._needs_delta_error()
                if delta:
                    headers["delta-sharing-capabilities"] = "responseformat=delta"
                    lines = self._delta_lines(t) + [self._delta_file(f) for f in snapshot.files]
                    return self._reply(200, lines, headers)
                lines = self._protocol_and_metadata(t)
                lines += [{"file": self._file_json(f)} for f in snapshot.files]
                return self._reply(200, lines, headers)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


# ---------------------------------------------------------------------------
# Fixtures: a partitioned table with rich types, three versions, and CDF
# ---------------------------------------------------------------------------

SCHEMA_STRING = json.dumps(
    {
        "type": "struct",
        "fields": [
            {"name": "id", "type": "long", "nullable": False, "metadata": {}},
            {"name": "qty", "type": "integer", "nullable": True, "metadata": {}},
            {"name": "name", "type": "string", "nullable": True, "metadata": {}},
            {"name": "price", "type": "decimal(10,2)", "nullable": True, "metadata": {}},
            {"name": "at", "type": "timestamp", "nullable": True, "metadata": {}},
            {
                "name": "tags",
                "type": {"type": "array", "elementType": "string", "containsNull": True},
                "nullable": True,
                "metadata": {},
            },
            {"name": "day", "type": "date", "nullable": True, "metadata": {}},
        ],
    }
)

MAPPED_SCHEMA = json.dumps(
    {
        "type": "struct",
        "fields": [
            {
                "name": "id",
                "type": "long",
                "nullable": True,
                "metadata": {
                    "delta.columnMapping.id": 1,
                    "delta.columnMapping.physicalName": "col-1",
                },
            },
            {
                "name": "label",
                "type": "string",
                "nullable": True,
                "metadata": {
                    "delta.columnMapping.id": 2,
                    "delta.columnMapping.physicalName": "col-2",
                },
            },
        ],
    }
)

T0, T1, T2 = "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z", "2024-03-01T00:00:00Z"


def _rows(ids: list[int], day: str) -> FakeFile:
    """One Parquet file with the non-partition columns, like a Delta writer makes."""
    n = len(ids)
    data = pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "qty": pa.array([i * 10 if i % 2 else None for i in ids], pa.int32()),
            "name": pa.array([f"n{i}" for i in ids]),
            "price": pa.array([Decimal(f"{i}.25") for i in ids], pa.decimal128(10, 2)),
            # Written as nanoseconds, as Spark's INT96 would come back.
            "at": pa.array(
                [datetime(2024, 1, i % 28 + 1, tzinfo=UTC) for i in ids],
                pa.timestamp("ns", tz="UTC"),
            ),
            "tags": pa.array([["a", str(i)] for i in ids]),
        }
    )
    assert data.num_rows == n
    return FakeFile(id=f"f-{day}-{ids[0]}", data=_parquet(data), partition_values={"day": day})


@pytest.fixture(scope="module")
def server() -> Any:
    srv = FakeSharingServer().start()
    f1 = _rows([1, 2, 3], "2024-01-01")
    f2 = _rows([4, 5], "2024-01-02")
    f3 = _rows([6], "2024-01-03")
    srv.add(
        FakeSharedTable(
            share="retail",
            schema="sales",
            name="orders",
            schema_string=SCHEMA_STRING,
            partition_columns=["day"],
            configuration={"delta.enableChangeDataFeed": "true"},
            versions=[
                FakeVersion(0, _ms(T0), [f1]),
                FakeVersion(1, _ms(T1), [f1, f2]),
                FakeVersion(2, _ms(T2), [f2, f3]),
            ],
            changes=[
                FakeChange("add", 1, _ms(T1), f2),
                FakeChange("remove", 2, _ms(T2), f1),
                FakeChange("add", 2, _ms(T2), f3),
            ],
        )
    )
    srv.add(
        FakeSharedTable(
            share="retail",
            schema="sales",
            name="snapshot_only",
            schema_string=SCHEMA_STRING,
            partition_columns=["day"],
            versions=[FakeVersion(7, _ms(T0), [_rows([9], "2024-05-05")])],
            history_shared=False,
        )
    )
    srv.add(
        FakeSharedTable(
            share="hr",
            schema="people",
            name="staff",
            schema_string=json.dumps(
                {"type": "struct", "fields": [{"name": "x", "type": "long", "nullable": True}]}
            ),
            versions=[FakeVersion(0, _ms(T0), [])],
        )
    )
    mapped = pa.table({"col-1": pa.array([1, 2, 3], pa.int64()), "col-2": ["a", "b", "c"]})
    srv.add(
        FakeSharedTable(
            share="retail",
            schema="mapped",
            name="renamed",
            schema_string=MAPPED_SCHEMA,
            configuration={"delta.columnMapping.mode": "name"},
            versions=[FakeVersion(4, _ms(T0), [FakeFile("mapped-1", _parquet(mapped))])],
            delta_reader_features=["columnMapping"],
        )
    )
    yield srv
    srv.stop()


@pytest.fixture
def profile_path(server: FakeSharingServer, tmp_path: Path) -> Path:
    path = tmp_path / "config.share"
    path.write_text(
        json.dumps(
            {"shareCredentialsVersion": 1, "endpoint": server.endpoint, "bearerToken": TOKEN}
        )
    )
    return path


@pytest.fixture
def catalog(profile_path: Path) -> SharingCatalog:
    return SharingCatalog.from_uri(f"sharing://{profile_path}")


@pytest.fixture
def orders(catalog: SharingCatalog) -> ResolvedTable:
    return catalog.resolve(parse_ref("retail.sales.orders"))


ENGINE = SharingEngine(num_retries=0)


def _read(reader: Any) -> Any:
    return pa.table(reader)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class TestCatalogConstruction:
    def test_scheme_mapping(self) -> None:
        for scheme in ("sharing", "sharing+https", "sharing+http"):
            assert scheme_to_catalog[scheme] == "sharing"

    def test_profile_path_uri(self, profile_path: Path, server: FakeSharingServer) -> None:
        catalog = SharingCatalog.from_uri(f"sharing://{profile_path}")
        assert catalog.endpoint == server.endpoint
        assert catalog.profile == str(profile_path)  # a path stays a path: no secret copied

    def test_profile_kwarg_dict_and_json(self, server: FakeSharingServer) -> None:
        doc = {"shareCredentialsVersion": 1, "endpoint": server.endpoint, "bearerToken": TOKEN}
        assert SharingCatalog.from_uri("sharing://", profile=doc).endpoint == server.endpoint
        by_json = SharingCatalog.from_uri(None, profile=json.dumps(doc))
        assert by_json.endpoint == server.endpoint

    def test_endpoint_plus_token(self, server: FakeSharingServer) -> None:
        body = server.endpoint.removeprefix("http://")
        catalog = SharingCatalog.from_uri(f"sharing+http://{body}", token=TOKEN)
        assert catalog.endpoint == server.endpoint
        assert catalog.list_catalogs() == ["hr", "retail"]

    def test_oauth_client_credentials_profile_parses(self) -> None:
        doc = {
            "shareCredentialsVersion": 2,
            "type": "oauth_client_credentials",
            "endpoint": "https://sharing.example.com/delta-sharing/",
            "tokenEndpoint": "https://login.example.com/token",
            "clientId": "id",
            "clientSecret": "secret",
        }
        catalog = SharingCatalog(doc)
        assert catalog.endpoint == "https://sharing.example.com/delta-sharing"

    def test_repr_hides_token(self, server: FakeSharingServer) -> None:
        doc = {"shareCredentialsVersion": 1, "endpoint": server.endpoint, "bearerToken": TOKEN}
        assert TOKEN not in repr(SharingCatalog(doc))

    @pytest.mark.parametrize(
        ("uri", "kwargs", "match"),
        [
            ("sharing+https://host/api", {}, "no bearer token"),
            ("sharing://", {}, "needs a profile"),
            ("sharing:///a.share", {"profile": "/b.share"}, "one or the other"),
            ("sharing:///definitely/not/here.share", {}, "does not exist"),
            ("sharing://", {"profile": '{"endpoint": "x"}'}, "not a valid profile"),
            ("sharing://", {"profile": "{not json"}, "does not parse"),
        ],
    )
    def test_bad_configuration_is_explained(
        self, uri: str, kwargs: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(InvalidReferenceError, match=match):
            SharingCatalog.from_uri(uri, **kwargs)

    def test_connect_style_kwargs_are_accepted(self, profile_path: Path) -> None:
        # connect() always passes these Databricks settings through.
        catalog = catalog_for_uri(
            f"sharing://{profile_path}",
            profile=None,
            host=None,
            token=None,
            config=None,
            catalog_name="deltaswamp.catalog.sharing:SharingCatalog",
        )
        assert isinstance(catalog, SharingCatalog)

    def test_entry_point(self, profile_path: Path) -> None:
        from deltaswamp.catalog.registry import available_catalogs

        if "sharing" not in available_catalogs():
            pytest.skip("distribution metadata predates the sharing entry point; reinstall")
        catalog = catalog_for_uri(f"sharing://{profile_path}", profile=None, host=None)
        assert isinstance(catalog, SharingCatalog)


class TestCatalogDiscovery:
    def test_listing(self, catalog: SharingCatalog) -> None:
        assert catalog.list_catalogs() == ["hr", "retail"]
        assert catalog.list_schemas("retail") == ["mapped", "sales"]
        tables = catalog.list_tables("retail", "sales")
        assert sorted(t.ref.table or "" for t in tables) == ["orders", "snapshot_only"]
        assert all(t.is_shared and t.location is None for t in tables)

    def test_resolve_fills_protocol_from_the_server(
        self, orders: ResolvedTable, profile_path: Path
    ) -> None:
        assert orders.location is None
        assert orders.data_source_format == "DELTA"
        assert orders.is_shared and orders.sharing_profile == str(profile_path)
        assert orders.partition_columns == ("day",)
        assert orders.properties == {"delta.enableChangeDataFeed": "true"}
        assert orders.min_reader_version == 1
        assert orders.table_uuid == "11111111-2222-3333-4444-555555555555"
        assert orders.external_write_supported is False

    def test_resolve_unknown_table(self, catalog: SharingCatalog) -> None:
        with pytest.raises(InvalidReferenceError, match="not visible through this share"):
            catalog.resolve(parse_ref("retail.sales.nope"))

    def test_resolve_rejects_paths(self, catalog: SharingCatalog) -> None:
        with pytest.raises(InvalidReferenceError, match=r"share\.schema\.table"):
            catalog.resolve(TableRef(kind=RefKind.PATH, path="/tmp/x"))

    def test_drop_is_refused(self, catalog: SharingCatalog) -> None:
        with pytest.raises(NotImplementedError, match="read-only"):
            catalog.drop_table(parse_ref("retail.sales.orders"))

    def test_bad_token_is_reported(self, server: FakeSharingServer) -> None:
        doc = {"shareCredentialsVersion": 1, "endpoint": server.endpoint, "bearerToken": "wrong"}
        with pytest.raises(UnreachableTableError, match="401"):
            SharingCatalog(doc).resolve(parse_ref("retail.sales.orders"))


# ---------------------------------------------------------------------------
# Engine: capabilities
# ---------------------------------------------------------------------------


class TestSupports:
    @pytest.mark.parametrize(
        "op", [Operation.SCAN, Operation.TIME_TRAVEL, Operation.CDF, Operation.DETAIL]
    )
    def test_reads_are_served(self, orders: ResolvedTable, op: Operation) -> None:
        cap = ENGINE.supports(op, orders)
        assert cap.ok and cap.engine is Engine.SHARING

    @pytest.mark.parametrize(
        "op", [Operation.APPEND, Operation.OVERWRITE, Operation.DELETE, Operation.MERGE]
    )
    def test_writes_are_refused_as_read_only(self, orders: ResolvedTable, op: Operation) -> None:
        cap = ENGINE.supports(op, orders)
        assert not cap.ok
        assert "read-only" in cap.reason and cap.remedy

    def test_history_is_not_in_the_protocol(self, orders: ResolvedTable) -> None:
        cap = ENGINE.supports(Operation.HISTORY, orders)
        assert not cap.ok and "no history endpoint" in cap.reason and "cdf" in cap.remedy

    def test_files_are_refused(self, orders: ResolvedTable) -> None:
        cap = ENGINE.supports(Operation.FILES, orders)
        assert not cap.ok and "presigned" in cap.reason

    def test_unshared_tables_are_refused(self) -> None:
        plain = ResolvedTable(ref=parse_ref("/tmp/table"), location="/tmp/table")
        cap = ENGINE.supports(Operation.SCAN, plain)
        assert not cap.ok and "not reached through Delta Sharing" in cap.reason
        assert "sharing://" in cap.remedy

    def test_flags(self) -> None:
        assert ENGINE.supports_predicates and ENGINE.supports_timestamp_travel
        assert not ENGINE.supports_distributed_scan
        assert not ENGINE.supports_schema_merge and not ENGINE.supports_idempotent_txn
        with pytest.raises(NotImplementedError):
            ENGINE.plan_scan(ResolvedTable(ref=parse_ref("a.b.c"), location=None))


# ---------------------------------------------------------------------------
# Engine: reads
# ---------------------------------------------------------------------------


class TestScan:
    def test_types_survive_including_partition_values(self, orders: ResolvedTable) -> None:
        reader = ENGINE.scan(orders)
        assert hasattr(reader, "__arrow_c_stream__")
        out = _read(reader)
        assert out.schema == delta_schema_to_arrow(SCHEMA_STRING)
        assert out.schema.field("at").type == pa.timestamp("us", tz="UTC")
        assert out.schema.field("price").type == pa.decimal128(10, 2)
        assert out.schema.field("day").type == pa.date32()
        assert out.schema.field("qty").type == pa.int32()  # not float64, as pandas would make it
        rows = sorted(out.to_pylist(), key=lambda r: r["id"])
        assert [r["id"] for r in rows] == [4, 5, 6]
        assert rows[0]["day"] == date(2024, 1, 2)
        assert rows[0]["qty"] is None and rows[1]["qty"] == 50
        assert rows[2]["price"] == Decimal("6.25")
        assert rows[0]["tags"] == ["a", "4"]

    def test_columns(self, orders: ResolvedTable) -> None:
        out = _read(ENGINE.scan(orders, columns=["day", "id"]))
        assert out.column_names == ["day", "id"]

    def test_unknown_column(self, orders: ResolvedTable) -> None:
        with pytest.raises(UnreachableTableError, match="no column"):
            ENGINE.scan(orders, columns=["nope"])

    def test_predicate_is_exact_and_sent_as_a_hint(
        self, orders: ResolvedTable, server: FakeSharingServer
    ) -> None:
        server.queries.clear()
        # id=4 and id=5 share a file: the server can only skip files, so an
        # exact filter must still drop id=5 from the file it returns.
        out = _read(ENGINE.scan(orders, columns=["id"], predicate="id = 4 AND name <> 'zz'"))
        assert out.column("id").to_pylist() == [4]
        assert out.column_names == ["id"]
        hint = json.loads(server.queries[-1]["jsonPredicateHints"])
        assert hint["op"] == "and"
        assert hint["children"][0] == {
            "op": "equal",
            "children": [
                {"op": "column", "name": "id", "valueType": "long"},
                {"op": "literal", "value": "4", "valueType": "long"},
            ],
        }

    def test_predicate_on_partition_column(self, orders: ResolvedTable) -> None:
        out = _read(ENGINE.scan(orders, predicate="day = DATE '2024-01-03'"))
        assert out.column("id").to_pylist() == [6]

    def test_time_travel_by_version(self, orders: ResolvedTable) -> None:
        assert sorted(_read(ENGINE.scan(orders, version=0)).column("id").to_pylist()) == [1, 2, 3]
        at1 = sorted(_read(ENGINE.scan(orders, version=1)).column("id").to_pylist())
        assert at1 == [1, 2, 3, 4, 5]

    def test_time_travel_by_timestamp(self, orders: ResolvedTable) -> None:
        out = _read(ENGINE.scan(orders, timestamp="2024-02-15T00:00:00Z"))
        assert sorted(out.column("id").to_pylist()) == [1, 2, 3, 4, 5]

    def test_time_travel_without_history_names_the_remedy(self, catalog: SharingCatalog) -> None:
        table = catalog.resolve(parse_ref("retail.sales.snapshot_only"))
        assert _read(ENGINE.scan(table)).column("id").to_pylist() == [9]
        with pytest.raises(UnreachableTableError, match="WITH HISTORY") as info:
            ENGINE.scan(table, version=3)
        assert "history sharing is not enabled" in str(info.value)

    def test_empty_table(self, catalog: SharingCatalog) -> None:
        out = _read(ENGINE.scan(catalog.resolve(parse_ref("hr.people.staff"))))
        assert out.num_rows == 0 and out.column_names == ["x"]


class TestCdf:
    def test_version_range(self, orders: ResolvedTable) -> None:
        out = _read(ENGINE.cdf(orders, starting_version=1, ending_version=2))
        assert out.schema.field("_commit_version").type == pa.int64()
        assert out.schema.field("_commit_timestamp").type == pa.timestamp("ms", tz="UTC")
        got = sorted((r["_commit_version"], r["_change_type"], r["id"]) for r in out.to_pylist())
        assert got == [
            (1, "insert", 4),
            (1, "insert", 5),
            (2, "delete", 1),
            (2, "delete", 2),
            (2, "delete", 3),
            (2, "insert", 6),
        ]
        first = next(r for r in out.to_pylist() if r["_commit_version"] == 1)
        assert first["_commit_timestamp"] == datetime(2024, 2, 1, tzinfo=UTC)
        assert first["day"] == date(2024, 1, 2)

    def test_columns_and_predicate(self, orders: ResolvedTable) -> None:
        out = _read(ENGINE.cdf(orders, starting_version=2, columns=["id"], predicate="id > 2"))
        assert out.column_names == ["id", "_change_type", "_commit_version", "_commit_timestamp"]
        assert sorted(out.column("id").to_pylist()) == [3, 6]

    def test_history_not_shared_surfaces_the_server_error(self, catalog: SharingCatalog) -> None:
        table = catalog.resolve(parse_ref("retail.sales.snapshot_only"))
        with pytest.raises(UnreachableTableError) as info:
            ENGINE.cdf(table, starting_version=0)
        message = str(info.value)
        assert "cdf is not enabled" in message and "400" in message
        assert "WITH HISTORY" in message


class TestDetail:
    def test_latest(self, orders: ResolvedTable) -> None:
        d = ENGINE.detail(orders)
        assert d["version"] == 2
        assert d["location"] is None
        assert d["partition_columns"] == ["day"]
        assert d["properties"] == {"delta.enableChangeDataFeed": "true"}
        assert d["min_reader_version"] == 1
        assert d["metadata_id"] == "11111111-2222-3333-4444-555555555555"

    def test_at_version(self, orders: ResolvedTable) -> None:
        assert ENGINE.detail(orders, version=1)["version"] == 1


# ---------------------------------------------------------------------------
# Through the router, as a Connection would use it
# ---------------------------------------------------------------------------


class TestRouted:
    def test_table_reads_route_to_sharing(self, catalog: SharingCatalog) -> None:
        from deltaswamp import Connection
        from deltaswamp.engine.deltars import DeltaRsEngine
        from deltaswamp.router import Router

        router = Router(
            engines={
                Engine.DELTARS: DeltaRsEngine(),
                Engine.SHARING: ENGINE,
            }
        )
        conn = Connection(catalog=catalog, router=router)
        t = conn.table("retail.sales.orders")
        assert t.to_arrow(predicate="id >= 5").num_rows == 2
        assert t.to_arrow(version=0).num_rows == 3
        assert t.version == 2
        caps = t.capabilities()
        assert caps[Operation.SCAN].engine is Engine.SHARING
        # SHARING is not in the write chains, so the router reports the other
        # engines' refusals; the sharing engine's own read-only reason is
        # asserted in TestSupports.
        assert not caps[Operation.APPEND].ok
        assert not caps[Operation.HISTORY].ok
        with pytest.raises(UnreachableTableError):
            t.append(pa.table({"id": [1]}))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestHints:
    def test_unexpressible_conjuncts_are_dropped(self) -> None:
        hint = json_predicate_hints("(id = 1 OR id = 2) AND name = 'it''s'", SCHEMA_STRING)
        assert hint is not None
        assert json.loads(hint) == {
            "op": "equal",
            "children": [
                {"op": "column", "name": "name", "valueType": "string"},
                {"op": "literal", "value": "it's", "valueType": "string"},
            ],
        }

    def test_type_mismatch_gives_no_hint(self) -> None:
        assert json_predicate_hints("id = 'x'", SCHEMA_STRING) is None
        assert json_predicate_hints("id = 1.5", SCHEMA_STRING) is None
        assert json_predicate_hints("nope = 1", SCHEMA_STRING) is None

    def test_null_checks_and_negation(self) -> None:
        hint = json.loads(json_predicate_hints("qty IS NOT NULL AND id != 3", SCHEMA_STRING) or "")
        assert [c["op"] for c in hint["children"]] == ["not", "not"]
        assert hint["children"][0]["children"][0]["op"] == "isNull"

    def test_between_is_not_mistranslated(self) -> None:
        # Splitting on AND must not turn "BETWEEN 1 AND 3" into a bogus hint.
        assert json_predicate_hints("id BETWEEN 1 AND 3", SCHEMA_STRING) is None


class TestExactFilter:
    def test_null_is_false_and_order_kept(self) -> None:
        table = pa.table({"a": pa.array([3, None, 1, 2], pa.int32())})
        assert filter_arrow_exact(table, "a > 1").column("a").to_pylist() == [3, 2]


# ---------------------------------------------------------------------------
# Delta-format responses (column mapping / deletion vectors)
# ---------------------------------------------------------------------------


class _FakeKernelScan:
    """Stands in for `delta_kernel_rust_sharing_wrapper`.

    The real wrapper resolves every add path against the (local) table root's
    object store, so it cannot fetch the plain-http URLs this test server hands
    out; real servers issue https presigned URLs. This stand-in replays the log
    the engine wrote -- protocol, metadata, adds -- downloads each add, and maps
    physical to logical column names, which is the part column mapping needs.
    """

    logs: ClassVar[list[list[dict[str, Any]]]] = []

    def __init__(self, snapshot: str) -> None:
        self.uri = snapshot

    def build(self) -> _FakeKernelScan:
        return self

    def execute(self, interface: object) -> Any:
        from urllib.request import urlopen

        root = Path(urlparse(self.uri).path)
        log = root / "_delta_log" / f"{0:020d}.json"
        actions = [json.loads(line) for line in log.read_text().splitlines()]
        _FakeKernelScan.logs.append(actions)
        schema = json.loads(actions[1]["metaData"]["schemaString"])
        logical = {
            f["metadata"]["delta.columnMapping.physicalName"]: f["name"] for f in schema["fields"]
        }
        tables = []
        for action in actions[2:]:
            with urlopen(action["add"]["path"]) as response:
                data = pq.read_table(pa.BufferReader(response.read()))
            tables.append(data.rename_columns([logical[n] for n in data.column_names]))
        table = pa.concat_tables(tables)
        return pa.RecordBatchReader.from_batches(table.schema, table.to_batches())


class _FakeKernelTable:
    def __init__(self, uri: str) -> None:
        self.uri = uri

    def snapshot(self, interface: object) -> str:
        return self.uri


@pytest.fixture
def fake_kernel(monkeypatch: pytest.MonkeyPatch) -> type[_FakeKernelScan]:
    import sys
    import types

    module = types.ModuleType("delta_kernel_rust_sharing_wrapper")
    module.PythonInterface = lambda uri: uri  # type: ignore[attr-defined]
    module.Table = _FakeKernelTable  # type: ignore[attr-defined]
    module.ScanBuilder = _FakeKernelScan  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "delta_kernel_rust_sharing_wrapper", module)
    _FakeKernelScan.logs.clear()
    return _FakeKernelScan


class TestDeltaFormat:
    @pytest.fixture
    def renamed(self, catalog: SharingCatalog) -> ResolvedTable:
        return catalog.resolve(parse_ref("retail.mapped.renamed"))

    def test_resolve_reports_the_full_protocol(self, renamed: ResolvedTable) -> None:
        assert renamed.reader_features == frozenset({"columnMapping"})
        assert renamed.min_reader_version == 3
        assert renamed.properties["delta.columnMapping.mode"] == "name"

    def test_scan_replays_the_log_with_logical_names(
        self, renamed: ResolvedTable, fake_kernel: type[_FakeKernelScan], server: Any
    ) -> None:
        server.request_headers.clear()
        out = _read(ENGINE.scan(renamed, columns=["label"], predicate="id >= 2"))
        assert out.to_pylist() == [{"label": "b"}, {"label": "c"}]
        # The query advertised delta responses with the reader features...
        capabilities = [h.get("delta-sharing-capabilities", "") for h in server.request_headers]
        assert any("responseformat=delta" in c and "columnmapping" in c for c in capabilities)
        # ...and the log written for the kernel holds the delta actions verbatim.
        protocol, metadata, add = fake_kernel.logs[-1]
        assert protocol["protocol"]["readerFeatures"] == ["columnMapping"]
        assert metadata["metaData"]["configuration"]["delta.columnMapping.mode"] == "name"
        assert add["add"]["path"].endswith("/files/mapped-1")

    def test_parquet_request_would_have_been_refused(self, renamed: ResolvedTable) -> None:
        # Guard for the fake itself: without the delta header the server refuses.
        forced = SharingEngine(num_retries=0)
        forced._needs_delta_format = staticmethod(lambda table: False)  # type: ignore[method-assign]
        with pytest.raises(UnreachableTableError, match="requires delta format"):
            forced.scan(renamed)

    def test_detail_at_version_parses_delta_lines(self, renamed: ResolvedTable) -> None:
        d = ENGINE.detail(renamed, version=4)
        assert d["version"] == 4
        assert d["reader_features"] == ["columnMapping"]
        assert d["min_reader_version"] == 3

    def test_missing_kernel_is_refused_up_front(
        self, renamed: ResolvedTable, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(SharingEngine, "_kernel_available", staticmethod(lambda: False))
        cap = ENGINE.supports(Operation.SCAN, renamed)
        assert not cap.ok and "columnMapping" in cap.reason
        assert "delta-kernel-rust-sharing-wrapper" in cap.remedy
