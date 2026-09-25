"""Regression tests for Delta Sharing defects (profile handling, requests, reads).

A small protocol server below lets each test switch on one misbehaviour
(repeated page tokens, missing headers, expiring URLs, odd response lines).
"""

from __future__ import annotations

import io
import json
import os
import threading
from datetime import UTC, date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("delta_sharing")

from deltaswamp.catalog.base import ResolvedTable  # noqa: E402
from deltaswamp.catalog.sharing import SharingCatalog, load_profile  # noqa: E402
from deltaswamp.engine import sharing as engine_module  # noqa: E402
from deltaswamp.engine.sharing import (  # noqa: E402
    SharingEngine,
    json_predicate_hints,
    sharing_timestamp,
)
from deltaswamp.errors import InvalidReferenceError, UnreachableTableError  # noqa: E402
from deltaswamp.identity import RefKind, TableRef, parse_ref  # noqa: E402

from .test_sharing import SCHEMA_STRING as RICH_SCHEMA  # noqa: E402
from .test_sharing import (  # noqa: E402
    FakeSharedTable,
    FakeSharingServer,
    FakeVersion,
    _ms,
    _rows,
)

TOKEN = "tok-123"
ENGINE = SharingEngine(num_retries=0)

SCHEMA = json.dumps(
    {
        "type": "struct",
        "fields": [
            {"name": "Id", "type": "long", "nullable": True, "metadata": {}},
            {"name": "s", "type": "string", "nullable": True, "metadata": {}},
            {"name": "p", "type": "integer", "nullable": True, "metadata": {}},
        ],
    }
)


def _parquet(table: Any) -> bytes:
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


class MiniServer:
    """A few endpoints of the protocol, each with a switchable misbehaviour."""

    def __init__(self) -> None:
        self.tables: dict[tuple[str, str, str], list[tuple[str, bytes, dict[str, Any]]]] = {}
        self.paths: list[str] = []
        self.queries: list[dict[str, Any]] = []
        self.query_count = 0
        self.expire_before = 0  # file URLs signed by an earlier query answer 403
        self.url_template: str | None = None
        self.repeat_page_token = False
        self.pages: list[list[str]] | None = None
        self.drop_version_header = False
        self.min_reader_version = 1
        self.status: int | None = None
        self._server: ThreadingHTTPServer | None = None

    @property
    def endpoint(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}/ds"

    def profile(self) -> dict[str, Any]:
        return {"shareCredentialsVersion": 1, "endpoint": self.endpoint, "bearerToken": TOKEN}

    def start(self) -> MiniServer:
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, code: int, lines: list[Any], headers: dict[str, str]) -> None:
                body = "".join(
                    (line if isinstance(line, str) else json.dumps(line)) + "\n" for line in lines
                ).encode()
                self.send_response(code)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _version(self) -> dict[str, str]:
                return {} if srv.drop_version_header else {"Delta-Table-Version": "3"}

            def _meta(self) -> list[Any]:
                return [
                    {"protocol": {"minReaderVersion": srv.min_reader_version}},
                    {
                        "metaData": {
                            "id": "abc",
                            "format": {"provider": "parquet"},
                            "schemaString": SCHEMA,
                            "partitionColumns": ["p"],
                        }
                    },
                ]

            def do_GET(self) -> None:
                srv.paths.append(self.path)
                parsed = urlparse(self.path)
                parts = [unquote(p) for p in parsed.path.split("/") if p][1:]
                if parts[:1] == ["files"]:
                    sig = int(parse_qs(parsed.query).get("sig", ["0"])[0])
                    if sig < srv.expire_before:
                        return self._send(403, [{"error": "expired"}], {})
                    for files in srv.tables.values():
                        for fid, data, _ in files:
                            if fid == parts[1]:
                                self.send_response(200)
                                self.send_header("Content-Length", str(len(data)))
                                self.end_headers()
                                self.wfile.write(data)
                                return None
                    return self._send(404, [], {})
                if srv.status is not None:
                    return self._send(srv.status, [{"errorCode": "X", "message": "no"}], {})
                if parts == ["shares"]:
                    if srv.repeat_page_token:
                        return self._send(
                            200, [{"items": [{"name": "a"}], "nextPageToken": "same"}], {}
                        )
                    token = parse_qs(parsed.query).get("pageToken", ["0"])[0]
                    pages = srv.pages or [["a"]]
                    index = int(token)
                    line: dict[str, Any] = {"items": [{"name": n} for n in pages[index]]}
                    if index + 1 < len(pages):
                        line["nextPageToken"] = str(index + 1)
                    return self._send(200, [line], {})
                if len(parts) == 3 and parts[2] == "schemas":
                    return self._send(200, [{"items": [{"name": "x y", "share": parts[1]}]}], {})
                if len(parts) == 5 and parts[4] == "tables":
                    items = [
                        {"name": n, "share": s, "schema": sc}
                        for (s, sc, n) in srv.tables
                        if (s, sc) == (parts[1], parts[3])
                    ]
                    return self._send(200, [{"items": items}], {})
                if len(parts) == 7 and parts[6] == "metadata":
                    if (parts[1], parts[3], parts[5]) not in srv.tables:
                        return self._send(404, [{"errorCode": "NF", "message": "nf"}], {})
                    return self._send(200, self._meta(), self._version())
                return self._send(404, [], {})

            def do_POST(self) -> None:
                srv.paths.append(self.path)
                parts = [unquote(p) for p in urlparse(self.path).path.split("/") if p][1:]
                length = int(self.headers.get("Content-Length") or 0)
                srv.queries.append(json.loads(self.rfile.read(length) or b"{}"))
                key = (parts[1], parts[3], parts[5])
                if key not in srv.tables:
                    return self._send(404, [{"errorCode": "NF", "message": "nf"}], {})
                srv.query_count += 1
                lines = self._meta()
                for fid, data, pv in srv.tables[key]:
                    url = (srv.url_template or "{endpoint}/files/{id}?sig={sig}").format(
                        endpoint=srv.endpoint, id=fid, sig=srv.query_count
                    )
                    lines.append(
                        {"file": {"url": url, "id": fid, "partitionValues": pv, "size": len(data)}}
                    )
                return self._send(200, lines, self._version())

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def _file(ids: list[int | None]) -> bytes:
    return _parquet(
        pa.table({"Id": pa.array(ids, pa.int64()), "s": pa.array([str(i) for i in ids])})
    )


@pytest.fixture
def mini() -> Any:
    srv = MiniServer().start()
    srv.tables[("my share", "sch/ema", "t#1?")] = [("f1", _file([1, 2]), {"p": "7"})]
    srv.tables[("a", "b", "c")] = [
        ("g1", _file([1, 2]), {"p": "1"}),
        ("g2", _file([3]), {"p": ""}),
    ]
    srv.tables[("a", "b", "dotted.name")] = [("h1", _file([5]), {"p": "1"})]
    yield srv
    srv.stop()


def _catalog(srv: MiniServer) -> SharingCatalog:
    return SharingCatalog(srv.profile())


def _ref(share: str, schema: str, table: str) -> TableRef:
    return TableRef(
        kind=RefKind.CATALOG, catalog=share, schema=schema, table=table, raw="x", scheme="sharing"
    )


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    rest = pytest.importorskip("delta_sharing.rest_client")
    monkeypatch.setattr(rest.time, "sleep", lambda _s: None)


# ---------------------------------------------------------------------------
# Catalog: names, pagination, errors
# ---------------------------------------------------------------------------


def test_resolve_percent_encodes_names_in_the_path(mini: MiniServer) -> None:
    resolved = _catalog(mini).resolve(_ref("my share", "sch/ema", "t#1?"))
    assert resolved.table_id == "abc"
    assert "/shares/my%20share/schemas/sch%2Fema/tables/t%231%3F/metadata" in mini.paths[-1]


def test_scan_percent_encodes_names_in_the_path(mini: MiniServer) -> None:
    resolved = _catalog(mini).resolve(_ref("my share", "sch/ema", "t#1?"))
    table = pa.table(ENGINE.scan(resolved))
    assert table.column("Id").to_pylist() == [1, 2]


def test_list_schemas_percent_encodes_the_share(mini: MiniServer) -> None:
    assert _catalog(mini).list_schemas("my share") == ["x y"]
    assert "/shares/my%20share/schemas" in mini.paths[-1]


def test_listing_follows_page_tokens(mini: MiniServer) -> None:
    mini.pages = [["a", "b"], ["c"], ["d"]]
    assert _catalog(mini).list_catalogs() == ["a", "b", "c", "d"]


def test_repeated_page_token_does_not_loop_forever(mini: MiniServer) -> None:
    mini.repeat_page_token = True
    with pytest.raises(UnreachableTableError, match="repeated the page token"):
        _catalog(mini).list_catalogs()


def test_listing_closes_its_http_session(mini: MiniServer, monkeypatch: pytest.MonkeyPatch) -> None:
    rest = pytest.importorskip("delta_sharing.rest_client")
    closed: list[int] = []
    original = rest.DataSharingRestClient.close

    def close(self: Any) -> Any:
        closed.append(1)
        return original(self)

    monkeypatch.setattr(rest.DataSharingRestClient, "close", close)
    _catalog(mini).list_catalogs()
    assert closed


def test_listing_http_error_is_wrapped(mini: MiniServer) -> None:
    mini.status = 403
    with pytest.raises(UnreachableTableError, match="HTTP 403"):
        _catalog(mini).list_catalogs()


def test_list_tables_quotes_dotted_names(mini: MiniServer) -> None:
    tables = _catalog(mini).list_tables("a", "b")
    by_name = {t.ref.table: t for t in tables}
    dotted = by_name["dotted.name"].ref
    assert parse_ref(dotted.raw).table == "dotted.name"


def test_resolve_missing_version_header_is_unreachable(mini: MiniServer) -> None:
    mini.drop_version_header = True
    with pytest.raises(UnreachableTableError, match="delta-table-version"):
        _catalog(mini).resolve(_ref("a", "b", "c"))


def test_resolve_reader_version_too_new_is_unreachable(mini: MiniServer) -> None:
    mini.min_reader_version = 9
    with pytest.raises(UnreachableTableError, match="newer version"):
        _catalog(mini).resolve(_ref("a", "b", "c"))


def test_resolve_unreachable_server_is_unreachable(no_sleep: None) -> None:
    catalog = SharingCatalog(
        {"shareCredentialsVersion": 1, "endpoint": "http://127.0.0.1:9/ds", "bearerToken": "t"}
    )
    with pytest.raises(UnreachableTableError, match="did not answer"):
        catalog.resolve(_ref("a", "b", "c"))


def test_list_unreachable_server_is_unreachable(no_sleep: None) -> None:
    catalog = SharingCatalog(
        {"shareCredentialsVersion": 1, "endpoint": "http://127.0.0.1:9/ds", "bearerToken": "t"}
    )
    with pytest.raises(UnreachableTableError, match="did not answer"):
        catalog.list_catalogs()


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("uri", ["sharing+https://", "sharing+https:///api/2.0/delta-sharing"])
def test_endpoint_uri_without_host_is_refused(uri: str) -> None:
    with pytest.raises(InvalidReferenceError, match="names no Delta Sharing host"):
        SharingCatalog.from_uri(uri, token="t")


def test_expired_profile_is_refused_at_connect() -> None:
    past = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    doc = {
        "shareCredentialsVersion": 1,
        "endpoint": "https://h/ds",
        "bearerToken": "secret-token-value",
        "expirationTime": past,
    }
    with pytest.raises(InvalidReferenceError, match="expired") as info:
        SharingCatalog(doc)
    assert "secret-token-value" not in str(info.value)


def test_unexpired_profile_is_accepted() -> None:
    future = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = {
        "shareCredentialsVersion": 1,
        "endpoint": "https://h/ds",
        "bearerToken": "t",
        "expirationTime": future,
    }
    assert SharingCatalog(doc).endpoint == "https://h/ds"


def test_401_names_the_token_expiry(mini: MiniServer) -> None:
    mini.status = 401
    future = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    catalog = SharingCatalog({**mini.profile(), "expirationTime": future})
    with pytest.raises(UnreachableTableError, match="bearer token expires at"):
        catalog.resolve(_ref("a", "b", "c"))


def test_home_directory_profile_path_is_expanded(
    mini: MiniServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "c.share").write_text(json.dumps(mini.profile()))
    assert SharingCatalog.from_uri("sharing://~/c.share").endpoint == mini.endpoint


def test_directory_as_profile_is_invalid_reference(tmp_path: Path) -> None:
    with pytest.raises(InvalidReferenceError, match="cannot be read"):
        load_profile(str(tmp_path))


def test_non_string_profile_is_invalid_reference() -> None:
    with pytest.raises(InvalidReferenceError, match="not int"):
        SharingCatalog(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# jsonPredicateHints
# ---------------------------------------------------------------------------

HINT_SCHEMA = {
    "fields": [
        {"name": "Id", "type": "long"},
        {"name": "n", "type": "integer"},
        {"name": "d", "type": "date"},
        {"name": "s", "type": "string"},
    ]
}


@pytest.mark.parametrize(
    "predicate",
    [
        "Id = 1 OR s = 'x' AND n = 2",
        "Id = 1 AND n = 2 OR s = 'x'",
        "n = 2 AND Id = 1 or s = 'y'",
    ],
)
def test_top_level_or_gives_no_hint(predicate: str) -> None:
    # A hint from one side of an OR would let the server skip matching files.
    assert json_predicate_hints(predicate, HINT_SCHEMA) is None


def test_or_inside_a_string_or_parens_still_hints() -> None:
    hint = json_predicate_hints("s = 'a or b' AND (n = 1 OR n = 2)", HINT_SCHEMA)
    assert hint is not None and json.loads(hint)["children"][1]["value"] == "a or b"


def test_hint_uses_the_schema_spelling_of_the_column() -> None:
    tree = json.loads(json_predicate_hints("id = 5", HINT_SCHEMA) or "")
    assert tree["children"][0]["name"] == "Id"


def test_out_of_range_int_literal_gives_no_hint() -> None:
    assert json_predicate_hints("n = 3000000000", HINT_SCHEMA) is None
    assert json_predicate_hints("Id = 3000000000", HINT_SCHEMA) is not None


def test_invalid_date_literal_gives_no_hint() -> None:
    assert json_predicate_hints("d = 'not-a-date'", HINT_SCHEMA) is None
    assert json_predicate_hints("d = '2024-01-02'", HINT_SCHEMA) is not None


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(2024, 2, 1, tzinfo=UTC), "2024-02-01T00:00:00Z"),
        (datetime(2024, 2, 1, 1, 0), "2024-02-01T01:00:00Z"),
        ("2024-02-01 00:00:00", "2024-02-01T00:00:00Z"),
        ("2024-02-01T02:00:00+02:00", "2024-02-01T00:00:00Z"),
        (date(2024, 2, 1), "2024-02-01T00:00:00Z"),
        (_ms("2024-02-01T00:00:00Z"), "2024-02-01T00:00:00Z"),
        ("2024-02-01T00:00:00.5Z", "2024-02-01T00:00:00.5Z"),
    ],
)
def test_sharing_timestamp(value: Any, expected: str) -> None:
    assert sharing_timestamp(value, what="t") == expected


def test_bad_timestamp_is_explained() -> None:
    with pytest.raises(UnreachableTableError, match="not an ISO-8601"):
        sharing_timestamp("yesterday", what="t")


@pytest.fixture(scope="module")
def rich() -> Any:
    srv = FakeSharingServer().start()
    f1, f2 = _rows([1, 2, 3], "2024-01-01"), _rows([4, 5], "2024-01-02")
    srv.add(
        FakeSharedTable(
            share="retail",
            schema="sales",
            name="orders",
            schema_string=RICH_SCHEMA,
            partition_columns=["day"],
            versions=[
                FakeVersion(0, _ms("2024-01-01T00:00:00Z"), [f1]),
                FakeVersion(1, _ms("2024-02-01T00:00:00Z"), [f1, f2]),
            ],
        )
    )
    yield srv
    srv.stop()


@pytest.fixture
def orders(rich: FakeSharingServer) -> ResolvedTable:
    catalog = SharingCatalog(
        {
            "shareCredentialsVersion": 1,
            "endpoint": rich.endpoint,
            "bearerToken": "s3cr3t-recipient-token",
        }
    )
    return catalog.resolve(parse_ref("retail.sales.orders"))


def test_time_travel_with_a_datetime(orders: ResolvedTable, rich: FakeSharingServer) -> None:
    table = pa.table(ENGINE.scan(orders, timestamp=datetime(2024, 1, 15, tzinfo=UTC)))
    assert sorted(table.column("id").to_pylist()) == [1, 2, 3]
    assert rich.queries[-1]["timestamp"] == "2024-01-15T00:00:00Z"


def test_time_travel_with_a_zoneless_string(orders: ResolvedTable, rich: FakeSharingServer) -> None:
    ENGINE.scan(orders, timestamp="2024-02-01 00:00:00")
    assert rich.queries[-1]["timestamp"] == "2024-02-01T00:00:00Z"


def test_cdf_accepts_datetime_bounds(orders: ResolvedTable) -> None:
    # Used to reach urllib.parse.quote(datetime) inside the client: TypeError.
    reader = ENGINE.cdf(orders, starting_timestamp=datetime(2024, 1, 1, tzinfo=UTC))
    assert "_change_type" in pa.table(reader).column_names


# ---------------------------------------------------------------------------
# Limits, projections, schema
# ---------------------------------------------------------------------------


def test_limit_is_sent_as_limit_hint_and_applied(
    orders: ResolvedTable, rich: FakeSharingServer
) -> None:
    table = pa.table(ENGINE.scan(orders, limit=2))
    assert table.num_rows == 2
    assert rich.queries[-1].get("limitHint") == 2


def test_limit_hint_is_not_sent_with_a_predicate(
    orders: ResolvedTable, rich: FakeSharingServer
) -> None:
    table = pa.table(ENGINE.scan(orders, predicate="id >= 4", limit=1))
    assert table.column("id").to_pylist() == [4]
    assert "limitHint" not in rich.queries[-1]


def test_empty_projection_keeps_the_row_count(orders: ResolvedTable) -> None:
    assert pa.table(ENGINE.scan(orders, columns=[])).num_rows == 5


def test_duplicate_projection_is_deduplicated(orders: ResolvedTable) -> None:
    assert pa.table(ENGINE.scan(orders, columns=["id", "ID"])).column_names == ["id"]


def test_schema_comes_from_metadata_without_reading_data(
    orders: ResolvedTable, rich: FakeSharingServer
) -> None:
    before = len(rich.queries)
    schema = ENGINE.snapshot(orders).schema()
    assert schema.field("price").type == pa.decimal128(10, 2)
    assert len(rich.queries) == before  # no /query, so no file listing or download


# ---------------------------------------------------------------------------
# Presigned files
# ---------------------------------------------------------------------------


def test_expired_url_is_reissued_mid_stream(mini: MiniServer) -> None:
    resolved = _catalog(mini).resolve(_ref("a", "b", "c"))
    reader = ENGINE.scan(resolved)
    first = reader.read_next_batch()
    mini.expire_before = mini.query_count + 1  # every URL issued so far expires
    rest = pa.Table.from_batches(list(reader))
    assert first.num_rows + rest.num_rows == 3
    assert mini.query_count == 2  # one refresh query


def test_file_scheme_url_is_refused(mini: MiniServer, tmp_path: Path) -> None:
    secret = tmp_path / "f1"
    secret.write_bytes(_file([1]))
    mini.url_template = f"file://{tmp_path}/{{id}}"
    resolved = _catalog(mini).resolve(_ref("a", "b", "c"))
    with pytest.raises(UnreachableTableError, match="not an HTTP"):
        pa.table(ENGINE.scan(resolved))


def test_unreachable_file_host_is_unreachable(mini: MiniServer) -> None:
    mini.url_template = "http://127.0.0.1:9/{id}"
    resolved = _catalog(mini).resolve(_ref("a", "b", "c"))
    with pytest.raises(UnreachableTableError, match="object store failed"):
        pa.table(SharingEngine(request_timeout=5).scan(resolved))


def test_empty_partition_value_is_null_for_non_strings(mini: MiniServer) -> None:
    resolved = _catalog(mini).resolve(_ref("a", "b", "c"))
    table = pa.table(ENGINE.scan(resolved)).sort_by("Id")
    assert table.column("p").to_pylist() == [1, 1, None]


def test_bad_partition_value_is_explained() -> None:
    with pytest.raises(UnreachableTableError, match="partition value 'x'"):
        engine_module._partition_column("x", pa.int32(), 1)


def test_lossy_cast_is_refused_not_wrapped() -> None:
    data = pa.table({"n": pa.array([2**40], pa.int64())})
    schema = pa.schema([pa.field("n", pa.int32())])
    with pytest.raises(UnreachableTableError, match="without loss"):
        engine_module._conform(data, schema, {})


def test_timestamp_narrowing_still_allowed() -> None:
    data = pa.table({"t": pa.array([1_000_000_001], pa.timestamp("ns"))})
    schema = pa.schema([pa.field("t", pa.timestamp("us", tz="UTC"))])
    assert engine_module._conform(data, schema, {}).num_rows == 1


def test_connection_failure_does_not_suggest_history(no_sleep: None) -> None:
    resolved = ResolvedTable(
        ref=_ref("a", "b", "c"),
        location=None,
        sharing_profile=json.dumps(
            {"shareCredentialsVersion": 1, "endpoint": "http://127.0.0.1:9/ds", "bearerToken": "t"}
        ),
    )
    with pytest.raises(UnreachableTableError) as info:
        ENGINE.scan(resolved, version=1)
    assert "did not answer" in str(info.value) and info.value.remedy is None


# ---------------------------------------------------------------------------
# Delta-format responses, replayed by the kernel
# ---------------------------------------------------------------------------

pytest.importorskip("delta_kernel_rust_sharing_wrapper")

PLAIN_SCHEMA = json.dumps(
    {
        "type": "struct",
        "fields": [
            {"name": "id", "type": "long", "nullable": True, "metadata": {}},
            {"name": "s", "type": "string", "nullable": True, "metadata": {}},
        ],
    }
)


def _delta_add(path: Path) -> dict[str, Any]:
    return {
        "add": {
            "path": path.as_uri(),
            "partitionValues": {},
            "size": os.path.getsize(path),
            "modificationTime": 0,
            "dataChange": True,
        }
    }


def _protocol_line() -> str:
    return json.dumps(
        {"protocol": {"deltaProtocol": {"minReaderVersion": 1, "minWriterVersion": 4}}}
    )


def _metadata_line(version: int | None = None) -> str:
    body: dict[str, Any] = {
        "deltaMetadata": {
            "id": "m",
            "format": {"provider": "parquet", "options": {}},
            "schemaString": PLAIN_SCHEMA,
            "partitionColumns": [],
            "configuration": {"delta.enableChangeDataFeed": "true"},
        }
    }
    if version is not None:
        body["version"] = version
    return json.dumps({"metaData": body})


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    path = tmp_path / "part-0.parquet"
    pq.write_table(pa.table({"id": pa.array([1, None], pa.int64()), "s": ["a", "b"]}), path)
    return path


def test_snapshot_replay_skips_blank_and_unknown_lines(data_file: Path) -> None:
    lines = [
        "",
        _metadata_line(),  # out of the usual order
        _protocol_line(),
        "   ",
        json.dumps({"endStreamAction": {"refreshToken": "x"}}),
        json.dumps({"file": {"id": "1", "deltaSingleAction": _delta_add(data_file)}}),
    ]
    table = ENGINE._replay_delta_log(lines, what="read", check_urls=False)
    assert table.column("id").to_pylist() == [1, None]


def test_snapshot_replay_without_protocol_is_explained() -> None:
    with pytest.raises(UnreachableTableError, match="no protocol or metaData"):
        ENGINE._replay_delta_log([], what="read")


def _cdf_file(version: int, ts: int, path: Path) -> str:
    return json.dumps(
        {
            "file": {
                "id": path.name,
                "version": version,
                "timestamp": ts,
                "deltaSingleAction": _delta_add(path),
            }
        }
    )


def test_cdf_replay_keeps_arrow_types(data_file: Path) -> None:
    ts = _ms("2024-03-01T12:00:00.250Z")
    lines = [_protocol_line(), _metadata_line(0), _cdf_file(0, ts, data_file)]
    table = ENGINE._replay_cdf_lines(lines, starting_version=0, what="cdf", check_urls=False)
    # Through pandas the nullable long came back as float64 [1.0, nan].
    assert table.schema.field("id").type == pa.int64()
    assert table.column("id").to_pylist() == [1, None]
    assert table.column("_change_type").to_pylist() == ["insert", "insert"]
    stamp = table.column("_commit_timestamp")[0].as_py()
    assert stamp == datetime(2024, 3, 1, 12, 0, 0, 250000, tzinfo=UTC)  # to the millisecond


def test_cdf_replay_from_a_later_version(data_file: Path) -> None:
    lines = [
        _protocol_line(),
        _metadata_line(5),
        _cdf_file(5, _ms("2024-03-01T00:00:00Z"), data_file),
    ]
    table = ENGINE._replay_cdf_lines(lines, starting_version=5, what="cdf", check_urls=False)
    assert table.column("_commit_version").to_pylist() == [5, 5]


def test_cdf_replay_with_no_changes_is_empty_with_schema() -> None:
    table = ENGINE._replay_cdf_lines(
        [_protocol_line(), _metadata_line(2)], starting_version=2, what="cdf", check_urls=False
    )
    assert table.num_rows == 0
    assert table.column_names == ["id", "s", "_change_type", "_commit_version", "_commit_timestamp"]


def test_nested_timestamp_narrowing_still_allowed() -> None:
    struct_ns = pa.struct([("t", pa.timestamp("ns"))])
    data = pa.table({"s": pa.array([{"t": 1_000_000_001}], struct_ns)})
    schema = pa.schema([pa.field("s", pa.struct([("t", pa.timestamp("us", tz="UTC"))]))])
    assert engine_module._conform(data, schema, {}).num_rows == 1


def test_version_and_timestamp_together_are_refused(orders: ResolvedTable) -> None:
    with pytest.raises(UnreachableTableError, match="one or the other"):
        ENGINE.scan(orders, version=0, timestamp="2024-01-01T00:00:00Z")


def test_cdf_version_and_timestamp_bounds_together_are_refused(orders: ResolvedTable) -> None:
    with pytest.raises(UnreachableTableError, match="starting_version and starting_timestamp"):
        ENGINE.cdf(orders, starting_version=0, starting_timestamp="2024-01-01T00:00:00Z")


def test_table_schema_and_count_do_not_read_everything_twice(
    rich: FakeSharingServer,
) -> None:
    from deltaswamp.capability import Engine
    from deltaswamp.connection import Connection
    from deltaswamp.router import Router

    catalog = SharingCatalog(
        {
            "shareCredentialsVersion": 1,
            "endpoint": rich.endpoint,
            "bearerToken": "s3cr3t-recipient-token",
        }
    )
    conn = Connection(catalog=catalog, router=Router(engines={Engine.SHARING: ENGINE}))
    t = conn.table("retail.sales.orders")
    before = len(rich.queries)
    assert t.schema().field("id").type == pa.int64()
    assert len(rich.queries) == before  # Table.schema() used to scan the whole table
    assert t.count() == 5
    assert len(rich.queries) == before + 1


def test_delta_response_local_paths_are_refused(data_file: Path) -> None:
    lines = [
        _protocol_line(),
        _metadata_line(),
        json.dumps({"file": {"id": "1", "deltaSingleAction": _delta_add(data_file)}}),
    ]
    with pytest.raises(UnreachableTableError, match="not an HTTP"):
        ENGINE._replay_delta_log(lines, what="read")
    cdf = [_protocol_line(), _metadata_line(0), _cdf_file(0, 0, data_file)]
    with pytest.raises(UnreachableTableError, match="not an HTTP"):
        ENGINE._replay_cdf_lines(cdf, starting_version=0, what="cdf")


def test_local_deletion_vector_path_is_refused() -> None:
    action = {
        "add": {
            "path": "https://h/f.parquet",
            "deletionVector": {"storageType": "p", "pathOrInlineDv": "file:///etc/passwd"},
        }
    }
    with pytest.raises(UnreachableTableError, match="not an HTTP"):
        SharingEngine._check_action_urls(action)


def test_unrequested_delta_answer_is_replayed(
    orders: ResolvedTable, monkeypatch: pytest.MonkeyPatch
) -> None:
    rest = pytest.importorskip("delta_sharing.rest_client")

    def delta_files(self: Any, table: Any, **kwargs: Any) -> Any:
        return rest.ListFilesInTableResponse(
            delta_table_version=3,
            protocol=None,
            metadata=None,
            add_files=[],
            lines=[_protocol_line(), _metadata_line()],
        )

    def delta_changes(self: Any, table: Any, options: Any) -> Any:
        return rest.ListTableChangesResponse(
            protocol=None, metadata=None, actions=None, lines=[_protocol_line(), _metadata_line(0)]
        )

    monkeypatch.setattr(rest.DataSharingRestClient, "list_files_in_table", delta_files)
    monkeypatch.setattr(rest.DataSharingRestClient, "list_table_changes", delta_changes)
    # Both used to fail with AttributeError on the missing parquet metadata.
    assert pa.table(ENGINE.scan(orders)).column_names == ["id", "s"]
    assert pa.table(ENGINE.cdf(orders)).num_rows == 0


# ---------------------------------------------------------------------------
# Predicates are never pasted into SQL
# ---------------------------------------------------------------------------

INJECTIONS = [
    "id = 1; DROP TABLE t",
    "id = 1 -- and the rest is a comment",
    "id = 1 /* comment */",
    "true) AS keep FROM __deltaswamp_rows UNION ALL SELECT (true",
    "s = 'x') AS keep FROM __deltaswamp_rows; SELECT ('",
    "(SELECT count(*) FROM read_text('/etc/passwd')) > 0",
    "id IN (SELECT 1)",
]


@pytest.mark.parametrize("predicate", INJECTIONS)
def test_filter_never_reaches_another_sql_engine(
    predicate: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deltaswamp.predicate import PredicateError

    duckdb = pytest.importorskip("duckdb")
    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: pytest.fail("reached DuckDB"))
    table = pa.table({"id": [1, 2], "s": ["x", "y"]})
    with pytest.raises(PredicateError):
        engine_module.filter_arrow_exact(table, predicate)


@pytest.mark.parametrize(
    "predicate",
    ["getenv('HOME') IS NOT NULL", "length(read_blob('/etc/passwd')::VARCHAR) > 0"],
)
def test_sandboxed_evaluator_has_no_environment_or_file_access(predicate: str) -> None:
    # Reaches DuckDB (it passes the screen) but the sandbox refuses it.
    from deltaswamp.predicate import PredicateError

    pytest.importorskip("duckdb")
    table = pa.table({"id": [1, 2], "s": ["x", "y"]})
    with pytest.raises(PredicateError):
        engine_module.filter_arrow_exact(table, predicate)


def test_function_predicates_still_filter_through_the_sandbox() -> None:
    # [regression] removing the DuckDB fallback outright refused every
    # predicate using a function or arithmetic, which used to work.
    pytest.importorskip("duckdb")
    table = pa.table({"id": [1, 2, 3], "s": ["Xa", "y", None]})
    out = engine_module.filter_arrow_exact(table, "lower(s) = 'xa' OR id + 1 = 4")
    assert out.column("id").to_pylist() == [1, 3]
    assert out.schema == table.schema


def test_function_predicate_scan_through_the_engine(orders: ResolvedTable) -> None:
    pytest.importorskip("duckdb")
    everything = pa.table(ENGINE.scan(orders))
    col = everything.column_names[0]
    first = everything.column(col)[0].as_py()
    expected = sum(1 for v in everything.column(col).to_pylist() if v == first)
    predicate = f"coalesce({col}, {col}) = {first!r}".replace('"', "'")
    assert pa.table(ENGINE.scan(orders, predicate=predicate)).num_rows == expected


@pytest.mark.parametrize("predicate", INJECTIONS[:4])
def test_injected_predicate_fails_before_any_request(
    predicate: str, orders: ResolvedTable, rich: FakeSharingServer
) -> None:
    from deltaswamp.predicate import PredicateError

    before = len(rich.queries)
    with pytest.raises(PredicateError):
        ENGINE.scan(orders, predicate=predicate)
    with pytest.raises(PredicateError):
        ENGINE.cdf(orders, predicate=predicate)
    assert len(rich.queries) == before


def test_quote_breakout_stays_a_literal(orders: ResolvedTable) -> None:
    # The doubled quote is part of the literal, not the end of it.
    table = pa.table(ENGINE.scan(orders, predicate="name = 'n1'' OR ''1''=''1'"))
    assert table.num_rows == 0
    assert pa.table(ENGINE.scan(orders, predicate="name = 'n1'")).num_rows == 1


# ---------------------------------------------------------------------------
# CDF ranges and timestamp formatting
# ---------------------------------------------------------------------------


def _record_cdf_options(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    rest = pytest.importorskip("delta_sharing.rest_client")
    seen: list[Any] = []
    original = rest.DataSharingRestClient.list_table_changes

    def recording(self: Any, table: Any, options: Any) -> Any:
        seen.append(options)
        return original(self, table, options)

    monkeypatch.setattr(rest.DataSharingRestClient, "list_table_changes", recording)
    return seen


def test_cdf_inverted_version_range_is_refused_before_the_request(
    orders: ResolvedTable, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _record_cdf_options(monkeypatch)
    with pytest.raises(UnreachableTableError, match="after ending_version"):
        ENGINE.cdf(orders, starting_version=3, ending_version=1)
    assert not seen


def test_cdf_inverted_timestamp_range_is_refused(orders: ResolvedTable) -> None:
    # Mixed forms: compared as instants, not as strings.
    with pytest.raises(UnreachableTableError, match="after ending_timestamp"):
        ENGINE.cdf(
            orders,
            starting_timestamp="2024-02-01T01:00:00+02:00",
            ending_timestamp=datetime(2024, 1, 31, 22, 0, tzinfo=UTC),
        )


def test_cdf_equal_instants_in_different_forms_are_accepted(orders: ResolvedTable) -> None:
    ENGINE.cdf(
        orders,
        starting_timestamp="2024-02-01T02:00:00+02:00",
        ending_timestamp=datetime(2024, 2, 1, tzinfo=UTC),
    )


def test_cdf_timestamps_reach_the_client_normalised(
    orders: ResolvedTable, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _record_cdf_options(monkeypatch)
    ENGINE.cdf(
        orders,
        starting_timestamp=datetime(2024, 1, 1, 5, 0),
        ending_timestamp="2024-03-01 00:00:00.250",
    )
    assert seen[-1].starting_timestamp == "2024-01-01T05:00:00Z"
    assert seen[-1].ending_timestamp == "2024-03-01T00:00:00.25Z"
    assert seen[-1].starting_version is None  # no default 0 next to a timestamp


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"starting_version": True}, "version number"),
        ({"starting_version": -1}, "non-negative"),
        ({"ending_version": 1.5}, "integer version"),
    ],
)
def test_cdf_bad_versions_are_refused(
    orders: ResolvedTable, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(UnreachableTableError, match=match):
        ENGINE.cdf(orders, **kwargs)


def test_numpy_versions_are_accepted(orders: ResolvedTable) -> None:
    np = pytest.importorskip("numpy")
    assert pa.table(ENGINE.scan(orders, version=np.int64(0))).num_rows == 3
    ENGINE.cdf(orders, starting_version=np.int64(0), ending_version=np.int64(1))


def test_scan_negative_version_is_refused(orders: ResolvedTable) -> None:
    with pytest.raises(UnreachableTableError, match="non-negative"):
        ENGINE.scan(orders, version=-1)


def test_out_of_range_epoch_is_explained() -> None:
    with pytest.raises(UnreachableTableError, match="in range"):
        sharing_timestamp(10**20, what="t")


# ---------------------------------------------------------------------------
# Profiles given as paths and URLs
# ---------------------------------------------------------------------------


def test_relative_profile_path_survives_a_chdir(
    mini: MiniServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "c.share").write_text(json.dumps(mini.profile()))
    monkeypatch.chdir(tmp_path)
    catalog = SharingCatalog.from_uri("sharing://c.share")
    assert catalog.profile == str(tmp_path / "c.share")
    monkeypatch.chdir("/")
    resolved = catalog.resolve(_ref("a", "b", "c"))
    assert pa.table(ENGINE.scan(resolved)).num_rows == 3  # re-reads the profile per request


def test_fsspec_url_profile(mini: MiniServer) -> None:
    fsspec = pytest.importorskip("fsspec")
    with fsspec.open("memory://profiles/c.share", "w") as handle:
        handle.write(json.dumps(mini.profile()))
    catalog = SharingCatalog.from_uri("sharing://memory://profiles/c.share")
    assert catalog.endpoint == mini.endpoint
    assert catalog.profile == "memory://profiles/c.share"


def test_unknown_fsspec_protocol_is_explained() -> None:
    with pytest.raises(InvalidReferenceError, match="no usable fsspec filesystem"):
        load_profile("nosuchproto://bucket/c.share")


def test_presigned_profile_url_is_not_echoed() -> None:
    url = "memory://nowhere/c.share?X-Amz-Signature=deadbeefsecret"
    with pytest.raises(InvalidReferenceError) as info:
        load_profile(url)
    assert "deadbeefsecret" not in str(info.value)
    assert "credentials hidden" in str(info.value)


def test_profile_error_does_not_echo_token() -> None:
    doc = {"shareCredentialsVersion": 2, "type": "bearer_token", "endpoint": "https://h"}
    with pytest.raises(InvalidReferenceError, match="bearerToken"):
        SharingCatalog(doc)


def test_json_array_profile_is_refused() -> None:
    with pytest.raises(InvalidReferenceError, match="must be an object"):
        SharingCatalog("[1, 2]")


def test_unserialisable_dict_profile_is_refused() -> None:
    with pytest.raises(InvalidReferenceError, match="not JSON-serialisable"):
        SharingCatalog({"shareCredentialsVersion": 1, "endpoint": object()})
