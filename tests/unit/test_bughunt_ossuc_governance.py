"""Regression tests for defects in catalog/ossuc.py, catalog/base.py and governance.py.

No network: `urllib.request.urlopen` inside the ossuc module is replaced by a
scripted fake that records every Request it is handed.
"""

from __future__ import annotations

import io
import json
import pickle
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from email.message import Message
from typing import Any

import pytest
from deltaswamp.catalog import ossuc
from deltaswamp.catalog.base import ResolvedTable, parse_commit_tail
from deltaswamp.catalog.ossuc import OSSUnityCatalog, OSSUnityCredentialProvider
from deltaswamp.credentials.base import Cloud, Operation
from deltaswamp.errors import (
    CorruptTableError,
    CredentialError,
    DeltaSwampError,
    InvalidReferenceError,
    PreflightError,
)
from deltaswamp.governance import (
    Grant,
    TableInfo,
    Volume,
    delta_schema_to_columns,
    staging_storage_options,
)
from deltaswamp.identity import RefKind, TableRef

BASE = "http://uc.test"


def ref(c: str = "main", s: str = "sch", t: str = "tbl") -> TableRef:
    return TableRef(kind=RefKind.CATALOG, catalog=c, schema=s, table=t, raw=f"{c}.{s}.{t}")


class _Resp:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


class FakeServer:
    """Routes (method, unquoted path) to a handler(query, body) -> (status, payload)."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Callable[[dict[str, str], Any], Any]] = {}
        self.requests: list[Any] = []

    def on(self, method: str, path: str, handler: Callable[[dict[str, str], Any], Any]) -> None:
        self.routes[(method, path)] = handler

    def urlopen(self, request: Any, timeout: float | None = None) -> _Resp:
        self.requests.append(request)
        parsed = urllib.parse.urlsplit(request.full_url)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        body = json.loads(request.data) if request.data else None
        key = (request.get_method(), urllib.parse.unquote(parsed.path))
        handler = self.routes.get(key)
        if handler is None:
            raise urllib.error.HTTPError(
                request.full_url, 404, "nf", Message(), io.BytesIO(b'{"message":"no route"}')
            )
        result = handler(query, body)
        if isinstance(result, BaseException):
            raise result
        status, payload = result
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        if status >= 400:
            raise urllib.error.HTTPError(
                request.full_url, status, "err", Message(), io.BytesIO(data)
            )
        return _Resp(data)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    fake = FakeServer()
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    return fake


UC = "/api/2.1/unity-catalog"


def _paged_handler(key: str, pages: list[list[dict[str, Any]]]) -> Callable[..., Any]:
    def handler(query: dict[str, str], body: Any) -> Any:
        index = int(query.get("page_token") or 0)
        payload: dict[str, Any] = {key: pages[index]}
        if index + 1 < len(pages):
            payload["next_page_token"] = str(index + 1)
        return 200, payload

    return handler


AWS_BODY = {
    "aws_temp_credentials": {
        "access_key_id": "AK",
        "secret_access_key": "SK",
        "session_token": "ST",
    },
    "expiration_time": 9999999999000,
}


# ------------------------------------------------------------------ _request


class TestRequest:
    def test_read_timeout_is_a_preflight_error(self, server: FakeServer) -> None:
        server.on("GET", f"{UC}/catalogs", lambda q, b: TimeoutError("timed out"))
        with pytest.raises(PreflightError, match="timed out"):
            OSSUnityCatalog(BASE).list_catalogs()

    def test_connection_reset_is_a_preflight_error(self, server: FakeServer) -> None:
        server.on("GET", f"{UC}/catalogs", lambda q, b: ConnectionResetError("reset"))
        with pytest.raises(PreflightError, match="reset"):
            OSSUnityCatalog(BASE).list_catalogs()

    def test_non_json_body_is_a_preflight_error(self, server: FakeServer) -> None:
        server.on("GET", f"{UC}/catalogs", lambda q, b: (200, b"<html>proxy</html>"))
        with pytest.raises(PreflightError, match="non-JSON"):
            OSSUnityCatalog(BASE).list_catalogs()

    def test_http_error_without_body(self, server: FakeServer) -> None:
        def boom(q: Any, b: Any) -> Any:
            return urllib.error.HTTPError(f"{BASE}/x", 500, "Server Error", {}, None)  # type: ignore[arg-type]

        server.on("GET", f"{UC}/catalogs", boom)
        with pytest.raises(ossuc.UnityCatalogHTTPError) as info:
            OSSUnityCatalog(BASE).list_catalogs()
        assert info.value.status == 500

    def test_non_http_url_refused_before_request(self, server: FakeServer) -> None:
        with pytest.raises(InvalidReferenceError):
            ossuc._request("file:///etc", "/passwd", None)
        assert server.requests == []


# ------------------------------------------------------------------ from_uri


def test_from_uri_without_scheme() -> None:
    cat = OSSUnityCatalog.from_uri("localhost:8080")
    assert cat._base_url == "http://localhost:8080"


def test_from_uri_with_no_host_is_refused() -> None:
    with pytest.raises(InvalidReferenceError):
        OSSUnityCatalog.from_uri("unity://")


# ----------------------------------------------------------------- resolve


class TestResolve:
    def test_missing_table_is_invalid_reference(self, server: FakeServer) -> None:
        with pytest.raises(InvalidReferenceError, match="does not exist"):
            OSSUnityCatalog(BASE).resolve(ref())

    def test_two_part_ref_refused(self, server: FakeServer) -> None:
        r = TableRef(kind=RefKind.CATALOG, catalog="main", schema="sch", table=None, raw="x")
        with pytest.raises(InvalidReferenceError):
            OSSUnityCatalog(BASE).resolve(r)

    def test_lowercase_table_type_parsed(self, server: FakeServer) -> None:
        server.on(
            "GET",
            f"{UC}/tables/main.sch.tbl",
            lambda q, b: (200, {"table_id": "id", "table_type": "managed"}),
        )
        assert OSSUnityCatalog(BASE).resolve(ref()).table_type is not None

    def test_property_values_stringified(self, server: FakeServer) -> None:
        server.on(
            "GET",
            f"{UC}/tables/main.sch.tbl",
            lambda q, b: (200, {"table_id": "id", "properties": {"a": 1, "b": None}}),
        )
        assert OSSUnityCatalog(BASE).resolve(ref()).properties == {"a": "1"}

    def test_commit_tail_path_is_percent_encoded(self, server: FakeServer) -> None:
        r = ref("main", "my schema", "t#1")
        server.on(
            "GET",
            f"{UC}/tables/main.my schema.t#1",
            lambda q, b: (
                200,
                {"table_id": "id", "properties": {"delta.feature.catalogManaged": "supported"}},
            ),
        )
        server.on(
            "GET",
            f"{UC}/delta/v1/catalogs/main/schemas/my schema/tables/t#1",
            lambda q, b: (200, {"latest-table-version": 3, "commits": []}),
        )
        resolved = OSSUnityCatalog(BASE).resolve(r)
        assert resolved.max_catalog_version == 3
        assert "%23" in server.requests[-1].full_url


# -------------------------------------------------------------- pagination


class TestPagination:
    def test_list_tables_follows_page_tokens(self, server: FakeServer) -> None:
        server.on(
            "GET",
            f"{UC}/tables",
            _paged_handler(
                "tables",
                [[{"name": "a", "table_id": "ida"}], [{"name": "b", "table_id": "idb"}]],
            ),
        )
        tables = OSSUnityCatalog(BASE).list_tables("main", "sch")
        assert [t.ref.table for t in tables] == ["a", "b"]

    def test_list_tables_keeps_table_id_on_provider(self, server: FakeServer) -> None:
        server.on(
            "GET",
            f"{UC}/tables",
            lambda q, b: (200, {"tables": [{"name": "a", "table_id": "ida"}]}),
        )
        (t,) = OSSUnityCatalog(BASE).list_tables("main", "sch")
        assert t.table_uuid == "ida"
        assert t.credential_provider is not None
        assert t.credential_provider.table_id == "ida"

    def test_list_tables_skips_nameless_entries(self, server: FakeServer) -> None:
        server.on("GET", f"{UC}/tables", lambda q, b: (200, {"tables": [{"table_id": "x"}]}))
        assert OSSUnityCatalog(BASE).list_tables("main", "sch") == []

    def test_list_catalogs_follows_page_tokens(self, server: FakeServer) -> None:
        server.on(
            "GET", f"{UC}/catalogs", _paged_handler("catalogs", [[{"name": "a"}], [{"name": "b"}]])
        )
        assert OSSUnityCatalog(BASE).list_catalogs() == ["a", "b"]

    def test_list_schemas_follows_page_tokens(self, server: FakeServer) -> None:
        server.on(
            "GET", f"{UC}/schemas", _paged_handler("schemas", [[{"name": "a"}], [{"name": "b"}]])
        )
        assert OSSUnityCatalog(BASE).list_schemas("main") == ["a", "b"]

    def test_repeated_page_token_does_not_loop_forever(self, server: FakeServer) -> None:
        server.on(
            "GET",
            f"{UC}/catalogs",
            lambda q, b: (200, {"catalogs": [{"name": "a"}], "next_page_token": "same"}),
        )
        with pytest.raises(PreflightError, match="repeated page token"):
            OSSUnityCatalog(BASE).list_catalogs()


# ----------------------------------------------------------- credentials


class TestCredentials:
    def _provider(self, r: TableRef | None = None, **kw: Any) -> OSSUnityCredentialProvider:
        return OSSUnityCredentialProvider(BASE, r or ref(), None, **kw)

    def test_credentials_path_is_percent_encoded(self, server: FakeServer) -> None:
        server.on(
            "GET",
            f"{UC}/delta/v1/catalogs/main/schemas/a b/tables/x?y/credentials",
            lambda q, b: (200, AWS_BODY),
        )
        self._provider(ref("main", "a b", "x?y")).credentials()
        assert "x%3Fy" in server.requests[-1].full_url

    def test_string_operation_accepted(self, server: FakeServer) -> None:
        server.on(
            "GET",
            f"{UC}/delta/v1/catalogs/main/schemas/sch/tables/tbl/credentials",
            lambda q, b: (200, AWS_BODY) if q["operation"] == "READ_WRITE" else (400, {}),
        )
        creds = self._provider().credentials("read_write")  # type: ignore[arg-type]
        assert creds.secrets["aws_access_key_id"] == "AK"

    def test_credentials_carry_table_id_and_location_scope(self, server: FakeServer) -> None:
        server.on(
            "GET",
            f"{UC}/delta/v1/catalogs/main/schemas/sch/tables/tbl/credentials",
            lambda q, b: (200, AWS_BODY),
        )
        creds = self._provider(table_id="tid", location="s3://b/t").credentials()
        assert creds.table_id == "tid"
        assert creds.scope_prefix == "s3://b/t"

    def test_provider_unpickles_from_old_state(self, server: FakeServer) -> None:
        p = self._provider(table_id="tid")
        state = p.__getstate__()
        state.pop("_location")
        q = OSSUnityCredentialProvider.__new__(OSSUnityCredentialProvider)
        q.__dict__.update(state)
        assert q._parse(AWS_BODY).table_id == "tid"
        pickle.loads(pickle.dumps(p))

    def test_path_ref_refused(self) -> None:
        r = TableRef(kind=RefKind.PATH, raw="s3://b/t", path="s3://b/t")
        with pytest.raises(InvalidReferenceError):
            OSSUnityCredentialProvider(BASE, r)

    def test_null_blocks_skipped(self) -> None:
        body = {
            "aws_temp_credentials": None,
            "gcp_oauth_token": {"oauth_token": "g"},
            "url": "gs://b/t",
        }
        assert ossuc._parse_vended(body).cloud is Cloud.GCP

    def test_aws_without_session_token(self) -> None:
        body = {"aws_temp_credentials": {"access_key_id": "a", "secret_access_key": "s"}}
        creds = ossuc._parse_vended(body)
        assert "aws_session_token" not in creds.secrets

    def test_incomplete_block_is_credential_error(self) -> None:
        with pytest.raises(CredentialError, match="secret_access_key"):
            ossuc._parse_vended({"aws_temp_credentials": {"access_key_id": "a"}})

    def test_bad_expiry_is_credential_error(self) -> None:
        with pytest.raises(CredentialError):
            ossuc._parse_vended({**AWS_BODY, "expiration_time": "soon"})

    def test_azure_endpoint_from_table_location(self) -> None:
        body = {"azure_user_delegation_sas": {"sas_token": "sv=x"}}
        creds = ossuc._parse_vended(body, fallback_url="abfss://c@acct.dfs.core.windows.net/t")
        assert creds.as_storage_options()["azure_endpoint"] == "https://acct.blob.core.windows.net"

    def test_delta_api_storage_credentials_shape(self) -> None:
        body = {
            "storage-credentials": [
                {
                    "prefix": "s3://b/t",
                    "operation": "READ",
                    "config": {"s3.access-key-id": "A", "s3.secret-access-key": "S"},
                    "expiration-time-ms": 5000,
                }
            ]
        }
        creds = ossuc._parse_vended(body, fallback_url="s3://b/t")
        assert creds.cloud is Cloud.AWS
        assert creds.secrets["aws_access_key_id"] == "A"
        assert creds.expires_at == 5.0

    def test_path_credentials_empty_url_filled(self, server: FakeServer) -> None:
        server.on(
            "POST",
            f"{UC}/temporary-path-credentials",
            lambda q, b: (200, {**AWS_BODY, "url": ""}),
        )
        creds = OSSUnityCatalog(BASE).path_credentials("s3://b/p")
        assert creds.url == "s3://b/p"


# ------------------------------------------------------------ permissions


class TestPermissions:
    def _patch(self, server: FakeServer) -> list[Any]:
        seen: list[Any] = []

        def handler(q: Any, body: Any) -> Any:
            seen.append(body)
            return 200, {"privilege_assignments": []}

        server.on("PATCH", f"{UC}/permissions/table/main.sch.tbl", handler)
        return seen

    def test_single_string_privilege(self, server: FakeServer) -> None:
        seen = self._patch(server)
        OSSUnityCatalog(BASE).grant(ref(), "alice", "SELECT")
        assert seen[0]["changes"][0]["add"] == ["SELECT"]

    def test_duplicate_privileges_collapsed(self, server: FakeServer) -> None:
        seen = self._patch(server)
        OSSUnityCatalog(BASE).grant(ref(), "alice", ["select", "SELECT", "use_schema"])
        assert seen[0]["changes"][0]["add"] == ["SELECT", "USE SCHEMA"]

    def test_empty_principal_refused(self, server: FakeServer) -> None:
        with pytest.raises(InvalidReferenceError, match="principal"):
            OSSUnityCatalog(BASE).grant(ref(), " ", ["SELECT"])
        assert server.requests == []

    def test_enum_securable_type(self, server: FakeServer) -> None:
        import enum

        class SecurableType(enum.Enum):
            SCHEMA = "SCHEMA"

        server.on(
            "GET",
            f"{UC}/permissions/schema/main.sch",
            lambda q, b: (200, {"privilege_assignments": []}),
        )
        OSSUnityCatalog(BASE).grants("main.sch", securable_type=SecurableType.SCHEMA)  # type: ignore[arg-type]

    def test_empty_securable_name_refused(self, server: FakeServer) -> None:
        with pytest.raises(InvalidReferenceError):
            OSSUnityCatalog(BASE).grants("", securable_type="SCHEMA")


# ---------------------------------------------------------------- volumes


def test_create_volume_bad_type_refused(server: FakeServer) -> None:
    with pytest.raises(InvalidReferenceError, match="MANAGED or EXTERNAL"):
        OSSUnityCatalog(BASE).create_volume("main", "sch", "v", volume_type="EXTRNAL")


def test_create_managed_volume_with_location_refused(server: FakeServer) -> None:
    with pytest.raises(InvalidReferenceError, match="EXTERNAL"):
        OSSUnityCatalog(BASE).create_volume("main", "sch", "v", storage_location="s3://b/v")


# --------------------------------------------------------------- register


SCHEMA = {
    "type": "struct",
    "fields": [
        {"name": "id", "type": "long", "nullable": True, "metadata": {}},
        {"name": "day", "type": "string", "nullable": True, "metadata": {}},
    ],
}


class TestRegister:
    def _server(self, server: FakeServer) -> list[Any]:
        seen: list[Any] = []

        def post(q: Any, body: Any) -> Any:
            seen.append(body)
            return 200, body

        server.on("POST", f"{UC}/tables", post)
        server.on("GET", f"{UC}/tables/main.sch.tbl", lambda q, b: (200, {"table_id": "i"}))
        return seen

    def test_string_partition_column(self, server: FakeServer) -> None:
        seen = self._server(server)
        OSSUnityCatalog(BASE).register_table(
            ref(), "s3://b/t", columns_schema_json=SCHEMA, partition_columns="day"
        )
        cols = {c["name"]: c.get("partition_index") for c in seen[0]["columns"]}
        assert cols == {"id": None, "day": 0}

    def test_properties_stringified(self, server: FakeServer) -> None:
        seen = self._server(server)
        OSSUnityCatalog(BASE).register_table(
            ref(),
            "s3://b/t",
            properties={"delta.appendOnly": True, "n": 3},  # type: ignore[dict-item]
        )
        assert seen[0]["properties"] == {"delta.appendOnly": "true", "n": "3"}

    def test_empty_location_refused(self, server: FakeServer) -> None:
        with pytest.raises(InvalidReferenceError, match="location"):
            OSSUnityCatalog(BASE).register_table(ref(), "")


# --------------------------------------------------------------- base.py


class TestCommitTail:
    def test_sorted_and_deduplicated(self) -> None:
        body = {
            "commits": [
                {"version": 3, "file-name": "c", "file-size": 1},
                {"version": 2, "file-name": "b", "file-size": 1},
                {"version": 3, "file-name": "c", "file-size": 1},
            ],
            "latest-table-version": 3,
        }
        entries, latest, _ = parse_commit_tail(body)
        assert [e.version for e in entries] == [2, 3]
        assert latest == 3

    def test_versionless_commit_is_clear_error(self) -> None:
        with pytest.raises(CorruptTableError, match="no version"):
            parse_commit_tail({"commits": [{"file-name": "x"}]})


class TestCatalogManagedProperty:
    def test_catalog_owned_preview_property(self) -> None:
        rt = ResolvedTable(
            ref=ref(), location=None, properties={"delta.feature.catalogOwned-preview": "supported"}
        )
        assert rt.is_catalog_managed

    def test_enabled_spelling(self) -> None:
        rt = ResolvedTable(
            ref=ref(), location=None, properties={"delta.feature.catalogManaged": "Enabled"}
        )
        assert rt.is_catalog_managed


# ---------------------------------------------------------- governance.py


class TestGrantParsing:
    def test_null_effective_privilege_skipped(self) -> None:
        body = {
            "privilege_assignments": [
                {"principal": "a", "privileges": [{"privilege": None}, {"privilege": "SELECT"}]}
            ]
        }
        (grant,) = Grant.list_from_api(body)
        assert grant.privileges == ("SELECT",)

    def test_duplicate_privileges_collapsed(self) -> None:
        body = {
            "privilege_assignments": [
                {"principal": "a", "privileges": ["USE SCHEMA", "USE_SCHEMA"]}
            ]
        }
        (grant,) = Grant.list_from_api(body)
        assert grant.privileges == ("USE_SCHEMA",)

    def test_slots_object_parsed(self) -> None:
        class Obj:
            __slots__ = ("privilege_assignments",)

            def __init__(self) -> None:
                self.privilege_assignments = [{"principal": "a", "privileges": ["SELECT"]}]

        assert Grant.list_from_api(Obj())[0].principal == "a"


def test_table_info_drops_null_properties() -> None:
    info = TableInfo.from_api({"name": "t", "properties": {"a": "1", "b": None}})
    assert info.properties == {"a": "1"}


class TestStagingOptions:
    def test_prefix_is_matched_on_a_path_boundary(self) -> None:
        creds = [
            {
                "prefix": "s3://b/t",
                "operation": "READ_WRITE",
                "config": {"s3.access-key-id": "WRONG"},
            },
            {"prefix": "s3://b/", "operation": "READ", "config": {"s3.access-key-id": "RIGHT"}},
        ]
        options, _ = staging_storage_options("s3://b/t2", creds)
        assert options["aws_access_key_id"] == "RIGHT"

    def test_single_mapping_accepted(self) -> None:
        cred = {"prefix": "s3://b/t", "config": {"s3.access-key-id": "A"}}
        options, _ = staging_storage_options("s3://b/t", cred)  # type: ignore[arg-type]
        assert options == {"aws_access_key_id": "A"}


class TestSchemaColumns:
    def test_string_partition_column(self) -> None:
        cols = delta_schema_to_columns(SCHEMA, "day")
        assert [c.get("partition_index") for c in cols] == [None, 0]

    def test_duplicate_partition_column_refused(self) -> None:
        with pytest.raises(DeltaSwampError, match="twice"):
            delta_schema_to_columns(SCHEMA, ["day", "day"])


class _Stream(io.BytesIO):
    closed_by_us = False

    def close(self) -> None:
        type(self).closed_by_us = True
        super().close()


class _Files:
    def __init__(self, listing: list[dict[str, Any]] | None = None) -> None:
        self.listing = listing or []
        self.downloaded: list[str] = []

    def download(self, file_path: str) -> Any:
        self.downloaded.append(file_path)

        class R:
            contents = _Stream(b"data")

        return R()

    def list_directory_contents(self, directory_path: str) -> list[dict[str, Any]]:
        return self.listing


class TestVolume:
    def test_read_closes_the_stream(self) -> None:
        vol = Volume("c.s.v", _Files())
        assert vol.read("x") == b"data"
        assert _Stream.closed_by_us

    def test_absolute_path_inside_volume(self) -> None:
        vol = Volume("c.s.v", _Files())
        assert vol.path("/Volumes/c/s/v/dir/f") == "/Volumes/c/s/v/dir/f"

    def test_listing_is_relative_when_server_lowercases(self) -> None:
        files = _Files([{"path": "/Volumes/main/s/v/a.txt", "name": "a.txt"}])
        vol = Volume("Main.s.v", files)
        assert vol.list()[0].path == "a.txt"


def test_operation_enum_value_query(server: FakeServer) -> None:
    server.on(
        "GET",
        f"{UC}/delta/v1/catalogs/main/schemas/sch/tables/tbl/credentials",
        lambda q, b: (200, AWS_BODY) if q["operation"] == "READ" else (400, {}),
    )
    OSSUnityCredentialProvider(BASE, ref()).credentials(Operation.READ)
