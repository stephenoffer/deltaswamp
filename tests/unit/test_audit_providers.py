"""Regressions for the provider-matrix audit (PV-*), with fakes: no servers."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest
from deltaswamp.capability import Engine, Operation
from deltaswamp.catalog.base import ResolvedTable
from deltaswamp.catalog.ossuc import OSSUnityCatalog, _parse_vended, _request
from deltaswamp.credentials.base import Cloud
from deltaswamp.errors import InvalidArgumentError, InvalidReferenceError, UnreachableTableError
from deltaswamp.identity import parse_ref

from tests.helpers import FakeEngine, resolved_table


def _router(*, warehouse_catalog: bool) -> Any:
    from deltaswamp.router import Router

    kinds = (Engine.KERNEL, Engine.DELTARS, Engine.SQL)
    return Router(
        engines={k: FakeEngine(k, yes=set()) for k in kinds},
        allow_sql_fallback=True,
        warehouse_catalog=warehouse_catalog,
    )


# ------------------------------------------------------------ PV-1 / PV-9


class TestWarehouseOnlyForDatabricks:
    def test_connect_refuses_the_fallback_on_oss_unity_catalog(self) -> None:
        with pytest.raises(InvalidArgumentError, match="Databricks SQL warehouse"):
            import deltaswamp as ds

            ds.connect("uc://http://localhost:1", allow_sql_fallback=True)

    def test_connect_refuses_the_fallback_on_a_path_connection(self) -> None:
        import deltaswamp as ds

        with pytest.raises(InvalidArgumentError, match="allow_sql_fallback"):
            ds.connect("file://", allow_sql_fallback=True)

    def test_router_never_asks_the_warehouse_for_another_catalog(self) -> None:
        from deltaswamp.router import Router

        # SQL would accept; the connection's catalog is not Databricks.
        router = Router(
            engines={Engine.SQL: FakeEngine(Engine.SQL)},
            allow_sql_fallback=True,
            warehouse_catalog=False,
        )
        verdict = router.capability(Operation.HISTORY, resolved_table("r3.m.t"))
        assert not verdict.ok
        assert "allow_sql_fallback" not in verdict.remedy
        assert "not one" in verdict.reason

    def test_refusals_never_suggest_the_fallback_there(self) -> None:
        router = _router(warehouse_catalog=False)
        router.allow_sql_fallback = False
        table = resolved_table("r3.m.t", reader_features=frozenset({"catalogManaged"}))
        for op in (Operation.HISTORY, Operation.DROP_FEATURE, Operation.ADD_COLUMN):
            verdict = router.capability(op, table)
            assert not verdict.ok
            assert "allow_sql_fallback" not in verdict.remedy, op
            with pytest.raises(UnreachableTableError) as info:
                router.engine_for(op, table)
            assert "allow_sql_fallback" not in str(info.value)

    def test_databricks_connections_still_suggest_it(self) -> None:
        router = _router(warehouse_catalog=True)
        router.allow_sql_fallback = False
        verdict = router.capability(Operation.DROP_FEATURE, resolved_table("main.s.t"))
        assert "allow_sql_fallback" in verdict.remedy


# ------------------------------------------------------------------ PV-2


def _shared(**fields: Any) -> ResolvedTable:
    fields.setdefault("sharing_profile", "{}")
    return ResolvedTable(ref=parse_ref("s.d.t"), location=None, **fields)


class TestSharingDeltaFormat:
    def test_legacy_column_mapping_needs_the_delta_format(self) -> None:
        from deltaswamp.engine.sharing import SharingEngine

        legacy = _shared(
            min_reader_version=2,
            min_writer_version=5,
            properties={"delta.columnMapping.mode": "name"},
        )
        assert SharingEngine._needs_delta_format(legacy)

    def test_reader_two_without_mapping_stays_parquet(self) -> None:
        from deltaswamp.engine.sharing import SharingEngine

        plain = _shared(min_reader_version=2, min_writer_version=5)
        assert not SharingEngine._needs_delta_format(plain)

    def test_physically_named_file_is_refused_not_null_filled(self) -> None:
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        from deltaswamp.engine.sharing import SharingEngine

        buf = io.BytesIO()
        pq.write_table(pa.table({"col-1a2b": [1, 2]}), buf)
        engine = SharingEngine()
        engine._download = lambda url: buf.getvalue()  # type: ignore[method-assign]
        schema = pa.schema([("id", pa.int64())])
        with pytest.raises(UnreachableTableError, match="physically named"):
            engine._read_file("https://h/x.parquet", {}, schema)


# ------------------------------------------------------------------ PV-3


class TestSharingNeverReadsLocalFiles:
    def _lines(self, path: str) -> list[str]:
        schema = json.dumps(
            {
                "type": "struct",
                "fields": [{"name": "id", "type": "long", "nullable": True, "metadata": {}}],
            }
        )
        return [
            json.dumps(
                {"protocol": {"deltaProtocol": {"minReaderVersion": 1, "minWriterVersion": 2}}}
            ),
            json.dumps(
                {
                    "metaData": {
                        "deltaMetadata": {
                            "id": "m",
                            "format": {"provider": "parquet", "options": {}},
                            "schemaString": schema,
                            "partitionColumns": [],
                            "configuration": {},
                        }
                    }
                }
            ),
            json.dumps(
                {
                    "file": {
                        "id": "1",
                        "deltaSingleAction": {
                            "add": {
                                "path": path,
                                "partitionValues": {},
                                "size": 1,
                                "modificationTime": 0,
                                "dataChange": True,
                            }
                        },
                    }
                }
            ),
        ]

    def test_https_url_is_downloaded_not_opened_locally(self, tmp_path: Any) -> None:
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        pytest.importorskip("delta_kernel_rust_sharing_wrapper")
        from deltaswamp.engine.sharing import SharingEngine

        secret = tmp_path / "secret.parquet"
        pq.write_table(pa.table({"id": pa.array([42], pa.int64())}), secret)
        served = io.BytesIO()
        pq.write_table(pa.table({"id": pa.array([7], pa.int64())}), served)
        fetched: list[str] = []
        engine = SharingEngine()

        def download(url: str) -> bytes:
            fetched.append(url)
            return served.getvalue()

        engine._download = download  # type: ignore[method-assign]
        url = "https://attacker.example" + str(secret)
        out = engine._replay_delta_log(self._lines(url), what="read").to_pylist()
        # Before: the wrapper read the local file and returned id 42.
        assert out == [{"id": 7}]
        assert fetched == [url]

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "/etc/passwd", "https:///etc/passwd"])
    def test_non_http_or_hostless_urls_are_refused(self, url: str) -> None:
        pytest.importorskip("delta_kernel_rust_sharing_wrapper")
        from deltaswamp.engine.sharing import SharingEngine

        with pytest.raises(UnreachableTableError, match="presigned"):
            SharingEngine()._replay_delta_log(self._lines(url), what="read")


# ------------------------------------------------------------ PV-5, PV-13


class TestSharingCanIsHonest:
    def test_cdf_refused_without_change_data_feed(self) -> None:
        pytest.importorskip("delta_sharing")
        from deltaswamp.engine.sharing import SharingEngine

        table = _shared(table_id="m", properties={})
        verdict = SharingEngine().supports(Operation.CDF, table)
        assert not verdict.ok and "change data feed" in verdict.reason

    def test_history_dependent_reads_name_the_dependency(self) -> None:
        pytest.importorskip("delta_sharing")
        from deltaswamp.engine.sharing import SharingEngine

        table = _shared(table_id="m", properties={"delta.enableChangeDataFeed": "true"})
        for op in (Operation.CDF, Operation.TIME_TRAVEL):
            verdict = SharingEngine().supports(op, table)
            assert verdict.ok and "WITH HISTORY" in verdict.reason

    def test_v2_bearer_token_profile_is_usable(self) -> None:
        pytest.importorskip("delta_sharing")
        from deltaswamp.catalog.sharing import load_profile, rest_client

        profile = load_profile(
            {
                "shareCredentialsVersion": 2,
                "type": "bearer_token",
                "endpoint": "http://localhost:1/",
                "bearerToken": "t",
            }
        )
        rest_client(profile).close()  # raised a bare RuntimeError before


# ----------------------------------------------------- PV-6, PV-7, PV-14


class TestOssUnityCatalog:
    def test_local_path_credentials_need_no_secrets(self) -> None:
        body = {
            "aws_temp_credentials": None,
            "azure_user_delegation_sas": None,
            "gcp_oauth_token": None,
            "url": "file:///tmp/x",
        }
        creds = _parse_vended(body)
        assert creds.cloud is Cloud.LOCAL and creds.secrets == {}

    def test_cloud_url_without_a_block_is_still_refused(self) -> None:
        from deltaswamp.errors import CredentialError

        with pytest.raises(CredentialError):
            _parse_vended({"aws_temp_credentials": None, "url": "s3://b/x"})

    def _serve(self, monkeypatch: pytest.MonkeyPatch, status: int, text: str) -> None:
        import urllib.error
        import urllib.request

        class Response(io.BytesIO):
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        def urlopen(request: Any, timeout: float = 0) -> Any:
            if status >= 400:
                raise urllib.error.HTTPError(
                    request.full_url,
                    status,
                    "err",
                    {},
                    io.BytesIO(text.encode()),  # type: ignore[arg-type]
                )
            return Response(text.encode())

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    def test_plain_text_delete_answer_is_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._serve(monkeypatch, 200, "200 OK")
        assert _request("http://h", "/x", None, method="DELETE") == {}
        OSSUnityCatalog("http://h").drop_table(parse_ref("a.b.c"))

    def test_grant_to_unknown_principal_names_the_principal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = {
            "error_code": "NOT_FOUND",
            "stack_trace": "x" * 900,
            "message": "User not found: a@x",
        }
        self._serve(monkeypatch, 404, json.dumps(body))
        with pytest.raises(InvalidReferenceError, match="knows no principal 'a@x'"):
            OSSUnityCatalog("http://h").grant(parse_ref("a.b.c"), "a@x", ["SELECT"])

    def test_preflight_takes_400_as_reachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._serve(monkeypatch, 400, '{"message": "Mandatory parameter is missing: catalog"}')
        assert OSSUnityCatalog("http://h").preflight() == []
        self._serve(monkeypatch, 404, "nope")
        assert OSSUnityCatalog("http://h").preflight()


# ------------------------------------------------------------ PV-4, PV-15


class TApplicationException(Exception):
    pass


class NoSuchObjectException(Exception):
    pass


@pytest.fixture
def hms4(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A Hive 4 metastore: get_table_req only, get_table is an invalid method."""
    import sys
    import types

    state: dict[str, Any] = {"dropped": [], "tables": {}}

    class GetTableRequest:
        def __init__(self, dbName: str, tblName: str, **_: Any) -> None:
            self.dbName, self.tblName = dbName, tblName

    class Client:
        def get_table(self, db: str, name: str) -> Any:
            raise TApplicationException("Invalid method name: 'get_table'")

        def get_table_req(self, req: Any) -> Any:
            key = (req.dbName, req.tblName)
            if key not in state["tables"]:
                raise NoSuchObjectException(f"{key} table not found")
            return types.SimpleNamespace(table=state["tables"][key])

        def get_all_databases(self) -> list[str]:
            return ["default", "db"]

        def drop_table(self, db: str, name: str, delete_data: bool) -> None:
            state["dropped"].append((db, name, delete_data))

    Client.__module__ = "fakehms.ThriftHiveMetastore"
    ttypes = types.ModuleType("fakehms.ttypes")
    ttypes.GetTableRequest = GetTableRequest  # type: ignore[attr-defined]

    class Connection:
        def __enter__(self) -> Any:
            return types.SimpleNamespace(client=Client())

        def __exit__(self, *exc: Any) -> None:
            return None

    module = types.ModuleType("pymetastore.metastore")
    module.HMS = types.SimpleNamespace(create=lambda host, port: Connection())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pymetastore", types.ModuleType("pymetastore"))
    monkeypatch.setitem(sys.modules, "pymetastore.metastore", module)
    monkeypatch.setitem(sys.modules, "fakehms", types.ModuleType("fakehms"))
    monkeypatch.setitem(sys.modules, "fakehms.ttypes", ttypes)
    state["tables"][("db", "t")] = types.SimpleNamespace(
        parameters={"spark.sql.sources.provider": "delta"},
        tableType="EXTERNAL_TABLE",
        viewOriginalText=None,
        sd=types.SimpleNamespace(
            location="s3://b/t", serdeInfo=types.SimpleNamespace(parameters=None)
        ),
    )
    return state


class TestHiveMetastore:
    def test_hive_4_tables_resolve(self, hms4: dict[str, Any]) -> None:
        from deltaswamp.catalog.hms import HiveMetastoreCatalog

        cat = HiveMetastoreCatalog("hms://h:9083")
        assert cat.resolve(parse_ref("hive_metastore.db.t")).location == "s3://b/t"
        with pytest.raises(InvalidReferenceError, match="does not exist"):
            cat.resolve(parse_ref("hive_metastore.db.nope"))
        assert cat.preflight() == []

    def test_two_part_names_and_foreign_catalog_parts(self, hms4: dict[str, Any]) -> None:
        import deltaswamp as ds

        conn = ds.connect("hms://h:9083")
        assert conn.table("db.t").resolved.location == "s3://b/t"
        with pytest.raises(InvalidReferenceError, match="only databases and tables"):
            conn.table("prod.db.t")

    def test_namespaces_and_drop(self, hms4: dict[str, Any]) -> None:
        from deltaswamp.catalog.hms import HiveMetastoreCatalog

        cat = HiveMetastoreCatalog("hms://h:9083")
        assert cat.list_catalogs() == ["hive_metastore"]
        assert cat.list_schemas("hive_metastore") == ["default", "db"]
        cat.drop_table(parse_ref("hive_metastore.db.t"))
        assert hms4["dropped"] == [("db", "t", False)]  # data kept


class TestGlue:
    def test_namespaces_drop_and_catalog_parts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types

        from deltaswamp.catalog.glue import GlueCatalog

        deleted: list[dict[str, Any]] = []

        class Paginator:
            def paginate(self, **kwargs: Any) -> Any:
                yield {"DatabaseList": [{"Name": "a"}]}
                yield {"DatabaseList": [{"Name": "b"}]}

        client = types.SimpleNamespace(
            get_paginator=lambda name: Paginator(),
            delete_table=lambda **kwargs: deleted.append(kwargs),
        )
        boto3 = types.ModuleType("boto3")
        boto3.client = lambda service, region_name=None: client  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "boto3", boto3)
        cat = GlueCatalog()
        assert cat.list_catalogs() == ["glue"]
        assert cat.list_schemas("glue") == ["a", "b"]
        cat.drop_table(parse_ref("glue://db.t"))
        assert deleted == [{"DatabaseName": "db", "Name": "t"}]
        with pytest.raises(InvalidReferenceError, match="only databases and tables"):
            cat.resolve(parse_ref("prod.db.t"))
