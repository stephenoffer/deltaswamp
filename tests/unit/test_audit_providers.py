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
