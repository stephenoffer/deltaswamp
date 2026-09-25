"""Second-pass regression tests: Databricks catalog and credential vending.

No network: the WorkspaceClient is a stub answering with the SDK's own
dataclasses and error types.
"""

from __future__ import annotations

import pickle
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("databricks.sdk")

from databricks.sdk import errors as sdk_errors
from databricks.sdk.service import catalog as uc
from deltaswamp.catalog import databricks as dbx_mod
from deltaswamp.catalog.databricks import DatabricksUnityCatalog
from deltaswamp.credentials import databricks as cred_mod
from deltaswamp.credentials.base import (
    Cloud,
    Credentials,
    Operation,
    StaticCredentialProvider,
)
from deltaswamp.credentials.databricks import (
    DatabricksCredentialProvider,
    _config_attributes,
    _error_kind,
    _expires_at_seconds,
    azure_endpoint_for,
)
from deltaswamp.errors import CredentialError, InvalidReferenceError, PreflightError
from deltaswamp.identity import parse_ref

REF = parse_ref("main.sales.orders")


class Api:
    def __init__(self, name: str, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self._name = name
        self._calls = calls
        self.responses: dict[str, Any] = {}

    def __getattr__(self, method: str) -> Any:
        if method.startswith("_"):
            raise AttributeError(method)

        def call(*args: Any, **kwargs: Any) -> Any:
            if args:
                kwargs = {"method": args[0], "path": args[1], **kwargs}
            self._calls.append((f"{self._name}.{method}", kwargs))
            response = self.responses.get(method)
            if isinstance(response, BaseException):
                raise response
            if callable(response):
                return response(**kwargs)
            return response

        return call


class FakeWorkspace:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tables = Api("tables", self.calls)
        self.grants = Api("grants", self.calls)
        self.metastores = Api("metastores", self.calls)
        self.catalogs = Api("catalogs", self.calls)
        self.schemas = Api("schemas", self.calls)
        self.temporary_table_credentials = Api("temporary_table_credentials", self.calls)
        self.temporary_path_credentials = Api("temporary_path_credentials", self.calls)
        self.api_client = Api("api_client", self.calls)
        self.config = SimpleNamespace(host="https://example.cloud.databricks.com")

    def called(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for n, kwargs in self.calls if n == name]


@pytest.fixture
def ws() -> FakeWorkspace:
    return FakeWorkspace()


@pytest.fixture
def dbx(ws: FakeWorkspace) -> DatabricksUnityCatalog:
    catalog = DatabricksUnityCatalog(host="https://example.cloud.databricks.com", token="t")
    catalog._client = ws
    return catalog


def _info(**overrides: Any) -> uc.TableInfo:
    fields: dict[str, Any] = {
        "name": "orders",
        "catalog_name": "main",
        "schema_name": "sales",
        "full_name": "main.sales.orders",
        "table_id": "tid-1",
        "table_type": uc.TableType.EXTERNAL,
        "data_source_format": uc.DataSourceFormat.DELTA,
        "storage_location": "s3://bucket/orders",
    }
    fields.update(overrides)
    return uc.TableInfo(**fields)


def _aws(expiration_time: Any = None) -> uc.GenerateTemporaryTableCredentialResponse:
    return uc.GenerateTemporaryTableCredentialResponse(
        url="s3://bucket/orders",
        expiration_time=expiration_time,
        aws_temp_credentials=uc.AwsCredentials(
            access_key_id="AK", secret_access_key="SK", session_token="ST"
        ),
    )


def _ms(seconds_from_now: float) -> int:
    return int((time.time() + seconds_from_now) * 1000)


class CountingProvider(DatabricksCredentialProvider):
    def __init__(self, responses: list[Any], **kwargs: Any) -> None:
        super().__init__("tid-1", host="https://h", token="t", **kwargs)
        self.responses = responses
        self.ops: list[Operation] = []

    def _vend(self, operation: Operation) -> Any:
        self.ops.append(operation)
        return self._to_credentials(
            self.responses[min(len(self.ops), len(self.responses)) - 1], operation
        )


# ================================================================ Azure endpoint


class TestAzureEndpoint:
    def test_dfs_host_becomes_the_blob_endpoint(self) -> None:
        # object_store speaks only the Blob API; the dfs host does not serve it.
        assert (
            azure_endpoint_for("abfss://c@acct.dfs.core.windows.net/t")
            == "https://acct.blob.core.windows.net"
        )

    def test_sovereign_and_fabric_suffixes_are_kept(self) -> None:
        assert (
            azure_endpoint_for("abfss://c@acct.dfs.core.chinacloudapi.cn/t")
            == "https://acct.blob.core.chinacloudapi.cn"
        )
        assert (
            azure_endpoint_for("abfss://c@acct.dfs.fabric.microsoft.com/t")
            == "https://acct.blob.fabric.microsoft.com"
        )

    def test_blob_and_custom_hosts_are_verbatim(self) -> None:
        assert (
            azure_endpoint_for("az://c@acct.blob.core.windows.net/t")
            == "https://acct.blob.core.windows.net"
        )
        assert azure_endpoint_for("abfss://c@storage.corp.example/t") == (
            "https://storage.corp.example"
        )

    def test_azurite_endpoint_keeps_the_account_segment(self) -> None:
        assert azure_endpoint_for("http://127.0.0.1:10000/devstoreaccount1/c/t") == (
            "http://127.0.0.1:10000/devstoreaccount1"
        )
        assert azure_endpoint_for("http://localhost:10000/devstoreaccount1") == (
            "http://localhost:10000/devstoreaccount1"
        )

    def test_http_endpoint_allows_http(self) -> None:
        p = DatabricksCredentialProvider("tid-1", host="https://h", token="t")
        resp = uc.GenerateTemporaryTableCredentialResponse(
            url="http://127.0.0.1:10000/devstoreaccount1/c/t",
            azure_user_delegation_sas=uc.AzureUserDelegationSas(sas_token="sv=1"),
        )
        opts = p._to_credentials(resp, Operation.READ).as_storage_options()
        assert opts["azure_allow_http"] == "true"

    def test_endpoint_alias_satisfies_the_explicit_endpoint_check(self) -> None:
        creds = Credentials(
            cloud=Cloud.AZURE,
            url="az://c/t",
            expires_at=None,
            secrets={"azure_storage_sas_key": "x", "azure_storage_endpoint": "https://e"},
        )
        assert creds.as_storage_options()["azure_storage_endpoint"] == "https://e"
        emulator = Credentials(
            cloud=Cloud.AZURE, url="az://c/t", expires_at=None, secrets={"use_emulator": "true"}
        )
        assert emulator.as_storage_options() == {"use_emulator": "true"}


# ================================================================ error kinds


class TestErrorKinds:
    def test_table_or_view_not_found_bad_request_is_not_found(self) -> None:
        exc = sdk_errors.BadRequest("Table or view not found", error_code="TABLE_OR_VIEW_NOT_FOUND")
        assert _error_kind(exc) == "not_found"

    def test_resolve_bad_request_not_found_is_invalid_reference(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = sdk_errors.BadRequest(
            "[TABLE_OR_VIEW_NOT_FOUND] main.sales.orders", error_code="TABLE_OR_VIEW_NOT_FOUND"
        )
        with pytest.raises(InvalidReferenceError):
            dbx.resolve(REF)

    def test_throttled_vend_is_not_blamed_on_privileges(self, ws: FakeWorkspace) -> None:
        p = DatabricksCredentialProvider("tid-1", host="https://h", token="t")
        p._client = ws
        ws.temporary_table_credentials.responses["generate_temporary_table_credentials"] = (
            sdk_errors.TooManyRequests("rate limit")
        )
        with pytest.raises(CredentialError, match="throttling") as info:
            p.credentials()
        assert "EXTERNAL USE SCHEMA" not in str(info.value)

    def test_throttled_resolve_is_not_a_denial(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = sdk_errors.TemporarilyUnavailable(
            "workspace does not have capacity right now"
        )
        with pytest.raises(PreflightError, match="throttling") as info:
            dbx.resolve(REF)
        assert "denied" not in str(info.value)
        assert not ws.called("grants.get_effective")

    def test_throttled_governance_call_is_not_a_denial(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.catalogs.responses["list"] = sdk_errors.TooManyRequests("does not have quota")
        with pytest.raises(PreflightError, match="throttling"):
            dbx.list_catalogs()
        assert not ws.called("grants.get_effective")

    def test_unconfigurable_auth_is_not_blamed_on_the_metastore(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deltaswamp.credentials.databricks as sdk  # imports workspace_client by name

        def refuse(**kwargs: Any) -> Any:
            raise ValueError("default auth: cannot configure default credentials")

        monkeypatch.setattr(sdk, "workspace_client", refuse)
        p = DatabricksCredentialProvider("tid-1")
        with pytest.raises(CredentialError, match="could not configure Databricks") as info:
            p.credentials()
        assert "external data access" not in str(info.value)

    def test_workspace_auth_failure_is_a_credential_error(self) -> None:
        def fail() -> dict[str, str]:
            raise RuntimeError("invalid_client")

        p = DatabricksCredentialProvider("tid-1", host="https://h", token="t")
        p._client = SimpleNamespace(config=SimpleNamespace(host="https://h", authenticate=fail))
        with pytest.raises(CredentialError, match="commit API"):
            p.workspace_auth()

    def test_commit_tail_unauthenticated_is_not_the_preview_gate(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = _info(properties={"delta.feature.catalogManaged": "supported"})
        ws.api_client.responses["do"] = sdk_errors.Unauthenticated("token expired")
        with pytest.raises(PreflightError, match="credentials were rejected") as info:
            dbx.resolve(REF)
        assert "preview" not in str(info.value)

    def test_commit_tail_endpoint_missing_keeps_the_preview_message(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = _info(properties={"delta.feature.catalogManaged": "supported"})
        ws.api_client.responses["do"] = sdk_errors.NotFound(
            "no such endpoint", error_code="ENDPOINT_NOT_FOUND"
        )
        with pytest.raises(PreflightError, match="preview"):
            dbx.resolve(REF)


# ================================================================ refresh


class TestRefresh:
    def test_valid_credential_served_while_another_thread_refreshes(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class Slow(CountingProvider):
            def _vend(self, operation: Operation) -> Any:
                if self.ops:
                    started.set()
                    release.wait(5)
                return super()._vend(operation)

        p = Slow([_aws(_ms(3600)), _aws(_ms(7200))])
        first = p.credentials()
        object.__setattr__(first, "expires_at", time.time() + 60)  # inside the margin
        p._vended_at[Operation.READ] = time.time() - 3500
        refresher = threading.Thread(target=p.credentials)
        refresher.start()
        assert started.wait(5)
        t0 = time.monotonic()
        # Previously this blocked on the provider lock until the vend finished.
        assert p.credentials() is first
        assert time.monotonic() - t0 < 1
        release.set()
        refresher.join(5)
        assert len(p.ops) == 2

    def test_failed_refresh_ahead_backs_off(self) -> None:
        p = CountingProvider([_aws(_ms(3600))])
        first = p.credentials()
        object.__setattr__(first, "expires_at", time.time() + 120)  # inside the margin
        p._vended_at[Operation.READ] = time.time() - 3500
        attempts: list[int] = []

        def boom(operation: Operation) -> Any:
            attempts.append(1)
            raise CredentialError("503")

        p._vend = boom  # type: ignore[method-assign]
        for _ in range(5):
            assert p.credentials() is first
        assert len(attempts) == 1

    def test_credential_without_expiry_is_eventually_revended(self) -> None:
        p = CountingProvider([_aws(None), _aws(None)])
        p.credentials()
        p.credentials()
        assert len(p.ops) == 1
        p._vended_at[Operation.READ] = time.time() - cred_mod.UNKNOWN_EXPIRY_MAX_AGE_SECONDS - 1
        p.credentials()
        assert len(p.ops) == 2

    def test_iso_and_datetime_expiration_are_parsed(self) -> None:
        assert _expires_at_seconds("2030-01-01T00:00:00Z") == pytest.approx(
            datetime(2030, 1, 1, tzinfo=UTC).timestamp()
        )
        assert _expires_at_seconds(datetime(2030, 1, 1)) == pytest.approx(
            datetime(2030, 1, 1, tzinfo=UTC).timestamp()
        )
        assert _expires_at_seconds(True) is None

    def test_clock_skew_does_not_revend_every_call(self) -> None:
        # The server says it expires in 10 minutes; our clock is 20 minutes ahead.
        p = CountingProvider([_aws(_ms(-600))])
        p.credentials()
        p.credentials()
        p.credentials()
        assert len(p.ops) == 1

    def test_pickled_provider_refreshes_after_unpickle(self) -> None:
        p = CountingProvider([_aws(_ms(3600))])
        p.credentials()
        clone = pickle.loads(pickle.dumps(p))
        assert clone._cache == {} and clone._retry_after == {} and clone._vend_locks == {}
        clone.credentials()
        assert len(clone.ops) == 2


# ================================================================ static / path


class TestStaticProvider:
    def test_read_only_path_credential_refuses_a_write(self) -> None:
        creds = Credentials(
            cloud=Cloud.AWS,
            url="s3://b/t",
            expires_at=time.time() + 3600,
            operation=Operation.READ,
        )
        provider = StaticCredentialProvider(creds)
        assert provider.credentials(Operation.READ) is creds
        with pytest.raises(CredentialError, match="read-only"):
            provider.credentials(Operation.READ_WRITE)

    def test_path_credentials_record_their_operation(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.temporary_path_credentials.responses["generate_temporary_path_credentials"] = (
            uc.GenerateTemporaryPathCredentialResponse(
                url="s3://b/t",
                expiration_time=_ms(3600),
                aws_temp_credentials=uc.AwsCredentials(access_key_id="A", secret_access_key="S"),
            )
        )
        assert dbx.path_credentials("s3://b/t", "PATH_READ").operation is Operation.READ
        created = dbx.path_credentials("s3://b/t", "PATH_CREATE_TABLE")
        assert created.operation is Operation.READ_WRITE


# ================================================================ catalog


class TestCatalog:
    def test_manifest_lookup_survives_concurrent_expiry(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        listed = _info(
            securable_kind_manifest=uc.SecurableKindManifest(
                capabilities=["HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT"]
            )
        )

        def listing(**kwargs: Any) -> Any:
            # A concurrent resolve expires the entry right after it is stored.
            def gen() -> Any:
                yield listed

            return gen()

        ws.tables.responses["list"] = listing
        calls = {"n": 0}

        class Evicting(dict):  # type: ignore[type-arg]
            def __setitem__(self, key: Any, value: Any) -> None:
                super().__setitem__(key, value)
                calls["n"] += 1
                super().pop(key, None)  # the other thread's pop

        dbx._manifest_cache = Evicting()
        caps = dbx._capabilities_for(REF, _info())
        assert caps is not None and "HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT" in caps

    def test_list_tables_governed_without_manifest_is_not_vendable(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["list"] = [
            _info(row_filter=uc.TableRowFilter(function_name="f", input_column_names=["a"]))
        ]
        (table,) = dbx.list_tables("main", "sales")
        assert table.external_read_supported is False
        assert table.external_write_supported is False

    def test_list_tables_degrades_on_an_older_sdk(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        def old_list(**kwargs: Any) -> Any:
            if "include_manifest_capabilities" in kwargs:
                raise TypeError("unexpected keyword argument")
            return [_info()]

        ws.tables.responses["list"] = old_list
        assert [t.ref.table for t in dbx.list_tables("main", "sales")] == ["orders"]

    def test_transient_region_failure_is_not_cached_forever(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws.metastores.responses["summary"] = sdk_errors.TemporarilyUnavailable("503")
        ws.metastores.responses["current"] = sdk_errors.TemporarilyUnavailable("503")
        assert dbx._metastore_region() is None
        # Within the back-off window it is not re-asked on every resolve.
        n = len(ws.called("metastores.summary"))
        assert dbx._metastore_region() is None
        assert len(ws.called("metastores.summary")) == n
        monkeypatch.setattr(dbx, "_region_retry_at", 0.0)
        ws.metastores.responses["summary"] = uc.GetMetastoreSummaryResponse(region="eu-west-1")
        assert dbx._metastore_region() == "eu-west-1"

    def test_denied_region_lookup_is_cached(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.metastores.responses["summary"] = sdk_errors.PermissionDenied("no")
        ws.metastores.responses["current"] = sdk_errors.PermissionDenied("no")
        assert dbx._metastore_region() is None
        assert dbx._region_cached

    def test_catalog_with_explicit_config_pickles(self) -> None:
        class UnpicklableConfig:
            host = "https://adb-1.azuredatabricks.net"

            def __init__(self) -> None:
                self._header_factory: Callable[[], dict[str, str]] = lambda: {}

            def as_dict(self) -> dict[str, Any]:
                return {"host": self.host, "token": "dapi", "auth_type": "pat"}

        catalog = DatabricksUnityCatalog(config=UnpicklableConfig())  # type: ignore[arg-type, unused-ignore]
        clone = pickle.loads(pickle.dumps(catalog))
        assert clone._explicit_config is None
        assert clone._config_kwargs["host"] == "https://adb-1.azuredatabricks.net"

    def test_pickled_catalog_drops_monotonic_timestamps(self, dbx: DatabricksUnityCatalog) -> None:
        dbx._manifest_cache[("main", "sales")] = {}
        dbx._manifest_cached_at[("main", "sales")] = time.monotonic() + 1e9
        dbx._region_retry_at = time.monotonic() + 1e9
        clone = pickle.loads(pickle.dumps(dbx))
        assert clone._manifest_cache == {} and clone._manifest_cached_at == {}
        assert clone._region_retry_at == 0.0

    def test_providers_share_the_catalogs_config(
        self, dbx: DatabricksUnityCatalog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        live = object()
        dbx._client = SimpleNamespace(config=live)
        monkeypatch.setattr(dbx_mod, "_is_sdk_config", lambda obj: obj is live)
        assert dbx._provider_kwargs()["config"] is live

    def test_pickled_config_does_not_name_the_profile(self) -> None:
        config = SimpleNamespace(
            host="https://h",
            as_dict=lambda: {
                "host": "https://h",
                "token": "t",
                "profile": "prod",
                "config_file": "/home/me/.databrickscfg",
                "auth_type": "pat",
            },
        )
        attrs = _config_attributes(config)
        assert "profile" not in attrs and "config_file" not in attrs
        assert attrs["host"] == "https://h" and attrs["token"] == "t"

    def test_from_uri_takes_only_the_host(self) -> None:
        c = DatabricksUnityCatalog.from_uri("databricks://h.cloud.databricks.com/?profile=p")
        assert c._host == "h.cloud.databricks.com"
        assert c._profile == "p"
        assert DatabricksUnityCatalog.from_uri("databricks://https://h.x/")._host == "https://h.x"
