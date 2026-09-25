"""Regression tests for Databricks Unity Catalog resolution and credential vending.

No network: the WorkspaceClient is a small stub answering with the SDK's own
dataclasses and error types.
"""

from __future__ import annotations

import pickle
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("databricks.sdk")

from databricks.sdk import errors as sdk_errors
from databricks.sdk.service import catalog as uc
from deltaswamp.catalog.databricks import DatabricksUnityCatalog, _url_name
from deltaswamp.credentials.base import Cloud, Operation
from deltaswamp.credentials.databricks import (
    DatabricksCredentialProvider,
    azure_endpoint_for,
)
from deltaswamp.errors import (
    CredentialError,
    InvalidArgumentError,
    InvalidReferenceError,
    PreflightError,
)
from deltaswamp.identity import parse_ref

REF = parse_ref("main.sales.orders")


# ------------------------------------------------------------------ helpers


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
        self.entity_tag_assignments = Api("entity_tag_assignments", self.calls)
        self.table_constraints = Api("table_constraints", self.calls)
        self.volumes = Api("volumes", self.calls)
        self.schemas = Api("schemas", self.calls)
        self.catalogs = Api("catalogs", self.calls)
        self.metastores = Api("metastores", self.calls)
        self.temporary_path_credentials = Api("temporary_path_credentials", self.calls)
        self.temporary_table_credentials = Api("temporary_table_credentials", self.calls)
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


def _provider(**kwargs: Any) -> DatabricksCredentialProvider:
    kwargs.setdefault("table_id", "tid-1")
    kwargs.setdefault("host", "https://h")
    kwargs.setdefault("token", "t")
    return DatabricksCredentialProvider(**kwargs)


def _aws(
    expiration_time: int | None = None, **creds: Any
) -> uc.GenerateTemporaryTableCredentialResponse:
    fields = {"access_key_id": "AK", "secret_access_key": "SK", "session_token": "ST"}
    fields.update(creds)
    return uc.GenerateTemporaryTableCredentialResponse(
        url="s3://bucket/orders",
        expiration_time=expiration_time,
        aws_temp_credentials=uc.AwsCredentials(**fields),
    )


class CountingProvider(DatabricksCredentialProvider):
    """Records each vend instead of calling the SDK."""

    def __init__(self, responses: list[Any], **kwargs: Any) -> None:
        super().__init__("tid-1", host="https://h", token="t", **kwargs)
        self.responses = responses
        self.ops: list[Operation] = []

    def _vend(self, operation: Operation) -> Any:
        self.ops.append(operation)
        return self._to_credentials(
            self.responses[min(len(self.ops), len(self.responses)) - 1], operation
        )


# ============================================================ credentials


class TestCredentialVending:
    def test_string_operation_vends_read_write(self) -> None:
        p = CountingProvider([_aws(int((time.time() + 3600) * 1000))])
        p.credentials("READ_WRITE")  # type: ignore[arg-type]
        assert p.ops == [Operation.READ_WRITE]
        assert p.ops[0] is Operation.READ_WRITE

    def test_unknown_operation_is_a_clear_error(self) -> None:
        p = CountingProvider([_aws()])
        with pytest.raises(CredentialError, match="unknown credential operation"):
            p.credentials("WRITE")  # type: ignore[arg-type]

    def test_vend_uses_sdk_read_write_enum_for_string_op(self, ws: FakeWorkspace) -> None:
        p = _provider()
        p._client = ws
        ws.temporary_table_credentials.responses["generate_temporary_table_credentials"] = _aws()
        p.credentials("READ_WRITE")  # type: ignore[arg-type]
        (call,) = ws.called("temporary_table_credentials.generate_temporary_table_credentials")
        assert call["operation"] is uc.TableOperation.READ_WRITE

    def test_expiration_in_seconds_is_not_read_as_1970(self) -> None:
        p = _provider()
        deadline = int(time.time()) + 3600
        creds = p._to_credentials(_aws(deadline), Operation.READ)
        assert creds.expires_at == pytest.approx(deadline)
        assert not creds.is_expired

    def test_expiration_in_milliseconds(self) -> None:
        p = _provider()
        deadline = time.time() + 3600
        creds = p._to_credentials(_aws(int(deadline * 1000)), Operation.READ)
        assert creds.expires_at == pytest.approx(deadline, abs=1)

    def test_short_lived_credential_is_not_revended_every_call(self) -> None:
        # 120s of life, under the 300s margin: previously every call re-vended.
        p = CountingProvider([_aws(int((time.time() + 120) * 1000))])
        p.credentials()
        p.credentials()
        p.credentials()
        assert len(p.ops) == 1

    def test_near_expiry_credential_is_revended(self) -> None:
        p = CountingProvider(
            [_aws(int((time.time() + 3600) * 1000)), _aws(int((time.time() + 7200) * 1000))]
        )
        first = p.credentials()
        # Pretend it was vended long ago and is now inside its margin.
        p._vended_at[Operation.READ] = time.time() - 3500
        object.__setattr__(first, "expires_at", time.time() + 60)
        p.credentials()
        assert len(p.ops) == 2

    def test_access_point_arn_becomes_an_endpoint_url(self) -> None:
        p = _provider(region="us-west-2")
        creds = p._to_credentials(
            _aws(access_point="arn:aws:s3:eu-central-1:123456789012:accesspoint/my-ap"),
            Operation.READ,
        )
        opts = creds.as_storage_options()
        assert opts["aws_endpoint_url"] == (
            "https://my-ap-123456789012.s3-accesspoint.eu-central-1.amazonaws.com"
        )
        assert opts["aws_virtual_hosted_style_request"] == "true"
        assert opts["aws_region"] == "eu-central-1"
        assert not opts["aws_endpoint_url"].startswith("arn:")

    def test_missing_session_token_is_dropped_not_none(self) -> None:
        p = _provider()
        creds = p._to_credentials(_aws(session_token=None), Operation.READ)
        assert "aws_session_token" not in creds.secrets
        assert all(isinstance(v, str) for v in creds.secrets.values())

    def test_missing_access_key_is_a_clear_error(self) -> None:
        p = _provider()
        with pytest.raises(CredentialError, match="aws_access_key_id"):
            p._to_credentials(_aws(access_key_id=None), Operation.READ)

    def test_r2_gets_its_account_endpoint(self) -> None:
        p = _provider()
        resp = uc.GenerateTemporaryTableCredentialResponse(
            url="r2://bkt@acct123.r2.cloudflarestorage.com/t",
            r2_temp_credentials=uc.R2Credentials(
                access_key_id="AK", secret_access_key="SK", session_token="ST"
            ),
        )
        creds = p._to_credentials(resp, Operation.READ)
        assert creds.cloud is Cloud.R2
        assert creds.secrets["aws_endpoint_url"] == "https://acct123.r2.cloudflarestorage.com"
        assert creds.secrets["aws_region"] == "auto"

    def test_azure_sas_leading_question_mark_stripped(self) -> None:
        p = _provider(table_url="abfss://c@acct.dfs.core.windows.net/t")
        resp = uc.GenerateTemporaryTableCredentialResponse(
            url="abfss://c@acct.dfs.core.windows.net/t",
            azure_user_delegation_sas=uc.AzureUserDelegationSas(sas_token="?sv=1&sig=x"),
        )
        creds = p._to_credentials(resp, Operation.READ)
        assert creds.secrets["azure_storage_sas_key"] == "sv=1&sig=x"

    def test_empty_gcp_token_is_refused(self) -> None:
        p = _provider()
        resp = uc.GenerateTemporaryTableCredentialResponse(
            url="gs://b/t", gcp_oauth_token=uc.GcpOauthToken(oauth_token=None)
        )
        with pytest.raises(CredentialError, match="no OAuth token"):
            p._to_credentials(resp, Operation.READ)

    def test_unrecognised_response_names_fields_without_crashing(self) -> None:
        p = _provider()
        with pytest.raises(CredentialError, match="no recognised credential block"):
            p._to_credentials(
                uc.GenerateTemporaryTableCredentialResponse(url="s3://b"), Operation.READ
            )
        with pytest.raises(CredentialError, match="no recognised credential block"):
            p._to_credentials({"url": "s3://b"}, Operation.READ)

    def test_vend_not_found_says_table_was_dropped(self, ws: FakeWorkspace) -> None:
        p = _provider()
        p._client = ws
        ws.temporary_table_credentials.responses["generate_temporary_table_credentials"] = (
            sdk_errors.NotFound("Table 'tid-1' does not exist.")
        )
        with pytest.raises(CredentialError, match="re-resolve"):
            p.credentials()

    def test_vend_unauthenticated_says_so(self, ws: FakeWorkspace) -> None:
        p = _provider()
        p._client = ws
        ws.temporary_table_credentials.responses["generate_temporary_table_credentials"] = (
            sdk_errors.Unauthenticated("Invalid access token.")
        )
        with pytest.raises(CredentialError, match="credentials were rejected"):
            p.credentials()

    def test_workspace_auth_does_not_leak_schemeless_token(self) -> None:
        p = _provider()
        p._client = SimpleNamespace(
            config=SimpleNamespace(
                host="https://h/", authenticate=lambda: {"Authorization": "dapiSECRET123"}
            )
        )
        with pytest.raises(CredentialError) as info:
            p.workspace_auth()
        assert "dapiSECRET123" not in str(info.value)

    def test_workspace_auth_header_case_insensitive(self) -> None:
        p = _provider()
        p._client = SimpleNamespace(
            config=SimpleNamespace(
                host="https://h/", authenticate=lambda: {"authorization": "Bearer abc"}
            )
        )
        assert p.workspace_auth() == ("https://h", "abc")

    def test_pickle_keeps_explicit_config_host(self) -> None:
        class FakeConfig:
            """A Config's picklable surface, plus live state that is not."""

            host = "https://adb-1.azuredatabricks.net"

            def __init__(self) -> None:
                self._lock = threading.Lock()

            def as_dict(self) -> dict[str, Any]:
                return {"host": self.host, "client_id": "sp", "auth_type": "oauth-m2m"}

        p = DatabricksCredentialProvider("tid-1", config=FakeConfig())  # type: ignore[arg-type]
        clone = pickle.loads(pickle.dumps(p))
        assert clone._explicit_config is None
        assert clone._config_kwargs["host"] == "https://adb-1.azuredatabricks.net"
        assert clone._config_kwargs["client_id"] == "sp"

    def test_pickled_provider_has_working_lock(self) -> None:
        clone = pickle.loads(pickle.dumps(_provider()))
        # Re-entrant: `credentials()` holds it while `_vend()` builds the client.
        assert clone._lock.acquire(timeout=1)
        try:
            assert clone._lock.acquire(timeout=1)
            clone._lock.release()
        finally:
            clone._lock.release()
        assert clone._vended_at == {}

    def test_workspace_client_built_once_under_concurrency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deltaswamp._sdk as sdk

        built: list[int] = []

        def slow_client(**kwargs: Any) -> Any:
            time.sleep(0.05)
            built.append(1)
            return SimpleNamespace()

        monkeypatch.setattr(sdk, "workspace_client", slow_client)
        p = _provider()
        threads = [threading.Thread(target=p._workspace) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(built) == 1


class TestAzureEndpoint:
    def test_container_only_url_has_no_endpoint(self) -> None:
        assert azure_endpoint_for("az://container/path") is None
        assert azure_endpoint_for("abfss://container/path") is None

    def test_account_url(self) -> None:
        assert (
            azure_endpoint_for("abfss://c@acct.dfs.core.windows.net/p")
            == "https://acct.blob.core.windows.net"
        )
        assert (
            azure_endpoint_for("https://acct.blob.core.windows.net/c/p")
            == "https://acct.blob.core.windows.net"
        )

    def test_http_emulator_keeps_http(self) -> None:
        assert azure_endpoint_for("http://127.0.0.1:10000/devstoreaccount1/c") == (
            "http://127.0.0.1:10000/devstoreaccount1"
        )


# ================================================================ catalog


class TestResolve:
    def test_permission_denied_without_code_in_text_names_privileges(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = sdk_errors.PermissionDenied(
            "User does not have SELECT on Table 'main.sales.orders'."
        )
        with pytest.raises(PreflightError, match="was denied"):
            dbx.resolve(REF)

    def test_unauthenticated_is_not_reported_as_missing_privileges(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = sdk_errors.Unauthenticated("Invalid access token.")
        with pytest.raises(PreflightError, match="credentials were rejected"):
            dbx.resolve(REF)
        assert not ws.called("grants.get_effective")

    def test_404_digits_in_a_permission_message_are_not_not_found(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = sdk_errors.PermissionDenied(
            "User does not have SELECT on Table 'main.sales.t404'."
        )
        with pytest.raises(PreflightError):
            dbx.resolve(parse_ref("main.sales.t404"))

    def test_table_name_with_hash_is_escaped_in_the_path(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = _info(name="a#b")
        ws.metastores.responses["summary"] = uc.GetMetastoreSummaryResponse(region="us-west-2")
        dbx.resolve(parse_ref("main.sales.`a#b`"))
        (call,) = ws.called("tables.get")
        assert call["full_name"] == "main.sales.a%23b"

    def test_url_name_leaves_plain_names_alone(self) -> None:
        assert _url_name("main.sales.orders") == "main.sales.orders"
        assert _url_name("main.sales.my-table") == "main.sales.my-table"
        assert _url_name("c.s.a%20b") == "c.s.a%2520b"

    def test_region_comes_from_summary_for_non_admins(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = _info()
        ws.metastores.responses["summary"] = uc.GetMetastoreSummaryResponse(region="eu-west-1")
        ws.metastores.responses["get"] = sdk_errors.PermissionDenied("metastore admin only")
        resolved = dbx.resolve(REF)
        assert resolved.credential_provider is not None
        assert resolved.credential_provider._region == "eu-west-1"  # type: ignore[attr-defined]

    def test_region_lookup_is_not_observed_half_done(self, ws: FakeWorkspace) -> None:
        catalog = DatabricksUnityCatalog(host="https://h", token="t")
        catalog._client = ws

        def slow_summary() -> Any:
            time.sleep(0.05)
            return uc.GetMetastoreSummaryResponse(region="ap-south-1")

        ws.metastores.responses["summary"] = slow_summary
        seen: list[str | None] = []
        threads = [
            threading.Thread(target=lambda: seen.append(catalog._metastore_region()))
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert seen == ["ap-south-1"] * 4
        assert len(ws.called("metastores.summary")) == 1

    def test_manifest_fallback_is_case_insensitive(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = _info(name="orders")
        ws.tables.responses["list"] = [
            _info(
                name="orders",
                securable_kind_manifest=uc.SecurableKindManifest(
                    capabilities=["HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT"]
                ),
            )
        ]
        resolved = dbx.resolve(parse_ref("Main.Sales.Orders"))
        assert resolved.external_read_supported is True

    def test_manifest_enum_capabilities_are_matched_by_value(self) -> None:
        import enum

        class Cap(enum.Enum):
            READ = "HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT"

        info = SimpleNamespace(securable_kind_manifest=SimpleNamespace(capabilities=[Cap.READ]))
        caps = DatabricksUnityCatalog._manifest_capabilities(info)
        assert caps == frozenset({"HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT"})

    def test_preview_catalog_owned_property_fetches_commit_tail(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = _info(
            table_type=uc.TableType.MANAGED,
            properties={"delta.feature.catalogOwned-preview": "supported"},
        )
        ws.api_client.responses["do"] = {"commits": [], "latest-table-version": 4}
        resolved = dbx.resolve(REF)
        assert resolved.max_catalog_version == 4

    def test_commit_tail_path_is_escaped(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = _info(
            name="a#b", properties={"delta.feature.catalogManaged": "supported"}
        )
        ws.api_client.responses["do"] = {"commits": [], "latest-table-version": 1}
        dbx.resolve(parse_ref("main.sales.`a#b`"))
        (call,) = ws.called("api_client.do")
        assert call["path"].endswith("/tables/a%23b")

    def test_empty_commit_tail_body_is_a_clear_error(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = _info(properties={"delta.feature.catalogManaged": "supported"})
        ws.api_client.responses["do"] = None
        with pytest.raises(PreflightError, match="commit tail"):
            dbx.resolve(REF)

    def test_r2_location_is_rewritten_to_s3(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = _info(
            storage_location="r2://bkt@acct.r2.cloudflarestorage.com/tables/t"
        )
        resolved = dbx.resolve(REF)
        assert resolved.location == "s3://bkt/tables/t"

    def test_list_tables_carries_provider_and_properties(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["list"] = [
            _info(properties={"delta.feature.catalogManaged": "supported"}),
            _info(name=None, table_id="x"),
        ]
        (listed,) = dbx.list_tables("main", "sales")
        assert listed.credential_provider is not None
        assert listed.is_catalog_managed

    def test_path_credentials_carry_metastore_region(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        ws.metastores.responses["summary"] = uc.GetMetastoreSummaryResponse(region="us-west-2")
        ws.temporary_path_credentials.responses["generate_temporary_path_credentials"] = (
            uc.GenerateTemporaryPathCredentialResponse(
                url="s3://bucket/new",
                aws_temp_credentials=uc.AwsCredentials(
                    access_key_id="AK", secret_access_key="SK", session_token="ST"
                ),
            )
        )
        creds = dbx.path_credentials("s3://bucket/new", "PATH_CREATE_TABLE")
        assert creds.secrets["aws_region"] == "us-west-2"


class TestGovernanceErrors:
    def test_create_schema_denial_looks_up_the_catalog(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.schemas.responses["create"] = sdk_errors.PermissionDenied(
            "User does not have CREATE SCHEMA on Catalog 'main'."
        )
        ws.grants.responses["get_effective"] = uc.EffectivePermissionsList(privilege_assignments=[])
        with pytest.raises(PreflightError, match="Effective privileges on main:"):
            dbx.create_schema("main", "new")
        (call,) = ws.called("grants.get_effective")
        assert call["full_name"] == "main"

    def test_unauthenticated_governance_call(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.grants.responses["get"] = sdk_errors.Unauthenticated("bad token")
        with pytest.raises(PreflightError, match="credentials were rejected"):
            dbx.grants(REF)
        assert not ws.called("grants.get_effective")

    def test_invalid_volume_type_is_a_clear_error(self, dbx: DatabricksUnityCatalog) -> None:
        with pytest.raises(InvalidArgumentError, match="volume_type"):
            dbx.create_volume("main", "sales", "v", volume_type="SHARED")

    def test_drop_table_invalidates_manifest_cache(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        dbx._manifest_cache[("main", "sales")] = {"orders": frozenset({"X"})}
        dbx.drop_table(REF)
        assert ("main", "sales") not in dbx._manifest_cache

    def test_drop_catalog_invalidates_its_schemas(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        dbx._manifest_cache[("main", "sales")] = {}
        dbx._manifest_cache[("other", "sales")] = {}
        dbx.drop_catalog("Main")
        assert ("main", "sales") not in dbx._manifest_cache
        assert ("other", "sales") in dbx._manifest_cache

    def test_tag_key_escaped_in_delete_path(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        dbx.unset_tags(REF, ["pii#level"])
        (call,) = ws.called("entity_tag_assignments.delete")
        assert call["tag_key"] == "pii%23level"

    def test_catalog_is_picklable(self, dbx: DatabricksUnityCatalog) -> None:
        clone = pickle.loads(pickle.dumps(dbx))
        assert clone._client is None
        with clone._lock:
            pass


class TestMoreCatalog:
    def test_row_filter_without_manifest_is_not_vendable(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = _info(
            row_filter=uc.TableRowFilter(function_name="f", input_column_names=["a"])
        )
        ws.tables.responses["list"] = sdk_errors.PermissionDenied("no list")
        resolved = dbx.resolve(REF)
        assert resolved.external_read_supported is False
        assert resolved.external_write_supported is False

    def test_column_mask_without_manifest_is_not_vendable(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["get"] = _info(
            columns=[uc.ColumnInfo(name="ssn", mask=uc.ColumnMask(function_name="m"))]
        )
        ws.tables.responses["list"] = []
        assert dbx.resolve(REF).external_read_supported is False

    def test_manifest_cache_expires(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deltaswamp.catalog.databricks as mod

        ws.tables.responses["get"] = _info()
        ws.tables.responses["list"] = []
        dbx.resolve(REF)
        dbx.resolve(REF)
        assert len(ws.called("tables.list")) == 1
        now = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: now + mod.MANIFEST_CACHE_TTL_SECONDS + 1)
        dbx.resolve(REF)
        assert len(ws.called("tables.list")) == 2

    def test_set_tags_sends_values_as_text(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.entity_tag_assignments.responses["list"] = []
        dbx.set_tags(REF, {"tier": 0, "flag": ""})  # type: ignore[dict-item]
        values = {
            c["tag_assignment"].tag_key: c["tag_assignment"].tag_value
            for c in ws.called("entity_tag_assignments.create")
        }
        assert values == {"tier": "0", "flag": None}

    def test_unset_absent_tag_is_a_noop(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.entity_tag_assignments.responses["delete"] = sdk_errors.NotFound("tag not found")
        ws.tables.responses["exists"] = uc.TableExistsResponse(table_exists=True)
        dbx.unset_tags(REF, ["missing"])

    def test_unset_tag_on_missing_table_still_raises(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.entity_tag_assignments.responses["delete"] = sdk_errors.NotFound("no table")
        ws.tables.responses["exists"] = uc.TableExistsResponse(table_exists=False)
        with pytest.raises(InvalidReferenceError):
            dbx.unset_tags(REF, ["k"])

    def test_invalid_volume_type_is_a_value_error(self, dbx: DatabricksUnityCatalog) -> None:
        with pytest.raises(ValueError, match="volume_type"):
            dbx.create_volume("main", "sales", "v", volume_type="SHARED")


class TestRefreshAndListing:
    def test_failed_refresh_ahead_serves_still_valid_credential(self) -> None:
        p = CountingProvider([_aws(int((time.time() + 3600) * 1000))])
        first = p.credentials()
        object.__setattr__(first, "expires_at", time.time() + 60)  # inside the margin

        def boom(operation: Operation) -> Any:
            raise CredentialError("503 temporarily unavailable")

        p._vend = boom  # type: ignore[method-assign]
        assert p.credentials() is first

    def test_failed_refresh_of_expired_credential_raises(self) -> None:
        p = CountingProvider([_aws(int((time.time() + 3600) * 1000))])
        first = p.credentials()
        object.__setattr__(first, "expires_at", time.time() - 1)

        def boom(operation: Operation) -> Any:
            raise CredentialError("503")

        p._vend = boom  # type: ignore[method-assign]
        with pytest.raises(CredentialError):
            p.credentials()

    def test_list_tables_denial_is_classified(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["list"] = sdk_errors.PermissionDenied("User does not have USE SCHEMA")
        with pytest.raises(PreflightError, match="denied"):
            dbx.list_tables("main", "sales")

    def test_list_schemas_missing_catalog_is_classified(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.schemas.responses["list"] = sdk_errors.NotFound("Catalog 'nope' does not exist.")
        with pytest.raises(InvalidReferenceError):
            dbx.list_schemas("nope")

    def test_list_catalogs_unauthenticated_is_classified(
        self, dbx: DatabricksUnityCatalog, ws: FakeWorkspace
    ) -> None:
        ws.catalogs.responses["list"] = sdk_errors.Unauthenticated("bad token")
        with pytest.raises(PreflightError, match="credentials were rejected"):
            dbx.list_catalogs()
