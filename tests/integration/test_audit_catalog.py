"""Regressions for the catalog audit (OC, LC and DC findings).

Each test failed before its fix. The live counterparts were checked against a
Databricks workspace; these reproduce them offline.
"""

from __future__ import annotations

import io
import logging
import pickle
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("deltaswamp")

from deltaswamp import Connection  # noqa: E402
from deltaswamp.catalog.base import ResolvedTable  # noqa: E402
from deltaswamp.catalog.filesystem import FilesystemCatalog  # noqa: E402
from deltaswamp.catalog.glue import GlueCatalog  # noqa: E402
from deltaswamp.catalog.hms import HiveMetastoreCatalog  # noqa: E402
from deltaswamp.credentials.base import Cloud, Credentials, StaticCredentialProvider  # noqa: E402
from deltaswamp.errors import (  # noqa: E402
    CredentialError,
    DeltaSwampError,
    InvalidArgumentError,
    InvalidReferenceError,
    PreflightError,
    UnreachableTableError,
)
from deltaswamp.identity import parse_ref  # noqa: E402

from tests.fake_uc import FakeTable, FakeUnityCatalog  # noqa: E402
from tests.helpers import direct_router  # noqa: E402


def _path_table(tmp_path: Any, name: str, who: str) -> str:
    path = str(tmp_path / name)
    fs = ds.connect("file://")
    fs.create_table(path, pa.schema([("who", pa.string())])).append(pa.table({"who": [who]}))
    return path


# --------------------------------------------------------------------- OC-1


class TestSchemeQualifiedReferences:
    """glue:// and hms:// names went to whatever catalog the connection had."""

    @pytest.fixture
    def uc(self, tmp_path: Any) -> Any:
        uc_path = _path_table(tmp_path, "uc_glue", "UC catalog 'glue'")
        glue_path = _path_table(tmp_path, "real_glue", "Glue")
        with FakeUnityCatalog() as server:
            server.add_schema("glue", "crm")
            server.add_schema("hive_metastore", "crm")
            server.add_table(
                "glue.crm.customers",
                FakeTable("customers", f"file://{uc_path}", str(uuid.uuid4())),
            )
            server.add_table(
                "hive_metastore.crm.customers",
                FakeTable("customers", f"file://{uc_path}", str(uuid.uuid4())),
            )
            yield SimpleNamespace(server=server, glue_path=glue_path)

    @staticmethod
    def _fake_glue(monkeypatch: Any, glue_path: str) -> list[Any]:
        seen: list[Any] = []

        def resolve(self: Any, ref: Any) -> ResolvedTable:
            seen.append((self, ref))
            return ResolvedTable(ref=ref, location=glue_path)

        monkeypatch.setattr(GlueCatalog, "resolve", resolve)
        return seen

    def test_glue_ref_on_a_uc_connection_reads_glue(self, uc: Any, monkeypatch: Any) -> None:
        seen = self._fake_glue(monkeypatch, uc.glue_path)
        conn = ds.connect(f"uc://{uc.server.url}")
        t = conn.table("glue://crm.customers")
        assert t.to_arrow().column("who").to_pylist() == ["Glue"]
        # One Glue catalog per connection, reused.
        conn.table("glue://crm.customers")
        assert seen[0][0] is seen[1][0] and isinstance(seen[0][0], GlueCatalog)

    def test_documented_cross_catalog_sql(self, uc: Any, monkeypatch: Any) -> None:
        self._fake_glue(monkeypatch, uc.glue_path)
        conn = ds.connect(f"uc://{uc.server.url}")
        got = conn.sql("SELECT who FROM c", tables={"c": "glue://crm.customers"})
        assert got.column("who").to_pylist() == ["Glue"]

    def test_glue_catalog_id_reaches_glue(self, uc: Any, monkeypatch: Any) -> None:
        seen = self._fake_glue(monkeypatch, uc.glue_path)
        conn = ds.connect(f"uc://{uc.server.url}")
        conn.table("glue://123456789012/crm.customers")
        catalog, ref = seen[0]
        assert catalog._catalog_id == "123456789012" and ref.endpoint == "123456789012"

    def test_hms_endpoint_is_honoured(self, uc: Any, monkeypatch: Any) -> None:
        seen: list[Any] = []

        def resolve(self: Any, ref: Any) -> ResolvedTable:
            seen.append((self._host, self._port))
            return ResolvedTable(ref=ref, location=uc.glue_path)

        monkeypatch.setattr(HiveMetastoreCatalog, "resolve", resolve)
        conn = ds.connect(f"uc://{uc.server.url}")
        conn.table("hms://thrift://othermetastore:9083/crm/customers")
        assert seen == [("othermetastore", 9083)]

    def test_hms_without_endpoint_is_refused_off_hms(self, uc: Any) -> None:
        conn = ds.connect(f"uc://{uc.server.url}")
        with pytest.raises(InvalidReferenceError, match="names no metastore endpoint"):
            conn.table("hms:///crm/customers")

    def test_create_of_a_glue_name_is_not_made_in_uc(self, uc: Any) -> None:
        conn = ds.connect(f"uc://{uc.server.url}")
        with pytest.raises(UnreachableTableError, match="glue catalog cannot register"):
            conn.create_table("glue://crm.newt", pa.schema([("id", pa.int64())]))
        assert "glue.crm.newt" not in uc.server.tables

    def test_filesystem_connection_refuses_uc(self) -> None:
        with pytest.raises(InvalidReferenceError, match="Unity Catalog table"):
            ds.connect("file://").table("uc://main.sales.orders")

    def test_glue_connection_refuses_uc_and_keeps_bare_names(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        path = _path_table(tmp_path, "g", "Glue")
        self._fake_glue(monkeypatch, path)
        conn = Connection(catalog=GlueCatalog(region_name="us-east-1"), router=direct_router())
        with pytest.raises(InvalidReferenceError, match="glue catalog"):
            conn.table("uc://main.sales.orders")
        assert conn.table("main.sales.orders").to_arrow().num_rows == 1

    def test_sharing_connection_refuses_uc(self) -> None:
        from deltaswamp.catalog.sharing import SharingCatalog

        profile = {"shareCredentialsVersion": 1, "endpoint": "https://x/", "bearerToken": "t"}
        conn = Connection(catalog=SharingCatalog(profile), router=direct_router())
        with pytest.raises(InvalidReferenceError, match="Unity Catalog table"):
            conn.table("uc://a.b.c")


# ------------------------------------------------------------------ LC-10/12


class TestConnectionArguments:
    def test_write_table_bare_name_with_defaults(self, monkeypatch: Any) -> None:
        # LC-10: the create-if-absent path re-parsed the name without defaults.
        conn = Connection(
            catalog=FilesystemCatalog(),
            router=direct_router(),
            default_catalog="main",
            default_schema="sales",
        )
        created: list[str] = []
        monkeypatch.setattr(conn, "table_exists", lambda name: False)

        def create_table(name: str, schema: Any, **_: Any) -> Any:
            created.append(name)
            raise UnreachableTableError("create", "stop here")

        monkeypatch.setattr(conn, "create_table", create_table)
        with pytest.raises(UnreachableTableError, match="stop here"):
            conn.write_table("orders", pa.table({"id": [1]}))
        assert created == ["orders"]

    @pytest.mark.parametrize("value", ["false", "0", 1, None])
    def test_allow_sql_fallback_must_be_a_bool(self, value: Any) -> None:
        # OC-9: "false" is truthy and turned on the paid fallback.
        with pytest.raises(InvalidArgumentError, match="allow_sql_fallback"):
            ds.connect("file://", allow_sql_fallback=value)

    def test_catalog_create_mode_is_validated(self) -> None:
        with pytest.raises(InvalidArgumentError, match="create mode"):
            ds.connect("file://").create_table(
                "main.s.t", pa.schema([("id", pa.int64())]), mode="bogus"
            )

    def test_zero_column_schema_is_refused(self, tmp_path: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="at least one column"):
            ds.connect("file://").create_table(str(tmp_path / "t"), pa.schema([]))

    def test_uc_slash_form_is_refused(self) -> None:
        with pytest.raises(InvalidReferenceError, match="dotted"):
            parse_ref("uc://main/sales/orders", default_catalog="c", default_schema="s")

    def test_bare_volume_name_says_volume(self) -> None:
        conn = ds.connect("file://")
        with pytest.raises(InvalidReferenceError, match="a volume is named"):
            conn.volume("landing")


class TestManagedCreateFallback:
    """LC-11: the warehouse create made a table that was not catalog-managed."""

    def test_fallback_create_is_catalog_managed(self, monkeypatch: Any) -> None:
        from deltaswamp.capability import Engine

        calls: list[dict[str, Any]] = []

        class Sql:
            def create_managed(self, name: str, schema: Any, **kwargs: Any) -> None:
                calls.append(kwargs)

        class Lifecycle:
            name = "lc"

            def create_staging_table(self, ref: Any) -> Any:
                raise PreflightError("User-Agent insufficient")

        router = direct_router()
        router.engines[Engine.SQL] = Sql()
        router.allow_sql_fallback = True
        conn = Connection(catalog=FilesystemCatalog(), router=router)
        monkeypatch.setattr(conn, "table", lambda name: name)
        ref = parse_ref("main.s.t")
        schema = pa.schema([("id", pa.int64())])
        conn._create_managed(Lifecycle(), ref, schema, None, None, {"a": "1"}, None)
        assert calls[0]["properties"] == {"a": "1", "delta.feature.catalogManaged": "supported"}
        # The caller's own choice stands.
        props = {"delta.feature.catalogManaged": "disabled"}
        conn._create_managed(Lifecycle(), ref, schema, None, None, props, None)
        assert calls[1]["properties"] == props


# --------------------------------------------------------------------- DC-2


class _FakeConfig:
    host = "https://fx-audit.cloud.databricks.com"

    def as_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "token": "dapi-x",
            "auth_type": "pat",
            # A resolved Config reports it; WorkspaceClient() does not take it.
            "discovery_url": "https://fx-audit.cloud.databricks.com/oidc",
        }


@pytest.fixture
def offline_sdk(monkeypatch: Any) -> None:
    pytest.importorskip("databricks.sdk")
    from databricks.sdk.config import Config
    from deltaswamp import _sdk

    monkeypatch.setattr(Config, "_resolve_host_metadata", lambda self: None)
    monkeypatch.setattr(_sdk, "_check_resolvable", lambda host: None)


class TestPickledProviderBuildsAClient:
    def test_unpickled_provider_builds_the_client(self, offline_sdk: None) -> None:
        from deltaswamp.credentials.databricks import DatabricksCredentialProvider

        provider = DatabricksCredentialProvider("tid-1", config=_FakeConfig())  # type: ignore[arg-type, unused-ignore]
        clone = pickle.loads(pickle.dumps(provider))
        client = clone._workspace()  # TypeError: multiple values for 'host'
        assert client.config.host == _FakeConfig.host
        assert client.config.token == "dapi-x"

    def test_unpickled_catalog_builds_the_client(self, offline_sdk: None) -> None:
        from deltaswamp.catalog.databricks import DatabricksUnityCatalog

        catalog = DatabricksUnityCatalog(config=_FakeConfig())  # type: ignore[arg-type, unused-ignore]
        clone = pickle.loads(pickle.dumps(catalog))
        assert clone.workspace.config.host == _FakeConfig.host

    def test_auth_failure_is_not_sent_to_the_fallback(self) -> None:
        from deltaswamp.router import _open_error_remedy

        table = ResolvedTable(
            ref=parse_ref("main.s.t"),
            location="s3://b/t",
            open_error="CredentialError: credential vending failed for table_id=x (READ): "
            "could not configure Databricks authentication",
        )
        assert "allow_sql_fallback" not in _open_error_remedy(table)


# ------------------------------------------------------------------ DC-5 / LC-14


def _creds() -> Credentials:
    return Credentials(
        cloud=Cloud.AWS,
        url="s3://b/t",
        expires_at=4e9,
        secrets={"aws_secret_access_key": "SECRET"},
        table_id="t1",
    )


class TestCredentialsDoNotPickle:
    def test_credentials_refuse(self) -> None:
        with pytest.raises(TypeError, match="not picklable"):
            pickle.dumps(_creds())

    def test_deliberate_carriers_still_travel(self) -> None:
        from deltaswamp.distributed import ShippedCredentials

        shipped = pickle.loads(pickle.dumps(ShippedCredentials(_creds(), "t1")))
        assert shipped.credentials().secrets == {"aws_secret_access_key": "SECRET"}
        static = pickle.loads(pickle.dumps(StaticCredentialProvider(_creds())))
        assert static.peek().table_id == "t1"


# --------------------------------------------------------------------- LC-05


class _SdkError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.error_code = code


class NotFound(_SdkError):
    pass


class InvalidParameterValue(_SdkError):
    pass


class PermissionDenied(_SdkError):
    pass


@pytest.fixture
def dbx() -> Any:
    from deltaswamp.catalog.databricks import DatabricksUnityCatalog

    catalog = DatabricksUnityCatalog(host="https://h")
    catalog._held_privileges = lambda *a: None  # type: ignore[method-assign]
    return catalog


class TestUnityErrorClassification:
    def test_missing_principal(self, dbx: Any) -> None:
        exc = NotFound("Could not find principal with name x@y", "PRINCIPAL_DOES_NOT_EXIST")
        err = dbx._error(exc, "grant privileges", "c.s.t")
        assert isinstance(err, InvalidArgumentError) and "principal" in str(err)
        assert "does not exist in Unity Catalog" not in str(err)

    def test_missing_constraint_or_file(self, dbx: Any) -> None:
        for message in (
            "Constraint fx_nope does not exist.",
            "The file being accessed is not found.",
        ):
            err = dbx._error(NotFound(message, "NOT_FOUND"), "act", "c.s.v")
            assert isinstance(err, InvalidReferenceError)
            assert "c.s.v does not exist" not in str(err) and message in str(err)

    def test_missing_securable_still_names_it(self, dbx: Any) -> None:
        err = dbx._error(NotFound("Volume 'c.s.v' does not exist.", "NOT_FOUND"), "act", "c.s.v")
        assert "c.s.v does not exist in Unity Catalog" in str(err)

    def test_invalid_parameter_is_an_argument_error(self, dbx: Any) -> None:
        exc = InvalidParameterValue(
            "Invalid input: RPC GetPermissions ... TABLEZ is not a valid securable type",
            "INVALID_PARAMETER_VALUE",
        )
        err = dbx._error(exc, "read grants", "c.s.t")
        assert isinstance(err, InvalidArgumentError) and "denied" not in str(err)

    def test_bad_token_is_unauthenticated(self, dbx: Any) -> None:
        # LC-08: Databricks answers a bad PAT with 403 PermissionDenied.
        err = dbx._error(PermissionDenied("Invalid access token.", "403"), "act", "c.s.t")
        assert "credentials were rejected" in str(err)


# --------------------------------------------------------------------- LC-06


class TestVolumePaths:
    @pytest.fixture
    def vol(self) -> Any:
        from deltaswamp.governance import Volume

        files = SimpleNamespace(upload=lambda **kw: None)
        return Volume("c.s.v", files)

    def test_other_volume_is_refused(self, vol: Any) -> None:
        with pytest.raises(InvalidReferenceError, match="outside volume"):
            vol.write("/Volumes/c/s/other/x.txt", b"x")
        assert vol.path("/Volumes/c/s/v/x.txt") == "/Volumes/c/s/v/x.txt"

    def test_trailing_slash_write_is_refused(self, vol: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="directory"):
            vol.write("trailing/", b"x")

    def test_text_is_refused(self, vol: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="bytes"):
            vol.write("s.txt", "text")


# --------------------------------------------------------------------- LC-07


class TestWarehouseErrors:
    def test_sdk_errors_are_wrapped(self) -> None:
        from deltaswamp.engine.sql_backend import sdk_error

        missing = sdk_error(NotFound("The warehouse w was not found.", "NOT_FOUND"), "warehouse w")
        assert isinstance(missing, UnreachableTableError)
        token = sdk_error(PermissionDenied("Invalid access token.", "403"), "warehouse w")
        assert isinstance(token, CredentialError)
        assert isinstance(sdk_error(RuntimeError("boom"), "warehouse w"), DeltaSwampError)

    def test_nonexistent_warehouse_is_refused_by_can(self) -> None:
        from deltaswamp.engine.sql import SqlEngine

        class Warehouses:
            def get(self, warehouse_id: str) -> Any:
                raise NotFound(
                    f"SQL warehouse {warehouse_id} does not exist.", "RESOURCE_DOES_NOT_EXIST"
                )

        engine = SqlEngine(warehouse_id="nope", client=SimpleNamespace(warehouses=Warehouses()))
        assert engine.warehouse_id is None
        assert "does not exist" in (engine._selection_error or "")


# --------------------------------------------------------------------- LC-08/09


class TestSdkClient:
    def test_unresolvable_host_fails_fast(self) -> None:
        from deltaswamp import _sdk

        with pytest.raises(PreflightError, match="does not resolve"):
            _sdk._check_resolvable("https://fx-nonexistent-xyz.invalid")

    def test_vended_secrets_are_redacted_in_sdk_debug_logs(self) -> None:
        from deltaswamp import _sdk

        _sdk._install_log_redaction()
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        logger = logging.getLogger("databricks.sdk")
        old = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            logger.debug(
                '< {\n<   "aws_temp_credentials": {\n<     "secret_access_key": "AB%s"', "CD"
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old)
        assert "ABCD" not in buf.getvalue() and "**REDACTED**" in buf.getvalue()


# --------------------------------------------------------------------- OC-2/3


class TestDatabricksManagedIceberg:
    """UC reports USING ICEBERG tables as DELTA + icebergWriterCompatV1 + catalogManaged."""

    @staticmethod
    def _table() -> ResolvedTable:
        return ResolvedTable(
            ref=parse_ref("main.s.ice"),
            location="s3://b/ice",
            data_source_format="DELTA",
            properties={
                "delta.enableIcebergWriterCompatV1": "true",
                "delta.feature.catalogManaged": "supported",
                "delta.universalFormat.enabledFormats": "iceberg",
            },
            writer_features=frozenset({"icebergCompatV2", "catalogManaged"}),
            iceberg_rest_uri="https://h/api/2.1/unity-catalog/iceberg-rest",
        )

    def test_appends_go_through_iceberg_rest(self) -> None:
        pytest.importorskip("pyiceberg")
        from deltaswamp.capability import Operation
        from deltaswamp.engine.iceberg import IcebergEngine

        table = self._table()
        assert table.is_managed_iceberg and not table.is_iceberg
        engine = IcebergEngine()
        assert engine.supports(Operation.APPEND, table).ok
        # Databricks' endpoint refuses the two snapshots an overwrite commits.
        assert not engine.supports(Operation.OVERWRITE, table).ok
        history = engine.supports(Operation.HISTORY, table)
        assert not history.ok and "Delta versions" in history.reason
