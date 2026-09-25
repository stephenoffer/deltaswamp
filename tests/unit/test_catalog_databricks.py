"""The Databricks Unity Catalog client, against a stub `WorkspaceClient`.

The stub records every call and answers with real SDK dataclasses, so request
shapes are checked against the SDK's own types and responses go through the
same `as_dict()` path a live answer would.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

pytest.importorskip("databricks.sdk")

from databricks.sdk.service import catalog as uc
from databricks.sdk.service import files as uc_files
from deltaswamp.catalog.databricks import DatabricksUnityCatalog
from deltaswamp.errors import InvalidReferenceError, PreflightError
from deltaswamp.governance import Grant, Lineage, StagingTable, TableInfo
from deltaswamp.identity import parse_ref

from tests.helpers import DELTA_SCHEMA

REF = parse_ref("main.sales.orders")


class NotFound(Exception):
    """Named like the SDK's, which is what classification keys on."""


class PermissionDenied(Exception):
    pass


class Api:
    """One SDK service: records calls, answers from `responses`.

    A response may be a value, an exception (raised), or a callable taking the
    call's kwargs.
    """

    def __init__(self, name: str, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self._name = name
        self._calls = calls
        self.responses: dict[str, Any] = {}

    def __getattr__(self, method: str) -> Callable[..., Any]:
        if method.startswith("_"):
            raise AttributeError(method)

        def call(*args: Any, **kwargs: Any) -> Any:
            if args:  # api_client.do(method, path, ...)
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
    """Just the WorkspaceClient services the catalog touches."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tables = Api("tables", self.calls)
        self.grants = Api("grants", self.calls)
        self.entity_tag_assignments = Api("entity_tag_assignments", self.calls)
        self.table_constraints = Api("table_constraints", self.calls)
        self.volumes = Api("volumes", self.calls)
        self.files = Api("files", self.calls)
        self.functions = Api("functions", self.calls)
        self.schemas = Api("schemas", self.calls)
        self.catalogs = Api("catalogs", self.calls)
        self.temporary_path_credentials = Api("temporary_path_credentials", self.calls)
        self.api_client = Api("api_client", self.calls)

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


def _table_info(**overrides: Any) -> uc.TableInfo:
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


class TestTableInfo:
    def test_request_asks_for_everything(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.tables.responses["get"] = _table_info()
        dbx.table_info(REF)
        (call,) = ws.called("tables.get")
        assert call == {
            "full_name": "main.sales.orders",
            "include_browse": True,
            "include_delta_metadata": True,
            "include_manifest_capabilities": True,
        }

    def test_normalises_the_full_record(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.tables.responses["get"] = _table_info(
            owner="alice@example.com",
            comment="all orders",
            created_at=1700000000000,
            created_by="alice@example.com",
            updated_at=1700000001000,
            updated_by="bob@example.com",
            properties={"delta.appendOnly": "false"},
            delta_runtime_properties_kvpairs=uc.DeltaRuntimePropertiesKvPairs(
                delta_runtime_properties={"delta.enableDeletionVectors": "true"}
            ),
            columns=[
                uc.ColumnInfo(
                    name="id",
                    type_text="bigint",
                    type_name=uc.ColumnTypeName.LONG,
                    nullable=False,
                    position=0,
                    comment="key",
                ),
                uc.ColumnInfo(
                    name="region",
                    type_text="string",
                    type_name=uc.ColumnTypeName.STRING,
                    nullable=True,
                    position=1,
                    partition_index=0,
                ),
                uc.ColumnInfo(
                    name="ssn",
                    type_text="string",
                    type_name=uc.ColumnTypeName.STRING,
                    position=2,
                    mask=uc.ColumnMask(
                        function_name="main.sec.mask_ssn", using_column_names=["region"]
                    ),
                ),
            ],
            row_filter=uc.TableRowFilter(
                function_name="main.sec.region_filter", input_column_names=["region"]
            ),
            effective_predictive_optimization_flag=uc.EffectivePredictiveOptimizationFlag(
                value=uc.EnablePredictiveOptimization.ENABLE,
                inherited_from_type=uc.EffectivePredictiveOptimizationFlagInheritedFromType.CATALOG,
                inherited_from_name="main",
            ),
            securable_kind_manifest=uc.SecurableKindManifest(
                securable_kind=uc.SecurableKind.TABLE_DELTA_EXTERNAL,
                capabilities=["HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT"],
            ),
            table_constraints=[
                uc.TableConstraint(
                    primary_key_constraint=uc.PrimaryKeyConstraint(
                        name="pk", child_columns=["id"], rely=True
                    )
                ),
                uc.TableConstraint(
                    foreign_key_constraint=uc.ForeignKeyConstraint(
                        name="fk",
                        child_columns=["region"],
                        parent_table="main.ref.regions",
                        parent_columns=["code"],
                    )
                ),
            ],
            pipeline_id="pipe-1",
            browse_only=False,
        )

        info = dbx.table_info(REF)

        assert isinstance(info, TableInfo)
        assert info.full_name == "main.sales.orders"
        assert (info.owner, info.comment, info.table_id) == (
            "alice@example.com",
            "all orders",
            "tid-1",
        )
        assert (info.table_type, info.data_source_format) == ("EXTERNAL", "DELTA")
        assert (info.created_at, info.updated_by) == (1700000000000, "bob@example.com")
        assert [c.name for c in info.columns] == ["id", "region", "ssn"]
        assert info.columns[0].nullable is False and info.columns[0].type_name == "LONG"
        assert info.partition_columns == ("region",)
        assert info.properties == {"delta.appendOnly": "false"}
        assert info.runtime_properties == {"delta.enableDeletionVectors": "true"}
        assert info.row_filter is not None
        assert info.row_filter.function_name == "main.sec.region_filter"
        assert info.row_filter.input_column_names == ("region",)
        assert info.column_masks["ssn"].function_name == "main.sec.mask_ssn"
        assert info.has_row_filter_or_mask
        assert info.predictive_optimization == "ENABLE"
        assert info.predictive_optimization_inherited_from == "CATALOG:main"
        assert info.securable_kind == "TABLE_DELTA_EXTERNAL"
        assert "HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT" in info.capabilities
        assert [(c.kind, c.name) for c in info.constraints] == [
            ("PRIMARY_KEY", "pk"),
            ("FOREIGN_KEY", "fk"),
        ]
        assert info.constraints[1].parent_table == "main.ref.regions"
        assert info.pipeline_id == "pipe-1"
        assert info.browse_only is False

    def test_denied_names_the_needed_and_held_privileges(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.tables.responses["get"] = PermissionDenied("403 PERMISSION_DENIED")
        ws.grants.responses["get_effective"] = uc.EffectivePermissionsList(
            privilege_assignments=[
                uc.EffectivePrivilegeAssignment(
                    principal="me",
                    privileges=[uc.EffectivePrivilege(privilege=uc.Privilege.USE_SCHEMA)],
                )
            ]
        )
        with pytest.raises(PreflightError) as excinfo:
            dbx.table_info(REF)
        message = str(excinfo.value)
        assert "SELECT" in message and "USE_SCHEMA" in message
        # The enum is rendered by value, never as "Privilege.USE_SCHEMA".
        assert "Privilege." not in message

    def test_missing_is_an_invalid_reference(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.tables.responses["get"] = NotFound("Table does not exist")
        with pytest.raises(InvalidReferenceError, match="does not exist"):
            dbx.table_info(REF)

    def test_resolve_denial_matches_held_privileges(self, dbx: Any, ws: FakeWorkspace) -> None:
        """Held privileges are compared by enum value, not by `str(enum)`."""
        ws.tables.responses["get"] = PermissionDenied("403")
        ws.grants.responses["get_effective"] = uc.EffectivePermissionsList(
            privilege_assignments=[
                uc.EffectivePrivilegeAssignment(
                    principal="me",
                    privileges=[
                        uc.EffectivePrivilege(privilege=uc.Privilege.SELECT),
                        uc.EffectivePrivilege(privilege=uc.Privilege.EXTERNAL_USE_SCHEMA),
                    ],
                )
            ]
        )
        with pytest.raises(PreflightError, match="metastore-level"):
            dbx.resolve(REF)

    def test_table_uuid_is_not_the_uc_table_id(self, dbx: Any, ws: FakeWorkspace) -> None:
        """UC's table_id names the securable, not the Delta log's Metadata.id."""
        ws.tables.responses["get"] = _table_info(
            table_id="6cf3e2d1-b087-4b25-b3fc-10615078d4a5",
            table_type=uc.TableType.MANAGED,
        )
        resolved = dbx.resolve(REF)
        assert resolved.table_id == "6cf3e2d1-b087-4b25-b3fc-10615078d4a5"
        assert resolved.table_uuid is None


class TestUnquotedRestNames:
    """Backticks are SQL syntax; REST paths carry the bare name."""

    def test_lookup(self, dbx: Any, ws: FakeWorkspace) -> None:
        ref = parse_ref("main.sales.`odd-name`")
        ws.tables.responses["get"] = _table_info(name="odd-name", full_name="main.sales.odd-name")
        dbx.resolve(ref)
        assert ws.called("tables.get")[0]["full_name"] == "main.sales.odd-name"

    def test_drop(self, dbx: Any, ws: FakeWorkspace) -> None:
        dbx.drop_table(parse_ref("main.sales.`odd-name`"))
        assert ws.called("tables.delete")[0]["full_name"] == "main.sales.odd-name"


class TestAccessPolicy:
    """Row filters and column masks are read from the table info and named."""

    @staticmethod
    def info(**kwargs: Any) -> Any:
        kwargs.setdefault("columns", [])
        return SimpleNamespace(**kwargs)

    def test_row_filter_is_detected(self) -> None:
        info = self.info(row_filter=SimpleNamespace(function_name="main.s.rf"))
        assert DatabricksUnityCatalog._access_policy(info) == "row filter main.s.rf"

    def test_column_masks_are_detected(self) -> None:
        column = SimpleNamespace(name="city", mask=SimpleNamespace(function_name="main.s.m"))
        info = self.info(row_filter=None, columns=[column, SimpleNamespace(name="id", mask=None)])
        assert DatabricksUnityCatalog._access_policy(info) == "column mask on city (main.s.m)"

    def test_plain_table_has_no_policy(self) -> None:
        assert DatabricksUnityCatalog._access_policy(self.info(row_filter=None)) is None


class TestDatabricksGrants:
    def test_grants_pages_through_and_normalises(self, dbx: Any, ws: FakeWorkspace) -> None:
        pages = iter(
            [
                uc.GetPermissionsResponse(
                    privilege_assignments=[
                        uc.PrivilegeAssignment(
                            principal="analysts",
                            privileges=[uc.Privilege.SELECT, uc.Privilege.MODIFY],
                        )
                    ],
                    next_page_token="p2",
                ),
                uc.GetPermissionsResponse(
                    privilege_assignments=[
                        uc.PrivilegeAssignment(principal="etl", privileges=[uc.Privilege.MODIFY])
                    ]
                ),
            ]
        )
        ws.grants.responses["get"] = lambda **_: next(pages)

        grants = dbx.grants(REF)

        assert grants == [
            Grant("analysts", ("MODIFY", "SELECT")),
            Grant("etl", ("MODIFY",)),
        ]
        first, second = ws.called("grants.get")
        assert first == {"securable_type": "TABLE", "full_name": "main.sales.orders"}
        assert second["page_token"] == "p2"

    def test_other_securables_and_principal_filter(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.grants.responses["get"] = uc.GetPermissionsResponse(privilege_assignments=[])
        dbx.grants("main.sales", "analysts", securable_type="schema")
        assert ws.called("grants.get") == [
            {"securable_type": "SCHEMA", "full_name": "main.sales", "principal": "analysts"}
        ]

    def test_effective_grants_keep_their_source(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.grants.responses["get_effective"] = uc.EffectivePermissionsList(
            privilege_assignments=[
                uc.EffectivePrivilegeAssignment(
                    principal="analysts",
                    privileges=[
                        uc.EffectivePrivilege(privilege=uc.Privilege.SELECT),
                        uc.EffectivePrivilege(
                            privilege=uc.Privilege.USE_SCHEMA,
                            inherited_from_type=uc.SecurableType.SCHEMA,
                            inherited_from_name="main.sales",
                        ),
                    ],
                )
            ]
        )
        grants = dbx.effective_grants(REF)
        assert Grant("analysts", ("SELECT",)) in grants
        assert Grant("analysts", ("USE_SCHEMA",), "SCHEMA", "main.sales") in grants

    def test_grant_sends_a_permissions_change(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.grants.responses["update"] = uc.UpdatePermissionsResponse(
            privilege_assignments=[
                uc.PrivilegeAssignment(principal="analysts", privileges=[uc.Privilege.SELECT])
            ]
        )
        after = dbx.grant(REF, "analysts", ["select", "external use schema", "SOME_NEW_PRIV"])

        (call,) = ws.called("grants.update")
        assert call["securable_type"] == "TABLE"
        assert call["full_name"] == "main.sales.orders"
        (change,) = call["changes"]
        # The SDK's own serialisation must accept what we built.
        assert change.as_dict() == {
            "principal": "analysts",
            "add": ["SELECT", "EXTERNAL_USE_SCHEMA", "SOME_NEW_PRIV"],
        }
        assert after == [Grant("analysts", ("SELECT",))]

    def test_revoke_removes(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.grants.responses["update"] = uc.UpdatePermissionsResponse(privilege_assignments=[])
        dbx.revoke("main.sales.vol", "analysts", ["READ_VOLUME"], securable_type="volume")
        (call,) = ws.called("grants.update")
        assert call["securable_type"] == "VOLUME"
        assert call["changes"][0].as_dict() == {"principal": "analysts", "remove": ["READ_VOLUME"]}

    def test_grant_denied_says_who_can(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.grants.responses["update"] = PermissionDenied("403 Forbidden")
        ws.grants.responses["get_effective"] = RuntimeError("cannot read either")
        with pytest.raises(PreflightError, match="ownership of the securable or MANAGE"):
            dbx.grant(REF, "analysts", ["SELECT"])

    def test_empty_privileges_refused(self, dbx: Any) -> None:
        with pytest.raises(InvalidReferenceError):
            dbx.grant(REF, "analysts", [])


# ===================================================================== tags


class TestTags:
    def test_tags_on_table_and_column(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.entity_tag_assignments.responses["list"] = lambda **kw: iter(
            [
                uc.EntityTagAssignment(
                    entity_name=kw["entity_name"],
                    tag_key="pii",
                    entity_type=kw["entity_type"],
                    tag_value="high",
                ),
                uc.EntityTagAssignment(
                    entity_name=kw["entity_name"], tag_key="gold", entity_type=kw["entity_type"]
                ),
            ]
        )
        assert dbx.tags(REF) == {"pii": "high", "gold": ""}
        dbx.tags(REF, column="ssn")
        assert ws.called("entity_tag_assignments.list") == [
            {"entity_type": "tables", "entity_name": "main.sales.orders"},
            {"entity_type": "columns", "entity_name": "main.sales.orders.ssn"},
        ]

    def test_set_tags_updates_existing_and_creates_new(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.entity_tag_assignments.responses["list"] = lambda **kw: iter(
            [uc.EntityTagAssignment(entity_name="x", tag_key="pii", entity_type="tables")]
        )
        dbx.set_tags(REF, {"pii": "low", "domain": "sales"})

        (update,) = ws.called("entity_tag_assignments.update")
        assert update["tag_key"] == "pii" and update["update_mask"] == "tag_value"
        assert update["tag_assignment"].tag_value == "low"
        (create,) = ws.called("entity_tag_assignments.create")
        assert create["tag_assignment"].as_dict() == {
            "entity_name": "main.sales.orders",
            "entity_type": "tables",
            "tag_key": "domain",
            "tag_value": "sales",
        }

    def test_unset_tags(self, dbx: Any, ws: FakeWorkspace) -> None:
        dbx.unset_tags(REF, ["a", "b"], column="ssn")
        assert ws.called("entity_tag_assignments.delete") == [
            {"entity_type": "columns", "entity_name": "main.sales.orders.ssn", "tag_key": "a"},
            {"entity_type": "columns", "entity_name": "main.sales.orders.ssn", "tag_key": "b"},
        ]


# ======================================================== owner and lineage


class TestOwnerAndLineage:
    def test_set_owner(self, dbx: Any, ws: FakeWorkspace) -> None:
        dbx.set_owner(REF, "data-eng")
        assert ws.called("tables.update") == [
            {"full_name": "main.sales.orders", "owner": "data-eng"}
        ]

    LINEAGE: ClassVar[dict[str, Any]] = {
        "upstreams": [
            {
                "tableInfo": {
                    "name": "raw_orders",
                    "catalog_name": "main",
                    "schema_name": "bronze",
                    "table_type": "TABLE",
                    "lineage_timestamp": "2026-01-01 00:00:00.0",
                },
                "notebookInfos": [{"workspace_id": 42, "notebook_id": 1001}],
                "jobInfos": [{"workspace_id": 42, "job_id": 7}],
            },
            {"fileInfo": {"path": "s3://landing/orders.csv"}},
        ],
        "downstreams": [
            {"tableInfo": {"name": "daily", "catalog_name": "main", "schema_name": "gold"}},
            {"dashboardV3Infos": [{"dashboard_id": "d1", "workspace_id": 42}]},
        ],
    }

    def test_lineage_request_and_normalisation(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.api_client.responses["do"] = self.LINEAGE
        lineage = dbx.lineage(REF)

        (call,) = ws.called("api_client.do")
        assert call["method"] == "GET"
        assert call["path"] == "/api/2.0/lineage-tracking/table-lineage"
        assert call["query"] == {"table_name": "main.sales.orders", "include_entity_lineage": True}

        assert isinstance(lineage, Lineage)
        assert lineage.upstream_tables == ("main.bronze.raw_orders",)
        assert lineage.downstream_tables == ("main.gold.daily",)
        kinds = {(e.kind, e.name) for e in lineage.upstream}
        assert ("notebook", "1001") in kinds
        assert ("job", "7") in kinds
        assert ("file", "s3://landing/orders.csv") in kinds
        assert ("dashboard", "d1") in {(e.kind, e.name) for e in lineage.downstream}
        assert {e.workspace_id for e in lineage.upstream if e.kind == "notebook"} == {42}

    def test_direction_filters(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.api_client.responses["do"] = self.LINEAGE
        assert dbx.lineage(REF, direction="upstream").downstream == ()
        assert dbx.lineage(REF, direction="downstream").upstream == ()
        with pytest.raises(InvalidReferenceError):
            dbx.lineage(REF, direction="sideways")

    def test_column_lineage(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.api_client.responses["do"] = {
            "upstream_cols": [
                {
                    "name": "id",
                    "catalog_name": "main",
                    "schema_name": "bronze",
                    "table_name": "raw_orders",
                    "table_type": "TABLE",
                }
            ],
            "downstream_cols": [],
        }
        lineage = dbx.column_lineage(REF, "id")
        (call,) = ws.called("api_client.do")
        assert call["path"] == "/api/2.0/lineage-tracking/column-lineage"
        assert call["query"] == {"table_name": "main.sales.orders", "column_name": "id"}
        assert [(e.table, e.column) for e in lineage.upstream] == [("main.bronze.raw_orders", "id")]


# ============================================================== constraints


class TestConstraints:
    def test_primary_key(self, dbx: Any, ws: FakeWorkspace) -> None:
        dbx.add_primary_key(REF, "orders_pk", ["id"], rely=True)
        (call,) = ws.called("table_constraints.create")
        assert call["full_name_arg"] == "main.sales.orders"
        assert call["constraint"].as_dict() == {
            "primary_key_constraint": {"name": "orders_pk", "child_columns": ["id"], "rely": True}
        }

    def test_foreign_key(self, dbx: Any, ws: FakeWorkspace) -> None:
        dbx.add_foreign_key(REF, "fk", ["region"], parse_ref("main.ref.regions"), ["code"])
        (call,) = ws.called("table_constraints.create")
        assert call["constraint"].as_dict()["foreign_key_constraint"] == {
            "name": "fk",
            "child_columns": ["region"],
            "parent_table": "main.ref.regions",
            "parent_columns": ["code"],
            "rely": False,
        }

    def test_foreign_key_arity_checked(self, dbx: Any) -> None:
        with pytest.raises(InvalidReferenceError, match="one to one"):
            dbx.add_foreign_key(REF, "fk", ["a", "b"], parse_ref("main.ref.r"), ["c"])

    def test_drop(self, dbx: Any, ws: FakeWorkspace) -> None:
        dbx.drop_table_constraint(REF, "fk", cascade=True)
        assert ws.called("table_constraints.delete") == [
            {"full_name": "main.sales.orders", "constraint_name": "fk", "cascade": True}
        ]


# =============================================================== namespaces


class TestDatabricksNamespaces:
    def test_catalog_and_schema_lifecycle(self, dbx: Any, ws: FakeWorkspace) -> None:
        dbx.create_catalog("sandbox", comment="play", storage_root="s3://b/sandbox")
        dbx.create_schema("sandbox", "s1", comment="c")
        dbx.drop_schema("sandbox", "s1", force=True)
        dbx.drop_catalog("sandbox", force=True)
        assert ws.called("catalogs.create") == [
            {"name": "sandbox", "comment": "play", "storage_root": "s3://b/sandbox"}
        ]
        assert ws.called("schemas.create") == [
            {"name": "s1", "catalog_name": "sandbox", "comment": "c", "storage_root": None}
        ]
        assert ws.called("schemas.delete") == [{"full_name": "sandbox.s1", "force": True}]
        assert ws.called("catalogs.delete") == [{"name": "sandbox", "force": True}]

    def test_table_exists(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.tables.responses["exists"] = uc.TableExistsResponse(table_exists=True)
        assert dbx.table_exists(REF) is True
        ws.tables.responses["exists"] = uc.TableExistsResponse(table_exists=False)
        assert dbx.table_exists(REF) is False
        ws.tables.responses["exists"] = NotFound("Schema 'main.sales' does not exist")
        assert dbx.table_exists(REF) is False
        ws.tables.responses["exists"] = PermissionDenied("403")
        with pytest.raises(PreflightError):
            dbx.table_exists(REF)

    def test_search_tables(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.tables.responses["list_summaries"] = iter(
            [
                uc.TableSummary(
                    full_name="main.sales.orders",
                    table_type=uc.TableType.MANAGED,
                    securable_kind_manifest=uc.SecurableKindManifest(
                        securable_kind=uc.SecurableKind.TABLE_DELTA,
                        capabilities=["HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT"],
                    ),
                )
            ]
        )
        (summary,) = dbx.search_tables("main", "sal%", "ord_rs")
        assert ws.called("tables.list_summaries") == [
            {
                "catalog_name": "main",
                "schema_name_pattern": "sal%",
                "table_name_pattern": "ord_rs",
                "include_manifest_capabilities": True,
            }
        ]
        assert summary.full_name == "main.sales.orders"
        assert summary.table_type == "MANAGED"
        assert summary.securable_kind == "TABLE_DELTA"
        assert summary.capabilities == frozenset({"HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT"})

    def test_functions_and_volumes(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.functions.responses["list"] = iter(
            [
                uc.FunctionInfo(
                    name="mask_ssn",
                    catalog_name="main",
                    schema_name="sec",
                    full_name="main.sec.mask_ssn",
                    data_type=uc.ColumnTypeName.STRING,
                    routine_body=uc.FunctionInfoRoutineBody.SQL,
                )
            ]
        )
        ws.volumes.responses["list"] = iter(
            [
                uc.VolumeInfo(
                    name="landing",
                    catalog_name="main",
                    schema_name="sales",
                    full_name="main.sales.landing",
                    volume_type=uc.VolumeType.MANAGED,
                    storage_location="s3://b/v",
                )
            ]
        )
        (fn,) = dbx.list_functions("main", "sec")
        assert (fn.full_name, fn.data_type, fn.routine_body) == (
            "main.sec.mask_ssn",
            "STRING",
            "SQL",
        )
        (vol,) = dbx.list_volumes("main", "sales")
        assert (vol.full_name, vol.volume_type) == ("main.sales.landing", "MANAGED")

    def test_volume_create_and_drop(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.volumes.responses["create"] = uc.VolumeInfo(
            name="v", catalog_name="main", schema_name="sales", volume_type=uc.VolumeType.EXTERNAL
        )
        summary = dbx.create_volume(
            "main", "sales", "v", volume_type="external", storage_location="s3://b/v"
        )
        assert summary.full_name == "main.sales.v"
        (call,) = ws.called("volumes.create")
        assert call["volume_type"] is uc.VolumeType.EXTERNAL
        assert call["storage_location"] == "s3://b/v"
        with pytest.raises(InvalidReferenceError, match="storage_location"):
            dbx.create_volume("main", "sales", "v2", volume_type="EXTERNAL")
        dbx.drop_volume("main", "sales", "v")
        assert ws.called("volumes.delete") == [{"name": "main.sales.v"}]


class TestVolumeFiles:
    def test_round_trip_through_the_files_api(self, dbx: Any, ws: FakeWorkspace) -> None:
        class Download:
            def __init__(self, data: bytes) -> None:
                import io

                self.contents = io.BytesIO(data)

        ws.files.responses["download"] = lambda **kw: Download(b"hello")
        ws.files.responses["list_directory_contents"] = iter(
            [
                uc_files.DirectoryEntry(
                    path="/Volumes/main/sales/landing/in/a.csv",
                    name="a.csv",
                    is_directory=False,
                    file_size=5,
                    last_modified=1700000000000,
                ),
                uc_files.DirectoryEntry(
                    path="/Volumes/main/sales/landing/in/sub/", name="sub", is_directory=True
                ),
            ]
        )
        volume = dbx.volume("main.sales.landing")
        assert repr(volume) == "Volume('main.sales.landing')"

        entries = volume.list("in")
        assert [(e.path, e.is_directory, e.size) for e in entries] == [
            ("in/a.csv", False, 5),
            ("in/sub/", True, None),
        ]
        assert volume.read("in/a.csv") == b"hello"
        volume.write("out/b.bin", b"\x00\x01", overwrite=True)
        volume.delete("in/a.csv")

        root = "/Volumes/main/sales/landing"
        assert ws.called("files.list_directory_contents") == [{"directory_path": f"{root}/in/"}]
        assert ws.called("files.download") == [{"file_path": f"{root}/in/a.csv"}]
        (upload,) = ws.called("files.upload")
        assert upload["file_path"] == f"{root}/out/b.bin"
        assert upload["contents"].getvalue() == b"\x00\x01"
        assert upload["overwrite"] is True
        assert ws.called("files.delete") == [{"file_path": f"{root}/in/a.csv"}]

    def test_paths_cannot_escape_the_volume(self, dbx: Any) -> None:
        volume = dbx.volume(parse_ref("main.sales.landing"))
        assert volume.path("a/./b") == "/Volumes/main/sales/landing/a/b"
        with pytest.raises(InvalidReferenceError, match="climbs out"):
            volume.path("../other/secret")

    def test_denied_names_volume_privileges(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.files.responses["download"] = PermissionDenied("403")
        ws.grants.responses["get_effective"] = RuntimeError("no")
        with pytest.raises(PreflightError, match="READ_VOLUME"):
            dbx.volume("main.sales.landing").read("x")

    def test_bad_volume_name(self, dbx: Any) -> None:
        with pytest.raises(InvalidReferenceError):
            dbx.volume("main.sales")


class TestDatabricksRegister:
    def test_register_creates_external_delta_then_resolves(
        self, dbx: Any, ws: FakeWorkspace
    ) -> None:
        ws.tables.responses["create"] = _table_info()
        ws.tables.responses["get"] = _table_info()
        ws.tables.responses["list"] = iter([])

        resolved = dbx.register_table(
            REF,
            "s3://bucket/orders",
            columns_schema_json=json.dumps(DELTA_SCHEMA),
            partition_columns=["region"],
            properties={"owner.team": "sales"},
        )

        (call,) = ws.called("tables.create")
        assert call["name"] == "orders"
        assert (call["catalog_name"], call["schema_name"]) == ("main", "sales")
        assert call["table_type"] is uc.TableType.EXTERNAL
        assert call["data_source_format"] is uc.DataSourceFormat.DELTA
        assert call["storage_location"] == "s3://bucket/orders"
        assert call["properties"] == {"owner.team": "sales"}
        assert [c.type_text for c in call["columns"]] == [
            "bigint",
            "decimal(10,2)",
            "array<string>",
            "string",
        ]
        assert call["columns"][3].partition_index == 0
        assert resolved.table_id == "tid-1"
        assert resolved.location == "s3://bucket/orders"

    def test_comment_goes_through_the_raw_body(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.api_client.responses["do"] = {}
        ws.tables.responses["get"] = _table_info()
        ws.tables.responses["list"] = iter([])
        dbx.register_table(REF, "s3://bucket/orders", comment="hello")
        (call,) = ws.called("api_client.do")
        assert (call["method"], call["path"]) == ("POST", "/api/2.1/unity-catalog/tables")
        assert call["body"]["comment"] == "hello"
        assert call["body"]["table_type"] == "EXTERNAL"

    def test_partitions_without_schema_refused(self, dbx: Any) -> None:
        with pytest.raises(InvalidReferenceError, match="columns_schema_json"):
            dbx.register_table(REF, "s3://b/t", partition_columns=["p"])

    def test_register_denied_names_external_use_schema(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.tables.responses["create"] = PermissionDenied("403")
        ws.grants.responses["get_effective"] = RuntimeError("no")
        with pytest.raises(PreflightError, match="EXTERNAL_USE_SCHEMA"):
            dbx.register_table(REF, "s3://b/t")


class TestPathCredentials:
    def test_aws(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.temporary_path_credentials.responses["generate_temporary_path_credentials"] = (
            uc.GenerateTemporaryPathCredentialResponse(
                aws_temp_credentials=uc.AwsCredentials(
                    access_key_id="AKIA", secret_access_key="SUPERSECRET", session_token="TOK"
                ),
                expiration_time=2000000000000,
                url="s3://bucket/new_table",
            )
        )
        creds = dbx.path_credentials("s3://bucket/new_table", "PATH_CREATE_TABLE")
        (call,) = ws.called("temporary_path_credentials.generate_temporary_path_credentials")
        assert call == {
            "url": "s3://bucket/new_table",
            "operation": uc.PathOperation.PATH_CREATE_TABLE,
        }
        assert creds.as_storage_options()["aws_secret_access_key"] == "SUPERSECRET"
        assert creds.expires_at == 2000000000.0
        assert creds.table_id is None
        assert "SUPERSECRET" not in repr(creds)

    def test_azure_gets_an_explicit_endpoint(self, dbx: Any, ws: FakeWorkspace) -> None:
        url = "abfss://c@acct.dfs.core.windows.net/t"
        ws.temporary_path_credentials.responses["generate_temporary_path_credentials"] = (
            uc.GenerateTemporaryPathCredentialResponse(
                azure_user_delegation_sas=uc.AzureUserDelegationSas(sas_token="sv=SECRET"),
                url=url,
            )
        )
        creds = dbx.path_credentials(url, "read")
        assert creds.as_storage_options()["azure_endpoint"] == "https://acct.blob.core.windows.net"
        assert "SECRET" not in repr(creds)

    def test_bad_operation(self, dbx: Any) -> None:
        with pytest.raises(InvalidReferenceError, match="PATH_READ"):
            dbx.path_credentials("s3://b/t", "WRITE_ONLY")


STAGING = {
    "table-id": "11111111-2222-3333-4444-555555555555",
    "table-type": "MANAGED",
    "location": "s3://bucket/metastore/tables/1111",
    "storage-credentials": [
        {
            "prefix": "s3://bucket/metastore/",
            "operation": "READ",
            "config": {"s3.access-key-id": "WRONG"},
        },
        {
            "prefix": "s3://bucket/metastore/tables/1111/",
            "operation": "READ_WRITE",
            "expiration-time-ms": 1900000000000,
            "config": {
                "s3.access-key-id": "AKIA",
                "s3.secret-access-key": "SUPERSECRET",
                "s3.session-token": "TOKEN",
            },
        },
    ],
    "required-protocol": {"min-reader-version": 3, "min-writer-version": 7},
    "required-properties": {"delta.feature.catalogManaged": "supported", "x.any": None},
    "suggested-properties": {"delta.enableDeletionVectors": "true"},
}


class TestDatabricksStaging:
    def test_create_staging_table(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.api_client.responses["do"] = STAGING
        staged = dbx.create_staging_table(REF)

        (call,) = ws.called("api_client.do")
        assert call["method"] == "POST"
        assert (
            call["path"]
            == "/api/2.1/unity-catalog/delta/v1/catalogs/main/schemas/sales/staging-tables"
        )
        assert call["body"] == {"name": "orders"}

        assert isinstance(staged, StagingTable)
        assert staged.table_id == STAGING["table-id"]
        assert staged.location == STAGING["location"]
        assert staged.storage_options == {
            "aws_access_key_id": "AKIA",
            "aws_secret_access_key": "SUPERSECRET",
            "aws_session_token": "TOKEN",
        }
        assert staged.expires_at == 1900000000.0
        assert staged.required_properties == {
            "delta.feature.catalogManaged": "supported",
            "x.any": None,
        }
        assert staged.required_protocol["min-reader-version"] == 3
        assert "SUPERSECRET" not in repr(staged)
        assert "SUPERSECRET" not in str(staged)

    def test_finalize_posts_then_resolves(self, dbx: Any, ws: FakeWorkspace) -> None:
        def do(**kw: Any) -> Any:
            if kw["method"] == "POST":
                return {"metadata": {}, "commits": []}
            return {"commits": [], "latest_table_version": 0, "location": "s3://b/t"}

        ws.api_client.responses["do"] = do
        ws.tables.responses["get"] = _table_info(
            table_type=uc.TableType.MANAGED,
            properties={"delta.feature.catalogManaged": "supported"},
        )
        body = {"name": "orders", "location": "s3://b/t", "table-type": "MANAGED"}
        resolved = dbx.finalize_managed_table(REF, body)

        post, get = ws.called("api_client.do")
        assert post["path"] == "/api/2.1/unity-catalog/delta/v1/catalogs/main/schemas/sales/tables"
        assert post["body"] == body
        assert get["method"] == "GET"
        assert resolved.is_catalog_managed and resolved.max_catalog_version == 0

    def test_finalize_refuses_a_body_for_another_table(self, dbx: Any) -> None:
        with pytest.raises(InvalidReferenceError, match="names 'other'"):
            dbx.finalize_managed_table(REF, {"name": "other"})

    def test_staging_denied_mentions_the_preview(self, dbx: Any, ws: FakeWorkspace) -> None:
        ws.api_client.responses["do"] = PermissionDenied("403")
        ws.grants.responses["get_effective"] = RuntimeError("no")
        with pytest.raises(PreflightError, match="preview"):
            dbx.create_staging_table(REF)
