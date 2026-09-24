"""Governance and table lifecycle on both Unity Catalogs, with no network.

Databricks is exercised through a stub WorkspaceClient that records every call
and answers with real SDK dataclasses, so request shapes are checked against
the SDK's own types and responses go through the same `as_dict()` path a live
answer would. Open-source Unity Catalog is exercised against `tests/fake_uc.py`,
which speaks the REST protocol over a real socket.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

pytest.importorskip("databricks.sdk")

from databricks.sdk.service import catalog as uc
from databricks.sdk.service import files as uc_files
from deltaswamp.catalog.base import (
    GovernedCatalog,
    NamespaceCatalog,
    TableLifecycleCatalog,
)
from deltaswamp.catalog.databricks import DatabricksUnityCatalog
from deltaswamp.catalog.ossuc import OSSUnityCatalog
from deltaswamp.errors import (
    DeltaSwampError,
    InvalidReferenceError,
    PreflightError,
)
from deltaswamp.governance import (
    Grant,
    Lineage,
    StagingTable,
    TableInfo,
    delta_schema_to_columns,
    normalize_privilege,
    staging_storage_options,
)
from deltaswamp.identity import parse_ref

from tests.fake_uc import FakeTable, FakeUnityCatalog

REF = parse_ref("main.sales.orders")


# ----------------------------------------------------------------- the stub


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


# =============================================================== table info


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

    def test_resolve_denial_now_matches_held_privileges(self, dbx: Any, ws: FakeWorkspace) -> None:
        """The old check compared "EXTERNAL USE SCHEMA" to str(enum) and never matched."""
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


# ============================================================== permissions


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


# ================================================================ lifecycle

SCHEMA = {
    "type": "struct",
    "fields": [
        {"name": "id", "type": "long", "nullable": False, "metadata": {"comment": "key"}},
        {"name": "amount", "type": "decimal(10,2)", "nullable": True, "metadata": {}},
        {
            "name": "tags",
            "type": {"type": "array", "elementType": "string", "containsNull": True},
            "nullable": True,
            "metadata": {},
        },
        {"name": "region", "type": "string", "nullable": True, "metadata": {}},
    ],
}


class TestSchemaToColumns:
    def test_spark_spelling_and_partitions(self) -> None:
        cols = delta_schema_to_columns(json.dumps(SCHEMA), ["region"])
        assert [(c["name"], c["type_text"], c["type_name"]) for c in cols] == [
            ("id", "bigint", "LONG"),
            ("amount", "decimal(10,2)", "DECIMAL"),
            ("tags", "array<string>", "ARRAY"),
            ("region", "string", "STRING"),
        ]
        assert cols[0]["nullable"] is False and cols[0]["comment"] == "key"
        assert (cols[1]["type_precision"], cols[1]["type_scale"]) == (10, 2)
        assert cols[3]["partition_index"] == 0
        assert "partition_index" not in cols[0]
        assert json.loads(cols[2]["type_json"])["type"]["elementType"] == "string"

    def test_nested_struct_and_map(self) -> None:
        schema = {
            "type": "struct",
            "fields": [
                {
                    "name": "m",
                    "type": {
                        "type": "map",
                        "keyType": "string",
                        "valueType": {
                            "type": "struct",
                            "fields": [{"name": "x", "type": "integer", "nullable": True}],
                        },
                        "valueContainsNull": True,
                    },
                    "nullable": True,
                }
            ],
        }
        (col,) = delta_schema_to_columns(schema)
        assert col["type_text"] == "map<string,struct<x:int>>"

    def test_unknown_partition_column(self) -> None:
        with pytest.raises(DeltaSwampError, match="not in the schema"):
            delta_schema_to_columns(SCHEMA, ["nope"])


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
            columns_schema_json=json.dumps(SCHEMA),
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
        assert creds.as_storage_options()["azure_endpoint"] == "https://acct.dfs.core.windows.net"
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


class TestStagingOptions:
    def test_azure_and_gcs(self) -> None:
        azure, _ = staging_storage_options(
            "abfss://c@acct.dfs.core.windows.net/t",
            [
                {
                    "prefix": "abfss://c@acct.dfs.core.windows.net/t",
                    "operation": "READ_WRITE",
                    "config": {"azure.sas-token": "sv=x"},
                }
            ],
        )
        assert azure == {
            "azure_storage_sas_key": "sv=x",
            "azure_endpoint": "https://acct.dfs.core.windows.net",
        }
        gcs, expires = staging_storage_options(
            "gs://b/t",
            [
                {
                    "prefix": "gs://b/t/",
                    "operation": "READ_WRITE",
                    "config": {"gcs.oauth-token": "y"},
                }
            ],
        )
        assert gcs == {"google_bearer_token": "y"} and expires is None

    def test_local_storage_has_no_options(self) -> None:
        assert staging_storage_options("file:///tmp/t", []) == ({}, None)


def test_privilege_spellings_converge() -> None:
    assert normalize_privilege("use schema") == "USE_SCHEMA"
    assert normalize_privilege(uc.Privilege.EXTERNAL_USE_SCHEMA) == "EXTERNAL_USE_SCHEMA"
    assert normalize_privilege("ALL-PRIVILEGES") == "ALL_PRIVILEGES"


def test_both_catalogs_satisfy_the_protocols(dbx: Any) -> None:
    oss = OSSUnityCatalog("http://localhost:1")
    for catalog in (dbx, oss):
        assert isinstance(catalog, GovernedCatalog)
        assert isinstance(catalog, NamespaceCatalog)
        assert isinstance(catalog, TableLifecycleCatalog)
    assert dbx.unsupported_operations == frozenset()
    assert "tags" in oss.unsupported_operations


# ================================================================ OSS Unity


@pytest.fixture
def fake(tmp_path: Path) -> Iterator[FakeUnityCatalog]:
    with FakeUnityCatalog(staging_root=tmp_path / "staging") as server:
        server.add_table(
            "main.sales.orders",
            FakeTable(
                name="orders",
                location="s3://bucket/orders",
                table_id="tid-9",
                owner="alice",
                comment="all orders",
                columns=[
                    {"name": "id", "type_text": "bigint", "type_name": "LONG", "position": 0},
                    {
                        "name": "region",
                        "type_text": "string",
                        "type_name": "STRING",
                        "position": 1,
                        "partition_index": 0,
                    },
                ],
                created_at=1700000000000,
            ),
        )
        yield server


@pytest.fixture
def oss(fake: FakeUnityCatalog) -> OSSUnityCatalog:
    return OSSUnityCatalog(fake.url, token="tok")


class TestOSSGovernance:
    def test_table_info(self, oss: OSSUnityCatalog) -> None:
        info = oss.table_info(REF)
        assert (info.full_name, info.owner, info.comment) == (
            "main.sales.orders",
            "alice",
            "all orders",
        )
        assert info.partition_columns == ("region",)
        assert info.created_at == 1700000000000

    def test_grant_revoke_round_trip_uses_space_spelling_on_the_wire(
        self, oss: OSSUnityCatalog, fake: FakeUnityCatalog
    ) -> None:
        after = oss.grant(REF, "analysts", ["SELECT", "USE_SCHEMA"])
        assert after == [Grant("analysts", ("SELECT", "USE_SCHEMA"))]
        method, path, body = fake.requests[-1]
        assert method == "PATCH"
        assert path == "/api/2.1/unity-catalog/permissions/table/main.sales.orders"
        assert body == {"changes": [{"principal": "analysts", "add": ["SELECT", "USE SCHEMA"]}]}

        oss.revoke(REF, "analysts", ["use schema"])
        assert oss.grants(REF) == [Grant("analysts", ("SELECT",))]
        assert oss.grants(REF, principal="nobody") == []
        oss.grant("main", "analysts", ["USE_CATALOG"], securable_type="CATALOG")
        assert oss.grants("main", securable_type="catalog") == [Grant("analysts", ("USE_CATALOG",))]

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("effective_grants", (REF,)),
            ("tags", (REF,)),
            ("set_tags", (REF, {"a": "b"})),
            ("unset_tags", (REF, ["a"])),
            ("set_owner", (REF, "bob")),
            ("lineage", (REF,)),
            ("column_lineage", (REF, "id")),
            ("add_primary_key", (REF, "pk", ["id"])),
            ("add_foreign_key", (REF, "fk", ["id"], REF, ["id"])),
            ("drop_table_constraint", (REF, "pk")),
            ("volume", ("main.sales.v",)),
        ],
    )
    def test_unsupported_says_what_oss_lacks(
        self, oss: OSSUnityCatalog, method: str, args: tuple[Any, ...]
    ) -> None:
        assert method in oss.unsupported_operations
        with pytest.raises(NotImplementedError, match="OSS Unity Catalog"):
            getattr(oss, method)(*args)


class TestOSSNamespaces:
    def test_catalog_and_schema_lifecycle(
        self, oss: OSSUnityCatalog, fake: FakeUnityCatalog
    ) -> None:
        oss.create_catalog("sandbox", comment="play")
        oss.create_schema("sandbox", "s1")
        assert "sandbox" in oss.list_catalogs()
        assert oss.list_schemas("sandbox") == ["s1"]
        assert fake.requests[0][2] == {"name": "sandbox", "comment": "play"}

        with pytest.raises(PreflightError, match="not empty"):
            oss.drop_catalog("sandbox")
        oss.drop_schema("sandbox", "s1")
        oss.drop_catalog("sandbox")
        assert "sandbox" not in oss.list_catalogs()

    def test_force_drop(self, oss: OSSUnityCatalog) -> None:
        oss.drop_schema("main", "sales", force=True)
        assert oss.list_tables("main", "sales") == []

    def test_missing_is_an_invalid_reference(self, oss: OSSUnityCatalog) -> None:
        with pytest.raises(InvalidReferenceError):
            oss.drop_catalog("nope")

    def test_table_exists(self, oss: OSSUnityCatalog) -> None:
        assert oss.table_exists(REF) is True
        assert oss.table_exists(parse_ref("main.sales.nope")) is False

    def test_search_tables_emulates_like(
        self, oss: OSSUnityCatalog, fake: FakeUnityCatalog
    ) -> None:
        fake.add_table(
            "main.hr.people", FakeTable(name="people", location="s3://b/p", table_id="x")
        )
        assert sorted(t.full_name for t in oss.search_tables("main")) == [
            "main.hr.people",
            "main.sales.orders",
        ]
        assert [t.full_name for t in oss.search_tables("main", "sa%")] == ["main.sales.orders"]
        assert [t.full_name for t in oss.search_tables("main", None, "p_ople")] == [
            "main.hr.people"
        ]

    def test_functions_and_volumes(self, oss: OSSUnityCatalog, fake: FakeUnityCatalog) -> None:
        fake.add_function("main.sales.f", data_type="INT", comment="adds")
        assert [(f.full_name, f.data_type) for f in oss.list_functions("main", "sales")] == [
            ("main.sales.f", "INT")
        ]
        created = oss.create_volume(
            "main", "sales", "landing", volume_type="EXTERNAL", storage_location="s3://b/v"
        )
        assert created.full_name == "main.sales.landing" and created.volume_type == "EXTERNAL"
        assert [v.full_name for v in oss.list_volumes("main", "sales")] == ["main.sales.landing"]
        oss.drop_volume("main", "sales", "landing")
        assert oss.list_volumes("main", "sales") == []


class TestOSSLifecycle:
    def test_register_external_then_resolve(
        self, oss: OSSUnityCatalog, fake: FakeUnityCatalog
    ) -> None:
        ref = parse_ref("main.sales.events")
        resolved = oss.register_table(
            ref,
            "s3://bucket/events",
            columns_schema_json=SCHEMA,
            partition_columns=["region"],
            properties={"k": "v"},
            comment="events",
        )
        _, path, body = next(r for r in fake.requests if r[0] == "POST")
        assert path == "/api/2.1/unity-catalog/tables"
        assert body["table_type"] == "EXTERNAL" and body["data_source_format"] == "DELTA"
        assert body["storage_location"] == "s3://bucket/events"
        assert body["comment"] == "events"
        assert body["columns"][3]["partition_index"] == 0
        assert resolved.location == "s3://bucket/events"
        assert resolved.table_id == fake.tables["main.sales.events"].table_id
        assert oss.table_info(ref).comment == "events"

    def test_register_twice_is_refused(self, oss: OSSUnityCatalog) -> None:
        with pytest.raises(PreflightError, match="409"):
            oss.register_table(REF, "s3://bucket/orders")

    def test_path_credentials(self, oss: OSSUnityCatalog, fake: FakeUnityCatalog) -> None:
        creds = oss.path_credentials("s3://bucket/new", "PATH_CREATE_TABLE")
        assert fake.requests[-1][1:] == (
            "/api/2.1/unity-catalog/temporary-path-credentials",
            {"url": "s3://bucket/new", "operation": "PATH_CREATE_TABLE"},
        )
        assert creds.url == "s3://bucket/new"
        assert creds.as_storage_options()["aws_secret_access_key"] == "path-secret"
        assert "path-secret" not in repr(creds)

    def _write_v0(self, staged: StagingTable) -> Path:
        from urllib.parse import urlparse

        root = Path(urlparse(staged.location).path)
        (root / "_delta_log").mkdir(parents=True)
        (root / "_delta_log" / f"{0:020d}.json").write_text("{}\n")
        return root

    def _request_body(self, staged: StagingTable, **overrides: Any) -> dict[str, Any]:
        properties = {
            k: v
            for k, v in staged.required_properties.items()
            if v is not None and not k.startswith("delta.feature.")
        }
        body: dict[str, Any] = {
            "name": "fresh",
            "location": staged.location,
            "table-type": "MANAGED",
            "columns": {"type": "struct", "fields": [{"name": "id", "type": "long"}]},
            "partition-columns": [],
            "protocol": {
                "min-reader-version": 3,
                "min-writer-version": 7,
                "reader-features": ["catalogManaged"],
                "writer-features": ["catalogManaged", "inCommitTimestamp"],
            },
            "properties": properties,
            "last-commit-timestamp-ms": 1700000000000,
        }
        body.update(overrides)
        return body

    def test_staging_then_finalize(self, oss: OSSUnityCatalog, fake: FakeUnityCatalog) -> None:
        ref = parse_ref("main.sales.fresh")
        staged = oss.create_staging_table(ref)
        assert fake.requests[-1][1] == (
            "/api/2.1/unity-catalog/delta/v1/catalogs/main/schemas/sales/staging-tables"
        )
        assert staged.location.startswith("file://")
        assert staged.storage_options == {}
        assert staged.required_properties["io.unitycatalog.tableId"] == staged.table_id
        assert staged.required_properties["delta.feature.catalogManaged"] == "supported"
        assert not oss.table_exists(ref)  # staged, not registered

        self._write_v0(staged)
        resolved = oss.finalize_managed_table(ref, self._request_body(staged))

        assert fake.requests[-3][1] == (
            "/api/2.1/unity-catalog/delta/v1/catalogs/main/schemas/sales/tables"
        )
        assert resolved.table_type is not None and resolved.table_type.value == "MANAGED"
        assert resolved.table_id == staged.table_id
        assert resolved.is_catalog_managed
        assert resolved.max_catalog_version == 0
        assert resolved.location == staged.location

    def test_finalize_without_v0_is_refused(self, oss: OSSUnityCatalog) -> None:
        ref = parse_ref("main.sales.fresh")
        staged = oss.create_staging_table(ref)
        with pytest.raises(PreflightError, match="version 0"):
            oss.finalize_managed_table(ref, self._request_body(staged))

    def test_finalize_without_the_feature_is_refused(self, oss: OSSUnityCatalog) -> None:
        ref = parse_ref("main.sales.fresh")
        staged = oss.create_staging_table(ref)
        self._write_v0(staged)
        body = self._request_body(
            staged, protocol={"min-reader-version": 1, "min-writer-version": 2}
        )
        with pytest.raises(PreflightError, match="catalogManaged"):
            oss.finalize_managed_table(ref, body)

    def test_staging_maps_cloud_credentials(
        self, oss: OSSUnityCatalog, fake: FakeUnityCatalog
    ) -> None:
        fake.staging_credential_config = {
            "s3.access-key-id": "AKIA",
            "s3.secret-access-key": "SUPERSECRET",
            "s3.session-token": "TOKEN",
        }
        staged = oss.create_staging_table(parse_ref("main.sales.fresh"))
        assert staged.storage_options["aws_secret_access_key"] == "SUPERSECRET"
        assert staged.expires_at == 9999999999.0
        assert "SUPERSECRET" not in repr(staged)

    def test_staging_into_a_missing_schema(self, oss: OSSUnityCatalog) -> None:
        with pytest.raises(InvalidReferenceError):
            oss.create_staging_table(parse_ref("main.nope.t"))
