"""Engine routing.

The routing table is the product. These tests assert the decisions and, just as
importantly, the *reasons* -- an unexplained refusal is the failure mode this
library exists to remove.
"""

from __future__ import annotations

import pytest
from deltaswamp.capability import OPERATION_ENGINES, Engine, Operation
from deltaswamp.catalog import ResolvedTable, TableType
from deltaswamp.errors import FallbackRequiredError, UnreachableTableError
from deltaswamp.identity import RefKind, TableRef, parse_ref
from deltaswamp.router import Router

from tests.helpers import FakeEngine, accepting_router
from tests.helpers import resolved_table as table


def router(*, fallback: bool = False, sql: bool = False) -> Router:
    engines: dict[Engine, object] = {
        Engine.KERNEL: FakeEngine(Engine.KERNEL, {Operation.SCAN, Operation.TIME_TRAVEL}),
        Engine.DELTARS: FakeEngine(Engine.DELTARS, {Operation.MERGE, Operation.VACUUM}),
    }
    if sql:
        engines[Engine.SQL] = FakeEngine(Engine.SQL)
    return Router(engines=engines, allow_sql_fallback=fallback)


class TestPreferenceOrder:
    def test_scan_prefers_kernel(self) -> None:
        got = router().capability(Operation.SCAN, table())
        assert got.ok and got.engine is Engine.KERNEL

    def test_merge_falls_through_to_deltars(self) -> None:
        got = router().capability(Operation.MERGE, table())
        assert got.ok and got.engine is Engine.DELTARS

    def test_unavailable_everywhere_collects_every_reason(self) -> None:
        got = router().capability(Operation.SCAN, table(external_read_supported=False))
        assert not got.ok
        assert got.reason

    def test_an_operation_with_no_engine_says_so(self) -> None:
        """INCREMENTAL is in the matrix but nothing implements it yet."""
        got = router().capability(Operation.INCREMENTAL, table())
        assert not got.ok
        assert got.reason


class TestFallbackIsOptIn:
    def test_sql_is_refused_when_disabled(self) -> None:
        got = router(sql=True, fallback=False).capability(Operation.DROP_COLUMN, table())
        assert not got.ok
        assert "allow_sql_fallback" in got.remedy

    def test_sql_is_used_when_enabled(self) -> None:
        got = router(sql=True, fallback=True).capability(Operation.DROP_COLUMN, table())
        assert got.ok and got.engine is Engine.SQL

    def test_engine_for_raises_fallback_required(self) -> None:
        r = router(sql=True, fallback=False)
        with pytest.raises((FallbackRequiredError, UnreachableTableError)) as excinfo:
            r.engine_for(Operation.DROP_COLUMN, table())
        assert "drop_column" in str(excinfo.value)


class TestCatalogLevelRefusals:
    """Decisions that must be made from catalog metadata, before opening the log."""

    @pytest.mark.parametrize(
        "table_type",
        [
            TableType.VIEW,
            TableType.MATERIALIZED_VIEW,
            TableType.METRIC_VIEW,
            TableType.STREAMING_TABLE,
        ],
    )
    def test_view_like_relations_have_no_file_surface(self, table_type: TableType) -> None:
        got = router().capability(Operation.SCAN, table(table_type=table_type))
        assert not got.ok
        assert "file surface" in got.reason

    def test_foreign_tables_are_refused(self) -> None:
        got = router().capability(Operation.SCAN, table(table_type=TableType.FOREIGN))
        assert not got.ok and "FOREIGN" in got.reason

    def test_manifest_absent_read_capability_is_decisive(self) -> None:
        got = router().capability(Operation.SCAN, table(external_read_supported=False))
        assert not got.ok
        assert "HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT" in got.reason

    def test_access_policy_is_named_in_the_refusal(self) -> None:
        """A row filter or mask makes vending refuse, while the manifest still
        claims direct reads -- so the catalog reports the policy itself."""
        t = table(external_read_supported=False, access_policy="row filter main.s.rf")
        got = router().capability(Operation.SCAN, t)
        assert not got.ok
        assert "row filter main.s.rf" in got.reason

    def test_manifest_write_capability_blocks_writes_only(self) -> None:
        t = table(external_write_supported=False, external_read_supported=True)
        assert router().capability(Operation.SCAN, t).ok
        assert not router().capability(Operation.MERGE, t).ok

    def test_unknown_manifest_capability_is_not_treated_as_refusal(self) -> None:
        """None means 'we did not ask', which must never be read as 'no'."""
        assert router().capability(Operation.SCAN, table(external_read_supported=None)).ok


class TestCapabilitiesReport:
    def test_every_operation_gets_a_verdict(self) -> None:
        report = router().capabilities(table())
        assert set(report) == set(Operation)

    def test_refusals_always_carry_a_reason(self) -> None:
        for op, cap in router().capabilities(table()).items():
            if not cap.ok:
                assert cap.reason, f"{op.value} was refused without a reason"


class TestCapabilityImpliesCallable:
    """A yes from `capabilities()` is backed by a real, publicly reachable method."""

    def test_engines_that_claim_an_operation_implement_it(self) -> None:
        from deltaswamp.capability import ENGINE_METHODS
        from deltaswamp.engine.deltars import DeltaRsEngine
        from deltaswamp.engine.kernel import KernelEngine
        from deltaswamp.engine.sql import SqlEngine

        missing: list[str] = []
        for engine_cls in (KernelEngine, DeltaRsEngine, SqlEngine):
            for operation, method in ENGINE_METHODS.items():
                routing = OPERATION_ENGINES.get(operation)
                if routing is None or engine_cls.kind not in routing.engines:
                    continue
                if not callable(getattr(engine_cls, method, None)):
                    missing.append(
                        f"{engine_cls.__name__} is routed {operation.value} but has no {method}()"
                    )
        assert not missing, "\n".join(missing)

    def test_every_operation_has_a_public_method(self) -> None:
        """An operation nobody can invoke is a matrix entry, not a feature."""
        from deltaswamp import Connection, Table

        # Operations surfaced under a different public name, or via kwargs.
        aliases = {
            Operation.TIME_TRAVEL: "scan",
            Operation.REPLACE_WHERE: "overwrite",
            Operation.MERGE_SCHEMA: "append",
            Operation.CREATE: "create_table",
            Operation.ZORDER: "z_order",
            Operation.DETAIL: "detail",
            Operation.INCREMENTAL: "cdf",
            Operation.LOG_COMPACTION: "compact_logs",
            Operation.CONVERT: "convert_to_delta",
        }
        unreachable = []
        for operation in Operation:
            name = aliases.get(operation, operation.value)
            if not (hasattr(Table, name) or hasattr(Connection, name)):
                unreachable.append(operation.value)
        assert not unreachable, f"no public method for: {sorted(unreachable)}"


class TestCatalogCreateDoesNotOrphan:
    """Creating a *catalog* table must not silently write only a Delta log.

    Writing the log without calling the catalog's create API leaves an orphaned
    table: the files exist, Unity Catalog knows nothing, and the call looked
    like it worked.
    """

    def test_catalog_name_with_location_is_refused(self) -> None:
        pa = pytest.importorskip("pyarrow")
        from deltaswamp import Connection
        from deltaswamp.catalog.filesystem import FilesystemCatalog
        from deltaswamp.errors import UnreachableTableError

        conn = Connection(catalog=FilesystemCatalog(), router=router())
        with pytest.raises(UnreachableTableError, match="orphaned"):
            conn.create_table(
                "main.sales.orders",
                pa.schema([("id", pa.int64())]),
                location="s3://bucket/t",
            )

    def test_catalog_name_without_location_is_refused(self) -> None:
        pa = pytest.importorskip("pyarrow")
        from deltaswamp import Connection
        from deltaswamp.catalog.filesystem import FilesystemCatalog
        from deltaswamp.errors import UnreachableTableError

        conn = Connection(catalog=FilesystemCatalog(), router=router())
        with pytest.raises(UnreachableTableError, match="cannot register"):
            conn.create_table("main.sales.orders", pa.schema([("id", pa.int64())]))


class TestUnityCatalogPreflight:
    """Refusals decidable from Unity Catalog metadata, in the right order."""

    def test_iceberg_tables_without_an_engine_name_the_extra(self) -> None:
        got = router().capability(Operation.SCAN, table(data_source_format="ICEBERG"))
        assert not got.ok
        assert "Iceberg" in got.reason
        assert "deltaswamp[iceberg]" in got.remedy

    def test_iceberg_tables_without_a_rest_endpoint_say_so(self) -> None:
        got = router().capability(Operation.SCAN, table(data_source_format="ICEBERG"))
        assert "no Iceberg REST endpoint" in got.reason

    @pytest.mark.parametrize("fmt", ["PARQUET", "CSV", "JSON", "AVRO", "ORC"])
    def test_non_delta_formats_are_refused(self, fmt: str) -> None:
        got = router().capability(Operation.SCAN, table(data_source_format=fmt))
        assert not got.ok
        assert "not Delta" in got.reason

    @pytest.mark.parametrize("fmt", ["DELTA", "DELTA_UNIFORM_ICEBERG", None])
    def test_delta_shaped_formats_are_allowed(self, fmt: str | None) -> None:
        assert router().capability(Operation.SCAN, table(data_source_format=fmt)).ok

    def test_manifest_overrides_the_view_refusal(self) -> None:
        """Databricks can make an MV externally readable via
        pipelines.externalMetadata. The manifest is authoritative, so a type
        refusal must not win over it."""
        readable_mv = table(table_type=TableType.MATERIALIZED_VIEW, external_read_supported=True)
        assert router().capability(Operation.SCAN, readable_mv).ok

    def test_view_without_manifest_support_is_still_refused(self) -> None:
        got = router().capability(
            Operation.SCAN, table(table_type=TableType.VIEW, external_read_supported=None)
        )
        assert not got.ok
        assert "file surface" in got.reason

    def test_hive_metastore_catalog_is_sql_only(self) -> None:
        legacy = ResolvedTable(
            ref=parse_ref("hive_metastore.analytics.events"), location="s3://bucket/t"
        )
        got = router().capability(Operation.SCAN, legacy)
        assert not got.ok
        assert "hive_metastore" in got.reason

    @pytest.mark.parametrize(
        "operation",
        [
            Operation.SET_PROPERTIES,
            Operation.ADD_FEATURE,
            Operation.ADD_COLUMN,
            Operation.RENAME_COLUMN,
            Operation.ADD_CONSTRAINT,
        ],
    )
    def test_catalog_managed_alter_is_refused(self, operation: Operation) -> None:
        """UCCommitter rejects protocol and metadata changes past version 0, so
        these are not ours to make."""
        managed = table(reader_features=frozenset({"catalogManaged"}))
        got = router().capability(operation, managed)
        assert not got.ok
        assert "catalog-managed" in got.reason

    def test_catalog_managed_reads_and_appends_are_unaffected(self) -> None:
        managed = table(reader_features=frozenset({"catalogManaged"}))
        assert router().capability(Operation.SCAN, managed).ok


class TestWarehouseOnlyShapes:
    """Shapes no direct engine may serve go to the warehouse when the fallback is on."""

    def test_access_policy(self) -> None:
        t = table(external_read_supported=False, access_policy="row filter main.s.rf")
        assert accepting_router(fallback=True).capability(Operation.SCAN, t).engine is Engine.SQL
        refused = accepting_router(fallback=False).capability(Operation.SCAN, t)
        assert not refused.ok
        assert "row filter main.s.rf" in refused.reason

    def test_unreadable_log(self) -> None:
        t = table(open_error="Unsupported Delta table type: 'geometry(OGC:CRS84)'")
        assert accepting_router(fallback=True).capability(Operation.SCAN, t).engine is Engine.SQL

    def test_catalog_managed_alter(self) -> None:
        t = table(
            reader_features=frozenset({"catalogManaged"}),
            writer_features=frozenset({"catalogManaged", "inCommitTimestamp"}),
        )
        got = accepting_router(fallback=True).capability(Operation.ADD_COLUMN, t)
        assert got.engine is Engine.SQL
        assert not accepting_router(fallback=False).capability(Operation.ADD_COLUMN, t).ok

    def test_foreign_and_hive_metastore(self) -> None:
        foreign = table(table_type=TableType.FOREIGN)
        got = accepting_router(fallback=True).capability(Operation.SCAN, foreign)
        assert got.engine is Engine.SQL
        hive = table(
            TableRef(kind=RefKind.CATALOG, catalog="hive_metastore", schema="s", table="t"),
            data_source_format="DELTA",
        )
        assert accepting_router(fallback=True).capability(Operation.SCAN, hive).engine is Engine.SQL
        assert not accepting_router(fallback=False).capability(Operation.SCAN, hive).ok


class TestCollatedTables:
    def test_predicated_scans_go_to_the_warehouse(self) -> None:
        """Only the warehouse evaluates a predicate under the column's collation."""
        t = table(writer_features=frozenset({"collations"}))
        r = accepting_router(fallback=True)
        predicated = r.capability(Operation.SCAN, t, needs=frozenset({"predicates"}))
        assert predicated.engine is Engine.SQL
        assert r.capability(Operation.SCAN, t).engine is Engine.KERNEL


class TestTableLevelProhibitions:
    """Operations the table itself forbids, so no engine may claim them."""

    @pytest.mark.parametrize(
        "operation",
        [Operation.DELETE, Operation.UPDATE, Operation.OVERWRITE, Operation.REPLACE_WHERE],
    )
    def test_append_only(self, operation: Operation) -> None:
        t = table(properties={"delta.appendOnly": "true"})
        verdict = accepting_router(fallback=True).capability(operation, t)
        assert not verdict.ok
        assert "append-only" in verdict.reason
        assert accepting_router(fallback=True).capability(Operation.APPEND, t).ok

    @pytest.mark.parametrize("operation", [Operation.TIME_TRAVEL, Operation.CDF, Operation.RESTORE])
    def test_access_policy_blocks_old_versions_but_not_history(self, operation: Operation) -> None:
        t = table(external_read_supported=False, access_policy="row filter main.s.rf")
        assert not accepting_router(fallback=True).capability(operation, t).ok
        assert accepting_router(fallback=True).capability(Operation.HISTORY, t).ok
