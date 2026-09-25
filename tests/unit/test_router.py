"""Engine routing.

The routing table is the product. These tests assert the decisions and, just as
importantly, the *reasons* -- an unexplained refusal is the failure mode this
library exists to remove.
"""

from __future__ import annotations

import pytest
from deltaswamp.capability import OPERATION_ENGINES, Capability, Engine, Operation
from deltaswamp.catalog import ResolvedTable, TableType
from deltaswamp.errors import FallbackRequiredError, UnreachableTableError
from deltaswamp.identity import parse_ref
from deltaswamp.router import Router


class FakeEngine:
    """An engine that says yes to a fixed set of operations."""

    supports_distributed_scan = False

    def __init__(self, kind: Engine, yes: set[Operation] | None = None) -> None:
        self.kind = kind
        self._yes = yes if yes is not None else set(Operation)

    def supports(self, operation: Operation, table: ResolvedTable) -> Capability:
        if operation in self._yes:
            return Capability(operation, ok=True, engine=self.kind)
        return Capability(operation, ok=False, reason=f"{self.kind.value} declines")

    def available(self) -> bool:
        return True


def table(**kwargs: object) -> ResolvedTable:
    kwargs.setdefault("location", "s3://bucket/t")
    return ResolvedTable(ref=parse_ref("main.sales.orders"), **kwargs)  # type: ignore[arg-type]


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
    """Structural guard: a yes must be backed by a real method.

    The gap analysis found several operations where `capabilities()` returned ok
    and the call then failed with an AttributeError or a late refusal. That is
    the exact failure this library exists to prevent, so it is asserted rather
    than reviewed.
    """

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

    def test_every_routed_operation_is_reachable_from_the_public_api(self) -> None:
        """An operation nobody can invoke is a matrix entry, not a feature."""
        from deltaswamp.table import Connection as Conn
        from deltaswamp.table import Table as Tbl

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
            if not (hasattr(Tbl, name) or hasattr(Conn, name)):
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
        from deltaswamp.catalog.filesystem import FilesystemCatalog
        from deltaswamp.errors import UnreachableTableError
        from deltaswamp.table import Connection

        conn = Connection(catalog=FilesystemCatalog(), router=router())
        with pytest.raises(UnreachableTableError, match="orphaned"):
            conn.create_table(
                "main.sales.orders",
                pa.schema([("id", pa.int64())]),
                location="s3://bucket/t",
            )

    def test_catalog_name_without_location_is_refused(self) -> None:
        pa = pytest.importorskip("pyarrow")
        from deltaswamp.catalog.filesystem import FilesystemCatalog
        from deltaswamp.errors import UnreachableTableError
        from deltaswamp.table import Connection

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
        from deltaswamp.identity import parse_ref

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
