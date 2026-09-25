"""Regression tests for defects found auditing capability.py and router.py."""

from __future__ import annotations

import pytest
from deltaswamp.capability import (
    FEATURE_DEPENDENCIES,
    FEATURE_SUPPORT,
    Capability,
    Engine,
    Operation,
    Support,
    TableFeature,
    feature_from_wire,
)
from deltaswamp.catalog import ResolvedTable, TableType
from deltaswamp.errors import FallbackRequiredError, UnreachableTableError
from deltaswamp.identity import parse_ref
from deltaswamp.router import Router


class YesEngine:
    """Says yes to everything, and records whether it was asked."""

    supports_predicates = True
    supports_distributed_scan = True

    def __init__(self, kind: Engine) -> None:
        self.kind = kind
        self.asked: list[Operation] = []

    def supports(self, operation: Operation, table: ResolvedTable, **_: object) -> Capability:
        self.asked.append(operation)
        return Capability(operation, ok=True, engine=self.kind)


def _table(ref: str = "main.sales.orders", **kwargs: object) -> ResolvedTable:
    kwargs.setdefault("location", "s3://bucket/t")
    return ResolvedTable(ref=parse_ref(ref), **kwargs)  # type: ignore[arg-type]


def _router(
    *, fallback: bool = True, sharing: bool = False
) -> tuple[Router, dict[Engine, YesEngine]]:
    engines = {k: YesEngine(k) for k in (Engine.KERNEL, Engine.DELTARS, Engine.SQL)}
    if sharing:
        engines[Engine.SHARING] = YesEngine(Engine.SHARING)
    return Router(engines=dict(engines), allow_sql_fallback=fallback), engines


# ---------------------------------------------------------------- capability


def test_timestamp_without_timezone_alias_is_known() -> None:
    """kernel parses the legacy 'timestampWithoutTimezone' as timestampNtz."""
    assert feature_from_wire("timestampWithoutTimezone") is TableFeature.TIMESTAMP_NTZ
    t = _table(reader_features=frozenset({"timestampWithoutTimezone"}))
    assert t.unknown_features == frozenset()


@pytest.mark.parametrize(
    "feature", [TableFeature.GEOSPATIAL, TableFeature.ADAPTIVE_METADATA_PREVIEW]
)
def test_dev_gated_kernel_features_are_not_claimed_readable(feature: TableFeature) -> None:
    """The native crate does not enable geo-type-in-dev / adaptive-metadata-in-dev,
    so the compiled kernel refuses to scan these ReaderWriter features."""
    row = FEATURE_SUPPORT[feature]
    assert row.kernel_read is Support.NO
    assert row.kernel_write is Support.NO


def test_variant_shredding_depends_on_variant_type() -> None:
    assert TableFeature.VARIANT_TYPE in FEATURE_DEPENDENCIES[TableFeature.VARIANT_SHREDDING]


# -------------------------------------------------------------------- router


def test_self_hosted_hms_tables_are_not_refused_as_uc_hive_metastore() -> None:
    r, _ = _router(fallback=False)
    t = _table("hms://metastore:9083/analytics/events")
    assert t.ref.catalog == "hive_metastore"
    got = r.capability(Operation.SCAN, t)
    assert got.ok and got.engine is Engine.KERNEL


def test_uc_hive_metastore_routes_to_sql_when_fallback_enabled() -> None:
    r, engines = _router(fallback=True)
    got = r.capability(Operation.SCAN, _table("hive_metastore.analytics.events"))
    assert got.ok and got.engine is Engine.SQL
    assert not engines[Engine.KERNEL].asked


def test_foreign_table_routes_to_sql_when_fallback_enabled() -> None:
    r, engines = _router(fallback=True)
    got = r.capability(Operation.SCAN, _table(table_type=TableType.FOREIGN))
    assert got.ok and got.engine is Engine.SQL
    assert not engines[Engine.KERNEL].asked and not engines[Engine.DELTARS].asked


def test_catalog_managed_alter_routes_to_sql_when_fallback_enabled() -> None:
    r, engines = _router(fallback=True)
    t = _table(reader_features=frozenset({"catalogManaged"}))
    got = r.capability(Operation.ADD_COLUMN, t)
    assert got.ok and got.engine is Engine.SQL
    assert not engines[Engine.KERNEL].asked and not engines[Engine.DELTARS].asked


def test_catalog_managed_alter_still_refused_without_fallback() -> None:
    r, _ = _router(fallback=False)
    got = r.capability(Operation.ADD_COLUMN, _table(reader_features=frozenset({"catalogManaged"})))
    assert not got.ok and "catalog-managed" in got.reason


def test_open_error_keeps_direct_engines_away_even_with_fallback() -> None:
    """An unreadable log means empty feature lists; a direct engine must not
    accept a write routed on them."""
    r, engines = _router(fallback=True)
    got = r.capability(Operation.APPEND, _table(open_error="boom"))
    assert got.ok and got.engine is Engine.SQL
    assert not engines[Engine.DELTARS].asked and not engines[Engine.KERNEL].asked


def test_non_delta_table_routes_to_sql_when_fallback_enabled() -> None:
    r, engines = _router(fallback=True)
    got = r.capability(Operation.SCAN, _table(data_source_format="PARQUET"))
    assert got.ok and got.engine is Engine.SQL
    assert not engines[Engine.DELTARS].asked


def test_view_with_location_but_no_manifest_skips_direct_engines_with_fallback() -> None:
    r, engines = _router(fallback=True)
    got = r.capability(Operation.SCAN, _table(table_type=TableType.MATERIALIZED_VIEW))
    assert got.ok and got.engine is Engine.SQL
    assert not engines[Engine.KERNEL].asked


def test_shared_table_honours_needs() -> None:
    r, engines = _router(fallback=False, sharing=True)
    engines[Engine.SHARING].supports_distributed_scan = False
    t = _table(sharing_profile="{}")
    assert t.is_shared
    got = r.capability(Operation.SCAN, t, needs=frozenset({"distributed_scan"}))
    assert not got.ok and "distributed_scan" in got.reason
    assert not engines[Engine.SHARING].asked


def test_catalog_level_fallback_refusal_raises_fallback_required() -> None:
    r, _ = _router(fallback=False)
    with pytest.raises(FallbackRequiredError):
        r.engine_for(Operation.SCAN, _table(table_type=TableType.VIEW))


def test_engine_for_explains_an_ok_verdict_from_an_unconfigured_engine() -> None:
    class Liar(YesEngine):
        def supports(self, operation: Operation, table: ResolvedTable, **_: object) -> Capability:
            return Capability(operation, ok=True, engine=Engine.ICEBERG)

    r = Router(engines={Engine.KERNEL: Liar(Engine.KERNEL)})
    with pytest.raises(UnreachableTableError, match="iceberg"):
        r.engine_for(Operation.SCAN, _table())


def test_missing_needs_are_reported_in_stable_order() -> None:
    class Bare:
        def supports(self, operation: Operation, table: ResolvedTable, **_: object) -> Capability:
            return Capability(operation, ok=True, engine=Engine.KERNEL)

    r = Router(engines={Engine.KERNEL: Bare()})
    got = r.capability(Operation.SCAN, _table(), needs=frozenset({"zeta", "alpha", "mid"}))
    assert "alpha, mid, zeta" in got.reason


@pytest.mark.parametrize("operation", [Operation.SCAN, Operation.APPEND])
def test_iceberg_table_never_reaches_deltars(operation: Operation) -> None:
    """delta-rs does not check the format; it would open (or write a _delta_log
    into) an Iceberg table's location ahead of the Iceberg engine."""
    engines = {k: YesEngine(k) for k in (Engine.KERNEL, Engine.DELTARS, Engine.ICEBERG)}
    r = Router(engines=dict(engines))
    t = _table(data_source_format="ICEBERG", iceberg_rest_uri="https://ws/iceberg")
    got = r.capability(operation, t)
    assert got.ok and got.engine is Engine.ICEBERG
    assert not engines[Engine.DELTARS].asked and not engines[Engine.KERNEL].asked
