"""Choosing an engine per operation, and explaining every refusal.

The router is where "you shouldn't have to care what connects to what" is either
kept honest or quietly becomes a lie. It follows three rules:

1. Ask each candidate engine, in the preference order the conformance matrix
   defines, and take the first that says yes.
2. Never guess. If the catalog's capability manifest says a table is not
   eligible for external access, that is decisive -- a row filter removes an
   otherwise ordinary managed Delta table from the eligible set, and no amount
   of protocol inspection reveals it.
3. When nothing can serve the request, raise an error carrying every reason
   collected along the way, plus the remedy if one exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .capability import (
    DATABRICKS_ONLY_OPERATIONS,
    METADATA_OPERATIONS,
    OPERATION_ENGINES,
    READ_OPERATIONS,
    Capability,
    Operation,
)
from .capability import (
    Engine as EngineKind,
)
from .catalog import ResolvedTable, TableType
from .errors import FallbackRequiredError, UnreachableTableError

__all__ = ["Router"]

#: Operations UCCommitter refuses on a catalog-managed table past version 0.
_CATALOG_MANAGED_ALTER_OPS: frozenset[Operation] = METADATA_OPERATIONS | {
    Operation.DROP_FEATURE,
    Operation.ADD_CONSTRAINT,
    Operation.MERGE_SCHEMA,
}


@dataclass
class Router:
    """Routes operations to engines for one connection."""

    engines: dict[EngineKind, object] = field(default_factory=dict)
    allow_sql_fallback: bool = False

    def capability(
        self,
        operation: Operation,
        table: ResolvedTable,
        *,
        needs: frozenset[str] = frozenset(),
        **shape: object,
    ) -> Capability:
        """Decide how `operation` would be served, without performing it.

        `needs` names request-shape requirements such as ``predicates`` or
        ``timestamp_travel``. An engine that cannot serve the shape is skipped
        here, rather than accepted and then raising once the call is underway.
        """
        blocked = self._catalog_level_block(operation, table)
        if blocked is not None:
            return blocked

        # A shared table has exactly one way in, so its engine's verdict is the
        # whole answer -- including "shares are read-only" for a write, which is
        # more useful than every other engine reporting it has no location.
        sharing = self.engines.get(EngineKind.SHARING)
        if table.is_shared and sharing is not None:
            verdict: Capability = sharing.supports(operation, table, **shape)  # type: ignore[attr-defined]
            return verdict

        reasons: list[str] = []
        routing = OPERATION_ENGINES.get(operation)
        if routing is None:
            return Capability(operation, ok=False, reason=f"unknown operation {operation.value}")

        for kind in routing.engines:
            engine = self.engines.get(kind)
            if engine is None:
                reasons.append(f"{kind.value}: engine not configured")
                continue

            if kind is EngineKind.SQL and not self.allow_sql_fallback:
                reasons.append(
                    "sql: the SQL warehouse fallback is disabled. It is opt-in because "
                    "rerouting through a warehouse changes latency and cost by orders "
                    "of magnitude"
                )
                continue

            missing = [need for need in needs if not getattr(engine, f"supports_{need}", False)]
            if missing:
                reasons.append(f"{kind.value}: does not support {', '.join(missing)}")
                continue

            result: Capability = engine.supports(operation, table, **shape)  # type: ignore[attr-defined]
            if result.ok:
                return result
            reasons.append(f"{kind.value}: {result.reason}")

        remedy = ""
        if not self.allow_sql_fallback and EngineKind.SQL in routing.engines:
            remedy = "ds.connect(..., allow_sql_fallback=True) would route this to a SQL warehouse"

        return Capability(
            operation,
            ok=False,
            reason="; ".join(reasons) if reasons else "no engine can serve this operation",
            remedy=remedy,
        )

    def engine_for(
        self,
        operation: Operation,
        table: ResolvedTable,
        *,
        needs: frozenset[str] = frozenset(),
        **shape: object,
    ) -> object:
        """Return the engine that will serve `operation`, or raise explaining why not."""
        capability = self.capability(operation, table, needs=needs, **shape)
        if capability.ok and capability.engine is not None:
            engine = self.engines.get(capability.engine)
            if engine is not None:
                return engine

        error = (
            FallbackRequiredError
            if "fallback is disabled" in capability.reason
            else UnreachableTableError
        )
        raise error(operation.value, capability.reason, capability.remedy or None)

    def capabilities(self, table: ResolvedTable) -> dict[Operation, Capability]:
        """Every operation's verdict. This is `Table.capabilities()`."""
        return {op: self.capability(op, table) for op in Operation}

    # ------------------------------------------------------------- pre-flight

    def _catalog_level_block(self, operation: Operation, table: ResolvedTable) -> Capability | None:
        """Refusals decidable from catalog metadata alone, before any log read."""

        # The capability manifest is authoritative, so it is consulted BEFORE the
        # type-based refusals. Databricks can make a materialized view or
        # streaming table externally readable (pipelines.externalMetadata), and
        # refusing on table_type first would contradict the manifest.
        manifest_says_readable = table.external_read_supported is True

        # A shared table is served by the Delta Sharing engine or not at all:
        # its files are presigned URLs, so nothing else has a way in.
        if table.is_shared:
            if EngineKind.SHARING in self.engines:
                return None
            return Capability(
                operation,
                ok=False,
                reason="the table is reached through Delta Sharing, and the sharing engine "
                "is not available",
                remedy="pip install 'deltaswamp[sharing]'",
            )

        # A non-Delta table has no Delta log to read, whatever else is true.
        if not table.is_delta:
            fmt = table.data_source_format or "unknown"
            if table.is_iceberg:
                if EngineKind.ICEBERG in self.engines and table.iceberg_rest_uri:
                    return None  # the Iceberg engine serves it through the catalog.
                return Capability(
                    operation,
                    ok=False,
                    reason=(
                        f"the table is Iceberg ({fmt}), not Delta, and no Iceberg engine is "
                        "available for it"
                        + (
                            ""
                            if table.iceberg_rest_uri
                            else " (its catalog advertises no Iceberg REST endpoint)"
                        )
                    ),
                    remedy="pip install 'deltaswamp[iceberg]' to read it through the catalog's "
                    "Iceberg REST endpoint with PyIceberg",
                )
            return Capability(
                operation,
                ok=False,
                reason=f"the table's format is {fmt}, not Delta, so it has no Delta log",
                remedy="ds.connect(..., allow_sql_fallback=True) can still query it",
            )

        if table.is_view_like and not manifest_says_readable:
            kind = table.table_type.value if table.table_type else "view"
            if self.allow_sql_fallback and EngineKind.SQL in self.engines:
                return None  # SQL can query it; let the normal path handle it.
            return Capability(
                operation,
                ok=False,
                reason=(
                    f"the table is a {kind}, which has no directly readable file surface. "
                    "Credential vending refuses views, materialized views, metric views "
                    "and streaming tables"
                ),
                remedy="ds.connect(..., allow_sql_fallback=True)",
            )

        if table.table_type is TableType.FOREIGN:
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table is FOREIGN (federated). Depending on subtype its data lives "
                    "in another system entirely, or is reachable only through the Iceberg "
                    "REST catalog; credential vending does not cover it"
                ),
                remedy="ds.connect(..., allow_sql_fallback=True)",
            )

        # The legacy hive_metastore catalog is not a Unity Catalog securable, so
        # it can be neither vended nor reached through the UC Delta API.
        if (table.ref.catalog or "").lower() == "hive_metastore":
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table is in the legacy hive_metastore catalog, which is not a "
                    "Unity Catalog securable and cannot be credential-vended"
                ),
                remedy="ds.connect(..., allow_sql_fallback=True)",
            )

        # UCCommitter rejects protocol, metadata and clustering-domain changes at
        # version >= 1, so ALTER on a catalog-managed table is not ours to make.
        if table.is_catalog_managed and operation in _CATALOG_MANAGED_ALTER_OPS:
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table is catalog-managed, and its commit protocol refuses "
                    "protocol, metadata and clustering changes after version 0. Schema "
                    "and property changes have to go through the catalog's own APIs"
                ),
                remedy="ds.connect(..., allow_sql_fallback=True)",
            )

        if table.external_read_supported is False:
            return Capability(
                operation,
                ok=False,
                reason=(
                    "Unity Catalog reports no external-engine read support for this table "
                    "(HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT absent). The usual cause is a "
                    "row filter or column mask, which makes credential vending refuse the "
                    "table outright"
                ),
                remedy="ds.connect(..., allow_sql_fallback=True)",
            )

        writing = operation not in READ_OPERATIONS
        if writing and table.external_write_supported is False:
            return Capability(
                operation,
                ok=False,
                reason=(
                    "Unity Catalog reports no external-engine write support for this table "
                    "(HAS_DIRECT_EXTERNAL_ENGINE_WRITE_SUPPORT absent)"
                ),
                remedy="ds.connect(..., allow_sql_fallback=True)",
            )

        # A table we could not open is not a table we can route. CREATE is
        # exempt: there is nothing to open yet.
        sql_available = self.allow_sql_fallback and EngineKind.SQL in self.engines
        if table.open_error is not None and operation is not Operation.CREATE and not sql_available:
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table's Delta log could not be read, so no direct engine can "
                    f"serve this ({table.open_error})"
                ),
                remedy="ds.connect(..., allow_sql_fallback=True)",
            )

        if operation in DATABRICKS_ONLY_OPERATIONS and not self.allow_sql_fallback:
            return Capability(
                operation,
                ok=False,
                reason=(
                    f"{operation.value} has no open-source implementation in either delta-rs "
                    "or delta-kernel; it exists only in Databricks"
                ),
                remedy="ds.connect(..., allow_sql_fallback=True)",
            )

        return None
