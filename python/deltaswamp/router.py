"""Choosing an engine per operation, and explaining every refusal.

The router follows three rules:

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
from .errors import SQL_FALLBACK_REMEDY, FallbackRequiredError, UnreachableTableError

__all__ = ["Router"]

#: Operations UCCommitter refuses on a catalog-managed table past version 0.
_CATALOG_MANAGED_ALTER_OPS: frozenset[Operation] = METADATA_OPERATIONS | {
    Operation.DROP_FEATURE,
    Operation.ADD_CONSTRAINT,
    Operation.MERGE_SCHEMA,
}


#: Engines that read and write storage directly, with a vended credential. The
#: SQL warehouse is not one: it runs inside Databricks.
_DIRECT_ENGINES: frozenset[EngineKind] = frozenset({EngineKind.KERNEL, EngineKind.DELTARS})
_COLLATIONS: frozenset[str] = frozenset({"collations", "collations-preview"})
_APPEND_ONLY_FORBIDS: frozenset[Operation] = frozenset(
    {Operation.DELETE, Operation.UPDATE, Operation.OVERWRITE, Operation.REPLACE_WHERE}
)
_ACCESS_POLICY_FORBIDS: frozenset[Operation] = frozenset(
    {Operation.TIME_TRAVEL, Operation.CDF, Operation.RESTORE}
)


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
        exclude: frozenset[EngineKind] = frozenset(),
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
            if kind in exclude:
                reasons.append(f"{kind.value}: already tried, and failed")
                continue
            engine = self.engines.get(kind)
            if engine is None:
                reasons.append(f"{kind.value}: engine not configured")
                continue

            if kind is not EngineKind.SQL:
                warehouse_only = self._warehouse_only(operation, table)
                if warehouse_only is not None:
                    reasons.append(f"{kind.value}: {warehouse_only}")
                    continue

            if kind is EngineKind.SQL and not self.allow_sql_fallback:
                reasons.append(
                    "sql: the SQL warehouse fallback is disabled. It is opt-in because "
                    "rerouting through a warehouse changes latency and cost by orders "
                    "of magnitude"
                )
                continue

            # A direct engine reaches the files with a vended credential, and
            # the manifest has already said vending will refuse this table. The
            # engine cannot know that -- it would accept the call and fail
            # mid-flight on a CredentialError -- so it is skipped here and the
            # warehouse, which needs no vending, serves instead.
            if kind in _DIRECT_ENGINES and not self._vendable(operation, table):
                reasons.append(
                    f"{kind.value}: Unity Catalog withdraws this table from credential "
                    "vending, and a direct engine cannot reach the files without it"
                )
                continue

            # Nor can a direct engine serve a table whose log it failed to read.
            # The catalog-level check lets this through when the warehouse is
            # available, so it has to be enforced per engine here: otherwise the
            # kernel is chosen on an empty feature list and fails the same way.
            if (
                kind in _DIRECT_ENGINES
                and table.open_error is not None
                and operation is not Operation.CREATE
            ):
                reasons.append(
                    f"{kind.value}: the Delta log could not be read ({table.open_error})"
                )
                continue

            missing = [need for need in needs if not getattr(engine, f"supports_{need}", False)]
            if missing:
                reasons.append(f"{kind.value}: does not support {', '.join(missing)}")
                continue

            if "predicates" in needs and kind in _DIRECT_ENGINES and table.features & _COLLATIONS:
                # A wrong answer, not an error: both engines compare bytes, so
                # `name = 'oslo'` misses 'Oslo' in a UTF8_LCASE column, and file
                # skipping on the same bounds can drop matching files outright.
                reasons.append(
                    f"{kind.value}: the table has collated columns, and a direct engine "
                    "evaluates predicates by byte order rather than by the collation"
                )
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
        exclude: frozenset[EngineKind] = frozenset(),
        **shape: object,
    ) -> object:
        """Return the engine that will serve `operation`, or raise explaining why not."""
        capability = self.capability(operation, table, needs=needs, exclude=exclude, **shape)
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

    @staticmethod
    def _vendable(operation: Operation, table: ResolvedTable) -> bool:
        """Whether credential vending will serve `operation` on this table.

        `None` means the catalog was never asked, which is not a refusal.
        """
        if table.external_read_supported is False:
            return False
        return not (operation not in READ_OPERATIONS and table.external_write_supported is False)

    @staticmethod
    def _warehouse_only(operation: Operation, table: ResolvedTable) -> str | None:
        """Why only the warehouse can serve this, or None.

        Kept apart from the refusal itself: with the fallback on, these shapes
        are served by SQL rather than refused, so the router skips every other
        engine instead.
        """
        if table.table_type is TableType.FOREIGN:
            return (
                "the table is FOREIGN (federated). Depending on subtype its data lives "
                "in another system entirely, or is reachable only through the Iceberg "
                "REST catalog; credential vending does not cover it"
            )
        # The legacy hive_metastore catalog is not a Unity Catalog securable, so
        # it can be neither vended nor reached through the UC Delta API.
        if (table.ref.catalog or "").lower() == "hive_metastore":
            return (
                "the table is in the legacy hive_metastore catalog, which is not a "
                "Unity Catalog securable and cannot be credential-vended"
            )
        # UCCommitter rejects protocol, metadata and clustering-domain changes at
        # version >= 1, so ALTER on a catalog-managed table is not ours to make.
        if table.is_catalog_managed and operation in _CATALOG_MANAGED_ALTER_OPS:
            return (
                "the table is catalog-managed, and its commit protocol refuses "
                "protocol, metadata and clustering changes after version 0. Schema "
                "and property changes have to go through the catalog's own APIs"
            )
        return None

    def _catalog_level_block(self, operation: Operation, table: ResolvedTable) -> Capability | None:
        """Refusals decidable from catalog metadata alone, before any log read."""

        # The capability manifest is authoritative, so it is consulted BEFORE the
        # type-based refusals. Databricks can make a materialized view or
        # streaming table externally readable (pipelines.externalMetadata), and
        # refusing on table_type first would contradict the manifest.
        manifest_says_readable = table.external_read_supported is True
        #: Whether a SQL warehouse is actually reachable for this connection.
        sql_fallback = self.allow_sql_fallback and EngineKind.SQL in self.engines

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

        # The manifest can say a materialized view or streaming table is
        # externally readable (pipelines.externalMetadata), so it overrides the
        # type -- but only when there is actually somewhere to read from.
        # Databricks returns no storage_location for these, and without one no
        # direct engine can do anything, so saying "the kernel found no
        # location" five times is worse than naming the real cause once.
        if table.is_view_like and not (manifest_says_readable and table.location):
            kind = table.table_type.value if table.table_type else "view"
            if sql_fallback:
                return None  # SQL can query it; let the normal path handle it.
            detail = (
                "Credential vending refuses views, materialized views, metric views "
                "and streaming tables"
            )
            if manifest_says_readable:
                detail = (
                    "Unity Catalog reports external read support for it but exposes no "
                    "storage location, so there are no files any direct engine can open"
                )
            return Capability(
                operation,
                ok=False,
                reason=(
                    f"the table is a {kind}, which has no directly readable file surface. {detail}"
                ),
                remedy=SQL_FALLBACK_REMEDY,
            )

        # Refusals no engine can get around, because the table itself forbids
        # the operation.
        if (
            operation in _APPEND_ONLY_FORBIDS
            and table.properties.get("delta.appendOnly", "").lower() == "true"
        ):
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table is append-only (delta.appendOnly=true), which forbids "
                    "removing or rewriting rows"
                ),
                remedy="t.set_properties({'delta.appendOnly': 'false'}) if that is intended",
            )
        if table.access_policy and operation in _ACCESS_POLICY_FORBIDS:
            return Capability(
                operation,
                ok=False,
                reason=(
                    f"the table has {table.access_policy}, and Databricks refuses time "
                    "travel, change data feed and RESTORE on a table with a row filter or "
                    "column mask, since earlier versions would escape the policy"
                ),
            )

        warehouse_only = self._warehouse_only(operation, table)
        if warehouse_only is not None and not sql_fallback:
            return Capability(
                operation,
                ok=False,
                reason=warehouse_only,
                remedy=SQL_FALLBACK_REMEDY,
            )

        # Both manifest flags describe DIRECT EXTERNAL ENGINE access: vending a
        # credential and touching the files yourself. A SQL warehouse is neither
        # -- it runs inside Databricks, where the row filter or mask that
        # withdrew the table from vending is simply evaluated. So these refusals
        # are conditional on the fallback being unavailable. They used to fire
        # unconditionally while naming `allow_sql_fallback=True` as the remedy,
        # which meant following the advice changed nothing.
        if table.external_read_supported is False and not sql_fallback:
            reason = (
                f"the table has {table.access_policy}, and Unity Catalog vends no "
                "credentials for tables with row filters or column masks"
                if table.access_policy
                else "Unity Catalog reports no external-engine read support for this table "
                "(HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT absent)"
            )
            return Capability(
                operation,
                ok=False,
                reason=reason,
                remedy=SQL_FALLBACK_REMEDY,
            )

        writing = operation not in READ_OPERATIONS
        if writing and table.external_write_supported is False and not sql_fallback:
            return Capability(
                operation,
                ok=False,
                reason=(
                    "Unity Catalog reports no external-engine write support for this table "
                    "(HAS_DIRECT_EXTERNAL_ENGINE_WRITE_SUPPORT absent)"
                ),
                remedy=SQL_FALLBACK_REMEDY,
            )

        # A table we could not open is not a table we can route. CREATE is
        # exempt: there is nothing to open yet.
        if table.open_error is not None and operation is not Operation.CREATE and not sql_fallback:
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table's Delta log could not be read, so no direct engine can "
                    f"serve this ({table.open_error})"
                ),
                remedy=SQL_FALLBACK_REMEDY,
            )

        if operation in DATABRICKS_ONLY_OPERATIONS and not self.allow_sql_fallback:
            return Capability(
                operation,
                ok=False,
                reason=(
                    f"{operation.value} has no open-source implementation in either delta-rs "
                    "or delta-kernel; it exists only in Databricks"
                ),
                remedy=SQL_FALLBACK_REMEDY,
            )

        return None
