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

import dataclasses
from dataclasses import dataclass, field

from .capability import (
    APPEND_ONLY_FORBIDDEN,
    COLUMN_MAPPING_REQUIRED,
    DATABRICKS_ONLY_OPERATIONS,
    DELTARS_DRY_RUN_OPERATIONS,
    DELTARS_OPERATION_EXEMPTIONS,
    DELTARS_VACUUM_DRY_RUN_EXEMPT,
    FEATURE_SUPPORT,
    KERNEL_LOG_WRITE_OPERATIONS,
    METADATA_OPERATIONS,
    OPERATION_ENGINES,
    OPERATION_FEATURE_BLOCKERS,
    READ_OPERATIONS,
    UNIFORM_STALE_WRITES,
    Capability,
    Operation,
    Support,
    TableFeature,
    feature_from_wire,
)
from .capability import (
    Engine as EngineKind,
)
from .catalog import ResolvedTable, TableType
from .errors import SQL_FALLBACK_REMEDY, FallbackRequiredError, UnreachableTableError
from .identity import RefKind

__all__ = ["Router"]

#: Operations UCCommitter refuses on a catalog-managed table past version 0.
_CATALOG_MANAGED_ALTER_OPS: frozenset[Operation] = METADATA_OPERATIONS | {
    Operation.DROP_FEATURE,
    Operation.ADD_CONSTRAINT,
    Operation.MERGE_SCHEMA,
}

#: Data writes a view, materialized view or metric view can never take: each is
#: a query over other tables, with no data of its own.
_QUERY_DEFINED = (TableType.VIEW, TableType.MATERIALIZED_VIEW, TableType.METRIC_VIEW)
_DATA_WRITES: frozenset[Operation] = frozenset(
    {
        Operation.APPEND,
        Operation.OVERWRITE,
        Operation.REPLACE_WHERE,
        Operation.DELETE,
        Operation.UPDATE,
        Operation.MERGE,
        Operation.MERGE_SCHEMA,
        Operation.RESTORE,
    }
)


def _is_uc_hive_metastore(table: ResolvedTable) -> bool:
    """Whether this is Databricks' legacy `hive_metastore` catalog.

    A table resolved from a self-hosted Hive Metastore (``hms://``) also carries
    the catalog name ``hive_metastore``, but its location comes straight from
    the metastore and direct engines read it; only the Unity Catalog one is
    unreachable without the warehouse.
    """
    return (table.ref.catalog or "").lower() == "hive_metastore" and table.ref.scheme != "hms"


#: Catalog reference schemes whose tables no Databricks SQL warehouse can name.
_NON_WAREHOUSE_SCHEMES: frozenset[str] = frozenset({"hms", "hive", "glue"})


def _warehouse_can_name(table: ResolvedTable) -> bool:
    """Whether a SQL warehouse could address this table at all.

    The warehouse takes a Unity Catalog three-part name. A storage path, a
    table from a self-hosted Hive metastore or from Glue has none, so telling
    the caller to enable the fallback for one sent them to an engine that then
    refused with "a SQL warehouse addresses tables by name".
    """
    if table.ref.kind is not RefKind.CATALOG:
        return False
    return (table.ref.scheme or "").lower() not in _NON_WAREHOUSE_SCHEMES


#: The remedy for a table the warehouse cannot name. Enabling the fallback on
#: its own changes nothing, so this is not a FallbackRequiredError.
_REGISTER_REMEDY = (
    "register the table in Unity Catalog (as an external table over its location) "
    "and address it by name through ds.connect(..., allow_sql_fallback=True)"
)


#: Open errors that say the log is absent or unreadable as data. The warehouse
#: reads the same storage, so it would fail the same way.
_UNFIXABLE_OPEN_ERRORS: tuple[str, ...] = (
    "tablenotfound",
    "not a delta table",
    "no such file or directory",
    "does not exist",
    "no files in log segment",
    "corrupt",
    "invalid json",
    "failed to parse",
    "has no such version",
    # A log missing or mangling what every reader needs: the warehouse reads
    # the same files, so enabling the fallback would fail the same way.
    "expected contiguous commit files",
    "no table metadata found",
    "no protocol found",
    "invalid protocol action",
    "unmasked nulls",
)


def _open_error_remedy(table: ResolvedTable) -> str:
    """The remedy for a table whose Delta log could not be opened."""
    error = (table.open_error or "").lower()
    if any(marker in error for marker in _UNFIXABLE_OPEN_ERRORS):
        return (
            f"check that {table.location or 'the table location'} holds a readable Delta "
            "log; the SQL warehouse reads the same storage and would fail the same way"
        )
    if _warehouse_can_name(table):
        return SQL_FALLBACK_REMEDY
    return ""


def _active_features(table: ResolvedTable) -> frozenset[TableFeature]:
    """The table's features as they act on a commit, not merely as listed.

    Column mapping on a legacy protocol (reader 2 / writer 5) is in no feature
    list, only in ``delta.columnMapping.mode``; and a listed feature that is
    switched off (mode ``none``, row tracking suspended) does not bind.
    """
    active = set(table.known_features)
    mode = str(table.properties.get("delta.columnMapping.mode", "none")).strip().lower()
    if mode in ("name", "id"):
        active.add(TableFeature.COLUMN_MAPPING)
    else:
        active.discard(TableFeature.COLUMN_MAPPING)
    if str(table.properties.get("delta.rowTrackingSuspended", "")).strip().lower() == "true":
        active.discard(TableFeature.ROW_TRACKING)
    return frozenset(active)


def _operation_blocker(
    kind: EngineKind, operation: Operation, table: ResolvedTable, engine: object = None
) -> str | None:
    """Why `kind` cannot run `operation` on this table although it handles every feature."""
    if (
        kind in _DIRECT_ENGINES
        and operation in APPEND_ONLY_FORBIDDEN
        and str(table.properties.get("delta.appendOnly", "")).strip().lower() == "true"
    ):
        return (
            "the table is append-only (delta.appendOnly=true), so no commit may remove "
            f"its data, and {operation.value} does"
        )
    if (
        kind in _DIRECT_ENGINES
        and operation in UNIFORM_STALE_WRITES
        and "iceberg"
        in str(table.properties.get("delta.universalFormat.enabledFormats", "")).lower()
    ):
        return (
            "the table has UniForm Iceberg metadata enabled, and only Databricks regenerates "
            "it after a write, so the Iceberg view would silently go stale"
        )
    if (
        kind in _DIRECT_ENGINES
        and operation in COLUMN_MAPPING_REQUIRED
        and TableFeature.COLUMN_MAPPING not in _active_features(table)
    ):
        return (
            f"{operation.value} without rewriting data needs column mapping, which the table "
            "does not have; set_properties({'delta.columnMapping.mode': 'name'}) first"
        )
    blocking = OPERATION_FEATURE_BLOCKERS.get((kind, operation), frozenset())
    hit = sorted(f.value for f in blocking & _active_features(table))
    if hit:
        return f"{operation.value} is not supported on a table with {', '.join(hit)}"
    if kind is EngineKind.KERNEL and operation in KERNEL_LOG_WRITE_OPERATIONS:
        writer = table.min_writer_version or 0
        if 3 <= writer <= 6:
            return (
                f"the table uses the legacy writer protocol version {writer}, which implies "
                "checkConstraints, and the kernel refuses to write its log"
            )
        unwritable = sorted(
            name
            for name in table.writer_features
            if (feature := feature_from_wire(name)) is None
            or FEATURE_SUPPORT[feature].kernel_write is Support.NO
        )
        if unwritable:
            return f"the kernel cannot write the log of a table with {', '.join(unwritable)}"
        has_invariants = getattr(engine, "_has_invariants", None)
        if (writer == 2 or "invariants" in table.writer_features) and callable(has_invariants):
            try:
                present = bool(has_invariants(table))
            except Exception:
                present = False
            if present:
                return "the table schema declares column invariants, which the kernel refuses"
    return None


def _exempted(
    kind: EngineKind, operation: Operation, table: ResolvedTable, shape: dict[str, object]
) -> ResolvedTable:
    """The table as `kind` should judge it for `operation`.

    delta-rs refuses a table carrying a feature it cannot *commit* to, but some
    operations never commit; for those the irrelevant features are hidden from
    its verdict (the operation itself still receives the real table).
    """
    if kind is not EngineKind.DELTARS:
        return table
    exempt = DELTARS_OPERATION_EXEMPTIONS.get(operation, frozenset())
    if operation in DELTARS_DRY_RUN_OPERATIONS and shape.get("dry_run") is True:
        exempt = exempt | DELTARS_VACUUM_DRY_RUN_EXEMPT
    if not exempt:
        return table
    names = {f.value for f in exempt}
    if not (table.features & names):
        return table
    return dataclasses.replace(
        table,
        reader_features=table.reader_features - names,
        writer_features=table.writer_features - names,
    )


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
            # The request shape still applies: a share that cannot serve it must
            # say so here rather than accept and then drop the requirement.
            unmet = sorted(
                need for need in needs if not getattr(sharing, f"supports_{need}", False)
            )
            if unmet:
                return Capability(
                    operation,
                    ok=False,
                    reason=f"{EngineKind.SHARING.value}: does not support {', '.join(unmet)}",
                )
            verdict: Capability = sharing.supports(operation, table, **shape)  # type: ignore[attr-defined]
            return verdict

        reasons: list[str] = []
        routing = OPERATION_ENGINES.get(operation)
        if routing is None:
            return Capability(operation, ok=False, reason=f"unknown operation {operation.value}")

        # Refusals `_catalog_level_block` waived only because the warehouse can
        # still serve the table. They still hold for every direct engine.
        direct_refusal = self._direct_refusal(operation, table)

        for kind in routing.engines:
            if kind in exclude:
                reasons.append(f"{kind.value}: already tried, and failed")
                continue
            engine = self.engines.get(kind)
            # Checked before "not configured": connect() only builds the SQL
            # engine when the fallback is on, so with it off the reason was
            # always "sql: engine not configured", which names no remedy.
            if kind is EngineKind.SQL and not _warehouse_can_name(table):
                # Enabling the fallback would not help: the warehouse takes a
                # Unity Catalog name, and this table has none.
                reasons.append(
                    "sql: a SQL warehouse addresses Unity Catalog tables by name, and this "
                    "table has none"
                )
                continue
            if kind is EngineKind.SQL and not self.allow_sql_fallback:
                reasons.append(
                    "sql: the SQL warehouse fallback is disabled. It is opt-in because "
                    "rerouting through a warehouse changes latency and cost by orders "
                    "of magnitude"
                )
                continue

            if engine is None:
                reasons.append(f"{kind.value}: engine not configured")
                continue

            # A direct engine reaches the files with a vended credential, and
            # the manifest has already said vending will refuse this table. The
            # engine cannot know that -- it would accept the call and fail
            # mid-flight on a CredentialError -- so it is skipped here and the
            # warehouse, which needs no vending, serves instead.
            if kind in _DIRECT_ENGINES and direct_refusal is not None:
                reasons.append(f"{kind.value}: {direct_refusal}")
                continue

            if kind in _DIRECT_ENGINES and not self._vendable(operation, table):
                reasons.append(
                    f"{kind.value}: Unity Catalog withdraws this table from credential "
                    "vending, and a direct engine cannot reach the files without it"
                )
                continue

            missing = sorted(
                need for need in needs if not getattr(engine, f"supports_{need}", False)
            )
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

            blocker = _operation_blocker(kind, operation, table, engine)
            if blocker is not None:
                reasons.append(f"{kind.value}: {blocker}")
                continue

            judged = _exempted(kind, operation, table, shape)
            result: Capability = engine.supports(operation, judged, **shape)  # type: ignore[attr-defined]
            if result.ok:
                return result
            reasons.append(f"{kind.value}: {result.reason}")

        remedy = ""
        if EngineKind.SQL in routing.engines and not table.is_shared:
            if not _warehouse_can_name(table):
                remedy = _REGISTER_REMEDY
            elif not self.allow_sql_fallback:
                remedy = (
                    "ds.connect(..., allow_sql_fallback=True) would route this to a SQL warehouse"
                )

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
        if capability.ok:
            engine = self.engines.get(capability.engine) if capability.engine else None
            if engine is not None:
                return engine
            raise UnreachableTableError(
                operation.value,
                f"an engine accepted the request as "
                f"{capability.engine.value if capability.engine else 'an unnamed engine'}, "
                "which is not configured on this connection",
            )

        # Either the loop reached the warehouse and found it switched off, or a
        # catalog-level refusal names the fallback as the one thing that helps.
        fallback_would_help = "fallback is disabled" in capability.reason or (
            not self.allow_sql_fallback
            and "allow_sql_fallback" in capability.remedy
            and capability.remedy != _REGISTER_REMEDY
        )
        error = FallbackRequiredError if fallback_would_help else UnreachableTableError
        raise error(operation.value, capability.reason, capability.remedy or None)

    def capabilities(self, table: ResolvedTable) -> dict[Operation, Capability]:
        """Every operation's verdict. This is `Table.capabilities()`."""
        return {op: self.capability(op, table) for op in Operation}

    # ------------------------------------------------------------- pre-flight

    @staticmethod
    def _direct_refusal(operation: Operation, table: ResolvedTable) -> str | None:
        """Why no direct engine may touch this table, whatever the fallback says.

        `_catalog_level_block` turns each of these into a refusal when the SQL
        warehouse is unavailable. When it is available the block steps aside so
        the warehouse can serve, but the direct engines must still be skipped:
        otherwise one accepts on an empty or irrelevant feature list and fails
        mid-flight, or worse, commits to a table whose protocol it never read.
        """
        if table.is_shared:
            return None  # the sharing engine owns the verdict.
        if not table.is_delta:
            return "the table is not Delta, so it has no Delta log"
        if table.is_view_like and not (table.external_read_supported is True and table.location):
            return "the relation is a view-like object with no directly readable file surface"
        if table.table_type is TableType.FOREIGN:
            return "the table is FOREIGN (federated), which credential vending does not cover"
        if _is_uc_hive_metastore(table):
            return "the legacy hive_metastore catalog cannot be credential-vended"
        if table.is_catalog_managed and operation in _CATALOG_MANAGED_ALTER_OPS:
            return (
                "the table is catalog-managed, and its commit protocol refuses protocol and "
                "metadata changes after version 0"
            )
        if table.open_error is not None and operation is not Operation.CREATE:
            return f"the table's Delta log could not be read ({table.open_error})"
        return None

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
        if _is_uc_hive_metastore(table):
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
            if table.is_iceberg and EngineKind.ICEBERG in self.engines and table.iceberg_rest_uri:
                return None  # the Iceberg engine serves it through the catalog.
            if sql_fallback:
                # The warehouse queries any format; the loop keeps the direct
                # engines, which only know Delta logs, away from it.
                return None
            if table.is_iceberg:
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
                remedy=(
                    "ds.connect(..., allow_sql_fallback=True) can still query it"
                    if _warehouse_can_name(table)
                    else ""
                ),
            )

        # The manifest can say a materialized view or streaming table is
        # externally readable (pipelines.externalMetadata), so it overrides the
        # type -- but only when there is actually somewhere to read from.
        # Databricks returns no storage_location for these, and without one no
        # direct engine can do anything, so saying "the kernel found no
        # location" five times is worse than naming the real cause once.
        if table.table_type in _QUERY_DEFINED and operation in _DATA_WRITES:
            # Named the SQL fallback as the remedy, and with it enabled sent
            # the write to a warehouse that refuses DML on a view -- or, for a
            # materialized view whose manifest says readable, to a direct
            # engine that would write into the view's own storage.
            kind = table.table_type.value
            refresh = table.table_type is TableType.MATERIALIZED_VIEW
            return Capability(
                operation,
                ok=False,
                reason=f"the table is a {kind}: it is defined by a query over other "
                f"tables and holds no data of its own, so there is nothing to "
                f"{operation.value} anywhere",
                remedy="write to the tables it reads from"
                + ("; then refresh() it" if refresh else ""),
            )
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
            and str(table.properties.get("delta.appendOnly", "")).strip().lower() == "true"
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
                remedy=_open_error_remedy(table),
            )

        if operation in DATABRICKS_ONLY_OPERATIONS and not _warehouse_can_name(table):
            return Capability(
                operation,
                ok=False,
                reason=(
                    f"{operation.value} has no open-source implementation in either delta-rs "
                    "or delta-kernel; it exists only in Databricks, and a SQL warehouse can "
                    "only reach tables by their Unity Catalog name"
                ),
                remedy=_REGISTER_REMEDY,
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
