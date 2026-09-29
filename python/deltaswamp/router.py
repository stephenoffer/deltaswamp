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
import re
from collections.abc import Mapping
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
    VALUE_CONSTRAINT_FEATURES,
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


#: Open errors that are this process's Databricks authentication failing.
#: The warehouse needs the same authentication, so the fallback cannot help.
_AUTH_OPEN_ERRORS: tuple[str, ...] = (
    "could not configure databricks authentication",
    "the databricks credentials were rejected",
    "cannot reach the databricks host",
)


def _geospatial(table: ResolvedTable) -> bool:
    """Whether the table uses the geospatial feature (a GEOMETRY / GEOGRAPHY column)."""
    if "geospatial" in table.features or "delta.feature.geospatial" in table.properties:
        return True
    return bool(re.search(r"type: '(geometry|geography)\(", table.open_error or ""))


def _open_error_remedy(table: ResolvedTable, *, warehouse: bool = True) -> str:
    """The remedy for a table whose Delta log could not be opened."""
    error = (table.open_error or "").lower()
    if any(marker in error for marker in _AUTH_OPEN_ERRORS):
        # A worker whose unpickled provider could not authenticate was told
        # to enable the SQL fallback, which authenticates the same way.
        return (
            "fix the Databricks authentication this process uses (host=/token=, "
            "DATABRICKS_HOST/DATABRICKS_TOKEN, a profile or OAuth settings); the SQL "
            "warehouse fallback needs the same credentials"
        )
    if any(marker in error for marker in _UNFIXABLE_OPEN_ERRORS):
        return (
            f"check that {table.location or 'the table location'} holds a readable Delta "
            "log; the SQL warehouse reads the same storage and would fail the same way"
        )
    if warehouse and _warehouse_can_name(table):
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
        if table.is_managed_iceberg:
            # Managed Iceberg (USING ICEBERG) carries the same property, but it
            # is not UniForm: its Iceberg metadata is the table, and a Delta
            # commit beside it would not reach it.
            return (
                "the table is managed Iceberg (USING ICEBERG): its Iceberg metadata is the "
                "table of record, and a Delta commit would not reach it"
            )
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
    if (
        kind in _DIRECT_ENGINES
        and operation is Operation.ALTER_COLUMN_TYPE
        and str(table.properties.get("delta.enableTypeWidening", "")).strip().lower() != "true"
    ):
        # The call refused this ("needs the typeWidening feature enabled")
        # while can() said ok on every table without it.
        return (
            "changing a type without rewriting data needs type widening, which the table "
            "does not enable; set_properties({'delta.enableTypeWidening': 'true'}) first"
        )
    blocking = OPERATION_FEATURE_BLOCKERS.get((kind, operation), frozenset())
    hit = sorted(f.value for f in blocking & _active_features(table))
    if hit:
        return f"{operation.value} is not supported on a table with {', '.join(hit)}"
    if kind is EngineKind.KERNEL and operation in KERNEL_LOG_WRITE_OPERATIONS:
        # A build that checkpoints past the value-constraint features sets
        # them aside (crates/native/src/restate.rs, log_writing_snapshot):
        # a checkpoint writes no row for them to bind.
        past = getattr(engine, "checkpoints_past_value_constraints", None)
        exempt = VALUE_CONSTRAINT_FEATURES if callable(past) and past() else frozenset()
        writer = table.min_writer_version or 0
        if 3 <= writer <= 6 and not exempt:
            return (
                f"the table uses the legacy writer protocol version {writer}, which implies "
                "checkConstraints, and the kernel refuses to write its log"
            )
        # checkConstraints binds a checkpoint too on an older build: the
        # kernel's data writes commit past it (constraints evaluated here),
        # its checkpoint writer did not.
        unwritable = sorted(
            name
            for name in table.writer_features
            if (feature := feature_from_wire(name)) is None
            or (
                feature not in exempt
                and (
                    FEATURE_SUPPORT[feature].kernel_write is Support.NO
                    or feature is TableFeature.CHECK_CONSTRAINTS
                )
            )
        )
        if unwritable:
            return f"the kernel cannot write the log of a table with {', '.join(unwritable)}"
        has_invariants = getattr(engine, "_has_invariants", None)
        if (
            TableFeature.INVARIANTS not in exempt
            and (writer == 2 or "invariants" in table.writer_features)
            and callable(has_invariants)
        ):
            try:
                present = bool(has_invariants(table))
            except Exception:
                present = False
            if present:
                return "the table schema declares column invariants, which the kernel refuses"
    return None


#: Properties whose true value turns a feature on (metadata._ENABLING).
_ENABLING_PROPERTIES = {
    "delta.enablechangedatafeed": "changeDataFeed",
    "delta.enabledeletionvectors": "deletionVectors",
    "delta.enabletypewidening": "typeWidening",
    "delta.enableincommittimestamps": "inCommitTimestamp",
    "delta.enablerowtracking": "rowTracking",
}


def _features_added(operation: Operation, shape: dict[str, object]) -> set[str]:
    """The table features a metadata change of this shape turns on, with dependencies."""
    from .capability import FEATURE_DEPENDENCIES

    names: set[str] = set()
    if operation is Operation.CLUSTER_BY:
        names.add("clustering")
    elif operation is Operation.ADD_CONSTRAINT:
        names.add("checkConstraints")
    elif operation is Operation.ADD_FEATURE:
        features = shape.get("features")
        items = [features] if isinstance(features, str) else features
        for item in items if isinstance(items, (list, tuple, set, frozenset)) else ():
            names.add(str(getattr(item, "value", item)))
    elif operation is Operation.SET_PROPERTIES:
        properties = shape.get("properties")
        for key, value in properties.items() if isinstance(properties, dict) else ():
            lowered = str(key).lower()
            on = str(value).lower()
            if lowered.startswith("delta.feature.") and on in ("supported", "enabled"):
                names.add(str(key)[len("delta.feature.") :])
            elif lowered in _ENABLING_PROPERTIES and on == "true":
                names.add(_ENABLING_PROPERTIES[lowered])
            elif lowered == "delta.checkpointpolicy" and on == "v2":
                names.add("v2Checkpoint")
            elif lowered == "delta.columnmapping.mode" and on in ("name", "id"):
                names.add("columnMapping")
    pending = list(names)
    while pending:
        feature = feature_from_wire(pending.pop())
        for dep in FEATURE_DEPENDENCIES.get(feature, ()) if feature is not None else ():
            if dep.value not in names:
                names.add(dep.value)
                pending.append(dep.value)
    return names


def _strands_legacy_table(
    operation: Operation, table: ResolvedTable, shape: dict[str, object]
) -> str | None:
    """Why a direct engine must not make this change: no local engine could write the result.

    Moving a legacy writer-3..6 protocol to table features keeps every feature
    the old version implied listed -- Databricks lists appendOnly, invariants,
    checkConstraints and (from 4) generatedColumns too, used or not. The
    kernel writes no table that lists checkConstraints, so what is left is
    delta-rs; a feature it cannot write (clustering, typeWidening, ...)
    left a change-feed table that no local engine could append to again.
    """
    if operation not in METADATA_OPERATIONS:
        return None
    writer = table.min_writer_version or 0
    if writer >= 7:
        return _strands_feature_table(operation, table, shape)
    if not 3 <= writer <= 6:
        return None
    from .engine.metadata import _LEGACY_WRITER

    implied = set(_LEGACY_WRITER.get(writer, ()))
    added = _features_added(operation, shape) - implied
    if added and _writes(implied | added, "kernel", table):
        # Writer 3 implies no more than the kernel writes (checkConstraints
        # included, enforced here), so it goes on writing the table.
        return None
    unwritable = sorted(
        name
        for name in added
        if (feature := feature_from_wire(name)) is None
        or FEATURE_SUPPORT[feature].deltars_write is Support.NO
    )
    if not unwritable:
        return None
    kept = ", ".join(n for n in sorted(implied) if not _writes({n}, "kernel", table))
    return (
        f"the table is at the legacy writer version {writer}, and turning on "
        f"{', '.join(unwritable)} moves it to table features that must keep listing {kept} "
        "(as Databricks keeps them). The kernel writes no table listing those, and delta-rs "
        f"cannot write {', '.join(unwritable)}, so no local engine could write the table "
        "afterwards. Make this change from Databricks, which can go on writing it, or copy "
        "the data into a table created with table features"
    )


def _writes(features: set[str], engine: str, table: ResolvedTable | None = None) -> bool:
    """Whether `engine` (``kernel`` or ``deltars``) writes a table listing `features`.

    With `table`, as the kernel writes that table: `generatedColumns` binds
    none of its writes where no column is generated (they commit past the
    kernel's refusal of it; see `engine.kernel._CHECKED_FEATURES`).
    """
    unused: set[str] = set()
    if engine == "kernel" and table is not None and not table.has_generated_columns:
        from .engine.kernel import _native_has

        if _native_has("check_constraints"):
            unused.add("generatedColumns")
    for name in features - unused:
        feature = feature_from_wire(name)
        if feature is None or getattr(FEATURE_SUPPORT[feature], f"{engine}_write") is Support.NO:
            return False
    return True


def _strands_feature_table(
    operation: Operation, table: ResolvedTable, shape: dict[str, object]
) -> str | None:
    """`_strands_legacy_table` for a table already on table features.

    A legacy table moved to features by one change (a change feed, then
    deletion vectors) keeps listing checkConstraints and generatedColumns,
    which the kernel does not write; a second change adding a feature delta-rs
    cannot write (typeWidening, clustering) then left no local engine able to
    write it. Checked only where one could before the change.
    """
    current = set(table.effective_writer_features)
    added = _features_added(operation, shape) - current
    if not added:
        return None
    after = current | added
    if not (_writes(current, "kernel", table) or _writes(current, "deltars")):
        return None
    if _writes(after, "kernel", table) or _writes(after, "deltars"):
        return None
    kernel_blocks = sorted(n for n in after if not _writes({n}, "kernel", table))
    deltars_blocks = sorted(n for n in after if not _writes({n}, "deltars"))
    return (
        f"turning on {', '.join(sorted(added))} would leave no local engine able to write the "
        f"table: the kernel does not write {', '.join(kernel_blocks)}, and delta-rs does not "
        f"write {', '.join(deltars_blocks)}. Make this change from Databricks, which can go on "
        "writing it, or copy the data into a new table created with the features it needs"
    )


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
#: Needs that describe the request to the router rather than ask anything of an
#: engine, so no engine is judged on them. ``collation_free``: the caller has
#: checked the predicate against the schema and it touches no collated column.
#: ``variant_free``: the read touches no VARIANT column. ``removes_rows``: the
#: MERGE's clauses rewrite or delete target rows, which an append-only table
#: forbids whichever engine runs it.
_ROUTER_HINTS: frozenset[str] = frozenset(
    {
        "collation_free",
        "variant_free",
        "removes_rows",
        "conditional_insert_with_feed",
        # Only delta-rs cannot serve it (`DeltaRsEngine.need_refusal`).
        "early_datetimes",
        # DML SQL beyond the kernel's grammar that DuckDB, in Spark's
        # dialect, binds over the table's schema (`_kernel_evaluates_sql`).
        "duckdb_sql",
    }
)
#: What a need means, where its name alone does not say.
_NEED_REASONS: dict[str, str] = {
    "interval_columns": (
        "SQL on an ANSI interval column: its files hold the interval as bare integers "
        "(months, or microseconds), which a direct engine would compare in place of the "
        "interval; filter after reading (the frames show the interval), or use the warehouse"
    ),
    "pinned_read": (
        "a DELETE, UPDATE or MERGE from a handle pinned to a past version, which only the "
        "kernel's deletion-vector DML commits (conflict-checked against the commits since)"
    ),
    "incremental_files": (
        "reading the files added between two versions (added_since), which only the "
        "kernel's incremental scan lists"
    ),
}
#: Operations on a directory with no Delta log yet: can("convert") refused a
#: Parquet directory because its (absent) log could not be read.
_NO_LOG_YET: frozenset[Operation] = frozenset({Operation.CREATE, Operation.CONVERT})
#: Reads that decode data files, so meet whatever a shredded variant file holds.
_DATA_READS: frozenset[Operation] = frozenset(
    {Operation.SCAN, Operation.TIME_TRAVEL, Operation.CDF, Operation.INCREMENTAL}
)


def _append_only(table: ResolvedTable) -> bool:
    return str(table.properties.get("delta.appendOnly", "")).strip().lower() == "true"


def shreds_variants(table: ResolvedTable) -> bool:
    """Whether Databricks may have written shredded VARIANT files into this table.

    Databricks sets ``delta.enableVariantShredding`` on every VARIANT table it
    creates and then shreds any file whose values share a shape. Neither direct
    engine decodes a shredded file (the kernel fails with an ArrowInvalid,
    delta-rs refuses the feature), and nothing in the log says which files are
    shredded, so the property is the only honest signal there is.
    """
    return table.properties.get("delta.enableVariantShredding", "").lower() == "true"


_APPEND_ONLY_FORBIDS: frozenset[Operation] = frozenset(
    {Operation.DELETE, Operation.UPDATE, Operation.OVERWRITE, Operation.REPLACE_WHERE}
)
_ACCESS_POLICY_FORBIDS: frozenset[Operation] = frozenset(
    {Operation.TIME_TRAVEL, Operation.CDF, Operation.RESTORE}
)


#: Operations through which delta-rs rewrites existing data files, copying the
#: rows it did not change as it read them.
_FILE_REWRITES: frozenset[Operation] = frozenset(
    {
        Operation.DELETE,
        Operation.UPDATE,
        Operation.MERGE,
        Operation.REPLACE_WHERE,
        Operation.OPTIMIZE,
        Operation.ZORDER,
    }
)


#: Operations delta-rs must not serve on a table it misreads (see
#: `Router._calendar_refusal`): the rewrites, and the reads that decode data.
_CALENDAR_BOUND: frozenset[Operation] = _FILE_REWRITES | _DATA_READS


#: DML that the kernel writes as deletion vectors when a table enables them.
_DV_DML: frozenset[Operation] = frozenset(
    {Operation.DELETE, Operation.UPDATE, Operation.REPLACE_WHERE, Operation.MERGE}
)


#: DML whose SQL (predicate, SET values) the kernel evaluates with DuckDB
#: when its own grammar does not read it.
_DUCKDB_DML: frozenset[Operation] = frozenset(
    {Operation.DELETE, Operation.UPDATE, Operation.REPLACE_WHERE}
)


def _kernel_evaluates_sql(
    kind: EngineKind, operation: Operation, needs: frozenset[str], engine: object
) -> bool:
    """Whether the kernel evaluates this DML's SQL beyond its grammar itself.

    A DELETE, UPDATE or replaceWhere predicate, or a SET value, that the
    kernel's grammar does not read (`n + 1`, `lower(s) = 'a'`) is respelled
    in Spark's dialect and evaluated by DuckDB over the rows the kernel reads,
    as the kernel MERGE evaluates its clauses. ``duckdb_sql`` says the request
    checked that DuckDB binds every such text over the table's schema; text
    the dialect cannot translate faithfully carries ``spark_sql`` instead,
    and goes to the warehouse.
    """
    return (
        kind is EngineKind.KERNEL
        and operation in _DUCKDB_DML
        and "duckdb_sql" in needs
        and "spark_sql" not in needs
        and bool(getattr(engine, "supports_dml_sql_expressions", False))
    )


def _preference(
    operation: Operation, table: ResolvedTable, engines: tuple[EngineKind, ...]
) -> tuple[EngineKind, ...]:
    """`engines` in the order to ask them for this table.

    On a table with deletion vectors enabled, row-level DML goes to the kernel
    first, which marks rows deleted as Databricks does. delta-rs would rewrite
    every touched file instead (it never emits deletion vectors), which is
    correct but costs a full rewrite of each file and ignores the table's
    setting. Where the kernel refuses, the usual order still applies.
    """
    from .engine.kernel import deletion_vectors_writable

    if operation in _DV_DML and EngineKind.KERNEL in engines and deletion_vectors_writable(table):
        return (EngineKind.KERNEL, *(k for k in engines if k is not EngineKind.KERNEL))
    if (
        operation is Operation.VACUUM
        and EngineKind.KERNEL in engines
        and (
            "deletionVectors" in table.effective_reader_features
            or not _writes(
                set(table.effective_reader_features | table.effective_writer_features), "deltars"
            )
        )
    ):
        # A dry run on these would reach delta-rs (it commits nothing), and
        # the real VACUUM the kernel: one table, two plans. The kernel plans
        # both. On a deletion-vector table delta-rs also cannot tell a live
        # vector file from an orphan, so it keeps every one.
        return (EngineKind.KERNEL, *(k for k in engines if k is not EngineKind.KERNEL))
    if (
        operation is Operation.RESTORE
        and EngineKind.KERNEL in engines
        and str(table.properties.get("delta.columnMapping.mode", "none")).lower()
        not in ("", "none")
    ):
        # delta-rs restores a column-mapped table's old metadata verbatim,
        # rewinding delta.columnMapping.maxColumnId, so it refuses whenever
        # the schema changed; the kernel keeps the id and restores the rest.
        return (EngineKind.KERNEL, *(k for k in engines if k is not EngineKind.KERNEL))
    if operation in _STATS_WRITES and EngineKind.KERNEL in engines and _deltars_drops_stats(table):
        # delta-rs matches delta.dataSkippingStatsColumns against top-level
        # leaf names only: a nested field ("s.a") or a whole struct ("s") got
        # no statistics, and on a column-mapped table no listed column did.
        # The kernel honors every form. (The table's properties do not say
        # which names are structs, so any list goes to the kernel.)
        return (EngineKind.KERNEL, *(k for k in engines if k is not EngineKind.KERNEL))
    return engines


_STATS_WRITES = frozenset({Operation.APPEND, Operation.OVERWRITE})


def _deltars_drops_stats(table: ResolvedTable) -> bool:
    """Whether delta-rs would write fewer statistics than the table asks for."""
    return bool(table.properties.get("delta.dataSkippingStatsColumns"))


@dataclass
class Router:
    """Routes operations to engines for one connection."""

    engines: dict[EngineKind, object] = field(default_factory=dict)
    allow_sql_fallback: bool = False
    #: Whether this connection's catalog is the Databricks workspace a SQL
    #: warehouse serves. An OSS Unity Catalog server names its tables the same
    #: way (``uc`` references, three parts), and with the fallback on a read
    #: or a write of ``r3.m.t`` went to the warehouse -- to whatever table of
    #: that name the *Databricks* workspace held. False for every catalog but
    #: Databricks, so the warehouse is never asked and never suggested.
    warehouse_catalog: bool = True

    def __post_init__(self) -> None:
        # Every engine is handed out behind the one error boundary, however it
        # is reached (engine_for, or the engines mapping), so a raw engine
        # exception never reaches a caller.
        from .engine.boundary import GuardedEngines

        if not isinstance(self.engines, GuardedEngines):
            self.engines = GuardedEngines(self.engines)

    def __setattr__(self, name: str, value: object) -> None:
        # `router.engines = {...}` after construction too: it replaced the
        # guarded mapping with a plain one of raw engines.
        if name == "engines":
            from .engine.boundary import GuardedEngines

            if not isinstance(value, GuardedEngines):
                value = GuardedEngines(value)
        object.__setattr__(self, name, value)

    def _warehouse_names(self, table: ResolvedTable) -> bool:
        """Whether the warehouse this connection talks to can address `table`."""
        return self.warehouse_catalog and _warehouse_can_name(table)

    def _fallback_remedy(self, table: ResolvedTable) -> str:
        """SQL_FALLBACK_REMEDY where following it can work, else nothing."""
        return SQL_FALLBACK_REMEDY if self._warehouse_names(table) else ""

    def _scrubbed(self, capability: Capability) -> Capability:
        """`capability` without advice to enable a fallback this connection cannot have.

        Engines phrase their own refusals and some name the warehouse as the
        remedy; on a connection to another catalog there is none, and taking
        the advice once sent the call to a same-named Databricks table.
        """
        if self.warehouse_catalog or "allow_sql_fallback" not in (capability.remedy or ""):
            return capability
        return dataclasses.replace(capability, remedy="")

    def capability(
        self,
        operation: Operation,
        table: ResolvedTable,
        *,
        needs: frozenset[str] = frozenset(),
        exclude: frozenset[EngineKind] = frozenset(),
        **shape: object,
    ) -> Capability:
        """Decide how `operation` would be served, without performing it."""
        return self._scrubbed(
            self._capability(operation, table, needs=needs, exclude=exclude, **shape)
        )

    def _capability(
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
        if "identity_insert" in needs:
            return Capability(
                operation,
                ok=False,
                reason="the data gives values for a GENERATED ALWAYS AS IDENTITY column, "
                "which Delta generates itself and refuses to take "
                "(DELTA_IDENTITY_COLUMNS_EXPLICIT_INSERT_NOT_SUPPORTED)",
                remedy="leave the identity column out of the data",
            )
        if "identity_update" in needs:
            return Capability(
                operation,
                ok=False,
                reason="when_matched_update_all() would assign the identity column the source "
                "carries, and Delta refuses to update an identity column "
                "(DELTA_IDENTITY_COLUMNS_UPDATE_NOT_SUPPORTED)",
                remedy="drop the identity column from the source, or name the columns to set "
                "with when_matched_update(...)",
            )
        if "removes_rows" in needs and _append_only(table):
            # A MERGE with an UPDATE or DELETE clause: delta-rs accepted it and
            # failed at commit with a raw CommitFailedError.
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table is append-only (delta.appendOnly=true), and this MERGE's "
                    "UPDATE or DELETE clauses would remove or rewrite rows"
                ),
                remedy="merge with WHEN NOT MATCHED INSERT clauses only, or "
                "t.set_properties({'delta.appendOnly': 'false'}) if that is intended",
            )

        # A shared table has exactly one way in, so its engine's verdict is the
        # whole answer -- including "shares are read-only" for a write, which is
        # more useful than every other engine reporting it has no location.
        sharing = self.engines.get(EngineKind.SHARING)
        if table.is_shared and sharing is not None:
            # The request shape still applies: a share that cannot serve it must
            # say so here rather than accept and then drop the requirement.
            unmet = sorted(
                need
                for need in needs - _ROUTER_HINTS
                if not getattr(sharing, f"supports_{need}", False)
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
        sql_remedy = ""
        routing = OPERATION_ENGINES.get(operation)
        if routing is None:
            return Capability(operation, ok=False, reason=f"unknown operation {operation.value}")

        # Refusals `_catalog_level_block` waived only because the warehouse can
        # still serve the table. They still hold for every direct engine.
        direct_refusal = self._direct_refusal(operation, table) or _strands_legacy_table(
            operation, table, shape
        )

        for kind in _preference(operation, table, routing.engines):
            if kind in exclude:
                reasons.append(f"{kind.value}: already tried, and failed")
                continue
            engine = self.engines.get(kind)
            # Checked before "not configured": connect() only builds the SQL
            # engine when the fallback is on, so with it off the reason was
            # always "sql: engine not configured", which names no remedy.
            if kind is EngineKind.SQL and not self.warehouse_catalog:
                reasons.append(
                    "sql: a SQL warehouse serves only the Databricks workspace's own Unity "
                    "Catalog, and this connection's catalog is not one"
                )
                continue
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
                reasons.append(f"{kind.value}: {self._unvendable_reason(operation, table)}")
                continue

            missing = sorted(
                need
                for need in needs - _ROUTER_HINTS
                if not getattr(engine, f"supports_{need}", False)
            )
            if missing == ["sql_expressions"] and (
                self._kernel_filters_sql(kind, operation, table, needs, engine, shape)
                or _kernel_evaluates_sql(kind, operation, needs, engine)
            ):
                missing = []
            if missing:
                described = [_NEED_REASONS.get(need, need) for need in missing]
                reasons.append(f"{kind.value}: does not support {', '.join(described)}")
                continue
            # A need the engine has in general but not for this table.
            need_refusal = getattr(engine, "need_refusal", None)
            refused = need_refusal(needs, table) if needs and need_refusal else None
            if refused is not None:
                reasons.append(f"{kind.value}: {refused}")
                continue

            if (
                "predicates" in needs
                and "collation_free" not in needs
                and kind in _DIRECT_ENGINES
                and table.features & _COLLATIONS
            ):
                # A wrong answer, not an error: both engines compare bytes, so
                # `name = 'oslo'` misses 'Oslo' in a UTF8_LCASE column, and file
                # skipping on the same bounds can drop matching files outright.
                reasons.append(
                    f"{kind.value}: the table has collated columns, and a direct engine "
                    "evaluates predicates by byte order rather than by the collation"
                )
                continue

            if (
                kind in _DIRECT_ENGINES
                and operation in _DATA_READS
                and "variant_free" not in needs
                and shreds_variants(table)
            ):
                reasons.append(
                    f"{kind.value}: the table shreds VARIANT values "
                    "(delta.enableVariantShredding), and a direct engine cannot decode a "
                    "shredded file; read columns other than the VARIANT ones, or use the "
                    "warehouse"
                )
                continue

            blocker = _operation_blocker(kind, operation, table, engine)
            if blocker is not None:
                reasons.append(f"{kind.value}: {blocker}")
                continue

            judged = _exempted(kind, operation, table, shape)
            result: Capability = engine.supports(operation, judged, **shape)  # type: ignore[attr-defined]
            if result.ok and kind is EngineKind.DELTARS and operation in _CALENDAR_BOUND:
                # Asked only once delta-rs would otherwise serve: the check
                # lists the table's files, which is wasted on a refusal.
                shifted = self._calendar_refusal(table, operation, shape)
                if shifted is not None:
                    reasons.append(f"{kind.value}: {shifted}")
                    continue
            if result.ok and kind is EngineKind.DELTARS and operation in _FILE_REWRITES:
                footerless = self._footerless_rewrite(table, operation, shape, engine)
                if footerless is not None:
                    reasons.append(f"{kind.value}: {footerless}")
                    continue
            if result.ok:
                return result
            reasons.append(f"{kind.value}: {result.reason}")
            if kind is EngineKind.SQL and result.remedy:
                # The warehouse would serve it once configured (a staging
                # volume, say): that is the one remedy that helps.
                sql_remedy = result.remedy

        remedy = sql_remedy
        if EngineKind.SQL in routing.engines and not table.is_shared:
            if not self.warehouse_catalog and table.ref.kind is RefKind.CATALOG:
                remedy = ""
            elif not _warehouse_can_name(table):
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

    def _calendar_refusal(
        self,
        table: ResolvedTable,
        operation: Operation,
        shape: Mapping[str, object],
    ) -> str | None:
        """Why delta-rs must not read or rewrite this table's files, if it must not.

        delta-rs reads a file Spark wrote in its legacy hybrid calendar without
        rebasing it, and INT96 timestamps as overflowing nanoseconds
        (``9999-12-31`` reads as ``1816-03-30``). A read through it returns
        those values, and filters on them; its DML and compaction then copy the
        rows they did not touch into new files, shifted, without the footer
        that said so. The kernel's reads rebase (and it finds the files), so
        its read and write paths and the warehouse serve these tables instead.
        """
        if not table.has_datetime_columns:
            return None
        kernel = self.engines.get(EngineKind.KERNEL)
        check = getattr(kernel, "legacy_calendar_files", None)
        if check is None:
            return None
        reading = operation not in _FILE_REWRITES
        at = {
            k: v
            for k, v in shape.items()
            if k in ("version", "timestamp") and v is not None and reading
        }
        try:
            found = check(table, **at) if at else check(table)
        except Exception as exc:
            if reading:
                # A read changes nothing, and a kernel that cannot open the
                # snapshot to look cannot serve the read either: delta-rs is
                # then the only way in, as it was before the check.
                return None
            # Unknown is not "none": a wrong guess rewrites the table shifted.
            return (
                "could not check whether its data files were written in Spark's legacy "
                f"hybrid calendar ({type(exc).__name__}: {str(exc)[:160]}), and a rewrite "
                "through delta-rs shifts the values of any that were"
            )
        if not found:
            return None
        held = (
            f"{len(found)} data file(s) (such as {found[0]}) hold dates before 1582-10-15 or "
            "timestamps before 1900 that Spark wrote in its legacy hybrid calendar, or "
            "INT96 timestamps (which delta-rs decodes as nanoseconds, overflowing before "
            "1677-09-21 and after 2262-04-11), and delta-rs reads them without rebasing"
        )
        if reading:
            return f"{held}; this read would return the shifted values and filter on them"
        return (
            f"{held}; this operation would copy them into new files shifted, corrupting the table"
        )

    def _footerless_rewrite(
        self,
        table: ResolvedTable,
        operation: Operation,
        shape: Mapping[str, object],
        engine: object,
    ) -> str | None:
        """Why delta-rs must not rewrite this table's files, if the footer is why.

        A rewrite copies the rows it does not change into new files, and
        delta-rs writes them without Spark's writer metadata. Databricks reads
        a date before 1582-10-15 (or, outside UTC, a timestamp before 1900)
        from such a file with its legacy calendar rebase: rows it read right
        from a Spark or kernel-written file read two to ten days off after
        delta-rs had copied them. By statistics, as `_calendar_refusal`; a
        table whose files cannot be listed is left to that check.
        """
        if not table.has_datetime_columns:
            return None
        kernel = self.engines.get(EngineKind.KERNEL)
        snapshot = getattr(kernel, "snapshot", None)
        if snapshot is None:
            return None
        writes_itself = getattr(engine, "writes_files_itself", None)
        if writes_itself is not None and not writes_itself(operation, table, shape):
            return None
        try:
            from .engine.calendar import early_datetime_files

            found = early_datetime_files(snapshot(table))
        except Exception as exc:
            # Unknown is not "none": a wrong guess copies early values into
            # files Databricks reads shifted.
            return (
                "could not check whether its data files hold dates before 1582-10-15 or "
                f"timestamps before 1900 ({type(exc).__name__}: {str(exc)[:160]}), which a "
                "rewrite through delta-rs copies into files Databricks reads shifted"
            )
        if not found:
            return None
        return (
            f"{len(found)} data file(s) (such as {found[0]}) may hold dates before "
            "1582-10-15 or timestamps before 1900, and this operation would copy them into "
            "files delta-rs writes without Spark's writer metadata, which Databricks reads "
            "with its legacy calendar rebase (0001-01-01 reads there as 0001-01-03)"
        )

    def _kernel_filters_sql(
        self,
        kind: EngineKind,
        operation: Operation,
        table: ResolvedTable,
        needs: frozenset[str],
        engine: object,
        shape: Mapping[str, object],
    ) -> bool:
        """Whether the kernel reads this table and filters by SQL outside its grammar.

        A read predicate the kernel's grammar does not parse (`abs(id) > 0`,
        `year(dt) = 1500`) goes to delta-rs, which evaluates SQL -- but not on a
        table delta-rs misreads (see `_calendar_refusal`). There the kernel
        reads, with the right values, and the predicate is evaluated exactly
        afterwards by DuckDB in Spark's dialect, so the call keeps its
        predicate rather than being refused.
        """
        return (
            kind is EngineKind.KERNEL
            and operation in (Operation.SCAN, Operation.TIME_TRAVEL)
            and "distributed_scan" not in needs
            and bool(getattr(engine, "supports_read_sql_expressions", False))
            and self._calendar_refusal(table, operation, shape) is not None
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
        if table.open_error is not None and operation not in _NO_LOG_YET:
            return f"the table's Delta log could not be read ({table.open_error})"
        return None

    @staticmethod
    def _unvendable_reason(operation: Operation, table: ResolvedTable) -> str:
        """Why a direct engine cannot reach the table, as the catalog's manifest says.

        Every such refusal used to say Unity Catalog "withdraws this table from
        credential vending", which on a managed table that vends reads fine
        pointed at the wrong cause; it is the missing external-write support.
        """
        if table.external_read_supported is False:
            if table.access_policy:
                return (
                    f"the table has {table.access_policy}, and Unity Catalog vends no "
                    "credentials for tables with row filters or column masks"
                )
            return (
                "Unity Catalog reports no external-engine read support for this table "
                "(HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT absent)"
            )
        return (
            "Unity Catalog reports no external-engine write support for this table "
            f"(HAS_DIRECT_EXTERNAL_ENGINE_WRITE_SUPPORT absent), so {operation.value} "
            "cannot commit from outside Databricks"
        )

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
        sql_fallback = (
            self.allow_sql_fallback and self.warehouse_catalog and EngineKind.SQL in self.engines
        )

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
                    if self._warehouse_names(table)
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
                remedy=self._fallback_remedy(table),
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
                remedy=self._fallback_remedy(table),
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
                remedy=self._fallback_remedy(table),
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
                remedy=self._fallback_remedy(table),
            )

        # A table we could not open is not a table we can route. CREATE and
        # CONVERT are exempt: there is no log to open yet.
        if table.open_error is not None and operation not in _NO_LOG_YET and not sql_fallback:
            reason = (
                "the table's Delta log could not be read, so no direct engine can "
                f"serve this ({table.open_error})"
            )
            if _geospatial(table):
                # The kernel fails parsing the schema's geometry(...) type
                # before the feature table (capability.py) is ever consulted.
                reason = (
                    "the table has a GEOMETRY or GEOGRAPHY column (the geospatial "
                    "feature), which no direct engine reads or writes"
                )
            return Capability(
                operation,
                ok=False,
                reason=reason,
                remedy=_open_error_remedy(table, warehouse=self.warehouse_catalog),
            )

        if (
            operation in DATABRICKS_ONLY_OPERATIONS
            and not self.warehouse_catalog
            and table.ref.kind is RefKind.CATALOG
        ):
            return Capability(
                operation,
                ok=False,
                reason=(
                    f"{operation.value} has no open-source implementation in either delta-rs "
                    "or delta-kernel; it exists only in Databricks, and this connection's "
                    "catalog is not Databricks Unity Catalog"
                ),
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
                remedy=self._fallback_remedy(table),
            )

        return None
