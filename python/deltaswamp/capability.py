"""The conformance matrix: which engine can serve which operation on which table.

This module is *data*, deliberately. Coverage claims that live only in prose rot
silently; here every claim is a row that a test asserts against.

The central finding it encodes: the two OSS engines fail in opposite directions.
delta-kernel reads through writer-only features that delta-rs refuses to open at
all (it does a flat set-difference, and `can_write_to` calls `can_read_from`
first). But delta-rs is far ahead on *doing* things -- kernel has no MERGE, no
OPTIMIZE, no VACUUM, no RESTORE. So kernel is the default read engine and
delta-rs the default DML/maintenance engine, composed per operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "ENGINE_METHODS",
    "FEATURE_DEPENDENCIES",
    "FEATURE_SIGNAL_PREFIX",
    "FEATURE_SUPPORT",
    "KERNEL_CREATE_FEATURES",
    "LEGACY_READER_FEATURES",
    "LEGACY_WRITER_FEATURES",
    "METADATA_OPERATIONS",
    "PROPERTY_SUPPORT",
    "Capability",
    "Engine",
    "FeatureKind",
    "FeatureSupport",
    "Operation",
    "PropertyEffect",
    "PropertySupport",
    "Support",
    "TableFeature",
    "feature_from_wire",
    "implied_features",
    "property_support",
]


class Engine(StrEnum):
    KERNEL = "kernel"
    DELTARS = "deltars"
    SQL = "sql"
    SHARING = "sharing"
    ICEBERG = "iceberg"
    PYARROW = "pyarrow"


class Support(StrEnum):
    YES = "yes"
    NO = "no"
    PARTIAL = "partial"

    def __bool__(self) -> bool:  # `if support:` means "usable without caveats"
        return self is Support.YES


class FeatureKind(StrEnum):
    """Protocol position. Decides whether a feature can block *reads*.

    WRITER features are invisible to kernel's read path -- that exemption is the
    single biggest reason kernel out-reads delta-rs. READER_WRITER features block
    both directions, which is why `vacuumProtocolCheck` blocks delta-rs reads and
    not merely VACUUM.
    """

    READER = "reader"
    WRITER = "writer"
    READER_WRITER = "readerWriter"


class Operation(StrEnum):
    # --- read
    SCAN = "scan"
    TIME_TRAVEL = "time_travel"
    CDF = "cdf"
    INCREMENTAL = "incremental"
    HISTORY = "history"
    DETAIL = "detail"
    FILES = "files"
    # --- write
    APPEND = "append"
    OVERWRITE = "overwrite"
    REPLACE_WHERE = "replace_where"
    CREATE = "create"
    MERGE_SCHEMA = "merge_schema"
    # --- dml
    DELETE = "delete"
    UPDATE = "update"
    MERGE = "merge"
    # --- ddl
    ADD_COLUMN = "add_column"
    DROP_COLUMN = "drop_column"
    RENAME_COLUMN = "rename_column"
    SET_PROPERTIES = "set_properties"
    ADD_FEATURE = "add_feature"
    DROP_FEATURE = "drop_feature"
    ADD_CONSTRAINT = "add_constraint"
    DROP_CONSTRAINT = "drop_constraint"
    UNSET_PROPERTIES = "unset_properties"
    SET_COMMENT = "set_comment"
    SET_COLUMN_COMMENT = "set_column_comment"
    ALTER_COLUMN_TYPE = "alter_column_type"
    SET_NOT_NULL = "set_not_null"
    DROP_NOT_NULL = "drop_not_null"
    CLUSTER_BY = "cluster_by"
    # --- maintenance
    OPTIMIZE = "optimize"
    ZORDER = "zorder"
    VACUUM = "vacuum"
    RESTORE = "restore"
    REPAIR = "repair"
    CHECKPOINT = "checkpoint"
    LOG_COMPACTION = "log_compaction"
    PUBLISH = "publish"
    REORG = "reorg"
    CLONE = "clone"
    CONVERT = "convert"
    GENERATE = "generate"
    CLEANUP_METADATA = "cleanup_metadata"
    ANALYZE = "analyze"
    SYNC_ICEBERG = "sync_iceberg"
    REFRESH = "refresh"


class TableFeature(StrEnum):
    """Delta table features, keyed by their on-the-wire name.

    Covers all 31 `delta_kernel::table_features::TableFeature` variants plus the
    ones that exist in Databricks or delta-rs but have NO kernel variant -- those
    are classified Unknown by kernel and therefore block writes on both engines.
    """

    APPEND_ONLY = "appendOnly"
    INVARIANTS = "invariants"
    CHECK_CONSTRAINTS = "checkConstraints"
    CHANGE_DATA_FEED = "changeDataFeed"
    GENERATED_COLUMNS = "generatedColumns"
    IDENTITY_COLUMNS = "identityColumns"
    IN_COMMIT_TIMESTAMP = "inCommitTimestamp"
    ROW_TRACKING = "rowTracking"
    DOMAIN_METADATA = "domainMetadata"
    ICEBERG_COMPAT_V1 = "icebergCompatV1"
    ICEBERG_COMPAT_V2 = "icebergCompatV2"
    ICEBERG_COMPAT_V3 = "icebergCompatV3"
    # NB: serialises as "clustering", NOT "clusteredTable". Easy to get wrong.
    CLUSTERING = "clustering"
    MATERIALIZE_PARTITION_COLUMNS = "materializePartitionColumns"
    ALLOW_COLUMN_DEFAULTS = "allowColumnDefaults"
    CATALOG_MANAGED = "catalogManaged"
    CATALOG_OWNED_PREVIEW = "catalogOwned-preview"
    COLUMN_MAPPING = "columnMapping"
    DELETION_VECTORS = "deletionVectors"
    TIMESTAMP_NTZ = "timestampNtz"
    TYPE_WIDENING = "typeWidening"
    TYPE_WIDENING_PREVIEW = "typeWidening-preview"
    V2_CHECKPOINT = "v2Checkpoint"
    VACUUM_PROTOCOL_CHECK = "vacuumProtocolCheck"
    VARIANT_TYPE = "variantType"
    VARIANT_TYPE_PREVIEW = "variantType-preview"
    VARIANT_SHREDDING = "variantShredding"
    VARIANT_SHREDDING_PREVIEW = "variantShredding-preview"
    ADAPTIVE_METADATA_PREVIEW = "adaptiveMetadata-preview"
    GEOSPATIAL = "geospatial"

    # --- No kernel 0.28 variant. Kernel classifies these Unknown -> writes blocked.
    COLLATIONS = "collations"
    COLLATIONS_PREVIEW = "collations-preview"
    CHECKPOINT_PROTECTION = "checkpointProtection"
    # delta-rs-only, non-standard extension.
    TIMESTAMP_NANOS = "timestampNanos"


#: Reader features a legacy `minReaderVersion` implies, cumulatively.
#:
#: Before version 3 a protocol has no `readerFeatures` list at all: the version
#: number alone says what the table uses. A connector that only reads the named
#: list sees an empty set and concludes the table is plain, which is how an
#: engine ends up accepting a write it cannot perform.
LEGACY_READER_FEATURES: dict[int, frozenset[TableFeature]] = {
    1: frozenset(),
    2: frozenset({TableFeature.COLUMN_MAPPING}),
}

#: Writer features a legacy `minWriterVersion` implies, cumulatively.
LEGACY_WRITER_FEATURES: dict[int, frozenset[TableFeature]] = {
    1: frozenset(),
    2: frozenset({TableFeature.APPEND_ONLY, TableFeature.INVARIANTS}),
    3: frozenset(
        {TableFeature.APPEND_ONLY, TableFeature.INVARIANTS, TableFeature.CHECK_CONSTRAINTS}
    ),
    4: frozenset(
        {
            TableFeature.APPEND_ONLY,
            TableFeature.INVARIANTS,
            TableFeature.CHECK_CONSTRAINTS,
            TableFeature.CHANGE_DATA_FEED,
            TableFeature.GENERATED_COLUMNS,
        }
    ),
    5: frozenset(
        {
            TableFeature.APPEND_ONLY,
            TableFeature.INVARIANTS,
            TableFeature.CHECK_CONSTRAINTS,
            TableFeature.CHANGE_DATA_FEED,
            TableFeature.GENERATED_COLUMNS,
            TableFeature.COLUMN_MAPPING,
        }
    ),
    6: frozenset(
        {
            TableFeature.APPEND_ONLY,
            TableFeature.INVARIANTS,
            TableFeature.CHECK_CONSTRAINTS,
            TableFeature.CHANGE_DATA_FEED,
            TableFeature.GENERATED_COLUMNS,
            TableFeature.COLUMN_MAPPING,
            TableFeature.IDENTITY_COLUMNS,
        }
    ),
}


def implied_features(
    min_reader: int | None, min_writer: int | None
) -> tuple[frozenset[str], frozenset[str]]:
    """The (reader, writer) features a legacy protocol version implies.

    Reader version 3 and writer version 7 are the feature-based protocols,
    where the named lists are authoritative and nothing is implied. Below
    those, the version number *is* the feature list.

    An unrecognised version is taken as the highest one we model rather than as
    "nothing": a table from a newer writer should not read as featureless.
    """
    readers: frozenset[TableFeature] = frozenset()
    if min_reader is not None and min_reader < 3:
        readers = LEGACY_READER_FEATURES.get(min_reader, LEGACY_READER_FEATURES[2])
    writers: frozenset[TableFeature] = frozenset()
    if min_writer is not None and min_writer < 7:
        writers = LEGACY_WRITER_FEATURES.get(min_writer, LEGACY_WRITER_FEATURES[6])
    return frozenset(f.value for f in readers), frozenset(f.value for f in writers)


@dataclass(frozen=True, slots=True)
class FeatureSupport:
    """One row of the feature matrix."""

    feature: TableFeature
    kind: FeatureKind
    kernel_read: Support
    kernel_write: Support
    deltars_read: Support
    deltars_write: Support
    note: str = ""
    issue: str = ""

    def read_engines(self) -> list[Engine]:
        out = []
        if self.kernel_read is not Support.NO:
            out.append(Engine.KERNEL)
        if self.deltars_read is not Support.NO:
            out.append(Engine.DELTARS)
        return out

    def write_engines(self) -> list[Engine]:
        out = []
        if self.kernel_write is not Support.NO:
            out.append(Engine.KERNEL)
        if self.deltars_write is not Support.NO:
            out.append(Engine.DELTARS)
        return out


_Y, _N, _P = Support.YES, Support.NO, Support.PARTIAL
_R, _W, _RW = FeatureKind.READER, FeatureKind.WRITER, FeatureKind.READER_WRITER


def _row(
    feature: TableFeature,
    kind: FeatureKind,
    kr: Support,
    kw: Support,
    dr: Support,
    dw: Support,
    note: str = "",
    issue: str = "",
) -> tuple[TableFeature, FeatureSupport]:
    return feature, FeatureSupport(feature, kind, kr, kw, dr, dw, note, issue)


FEATURE_SUPPORT: dict[TableFeature, FeatureSupport] = dict(
    [
        _row(TableFeature.APPEND_ONLY, _W, _Y, _Y, _Y, _Y),
        _row(
            TableFeature.INVARIANTS,
            _W,
            _Y,
            _P,
            _Y,
            _Y,
            "kernel is nominally Supported but fails any write when invariants are "
            "actually present in the schema; route those writes to delta-rs",
        ),
        _row(
            TableFeature.CHECK_CONSTRAINTS,
            _W,
            _Y,
            _N,
            _Y,
            _Y,
            "delta-rs is ahead of kernel here: add_constraint/drop_constraint exist",
        ),
        _row(TableFeature.CHANGE_DATA_FEED, _W, _Y, _Y, _Y, _Y),
        _row(
            TableFeature.GENERATED_COLUMNS,
            _W,
            _Y,
            _N,
            _Y,
            _Y,
            "delta-rs evaluates generated columns via DataFusion; kernel refuses to write",
        ),
        _row(
            TableFeature.IDENTITY_COLUMNS,
            _W,
            _Y,
            _N,
            _N,
            _N,
            "delta-rs has the enum variant but it is commented out of writer_features",
            "delta-rs#3249",
        ),
        _row(TableFeature.IN_COMMIT_TIMESTAMP, _W, _Y, _Y, _N, _N, "", "delta-rs#3253"),
        _row(
            TableFeature.ROW_TRACKING,
            _W,
            _Y,
            _P,
            _N,
            _N,
            "kernel assigns fresh row IDs only; it rejects commits with staged removes "
            "because it cannot preserve IDs across a rewrite. Requires domainMetadata.",
            "delta-rs#3254, delta-rs#3928",
        ),
        _row(
            TableFeature.DOMAIN_METADATA,
            _W,
            _Y,
            _Y,
            _N,
            _N,
            "delta-rs has no domain metadata support, so EVERY liquid-clustered and "
            "row-tracked table is delta-rs-unwritable",
            "delta-rs#3249",
        ),
        _row(TableFeature.ICEBERG_COMPAT_V1, _W, _Y, _N, _N, _N, "", "delta-rs#3249"),
        _row(TableFeature.ICEBERG_COMPAT_V2, _W, _Y, _N, _N, _N, "", "delta-rs#3249"),
        _row(
            TableFeature.ICEBERG_COMPAT_V3,
            _W,
            _Y,
            _Y,
            _N,
            _N,
            "requires columnMapping + rowTracking; permits deletion vectors (V1/V2 do not)",
        ),
        _row(
            TableFeature.CLUSTERING,
            _W,
            _Y,
            _Y,
            _N,
            _N,
            "wire name is 'clustering', not 'clusteredTable'. Requires domainMetadata. "
            "UCCommitter rejects clustering changes at version >= 1.",
            "delta-rs#2043",
        ),
        _row(TableFeature.MATERIALIZE_PARTITION_COLUMNS, _W, _Y, _Y, _N, _N),
        _row(
            TableFeature.ALLOW_COLUMN_DEFAULTS,
            _W,
            _Y,
            _Y,
            _N,
            _N,
            "kernel requires ack_column_defaults() before write; connector materialises "
            "the defaults itself. ALTER TABLE is rejected on such tables.",
        ),
        _row(
            TableFeature.CATALOG_MANAGED,
            _RW,
            _Y,
            _Y,
            _N,
            _N,
            "requires inCommitTimestamp enabled. CDF explicitly errors on these tables. "
            "delta-rs cannot even open one.",
            "delta-rs#4549",
        ),
        _row(TableFeature.CATALOG_OWNED_PREVIEW, _RW, _Y, _Y, _N, _N, "legacy preview name"),
        _row(
            TableFeature.COLUMN_MAPPING,
            _RW,
            _Y,
            _Y,
            _Y,
            _Y,
            "delta-rs read/write OK, but to_pyarrow_dataset() hard-rejects it, and it "
            "cannot be enabled from write_deltalake",
            "delta-rs#3936",
        ),
        _row(
            TableFeature.DELETION_VECTORS,
            _RW,
            _Y,
            _P,
            _Y,
            _P,
            "kernel installs connector-authored DV descriptors but computes no bitmaps; "
            "delta-rs reads and preserves DVs but never emits them (DML is copy-on-write)",
            "delta-rs#4512",
        ),
        _row(
            TableFeature.TIMESTAMP_NTZ,
            _RW,
            _Y,
            _Y,
            _Y,
            _Y,
            "kernel accepts the non-canonical 'timestampWithoutTimezone' on read but "
            "always writes 'timestampNtz'",
        ),
        _row(TableFeature.TYPE_WIDENING, _RW, _Y, _Y, _N, _N, "", "delta-rs#2464"),
        _row(TableFeature.TYPE_WIDENING_PREVIEW, _RW, _Y, _Y, _N, _N, "", "delta-rs#2464"),
        _row(TableFeature.V2_CHECKPOINT, _RW, _Y, _Y, _Y, _Y),
        _row(
            TableFeature.VACUUM_PROTOCOL_CHECK,
            _RW,
            _Y,
            _Y,
            _N,
            _N,
            "ReaderWriter, so it blocks delta-rs READS, not merely VACUUM. Easy to miss.",
            "delta-rs#3249",
        ),
        _row(TableFeature.VARIANT_TYPE, _RW, _Y, _Y, _Y, _Y),
        _row(TableFeature.VARIANT_TYPE_PREVIEW, _RW, _Y, _Y, _Y, _Y),
        _row(TableFeature.VARIANT_SHREDDING, _RW, _Y, _Y, _N, _N),
        _row(TableFeature.VARIANT_SHREDDING_PREVIEW, _RW, _Y, _Y, _N, _N),
        # Both sit behind a kernel cargo feature this build deliberately does not
        # enable (see crates/native/Cargo.toml), so for *this* binary they are
        # unsupported in both directions. Recording what kernel could do behind
        # a flag we do not compile made these pass the write check and fail at
        # commit instead -- after the data was written.
        _row(
            TableFeature.ADAPTIVE_METADATA_PREVIEW,
            _RW,
            _N,
            _N,
            _N,
            _N,
            "kernel gates this behind the adaptive-metadata-in-dev cargo feature, which "
            "this build does not enable; not production-ready",
        ),
        _row(
            TableFeature.GEOSPATIAL,
            _RW,
            _N,
            _N,
            _N,
            _N,
            "kernel gates reads behind geo-type-in-dev, which this build does not "
            "enable, and errors on writes regardless. One feature covers both "
            "geometry and geography.",
        ),
        # --- no kernel variant at all
        _row(
            TableFeature.COLLATIONS,
            _RW,
            _N,
            _N,
            _N,
            _N,
            "Databricks DBR 16.1. No kernel 0.28 variant -> classified Unknown -> "
            "writes blocked on both engines. Route to SQL.",
        ),
        _row(TableFeature.COLLATIONS_PREVIEW, _RW, _N, _N, _N, _N, "see collations"),
        _row(
            TableFeature.CHECKPOINT_PROTECTION,
            _W,
            _N,
            _N,
            _N,
            _N,
            "Databricks DBR 16.3, added when a table feature is dropped. Ignoring it "
            "during metadata cleanup can delete the checkpoint that makes a downgraded "
            "table readable.",
            "delta-rs#4462",
        ),
        _row(
            TableFeature.TIMESTAMP_NANOS,
            _RW,
            _N,
            _N,
            _Y,
            _Y,
            "delta-rs-only non-standard extension behind the nanosecond-timestamps "
            "cargo feature; no kernel variant",
        ),
    ]
)


# Kernel enforces these before a write; violating them produces a table other
# engines will reject. Keys require the listed features to be present/enabled.
FEATURE_DEPENDENCIES: dict[TableFeature, frozenset[TableFeature]] = {
    TableFeature.ROW_TRACKING: frozenset({TableFeature.DOMAIN_METADATA}),
    TableFeature.CLUSTERING: frozenset({TableFeature.DOMAIN_METADATA}),
    TableFeature.CATALOG_MANAGED: frozenset({TableFeature.IN_COMMIT_TIMESTAMP}),
    TableFeature.CATALOG_OWNED_PREVIEW: frozenset({TableFeature.IN_COMMIT_TIMESTAMP}),
    TableFeature.ICEBERG_COMPAT_V1: frozenset({TableFeature.COLUMN_MAPPING}),
    TableFeature.ICEBERG_COMPAT_V2: frozenset({TableFeature.COLUMN_MAPPING}),
    TableFeature.ICEBERG_COMPAT_V3: frozenset(
        {TableFeature.COLUMN_MAPPING, TableFeature.ROW_TRACKING}
    ),
    TableFeature.ADAPTIVE_METADATA_PREVIEW: frozenset(
        {
            TableFeature.COLUMN_MAPPING,
            TableFeature.ROW_TRACKING,
            TableFeature.DOMAIN_METADATA,
            TableFeature.DELETION_VECTORS,
            TableFeature.IN_COMMIT_TIMESTAMP,
        }
    ),
}

# Mutually exclusive: enabling one forbids the others.
FEATURE_CONFLICTS: dict[TableFeature, frozenset[TableFeature]] = {
    TableFeature.ICEBERG_COMPAT_V1: frozenset(
        {
            TableFeature.ICEBERG_COMPAT_V2,
            TableFeature.ICEBERG_COMPAT_V3,
            TableFeature.DELETION_VECTORS,
        }
    ),
    TableFeature.ICEBERG_COMPAT_V2: frozenset(
        {
            TableFeature.ICEBERG_COMPAT_V1,
            TableFeature.ICEBERG_COMPAT_V3,
            TableFeature.DELETION_VECTORS,
        }
    ),
    TableFeature.ICEBERG_COMPAT_V3: frozenset(
        {TableFeature.ICEBERG_COMPAT_V1, TableFeature.ICEBERG_COMPAT_V2}
    ),
}


def feature_from_wire(name: str) -> TableFeature | None:
    """Map an on-the-wire feature name to a known feature, or None if unknown.

    Returning None rather than raising is deliberate: unknown features must be
    tolerated on the read path (kernel ignores writer-only unknowns) and only
    block writes. Forward compatibility depends on not choking here.
    """
    try:
        return TableFeature(name)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Capability:
    """Whether one operation is servable, by which engine, and if not, why not."""

    operation: Operation
    ok: bool
    engine: Engine | None = None
    reason: str = ""
    remedy: str = ""
    blockers: tuple[TableFeature, ...] = field(default=())

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        if self.ok:
            return f"{self.operation.value}: via {self.engine.value if self.engine else '?'}"
        s = f"{self.operation.value}: unavailable -- {self.reason}"
        if self.remedy:
            s += f" (remedy: {self.remedy})"
        return s


# ---------------------------------------------------------------------------
# Operation routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OperationSupport:
    """Which engines can serve an operation, in preference order."""

    operation: Operation
    engines: tuple[Engine, ...]
    rationale: str = ""

    @property
    def primary(self) -> Engine | None:
        return self.engines[0] if self.engines else None


def _op(
    operation: Operation, engines: tuple[Engine, ...], rationale: str = ""
) -> tuple[Operation, OperationSupport]:
    return operation, OperationSupport(operation, engines, rationale)


_K, _D, _S, _I, _SH = Engine.KERNEL, Engine.DELTARS, Engine.SQL, Engine.ICEBERG, Engine.SHARING

# Engines listed in preference order. The SQL entry is only reachable when the
# connection opted into fallback; the router enforces that, not this table.
#
# ICEBERG and SHARING serve only tables that are theirs -- Iceberg tables behind
# a catalog's Iceberg REST endpoint, and tables reached through a Delta Sharing
# profile -- and refuse everything else, so their position in a chain only
# matters for tables more than one engine could serve.
OPERATION_ENGINES: dict[Operation, OperationSupport] = dict(
    [
        # --- read: kernel first, because it reads through writer-only features
        _op(
            Operation.SCAN,
            (_K, _D, _SH, _I, _S),
            "kernel reads through writer-only features delta-rs rejects",
        ),
        _op(
            Operation.TIME_TRAVEL,
            (_K, _D, _SH, _I, _S),
            "history_manager handles the ICT-enablement boundary",
        ),
        _op(
            Operation.CDF,
            (_D, _K, _SH, _S),
            "delta-rs serves path tables it can open; the kernel's TableChanges takes the "
            "ones it cannot. Catalog-managed tables have no CDF outside Databricks",
        ),
        _op(
            Operation.INCREMENTAL,
            (),
            "reading only the files added since a version needs the kernel's "
            "incremental_scan, which is not bound yet; Table.changes() follows the change "
            "data feed instead",
        ),
        _op(
            Operation.HISTORY,
            (_D, _I, _S),
            "kernel exposes no history() API, only commit_range primitives",
        ),
        _op(
            Operation.DETAIL,
            (_K, _D, _SH, _I, _S),
            "kernel CRC path gives O(1) stats with zero I/O when a .crc exists",
        ),
        _op(
            Operation.FILES,
            (_D, _K),
            "delta-rs lists add actions with stats; the kernel lists the files of tables "
            "delta-rs cannot open",
        ),
        # --- write
        _op(
            Operation.APPEND,
            (_D, _K, _I, _S),
            "delta-rs for path and external tables; it refuses catalog-managed tables, "
            "which then fall through to kernel and UCCommitter. The warehouse loads "
            "through a staging volume when neither can",
        ),
        _op(
            Operation.CREATE,
            (_D, _K),
            "delta-rs creates path and external tables; the kernel takes over when the "
            "properties or clustering exceed what delta-rs accepts, and for managed "
            "tables, whose storage the catalog allocates through its staging-table API",
        ),
        _op(
            Operation.OVERWRITE,
            (_D, _K, _I, _S),
            "delta-rs first; the kernel replaces a catalog-managed table in one commit",
        ),
        _op(
            Operation.REPLACE_WHERE,
            (_D, _I, _S, _K),
            "delta-rs, PyIceberg for Iceberg tables, the warehouse; last, the kernel "
            "rewrites the whole table in one commit, bounded in size",
        ),
        _op(
            Operation.MERGE_SCHEMA,
            (_D, _S),
            "kernel has no mergeSchema on the write path; the warehouse uses INSERT WITH "
            "SCHEMA EVOLUTION",
        ),
        # --- dml: delta-rs (copy-on-write) or kernel + our own DV authoring
        _op(
            Operation.DELETE,
            (_D, _S, _K),
            "delta-rs copy-on-write, then the warehouse; last, a whole-table rewrite "
            "through the kernel, since kernel 0.28 cannot author deletion vectors",
        ),
        _op(
            Operation.UPDATE,
            (_D, _S, _K),
            "delta-rs, then the warehouse; last, a whole-table rewrite through the kernel "
            "with literal or column assignments",
        ),
        _op(
            Operation.MERGE,
            (_D, _S),
            "kernel has no MERGE at all; the warehouse merges from a source staged in a volume",
        ),
        # --- ddl. The kernel rows are metadata-only commits this library writes
        # itself, for path tables delta-rs cannot alter or cannot express.
        _op(
            Operation.ADD_COLUMN, (_D, _K, _S), "delta-rs first; kernel for tables it cannot write"
        ),
        _op(
            Operation.SET_PROPERTIES,
            (_D, _K, _S),
            "delta-rs rejects most of the property surface on ALTER; the kernel path "
            "accepts what it can validate",
        ),
        _op(Operation.UNSET_PROPERTIES, (_K, _S), "delta-rs has no way to remove a property"),
        _op(Operation.ADD_FEATURE, (_D, _K, _S), "delta-rs first; kernel for the rest"),
        _op(
            Operation.ADD_CONSTRAINT,
            (_D, _S),
            "kernel marks checkConstraints NotSupported for writes, and adding one means "
            "validating every existing row",
        ),
        _op(Operation.DROP_CONSTRAINT, (_D, _K, _S), "a metadata-only change"),
        _op(
            Operation.SET_COMMENT,
            (_D, _K, _S),
            "the table description in the Metadata action",
        ),
        _op(
            Operation.SET_COLUMN_COMMENT,
            (_D, _K, _S),
            "a 'comment' entry in the column's field metadata",
        ),
        _op(
            Operation.ALTER_COLUMN_TYPE,
            (_K, _S),
            "type widening: metadata-only under the typeWidening feature; delta-rs "
            "cannot read such a table at all",
        ),
        _op(
            Operation.SET_NOT_NULL,
            (_K, _S),
            "needs every existing row checked for nulls before the commit",
        ),
        _op(Operation.DROP_NOT_NULL, (_D, _K, _S), "a metadata-only change"),
        _op(
            Operation.CLUSTER_BY,
            (_K, _S),
            "the delta.clustering domain; delta-rs has no domain metadata support",
        ),
        _op(
            Operation.DROP_COLUMN,
            (_K, _S),
            "metadata-only under column mapping, which the kernel path writes; delta-rs "
            "has no DROP COLUMN",
        ),
        _op(
            Operation.RENAME_COLUMN,
            (_K, _S),
            "metadata-only under column mapping, which the kernel path writes; delta-rs "
            "has no RENAME COLUMN",
        ),
        _op(Operation.DROP_FEATURE, (_S,), "Databricks-only (DROP FEATURE ... TRUNCATE HISTORY)"),
        # --- maintenance
        _op(
            Operation.OPTIMIZE,
            (_D, _S),
            "kernel has no OPTIMIZE; the warehouse runs it on managed and clustered tables",
        ),
        _op(Operation.ZORDER, (_D, _S), "kernel has no Z-ORDER"),
        _op(Operation.VACUUM, (_D, _S), "kernel has no VACUUM"),
        _op(Operation.RESTORE, (_D, _S), "kernel has no RESTORE"),
        _op(Operation.REPAIR, (_D, _S), "kernel has no FSCK"),
        _op(Operation.CONVERT, (_D,), "kernel has no CONVERT TO DELTA"),
        _op(Operation.GENERATE, (_D,), "kernel has no manifest generation"),
        _op(
            Operation.CHECKPOINT,
            (_D, _K),
            "delta-rs for tables it can open; the kernel for the rest, including "
            "catalog-managed tables, which it publishes first",
        ),
        _op(
            Operation.LOG_COMPACTION,
            (_D,),
            "kernel's log_compaction_writer is a no-op stub (kernel#2337)",
        ),
        _op(
            Operation.CLEANUP_METADATA,
            (_D,),
            "delta-rs removes log files older than delta.logRetentionDuration",
        ),
        _op(
            Operation.PUBLISH, (_K,), "Snapshot::publish; only kernel implements staged->published"
        ),
        _op(Operation.REORG, (_S,), "Databricks-only (REORG ... APPLY PURGE / UPGRADE UNIFORM)"),
        _op(Operation.CLONE, (_S,), "Databricks-only (shallow and deep CLONE)"),
        _op(
            Operation.ANALYZE,
            (_S,),
            "Databricks-only (ANALYZE TABLE ... COMPUTE [DELTA] STATISTICS)",
        ),
        _op(
            Operation.SYNC_ICEBERG,
            (_S,),
            "Databricks-only (MSCK REPAIR TABLE ... SYNC METADATA regenerates UniForm "
            "Iceberg metadata)",
        ),
        _op(
            Operation.REFRESH,
            (_S,),
            "Databricks-only (REFRESH of a materialized view or streaming table)",
        ),
    ]
)

# Operations that no OSS engine implements -- there is nothing to degrade to, so
# these always require the SQL fallback and say so explicitly.
DATABRICKS_ONLY_OPERATIONS: frozenset[Operation] = frozenset(
    op for op, sup in OPERATION_ENGINES.items() if sup.engines == (Engine.SQL,)
)

# The method an engine must implement to serve each operation. An engine may
# only claim an operation it has a method for; `engine.base.missing_method`
# enforces that, and a unit test checks every engine against this table.
ENGINE_METHODS: dict[Operation, str] = {
    Operation.SCAN: "scan",
    Operation.TIME_TRAVEL: "scan",
    Operation.CDF: "cdf",
    Operation.INCREMENTAL: "incremental",
    Operation.HISTORY: "history",
    Operation.DETAIL: "detail",
    Operation.FILES: "files",
    Operation.APPEND: "append",
    Operation.OVERWRITE: "overwrite",
    Operation.REPLACE_WHERE: "overwrite",
    Operation.CREATE: "create",
    Operation.MERGE_SCHEMA: "append",
    Operation.DELETE: "delete",
    Operation.UPDATE: "update",
    Operation.MERGE: "merge",
    Operation.ADD_COLUMN: "add_columns",
    Operation.DROP_COLUMN: "drop_column",
    Operation.RENAME_COLUMN: "rename_column",
    Operation.SET_PROPERTIES: "set_properties",
    Operation.UNSET_PROPERTIES: "unset_properties",
    Operation.ADD_FEATURE: "add_feature",
    Operation.DROP_FEATURE: "drop_feature",
    Operation.ADD_CONSTRAINT: "add_constraint",
    Operation.DROP_CONSTRAINT: "drop_constraint",
    Operation.SET_COMMENT: "set_comment",
    Operation.SET_COLUMN_COMMENT: "set_column_comment",
    Operation.ALTER_COLUMN_TYPE: "alter_column_type",
    Operation.SET_NOT_NULL: "set_not_null",
    Operation.DROP_NOT_NULL: "drop_not_null",
    Operation.CLUSTER_BY: "cluster_by",
    Operation.OPTIMIZE: "optimize",
    Operation.ZORDER: "zorder",
    Operation.VACUUM: "vacuum",
    Operation.RESTORE: "restore",
    Operation.REPAIR: "repair",
    Operation.CHECKPOINT: "checkpoint",
    Operation.LOG_COMPACTION: "compact_logs",
    Operation.CLEANUP_METADATA: "cleanup_metadata",
    Operation.PUBLISH: "publish",
    Operation.REORG: "reorg",
    Operation.CLONE: "clone",
    Operation.CONVERT: "convert",
    Operation.GENERATE: "generate",
    Operation.ANALYZE: "analyze",
    Operation.SYNC_ICEBERG: "sync_iceberg_metadata",
    Operation.REFRESH: "refresh",
}

READ_OPERATIONS: frozenset[Operation] = frozenset(
    {
        Operation.SCAN,
        Operation.TIME_TRAVEL,
        Operation.CDF,
        Operation.INCREMENTAL,
        Operation.HISTORY,
        Operation.DETAIL,
        Operation.FILES,
    }
)

#: Operations that change only the table's metadata, never its data files.
#: The kernel engine serves these for path tables by writing the commit itself.
METADATA_OPERATIONS: frozenset[Operation] = frozenset(
    {
        Operation.ADD_COLUMN,
        Operation.DROP_COLUMN,
        Operation.RENAME_COLUMN,
        Operation.SET_PROPERTIES,
        Operation.UNSET_PROPERTIES,
        Operation.ADD_FEATURE,
        Operation.DROP_CONSTRAINT,
        Operation.SET_COMMENT,
        Operation.SET_COLUMN_COMMENT,
        Operation.ALTER_COLUMN_TYPE,
        Operation.SET_NOT_NULL,
        Operation.DROP_NOT_NULL,
        Operation.CLUSTER_BY,
    }
)


# ---------------------------------------------------------------------------
# Table properties
# ---------------------------------------------------------------------------
#
# Delta configuration is where the two engines diverge most sharply and most
# quietly. delta-rs rejects about half of the spec with one opaque message --
# "Kernel: Generic delta kernel error: Error parsing property" -- and panics
# outright on `delta.minReaderVersion`. The kernel accepts nearly all of it.
#
# The rows below come from probing the installed delta-rs and from reading
# kernel's ALLOWED_DELTA_FEATURES / ALLOWED_DELTA_PROPERTIES. Probe tests in
# `tests/integration/test_properties.py` re-run them, so an engine upgrade that
# changes behavior fails loudly instead of drifting.


class PropertyEffect(StrEnum):
    """What an engine does with a table property."""

    #: Stored, and it changes the protocol or the writer's behavior.
    HONORED = "honored"
    #: Stored verbatim, but nothing here acts on it. Databricks may.
    STORED = "stored"
    #: The engine raises. We refuse first, with a message that names the key.
    REJECTED = "rejected"
    #: The engine panics through the FFI boundary. Never let one reach a user.
    CRASH = "crash"
    #: The engine has no path for this operation at all.
    UNSUPPORTED = "unsupported"

    def usable(self) -> bool:
        return self in (PropertyEffect.HONORED, PropertyEffect.STORED)


@dataclass(frozen=True, slots=True)
class PropertySupport:
    """One row of the property matrix."""

    key: str
    deltars_create: PropertyEffect
    deltars_set: PropertyEffect
    kernel_create: PropertyEffect
    databricks_only: bool = False
    note: str = ""


_H, _ST, _RJ, _CR, _UN = (
    PropertyEffect.HONORED,
    PropertyEffect.STORED,
    PropertyEffect.REJECTED,
    PropertyEffect.CRASH,
    PropertyEffect.UNSUPPORTED,
)


def _prop(
    key: str,
    dc: PropertyEffect,
    dset: PropertyEffect,
    kc: PropertyEffect,
    databricks_only: bool = False,
    note: str = "",
) -> tuple[str, PropertySupport]:
    return key, PropertySupport(key, dc, dset, kc, databricks_only, note)


PROPERTY_SUPPORT: dict[str, PropertySupport] = dict(
    [
        # --- both engines handle these
        _prop("delta.appendOnly", _H, _H, _H),
        _prop("delta.columnMapping.mode", _H, _RJ, _H),
        _prop("delta.enableChangeDataFeed", _H, _RJ, _H),
        _prop(
            "delta.enableDeletionVectors",
            _H,
            _RJ,
            _H,
            False,
            "delta-rs also writes duplicate feature entries and an unexpected "
            "variantType into the protocol when this is enabled at create",
        ),
        _prop("delta.dataSkippingNumIndexedCols", _H, _RJ, _H),
        _prop("delta.checkpoint.writeStatsAsStruct", _H, _RJ, _H),
        # --- delta-rs stores but does not act on
        _prop(
            "delta.checkpointPolicy",
            _ST,
            _RJ,
            _H,
            False,
            "delta-rs stores it but adds no v2Checkpoint feature, so a v2 policy "
            "is inert there; the kernel honors it",
        ),
        _prop("delta.checkpointInterval", _ST, _RJ, _H),
        _prop("delta.logRetentionDuration", _H, _H, _H),
        _prop("delta.deletedFileRetentionDuration", _H, _H, _H),
        _prop("delta.enableExpiredLogCleanup", _ST, _RJ, _H),
        _prop("delta.setTransactionRetentionDuration", _ST, _RJ, _H),
        _prop("delta.dataSkippingStatsColumns", _ST, _RJ, _H),
        _prop("delta.checkpoint.writeStatsAsJson", _ST, _RJ, _H),
        _prop("delta.targetFileSize", _H, _H, _UN, False, "delta-rs-only writer hint"),
        _prop("delta.isolationLevel", _H, _H, _UN),
        _prop("delta.tuneFileSizesForRewrites", _ST, _RJ, _UN, True),
        _prop("delta.autoOptimize.optimizeWrite", _ST, _RJ, _UN, True),
        _prop("delta.autoOptimize.autoCompact", _ST, _RJ, _UN, True),
        _prop("delta.randomizeFilePrefixes", _ST, _RJ, _UN, True),
        # --- kernel only: delta-rs rejects these outright
        _prop("delta.enableRowTracking", _RJ, _RJ, _H),
        _prop("delta.enableInCommitTimestamps", _RJ, _RJ, _H),
        _prop("delta.enableTypeWidening", _RJ, _RJ, _H),
        _prop("delta.enableIcebergCompatV3", _RJ, _RJ, _H),
        _prop("delta.parquet.format.version", _RJ, _RJ, _H),
        # --- neither engine
        _prop(
            "delta.enableIcebergCompatV2",
            _RJ,
            _RJ,
            _UN,
            False,
            "V2 is superseded by V3, which the kernel supports",
        ),
        _prop(
            "delta.universalFormat.enabledFormats",
            _RJ,
            _RJ,
            _UN,
            True,
            "UniForm metadata generation is a Databricks-side job",
        ),
        _prop("delta.parquet.compression.codec", _RJ, _RJ, _UN),
        # --- protocol versions: never set these by hand
        _prop(
            "delta.minReaderVersion",
            _CR,
            _RJ,
            _UN,
            False,
            "delta-rs PANICS ('Reader features should be present in writer "
            "features'), which is not a catchable Python exception. Enable the "
            "feature you want and let the writer raise the version",
        ),
        _prop(
            "delta.minWriterVersion",
            _H,
            _RJ,
            _UN,
            False,
            "raises the protocol without adding the matching features; prefer "
            "enabling features by name",
        ),
    ]
)

#: Feature signals (`delta.feature.<name> = supported`). delta-rs rejects every
#: one; the kernel accepts this subset at create.
KERNEL_CREATE_FEATURES: frozenset[TableFeature] = frozenset(
    {
        TableFeature.DOMAIN_METADATA,
        TableFeature.COLUMN_MAPPING,
        TableFeature.IN_COMMIT_TIMESTAMP,
        TableFeature.VACUUM_PROTOCOL_CHECK,
        TableFeature.CATALOG_MANAGED,
        TableFeature.DELETION_VECTORS,
        TableFeature.V2_CHECKPOINT,
        TableFeature.APPEND_ONLY,
        TableFeature.CHANGE_DATA_FEED,
        TableFeature.TYPE_WIDENING,
        TableFeature.ROW_TRACKING,
        TableFeature.VARIANT_TYPE,
        TableFeature.VARIANT_SHREDDING,
        TableFeature.INVARIANTS,
        TableFeature.MATERIALIZE_PARTITION_COLUMNS,
        TableFeature.ICEBERG_COMPAT_V3,
    }
)

FEATURE_SIGNAL_PREFIX = "delta.feature."


def property_support(key: str) -> PropertySupport:
    """Look up a property, applying the prefix rules for keys with no row.

    Three rules cover everything not listed explicitly:

    * ``delta.feature.<name>`` -- delta-rs rejects all of them. The kernel
      accepts the names in `KERNEL_CREATE_FEATURES`. Note `clustering` is
      deliberately excluded: the kernel wants clustering columns through its
      data layout, not through a feature signal.
    * any other unknown ``delta.*`` key -- delta-rs rejects it; the kernel
      rejects it too, since its allow-list is closed.
    * a custom key outside the ``delta.`` namespace -- delta-rs rejects it,
      which surprises people; the kernel stores it verbatim.
    """
    known = PROPERTY_SUPPORT.get(key)
    if known is not None:
        return known

    if key.startswith(FEATURE_SIGNAL_PREFIX):
        name = key[len(FEATURE_SIGNAL_PREFIX) :]
        feature = feature_from_wire(name)
        kernel = (
            PropertyEffect.HONORED
            if feature is not None and feature in KERNEL_CREATE_FEATURES
            else PropertyEffect.REJECTED
        )
        note = "delta-rs rejects every delta.feature.* signal"
        if name == "clustering":
            note += (
                "; the kernel rejects this one too -- pass cluster_by= instead, "
                "which sets clustering through its data layout"
            )
        return PropertySupport(key, _RJ, _RJ, kernel, note=note)

    if key.startswith("delta."):
        return PropertySupport(
            key, _RJ, _RJ, _RJ, note="unrecognized delta.* key; both allow-lists are closed"
        )

    return PropertySupport(
        key,
        _RJ,
        _RJ,
        _ST,
        note="a custom key outside the delta. namespace: delta-rs rejects it, the kernel stores it",
    )
