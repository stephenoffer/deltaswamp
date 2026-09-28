"""The delta-kernel engine: the default read path.

The kernel checks feature support per operation and ignores writer-only
features when reading, so it opens tables delta-rs refuses: every
`catalogManaged` table, and anything with type widening, shredded variants or
`vacuumProtocolCheck`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, ClassVar

from .. import predicate as sqlpred
from .._sdk import PRODUCT, sdk_version
from .._storage import engine_options, location_refusal, store_options, write_refusal
from .._util import commit_backoff, timestamp_ms
from ..capability import (
    FEATURE_DEPENDENCIES,
    FEATURE_SUPPORT,
    METADATA_OPERATIONS,
    READ_OPERATIONS,
    Capability,
    Operation,
    Support,
    TableFeature,
    feature_from_wire,
    implied_features,
)
from ..capability import (
    Engine as EngineKind,
)
from ..catalog import ResolvedTable
from ..credentials import Operation as CredentialOperation
from ..errors import (
    SQL_FALLBACK_REMEDY,
    CommitConflictError,
    EngineLimitError,
    InvalidArgumentError,
    MetadataChangedError,
    UnreachableTableError,
)
from ..properties import (
    CHECKPOINT_STATS_REMEDY,
    KERNEL_CREATE_DEFERRED,
    checkpoint_drops_stats,
    effect_for,
    parse_byte_size,
)
from . import metadata as meta
from .base import missing_file_error, missing_method
from .boundary import conflict_version, engine_cause, recorded, translating
from .metadata import CLUSTERING_DOMAIN, TableState, arrow_to_delta_field, build_actions

__all__ = ["KernelEngine"]


def _engine_info() -> str:
    """The `engineInfo` recorded in every commit this library writes."""
    return f"{PRODUCT}/{sdk_version()}"


def _as_record_batch_reader(data: Any) -> Any:
    """Coerce any Arrow-ish input into a RecordBatchReader.

    Accepts anything exporting the Arrow PyCapsule interface, which is the point
    of the interface: pandas, Polars, DuckDB and pyarrow objects all qualify
    without us importing any of them.
    """
    if hasattr(data, "to_reader"):
        return data.to_reader()
    if hasattr(data, "__arrow_c_stream__"):
        return data
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover
        raise UnreachableTableError(
            "write",
            "the data does not export the Arrow PyCapsule interface and pyarrow is not "
            "installed to convert it",
        ) from exc
    batches = isinstance(data, (list, tuple)) and data
    if batches and all(isinstance(b, pa.RecordBatch) for b in data):
        # A list of batches, as delta-rs takes one: pa.table() reads a list
        # as columns and raised "Must pass names or schema".
        return pa.RecordBatchReader.from_batches(data[0].schema, data)
    return pa.table(data).to_reader()


# Operations that write through the kernel. Each is checked against the
# table's writer features before it is claimed.
_WRITE_OPS: frozenset[Operation] = frozenset(
    {
        Operation.APPEND,
        Operation.CREATE,
        Operation.OVERWRITE,
        Operation.REPLACE_WHERE,
        Operation.DELETE,
        Operation.UPDATE,
        Operation.MERGE,
    }
)

#: Served by rewriting the whole table in one commit: correct on any table the
#: kernel can write, and bounded by `KernelEngine.rewrite_max_bytes`.
_REWRITE_OPS: frozenset[Operation] = frozenset(
    {Operation.REPLACE_WHERE, Operation.DELETE, Operation.UPDATE}
)

#: Call options the kernel's write paths do not implement, by operation. One
#: table for both sides: `supports()` refuses a call that passes one, so the
#: router moves on to an engine that takes it and `Table.can()` says so, and
#: the write path refuses the same set rather than dropping it. Refused only
#: inside the write, can() named the kernel and the call then failed.
_UNIMPLEMENTED_OPTIONS: dict[Operation, frozenset[str]] = {
    Operation.APPEND: frozenset({"target_file_size", "writer_properties", "partition_by"}),
    Operation.OVERWRITE: frozenset({"target_file_size", "writer_properties", "schema_mode"}),
    Operation.REPLACE_WHERE: frozenset(
        {"target_file_size", "writer_properties", "schema_mode", "max_commit_retries"}
    ),
    # A rewrite is staged against the snapshot it read and is not re-staged on
    # a conflict, so it has no retries to bound.
    Operation.DELETE: frozenset({"writer_properties", "max_commit_retries"}),
    Operation.UPDATE: frozenset(
        {"writer_properties", "max_commit_retries", "error_on_type_mismatch"}
    ),
}


def _options_refusal(operation: Operation, shape: dict[str, Any]) -> str | None:
    given = sorted(k for k in _UNIMPLEMENTED_OPTIONS.get(operation, ()) if shape.get(k) is not None)
    if not given:
        return None
    return f"the kernel {operation.value} path does not implement {', '.join(given)}"


#: What Unity Catalog's committer checks on every commit to a catalog-managed
#: table, and refuses with a generic error once the data files are written.
_UC_COMMIT_FEATURES = ("vacuumProtocolCheck", "inCommitTimestamp")


def _uc_commit_refusal(table: ResolvedTable) -> str | None:
    missing = [f for f in _UC_COMMIT_FEATURES if f not in table.effective_writer_features]
    if "vacuumProtocolCheck" not in table.effective_reader_features:
        missing = sorted({*missing, "vacuumProtocolCheck"})
    if missing:
        return (
            "Unity Catalog's committer requires a catalog-managed table to carry the "
            f"{', '.join(missing)} table feature(s), and this one does not"
        )
    if table.properties.get("io.unitycatalog.tableId") is None:
        return (
            "Unity Catalog's committer requires io.unitycatalog.tableId in the table's "
            "configuration, and this catalog-managed table has none"
        )
    if str(table.properties.get("delta.enableInCommitTimestamps", "")).lower() != "true":
        return (
            "Unity Catalog's committer requires delta.enableInCommitTimestamps=true on a "
            "catalog-managed table"
        )
    return None


#: Operations whose commit stages remove actions. Kernel 0.28 refuses those on
#: a row-tracked table: it cannot preserve the row ids of what it removes, and
#: it refuses at commit -- after the data files are already written.
_REMOVE_OPS: frozenset[Operation] = _REWRITE_OPS | {Operation.OVERWRITE}


def _row_tracking_enabled(table: ResolvedTable) -> bool:
    return str(table.properties.get("delta.enableRowTracking", "false")).lower() == "true"


def _keeps_row_ids(table: ResolvedTable) -> bool:
    """Whether DV DML here can keep every row id stable.

    An UPDATE on a table with row tracking enabled needs the table's
    materialized row-id column to carry the old ids into the new files, and a
    native build that writes it. Where the feature is merely supported, ids are
    assigned but not promised stable, so nothing needs carrying.
    """
    if not _row_tracking_enabled(table):
        return True
    return bool(
        table.properties.get("delta.rowTracking.materializedRowIdColumnName")
    ) and _native_has("materialized_row_ids")


def deletion_vectors_writable(table: ResolvedTable) -> bool:
    """Whether DML on `table` may be written as deletion vectors.

    The Delta protocol permits DV writes only when the table supports the
    feature on both sides and `delta.enableDeletionVectors` is true; a table
    that merely supports the feature still takes copy-on-write, as in Spark.
    """
    return (
        "deletionVectors" in table.reader_features
        and "deletionVectors" in table.writer_features
        and str(table.properties.get("delta.enableDeletionVectors", "false")).lower() == "true"
    )


#: Features that block a write only when the table really uses them.
#:
#: Just one qualifies, and the distinction is the kernel's, not ours. Writer
#: version 2 implies `invariants` for nearly every legacy table and the kernel
#: writes those happily: it inspects the schema and fails only where one really
#: exists. `checkConstraints` is the opposite -- the kernel refuses any table
#: whose protocol carries it, used or not, so a table at writer version 3 or
#: above cannot take a kernel write at all even with no constraint defined.
#: Verified rather than assumed: a version 4 change-data-feed table with no
#: constraints is still refused by the kernel, so gating that one on usage
#: would put the refusal back after the data was written.
_USAGE_GATED: dict[TableFeature, tuple[str, str]] = {
    TableFeature.INVARIANTS: (
        "has_invariants",
        "a column of this table carries a Delta invariant, which the kernel cannot evaluate",
    ),
}

#: Features a create's column metadata can call for that the kernel's
#: CREATE TABLE refuses to declare (`delta.feature.X` is rejected at create,
#: and invariants fail schema validation).
_CREATE_UNDECLARABLE: frozenset[str] = frozenset(
    {"generatedColumns", "identityColumns", "allowColumnDefaults", "invariants"}
)

_IMPLEMENTED: frozenset[Operation] = frozenset(
    {
        Operation.SCAN,
        Operation.TIME_TRAVEL,
        Operation.DETAIL,
        Operation.APPEND,
        Operation.CREATE,
        Operation.OVERWRITE,
        Operation.PUBLISH,
        Operation.MERGE,
    }
    | _REWRITE_OPS
)


_SHREDDING: frozenset[str] = frozenset({"variantShredding", "variantShredding-preview"})


#: Features that block even a metadata-only commit written here: their
#: semantics live in the schema or the log in ways this path does not model.
_METADATA_BLOCKERS: frozenset[TableFeature] = frozenset(
    {
        TableFeature.COLLATIONS,
        TableFeature.COLLATIONS_PREVIEW,
        TableFeature.CATALOG_MANAGED,
        TableFeature.CATALOG_OWNED_PREVIEW,
        TableFeature.ADAPTIVE_METADATA_PREVIEW,
        TableFeature.GEOSPATIAL,
        # Governs which checkpoints may be removed; readable, but a commit
        # written without modelling it can corrupt history.
        TableFeature.CHECKPOINT_PROTECTION,
    }
)


#: The process that first called into the native runtime (its pid, once set).
_RUNTIME_OWNER: list[int] = []


#: Keys the store fingerprint with a secret of this process's own, so the
#: cache holds no digest of a credential that could be checked offline.
_FINGERPRINT_KEY = os.urandom(32)


def _store_fingerprint(options: dict[str, str]) -> str:
    """Which store `options` reach, as a keyed digest: the snapshot cache's key.

    Every option counts -- endpoint, region, account, and the credential
    itself -- because any of them can make one URL a different table (two
    endpoints, two accounts) or an unreadable one (a principal without
    access). Secrets go into a keyed hash, never into the key in the clear.
    """
    import hashlib

    canonical = json.dumps(sorted((str(k).lower(), str(v)) for k, v in options.items()))
    return hashlib.blake2b(canonical.encode(), key=_FINGERPRINT_KEY, digest_size=16).hexdigest()


def _forked_on_macos() -> bool:
    """A macOS child forked from a process whose native runtime had started.

    macOS kills such a child with SIGTRAP on its first native call (the
    runtime's thread machinery does not survive fork there) -- an
    uncatchable crash of the worker, where Linux gets a fresh runtime.
    """
    import os
    import sys

    return sys.platform == "darwin" and bool(_RUNTIME_OWNER) and _RUNTIME_OWNER[0] != os.getpid()


def _enter_native(what: str) -> None:
    """Record the runtime's owner, or refuse a call that would crash the process."""
    import os

    if not _RUNTIME_OWNER:
        _RUNTIME_OWNER.append(os.getpid())
    elif _forked_on_macos():
        raise EngineLimitError(
            what,
            "the kernel cannot run in a process forked on macOS from one that had already "
            "used it: the child would crash (SIGTRAP) on its first native call",
            "start worker processes with multiprocessing's 'spawn' (the macOS default) "
            "or 'forkserver' method",
        )


def _commit_info(*, blind: bool, **parameters: Any) -> dict[str, Any]:
    """`operationParameters` and `isBlindAppend` for a kernel commit, as the binding takes them.

    Kernel 0.28 writes an empty parameter map and writes `isBlindAppend` only
    when true, so a replaceWhere that matched nothing (only adds) read as a
    blind append to every concurrent writer's conflict check, and four
    concurrent loads of one range each committed. Values follow Spark's
    history: strings as they are, anything else as JSON (``partitionBy``
    ``[]``, ``predicate`` ``["day = '2026-09-27'"]``).
    """
    if not _native_has("commit_info_patch"):
        return {}
    return {
        "operation_parameters": {
            key: value if isinstance(value, str) else json.dumps(value)
            for key, value in parameters.items()
            if value is not None
        },
        "blind_append": blind,
    }


def _blind_append(info: dict[str, Any]) -> bool:
    """Whether a winner's commitInfo says it was a blind append (given it only adds).

    `isBlindAppend` decides wherever it is written: Spark writes it, false
    for an INSERT ... SELECT, and so does this library. delta-rs never writes
    it, but records ``mode: Append`` for exactly its appends -- writes built
    from data it was handed, which read nothing of the table, as Spark's
    DataFrame appends are blind. A WRITE without either (a replaceWhere, or a
    kernel commit from before the flag was written) is not.
    """
    flag = info.get("isBlindAppend")
    if flag is not None:
        return flag is True
    parameters = info.get("operationParameters") or {}
    return (
        str(info.get("operation", "")).upper() == "WRITE"
        and str(parameters.get("mode", "")).lower() == "append"
        and not parameters.get("predicate")
    )


def _write_info(snapshot: Any, *, overwrite: bool, predicate: str | None = None) -> dict[str, Any]:
    """`_commit_info` for a WRITE: an append (blind) or an overwrite / replaceWhere."""
    try:
        partitions = list(snapshot.partition_columns)
    except Exception:
        partitions = []
    return _commit_info(
        blind=not overwrite and predicate is None,
        mode="Overwrite" if overwrite or predicate is not None else "Append",
        partitionBy=partitions,
        predicate=None if predicate is None else [predicate],
    )


def _dml_info(operation: str, predicate: str | None, snapshot: Any) -> dict[str, Any]:
    """`_commit_info` for a DELETE, UPDATE, MERGE or replaceWhere: never a blind append."""
    if operation.upper() == "WRITE":
        return _write_info(snapshot, overwrite=True, predicate=predicate)
    return _commit_info(blind=False, predicate=[predicate] if predicate is not None else [])


def _native_has(*features: str) -> bool:
    """Whether the compiled extension advertises every one of `features`.

    Capabilities that depend on newer native functions are claimed only when
    the installed build actually has them, so a stale build refuses cleanly
    instead of failing with AttributeError halfway through.
    """
    try:
        from deltaswamp import _native
    except ImportError:
        return False
    return set(features) <= set(getattr(_native, "FEATURES", ()))


def write_checksum(
    location: str | None, options: dict[str, str] | None, version: int | None = None
) -> bool:
    """Write `_delta_log/<version>.crc` for a commit the kernel did not make; best effort.

    Kernel commits write their own (`crate::checksum`); delta-rs and a raw
    commit write none, and Databricks, which keeps one per version, read a
    table this library wrote as one whose chain broke at every append
    (delta-rs#4190). Written only where the native side finds it cheap (a
    checksum a few commits back, or a short log), and never raising: the
    commit it follows has already succeeded.
    """
    if location is None or not _native_has("write_checksum"):
        return False
    try:
        _enter_native("write a version checksum")
        from deltaswamp._native import Snapshot

        snapshot = Snapshot.resolve(
            location, options=options or None, version=None if version is None else int(version)
        )
        return bool(snapshot.write_checksum())
    except Exception:
        return False


def _implemented() -> frozenset[Operation]:
    ops = set(_IMPLEMENTED)
    if _native_has("commit_raw", "metadata_json"):
        ops |= METADATA_OPERATIONS
    if _native_has("table_changes"):
        ops.add(Operation.CDF)
    if _native_has("files"):
        ops.add(Operation.FILES)
    if _native_has("checkpoint"):
        ops.add(Operation.CHECKPOINT)
    if _native_has(*_COMPACTION_NATIVE):
        ops |= {Operation.OPTIMIZE, Operation.ZORDER}
    return frozenset(ops)


class KernelEngine:
    """Reads Delta tables through delta-kernel-rs."""

    kind = EngineKind.KERNEL

    @property
    def supports_distributed_scan(self) -> bool:
        """Workers can each read a planned subset of a snapshot's files."""
        return _native_has("file_restricted_scan", "files")

    @property
    def supports_distributed_write(self) -> bool:
        """Workers can write data files that a coordinator commits together."""
        return _native_has("distributed_write")

    #: SQL predicates are parsed here: the kernel skips files with the
    #: structured form, and the exact row filter is applied afterwards.
    supports_predicates = True
    #: None of these are bound in the native extension yet. Declaring them false
    #: makes the router divert the call rather than letting it be dropped.
    supports_schema_merge = False
    supports_schema_overwrite = False
    supports_idempotent_txn = True
    supports_commit_metadata = True
    supports_writer_properties = False
    supports_dynamic_overwrite = False
    #: Negative fractional decimal partition values are serialized correctly.
    supports_negative_decimal_partition_values = True
    #: history_manager resolves a timestamp to the latest recreatable version,
    #: honoring in-commit timestamps.
    supports_timestamp_travel = True
    #: DELETE/UPDATE/replaceWhere evaluate their SQL here, and only the
    #: predicate grammar (comparisons, IN, BETWEEN, LIKE, IS NULL, AND/OR/NOT
    #: over columns and literals; SET values a literal or a column). Arithmetic
    #: and function calls go to an engine that evaluates SQL.
    supports_sql_expressions = False

    #: Resolved snapshots kept per (location, version, store) for reuse.
    snapshot_cache_size = 16
    # Every metadata call and every routing decision resolved the log from
    # scratch: 0.5 s each on a table 5000 commits past its last checkpoint,
    # four of them for one UPDATE, and one per WritePlan.write(). A cached
    # snapshot is revalidated and brought up to date instead (see
    # Snapshot.refresh: a real read of the strong identity of the commit file
    # it ends at, then one listing for the latest), so reads still see every
    # commit, anyone's. Per process rather than per engine: a distributed write
    # unpickles a fresh engine with every task. Keyed by the store as well as
    # the path: one s3:// URL on two endpoints or accounts is two tables, and
    # sharing entries handed one connection's schema -- and its files -- to the
    # other, or to a connection that could not reach storage at all. Commits
    # never reuse an entry (see `snapshot(fresh=)`).
    _snapshots: ClassVar[OrderedDict[tuple[Any, ...], Any]] = OrderedDict()
    _snapshots_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, *, storage_options: dict[str, str] | None = None) -> None:
        self._base_options = dict(storage_options or {})

    # ----------------------------------------------------------- capabilities

    def supports(self, operation: Operation, table: ResolvedTable, **shape: Any) -> Capability:
        if not self.available():
            return Capability(
                operation,
                ok=False,
                reason="the deltaswamp native extension is not installed",
                remedy="pip install deltaswamp (a wheel with the compiled kernel binding)",
            )
        if _forked_on_macos():
            return Capability(
                operation,
                ok=False,
                reason="the kernel cannot run in this process: it was forked on macOS from "
                "one that had already used it, and would crash on its first call",
                remedy="start worker processes with multiprocessing's 'spawn' or 'forkserver' "
                "method",
            )

        # Structural guard: never claim an operation with no method behind it.
        gap = missing_method(self, operation)
        if gap is not None:
            return gap

        if operation not in _implemented():
            return Capability(
                operation,
                ok=False,
                reason=f"the kernel engine does not implement {operation.value} yet",
            )

        if operation is Operation.CHECKPOINT and checkpoint_drops_stats(table.properties):
            return Capability(
                operation,
                ok=False,
                reason=_DROPS_STATS_REASON,
                remedy=CHECKPOINT_STATS_REMEDY,
            )

        if not table.is_delta:
            return Capability(operation, ok=False, reason="the table is not a Delta table")

        # The kernel create writes version 0 only; claiming mode='overwrite'
        # here and refusing in create() left no engine to fall through to.
        if operation is Operation.CREATE and shape.get("mode") not in (None, "error", "create"):
            return Capability(
                operation,
                ok=False,
                reason=f"the kernel create path writes version 0 of a new table only, "
                f"not mode={shape.get('mode')!r}",
            )

        if table.location is None:
            return Capability(
                operation,
                ok=False,
                reason="the table has no storage location, so there are no files to read",
            )

        unreachable = location_refusal(table.location)
        if unreachable is not None:
            return Capability(operation, ok=False, reason=unreachable)

        if operation not in READ_OPERATIONS and not table.is_catalog_managed:
            unsafe = write_refusal(table.location, self._base_options, table.credential_provider)
            if unsafe is not None:
                # The remedy goes in the reason too: the router reports only
                # the reasons when every engine refuses.
                reason, remedy = unsafe
                return Capability(operation, ok=False, reason=f"{reason}; {remedy}", remedy=remedy)

        # Writer-only features never block a read; only reader and
        # reader-writer features can. This asymmetry is the whole point.
        blockers: list[str] = []
        for name in table.effective_reader_features:
            feature = feature_from_wire(name)
            if feature is None:
                blockers.append(f"{name} (unrecognized reader feature)")
            elif FEATURE_SUPPORT[feature].kernel_read is Support.NO:
                blockers.append(name)

        if blockers:
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table requires reader features the kernel cannot read: "
                    + ", ".join(sorted(blockers))
                ),
            )

        if operation in (Operation.OPTIMIZE, Operation.ZORDER):
            # Its own gate, not the write gate below: see _compaction_capability.
            return self._compaction_capability(operation, table, shape)

        if operation in METADATA_OPERATIONS:
            refusal = self._metadata_refusal(operation, table)
            if refusal is not None:
                return refusal

        if operation is Operation.ADD_FEATURE and shape.get("features") is not None:
            refusal = self._add_feature_refusal(table, shape["features"])
            if refusal is not None:
                return refusal

        unimplemented = _options_refusal(operation, shape)
        if unimplemented is not None:
            return Capability(operation, ok=False, reason=unimplemented)

        if (
            (operation in _WRITE_OPS or operation in METADATA_OPERATIONS)
            and operation is not Operation.CREATE
            and table.is_catalog_managed
        ):
            uncommittable = _uc_commit_refusal(table)
            if uncommittable is not None:
                return Capability(operation, ok=False, reason=uncommittable)

        if operation is Operation.MERGE:
            refusal = self._merge_refusal(table)
            if refusal is not None:
                return refusal

        by_dv = operation in _REWRITE_OPS and self._dv_path(table)
        row_tracked = "rowTracking" in table.effective_writer_features
        if row_tracked and operation in _REMOVE_OPS and not (by_dv and _keeps_row_ids(table)):
            # Checked for every remove-staging operation, not just the rewrites:
            # a kernel overwrite removes every visible file in the same commit,
            # and the commit is refused after the data is written. Through
            # deletion vectors no file is removed and every surviving row keeps
            # its baseRowId; an UPDATE also writes each rewritten row's old id
            # into the materialized row-id column, so ids stay stable.
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table tracks row ids, and this commit would remove rows whose "
                    "ids delta-kernel 0.28 cannot preserve"
                ),
                remedy=SQL_FALLBACK_REMEDY,
            )

        if operation in _REWRITE_OPS:
            refusal = self._rewrite_refusal(operation, table, by_dv=by_dv)
            if refusal is not None:
                return refusal

        if operation in _WRITE_OPS or operation in METADATA_OPERATIONS:
            write_blockers: list[str] = []
            usage_blockers: list[str] = []
            metadata_only = operation in METADATA_OPERATIONS
            for name in table.effective_writer_features:
                feature = feature_from_wire(name)
                if feature is None:
                    write_blockers.append(f"{name} (unrecognized writer feature)")
                elif metadata_only:
                    # A metadata-only commit writes no data, so features that
                    # govern data (constraints, generated and identity columns)
                    # do not block it. Ones that change what a schema *means* do.
                    #
                    # A feature the kernel can neither read nor write is one it
                    # cannot model at all, and the protocol is explicit that a
                    # writer must not write to a table carrying a writer feature
                    # it does not support -- metadata-only commits included,
                    # since an unmodelled feature may constrain every commit.
                    # `checkpointProtection` is exactly that: it governs which
                    # checkpoints may be removed, so writing blind can corrupt
                    # history rather than merely losing an edit.
                    support = FEATURE_SUPPORT[feature]
                    unmodelled = (
                        support.kernel_read is Support.NO and support.kernel_write is Support.NO
                    )
                    if feature in _METADATA_BLOCKERS or unmodelled:
                        write_blockers.append(name)
                elif feature in _USAGE_GATED:
                    flag, why = _USAGE_GATED[feature]
                    if getattr(table, flag, False):
                        usage_blockers.append(why)
                elif FEATURE_SUPPORT[feature].kernel_write is Support.NO:
                    write_blockers.append(name)
            if write_blockers:
                return Capability(
                    operation,
                    ok=False,
                    reason=(
                        "the kernel cannot write these table features: "
                        + ", ".join(sorted(write_blockers))
                        + self._implied_only_note(table, write_blockers)
                    ),
                    remedy="write through delta-rs, which serves this table",
                )
            if usage_blockers and not metadata_only:
                # These refuse at commit, after the data files are written, or
                # worse write data the feature should have checked. delta-rs
                # evaluates them, so it serves these tables.
                return Capability(
                    operation,
                    ok=False,
                    reason="; ".join(sorted(usage_blockers)),
                    remedy="write through delta-rs, which evaluates these",
                )
            if operation in _WRITE_OPS and operation is not Operation.CREATE:
                refusal = self._data_write_refusal(operation, table)
                if refusal is not None:
                    return refusal
            if (
                table.partition_columns
                and (
                    operation in (Operation.APPEND, Operation.OVERWRITE)
                    or operation in _REWRITE_OPS
                )
                and not _native_has("partitioned_append")
            ):
                # The write path here builds one unpartitioned context, so this
                # would put every row in the root with no partition values.
                return Capability(
                    operation,
                    ok=False,
                    reason=(
                        "the table is partitioned by "
                        f"{', '.join(table.partition_columns)}, and the kernel write "
                        "path handles unpartitioned writes only"
                    ),
                )

        if (
            operation is Operation.CDF
            and table.properties.get("delta.enableChangeDataFeed", "false").lower() != "true"
        ):
            # Decided here, not in cdf(). A table without a change feed has
            # nothing for any engine to read, and reporting it only when the
            # call is made makes capabilities() claim a feed that is not there
            # -- which is the whole reason to ask before calling.
            return Capability(
                operation,
                ok=False,
                reason=(
                    "delta.enableChangeDataFeed is not enabled on this table, so there is "
                    "no change feed to read. Enabling it is not retroactive: only changes "
                    "after enablement are recorded"
                ),
            )

        if operation in (Operation.SCAN, Operation.TIME_TRAVEL) and table.is_shallow_clone:
            # Its add actions point at the source table's files by absolute path.
            # Kernel resolves those before we see them, so we cannot scope
            # credentials correctly or tell borrowed files from owned ones.
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table is a shallow clone; its data files belong to the source "
                    "table and are referenced by absolute path, which cannot be "
                    "credential-scoped reliably"
                ),
                remedy="read the source table directly, or use Compatibility Mode",
            )

        if operation is Operation.CDF and table.is_catalog_managed:
            # cdf() refuses these; claiming them here kept the router from
            # diverting to an engine that can serve them.
            return Capability(
                operation,
                ok=False,
                reason="the kernel's TableChanges takes no catalog commit tail, so it would "
                "miss ratified-but-unpublished commits of a catalog-managed table",
                remedy="ds.connect(..., allow_sql_fallback=True) reads it with table_changes()",
            )

        if operation is Operation.CREATE:
            unusable = [
                key
                for key in (shape.get("properties") or {})
                if not effect_for(key, EngineKind.KERNEL, operation).usable()
            ]
            if unusable:
                return Capability(
                    operation,
                    ok=False,
                    reason=(
                        "the kernel cannot handle these table properties at create: "
                        + ", ".join(sorted(unusable))
                    ),
                )
            deferred = sorted(set(shape.get("properties") or {}) & KERNEL_CREATE_DEFERRED)
            if deferred and table.is_catalog_managed:
                # create() commits these as version 1, a metadata commit that
                # on a catalog-managed table would bypass the catalog.
                return Capability(
                    operation,
                    ok=False,
                    reason=(
                        "the kernel refuses these table properties at create, and the "
                        "follow-up metadata commit that applies them would bypass the "
                        "catalog: " + ", ".join(deferred)
                    ),
                    remedy="create the table, then set them through the catalog",
                )
            declared = sorted(
                (meta.create_schema_features(shape.get("schema")) or set()) & _CREATE_UNDECLARABLE
            )
            if declared:
                # delta-kernel 0.28 refuses these features at CREATE, and left
                # to itself committed the column metadata without them: a
                # table whose generation expressions nothing enforces.
                return Capability(
                    operation,
                    ok=False,
                    reason=(
                        "the schema's column metadata needs the "
                        + ", ".join(declared)
                        + " feature, which the kernel cannot declare when it creates a table"
                    ),
                )

        return Capability(operation, ok=True, engine=self.kind)

    @staticmethod
    def available() -> bool:
        try:
            import deltaswamp._native  # noqa: F401
        except ImportError:
            return False
        return True

    # ------------------------------------------------------------------- read

    def snapshot(
        self,
        table: ResolvedTable,
        *,
        version: int | None = None,
        timestamp: Any = None,
        write: bool = False,
        fresh: bool | None = None,
    ) -> Any:
        """Resolve a kernel snapshot, supplying the catalog tail when needed.

        `fresh` (default: `write`) resolves the log from storage rather than
        reusing a cached snapshot. Every commit path passes write=True and so
        builds on the table as it is now: a commit built on a revalidated but
        cached snapshot trusts that revalidation with the table's contents,
        and a stale one removed rows from a table re-created at the same path
        and added files written for the old schema.
        """
        from deltaswamp._native import Snapshot

        _enter_native("open the table with the kernel")
        if table.location is None:
            raise UnreachableTableError("open", "the table has no storage location", None)
        if version is not None and timestamp is not None:
            raise UnreachableTableError("time travel", "pass a version or a timestamp, not both")
        if version is not None:
            # operator.index keeps numpy integers (e.g. from a history frame)
            # working, as the binding's own integer extraction accepted them.
            import operator

            try:
                if isinstance(version, bool):
                    raise TypeError
                version = operator.index(version)
            except TypeError:
                raise UnreachableTableError(
                    f"read version {version!r}", "a version is an integer"
                ) from None
        if version is not None and version < 0:
            # The binding takes an unsigned version and raised OverflowError.
            raise UnreachableTableError(f"read version {version}", "versions start at 0")

        # `log_tail` and `max_catalog_version` are what make a catalog-managed
        # table readable; both are meaningless (and omitted) otherwise.
        log_tail = [
            (entry.version, entry.path, entry.timestamp or 0, entry.size)
            for entry in table.log_tail
        ] or None

        location: str = table.location
        # Only a path-based read by version (or latest) is cached: a
        # catalog-managed table's latest version is the catalog's to say,
        # which a log listing cannot see.
        cacheable = log_tail is None and table.max_catalog_version is None and timestamp is None
        reuse = cacheable and not (write if fresh is None else fresh)

        def resolve() -> Any:
            options = self._options(table, write=write)
            key = (location, version, _store_fingerprint(options), table.table_id)
            if reuse:
                with self._snapshots_lock:
                    cached = self._snapshots.get(key)
                if cached is not None:
                    try:
                        refreshed = cached.refresh(options, latest=version is None)
                    except Exception:
                        refreshed = None  # resolved afresh below, with its own errors
                    if refreshed is not None:
                        self._remember(key, refreshed)
                        return refreshed
            snapshot = Snapshot.resolve(
                location,
                options=options,
                version=version,
                log_tail=log_tail,
                max_catalog_version=table.max_catalog_version,
                timestamp_ms=timestamp_ms(timestamp) if timestamp is not None else None,
                # Records the identity a later reuse is revalidated against.
                identify=cacheable,
            )
            if cacheable:
                self._remember(key, snapshot)
            return snapshot

        try:
            try:
                return resolve()
            except Exception as exc:
                from ..errors import DeltaSwampError

                provider = table.credential_provider
                if (
                    provider is None
                    # Vending itself failed: re-vending at once would only
                    # repeat it (the SDK has already retried).
                    or isinstance(exc, DeltaSwampError)
                    or not _rejected_credential(str(exc))
                ):
                    raise
                # Storage refused a credential the cache still thought valid
                # (revoked, clocks disagreeing, expired early). Nothing called
                # the provider's invalidate(), so the dead credential kept
                # being served -- every call failing -- until its stated
                # expiry. Re-vend once and try again.
                provider.invalidate()
                return resolve()
        except ValueError as exc:
            if "earliest recreatable" in str(exc):
                raise UnreachableTableError(f"read the table as of {timestamp}", str(exc)) from exc
            if version is not None and (
                "not the same as the specified end version" in str(exc)
                # A catalog-managed table past its ratified version: this
                # escaped as a bare ValueError, not a library error.
                or "exceeds max catalog version" in str(exc)
            ):
                raise UnreachableTableError(
                    f"read version {version}",
                    f"the table has no such version ({exc})",
                    "time travel to a committed version, or omit it for the latest",
                ) from exc
            if version is not None and "No files in log segment" in str(exc):
                # The commits up to `version` may be gone while the table is
                # not; that escaped as a bare ValueError. Tell the two apart.
                try:
                    Snapshot.resolve(
                        table.location,
                        options=self._options(table, write=write),
                        log_tail=log_tail,
                        max_catalog_version=table.max_catalog_version,
                    )
                except Exception:
                    raise exc from None
                raise UnreachableTableError(
                    f"read version {version}",
                    "the table exists, but its log no longer holds the commits or checkpoint "
                    "to reconstruct this version: they were removed by log retention or "
                    "metadata cleanup",
                    "time travel to a version at or after the table's oldest checkpoint",
                ) from exc
            raise

    def forget(self, location: str) -> None:
        """Drop every snapshot cached for the table at `location`."""
        with self._snapshots_lock:
            for key in [k for k in self._snapshots if k[0] == location]:
                self._snapshots.pop(key, None)

    def _remember(self, key: tuple[Any, ...], snapshot: Any) -> None:
        if getattr(snapshot, "commit_identity", None) is None:
            # Nothing to revalidate it against (no commit file at its version,
            # or a store without strong change tokens): never reused.
            with self._snapshots_lock:
                self._snapshots.pop(key, None)
            return
        with self._snapshots_lock:
            self._snapshots[key] = snapshot
            self._snapshots.move_to_end(key)
            while len(self._snapshots) > self.snapshot_cache_size:
                self._snapshots.popitem(last=False)

    def scan(
        self,
        table: ResolvedTable,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: Any = None,
        limit: int | None = None,
    ) -> Any:
        """Read with deletion vectors applied; a predicate skips files, then filters.

        The kernel only uses a predicate to skip files and never drops a row,
        so the exact filter is applied here -- from the same parsed predicate,
        so both halves mean the same thing. Columns the predicate needs but the
        caller did not ask for are read and then dropped.
        """
        stream = self._scan(table, columns, predicate, version, timestamp)
        if table.features & _SHREDDING and importlib.util.find_spec("pyarrow") is not None:
            # Databricks puts variantShredding (and enableVariantShredding) on
            # every VARIANT table, but whether a file is actually shredded
            # depends on its data, and the kernel fails on the ones that are.
            # Reading eagerly raises that failure here, inside the call, where
            # Table can hand the read to the next engine -- a lazy stream would
            # raise it mid-iteration, past the point of any retry.
            import pyarrow as pa

            try:
                return pa.table(stream)
            except pa.ArrowInvalid as exc:
                if "shredded" not in str(exc).lower():
                    raise
                # A limit of this engine, not of the request: Table moves on to
                # the warehouse if there is one, and otherwise says why.
                raise EngineLimitError(
                    "read a VARIANT column",
                    "a data file holds shredded VARIANT values, which the kernel does not "
                    f"decode ({str(exc).splitlines()[0][:160]})",
                    "read the other columns (columns=[...]), or ds.connect(..., "
                    "allow_sql_fallback=True) to read through a SQL warehouse",
                ) from exc
        return stream

    def _scan(
        self,
        table: ResolvedTable,
        columns: list[str] | None,
        predicate: str | None,
        version: int | None,
        timestamp: Any,
    ) -> Any:
        snapshot = self.snapshot(table, version=version, timestamp=timestamp)
        return _planned_read(snapshot, columns, predicate)

    def legacy_calendar_files(
        self, table: ResolvedTable, *, version: int | None = None, timestamp: Any = None
    ) -> tuple[str, ...] | None:
        """Live files a reader that does not rebase would misread; None if unknowable.

        Files Spark wrote in its legacy hybrid calendar with a value the rebase
        moves, or with INT96 timestamps that overflow as nanoseconds (before
        1677 or after 2262). See `engine/calendar.py`.
        """
        if not self.available() or not _native_has("legacy_calendar_files"):
            return None
        from .calendar import legacy_calendar_files

        return legacy_calendar_files(self.snapshot(table, version=version, timestamp=timestamp))

    @property
    def supports_read_sql_expressions(self) -> bool:
        """A read can filter by SQL outside the grammar, evaluated by DuckDB.

        Only the router asks, and only for a table delta-rs misreads (see
        `Router._kernel_filters_sql`); elsewhere such a predicate goes to
        delta-rs, which evaluates it itself.
        """
        return importlib.util.find_spec("duckdb") is not None

    def files(
        self,
        table: ResolvedTable,
        *,
        version: int | None = None,
        predicate: str | None = None,
    ) -> Any:
        """Live data files, with stats and deletion-vector descriptors.

        `num_records` counts rows before deletion vectors are applied.
        """
        snapshot = self.snapshot(table, version=version)
        skipping = (
            sqlpred.to_kernel_json(sqlpred.parse(predicate), _arrow_schema(snapshot))
            if predicate
            else None
        )
        return snapshot.files(predicate=skipping)

    def cdf(
        self,
        table: ResolvedTable,
        *,
        starting_version: int | None = None,
        ending_version: int | None = None,
        starting_timestamp: Any = None,
        ending_timestamp: Any = None,
        columns: list[str] | None = None,
        predicate: str | None = None,
        **unsupported: Any,
    ) -> Any:
        """The change data feed through the kernel's TableChanges.

        Serves path tables delta-rs cannot open. It cannot serve a
        catalog-managed table: TableChanges lists the log itself and takes no
        catalog commit tail, so it would miss ratified-but-unpublished commits.
        """
        from deltaswamp import _native

        # `v not in (None, False)` also dropped 0, since 0 == False.
        given = {k: v for k, v in unsupported.items() if v is not None and v is not False}
        if given:
            raise EngineLimitError(
                f"read the change data feed with {', '.join(sorted(given))}",
                "the kernel change feed does not implement these options",
            )
        if table.is_catalog_managed:
            raise UnreachableTableError(
                "read the change data feed of a catalog-managed table",
                "the kernel's TableChanges lists the log directly and takes no catalog "
                "commit tail, so it would silently miss unpublished commits",
                "ds.connect(..., allow_sql_fallback=True) reads it with table_changes()",
            )
        # A path-resolved table carries no properties until it is enriched, so
        # an absent key means "ask the log", not "disabled".
        enabled = table.properties.get("delta.enableChangeDataFeed")
        snapshot = None
        if enabled is None or columns is not None or predicate:
            snapshot = self.snapshot(table)
            enabled = snapshot.table_properties().get("delta.enableChangeDataFeed", enabled)
        if str(enabled or "false").lower() != "true":
            raise UnreachableTableError(
                "read the change data feed",
                "delta.enableChangeDataFeed is not enabled on this table, and enabling "
                "it is not retroactive -- only changes after enablement are recorded",
            )
        if starting_version is not None and starting_timestamp is not None:
            raise UnreachableTableError(
                "read the change data feed", "pass a starting version or timestamp, not both"
            )
        if ending_version is not None and ending_timestamp is not None:
            raise UnreachableTableError(
                "read the change data feed", "pass an ending version or timestamp, not both"
            )
        meta = ["_change_type", "_commit_version", "_commit_timestamp"]
        node, read_columns, keep = None, columns, None
        if snapshot is not None and (columns is not None or predicate):
            node, read_columns, wanted = _read_plan(snapshot, columns, predicate or None)
            if read_columns is not None:
                # The feed always carries its metadata columns, so they are
                # never projected, and each is kept exactly once.
                wanted = read_columns if wanted is None else wanted
                keep = [*wanted, *(m for m in meta if m not in wanted)]
                read_columns = [c for c in read_columns if c not in meta]
                read_columns = _with_data_column(snapshot, read_columns)
        assert table.location is not None  # supports() refused otherwise
        kernel_predicate = (
            sqlpred.to_kernel_json(node, _arrow_schema(snapshot)) if node is not None else None
        )

        location = table.location

        def changes(start: int | None) -> Any:
            return _native.table_changes(
                location,
                options=self._options(table, write=False) or None,
                start_version=start,
                end_version=ending_version,
                columns=read_columns,
                predicate=kernel_predicate,
                start_timestamp_ms=(
                    timestamp_ms(starting_timestamp) if starting_timestamp is not None else None
                ),
                end_timestamp_ms=(
                    timestamp_ms(ending_timestamp) if ending_timestamp is not None else None
                ),
            )

        start = starting_version
        while True:
            try:
                stream = changes(start)
                break
            except ValueError as exc:
                message = str(exc)
                off = re.search(r"feed is unsupported for the table at version (\d+)", message)
                if off is not None and starting_version is None and starting_timestamp is None:
                    # No start was given and the feed was switched on after
                    # that version: start where it is on, as delta-rs does,
                    # rather than failing on the default start of 0.
                    after = int(off.group(1)) + 1
                    # Only while it is off at the start (enabled later); a gap
                    # after the feed was on is refused, not silently skipped.
                    if after == (start or 0) + 1 and (
                        ending_version is None or after <= ending_version
                    ):
                        start = after
                        continue
                if off is not None:
                    error = UnreachableTableError(
                        "read the change data feed",
                        f"the change data feed was not enabled at version {off.group(1)}, "
                        "which the requested range includes",
                        "start the range after the feed was enabled",
                    )
                    # Named, as a schema change names its version, so a
                    # follower can read up to it.
                    error.version = int(off.group(1))  # type: ignore[attr-defined]
                    raise error from exc
                if "Start and end version schemas are different" in message:
                    # Turning the feed off is a metadata change too, which the
                    # kernel reports as a schema change. Name the real cause.
                    gap = self._feed_gap(table, start or 0, ending_version)
                    if gap is not None:
                        error = UnreachableTableError(
                            "read the change data feed",
                            f"the change data feed was not enabled at version {gap}, "
                            "which the requested range includes",
                            "start the range after the feed was re-enabled",
                        )
                        error.version = gap  # type: ignore[attr-defined]
                        raise error from exc
                    raise EngineLimitError(
                        "read the change data feed",
                        "the table's schema changed within the requested range, and the "
                        "kernel reads a change feed across one schema only",
                        "read the ranges before and after the schema change separately",
                    ) from exc
                gone = re.search(
                    r"Expected the first commit to have version (\d+), got Some\((\d+)\)", message
                )
                if gone is not None and int(gone.group(2)) > int(gone.group(1)):
                    earliest = int(gone.group(2))
                    if starting_version is None and starting_timestamp is None and start is None:
                        # No start given: the readable feed begins at the
                        # oldest commit log retention has kept.
                        start = earliest
                        continue
                    raise UnreachableTableError(
                        "read the change data feed",
                        f"version {gone.group(1)} is no longer in the log (log retention "
                        f"removed it); the oldest commit still there is {earliest}",
                        f"pass starting_version={earliest} or later",
                    ) from exc
                raise
        if node is not None:
            stream = sqlpred.filter_stream(stream, node)
        if keep is not None:
            import pyarrow as pa

            have = pa.RecordBatchReader.from_stream(stream)
            if list(have.schema.names) == keep:
                return have
            return _project(have, keep)
        return stream

    def _feed_gap(self, table: ResolvedTable, start: int, end: int | None) -> int | None:
        """The first version in `start..end` with the change feed off, or None.

        Only asked on the error path, and bounded, since it opens one snapshot
        per version.
        """
        try:
            last = int(self.snapshot(table).version) if end is None else end
            for version in range(start, min(last, start + 1000) + 1):
                props = self.snapshot(table, version=version).table_properties() or {}
                if str(props.get("delta.enableChangeDataFeed", "false")).lower() != "true":
                    return version
        except Exception:
            return None
        return None

    def checkpoint(self, table: ResolvedTable) -> bool:
        """Write a checkpoint at the latest version. False if one already existed.

        A catalog-managed table is published first: the kernel checkpoints only
        published versions, and publishing is owed to the catalog anyway.
        """
        if table.is_catalog_managed:
            self.publish(table)
        written: bool = self.snapshot(table, write=True).checkpoint()
        return written

    def detail(self, table: ResolvedTable, *, version: int | None = None) -> dict[str, Any]:
        """Table metadata. Cheap: kernel serves stats from a CRC when present."""
        snap = self.snapshot(table, version=version)
        min_reader, min_writer, reader_features, writer_features = snap.protocol()
        return {
            "version": snap.version,
            "location": snap.table_root,
            "is_catalog_managed": snap.is_catalog_managed,
            "min_reader_version": min_reader,
            "min_writer_version": min_writer,
            "reader_features": reader_features,
            "writer_features": writer_features,
            "properties": snap.table_properties(),
            "partition_columns": list(snap.partition_columns),
            "metadata_id": snap.metadata_id,
            **_feature_usage(snap),
        }

    # ------------------------------------------------------------------ write

    def _uc_commit_config(self, table: ResolvedTable, *, staging: bool = False) -> Any:
        """Build the UC committer config, or None for a path-based table.

        A catalog-managed table *must* commit through the catalog: staging a
        file and hoping object-store atomicity carries it is not a fallback,
        it produces a commit nobody ratifies.
        """
        if not table.is_catalog_managed:
            return None

        from deltaswamp._native import UcCommitConfig

        provider = table.credential_provider
        auth = getattr(provider, "workspace_auth", None)
        if auth is None and staging:
            # A worker holding a plan's shipped storage credential: the write
            # context needs a committer to exist, never to commit, so the
            # workspace URL without a token is enough -- and all it gets.
            auth = getattr(provider, "staging_auth", None)
        if provider is None or auth is None:
            raise UnreachableTableError(
                "commit to a catalog-managed table",
                "no Unity Catalog credentials are available; the commit must be ratified "
                "by the catalog and cannot be written directly to object storage",
            )
        if table.table_id is None:
            raise UnreachableTableError(
                "commit to a catalog-managed table",
                "the catalog did not report a table_id, which the UC commit API requires",
            )

        ref = table.ref
        if not (ref.catalog and ref.schema and ref.table):
            raise UnreachableTableError(
                "commit to a catalog-managed table",
                f"the UC commit API addresses tables by three-level name, but {ref} "
                "does not have one",
            )

        workspace_url, token = auth()
        return UcCommitConfig(
            workspace_url, token, table.table_id, ref.catalog, ref.schema, ref.table
        )

    def append(
        self,
        table: ResolvedTable,
        data: Any,
        *,
        engine_info: str | None = None,
        operation: str = "WRITE",
        overwrite: bool = False,
        txn: tuple[str, int] | None = None,
        commit_metadata: dict[str, str] | None = None,
        **unsupported: Any,
    ) -> int:
        """Append data and commit. Returns the committed version."""
        # Refuse options this path would otherwise ignore.
        retries = unsupported.pop("max_commit_retries", None)
        # The binding extracts only a tuple; txn=["job", 1] passed every check
        # and then failed with TypeError at the commit, after the files.
        txn = (txn[0], int(txn[1])) if txn is not None else None
        given = {k: v for k, v in unsupported.items() if v is not None}
        if given:
            raise UnreachableTableError(
                f"append with {', '.join(sorted(given))}",
                "the kernel write path does not implement these options",
                "the router sends these to delta-rs when the table allows it; a "
                "catalog-managed table needs the SQL fallback",
            )
        from deltaswamp import _native

        # A blind append commutes with any concurrent commit, so losing the race
        # means re-staging it on the new snapshot -- which delta-rs does too.
        # Only re-readable data can be re-staged, an overwrite must surface the
        # conflict, and a catalog-managed table's tail cannot be refreshed here.
        if retries is not None and not hasattr(data, "to_reader") and not overwrite:
            import pyarrow as pa

            data = pa.table(_as_record_batch_reader(data))
        replayable = hasattr(data, "to_reader") and not overwrite and not table.is_catalog_managed
        attempts = 1 + max(0, self.append_commit_retries if retries is None else int(retries))
        attempts = attempts if replayable else 1
        with translating(EngineKind.KERNEL, "commit"):
            for attempt in range(attempts):
                snapshot = self.snapshot(table, write=True)
                if txn is not None and hasattr(snapshot, "app_id_version"):
                    # The caller's txn check read an older snapshot: a writer that
                    # committed this batch in between left it in the one we are
                    # about to commit on, and committing anyway appended it twice.
                    last = snapshot.app_id_version(txn[0])
                    if last is not None and int(last) >= int(txn[1]):
                        raise CommitConflictError(
                            int(snapshot.version),
                            f"transaction {txn[0]!r} version {txn[1]} was committed by a "
                            f"concurrent writer (the table records version {last}); "
                            "this batch is already in the table",
                        )
                reader = recorded(_as_record_batch_reader(data))
                try:
                    version: int = snapshot.append(
                        reader,
                        uc=self._uc_commit_config(table),
                        engine_info=engine_info or _engine_info(),
                        operation=operation,
                        overwrite=overwrite,
                        txn=txn,
                        commit_metadata={k: str(v) for k, v in (commit_metadata or {}).items()}
                        or None,
                        **_write_info(snapshot, overwrite=overwrite),
                    )
                except _native.CommitConflictError:
                    if attempt + 1 >= attempts or self._txn_won_race(snapshot, table, txn):
                        # A concurrent commit that recorded this txn (or a later
                        # one) already wrote this batch: re-staging would append it twice.
                        raise
                    # Re-staged on the next attempt's snapshot, so the batch is
                    # conformed to whatever schema a concurrent commit left.
                    commit_backoff(attempt)
                    continue
                break
        self._maybe_checkpoint(table, version, snapshot)
        return version

    def _txn_won_race(self, snapshot: Any, table: ResolvedTable, txn: Any) -> bool:
        if txn is None:
            return False
        try:
            fresh = self.snapshot(table)
            last = fresh.app_id_version(txn[0]) if hasattr(fresh, "app_id_version") else None
        except Exception:
            return True  # cannot tell; surfacing the conflict is the safe answer
        return last is not None and int(last) >= int(txn[1])

    @property
    def txn_version(self) -> Any:
        """`txn_version(table, app_id)`: the last version committed under `app_id`.

        None (not a method) on a build without the binding, so callers that
        test for the method fall back instead of calling one that cannot work.
        Without it the append dedup had nothing to ask on tables only the
        kernel can open, and a replayed txn appended twice.
        """
        return self._txn_version if _native_has("app_id_version") else None

    def _txn_version(self, table: ResolvedTable, app_id: str) -> int | None:
        # The snapshot includes a catalog-managed table's ratified tail, so an
        # unpublished commit's txn counts too.
        version = self.snapshot(table).app_id_version(app_id)
        return None if version is None else int(version)

    #: Re-stagings of a blind append that lost a commit race, by default.
    #: delta-rs's default: five, with no pause between them, left eight
    #: threads appending at once with conflicts after every retry was spent.
    append_commit_retries = 15

    def create(
        self,
        table: ResolvedTable,
        schema: Any,
        *,
        partition_by: list[str] | None = None,
        cluster_by: list[str] | None = None,
        mode: str = "error",
        properties: dict[str, str] | None = None,
        engine_info: str | None = None,
        description: str | None = None,
        **unsupported: Any,
    ) -> int:
        """Create a table and commit version 0.

        The kernel accepts nearly the whole property surface, including the
        `delta.feature.*` signals and custom keys delta-rs refuses. Clustering
        is not a property here: it goes through the data layout.
        """
        from deltaswamp._native import create_table

        _enter_native("create the table with the kernel")
        if table.location is None:
            raise UnreachableTableError("create", "no storage location was given for the new table")
        if mode not in ("error", "create"):
            raise UnreachableTableError(
                f"create with mode={mode!r}",
                "the kernel create path writes version 0 of a new table only",
                "use mode='error', or write through delta-rs for overwrite semantics",
            )
        # A bare **_ignored used to drop these (a description included) silently.
        _refuse_options("create", unsupported)
        if description is not None and table.is_catalog_managed:
            raise UnreachableTableError(
                "create a catalog-managed table with a description",
                "the kernel create takes no description, and a follow-up metadata commit "
                "on a catalog-managed table would bypass the catalog",
                "set the comment through the catalog after creating the table",
            )

        # delta-kernel refuses these in CREATE TABLE although its metadata
        # commit stores them; they land as version 1, before this returns.
        deferred = {k: v for k, v in (properties or {}).items() if k in KERNEL_CREATE_DEFERRED}
        if deferred and table.is_catalog_managed:
            raise UnreachableTableError(
                "create a catalog-managed table with " + ", ".join(sorted(deferred)),
                "the kernel create refuses these properties, and a follow-up metadata commit "
                "on a catalog-managed table would bypass the catalog",
                "create the table, then set them through the catalog",
            )
        properties = {k: v for k, v in (properties or {}).items() if k not in deferred}
        version: int = create_table(
            table.location,
            schema,
            # Vended credentials too: a catalog create writes version 0 to a
            # location only they can reach.
            options=self._options(table, write=True) or None,
            properties=properties or None,
            partition_by=partition_by or None,
            cluster_by=cluster_by or None,
            uc=self._uc_commit_config(table),
            engine_info=engine_info or _engine_info(),
        )
        if deferred:
            version = self.set_properties(table, deferred)
        if description is not None:
            version = self.set_comment(table, description)
        return version

    #: Fallback when the table sets no interval. Matches Delta's own default.
    default_checkpoint_interval = 10

    def checkpoint_interval(
        self, table: ResolvedTable, properties: dict[str, str] | None = None
    ) -> int:
        # `properties` are the snapshot's own: a path-resolved table carries
        # none, and its interval used to be read as the default.
        raw = {**table.properties, **(properties or {})}.get("delta.checkpointInterval")
        try:
            interval = int(raw) if raw else self.default_checkpoint_interval
        except ValueError:
            interval = self.default_checkpoint_interval
        return max(interval, 1)

    def _maybe_checkpoint(self, table: ResolvedTable, version: int, snapshot: Any = None) -> None:
        """Checkpoint after a commit at the table's checkpoint interval.

        Kernel commits never checkpoint on their own. The checkpoint is written
        at exactly `version`, the commit just made.

        A catalog-managed table is skipped: the commit just ratified is not in
        this ResolvedTable's log tail, so publishing from here stops one version
        short, and the checkpoint used to land on the previous version. A
        checkpoint cannot cover an unpublished commit, so `checkpoint()` on a
        freshly resolved table (which publishes first) is the way to take one.
        """
        if version == 0 or table.is_catalog_managed:
            return
        try:
            properties = snapshot.table_properties() if snapshot is not None else None
        except Exception:
            properties = None
        if version % self.checkpoint_interval(table, properties) != 0:
            return
        if checkpoint_drops_stats({**table.properties, **(properties or {})}):
            return  # it would erase every file's statistics; see supports()
        try:
            self.snapshot(table, version=version, write=True).checkpoint()
        except Exception:
            # A checkpoint is an optimization; never fail a commit that worked.
            return

    def overwrite(
        self,
        table: ResolvedTable,
        data: Any,
        *,
        predicate: str | None = None,
        partition_overwrite: str = "static",
        **kwargs: Any,
    ) -> int:
        """Replace the table's contents in a single commit.

        Every file the snapshot can see is removed in the same transaction that
        adds the new ones, so readers never observe an empty table. This is the
        only path that can overwrite a catalog-managed table, since delta-rs
        cannot open one.
        """
        if partition_overwrite != "static":
            raise UnreachableTableError(
                "overwrite partitions dynamically with the kernel engine",
                "dynamic partition overwrite is emulated on delta-rs; pass an explicit "
                "predicate here instead",
            )
        if predicate is not None:
            import pyarrow as pa

            # These used to be dropped here: a txn lost its idempotency record
            # and commit metadata vanished, while the plain overwrite kept both.
            passthrough = {
                k: kwargs.pop(k, None) for k in ("txn", "commit_metadata", "engine_info")
            }
            _refuse_options("overwrite with a predicate", kwargs)
            incoming = pa.table(_as_record_batch_reader(data))
            if self._dv_path(table):
                result = self._dv_dml(
                    table,
                    predicate,
                    operation="WRITE",
                    replacement=incoming,
                    **passthrough,
                )
                return int(result["version"])

            def replace(current: Any, keep: Any) -> Any:
                import pyarrow.compute as pc

                from .. import predicate as sqlpred

                kept = current.filter(keep)
                new = _conform(incoming, kept.schema)
                # replaceWhere: every new row must satisfy the predicate, as
                # Databricks (replaceWhere.constraintCheck) and delta-rs enforce.
                # Accepting others writes rows outside the range being replaced.
                node = _canonical_node(sqlpred.parse(predicate), new.schema)
                ok = pc.fill_null(_evaluate(new, sqlpred.to_arrow(node, new.schema)), False)
                bad = new.num_rows - int(pc.sum(ok).as_py() or 0)
                if bad:
                    # The request's data, not the table: the type delta-rs raises
                    # for it too, so one except clause catches both engines.
                    raise InvalidArgumentError(
                        f"overwrite where {predicate}: {bad} row(s) of the new data do not "
                        "satisfy the predicate, so writing them would add rows outside the "
                        "range being replaced; filter the data to the predicate first"
                    )
                return pa.concat_tables([kept, new])

            result = self._rewrite(table, predicate, replace, operation="WRITE", **passthrough)
            return int(result["version"])
        return self.append(table, data, operation="WRITE", overwrite=True, **kwargs)

    # ------------------------------------------------ copy-on-write rewrites

    #: The largest table (by live data-file bytes) a copy-on-write rewrite will
    #: take on. The rewrite holds the table in memory, so past this size the
    #: warehouse, or delta-rs on a table it can open, is the right tool.
    rewrite_max_bytes = 1 << 30

    def _merge_refusal(self, table: ResolvedTable) -> Capability | None:
        """Why the kernel cannot MERGE into `table`, or None if it can."""
        if not self._dv_path(table):
            return Capability(
                Operation.MERGE,
                ok=False,
                reason="the kernel serves MERGE only by writing deletion vectors, and this "
                "table does not enable them (delta.enableDeletionVectors)",
                remedy=SQL_FALLBACK_REMEDY,
            )
        if (
            importlib.util.find_spec("duckdb") is None
            or importlib.util.find_spec("pyarrow") is None
        ):
            return Capability(
                Operation.MERGE,
                ok=False,
                reason="the kernel MERGE evaluates its clauses with DuckDB, which is not installed",
                remedy="pip install 'deltaswamp[duckdb]'",
            )
        if table.is_shallow_clone:
            return Capability(
                Operation.MERGE,
                ok=False,
                reason="a shallow clone's data files belong to the source table (referenced "
                "by absolute path), which cannot be credential-scoped reliably",
                remedy=SQL_FALLBACK_REMEDY,
            )
        if "rowTracking" in table.effective_writer_features and not _keeps_row_ids(table):
            return Capability(
                Operation.MERGE,
                ok=False,
                reason="the table tracks row ids but names no materialized row-id column, "
                "so updated rows could not keep theirs",
                remedy=SQL_FALLBACK_REMEDY,
            )
        return None

    def _dv_path(self, table: ResolvedTable) -> bool:
        """Whether DELETE/UPDATE/replaceWhere go through deletion vectors here."""
        return deletion_vectors_writable(table) and _native_has("deletion_vector_dml")

    def _rewrite_refusal(
        self, operation: Operation, table: ResolvedTable, *, by_dv: bool = False
    ) -> Capability | None:
        """Whether a rewrite may serve DELETE/UPDATE/replaceWhere.

        Row tracking is handled before this, since it rules out every commit
        that stages a remove, not only the rewrites. Through deletion vectors
        only the files the predicate cannot skip are read, and only matching
        rows are held, so the whole-table size bound does not apply.
        """
        if table.is_shallow_clone:
            # A rewrite reads every live file, and a shallow clone's are the
            # source table's, by absolute path: the very read SCAN refuses.
            # Claiming it sent the rewrite into a storage 403 on borrowed files.
            return Capability(
                operation,
                ok=False,
                reason=f"the kernel serves {operation.value} by rewriting the table, and a "
                "shallow clone's data files belong to the source table (referenced by "
                "absolute path), which cannot be credential-scoped reliably",
                remedy="ds.connect(..., allow_sql_fallback=True) runs it on Databricks",
            )
        if table.location is None or by_dv:
            return None
        try:
            size = self._live_bytes(table)
        except Exception as exc:  # cannot size it: do not claim it
            return Capability(operation, ok=False, reason=f"could not size the table: {exc}")
        if size > self.rewrite_max_bytes:
            return Capability(
                operation,
                ok=False,
                reason=f"the kernel serves {operation.value} by rewriting the whole table, and "
                f"this one holds {size:,} bytes, above the {self.rewrite_max_bytes:,}-byte limit",
                remedy="ds.connect(..., allow_sql_fallback=True), or raise "
                "KernelEngine.rewrite_max_bytes",
            )
        return None

    def _data_write_refusal(self, operation: Operation, table: ResolvedTable) -> Capability | None:
        """Tables whose data the kernel's transaction refuses to write at commit.

        Each of these used to be claimed and then fail after the data was
        staged, with no way for the router to divert the call.
        """
        writer = table.min_writer_version or 0
        if 3 <= writer <= 6:
            # A legacy protocol implies checkConstraints (and, from 4, CDF and
            # generated columns), which the kernel writer does not support.
            return Capability(
                operation,
                ok=False,
                reason=f"the table uses the legacy writer protocol version {writer}, which "
                "implies checkConstraints; the kernel writer does not support it",
            )
        append_only = str(table.properties.get("delta.appendOnly", "false")).lower() == "true"
        if append_only and operation is not Operation.APPEND:
            return Capability(
                operation,
                ok=False,
                reason="the table is append-only (delta.appendOnly=true), so no commit may "
                "remove or rewrite its data",
            )
        cdf = str(table.properties.get("delta.enableChangeDataFeed", "false")).lower() == "true"
        dv_delete = operation is Operation.DELETE and self._dv_path(table)
        if cdf and operation is not Operation.APPEND and not dv_delete:
            # A DELETE through deletion vectors is exempt: its commit adds no
            # data, and change-feed readers derive the deleted rows from the
            # difference between each file's old and new vector.
            return Capability(
                operation,
                ok=False,
                reason="the table has the change data feed enabled, and the kernel cannot "
                "write the CDC files a commit that removes data must carry",
            )
        if table.has_generated_columns:
            # Normally refused by the generatedColumns feature itself; this
            # covers the tables an earlier create left with the expressions but
            # not the feature, where the kernel wrote whatever it was given.
            return Capability(
                operation,
                ok=False,
                reason="the table schema declares generated columns, whose values the "
                "kernel writer neither computes nor checks",
                remedy="write through delta-rs, which evaluates them",
            )
        if (writer == 2 or "invariants" in table.writer_features) and self._has_invariants(table):
            return Capability(
                operation,
                ok=False,
                reason="the table schema declares column invariants, which the kernel "
                "writer does not enforce and therefore refuses",
            )
        return None

    def _has_invariants(self, table: ResolvedTable) -> bool:
        import json

        try:
            schema = json.loads(json.loads(self.snapshot(table).metadata_json())["schemaString"])
        except Exception:
            return False  # cannot tell; the commit itself will refuse if so

        def walk(node: Any) -> bool:
            if isinstance(node, dict):
                metadata = node.get("metadata")
                if isinstance(metadata, dict) and "delta.invariants" in metadata:
                    return True
                return any(walk(v) for v in node.values())
            if isinstance(node, list):
                return any(walk(v) for v in node)
            return False

        return walk(schema)

    def _live_bytes(self, table: ResolvedTable) -> int:
        import pyarrow as pa

        files = pa.table(self.snapshot(table).files())
        return int(sum(files.column("size").to_pylist())) if files.num_rows else 0

    def _rewrite(
        self,
        table: ResolvedTable,
        predicate: str | None,
        transform: Any,
        *,
        operation: str,
        txn: tuple[str, int] | None = None,
        commit_metadata: dict[str, Any] | None = None,
        engine_info: str | None = None,
    ) -> dict[str, Any]:
        """Copy-on-write through one kernel transaction.

        Reads the snapshot, computes the new contents, and commits them with
        every old file removed -- against that same snapshot, so a concurrent
        writer makes the commit conflict (a 409 from the catalog, or a lost
        put-if-absent) rather than being overwritten. Deletion vectors are
        applied on the read and none are written.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        refusal = self._rewrite_refusal(Operation.DELETE, table)
        if refusal is not None:
            raise UnreachableTableError(
                f"{operation.lower()} via a kernel rewrite", refusal.reason, refusal.remedy or None
            )

        snapshot = self.snapshot(table, write=True)
        attempt = 0
        while True:
            current = pa.table(snapshot.scan())
            node = None
            if predicate is None:
                matched = pa.array([True] * current.num_rows, pa.bool_())
            else:
                node = _canonical_node(sqlpred.parse(predicate), current.schema)
                expr = sqlpred.to_arrow(node, current.schema)
                matched = pc.fill_null(_evaluate(current, expr), False)
            keep = pc.invert(matched)
            replacement = transform(current, keep)
            touched = int(pc.sum(matched).as_py() or 0)
            if touched == 0 and replacement.num_rows == current.num_rows:
                return {"version": int(snapshot.version), "num_affected_rows": 0}
            try:
                with translating(EngineKind.KERNEL, "commit"):
                    version = snapshot.append(
                        replacement.to_reader(),
                        uc=self._uc_commit_config(table),
                        engine_info=engine_info or _engine_info(),
                        operation=operation,
                        overwrite=True,
                        txn=txn,
                        commit_metadata={k: str(v) for k, v in (commit_metadata or {}).items()}
                        or None,
                        **_dml_info(operation, predicate, snapshot),
                    )
                break
            except CommitConflictError:
                # The overwrite removes every file of the snapshot it read, so
                # it cannot be re-committed as staged; but when the winners
                # were ones Delta lets it survive (blind appends, under
                # WriteSerializable), running it again on the new snapshot is
                # the same outcome, and without it a DELETE under steady
                # appends never committed.
                attempt += 1
                if attempt > self.dml_commit_retries:
                    raise
                paths = set(pa.table(snapshot.files()).column("path").to_pylist())
                reads = sqlpred.to_kernel_json(node, _arrow_schema(snapshot)) if node else None
                rebased = self._rebase_dv_commit(table, snapshot, paths, txn, reads)
                if rebased is None:
                    raise
                snapshot = rebased
                commit_backoff(attempt - 1)
        self._maybe_checkpoint(table, version, snapshot)
        return {"version": int(version), "num_affected_rows": touched}

    def _dv_dml(
        self,
        table: ResolvedTable,
        predicate: str | None,
        *,
        operation: str,
        transform: Any = None,
        replacement: Any = None,
        txn: tuple[str, int] | None = None,
        commit_metadata: dict[str, Any] | None = None,
        engine_info: str | None = None,
    ) -> dict[str, Any]:
        """DELETE, UPDATE or replaceWhere as deletion vectors, as Databricks writes them.

        Reads only the files the predicate cannot skip, with each row tagged by
        its file and physical position, and marks the matching rows deleted.
        `transform` turns the matched rows into their updated form, which is
        appended in the same commit; `replacement` is appended as given (after
        checking every row satisfies the predicate, as replaceWhere requires).
        The commit is staged against the snapshot read, so a concurrent writer
        makes it conflict rather than be lost.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        snapshot = self.snapshot(table, write=True)
        schema = _arrow_schema(snapshot)
        if predicate is None and transform is None and replacement is None:
            emptied = self._delete_every_file(table, snapshot, txn, commit_metadata, engine_info)
            if emptied is not None:
                return emptied
        node = _canonical_node(sqlpred.parse(predicate), schema) if predicate is not None else None
        if transform is None:
            # Only what the predicate reads: a DELETE never needs the rest.
            wanted = sorted({path[0] for path in sqlpred.columns_of(node)}) if node else []
            columns: list[str] | None = _with_data_column(snapshot, wanted)
        else:
            columns = None
        skipping = sqlpred.to_kernel_json(node, schema) if node is not None else None
        # An UPDATE on a row-tracked table carries each row's id into the new
        # file, so the rewritten row keeps the id it had.
        row_ids = transform is not None and _row_tracking_enabled(table)
        extra = {"row_ids": True} if row_ids else {}
        stream = snapshot.scan(columns=columns, predicate=skipping, row_positions=True, **extra)
        if node is not None:
            stream = sqlpred.filter_stream(stream, node)
        reader = pa.RecordBatchReader.from_stream(stream)

        positions = [_FILE_COLUMN, _ROW_INDEX_COLUMN]
        matched_batches = []
        deletion_batches = []
        for batch in reader:
            if batch.num_rows == 0:
                continue
            deletion_batches.append(
                pa.record_batch(
                    [batch.column(_FILE_COLUMN), batch.column(_ROW_INDEX_COLUMN)],
                    names=["path", "row_index"],
                )
            )
            if transform is not None:
                matched_batches.append(batch.drop_columns(positions))
        touched = sum(b.num_rows for b in deletion_batches)

        data = None
        if transform is not None and matched_batches:
            data = transform(pa.Table.from_batches(matched_batches))
        elif replacement is not None:
            data_schema = pa.schema([f for f in schema])
            data = _conform(replacement, data_schema)
            if node is not None:
                ok = pc.fill_null(_evaluate(data, sqlpred.to_arrow(node, data.schema)), False)
                bad = data.num_rows - int(pc.sum(ok).as_py() or 0)
                if bad:
                    # The request's data, not the table: the type delta-rs raises
                    # for it too, so one except clause catches both engines.
                    raise InvalidArgumentError(
                        f"overwrite where {predicate}: {bad} row(s) of the new data do not "
                        "satisfy the predicate, so writing them would add rows outside the "
                        "range being replaced; filter the data to the predicate first"
                    )
            if data.num_rows == 0:
                data = None
        if touched == 0 and data is None:
            return {"version": int(snapshot.version), "num_affected_rows": 0}

        deletions = pa.Table.from_batches(
            deletion_batches,
            schema=pa.schema([("path", pa.string()), ("row_index", pa.int64())]),
        )
        version = self._commit_dv_changes(
            table,
            snapshot,
            deletions,
            data,
            operation=operation,
            txn=txn,
            commit_metadata=commit_metadata,
            engine_info=engine_info,
            read_predicate=skipping,
            predicate=predicate,
        )
        return {"version": version, "num_affected_rows": touched}

    def _delete_every_file(
        self,
        table: ResolvedTable,
        snapshot: Any,
        txn: tuple[str, int] | None,
        commit_metadata: dict[str, Any] | None,
        engine_info: str | None,
    ) -> dict[str, Any] | None:
        """DELETE with no predicate: every file goes, without reading a row.

        Returns None when a file's row count is unknown (no statistics), so
        the caller counts rows the slow way.
        """
        import json

        import pyarrow as pa

        files = pa.table(snapshot.files())
        live = 0
        for records, dv in zip(
            files.column("num_records").to_pylist(),
            files.column("deletion_vector").to_pylist(),
            strict=True,
        ):
            if records is None:
                return None
            live += int(records) - (int(json.loads(dv)["cardinality"]) if dv else 0)
        if live == 0:
            return {"version": int(snapshot.version), "num_affected_rows": 0}
        empty = pa.table({"path": pa.array([], pa.string()), "row_index": pa.array([], pa.int64())})
        version = self._commit_dv_changes(
            table,
            snapshot,
            empty,
            None,
            operation="DELETE",
            whole_files=files.column("path").to_pylist(),
            txn=txn,
            commit_metadata=commit_metadata,
            engine_info=engine_info,
        )
        return {"version": version, "num_affected_rows": live}

    def _commit_dv_changes(
        self,
        table: ResolvedTable,
        snapshot: Any,
        deletions: Any,
        data: Any,
        *,
        operation: str,
        whole_files: list[str] | None = None,
        txn: tuple[str, int] | None = None,
        commit_metadata: dict[str, Any] | None = None,
        engine_info: str | None = None,
        read_predicate: str | None = None,
        predicate: str | None = None,
    ) -> int:
        """Commit `deletions` (path, row_index) as vectors plus `data`, in one transaction.

        A touched file whose add has no `numRecords` statistic (every add in a
        Databricks checkpoint: its tables write stats only as `stats_parsed`)
        still takes a vector; the native commit reads the count from the
        file's Parquet footer. `read_predicate` is the kernel skipping
        predicate the rows were read with (None: every file was read), which
        decides which concurrently added files this commit must conflict with.
        """
        touched = set(deletions.column("path").to_pylist()) | set(whole_files or ())
        attempt = 0
        try:
            while True:
                try:
                    with translating(EngineKind.KERNEL, "commit"):
                        version, _deleted, _dvs, _removed = snapshot.commit_dml(
                            deletions.to_reader(),
                            data=data.to_reader() if data is not None else None,
                            whole_files=whole_files or None,
                            uc=self._uc_commit_config(table),
                            engine_info=engine_info or _engine_info(),
                            operation=operation,
                            txn=txn,
                            commit_metadata={k: str(v) for k, v in (commit_metadata or {}).items()}
                            or None,
                            **_dml_info(operation, predicate, snapshot),
                        )
                    break
                except CommitConflictError:
                    attempt += 1
                    if attempt > self.dml_commit_retries:
                        raise
                    rebased = self._rebase_dv_commit(table, snapshot, touched, txn, read_predicate)
                    if rebased is None:
                        raise
                    snapshot = rebased
                    commit_backoff(attempt - 1)
        except ValueError as exc:
            # Refused before anything is written; the request's mistake, not
            # the engine's, so it is reported as one, saying which.
            if engine_cause(exc) is exc or "non-nullable" not in str(exc):
                raise
            raise InvalidArgumentError(
                f"{operation} would write NULL into a NOT NULL column: {exc}"
            ) from exc
        if int(version) != int(snapshot.version):
            self._maybe_checkpoint(table, version, snapshot)
        return int(version)

    #: Re-commits of a DELETE/UPDATE/MERGE that lost to writers which left
    #: every file it touched alone and added nothing it should have read.
    dml_commit_retries = 15

    def _rebase_dv_commit(
        self,
        table: ResolvedTable,
        read: Any,
        touched: set[str],
        txn: tuple[str, int] | None,
        read_predicate: str | None = None,
        *,
        compaction: bool = False,
    ) -> Any:
        """The latest snapshot, when the commits that won left this one valid; else None.

        Delta's conflict check under WriteSerializable, its default: a DML
        commit survives concurrent winners that changed neither the metadata
        nor the protocol, neither removed nor re-vectored any file it
        touched, and -- unless they were blind appends -- added no file its
        read could have matched (Spark's checkForAddedFilesThatShouldHaveBeen-
        ReadByCurrentTxn). Rows a concurrent blind append added are not
        deleted or updated -- the transaction never read them, and
        WriteSerializable orders it before the append. A MERGE, UPDATE or
        DELETE that won is not blind: its new rows (an inserted key, an
        updated value) are what this transaction's read would have seen, and
        rebasing over them let four concurrent upserts of one key each insert
        it. Under `delta.isolationLevel=Serializable` any concurrent append
        conflicts, as Spark decides it. Without the rebase the kernel path
        failed on any concurrent append, where delta-rs retried. A
        `compaction` changes no rows, so what the winners added is no concern
        of it (Spark checks it under snapshot isolation); only the files it
        removes must be as it read them.
        """
        import pyarrow as pa

        if table.is_catalog_managed:
            # As for appends: a catalog commit is not re-staged here.
            return None
        if txn is not None and self._txn_won_race(read, table, txn):
            return None
        metadata_changed = False
        try:
            fresh = self.snapshot(table, write=True)
            if int(fresh.version) <= int(read.version):
                return None
            level = (fresh.table_properties() or {}).get("delta.isolationLevel", "")
            if level.lower() == "serializable" and not compaction:
                return None
            metadata_changed = fresh.metadata_json() != read.metadata_json()
            if metadata_changed and compaction:
                return None  # planned again from the new layout
            if metadata_changed:
                raise _MetadataMoved(int(fresh.version))
            if fresh.protocol_json() != read.protocol_json():
                return None

            def vectors(snapshot: Any) -> dict[str, Any]:
                files = pa.table(snapshot.files()).select(["path", "deletion_vector"])
                return {
                    path: dv
                    for path, dv in zip(
                        files.column("path").to_pylist(),
                        files.column("deletion_vector").to_pylist(),
                        strict=True,
                    )
                    if path in touched
                }

            before = vectors(read)
            if before != vectors(fresh) or set(before) != touched:
                return None
            added = not compaction and self._added_what_was_read(read, fresh, read_predicate)
        except _MetadataMoved as moved:
            # Spark's MetadataChangedException, and what the delta-rs path
            # raises for the same race: a caller that re-plans on
            # MetadataChangedError caught it on one engine only.
            raise MetadataChangedError(
                moved.version,
                f"another writer changed the table's metadata (schema, partitioning or "
                f"properties) in version {moved.version}, after this commit read it; "
                "plan the operation again against the table as it is now",
            ) from None
        except Exception:
            return None  # cannot tell; surfacing the conflict is the safe answer
        if added:
            # Spark's ConcurrentAppendException, said plainly: the engine's
            # own "committed first" reads as a race a retry wins as it stands.
            raise CommitConflictError(
                int(fresh.version),
                "a concurrent commit (not a blind append) added rows this transaction's "
                "read could have matched, so committing it as read would ignore them; "
                "nothing was committed. Re-read the table and run it again",
            )
        return fresh

    def _added_what_was_read(self, read: Any, fresh: Any, read_predicate: str | None) -> bool:
        """Whether a commit after `read` that was not a blind append added rows it could read.

        Judged from the winners' own commit files: which were blind appends
        (only adds, and `_blind_append` says so), and which data files the
        others added. A replaceWhere or INSERT ... SELECT is a WRITE with only
        adds too, and taking every such WRITE for a blind append let four
        concurrent replaceWhere loads of one range each commit their rows. An
        added file
        the read predicate skips could not hold a row this transaction would
        have matched; one already gone again (compacted since) cannot be
        judged and counts as read.
        """
        import pyarrow as pa

        if not _native_has("commit_log"):
            return True
        live_at_read = set(pa.table(read.files()).column("path").to_pylist())
        added: set[str] = set()
        for _version, text in fresh.commit_log(int(read.version)):
            actions = [json.loads(line) for line in text.splitlines() if line.strip()]
            info = next((a["commitInfo"] for a in actions if "commitInfo" in a), None) or {}
            if not any(
                k in a for a in actions for k in ("remove", "metaData", "protocol")
            ) and _blind_append(info):
                continue
            added.update(
                a["add"]["path"]
                for a in actions
                if "add" in a
                and a["add"].get("dataChange", True)
                # A file re-added with a new vector holds no new rows.
                and a["add"]["path"] not in live_at_read
            )
        if not added:
            return False
        if read_predicate is None:
            return True
        candidates = set(pa.table(fresh.files(predicate=read_predicate)).column("path").to_pylist())
        live = set(pa.table(fresh.files()).column("path").to_pylist())
        return bool(added & candidates) or not added <= live

    # ------------------------------------------------------------ compaction

    def compaction_refusal(self, table: ResolvedTable, **shape: Any) -> str | None:
        """Why the kernel cannot run this OPTIMIZE of this table, or None.

        delta-rs's own OPTIMIZE commit rebases over a concurrent compaction
        of the same files (its conflict check ignores `dataChange=false`
        removes) and duplicates every row both compacted; the kernel commits
        on the snapshot it read, so the loser conflicts instead. It is the one
        compaction path: `supports(OPTIMIZE)` and delta-rs's `optimize` both
        ask this.
        """
        cap = self.supports(Operation.OPTIMIZE, table, **shape)
        return None if cap.ok else cap.reason

    def _compaction_capability(
        self, operation: Operation, table: ResolvedTable, shape: dict[str, Any]
    ) -> Capability:
        """`supports()` for OPTIMIZE and Z-ORDER, past the checks every operation gets.

        Not the write gate: a compaction writes back the values it read, so
        CHECK constraints, generated and identity columns and invariants (and
        the writer versions 3-6 that imply them) do not constrain it -- the
        native commit sets exactly those aside (`compaction_snapshot`).
        """
        refusal = (
            _compaction_option_refusal(shape)
            or self._compaction_table_refusal(table)
            or _compaction_feature_refusal(table.effective_writer_features, table.properties)
        )
        if refusal is not None:
            reason, remedy = refusal
            return Capability(operation, ok=False, reason=reason, remedy=remedy or "")
        return Capability(operation, ok=True, engine=self.kind)

    def _compaction_table_refusal(self, table: ResolvedTable) -> tuple[str, str | None] | None:
        if not _native_has(*_COMPACTION_NATIVE):
            return "the native extension has no streaming kernel compaction", None
        if table.is_catalog_managed:
            return (
                "a catalog-managed table's commits are ratified by Unity Catalog, and the "
                "kernel's compaction commits to storage directly",
                SQL_FALLBACK_REMEDY,
            )
        if table.is_shallow_clone:
            return "a shallow clone's data files belong to its source table", None
        if table.properties.get("delta.enableVariantShredding", "").lower() == "true":
            return (
                "the table shreds VARIANT values (delta.enableVariantShredding), and the "
                "kernel cannot decode a shredded file to rewrite it",
                SQL_FALLBACK_REMEDY,
            )
        # Legacy-calendar and INT96 files compact too: the scan rebases them,
        # and the kernel writer marks every file it writes with the Spark
        # version, so Databricks reads the rewritten values as written.
        return None

    def _legacy_files(self, table: ResolvedTable, snapshot: Any = None) -> tuple[str, ...]:
        """Live files written in Spark's legacy calendar or with INT96 timestamps."""
        if not _native_has("legacy_calendar_files"):
            return ()
        from .calendar import legacy_calendar_files

        try:
            return tuple(legacy_calendar_files(snapshot or self.snapshot(table)))
        except Exception as exc:
            # Unknown is not "none": a wrong guess rewrites them shifted.
            return (f"<could not check: {type(exc).__name__}: {str(exc)[:120]}>",)

    #: Commits of one OPTIMIZE's rewritten files that lost a race, re-made on
    #: the new snapshot (or re-planned from it) before giving up.
    compaction_commit_retries = 15

    #: Input (compressed file bytes) committed per step; a larger OPTIMIZE
    #: commits in several. Rows are streamed, so this bounds how much work a
    #: lost race repeats, not memory.
    compaction_batch_bytes = 1 << 30

    #: Decoded (Arrow) bytes held for one output file, and for one Z-order
    #: sort. The kernel writes each output file from a single batch, so this
    #: bounds what an OPTIMIZE holds in memory: about twice it, whatever the
    #: size of the table or of a partition (the whole bin, and for a Z-order
    #: the whole partition, used to be read into memory at once). Where rows
    #: decode to more than this per target-sized file, output files come out
    #: smaller than the target; the planner then leaves such files alone.
    compaction_max_file_bytes = 512 << 20

    def optimize(
        self,
        table: ResolvedTable,
        *,
        zorder_by: list[str] | str | None = None,
        full: bool = False,
        predicate: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """OPTIMIZE (bin-packing, or Z-order with `zorder_by`), committed by the kernel.

        delta-rs's options as delta-rs takes them. `partition_filters` scopes
        it; `target_size` (default `delta.targetFileSize`, else 100 MiB) sizes
        the output; `commit_properties` / `commit_metadata` /
        `max_commit_retries` and `post_commithook_properties(create_checkpoint=)`
        are honoured. delta-rs's executor knobs (`max_concurrent_tasks`,
        `max_spill_size`, `max_temp_directory_size`) have nothing to bound
        here. `writer_properties`, `min_commit_interval`, app transactions and
        log cleanup are refused (see `supports`).
        """
        if isinstance(zorder_by, str):
            zorder_by = [zorder_by]
        shape = {"zorder_by": zorder_by, "full": full, "predicate": predicate, **kwargs}
        unknown = sorted(set(kwargs) - _COMPACTION_OPTIONS)
        if unknown:
            raise InvalidArgumentError(
                f"optimize got unexpected option(s) {unknown}; it takes "
                f"{sorted(_COMPACTION_OPTIONS)}"
            )
        refusal = _compaction_option_refusal(shape)
        if refusal is not None:
            raise EngineLimitError("optimize on the kernel", refusal[0], refusal[1])
        properties = kwargs.get("commit_properties")
        metadata = {
            **dict(getattr(properties, "custom_metadata", None) or {}),
            **dict(kwargs.get("commit_metadata") or {}),
        }
        retries = kwargs.get("max_commit_retries")
        if retries is None:
            retries = getattr(properties, "max_commit_retries", None)
        target = kwargs.get("target_size")
        if target is None:
            target = parse_byte_size(table.properties.get("delta.targetFileSize"))
        hooks = kwargs.get("post_commithook_properties")
        return self.compact(
            table,
            zorder_by=list(zorder_by) if zorder_by else None,
            target_size=target,
            partition_filters=kwargs.get("partition_filters"),
            commit_metadata=metadata or None,
            max_commit_retries=retries,
            checkpoint=getattr(hooks, "create_checkpoint", True) is not False,
        )

    def zorder(
        self, table: ResolvedTable, columns: list[str] | str, **kwargs: Any
    ) -> dict[str, Any]:
        columns = [columns] if isinstance(columns, str) else list(columns or [])
        if not columns:
            raise InvalidArgumentError("Z-ORDER needs at least one column")
        return self.optimize(table, zorder_by=columns, **kwargs)

    def compact(
        self,
        table: ResolvedTable,
        *,
        zorder_by: list[str] | None = None,
        target_size: int | None = None,
        partition_filters: list[Any] | None = None,
        commit_metadata: dict[str, Any] | None = None,
        max_commit_retries: int | None = None,
        checkpoint: bool = True,
    ) -> dict[str, Any]:
        """OPTIMIZE (bin-packing, or Z-order with `zorder_by`), committed by the kernel.

        Plans from one snapshot and commits the rewritten files against it,
        `dataChange=false`. When a concurrent commit wins, the same files are
        committed on the new snapshot if it left every file this one removes
        as it was read (Delta's rule for compactions); otherwise the rest is
        planned again from the new snapshot, so a file some other OPTIMIZE
        compacted first is never compacted twice. Each step's rows stream from
        one scan (one log replay) into the commit that writes them.
        """
        import pyarrow as pa

        _enter_native("compact the table with the kernel")
        retries = (
            self.compaction_commit_retries
            if max_commit_retries is None
            else int(max_commit_retries)
        )
        target = int(target_size or _DEFAULT_TARGET_SIZE)
        metrics: dict[str, Any] = {
            "numFilesAdded": 0,
            "numFilesRemoved": 0,
            "partitionsOptimized": 0,
            "numBatches": 0,
            "totalConsideredFiles": 0,
            "totalFilesSkipped": 0,
            "preserveInsertionOrder": not zorder_by,
        }
        added_sizes: list[int] = []
        removed_sizes: list[int] = []
        done: set[str] = set()
        partitions: set[Any] = set()
        conflicts = 0
        first_plan = True
        read = None
        while True:
            snapshot = self.snapshot(table, write=True)
            if read is not None and (
                snapshot.protocol_json() != read.protocol_json()
                or snapshot.metadata_json() != read.metadata_json()
            ):
                # A concurrent protocol or metadata change (row tracking
                # enabled, a feature the kernel cannot write) won the race;
                # the table may no longer be one this path can compact, and
                # re-planning into the native commit's refusal raised an
                # untyped ValueError.
                refused = _snapshot_compaction_refusal(snapshot)
                if refused is not None:
                    raise CommitConflictError(
                        int(snapshot.version),
                        "a concurrent commit changed the table's protocol or metadata, and "
                        f"the kernel can no longer compact it: {refused}. What this OPTIMIZE "
                        "committed before then stands; nothing else was committed",
                    )
            read = snapshot
            files = pa.table(snapshot.files())
            bins, considered, skipped = self._plan_compaction(
                snapshot, files, zorder_by, target, partition_filters, done
            )
            if first_plan:
                metrics["totalConsideredFiles"] = considered
                metrics["totalFilesSkipped"] = skipped
                first_plan = False
            if not bins:
                break
            step: list[_Bin] = []
            budget = 0
            for item in bins:
                step.append(item)
                budget += sum(item.sizes)
                if budget >= self.compaction_batch_bytes:
                    break
            removing = [p for item in step for p in item.paths]
            before = set(files.column("path").to_pylist())
            while True:
                try:
                    version = self._commit_compaction(
                        table,
                        snapshot,
                        step,
                        zorder_by,
                        target,
                        commit_metadata,
                        checkpoint,
                        [_filter_text(f) for f in partition_filters or []],
                    )
                    break
                except CommitConflictError as exc:
                    conflicts += 1
                    if conflicts > retries:
                        raise
                    rebased = self._rebase_dv_commit(
                        table, snapshot, set(removing), None, compaction=True
                    )
                    missing = getattr(exc, "missing", None)
                    if missing is not None and rebased is not None:
                        # Still in the table, yet gone from storage: no race,
                        # a dangling file, which planning again reads again.
                        raise missing from exc
                    commit_backoff(conflicts - 1)
                    if rebased is None:
                        # Another writer changed a file this one would remove
                        # (another OPTIMIZE compacted it, most likely): the
                        # rows read are stale, so plan again from the table
                        # as it is now.
                        version = None
                        break
                    # Every file this step removes is as it was read: the same
                    # rows, streamed again, commit on the new snapshot.
                    snapshot = rebased
                    before = set(pa.table(snapshot.files()).column("path").to_pylist())
            if version is None:
                continue
            done.update(removing)
            partitions.update(item.key for item in step)
            metrics["numBatches"] += len(step)
            metrics["numFilesRemoved"] += len(removing)
            removed_sizes.extend(size for item in step for size in item.sizes)
            after = pa.table(self.snapshot(table, version=version).files())
            for path, size in zip(
                after.column("path").to_pylist(), after.column("size").to_pylist(), strict=True
            ):
                if path not in before:
                    added_sizes.append(int(size))
                    # Written by this OPTIMIZE: planning again (a Z-order
                    # rewrites whole partitions) must not take it back up.
                    done.add(path)
        metrics["numFilesAdded"] = len(added_sizes)
        metrics["partitionsOptimized"] = len(partitions)
        metrics["filesAdded"] = _size_stats(added_sizes)
        metrics["filesRemoved"] = _size_stats(removed_sizes)
        return _Compacted(metrics)

    def _plan_compaction(
        self,
        snapshot: Any,
        files: Any,
        zorder_by: list[str] | None,
        target: int,
        partition_filters: list[Any] | None,
        exclude: set[str],
    ) -> tuple[list[_Bin], int, int]:
        """Bins of files to rewrite together, and the considered and skipped counts.

        As delta-rs plans them: per partition, files below the target size
        packed greedily up to it, a bin of one file left alone -- unless it
        carries a deletion vector, whose deleted rows the rewrite drops. A
        Z-order rewrites every file of each partition as one bin.
        """
        physical = _physical_partition_names(snapshot)
        keep = _partition_filter(snapshot, partition_filters, physical)
        width = _decoded_row_bytes(_arrow_schema(snapshot))
        groups: dict[Any, list[tuple[str, int, bool, int | None]]] = {}
        considered = skipped = 0
        for path, size, values, dv, records in zip(
            files.column("path").to_pylist(),
            files.column("size").to_pylist(),
            files.column("partition_values").to_pylist(),
            files.column("deletion_vector").to_pylist(),
            files.column("num_records").to_pylist(),
            strict=True,
        ):
            values = dict(values or [])
            if path in exclude or not keep(values):
                continue
            considered += 1
            key = tuple(sorted(values.items()))
            live = None
            if records is not None:
                live = int(records) - (int(json.loads(dv).get("cardinality", 0)) if dv else 0)
            groups.setdefault(key, []).append((path, int(size), dv is not None, live))
        bins: list[_Bin] = []
        for key in sorted(groups, key=lambda k: [(n, v is None, v or "") for n, v in k]):
            members = groups[key]
            if zorder_by:
                bins.append(_Bin.of(key, members))
                continue
            current: list[tuple[str, int, bool, int | None]] = []
            total = 0
            packed: list[list[tuple[str, int, bool, int | None]]] = []
            for member in sorted(members, key=lambda m: m[1]):
                if member[1] >= target and not member[2]:
                    skipped += 1
                    continue
                if current and total + member[1] > target:
                    packed.append(current)
                    current, total = [], 0
                current.append(member)
                total += member[1]
            if current:
                packed.append(current)
            for group in packed:
                if len(group) == 1 and not group[0][2]:
                    skipped += 1
                    continue
                if not any(m[2] for m in group) and self._no_fewer_files(group, width):
                    # Files the memory bound cut apart: rewriting them would
                    # write as many again, on every OPTIMIZE.
                    skipped += len(group)
                    continue
                bins.append(_Bin.of(key, group))
        return bins, considered, skipped

    def _no_fewer_files(self, group: list[tuple[str, int, bool, int | None]], width: int) -> bool:
        """Whether compacting `group` would write at least as many files as it holds.

        Judged by its rows at `width` decoded bytes each against the memory
        bound on one output file (`compaction_max_file_bytes`).
        """
        rows = [m[3] for m in group]
        if any(r is None for r in rows):
            return False
        decoded = sum(r for r in rows if r is not None) * width
        return -(-decoded // max(1, int(self.compaction_max_file_bytes))) >= len(group)

    def _compacted_batches(
        self,
        snapshot: Any,
        step: list[_Bin],
        zorder_by: list[str] | None,
        target: int,
    ) -> Any:
        """The rows of each bin, as a stream of batches that are one output file each.

        One scan reads every file of the step, in bin order (a log replay per
        call used to cost 0.07-0.7 s per bin), with each batch tagged by its
        file; the rows of a bin are held only until they make an output file,
        and at most `compaction_max_file_bytes` of them at once -- about
        twice that at the moment one is joined into a file's single batch.
        """
        import pyarrow as pa

        order = [p for item in step for p in item.paths]
        owner = {p: i for i, item in enumerate(step) for p in item.paths}
        stream = pa.RecordBatchReader.from_stream(
            snapshot.scan(
                files=order,
                # Each batch tagged by its file, and a bin's files merged into
                # few batches rather than one each.
                file_groups=[len(item.paths) for item in step],
            )
        )
        positions = [_FILE_COLUMN]
        schema = pa.schema([f for f in stream.schema if f.name not in positions])
        cap = max(1, int(self.compaction_max_file_bytes))
        codec = _pyarrow_codec((snapshot.table_properties() or {}).get(_CODEC_PROPERTY))

        def files_of(index: int, batches: list[Any], final: bool) -> tuple[list[Any], list[Any]]:
            """Output files cut from `batches` of bin `index`, and the rows left over.

            At the bin's end (or at the memory cap) every row goes, in equal
            slices so the last is not a sliver next to full files; before
            it, only whole files of the size the bin's statistics give.
            """
            item = step[index]
            rows = pa.Table.from_batches(batches, schema=schema)
            per_file = item.rows_per_file(target, rows.num_rows)
            if not final:
                whole = rows.num_rows // per_file
                if whole == 0:
                    return [], batches
                out = [_one_batch(rows.slice(i * per_file, per_file)) for i in range(whole)]
                rest = rows.slice(whole * per_file)
                return out, rest.to_batches() if rest.num_rows else []
            if zorder_by:
                rows = rows.take(_zorder_indices(rows, zorder_by))
                # Sorted rows compress worse than the input they came from
                # (files came out 40% over the target): size them by how the
                # first file's worth of them (a sample, at most) encodes.
                sample = rows.slice(0, min(per_file, _SIZE_SAMPLE_ROWS))
                per_file = _encoded_rows_per_file(sample, target, per_file, codec)
            # As many files as the target size asks, and enough that none
            # holds more than the memory bound.
            count = max(-(-rows.num_rows // per_file), -(-rows.nbytes // cap), 1)
            size = max(1, -(-rows.num_rows // count))
            return [_one_batch(rows.slice(i, size)) for i in range(0, rows.num_rows, size)], []

        def inputs() -> Any:
            try:
                yield from stream
            except Exception as exc:
                missing = missing_file_error(exc, "the files this compaction rewrites")
                if missing is None:
                    raise
                # Another writer removed the file and VACUUM deleted it after
                # this step was planned: a lost race (Spark's
                # ConcurrentDeleteReadException), planned again from the
                # table as it is now, not a bad argument.
                conflict = CommitConflictError(
                    int(snapshot.version),
                    f"a file this compaction reads is gone from storage: {missing}",
                )
                conflict.missing = missing  # type: ignore[attr-defined]
                raise conflict from exc

        def generate() -> Any:
            current: int | None = None
            held: list[Any] = []
            held_bytes = 0
            for batch in inputs():
                if batch.num_rows == 0:
                    continue
                paths = batch.column(_FILE_COLUMN)
                index = owner[paths[0].as_py()]
                data = batch.drop_columns(positions)
                if current is not None and index != current and held:
                    out, held = files_of(current, held, final=True)
                    yield from out
                    held_bytes = 0
                current = index
                held.append(data)
                held_bytes += data.nbytes
                if held_bytes >= cap:
                    # Past what one file (or one Z-order sort) may hold: cut
                    # files from what is here rather than hold the whole bin.
                    out, held = files_of(current, held, final=True)
                    yield from out
                    held_bytes = sum(b.nbytes for b in held)
                elif not zorder_by and step[current].splits(target):
                    out, held = files_of(current, held, final=False)
                    yield from out
                    held_bytes = sum(b.nbytes for b in held)
            if current is not None and held:
                out, _ = files_of(current, held, final=True)
                yield from out

        return pa.RecordBatchReader.from_batches(schema, generate())

    def _commit_compaction(
        self,
        table: ResolvedTable,
        snapshot: Any,
        step: list[_Bin],
        zorder_by: list[str] | None,
        target: int,
        commit_metadata: dict[str, Any] | None,
        checkpoint: bool = True,
        predicate: list[str] | None = None,
    ) -> int:
        import pyarrow as pa

        removing = [p for item in step for p in item.paths]
        data = self._compacted_batches(snapshot, step, zorder_by, target)
        empty = pa.table({"path": pa.array([], pa.string()), "row_index": pa.array([], pa.int64())})
        with translating(EngineKind.KERNEL, "commit"):
            version, _deleted, _dvs, _removed = snapshot.commit_dml(
                empty.to_reader(),
                data=recorded(data),
                whole_files=removing,
                engine_info=_engine_info(),
                operation="OPTIMIZE",
                commit_metadata={k: str(v) for k, v in (commit_metadata or {}).items()} or None,
                data_change=False,
                # As Spark records an OPTIMIZE; Databricks' history showed {}.
                **_commit_info(
                    blind=False,
                    predicate=predicate or [],
                    zOrderBy=list(zorder_by or []),
                    auto="false",
                ),
            )
        if checkpoint and int(version) != int(snapshot.version):
            self._maybe_checkpoint(table, int(version), snapshot)
        return int(version)

    def merge(
        self,
        table: ResolvedTable,
        source: Any,
        predicate: str,
        **kwargs: Any,
    ) -> Any:
        """Start a MERGE, written as deletion vectors. Mirrors delta-rs's clause API."""
        from .kernel_merge import KernelMerger

        _require_pyarrow("merge on the kernel path")
        return KernelMerger(self, table, _as_record_batch_reader(source), predicate, **kwargs)

    def delete(
        self,
        table: ResolvedTable,
        predicate: str | None = None,
        *,
        commit_metadata: dict[str, Any] | None = None,
        **unsupported: Any,
    ) -> dict[str, Any]:
        """DELETE by rewriting the table without the matching rows.

        SQL semantics: a row is deleted only where the predicate is TRUE; a
        NULL result keeps it.
        """
        _refuse_options("delete", unsupported)
        if self._dv_path(table):
            result = self._dv_dml(
                table, predicate, operation="DELETE", commit_metadata=commit_metadata
            )
        else:
            result = self._rewrite(
                table,
                predicate,
                lambda current, keep: current.filter(keep),
                operation="DELETE",
                commit_metadata=commit_metadata,
            )
        return {"num_deleted_rows": result["num_affected_rows"], "version": result["version"]}

    def update(
        self,
        table: ResolvedTable,
        *,
        updates: dict[str, str] | None = None,
        new_values: dict[str, Any] | None = None,
        predicate: str | None = None,
        commit_metadata: dict[str, Any] | None = None,
        **unsupported: Any,
    ) -> dict[str, Any]:
        """UPDATE by rewriting the table. Assignments are plain values.

        `updates` (SQL expressions) are accepted only when each is a literal or
        a column reference, since nothing here evaluates arbitrary SQL.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        from .kernel_merge import store_cast

        _refuse_options("update", unsupported)
        assignments: dict[str, Any] = dict(new_values or {})
        for column, expression in (updates or {}).items():
            value = sqlpred.parse_value(expression)
            assignments[column] = value
        if not assignments:
            raise UnreachableTableError("update", "no assignments given")
        from .._variant import log_variant_columns, string_variant, variant_column

        variants: frozenset[str] = frozenset()
        if any(
            isinstance(v.value if isinstance(v, sqlpred.Literal) else v, str)
            for v in assignments.values()
        ) and table.features & {"variantType", "variantType-preview"}:
            variants = log_variant_columns(getattr(self.snapshot(table), "metadata_json", dict)())

        def column_index(schema: Any, name: str, what: str) -> int:
            # Delta column names are case-insensitive.
            index = int(schema.get_field_index(_canonical_path(schema, (name,))[0]))
            if index < 0:
                raise InvalidArgumentError(f"cannot {what}: the table has no column {name!r}")
            return index

        def assign(current: Any, keep: Any) -> Any:
            out = current
            for column, value in assignments.items():
                index = column_index(out.schema, column, f"update {column}")
                field = out.schema.field(index)
                if isinstance(value, sqlpred.Column):
                    # SQL reads every right-hand side from the row as it was:
                    # `SET a = b, b = a` swaps. Reading `out` chained them.
                    if len(value.path) != 1:
                        raise UnreachableTableError(
                            f"update {column}", "assigning from a nested field is not supported"
                        )
                    what = f"update {column} from {value.path[0]}"
                    source_index = column_index(current.schema, value.path[0], what)
                    source = store_cast(current.column(source_index), field.type, field.name)
                else:
                    # Stored as Spark stores it: 12.345 rounds into a
                    # DECIMAL(10,2), where Arrow's cast refused it.
                    raw = value.value if isinstance(value, sqlpred.Literal) else value
                    if field.name.lower() in variants and isinstance(raw, str):
                        # A new_values string is JSON text, as every write
                        # takes a VARIANT; a SQL string literal is a variant
                        # string, as Spark stores one.
                        text = string_variant(raw) if isinstance(value, sqlpred.Literal) else raw
                        encoded = variant_column(pa, pa.array([text] * out.num_rows, pa.string()))
                        source = store_cast(encoded, field.type, field.name)
                        new = pc.if_else(keep, out.column(index), source)
                        out = out.set_column(index, field, new)
                        continue
                    try:
                        literal = pa.array([raw] * out.num_rows)
                    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
                        raise InvalidArgumentError(
                            f"update {column}: {raw!r} is not a value Arrow can hold: {exc}"
                        ) from exc
                    source = store_cast(literal, field.type, field.name)
                new = pc.if_else(keep, out.column(index), source)
                out = out.set_column(index, field, new)
            return out

        if self._dv_path(table):
            result = self._dv_dml(
                table,
                predicate,
                operation="UPDATE",
                transform=lambda matched: assign(
                    matched, pa.nulls(matched.num_rows, pa.bool_()).fill_null(False)
                ),
                commit_metadata=commit_metadata,
            )
        else:
            result = self._rewrite(
                table, predicate, assign, operation="UPDATE", commit_metadata=commit_metadata
            )
        return {"num_updated_rows": result["num_affected_rows"], "version": result["version"]}

    def publish(self, table: ResolvedTable) -> int:
        """Publish ratified-but-unpublished commits into the Delta log."""
        snapshot = self.snapshot(table, write=True)
        with translating(EngineKind.KERNEL, "commit"):
            version: int = snapshot.publish(uc=self._uc_commit_config(table))
        return version

    # ---------------------------------------------------- metadata-only DDL

    @staticmethod
    def _implied_only_note(table: ResolvedTable, blockers: list[str]) -> str:
        """Explain any blocker the table never named and does not actually use.

        A legacy writer version implies a whole set of features. Version 4 is
        reached by enabling change data feed alone, and it implies
        `checkConstraints`, which the kernel refuses whether or not a single
        constraint exists. Reporting only the feature name leaves the owner of a
        CDF table with no constraints hunting for constraints they never wrote.

        Only the features that are genuinely implied *and* unused are named, so
        a table that really does have a constraint is not told otherwise.
        """
        in_use = {
            "checkConstraints": table.has_check_constraints,
            "generatedColumns": table.has_generated_columns,
            "invariants": table.has_invariants,
        }
        implied_only = sorted(
            name for name in set(blockers) - set(table.writer_features) if in_use.get(name) is False
        )
        if not implied_only:
            return ""
        which = ", ".join(implied_only)
        it = "them" if len(implied_only) > 1 else "it"
        return (
            f". The table neither names nor uses {which}: writer version "
            f"{table.min_writer_version} implies {it}, and the kernel refuses the table "
            "on that alone"
        )

    def _metadata_refusal(self, operation: Operation, table: ResolvedTable) -> Capability | None:
        """Refusals specific to the commits this engine writes itself."""
        if table.is_catalog_managed:
            return Capability(
                operation,
                ok=False,
                reason="the table is catalog-managed; its metadata changes go through the "
                "catalog, which refuses them from external writers after version 0",
            )
        if table.has_iceberg_compat:
            return Capability(
                operation,
                ok=False,
                reason="the table has Iceberg reads enabled, and a metadata change written "
                "here would leave its Iceberg metadata stale",
                remedy="perform this change from Databricks, or enable the SQL fallback",
            )
        if operation in (Operation.DROP_COLUMN, Operation.RENAME_COLUMN) and table.properties.get(
            "delta.columnMapping.mode", "none"
        ).lower() not in ("name", "id"):
            # Decided here rather than inside the commit. Without column mapping
            # a column's name is also its name in every Parquet file, so the
            # change would need the data rewritten -- which this path does not
            # do. The mode is a table property, so saying so costs nothing.
            return Capability(
                operation,
                ok=False,
                # The remedy is carried in the reason as well: the router keeps
                # only its own remedy when it aggregates engine verdicts, and
                # turning column mapping on is far more use here than being
                # told a SQL warehouse could do it.
                reason=(
                    "renaming or dropping a column without rewriting data needs column "
                    "mapping, and this table does not use it: the column's name is also "
                    "its name in every Parquet file. Set "
                    "delta.columnMapping.mode='name' first"
                ),
                remedy="t.set_properties({'delta.columnMapping.mode': 'name'}) first",
            )
        if (
            operation in (Operation.DROP_COLUMN, Operation.RENAME_COLUMN)
            and "columnMapping" not in table.effective_writer_features
        ):
            # The mode alone is not column mapping: a protocol without the
            # feature tells every writer to use logical names in the Parquet
            # files (delta-rs's create once wrote exactly such tables), and a
            # rename then read the renamed column back as NULL.
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table sets delta.columnMapping.mode, but its protocol does not "
                    "support the columnMapping feature, so its data files are keyed by "
                    "logical column names and renaming or dropping one would lose its data"
                ),
                remedy="rewrite the table into a new one created with column mapping",
            )

        if operation is Operation.CLUSTER_BY and table.partition_columns:
            return Capability(
                operation,
                ok=False,
                reason="the table is partitioned; a table is either partitioned or clustered",
            )
        return None

    def _state(self, table: ResolvedTable) -> tuple[Any, Any]:
        """(snapshot, TableState) for the latest version."""
        snapshot = self.snapshot(table, write=True)
        clustering_raw = snapshot.domain_metadata(CLUSTERING_DOMAIN)
        state = TableState(
            version=snapshot.version,
            protocol=json.loads(snapshot.protocol_json()),
            metadata=json.loads(snapshot.metadata_json()),
            timestamp=snapshot.timestamp(),
            clustering=json.loads(clustering_raw) if clustering_raw else None,
        )
        return snapshot, state

    #: Attempts before a metadata change gives up on a busy table. Each retry
    #: recomputes the change against the state another writer just committed.
    metadata_commit_attempts = 5

    def _commit_metadata(
        self,
        table: ResolvedTable,
        mutate: Any,
        *,
        precheck: Any = None,
    ) -> int:
        """Compute a metadata change against the latest state and commit it.

        The commit is a put-if-absent of the next log file, so a concurrent
        writer makes it fail instead of being overwritten. The change is a pure
        function of the state, so recomputing it against the new state is
        exactly what serial execution would have produced.
        """
        from deltaswamp import _native

        last_error: Exception | None = None
        for _ in range(self.metadata_commit_attempts):
            snapshot, state = self._state(table)
            if precheck is not None:
                precheck(snapshot, state)
            change = mutate(state)
            if change.protocol is None and change.metadata is None and not change.domains:
                return int(state.version)
            actions = build_actions(state, change, engine_info=_engine_info())
            try:
                assert table.location is not None  # supports() refused otherwise
                version: int = _native.commit_raw(
                    table.location,
                    state.version + 1,
                    actions,
                    options=self._options(table, write=True) or None,
                )
            except _native.CommitConflictError as exc:
                last_error = exc
                continue
            if not table.is_catalog_managed:
                write_checksum(table.location, self._options(table, write=True), version)
            return version
        # A lost race is a conflict, not an unreachable table: callers that
        # catch CommitConflictError to retry never saw this one.
        raise CommitConflictError(
            conflict_version(str(last_error)),
            f"cannot commit a metadata change: another writer committed first on each "
            f"of {self.metadata_commit_attempts} attempts ({last_error}); retry when the "
            "table is less busy",
        )

    #: Warn when a scan starts with less than this much credential life left.
    #: A long read that outlives its credential fails partway through, and the
    #: storage layer reports only a 403.
    expiry_warning_seconds: float = 300.0

    def _options(self, table: ResolvedTable, *, write: bool) -> dict[str, str]:
        # Merged by the one rule every engine shares (_storage): a plain
        # update() kept alias spellings side by side (AWS_REGION next to a
        # vended aws_region), and object_store then chose between them in
        # HashMap order.
        vended = None
        if table.credential_provider is not None:
            op = CredentialOperation.READ_WRITE if write else CredentialOperation.READ
            credentials = table.credential_provider.credentials(op)
            self._warn_if_short_lived(credentials)
            vended = credentials.as_storage_options()
        return store_options(engine_options(self._base_options, vended, table.location))

    def _warn_if_short_lived(self, credentials: Any) -> None:
        """Say so when the credential may not outlive the read it is about to serve.

        The object store is built once per snapshot and holds this credential
        for the whole scan, so there is no refresh to rescue a long read.
        """
        remaining = getattr(credentials, "expires_at", None)
        if remaining is None or not credentials.expires_within(self.expiry_warning_seconds):
            return
        import time
        import warnings

        from ..errors import CredentialExpiryWarning

        left = max(0.0, remaining - time.time())
        warnings.warn(
            f"the vended credential for this table expires in {left:.0f}s, and it is "
            "held for the whole scan: the object store is built once per snapshot, so "
            "a read that runs longer fails partway through with a 403 from storage. "
            "Split the read with plan_scan()/to_ray_dataset(), where each worker vends "
            "its own, or re-open the table to mint a fresh one.",
            CredentialExpiryWarning,
            stacklevel=4,
        )

    def add_columns(self, table: ResolvedTable, fields: Any, **_: Any) -> int:
        new_fields = _delta_fields(fields)
        return self._commit_metadata(table, lambda s: meta.add_columns(s, new_fields))

    def drop_column(self, table: ResolvedTable, column: str) -> int:
        return self._commit_metadata(table, lambda s: meta.drop_column(s, column))

    def rename_column(self, table: ResolvedTable, old: str, new: str) -> int:
        return self._commit_metadata(table, lambda s: meta.rename_column(s, old, new))

    def set_properties(self, table: ResolvedTable, properties: dict[str, str], **_: Any) -> int:
        return self._commit_metadata(table, lambda s: meta.set_properties(s, properties))

    def unset_properties(
        self, table: ResolvedTable, keys: list[str], *, if_exists: bool = True
    ) -> int:
        return self._commit_metadata(
            table, lambda s: meta.unset_properties(s, keys, if_exists=if_exists)
        )

    @staticmethod
    def _add_feature_refusal(table: ResolvedTable, features: Any) -> Capability | None:
        """The features this path cannot add, decided before the call as the call decides.

        One the protocol already supports -- named, or implied by a legacy
        version (writer 4 implies generatedColumns, (2, 5) columnMapping) --
        needs nothing added, so it never blocks.
        """
        names = features if isinstance(features, (list, tuple, set, frozenset)) else [features]
        have = table.effective_writer_features | table.effective_reader_features
        blocked = []
        for name in names:
            wire = meta._canonical_feature(str(getattr(name, "value", name)))
            if wire in have:
                continue
            if wire in meta._NOT_ADDABLE:
                blocked.append(f"{wire} ({meta._NOT_ADDABLE[wire]})")
            elif wire not in meta._ADDABLE_FEATURES:
                blocked.append(f"{wire} (not a feature a metadata commit here can add)")
        if not blocked:
            return None
        return Capability(
            Operation.ADD_FEATURE,
            ok=False,
            reason="the kernel path cannot add " + "; ".join(blocked),
        )

    def add_feature(self, table: ResolvedTable, feature: Any, **_: Any) -> int:
        names = feature if isinstance(feature, (list, tuple, set, frozenset)) else [feature]
        wires = {meta._canonical_feature(str(getattr(n, "value", n))) for n in names}
        # A feature arrives with what it depends on (rowTracking needs
        # domainMetadata), or the commit is one other engines reject.
        pending = list(wires)
        while pending:
            known = feature_from_wire(pending.pop())
            for dep in FEATURE_DEPENDENCIES.get(known, frozenset()) if known else ():
                if dep.value not in wires:
                    wires.add(dep.value)
                    pending.append(dep.value)

        def mutate(state: Any) -> Any:
            # A feature the protocol already supports, by name or by legacy
            # version, is a no-op, as Spark treats it -- not a refusal.
            have = meta.supported_features(state.protocol) | set(
                state.protocol.get("readerFeatures") or ()
            )
            props = {f"delta.feature.{n}": "supported" for n in sorted(wires - have)}
            return meta.set_properties(state, props)

        return self._commit_metadata(table, mutate)

    def drop_constraint(self, table: ResolvedTable, name: str, *, if_exists: bool = False) -> int:
        return self._commit_metadata(
            table, lambda s: meta.drop_constraint(s, name, if_exists=if_exists)
        )

    def set_comment(self, table: ResolvedTable, comment: str | None) -> int:
        return self._commit_metadata(table, lambda s: meta.set_comment(s, comment))

    def set_column_comment(self, table: ResolvedTable, column: str, comment: str | None) -> int:
        return self._commit_metadata(table, lambda s: meta.set_column_comment(s, column, comment))

    def alter_column_type(self, table: ResolvedTable, column: str, new_type: str) -> int:
        return self._commit_metadata(table, lambda s: meta.alter_column_type(s, column, new_type))

    def drop_not_null(self, table: ResolvedTable, column: str) -> int:
        return self._commit_metadata(table, lambda s: meta.set_nullability(s, column, True))

    def set_not_null(self, table: ResolvedTable, column: str) -> int:
        """SET NOT NULL, after proving no existing row is null.

        The check runs against the same snapshot the commit is computed from,
        and the put-if-absent commit fails if anything was written since -- so
        a null cannot slip in between the check and the change.
        """

        def precheck(snapshot: Any, state: Any) -> None:
            import pyarrow as pa

            # The change's own refusals (no such column, a nested field) come
            # first; the scan below answered them with a native "column is
            # not in the table schema".
            meta.set_nullability(state, column, False)
            nulls = 0
            # The metadata change matches names case-insensitively; the
            # native projection does not, and failed on `ID` for `id`.
            name = _canonical_path(pa.schema(snapshot.schema()), (column,))[0]
            for batch in pa.RecordBatchReader.from_stream(snapshot.scan(columns=[name])):
                nulls += batch.column(0).null_count
            if nulls:
                raise UnreachableTableError(
                    f"SET NOT NULL on {column}",
                    f"{nulls} existing row(s) have a null {column}",
                    "update or delete those rows first",
                )

        return self._commit_metadata(
            table, lambda s: meta.set_nullability(s, column, False), precheck=precheck
        )

    def cluster_by(self, table: ResolvedTable, columns: Any) -> int:
        if isinstance(columns, str):
            if columns.lower() == "auto":
                raise UnreachableTableError(
                    "CLUSTER BY AUTO",
                    "automatic key selection is Databricks predictive optimization, which "
                    "runs server-side",
                    SQL_FALLBACK_REMEDY,
                )
            columns = [columns]
        return self._commit_metadata(table, lambda s: meta.cluster_by(s, list(columns or [])))

    def metadata_count(
        self, table: ResolvedTable, *, predicate: str | None = None, version: int | None = None
    ) -> int | None:
        """The exact row count from the log alone, or None when it cannot be had.

        Delta's `numRecords` is exact, and a deletion vector's cardinality is
        exactly the rows it hides, so with both on every file no data file
        need be opened -- how Databricks answers `count(*)`. None (so the
        caller scans) when any file lacks `numRecords`, or the predicate
        touches anything but partition columns of types whose string form
        converts losslessly; a partition predicate is then applied exactly to
        each file's partition values, with SQL's three-valued logic.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        snapshot = self.snapshot(table, version=version)
        schema = _arrow_schema(snapshot)
        node = sqlpred.parse(predicate) if predicate is not None else None
        partitions: dict[str, Any] = {}
        if node is not None:
            by_lower = {name.lower(): name for name in snapshot.partition_columns}
            for path in sqlpred.columns_of(node):
                name = by_lower.get(path[0].lower()) if len(path) == 1 else None
                if name is None:
                    return None  # a data column: only a scan can decide it
                field = schema.field(name)
                t = field.type
                if not (
                    pa.types.is_string(t)
                    or pa.types.is_integer(t)
                    or pa.types.is_date(t)
                    or pa.types.is_boolean(t)
                ):
                    return None
                partitions[name] = field
        skipping = sqlpred.to_kernel_json(node, schema) if node is not None else None
        files = pa.table(snapshot.files(predicate=skipping))
        records = files.column("num_records")
        if records.null_count:
            return None
        live = pc.subtract(records, _dv_cardinalities(files.column("deletion_vector")))
        if not partitions:
            return int(pc.sum(live).as_py() or 0)
        physical = {
            fld.name: (fld.metadata or {}).get(b"delta.columnMapping.physicalName", b"").decode()
            or fld.name
            for fld in partitions.values()
        }
        values = files.column("partition_values").to_pylist()
        columns = {}
        for name, fld in partitions.items():
            raw = [dict(v or ()).get(physical[name]) for v in values]
            try:
                columns[name] = pa.array(raw, pa.string()).cast(fld.type)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                return None
        columns["__live_rows"] = live
        assert node is not None  # partitions are only collected from a predicate
        kept = sqlpred.filter_table(pa.table(columns), node)
        return int(pc.sum(kept.column("__live_rows")).as_py() or 0)

    def plan_scan(
        self,
        table: ResolvedTable,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: Any = None,
    ) -> list[Any]:
        """Enumerate the files a scan would read, as serializable splits.

        Every split is pinned to one snapshot version, so workers that
        re-resolve the table read exactly the snapshot that was planned, even
        if it has been written to since. Files the predicate's statistics rule
        out are not planned at all.
        """
        from ..table import _require

        pa = _require("pyarrow", "pyarrow", "plan_scan")

        from .base import DeletionVectorDescriptor, ScanSplit

        if not self.supports_distributed_scan:
            raise NotImplementedError(
                "the installed native extension cannot restrict a scan to planned files"
            )
        snapshot = self.snapshot(table, version=version, timestamp=timestamp)
        # Settle on the driver what every worker would otherwise fail on
        # separately: an unknown column, or a predicate that cannot be
        # evaluated against this schema.
        _check_planned_read(snapshot, columns, predicate)
        skipping = (
            sqlpred.to_kernel_json(sqlpred.parse(predicate), _arrow_schema(snapshot))
            if predicate
            else None
        )
        files = pa.table(snapshot.files(predicate=skipping)).to_pylist()
        # The log keys partition values by *physical* name, which under column
        # mapping is a `col-<uuid>`; splits report the logical column name.
        logical = {}
        for fld in pa.schema(snapshot.schema()):
            physical = (fld.metadata or {}).get(b"delta.columnMapping.physicalName")
            if physical is not None:
                logical[physical.decode()] = fld.name
        splits = []
        for f in files:
            dv = f.get("deletion_vector")
            descriptor = None
            if dv:
                raw = json.loads(dv)
                descriptor = DeletionVectorDescriptor(
                    storage_type=raw.get("storageType", ""),
                    path_or_inline=raw.get("pathOrInlineDv", ""),
                    size_in_bytes=int(raw.get("sizeInBytes", 0)),
                    cardinality=int(raw.get("cardinality", 0)),
                    offset=raw.get("offset"),
                )
            partition_values = f.get("partition_values") or {}
            if isinstance(partition_values, list):  # an Arrow map arrives as pairs
                partition_values = dict(partition_values)
            splits.append(
                ScanSplit(
                    path=f["path"],
                    size=int(f["size"]),
                    partition_values={logical.get(k, k): v for k, v in partition_values.items()},
                    deletion_vector=descriptor,
                    commit_version=int(snapshot.version),
                )
            )
        return splits

    def write_files(
        self,
        table: ResolvedTable,
        data: Any,
        *,
        version: int | None = None,
        table_identity: str | None = None,
    ) -> bytes:
        """Write data files for `table` without committing them.

        Returns opaque fragment bytes describing what was written. The files
        exist and are durable once this returns; they belong to no version
        until `commit_files` accepts them, so a coordinator that abandons the
        write leaves them behind.

        `version` writes with that snapshot's layout (the one the write was
        planned at). A pinned snapshot is resolved once per process and then
        reused (revalidated against storage each time), where the latest had
        to be listed again for every call; the commit still refuses fragments
        whose layout the table has since left. `table_identity` is the metaData
        id the write was planned against: a table re-created at the same path
        is refused here rather than written for.
        """
        if not self.supports_distributed_write:
            raise NotImplementedError(
                "the installed native extension cannot write files without committing"
            )
        # The one write path that reuses a cached snapshot: it writes files and
        # commits nothing, and the identity check below ties the reuse to the
        # table the write was planned for.
        snapshot = self.snapshot(table, version=version, write=True, fresh=False)
        _refuse_other_table(snapshot, table_identity, "write files for this plan")
        result: bytes = snapshot.write_files(
            _as_record_batch_reader(data), uc=self._uc_commit_config(table, staging=True)
        )
        # Record the layout these files were written under, from the very
        # snapshot that wrote them, so the commit can tell whether the table
        # still has it.
        return _stamp_fragment(result, _write_layout(snapshot))

    def commit_files(
        self,
        table: ResolvedTable,
        fragments: list[bytes],
        *,
        overwrite: bool = False,
        engine_info: str | None = None,
        operation: str = "WRITE",
        txn: tuple[str, int] | None = None,
        commit_metadata: dict[str, Any] | None = None,
        version: int | None = None,
        table_identity: str | None = None,
    ) -> int:
        """Commit fragments from `write_files` as one transaction.

        Every fragment lands at a single version, so a distributed write is
        atomic: a reader sees all of it or none of it. With `version`, the
        commit is built on that snapshot, so it conflicts if the table has
        moved past it -- what a guarded overwrite needs. The snapshot is read
        from storage, never reused from the cache, and must be the table
        `table_identity` (the planned metaData id) names.
        """
        if not self.supports_distributed_write:
            raise NotImplementedError(
                "the installed native extension cannot commit externally written files"
            )
        snapshot = self.snapshot(table, version=version, write=True)
        _refuse_other_table(snapshot, table_identity, "commit these fragments", committing=True)
        # On the snapshot the commit is built on, like the txn check below: a
        # schema change that lands after it makes the commit conflict, and the
        # retry checks again.
        _refuse_changed_layout(snapshot, fragments)
        if txn is not None and hasattr(snapshot, "app_id_version"):
            # Checked on the very snapshot the commit is built on: a writer
            # that records the txn after this makes the commit conflict, and
            # the retry re-checks. Checking only at plan time let two runs of
            # the same job both commit, and the rows landed twice.
            last = snapshot.app_id_version(txn[0])
            if last is not None and int(last) >= int(txn[1]):
                raise UnreachableTableError(
                    f"commit the idempotent write for {txn[0]!r} at version {txn[1]}",
                    f"that transaction is already committed (the table records {last}), so "
                    "committing these fragments would duplicate rows already in the table",
                    "drop the fragments; their files are unreferenced and VACUUM removes them",
                )
        # A refused fragment is the caller's input: the extension raises it as
        # InvalidInputError, which the translation makes InvalidArgumentError.
        with translating(EngineKind.KERNEL, "commit"):
            committed: int = snapshot.commit_files(
                list(fragments),
                uc=self._uc_commit_config(table),
                engine_info=engine_info or _engine_info(),
                operation=operation,
                overwrite=overwrite,
                txn=txn,
                commit_metadata={k: str(v) for k, v in (commit_metadata or {}).items()} or None,
                **_write_info(snapshot, overwrite=overwrite),
            )
        self._maybe_checkpoint(table, committed, snapshot)
        return committed

    def execute_scan(
        self,
        table: ResolvedTable,
        splits: list[Any],
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
    ) -> Any:
        """Read planned splits: same semantics as `scan`, restricted to their files.

        Deletion vectors, column mapping and partition values are handled by
        the kernel exactly as in a full scan; the predicate is applied exactly
        afterwards. `version` pins the snapshot when there are no splits to
        carry it, so an empty plan reads the planned schema, not the latest.
        """
        versions = {s.commit_version for s in splits}
        if len(versions) > 1:
            raise UnreachableTableError(
                "execute scan splits",
                # key=str: a hand-built split with no version made sorted()
                # raise TypeError comparing None with an int.
                f"the splits were planned against different versions ({sorted(versions, key=str)})",
                "plan once and distribute that plan",
            )
        if versions:
            version = next(iter(versions))
        snapshot = self.snapshot(table, version=version)
        paths = [s.path for s in splits]
        return _planned_read(snapshot, columns, predicate, files=paths)


#: Fragment schema-metadata key: the table layout its files were written
#: under, as JSON. Added here, beside the native table-identity keys.
_FRAGMENT_LAYOUT = "deltaswamp.write_layout"


def _write_layout(snapshot: Any) -> str | None:
    """What a data file written on `snapshot` depends on, as canonical JSON.

    The schema (with column-mapping ids and physical names), the partition
    columns and the configuration that governs how files are written and
    read. A column comment and an identity column's high-water mark change on
    their own and do not affect a single file, so they are left out -- as is
    every other property, so an unrelated SET TBLPROPERTIES does not fail a
    job. None when the binding cannot report the metadata.
    """
    try:
        metadata = json.loads(snapshot.metadata_json())
        schema = json.loads(metadata.get("schemaString") or "{}")
    except Exception:
        return None

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            out = {k: strip(v) for k, v in node.items()}
            field_meta = out.get("metadata")
            if isinstance(field_meta, dict):
                out["metadata"] = {
                    k: v
                    for k, v in field_meta.items()
                    if k not in ("comment", "delta.identity.highWaterMark")
                }
            return out
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    configuration = metadata.get("configuration") or {}
    kept = {
        k: v
        for k, v in configuration.items()
        if k == "delta.columnMapping.mode" or k.startswith("delta.constraints.")
    }
    return json.dumps(
        {
            "fields": strip(schema).get("fields") or [],
            "partitionColumns": list(metadata.get("partitionColumns") or []),
            "configuration": kept,
        },
        sort_keys=True,
    )


def _refuse_other_table(
    snapshot: Any, identity: str | None, action: str, *, committing: bool = False
) -> None:
    """Refuse when `snapshot` is not the table `identity` (a metaData id) names.

    Writing, MetadataChangedError: nothing was written, and the write must be
    planned again. Committing, InvalidArgumentError, as the binding refuses a
    fragment stamped with another table's id -- the same mistake, caught even
    when no fragment carries a file (an empty overwrite would otherwise empty
    the new table).
    """
    if identity is None:
        return
    now = getattr(snapshot, "metadata_id", None)
    if now is None or str(now) == identity:
        return
    from ..errors import InvalidArgumentError, MetadataChangedError

    reason = (
        f"cannot {action}: "
        + ("they were written for a different table -- " if committing else "")
        + "the table at this location was dropped and re-created since the write was "
        f"planned (its table id was {identity!r} and is {str(now)!r} now). Plan the write "
        "again against the new table; files already written are unreferenced and VACUUM "
        "removes them"
    )
    if committing:
        raise InvalidArgumentError(reason)
    raise MetadataChangedError(int(snapshot.version), reason)


def _layout_still_fits(written: str, current: str) -> bool:
    """Whether files written under layout `written` are valid in `current`.

    The same layout, or one that only added nullable top-level columns: a
    file without a column reads it as null, which is what Delta does for
    every file written before an ADD COLUMN. A generated or identity column
    is not a plain null, so adding one counts as a change.
    """
    if written == current:
        return True
    try:
        old, new = json.loads(written), json.loads(current)
    except ValueError:
        return False
    if (old["partitionColumns"], old["configuration"]) != (
        new["partitionColumns"],
        new["configuration"],
    ):
        return False
    before, after = old["fields"], new["fields"]
    if after[: len(before)] != before:
        return False
    for field in after[len(before) :]:
        extra = field.get("metadata") or {}
        if not field.get("nullable", True) or any(
            k.startswith(("delta.generationExpression", "delta.identity")) for k in extra
        ):
            return False
    return True


def _stamp_fragment(fragment: bytes, layout: str | None) -> bytes:
    """`fragment` with `layout` added to its schema metadata."""
    if not fragment or layout is None:
        return fragment  # no files, or nothing to record
    import pyarrow as pa

    reader = pa.ipc.open_stream(fragment)
    schema = reader.schema.with_metadata(
        {**(reader.schema.metadata or {}), _FRAGMENT_LAYOUT.encode(): layout.encode()}
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        for batch in reader:
            writer.write_batch(pa.RecordBatch.from_arrays(batch.columns, schema=schema))
    return bytes(sink.getvalue().to_pybytes())


def _refuse_changed_layout(snapshot: Any, fragments: list[bytes]) -> None:
    """Refuse fragments written under a layout `snapshot` no longer has.

    A commit rebases onto the latest snapshot, which is right for an append --
    but not across a concurrent metaData action. Files written with the old
    types made the whole table unreadable (Expected Utf8, got Int64), and on a
    column-mapping table a dropped and re-added column's values were silently
    read as null: the files carry the old physical name. Spark conflicts here
    (MetadataChangedException). Adding a nullable column is the one change
    let through: the files simply lack it, as every older file does. A
    fragment from a release that did not stamp its layout cannot be checked,
    and commits as before.
    """
    from ..errors import MetadataChangedError

    current: str | None = None
    for fragment in fragments:
        if not fragment:
            continue
        try:
            import pyarrow as pa

            metadata = pa.ipc.open_stream(fragment).schema.metadata or {}
        except Exception:
            continue  # the binding names a malformed fragment itself
        written = metadata.get(_FRAGMENT_LAYOUT.encode())
        if written is None:
            continue
        if current is None:
            current = _write_layout(snapshot)
            if current is None:
                return
        if not _layout_still_fits(written.decode(), current):
            raise MetadataChangedError(
                int(snapshot.version),
                "the table's schema, partitioning or column mapping changed after these "
                f"fragments were written (it is at version {snapshot.version} now), so "
                "their files no longer match it: committing them would leave the table "
                "unreadable or put values in the wrong columns. Re-plan the write and "
                "write the data again; the old fragments' files are unreferenced and "
                "VACUUM removes them",
            )


#: The columns a positional scan (`row_positions=True`) adds to each row.
_FILE_COLUMN = "__deltaswamp_file"
_ROW_INDEX_COLUMN = "__deltaswamp_row_index"


def _canonical_path(schema: Any, path: tuple[str, ...]) -> tuple[str, ...]:
    """`path` spelled as `schema` spells it. Delta names are case-insensitive.

    The kernel resolves a projection and the exact row filter resolves a
    column by exact name, so `ID` against a column `id` used to fail on one
    side and match on the other. Unmatched segments are left as written, for
    the reader to reject.
    """
    import pyarrow as pa

    out: list[str] = []
    current: Any = schema
    for name in path:
        if isinstance(current, pa.Schema):
            fields = list(current)
        elif isinstance(current, pa.StructType):
            fields = [current.field(i) for i in range(current.num_fields)]
        else:
            return (*out, *path[len(out) :])
        match = next((f for f in fields if f.name == name), None) or next(
            (f for f in fields if f.name.lower() == name.lower()), None
        )
        if match is None:
            return (*out, *path[len(out) :])
        out.append(match.name)
        current = match.type
    return tuple(out)


def _check_planned_read(snapshot: Any, columns: list[str] | None, predicate: str | None) -> None:
    """Raise now for a projection or predicate the snapshot cannot serve."""
    import pyarrow as pa

    from .. import predicate as sqlpred
    from ..errors import InvalidArgumentError

    schema = pa.schema(snapshot.schema())
    if columns:
        known = {name.lower() for name in schema.names}
        missing = [c for c in columns if isinstance(c, str) and c.lower() not in known]
        if missing:
            raise InvalidArgumentError(
                f"column {missing[0]!r} is not in the table schema; columns are {schema.names}"
            )
    if predicate is not None:
        sqlpred.to_arrow(_canonical_node(sqlpred.parse(predicate), schema), schema)


def _canonical_node(node: Any, schema: Any) -> Any:
    """`node` with every column reference spelled as `schema` spells it."""
    import dataclasses

    from .. import predicate as sqlpred

    if isinstance(node, sqlpred.Column):
        return sqlpred.Column(_canonical_path(schema, node.path))
    if isinstance(node, sqlpred.Node):
        return dataclasses.replace(node, args=tuple(_canonical_node(a, schema) for a in node.args))
    return node


def _read_plan(
    snapshot: Any, columns: list[str] | None, predicate: str | None
) -> tuple[Any, list[str] | None, list[str] | None]:
    """`(node, read_columns, keep)` for a projected, filtered read.

    `read_columns` adds what the predicate needs to what was asked for, and is
    never empty: an empty projection panics inside the native reader, so one
    column is read and dropped. `keep` is the final projection, or None when
    the read already is it.
    """
    from .. import predicate as sqlpred

    node = sqlpred.parse(predicate) if predicate is not None else None
    if columns is None and node is None:
        return None, None, None
    import pyarrow as pa

    schema = pa.schema(snapshot.schema())
    if node is not None:
        node = _canonical_node(node, schema)
    if columns is None:
        return node, None, None
    wanted = list(dict.fromkeys(_canonical_path(schema, (c,))[0] for c in columns))
    read = list(wanted)
    if node is not None:
        for path in sorted(sqlpred.columns_of(node)):
            if path[0] not in read:
                read.append(path[0])
    read = _with_data_column(snapshot, read)
    return node, read, (wanted if read != wanted else None)


_DROPS_STATS_REASON = (
    "the table sets delta.checkpoint.writeStatsAsJson=false and leaves "
    "writeStatsAsStruct unset: Spark reads that as struct stats on, this library's "
    "checkpoint writers as off, so the checkpoint would keep no file statistics and "
    "data skipping would stop working for every file it covers, on Databricks too"
)


def _dv_cardinalities(column: Any) -> Any:
    """Rows each file's deletion vector (a descriptor JSON, or null) removes."""
    import pyarrow as pa

    return pa.array(
        [
            0 if dv is None else int(json.loads(dv).get("cardinality") or 0)
            for dv in column.to_pylist()
        ],
        pa.int64(),
    )


#: delta-rs's OPTIMIZE target when neither the call nor delta.targetFileSize sets one.
_DEFAULT_TARGET_SIZE = 104_857_600


#: Native functions the kernel's OPTIMIZE needs: the DV commit with
#: data_change=False, streamed and past the value-constraint features; commit
#: logs for the conflict rules; ordered file-restricted scans.
_COMPACTION_NATIVE = (
    "compaction",
    "commit_log",
    "streaming_compaction",
    "commit_info_patch",
    "file_restricted_scan",
)

#: The OPTIMIZE options the kernel path takes (some only to refuse them).
_COMPACTION_OPTIONS = frozenset(
    {
        "partition_filters",
        "target_size",
        "max_concurrent_tasks",
        "max_spill_size",
        "max_temp_directory_size",
        "min_commit_interval",
        "writer_properties",
        "commit_metadata",
        "max_commit_retries",
        "commit_properties",
        "post_commithook_properties",
    }
)


def _compaction_option_refusal(shape: dict[str, Any]) -> tuple[str, str | None] | None:
    """An OPTIMIZE option the kernel path does not implement, as (reason, remedy).

    Refused rather than handed to delta-rs, which implements them all: its
    OPTIMIZE commit rebases over a concurrent compaction of the same files
    and leaves their rows in the table twice.
    """
    if shape.get("full"):
        return (
            "OPTIMIZE ... FULL reclusters a liquid-clustered table, which the kernel does "
            "not implement",
            SQL_FALLBACK_REMEDY,
        )
    if shape.get("predicate") is not None:
        return (
            "the kernel scopes OPTIMIZE by partition filters, not by a SQL predicate",
            "pass partition_filters=[('col', '=', 'value')] instead",
        )
    if shape.get("writer_properties") is not None:
        return (
            "the kernel's OPTIMIZE writes Parquet with the kernel writer's settings and "
            "does not take writer_properties (delta-rs's OPTIMIZE, which does, duplicates "
            "rows under a concurrent compaction)",
            "drop writer_properties",
        )
    if shape.get("min_commit_interval") is not None:
        return (
            "the kernel's OPTIMIZE commits after every compaction_batch_bytes of input, "
            "not on a timer, and does not take min_commit_interval",
            "drop min_commit_interval",
        )
    if getattr(shape.get("commit_properties"), "app_transactions", None):
        return (
            "an OPTIMIZE commits in steps, and the kernel's records no app transactions on them",
            "drop app_transactions from commit_properties",
        )
    if getattr(shape.get("post_commithook_properties"), "cleanup_expired_logs", None):
        return (
            "the kernel's OPTIMIZE does not clean up expired log files after its commit",
            "drop cleanup_expired_logs, or pass it as False",
        )
    return None


#: Features that constrain only the values a commit writes. A compaction
#: writes back the values it read, so none constrains it; the native commit
#: sets them aside (`compaction_snapshot` in crates/native/src/dml.rs).
_COMPACTION_NEUTRAL: frozenset[TableFeature] = frozenset(
    {
        TableFeature.CHECK_CONSTRAINTS,
        TableFeature.GENERATED_COLUMNS,
        TableFeature.IDENTITY_COLUMNS,
        TableFeature.INVARIANTS,
    }
)


def _compaction_feature_refusal(
    writer_features: Any, properties: dict[str, Any]
) -> tuple[str, str | None] | None:
    """Why the kernel cannot compact a table with these writer features, or None."""
    mode = str(properties.get("delta.columnMapping.mode", "none")).strip().lower()
    mapped = mode not in ("", "none")
    blockers: list[str] = []
    for name in sorted(writer_features):
        feature = feature_from_wire(name)
        if feature is None:
            blockers.append(f"{name} (unrecognized writer feature)")
            continue
        if feature in _COMPACTION_NEUTRAL:
            continue
        if feature is TableFeature.ROW_TRACKING:
            return (
                "the table tracks row ids, and rows a compaction moves to new files would "
                "get new ids: delta-kernel 0.28 cannot carry them",
                SQL_FALLBACK_REMEDY,
            )
        if feature is TableFeature.COLUMN_MAPPING and mapped:
            # Protocol-correct files, but not ones every reader takes: DuckDB's
            # delta reader resolves the Parquet field ids the kernel writes and
            # reads each partition column of such a file as NULL.
            return (
                "the table maps column names (columnMapping), and the files the kernel's "
                "compaction writes there are not yet read correctly by every reader "
                "(DuckDB's delta_scan reads their partition columns as NULL)",
                SQL_FALLBACK_REMEDY,
            )
        if feature is TableFeature.CLUSTERING:
            return (
                "the table is liquid-clustered, where OPTIMIZE reclusters its files; the "
                "kernel only bin-packs and Z-orders",
                SQL_FALLBACK_REMEDY,
            )
        if feature in _METADATA_BLOCKERS or FEATURE_SUPPORT[feature].kernel_write is Support.NO:
            blockers.append(name)
    if blockers:
        return (
            "the kernel cannot compact a table with these features: " + ", ".join(blockers),
            SQL_FALLBACK_REMEDY,
        )
    return None


def _snapshot_compaction_refusal(snapshot: Any) -> str | None:
    """`_compaction_feature_refusal` for the table as `snapshot` has it now."""
    reader, writer, _readers, writers = snapshot.protocol()
    _, implied = implied_features(reader, writer)
    properties = snapshot.table_properties() or {}
    refusal = _compaction_feature_refusal(set(writers or ()) | implied, properties)
    if refusal is not None:
        return refusal[0]
    if properties.get("delta.enableVariantShredding", "").lower() == "true":
        return "the table now shreds VARIANT values, which the kernel cannot decode"
    return None


def _decoded_row_bytes(schema: Any) -> int:
    """A lower estimate of one row's Arrow size: fixed-width columns exactly, others 16 bytes."""
    import pyarrow as pa

    def width(kind: Any) -> int:
        if pa.types.is_struct(kind):
            return sum(width(kind.field(i).type) for i in range(kind.num_fields))
        try:
            return max(1, int(kind.bit_width) // 8)
        except ValueError:
            return 16  # variable width: its offsets, and some of its bytes

    return max(1, sum(width(field.type) for field in schema))


#: Rows a Z-order encodes to learn how its sorted rows compress.
_SIZE_SAMPLE_ROWS = 256 * 1024


#: The table property naming the codec the kernel's writer compresses with
#: (`crates/native/src/writer.rs`, snappy when unset).
_CODEC_PROPERTY = "delta.parquet.compression.codec"


def _pyarrow_codec(name: Any) -> str:
    """pyarrow's name for the codec the kernel writes a table's files with."""
    lowered = str(name or "").strip().lower()
    if lowered in ("uncompressed", "none"):
        return "none"
    if lowered in ("gzip", "zstd", "brotli", "lz4"):
        return lowered
    if lowered in ("lz4_raw", "lz4raw"):
        return "lz4"
    return "snappy"


class _MetadataMoved(Exception):
    """Internal: a concurrent commit changed the metadata a DML commit read."""

    def __init__(self, version: int) -> None:
        super().__init__(version)
        self.version = version


def _encoded_rows_per_file(sample: Any, target: int, fallback: int, codec: str = "snappy") -> int:
    """Rows per `target` bytes of Parquet, judged by encoding `sample` as the kernel writes.

    The kernel's writer compresses pages with the table's codec, dictionary-
    encoded where it helps; pyarrow's with the same settings comes within a
    few percent.
    """
    import io

    import pyarrow.parquet as pq

    if sample.num_rows == 0:
        return fallback
    buffer = io.BytesIO()
    try:
        pq.write_table(sample, buffer, compression=codec)
    except Exception:
        return fallback
    size = buffer.tell()
    return max(1, int(sample.num_rows * target / size)) if size else fallback


def _filter_text(item: Any) -> str:
    """One partition filter as the OPTIMIZE's recorded predicate shows it."""
    column, op, value = item
    return f"{column} {op} {value!r}"


def _one_batch(rows: Any) -> Any:
    """`rows` (a table) as one record batch: the native writer writes a batch per file."""
    import pyarrow as pa

    batches = rows.combine_chunks().to_batches()
    return batches[0] if len(batches) == 1 else pa.concat_batches(batches)


class _Compacted(dict):  # type: ignore[type-arg]
    """An OPTIMIZE's metrics, which say the kernel committed it, whichever engine was asked."""

    served_by = EngineKind.KERNEL


@dataclass(frozen=True)
class _Bin:
    """Files an OPTIMIZE rewrites together: one partition's, up to the target size."""

    key: Any
    paths: list[str]
    sizes: list[int]
    #: Live rows by the files' statistics; None when a file has none.
    expected_rows: int | None

    @classmethod
    def of(cls, key: Any, members: list[tuple[str, int, bool, int | None]]) -> _Bin:
        counts = [m[3] for m in members]
        return cls(
            key,
            [m[0] for m in members],
            [m[1] for m in members],
            None if any(c is None for c in counts) else sum(c for c in counts if c is not None),
        )

    def rows_per_file(self, target: int, seen: int) -> int:
        """Rows per output file: the target size at the input's compression, evened out."""
        rows = self.expected_rows if self.expected_rows is not None else seen
        rows = max(1, rows)
        raw = max(1, int(rows * target / max(1, sum(self.sizes))))
        count = max(1, -(-rows // raw))
        return max(1, -(-rows // count))

    def splits(self, target: int) -> bool:
        """Whether this bin makes more than one output file (by its statistics)."""
        return (
            self.expected_rows is not None
            and self.rows_per_file(target, self.expected_rows) < self.expected_rows
        )


def _size_stats(sizes: list[int]) -> str:
    """delta-rs's filesAdded/filesRemoved metric: size statistics as a JSON string."""
    if not sizes:
        return json.dumps({"avg": 0.0, "max": 0, "min": 0, "totalFiles": 0, "totalSize": 0})
    return json.dumps(
        {
            "avg": sum(sizes) / len(sizes),
            "max": max(sizes),
            "min": min(sizes),
            "totalFiles": len(sizes),
            "totalSize": sum(sizes),
        }
    )


def _physical_partition_names(snapshot: Any) -> dict[str, str]:
    """Each partition column's logical name -> the key `files()` reports its value under."""
    names = {c: c for c in snapshot.partition_columns}
    try:
        schema = json.loads(json.loads(snapshot.metadata_json())["schemaString"])
    except Exception:
        return names
    for field in schema.get("fields", []):
        name = field.get("name")
        if name in names:
            names[name] = (field.get("metadata") or {}).get(
                "delta.columnMapping.physicalName", name
            )
    return names


def _partition_value(raw: Any, kind: Any, column: str) -> Any:
    """A partition value (as the log stores it, or as a filter gives it) as `kind`.

    As delta-rs reads them: an empty string is NULL, a timestamp is ISO 8601
    in either the log's form (``2024-01-01 01:30:15.123456``, UTC) or with a
    ``T`` and an offset, and a value that is not of the column's type is the
    caller's mistake, not a filter that matches nothing.
    """
    import datetime as dt

    import pyarrow as pa
    import pyarrow.compute as pc

    if raw is None or raw == "":
        return None
    text = str(raw).strip()
    try:
        if pa.types.is_timestamp(kind):
            value = dt.datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
            if kind.tz is None:
                # timestamp_ntz: wall-clock values, compared as given.
                return value.replace(tzinfo=None) if value.tzinfo is None else value
            return value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)
        if pa.types.is_date(kind):
            return dt.date.fromisoformat(text)
        return pc.cast(pa.array([text]), kind)[0].as_py()
    except (ValueError, TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise InvalidArgumentError(
            f"partition_filters: {raw!r} is not a {kind} value for partition column {column!r}"
        ) from exc


def _partition_filter(snapshot: Any, filters: list[Any] | None, physical: dict[str, str]) -> Any:
    """A test of a file's partition values against delta-rs-style `(column, op, value)` filters.

    delta-rs's semantics, which are SQL's: values compare as the column's
    type, and a NULL partition value satisfies no comparison -- ``!= 1``,
    ``< 2`` and ``not in`` included. A NULL (or empty-string) filter value
    tests for NULL instead: ``= ''`` / ``= None`` matches the NULL partition,
    ``!= ''`` every other; as an IN-list element it matches nothing, and a
    NOT IN list holding one matches nothing at all.
    """
    if not filters:
        return lambda values: True
    schema = _arrow_schema(snapshot)
    by_lower = {n.lower(): n for n in physical}
    tests = []
    for column, op, value in filters:
        name = by_lower.get(str(column).lower())
        if name is None:
            raise InvalidArgumentError(f"partition_filters name {column!r}, not a partition column")
        kind = schema.field(name).type
        op = str(op).strip().lower()
        if op not in ("=", "==", "!=", "<>", "in", "not in", "<", "<=", ">", ">="):
            raise InvalidArgumentError(f"partition_filters: unknown operator {op!r}")
        if op in ("in", "not in"):
            if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set)):
                raise InvalidArgumentError(
                    f"partition_filters: {op!r} takes a list of values, got {value!r}"
                )
            wanted: Any = [_partition_value(v, kind, name) for v in value]
        else:
            wanted = _partition_value(value, kind, name)
        tests.append((physical[name], op, wanted, kind, name))

    def keep(values: dict[str, Any]) -> bool:
        for key, op, wanted, kind, name in tests:
            try:
                have = _partition_value(values.get(key), kind, name)
            except InvalidArgumentError:
                return False  # a value the log holds that no filter can equal
            if op in ("=", "=="):
                ok = have is None if wanted is None else have == wanted
            elif op in ("!=", "<>"):
                ok = have is not None if wanted is None else have is not None and have != wanted
            elif op == "in":
                ok = have is not None and have in [w for w in wanted if w is not None]
            elif op == "not in":
                ok = have is not None and None not in wanted and have not in wanted
            elif have is None or wanted is None:
                ok = False
            elif op == "<":
                ok = have < wanted
            elif op == "<=":
                ok = have <= wanted
            elif op == ">":
                ok = have > wanted
            else:
                ok = have >= wanted
            if not ok:
                return False
        return True

    return keep


def _zorder_indices(rows: Any, columns: list[str]) -> Any:
    """The order that sorts `rows` along a Z-order curve over `columns`.

    Each column's values become their dense rank (NULLs last), scaled into an
    equal share of 64 bits, and the bits are interleaved; sorting by the
    result clusters rows close in every column, which is what lets each
    output file's min/max statistics skip well on any of them.
    """
    from ..table import _require

    np = _require("numpy", "pyarrow", "Z-ORDER")
    import pyarrow.compute as pc

    by_lower = {n.lower(): n for n in rows.column_names}
    bits = max(1, 64 // len(columns))
    keys = []
    for column in columns:
        name = by_lower.get(column.lower(), column)
        # Ascending ranks put NULLs last by default.
        ranks = pc.rank(rows.column(name), tiebreaker="dense")
        r = np.asarray(ranks.to_numpy(zero_copy_only=False), dtype=np.uint64) - np.uint64(1)
        distinct = int(r.max()) + 1 if len(r) else 1
        if bits < 64 and distinct > (1 << bits):
            r = (r.astype(np.float64) * float(1 << bits) / distinct).astype(np.uint64)
        keys.append(r)
    z = np.zeros(rows.num_rows, dtype=np.uint64)
    for bit in range(bits - 1, -1, -1):
        for r in keys:
            z = (z << np.uint64(1)) | ((r >> np.uint64(bit)) & np.uint64(1))
    return np.argsort(z, kind="stable")


def _arrow_schema(snapshot: Any) -> Any:
    """The snapshot's schema as a pyarrow Schema (skipping needs column types)."""
    import pyarrow as pa

    return pa.schema(snapshot.schema())


def _with_data_column(snapshot: Any, read: list[str]) -> list[str]:
    """`read`, plus one data-file column when it has none.

    The native reader panics on a projection with no data-file columns --
    an empty one, or only partition columns -- so one is read and dropped.
    """
    partitions = set(snapshot.partition_columns)
    if read and not set(read) <= partitions:
        return read
    data = [n for n in snapshot.schema().names if n not in partitions]
    return [*read, data[0]] if data else read


def _project(stream: Any, keep: list[str]) -> Any:
    """`stream` narrowed to `keep`, in that order. `[]` keeps row counts only."""
    import pyarrow as pa

    reader = pa.RecordBatchReader.from_stream(stream)
    schema = pa.schema([reader.schema.field(name) for name in keep])
    return pa.RecordBatchReader.from_batches(schema, (b.select(keep) for b in reader))


def _planned_read(
    snapshot: Any,
    columns: list[str] | None,
    predicate: str | None,
    *,
    files: list[str] | None = None,
) -> Any:
    """Scan `snapshot`: files skipped by the predicate, rows filtered exactly."""
    from .. import predicate as sqlpred

    extra = {} if files is None else {"files": files}
    if predicate is not None:
        _require_pyarrow("filter rows with a predicate on the kernel path")
        sql = _outside_grammar(predicate)
        if sql is not None:
            return _sql_filtered_read(snapshot, columns, sql, extra)
    elif columns is not None and not columns:
        _require_pyarrow("read an empty projection on the kernel path")
    if predicate is None and columns and all(isinstance(c, str) for c in columns):
        names = set(snapshot.schema().names)
        data = names - set(snapshot.partition_columns)
        exact = set(columns) <= names and len(set(columns)) == len(columns)
        if exact and data & set(columns):
            return snapshot.scan(columns=columns, **extra)
    node, read, keep = _read_plan(snapshot, columns, predicate)
    skipping = sqlpred.to_kernel_json(node, _arrow_schema(snapshot)) if node is not None else None
    stream = snapshot.scan(columns=read, predicate=skipping, **extra)
    if node is not None:
        stream = sqlpred.filter_stream(stream, node)
    return stream if keep is None else _project(stream, keep)


def _outside_grammar(predicate: str) -> str | None:
    """`predicate` respelled for DuckDB when the kernel's grammar does not read it.

    None when the grammar does (the usual path), or when DuckDB is not
    installed (the grammar's own PredicateError then stands). The router sends
    such a predicate here only for a table delta-rs misreads.
    """
    try:
        sqlpred.parse(predicate)
    except sqlpred.PredicateError as exc:
        if not exc.beyond_grammar or importlib.util.find_spec("duckdb") is None:
            raise
    else:
        return None
    from . import sharing
    from .dialect import to_duckdb

    # Spark SQL DuckDB cannot be made to evaluate as Spark does is refused
    # (EngineLimitError), rather than filtered by another meaning.
    text = to_duckdb(predicate)
    sharing._screen_expression(text)
    return text


def _sql_filtered_read(
    snapshot: Any, columns: list[str] | None, text: str, extra: dict[str, Any]
) -> Any:
    """Every row of `snapshot` (no file skipping), kept where DuckDB finds `text` true.

    The columns the expression reads are not known without a parse, so the
    whole row is read and the projection applied after the filter.
    """
    import pyarrow as pa

    reader = pa.RecordBatchReader.from_stream(snapshot.scan(**extra))
    keep = None
    if columns is not None:
        keep = list(dict.fromkeys(_canonical_path(reader.schema, (c,))[0] for c in columns))
        unknown = [c for c in keep if c not in reader.schema.names]
        if unknown:
            from ..errors import InvalidArgumentError

            raise InvalidArgumentError(
                f"column {unknown[0]!r} is not in the table schema; columns are "
                f"{reader.schema.names}"
            )
    from .duckfilter import RowFilter

    return RowFilter(text, reader.schema, spark=True).filtered(reader, keep)


_REJECTED_CREDENTIAL_MARKERS = (
    "expiredtoken",
    "token has expired",
    "invalidaccesskeyid",
    "signaturedoesnotmatch",
    "authenticationfailed",
    "invalidauthenticationinfo",
    "accessdenied",
    "access denied",
    "forbidden",
    "403",
)


def _rejected_credential(message: str) -> bool:
    """Whether a storage error reads as a credential the store refused."""
    lowered = message.lower()
    return any(marker in lowered for marker in _REJECTED_CREDENTIAL_MARKERS)


def _feature_usage(snapshot: Any) -> dict[str, bool]:
    """Which version-implied features the table actually uses.

    A legacy writer version implies a whole set whether or not any is used:
    version 2 implies `invariants`, version 4 implies `checkConstraints` and
    `generatedColumns`. Enabling change data feed alone puts a table at version
    4, so refusing on the implied name would push every CDF table off the kernel
    write path. These look at the schema and configuration instead.
    """
    usage = {
        "has_invariants": False,
        "has_check_constraints": False,
        "has_generated_columns": False,
        "has_binary_partitions": False,
        "has_datetime_columns": True,
    }
    properties = snapshot.table_properties() or {}
    usage["has_check_constraints"] = any(
        key.lower().startswith("delta.constraints.") for key in properties
    )

    metadata = json.loads(snapshot.metadata_json())
    schema = metadata.get("schemaString") or metadata.get("schema_string")
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except ValueError:
            return usage
    if not isinstance(schema, dict):
        return usage
    from .calendar import has_datetime_columns

    usage["has_datetime_columns"] = has_datetime_columns(schema)
    found = _field_metadata_keys(schema.get("fields") or [])
    usage["has_invariants"] = "delta.invariants" in found
    usage["has_generated_columns"] = "delta.generationExpression" in found
    columns = metadata.get("partitionColumns") or metadata.get("partition_columns") or []
    partitions = {str(c).lower() for c in columns}
    usage["has_binary_partitions"] = any(
        isinstance(f, dict)
        and str(f.get("name", "")).lower() in partitions
        and f.get("type") == "binary"
        for f in schema.get("fields") or []
    )
    return usage


def _field_metadata_keys(fields: Any) -> set[str]:
    """Every field-metadata key in a Delta schema, nested types included."""
    keys: set[str] = set()
    if isinstance(fields, dict):
        fields = [fields]
    for field in fields or []:
        if not isinstance(field, dict):
            continue
        keys.update((field.get("metadata") or {}).keys())
        dtype = field.get("type")
        if isinstance(dtype, dict):
            keys |= _field_metadata_keys(dtype.get("fields") or [])
            for nested in ("elementType", "valueType", "keyType"):
                inner = dtype.get(nested)
                if isinstance(inner, dict):
                    keys |= _field_metadata_keys(inner.get("fields") or [])
    return keys


def _delta_fields(fields: Any) -> list[dict[str, Any]]:
    """Normalise what `add_columns` accepts into Delta schema field dicts.

    Takes a pyarrow Schema or Field (or a list of them), delta-rs `Field`
    objects, or a `{name: delta_type}` mapping of primitive types.
    """
    if isinstance(fields, dict):
        # `str(dtype)` used to commit whatever it produced -- `int64` for
        # pa.int64(), `bigint` as typed -- and a schema holding a type Delta
        # does not know makes the whole table unreadable.
        return [
            {"name": name, "type": _delta_type(name, dtype), "nullable": True, "metadata": {}}
            for name, dtype in fields.items()
        ]
    if not isinstance(fields, (list, tuple)) and isinstance(getattr(fields, "fields", None), list):
        fields = fields.fields  # a deltalake Schema: its fields, not itself as one
    items = (
        list(fields) if isinstance(fields, (list, tuple)) or hasattr(fields, "names") else [fields]
    )
    out: list[dict[str, Any]] = []
    for item in items:
        to_json = getattr(item, "to_json", None)
        if callable(to_json):
            out.append(json.loads(to_json()))
        elif hasattr(item, "type") and hasattr(item, "nullable"):
            out.append(arrow_to_delta_field(item))
        else:
            raise InvalidArgumentError(
                f"cannot add columns: cannot interpret {type(item).__name__} as a column "
                "definition; pass pyarrow fields, deltalake Fields, or a {name: type} mapping"
            )
    return out


def _delta_type(name: str, dtype: Any) -> Any:
    """A Delta schema type from a `{name: type}` value, or a refusal."""
    if isinstance(dtype, dict):
        return dtype  # already Delta JSON (struct, array, map)
    if not isinstance(dtype, str):
        from .metadata import arrow_to_delta_type

        return arrow_to_delta_type(dtype)
    try:
        return meta.sql_type_to_delta(dtype)
    except ValueError as exc:
        raise InvalidArgumentError(
            f"cannot add column {name}: {dtype!r} is not a Delta type ({exc}); use a Delta or "
            "SQL type (long, bigint, string, decimal(p,s), array<int>, ...), a pyarrow type, "
            "or a Delta JSON type"
        ) from exc


def _require_pyarrow(what: str) -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise UnreachableTableError(
            what,
            "pyarrow is needed to evaluate the row filter",
            "pip install 'deltaswamp[pyarrow]'",
        ) from exc


def _evaluate(table: Any, expr: Any) -> Any:
    """Evaluate a boolean compute expression over a table, as one array."""
    import pyarrow as pa
    import pyarrow.dataset as ds

    if table.num_rows == 0:
        return pa.array([], pa.bool_())
    result = ds.dataset(table).to_table(columns={"_m": expr}).column("_m")
    return result.combine_chunks() if isinstance(result, pa.ChunkedArray) else result


def _conform(data: Any, schema: Any) -> Any:
    """`data` with `schema`'s columns, names and types, for a rewrite commit.

    Names match case-insensitively; a missing nullable column is null. An
    extra column, or a missing non-nullable one, is refused: selecting only
    the table's columns used to drop the extra data without a word.
    """
    import pyarrow as pa

    by_lower = {name.lower(): name for name in data.column_names}
    extra = sorted(set(by_lower) - {f.name.lower() for f in schema})
    if extra:
        raise UnreachableTableError(
            "overwrite with a predicate",
            f"the data has columns the table does not: {', '.join(extra)}",
            "drop them, or add them to the table first",
        )
    columns = []
    for field in schema:
        source = by_lower.get(field.name.lower())
        if source is None:
            if not field.nullable:
                raise UnreachableTableError(
                    "overwrite with a predicate",
                    f"the data has no {field.name!r} column, which the table requires",
                )
            columns.append(pa.nulls(data.num_rows, field.type))
        else:
            columns.append(data.column(source).cast(field.type))
    return pa.Table.from_arrays(columns, schema=schema)


def _refuse_options(what: str, options: dict[str, Any]) -> None:
    given = sorted(k for k, v in options.items() if v is not None)
    if given:
        raise UnreachableTableError(
            f"{what} with {', '.join(given)}",
            "the kernel rewrite path does not implement these options",
        )
