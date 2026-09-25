"""The delta-kernel engine: the default read path.

The kernel checks feature support per operation and ignores writer-only
features when reading, so it opens tables delta-rs refuses: every
`catalogManaged` table, and anything with type widening, shredded variants or
`vacuumProtocolCheck`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import re
from collections.abc import Iterator
from typing import Any

from .. import predicate as sqlpred
from .._sdk import PRODUCT, sdk_version
from .._util import timestamp_ms
from ..capability import (
    FEATURE_DEPENDENCIES,
    FEATURE_SUPPORT,
    METADATA_OPERATIONS,
    Capability,
    Operation,
    Support,
    TableFeature,
    feature_from_wire,
)
from ..capability import (
    Engine as EngineKind,
)
from ..catalog import ResolvedTable
from ..credentials import Operation as CredentialOperation
from ..errors import (
    SQL_FALLBACK_REMEDY,
    BackfillRequiredError,
    CommitConflictError,
    DeltaSwampError,
    EngineLimitError,
    TransientCommitError,
    UnreachableTableError,
)
from ..properties import effect_for
from . import metadata as meta
from .base import missing_method
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

        if operation in METADATA_OPERATIONS:
            refusal = self._metadata_refusal(operation, table)
            if refusal is not None:
                return refusal

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
    ) -> Any:
        """Resolve a kernel snapshot, supplying the catalog tail when needed."""
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

        def resolve() -> Any:
            return Snapshot.resolve(
                location,
                options=self._options(table, write=write),
                version=version,
                log_tail=log_tail,
                max_catalog_version=table.max_catalog_version,
                timestamp_ms=timestamp_ms(timestamp) if timestamp is not None else None,
            )

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

            return pa.table(stream)
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
                    raise UnreachableTableError(
                        "read the change data feed",
                        f"the change data feed was not enabled at version {off.group(1)}, "
                        "which the requested range includes",
                        "start the range after the feed was enabled",
                    ) from exc
                if "Start and end version schemas are different" in message:
                    # Turning the feed off is a metadata change too, which the
                    # kernel reports as a schema change. Name the real cause.
                    gap = self._feed_gap(table, start or 0, ending_version)
                    if gap is not None:
                        raise UnreachableTableError(
                            "read the change data feed",
                            f"the change data feed was not enabled at version {gap}, "
                            "which the requested range includes",
                            "start the range after the feed was re-enabled",
                        ) from exc
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
        with _library_commit_errors():
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
                reader = _as_record_batch_reader(data)
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
                    )
                except _native.CommitConflictError:
                    if attempt + 1 >= attempts or self._txn_won_race(snapshot, table, txn):
                        # A concurrent commit that recorded this txn (or a later
                        # one) already wrote this batch: re-staging would append it twice.
                        raise
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
    append_commit_retries = 5

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
                    raise UnreachableTableError(
                        f"overwrite where {predicate}",
                        f"{bad} row(s) of the new data do not satisfy the predicate, so "
                        "writing them would add rows outside the range being replaced",
                        "filter the data to the predicate first",
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
        current = pa.table(snapshot.scan())
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
        with _library_commit_errors():
            version = snapshot.append(
                replacement.to_reader(),
                uc=self._uc_commit_config(table),
                engine_info=engine_info or _engine_info(),
                operation=operation,
                overwrite=True,
                txn=txn,
                commit_metadata={k: str(v) for k, v in (commit_metadata or {}).items()} or None,
            )
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
                    raise UnreachableTableError(
                        f"overwrite where {predicate}",
                        f"{bad} row(s) of the new data do not satisfy the predicate, so "
                        "writing them would add rows outside the range being replaced",
                        "filter the data to the predicate first",
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
    ) -> int:
        """Commit `deletions` (path, row_index) as vectors plus `data`, in one transaction.

        A touched file with no `numRecords` statistic cannot take a vector (its
        cardinality is tied to that count), so it is rewritten instead, as
        copy-on-write does: its surviving rows join `data` and the file is
        removed in the same commit.
        """
        deletions, data, whole_files = _rewrite_unsized_files(
            table, snapshot, deletions, data, list(whole_files or [])
        )
        with _library_commit_errors():
            version, _deleted, _dvs, _removed = snapshot.commit_dml(
                deletions.to_reader(),
                data=data.to_reader() if data is not None else None,
                whole_files=whole_files or None,
                uc=self._uc_commit_config(table),
                engine_info=engine_info or _engine_info(),
                operation=operation,
                txn=txn,
                commit_metadata={k: str(v) for k, v in (commit_metadata or {}).items()} or None,
            )
        if int(version) != int(snapshot.version):
            self._maybe_checkpoint(table, version, snapshot)
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
        self, table: ResolvedTable, predicate: str | None = None, **unsupported: Any
    ) -> dict[str, Any]:
        """DELETE by rewriting the table without the matching rows.

        SQL semantics: a row is deleted only where the predicate is TRUE; a
        NULL result keeps it.
        """
        _refuse_options("delete", unsupported)
        if self._dv_path(table):
            result = self._dv_dml(table, predicate, operation="DELETE")
        else:
            result = self._rewrite(
                table, predicate, lambda current, keep: current.filter(keep), operation="DELETE"
            )
        return {"num_deleted_rows": result["num_affected_rows"], "version": result["version"]}

    def update(
        self,
        table: ResolvedTable,
        *,
        updates: dict[str, str] | None = None,
        new_values: dict[str, Any] | None = None,
        predicate: str | None = None,
        **unsupported: Any,
    ) -> dict[str, Any]:
        """UPDATE by rewriting the table. Assignments are plain values.

        `updates` (SQL expressions) are accepted only when each is a literal or
        a column reference, since nothing here evaluates arbitrary SQL.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        _refuse_options("update", unsupported)
        assignments: dict[str, Any] = dict(new_values or {})
        for column, expression in (updates or {}).items():
            value = sqlpred.parse_value(expression)
            assignments[column] = value
        if not assignments:
            raise UnreachableTableError("update", "no assignments given")

        def column_index(schema: Any, name: str, what: str) -> int:
            # Delta column names are case-insensitive.
            index = int(schema.get_field_index(_canonical_path(schema, (name,))[0]))
            if index < 0:
                raise UnreachableTableError(what, f"the table has no column {name!r}")
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
                    source = current.column(source_index).cast(field.type)
                else:
                    raw = value.value if isinstance(value, sqlpred.Literal) else value
                    source = pa.array([raw] * out.num_rows).cast(field.type)
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
            )
        else:
            result = self._rewrite(table, predicate, assign, operation="UPDATE")
        return {"num_updated_rows": result["num_affected_rows"], "version": result["version"]}

    def publish(self, table: ResolvedTable) -> int:
        """Publish ratified-but-unpublished commits into the Delta log."""
        snapshot = self.snapshot(table, write=True)
        with _library_commit_errors():
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
            return version
        # A lost race is a conflict, not an unreachable table: callers that
        # catch CommitConflictError to retry never saw this one.
        raise CommitConflictError(
            _conflict_version(str(last_error)),
            f"cannot commit a metadata change: another writer committed first on each "
            f"of {self.metadata_commit_attempts} attempts ({last_error}); retry when the "
            "table is less busy",
        )

    #: Warn when a scan starts with less than this much credential life left.
    #: A long read that outlives its credential fails partway through, and the
    #: storage layer reports only a 403.
    expiry_warning_seconds: float = 300.0

    def _options(self, table: ResolvedTable, *, write: bool) -> dict[str, str]:
        options = dict(self._base_options)
        if table.credential_provider is not None:
            op = CredentialOperation.READ_WRITE if write else CredentialOperation.READ
            credentials = table.credential_provider.credentials(op)
            self._warn_if_short_lived(credentials)
            options.update(credentials.as_storage_options())
        return options

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

    def add_feature(self, table: ResolvedTable, feature: Any, **_: Any) -> int:
        names = feature if isinstance(feature, (list, tuple, set, frozenset)) else [feature]
        wires = {str(getattr(n, "value", n)) for n in names}
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
            props = {f"delta.feature.{n}": "supported" for n in sorted(wires)}
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
        import pyarrow as pa

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

    def write_files(self, table: ResolvedTable, data: Any) -> bytes:
        """Write data files for `table` without committing them.

        Returns opaque fragment bytes describing what was written. The files
        exist and are durable once this returns; they belong to no version
        until `commit_files` accepts them, so a coordinator that abandons the
        write leaves them behind.
        """
        if not self.supports_distributed_write:
            raise NotImplementedError(
                "the installed native extension cannot write files without committing"
            )
        snapshot = self.snapshot(table, write=True)
        result: bytes = snapshot.write_files(
            _as_record_batch_reader(data), uc=self._uc_commit_config(table, staging=True)
        )
        return result

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
    ) -> int:
        """Commit fragments from `write_files` as one transaction.

        Every fragment lands at a single version, so a distributed write is
        atomic: a reader sees all of it or none of it. With `version`, the
        commit is built on that snapshot, so it conflicts if the table has
        moved past it -- what a guarded overwrite needs.
        """
        if not self.supports_distributed_write:
            raise NotImplementedError(
                "the installed native extension cannot commit externally written files"
            )
        snapshot = self.snapshot(table, version=version, write=True)
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
        try:
            with _library_commit_errors():
                committed: int = snapshot.commit_files(
                    list(fragments),
                    uc=self._uc_commit_config(table),
                    engine_info=engine_info or _engine_info(),
                    operation=operation,
                    overwrite=overwrite,
                    txn=txn,
                    commit_metadata={k: str(v) for k, v in (commit_metadata or {}).items()} or None,
                )
        except ValueError as exc:
            # A refused fragment is the caller's input, and reached callers as
            # a bare ValueError that `except DeltaSwampError` did not catch.
            from ..errors import DeltaSwampError, InvalidArgumentError

            if isinstance(exc, DeltaSwampError) or "fragment" not in str(exc):
                raise
            raise InvalidArgumentError(str(exc)) from exc
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


def _rewrite_unsized_files(
    table: ResolvedTable, snapshot: Any, deletions: Any, data: Any, whole_files: list[str]
) -> tuple[Any, Any, list[str]]:
    """Move deletions from files without `numRecords` into a rewrite.

    Returns the deletions left for vectors, the data with each such file's
    surviving rows added, and the files to remove whole. Row tracking forbids
    the remove, so there the native commit refuses with the reason instead.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if deletions.num_rows == 0 or _row_tracking_enabled(table):
        return deletions, data, whole_files
    files = pa.table(snapshot.files())
    unsized = set(files.filter(pc.is_null(files.column("num_records"))).column("path").to_pylist())
    touched = set(pc.unique(deletions.column("path")).to_pylist())
    rewrite = sorted(unsized & touched)
    if not rewrite:
        return deletions, data, whole_files
    in_rewrite = pc.is_in(deletions.column("path"), pa.array(rewrite, pa.string()))
    gone = deletions.filter(in_rewrite)
    rest = pa.table(snapshot.scan(files=rewrite, row_positions=True))
    keys = pc.binary_join_element_wise(
        rest.column(_FILE_COLUMN), pc.cast(rest.column(_ROW_INDEX_COLUMN), pa.string()), "\x00"
    )
    gone_keys = pc.binary_join_element_wise(
        gone.column("path"), pc.cast(gone.column("row_index"), pa.string()), "\x00"
    )
    survivors = rest.filter(pc.invert(pc.is_in(keys, gone_keys))).drop_columns(
        [_FILE_COLUMN, _ROW_INDEX_COLUMN]
    )
    if data is None:
        data = survivors
    elif survivors.num_rows:
        data = pa.concat_tables([data, survivors.cast(data.schema)])
    return deletions.filter(pc.invert(in_rewrite)), data, whole_files + rewrite


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


@contextlib.contextmanager
def _library_commit_errors() -> Iterator[None]:
    """Raise this library's error types instead of the extension's.

    `errors.py` defines `CommitConflictError` and `BackfillRequiredError` so a
    caller can catch `DeltaSwampError` and tell a lost race from backpressure.
    The native ones are plain `RuntimeError`s, and they were reaching callers
    untranslated -- so `except DeltaSwampError` around a commit caught nothing,
    which is precisely the case it exists for.
    """
    from deltaswamp import _native

    try:
        yield
    except _native.BackfillRequiredError as exc:
        raise BackfillRequiredError(str(exc)) from exc
    except _native.CommitConflictError as exc:
        raise CommitConflictError(_conflict_version(str(exc)), str(exc)) from exc
    except _native.RetryableError as exc:
        raise TransientCommitError(str(exc)) from exc
    except ValueError as exc:
        if isinstance(exc, DeltaSwampError):
            raise
        translated = _uc_commit_http_error(str(exc))
        if translated is None and any(m in str(exc) for m in _BAD_DATA_MARKERS):
            # The data does not fit the table: the caller's input, reported as
            # a bare ValueError that `except DeltaSwampError` did not catch.
            # InvalidArgumentError is still a ValueError.
            from ..errors import InvalidArgumentError

            translated = InvalidArgumentError(str(exc))
        if translated is None and isinstance(exc, getattr(_native, "CatalogCommitError", ())):
            from ..errors import CredentialError, InvalidReferenceError, PreflightError

            # The extension types what the UC client words without a status
            # (a 401 is "Authentication failed"); anything else -- a 400, say
            # -- is still the catalog's refusal, with nothing committed.
            if isinstance(exc, getattr(_native, "CatalogNotFoundError", ())):
                translated = InvalidReferenceError(
                    "the catalog no longer has this table: it was dropped (or renamed) "
                    f"after it was opened, and the commit was refused. Re-resolve it. ({exc})"
                )
            elif isinstance(exc, getattr(_native, "CatalogPermissionError", ())):
                if "authentication failed" in str(exc).lower() or "401" in str(exc):
                    translated = CredentialError(
                        "the catalog rejected the commit's credentials (expired or invalid "
                        f"token); nothing was committed. ({exc})"
                    )
                else:
                    translated = PreflightError(
                        "the catalog refused the commit: the principal may not modify this "
                        f"table; nothing was committed. ({exc})"
                    )
            else:
                translated = UnreachableTableError(
                    "commit to the catalog", f"the catalog refused the commit ({exc})"
                )
        if translated is None:
            raise
        raise translated from exc


#: Messages the extension raises (as ValueError) for data that does not fit.
_BAD_DATA_MARKERS = (
    "that are not in the table schema",
    "cannot be written as the table's type",
)


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


def _uc_commit_http_error(message: str) -> Exception | None:
    """A catalog refusal of a commit, as this library's error type.

    The extension classifies only 409 and 429; any other status from the UC
    commit API (the table dropped under the writer, an expired token, a lost
    privilege) reached callers as a bare ValueError that `except
    DeltaSwampError` did not catch, after the data files were written.
    """
    import re

    from ..errors import CredentialError, InvalidReferenceError, PreflightError

    if "UC update_table error" not in message:
        return None
    found = re.search(r"status\D{0,3}(\d{3})", message)
    status = int(found.group(1)) if found else None
    if status == 404:
        return InvalidReferenceError(
            "the catalog no longer has this table: it was dropped (or renamed) after it was "
            f"opened, and the commit was refused. Re-resolve it. ({message})"
        )
    if status == 401:
        return CredentialError(
            "the catalog rejected the commit's credentials (expired or invalid token); "
            f"nothing was committed. ({message})"
        )
    if status == 403:
        return PreflightError(
            "the catalog refused the commit: the principal may not modify this table "
            f"(MODIFY, plus USE SCHEMA and USE CATALOG); nothing was committed. ({message})"
        )
    # A 5xx is left alone: the catalog may have ratified the commit before
    # failing, so "unchanged, retry" (TransientCommitError) would be a guess.
    return None


def _conflict_version(message: str) -> int:
    """The version someone else won, if the message names one; -1 otherwise."""
    import re

    found = re.search(r"version (\d+)", message)
    return int(found.group(1)) if found else -1


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
    found = _field_metadata_keys(schema.get("fields") or [])
    usage["has_invariants"] = "delta.invariants" in found
    usage["has_generated_columns"] = "delta.generationExpression" in found
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
            raise UnreachableTableError(
                "add columns",
                f"cannot interpret {type(item).__name__} as a column definition",
                "pass pyarrow fields, deltalake Fields, or a {name: type} mapping",
            )
    return out


_DELTA_PRIMITIVES = frozenset(
    {
        "string",
        "long",
        "integer",
        "short",
        "byte",
        "float",
        "double",
        "boolean",
        "binary",
        "date",
        "timestamp",
        "timestamp_ntz",
    }
)
_SQL_ALIASES = {
    "bigint": "long",
    "int": "integer",
    "smallint": "short",
    "tinyint": "byte",
    "bool": "boolean",
    "real": "float",
    "varchar": "string",
    "char": "string",
    "text": "string",
    "int64": "long",
    "int32": "integer",
    "int16": "short",
    "int8": "byte",
    "float32": "float",
    "float64": "double",
    "utf8": "string",
    "large_string": "string",
    "timestampntz": "timestamp_ntz",
    "date32": "date",
}


def _delta_type(name: str, dtype: Any) -> Any:
    """A Delta schema type from a `{name: type}` value, or a refusal."""
    import re

    if isinstance(dtype, dict):
        return dtype  # already Delta JSON (struct, array, map)
    if not isinstance(dtype, str):
        from .metadata import arrow_to_delta_type

        return arrow_to_delta_type(dtype)
    text = dtype.strip().lower()
    text = _SQL_ALIASES.get(text, text)
    decimal = re.fullmatch(r"(?:decimal|numeric)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if decimal:
        return f"decimal({int(decimal.group(1))},{int(decimal.group(2))})"
    if text in _DELTA_PRIMITIVES:
        return text
    raise UnreachableTableError(
        f"add column {name}",
        f"{dtype!r} is not a Delta type",
        "use a Delta primitive (long, integer, string, double, decimal(p,s), ...), "
        "a pyarrow type, or a Delta JSON type",
    )


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
