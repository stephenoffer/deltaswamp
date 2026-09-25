"""The delta-kernel engine: the default read path.

Kernel earns that position because it evaluates feature support *per operation*
and ignores writer-only features when reading. delta-rs does a flat
set-difference against a hardcoded list and refuses to open 19 features kernel
reads without complaint -- including every `catalogManaged` table and, because
it is a ReaderWriter feature, anything carrying `vacuumProtocolCheck`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from ..capability import (
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
    BackfillRequiredError,
    CommitConflictError,
    TransientCommitError,
    UnreachableTableError,
)
from ..properties import effect_for
from .base import missing_method

__all__ = ["KernelEngine"]


def _version() -> str:
    from .. import __version__

    return __version__


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
    }
)

#: Served by rewriting the whole table in one commit: correct on any table the
#: kernel can write, and bounded by `KernelEngine.rewrite_max_bytes`.
_REWRITE_OPS: frozenset[Operation] = frozenset(
    {Operation.REPLACE_WHERE, Operation.DELETE, Operation.UPDATE}
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
    }
    | _REWRITE_OPS
)


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
    }
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
    #: history_manager resolves a timestamp to the latest recreatable version,
    #: honouring in-commit timestamps.
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
                blockers.append(f"{name} (unrecognised reader feature)")
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

        if operation in _REWRITE_OPS:
            refusal = self._rewrite_refusal(operation, table)
            if refusal is not None:
                return refusal

        if operation in _WRITE_OPS or operation in METADATA_OPERATIONS:
            write_blockers: list[str] = []
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
                elif FEATURE_SUPPORT[feature].kernel_write is Support.NO:
                    write_blockers.append(name)
            if write_blockers:
                return Capability(
                    operation,
                    ok=False,
                    reason=(
                        "the kernel cannot write these table features: "
                        + ", ".join(sorted(write_blockers))
                    ),
                )
            if (
                table.partition_columns
                and operation in (Operation.APPEND, Operation.OVERWRITE)
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

        if operation is Operation.SCAN and table.is_shallow_clone:
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

        if table.location is None:
            raise UnreachableTableError("open", "the table has no storage location", None)
        if version is not None and timestamp is not None:
            raise UnreachableTableError("time travel", "pass a version or a timestamp, not both")

        # `log_tail` and `max_catalog_version` are what make a catalog-managed
        # table readable; both are meaningless (and omitted) otherwise.
        log_tail = [
            (entry.version, entry.path, entry.timestamp or 0, entry.size)
            for entry in table.log_tail
        ] or None

        try:
            return Snapshot.resolve(
                table.location,
                options=self._options(table, write=write),
                version=version,
                log_tail=log_tail,
                max_catalog_version=table.max_catalog_version,
                timestamp_ms=_timestamp_ms(timestamp) if timestamp is not None else None,
            )
        except ValueError as exc:
            if "earliest recreatable" in str(exc):
                raise UnreachableTableError(f"read the table as of {timestamp}", str(exc)) from exc
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
        snapshot = self.snapshot(table, version=version, timestamp=timestamp)
        if predicate is None:
            return snapshot.scan(columns=columns)

        from .. import predicate as sqlpred

        node = sqlpred.parse(predicate)
        read_columns = columns
        if columns is not None:
            needed = {path[0] for path in sqlpred.columns_of(node)}
            lowered = {c.lower() for c in columns}
            read_columns = list(columns) + sorted(c for c in needed if c.lower() not in lowered)
        stream = snapshot.scan(columns=read_columns, predicate=sqlpred.to_kernel_json(node))
        _require_pyarrow("filter rows with a predicate on the kernel path")
        return sqlpred.filter_stream(
            stream, node, keep=list(columns) if columns is not None else None
        )

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
        from .. import predicate as sqlpred

        skipping = sqlpred.to_kernel_json(sqlpred.parse(predicate)) if predicate else None
        return self.snapshot(table, version=version).files(predicate=skipping)

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

        given = {k: v for k, v in unsupported.items() if v not in (None, False)}
        if given:
            raise UnreachableTableError(
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
        if table.properties.get("delta.enableChangeDataFeed", "false").lower() != "true":
            raise UnreachableTableError(
                "read the change data feed",
                "delta.enableChangeDataFeed is not enabled on this table, and enabling "
                "it is not retroactive -- only changes after enablement are recorded",
            )
        if starting_version is not None and starting_timestamp is not None:
            raise UnreachableTableError(
                "read the change data feed", "pass a starting version or timestamp, not both"
            )
        from .. import predicate as sqlpred

        node = sqlpred.parse(predicate) if predicate else None
        read_columns = columns
        if columns is not None and node is not None:
            needed = {path[0] for path in sqlpred.columns_of(node)}
            read_columns = list(columns) + sorted(needed - set(columns))
        assert table.location is not None  # supports() refused otherwise
        stream = _native.table_changes(
            table.location,
            options=self._options(table, write=False) or None,
            start_version=starting_version,
            end_version=ending_version,
            columns=read_columns,
            predicate=sqlpred.to_kernel_json(node) if node is not None else None,
            start_timestamp_ms=(
                _timestamp_ms(starting_timestamp) if starting_timestamp is not None else None
            ),
            end_timestamp_ms=(
                _timestamp_ms(ending_timestamp) if ending_timestamp is not None else None
            ),
        )
        if node is None:
            return stream
        keep = None
        if columns is not None:
            keep = [*columns, "_change_type", "_commit_version", "_commit_timestamp"]
        return sqlpred.filter_stream(stream, node, keep=keep)

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
        }

    # ------------------------------------------------------------------ write

    def _uc_commit_config(self, table: ResolvedTable) -> Any:
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
        # A bare **_ here used to swallow schema_mode, writer_properties and the
        # rest, so they silently did nothing on a catalog-managed table.
        given = {k: v for k, v in unsupported.items() if v is not None}
        if given:
            raise UnreachableTableError(
                f"append with {', '.join(sorted(given))}",
                "the kernel write path does not implement these options",
                "the router sends these to delta-rs when the table allows it; a "
                "catalog-managed table needs the SQL fallback",
            )
        reader = _as_record_batch_reader(data)
        snapshot = self.snapshot(table, write=True)

        with _library_commit_errors():
            version: int = snapshot.append(
                reader,
                uc=self._uc_commit_config(table),
                engine_info=engine_info or f"deltaswamp/{_version()}",
                operation=operation,
                overwrite=overwrite,
                txn=txn,
                commit_metadata={k: str(v) for k, v in (commit_metadata or {}).items()} or None,
            )
        self._maybe_checkpoint(table, version)
        return version

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
        **_ignored: Any,
    ) -> int:
        """Create a table and commit version 0.

        The kernel accepts nearly the whole property surface, including the
        `delta.feature.*` signals and custom keys delta-rs refuses. Clustering
        is not a property here: it goes through the data layout.
        """
        from deltaswamp._native import create_table

        if table.location is None:
            raise UnreachableTableError("create", "no storage location was given for the new table")
        if mode not in ("error", "create"):
            raise UnreachableTableError(
                f"create with mode={mode!r}",
                "the kernel create path writes version 0 of a new table only",
                "use mode='error', or write through delta-rs for overwrite semantics",
            )

        version: int = create_table(
            table.location,
            schema,
            options=self._base_options or None,
            properties=properties or None,
            partition_by=partition_by or None,
            cluster_by=cluster_by or None,
            uc=self._uc_commit_config(table),
            engine_info=engine_info or f"deltaswamp/{_version()}",
        )
        return version

    #: Fallback when the table sets no interval. Matches Delta's own default.
    default_checkpoint_interval = 10

    def checkpoint_interval(self, table: ResolvedTable) -> int:
        raw = table.properties.get("delta.checkpointInterval")
        try:
            interval = int(raw) if raw else self.default_checkpoint_interval
        except ValueError:
            interval = self.default_checkpoint_interval
        return max(interval, 1)

    def _maybe_checkpoint(self, table: ResolvedTable, version: int) -> None:
        """Checkpoint after a commit at the table's checkpoint interval.

        Kernel commits never checkpoint on their own. A catalog-managed table is
        checkpointed here too (after publishing), since nothing else outside
        Databricks can; without it the log grows and opening slows.
        """
        if version == 0 or version % self.checkpoint_interval(table) != 0:
            return
        try:
            self.checkpoint(table)
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

            incoming = pa.table(_as_record_batch_reader(data))

            def replace(current: Any, keep: Any) -> Any:
                kept = current.filter(keep)
                return pa.concat_tables(
                    [kept, incoming.select(kept.column_names).cast(kept.schema)]
                )

            return int(self._rewrite(table, predicate, replace, operation="WRITE")["version"])
        return self.append(table, data, operation="WRITE", overwrite=True, **kwargs)

    # ------------------------------------------------ copy-on-write rewrites

    #: The largest table (by live data-file bytes) a copy-on-write rewrite will
    #: take on. The rewrite holds the table in memory, so past this size the
    #: warehouse, or delta-rs on a table it can open, is the right tool.
    rewrite_max_bytes = 1 << 30

    def _rewrite_refusal(self, operation: Operation, table: ResolvedTable) -> Capability | None:
        """Whether a whole-table rewrite may serve DELETE/UPDATE/replaceWhere."""
        if "rowTracking" in table.writer_features:
            return Capability(
                operation,
                ok=False,
                reason="the table tracks row ids, and a rewrite through the kernel would have "
                "to preserve them, which delta-kernel 0.28 cannot do",
            )
        if table.location is None:
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

        from .. import predicate as sqlpred

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
            expr = sqlpred.to_arrow(sqlpred.parse(predicate), current.schema)
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
                engine_info=f"deltaswamp/{_version()}",
                operation=operation,
                overwrite=True,
            )
        self._maybe_checkpoint(table, version)
        return {"version": int(version), "num_affected_rows": touched}

    def delete(
        self, table: ResolvedTable, predicate: str | None = None, **unsupported: Any
    ) -> dict[str, Any]:
        """DELETE by rewriting the table without the matching rows.

        SQL semantics: a row is deleted only where the predicate is TRUE; a
        NULL result keeps it.
        """
        _refuse_options("delete", unsupported)
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

        from .. import predicate as sqlpred

        _refuse_options("update", unsupported)
        assignments: dict[str, Any] = dict(new_values or {})
        for column, expression in (updates or {}).items():
            value = sqlpred.parse_value(expression)
            assignments[column] = value
        if not assignments:
            raise UnreachableTableError("update", "no assignments given")

        def assign(current: Any, keep: Any) -> Any:
            out = current
            for column, value in assignments.items():
                index = out.schema.get_field_index(column)
                if index < 0:
                    raise UnreachableTableError(f"update {column}", "the table has no such column")
                field = out.schema.field(index)
                if isinstance(value, sqlpred.Column):
                    source = out.column(".".join(value.path)).cast(field.type)
                else:
                    raw = value.value if isinstance(value, sqlpred.Literal) else value
                    source = pa.array([raw] * out.num_rows).cast(field.type)
                new = pc.if_else(keep, out.column(index), source)
                out = out.set_column(index, field, new)
            return out

        result = self._rewrite(table, predicate, assign, operation="UPDATE")
        return {"num_updated_rows": result["num_affected_rows"], "version": result["version"]}

    def publish(self, table: ResolvedTable) -> int:
        """Publish ratified-but-unpublished commits into the Delta log."""
        snapshot = self.snapshot(table, write=True)
        with _library_commit_errors():
            version: int = snapshot.publish(uc=self._uc_commit_config(table))
        return version

    # ---------------------------------------------------- metadata-only DDL

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
        if operation is Operation.CLUSTER_BY and table.partition_columns:
            return Capability(
                operation,
                ok=False,
                reason="the table is partitioned; a table is either partitioned or clustered",
            )
        return None

    def _state(self, table: ResolvedTable) -> tuple[Any, Any]:
        """(snapshot, TableState) for the latest version."""
        import json

        from .metadata import CLUSTERING_DOMAIN, TableState

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

        from .metadata import build_actions

        last_error: Exception | None = None
        for _ in range(self.metadata_commit_attempts):
            snapshot, state = self._state(table)
            if precheck is not None:
                precheck(snapshot, state)
            change = mutate(state)
            if change.protocol is None and change.metadata is None and not change.domains:
                return int(state.version)
            actions = build_actions(state, change, engine_info=f"deltaswamp/{_version()}")
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
        raise UnreachableTableError(
            "commit a metadata change",
            f"another writer committed first on each of {self.metadata_commit_attempts} "
            f"attempts ({last_error})",
            "retry when the table is less busy",
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
        from . import metadata as m

        new_fields = _delta_fields(fields)
        return self._commit_metadata(table, lambda s: m.add_columns(s, new_fields))

    def drop_column(self, table: ResolvedTable, column: str) -> int:
        from . import metadata as m

        return self._commit_metadata(table, lambda s: m.drop_column(s, column))

    def rename_column(self, table: ResolvedTable, old: str, new: str) -> int:
        from . import metadata as m

        return self._commit_metadata(table, lambda s: m.rename_column(s, old, new))

    def set_properties(self, table: ResolvedTable, properties: dict[str, str], **_: Any) -> int:
        from . import metadata as m

        return self._commit_metadata(table, lambda s: m.set_properties(s, properties))

    def unset_properties(
        self, table: ResolvedTable, keys: list[str], *, if_exists: bool = True
    ) -> int:
        from . import metadata as m

        return self._commit_metadata(
            table, lambda s: m.unset_properties(s, keys, if_exists=if_exists)
        )

    def add_feature(self, table: ResolvedTable, feature: Any, **_: Any) -> int:
        from . import metadata as m

        names = feature if isinstance(feature, (list, tuple, set, frozenset)) else [feature]

        def mutate(state: Any) -> Any:
            props = {f"delta.feature.{getattr(n, 'value', n)}": "supported" for n in names}
            return m.set_properties(state, props)

        return self._commit_metadata(table, mutate)

    def drop_constraint(self, table: ResolvedTable, name: str, *, if_exists: bool = False) -> int:
        from . import metadata as m

        return self._commit_metadata(
            table, lambda s: m.drop_constraint(s, name, if_exists=if_exists)
        )

    def set_comment(self, table: ResolvedTable, comment: str | None) -> int:
        from . import metadata as m

        return self._commit_metadata(table, lambda s: m.set_comment(s, comment))

    def set_column_comment(self, table: ResolvedTable, column: str, comment: str | None) -> int:
        from . import metadata as m

        return self._commit_metadata(table, lambda s: m.set_column_comment(s, column, comment))

    def alter_column_type(self, table: ResolvedTable, column: str, new_type: str) -> int:
        from . import metadata as m

        return self._commit_metadata(table, lambda s: m.alter_column_type(s, column, new_type))

    def drop_not_null(self, table: ResolvedTable, column: str) -> int:
        from . import metadata as m

        return self._commit_metadata(table, lambda s: m.set_nullability(s, column, True))

    def set_not_null(self, table: ResolvedTable, column: str) -> int:
        """SET NOT NULL, after proving no existing row is null.

        The check runs against the same snapshot the commit is computed from,
        and the put-if-absent commit fails if anything was written since -- so
        a null cannot slip in between the check and the change.
        """
        from . import metadata as m

        def precheck(snapshot: Any, state: Any) -> None:
            import pyarrow as pa

            nulls = 0
            for batch in pa.RecordBatchReader.from_stream(snapshot.scan(columns=[column])):
                nulls += batch.column(0).null_count
            if nulls:
                raise UnreachableTableError(
                    f"SET NOT NULL on {column}",
                    f"{nulls} existing row(s) have a null {column}",
                    "update or delete those rows first",
                )

        return self._commit_metadata(
            table, lambda s: m.set_nullability(s, column, False), precheck=precheck
        )

    def cluster_by(self, table: ResolvedTable, columns: Any) -> int:
        from . import metadata as m

        if isinstance(columns, str):
            if columns.lower() == "auto":
                raise UnreachableTableError(
                    "CLUSTER BY AUTO",
                    "automatic key selection is Databricks predictive optimization, which "
                    "runs server-side",
                    "ds.connect(..., allow_sql_fallback=True)",
                )
            columns = [columns]
        return self._commit_metadata(table, lambda s: m.cluster_by(s, list(columns or [])))

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
        import json

        import pyarrow as pa

        from .. import predicate as sqlpred
        from .base import DeletionVectorDescriptor, ScanSplit

        if not self.supports_distributed_scan:
            raise NotImplementedError(
                "the installed native extension cannot restrict a scan to planned files"
            )
        snapshot = self.snapshot(table, version=version, timestamp=timestamp)
        skipping = sqlpred.to_kernel_json(sqlpred.parse(predicate)) if predicate else None
        files = pa.table(snapshot.files(predicate=skipping)).to_pylist()
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
                    partition_values={k: v for k, v in partition_values.items()},
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
            _as_record_batch_reader(data), uc=self._uc_commit_config(table)
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
    ) -> int:
        """Commit fragments from `write_files` as one transaction.

        Every fragment lands at a single version, so a distributed write is
        atomic: a reader sees all of it or none of it.
        """
        if not self.supports_distributed_write:
            raise NotImplementedError(
                "the installed native extension cannot commit externally written files"
            )
        snapshot = self.snapshot(table, write=True)
        with _library_commit_errors():
            version: int = snapshot.commit_files(
                list(fragments),
                uc=self._uc_commit_config(table),
                engine_info=engine_info or f"deltaswamp/{_version()}",
                operation=operation,
                overwrite=overwrite,
                txn=txn,
                commit_metadata={k: str(v) for k, v in (commit_metadata or {}).items()} or None,
            )
        self._maybe_checkpoint(table, version)
        return version

    def execute_scan(
        self,
        table: ResolvedTable,
        splits: list[Any],
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
    ) -> Any:
        """Read planned splits: same semantics as `scan`, restricted to their files.

        Deletion vectors, column mapping and partition values are handled by
        the kernel exactly as in a full scan; the predicate is applied exactly
        afterwards.
        """
        from .. import predicate as sqlpred

        versions = {s.commit_version for s in splits}
        if len(versions) > 1:
            raise UnreachableTableError(
                "execute scan splits",
                f"the splits were planned against different versions ({sorted(versions)})",
                "plan once and distribute that plan",
            )
        version = next(iter(versions)) if versions else None
        snapshot = self.snapshot(table, version=version)
        paths = [s.path for s in splits]
        if predicate is None:
            return snapshot.scan(columns=columns, files=paths)
        node = sqlpred.parse(predicate)
        read_columns = columns
        if columns is not None:
            needed = {path[0] for path in sqlpred.columns_of(node)}
            read_columns = list(columns) + sorted(needed - set(columns))
        stream = snapshot.scan(
            columns=read_columns, predicate=sqlpred.to_kernel_json(node), files=paths
        )
        return sqlpred.filter_stream(stream, node, keep=list(columns) if columns else None)


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


def _conflict_version(message: str) -> int:
    """The version someone else won, if the message names one; -1 otherwise."""
    import re

    found = re.search(r"version (\d+)", message)
    return int(found.group(1)) if found else -1


def _delta_fields(fields: Any) -> list[dict[str, Any]]:
    """Normalise what `add_columns` accepts into Delta schema field dicts.

    Takes a pyarrow Schema or Field (or a list of them), delta-rs `Field`
    objects, or a `{name: delta_type}` mapping of primitive types.
    """
    import json

    from .metadata import arrow_to_delta_field

    if isinstance(fields, dict):
        return [
            {"name": name, "type": str(dtype), "nullable": True, "metadata": {}}
            for name, dtype in fields.items()
        ]
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


def _require_pyarrow(what: str) -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise UnreachableTableError(
            what,
            "pyarrow is needed to evaluate the row filter",
            "pip install 'deltaswamp[pyarrow]'",
        ) from exc


def _timestamp_ms(value: Any) -> int:
    """Epoch milliseconds from a datetime, an ISO-8601 string, or a number.

    A naive datetime or zoneless string is read as UTC, the only defensible
    choice without a session time zone.
    """
    import datetime as dt

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        try:
            value = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise UnreachableTableError(
                f"time travel to {value!r}", "not an ISO-8601 timestamp"
            ) from exc
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        value = dt.datetime(value.year, value.month, value.day)
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        return int(value.timestamp() * 1000)
    raise UnreachableTableError(
        f"time travel to {value!r}", "expected a datetime, an ISO-8601 string or epoch millis"
    )


def _evaluate(table: Any, expr: Any) -> Any:
    """Evaluate a boolean compute expression over a table, as one array."""
    import pyarrow as pa
    import pyarrow.dataset as ds

    if table.num_rows == 0:
        return pa.array([], pa.bool_())
    result = ds.dataset(table).to_table(columns={"_m": expr}).column("_m")
    return result.combine_chunks() if isinstance(result, pa.ChunkedArray) else result


def _refuse_options(what: str, options: dict[str, Any]) -> None:
    given = sorted(k for k, v in options.items() if v is not None)
    if given:
        raise UnreachableTableError(
            f"{what} with {', '.join(given)}",
            "the kernel rewrite path does not implement these options",
        )
