"""The delta-kernel engine: the default read path.

Kernel earns that position because it evaluates feature support *per operation*
and ignores writer-only features when reading. delta-rs does a flat
set-difference against a hardcoded list and refuses to open 19 features kernel
reads without complaint -- including every `catalogManaged` table and, because
it is a ReaderWriter feature, anything carrying `vacuumProtocolCheck`.
"""

from __future__ import annotations

from typing import Any

from ..capability import (
    FEATURE_SUPPORT,
    Capability,
    Operation,
    Support,
    feature_from_wire,
)
from ..capability import (
    Engine as EngineKind,
)
from ..catalog import ResolvedTable
from ..credentials import Operation as CredentialOperation
from ..errors import UnreachableTableError
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


# Operations the kernel engine implements today. Write support arrives with the
# UC committer; until then `supports()` reports the gap rather than failing late.
_WRITE_OPS: frozenset[Operation] = frozenset({Operation.APPEND, Operation.CREATE})

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
)


class KernelEngine:
    """Reads Delta tables through delta-kernel-rs."""

    kind = EngineKind.KERNEL
    supports_distributed_scan = False
    #: Kernel takes kernel `Predicate` objects, not SQL strings, and nothing
    #: here builds one yet.
    supports_predicates = False
    #: None of these are bound in the native extension yet. Declaring them false
    #: makes the router divert the call rather than letting it be dropped.
    supports_schema_merge = False
    supports_schema_overwrite = False
    supports_idempotent_txn = True
    supports_commit_metadata = True
    supports_writer_properties = False
    supports_dynamic_overwrite = False
    #: Timestamp travel needs history_manager plumbing that is not exposed.
    supports_timestamp_travel = False

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

        if operation not in _IMPLEMENTED:
            return Capability(
                operation,
                ok=False,
                reason=f"the kernel engine does not implement {operation.value} yet",
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
        for name in table.reader_features:
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

        if operation in _WRITE_OPS:
            write_blockers: list[str] = []
            for name in table.writer_features:
                feature = feature_from_wire(name)
                if feature is None:
                    write_blockers.append(f"{name} (unrecognized writer feature)")
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
            if table.partition_columns:
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
        write: bool = False,
    ) -> Any:
        """Resolve a kernel snapshot, supplying the catalog tail when needed."""
        from deltaswamp._native import Snapshot

        if table.location is None:
            raise UnreachableTableError("open", "the table has no storage location", None)

        options = dict(self._base_options)
        if table.credential_provider is not None:
            op = CredentialOperation.READ_WRITE if write else CredentialOperation.READ
            creds = table.credential_provider.credentials(op)
            options.update(creds.as_storage_options())

        # `log_tail` and `max_catalog_version` are what make a catalog-managed
        # table readable; both are meaningless (and omitted) otherwise.
        log_tail = [
            (entry.version, entry.path, entry.timestamp or 0, entry.size)
            for entry in table.log_tail
        ] or None

        return Snapshot.resolve(
            table.location,
            options=options,
            version=version,
            log_tail=log_tail,
            max_catalog_version=table.max_catalog_version,
        )

    def scan(
        self,
        table: ResolvedTable,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: str | None = None,
    ) -> Any:
        # The router normally keeps these away from this engine; the checks stay
        # as a guard for anyone calling the engine directly.
        if predicate is not None:
            raise UnreachableTableError(
                "scan with a predicate",
                "the kernel engine does not accept predicates yet",
                "omit the predicate, or use a table the delta-rs engine can open",
            )
        if timestamp is not None:
            raise UnreachableTableError(
                "scan at a timestamp",
                "timestamp travel is not wired up in the kernel engine yet",
                "pass an explicit version instead",
            )
        return self.snapshot(table, version=version).scan(columns=columns)

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

        # The kernel writer we drive here builds one unpartitioned write context.
        # Writing a partitioned table through it would put every row in the root
        # directory with no partition values -- silently wrong rather than an
        # error, so refuse explicitly.
        partitions = list(snapshot.partition_columns)
        if partitions:
            raise UnreachableTableError(
                "append to a partitioned table via the kernel engine",
                f"the table is partitioned by {', '.join(partitions)}, and the kernel "
                "write path here handles unpartitioned writes only",
                "path-based tables route to delta-rs, which handles partitioning; a "
                "partitioned catalog-managed table needs the SQL fallback",
            )
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
        """Checkpoint after a commit, where anything can.

        Kernel commits never checkpoint on their own, and binding
        `Snapshot::checkpoint` is not currently possible: it deadlocks, both on
        and off the shared runtime, against the default engine's background
        executor. So the fallback is delta-rs, which cannot open a
        catalog-managed table at all.

        The consequence, stated plainly because it matters operationally: a
        catalog-managed table written only through this path accumulates log
        entries with no checkpoint, and gets slower to open over time. Run
        OPTIMIZE or a checkpoint from Databricks periodically until the kernel
        binding is fixed.
        """
        if version == 0 or version % self.checkpoint_interval(table) != 0:
            return
        if table.is_catalog_managed:
            return
        try:
            from .deltars import DeltaRsEngine

            DeltaRsEngine(storage_options=self._base_options).checkpoint(table)
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
        if predicate is not None or partition_overwrite != "static":
            raise UnreachableTableError(
                "overwrite part of a table with the kernel engine",
                "the kernel path removes every file in the snapshot; it cannot scope "
                "the removal to a predicate or to individual partitions",
                "for a path-based table this routes to delta-rs; a catalog-managed "
                "table needs the SQL fallback",
            )
        return self.append(table, data, operation="WRITE", overwrite=True, **kwargs)

    def publish(self, table: ResolvedTable) -> int:
        """Publish ratified-but-unpublished commits into the Delta log."""
        snapshot = self.snapshot(table, write=True)
        version: int = snapshot.publish(uc=self._uc_commit_config(table))
        return version

    def plan_scan(self, table: ResolvedTable, **kwargs: Any) -> list[Any]:
        raise NotImplementedError(
            "split planning is not exposed by the native extension yet; it lands with "
            "the distributed read path"
        )

    def execute_scan(self, table: ResolvedTable, splits: list[Any], **kwargs: Any) -> Any:
        raise NotImplementedError("see plan_scan")
