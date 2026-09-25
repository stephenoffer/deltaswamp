"""`Table`: one table, every operation, routed per operation.

Protocol features are discovered in two stages. The catalog supplies enough to
make the decisions that must precede opening the log (is this a view, a shallow
clone, a table vending refuses). The full reader/writer feature lists exist only
in the log, so the first operation enriches the resolved table with them and
everything afterward routes on the complete picture.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .capability import FEATURE_SUPPORT, Capability, FeatureKind, Operation, feature_from_wire
from .capability import Engine as EngineKind
from .catalog import ResolvedTable, TableType
from .credentials import Operation as CredentialOperation
from .errors import (
    SQL_FALLBACK_REMEDY,
    CorruptTableError,
    DeltaSwampError,
    EngineFallbackWarning,
    FallbackRequiredError,
    UnreachableTableError,
)
from .identity import parse_ref

if TYPE_CHECKING:
    from .connection import Connection

__all__ = ["Table"]


def _plain_views(table: Any) -> Any:
    """Cast Arrow view types (string_view, binary_view) to their plain forms.

    delta-rs returns views, and several pyarrow kernels (sort, take) have no
    implementation for them yet.
    """
    import pyarrow as pa

    fields = []
    for field in table.schema:
        if pa.types.is_string_view(field.type):
            field = field.with_type(pa.string())
        elif pa.types.is_binary_view(field.type):
            field = field.with_type(pa.binary())
        fields.append(field)
    target = pa.schema(fields, metadata=table.schema.metadata)
    return table if target.equals(table.schema) else table.cast(target)


def _given(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop None-valued keyword arguments, so engines see only what was asked."""
    return {k: v for k, v in kwargs.items() if v is not None}


def _require(module: str, extra: str) -> Any:
    """Import an optional dependency, or say which extra provides it.

    Without this the failure surfaces as a ModuleNotFoundError raised from deep
    inside pyarrow, which does not tell you what to install.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} is needed for this conversion but is not installed. "
            f"Install it with: pip install 'deltaswamp[{extra}]'"
        ) from exc


def _protocol_from_properties(resolved: ResolvedTable) -> dict[str, Any]:
    """The protocol as the catalog records it, for a log no engine could read.

    Unity Catalog mirrors `delta.minReaderVersion`, `delta.minWriterVersion` and
    one `delta.feature.<name>` per feature into the table's properties. That is
    enough to name the feature that makes the log unreadable -- which beats
    reporting an empty feature set.
    """
    if resolved.reader_features or resolved.writer_features:
        return {}
    props = resolved.properties
    names = {
        key.removeprefix("delta.feature.")
        for key, value in props.items()
        if key.startswith("delta.feature.") and value.lower() in ("supported", "enabled")
    }
    if not names:
        return {}

    def version(key: str) -> int | None:
        try:
            return int(props[key])
        except (KeyError, ValueError):
            return None

    reader_version = version("delta.minReaderVersion")
    readers = set()
    if reader_version is not None and reader_version >= 3:
        for name in names:
            feature = feature_from_wire(name)
            # Unknown names are kept as reader features: refusing too much is
            # recoverable, claiming a read that fails is not.
            if feature is None or FEATURE_SUPPORT[feature].kind is not FeatureKind.WRITER:
                readers.add(name)
    return {
        "min_reader_version": reader_version,
        "min_writer_version": version("delta.minWriterVersion"),
        "reader_features": frozenset(readers),
        "writer_features": frozenset(names),
    }


class Table:
    """One table: reads, writes, DDL and maintenance, routed per operation."""

    def __init__(
        self, connection: Connection, resolved: ResolvedTable, *, version: int | None = None
    ) -> None:
        self._connection = connection
        self._resolved = resolved
        self._version = version
        self._enriched = False
        #: Set by a commit through this handle on a table whose commit tail lives
        #: in the catalog; the next operation re-resolves before reading.
        self._catalog_stale = False
        # What the catalog said, kept apart from what the log says: the log is
        # the truth for table properties, and re-enrichment after an ALTER must
        # not let a stale earlier read win.
        self._catalog_properties = dict(resolved.properties)

    # --------------------------------------------------------------- identity

    def __repr__(self) -> str:
        return f"Table({self._resolved.ref}, location={self._resolved.location!r})"

    @property
    def resolved(self) -> ResolvedTable:
        return self._resolved

    @property
    def location(self) -> str | None:
        return self._resolved.location

    @property
    def table_type(self) -> str | None:
        t = self._resolved.table_type
        return t.value if t else None

    @property
    def securable_kind(self) -> str | None:
        return self._resolved.securable_kind

    @property
    def is_catalog_managed(self) -> bool:
        return self._resolved.is_catalog_managed

    # ------------------------------------------------------------- enrichment

    def _check_identity(self, metadata_id: str | None) -> None:
        """Refuse a table whose log identity does not match the catalog's.

        A table dropped and re-created under the same name keeps the name and
        gets a new id. Reading on with a cached id means reading a different
        table while believing it is the same one.
        """
        expected = self._resolved.table_uuid
        # Only a managed table's log carries the catalog's id; the catalog gives
        # a registered external table an id of its own.
        if self._resolved.table_type not in (TableType.MANAGED, None):
            return
        if expected and metadata_id and expected != metadata_id:
            raise CorruptTableError(
                f"{self._resolved.ref} resolves to a table whose log id is "
                f"{metadata_id!r}, but the catalog reports {expected!r}. The table was "
                "most likely dropped and re-created; re-open it to pick up the new one."
            )

    def _invalidate(self) -> None:
        """Forget cached protocol state after a commit.

        Enrichment caches the feature lists and properties read from the log.
        Any write or ALTER changes them, so a Table that kept the cache would
        keep reporting what was true before the call it just made.

        A catalog-managed table needs more: its snapshot is pinned to the
        catalog version captured at resolve time, so the kernel would keep
        reading the table as it was before this very write. Its
        commit tail is re-fetched on the next operation, not here, so a failure
        to reach the catalog surfaces there rather than after a write that
        succeeded.
        """
        self._enriched = False
        if self._version is None and (
            self._resolved.is_catalog_managed or self._resolved.max_catalog_version is not None
        ):
            self._catalog_stale = True

    def _enrich(self) -> ResolvedTable:
        """Fill in the protocol feature lists by reading the log once.

        Does not go through the router: routing depends on these
        features, so asking the router first would be circular. We try kernel,
        then delta-rs, and fall back to catalog metadata alone if neither can
        open the table -- in which case routing proceeds on partial information
        and the engines themselves produce the refusal.
        """
        if self._catalog_stale:
            fresh = self._connection._reresolve(self).resolved
            self._resolved = fresh
            self._catalog_properties = dict(fresh.properties)
            self._catalog_stale = False
        if self._enriched or self._resolved.location is None:
            return self._resolved

        self._enriched = True
        last_error: str | None = None
        for kind in (EngineKind.KERNEL, EngineKind.DELTARS):
            engine = self._connection.router.engines.get(kind)
            if engine is None or not engine.available():  # type: ignore[attr-defined]
                continue
            try:
                detail = engine.detail(self._resolved, version=self._version)  # type: ignore[attr-defined]
            except Exception as exc:
                # Losing this is how a vending failure turns into an empty
                # feature set and a confident, wrong "yes".
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            self._resolved = dataclasses.replace(
                self._resolved,
                min_reader_version=detail.get("min_reader_version"),
                min_writer_version=detail.get("min_writer_version"),
                reader_features=frozenset(detail.get("reader_features") or ()),
                writer_features=frozenset(detail.get("writer_features") or ()),
                properties={**self._catalog_properties, **(detail.get("properties") or {})},
                partition_columns=tuple(detail.get("partition_columns") or ()),
                has_invariants=bool(detail.get("has_invariants")),
                has_check_constraints=bool(detail.get("has_check_constraints")),
                has_generated_columns=bool(detail.get("has_generated_columns")),
            )
            self._check_identity(detail.get("metadata_id"))
            return self._resolved

        if last_error is not None:
            self._resolved = dataclasses.replace(
                self._resolved, open_error=last_error, **_protocol_from_properties(self._resolved)
            )
        return self._resolved

    # ------------------------------------------------------------ capabilities

    def capabilities(self) -> dict[Operation, Capability]:
        """What can and cannot be done with this table, and why.

        Every refusal names the blocker and, where one exists, the remedy.
        """
        return self._connection.router.capabilities(self._enrich())

    def can(self, operation: Operation | str, **shape: Any) -> Capability:
        """Whether an operation is possible, optionally for a specific request.

        `shape` takes the same arguments as the call itself, so you can
        preflight the write you actually intend::

            t.can("create", properties={"delta.enableRowTracking": "true"})
            t.can("append", schema_mode="merge")
        """
        op = Operation(operation) if isinstance(operation, str) else operation
        return self._connection.router.capability(op, self._enrich(), **shape)

    def _engine(
        self,
        operation: Operation,
        needs: frozenset[str] = frozenset(),
        **shape: Any,
    ) -> Any:
        return self._connection.router.engine_for(operation, self._enrich(), needs=needs, **shape)

    def _read(
        self,
        operation: Operation,
        call: Callable[[Any], Any],
        needs: frozenset[str] = frozenset(),
    ) -> Any:
        """Serve a read-only operation, moving to the next engine if one breaks.

        Routing decides from the protocol, but an engine can still choke on a
        table it claims -- delta-rs cannot parse the file statistics Databricks
        writes for a CLONE, for one. A read changes nothing, so trying the next
        engine that claims it is safe. Refusals raised by this library are
        deliberate and propagate as they are.
        """
        tried: set[EngineKind] = set()
        while True:
            engine: Any = self._connection.router.engine_for(
                operation, self._enrich(), needs=needs, exclude=frozenset(tried)
            )
            try:
                return call(engine)
            except DeltaSwampError:
                raise
            except Exception as exc:
                tried.add(engine.kind)
                try:
                    self._connection.router.engine_for(
                        operation, self._resolved, needs=needs, exclude=frozenset(tried)
                    )
                except DeltaSwampError:
                    raise exc from None
                warnings.warn(
                    f"{engine.kind.value} failed to serve {operation.value} "
                    f"({type(exc).__name__}: {str(exc)[:200]}); trying the next engine",
                    EngineFallbackWarning,
                    stacklevel=3,
                )

    # ------------------------------------------------------------------- read

    def scan(
        self,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: str | None = None,
        limit: int | None = None,
    ) -> Any:
        """Read the table as an Arrow stream (exports `__arrow_c_stream__`).

        `limit` is a hint passed to the engine. Streaming engines ignore it and
        the caller simply stops reading; the SQL warehouse turns it into a real
        `LIMIT`, because it computes the result set before streaming any of it.
        """
        # `version or timestamp` would treat version 0 as no time travel.
        travelling = version is not None or timestamp is not None
        op = Operation.TIME_TRAVEL if travelling else Operation.SCAN
        # A predicate or a timestamp narrows which engines can serve the call,
        # so say so up front instead of letting one accept and then raise.
        needs = set()
        if predicate is not None:
            needs.add("predicates")
        if timestamp is not None:
            needs.add("timestamp_travel")
        return self._read(
            op,
            lambda engine: engine.scan(
                self._resolved,
                columns=columns,
                predicate=predicate,
                version=version if version is not None else self._version,
                timestamp=timestamp,
                limit=limit,
            ),
            frozenset(needs),
        )

    def to_arrow(self, **kwargs: Any) -> Any:
        pa = _require("pyarrow", "pyarrow")
        return pa.table(self.scan(**kwargs))

    def to_pandas(self, **kwargs: Any) -> Any:
        _require("pandas", "pandas")
        return self.to_arrow(**kwargs).to_pandas()

    def to_polars(self, *, lazy: bool = False, **kwargs: Any) -> Any:
        """A Polars DataFrame, or a LazyFrame over it with ``lazy=True``.

        The lazy form still reads through this library, so it works on the
        tables `polars.scan_delta` cannot open (catalog-managed, row-tracked,
        vacuumProtocolCheck, ...).
        """
        pl = _require("polars", "polars")
        frame = pl.DataFrame(self.scan(**kwargs))
        return frame.lazy() if lazy else frame

    def to_duckdb(self, connection: Any = None, *, name: str | None = None, **kwargs: Any) -> Any:
        """A DuckDB relation over the table. With `name`, also a view of that name.

        DuckDB's own delta extension is C++ and knows nothing of Unity Catalog
        credentials or catalog-managed commits; this hands it the rows instead.
        """
        duckdb = _require("duckdb", "duckdb")
        pa = _require("pyarrow", "pyarrow")
        con = connection if connection is not None else duckdb.connect()
        data = pa.table(self.scan(**kwargs))
        relation = con.from_arrow(data)
        if name is not None:
            relation.create_view(name, replace=True)
        return relation

    def plan_write(
        self,
        *,
        mode: str = "append",
        txn: tuple[str, int] | None = None,
        commit_metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Plan a distributed write, refusing now if the table will not accept it.

        Returns a picklable `WritePlan`. Ship it to workers, call
        `plan.write(batch)` there, send the fragments back, and commit them all
        at once with `plan.commit(fragments)`. The write lands at a single
        version: a reader sees the whole job or none of it.

        The refusal happens here, on the driver, before any worker runs. That
        is the point of planning separately -- the usual way a distributed Delta
        write fails is to discover at commit time that the table rejects it,
        after the compute is spent, leaving orphaned Parquet behind. Anything
        `can()` reports as unavailable is raised here instead, with the reason.

        `mode` is ``append`` or ``overwrite``; overwrite removes every file
        visible in the planned snapshot in the same commit.
        """
        from .distributed import WritePlan

        if mode not in ("append", "overwrite"):
            raise UnreachableTableError(
                f"plan a write with mode={mode!r}",
                "a distributed write is 'append' or 'overwrite'",
            )
        operation = Operation.OVERWRITE if mode == "overwrite" else Operation.APPEND
        needs = {"distributed_write"}
        if txn is not None:
            needs.add("idempotent_txn")
        if commit_metadata is not None:
            needs.add("commit_metadata")
        if txn is not None and self._already_committed(txn):
            raise UnreachableTableError(
                f"plan an idempotent write for {txn[0]!r} at version {txn[1]}",
                "that transaction is already committed, so running the job would "
                "duplicate work whose result is already in the table",
                "raise the txn version, or drop txn= to write unconditionally",
            )
        engine = self._engine(operation, frozenset(needs))
        return WritePlan(
            engine=engine,
            table=self._enrich(),
            mode=mode,
            version=self.version,
            txn=txn,
            commit_metadata=commit_metadata,
        )

    def plan_scan(
        self,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: Any = None,
    ) -> Any:
        """Plan a distributed read: a picklable `ScanPlan` of per-file splits.

        Ship the plan (or parts of it, via `plan.partitions(n)`) to workers and
        call `plan.read(splits)` there. Each worker re-resolves the same
        snapshot version and vends its own credentials.
        """
        from .distributed import ScanPlan

        needs = {"distributed_scan"}
        if predicate is not None:
            needs.add("predicates")
        engine = self._engine(Operation.SCAN, frozenset(needs))
        splits = engine.plan_scan(
            self._resolved,
            columns=columns,
            predicate=predicate,
            version=version if version is not None else self._version,
            timestamp=timestamp,
        )
        return ScanPlan(
            engine=engine,
            table=self._resolved,
            splits=tuple(splits),
            columns=tuple(columns) if columns is not None else None,
            predicate=predicate,
        )

    def to_ray_dataset(self, *, override_num_blocks: int | None = None, **kwargs: Any) -> Any:
        """A Ray Dataset, read in parallel by Ray workers.

        The scan is planned on the driver and each read task reads a
        byte-balanced group of files. Where no engine can plan a distributed
        read, the table is read on the driver instead.
        """
        ray_data = _require("ray.data", "ray")
        from .distributed import DeltaSwampDatasource

        if self.can(Operation.SCAN).engine is not None:
            try:
                plan = self.plan_scan(**kwargs)
            except UnreachableTableError:
                plan = None
            if plan is not None:
                return ray_data.read_datasource(
                    DeltaSwampDatasource(plan), override_num_blocks=override_num_blocks
                )
        return ray_data.from_arrow(self.to_arrow(**kwargs))

    def to_daft(self, **kwargs: Any) -> Any:
        """A Daft DataFrame."""
        daft = _require("daft", "daft")
        return daft.from_arrow(self.to_arrow(**kwargs))

    def to_pyarrow_dataset(self, **kwargs: Any) -> Any:
        dataset = _require("pyarrow.dataset", "pyarrow")
        return dataset.dataset(self.to_arrow(**kwargs))

    def head(self, n: int = 5, **kwargs: Any) -> Any:
        """The first `n` rows.

        Consumes the scan stream batch by batch and stops as soon as `n` rows
        are in hand, so this costs one batch on a table of any size.

        The stream is the engine's output, so deletion vectors, column mapping
        and partition values are already applied; stopping early here is not the
        limit pushdown the scan layer refuses, which would break
        the positional mapping a deletion vector depends on.
        """
        pa = _require("pyarrow", "pyarrow")
        # The limit is a hint to the engine as well as a client-side stop: the
        # warehouse would otherwise compute the entire result set first.
        kwargs.setdefault("limit", n)
        reader = pa.RecordBatchReader.from_stream(self.scan(**kwargs))
        batches, taken = [], 0
        if n > 0:
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                batches.append(batch)
                taken += batch.num_rows
                if taken >= n:
                    break
        table = (
            pa.Table.from_batches(batches, reader.schema)
            if batches
            else reader.schema.empty_table()
        )
        return table.slice(0, n)

    def count(self, *, predicate: str | None = None) -> int:
        """Exact row count.

        Streams the narrowest column rather than materializing the table; the
        engines' statistics-based counts are approximate by their own
        documentation (a file without stats counts as zero rows), so they are
        not used here.
        """
        pa = _require("pyarrow", "pyarrow")
        schema = self.schema()
        names = list(getattr(schema, "names", None) or [f.name for f in schema])
        narrow = [names[0]] if names and predicate is None else None
        total = 0
        for batch in pa.RecordBatchReader.from_stream(
            self.scan(columns=narrow, predicate=predicate)
        ):
            total += batch.num_rows
        return total

    def files(self) -> Any:
        """The table's live data files: path, size, partition values and statistics."""
        return self._read(
            Operation.FILES, lambda engine: engine.files(self._resolved, version=self._version)
        )

    def history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Commit history, newest first. `timestamp` is epoch milliseconds.

        Milliseconds is what the Delta log records and what delta-rs and the
        Iceberg engine return; the warehouse returns a datetime, so it is
        converted here and the type no longer depends on which engine served.
        """
        result: list[dict[str, Any]] = self._read(
            Operation.HISTORY, lambda engine: engine.history(self._resolved, limit=limit)
        )
        for entry in result:
            stamp: Any = entry.get("timestamp")
            if hasattr(stamp, "timestamp"):
                entry["timestamp"] = round(stamp.timestamp() * 1000)
        return result

    def detail(self) -> dict[str, Any]:
        result: dict[str, Any] = self._read(
            Operation.DETAIL, lambda engine: engine.detail(self._resolved, version=self._version)
        )
        return result

    def cdf(self, **kwargs: Any) -> Any:
        """The change data feed, by version or timestamp range.

        Rows carry `_change_type`, `_commit_version` and `_commit_timestamp`.
        """
        given = _given(kwargs)
        return self._read(Operation.CDF, lambda engine: engine.cdf(self._resolved, **given))

    def changes(
        self,
        starting_version: int,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        poll_interval: float | None = None,
    ) -> Any:
        """Follow the change feed, one committed version at a time.

        Yields ``(version, pyarrow.Table)`` for every version from
        `starting_version` on, in order. With `poll_interval` (seconds) it keeps
        waiting for new commits, like a streaming read with a change-feed
        source; without, it stops at the latest version. Record the last
        version you processed and pass the next one to resume.
        """
        import time

        pa = _require("pyarrow", "pyarrow")
        next_version = starting_version
        while True:
            current = self._connection._reresolve(self)
            latest = current.version
            if latest is not None and latest >= next_version:
                changes = pa.table(
                    current.cdf(
                        starting_version=next_version,
                        ending_version=latest,
                        columns=columns,
                        predicate=predicate,
                    )
                )
                if changes.num_rows:
                    changes = _plain_views(changes).sort_by("_commit_version")
                    versions = changes.column("_commit_version").to_pylist()
                    for version in sorted(set(versions)):
                        mask = pa.compute.equal(changes.column("_commit_version"), version)
                        yield int(version), changes.filter(mask)
                next_version = latest + 1
            if poll_interval is None:
                return
            time.sleep(poll_interval)

    # -------------------------------------------------------------- metadata

    def schema(self) -> Any:
        engine = self._engine(Operation.SCAN)
        snapshot = getattr(engine, "snapshot", None)
        if snapshot is not None:
            return snapshot(self._resolved, version=self._version).schema()
        return self.to_arrow().schema

    def protocol(self) -> tuple[int | None, int | None]:
        r = self._enrich()
        return (r.min_reader_version, r.min_writer_version)

    def features(self) -> frozenset[str]:
        """Every table feature in force, including those a legacy protocol implies.

        A protocol below reader 3 / writer 7 names no features: its version
        number is the feature set. Databricks' DESCRIBE DETAIL reports the
        implied ones (a (1, 2) table lists appendOnly and invariants), and so
        does this, so the two agree on every table.
        """
        r = self._enrich()
        return r.effective_reader_features | r.effective_writer_features

    def properties(self) -> dict[str, str]:
        return dict(self._enrich().properties)

    @property
    def version(self) -> int | None:
        """The current version; None for an Iceberg table with no snapshot yet."""
        value = self.detail().get("version")
        return None if value is None else int(value)

    # ------------------------------------------------------------------ write

    @staticmethod
    def _write_needs(
        schema_mode: str | None,
        commit_metadata: dict[str, Any] | None,
        txn: tuple[str, int] | None,
        writer_properties: Any,
        partition_overwrite: str,
    ) -> frozenset[str]:
        """Translate write arguments into engine capabilities they require."""
        needs: set[str] = set()
        if schema_mode == "merge":
            needs.add("schema_merge")
        elif schema_mode == "overwrite":
            needs.add("schema_overwrite")
        if commit_metadata is not None:
            needs.add("commit_metadata")
        if txn is not None:
            needs.add("idempotent_txn")
        if writer_properties is not None:
            needs.add("writer_properties")
        if partition_overwrite == "dynamic":
            needs.add("dynamic_overwrite")
        return frozenset(needs)

    def append(
        self,
        data: Any,
        *,
        schema_mode: str | None = None,
        partition_by: list[str] | None = None,
        target_file_size: int | None = None,
        writer_properties: Any = None,
        commit_metadata: dict[str, Any] | None = None,
        txn: tuple[str, int] | None = None,
        max_commit_retries: int | None = None,
    ) -> None:
        """Append data.

        `schema_mode="merge"` widens the table schema to fit the data, and
        routes as MERGE_SCHEMA rather than a plain append.

        `txn=(app_id, version)` makes the write idempotent. Note that neither
        engine enforces this: delta-rs records the transaction identifier but
        happily appends the same one twice. So the check happens here, by
        comparing against the last committed version before writing. That
        removes the common replay case; it is not a substitute for engine-level
        enforcement, because a concurrent writer could still commit in between.
        """
        if txn is not None and self._already_committed(txn):
            return
        needs = self._write_needs(schema_mode, commit_metadata, txn, writer_properties, "static")
        op = Operation.MERGE_SCHEMA if schema_mode == "merge" else Operation.APPEND
        self._engine(op, needs).append(
            self._resolved,
            data,
            schema_mode=schema_mode,
            partition_by=partition_by,
            target_file_size=target_file_size,
            writer_properties=writer_properties,
            commit_metadata=commit_metadata,
            txn=txn,
            max_commit_retries=max_commit_retries,
        )
        self._invalidate()

    def overwrite(
        self,
        data: Any,
        *,
        predicate: str | None = None,
        partition_overwrite: str = "static",
        schema_mode: str | None = None,
        target_file_size: int | None = None,
        writer_properties: Any = None,
        commit_metadata: dict[str, Any] | None = None,
        txn: tuple[str, int] | None = None,
        max_commit_retries: int | None = None,
    ) -> None:
        """Replace data.

        With `predicate`, replaces only matching rows (`replaceWhere`). With
        `partition_overwrite="dynamic"`, replaces exactly the partitions present
        in `data` and leaves the rest alone, which is Spark's
        `partitionOverwriteMode=dynamic`.
        """
        if txn is not None and self._already_committed(txn):
            return
        needs = self._write_needs(
            schema_mode, commit_metadata, txn, writer_properties, partition_overwrite
        )
        op = (
            Operation.REPLACE_WHERE
            if (predicate or partition_overwrite == "dynamic")
            else Operation.OVERWRITE
        )
        self._engine(op, needs).overwrite(
            self._resolved,
            data,
            predicate=predicate,
            partition_overwrite=partition_overwrite,
            schema_mode=schema_mode,
            target_file_size=target_file_size,
            writer_properties=writer_properties,
            commit_metadata=commit_metadata,
            txn=txn,
            max_commit_retries=max_commit_retries,
        )
        self._invalidate()

    def replace(self, data: Any, **kwargs: Any) -> None:
        """Replace the table's contents and schema. REPLACE TABLE / RTAS."""
        self.overwrite(data, schema_mode="overwrite", **kwargs)

    def _already_committed(self, txn: tuple[str, int]) -> bool:
        """True if `txn` was already committed, so the write should be skipped.

        Neither engine deduplicates on its own: delta-rs 1.6.5 records the txn
        action and appends anyway.
        """
        app_id, version = txn
        last = self.txn_version(app_id)
        return last is not None and version <= last

    def txn_version(self, app_id: str) -> int | None:
        """The last version committed under `app_id`, or None if never.

        Pair with `txn=` on a write to make a pipeline exactly-once::

            if t.txn_version("nightly-load") != batch_id:
                t.append(data, txn=("nightly-load", batch_id))
        """
        engine = self._engine(Operation.APPEND, frozenset({"idempotent_txn"}))
        version: int | None = engine.txn_version(self._resolved, app_id)
        return version

    def delete(self, predicate: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """DELETE rows matching a SQL predicate (every row when None)."""
        result: dict[str, Any] = self._engine(Operation.DELETE).delete(
            self._resolved, predicate, **_given(kwargs)
        )
        self._invalidate()
        return result

    def update(
        self,
        updates: dict[str, str] | None = None,
        *,
        new_values: dict[str, Any] | None = None,
        predicate: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """UPDATE. `updates` maps columns to SQL expressions; `new_values` to
        plain Python values, which need no quoting."""
        if new_values is not None:
            kwargs["new_values"] = new_values
        result: dict[str, Any] = self._engine(Operation.UPDATE).update(
            self._resolved, updates=updates, predicate=predicate, **_given(kwargs)
        )
        self._invalidate()
        return result

    def merge(self, source: Any, predicate: str, **kwargs: Any) -> Any:
        """MERGE INTO. Returns a builder with the delta-rs clause API
        (``when_matched_update_all()`` ... ``execute()``) whichever engine serves it."""
        builder = self._engine(Operation.MERGE).merge(self._resolved, source, predicate, **kwargs)
        return _InvalidatingMerger(builder, self._invalidate)

    # ------------------------------------------------------------ maintenance

    def optimize(
        self,
        *,
        zorder_by: list[str] | None = None,
        full: bool = False,
        predicate: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """OPTIMIZE: compaction, or Z-ordering with `zorder_by`.

        `full=True` is OPTIMIZE ... FULL, reclustering every file of a
        liquid-clustered table. `predicate` scopes the work (SQL fallback);
        delta-rs takes ``partition_filters=`` instead.
        """
        op = Operation.ZORDER if zorder_by else Operation.OPTIMIZE
        needs = set()
        if full:
            needs.add("optimize_full")
        if predicate is not None:
            needs.add("optimize_predicate")
        result: dict[str, Any] = self._engine(op, frozenset(needs)).optimize(
            self._resolved, zorder_by=zorder_by, full=full, predicate=predicate, **kwargs
        )
        self._invalidate()
        return result

    def z_order(self, columns: list[str], **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.ZORDER).zorder(
            self._resolved, columns, **kwargs
        )
        self._invalidate()
        return result

    def vacuum(
        self,
        *,
        retention_hours: int | None = None,
        dry_run: bool = True,
        lite: bool = False,
        **kwargs: Any,
    ) -> Any:
        """VACUUM. A dry run by default, because the real thing deletes files.

        `lite=True` considers only files the log records as removed (VACUUM
        LITE), which is cheaper than listing storage for orphans.
        """
        result = self._engine(Operation.VACUUM).vacuum(
            self._resolved, retention_hours=retention_hours, dry_run=dry_run, lite=lite, **kwargs
        )
        self._invalidate()
        return result

    def restore(self, target: Any, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.RESTORE).restore(
            self._resolved, target, **kwargs
        )
        self._invalidate()
        return result

    def repair(self, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.REPAIR).repair(self._resolved, **kwargs)
        self._invalidate()
        return result

    # ----------------------------------------------------------------- schema

    def add_column(self, fields: Any, **kwargs: Any) -> None:
        """Add columns. `fields` is a list of Arrow/Delta fields, or a
        {name: sql_type} mapping when the SQL fallback serves it."""
        self._engine(Operation.ADD_COLUMN).add_columns(self._resolved, fields, **kwargs)
        self._invalidate()

    def drop_column(self, column: str) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.DROP_COLUMN).drop_column(
            self._resolved, column
        )
        self._invalidate()
        return result

    def rename_column(self, old: str, new: str) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.RENAME_COLUMN).rename_column(
            self._resolved, old, new
        )
        self._invalidate()
        return result

    def set_properties(self, properties: dict[str, str], **kwargs: Any) -> None:
        self._engine(Operation.SET_PROPERTIES, properties=properties).set_properties(
            self._resolved, properties, **kwargs
        )
        self._invalidate()

    def add_feature(self, feature: Any, **kwargs: Any) -> None:
        names = list(feature) if isinstance(feature, (list, tuple, set, frozenset)) else [feature]
        self._engine(Operation.ADD_FEATURE, features=names).add_feature(
            self._resolved, feature, **kwargs
        )
        self._invalidate()

    def drop_feature(self, feature: str, **kwargs: Any) -> dict[str, Any]:
        """Drop a table feature. Databricks-only, so it needs the SQL fallback."""
        result: dict[str, Any] = self._engine(Operation.DROP_FEATURE).drop_feature(
            self._resolved, feature, **kwargs
        )
        self._invalidate()
        return result

    def add_constraint(self, constraints: dict[str, str], **kwargs: Any) -> None:
        self._engine(Operation.ADD_CONSTRAINT).add_constraint(self._resolved, constraints, **kwargs)
        self._invalidate()

    def drop_constraint(self, name: str, *, if_exists: bool = False) -> None:
        self._engine(Operation.DROP_CONSTRAINT).drop_constraint(
            self._resolved, name, if_exists=if_exists
        )
        self._invalidate()

    def unset_properties(self, keys: list[str] | str, *, if_exists: bool = True) -> None:
        """ALTER TABLE ... UNSET TBLPROPERTIES. delta-rs cannot remove a property."""
        names = [keys] if isinstance(keys, str) else list(keys)
        self._engine(Operation.UNSET_PROPERTIES).unset_properties(
            self._resolved, names, if_exists=if_exists
        )
        self._invalidate()

    def set_comment(self, comment: str | None) -> None:
        """The table comment (the Metadata action's description)."""
        self._engine(Operation.SET_COMMENT).set_comment(self._resolved, comment)
        self._invalidate()

    def set_column_comment(self, column: str, comment: str | None) -> None:
        self._engine(Operation.SET_COLUMN_COMMENT).set_column_comment(
            self._resolved, column, comment
        )
        self._invalidate()

    def alter_column_type(self, column: str, new_type: str) -> None:
        """Widen a column's type without rewriting data (type widening).

        Allowed: byte->short->int->long, float->double, byte/short/int->double,
        date->timestamp_ntz, and decimals whose precision and scale do not
        shrink. The table needs ``delta.enableTypeWidening = true``.
        """
        self._engine(Operation.ALTER_COLUMN_TYPE).alter_column_type(
            self._resolved, column, new_type
        )
        self._invalidate()

    def set_not_null(self, column: str) -> None:
        """Add a NOT NULL constraint, after checking no existing row is null."""
        self._engine(Operation.SET_NOT_NULL).set_not_null(self._resolved, column)
        self._invalidate()

    def drop_not_null(self, column: str) -> None:
        self._engine(Operation.DROP_NOT_NULL).drop_not_null(self._resolved, column)
        self._invalidate()

    def cluster_by(self, columns: list[str] | str | None) -> None:
        """Set the liquid-clustering keys (ALTER TABLE ... CLUSTER BY).

        ``None`` or ``[]`` is CLUSTER BY NONE. ``"auto"`` asks Databricks to
        choose keys, which only the SQL fallback can do. New keys apply to data
        written afterwards; existing files are reclustered by OPTIMIZE.
        """
        auto = isinstance(columns, str) and columns.lower() == "auto"
        needs = frozenset({"auto_clustering"}) if auto else frozenset()
        self._engine(Operation.CLUSTER_BY, needs).cluster_by(self._resolved, columns)
        self._invalidate()

    # ------------------------------------------------------- log and layout

    def checkpoint(self) -> None:
        self._engine(Operation.CHECKPOINT).checkpoint(self._resolved)

    def compact_logs(self, start: int | None = None, end: int | None = None) -> Any:
        return self._engine(Operation.LOG_COMPACTION).compact_logs(self._resolved, start, end)

    def cleanup_metadata(self) -> None:
        """Delete log files older than ``delta.logRetentionDuration``.

        This is what makes versions past log retention unreachable by time
        travel, so it is never done implicitly.
        """
        self._engine(Operation.CLEANUP_METADATA).cleanup_metadata(self._resolved)

    def analyze(self, *, columns: list[str] | None = None, delta_statistics: bool = False) -> Any:
        """ANALYZE TABLE. Databricks-only, so it needs the SQL fallback."""
        return self._engine(Operation.ANALYZE).analyze(
            self._resolved, columns=columns, delta_statistics=delta_statistics
        )

    def sync_iceberg(self) -> Any:
        """Regenerate UniForm Iceberg metadata (MSCK REPAIR TABLE ... SYNC METADATA).

        Needed after anything other than Databricks writes to a table with
        Iceberg reads enabled. Databricks-only, so it needs the SQL fallback.
        """
        return self._engine(Operation.SYNC_ICEBERG).sync_iceberg_metadata(self._resolved)

    def refresh(self, *, full: bool = False) -> Any:
        """REFRESH a materialized view or streaming table. Needs the SQL fallback."""
        return self._engine(Operation.REFRESH).refresh(self._resolved, full=full)

    def generate(self) -> None:
        """Write symlink manifests, for engines that read those instead of the log."""
        self._engine(Operation.GENERATE).generate(self._resolved)

    def reorg(self, **kwargs: Any) -> dict[str, Any]:
        """REORG TABLE. Databricks-only, so it needs the SQL fallback."""
        result: dict[str, Any] = self._engine(Operation.REORG).reorg(self._resolved, **kwargs)
        self._invalidate()
        return result

    def clone(self, target: str, **kwargs: Any) -> dict[str, Any]:
        """CLONE. Databricks-only, so it needs the SQL fallback."""
        result: dict[str, Any] = self._engine(Operation.CLONE).clone(
            self._resolved, target, **kwargs
        )
        self._invalidate()
        return result

    def publish(self) -> int:
        """Publish ratified-but-unpublished commits into `_delta_log/`.

        Required on catalog-managed tables, not optional housekeeping. The
        catalog caps how many unbackfilled commits it will hold and starts
        refusing writes past the limit, and checkpoints only run on published
        versions. `BackfillRequiredError` means do this.
        """
        version: int = self._engine(Operation.PUBLISH).publish(self._resolved)
        self._invalidate()
        return version

    # ------------------------------------------------------------- governance

    def _governance(self, what: str) -> Any:
        return _governed(self._connection.catalog, "GovernedCatalog", what)

    def info(self) -> Any:
        """The catalog's view of the table: owner, comment, columns, row filter,
        column masks, predictive optimization, audit timestamps."""
        cat = self._governance("read table info")
        return _call("read table info", cat.table_info, self._resolved.ref)

    def grants(self, principal: str | None = None) -> list[Any]:
        cat = self._governance("read grants")
        return list(_call("read grants", cat.grants, self._resolved.ref, principal))

    def effective_grants(self, principal: str | None = None) -> list[Any]:
        """Grants including those inherited from the schema and catalog."""
        cat = self._governance("read effective grants")
        return list(
            _call("read effective grants", cat.effective_grants, self._resolved.ref, principal)
        )

    def grant(self, principal: str, privileges: list[str] | str) -> list[Any]:
        names = [privileges] if isinstance(privileges, str) else list(privileges)
        cat = self._governance("grant")
        return list(_call("grant", cat.grant, self._resolved.ref, principal, names))

    def revoke(self, principal: str, privileges: list[str] | str) -> list[Any]:
        names = [privileges] if isinstance(privileges, str) else list(privileges)
        cat = self._governance("revoke")
        return list(_call("revoke", cat.revoke, self._resolved.ref, principal, names))

    def tags(self, column: str | None = None) -> dict[str, str]:
        cat = self._governance("read tags")
        return dict(_call("read tags", cat.tags, self._resolved.ref, column))

    def set_tags(self, tags: dict[str, str], *, column: str | None = None) -> None:
        cat = self._governance("set tags")
        _call("set tags", cat.set_tags, self._resolved.ref, tags, column)

    def unset_tags(self, keys: list[str] | str, *, column: str | None = None) -> None:
        names = [keys] if isinstance(keys, str) else list(keys)
        cat = self._governance("unset tags")
        _call("unset tags", cat.unset_tags, self._resolved.ref, names, column)

    def set_owner(self, principal: str) -> None:
        cat = self._governance("set owner")
        _call("set owner", cat.set_owner, self._resolved.ref, principal)

    def lineage(self, direction: str = "both") -> Any:
        """Upstream and downstream tables, notebooks, jobs and dashboards."""
        cat = self._governance("read lineage")
        return _call("read lineage", cat.lineage, self._resolved.ref, direction)

    def column_lineage(self, column: str, direction: str = "both") -> Any:
        cat = self._governance("read column lineage")
        return _call(
            "read column lineage", cat.column_lineage, self._resolved.ref, column, direction
        )

    def add_primary_key(self, name: str, columns: list[str], *, rely: bool = False) -> None:
        """An informational PRIMARY KEY constraint in Unity Catalog (not enforced)."""
        cat = self._governance("add a primary key")
        _call(
            "add a primary key", cat.add_primary_key, self._resolved.ref, name, columns, rely=rely
        )

    def add_foreign_key(
        self,
        name: str,
        columns: list[str],
        parent: str | Table,
        parent_columns: list[str],
        *,
        rely: bool = False,
    ) -> None:
        """An informational FOREIGN KEY constraint in Unity Catalog (not enforced)."""
        parent_ref = (
            parent.resolved.ref
            if isinstance(parent, Table)
            else parse_ref(
                parent,
                default_catalog=self._connection.default_catalog,
                default_schema=self._connection.default_schema,
            )
        )
        cat = self._governance("add a foreign key")
        _call(
            "add a foreign key",
            cat.add_foreign_key,
            self._resolved.ref,
            name,
            columns,
            parent_ref,
            parent_columns,
            rely=rely,
        )

    def drop_key_constraint(self, name: str, *, cascade: bool = False) -> None:
        """Drop an informational PRIMARY/FOREIGN KEY. CHECK constraints: `drop_constraint`."""
        cat = self._governance("drop a key constraint")
        _call(
            "drop a key constraint",
            cat.drop_table_constraint,
            self._resolved.ref,
            name,
            cascade=cascade,
        )

    def _warehouse(self, what: str) -> Any:
        engine = self._connection.router.engines.get(EngineKind.SQL)
        if engine is None or not self._connection.router.allow_sql_fallback:
            raise FallbackRequiredError(
                what,
                "row filters and column masks are defined in SQL and enforced by Databricks",
                SQL_FALLBACK_REMEDY,
            )
        return engine

    def set_row_filter(self, function_name: str, columns: list[str]) -> None:
        self._warehouse("set a row filter").set_row_filter(self._resolved, function_name, columns)

    def drop_row_filter(self) -> None:
        self._warehouse("drop a row filter").drop_row_filter(self._resolved)

    def set_column_mask(
        self, column: str, function_name: str, *, using_columns: list[str] | None = None
    ) -> None:
        self._warehouse("set a column mask").set_column_mask(
            self._resolved, column, function_name, using_columns
        )

    def drop_column_mask(self, column: str) -> None:
        self._warehouse("drop a column mask").drop_column_mask(self._resolved, column)

    # ------------------------------------------------------------ credentials

    def credentials(self, *, write: bool = False) -> Any:
        """The vended credential currently in force, for debugging.

        Secrets are redacted in `repr`; call `.secrets` deliberately if you
        really need them.
        """
        provider = self._resolved.credential_provider
        if provider is None:
            raise UnreachableTableError(
                "vend credentials",
                "this table has no credential provider (it is not "
                "governed by a catalog that vends them)",
            )
        op = CredentialOperation.READ_WRITE if write else CredentialOperation.READ
        return provider.credentials(op)


def _governed(catalog: Any, protocol_name: str, what: str) -> Any:
    """The catalog, if it implements `protocol_name`; else a refusal naming it."""
    from .catalog import base

    protocol = getattr(base, protocol_name)
    if not isinstance(catalog, protocol):
        raise UnreachableTableError(
            what,
            f"the {getattr(catalog, 'name', type(catalog).__name__)} catalog has no "
            "governance API for this",
            "Unity Catalog (Databricks or open source) provides it",
        )
    return catalog


def _call(what: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke a catalog method, turning 'this catalog lacks it' into a refusal."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        raise UnreachableTableError(what, str(exc) or "the catalog does not implement it") from exc


class _InvalidatingMerger:
    """Wraps a merge builder so the table forgets cached state once it executes.

    A MERGE runs at `execute()`, not when the builder is created, so
    invalidating any earlier would let a read in between re-cache the
    pre-merge protocol and properties.
    """

    def __init__(self, builder: Any, invalidate: Any) -> None:
        self._builder = builder
        self._invalidate = invalidate

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._builder, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            if name == "execute":
                self._invalidate()
                return result
            # Clause methods return the builder; keep the wrapper in the chain.
            if result is self._builder or type(result) is type(self._builder):
                self._builder = result
                return self
            return result

        return call
