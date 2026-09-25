"""The delta-rs engine: the default DML and maintenance path.

delta-rs has MERGE, UPDATE and DELETE by predicate, `replaceWhere`, schema
evolution, OPTIMIZE, Z-ORDER, VACUUM, RESTORE, FSCK, CONVERT, manifest
generation and `history()`, none of which the kernel implements.

Every operation opens a fresh `DeltaTable`. delta-rs bakes `storage_options`
into the object store at construction and has no credential-provider hook, so
re-opening is the only way to pick up a re-vended credential.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import threading
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from numbers import Integral
from typing import Any, Literal

from ..capability import (
    FEATURE_DEPENDENCIES,
    FEATURE_SUPPORT,
    OPERATION_ENGINES,
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
from ..errors import (
    CommitConflictError,
    EnginePanicError,
    InvalidArgumentError,
    UnreachableTableError,
)
from ..properties import effect_for, validate_properties
from .base import missing_method

__all__ = ["DeltaRsEngine"]

_RESTORE_DV_REASON = (
    "the table has deletion vectors, and delta-rs RESTORE silently fails to revert DV "
    "changes -- it reports success and leaves the rows deleted (delta-rs#4613)"
)

_READ_ONLY_OPS: frozenset[Operation] = frozenset(
    {
        Operation.SCAN,
        Operation.TIME_TRAVEL,
        Operation.CDF,
        Operation.HISTORY,
        Operation.DETAIL,
        # Listing add actions writes nothing; without this, files() on a
        # UniForm table or one with an unwritable writer feature was refused.
        Operation.FILES,
    }
)


class DeltaRsEngine:
    """Reads and writes Delta tables through the `deltalake` package."""

    kind = EngineKind.DELTARS
    supports_distributed_scan = False
    supports_predicates = True
    supports_timestamp_travel = True
    supports_schema_merge = True
    supports_schema_overwrite = True
    supports_idempotent_txn = True
    supports_commit_metadata = True
    supports_writer_properties = True
    #: Emulated: delta-rs has no partitionOverwriteMode, so we build a
    #: replaceWhere from the partition values present in the incoming data.
    supports_dynamic_overwrite = True
    #: delta-rs 1.x writes -1.5 as the partition value '-1.-50' and commits it,
    #: leaving the table unreadable, so such writes route elsewhere.
    supports_negative_decimal_partition_values = False

    def __init__(self, *, storage_options: dict[str, str] | None = None) -> None:
        self._base_options = dict(storage_options or {})

    # ----------------------------------------------------------- capabilities

    def supports(self, operation: Operation, table: ResolvedTable, **shape: Any) -> Capability:
        if os.getpid() not in _FORKED_RUNTIME and not self.available():
            return Capability(
                operation,
                ok=False,
                reason="the deltalake package is not installed",
                remedy="pip install deltalake",
            )
        if os.getpid() in _FORKED_RUNTIME:
            # Every call would fail the same way; let the next engine serve.
            return Capability(
                operation,
                ok=False,
                reason="delta-rs cannot run in this process: it was forked from one that "
                "had already started delta-rs's runtime",
                remedy="start worker processes with multiprocessing's 'spawn' or "
                "'forkserver' method",
            )

        # Structural guard: never claim an operation with no method behind it.
        gap = missing_method(self, operation)
        if gap is not None:
            return gap

        routing = OPERATION_ENGINES.get(operation)
        if routing is None or self.kind not in routing.engines:
            return Capability(
                operation,
                ok=False,
                reason=f"delta-rs does not implement {operation.value}"
                + (f" ({routing.rationale})" if routing and routing.rationale else ""),
            )

        if table.location is None:
            return Capability(
                operation,
                ok=False,
                reason="the table has no storage location, so there are no files to read",
            )

        if not table.is_delta:
            # The router guards this too; an engine must still never claim a
            # table whose data is not Delta at all (Iceberg, Parquet, ...).
            return Capability(
                operation,
                ok=False,
                reason=f"the table is {table.data_source_format}, not Delta",
            )

        if table.location.split("://", 1)[0].lower() in ("gs", "gcs") and _may_vend_gcs_bearer(
            table.credential_provider
        ):
            # Catalog-vended GCS credentials are OAuth bearer tokens, and the
            # installed delta-rs object store has no option that accepts one.
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table is on GCS with catalog-vended credentials (an OAuth bearer "
                    "token), which delta-rs's object store cannot use"
                ),
                remedy="this routes to the kernel engine",
            )

        if table.is_catalog_managed:
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table has the catalogManaged feature; delta-rs cannot open one "
                    "at all, because commits are ratified by the catalog rather than by "
                    "object-store listing (delta-rs#4549)"
                ),
                remedy="this routes to the kernel engine automatically",
            )

        if table.is_shallow_clone:
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table is a shallow clone, whose add actions reference the source "
                    "table's files by absolute path; delta-rs cannot read those"
                ),
                remedy="read the source table directly",
            )

        if _vends_gcs_bearer_token(table):
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table is on GCS and its credential is a vended OAuth bearer token, "
                    "which deltalake cannot use: it ignores the token and falls back to the "
                    "GCE metadata server"
                ),
                remedy="reads route to the kernel, whose store accepts bearer tokens",
            )

        writing = operation not in _READ_ONLY_OPS

        if writing and table.has_iceberg_compat:
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table has Iceberg reads enabled. An external write would require "
                    "MSCK REPAIR TABLE ... SYNC METADATA afterwards to regenerate Iceberg "
                    "metadata, which only Databricks can run -- so the Iceberg view would "
                    "silently go stale"
                ),
                remedy="perform this write from Databricks, or enable the SQL fallback",
            )

        # Reader features gate everything; writer features gate only writes.
        blockers: list[str] = []
        for name in table.effective_reader_features:
            feature = feature_from_wire(name)
            if feature is None or FEATURE_SUPPORT[feature].deltars_read is Support.NO:
                blockers.append(name)
        if writing:
            for name in table.effective_writer_features:
                feature = feature_from_wire(name)
                if feature is None or FEATURE_SUPPORT[feature].deltars_write is Support.NO:
                    blockers.append(name)

        if blockers:
            unique = sorted(set(blockers))
            hint = ""
            if "domainMetadata" in unique:
                hint = (
                    " (domainMetadata is required by row tracking and liquid clustering, "
                    "so every such table is delta-rs-unwritable)"
                )
            return Capability(
                operation,
                ok=False,
                reason=f"delta-rs cannot handle table features: {', '.join(unique)}{hint}",
                remedy="reads route to the kernel engine; writes need the SQL fallback",
            )

        if operation is Operation.CREATE and shape.get("cluster_by"):
            return Capability(
                operation,
                ok=False,
                reason="delta-rs cannot create a liquid-clustered table",
                remedy="the kernel sets clustering through its data layout",
            )

        if operation is Operation.FILES and "deletionVectors" in table.effective_reader_features:
            # get_add_actions() carries no deletion-vector descriptor, so each
            # file's num_records silently counted its deleted rows too.
            return Capability(
                operation,
                ok=False,
                reason=(
                    "delta-rs lists add actions without their deletion vectors, so "
                    "num_records would include deleted rows"
                ),
                remedy="this routes to the kernel engine, which lists them",
            )
        # Refusals that depend on the table's shape belong here rather than
        # inside the operation: raised at run time, they stop the router from
        # trying the next engine, which can often serve the same request.
        if operation is Operation.CDF:
            try:
                self._guard_cdf(table)
            except UnreachableTableError as exc:
                return Capability(operation, ok=False, reason=exc.reason, remedy=exc.remedy or "")
        if operation is Operation.RESTORE and "deletionVectors" in table.reader_features:
            return Capability(operation, ok=False, reason=_RESTORE_DV_REASON)
        if operation is Operation.ADD_FEATURE and shape.get("features") is not None:
            refusal = _add_feature_refusal(table, shape["features"])
            if refusal is not None:
                return Capability(
                    operation,
                    ok=False,
                    reason=refusal,
                    remedy="the kernel adds the feature together with its dependencies",
                )

        unusable = self._unusable_properties(shape.get("properties"), operation)
        if unusable:
            return Capability(
                operation,
                ok=False,
                reason=(
                    "delta-rs cannot handle these table properties: " + ", ".join(sorted(unusable))
                ),
                remedy="the kernel engine accepts most of them at create",
            )

        return Capability(operation, ok=True, engine=self.kind)

    @staticmethod
    def _unusable_properties(properties: Any, operation: Operation) -> list[str]:
        """Property keys this engine would reject or crash on."""
        if not properties:
            return []
        return [
            key for key in properties if not effect_for(key, EngineKind.DELTARS, operation).usable()
        ]

    @staticmethod
    def available() -> bool:
        try:
            import deltalake  # noqa: F401
        except ImportError:
            return False
        return os.getpid() not in _FORKED_RUNTIME

    # --------------------------------------------------------------- internals

    def _storage_options(self, table: ResolvedTable, *, write: bool) -> dict[str, str]:
        options = dict(self._base_options)
        if table.credential_provider is not None:
            op = CredentialOperation.READ_WRITE if write else CredentialOperation.READ
            options.update(table.credential_provider.credentials(op).as_storage_options())
            _pin_s3_endpoint(options)
        return options

    def _open(
        self,
        table: ResolvedTable,
        *,
        version: int | None = None,
        write: bool = False,
    ) -> Any:
        """Open a fresh DeltaTable. Never cached -- see the module docstring."""
        from deltalake import DeltaTable

        if table.location is None:
            raise UnreachableTableError("open", "the table has no storage location")
        options = _object_store_options(self._storage_options(table, write=write))
        with _no_panics("open the table"):
            dt = DeltaTable(table.location, version=version, storage_options=options)
        return dt

    # ------------------------------------------------------------------- read

    def scan(
        self,
        table: ResolvedTable,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: str | None = None,
        limit: int | None = None,
    ) -> Any:
        if version is not None and timestamp is not None:
            # Opening at `version` and then loading `timestamp` silently
            # served the timestamp and ignored the version.
            raise UnreachableTableError(
                "time travel by both version and timestamp",
                "a read can be pinned to a version or to a timestamp, not both",
            )
        dt = self._open(table, version=version)
        if timestamp is not None:
            dt.load_as_version(_timestamp_arg(timestamp))
        # `scan()` is the only delta-rs read path that handles deletion vectors
        # and column mapping; to_pyarrow_dataset() hard-rejects both.
        stream = dt.scan(
            columns=_canonical_columns(dt, columns),
            predicate=_datafusion_predicate(dt, predicate),
        )
        return _without_view_types(stream)

    def history(self, table: ResolvedTable, *, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None and limit < 0:
            # delta-rs surfaces this as "can't convert negative int to unsigned".
            raise InvalidArgumentError(f"history limit must be zero or more, got {limit}")
        dt = self._open(table)
        try:
            result: list[dict[str, Any]] | None = dt.history(limit)
            failure: Exception | None = None
        except Exception as exc:
            result, failure = None, exc
        rebuilt = _history_from_log(dt, limit, result)
        if rebuilt is not None:
            return rebuilt
        if failure is not None:
            raise failure
        assert result is not None
        return result

    def detail(self, table: ResolvedTable, *, version: int | None = None) -> dict[str, Any]:
        dt = self._open(table, version=version)
        protocol = dt.protocol()
        metadata = dt.metadata()
        return {
            # Table._check_identity compares this with the catalog's id; without
            # it a dropped-and-recreated table went unnoticed on this engine.
            "metadata_id": metadata.id,
            "version": dt.version(),
            "location": dt.table_uri,
            "min_reader_version": protocol.min_reader_version,
            "min_writer_version": protocol.min_writer_version,
            "reader_features": list(protocol.reader_features or []),
            "writer_features": list(protocol.writer_features or []),
            "properties": dict(metadata.configuration),
            "partition_columns": list(metadata.partition_columns),
        }

    def cdf(
        self,
        table: ResolvedTable,
        *,
        starting_version: int | None = None,
        ending_version: int | None = None,
        starting_timestamp: str | None = None,
        ending_timestamp: str | None = None,
        columns: list[str] | None = None,
        predicate: str | None = None,
        allow_out_of_range: bool = False,
    ) -> Any:
        self._guard_cdf(table)
        # delta-rs lets a version silently override a timestamp on the same
        # bound, so a caller passing both got a range they never asked for.
        for bound, ver, ts in (
            ("start", starting_version, starting_timestamp),
            ("end", ending_version, ending_timestamp),
        ):
            if ver is not None and ts is not None:
                raise UnreachableTableError(
                    "read the change data feed",
                    f"both a {bound} version and a {bound} timestamp were given; pass one",
                )
        dt = self._open(table)

        def load(start: int) -> Any:
            while True:
                try:
                    return _cdf_commit_times(
                        dt,
                        _without_view_types(
                            dt.load_cdf(
                                starting_version=start,
                                ending_version=ending_version,
                                starting_timestamp=(
                                    None
                                    if starting_timestamp is None
                                    else _timestamp_arg(starting_timestamp)
                                ),
                                ending_timestamp=(
                                    None
                                    if ending_timestamp is None
                                    else _timestamp_arg(ending_timestamp)
                                ),
                                columns=_cdf_columns(dt, columns),
                                predicate=_datafusion_predicate(dt, predicate, cdf=True),
                                allow_out_of_range=allow_out_of_range,
                            ),
                            keep=_canonical_columns(dt, columns),
                        ),
                    )
                except Exception as exc:
                    # The feed was switched off inside the range. delta-rs said
                    # so as a bare DeltaError the caller could not tell apart.
                    off = re.search(r"Change Data not enabled for version: (\d+)", str(exc))
                    if off is None:
                        raise
                    after = int(off.group(1)) + 1
                    if (
                        starting_version is None
                        and starting_timestamp is None
                        and after == start + 1
                        and (ending_version is None or after <= ending_version)
                    ):
                        # No start given and the feed is off at the start: it
                        # was enabled later, so begin there. A gap after it was
                        # on is refused below -- skipping it would drop changes.
                        start = after
                        continue
                    raise UnreachableTableError(
                        "read the change data feed",
                        f"the change data feed was not enabled at version {off.group(1)}, "
                        "which the requested range includes",
                        "start the range after the feed was re-enabled",
                    ) from exc

        def cleaned(exc: Exception, start: int) -> int | None:
            """The oldest retained commit, when log retention removed `start`."""
            if f"Invalid table version: {start}" not in str(exc):
                return None
            log = _DeltaLog.open(dt)
            earliest = log.commits[0][0] if log is not None and log.commits else None
            return earliest if earliest is not None and earliest > start else None

        if starting_version is not None or starting_timestamp is not None:
            start = starting_version if starting_version is not None else 0
            try:
                return load(start)
            except Exception as exc:
                earliest = cleaned(exc, start) if starting_version is not None else None
                if earliest is None:
                    raise
                raise UnreachableTableError(
                    "read the change data feed",
                    f"version {start} is no longer in the log (log retention removed it); "
                    f"the oldest commit still there is {earliest}",
                    f"pass starting_version={earliest} or later",
                ) from exc
        try:
            return load(0)
        except Exception as exc:
            earliest = cleaned(exc, 0)
            if earliest is not None:
                # No start given: the feed still readable begins at the oldest
                # retained commit, not at a version log retention removed.
                return load(earliest)
            # No start given and CDF was switched on after version 0: delta-rs
            # refuses to start at 0, so start where the feed actually begins.
            if "does not have change data enabled" not in str(exc):
                raise
            try:
                since = _cdf_enabled_since(dt)
            except Exception:
                since = None  # an unreadable commit: fall through to the clear error
            if since is None:
                raise UnreachableTableError(
                    "read the change data feed",
                    "change data feed was enabled after version 0 and the enabling "
                    "version could not be found in the log",
                    "pass starting_version= explicitly",
                ) from exc
            return load(since)

    def files(self, table: ResolvedTable, *, version: int | None = None) -> Any:
        """One row per live data file: path, size, partition values, stats.

        Flattened, so partition values and column statistics are ordinary
        columns (`partition.<col>`, `min.<col>`, `max.<col>`, `null_count.<col>`).
        """
        dt = self._open(table, version=version)
        actions = dt.get_add_actions(flatten=True)
        # Under column mapping delta-rs keys the statistics by *physical*
        # names (`min.col-<uuid>`); callers know only the logical ones.
        try:
            logical = _physical_to_logical(json.loads(dt.schema().to_json()))
        except Exception:
            logical = {}
        if not logical:
            return actions
        import pyarrow as pa

        arrow = pa.table(actions)
        return arrow.rename_columns([_logical_stat_name(n, logical) for n in arrow.column_names])

    @staticmethod
    def _guard_cdf(table: ResolvedTable) -> None:
        """Refuse CDF reads that would silently mislead.

        Two traps: a table whose data files can be vacuumed away while their
        commits survive (the read then fails deep in the reader with a missing
        file), and column mapping, which CDF does not support.
        """
        props = table.properties
        if props.get("delta.enableChangeDataFeed", "false").lower() != "true":
            raise UnreachableTableError(
                "read the change data feed",
                "delta.enableChangeDataFeed is not enabled on this table, and enabling "
                "it is not retroactive -- only changes after enablement are recorded",
            )
        if props.get("delta.columnMapping.mode", "none").lower() not in ("none", ""):
            raise UnreachableTableError(
                "read the change data feed",
                "the table uses column mapping, which is incompatible with CDF",
            )
        deleted = props.get("delta.deletedFileRetentionDuration")
        log = props.get("delta.logRetentionDuration")
        deleted_days = _duration_days(deleted) if deleted else None
        log_days = _duration_days(log) if log else None
        if deleted_days is not None and log_days is not None and deleted_days < log_days:
            raise UnreachableTableError(
                "read the change data feed",
                f"delta.deletedFileRetentionDuration ({deleted}) is shorter than "
                f"delta.logRetentionDuration ({log}), so data files can be vacuumed "
                "while their commits remain -- a CDF read over that range would fail "
                "on missing files",
                "raise deletedFileRetentionDuration to at least logRetentionDuration",
            )

    # ------------------------------------------------------------------ write

    def append(self, table: ResolvedTable, data: Any, **kwargs: Any) -> None:
        self._write(table, data, mode="append", **kwargs)

    def overwrite(
        self,
        table: ResolvedTable,
        data: Any,
        *,
        predicate: str | None = None,
        partition_overwrite: str = "static",
        **kwargs: Any,
    ) -> None:
        if partition_overwrite == "dynamic":
            if predicate is not None:
                raise UnreachableTableError(
                    "overwrite dynamically with a predicate",
                    "dynamic partition overwrite derives its own predicate from the "
                    "data, so it cannot be combined with an explicit one",
                )
            # Materialise once: deriving the predicate reads the data, and a
            # one-shot stream read twice wrote nothing after deleting the
            # partitions it named.
            data = _to_arrow_table(data)
            predicate = self._dynamic_partition_predicate(table, data)
        elif partition_overwrite != "static":
            raise UnreachableTableError(
                f"overwrite with partition_overwrite={partition_overwrite!r}",
                "the only modes are 'static' and 'dynamic'",
            )
        elif predicate is not None and predicate.strip():
            predicate = _datafusion_predicate(self._open(table), predicate)
        self._write(table, data, mode="overwrite", predicate=predicate, **kwargs)

    #: Refuse to build a predicate wider than this. A thousand OR-ed partition
    #: clauses is a sign the caller meant a full overwrite.
    max_dynamic_partitions = 1000

    def _dynamic_partition_predicate(self, table: ResolvedTable, data: Any) -> str:
        """Emulate `partitionOverwriteMode=dynamic`.

        delta-rs has no such mode, so we read the distinct partition tuples out
        of the incoming data and overwrite exactly those partitions with a
        replaceWhere. Anything not present in the data is left alone, which is
        the semantic Spark gives you.
        """
        partitions = list(table.partition_columns)
        if not partitions:
            raise UnreachableTableError(
                "overwrite dynamically",
                "the table is not partitioned, so there are no partitions to replace",
                "use a plain overwrite, or a predicate",
            )

        arrow = _to_arrow_table(data)
        missing = [c for c in partitions if c not in arrow.column_names]
        if missing:
            raise UnreachableTableError(
                "overwrite dynamically",
                f"the data has no {', '.join(missing)} column, so the partitions it "
                "would replace cannot be determined",
            )

        # Sorted by rendered literal: a NULL partition next to a real value
        # made plain sorted() raise TypeError comparing None with str.
        tuples = sorted(
            {
                tuple(row[column] for column in partitions)
                for row in arrow.select(partitions).to_pylist()
            },
            key=lambda values: tuple(_sql_literal(v) for v in values),
        )
        if not tuples:
            raise UnreachableTableError(
                "overwrite dynamically", "the data is empty, so no partitions are implied"
            )
        if len(tuples) > self.max_dynamic_partitions:
            raise UnreachableTableError(
                "overwrite dynamically",
                f"the data spans {len(tuples)} partitions, above the "
                f"{self.max_dynamic_partitions} limit for a generated predicate",
                "overwrite the whole table, or raise DeltaRsEngine.max_dynamic_partitions",
            )

        clauses = [
            "("
            + " AND ".join(
                # `col = NULL` is never true (delta-rs rejects it outright), and
                # an unquoted name breaks on spaces, dots and reserved words.
                f"{_sql_ident(col)} IS NULL"
                if val is None
                else f"{_sql_ident(col)} = {_sql_literal(val)}"
                for col, val in zip(partitions, values, strict=True)
            )
            + ")"
            for values in tuples
        ]
        return " OR ".join(clauses)

    def _write(
        self,
        table: ResolvedTable,
        data: Any,
        *,
        mode: Literal["append", "overwrite"],
        schema_mode: str | None = None,
        predicate: str | None = None,
        partition_by: list[str] | None = None,
        target_file_size: int | None = None,
        writer_properties: Any = None,
        commit_metadata: dict[str, Any] | None = None,
        txn: tuple[str, int] | None = None,
        max_commit_retries: int | None = None,
        configuration: dict[str, str] | None = None,
    ) -> None:
        from deltalake import write_deltalake

        if table.location is None:
            raise UnreachableTableError("write", "the table has no storage location")
        if configuration:
            # delta-rs accepts this and then discards it, so saying nothing
            # would leave the caller believing the properties were applied.
            raise UnreachableTableError(
                "set table properties during a write",
                "delta-rs silently ignores configuration= on a write",
                "call set_properties() separately, or pass properties at create",
            )

        # delta-rs overloads write_deltalake on `mode`: only the overwrite
        # signature accepts `predicate`, so the two are called separately rather
        # than passing a union that matches neither.
        if predicate is not None and not predicate.strip():
            # delta-rs answers with an opaque SQL parser error.
            raise InvalidArgumentError(
                "overwrite predicate is empty; pass None to replace the whole table"
            )
        data = _plain_data(data)
        _refuse_long_float_partitions(self, table, data, partition_by)
        common: dict[str, Any] = {
            "schema_mode": schema_mode,
            "partition_by": partition_by,
            "target_file_size": target_file_size,
            "writer_properties": _writer_properties(writer_properties),
            "commit_properties": _commit_properties(commit_metadata, txn, max_commit_retries),
            "storage_options": _object_store_options(self._storage_options(table, write=True)),
        }
        extra: dict[str, Any] = {"predicate": predicate} if mode == "overwrite" else {}
        if mode == "overwrite" and max_commit_retries is None:
            # delta-rs's rebase misses a concurrent compaction's removes: an
            # overwrite that lost the race to an OPTIMIZE committed without
            # removing the compacted file, so the "replaced" rows were still
            # there next to the new ones. Committed on its own snapshot, the
            # overwrite conflicts instead (CommitConflictError).
            common["commit_properties"] = _commit_properties(commit_metadata, txn, 0)
        if txn is None:
            with _no_panics(f"{mode} to the table"):
                write_deltalake(table.location, data, mode=mode, **common, **extra)
            return
        # delta-rs's own commit retry does not look at transaction ids: a writer
        # that lost the race to one committing the same (app_id, version)
        # rebased onto it and committed the batch a second time. So with txn=
        # each attempt commits against the snapshot its txn check read, with
        # delta-rs retries off, and a lost race is re-checked here.
        retries = 15 if max_commit_retries is None else max(0, int(max_commit_retries))
        replayable = hasattr(data, "to_reader") or type(data).__name__ == "RecordBatch"
        # Only an append is re-run: an overwrite re-run on a newer snapshot would
        # silently remove whatever the writer that beat it just committed.
        attempts = 1 + retries if replayable and mode == "append" else 1
        common["commit_properties"] = _commit_properties(commit_metadata, txn, 0)
        del common["storage_options"]  # the DeltaTable carries them
        for attempt in range(attempts):
            dt = self._open(table, write=True)
            last = dt.transaction_version(txn[0])
            if last is not None and int(last) >= int(txn[1]):
                raise CommitConflictError(
                    int(dt.version()),
                    f"transaction {txn[0]!r} version {txn[1]} was committed by a concurrent "
                    f"writer (the table records version {last}); this batch is already "
                    "in the table",
                )
            try:
                with _no_panics(f"{mode} to the table"):
                    write_deltalake(dt, data, mode=mode, **common, **extra)
            except CommitConflictError:
                if attempt + 1 >= attempts:
                    raise
                continue
            return

    def txn_version(self, table: ResolvedTable, app_id: str) -> int | None:
        """The last version committed under `app_id`, or None.

        This is what makes an exactly-once pipeline possible: read the last
        version you committed, and skip the work if it is already there.
        """
        version: int | None = self._open(table).transaction_version(app_id)
        return version

    def create(
        self,
        table: ResolvedTable,
        schema: Any,
        *,
        partition_by: list[str] | None = None,
        mode: Literal["error", "append", "overwrite", "ignore"] = "error",
        properties: dict[str, str] | None = None,
        name: str | None = None,
        description: str | None = None,
        cluster_by: list[str] | None = None,
    ) -> None:
        from deltalake import DeltaTable

        if table.location is None:
            raise UnreachableTableError("create", "no storage location was given for the new table")
        if cluster_by:
            raise UnreachableTableError(
                "create a clustered table with delta-rs",
                "delta-rs has no liquid clustering support",
                "the kernel sets clustering columns through its data layout",
            )
        # Check first: delta-rs reports every property problem with one opaque
        # message, and panics on delta.minReaderVersion.
        validate_properties(properties, EngineKind.DELTARS, Operation.CREATE)
        with _no_panics("create the table"):
            DeltaTable.create(
                table.location,
                schema,
                mode=mode,
                partition_by=partition_by,
                name=name or (table.ref.table if table.ref.kind.value == "catalog" else None),
                description=description,
                configuration=properties,
                storage_options=_object_store_options(self._storage_options(table, write=True)),
            )

    def delete(
        self,
        table: ResolvedTable,
        predicate: str | None = None,
        *,
        commit_metadata: dict[str, Any] | None = None,
        writer_properties: Any = None,
        max_commit_retries: int | None = None,
    ) -> dict[str, Any]:
        if predicate is not None and not predicate.strip():
            # delta-rs answers with an opaque SQL parser error.
            raise InvalidArgumentError("delete predicate is empty; pass None to delete every row")
        if predicate is None and max_commit_retries is None:
            # A delete of every row reads nothing, so delta-rs's rebase checks
            # nothing: one that lost the race to an OPTIMIZE removed only the
            # pre-compaction files and reported success with every row still
            # in the compacted one. On its own snapshot it conflicts instead.
            max_commit_retries = 0
        with _no_panics("delete"):
            dt = self._open(table, write=True)
            result: dict[str, Any] = dt.delete(
                _datafusion_predicate(dt, predicate, dml="delete"),
                writer_properties=_writer_properties(writer_properties),
                commit_properties=_commit_properties(commit_metadata, None, max_commit_retries),
            )
        if "num_deleted_rows" not in result:
            # A delete that only drops whole files (a partition predicate) on
            # files without statistics reports no row count at all, and
            # result["num_deleted_rows"] raised KeyError on such tables.
            result["num_deleted_rows"] = _rows_removed(dt, result)
        return result

    def update(
        self,
        table: ResolvedTable,
        *,
        updates: dict[str, str] | None = None,
        new_values: dict[str, Any] | None = None,
        predicate: str | None = None,
        commit_metadata: dict[str, Any] | None = None,
        writer_properties: Any = None,
        error_on_type_mismatch: bool = True,
        max_commit_retries: int | None = None,
    ) -> dict[str, Any]:
        if updates is not None and new_values is not None:
            raise InvalidArgumentError("pass updates (SQL expressions) or new_values, not both")
        if new_values is not None:
            # delta-rs renders these itself, and gets it wrong: no quote
            # escaping, no NULL, naive datetimes shifted by the local timezone,
            # NaN as a column name, and no date/Decimal/bytes at all.
            updates = {column: _sql_value(value) for column, value in new_values.items()}
        if not updates:
            # An empty mapping is a silent no-op in delta-rs.
            raise InvalidArgumentError("update needs at least one column to set")
        with _no_panics("update"):
            dt = self._open(table, write=True)
            updates = _datafusion_updates(dt, updates, rendered=new_values is not None)
            if new_values is None:
                names = _column_names(dt)
                updates = {
                    k: _exact_decimals(_fold_case(v, {None: names})) for k, v in updates.items()
                }
            updates = _with_generated(dt, updates)
            result: dict[str, Any] = dt.update(
                updates=updates,
                predicate=_datafusion_predicate(dt, predicate, dml="update"),
                writer_properties=_writer_properties(writer_properties),
                error_on_type_mismatch=error_on_type_mismatch,
                commit_properties=_commit_properties(commit_metadata, None, max_commit_retries),
            )
        return result

    def merge(
        self,
        table: ResolvedTable,
        source: Any,
        predicate: str,
        *,
        commit_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if commit_metadata is not None:
            kwargs["commit_metadata"] = commit_metadata
        _commit_kwargs(kwargs)
        if isinstance(kwargs.get("writer_properties"), dict):
            kwargs["writer_properties"] = _writer_properties(kwargs["writer_properties"])
        dt = self._open(table, write=True)
        data = _respell_source(_plain_data(source), _column_names(dt))
        columns = _merge_columns(dt, data, kwargs)
        if isinstance(predicate, str):
            predicate = _exact_decimals(_fold_case(predicate, columns))
        merger = dt.merge(data, predicate, **kwargs)
        return _CheckedMerger(
            merger,
            columns,
            _column_names(dt),
            generated=_generated_columns(dt),
            source_alias=kwargs.get("source_alias"),
            target_alias=kwargs.get("target_alias"),
        )

    # ------------------------------------------------------------ maintenance

    def optimize(
        self,
        table: ResolvedTable,
        *,
        zorder_by: list[str] | str | None = None,
        full: bool = False,
        predicate: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if predicate is not None:
            raise UnreachableTableError(
                "optimize with a SQL predicate",
                "delta-rs scopes OPTIMIZE by partition filters, not by a SQL predicate",
                "pass partition_filters=[('col', '=', 'value')] instead",
            )
        if full:
            raise UnreachableTableError(
                "OPTIMIZE ... FULL",
                "a full reclustering rewrite is a liquid-clustering operation, and delta-rs "
                "cannot write liquid-clustered tables",
            )
        if isinstance(zorder_by, str):
            # delta-rs calls list() on it, turning "region" into r, e, g, ...
            zorder_by = [zorder_by]
        _commit_kwargs(kwargs)
        if zorder_by:
            order = list(zorder_by)
            return self._rewrite(
                table, "optimize", lambda dt, kw: dt.optimize.z_order(order, **kw), kwargs
            )
        return self._rewrite(table, "optimize", lambda dt, kw: dt.optimize.compact(**kw), kwargs)

    def _rewrite(
        self, table: ResolvedTable, what: str, run: Any, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a delta-rs compaction, refusing to leave its data duplicated.

        delta-rs 1.6.5's conflict check ignores a concurrent commit's
        ``dataChange=false`` removes: two OPTIMIZE runs over the same files
        both commit, and every compacted row is then in the table twice
        (verified: 5 rows became 10). In one process the runs are serialised
        here; across processes the commit is found by a tag and checked, and
        rolled back while it is still the latest one.
        """
        import uuid

        tag = uuid.uuid4().hex
        kwargs = dict(kwargs)
        kwargs["commit_properties"] = _tagged(kwargs.get("commit_properties"), tag)
        with _rewrite_lock(table.location or ""):
            dt = self._open(table, write=True)
            before = int(dt.version())
            with _no_panics(what):
                result: dict[str, Any] = run(dt, kwargs)
            self._check_rewrite(table, before, tag, result, what)
        return result

    def _check_rewrite(
        self, table: ResolvedTable, before: int, tag: str, result: Any, what: str
    ) -> None:
        from ..errors import CorruptTableError

        try:
            claimed = int((result or {}).get("numFilesRemoved") or 0)
        except (TypeError, ValueError, AttributeError):
            return
        if not claimed:
            return
        latest = self._open(table, write=True)
        entry = next(
            (
                e
                for e in latest.history(max(1, int(latest.version()) - before))
                if e.get(_REWRITE_TAG) == tag
            ),
            None,
        )
        if entry is None or entry.get("version") is None:
            return  # cannot find it; nothing to compare
        version, read = int(entry["version"]), entry.get("readVersion")
        if read is not None and version == int(read) + 1:
            return  # committed on the snapshot it planned from: nothing was rebased
        try:
            # Acted on only while this compaction is still the newest commit:
            # history's numbering further back proved unreliable, and a later
            # writer's commit must never be rolled back with it.
            newest = latest.history(1)
            if not newest or newest[0].get(_REWRITE_TAG) != tag:
                return
            version = int(latest.version())
        except Exception:
            return
        previous = set(self._open(table, version=version - 1).file_uris())
        current = set(self._open(table, version=version).file_uris())
        if len(previous - current) >= claimed:
            return
        if int(latest.version()) == version and "deletionVectors" not in table.reader_features:
            from deltalake import CommitProperties

            with _no_panics(f"roll back the duplicating {what}"):
                latest.restore(
                    version - 1, commit_properties=CommitProperties(max_commit_retries=0)
                )
            raise CommitConflictError(
                version,
                f"a concurrent OPTIMIZE compacted the same files first, and delta-rs "
                f"committed this {what} anyway (version {version}), duplicating their rows; "
                f"it has been rolled back to version {version - 1}. Nothing is lost; re-run "
                f"the {what} if there is still something to compact",
            )
        raise CorruptTableError(
            f"{what} committed version {version} over a concurrent compaction of the same "
            f"files (delta-rs does not detect that conflict), so rows compacted by both are "
            f"now in the table twice. Restore the table to version {version - 1} and "
            "re-apply any commits made after it"
        )

    def zorder(
        self, table: ResolvedTable, columns: list[str] | str, **kwargs: Any
    ) -> dict[str, Any]:
        if isinstance(columns, str):
            columns = [columns]
        if not columns:
            raise InvalidArgumentError("Z-ORDER needs at least one column")
        _commit_kwargs(kwargs)
        order = list(columns)
        return self._rewrite(
            table, "z-order", lambda dt, kw: dt.optimize.z_order(order, **kw), kwargs
        )

    def vacuum(
        self,
        table: ResolvedTable,
        *,
        retention_hours: float | None = None,
        dry_run: bool = True,
        lite: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """VACUUM. `lite=True` only considers files the log says were removed.

        delta-rs calls the two modes `full=True` (Delta's standard VACUUM,
        which also lists storage for unreferenced files) and `full=False`
        (Databricks' VACUUM LITE). The default here is the standard one, so the
        same call means the same thing on every engine.
        """
        if table.is_shallow_clone:
            raise UnreachableTableError(
                "vacuum",
                "the table is a shallow clone and borrows the source table's files; "
                "vacuuming it risks deleting data the source still owns",
            )
        if retention_hours is not None and not isinstance(retention_hours, int):
            # delta-rs takes whole hours only and raises TypeError on 0.5.
            # Round up: keeping a little more history is the safe direction.
            retention_hours = math.ceil(retention_hours)
        _commit_kwargs(kwargs)
        with _no_panics("vacuum"):
            result: list[str] = self._open(table, write=True).vacuum(
                retention_hours=retention_hours, dry_run=dry_run, full=not lite, **kwargs
            )
        return result

    def restore(self, table: ResolvedTable, target: Any, **kwargs: Any) -> dict[str, Any]:
        if isinstance(target, bool) or not isinstance(target, (Integral, str, datetime, date)):
            # bool is an int: restore(True) silently restored version 1.
            raise TypeError(
                f"restore target must be a version or a timestamp, got {type(target).__name__}"
            )
        target = int(target) if isinstance(target, Integral) else _timestamp_arg(target)
        _commit_kwargs(kwargs)
        if "deletionVectors" in table.reader_features:
            # delta-rs 0.31+ proceeds without error and leaves DV changes in
            # place, so the table reads wrong afterwards (delta-rs#4613).
            raise UnreachableTableError(
                "restore", _RESTORE_DV_REASON, "perform the restore from Databricks"
            )
        with _no_panics("restore"):
            result: dict[str, Any] = self._open(table, write=True).restore(target, **kwargs)
        return result

    def repair(self, table: ResolvedTable, **kwargs: Any) -> dict[str, Any]:
        _commit_kwargs(kwargs)
        with _no_panics("repair"):
            result: dict[str, Any] = self._open(table, write=True).repair(**kwargs)
        return result

    # ---------------------------------------------------------------- schema

    def add_columns(self, table: ResolvedTable, fields: Any, **kwargs: Any) -> None:
        converted = _deltars_fields(fields)
        required = [f.name for f in converted if not f.nullable]
        if required:
            # delta-rs accepts this, and every existing file then lacks a
            # non-nullable column: the table no longer reads at all.
            raise UnreachableTableError(
                f"add NOT NULL column(s) {', '.join(required)}",
                "existing rows have no value for a new column, so it must be nullable",
                "add the column as nullable, backfill it, then set NOT NULL",
            )
        _commit_kwargs(kwargs)
        self._alter(table, "add columns", lambda dt: dt.alter.add_columns(converted, **kwargs))

    #: Attempts for a metadata change that keeps losing to concurrent ones.
    metadata_commit_attempts = 5

    def _alter(self, table: ResolvedTable, what: str, change: Any) -> None:
        """Commit a metadata change, recomputed on a fresh snapshot after a lost race.

        delta-rs does not rebase an ALTER over a concurrent metadata change
        ("Metadata changed since last commit"): with five processes adding
        columns, most failed where the kernel's metadata path, which
        recomputes the change against the new state, succeeded every time.
        """
        for attempt in range(self.metadata_commit_attempts):
            dt = self._open(table, write=True)
            try:
                with _no_panics(what):
                    change(dt)
                return
            except CommitConflictError:
                if attempt + 1 >= self.metadata_commit_attempts:
                    raise

    def set_properties(
        self, table: ResolvedTable, properties: dict[str, str], **kwargs: Any
    ) -> None:
        validate_properties(properties, EngineKind.DELTARS, Operation.SET_PROPERTIES)
        _commit_kwargs(kwargs)
        self._alter(
            table,
            "set table properties",
            lambda dt: dt.alter.set_table_properties(properties, **kwargs),
        )

    def add_feature(self, table: ResolvedTable, feature: Any, **kwargs: Any) -> None:
        from deltalake import TableFeatures

        names = feature if isinstance(feature, (list, tuple, set, frozenset)) else [feature]
        members = []
        for name in names:
            wire = _wire_name(name)
            if wire not in _DELTARS_FEATURES:
                raise UnreachableTableError(
                    f"add table feature {wire!r} with delta-rs",
                    "delta-rs's TableFeatures has no member for it",
                )
            members.append(getattr(TableFeatures, _DELTARS_FEATURES[wire]))
        kwargs.setdefault("allow_protocol_versions_increase", True)
        self._open(table, write=True).alter.add_feature(members, **kwargs)

    def add_constraint(
        self, table: ResolvedTable, constraints: dict[str, str], **kwargs: Any
    ) -> None:
        _commit_kwargs(kwargs)
        self._alter(
            table, "add a constraint", lambda dt: dt.alter.add_constraint(constraints, **kwargs)
        )

    def drop_constraint(self, table: ResolvedTable, name: str, *, if_exists: bool = False) -> None:
        self._alter(
            table,
            "drop a constraint",
            lambda dt: dt.alter.drop_constraint(name, raise_if_not_exists=not if_exists),
        )

    def set_comment(self, table: ResolvedTable, comment: str | None) -> None:
        self._alter(
            table,
            "set the table comment",
            lambda dt: dt.alter.set_table_description(comment or ""),
        )

    def set_column_comment(self, table: ResolvedTable, column: str, comment: str | None) -> None:
        dt = self._open(table, write=True)
        # A dot means a nested path only when no top-level column has that
        # name; "a.b" is a legal column name and was refused outright.
        if "." in column and column not in dt.schema().to_arrow().names:
            raise UnreachableTableError(
                f"comment on nested column {column!r} with delta-rs",
                "delta-rs sets field metadata on top-level columns only",
            )
        self._alter(
            table,
            "set a column comment",
            lambda fresh: fresh.alter.set_column_metadata(column, {"comment": comment or ""}),
        )

    def drop_not_null(self, table: ResolvedTable, column: str) -> None:
        """DROP NOT NULL. A no-op on a column that is already nullable, as in Spark."""
        try:
            self._alter(table, "drop NOT NULL", lambda dt: dt.alter.drop_column_not_null(column))
        except Exception as exc:
            if "already nullable" not in str(exc):
                raise

    # ----------------------------------------------------------- log upkeep

    def checkpoint(self, table: ResolvedTable) -> None:
        self._open(table, write=True).create_checkpoint()

    def compact_logs(
        self, table: ResolvedTable, start: int | None = None, end: int | None = None
    ) -> Any:
        # delta-rs wants concrete versions and rejects a degenerate range, so a
        # table with a single commit has nothing to compact.
        if start is not None and end is not None and end < start:
            # An inverted range the caller spelled out is a mistake, not a
            # table with nothing to compact.
            raise InvalidArgumentError(f"compact_logs range is inverted: start={start} > end={end}")
        dt = self._open(table, write=True)
        first = 0 if start is None else start
        last = dt.version() if end is None else end
        if last <= first:
            return None
        return dt.compact_logs(first, last)

    def cleanup_metadata(self, table: ResolvedTable) -> None:
        """Delete log files older than `delta.logRetentionDuration`.

        Log entries past retention are what keep old versions time-travelable,
        so this is the step that makes them unreachable.
        """
        self._open(table, write=True).cleanup_metadata()

    def generate(self, table: ResolvedTable) -> None:
        """Write symlink manifests for engines that read them (Presto, Athena)."""
        blocker = next(
            (
                why
                for feature, why in (
                    ("deletionVectors", "manifest readers would return the deleted rows"),
                    ("columnMapping", "manifest readers would see physical column names"),
                )
                if feature in table.features
            ),
            None,
        )
        if blocker is not None:
            # Spark refuses both; delta-rs writes the manifest regardless.
            raise UnreachableTableError(
                "generate a symlink manifest", f"the table uses a feature for which {blocker}"
            )
        self._open(table, write=True).generate()

    def convert(
        self,
        location: str,
        *,
        partition_by: Any = None,
        partition_strategy: Literal["hive", "directory"] = "hive",
        **kwargs: Any,
    ) -> None:
        """Turn a directory of Parquet into a Delta table in place."""
        from deltalake import convert_to_deltalake

        convert_to_deltalake(
            location,
            partition_by=partition_by,
            partition_strategy=partition_strategy,
            # Merged, not duplicated: a caller's storage_options= raised
            # "got multiple values for keyword argument".
            storage_options={**self._base_options, **(kwargs.pop("storage_options", None) or {})}
            or None,
            **kwargs,
        )

    def plan_scan(self, table: ResolvedTable, **kwargs: Any) -> list[Any]:
        raise NotImplementedError("the delta-rs engine has no split-planning surface")

    def execute_scan(self, table: ResolvedTable, splits: list[Any], **kwargs: Any) -> Any:
        raise NotImplementedError("see plan_scan")


#: Wire name -> member of `deltalake.TableFeatures`, which is all delta-rs's
#: `alter.add_feature` accepts; it raises a TypeError on a plain string.
_DELTARS_FEATURES: dict[str, str] = {
    "appendOnly": "AppendOnly",
    "changeDataFeed": "ChangeDataFeed",
    "checkConstraints": "CheckConstraints",
    "columnMapping": "ColumnMapping",
    "deletionVectors": "DeletionVectors",
    "domainMetadata": "DomainMetadata",
    "generatedColumns": "GeneratedColumns",
    "icebergCompatV1": "IcebergCompatV1",
    "identityColumns": "IdentityColumns",
    "invariants": "Invariants",
    "rowTracking": "RowTracking",
    "timestampNanos": "TimestampNanos",
    "timestampNtz": "TimestampWithoutTimezone",
    "v2Checkpoint": "V2Checkpoint",
    "variantType": "VariantType",
    "variantType-preview": "VariantTypePreview",
}


def _wire_name(feature: Any) -> str:
    value = getattr(feature, "value", feature)
    return str(value)


def _add_feature_refusal(table: ResolvedTable, features: Any) -> str | None:
    """Why delta-rs must not add these features, or None if it may.

    delta-rs adds whatever it is told, including features it cannot then write
    and without their dependencies: rowTracking arrives without domainMetadata,
    and the result is a table neither engine will write.
    """
    names = features if isinstance(features, (list, tuple, set, frozenset)) else [features]
    wires = {_wire_name(n) for n in names}
    present = set(table.reader_features) | set(table.writer_features) | wires
    for wire in sorted(wires):
        feature = feature_from_wire(wire)
        if feature is None or wire not in _DELTARS_FEATURES:
            return f"delta-rs cannot add {wire!r}: it has no such table feature"
        if FEATURE_SUPPORT[feature].deltars_write is not Support.YES:
            return f"adding {wire!r} with delta-rs would leave a table delta-rs cannot write"
        missing = {d.value for d in FEATURE_DEPENDENCIES.get(feature, frozenset())} - present
        if missing:
            return (
                f"{wire!r} requires {', '.join(sorted(missing))}, which delta-rs does not "
                "add alongside it"
            )
    return None


def _vends_gcs_bearer_token(table: ResolvedTable) -> bool:
    """Whether reaching this table means a GCS OAuth bearer token.

    Both Unity Catalog flavours vend GCS access as a raw token. object_store has
    no option key for one -- the native crate builds its GCS store by hand for
    exactly this -- and deltalake silently drops it.
    """
    location = table.location or ""
    provider = table.credential_provider
    if not location.startswith("gs://") or provider is None:
        return False
    from ..credentials.base import StaticCredentialProvider

    if isinstance(provider, StaticCredentialProvider):
        # Held, not minted, so reading it costs nothing -- and an expired one
        # must not turn a routing question into a raise.
        return "google_bearer_token" in provider.peek().secrets
    return True


_S3_ENDPOINT_KEYS = frozenset({"aws_endpoint", "aws_endpoint_url", "endpoint", "endpoint_url"})


def _pin_s3_endpoint(options: dict[str, str]) -> None:
    """Give vended S3 keys an explicit regional endpoint.

    With no endpoint, delta-rs treats the store as real AWS and builds an AWS SDK
    config via `aws_config::from_env()`, whose region chain ignores the
    `aws_region` we pass and ends at EC2 instance metadata. Off EC2 that costs
    three one-second connect timeouts on the first open in every process, and
    the SDK config is pointless here: the credentials are already static. An
    endpoint makes delta-rs hand the keys straight to object_store instead.
    """
    region = options.get("aws_region")
    if not region or "aws_access_key_id" not in options:
        return
    if any(k.lower() in _S3_ENDPOINT_KEYS for k in options):
        return
    suffix = "amazonaws.com.cn" if region.startswith("cn-") else "amazonaws.com"
    options["aws_endpoint"] = f"https://s3.{region}.{suffix}"


def _sql_literal(value: Any) -> str:
    """Render a partition value as a SQL literal.

    Dates and timestamps render as quoted strings, which DataFusion casts to
    the column's type when comparing.
    """
    if isinstance(value, (datetime, date)):
        return _sql_string(str(value))
    return _sql_value(value)


def _datafusion_updates(dt: Any, updates: dict[str, str], *, rendered: bool) -> dict[str, str]:
    """UPDATE assignments with columns spelled as the table spells them, and
    each plain-value right-hand side (Spark SQL) rendered as DataFusion reads it.

    delta-rs parsed `'a\\b'` without Spark's escapes, `"text"` as a column,
    `12345678901234.5678` as a DOUBLE (so a DECIMAL(18,4) column received
    ...5680), and rejected TIMESTAMP_NTZ. Expressions the parser does not
    cover pass through as DataFusion SQL; `rendered` values (from
    new_values=) are DataFusion already.
    """
    import pyarrow as pa

    from .. import predicate as sqlpred

    keys = _canonical_columns(dt, list(updates)) or []
    try:
        schema = pa.schema(dt.schema().to_arrow())
    except Exception:
        return dict(zip(keys, updates.values(), strict=True))
    out: dict[str, str] = {}
    for key, expression in zip(keys, updates.values(), strict=True):
        out[key] = expression
        if rendered or schema.get_field_index(key) < 0:
            continue
        try:
            value = sqlpred.parse_value(expression)
        except sqlpred.PredicateError:
            continue
        text = sqlpred.to_datafusion_value(value, schema, schema.field(key).type)
        if text is not None:
            out[key] = text
    return out


def _canonical_columns(dt: Any, columns: list[str] | None) -> list[str] | None:
    """`columns` spelled as the table spells them.

    Delta column names are case-insensitive, and the kernel path resolves
    `columns=["ID"]` against `id`; delta-rs failed with "No field named ID".
    """
    if not columns:
        return columns
    try:
        names = [f.name for f in dt.schema().fields]
    except Exception:
        return columns
    folded: dict[str, list[str]] = {}
    for name in names:
        folded.setdefault(name.lower(), []).append(name)
    out = []
    for column in columns:
        matches = folded.get(column.lower(), []) if column not in names else [column]
        out.append(matches[0] if len(matches) == 1 else column)
    return out


def _plain_type(pa: Any, t: Any) -> Any:
    """`t` with Arrow view types (string_view, binary_view) replaced, recursively."""
    types = pa.types
    if hasattr(types, "is_string_view") and types.is_string_view(t):
        return pa.string()
    if hasattr(types, "is_binary_view") and types.is_binary_view(t):
        return pa.binary()
    if types.is_struct(t):
        return pa.struct([f.with_type(_plain_type(pa, f.type)) for f in t])
    if types.is_large_list(t):
        return pa.large_list(t.value_field.with_type(_plain_type(pa, t.value_type)))
    if types.is_list(t):
        return pa.list_(t.value_field.with_type(_plain_type(pa, t.value_type)))
    if types.is_map(t):
        return pa.map_(
            t.key_field.with_type(_plain_type(pa, t.key_type)),
            t.item_field.with_type(_plain_type(pa, t.item_type)),
            keys_sorted=t.keys_sorted,
        )
    return t


def _cdf_columns(dt: Any, columns: list[str] | None) -> list[str] | None:
    """The projection to hand load_cdf(), or None to read every column.

    load_cdf() puts the names into SQL unquoted, where DataFusion lower-cases
    them, so a column `Name` failed with "No field named name" however it was
    spelled; such projections are applied afterwards instead.
    """
    canonical = _canonical_columns(dt, columns)
    if canonical and any(c != c.lower() for c in canonical):
        return None
    return canonical


def _without_view_types(stream: Any, *, keep: list[str] | None = None) -> Any:
    """`stream` with string_view/binary_view columns read as string/binary.

    delta-rs scans return view types, so the same table came back as
    `string` through the kernel and `string_view` through delta-rs, and
    pyarrow compute has no kernels for views in common operations
    (`take`, hence `Table.filter`/`sort_by`, failed with ArrowNotImplementedError).
    """
    import pyarrow as pa

    reader = pa.RecordBatchReader.from_stream(stream)
    if keep and reader.schema.names[: len(keep)] != keep:
        # A CDF projection applied here (see _cdf_columns): the requested
        # columns, then the feed's metadata columns, as load_cdf() orders them.
        meta = ("_change_type", "_commit_version", "_commit_timestamp")
        names = [*keep, *(n for n in meta if n not in keep)]
        names = [n for n in names if n in reader.schema.names]
        source = reader
        reader = pa.RecordBatchReader.from_batches(
            pa.schema([source.schema.field(n) for n in names]),
            (b.select(names) for b in source),
        )
    schema = reader.schema
    target = pa.schema(
        [f.with_type(_plain_type(pa, f.type)) for f in schema], metadata=schema.metadata
    )
    if target == schema:
        return reader
    return pa.RecordBatchReader.from_batches(target, (b.cast(target) for b in reader))


def _cdf_commit_times(dt: Any, reader: Any) -> Any:
    """`reader` with `_commit_timestamp` taken from each commit file's mtime.

    delta-rs fills it from commitInfo.timestamp, the writer's clock; the
    kernel -- and time travel on both engines -- use the commit file's
    modification time when in-commit timestamps are off. The two straddle a
    millisecond often enough that the same feed disagreed across engines, and
    a `_commit_timestamp` fed back to `starting_timestamp=` could miss its
    own commit. With in-commit timestamps on, delta-rs is left alone.
    """
    import pyarrow as pa

    config = dict(getattr(dt.metadata(), "configuration", {}) or {})
    if str(config.get("delta.enableInCommitTimestamps", "")).lower() == "true":
        return reader
    names = reader.schema.names
    if "_commit_timestamp" not in names or "_commit_version" not in names:
        return reader
    log = _DeltaLog.open(dt)
    if log is None or not log.mtimes:
        return reader
    ts_index = names.index("_commit_timestamp")
    ts_type = reader.schema.field(ts_index).type
    mtimes = log.mtimes

    def fixed() -> Iterator[Any]:
        for batch in reader:
            versions = batch.column(names.index("_commit_version")).to_pylist()
            if not all(v in mtimes for v in versions if v is not None):
                yield batch
                continue
            millis = pa.array([None if v is None else mtimes[v] for v in versions], type=pa.int64())
            column = millis.cast(pa.timestamp("ms", tz=getattr(ts_type, "tz", None))).cast(ts_type)
            yield batch.set_column(ts_index, reader.schema.field(ts_index), column)

    return pa.RecordBatchReader.from_batches(reader.schema, fixed())


def _datafusion_predicate(
    dt: Any, predicate: str | None, *, cdf: bool = False, dml: str | None = None
) -> str | None:
    """`predicate` (Spark SQL, as every engine takes it) as DataFusion SQL.

    delta-rs hands the string to DataFusion, whose dialect differs from
    Spark's in ways that change which rows match -- or which rows a DELETE
    removes: case-sensitive column names, `1.5` read as a DOUBLE (so
    `d = 12345678901234.5678` matched nothing), `5L`/`1.5D`/TIMESTAMP_NTZ
    rejected, `x NOT IN (1, NULL)` true for most rows, NaN rows pruned by
    statistics. The predicate is parsed once and rendered with the table's
    column spellings and typed literals, exactly as the kernel path filters.
    Text the parser does not cover (functions, arithmetic) is DataFusion's
    to read, and passes through unchanged.
    """
    if predicate is None or not predicate.strip():
        return predicate
    import pyarrow as pa

    from .. import predicate as sqlpred

    try:
        schema = pa.schema(dt.schema().to_arrow())
    except Exception:
        return predicate
    if cdf:
        schema = schema.append(pa.field("_change_type", pa.string()))
        schema = schema.append(pa.field("_commit_version", pa.int64()))
        schema = schema.append(pa.field("_commit_timestamp", pa.timestamp("us", tz="UTC")))
    try:
        node = sqlpred.parse(predicate)
    except sqlpred.PredicateError:
        # Functions and arithmetic are DataFusion's to read; the column names
        # in them still resolve case-insensitively, as in Delta.
        return _shield_stats(
            _exact_decimals(
                _fold_case(_refuse_null_literals(predicate), {None: list(schema.names)})
            ),
            schema,
        )
    if dml is not None:
        # delta-rs plans DELETE/UPDATE file skipping through the kernel, which
        # knows only leaf columns: `st IS NULL` on a struct failed with
        # "Predicate references unknown column: st", naming a column that exists.
        for path in sqlpred.columns_of(node):
            field = next((f for f in schema if f.name.lower() == path[0].lower()), None)
            if len(path) == 1 and field is not None and pa.types.is_nested(field.type):
                raise InvalidArgumentError(
                    f"delta-rs cannot {dml} by a predicate on the whole {field.type} column "
                    f"{field.name!r}; compare one of its fields instead (e.g. "
                    f"{field.name}.<field> IS NULL)"
                )
    rendered = sqlpred.to_datafusion(node, schema)
    if rendered is None:
        return _shield_stats(
            _exact_decimals(
                _fold_case(_refuse_null_literals(predicate), {None: list(schema.names)})
            ),
            schema,
        )
    return rendered


def _generated_columns(dt: Any) -> dict[str, str]:
    """Generated column name -> its generation expression."""
    try:
        fields = list(dt.schema().fields)
    except Exception:
        return {}
    out: dict[str, str] = {}
    for f in fields:
        expr = (f.metadata or {}).get("delta.generationExpression")
        if isinstance(expr, str) and expr.strip():
            out[f.name] = expr
    return out


def _with_generated(
    dt: Any,
    updates: dict[str, str],
    *,
    generated: dict[str, str] | None = None,
    target_alias: str | None = None,
) -> dict[str, str]:
    """Add SETs recomputing the generated columns an UPDATE's SETs feed.

    Delta recomputes a generated column when a column it derives from
    changes; delta-rs only checks, so ``SET id = id + 10`` on a table with
    ``g GENERATED ALWAYS AS (id * 2)`` failed "rows failed validation
    check". Each such column is set to its expression with the updated
    columns replaced by their new values (SET sees the old row). In a MERGE
    (`target_alias` given) the columns left alone are qualified with the
    target alias, since the source may have columns of the same name.
    """
    generated = _generated_columns(dt) if generated is None else generated
    if not generated:
        return updates
    changed = {k.strip("`").lower(): v for k, v in updates.items()}
    out = dict(updates)
    for name, expr in generated.items():
        if name.lower() in changed:
            continue
        pieces: list[str] = []
        pos = 0
        hit = False
        for match in _SQL_TOKEN.finditer(expr):
            if match.group(1) is not None:
                continue  # a string literal
            token = match.group(3) or (match.group(2) or "")[1:-1]
            if match.group(3) is not None and (
                expr[match.end() :].lstrip().startswith("(") or token.lower() in _SQL_WORDS
            ):
                continue  # a function name or a keyword
            if token.lower() in changed:
                pieces += [expr[pos : match.start()], f"({changed[token.lower()]})"]
                pos = match.end()
                hit = True
            elif target_alias is not None and "." not in token:
                quoted = "`" + token.replace("`", "``") + "`"
                pieces += [expr[pos : match.start()], f"{target_alias}.{quoted}"]
                pos = match.end()
        if hit:
            out[name] = "".join([*pieces, expr[pos:]])
    return out


def _column_names(dt: Any) -> list[str]:
    try:
        return [f.name for f in dt.schema().fields]
    except Exception:
        return []


def _respell_source(data: Any, target: list[str]) -> Any:
    """Rename source columns to the target's spelling where only case differs.

    delta-rs pairs source and target columns by exact name, so a source
    column ``barfoo`` for the target's ``BarFoo`` was silently skipped by
    ``when_matched_update_all()`` -- the rows counted as updated, unchanged.
    """
    try:
        import pyarrow as pa
    except ImportError:
        return data
    if not isinstance(data, (pa.Table, pa.RecordBatch)) or not target:
        return data
    names = list(data.schema.names)
    exact = set(target)
    folded: dict[str, list[str]] = {}
    for name in target:
        folded.setdefault(name.lower(), []).append(name)
    renamed = []
    for name in names:
        match = folded.get(name.lower(), [])
        if name in exact or len(match) != 1 or match[0] in names:
            renamed.append(name)
        else:
            renamed.append(match[0])
    if renamed == names or len(set(renamed)) != len(renamed):
        return data
    return data.rename_columns(renamed)


def _merge_columns(dt: Any, source: Any, kwargs: dict[str, Any]) -> dict[str | None, list[str]]:
    """What each alias of a MERGE may name, for `_fold_case`."""
    target = _column_names(dt)
    try:
        import pyarrow as pa

        schema = getattr(source, "schema", None)
        if not isinstance(schema, pa.Schema):
            schema = pa.schema(schema) if schema is not None else None
        src = list(schema.names) if schema is not None else []
    except Exception:
        src = []
    out: dict[str | None, list[str]] = {}
    source_alias, target_alias = kwargs.get("source_alias"), kwargs.get("target_alias")
    if isinstance(source_alias, str) and src:
        out[source_alias.lower()] = src
    if isinstance(target_alias, str) and target:
        out[target_alias.lower()] = target
    return out


_SQL_TOKEN = re.compile(
    r"""('(?:[^']|'')*')"""  # string literal
    r"""|("(?:[^"]|"")*"|`(?:[^`]|``)*`)"""  # quoted identifier
    r"""|([A-Za-z_][A-Za-z0-9_]*(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)*)"""  # bare (dotted) name
)

#: Words a bare name must never be rewritten as, even if a column is so named.
_SQL_WORDS = frozenset(
    {
        *("and", "or", "not", "is", "null", "true", "false", "in", "like", "ilike"),
        *("between", "case", "when", "then", "else", "end", "cast", "as", "interval"),
        *("distinct", "exists", "similar", "escape", "date", "timestamp"),
    }
)


def _fold_case(expr: str, columns: dict[str | None, list[str]]) -> str:
    """Respell bare column names in DataFusion SQL with the table's own case.

    Delta (like Spark) resolves column names case-insensitively; DataFusion
    does not, so ``SET FooBar = FOOBAR + 1`` or a merge on ``s.ID = t.id``
    failed with "No field named". `columns` maps a lower-cased alias (None
    for unqualified names) to the column names it may refer to. Names that
    match exactly, match nothing or match ambiguously, function names,
    keywords and quoted text are left alone.
    """

    def spell(name: str, alias: str | None) -> str | None:
        candidates = columns.get(alias.lower() if alias is not None else None) or []
        if name in candidates:
            return None
        found = [c for c in candidates if c.lower() == name.lower()]
        return found[0] if len(found) == 1 else None

    out: list[str] = []
    pos = 0
    for match in _SQL_TOKEN.finditer(expr):
        bare = match.group(3)
        if bare is None:
            continue
        if expr[match.end() :].lstrip().startswith("(") or bare.lower() in _SQL_WORDS:
            continue  # a function call or a keyword
        parts = [p.strip() for p in bare.split(".")]
        if len(parts) > 2:
            continue
        alias, name = (parts[0], parts[1]) if len(parts) == 2 else (None, parts[0])
        right = spell(name, alias)
        if right is None:
            continue
        quoted = "`" + right.replace("`", "``") + "`"
        out.append(expr[pos : match.start()])
        out.append(f"{alias}.{quoted}" if alias is not None else quoted)
        pos = match.end()
    out.append(expr[pos:])
    return "".join(out)


_SQL_NUMBER = re.compile(
    r"""('(?:[^']|'')*'|"(?:[^"]|"")*"|`(?:[^`]|``)*`)"""  # skipped: quoted text
    r"""|((?<![\w.])(?:\d+\.\d*|\.\d+)(?![\w.]))"""  # an unsuffixed decimal literal
)


def _exact_decimals(expr: str) -> str:
    """Unsuffixed fractional literals in DataFusion SQL typed as Spark types them.

    Spark reads `12345678901234.5678` as DECIMAL(18,4); DataFusion reads it
    as a DOUBLE (12345678901234.568), so in text passed through to delta-rs
    `w <> 12345678901234.5678` was true for the row holding exactly that
    value and a DELETE removed it, and `SET w = 12345678901234.5678` wrote
    ...5680. Exponent forms (1.5e3) stay DOUBLE, as in Spark, and so do
    literals short enough for a DOUBLE to compare exactly.
    """
    if not isinstance(expr, str) or "." not in expr:
        return expr

    def typed(match: re.Match[str]) -> str:
        number = match.group(2)
        if number is None:
            return match.group(0)
        whole, _, frac = number.partition(".")
        digits = (whole.lstrip("0") or "") + frac
        precision = max(len(digits), len(frac), 1)
        if len(digits.lstrip("0")) <= 15 or precision > 38:
            # Up to 15 significant digits a DOUBLE holds the value exactly
            # enough to compare as the DECIMAL would, and stays a DOUBLE:
            # DataFusion coerces a FLOAT column to DECIMAL against a DECIMAL
            # literal, which fails on NaN and infinity.
            return number
        text = f"{whole or '0'}.{frac}" if frac else (whole or "0")
        return f"CAST('{text}' AS DECIMAL({precision}, {len(frac)}))"

    return _SQL_NUMBER.sub(typed, expr)


def _refuse_null_literals(expr: str) -> str:
    """`expr`, unless it holds a NULL delta-rs would mis-evaluate.

    DataFusion folds a NULL literal in a predicate into forms delta-rs then
    gets wrong: DELETE ... WHERE abs(id) = NULL deleted every row (SQL deletes
    none), `x > 0 AND CAST(NULL AS BOOLEAN)` scanned every row, and on a
    partition column `CASE WHEN p = 1 THEN TRUE ELSE NULL END` matched them
    all. Predicates deltaswamp can parse are rendered without such NULLs;
    text passed through verbatim is refused instead. `IS [NOT] NULL` is fine.
    """
    for match in _SQL_TOKEN.finditer(expr):
        bare = (match.group(3) or "").lower()
        is_nullif = bare == "nullif" and expr[match.end() :].lstrip().startswith("(")
        if not is_nullif and (
            bare != "null"
            or re.search(r"\bis\s+(?:not\s+)?$", expr[: match.start()], re.IGNORECASE)
        ):
            continue
        raise InvalidArgumentError(
            f"the predicate {expr!r} uses a NULL literal in an expression delta-rs "
            "mis-evaluates (it can match -- or delete -- every row); write it with IS [NOT] "
            "NULL, or without the function so deltaswamp can evaluate it"
        )
    return expr


def _shield_stats(expr: str, schema: Any) -> str:
    """Hide columns with inexact statistics from delta-rs's stats substitution.

    When a predicate compares a column whose file min equals its max, delta-rs
    replaces the column with that statistic. For FLOAT/DOUBLE (NaN left out
    of the stats) and DECIMAL wider than 15 digits (stats stored as JSON
    doubles) the statistic is not the value: a DELETE by
    ``w <> CAST('12345678901234.5678' AS DECIMAL(18,4)) AND abs(id) > 0``
    removed the row holding exactly that value, and an UPDATE by it rewrote
    the row with ...5680. Text the predicate parser cannot render (it has a
    function) reaches DataFusion as written, so each reference to such a
    column is wrapped in an exact VARCHAR round trip the stats cannot see
    through.
    """
    import pyarrow as pa

    from .. import predicate as sqlpred

    risky: dict[str, tuple[str, Any]] = {}
    for field in schema:
        t = field.type
        if pa.types.is_floating(t) or (pa.types.is_decimal(t) and t.precision > 15):
            risky[field.name.lower()] = (field.name, t)
    if not risky:
        return expr
    out: list[str] = []
    pos = 0
    for match in _SQL_TOKEN.finditer(expr):
        token = match.group(2) or match.group(3)
        if token is None or "." in (match.group(3) or ""):
            continue
        name = token[1:-1].replace(token[0] * 2, token[0]) if match.group(2) else token
        if match.group(3) is not None and name.lower() in _SQL_WORDS:
            continue
        if expr[match.end() :].lstrip().startswith("("):
            continue  # a function call
        before = expr[: match.start()].rstrip()
        if before.endswith("."):
            continue  # a field of something else
        found = risky.get(name.lower())
        if found is None or (match.group(2) is not None and found[0] != name):
            continue
        column, type_ = found
        try:
            spelled = sqlpred._df_type(pa, type_)
        except Exception:
            continue
        quoted = '"' + column.replace('"', '""') + '"'
        out.append(expr[pos : match.start()])
        out.append(f"arrow_cast(CAST({quoted} AS VARCHAR), '{spelled}')")
        pos = match.end()
    out.append(expr[pos:])
    return "".join(out)


def _sql_string(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _sql_ident(name: str) -> str:
    """Quote a column name for a DataFusion predicate."""
    return '"' + name.replace('"', '""') + '"'


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECOND = timedelta(microseconds=1)


def _sql_value(value: Any) -> str:
    """Render a Python value as a DataFusion SQL expression."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "CAST('NaN' AS DOUBLE)"
        if math.isinf(value):
            return "CAST('inf' AS DOUBLE)" if value > 0 else "CAST('-inf' AS DOUBLE)"
        return repr(value)
    if isinstance(value, Decimal):
        return _sql_string(str(value))
    if isinstance(value, datetime):
        # Microseconds since the epoch, exact. A naive value is wall-clock
        # time taken as UTC -- delta-rs's own rendering used the machine's
        # local timezone, silently shifting every naive timestamp.
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return str((aware - _EPOCH) // _MICROSECOND)
    if isinstance(value, date):
        return _sql_string(value.isoformat())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "X'" + bytes(value).hex() + "'"
    if isinstance(value, str):
        return _sql_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_sql_value(v) for v in value) + "]"
    raise TypeError(f"cannot render a {type(value).__name__} as a SQL value")


def _timestamp_arg(value: Any) -> str:
    """An RFC 3339 timestamp string for delta-rs.

    delta-rs accepts only a string, and only one with an offset: a datetime
    object raised TypeError and '2024-01-01' or a naive ISO string failed with
    "premature end of input". Naive values are taken as UTC, as delta-rs's own
    `load_as_version` does for a naive datetime.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.strip())
        except ValueError:
            return value  # let delta-rs report what it makes of it
    else:
        raise TypeError(f"expected a timestamp string or datetime, got {type(value).__name__}")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.isoformat()


#: The longest file or directory name common filesystems accept, in bytes.
_MAX_PATH_COMPONENT = 255


def _refuse_long_float_partitions(
    engine: Any, table: ResolvedTable, data: Any, partition_by: list[str] | None
) -> None:
    """Refuse FLOAT/DOUBLE partition values delta-rs cannot write.

    delta-rs spells a float partition value positionally, so 1e-300 becomes a
    302-character `f=0.000...1` directory name: past the filesystem's limit,
    the write fails with "Unable to open file" after files may already have
    been staged. (Spark and the kernel write `1.0E-300`.)
    """
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
    except ImportError:
        return
    if isinstance(data, pa.RecordBatch):
        data = pa.Table.from_batches([data])
    if not isinstance(data, pa.Table):
        return  # a stream cannot be inspected without consuming it
    parts = list(partition_by or [])
    if not parts:
        try:
            parts = list(engine._open(table).metadata().partition_columns)
        except Exception:
            return
    for name in parts:
        if name not in data.column_names or not pa.types.is_floating(data.schema.field(name).type):
            continue
        for value in pc.unique(data.column(name)).to_pylist():
            if value is None or not math.isfinite(value):
                continue
            text = format(Decimal(repr(float(value))), "f")
            if len(f"{name}={text}".encode()) > _MAX_PATH_COMPONENT:
                raise InvalidArgumentError(
                    f"partition value {value!r} of FLOAT/DOUBLE column {name!r} cannot be "
                    f"written by delta-rs: it spells it as a {len(text)}-character directory "
                    "name, past the filesystem's limit. Partition by a rounded or string "
                    "column instead."
                )


def _plain_data(data: Any) -> Any:
    """Convert a pandas DataFrame so its index is not written as a column.

    delta-rs goes through pyarrow's default, which stores any non-range index
    as an `__index_level_0__` column -- a junk column on create and a schema
    mismatch on append. A *named* index is kept: that is data.
    """
    module = type(data).__module__ or ""
    if module.startswith("pandas") and type(data).__name__ == "DataFrame":
        import pyarrow as pa

        unnamed = all(name is None for name in data.index.names)
        # preserve_index=None keeps a range-like named index (set_index("id")
        # over 1, 2, ...) only as schema metadata, silently dropping the column.
        return pa.Table.from_pandas(data, preserve_index=not unnamed)
    return data


def _to_arrow_table(data: Any) -> Any:
    """Materialise write input as a pyarrow Table (consuming a stream once)."""
    import pyarrow as pa

    data = _plain_data(data)
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatchReader):
        return data.read_all()
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], pa.RecordBatch):
        return pa.Table.from_batches(list(data))
    return pa.table(data)


def _commit_kwargs(kwargs: dict[str, Any]) -> None:
    """Turn `commit_metadata=`/`max_commit_retries=` into delta-rs commit_properties.

    Only the write and DML paths translated these, so passing them to
    optimize, vacuum, restore or repair raised an unexpected-keyword TypeError
    even though the engine advertises commit metadata support.
    """
    if isinstance(kwargs.get("writer_properties"), dict):
        kwargs["writer_properties"] = _writer_properties(kwargs["writer_properties"])
    metadata = kwargs.pop("commit_metadata", None)
    retries = kwargs.pop("max_commit_retries", None)
    if metadata is None and retries is None:
        return
    if kwargs.get("commit_properties") is not None:
        raise InvalidArgumentError(
            "pass commit_metadata/max_commit_retries or commit_properties, not both"
        )
    kwargs["commit_properties"] = _commit_properties(metadata, None, retries)


def _commit_properties(
    commit_metadata: dict[str, Any] | None,
    txn: tuple[str, int] | None,
    max_commit_retries: int | None,
) -> Any:
    """Build delta-rs CommitProperties, or None when nothing was asked for."""
    if commit_metadata is None and txn is None and max_commit_retries is None:
        return None
    from deltalake import CommitProperties, Transaction

    app_transactions = None
    if txn is not None:
        app_id, version = txn
        app_transactions = [Transaction(app_id=app_id, version=version)]
    return CommitProperties(
        custom_metadata=commit_metadata,
        max_commit_retries=max_commit_retries,
        app_transactions=app_transactions,
    )


@contextlib.contextmanager
def _no_panics(what: str) -> Iterator[None]:
    """Turn a Rust panic into a catchable error.

    `pyo3_runtime.PanicException` inherits from BaseException, so it sails
    straight through `except Exception` and past any retry logic.
    """
    try:
        yield
    except Exception as exc:
        conflict = _as_commit_conflict(exc)
        if conflict is not None:
            raise conflict from exc
        raise
    except BaseException as exc:
        if type(exc).__name__ != "PanicException":
            raise
        if "Forked process detected" in str(exc):
            # delta-rs keeps one tokio runtime per process and refuses to run
            # in a forked child of a process that already used it. It is not
            # a bug in the input or in the engine's logic, and the fix is the
            # caller's start method.
            _FORKED_RUNTIME.add(os.getpid())
            raise UnreachableTableError(
                what,
                "delta-rs cannot run in a process forked from one that already used it "
                "(its tokio runtime does not survive fork)",
                "start worker processes with multiprocessing's 'spawn' or 'forkserver' "
                "method, or open the table only in the children",
            ) from exc
        raise EnginePanicError(
            f"delta-rs panicked while trying to {what}: {exc}. This is a bug in the "
            "engine rather than in your input; deltaswamp validates properties up "
            "front to avoid the known cases."
        ) from exc


#: commitInfo key marking a compaction this process ran, to find its commit.
_REWRITE_TAG = "deltaswamp.rewriteId"


def _tagged(properties: Any, tag: str) -> Any:
    """`properties` (delta-rs CommitProperties or None) with the rewrite tag added."""
    from deltalake import CommitProperties

    meta = dict(getattr(properties, "custom_metadata", None) or {})
    meta[_REWRITE_TAG] = tag
    return CommitProperties(
        custom_metadata=meta,
        max_commit_retries=getattr(properties, "max_commit_retries", None),
        app_transactions=getattr(properties, "app_transactions", None),
    )


_REWRITE_LOCKS: dict[str, threading.Lock] = {}
_REWRITE_LOCKS_GUARD = threading.Lock()


def _rewrite_lock(location: str) -> threading.Lock:
    """One lock per table location for compactions run from this process."""
    key = location.rstrip("/")
    with _REWRITE_LOCKS_GUARD:
        lock = _REWRITE_LOCKS.get(key)
        if lock is None:
            lock = _REWRITE_LOCKS[key] = threading.Lock()
        return lock


#: Processes in which delta-rs refused to run because they were forked from
#: one that had already started its runtime. Routing skips delta-rs there.
_FORKED_RUNTIME: set[int] = set()

#: delta-rs's wording for a commit that lost its race: a conflict found by its
#: checker, a version someone else wrote, or retries used up (the bare number).
_CONFLICT_MESSAGE = re.compile(
    r"concurrent|changed since last commit|existing table version|"
    r"Failed to commit transaction: \d+\s*$",
    re.IGNORECASE,
)


def _as_commit_conflict(exc: BaseException) -> CommitConflictError | None:
    """This library's CommitConflictError for a delta-rs lost commit race, else None.

    delta-rs raises its own CommitFailedError, so `except CommitConflictError`
    around a write caught the kernel's lost races but never delta-rs's.
    """
    if type(exc).__name__ != "CommitFailedError" or isinstance(exc, CommitConflictError):
        return None
    message = str(exc)
    if not _CONFLICT_MESSAGE.search(message):
        return None
    found = re.search(r"version:? (\d+)", message)
    return CommitConflictError(
        int(found.group(1)) if found else -1,
        f"another writer committed first: {message}. Re-read the table and retry",
    )


_COMMIT_FILE = re.compile(r"(?:^|/)(\d{20})\.json$")


class _CheckedMerger:
    """delta-rs's TableMerger, with `except_cols` checked before it is used.

    delta-rs tests `col.name not in except_cols`: a misspelled or wrong-case
    name is silently ignored (so the column is overwritten after all), and a
    bare string is a substring test that excludes columns by accident.
    """

    def __init__(
        self,
        merger: Any,
        columns: dict[str | None, list[str]] | None = None,
        target: list[str] | None = None,
        *,
        generated: dict[str, str] | None = None,
        source_alias: str | None = None,
        target_alias: str | None = None,
    ) -> None:
        self._merger = merger
        #: alias -> column names, for respelling clause SQL case-insensitively.
        self._columns = columns or {}
        self._target = target or []
        #: generated column -> expression, recomputed by matched updates.
        self._generated = generated or {}
        self._source_alias = source_alias
        self._target_alias = target_alias

    def _recompute(self, updates: dict[str, str]) -> dict[str, str]:
        """`updates` plus SETs recomputing the generated columns they feed.

        delta-rs only validates generated columns on a matched update, so a
        SET of a column one derives from failed "rows failed validation
        check" where Delta recomputes it.
        """
        if not self._generated:
            return updates
        out = _with_generated(
            None, updates, generated=self._generated, target_alias=self._target_alias
        )
        if out != updates and self._target_alias is None:
            raise InvalidArgumentError(
                "this MERGE updates columns that generated columns "
                f"({', '.join(sorted(set(out) - set(updates)))}) derive from; recomputing "
                "them needs merge(..., target_alias=...) so the target's own columns can be "
                "named unambiguously"
            )
        return out

    def when_matched_update(self, updates: Any, predicate: str | None = None) -> _CheckedMerger:
        folded = self._fold(updates)
        if isinstance(folded, dict):
            folded = self._recompute(folded)
        with _no_panics("merge (when_matched_update)"):
            self._merger.when_matched_update(folded, self._fold(predicate))
        return self

    def _fold(self, value: Any) -> Any:
        if isinstance(value, str):
            return _exact_decimals(_fold_case(value, self._columns))
        if isinstance(value, dict):
            # SET/INSERT keys name target columns, unqualified.
            return {
                (
                    _fold_case(k, {None: self._target}).strip("`")
                    if isinstance(k, str) and "`" not in k
                    else k
                ): self._fold(v)
                for k, v in value.items()
            }
        return value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in ("_merger", "_columns", "_target"):
            raise AttributeError(name)
        attr = getattr(self._merger, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            if name.startswith("when_"):
                args = tuple(self._fold(a) for a in args)
                kwargs = {
                    k: self._fold(v) if k in ("updates", "predicate") else v
                    for k, v in kwargs.items()
                }
            # execute() commits: a lost race is a CommitConflictError here too.
            with _no_panics(f"merge ({name})"):
                result = attr(*args, **kwargs)
            return self if result is self._merger else result

        return call

    def _except(self, except_cols: Any) -> list[str] | None:
        if except_cols is None:
            return None
        cols = [except_cols] if isinstance(except_cols, str) else list(except_cols)
        try:
            names = [field.name for field in self._merger._builder.arrow_schema]
        except Exception:
            return cols  # a deltalake without this private hook: pass through unchecked
        unknown = [c for c in cols if c not in names]
        if unknown:
            raise InvalidArgumentError(
                f"except_cols names columns the merge source does not have: {unknown}; "
                f"source columns are {names}"
            )
        return cols

    def when_matched_update_all(
        self, predicate: str | None = None, except_cols: Any = None
    ) -> _CheckedMerger:
        excluded = self._except(except_cols)
        if self._generated:
            # Spelled out as explicit SETs so the generated columns the source
            # does not carry can be recomputed alongside (see _recompute).
            source = self._columns.get((self._source_alias or "").lower()) or []
            by_lower = {c.lower(): c for c in self._target}
            sets = {
                by_lower[c.lower()]: f"{self._source_alias}.`{c.replace('`', '``')}`"
                for c in source
                if c.lower() in by_lower and c not in (excluded or [])
            }
            if sets and _with_generated(None, sets, generated=self._generated) != sets:
                if self._source_alias is None:
                    raise InvalidArgumentError(
                        "when_matched_update_all() on a table with generated columns needs "
                        "merge(..., source_alias=..., target_alias=...) to recompute them"
                    )
                return self.when_matched_update(sets, predicate)
        self._merger.when_matched_update_all(self._fold(predicate), except_cols=excluded)
        return self

    def when_not_matched_insert_all(
        self, predicate: str | None = None, except_cols: Any = None
    ) -> _CheckedMerger:
        self._merger.when_not_matched_insert_all(
            self._fold(predicate), except_cols=self._except(except_cols)
        )
        return self


def _writer_properties(value: Any) -> Any:
    """Accept a plain dict of WriterProperties arguments.

    delta-rs reads attributes off the object, so a dict failed with
    "'dict' object has no attribute 'data_page_size_limit'".
    """
    if isinstance(value, dict):
        from deltalake import WriterProperties

        return WriterProperties(**value)
    return value


class _DeltaLog:
    """The commit files under `_delta_log`, listed through the table's own store."""

    def __init__(
        self,
        handler: Any,
        commits: list[tuple[int, str]],
        mtimes: dict[int, int] | None = None,
    ) -> None:
        self._handler = handler
        #: (version, path), ascending.
        self.commits = commits
        #: version -> commit file modification time, epoch milliseconds.
        self.mtimes = mtimes or {}

    @classmethod
    def open(cls, dt: Any) -> _DeltaLog | None:
        try:
            from deltalake._internal import DeltaFileSystemHandler

            handler = DeltaFileSystemHandler.from_table(
                dt._table, getattr(dt, "_storage_options", None), None
            )
            infos = handler.get_file_info_selector("_delta_log", False, False)
        except Exception:
            return None
        latest = dt.version()
        commits = []
        mtimes: dict[int, int] = {}
        for info in infos:
            match = _COMMIT_FILE.search(info.path)
            if match and int(match.group(1)) <= latest:
                version = int(match.group(1))
                commits.append((version, info.path))
                mtime_ns = getattr(info, "mtime_ns", None)
                if isinstance(mtime_ns, int):
                    mtimes[version] = mtime_ns // 1_000_000
        return cls(handler, sorted(commits), mtimes)

    def actions(self, path: str) -> list[dict[str, Any]]:
        stream = self._handler.open_input_file(path)
        try:
            raw = stream.read()
        finally:
            with contextlib.suppress(Exception):
                stream.close()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _rows_removed(dt: Any, result: dict[str, Any]) -> int | None:
    """Live rows in the files the latest commit removed, from their footers."""
    if not result.get("num_removed_files"):
        return 0
    try:
        import urllib.parse

        import pyarrow.parquet as pq

        dt.update_incremental()
        log = _DeltaLog.open(dt)
        if log is None or not log.commits:
            return None
        total = 0
        for action in log.actions(log.commits[-1][1]):
            remove = action.get("remove")
            if remove is None or remove.get("dataChange") is False:
                continue
            path = urllib.parse.unquote(remove["path"])
            if "://" in path:
                return None
            stream = log._handler.open_input_file(path)
            try:
                total += pq.ParquetFile(stream).metadata.num_rows
            finally:
                with contextlib.suppress(Exception):
                    stream.close()
            total -= int((remove.get("deletionVector") or {}).get("cardinality") or 0)
        return total
    except Exception:
        return None


def _history_from_log(
    dt: Any, limit: int | None, reported: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]] | None:
    """Commit history with each entry's version taken from its commit file.

    delta-rs numbers history entries by counting down from the latest version,
    so one commit without a commitInfo action (it is optional in the
    protocol) shifts every older entry onto the wrong version. None when the
    log cannot be listed; the caller then falls back to delta-rs.
    """
    log = _DeltaLog.open(dt)
    if log is None or not log.commits:
        return None
    newest_first = list(reversed(log.commits))
    if limit is not None:
        newest_first = newest_first[:limit]
    if (
        reported is not None
        and len(reported) == len(newest_first)
        and [h.get("version") for h in reported] == [v for v, _ in newest_first]
    ):
        # Every commit in the window has a commitInfo, so delta-rs numbered
        # them correctly: no need to re-read each commit file.
        return [
            _travel_timestamp(dict(h), log.mtimes.get(v))
            for h, (v, _) in zip(reported, newest_first, strict=True)
        ]
    from concurrent.futures import ThreadPoolExecutor

    def entry(item: tuple[int, str]) -> dict[str, Any]:
        version, path = item
        info = next((a["commitInfo"] for a in log.actions(path) if "commitInfo" in a), None)
        return _travel_timestamp({**(info or {}), "version": version}, log.mtimes.get(version))

    try:
        # Concurrently: one GET per commit, which on object storage is slow
        # when read one after another.
        with ThreadPoolExecutor(max_workers=min(16, len(newest_first))) as pool:
            return list(pool.map(entry, newest_first))
    except Exception:
        return None


def _travel_timestamp(entry: dict[str, Any], mtime_ms: int | None) -> dict[str, Any]:
    """`entry` with the timestamp time travel resolves against.

    commitInfo.timestamp is the writer's clock, often a millisecond or so
    before the commit file's modification time, which is what delta-rs time
    travel compares; feeding it back to scan(timestamp=) read the previous
    version. With in-commit timestamps the commit's own inCommitTimestamp is
    the authority.
    """
    ict = entry.get("inCommitTimestamp")
    if isinstance(ict, int) and not isinstance(ict, bool):
        entry["timestamp"] = ict
    elif mtime_ms is not None:
        entry["timestamp"] = mtime_ms
    return entry


def _cdf_enabled_since(dt: Any) -> int | None:
    """The first version of the latest run of commits with CDF enabled."""
    log = _DeltaLog.open(dt)
    if log is None or not log.commits:
        return None
    since: int | None = None
    for version, path in reversed(log.commits):
        for action in log.actions(path):
            metadata = action.get("metaData")
            if metadata is None:
                continue
            value = (metadata.get("configuration") or {}).get("delta.enableChangeDataFeed")
            if str(value).lower() != "true":
                return since
            since = version
    # Enabled since before the oldest retained commit.
    return log.commits[0][0]


def _physical_to_logical(schema: dict[str, Any]) -> dict[str, str]:
    """Dotted physical column path -> dotted logical path, for mapped columns."""
    out: dict[str, str] = {}

    def walk(fields: Any, physical: tuple[str, ...], logical: tuple[str, ...]) -> None:
        for field in fields or ():
            name = str(field["name"])
            phys = (field.get("metadata") or {}).get("delta.columnMapping.physicalName") or name
            p, lg = (*physical, str(phys)), (*logical, name)
            if p != lg:
                out[".".join(p)] = ".".join(lg)
            kind = field.get("type")
            if isinstance(kind, dict) and kind.get("type") == "struct":
                walk(kind.get("fields"), p, lg)

    walk(schema.get("fields"), (), ())
    return out


def _logical_stat_name(column: str, logical: dict[str, str]) -> str:
    for prefix in ("min.", "max.", "null_count."):
        if column.startswith(prefix):
            rest = column[len(prefix) :]
            return prefix + logical.get(rest, rest)
    return column


def _may_vend_gcs_bearer(provider: Any) -> bool:
    """Whether `provider` may hand delta-rs a GCS bearer token (unusable there).

    A static provider's credential is inspectable without vending, so one
    carrying a service-account key (which delta-rs accepts) is not refused.
    """
    if provider is None:
        return False
    static = getattr(provider, "_credentials", None)
    secrets = getattr(static, "secrets", None)
    if isinstance(secrets, dict):
        return "google_bearer_token" in secrets
    return True


def _object_store_options(options: dict[str, str] | None) -> dict[str, str] | None:
    """Adapt vended storage options to what delta-rs's object_store accepts.

    `google_bearer_token` is not an object_store GoogleConfigKey in the
    installed deltalake: it is silently dropped and the client falls back to
    the GCE metadata server, failing far from the cause. There is no key that
    takes a bearer token, so refuse clearly instead.
    """
    if not options:
        return None
    if "google_bearer_token" in options:
        raise UnreachableTableError(
            "open a GCS table with delta-rs",
            "the catalog vended a GCS OAuth bearer token, which delta-rs's object store "
            "has no option for (it only takes a service-account key)",
            "reads route to the kernel engine; or supply google_service_account_key",
        )
    endpoint = options.get("aws_endpoint") or options.get("aws_endpoint_url") or ""
    if "r2.cloudflarestorage.com" in endpoint and "aws_conditional_put" not in options:
        # R2 supports If-None-Match; make the commit's put-if-absent explicit
        # rather than depending on object_store's default.
        options = {**options, "aws_conditional_put": "etag"}
    return options


_DURATION_UNITS_DAYS = {
    "week": 7.0,
    "day": 1.0,
    "hour": 1 / 24,
    "minute": 1 / 1440,
    "second": 1 / 86400,
    "millisecond": 1 / 86_400_000,
    "microsecond": 1 / 86_400_000_000,
}


def _duration_days(value: str) -> float | None:
    """Parse a Delta interval like 'interval 7 days' into days.

    None when any part is not understood: guessing zero for an unknown unit
    ("interval 10080 minutes") made a week look shorter than a day.
    """
    parts = value.lower().replace("interval", "").split()
    total = 0.0
    number: float | None = None
    seen = False
    for part in parts:
        try:
            number = float(part)
            continue
        except ValueError:
            pass
        unit = part[:-1] if part.endswith("s") else part
        if number is None or unit not in _DURATION_UNITS_DAYS:
            return None
        total += number * _DURATION_UNITS_DAYS[unit]
        number = None
        seen = True
    if number is not None or not seen:
        return None
    return total


def _deltars_fields(fields: Any) -> list[Any]:
    """Normalise column definitions into the `deltalake.Field`s delta-rs wants.

    The kernel path accepts pyarrow fields, delta-rs fields or a
    ``{name: type}`` mapping; delta-rs accepts only its own `Field`. Passing
    them straight through made `add_column` succeed or fail on the same
    argument depending on which engine the router happened to pick, which is
    exactly the kind of difference a caller cannot see.
    """
    from deltalake import Field, Schema

    if isinstance(fields, dict):
        return [
            Field.from_json(
                json.dumps({"name": name, "type": str(dtype), "nullable": True, "metadata": {}})
            )
            for name, dtype in fields.items()
        ]

    items = (
        list(fields) if isinstance(fields, (list, tuple)) or hasattr(fields, "names") else [fields]
    )
    out: list[Any] = []
    for item in items:
        if isinstance(item, Field):
            out.append(item)
        elif hasattr(item, "type") and hasattr(item, "nullable"):
            # A pyarrow Field: a one-field Arrow schema converts cleanly.
            import pyarrow as pa

            out.append(Schema.from_arrow(pa.schema([item])).fields[0])
        else:
            raise UnreachableTableError(
                "add columns",
                f"cannot interpret {type(item).__name__} as a column definition",
                "pass pyarrow fields, deltalake Fields, or a {name: type} mapping",
            )
    return out
