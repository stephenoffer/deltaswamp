"""The delta-rs engine: the default DML and maintenance path.

delta-rs is far ahead of the kernel on *doing* things -- MERGE with all six
clause types, UPDATE/DELETE by predicate, `replaceWhere`, schema evolution,
OPTIMIZE, Z-ORDER, VACUUM, RESTORE, FSCK, CONVERT, manifest generation,
`history()` -- none of which the kernel implements.

Two habits this module keeps deliberately:

* **A fresh `DeltaTable` per operation.** delta-rs bakes `storage_options` into
  the object store at construction and offers no credential-provider hook, so a
  long-lived handle dies when its vended credential expires. Re-resolving is the
  only available refresh mechanism.
* **Refusing rather than diverging.** A write to an Iceberg-reads table needs
  `MSCK REPAIR TABLE ... SYNC METADATA` afterwards, which only Databricks can
  run. Writing anyway leaves the Iceberg view silently stale, so we decline.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from numbers import Integral
from typing import Any, Literal

from ..capability import (
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
from ..errors import EnginePanicError, UnreachableTableError
from ..properties import effect_for, validate_properties
from .base import missing_method

__all__ = ["DeltaRsEngine"]

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

    def __init__(self, *, storage_options: dict[str, str] | None = None) -> None:
        self._base_options = dict(storage_options or {})

    # ----------------------------------------------------------- capabilities

    def supports(self, operation: Operation, table: ResolvedTable, **shape: Any) -> Capability:
        if not self.available():
            return Capability(
                operation,
                ok=False,
                reason="the deltalake package is not installed",
                remedy="pip install deltalake",
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
        return True

    # --------------------------------------------------------------- internals

    def _storage_options(self, table: ResolvedTable, *, write: bool) -> dict[str, str]:
        options = dict(self._base_options)
        if table.credential_provider is not None:
            op = CredentialOperation.READ_WRITE if write else CredentialOperation.READ
            options.update(table.credential_provider.credentials(op).as_storage_options())
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
        dt = DeltaTable(
            table.location,
            version=version,
            storage_options=_object_store_options(self._storage_options(table, write=write)),
        )
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
        return dt.scan(columns=columns, predicate=predicate)

    def history(self, table: ResolvedTable, *, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None and limit < 0:
            # delta-rs surfaces this as "can't convert negative int to unsigned".
            raise ValueError(f"history limit must be zero or more, got {limit}")
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
            return dt.load_cdf(
                starting_version=start,
                ending_version=ending_version,
                starting_timestamp=(
                    None if starting_timestamp is None else _timestamp_arg(starting_timestamp)
                ),
                ending_timestamp=(
                    None if ending_timestamp is None else _timestamp_arg(ending_timestamp)
                ),
                columns=columns,
                predicate=predicate,
                allow_out_of_range=allow_out_of_range,
            )

        if starting_version is not None or starting_timestamp is not None:
            return load(starting_version if starting_version is not None else 0)
        try:
            return load(0)
        except Exception as exc:
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
            raise ValueError("overwrite predicate is empty; pass None to replace the whole table")
        data = _plain_data(data)
        common: dict[str, Any] = {
            "schema_mode": schema_mode,
            "partition_by": partition_by,
            "target_file_size": target_file_size,
            "writer_properties": _writer_properties(writer_properties),
            "commit_properties": _commit_properties(commit_metadata, txn, max_commit_retries),
            "storage_options": _object_store_options(self._storage_options(table, write=True)),
        }
        with _no_panics(f"{mode} to the table"):
            if mode == "overwrite":
                write_deltalake(
                    table.location, data, mode="overwrite", predicate=predicate, **common
                )
            else:
                write_deltalake(table.location, data, mode="append", **common)

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
            raise ValueError("delete predicate is empty; pass None to delete every row")
        with _no_panics("delete"):
            result: dict[str, Any] = self._open(table, write=True).delete(
                predicate,
                writer_properties=_writer_properties(writer_properties),
                commit_properties=_commit_properties(commit_metadata, None, max_commit_retries),
            )
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
            raise ValueError("pass updates (SQL expressions) or new_values, not both")
        if new_values is not None:
            # delta-rs renders these itself, and gets it wrong: no quote
            # escaping, no NULL, naive datetimes shifted by the local timezone,
            # NaN as a column name, and no date/Decimal/bytes at all.
            updates = {column: _sql_value(value) for column, value in new_values.items()}
        if not updates:
            # An empty mapping is a silent no-op in delta-rs.
            raise ValueError("update needs at least one column to set")
        with _no_panics("update"):
            result: dict[str, Any] = self._open(table, write=True).update(
                updates=updates,
                predicate=predicate,
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
        merger = self._open(table, write=True).merge(_plain_data(source), predicate, **kwargs)
        return _CheckedMerger(merger)

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
        dt = self._open(table, write=True)
        with _no_panics("optimize"):
            if zorder_by:
                result: dict[str, Any] = dt.optimize.z_order(zorder_by, **kwargs)
            else:
                result = dt.optimize.compact(**kwargs)
        return result

    def zorder(
        self, table: ResolvedTable, columns: list[str] | str, **kwargs: Any
    ) -> dict[str, Any]:
        if isinstance(columns, str):
            columns = [columns]
        if not columns:
            raise ValueError("Z-ORDER needs at least one column")
        _commit_kwargs(kwargs)
        with _no_panics("z-order"):
            result: dict[str, Any] = self._open(table, write=True).optimize.z_order(
                columns, **kwargs
            )
        return result

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
                "restore",
                "the table has deletion vectors, and delta-rs RESTORE silently fails to "
                "revert DV changes -- it reports success and leaves the rows deleted "
                "(delta-rs#4613)",
                "perform the restore from Databricks",
            )
        result: dict[str, Any] = self._open(table, write=True).restore(target, **kwargs)
        return result

    def repair(self, table: ResolvedTable, **kwargs: Any) -> dict[str, Any]:
        _commit_kwargs(kwargs)
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
        self._open(table, write=True).alter.add_columns(converted, **kwargs)

    def set_properties(
        self, table: ResolvedTable, properties: dict[str, str], **kwargs: Any
    ) -> None:
        validate_properties(properties, EngineKind.DELTARS, Operation.SET_PROPERTIES)
        _commit_kwargs(kwargs)
        with _no_panics("set table properties"):
            self._open(table, write=True).alter.set_table_properties(properties, **kwargs)

    def add_feature(self, table: ResolvedTable, feature: Any, **kwargs: Any) -> None:
        self._open(table, write=True).alter.add_feature(feature, **kwargs)

    def add_constraint(
        self, table: ResolvedTable, constraints: dict[str, str], **kwargs: Any
    ) -> None:
        _commit_kwargs(kwargs)
        self._open(table, write=True).alter.add_constraint(constraints, **kwargs)

    def drop_constraint(self, table: ResolvedTable, name: str, *, if_exists: bool = False) -> None:
        self._open(table, write=True).alter.drop_constraint(name, raise_if_not_exists=not if_exists)

    def set_comment(self, table: ResolvedTable, comment: str | None) -> None:
        self._open(table, write=True).alter.set_table_description(comment or "")

    def set_column_comment(self, table: ResolvedTable, column: str, comment: str | None) -> None:
        dt = self._open(table, write=True)
        # A dot means a nested path only when no top-level column has that
        # name; "a.b" is a legal column name and was refused outright.
        if "." in column and column not in dt.schema().to_arrow().names:
            raise UnreachableTableError(
                f"comment on nested column {column!r} with delta-rs",
                "delta-rs sets field metadata on top-level columns only",
            )
        dt.alter.set_column_metadata(column, {"comment": comment or ""})

    def drop_not_null(self, table: ResolvedTable, column: str) -> None:
        """DROP NOT NULL. A no-op on a column that is already nullable, as in Spark."""
        try:
            self._open(table, write=True).alter.drop_column_not_null(column)
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
            raise ValueError(f"compact_logs range is inverted: start={start} > end={end}")
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


def _sql_literal(value: Any) -> str:
    """Render a partition value as a SQL literal.

    Dates and timestamps render as quoted strings, which DataFusion casts to
    the column's type when comparing.
    """
    if isinstance(value, (datetime, date)):
        return _sql_string(str(value))
    return _sql_value(value)


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
        raise ValueError("pass commit_metadata/max_commit_retries or commit_properties, not both")
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
    except Exception:
        raise
    except BaseException as exc:
        if type(exc).__name__ != "PanicException":
            raise
        raise EnginePanicError(
            f"delta-rs panicked while trying to {what}: {exc}. This is a bug in the "
            "engine rather than in your input; deltaswamp validates properties up "
            "front to avoid the known cases."
        ) from exc


_COMMIT_FILE = re.compile(r"(?:^|/)(\d{20})\.json$")


class _CheckedMerger:
    """delta-rs's TableMerger, with `except_cols` checked before it is used.

    delta-rs tests `col.name not in except_cols`: a misspelled or wrong-case
    name is silently ignored (so the column is overwritten after all), and a
    bare string is a substring test that excludes columns by accident.
    """

    def __init__(self, merger: Any) -> None:
        self._merger = merger

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name == "_merger":
            raise AttributeError(name)
        attr = getattr(self._merger, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
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
            raise ValueError(
                f"except_cols names columns the merge source does not have: {unknown}; "
                f"source columns are {names}"
            )
        return cols

    def when_matched_update_all(
        self, predicate: str | None = None, except_cols: Any = None
    ) -> _CheckedMerger:
        self._merger.when_matched_update_all(predicate, except_cols=self._except(except_cols))
        return self

    def when_not_matched_insert_all(
        self, predicate: str | None = None, except_cols: Any = None
    ) -> _CheckedMerger:
        self._merger.when_not_matched_insert_all(predicate, except_cols=self._except(except_cols))
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
