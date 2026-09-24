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
from collections.abc import Iterator
from typing import Any

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
    {Operation.SCAN, Operation.TIME_TRAVEL, Operation.CDF, Operation.HISTORY, Operation.DETAIL}
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
        for name in table.reader_features:
            feature = feature_from_wire(name)
            if feature is None or FEATURE_SUPPORT[feature].deltars_read is Support.NO:
                blockers.append(name)
        if writing:
            for name in table.writer_features:
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
            storage_options=self._storage_options(table, write=write) or None,
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
    ) -> Any:
        dt = self._open(table, version=version)
        if timestamp is not None:
            dt.load_as_version(timestamp)
        # `scan()` is the only delta-rs read path that handles deletion vectors
        # and column mapping; to_pyarrow_dataset() hard-rejects both.
        return dt.scan(columns=columns, predicate=predicate)

    def history(self, table: ResolvedTable, *, limit: int | None = None) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._open(table).history(limit)
        return result

    def detail(self, table: ResolvedTable, *, version: int | None = None) -> dict[str, Any]:
        dt = self._open(table, version=version)
        protocol = dt.protocol()
        return {
            "version": dt.version(),
            "location": dt.table_uri,
            "min_reader_version": protocol.min_reader_version,
            "min_writer_version": protocol.min_writer_version,
            "reader_features": list(protocol.reader_features or []),
            "writer_features": list(protocol.writer_features or []),
            "properties": dict(dt.metadata().configuration),
            "partition_columns": list(dt.metadata().partition_columns),
        }

    def cdf(
        self,
        table: ResolvedTable,
        *,
        starting_version: int | None = None,
        ending_version: int | None = None,
        columns: list[str] | None = None,
    ) -> Any:
        self._guard_cdf(table)
        return self._open(table).load_cdf(
            starting_version=starting_version if starting_version is not None else 0,
            ending_version=ending_version,
            columns=columns,
        )

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
        if deleted and log and _duration_days(deleted) < _duration_days(log):
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

        import pyarrow as pa

        arrow = data if isinstance(data, pa.Table) else pa.table(data)
        missing = [c for c in partitions if c not in arrow.column_names]
        if missing:
            raise UnreachableTableError(
                "overwrite dynamically",
                f"the data has no {', '.join(missing)} column, so the partitions it "
                "would replace cannot be determined",
            )

        tuples = sorted(
            {
                tuple(row[column] for column in partitions)
                for row in arrow.select(partitions).to_pylist()
            }
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
                f"{col} = {_sql_literal(val)}" for col, val in zip(partitions, values, strict=True)
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
        mode: str,
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

        write_deltalake(
            table.location,
            data,
            mode=mode,
            schema_mode=schema_mode,
            predicate=predicate,
            partition_by=partition_by,
            target_file_size=target_file_size,
            writer_properties=writer_properties,
            commit_properties=_commit_properties(commit_metadata, txn, max_commit_retries),
            storage_options=self._storage_options(table, write=True) or None,
        )

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
        mode: str = "error",
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
                storage_options=self._storage_options(table, write=True) or None,
            )

    def delete(self, table: ResolvedTable, predicate: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = self._open(table, write=True).delete(predicate)
        return result

    def update(
        self,
        table: ResolvedTable,
        *,
        updates: dict[str, str] | None = None,
        predicate: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = self._open(table, write=True).update(
            updates=updates, predicate=predicate
        )
        return result

    def merge(self, table: ResolvedTable, source: Any, predicate: str, **kwargs: Any) -> Any:
        return self._open(table, write=True).merge(source, predicate, **kwargs)

    # ------------------------------------------------------------ maintenance

    def optimize(self, table: ResolvedTable, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._open(table, write=True).optimize.compact(**kwargs)
        return result

    def zorder(self, table: ResolvedTable, columns: list[str], **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._open(table, write=True).optimize.z_order(columns, **kwargs)
        return result

    def vacuum(self, table: ResolvedTable, **kwargs: Any) -> list[str]:
        if table.is_shallow_clone:
            raise UnreachableTableError(
                "vacuum",
                "the table is a shallow clone and borrows the source table's files; "
                "vacuuming it risks deleting data the source still owns",
            )
        result: list[str] = self._open(table, write=True).vacuum(**kwargs)
        return result

    def restore(self, table: ResolvedTable, target: Any, **kwargs: Any) -> dict[str, Any]:
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
        result: dict[str, Any] = self._open(table, write=True).repair(**kwargs)
        return result

    # ---------------------------------------------------------------- schema

    def add_columns(self, table: ResolvedTable, fields: Any, **kwargs: Any) -> None:
        self._open(table, write=True).alter.add_columns(fields, **kwargs)

    def set_properties(
        self, table: ResolvedTable, properties: dict[str, str], **kwargs: Any
    ) -> None:
        validate_properties(properties, EngineKind.DELTARS, Operation.SET_PROPERTIES)
        with _no_panics("set table properties"):
            self._open(table, write=True).alter.set_table_properties(properties, **kwargs)

    def add_feature(self, table: ResolvedTable, feature: Any, **kwargs: Any) -> None:
        self._open(table, write=True).alter.add_feature(feature, **kwargs)

    def add_constraint(
        self, table: ResolvedTable, constraints: dict[str, str], **kwargs: Any
    ) -> None:
        self._open(table, write=True).alter.add_constraint(constraints, **kwargs)

    # ----------------------------------------------------------- log upkeep

    def checkpoint(self, table: ResolvedTable) -> None:
        self._open(table, write=True).create_checkpoint()

    def compact_logs(
        self, table: ResolvedTable, start: int | None = None, end: int | None = None
    ) -> Any:
        # delta-rs wants concrete versions and rejects a degenerate range, so a
        # table with a single commit has nothing to compact.
        dt = self._open(table, write=True)
        first = 0 if start is None else start
        last = dt.version() if end is None else end
        if last <= first:
            return None
        return dt.compact_logs(first, last)

    def generate(self, table: ResolvedTable) -> None:
        """Write symlink manifests for engines that read them (Presto, Athena)."""
        self._open(table, write=True).generate()

    def convert(
        self,
        location: str,
        *,
        partition_by: Any = None,
        partition_strategy: str = "hive",
        **kwargs: Any,
    ) -> None:
        """Turn a directory of Parquet into a Delta table in place."""
        from deltalake import convert_to_deltalake

        convert_to_deltalake(
            location,
            partition_by=partition_by,
            partition_strategy=partition_strategy,
            storage_options=self._base_options or None,
            **kwargs,
        )

    def plan_scan(self, table: ResolvedTable, **kwargs: Any) -> list[Any]:
        raise NotImplementedError("the delta-rs engine has no split-planning surface")

    def execute_scan(self, table: ResolvedTable, splits: list[Any], **kwargs: Any) -> Any:
        raise NotImplementedError("see plan_scan")


def _sql_literal(value: Any) -> str:
    """Render a partition value as a SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


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


def _duration_days(value: str) -> float:
    """Parse a Delta interval like 'interval 7 days' into days. Best effort."""
    parts = value.lower().replace("interval", "").split()
    total = 0.0
    number: float | None = None
    units = {"day": 1.0, "days": 1.0, "hour": 1 / 24, "hours": 1 / 24, "week": 7.0, "weeks": 7.0}
    for part in parts:
        try:
            number = float(part)
            continue
        except ValueError:
            pass
        if number is not None and part in units:
            total += number * units[part]
            number = None
    return total
