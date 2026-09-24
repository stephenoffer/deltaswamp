"""The Databricks SQL warehouse fallback.

Opt-in, never automatic. Everything works here -- views, materialized views,
row-filtered tables, catalog-managed tables, DROP FEATURE, REORG, CLONE, column
rename -- because the work happens server-side. That is also why it is off by
default: rerouting a scan through a warehouse changes latency, egress and DBU
cost by orders of magnitude, and a silent reroute would turn a performance cliff
into a mystery.

When this engine serves an operation, it says so, and names the warehouse.

Statements run through the Statement Execution API of `databricks-sdk` (see
`sql_backend`), so the fallback needs no optional driver. Three rules govern
the SQL this module writes:

* **Identifiers are always backtick-quoted**, with embedded backticks doubled.
  A column called ``my col`` or ``a.b`` or ``x`y`` is one identifier, never
  three tokens. A nested field is addressed by passing a list of parts.
* **Plain values are bound as `:name` parameters** wherever Databricks SQL
  accepts a marker (queries, DML, time travel, table functions). DDL clauses
  that grammatically demand a string literal (TBLPROPERTIES, COMMENT, TAGS)
  get a fully escaped literal instead.
* **`predicate` strings and `updates` values are SQL expressions by contract**,
  exactly as in delta-rs: ``"price * 1.1"`` is an expression, not a string. They
  are passed through verbatim, so never build them from untrusted input. Use
  `new_values=` on `update` to set plain values; those are parameterized.

Bulk writes (append, overwrite, replaceWhere, MERGE) need the data on the
server. They go through a Unity Catalog volume: the Arrow data is written to a
Parquet file in memory, uploaded to ``/Volumes/<staging_volume>/deltaswamp-
staging/``, read back with `read_files`, and the staged file is deleted
afterwards whether or not the statement succeeded. Without `staging_volume=`
those operations are refused up front so the router moves on.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import io
import re
import uuid
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from ..capability import OPERATION_ENGINES, READ_OPERATIONS, Capability, Operation
from ..capability import Engine as EngineKind
from ..catalog import ResolvedTable, TableType
from ..errors import UnreachableTableError
from ..identity import RefKind, split_identifier
from .base import missing_method
from .sql_backend import (
    ParameterBinder,
    SdkStatementBackend,
    SqlParameter,
    SqlStatementError,
    StatementBackend,
    parameters_from_mapping,
)

__all__ = ["SqlEngine", "SqlFallbackWarning", "SqlMerger", "SqlStatementError"]

#: Operations that need the incoming data uploaded to a staging volume first.
_STAGED_OPERATIONS: frozenset[Operation] = frozenset(
    {
        Operation.APPEND,
        Operation.OVERWRITE,
        Operation.REPLACE_WHERE,
        Operation.MERGE,
        Operation.MERGE_SCHEMA,
    }
)

#: Table types with no writable storage of their own.
_NOT_WRITABLE_TYPES = frozenset(
    {TableType.VIEW, TableType.METRIC_VIEW, TableType.MATERIALIZED_VIEW}
)

#: Reads that only make sense on a Delta table with a transaction log.
_LOG_READS: frozenset[Operation] = frozenset(
    {Operation.TIME_TRAVEL, Operation.CDF, Operation.HISTORY}
)

#: Tuning knobs from the delta-rs signatures that have no meaning on a
#: warehouse and change nothing about the result, so they are accepted.
_IGNORABLE_TUNING = frozenset(
    {
        "max_concurrent_tasks",
        "max_spill_size",
        "min_commit_interval",
        "post_commithook_properties",
        "max_commit_retries",
        "streamed_exec",
        "error_on_type_mismatch",
    }
)

_PRIVILEGE = re.compile(r"^[A-Za-z][A-Za-z _]*$")
_FORBIDDEN_IN_TYPE = re.compile(r"(;|--|/\*|\*/|'|\"|`)")
_STAGING_DIR = "deltaswamp-staging"

#: delta-rs `TableFeatures` members whose wire name is not just camelCase.
_FEATURE_ALIASES = {"TimestampWithoutTimezone": "timestampNtz"}


# ------------------------------------------------------------------ quoting


def _quote(identifier: str) -> str:
    """Backtick-quote an identifier, doubling any embedded backtick."""
    return "`" + str(identifier).replace("`", "``") + "`"


def _column(column: str | Sequence[str]) -> str:
    """Quote a column. A str is one identifier; a list is a nested field path."""
    if isinstance(column, str):
        return _quote(column)
    parts = list(column)
    if not parts:
        raise ValueError("a column path needs at least one part")
    return ".".join(_quote(p) for p in parts)


def _columns(columns: Sequence[str | Sequence[str]]) -> str:
    return ", ".join(_column(c) for c in columns)


def _qualified(name: str) -> str:
    """Quote a dotted, possibly backtick-quoted, name part by part."""
    return ".".join(_quote(p) for p in split_identifier(name))


def _name(table: ResolvedTable) -> str:
    ref = table.ref
    if ref.kind is not RefKind.CATALOG:
        raise UnreachableTableError(
            "address the table by name",
            "a SQL warehouse addresses tables by name; this is a path reference",
        )
    return ".".join(_quote(p) for p in (ref.catalog, ref.schema, ref.table) if p is not None)


def _literal(value: str | None) -> str:
    """A string literal for grammar positions that refuse parameter markers.

    Databricks string literals honour backslash escapes, so both the backslash
    and the quote must be escaped or ``\\'`` would close the literal early.
    """
    if value is None:
        return "NULL"
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _sql_type(type_text: str) -> str:
    """Pass a SQL type through, refusing text that could end the statement."""
    text = str(type_text).strip()
    if not text or _FORBIDDEN_IN_TYPE.search(text):
        raise ValueError(f"{type_text!r} is not a SQL type")
    return text


def _principal(principal: str) -> str:
    return _quote(principal)


def _privileges(privileges: str | Sequence[str]) -> str:
    items = [privileges] if isinstance(privileges, str) else list(privileges)
    if not items:
        raise ValueError("at least one privilege is required")
    for item in items:
        if not _PRIVILEGE.match(item):
            raise ValueError(f"{item!r} is not a privilege name")
    return ", ".join(" ".join(i.upper().split()) for i in items)


def _feature_name(feature: Any) -> str:
    """The wire name of a feature, from a string or a delta-rs `TableFeatures`."""
    if isinstance(feature, str):
        return feature
    text = str(getattr(feature, "value", feature))
    member = text.rsplit(".", 1)[-1]
    if member in _FEATURE_ALIASES:
        return _FEATURE_ALIASES[member]
    return member[:1].lower() + member[1:]


def _timestamp_text(value: Any) -> str:
    if isinstance(value, _dt.datetime | _dt.date):
        return value.isoformat()
    return str(value)


def _reject_unsupported(what: str, kwargs: Mapping[str, Any]) -> None:
    """Refuse arguments the warehouse cannot honour rather than ignoring them."""
    unsupported = sorted(
        k for k, v in kwargs.items() if k not in _IGNORABLE_TUNING and v not in (None, False)
    )
    if unsupported:
        raise UnreachableTableError(
            f"{what} via a SQL warehouse",
            f"these arguments have no SQL equivalent here: {', '.join(unsupported)}",
            "drop them, or route this to an engine that supports them",
        )


def _rows(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    if hasattr(result, "to_pylist"):
        rows: list[dict[str, Any]] = result.to_pylist()
        return rows
    return [dict(r) for r in result]


def _first(result: Any) -> dict[str, Any]:
    rows = _rows(result)
    return rows[0] if rows else {}


_OK: dict[str, Any] = {"status": "ok"}


def _ok() -> dict[str, Any]:
    return dict(_OK)


# ------------------------------------------------------------------ arrow


def _to_arrow(data: Any) -> Any:
    """Coerce whatever the caller passed into a `pyarrow.Table`."""
    import pyarrow as pa

    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])
    if isinstance(data, pa.RecordBatchReader):
        return data.read_all()
    module = type(data).__module__
    if module.startswith("pandas"):
        return pa.Table.from_pandas(data, preserve_index=False)
    if hasattr(data, "__arrow_c_stream__"):
        return pa.table(data)
    if module.startswith("polars") and hasattr(data, "to_arrow"):
        return data.to_arrow()
    return pa.table(data)


def _parquet_bytes(table: Any) -> bytes:
    import pyarrow.parquet as pq

    buffer = io.BytesIO()
    # Spark reads nanosecond Parquet timestamps badly. Coercing to micros raises
    # instead of truncating, so a real sub-microsecond value is never lost.
    pq.write_table(table, buffer, coerce_timestamps="us")
    return buffer.getvalue()


# ------------------------------------------------------------------ engine


class SqlFallbackWarning(UserWarning):
    """Raised when an operation is served by a SQL warehouse rather than directly."""


class SqlEngine:
    """Executes operations as SQL against a Databricks warehouse."""

    kind = EngineKind.SQL
    supports_distributed_scan = False
    supports_predicates = True
    supports_timestamp_travel = True
    #: `INSERT WITH SCHEMA EVOLUTION` / `MERGE WITH SCHEMA EVOLUTION`.
    supports_schema_merge = True
    #: A statement cannot replace a table's schema without also discarding its
    #: properties, comments and grants, so this is refused rather than done.
    supports_schema_overwrite = False
    #: Neither is expressible through a stateless statement API: there is no
    #: session to carry `userMetadata`, and no SQL surface for txn actions.
    supports_idempotent_txn = False
    supports_commit_metadata = False
    supports_writer_properties = False
    #: Emulated with REPLACE WHERE over the partition values present in the data.
    supports_dynamic_overwrite = True
    #: Server-side-only request shapes: OPTIMIZE FULL, OPTIMIZE ... WHERE and
    #: CLUSTER BY AUTO. Declared so the router sends them here and nowhere else.
    supports_optimize_full = True
    supports_optimize_predicate = True
    supports_auto_clustering = True

    #: Refuse to build a dynamic-overwrite predicate wider than this.
    max_dynamic_partitions = 1000

    def __init__(
        self,
        *,
        warehouse_id: str | None = None,
        http_path: str | None = None,
        profile: str | None = None,
        host: str | None = None,
        token: str | None = None,
        config: Any = None,
        warn_on_use: bool = True,
        staging_volume: str | None = None,
        client: Any = None,
        backend: StatementBackend | None = None,
        wait_timeout: str = "30s",
        timeout: float | None = 3600.0,
        auto_select_warehouse: bool = True,
    ) -> None:
        self._warehouse_id = warehouse_id or _warehouse_from_http_path(http_path)
        self._http_path = http_path
        self._profile = profile
        self._host = host
        self._token = token
        self._config = config
        self._warn_on_use = warn_on_use
        self._staging_volume = _check_volume(staging_volume)
        self._client = client
        self._backend: StatementBackend | None = backend
        self._wait_timeout = wait_timeout
        self._timeout = timeout
        self._auto_select = auto_select_warehouse
        self._warehouse_label: str | None = (
            f"warehouse {self._warehouse_id}" if self._warehouse_id else None
        )
        self._selection_error: str | None = None

    # ----------------------------------------------------------- capabilities

    def supports(self, operation: Operation, table: ResolvedTable, **shape: Any) -> Capability:
        if self._backend is None and not self.available():
            return Capability(
                operation,
                ok=False,
                reason="the SQL fallback needs databricks-sdk and pyarrow, and one is missing",
                remedy="pip install 'deltaswamp[pyarrow]'",
            )
        # Structural guard: never claim an operation with no method behind it.
        gap = missing_method(self, operation)
        if gap is not None:
            return gap

        routing = OPERATION_ENGINES.get(operation)
        if routing is None or self.kind not in routing.engines:
            return Capability(
                operation, ok=False, reason=f"{operation.value} is not expressible as SQL here"
            )
        if table.ref.kind is not RefKind.CATALOG:
            return Capability(
                operation,
                ok=False,
                reason="a SQL warehouse addresses tables by name; this is a path reference",
            )
        if operation in _STAGED_OPERATIONS and self._staging_volume is None:
            return Capability(
                operation,
                ok=False,
                reason=(
                    f"{operation.value} through a SQL warehouse needs the data on the server, "
                    "and no staging volume is configured to upload it to"
                ),
                remedy="SqlEngine(..., staging_volume='<catalog>.<schema>.<volume>')",
            )
        refusal = self._table_type_refusal(operation, table)
        if refusal is not None:
            return refusal
        if self._backend is None and self._resolve_warehouse() is None:
            return Capability(
                operation,
                ok=False,
                reason=self._selection_error or "no SQL warehouse configured",
                remedy="ds.connect(..., allow_sql_fallback=True, warehouse_id='<id>')",
            )
        return Capability(operation, ok=True, engine=self.kind)

    @staticmethod
    def _table_type_refusal(operation: Operation, table: ResolvedTable) -> Capability | None:
        kind = table.table_type
        if operation is Operation.REFRESH:
            if kind in (TableType.MATERIALIZED_VIEW, TableType.STREAMING_TABLE):
                return None
            return Capability(
                operation,
                ok=False,
                reason=(
                    f"REFRESH applies to materialized views and streaming tables; this is "
                    f"{kind.value if kind else 'a table of unknown type'}"
                ),
            )
        if kind in (TableType.VIEW, TableType.METRIC_VIEW) and operation in _LOG_READS:
            return Capability(
                operation,
                ok=False,
                reason=f"the table is a {kind.value}, which has no Delta log to read",
            )
        if kind in _NOT_WRITABLE_TYPES and operation not in READ_OPERATIONS:
            return Capability(
                operation,
                ok=False,
                reason=f"the table is a {kind.value}, which cannot be written or altered",
                remedy="change the underlying tables, or its definition",
            )
        return None

    @staticmethod
    def available() -> bool:
        try:
            import databricks.sdk  # noqa: F401
            import pyarrow  # noqa: F401
        except ImportError:
            return False
        return True

    # ---------------------------------------------------- warehouse selection

    def _workspace(self) -> Any:
        if self._client is None:
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.core import Config

            cfg = self._config or Config(profile=self._profile, host=self._host, token=self._token)
            self._client = WorkspaceClient(config=cfg)
        return self._client

    def _resolve_warehouse(self) -> str | None:
        """The warehouse id to use, choosing one if none was given.

        Never raises: a failure is recorded as the refusal reason instead,
        because `supports()` must answer rather than throw.
        """
        if self._warehouse_id is not None:
            return self._warehouse_id
        if self._selection_error is not None:
            return None
        if not self._auto_select:
            self._selection_error = "no SQL warehouse configured"
            return None
        try:
            chosen = _choose_warehouse(list(self._workspace().warehouses.list()))
        except Exception as exc:
            self._selection_error = f"could not list SQL warehouses to choose one: {exc}"
            return None
        if chosen is None:
            self._selection_error = "no SQL warehouse configured, and the workspace has none"
            return None
        warehouse, why = chosen
        self._warehouse_id = str(warehouse.id)
        self._warehouse_label = (
            f"warehouse {getattr(warehouse, 'name', None) or '?'!r} ({warehouse.id}), "
            f"chosen automatically as {why} because no warehouse_id was given"
        )
        return self._warehouse_id

    @property
    def warehouse_id(self) -> str | None:
        """The warehouse in use, choosing one now if it was left to us."""
        return self._resolve_warehouse()

    def _statement_backend(self) -> StatementBackend:
        if self._backend is None:
            warehouse = self._resolve_warehouse()
            if warehouse is None:
                raise UnreachableTableError(
                    "run SQL",
                    self._selection_error or "no SQL warehouse configured",
                    "ds.connect(..., allow_sql_fallback=True, warehouse_id='<id>')",
                )
            self._backend = SdkStatementBackend(
                self._workspace(),
                warehouse,
                wait_timeout=self._wait_timeout,
                timeout=self._timeout,
            )
        return self._backend

    # --------------------------------------------------------------- plumbing

    def _notify(self, operation: Operation | str) -> None:
        if self._warn_on_use:
            label = operation.value if isinstance(operation, Operation) else operation
            where = self._warehouse_label or "a Databricks SQL warehouse"
            warnings.warn(
                f"{label} is being served by {where} rather than by direct object-store "
                "access. Expect materially different latency and cost.",
                SqlFallbackWarning,
                stacklevel=4,
            )

    def execute(
        self,
        operation: Operation | str,
        statement: str,
        fetch: bool = True,
        *,
        parameters: Sequence[SqlParameter] | Mapping[str, Any] | None = None,
    ) -> Any:
        """Run one statement, announcing the fallback. Returns Arrow when `fetch`."""
        backend = self._statement_backend()
        self._notify(operation)
        if isinstance(parameters, Mapping):
            params: Sequence[SqlParameter] = parameters_from_mapping(parameters)
        else:
            params = list(parameters or ())
        return backend.execute(statement, params, fetch=fetch)

    def _run(
        self, operation: Operation, statement: str, binder: ParameterBinder | None = None
    ) -> None:
        self.execute(
            operation, statement, fetch=False, parameters=binder.parameters if binder else None
        )

    def _query(
        self, operation: Operation | str, statement: str, binder: ParameterBinder | None = None
    ) -> Any:
        return self.execute(operation, statement, parameters=binder.parameters if binder else None)

    # ------------------------------------------------------------------ extras

    def query(self, statement: str, parameters: Mapping[str, Any] | None = None) -> Any:
        """Run arbitrary SQL and return a `pyarrow.Table`.

        Bind values with `:name` markers and `parameters={"name": value}`
        rather than formatting them into `statement`.
        """
        return self.execute("query", statement, parameters=parameters)

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
        """SELECT the table. `predicate` is a SQL expression, passed verbatim."""
        binder = ParameterBinder()
        projection = _columns(columns) if columns else "*"
        sql = f"SELECT {projection} FROM {_name(table)}"
        if version is not None:
            sql += f" VERSION AS OF {int(version)}"
        elif timestamp is not None:
            sql += f" TIMESTAMP AS OF {binder.bind(_timestamp_text(timestamp), type='TIMESTAMP')}"
        if predicate:
            sql += f" WHERE {predicate}"
        op = Operation.TIME_TRAVEL if version is not None or timestamp else Operation.SCAN
        return self._query(op, sql, binder)

    def history(self, table: ResolvedTable, *, limit: int | None = None) -> list[dict[str, Any]]:
        sql = f"DESCRIBE HISTORY {_name(table)}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return _rows(self._query(Operation.HISTORY, sql))

    def detail(self, table: ResolvedTable, *, version: int | None = None) -> dict[str, Any]:
        """DESCRIBE DETAIL, plus the fields the other engines report.

        DESCRIBE DETAIL has no version, so it is fetched from the history, and
        it cannot describe an older version at all, so asking for one is refused.
        """
        if version is not None:
            raise UnreachableTableError(
                f"describe version {version} via SQL",
                "DESCRIBE DETAIL only describes the current version of a table",
                "open the table without a version",
            )
        name = _name(table)
        row = _first(self._query(Operation.DETAIL, f"DESCRIBE DETAIL {name}"))
        out = dict(row)
        if table.table_type not in (TableType.VIEW, TableType.METRIC_VIEW):
            latest = _first(self._query(Operation.DETAIL, f"DESCRIBE HISTORY {name} LIMIT 1"))
            if "version" in latest:
                out["version"] = latest["version"]
        features = list(row.get("tableFeatures") or [])
        out.setdefault("location", row.get("location"))
        out["min_reader_version"] = row.get("minReaderVersion")
        out["min_writer_version"] = row.get("minWriterVersion")
        out["table_features"] = features
        props = row.get("properties") or {}
        out["properties"] = dict(props.items() if hasattr(props, "items") else props)
        out["partition_columns"] = list(row.get("partitionColumns") or [])
        return out

    def describe_extended(self, table: ResolvedTable) -> dict[str, Any]:
        """DESCRIBE TABLE EXTENDED, as ``{"columns": [...], <detail>: <value>}``."""
        rows = _rows(self._query("describe_extended", f"DESCRIBE TABLE EXTENDED {_name(table)}"))
        columns: list[dict[str, Any]] = []
        info: dict[str, Any] = {}
        section = "columns"
        for row in rows:
            key = (row.get("col_name") or "").strip()
            if not key:
                continue
            if key.startswith("#"):
                section = "partition" if "Partition" in key else "info"
                continue
            if section == "columns":
                columns.append(
                    {"name": key, "type": row.get("data_type"), "comment": row.get("comment")}
                )
            elif section == "info":
                info[key] = row.get("data_type")
        return {"columns": columns, **info}

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
        **kwargs: Any,
    ) -> Any:
        """`table_changes(...)`. Versions and timestamps are bound as parameters."""
        _reject_unsupported("read the change data feed", kwargs)
        if starting_version is not None and starting_timestamp is not None:
            raise ValueError("pass starting_version or starting_timestamp, not both")
        if ending_version is not None and ending_timestamp is not None:
            raise ValueError("pass ending_version or ending_timestamp, not both")
        binder = ParameterBinder()
        if starting_timestamp is not None:
            start = binder.bind(_timestamp_text(starting_timestamp), type="TIMESTAMP")
        else:
            start = binder.bind(0 if starting_version is None else int(starting_version))
        args = [_literal(_name(table)), start]
        if ending_timestamp is not None:
            args.append(binder.bind(_timestamp_text(ending_timestamp), type="TIMESTAMP"))
        elif ending_version is not None:
            args.append(binder.bind(int(ending_version)))
        projection = _columns(columns) if columns else "*"
        sql = f"SELECT {projection} FROM table_changes({', '.join(args)})"
        if predicate:
            sql += f" WHERE {predicate}"
        return self._query(Operation.CDF, sql, binder)

    # ------------------------------------------------------------ staging

    @contextlib.contextmanager
    def _staged(self, data: Any) -> Iterator[tuple[str, Any]]:
        """Upload `data` as Parquet to the staging volume; always delete it after.

        Yields the `read_files(...)` relation and the Arrow table uploaded.
        """
        if self._staging_volume is None:
            raise UnreachableTableError(
                "stage data for a SQL write",
                "no staging volume is configured",
                "SqlEngine(..., staging_volume='<catalog>.<schema>.<volume>')",
            )
        arrow = _to_arrow(data)
        payload = _parquet_bytes(arrow)
        catalog, schema, volume = self._staging_volume
        path = f"/Volumes/{catalog}/{schema}/{volume}/{_STAGING_DIR}/{uuid.uuid4().hex}.parquet"
        files = self._workspace().files
        files.upload(path, io.BytesIO(payload), overwrite=False)
        try:
            yield f"read_files({_literal(path)}, format => 'parquet')", arrow
        finally:
            try:
                files.delete(path)
            except Exception as exc:
                warnings.warn(
                    f"could not delete the staged file {path}: {exc}. Remove it by hand.",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def _table_columns(self, table: ResolvedTable) -> list[str]:
        result = self._query(Operation.SCAN, f"SELECT * FROM {_name(table)} LIMIT 0")
        return list(result.schema.names)

    # ------------------------------------------------------------------ write

    def append(
        self,
        table: ResolvedTable,
        data: Any,
        *,
        schema_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """INSERT INTO ... BY NAME, so column order in the data does not matter."""
        self._check_write_args("append", schema_mode, kwargs)
        evolve = " WITH SCHEMA EVOLUTION" if schema_mode == "merge" else ""
        with self._staged(data) as (source, _):
            self._run(
                Operation.APPEND,
                f"INSERT{evolve} INTO {_name(table)} BY NAME SELECT * FROM {source}",
            )

    def overwrite(
        self,
        table: ResolvedTable,
        data: Any,
        *,
        predicate: str | None = None,
        partition_overwrite: str = "static",
        schema_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """INSERT OVERWRITE, or INSERT INTO ... REPLACE WHERE with a predicate.

        `partition_overwrite="dynamic"` replaces exactly the partitions present
        in the data, by building a parameterized REPLACE WHERE from them.
        """
        self._check_write_args("overwrite", schema_mode, kwargs)
        if partition_overwrite not in ("static", "dynamic"):
            raise UnreachableTableError(
                f"overwrite with partition_overwrite={partition_overwrite!r}",
                "the only modes are 'static' and 'dynamic'",
            )
        if partition_overwrite == "dynamic" and predicate is not None:
            raise UnreachableTableError(
                "overwrite dynamically with a predicate",
                "dynamic partition overwrite derives its own predicate from the data, so "
                "it cannot be combined with an explicit one",
            )
        name = _name(table)
        with self._staged(data) as (source, arrow):
            if partition_overwrite == "static" and predicate is None:
                self._run(
                    Operation.OVERWRITE, f"INSERT OVERWRITE {name} BY NAME SELECT * FROM {source}"
                )
                return
            binder = ParameterBinder()
            if predicate is None:
                predicate = self._dynamic_predicate(table, arrow, binder)
            # REPLACE WHERE takes no column list, so order the projection to
            # match the table rather than trusting the data's column order.
            target = self._table_columns(table)
            missing = [c for c in target if c not in arrow.column_names]
            extra = [c for c in arrow.column_names if c not in target]
            if missing or extra:
                raise UnreachableTableError(
                    "overwrite with REPLACE WHERE",
                    "the data's columns do not match the table's"
                    + (f"; missing {', '.join(missing)}" if missing else "")
                    + (f"; unexpected {', '.join(extra)}" if extra else ""),
                )
            self._run(
                Operation.REPLACE_WHERE,
                f"INSERT INTO {name} REPLACE WHERE {predicate} "
                f"SELECT {_columns(target)} FROM {source}",
                binder,
            )

    def _dynamic_predicate(self, table: ResolvedTable, arrow: Any, binder: ParameterBinder) -> str:
        partitions = list(table.partition_columns)
        if not partitions:
            raise UnreachableTableError(
                "overwrite dynamically",
                "the table is not partitioned, so there are no partitions to replace",
                "use a plain overwrite, or a predicate",
            )
        missing = [c for c in partitions if c not in arrow.column_names]
        if missing:
            raise UnreachableTableError(
                "overwrite dynamically",
                f"the data has no {', '.join(missing)} column, so the partitions it would "
                "replace cannot be determined",
            )
        tuples = {tuple(r[c] for c in partitions) for r in arrow.select(partitions).to_pylist()}
        if not tuples:
            raise UnreachableTableError(
                "overwrite dynamically", "the data is empty, so no partitions are implied"
            )
        if len(tuples) > self.max_dynamic_partitions:
            raise UnreachableTableError(
                "overwrite dynamically",
                f"the data spans {len(tuples)} partitions, above the "
                f"{self.max_dynamic_partitions} limit for a generated predicate",
                "overwrite the whole table, or raise SqlEngine.max_dynamic_partitions",
            )
        clauses = []
        for values in sorted(tuples, key=repr):
            terms = [
                f"{_quote(c)} IS NULL" if v is None else f"{_quote(c)} = {binder.bind(v)}"
                for c, v in zip(partitions, values, strict=True)
            ]
            clauses.append("(" + " AND ".join(terms) + ")")
        return " OR ".join(clauses)

    @staticmethod
    def _check_write_args(what: str, schema_mode: str | None, kwargs: Mapping[str, Any]) -> None:
        if schema_mode == "overwrite":
            raise UnreachableTableError(
                f"{what} with schema_mode='overwrite' via SQL",
                "replacing a table's schema from a statement also discards its properties, "
                "comments and grants",
                "use CREATE OR REPLACE TABLE deliberately, via query()",
            )
        if schema_mode not in (None, "merge"):
            raise UnreachableTableError(
                f"{what} with schema_mode={schema_mode!r}", "the modes are 'merge' and 'overwrite'"
            )
        # partition_by / target_file_size describe how files are laid out, which
        # the table already decides on the server; neither changes the rows.
        rest = {k: v for k, v in kwargs.items() if k not in ("partition_by", "target_file_size")}
        _reject_unsupported(what, rest)

    def delete(self, table: ResolvedTable, predicate: str | None = None) -> dict[str, Any]:
        """DELETE FROM. `predicate` is a SQL expression, passed verbatim."""
        sql = f"DELETE FROM {_name(table)}"
        if predicate:
            sql += f" WHERE {predicate}"
        return _first(self._query(Operation.DELETE, sql)) or _ok()

    def update(
        self,
        table: ResolvedTable,
        *,
        updates: dict[str, str] | None = None,
        new_values: dict[str, Any] | None = None,
        predicate: str | None = None,
    ) -> dict[str, Any]:
        """UPDATE ... SET.

        `updates` maps column to a SQL *expression* (``{"n": "n + 1"}``), as in
        delta-rs. `new_values` maps column to a plain value, bound as a
        parameter (``{"city": "O'Hare"}``). Column names are always quoted.
        """
        if not updates and not new_values:
            raise UnreachableTableError("update", "no updates given")
        binder = ParameterBinder()
        assignments = [f"{_quote(k)} = {v}" for k, v in (updates or {}).items()]
        assignments += [f"{_quote(k)} = {binder.bind(v)}" for k, v in (new_values or {}).items()]
        sql = f"UPDATE {_name(table)} SET {', '.join(assignments)}"
        if predicate:
            sql += f" WHERE {predicate}"
        return _first(self._query(Operation.UPDATE, sql, binder)) or _ok()

    def merge(
        self,
        table: ResolvedTable,
        source: Any,
        predicate: str,
        *,
        source_alias: str | None = None,
        target_alias: str | None = None,
        merge_schema: bool = False,
        **kwargs: Any,
    ) -> SqlMerger:
        """Start a MERGE. Mirrors delta-rs: add clauses, then `execute()`."""
        _reject_unsupported("merge", kwargs)
        if self._staging_volume is None:
            raise UnreachableTableError(
                "merge via SQL",
                "the source has to be uploaded, and no staging volume is configured",
                "SqlEngine(..., staging_volume='<catalog>.<schema>.<volume>')",
            )
        return SqlMerger(
            self,
            table,
            source,
            predicate,
            source_alias=source_alias or "source",
            target_alias=target_alias or "target",
            merge_schema=merge_schema,
        )

    # ------------------------------------------------------------ maintenance

    def optimize(
        self,
        table: ResolvedTable,
        *,
        zorder_by: list[str] | None = None,
        full: bool = False,
        predicate: str | None = None,
        partition_filters: list[tuple[str, str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """OPTIMIZE [FULL] [WHERE ...] [ZORDER BY (...)].

        `partition_filters` (delta-rs style ``[("col", "=", value)]``) become a
        parameterized WHERE; `predicate` is a raw SQL expression.
        """
        _reject_unsupported("optimize", kwargs)
        binder = ParameterBinder()
        sql = f"OPTIMIZE {_name(table)}"
        if full:
            sql += " FULL"
        conditions = [f"({predicate})"] if predicate else []
        conditions += [_filter_sql(f, binder) for f in partition_filters or []]
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        if zorder_by:
            sql += f" ZORDER BY ({_columns(zorder_by)})"
        return _first(
            self._query(Operation.ZORDER if zorder_by else Operation.OPTIMIZE, sql, binder)
        )

    def zorder(self, table: ResolvedTable, columns: list[str], **kwargs: Any) -> dict[str, Any]:
        if not columns:
            raise ValueError("Z-ORDER needs at least one column")
        return self.optimize(table, zorder_by=columns, **kwargs)

    def vacuum(
        self,
        table: ResolvedTable,
        *,
        retention_hours: float | None = None,
        dry_run: bool = True,
        lite: bool = False,
        full: bool = False,
        enforce_retention_duration: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """VACUUM [LITE|FULL] [RETAIN n HOURS] [DRY RUN]. Returns the file paths."""
        _reject_unsupported("vacuum", kwargs)
        if lite and full:
            raise ValueError("VACUUM is LITE or FULL, not both")
        if not enforce_retention_duration:
            raise UnreachableTableError(
                "vacuum below the retention threshold via SQL",
                "the retention-duration check is a session setting, and the statement API "
                "has no session to change it in",
                "VACUUM with a retention at or above delta.deletedFileRetentionDuration",
            )
        sql = f"VACUUM {_name(table)}"
        if lite:
            sql += " LITE"
        elif full:
            sql += " FULL"
        if retention_hours is not None:
            hours = float(retention_hours)
            sql += f" RETAIN {int(hours) if hours.is_integer() else hours} HOURS"
        if dry_run:
            sql += " DRY RUN"
        rows = _rows(self._query(Operation.VACUUM, sql))
        return [str(r["path"]) for r in rows if r.get("path") is not None]

    def restore(self, table: ResolvedTable, target: Any, **kwargs: Any) -> dict[str, Any]:
        """RESTORE TABLE ... TO VERSION AS OF n | TIMESTAMP AS OF '...'."""
        _reject_unsupported("restore", kwargs)
        if isinstance(target, bool):
            raise TypeError("restore target must be a version or a timestamp")
        if isinstance(target, int):
            clause = f"VERSION AS OF {int(target)}"
        else:
            clause = f"TIMESTAMP AS OF {_literal(_timestamp_text(target))}"
        return _first(self._query(Operation.RESTORE, f"RESTORE TABLE {_name(table)} TO {clause}"))

    def repair(
        self, table: ResolvedTable, *, dry_run: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """FSCK REPAIR TABLE [DRY RUN]. Shaped like delta-rs's result."""
        _reject_unsupported("repair", kwargs)
        sql = f"FSCK REPAIR TABLE {_name(table)}" + (" DRY RUN" if dry_run else "")
        rows = _rows(self._query(Operation.REPAIR, sql))
        removed = [str(r.get("dataFilePath") or r.get("path")) for r in rows if r]
        return {"dry_run": dry_run, "files_removed": removed}

    def reorg(
        self,
        table: ResolvedTable,
        *,
        purge: bool = True,
        iceberg_compat_version: int | None = None,
        predicate: str | None = None,
    ) -> dict[str, Any]:
        """REORG TABLE ... APPLY (PURGE) or APPLY (UPGRADE UNIFORM(...))."""
        if iceberg_compat_version is not None:
            clause = (
                f"APPLY (UPGRADE UNIFORM(ICEBERG_COMPAT_VERSION={int(iceberg_compat_version)}))"
            )
        elif purge:
            clause = "APPLY (PURGE)"
        else:
            raise ValueError("REORG needs purge=True or an iceberg_compat_version")
        where = f" WHERE {predicate}" if predicate else ""
        self._run(Operation.REORG, f"REORG TABLE {_name(table)}{where} {clause}")
        return _ok()

    def clone(
        self,
        table: ResolvedTable,
        target: str,
        *,
        shallow: bool = True,
        replace: bool = False,
        if_not_exists: bool = False,
        version: int | None = None,
        timestamp: Any = None,
    ) -> dict[str, Any]:
        """CREATE [OR REPLACE] TABLE target {SHALLOW|DEEP} CLONE source [AS OF]."""
        if replace and if_not_exists:
            raise ValueError("replace and if_not_exists are mutually exclusive")
        verb = "CREATE OR REPLACE TABLE" if replace else "CREATE TABLE"
        if if_not_exists:
            verb += " IF NOT EXISTS"
        kind = "SHALLOW CLONE" if shallow else "DEEP CLONE"
        sql = f"{verb} {_qualified(target)} {kind} {_name(table)}"
        if version is not None:
            sql += f" VERSION AS OF {int(version)}"
        elif timestamp is not None:
            sql += f" TIMESTAMP AS OF {_literal(_timestamp_text(timestamp))}"
        return _first(self._query(Operation.CLONE, sql)) or _ok()

    def analyze(
        self,
        table: ResolvedTable,
        *,
        columns: list[str] | str | None = None,
        delta_statistics: bool = False,
        noscan: bool = False,
    ) -> dict[str, Any]:
        """ANALYZE TABLE ... COMPUTE [DELTA] STATISTICS [NOSCAN | FOR COLUMNS | FOR ALL COLUMNS].

        `columns="all"` computes every column's statistics.
        """
        sql = f"ANALYZE TABLE {_name(table)} COMPUTE"
        sql += " DELTA STATISTICS" if delta_statistics else " STATISTICS"
        if isinstance(columns, str):
            if columns.lower() not in ("all", "*"):
                raise ValueError("columns must be a list of names, or 'all'")
            sql += " FOR ALL COLUMNS"
        elif columns:
            sql += f" FOR COLUMNS {_columns(columns)}"
        if noscan:
            if columns:
                raise ValueError("NOSCAN computes table-level statistics only")
            sql += " NOSCAN"
        self._run(Operation.ANALYZE, sql)
        return _ok()

    def sync_iceberg_metadata(self, table: ResolvedTable) -> dict[str, Any]:
        """Regenerate Iceberg metadata after an external write.

        Required when anything other than Databricks writes to a table with
        Iceberg reads enabled; without it the Iceberg view silently goes stale.
        """
        self._run(Operation.SYNC_ICEBERG, f"MSCK REPAIR TABLE {_name(table)} SYNC METADATA")
        return _ok()

    def refresh(self, table: ResolvedTable, *, full: bool = False) -> dict[str, Any]:
        """REFRESH MATERIALIZED VIEW / STREAMING TABLE [FULL], by table type."""
        if table.table_type is TableType.MATERIALIZED_VIEW:
            kind = "MATERIALIZED VIEW"
        elif table.table_type is TableType.STREAMING_TABLE:
            kind = "STREAMING TABLE"
        else:
            raise UnreachableTableError(
                "refresh",
                "REFRESH applies to materialized views and streaming tables only",
            )
        self._run(Operation.REFRESH, f"REFRESH {kind} {_name(table)}" + (" FULL" if full else ""))
        return _ok()

    # ------------------------------------------------------------------ ddl

    def _alter(
        self, operation: Operation | str, table: ResolvedTable, clause: str
    ) -> dict[str, Any]:
        self.execute(operation, f"ALTER TABLE {_name(table)} {clause}", fetch=False)
        return _ok()

    def add_columns(self, table: ResolvedTable, fields: Any, **kwargs: Any) -> dict[str, Any]:
        """ADD COLUMNS. `fields` is ``{name: sql_type}`` or Arrow fields/schema."""
        _reject_unsupported("add columns", kwargs)
        pairs = [f"{_quote(n)} {_sql_type(t)}" for n, t in _column_types(fields)]
        if not pairs:
            raise ValueError("no columns given")
        return self._alter(Operation.ADD_COLUMN, table, f"ADD COLUMNS ({', '.join(pairs)})")

    def drop_column(self, table: ResolvedTable, column: str | Sequence[str]) -> dict[str, Any]:
        return self._alter(Operation.DROP_COLUMN, table, f"DROP COLUMN {_column(column)}")

    def rename_column(self, table: ResolvedTable, old: str, new: str) -> dict[str, Any]:
        return self._alter(
            Operation.RENAME_COLUMN, table, f"RENAME COLUMN {_column(old)} TO {_quote(new)}"
        )

    def set_properties(
        self, table: ResolvedTable, properties: dict[str, str], **kwargs: Any
    ) -> dict[str, Any]:
        """SET TBLPROPERTIES. Keys and values are escaped literals (DDL takes no markers)."""
        kwargs.pop("raise_if_not_exists", None)
        _reject_unsupported("set table properties", kwargs)
        if not properties:
            raise ValueError("no properties given")
        pairs = ", ".join(f"{_literal(k)} = {_literal(str(v))}" for k, v in properties.items())
        return self._alter(Operation.SET_PROPERTIES, table, f"SET TBLPROPERTIES ({pairs})")

    def unset_properties(
        self, table: ResolvedTable, keys: Sequence[str], *, if_exists: bool = True
    ) -> dict[str, Any]:
        keys = [keys] if isinstance(keys, str) else list(keys)
        if not keys:
            raise ValueError("no property keys given")
        guard = "IF EXISTS " if if_exists else ""
        listed = ", ".join(_literal(k) for k in keys)
        return self._alter(
            Operation.UNSET_PROPERTIES, table, f"UNSET TBLPROPERTIES {guard}({listed})"
        )

    def add_feature(self, table: ResolvedTable, feature: Any, **kwargs: Any) -> dict[str, Any]:
        """Enable one feature, or several, via ``delta.feature.<name> = 'supported'``."""
        kwargs.pop("allow_protocol_versions_increase", None)
        _reject_unsupported("add a table feature", kwargs)
        features = feature if isinstance(feature, list | tuple) else [feature]
        return self.set_properties(
            table, {f"delta.feature.{_feature_name(f)}": "supported" for f in features}
        )

    def drop_feature(
        self, table: ResolvedTable, feature: Any, *, truncate_history: bool = False
    ) -> dict[str, Any]:
        clause = f"DROP FEATURE {_quote(_feature_name(feature))}"
        if truncate_history:
            clause += " TRUNCATE HISTORY"
        return self._alter(Operation.DROP_FEATURE, table, clause)

    def add_constraint(
        self, table: ResolvedTable, constraints: dict[str, str], **kwargs: Any
    ) -> dict[str, Any]:
        """ADD CONSTRAINT ... CHECK. The expression is SQL by contract."""
        _reject_unsupported("add a constraint", kwargs)
        for name, expression in constraints.items():
            self._alter(
                Operation.ADD_CONSTRAINT,
                table,
                f"ADD CONSTRAINT {_quote(name)} CHECK ({expression})",
            )
        return _ok()

    def drop_constraint(
        self, table: ResolvedTable, name: str, *, if_exists: bool = False
    ) -> dict[str, Any]:
        guard = "IF EXISTS " if if_exists else ""
        return self._alter(
            Operation.DROP_CONSTRAINT, table, f"DROP CONSTRAINT {guard}{_quote(name)}"
        )

    def set_comment(self, table: ResolvedTable, comment: str | None) -> dict[str, Any]:
        self._run(Operation.SET_COMMENT, f"COMMENT ON TABLE {_name(table)} IS {_literal(comment)}")
        return _ok()

    def set_column_comment(
        self, table: ResolvedTable, column: str | Sequence[str], comment: str | None
    ) -> dict[str, Any]:
        return self._alter(
            Operation.SET_COLUMN_COMMENT,
            table,
            f"ALTER COLUMN {_column(column)} COMMENT {_literal(comment or '')}",
        )

    def alter_column_type(
        self, table: ResolvedTable, column: str | Sequence[str], new_type: str
    ) -> dict[str, Any]:
        return self._alter(
            Operation.ALTER_COLUMN_TYPE,
            table,
            f"ALTER COLUMN {_column(column)} TYPE {_sql_type(new_type)}",
        )

    def set_not_null(self, table: ResolvedTable, column: str | Sequence[str]) -> dict[str, Any]:
        return self._alter(
            Operation.SET_NOT_NULL, table, f"ALTER COLUMN {_column(column)} SET NOT NULL"
        )

    def drop_not_null(self, table: ResolvedTable, column: str | Sequence[str]) -> dict[str, Any]:
        return self._alter(
            Operation.DROP_NOT_NULL, table, f"ALTER COLUMN {_column(column)} DROP NOT NULL"
        )

    def cluster_by(
        self, table: ResolvedTable, columns: Sequence[str] | str | None
    ) -> dict[str, Any]:
        """CLUSTER BY (cols); an empty list or None is NONE; ``"auto"`` is AUTO."""
        if isinstance(columns, str):
            if columns.lower() != "auto":
                raise ValueError("columns must be a list of names, None, or 'auto'")
            clause = "CLUSTER BY AUTO"
        elif not columns:
            clause = "CLUSTER BY NONE"
        else:
            clause = f"CLUSTER BY ({_columns(columns)})"
        return self._alter(Operation.CLUSTER_BY, table, clause)

    # ---------------------------------------------------- governance extras

    def undrop(self, full_name: str) -> dict[str, Any]:
        """UNDROP TABLE, within the catalog's retention window."""
        self.execute("undrop", f"UNDROP TABLE {_qualified(full_name)}", fetch=False)
        return _ok()

    def set_row_filter(
        self, table: ResolvedTable, function_name: str, columns: Sequence[str]
    ) -> dict[str, Any]:
        clause = f"SET ROW FILTER {_qualified(function_name)} ON ({_columns(columns)})"
        return self._alter("set_row_filter", table, clause)

    def drop_row_filter(self, table: ResolvedTable) -> dict[str, Any]:
        return self._alter("drop_row_filter", table, "DROP ROW FILTER")

    def set_column_mask(
        self,
        table: ResolvedTable,
        column: str | Sequence[str],
        function_name: str,
        using_columns: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        clause = f"ALTER COLUMN {_column(column)} SET MASK {_qualified(function_name)}"
        if using_columns:
            clause += f" USING COLUMNS ({_columns(using_columns)})"
        return self._alter("set_column_mask", table, clause)

    def drop_column_mask(self, table: ResolvedTable, column: str | Sequence[str]) -> dict[str, Any]:
        return self._alter("drop_column_mask", table, f"ALTER COLUMN {_column(column)} DROP MASK")

    def set_tags(
        self,
        table: ResolvedTable,
        tags: Mapping[str, str],
        column: str | Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if not tags:
            raise ValueError("no tags given")
        pairs = ", ".join(f"{_literal(k)} = {_literal(str(v))}" for k, v in tags.items())
        target = f"ALTER COLUMN {_column(column)} " if column is not None else ""
        return self._alter("set_tags", table, f"{target}SET TAGS ({pairs})")

    def unset_tags(
        self,
        table: ResolvedTable,
        keys: Sequence[str],
        column: str | Sequence[str] | None = None,
    ) -> dict[str, Any]:
        keys = [keys] if isinstance(keys, str) else list(keys)
        if not keys:
            raise ValueError("no tag keys given")
        target = f"ALTER COLUMN {_column(column)} " if column is not None else ""
        listed = ", ".join(_literal(k) for k in keys)
        return self._alter("unset_tags", table, f"{target}UNSET TAGS ({listed})")

    def set_owner(self, table: ResolvedTable, principal: str) -> dict[str, Any]:
        return self._alter("set_owner", table, f"OWNER TO {_principal(principal)}")

    def grant(
        self, table: ResolvedTable, privileges: str | Sequence[str], principal: str
    ) -> dict[str, Any]:
        sql = f"GRANT {_privileges(privileges)} ON TABLE {_name(table)} TO {_principal(principal)}"
        self.execute("grant", sql, fetch=False)
        return _ok()

    def revoke(
        self, table: ResolvedTable, privileges: str | Sequence[str], principal: str
    ) -> dict[str, Any]:
        sql = (
            f"REVOKE {_privileges(privileges)} ON TABLE {_name(table)} FROM {_principal(principal)}"
        )
        self.execute("revoke", sql, fetch=False)
        return _ok()

    def create_table_as(
        self,
        name: str,
        select_sql: str,
        *,
        replace: bool = False,
        comment: str | None = None,
        properties: Mapping[str, str] | None = None,
        cluster_by: Sequence[str] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """CREATE [OR REPLACE] TABLE name ... AS <select_sql>.

        `select_sql` is SQL by contract; bind values in it with `:name` markers.
        """
        sql = ("CREATE OR REPLACE TABLE " if replace else "CREATE TABLE ") + _qualified(name)
        if cluster_by:
            sql += f" CLUSTER BY ({_columns(cluster_by)})"
        if comment is not None:
            sql += f" COMMENT {_literal(comment)}"
        if properties:
            pairs = ", ".join(f"{_literal(k)} = {_literal(str(v))}" for k, v in properties.items())
            sql += f" TBLPROPERTIES ({pairs})"
        sql += f" AS {select_sql}"
        self.execute("create_table_as", sql, fetch=False, parameters=parameters)
        return _ok()

    def plan_scan(self, table: ResolvedTable, **kwargs: Any) -> list[Any]:
        raise NotImplementedError("the SQL engine has no split-planning surface")

    def execute_scan(self, table: ResolvedTable, splits: list[Any], **kwargs: Any) -> Any:
        raise NotImplementedError("see plan_scan")


# ------------------------------------------------------------------ MERGE


class SqlMerger:
    """Builds one MERGE statement, with the same surface as delta-rs's `TableMerger`.

    Clause `updates` values and every `predicate` are SQL expressions by
    contract, referring to the target and source by their aliases. Nothing runs
    until `execute()`, which stages the source, runs the MERGE and deletes the
    staged file.
    """

    def __init__(
        self,
        engine: SqlEngine,
        table: ResolvedTable,
        source: Any,
        predicate: str,
        *,
        source_alias: str = "source",
        target_alias: str = "target",
        merge_schema: bool = False,
    ) -> None:
        self._engine = engine
        self._table = table
        self._source = source
        self._predicate = predicate
        self._source_alias = source_alias
        self._target_alias = target_alias
        self._merge_schema = merge_schema
        self._arrow: Any = None
        # (kind, predicate, action), resolved against the source columns at execute().
        self._clauses: list[tuple[str, str | None, Any]] = []

    def _add(self, kind: str, predicate: str | None, action: Any) -> SqlMerger:
        self._clauses.append((kind, predicate, action))
        return self

    def when_matched_update(
        self, updates: dict[str, str], predicate: str | None = None
    ) -> SqlMerger:
        return self._add("MATCHED", predicate, ("UPDATE", dict(updates)))

    def when_matched_update_all(
        self, predicate: str | None = None, except_cols: list[str] | None = None
    ) -> SqlMerger:
        return self._add("MATCHED", predicate, ("UPDATE_ALL", list(except_cols or [])))

    def when_matched_delete(self, predicate: str | None = None) -> SqlMerger:
        return self._add("MATCHED", predicate, ("DELETE", None))

    def when_not_matched_insert(
        self, updates: dict[str, str], predicate: str | None = None
    ) -> SqlMerger:
        return self._add("NOT MATCHED", predicate, ("INSERT", dict(updates)))

    def when_not_matched_insert_all(
        self, predicate: str | None = None, except_cols: list[str] | None = None
    ) -> SqlMerger:
        return self._add("NOT MATCHED", predicate, ("INSERT_ALL", list(except_cols or [])))

    def when_not_matched_by_source_update(
        self, updates: dict[str, str], predicate: str | None = None
    ) -> SqlMerger:
        return self._add("NOT MATCHED BY SOURCE", predicate, ("UPDATE", dict(updates)))

    def when_not_matched_by_source_delete(self, predicate: str | None = None) -> SqlMerger:
        return self._add("NOT MATCHED BY SOURCE", predicate, ("DELETE", None))

    def _action(self, action: Any, source_columns: list[str]) -> str:
        verb, arg = action
        tgt, src = _quote(self._target_alias), _quote(self._source_alias)
        if verb == "DELETE":
            return "DELETE"
        if verb == "UPDATE":
            return "UPDATE SET " + ", ".join(f"{tgt}.{_quote(k)} = {v}" for k, v in arg.items())
        if verb == "INSERT":
            names = ", ".join(_quote(k) for k in arg)
            return f"INSERT ({names}) VALUES ({', '.join(arg.values())})"
        # *_ALL: spell the columns out, so except_cols works on every runtime.
        cols = [c for c in source_columns if c not in set(arg)]
        if not cols:
            raise ValueError("except_cols excludes every source column")
        if verb == "UPDATE_ALL":
            if not arg:
                return "UPDATE SET *"
            return "UPDATE SET " + ", ".join(f"{tgt}.{_quote(c)} = {src}.{_quote(c)}" for c in cols)
        if not arg:
            return "INSERT *"
        names = ", ".join(_quote(c) for c in cols)
        values = ", ".join(f"{src}.{_quote(c)}" for c in cols)
        return f"INSERT ({names}) VALUES ({values})"

    def statement(self, relation: str, source_columns: list[str]) -> str:
        """The MERGE text for a source `relation`. Exposed for inspection and tests."""
        if not self._clauses:
            raise ValueError("a MERGE needs at least one WHEN clause")
        evolve = " WITH SCHEMA EVOLUTION" if self._merge_schema else ""
        sql = (
            f"MERGE{evolve} INTO {_name(self._table)} AS {_quote(self._target_alias)} "
            f"USING (SELECT * FROM {relation}) AS {_quote(self._source_alias)} "
            f"ON {self._predicate}"
        )
        for kind, predicate, action in self._clauses:
            condition = f" AND {predicate}" if predicate else ""
            sql += f" WHEN {kind}{condition} THEN {self._action(action, source_columns)}"
        return sql

    def execute(self) -> dict[str, Any]:
        """Stage the source, run the MERGE, delete the staged file. Returns metrics."""
        if not self._clauses:
            raise ValueError("a MERGE needs at least one WHEN clause")
        with self._engine._staged(self._source) as (relation, arrow):
            sql = self.statement(relation, list(arrow.column_names))
            result = self._engine._query(Operation.MERGE, sql)
        return _first(result) or _ok()


# ------------------------------------------------------------------ helpers


def _filter_sql(spec: tuple[str, str, Any], binder: ParameterBinder) -> str:
    """One delta-rs partition filter, as a parameterized SQL condition."""
    column, op, value = spec
    op = op.strip().lower()
    if op in ("=", "!=", "<", "<=", ">", ">="):
        if value is None:
            if op not in ("=", "!="):
                raise ValueError(f"cannot compare {column} {op} NULL")
            return f"{_quote(column)} IS {'NOT ' if op == '!=' else ''}NULL"
        return f"{_quote(column)} {op} {binder.bind(value)}"
    if op in ("in", "not in"):
        values = list(value)
        if not values:
            raise ValueError(f"an '{op}' filter on {column} needs at least one value")
        markers = ", ".join(binder.bind(v) for v in values)
        return f"{_quote(column)} {op.upper()} ({markers})"
    raise ValueError(f"unsupported partition filter operator {op!r}")


def _column_types(fields: Any) -> list[tuple[str, str]]:
    """``{name: sql_type}``, an Arrow schema, or a list of Arrow fields."""
    if isinstance(fields, Mapping):
        return [(str(k), str(v)) for k, v in fields.items()]
    if hasattr(fields, "names") and hasattr(fields, "field"):  # an Arrow schema
        items = [fields.field(i) for i in range(len(fields.names))]
    elif hasattr(fields, "name") and hasattr(fields, "type"):  # a single field
        items = [fields]
    else:
        items = list(fields)
    out: list[tuple[str, str]] = []
    for field in items:
        name = getattr(field, "name", None)
        arrow_type = getattr(field, "type", None)
        if name is None or arrow_type is None:
            raise UnreachableTableError(
                "add columns via SQL",
                "the SQL engine needs a {name: sql_type} mapping or Arrow fields",
            )
        out.append((str(name), _arrow_to_sql(arrow_type)))
    return out


def _arrow_to_sql(arrow_type: Any) -> str:
    import pyarrow.types as t

    if t.is_boolean(arrow_type):
        return "BOOLEAN"
    if t.is_int8(arrow_type):
        return "TINYINT"
    if t.is_int16(arrow_type):
        return "SMALLINT"
    if t.is_int32(arrow_type):
        return "INT"
    if t.is_int64(arrow_type):
        return "BIGINT"
    if t.is_float32(arrow_type):
        return "FLOAT"
    if t.is_float64(arrow_type):
        return "DOUBLE"
    if t.is_decimal(arrow_type):
        return f"DECIMAL({arrow_type.precision},{arrow_type.scale})"
    if t.is_string(arrow_type) or t.is_large_string(arrow_type):
        return "STRING"
    if t.is_binary(arrow_type) or t.is_large_binary(arrow_type):
        return "BINARY"
    if t.is_date(arrow_type):
        return "DATE"
    if t.is_timestamp(arrow_type):
        return "TIMESTAMP" if arrow_type.tz else "TIMESTAMP_NTZ"
    if t.is_list(arrow_type) or t.is_large_list(arrow_type):
        return f"ARRAY<{_arrow_to_sql(arrow_type.value_type)}>"
    if t.is_map(arrow_type):
        return f"MAP<{_arrow_to_sql(arrow_type.key_type)}, {_arrow_to_sql(arrow_type.item_type)}>"
    if t.is_struct(arrow_type):
        inner = ", ".join(
            f"{_quote(arrow_type.field(i).name)}: {_arrow_to_sql(arrow_type.field(i).type)}"
            for i in range(arrow_type.num_fields)
        )
        return f"STRUCT<{inner}>"
    raise UnreachableTableError("add columns via SQL", f"no SQL type for Arrow type {arrow_type}")


def _check_volume(volume: str | None) -> tuple[str, str, str] | None:
    if volume is None:
        return None
    parts = split_identifier(volume)
    if len(parts) != 3:
        raise ValueError(f"staging_volume must be '<catalog>.<schema>.<volume>', got {volume!r}")
    for part in parts:
        if "/" in part or part in (".", ".."):
            raise ValueError(f"staging_volume part {part!r} cannot appear in a volume path")
    return parts[0], parts[1], parts[2]


def _warehouse_from_http_path(http_path: str | None) -> str | None:
    if not http_path:
        return None
    tail = http_path.rstrip("/").rsplit("/", 1)[-1]
    if "/warehouses/" not in http_path and "/endpoints/" not in http_path:
        raise ValueError(
            f"{http_path!r} is not a SQL warehouse HTTP path (/sql/1.0/warehouses/<id>)"
        )
    return tail


def _choose_warehouse(warehouses: list[Any]) -> tuple[Any, str] | None:
    """Prefer a running serverless warehouse, then any running one, then any.

    Starting a stopped warehouse is left to the first statement, which starts
    it on demand; we never start one ourselves just to answer `supports()`.
    """

    def state(w: Any) -> str:
        return str(getattr(getattr(w, "state", None), "value", getattr(w, "state", "")) or "")

    usable = [w for w in warehouses if state(w) not in ("DELETED", "DELETING")]
    running = [w for w in usable if state(w) == "RUNNING"]
    serverless = [w for w in running if getattr(w, "enable_serverless_compute", False)]
    if serverless:
        return serverless[0], "the running serverless warehouse"
    if running:
        return running[0], "a running warehouse"
    if usable:
        return usable[0], "the first available warehouse (it is not running and will start)"
    return None
