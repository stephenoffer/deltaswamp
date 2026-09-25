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
import io
import numbers
import uuid
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from ..capability import OPERATION_ENGINES, READ_OPERATIONS, Capability, Operation
from ..capability import Engine as EngineKind
from ..catalog import ResolvedTable, TableType
from ..errors import InvalidArgumentError, SqlFallbackWarning, UnreachableTableError
from ..identity import RefKind, split_identifier
from . import sql_text as sq
from .base import missing_method
from .sql_backend import (
    ParameterBinder,
    SdkStatementBackend,
    SqlParameter,
    SqlStatementError,
    StatementBackend,
    _check_timeout,
    _check_wait_timeout,
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
    {Operation.TIME_TRAVEL, Operation.CDF, Operation.HISTORY, Operation.DETAIL}
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


_STAGING_DIR = "deltaswamp-staging"


def _reject_unsupported(what: str, kwargs: Mapping[str, Any]) -> None:
    """Refuse arguments the warehouse cannot honor rather than ignoring them."""
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


def _history_entry(row: dict[str, Any]) -> dict[str, Any]:
    """One DESCRIBE HISTORY row in the shape delta-rs's history() reports.

    Arrow hands the map columns back as lists of pairs and the timestamp as a
    datetime, where every other engine gives dicts and epoch milliseconds, so
    ``entry["operationParameters"]["mode"]`` raised on this engine alone.
    """
    import datetime as _dt

    out = dict(row)
    for key in ("operationParameters", "operationMetrics"):
        value = out.get(key)
        if isinstance(value, list):
            out[key] = dict(value)
    stamp = out.get("timestamp")
    if isinstance(stamp, _dt.datetime):
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=_dt.UTC)
        out["timestamp"] = int(stamp.timestamp() * 1000)
    if out.get("version") is not None:
        out["version"] = int(out["version"])
    return out


#: The warehouse's DML metric names, and the delta-rs names for the same count.
_DML_ALIASES = {
    "num_updated_rows": "num_target_rows_updated",
    "num_deleted_rows": "num_target_rows_deleted",
    "num_inserted_rows": "num_target_rows_inserted",
}


def _dml_metrics(row: dict[str, Any], affected_as: str | None) -> dict[str, Any]:
    """Warehouse DML metrics, also under the names delta-rs and the kernel use.

    DELETE and UPDATE report only ``num_affected_rows`` and MERGE its own
    names, so ``t.delete(...)["num_deleted_rows"]`` was a KeyError exactly
    when the warehouse served the call.
    """
    if not row:
        return _ok()
    out = dict(row)
    if affected_as is not None and "num_affected_rows" in out:
        out.setdefault(affected_as, out["num_affected_rows"])
    if affected_as is None:
        for ours, theirs in _DML_ALIASES.items():
            if ours in out:
                out.setdefault(theirs, out[ours])
    return out


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
        # A *named* index is data (every other engine keeps it as a column);
        # only an anonymous one is dropped rather than written as
        # `__index_level_0__`.
        unnamed = all(name is None for name in getattr(data, "index", data).names or [None])
        # preserve_index=None would still keep a range-like named index as
        # metadata only, dropping the column.
        return pa.Table.from_pandas(data, preserve_index=not unnamed)
    if hasattr(data, "__arrow_c_stream__"):
        return pa.table(data)
    if module.startswith("polars") and hasattr(data, "to_arrow"):
        return data.to_arrow()
    if isinstance(data, list) and data and all(isinstance(r, Mapping) for r in data):
        # Rows as dicts; `pa.table(list)` rejects them with a confusing TypeError.
        return pa.Table.from_pylist(data)
    return pa.table(data)


def _unstageable(arrow_type: Any) -> bool:
    import pyarrow.types as t

    if t.is_time(arrow_type) or t.is_duration(arrow_type) or t.is_interval(arrow_type):
        return True
    if t.is_dictionary(arrow_type):
        return _unstageable(arrow_type.value_type)
    if t.is_list(arrow_type) or t.is_large_list(arrow_type) or t.is_fixed_size_list(arrow_type):
        return _unstageable(arrow_type.value_type)
    if t.is_map(arrow_type):
        return _unstageable(arrow_type.key_type) or _unstageable(arrow_type.item_type)
    if t.is_struct(arrow_type):
        return any(_unstageable(arrow_type.field(i).type) for i in range(arrow_type.num_fields))
    return False


def _stageable_type(arrow_type: Any) -> Any:
    """The type to write to Parquet so Spark's reader understands it."""
    import pyarrow as pa
    import pyarrow.types as t

    if isinstance(arrow_type, pa.BaseExtensionType):
        # e.g. arrow.uuid is written with Parquet's UUID annotation, which
        # Spark refuses to read; the storage type carries the same bytes.
        return _stageable_type(arrow_type.storage_type)
    if t.is_float16(arrow_type):
        # Parquet's FLOAT16 logical type is not readable by Spark/read_files.
        return pa.float32()
    if t.is_decimal(arrow_type) and arrow_type.precision > 38:
        raise UnreachableTableError(
            "write via a SQL warehouse",
            f"{arrow_type} has more than the 38 digits of precision a Databricks DECIMAL holds",
            "cast it to DECIMAL(38, s), DOUBLE or STRING first",
        )
    if t.is_decimal(arrow_type) and arrow_type.bit_width == 256:
        return pa.decimal128(arrow_type.precision, arrow_type.scale)
    if t.is_dictionary(arrow_type):
        return pa.dictionary(arrow_type.index_type, _stageable_type(arrow_type.value_type))
    if t.is_list(arrow_type):
        return pa.list_(arrow_type.value_field.with_type(_stageable_type(arrow_type.value_type)))
    if t.is_large_list(arrow_type):
        return pa.large_list(
            arrow_type.value_field.with_type(_stageable_type(arrow_type.value_type))
        )
    if t.is_fixed_size_list(arrow_type):
        return pa.list_(arrow_type.value_field.with_type(_stageable_type(arrow_type.value_type)))
    if t.is_map(arrow_type):
        return pa.map_(
            arrow_type.key_field.with_type(_stageable_type(arrow_type.key_type)),
            arrow_type.item_field.with_type(_stageable_type(arrow_type.item_type)),
        )
    if t.is_struct(arrow_type):
        fields = [arrow_type.field(i) for i in range(arrow_type.num_fields)]
        return pa.struct([f.with_type(_stageable_type(f.type)) for f in fields])
    return arrow_type


def _stageable(table: Any) -> Any:
    """Refuse column sets Spark cannot tell apart; cast types it cannot read."""
    import pyarrow as pa

    if not table.column_names:
        raise UnreachableTableError(
            "write via a SQL warehouse", "the data has no columns", "pass at least one column"
        )
    seen: dict[str, str] = {}
    for column in table.column_names:
        other = seen.get(column.lower())
        seen.setdefault(column.lower(), column)
        if other is not None:
            # Databricks resolves names case-insensitively: "id" and "ID" (or a
            # repeated name) are one ambiguous column once staged.
            raise UnreachableTableError(
                "write via a SQL warehouse",
                f"the data has columns {other!r} and {column!r}, which Databricks treats as "
                "the same column",
                "rename one of them",
            )
    schema = pa.schema(
        [f.with_type(_stageable_type(f.type)) for f in table.schema], metadata=table.schema.metadata
    )
    return table if schema.equals(table.schema) else table.cast(schema)


def _parquet_bytes(table: Any) -> bytes:
    import pyarrow.parquet as pq

    # Databricks SQL has no TIME type and reads no Parquet TIME/duration
    # column; refuse here with the column named, not after an upload with an
    # opaque read_files error (or a pyarrow one for duration).
    bad = [f.name for f in table.schema if _unstageable(f.type)]
    if bad:
        raise UnreachableTableError(
            "write via a SQL warehouse",
            f"column(s) {', '.join(bad)} have a time-of-day, duration or interval type, "
            "which Databricks SQL cannot store",
            "cast them to STRING, BIGINT or TIMESTAMP first",
        )

    buffer = io.BytesIO()
    # Spark reads nanosecond Parquet timestamps badly. Coercing to micros raises
    # instead of truncating, so a real sub-microsecond value is never lost.
    pq.write_table(table, buffer, coerce_timestamps="us")
    return buffer.getvalue()


# ------------------------------------------------------------------ engine


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
    #: The warehouse formats partition values itself.
    supports_negative_decimal_partition_values = True
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
        # Validate now: a bad value otherwise surfaced only at the first query.
        self._wait_timeout = _check_wait_timeout(wait_timeout)
        _check_timeout(timeout)
        self._timeout = timeout
        self._auto_select = auto_select_warehouse
        self._warehouse_label: str | None = (
            f"warehouse {self._warehouse_id}" if self._warehouse_id else None
        )
        self._selection_error: str | None = None
        self._last_listing_error: str | None = None

    def __getstate__(self) -> dict[str, Any]:
        # The live WorkspaceClient (and the statement backend built on it) and
        # an explicit SDK Config cannot be pickled, so a Connection or Table
        # with the fallback enabled failed to pickle once the warehouse had
        # been used -- or at all, given config=. Ship plain configuration; a
        # copy rebuilds the client on first use.
        state = self.__dict__.copy()
        state["_client"] = None
        if isinstance(state.get("_backend"), SdkStatementBackend):
            state["_backend"] = None
        config = state.get("_config")
        if config is not None:
            from ..credentials.databricks import _config_attributes

            state["_config"] = None
            state["_config_kwargs"] = _config_attributes(config)
        return state

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
                reason=self._selection_error
                or self._last_listing_error
                or "no SQL warehouse configured",
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
        if kind is TableType.MATERIALIZED_VIEW and operation in _LOG_READS:
            # A materialized view is backed by Delta, but its log belongs to the
            # pipeline: DESCRIBE HISTORY and DESCRIBE DETAIL both fail with
            # EXPECT_TABLE_NOT_VIEW.
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table is a materialized view, whose Delta log is managed by its "
                    "pipeline; the warehouse refuses DESCRIBE HISTORY and DESCRIBE DETAIL on it"
                ),
            )
        widening = {"typeWidening", "typeWidening-preview"} & table.features or (
            table.properties.get("delta.enableTypeWidening", "").lower() == "true"
        )
        if operation is Operation.ALTER_COLUMN_TYPE and table.features and not widening:
            # Databricks answers DELTA_UNSUPPORTED_ALTER_TABLE_CHANGE_COL_OP even
            # for INT -> BIGINT until type widening is on.
            return Capability(
                operation,
                ok=False,
                reason="type widening is not enabled on this table, and Databricks changes "
                "a column's type only under it",
                remedy="t.set_properties({'delta.enableTypeWidening': 'true'}) first",
            )
        mapping = table.properties.get("delta.columnMapping.mode", "none").lower()
        if (
            operation in (Operation.DROP_COLUMN, Operation.RENAME_COLUMN)
            and table.features
            and mapping not in ("name", "id")
        ):
            # DELTA_UNSUPPORTED_DROP_COLUMN otherwise: without
            # column mapping a column's name is also its name in every file.
            return Capability(
                operation,
                ok=False,
                reason="the table does not use column mapping, and Databricks renames or "
                "drops a column only under it",
                remedy="t.set_properties({'delta.columnMapping.mode': 'name'}) first",
            )
        if operation is Operation.ZORDER and "clustering" in table.features:
            return Capability(
                operation,
                ok=False,
                reason=(
                    "the table is liquid-clustered, and Databricks refuses Z-ORDER on it "
                    "(DELTA_CLUSTERING_WITH_ZORDER_BY)"
                ),
                remedy="t.optimize() clusters by the table's clustering keys; "
                "t.cluster_by([...]) changes them",
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
            from .._sdk import workspace_client

            carried: dict[str, Any] = getattr(self, "_config_kwargs", None) or {}
            if self._config is None and carried:
                # An unpickled copy of an engine built from an explicit Config.
                from .._sdk import PRODUCT, sdk_version

                identity = {"product": PRODUCT, "product_version": sdk_version()}
                self._client = workspace_client(**{**carried, **identity})
            else:
                self._client = workspace_client(
                    config=self._config, profile=self._profile, host=self._host, token=self._token
                )
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
            # Not cached: a transient listing failure (network, throttling)
            # used to disable the fallback for the rest of the process.
            self._last_listing_error = f"could not list SQL warehouses to choose one: {exc}"
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
                    self._selection_error
                    or self._last_listing_error
                    or "no SQL warehouse configured",
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
        limit: int | None = None,
    ) -> Any:
        """SELECT the table. `predicate` is a SQL expression, passed verbatim.

        `limit` becomes a real `LIMIT`. The warehouse computes the whole result
        before any of it is streamed back, so without this a `head(3)` on a big
        table is a full scan billed to the warehouse -- and slow enough to hit
        the statement timeout.
        """
        if version is not None and timestamp is not None:
            # One of them was silently ignored before.
            raise InvalidArgumentError("pass version or timestamp, not both")
        binder = ParameterBinder()
        projection = sq.columns(columns) if columns else "*"
        sql = f"SELECT {projection} FROM {sq.name(table)}"
        if version is not None:
            sql += f" VERSION AS OF {int(version)}"
        elif timestamp is not None:
            sql += f" TIMESTAMP AS OF {binder.bind(sq.timestamp_text(timestamp), type='TIMESTAMP')}"
        if sq.predicate(predicate) is not None:
            sql += f" WHERE {predicate}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        op = Operation.TIME_TRAVEL if version is not None or timestamp else Operation.SCAN
        return self._query(op, sql, binder)

    def history(self, table: ResolvedTable, *, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None:
            limit = int(limit)
            if limit < 0:
                raise InvalidArgumentError(f"history limit must be >= 0, got {limit}")
            if limit == 0:
                # `if limit:` read 0 as "no limit" and returned the whole history.
                return []
        sql = f"DESCRIBE HISTORY {sq.name(table)}"
        if limit is not None:
            sql += f" LIMIT {limit}"
        return [_history_entry(row) for row in _rows(self._query(Operation.HISTORY, sql))]

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
        name = sq.name(table)
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
        rows = _rows(self._query("describe_extended", f"DESCRIBE TABLE EXTENDED {sq.name(table)}"))
        columns: list[dict[str, Any]] = []
        info: dict[str, Any] = {}
        section = "columns"
        for row in rows:
            key = (row.get("col_name") or "").strip()
            if not key:
                continue
            if key.startswith("#"):
                # "# col_name" is the column header *inside* the partition (or
                # clustering) section; it must not end that section, or every
                # partition column landed in the table details as a property.
                if key.lower().startswith("# col_name"):
                    continue
                section = "partition" if "Partition" in key or "Clustering" in key else "info"
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
            raise InvalidArgumentError("pass starting_version or starting_timestamp, not both")
        if ending_version is not None and ending_timestamp is not None:
            raise InvalidArgumentError("pass ending_version or ending_timestamp, not both")
        binder = ParameterBinder()
        if starting_timestamp is not None:
            start = binder.bind(sq.timestamp_text(starting_timestamp), type="TIMESTAMP")
        else:
            start = binder.bind(0 if starting_version is None else int(starting_version))
        args = [sq.literal(sq.name(table)), start]
        if ending_timestamp is not None:
            args.append(binder.bind(sq.timestamp_text(ending_timestamp), type="TIMESTAMP"))
        elif ending_version is not None:
            args.append(binder.bind(int(ending_version)))
        projection = sq.columns(columns) if columns else "*"
        sql = f"SELECT {projection} FROM table_changes({', '.join(args)})"
        if sq.predicate(predicate) is not None:
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
        arrow = _stageable(_to_arrow(data))
        payload = _parquet_bytes(arrow)
        catalog, schema, volume = self._staging_volume
        path = f"/Volumes/{catalog}/{schema}/{volume}/{_STAGING_DIR}/{uuid.uuid4().hex}.parquet"
        files = self._workspace().files
        try:
            files.upload(path, io.BytesIO(payload), overwrite=False)
        except BaseException:
            # A multipart upload that fails part-way can leave a partial file.
            with contextlib.suppress(Exception):
                files.delete(path)
            raise
        # Each statement projects the staged columns itself (`_staged_columns`):
        # read_files adds a `_rescued_data` column of its own.
        relation = f"read_files({sq.literal(path)}, format => 'parquet')"
        try:
            yield relation, arrow
        finally:
            try:
                files.delete(path)
            except Exception as exc:
                warnings.warn(
                    f"could not delete the staged file {path}: {exc}. Remove it by hand.",
                    RuntimeWarning,
                    stacklevel=3,
                )

    @staticmethod
    def _staged_columns(arrow: Any) -> str:
        """The staged file's own columns, spelled out.

        `read_files` with an inferred schema can add a `_rescued_data` column,
        so `SELECT *` over it fed that column to INSERT BY NAME (an unknown
        column error, or a new column under schema evolution) and to MERGE's
        `UPDATE SET *` / `INSERT *`. The Arrow schema staged is known exactly.
        """
        return sq.columns(list(arrow.column_names))

    def _table_columns(self, table: ResolvedTable) -> list[str]:
        result = self._query(Operation.SCAN, f"SELECT * FROM {sq.name(table)} LIMIT 0")
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
        with self._staged(data) as (source, arrow):
            self._run(
                Operation.APPEND,
                f"INSERT{evolve} INTO {sq.name(table)} BY NAME "
                f"SELECT {self._staged_columns(arrow)} FROM {source}",
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
        predicate = sq.predicate(predicate)
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
        name = sq.name(table)
        # schema_mode="merge" was accepted and then dropped, so new columns in
        # the data failed the overwrite instead of evolving the schema.
        evolve = " WITH SCHEMA EVOLUTION" if schema_mode == "merge" else ""
        with self._staged(data) as (source, arrow):
            if partition_overwrite == "static" and predicate is None:
                self._run(
                    Operation.OVERWRITE,
                    f"INSERT{evolve} OVERWRITE {name} BY NAME "
                    f"SELECT {self._staged_columns(arrow)} FROM {source}",
                )
                return
            binder = ParameterBinder()
            if predicate is None:
                predicate = self._dynamic_predicate(table, arrow, binder)
            # REPLACE WHERE takes no column list, so order the projection to
            # match the table rather than trusting the data's column order.
            # Column names resolve case-insensitively on Databricks, so match
            # them that way too, projecting the data's own spelling.
            target = self._table_columns(table)
            by_lower = {c.lower(): c for c in arrow.column_names}
            target_lower = {c.lower() for c in target}
            missing = [c for c in target if c.lower() not in by_lower]
            extra = [c for c in arrow.column_names if c.lower() not in target_lower]
            if missing or (extra and schema_mode != "merge"):
                raise UnreachableTableError(
                    "overwrite with REPLACE WHERE",
                    "the data's columns do not match the table's"
                    + (f"; missing {', '.join(missing)}" if missing else "")
                    + (f"; unexpected {', '.join(extra)}" if extra else ""),
                    "pass schema_mode='merge' to add new columns"
                    if extra and not missing
                    else None,
                )
            projection = [by_lower[c.lower()] for c in target] + extra
            self._run(
                Operation.REPLACE_WHERE,
                f"INSERT{evolve} INTO {name} REPLACE WHERE {predicate} "
                f"SELECT {sq.columns(projection)} FROM {source}",
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
        by_lower = {c.lower(): c for c in arrow.column_names}
        missing = [c for c in partitions if c.lower() not in by_lower]
        if missing:
            raise UnreachableTableError(
                "overwrite dynamically",
                f"the data has no {', '.join(missing)} column, so the partitions it would "
                "replace cannot be determined",
            )
        # The catalog may spell a partition column differently from the data
        # ("Region" vs "region"); Databricks resolves names case-insensitively.
        source_cols = [by_lower[c.lower()] for c in partitions]
        tuples = {tuple(r[c] for c in source_cols) for r in arrow.select(source_cols).to_pylist()}
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
                f"{sq.quote(c)} IS NULL" if v is None else f"{sq.quote(c)} = {binder.bind(v)}"
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
                "use CREATE OR REPLACE TABLE explicitly, via query()",
            )
        if schema_mode not in (None, "merge"):
            raise UnreachableTableError(
                f"{what} with schema_mode={schema_mode!r}", "the modes are 'merge' and 'overwrite'"
            )
        # partition_by / target_file_size describe how files are laid out, which
        # the table already decides on the server; neither changes the rows.
        rest = {k: v for k, v in kwargs.items() if k not in ("partition_by", "target_file_size")}
        _reject_unsupported(what, rest)

    def delete(
        self, table: ResolvedTable, predicate: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """DELETE FROM. `predicate` is a SQL expression, passed verbatim."""
        # delta-rs arguments arrive here via Table.delete(**kwargs); without
        # this they were a bare TypeError, and harmless tuning was refused.
        _reject_unsupported("delete", kwargs)
        sql = f"DELETE FROM {sq.name(table)}"
        if sq.predicate(predicate) is not None:
            sql += f" WHERE {predicate}"
        return _dml_metrics(_first(self._query(Operation.DELETE, sql)), "num_deleted_rows")

    def update(
        self,
        table: ResolvedTable,
        *,
        updates: dict[str, str] | None = None,
        new_values: dict[str, Any] | None = None,
        predicate: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """UPDATE ... SET.

        `updates` maps column to a SQL *expression* (``{"n": "n + 1"}``), as in
        delta-rs. `new_values` maps column to a plain value, bound as a
        parameter (``{"city": "O'Hare"}``). Column names are always quoted.
        """
        _reject_unsupported("update", kwargs)
        if not updates and not new_values:
            raise UnreachableTableError("update", "no updates given")
        binder = ParameterBinder()
        # A None expression is NULL; spliced as text it was the identifier `None`.
        assignments = [
            f"{sq.quote(k)} = {'NULL' if v is None else v}" for k, v in (updates or {}).items()
        ]
        # None is written as the NULL literal: a STRING-typed NULL marker is
        # refused by ANSI store assignment into an INT/DATE/... column.
        assignments += [
            f"{sq.quote(k)} = {'NULL' if v is None else binder.bind(v)}"
            for k, v in (new_values or {}).items()
        ]
        both = {str(k).lower() for k in (updates or {})} & {
            str(k).lower() for k in (new_values or {})
        }
        if both:
            # Two assignments to one column fail on the warehouse only.
            raise InvalidArgumentError(
                f"column(s) {', '.join(sorted(both))} are set by both updates and new_values"
            )
        sql = f"UPDATE {sq.name(table)} SET {', '.join(assignments)}"
        if sq.predicate(predicate) is not None:
            sql += f" WHERE {predicate}"
        return _dml_metrics(_first(self._query(Operation.UPDATE, sql, binder)), "num_updated_rows")

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
        # An empty ON clause was a syntax error only after the source was staged.
        if not isinstance(predicate, str) or not predicate.strip():
            raise InvalidArgumentError(
                f"a MERGE needs a join predicate (its ON condition), got {predicate!r}"
            )
        source_alias = source_alias or "source"
        target_alias = target_alias or "target"
        if source_alias.lower() == target_alias.lower():
            # Aliases resolve case-insensitively: every reference is ambiguous.
            raise InvalidArgumentError(f"the source and target aliases are both {source_alias!r}")
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
        sql = f"OPTIMIZE {sq.name(table)}"
        if full:
            sql += " FULL"
        conditions = [f"({predicate})"] if sq.predicate(predicate) is not None else []
        conditions += [_filter_sql(f, binder) for f in partition_filters or []]
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        if zorder_by:
            sql += f" ZORDER BY ({sq.columns(zorder_by)})"
        return _first(
            self._query(Operation.ZORDER if zorder_by else Operation.OPTIMIZE, sql, binder)
        )

    def zorder(self, table: ResolvedTable, columns: list[str], **kwargs: Any) -> dict[str, Any]:
        if not columns:
            raise InvalidArgumentError("Z-ORDER needs at least one column")
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
            raise InvalidArgumentError("VACUUM is LITE or FULL, not both")
        if not enforce_retention_duration:
            raise UnreachableTableError(
                "vacuum below the retention threshold via SQL",
                "the retention-duration check is a session setting, and the statement API "
                "has no session to change it in",
                "VACUUM with a retention at or above delta.deletedFileRetentionDuration",
            )
        sql = f"VACUUM {sq.name(table)}"
        if lite:
            sql += " LITE"
        elif full:
            sql += " FULL"
        if retention_hours is not None:
            hours = float(retention_hours)
            if not 0 <= hours < float("inf"):
                # "RETAIN nan HOURS" / "RETAIN -1 HOURS" failed on the warehouse
                # with a parse error that named neither the argument nor the value.
                raise InvalidArgumentError(
                    f"retention_hours must be a number >= 0, got {retention_hours!r}"
                )
            sql += f" RETAIN {int(hours) if hours.is_integer() else hours} HOURS"
        if dry_run:
            sql += " DRY RUN"
        rows = _rows(self._query(Operation.VACUUM, sql))
        paths = [str(r["path"]) for r in rows if r.get("path") is not None]
        if not dry_run:
            # A real VACUUM answers with one row: the table's own location, not
            # the files it deleted. Reporting that as a deleted file was wrong.
            location = str(table.location or "").rstrip("/")
            if not location and len(paths) == 1:
                return []
            paths = [p for p in paths if p.rstrip("/") != location]
        return paths

    def restore(self, table: ResolvedTable, target: Any, **kwargs: Any) -> dict[str, Any]:
        """RESTORE TABLE ... TO VERSION AS OF n | TIMESTAMP AS OF '...'."""
        _reject_unsupported("restore", kwargs)
        if isinstance(target, bool):
            raise TypeError("restore target must be a version or a timestamp")
        # numbers.Integral, so a NumPy integer is a version, not the timestamp '3'.
        if isinstance(target, numbers.Integral):
            clause = f"VERSION AS OF {int(target)}"
        else:
            clause = f"TIMESTAMP AS OF {sq.literal(sq.timestamp_text(target))}"
        return _first(self._query(Operation.RESTORE, f"RESTORE TABLE {sq.name(table)} TO {clause}"))

    def repair(
        self, table: ResolvedTable, *, dry_run: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """FSCK REPAIR TABLE [DRY RUN]. Shaped like delta-rs's result."""
        _reject_unsupported("repair", kwargs)
        sql = f"FSCK REPAIR TABLE {sq.name(table)}" + (" DRY RUN" if dry_run else "")
        rows = _rows(self._query(Operation.REPAIR, sql))
        # A row naming no file used to be reported as the file "None".
        removed = [str(p) for r in rows if r for p in [r.get("dataFilePath") or r.get("path")] if p]
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
            raise InvalidArgumentError("REORG needs purge=True or an iceberg_compat_version")
        where = f" WHERE {predicate}" if sq.predicate(predicate) is not None else ""
        self._run(Operation.REORG, f"REORG TABLE {sq.name(table)}{where} {clause}")
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
            raise InvalidArgumentError("replace and if_not_exists are mutually exclusive")
        if version is not None and timestamp is not None:
            # The timestamp was silently ignored.
            raise InvalidArgumentError("pass version or timestamp, not both")
        verb = "CREATE OR REPLACE TABLE" if replace else "CREATE TABLE"
        if if_not_exists:
            verb += " IF NOT EXISTS"
        kind = "SHALLOW CLONE" if shallow else "DEEP CLONE"
        sql = f"{verb} {sq.qualified(target)} {kind} {sq.name(table)}"
        if version is not None:
            sql += f" VERSION AS OF {int(version)}"
        elif timestamp is not None:
            sql += f" TIMESTAMP AS OF {sq.literal(sq.timestamp_text(timestamp))}"
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
        sql = f"ANALYZE TABLE {sq.name(table)} COMPUTE"
        sql += " DELTA STATISTICS" if delta_statistics else " STATISTICS"
        if isinstance(columns, str):
            if columns.lower() not in ("all", "*"):
                raise InvalidArgumentError("columns must be a list of names, or 'all'")
            sql += " FOR ALL COLUMNS"
        elif columns:
            sql += f" FOR COLUMNS {sq.columns(columns)}"
        if noscan:
            if columns:
                raise InvalidArgumentError("NOSCAN computes table-level statistics only")
            sql += " NOSCAN"
        self._run(Operation.ANALYZE, sql)
        return _ok()

    def sync_iceberg_metadata(self, table: ResolvedTable) -> dict[str, Any]:
        """Regenerate Iceberg metadata after an external write.

        Required when anything other than Databricks writes to a table with
        Iceberg reads enabled; without it the Iceberg view silently goes stale.
        """
        self._run(Operation.SYNC_ICEBERG, f"MSCK REPAIR TABLE {sq.name(table)} SYNC METADATA")
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
        self._run(
            Operation.REFRESH,
            f"REFRESH {kind} {sq.name(table)}" + (" FULL" if full else ""),
        )
        return _ok()

    # ------------------------------------------------------------------ ddl

    def _alter(
        self, operation: Operation | str, table: ResolvedTable, clause: str
    ) -> dict[str, Any]:
        self.execute(operation, f"ALTER TABLE {sq.name(table)} {clause}", fetch=False)
        return _ok()

    def create_managed(
        self,
        full_name: str,
        schema: Any,
        *,
        partition_by: list[str] | None = None,
        cluster_by: list[str] | None = None,
        properties: dict[str, str] | None = None,
        comment: str | None = None,
    ) -> None:
        """CREATE TABLE through the warehouse: an ordinary Databricks managed table.

        The route when Unity Catalog's staging-table API refuses this client,
        which Databricks does for any connector it has not allowlisted.
        """
        if partition_by and cluster_by:
            raise UnreachableTableError(
                f"create {full_name}", "a table is partitioned or clustered, not both"
            )
        nullable = {}
        if hasattr(schema, "names") and hasattr(schema, "field"):
            nullable = {schema.field(i).name: schema.field(i).nullable for i in range(len(schema))}
        columns = ", ".join(
            f"{sq.quote(n)} {sq.sql_type(t)}" + ("" if nullable.get(n, True) else " NOT NULL")
            for n, t in sq.column_types(schema)
        )
        sql = f"CREATE TABLE {sq.qualified(full_name)} ({columns}) USING DELTA"
        if partition_by:
            sql += f" PARTITIONED BY ({sq.columns(partition_by)})"
        if cluster_by:
            sql += f" CLUSTER BY ({sq.columns(cluster_by)})"
        if comment:
            sql += f" COMMENT {sq.literal(comment)}"
        if properties:
            sql += f" TBLPROPERTIES ({sq.properties(properties)})"
        self.execute(Operation.CREATE, sql, fetch=False)

    def add_columns(self, table: ResolvedTable, fields: Any, **kwargs: Any) -> dict[str, Any]:
        """ADD COLUMNS. `fields` is ``{name: sql_type}`` or Arrow fields/schema."""
        _reject_unsupported("add columns", kwargs)
        pairs = [f"{sq.quote(n)} {sq.sql_type(t)}" for n, t in sq.column_types(fields)]
        if not pairs:
            raise InvalidArgumentError("no columns given")
        return self._alter(Operation.ADD_COLUMN, table, f"ADD COLUMNS ({', '.join(pairs)})")

    def drop_column(self, table: ResolvedTable, column: str | Sequence[str]) -> dict[str, Any]:
        return self._alter(Operation.DROP_COLUMN, table, f"DROP COLUMN {sq.column(column)}")

    def rename_column(self, table: ResolvedTable, old: str, new: str) -> dict[str, Any]:
        return self._alter(
            Operation.RENAME_COLUMN,
            table,
            f"RENAME COLUMN {sq.column(old)} TO {sq.quote(new)}",
        )

    def set_properties(
        self, table: ResolvedTable, properties: dict[str, str], **kwargs: Any
    ) -> dict[str, Any]:
        """SET TBLPROPERTIES. Keys and values are escaped literals (DDL takes no markers)."""
        kwargs.pop("raise_if_not_exists", None)
        _reject_unsupported("set table properties", kwargs)
        if not properties:
            raise InvalidArgumentError("no properties given")
        pairs = sq.properties(properties)
        return self._alter(Operation.SET_PROPERTIES, table, f"SET TBLPROPERTIES ({pairs})")

    def unset_properties(
        self, table: ResolvedTable, keys: Sequence[str], *, if_exists: bool = True
    ) -> dict[str, Any]:
        keys = [keys] if isinstance(keys, str) else list(keys)
        if not keys:
            raise InvalidArgumentError("no property keys given")
        guard = "IF EXISTS " if if_exists else ""
        listed = ", ".join(sq.literal(k) for k in keys)
        return self._alter(
            Operation.UNSET_PROPERTIES, table, f"UNSET TBLPROPERTIES {guard}({listed})"
        )

    def add_feature(self, table: ResolvedTable, feature: Any, **kwargs: Any) -> dict[str, Any]:
        """Enable one feature, or several, via ``delta.feature.<name> = 'supported'``."""
        kwargs.pop("allow_protocol_versions_increase", None)
        _reject_unsupported("add a table feature", kwargs)
        # Any collection of features (a set too), but a str is one feature.
        if isinstance(feature, str) or not isinstance(feature, list | tuple | set | frozenset):
            features = [feature]
        else:
            features = list(feature)
        return self.set_properties(
            table, {f"delta.feature.{sq.feature_name(f)}": "supported" for f in features}
        )

    def drop_feature(
        self, table: ResolvedTable, feature: Any, *, truncate_history: bool = False
    ) -> dict[str, Any]:
        clause = f"DROP FEATURE {sq.quote(sq.feature_name(feature))}"
        if truncate_history:
            clause += " TRUNCATE HISTORY"
        return self._alter(Operation.DROP_FEATURE, table, clause)

    def add_constraint(
        self, table: ResolvedTable, constraints: dict[str, str], **kwargs: Any
    ) -> dict[str, Any]:
        """ADD CONSTRAINT ... CHECK. The expression is SQL by contract."""
        _reject_unsupported("add a constraint", kwargs)
        if not constraints:
            raise InvalidArgumentError("no constraints given")
        for name, expression in constraints.items():
            # CHECK () or CHECK (None) failed on the warehouse with a parse error.
            if sq.predicate(expression) is None or not str(name).strip():
                raise InvalidArgumentError(
                    f"constraint {name!r} needs a name and a CHECK expression"
                )
        for name, expression in constraints.items():
            self._alter(
                Operation.ADD_CONSTRAINT,
                table,
                f"ADD CONSTRAINT {sq.quote(name)} CHECK ({expression})",
            )
        return _ok()

    def drop_constraint(
        self, table: ResolvedTable, name: str, *, if_exists: bool = False
    ) -> dict[str, Any]:
        guard = "IF EXISTS " if if_exists else ""
        return self._alter(
            Operation.DROP_CONSTRAINT, table, f"DROP CONSTRAINT {guard}{sq.quote(name)}"
        )

    def set_comment(self, table: ResolvedTable, comment: str | None) -> dict[str, Any]:
        self._run(
            Operation.SET_COMMENT,
            f"COMMENT ON TABLE {sq.name(table)} IS {sq.literal(comment)}",
        )
        return _ok()

    def set_column_comment(
        self, table: ResolvedTable, column: str | Sequence[str], comment: str | None
    ) -> dict[str, Any]:
        return self._alter(
            Operation.SET_COLUMN_COMMENT,
            table,
            f"ALTER COLUMN {sq.column(column)} COMMENT {sq.literal(comment or '')}",
        )

    def alter_column_type(
        self, table: ResolvedTable, column: str | Sequence[str], new_type: str
    ) -> dict[str, Any]:
        return self._alter(
            Operation.ALTER_COLUMN_TYPE,
            table,
            f"ALTER COLUMN {sq.column(column)} TYPE {sq.sql_type(sq.type_text(new_type))}",
        )

    def set_not_null(self, table: ResolvedTable, column: str | Sequence[str]) -> dict[str, Any]:
        return self._alter(
            Operation.SET_NOT_NULL, table, f"ALTER COLUMN {sq.column(column)} SET NOT NULL"
        )

    def drop_not_null(self, table: ResolvedTable, column: str | Sequence[str]) -> dict[str, Any]:
        return self._alter(
            Operation.DROP_NOT_NULL, table, f"ALTER COLUMN {sq.column(column)} DROP NOT NULL"
        )

    def cluster_by(
        self, table: ResolvedTable, columns: Sequence[str] | str | None
    ) -> dict[str, Any]:
        """CLUSTER BY (cols); an empty list or None is NONE; ``"auto"`` is AUTO."""
        if isinstance(columns, str):
            if columns.lower() != "auto":
                raise InvalidArgumentError("columns must be a list of names, None, or 'auto'")
            clause = "CLUSTER BY AUTO"
        elif not columns:
            clause = "CLUSTER BY NONE"
        else:
            clause = f"CLUSTER BY ({sq.columns(columns)})"
        return self._alter(Operation.CLUSTER_BY, table, clause)

    # ---------------------------------------------------- governance extras

    def undrop(self, full_name: str) -> dict[str, Any]:
        """UNDROP TABLE, within the catalog's retention window."""
        self.execute("undrop", f"UNDROP TABLE {sq.qualified(full_name)}", fetch=False)
        return _ok()

    def set_row_filter(
        self, table: ResolvedTable, function_name: str, columns: Sequence[str]
    ) -> dict[str, Any]:
        clause = f"SET ROW FILTER {sq.qualified(function_name)} ON ({sq.columns(columns)})"
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
        clause = f"ALTER COLUMN {sq.column(column)} SET MASK {sq.qualified(function_name)}"
        if using_columns:
            clause += f" USING COLUMNS ({sq.columns(using_columns)})"
        return self._alter("set_column_mask", table, clause)

    def drop_column_mask(self, table: ResolvedTable, column: str | Sequence[str]) -> dict[str, Any]:
        return self._alter("drop_column_mask", table, f"ALTER COLUMN {sq.column(column)} DROP MASK")

    def set_tags(
        self,
        table: ResolvedTable,
        tags: Mapping[str, str],
        column: str | Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if not tags:
            raise InvalidArgumentError("no tags given")
        # A key-only tag has an empty value; str(None) stored the text 'None'.
        pairs = ", ".join(
            f"{sq.literal(k)} = {sq.literal('' if v is None else str(v))}" for k, v in tags.items()
        )
        target = f"ALTER COLUMN {sq.column(column)} " if column is not None else ""
        return self._alter("set_tags", table, f"{target}SET TAGS ({pairs})")

    def unset_tags(
        self,
        table: ResolvedTable,
        keys: Sequence[str],
        column: str | Sequence[str] | None = None,
    ) -> dict[str, Any]:
        keys = [keys] if isinstance(keys, str) else list(keys)
        if not keys:
            raise InvalidArgumentError("no tag keys given")
        target = f"ALTER COLUMN {sq.column(column)} " if column is not None else ""
        listed = ", ".join(sq.literal(k) for k in keys)
        return self._alter("unset_tags", table, f"{target}UNSET TAGS ({listed})")

    def set_owner(self, table: ResolvedTable, principal: str) -> dict[str, Any]:
        return self._alter("set_owner", table, f"OWNER TO {sq.principal(principal)}")

    def grant(
        self, table: ResolvedTable, privileges: str | Sequence[str], principal: str
    ) -> dict[str, Any]:
        target = f"TABLE {sq.name(table)} TO {sq.principal(principal)}"
        sql = f"GRANT {sq.privileges(privileges)} ON {target}"
        self.execute("grant", sql, fetch=False)
        return _ok()

    def revoke(
        self, table: ResolvedTable, privileges: str | Sequence[str], principal: str
    ) -> dict[str, Any]:
        source = f"TABLE {sq.name(table)} FROM {sq.principal(principal)}"
        sql = f"REVOKE {sq.privileges(privileges)} ON {source}"
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
        sql = ("CREATE OR REPLACE TABLE " if replace else "CREATE TABLE ") + sq.qualified(name)
        if cluster_by:
            sql += f" CLUSTER BY ({sq.columns(cluster_by)})"
        if comment is not None:
            sql += f" COMMENT {sq.literal(comment)}"
        if properties:
            sql += f" TBLPROPERTIES ({sq.properties(properties)})"
        sql += f" AS {select_sql}"
        self.execute("create_table_as", sql, fetch=False, parameters=parameters)
        return _ok()

    def plan_scan(self, table: ResolvedTable, **kwargs: Any) -> list[Any]:
        raise NotImplementedError("the SQL engine has no split-planning surface")

    def execute_scan(self, table: ResolvedTable, splits: list[Any], **kwargs: Any) -> Any:
        raise NotImplementedError("see plan_scan")


# ------------------------------------------------------------------ MERGE


def _name_list(names: str | Sequence[str] | None) -> list[str]:
    """A column list where a bare str is one name: list("ts") was ['t', 's']."""
    if names is None:
        return []
    if isinstance(names, str):
        return [names]
    return list(names)


_CLAUSE_ORDER = {"MATCHED": 0, "NOT MATCHED": 1, "NOT MATCHED BY SOURCE": 2}


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
        # Checked now, not after the source has been uploaded at execute().
        sq.predicate(predicate)
        verb, arg = action
        if verb in ("UPDATE", "INSERT") and not arg:
            raise InvalidArgumentError(f"a MERGE {verb} clause needs at least one column to set")
        self._clauses.append((kind, predicate, action))
        return self

    def when_matched_update(
        self, updates: dict[str, str], predicate: str | None = None
    ) -> SqlMerger:
        return self._add("MATCHED", predicate, ("UPDATE", dict(updates)))

    def when_matched_update_all(
        self, predicate: str | None = None, except_cols: list[str] | None = None
    ) -> SqlMerger:
        return self._add("MATCHED", predicate, ("UPDATE_ALL", _name_list(except_cols)))

    def when_matched_delete(self, predicate: str | None = None) -> SqlMerger:
        return self._add("MATCHED", predicate, ("DELETE", None))

    def when_not_matched_insert(
        self, updates: dict[str, str], predicate: str | None = None
    ) -> SqlMerger:
        return self._add("NOT MATCHED", predicate, ("INSERT", dict(updates)))

    def when_not_matched_insert_all(
        self, predicate: str | None = None, except_cols: list[str] | None = None
    ) -> SqlMerger:
        return self._add("NOT MATCHED", predicate, ("INSERT_ALL", _name_list(except_cols)))

    def when_not_matched_by_source_update(
        self, updates: dict[str, str], predicate: str | None = None
    ) -> SqlMerger:
        return self._add("NOT MATCHED BY SOURCE", predicate, ("UPDATE", dict(updates)))

    def when_not_matched_by_source_delete(self, predicate: str | None = None) -> SqlMerger:
        return self._add("NOT MATCHED BY SOURCE", predicate, ("DELETE", None))

    def _target_column(self, key: Any) -> str:
        """A clause key, without the target alias delta-rs lets it carry.

        delta-rs resolves ``{"target.v": ...}`` as column `v` of the target;
        quoted whole it became the unknown column `target`.`target.v`.
        """
        text = str(key)
        prefix = self._target_alias + "."
        if text.lower().startswith(prefix.lower()) and len(text) > len(prefix):
            return text[len(prefix) :]
        return text

    def _action(self, action: Any, source_columns: list[str]) -> str:
        verb, arg = action
        tgt, src = sq.quote(self._target_alias), sq.quote(self._source_alias)
        if verb == "DELETE":
            return "DELETE"
        if verb in ("UPDATE", "INSERT") and not arg:
            raise InvalidArgumentError(f"a MERGE {verb} clause needs at least one column to set")
        if verb == "UPDATE":
            return "UPDATE SET " + ", ".join(
                f"{tgt}.{sq.quote(self._target_column(k))} = {sq.expression(v)}"
                for k, v in arg.items()
            )
        if verb == "INSERT":
            names = ", ".join(sq.quote(self._target_column(k)) for k in arg)
            return f"INSERT ({names}) VALUES ({', '.join(sq.expression(v) for v in arg.values())})"
        # *_ALL: spell the columns out, so except_cols works on every runtime.
        # Names resolve case-insensitively, so except_cols=["ID"] excludes "id".
        excluded = {str(c).lower() for c in arg}
        cols = [c for c in source_columns if c.lower() not in excluded]
        if not cols:
            raise InvalidArgumentError("except_cols excludes every source column")
        if verb == "UPDATE_ALL":
            if not arg:
                return "UPDATE SET *"
            return "UPDATE SET " + ", ".join(
                f"{tgt}.{sq.quote(c)} = {src}.{sq.quote(c)}" for c in cols
            )
        if not arg:
            return "INSERT *"
        names = ", ".join(sq.quote(c) for c in cols)
        values = ", ".join(f"{src}.{sq.quote(c)}" for c in cols)
        return f"INSERT ({names}) VALUES ({values})"

    def statement(self, relation: str, source_columns: list[str]) -> str:
        """The MERGE text for a source `relation`. Exposed for inspection and tests."""
        if not self._clauses:
            raise InvalidArgumentError("a MERGE needs at least one WHEN clause")
        evolve = " WITH SCHEMA EVOLUTION" if self._merge_schema else ""
        sql = (
            f"MERGE{evolve} INTO {sq.name(self._table)} AS {sq.quote(self._target_alias)} "
            f"USING (SELECT {sq.columns(source_columns)} FROM {relation}) "
            f"AS {sq.quote(self._source_alias)} "
            f"ON {self._predicate}"
        )
        # Spark's grammar takes WHEN MATCHED, then WHEN NOT MATCHED, then WHEN
        # NOT MATCHED BY SOURCE clauses, in that order; delta-rs lets them be
        # added in any order. Clauses of different kinds never compete for a
        # row, so a stable sort by kind keeps the meaning and makes it parse.
        ordered = sorted(self._clauses, key=lambda c: _CLAUSE_ORDER[c[0]])
        for kind, predicate, action in ordered:
            condition = f" AND {predicate}" if sq.predicate(predicate) is not None else ""
            sql += f" WHEN {kind}{condition} THEN {self._action(action, source_columns)}"
        return sql

    def execute(self) -> dict[str, Any]:
        """Stage the source, run the MERGE, delete the staged file. Returns metrics."""
        if not self._clauses:
            raise InvalidArgumentError("a MERGE needs at least one WHEN clause")
        with self._engine._staged(self._source) as (relation, arrow):
            sql = self.statement(relation, list(arrow.column_names))
            result = self._engine._query(Operation.MERGE, sql)
        return _dml_metrics(_first(result), None)


# ------------------------------------------------------------------ helpers


def _filter_sql(spec: tuple[str, str, Any], binder: ParameterBinder) -> str:
    """One delta-rs partition filter, as a parameterized SQL condition."""
    column, op, value = spec
    op = op.strip().lower()
    if op in ("=", "!=", "<", "<=", ">", ">="):
        if value is None:
            if op not in ("=", "!="):
                raise InvalidArgumentError(f"cannot compare {column} {op} NULL")
            return f"{sq.quote(column)} IS {'NOT ' if op == '!=' else ''}NULL"
        return f"{sq.quote(column)} {op} {binder.bind(value)}"
    if op in ("in", "not in"):
        if isinstance(value, str | bytes) or not hasattr(value, "__iter__"):
            # list("abc") is ['a', 'b', 'c']: a lone string would silently
            # become an IN over its characters.
            raise InvalidArgumentError(
                f"an '{op}' filter on {column} needs a list of values, got {value!r}"
            )
        values = list(value)
        if not values:
            raise InvalidArgumentError(f"an '{op}' filter on {column} needs at least one value")
        markers = ", ".join(binder.bind(v) for v in values)
        return f"{sq.quote(column)} {op.upper()} ({markers})"
    raise InvalidArgumentError(f"unsupported partition filter operator {op!r}")


def _check_volume(volume: str | None) -> tuple[str, str, str] | None:
    if volume is None:
        return None
    parts = split_identifier(volume)
    if len(parts) != 3:
        raise InvalidArgumentError(
            f"staging_volume must be '<catalog>.<schema>.<volume>', got {volume!r}"
        )
    for part in parts:
        if not part.strip() or "/" in part or "\x00" in part or part in (".", ".."):
            raise InvalidArgumentError(
                f"staging_volume part {part!r} cannot appear in a volume path"
            )
    return parts[0], parts[1], parts[2]


def _warehouse_from_http_path(http_path: str | None) -> str | None:
    if not http_path:
        return None
    # A path copied from the UI can carry "?o=<workspace id>".
    http_path = http_path.split("?", 1)[0].split("#", 1)[0].strip()
    tail = http_path.rstrip("/").rsplit("/", 1)[-1]
    if ("/warehouses/" not in http_path and "/endpoints/" not in http_path) or tail in (
        "",
        "warehouses",
        "endpoints",
    ):
        raise InvalidArgumentError(
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
