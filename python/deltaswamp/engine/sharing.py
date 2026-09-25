"""The Delta Sharing engine: read-only access to tables a provider has shared.

A shared table has no storage location a recipient can open. The server answers
a query with the table's protocol, metadata and a list of data files as
short-lived presigned URLs, so this engine reads those files itself:

* **parquet responses** (the common case) are read file by file with pyarrow
  and conformed to the table's Delta schema -- partition values are added back
  as typed columns, and missing columns become nulls. Going through the
  client's pandas path instead would lose types (timestamps, decimals, nested
  types and integer nullability all degrade), so it is avoided.
* **delta responses** are required once a table uses deletion vectors or column
  mapping, since a plain file list cannot express either. The server then
  returns Delta log actions, which are replayed with the delta-kernel wrapper
  the client ships (`delta_kernel_rust_sharing_wrapper`) -- the same path as
  ``load_as_pandas(use_delta_format=True)``, but kept in Arrow.

Predicates are sent as ``jsonPredicateHints`` for file skipping only. The
protocol lets a server ignore hints, so the predicate is always applied exactly
to the rows afterwards.

History, writes and file listing are not part of the sharing protocol and are
refused with a reason.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import re
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .. import predicate as sqlpred
from .._util import http_error_text
from ..capability import READ_OPERATIONS, Capability, Operation
from ..capability import Engine as EngineKind
from ..catalog import ResolvedTable
from ..catalog.sharing import http_error_type, load_profile, sharing_module
from ..errors import UnreachableTableError
from .base import missing_method

__all__ = [
    "SharingEngine",
    "delta_schema_to_arrow",
    "filter_arrow_exact",
    "json_predicate_hints",
]

_SERVED: frozenset[Operation] = frozenset(
    {Operation.SCAN, Operation.TIME_TRAVEL, Operation.CDF, Operation.DETAIL}
)
_WRITES_REMEDY = "ask the provider to make the change on their side; a share is read-only"

#: Reader features a parquet file list cannot express; they force a delta response.
_DELTA_FORMAT_FEATURES = frozenset({"deletionVectors", "columnMapping"})

_HISTORY_REMEDY = (
    "the provider must share the table with its history (ALTER SHARE <share> ADD TABLE "
    "<table> WITH HISTORY), and change data feed needs delta.enableChangeDataFeed=true on "
    "the provider's table over the requested versions"
)

CHANGE_TYPE, COMMIT_VERSION, COMMIT_TIMESTAMP = (
    "_change_type",
    "_commit_version",
    "_commit_timestamp",
)


def _pa() -> Any:
    try:
        return importlib.import_module("pyarrow")
    except ImportError as exc:
        raise ImportError(
            "reading a shared table needs pyarrow. Install it with: "
            "pip install 'deltaswamp[sharing,pyarrow]'"
        ) from exc


# ---------------------------------------------------------------------------
# Exact predicate evaluation
# ---------------------------------------------------------------------------


def filter_arrow_exact(table: Any, predicate: str) -> Any:
    """Keep the rows of an Arrow table for which the SQL `predicate` is TRUE.

    Uses ``deltaswamp.predicate.filter_table``, the library's one SQL-predicate
    evaluator. For SQL it declines (``PredicateError``), DuckDB evaluates the
    predicate to a boolean column -- one value per input row, in input order --
    and pyarrow filters on that, so row order and the table's exact Arrow types
    are kept either way. NULL counts as false, as in a WHERE clause.
    """
    try:
        return sqlpred.filter_table(table, predicate)
    except sqlpred.PredicateError as declined:
        try:
            duckdb = importlib.import_module("duckdb")
        except ImportError:
            raise declined from None
    return _duckdb_filter(duckdb, table, predicate)


def _duckdb_filter(duckdb: Any, table: Any, predicate: str) -> Any:
    pa = _pa()
    if table.num_rows == 0:
        return table
    pc = importlib.import_module("pyarrow.compute")
    con = duckdb.connect()
    try:
        con.register("__deltaswamp_rows", table)
        result = con.execute(f"SELECT ({predicate}) AS keep FROM __deltaswamp_rows").arrow()
        if isinstance(result, pa.RecordBatchReader):
            result = result.read_all()
    finally:
        con.close()
    mask = pc.fill_null(result.column("keep").cast(pa.bool_()), False)
    return table.filter(mask)


# ---------------------------------------------------------------------------
# Delta schema -> Arrow
# ---------------------------------------------------------------------------

_PRIMITIVES: dict[str, str] = {
    "string": "string",
    "long": "int64",
    "integer": "int32",
    "short": "int16",
    "byte": "int8",
    "float": "float32",
    "double": "float64",
    "boolean": "bool_",
    "binary": "binary",
    "date": "date32",
}
_DECIMAL = re.compile(r"^decimal\((\d+),\s*(\d+)\)$")


def _delta_type_to_arrow(delta_type: Any) -> Any:
    pa = _pa()
    if isinstance(delta_type, str):
        if delta_type in _PRIMITIVES:
            return getattr(pa, _PRIMITIVES[delta_type])()
        if delta_type == "timestamp":
            return pa.timestamp("us", tz="UTC")
        if delta_type == "timestamp_ntz":
            return pa.timestamp("us")
        match = _DECIMAL.match(delta_type)
        if match:
            return pa.decimal128(int(match.group(1)), int(match.group(2)))
        raise UnreachableTableError(
            "read a shared table",
            f"its schema uses the Delta type {delta_type!r}, which has no Arrow mapping here",
        )
    kind = delta_type.get("type")
    if kind == "struct":
        return pa.struct([_delta_field_to_arrow(f) for f in delta_type["fields"]])
    if kind == "array":
        element = pa.field(
            "element",
            _delta_type_to_arrow(delta_type["elementType"]),
            nullable=bool(delta_type.get("containsNull", True)),
        )
        return pa.list_(element)
    if kind == "map":
        return pa.map_(
            _delta_type_to_arrow(delta_type["keyType"]),
            pa.field(
                "value",
                _delta_type_to_arrow(delta_type["valueType"]),
                nullable=bool(delta_type.get("valueContainsNull", True)),
            ),
        )
    raise UnreachableTableError(
        "read a shared table", f"its schema has an unrecognized type {delta_type!r}"
    )


def _delta_field_to_arrow(field: dict[str, Any]) -> Any:
    return _pa().field(
        field["name"],
        _delta_type_to_arrow(field["type"]),
        nullable=bool(field.get("nullable", True)),
    )


def delta_schema_to_arrow(schema: str | dict[str, Any]) -> Any:
    """Convert a Delta ``schemaString`` (or its parsed form) to an Arrow schema."""
    parsed = json.loads(schema) if isinstance(schema, str) else schema
    return _pa().schema([_delta_field_to_arrow(f) for f in parsed.get("fields", [])])


# ---------------------------------------------------------------------------
# jsonPredicateHints
# ---------------------------------------------------------------------------

_HINT_VALUE_TYPES: dict[str, str] = {
    "boolean": "bool",
    "byte": "int",
    "short": "int",
    "integer": "int",
    "long": "long",
    "float": "float",
    "double": "double",
    "string": "string",
    "date": "date",
}
_HINT_OPS: dict[str, tuple[str, bool]] = {
    "eq": ("equal", False),
    "ne": ("equal", True),
    "lt": ("lessThan", False),
    "le": ("lessThanOrEqual", False),
    "gt": ("greaterThan", False),
    "ge": ("greaterThanOrEqual", False),
}


def _conjuncts(node: sqlpred.Node) -> Iterator[sqlpred.Node]:
    if node.op == "and":
        for child in node.args:
            yield from _conjuncts(child)
    else:
        yield node


def _hint_literal(literal: sqlpred.Literal, value_type: str) -> str | None:
    """The literal as the protocol spells it, or None if the types disagree."""
    if literal.type == "string":
        return str(literal.value) if value_type in ("string", "date") else None
    if literal.type == "date":
        return str(literal.value.isoformat()) if value_type == "date" else None
    if literal.type == "boolean":
        return str(literal.value).lower() if value_type == "bool" else None
    if literal.type == "long":
        return str(literal.value) if value_type in ("int", "long", "float", "double") else None
    if literal.type in ("decimal", "double"):
        return str(literal.value) if value_type in ("float", "double") else None
    return None


def _hint_leaf(node: sqlpred.Node, types: dict[str, str]) -> dict[str, Any] | None:
    column = node.args[0] if node.args else None
    if not isinstance(column, sqlpred.Column) or len(column.path) != 1:
        return None
    name = column.path[0]
    value_type = types.get(name.lower())
    if value_type is None:
        return None
    ref = {"op": "column", "name": name, "valueType": value_type}

    if node.op in ("is_null", "is_not_null"):
        leaf: dict[str, Any] = {"op": "isNull", "children": [ref]}
        return {"op": "not", "children": [leaf]} if node.op == "is_not_null" else leaf

    if node.op not in _HINT_OPS or not isinstance(node.args[1], sqlpred.Literal):
        return None
    value = _hint_literal(node.args[1], value_type)
    if value is None:
        return None
    op, negate = _HINT_OPS[node.op]
    compare = {
        "op": op,
        "children": [ref, {"op": "literal", "value": value, "valueType": value_type}],
    }
    return {"op": "not", "children": [compare]} if negate else compare


def json_predicate_hints(predicate: str | None, schema: str | dict[str, Any]) -> str | None:
    """Translate the conjuncts of a SQL predicate the protocol can express.

    Only top-level ``AND``-ed comparisons of a top-level column against a
    literal of the column's own type, and ``IS [NOT] NULL``, are translated.
    Dropping a conjunct only weakens the hint, which is safe because the exact
    predicate is applied to the rows afterward.
    """
    if not predicate:
        return None
    try:
        node = sqlpred.parse(predicate)
    except sqlpred.PredicateError:
        return None
    parsed = json.loads(schema) if isinstance(schema, str) else schema
    types: dict[str, str] = {}
    for field in parsed.get("fields", []):
        kind = field.get("type")
        if isinstance(kind, str) and kind in _HINT_VALUE_TYPES:
            types[str(field["name"]).lower()] = _HINT_VALUE_TYPES[kind]
    leaves = [leaf for part in _conjuncts(node) if (leaf := _hint_leaf(part, types))]
    if not leaves:
        return None
    tree = leaves[0] if len(leaves) == 1 else {"op": "and", "children": leaves}
    return json.dumps(tree, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Reading presigned files
# ---------------------------------------------------------------------------


def _partition_column(value: str | None, arrow_type: Any, rows: int) -> Any:
    pa = _pa()
    if value is None:
        return pa.nulls(rows, arrow_type)
    strings = pa.array([value] * rows, pa.string())
    if pa.types.is_string(arrow_type):
        return strings
    if pa.types.is_binary(arrow_type):
        return pa.array([value.encode()] * rows, arrow_type)
    try:
        return strings.cast(arrow_type)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        if pa.types.is_timestamp(arrow_type):
            from datetime import UTC, datetime

            parsed = datetime.fromisoformat(value.replace(" ", "T"))
            if arrow_type.tz is not None and parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return pa.array([parsed] * rows, arrow_type)
        raise


def _conform(data: Any, schema: Any, partition_values: dict[str, str | None]) -> Any:
    """Shape one file's rows into `schema`: typed partition values, nulls for gaps."""
    pa = _pa()
    rows = data.num_rows
    by_lower = {name.lower(): name for name in data.column_names}
    lowered_partitions = {k.lower(): v for k, v in partition_values.items()}
    columns = []
    for field in schema:
        key = field.name.lower()
        if key in by_lower:
            column = data.column(by_lower[key])
            if column.type != field.type:
                try:
                    column = column.cast(field.type)
                except pa.ArrowInvalid:
                    # e.g. INT96 nanosecond timestamps narrowing to microseconds.
                    column = column.cast(field.type, safe=False)
            columns.append(column)
        elif key in lowered_partitions:
            columns.append(_partition_column(lowered_partitions[key], field.type, rows))
        else:
            columns.append(pa.nulls(rows, field.type))
    return pa.Table.from_arrays(columns, schema=schema)


def _project(schema: Any, columns: list[str] | None) -> Any:
    if columns is None:
        return schema
    pa = _pa()
    by_lower = {f.name.lower(): f for f in schema}
    missing = [c for c in columns if c.lower() not in by_lower]
    if missing:
        raise UnreachableTableError(
            "read the shared table",
            f"it has no column(s) {', '.join(missing)}; its columns are {', '.join(schema.names)}",
        )
    return pa.schema([by_lower[c.lower()] for c in columns])


class SharingEngine:
    """Reads tables reached through a Delta Sharing profile. Never writes."""

    kind = EngineKind.SHARING
    supports_distributed_scan = False
    #: Exact: hints go to the server, the predicate is re-applied to the rows.
    supports_predicates = True
    supports_timestamp_travel = True
    supports_schema_merge = False
    supports_schema_overwrite = False
    supports_idempotent_txn = False
    supports_commit_metadata = False
    supports_writer_properties = False
    supports_dynamic_overwrite = False

    def __init__(self, *, request_timeout: float = 300.0, num_retries: int = 10) -> None:
        """`request_timeout` bounds each presigned-file download, in seconds.
        `num_retries` is passed to the client, which retries 429s and 5xxs."""
        self._timeout = request_timeout
        self._num_retries = num_retries

    # ----------------------------------------------------------- capabilities

    @staticmethod
    def available() -> bool:
        try:
            importlib.import_module("delta_sharing")
        except ImportError:
            return False
        return True

    @staticmethod
    def _kernel_available() -> bool:
        try:
            importlib.import_module("delta_kernel_rust_sharing_wrapper")
        except ImportError:
            return False
        return True

    def supports(self, operation: Operation, table: ResolvedTable, **shape: Any) -> Capability:
        if not self.available():
            return Capability(
                operation,
                ok=False,
                reason="the delta-sharing package is not installed",
                remedy="pip install 'deltaswamp[sharing]'",
            )

        gap = missing_method(self, operation)
        if gap is not None:
            if operation is Operation.HISTORY:
                return Capability(
                    operation,
                    ok=False,
                    reason="the Delta Sharing protocol has no history endpoint",
                    remedy="read changes version by version with cdf(), or ask the provider "
                    "for DESCRIBE HISTORY",
                )
            if operation is Operation.FILES:
                return Capability(
                    operation,
                    ok=False,
                    reason="a share exposes data files only as short-lived presigned URLs, "
                    "not as table paths",
                )
            if operation in READ_OPERATIONS:
                return gap
            return Capability(
                operation,
                ok=False,
                reason=f"shares are read-only: a Delta Sharing recipient cannot "
                f"{operation.value.replace('_', ' ')}",
                remedy=_WRITES_REMEDY,
            )

        if not table.is_shared:
            return Capability(
                operation,
                ok=False,
                reason="the table is not reached through Delta Sharing, and this engine "
                "reads only shared tables",
                remedy="open shared tables with ds.connect('sharing:///path/to/config.share')",
            )

        if operation not in _SERVED:
            return Capability(
                operation,
                ok=False,
                reason=f"Delta Sharing has no {operation.value} operation",
            )

        if self._needs_delta_format(table) and not self._kernel_available():
            return Capability(
                operation,
                ok=False,
                reason="the table uses "
                + ", ".join(sorted(_DELTA_FORMAT_FEATURES & table.reader_features))
                + ", which Delta Sharing serves only as Delta log actions, and the kernel "
                "wrapper that replays them is not installed",
                remedy="pip install delta-kernel-rust-sharing-wrapper",
            )

        return Capability(operation, ok=True, engine=self.kind)

    @staticmethod
    def _needs_delta_format(table: ResolvedTable) -> bool:
        return bool(_DELTA_FORMAT_FEATURES & table.reader_features)

    # --------------------------------------------------------------- plumbing

    def _client(self, table: ResolvedTable) -> Any:
        if table.sharing_profile is None:
            raise UnreachableTableError(
                "read the table through Delta Sharing", "it carries no sharing profile"
            )
        rest = sharing_module("delta_sharing.rest_client")
        return rest.DataSharingRestClient(
            load_profile(table.sharing_profile), num_retries=self._num_retries
        )

    @staticmethod
    def _shared(table: ResolvedTable) -> Any:
        ref = table.ref
        if not (ref.catalog and ref.schema and ref.table):
            raise UnreachableTableError(
                "read the table through Delta Sharing",
                f"{ref} is not a share.schema.table name",
            )
        return sharing_module("delta_sharing.protocol").Table(
            name=ref.table, share=ref.catalog, schema=ref.schema
        )

    @staticmethod
    def _refusal(what: str, exc: BaseException, *, history: bool = False) -> UnreachableTableError:
        remedy = _HISTORY_REMEDY if history else None
        return UnreachableTableError(
            what, f"the sharing server refused the request ({http_error_text(exc)})", remedy
        )

    def _download(self, url: str) -> bytes:
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as response:
                body: bytes = response.read()
                return body
        except urllib.error.HTTPError as exc:
            raise UnreachableTableError(
                "read a shared data file",
                f"the presigned URL was refused (HTTP {exc.code}). Presigned URLs expire, "
                "typically within an hour of the query that issued them",
                "re-run the read; a new query issues fresh URLs",
            ) from exc

    def _read_file(self, url: str, partition_values: dict[str, str | None], schema: Any) -> Any:
        parquet = importlib.import_module("pyarrow.parquet")
        pa = _pa()
        source = pa.BufferReader(self._download(url))
        file_names = parquet.ParquetFile(source).schema_arrow.names
        wanted = {f.name.lower() for f in schema}
        present = [name for name in file_names if name.lower() in wanted]
        source.seek(0)
        data = parquet.read_table(source, columns=present)
        return _conform(data, schema, partition_values)

    def _hints(self, client: Any, shared: Any, predicate: str | None) -> str | None:
        """Hints need the column types, so this costs one metadata request."""
        if not predicate:
            return None
        try:
            metadata = client.query_table_metadata(shared).metadata
        except Exception:
            return None  # a hint is an optimization; never fail the read over one
        return json_predicate_hints(predicate, metadata.schema_string)

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
        """Read a shared table as a `pyarrow.RecordBatchReader`.

        Parquet responses stream: files are downloaded one at a time as the
        reader is consumed. Delta responses are replayed by the kernel into
        memory first, because the replay needs the log on local disk.
        """
        if self._needs_delta_format(table):
            return self._scan_delta_format(
                table, columns=columns, predicate=predicate, version=version, timestamp=timestamp
            )

        pa = _pa()
        shared = self._shared(table)
        client = self._client(table)
        travelling = version is not None or timestamp is not None
        try:
            hints = self._hints(client, shared, predicate)
            response = client.list_files_in_table(
                shared, jsonPredicateHints=hints, version=version, timestamp=timestamp
            )
        except http_error_type() as exc:
            raise self._refusal("read the shared table", exc, history=travelling) from exc
        finally:
            client.close()

        full = delta_schema_to_arrow(response.metadata.schema_string)
        out = _project(full, columns)
        # The predicate may reference columns the caller did not ask for.
        read = full if predicate else out
        files = list(response.add_files)

        def batches() -> Iterator[Any]:
            for f in files:
                rows = self._read_file(f.url, dict(f.partition_values), read)
                if predicate:
                    rows = filter_arrow_exact(rows, predicate).select(out.names)
                yield from rows.to_batches()

        return pa.RecordBatchReader.from_batches(out, batches())

    def _replay_delta_log(self, lines: list[str], *, what: str) -> Any:
        """Replay a delta-format query response with the kernel, into Arrow."""
        pa = _pa()
        try:
            kernel = importlib.import_module("delta_kernel_rust_sharing_wrapper")
        except ImportError as exc:
            raise UnreachableTableError(
                what,
                "the server answered with Delta log actions, and the kernel wrapper that "
                "replays them is not installed",
                "pip install delta-kernel-rust-sharing-wrapper",
            ) from exc
        protocol = json.loads(lines[0])["protocol"]["deltaProtocol"]
        metadata = json.loads(lines[1])["metaData"]["deltaMetadata"]
        with tempfile.TemporaryDirectory(prefix="deltaswamp-sharing-") as root:
            log = Path(root) / "_delta_log"
            log.mkdir()
            with (log / f"{0:020d}.json").open("w") as out:
                out.write(json.dumps({"protocol": protocol}) + "\n")
                out.write(json.dumps({"metaData": metadata}) + "\n")
                for line in lines[2:]:
                    action = json.loads(line)["file"]["deltaSingleAction"]
                    out.write(json.dumps(action) + "\n")
            uri = Path(root).as_uri()
            interface = kernel.PythonInterface(uri)
            snapshot = kernel.Table(uri).snapshot(interface)
            result = kernel.ScanBuilder(snapshot).build().execute(interface)
            if isinstance(result, pa.RecordBatchReader):
                return result.read_all()
            batches = list(result)
            if batches:
                return pa.Table.from_batches(batches)
            return delta_schema_to_arrow(metadata["schemaString"]).empty_table()

    def _scan_delta_format(
        self,
        table: ResolvedTable,
        *,
        columns: list[str] | None,
        predicate: str | None,
        version: int | None,
        timestamp: str | None,
    ) -> Any:
        pa = _pa()
        shared = self._shared(table)
        client = self._client(table)
        travelling = version is not None or timestamp is not None
        try:
            client.set_delta_format_header()
            hints = self._hints(client, shared, predicate)
            response = client.list_files_in_table(
                shared, jsonPredicateHints=hints, version=version, timestamp=timestamp
            )
        except http_error_type() as exc:
            raise self._refusal("read the shared table", exc, history=travelling) from exc
        finally:
            client.close()

        data = self._replay_delta_log(list(response.lines), what="read the shared table")
        if predicate:
            data = filter_arrow_exact(data, predicate)
        if columns is not None:
            data = data.select(_project(data.schema, columns).names)
        return pa.RecordBatchReader.from_batches(data.schema, data.to_batches())

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
    ) -> Any:
        """The table's change data feed as a `pyarrow.RecordBatchReader`.

        Adds ``_change_type`` (insert, delete, update_preimage,
        update_postimage), ``_commit_version`` and ``_commit_timestamp``, as
        Delta's own CDF does. Ranges are inclusive. With neither start given,
        reads from version 0.
        """
        pa = _pa()
        if starting_version is None and starting_timestamp is None:
            starting_version = 0
        protocol = sharing_module("delta_sharing.protocol")
        options = protocol.CdfOptions(
            starting_version=starting_version,
            ending_version=ending_version,
            starting_timestamp=starting_timestamp,
            ending_timestamp=ending_timestamp,
        )
        shared = self._shared(table)
        what = "read the change data feed of the shared table"

        if self._needs_delta_format(table):
            data = self._cdf_delta_format(table, shared, options, what)
        else:
            client = self._client(table)
            try:
                response = client.list_table_changes(shared, options)
            except http_error_type() as exc:
                raise self._refusal(what, exc, history=True) from exc
            finally:
                client.close()
            data = self._cdf_parquet(response)

        if predicate:
            data = filter_arrow_exact(data, predicate)
        if columns is not None:
            keep = [*columns, *(c for c in (CHANGE_TYPE, COMMIT_VERSION, COMMIT_TIMESTAMP))]
            keep = list(dict.fromkeys(keep))
            data = data.select(_project(data.schema, keep).names)
        return pa.RecordBatchReader.from_batches(data.schema, data.to_batches())

    def _cdf_parquet(self, response: Any) -> Any:
        pa = _pa()
        protocol = sharing_module("delta_sharing.protocol")
        data_schema = delta_schema_to_arrow(response.metadata.schema_string)
        change_field = pa.field(CHANGE_TYPE, pa.string())
        out = pa.schema(
            [
                *data_schema,
                change_field,
                pa.field(COMMIT_VERSION, pa.int64()),
                pa.field(COMMIT_TIMESTAMP, pa.timestamp("ms", tz="UTC")),
            ]
        )
        pieces = []
        for action in response.actions:
            if action is None:
                continue
            partitions = dict(action.partition_values)
            if isinstance(action, protocol.AddCdcFile):
                rows = self._read_file(
                    action.url, partitions, pa.schema([*data_schema, change_field])
                )
            else:
                rows = self._read_file(action.url, partitions, data_schema)
                change = action.get_change_type_col_value()
                rows = rows.append_column(change_field, pa.array([change] * rows.num_rows))
            n = rows.num_rows
            rows = rows.append_column(
                out.field(COMMIT_VERSION),
                pa.array([action.version] * n, pa.int64()),
            ).append_column(
                out.field(COMMIT_TIMESTAMP),
                pa.array([action.timestamp] * n, pa.int64()).cast(pa.timestamp("ms", tz="UTC")),
            )
            pieces.append(rows)
        if not pieces:
            return out.empty_table()
        return pa.concat_tables(pieces)

    def _cdf_delta_format(self, table: ResolvedTable, shared: Any, options: Any, what: str) -> Any:
        """Column-mapped or DV tables: the client's kernel path, then back to Arrow.

        This one goes through pandas because the client's CDF replay (which
        also has to fabricate a checkpoint to start mid-log) returns nothing
        else; the result is cast back to the Delta schema where it can be.
        """
        pa = _pa()
        reader_module = sharing_module("delta_sharing.reader")
        protocol = sharing_module("delta_sharing.protocol")
        client = self._client(table)
        try:
            reader = reader_module.DeltaSharingReader(
                table=shared, rest_client=client, use_delta_format=True
            )
            # The client prints replay statistics to stdout; keep them out of ours.
            with contextlib.redirect_stdout(io.StringIO()):
                frame = reader.table_changes_to_pandas(
                    protocol.CdfOptions(
                        starting_version=options.starting_version,
                        ending_version=options.ending_version,
                        starting_timestamp=options.starting_timestamp,
                        ending_timestamp=options.ending_timestamp,
                        include_historical_metadata=True,
                    )
                )
        except http_error_type() as exc:
            raise self._refusal(what, exc, history=True) from exc
        finally:
            client.close()
        return pa.Table.from_pandas(frame, preserve_index=False)

    # --------------------------------------------------------------- metadata

    def detail(self, table: ResolvedTable, *, version: int | None = None) -> dict[str, Any]:
        """Version, protocol and metadata as the sharing server reports them."""
        protocol_module = sharing_module("delta_sharing.protocol")
        shared = self._shared(table)
        client = self._client(table)
        try:
            if version is None:
                client.set_sharing_capabilities_header()
                response = client.query_table_metadata(shared)
                at, protocol, metadata = (
                    response.delta_table_version,
                    response.protocol,
                    response.metadata,
                )
            else:
                if self._needs_delta_format(table):
                    client.set_delta_format_header()
                response = client.list_files_in_table(shared, version=version, limitHint=1)
                at = response.delta_table_version
                if response.protocol is not None:
                    protocol, metadata = response.protocol, response.metadata
                else:
                    lines = list(response.lines)
                    protocol = protocol_module.Protocol.from_json(json.loads(lines[0])["protocol"])
                    metadata = protocol_module.Metadata.from_json(json.loads(lines[1])["metaData"])
        except http_error_type() as exc:
            raise self._refusal(
                "describe the shared table", exc, history=version is not None
            ) from exc
        finally:
            client.close()

        ref = table.ref
        return {
            "version": at,
            "location": None,
            "share": ref.catalog,
            "schema": ref.schema,
            "table": ref.table,
            "format": "delta",
            "min_reader_version": protocol.min_reader_version,
            "min_writer_version": protocol.min_writer_version,
            "reader_features": list(protocol.reader_features or []),
            "writer_features": list(protocol.writer_features or []),
            "properties": dict(metadata.configuration or {}),
            "partition_columns": list(metadata.partition_columns or []),
            "metadata_id": metadata.id,
            "name": metadata.name,
            "description": metadata.description,
            "schema_string": metadata.schema_string,
            "size_in_bytes": metadata.size,
            "num_files": metadata.num_files,
        }

    # ------------------------------------------------------ distributed path

    def plan_scan(self, table: ResolvedTable, **kwargs: Any) -> list[Any]:
        raise NotImplementedError(
            "the sharing engine has no split-planning surface; presigned URLs would also "
            "expire in transit to workers"
        )

    def execute_scan(self, table: ResolvedTable, splits: list[Any], **kwargs: Any) -> Any:
        raise NotImplementedError("see plan_scan")
