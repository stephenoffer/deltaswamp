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

import importlib
import itertools
import json
import operator
import os
import re
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .. import predicate as sqlpred
from ..capability import READ_OPERATIONS, Capability, Operation
from ..capability import Engine as EngineKind
from ..catalog import ResolvedTable
from ..catalog.sharing import (
    error_text,
    http_error_type,
    load_profile,
    quote_name,
    request_error_types,
    sharing_module,
)
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


def filter_arrow_exact(table: Any, predicate: Any) -> Any:
    """Keep the rows of an Arrow table for which the SQL `predicate` is TRUE.

    `predicate` is a SQL string, an already parsed ``deltaswamp.predicate.Node``,
    or what `_parse_predicate` returned. It is evaluated by
    ``deltaswamp.predicate.filter_table``, the library's own parser and Arrow
    evaluator, so row order and the table's exact Arrow types are kept and NULL
    counts as false, as in a WHERE clause.

    SQL the parser does not represent (functions such as ``lower(s)``,
    arithmetic) is evaluated by DuckDB when it is installed, but only as a
    single *expression* in a sandboxed connection: no file, network or
    environment access, no extension loading, configuration locked, and text
    that could leave the expression (``;``, comments, a subquery) is refused
    before DuckDB sees it. The old fallback pasted the text into
    ``SELECT (<predicate>) FROM t``, which let a predicate close the
    parenthesis and run arbitrary DuckDB SQL.
    """
    if isinstance(predicate, _SandboxedSql):
        return _sandboxed_filter(table, predicate.text)
    try:
        return sqlpred.filter_table(table, predicate)
    except sqlpred.PredicateError:
        if not isinstance(predicate, str) or _duckdb() is None:
            raise
        _screen_expression(predicate)
    return _sandboxed_filter(table, predicate)


class _SandboxedSql:
    """A predicate the parser declined, screened for the sandboxed evaluator."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


def _duckdb() -> Any:
    try:
        return importlib.import_module("duckdb")
    except ImportError:
        return None


_SQL_LITERAL = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_OUTSIDE_EXPRESSION = re.compile(
    r";|--|/\*|\*/|\b(?:select|from|with|pragma|set|attach|copy)\b", re.I
)


def _screen_expression(text: str) -> None:
    """Refuse text that could be more than one boolean expression over the row."""
    predicate_module = importlib.import_module("deltaswamp.predicate")
    bare = _SQL_LITERAL.sub(" 0 ", text)
    if "'" in bare or '"' in bare:
        raise predicate_module.PredicateError(f"unbalanced quotes in predicate {text!r}")
    found = _OUTSIDE_EXPRESSION.search(bare)
    if found is not None:
        raise predicate_module.PredicateError(
            f"the predicate {text!r} contains {found.group(0)!r}; a filter is one expression "
            "over the row (no statements, comments or subqueries)"
        )


def _sandboxed_filter(table: Any, text: str) -> Any:
    pa = _pa()
    pc = importlib.import_module("pyarrow.compute")
    predicate_module = importlib.import_module("deltaswamp.predicate")
    duckdb = _duckdb()
    if duckdb is None:
        raise predicate_module.PredicateError(
            f"the predicate {text!r} needs DuckDB to evaluate; pip install duckdb"
        )
    if table.num_rows == 0:
        return table
    con = duckdb.connect(
        ":memory:",
        config={
            "enable_external_access": False,
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
            "lock_configuration": True,
        },
    )
    try:
        # The relational API parses an expression list, never a statement.
        result = con.from_arrow(table).project(f"CAST(({text}) AS BOOLEAN) AS __deltaswamp_keep")
        result = result.arrow()
        if isinstance(result, pa.RecordBatchReader):
            result = result.read_all()
    except duckdb.Error as exc:
        raise predicate_module.PredicateError(f"cannot evaluate {text!r}: {exc}") from None
    finally:
        con.close()
    if result.column_names != ["__deltaswamp_keep"] or result.num_rows != table.num_rows:
        raise predicate_module.PredicateError(f"{text!r} is not one boolean expression per row")
    mask = pc.fill_null(result.column(0).cast(pa.bool_()), False)
    return table.filter(mask)


def _parse_predicate(predicate: str | None) -> Any:
    """Parse once, before any request, so bad SQL fails fast and as itself.

    Raised later, inside a streaming read, the error would reach the caller
    only as pyarrow's ArrowInvalid. SQL the parser declines is screened for the
    sandboxed DuckDB evaluator instead (column binding is checked on the first
    file, which is read before the stream is returned).
    """
    if not predicate:
        return None
    predicate_module = importlib.import_module("deltaswamp.predicate")
    try:
        return predicate_module.parse(predicate)
    except predicate_module.PredicateError:
        if _duckdb() is None:
            raise
        _screen_expression(predicate)
        return _SandboxedSql(predicate)


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
    """The top-level AND-ed terms. The parser applies SQL precedence, so in
    ``a = 1 OR b = 2 AND c = 3`` the whole OR is one term and is not hinted:
    sending ``c = 3`` alone would let the server skip matching files."""
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


_INT_RANGES: dict[str, tuple[int, int]] = {
    "byte": (-(2**7), 2**7 - 1),
    "short": (-(2**15), 2**15 - 1),
    "integer": (-(2**31), 2**31 - 1),
    "long": (-(2**63), 2**63 - 1),
}


def _hint_leaf(node: sqlpred.Node, types: dict[str, tuple[str, str, str]]) -> dict[str, Any] | None:
    """One conjunct as a hint node, or None. `types` maps a lower-cased column
    name to (schema name, hint value type, Delta type)."""
    column = node.args[0] if node.args else None
    if not isinstance(column, sqlpred.Column) or len(column.path) != 1:
        return None
    known = types.get(column.path[0].lower())
    if known is None:
        return None
    # The schema's spelling: the server matches column names exactly.
    name, value_type, delta_type = known
    ref = {"op": "column", "name": name, "valueType": value_type}

    if node.op in ("is_null", "is_not_null"):
        leaf: dict[str, Any] = {"op": "isNull", "children": [ref]}
        return {"op": "not", "children": [leaf]} if node.op == "is_not_null" else leaf

    if node.op not in _HINT_OPS or not isinstance(node.args[1], sqlpred.Literal):
        return None
    literal = node.args[1]
    value = _hint_literal(literal, value_type)
    if value is None:
        return None
    if value_type == "date" and literal.type == "string":
        from datetime import date

        try:
            date.fromisoformat(value)
        except ValueError:
            return None  # a date the server cannot parse; the hint is optional
    if delta_type in _INT_RANGES and literal.type == "long":
        low, high = _INT_RANGES[delta_type]
        if not low <= int(literal.value) <= high:
            # Out of the column's range: the server cannot represent the
            # literal in the column's type.
            return None
    op, negate = _HINT_OPS[node.op]
    if value_type in ("float", "double") and (negate or op.startswith("greaterThan")):
        # Statistics leave NaN out of min/max, and NaN (above every value)
        # satisfies `f > x` and `f != x`: a server skipping on such a hint
        # drops files whose NaN rows the exact filter keeps.
        return None
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
    types: dict[str, tuple[str, str, str]] = {}
    for field in parsed.get("fields", []):
        kind = field.get("type")
        if isinstance(kind, str) and kind in _HINT_VALUE_TYPES:
            name = str(field["name"])
            types[name.lower()] = (name, _HINT_VALUE_TYPES[kind], kind)
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
    if value is None or (value == "" and not pa.types.is_string(arrow_type)):
        # Delta writes a null partition as a missing or empty value; an empty
        # string cannot be cast to a number, date or timestamp anyway.
        return pa.nulls(rows, arrow_type)
    strings = pa.array([value] * rows, pa.string())
    if pa.types.is_string(arrow_type):
        return strings
    if pa.types.is_binary(arrow_type):
        return pa.array([value.encode()] * rows, arrow_type)
    try:
        return strings.cast(arrow_type)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        if pa.types.is_timestamp(arrow_type):
            from datetime import UTC, datetime

            try:
                parsed = datetime.fromisoformat(value.strip().replace(" ", "T"))
            except ValueError:
                pass
            else:
                if arrow_type.tz is not None and parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                elif arrow_type.tz is None and parsed.tzinfo is not None:
                    parsed = parsed.astimezone(UTC).replace(tzinfo=None)
                return pa.array([parsed] * rows, arrow_type)
        raise UnreachableTableError(
            "read a shared data file",
            f"its partition value {value!r} is not a valid {arrow_type} ({exc})",
        ) from exc


def _has_timestamp(arrow_type: Any) -> bool:
    pa = _pa()
    if pa.types.is_timestamp(arrow_type):
        return True
    if pa.types.is_struct(arrow_type):
        return any(_has_timestamp(arrow_type.field(i).type) for i in range(arrow_type.num_fields))
    if pa.types.is_map(arrow_type):
        return _has_timestamp(arrow_type.key_type) or _has_timestamp(arrow_type.item_type)
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return _has_timestamp(arrow_type.value_type)
    return False


def _cast_column(column: Any, target: Any, name: str) -> Any:
    """Cast one file column to the table's type, refusing lossy casts.

    A lossless cast is always tried first. Only timestamps fall back to an
    unchecked cast (Spark's INT96 nanoseconds narrowing to Delta's
    microseconds); anywhere else an unchecked cast would silently wrap
    overflowing integers or drop decimal digits.
    """
    pa = _pa()
    try:
        return column.cast(target)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        if _has_timestamp(column.type) and _has_timestamp(target):
            # Also covers INT96 timestamps nested in a struct, list or map.
            return column.cast(target, safe=False)
        raise UnreachableTableError(
            "read a shared data file",
            f"its column {name!r} holds {column.type}, which does not convert to the "
            f"table's type {target} without loss ({exc})",
        ) from exc


def _conform(data: Any, schema: Any, partition_values: dict[str, str | None]) -> Any:
    """Shape one file's rows into `schema`: typed partition values, nulls for gaps."""
    pa = _pa()
    rows = data.num_rows
    if len(schema) == 0:
        # No columns asked for (a row count): keep the row count, which
        # `Table.from_arrays` with no arrays would reset to zero.
        return data.select([])
    by_lower = {name.lower(): name for name in data.column_names}
    lowered_partitions = {k.lower(): v for k, v in partition_values.items()}
    columns = []
    for field in schema:
        key = field.name.lower()
        if key in by_lower:
            column = data.column(by_lower[key])
            if column.type != field.type:
                column = _cast_column(column, field.type, field.name)
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
    # A column named twice is returned once, as a SELECT list would not do but
    # every other engine here does.
    wanted = list(dict.fromkeys(c.lower() for c in columns))
    return pa.schema([by_lower[c] for c in wanted])


def _utc_moment(value: Any, *, what: str) -> Any:
    """`value` as an aware UTC datetime; see `sharing_timestamp` for the forms."""
    from datetime import UTC, date, datetime

    moment: datetime
    try:
        if isinstance(value, bool):
            raise UnreachableTableError(what, f"{value!r} is not a timestamp")
        if isinstance(value, (int, float)):
            moment = datetime.fromtimestamp(value / 1000, UTC)
        elif isinstance(value, str):
            text = value.strip()
            if text.lstrip("-").isdigit():
                moment = datetime.fromtimestamp(int(text) / 1000, UTC)
            else:
                moment = datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
        elif isinstance(value, datetime):
            moment = value
        elif isinstance(value, date):
            moment = datetime(value.year, value.month, value.day)
        else:
            raise UnreachableTableError(
                what,
                f"expected a datetime, an ISO-8601 string or epoch milliseconds, not "
                f"{type(value).__name__}",
            )
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.astimezone(UTC)
    except (ValueError, OverflowError, OSError) as exc:
        # Not ISO-8601, or epoch milliseconds outside the datetime range.
        raise UnreachableTableError(
            what, f"{value!r} is not an ISO-8601 timestamp or epoch milliseconds in range"
        ) from exc


def sharing_timestamp(value: Any, *, what: str) -> str:
    """A timestamp as the protocol wants it: ISO-8601 in UTC, ``...Z``.

    Accepts a datetime, a date, an ISO-8601 string or epoch milliseconds (as a
    number or a string of digits). A naive datetime or zoneless string is read
    as UTC, as the other engines here read it. The client would otherwise send
    a `datetime` through ``urllib.parse.quote`` (a TypeError) or pass a string
    such as ``2024-01-01 00:00:00`` straight to a server that expects an
    offset.
    """
    moment = _utc_moment(value, what=what)
    text: str = moment.strftime("%Y-%m-%dT%H:%M:%S")
    if moment.microsecond:
        text += f".{moment.microsecond:06d}".rstrip("0")
    return text + "Z"


def _version_arg(value: Any, name: str, what: str) -> int | None:
    """A table version argument as a non-negative int (a numpy integer is fine).

    The client pastes it into the query string with an f-string, so ``True``
    became ``startingVersion=True`` and ``1.5`` became ``1.5``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise UnreachableTableError(what, f"{name} must be a version number, not {value!r}")
    try:
        number = operator.index(value)
    except TypeError:
        raise UnreachableTableError(
            what, f"{name} must be an integer version, not {value!r}"
        ) from None
    if number < 0:
        raise UnreachableTableError(what, f"{name} must be non-negative, not {number}")
    return number


def _parse_lines(lines: Any, what: str) -> list[dict[str, Any]]:
    """Decode NDJSON response lines, skipping blank keep-alive lines."""
    out = []
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode()
        if not line or not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError as exc:
            raise UnreachableTableError(
                what, f"the sharing server sent a response line that is not JSON ({exc})"
            ) from exc
    return out


def _unwrap(action: dict[str, Any], key: str, inner: str) -> dict[str, Any]:
    body: dict[str, Any] = action[key]
    value: dict[str, Any] = body.get(inner, body)
    return value


class _ExpiredUrlError(UnreachableTableError):
    """A presigned URL was refused with a status an expired signature gets."""


#: Statuses object stores answer an expired presigned URL with: S3 and Azure
#: say 403, GCS says 400, some gateways 401.
_EXPIRED_STATUSES = frozenset({400, 401, 403})


class _UrlBook:
    """Presigned URLs by file key, re-issued by a fresh query when they expire.

    A parquet scan streams, so a slow consumer can outlive the URLs the query
    issued (typically an hour). On an expiry the book re-runs the query pinned
    to the same table version and continues with the new URLs.
    """

    def __init__(self, refresh: Any) -> None:
        self._refresh = refresh
        self._urls: dict[Any, str] = {}

    def url(self, key: Any, original: str) -> str:
        return self._urls.get(key, original)

    def renew(self, key: Any, stale: str) -> str | None:
        try:
            self._urls = dict(self._refresh())
        except Exception:
            return None
        fresh = self._urls.get(key)
        return fresh if fresh and fresh != stale else None


class _SharedSnapshot:
    """What `Table.schema()` asks of an engine: the schema, from metadata alone.

    Without it `Table.schema()` falls back to reading the entire table.
    """

    def __init__(self, engine: SharingEngine, table: ResolvedTable, version: int | None) -> None:
        self._engine, self._table, self._version = engine, table, version

    def schema(self) -> Any:
        detail = self._engine.detail(self._table, version=self._version)
        return delta_schema_to_arrow(detail["schema_string"])


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
        # Quoted: the client puts these into the request path verbatim.
        return sharing_module("delta_sharing.protocol").Table(
            name=quote_name(ref.table), share=quote_name(ref.catalog), schema=quote_name(ref.schema)
        )

    @staticmethod
    def _refusal(
        what: str, exc: BaseException, *, history: bool = False, client: Any = None
    ) -> UnreachableTableError:
        profile = getattr(client, "_profile", None)
        if isinstance(exc, http_error_type()) and getattr(exc, "response", None) is not None:
            text = error_text(exc, profile)
            reason = f"the sharing server refused the request ({text})"
            remedy = _HISTORY_REMEDY if history else None
            lowered = text.lower()
            if history and any(
                marker in lowered
                for marker in (
                    "no such version",
                    "cannot time travel",
                    "available versions",
                    "greater than the latest",
                )
            ):
                # The server named the version itself ("no such version", past
                # the latest): the table's history is shared, and telling the
                # recipient to ask the provider to share it sent them after the
                # wrong fix.
                remedy = "time travel to a version the table has; detail() gives the latest"
        else:
            # A connection failure, a timeout, or a malformed response: nothing
            # to do with history sharing, so do not suggest it.
            reason = f"the sharing server did not answer usably ({error_text(exc, profile)})"
            remedy = None
        return UnreachableTableError(what, reason, remedy)

    @staticmethod
    def _check_url(url: str) -> None:
        scheme = urlparse(url).scheme.lower()
        if scheme not in ("https", "http"):
            # A presigned URL is always HTTP(S); anything else (file://, ftp://)
            # from a server would make this process read local or foreign
            # resources on the server's behalf.
            raise UnreachableTableError(
                "read a shared data file",
                f"the sharing server issued a {scheme or 'relative'!s} URL, not an HTTP(S) "
                "presigned URL",
            )

    @classmethod
    def _check_action_urls(cls, action: dict[str, Any]) -> None:
        """The kernel opens a delta response's paths itself, so check them too."""
        for kind in ("add", "remove", "cdc"):
            body = action.get(kind)
            if not isinstance(body, dict):
                continue
            if body.get("path") is not None:
                cls._check_url(str(body["path"]))
            dv = body.get("deletionVector")
            if isinstance(dv, dict) and dv.get("storageType") == "p":
                cls._check_url(str(dv.get("pathOrInlineDv", "")))

    def _download(self, url: str) -> bytes:
        self._check_url(url)
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as response:
                body: bytes = response.read()
                return body
        except urllib.error.HTTPError as exc:
            kind = _ExpiredUrlError if exc.code in _EXPIRED_STATUSES else UnreachableTableError
            raise kind(
                "read a shared data file",
                f"the presigned URL was refused (HTTP {exc.code}). Presigned URLs expire, "
                "typically within an hour of the query that issued them",
                "re-run the read; a new query issues fresh URLs",
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # The URL itself carries the signature, so it is not repeated here.
            reason = getattr(exc, "reason", exc)
            raise UnreachableTableError(
                "read a shared data file",
                f"downloading it from the object store failed ({type(exc).__name__}: {reason})",
            ) from exc

    def _read_file(
        self,
        url: str,
        partition_values: dict[str, str | None],
        schema: Any,
        *,
        key: Any = None,
        book: _UrlBook | None = None,
    ) -> Any:
        parquet = importlib.import_module("pyarrow.parquet")
        pa = _pa()
        current = book.url(key, url) if book is not None else url
        try:
            payload = self._download(current)
        except _ExpiredUrlError:
            fresh = book.renew(key, current) if book is not None else None
            if fresh is None:
                raise
            payload = self._download(fresh)
        source = pa.BufferReader(payload)
        file_names = parquet.ParquetFile(source).schema_arrow.names
        wanted = {f.name.lower() for f in schema}
        present = [name for name in file_names if name.lower() in wanted]
        source.seek(0)
        data = parquet.read_table(source, columns=present)
        return _conform(data, schema, dict(partition_values or {}))

    def _hints(self, client: Any, shared: Any, predicate: str | None) -> str | None:
        """Hints need the column types, so this costs one metadata request."""
        if not predicate:
            return None
        try:
            metadata = client.query_table_metadata(shared).metadata
            return json_predicate_hints(predicate, metadata.schema_string)
        except Exception:
            return None  # a hint is an optimization; never fail the read over one

    # ------------------------------------------------------------------- read

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
        """Read a shared table as a `pyarrow.RecordBatchReader`.

        Parquet responses stream: files are downloaded one at a time as the
        reader is consumed, and URLs that expire mid-stream are re-issued.
        Delta responses are replayed by the kernel into memory first, because
        the replay needs the log on local disk.

        `limit` is sent as the protocol's ``limitHint`` when there is no
        predicate (with one, the server cannot know which files hold matching
        rows), and at most `limit` rows are returned.
        """
        what = "read the shared table"
        if limit is not None:
            try:
                limit = operator.index(limit)  # an int, a numpy integer...
            except TypeError:
                limit = -1
            if isinstance(limit, bool) or limit < 0:
                raise UnreachableTableError(what, "limit must be a non-negative int")
        if timestamp is not None:
            timestamp = sharing_timestamp(timestamp, what=f"time travel to {timestamp!r}")
        if version is not None and timestamp is not None:
            raise UnreachableTableError(
                what,
                f"both version={version} and timestamp={timestamp!r} were given; the sharing "
                "protocol takes one or the other",
            )
        version = _version_arg(version, "version", what)
        node = _parse_predicate(predicate)
        limit_hint = limit if predicate is None else None
        if self._needs_delta_format(table):
            return self._scan_delta_format(
                table,
                columns=columns,
                predicate=predicate,
                version=version,
                timestamp=timestamp,
                limit=limit,
                limit_hint=limit_hint,
            )

        pa = _pa()
        shared = self._shared(table)
        client = self._client(table)
        travelling = version is not None or timestamp is not None
        try:
            hints = self._hints(client, shared, predicate)
            response = client.list_files_in_table(
                shared,
                jsonPredicateHints=hints,
                limitHint=limit_hint,
                version=version,
                timestamp=timestamp,
            )
        except request_error_types() as exc:
            raise self._refusal(what, exc, history=travelling, client=client) from exc
        finally:
            client.close()

        if response.metadata is None:
            # The server answered with Delta log actions although parquet was
            # the default asked for (it may require them for this table).
            data = self._replay_delta_log(list(response.lines or ()), what=what)
            return self._finish(data, columns=columns, predicate=predicate, limit=limit)

        full = delta_schema_to_arrow(response.metadata.schema_string)
        out = _project(full, columns)
        # The predicate may reference columns the caller did not ask for.
        read = full if predicate else out
        files = list(response.add_files)
        # Checked before streaming: an error raised inside the stream reaches
        # the caller only as pyarrow's ArrowInvalid.
        for f in files:
            self._check_url(f.url)
        pinned = response.delta_table_version

        def refresh() -> dict[Any, str]:
            again = self._client(table)
            try:
                fresh = again.list_files_in_table(
                    shared,
                    jsonPredicateHints=hints,
                    limitHint=limit_hint,
                    version=version,
                    timestamp=timestamp,
                )
                if fresh.delta_table_version != pinned:
                    # The table moved on; ask for the version already being read.
                    fresh = again.list_files_in_table(
                        shared, jsonPredicateHints=hints, limitHint=limit_hint, version=pinned
                    )
            finally:
                again.close()
            return {f.id: f.url for f in fresh.add_files}

        book = _UrlBook(refresh)

        def batches() -> Iterator[Any]:
            remaining = limit
            for f in files:
                if remaining is not None and remaining <= 0:
                    return
                rows = self._read_file(
                    f.url, dict(f.partition_values or {}), read, key=f.id, book=book
                )
                if node is not None:
                    rows = filter_arrow_exact(rows, node).select(out.names)
                if remaining is not None:
                    rows = rows.slice(0, remaining)
                    remaining -= rows.num_rows
                yield from rows.to_batches()

        # The first batch is read now, so the usual failures (an expired or
        # unreachable URL, a file that does not match the schema) raise as
        # themselves. Raised later, inside the stream, pyarrow reports them as
        # ArrowInvalid.
        stream = batches()
        first = next(stream, None)
        head = [] if first is None else [first]
        return pa.RecordBatchReader.from_batches(out, itertools.chain(head, stream))

    def _replay_delta_log(self, lines: list[str], *, what: str, check_urls: bool = True) -> Any:
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
        protocol = metadata = None
        actions = []
        # Dispatch on each line's key rather than its position: blank keep-alive
        # lines and actions this release does not know are skipped.
        for obj in _parse_lines(lines, what):
            if "protocol" in obj:
                protocol = _unwrap(obj, "protocol", "deltaProtocol")
            elif "metaData" in obj:
                metadata = _unwrap(obj, "metaData", "deltaMetadata")
            elif "file" in obj and "deltaSingleAction" in obj["file"]:
                if check_urls:
                    self._check_action_urls(obj["file"]["deltaSingleAction"])
                actions.append(obj["file"]["deltaSingleAction"])
        if protocol is None or metadata is None:
            raise UnreachableTableError(
                what, "the sharing server's delta response has no protocol or metaData line"
            )
        with tempfile.TemporaryDirectory(prefix="deltaswamp-sharing-") as root:
            log = Path(root) / "_delta_log"
            log.mkdir()
            with (log / f"{0:020d}.json").open("w") as out:
                out.write(json.dumps({"protocol": protocol}) + "\n")
                out.write(json.dumps({"metaData": metadata}) + "\n")
                for action in actions:
                    out.write(json.dumps(action) + "\n")
            uri = Path(root).as_uri()
            try:
                interface = kernel.PythonInterface(uri)
                snapshot = kernel.Table(uri).snapshot(interface)
                result = kernel.ScanBuilder(snapshot).build().execute(interface)
                if isinstance(result, pa.RecordBatchReader):
                    return result.read_all()
                batches = list(result)
            except Exception as exc:
                raise UnreachableTableError(
                    what, f"replaying the shared table's log failed ({exc})"
                ) from exc
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
        limit: int | None = None,
        limit_hint: int | None = None,
    ) -> Any:
        what = "read the shared table"
        shared = self._shared(table)
        client = self._client(table)
        travelling = version is not None or timestamp is not None
        try:
            client.set_delta_format_header()
            hints = self._hints(client, shared, predicate)
            response = client.list_files_in_table(
                shared,
                jsonPredicateHints=hints,
                limitHint=limit_hint,
                version=version,
                timestamp=timestamp,
            )
        except request_error_types() as exc:
            raise self._refusal(what, exc, history=travelling, client=client) from exc
        finally:
            client.close()

        data = self._replay_delta_log(list(response.lines), what=what)
        return self._finish(data, columns=columns, predicate=predicate, limit=limit)

    @staticmethod
    def _finish(
        data: Any, *, columns: list[str] | None, predicate: str | None, limit: int | None
    ) -> Any:
        pa = _pa()
        if predicate:
            data = filter_arrow_exact(data, predicate)
        if columns is not None:
            data = data.select(_project(data.schema, columns).names)
        if limit is not None:
            data = data.slice(0, limit)
        return pa.RecordBatchReader.from_batches(data.schema, data.to_batches())

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
    ) -> Any:
        """The table's change data feed as a `pyarrow.RecordBatchReader`.

        Adds ``_change_type`` (insert, delete, update_preimage,
        update_postimage), ``_commit_version`` and ``_commit_timestamp``, as
        Delta's own CDF does. Ranges are inclusive. With neither start given,
        reads from version 0.
        """
        pa = _pa()
        what = "read the change data feed of the shared table"
        starting_version = _version_arg(starting_version, "starting_version", what)
        ending_version = _version_arg(ending_version, "ending_version", what)
        if (
            starting_timestamp is not None
            and ending_timestamp is not None
            and _utc_moment(starting_timestamp, what=what)
            > _utc_moment(ending_timestamp, what=what)
        ):
            raise UnreachableTableError(
                what,
                f"starting_timestamp {starting_timestamp!r} is after ending_timestamp "
                f"{ending_timestamp!r}",
            )
        if (
            starting_version is not None
            and ending_version is not None
            and starting_version > ending_version
        ):
            raise UnreachableTableError(
                what,
                f"starting_version {starting_version} is after ending_version {ending_version} "
                "(the range is inclusive)",
            )
        if starting_timestamp is not None:
            starting_timestamp = sharing_timestamp(starting_timestamp, what=what)
        if ending_timestamp is not None:
            ending_timestamp = sharing_timestamp(ending_timestamp, what=what)
        for bound, by_version, by_time in (
            ("starting", starting_version, starting_timestamp),
            ("ending", ending_version, ending_timestamp),
        ):
            if by_version is not None and by_time is not None:
                raise UnreachableTableError(
                    what,
                    f"both {bound}_version and {bound}_timestamp were given; the sharing "
                    "protocol takes one or the other",
                )
        _parse_predicate(predicate)  # fail before the request, not after the download
        if starting_version is None and starting_timestamp is None:
            starting_version = 0
        protocol = sharing_module("delta_sharing.protocol")
        shared = self._shared(table)

        if self._needs_delta_format(table):
            options = protocol.CdfOptions(
                starting_version=starting_version,
                ending_version=ending_version,
                starting_timestamp=starting_timestamp,
                ending_timestamp=ending_timestamp,
                # The replay needs every schema change in the range.
                include_historical_metadata=True,
            )
            data = self._cdf_delta_format(table, shared, options, what)
        else:
            options = protocol.CdfOptions(
                starting_version=starting_version,
                ending_version=ending_version,
                starting_timestamp=starting_timestamp,
                ending_timestamp=ending_timestamp,
            )
            client = self._client(table)
            try:
                response = client.list_table_changes(shared, options)
            except request_error_types() as exc:
                raise self._refusal(what, exc, history=True, client=client) from exc
            finally:
                client.close()
            if response.actions is None:
                # Answered with Delta log actions although parquet was asked for.
                data = self._replay_cdf_lines(
                    list(response.lines or ()),
                    starting_version=starting_version,
                    what=what,
                    schema_fallback=lambda: str(self.detail(table)["schema_string"]),
                )
            else:
                data = self._cdf_parquet(response, table=table, shared=shared, options=options)

        if predicate:
            data = filter_arrow_exact(data, predicate)
        if columns is not None:
            keep = [*columns, *(c for c in (CHANGE_TYPE, COMMIT_VERSION, COMMIT_TIMESTAMP))]
            data = data.select(_project(data.schema, keep).names)
        return pa.RecordBatchReader.from_batches(data.schema, data.to_batches())

    @staticmethod
    def _action_key(action: Any) -> Any:
        # The same file id appears as an add in one version and a remove in a
        # later one, so the id alone does not name an action.
        return (type(action).__name__, action.id, action.version)

    def _cdf_parquet(
        self,
        response: Any,
        *,
        table: ResolvedTable | None = None,
        shared: Any = None,
        options: Any = None,
    ) -> Any:
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

        def refresh() -> dict[Any, str]:
            if table is None:
                return {}
            again = self._client(table)
            try:
                fresh = again.list_table_changes(shared, options)
            finally:
                again.close()
            return {self._action_key(a): a.url for a in fresh.actions if a is not None}

        book = _UrlBook(refresh)
        pieces = []
        for action in response.actions or ():
            if action is None:
                continue
            partitions = dict(action.partition_values or {})
            key = self._action_key(action)
            if isinstance(action, protocol.AddCdcFile):
                rows = self._read_file(
                    action.url,
                    partitions,
                    pa.schema([*data_schema, change_field]),
                    key=key,
                    book=book,
                )
            else:
                rows = self._read_file(action.url, partitions, data_schema, key=key, book=book)
                change = action.get_change_type_col_value()
                rows = rows.append_column(
                    change_field, pa.array([change] * rows.num_rows, pa.string())
                )
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
        """Column-mapped or DV tables: the change log replayed by the kernel.

        Kept in Arrow end to end. (The client's ``table_changes_to_pandas``
        goes through pandas, which turns a nullable integer into float64 and
        loses the Delta types; it also stamps commit timestamps at one-second
        resolution.)
        """
        client = self._client(table)
        try:
            client.set_delta_format_header(for_cdf=True)
            response = client.list_table_changes(shared, options)
        except request_error_types() as exc:
            raise self._refusal(what, exc, history=True, client=client) from exc
        finally:
            client.close()

        def current_schema() -> str:
            return str(self.detail(table)["schema_string"])

        return self._replay_cdf_lines(
            list(response.lines or ()),
            starting_version=options.starting_version,
            what=what,
            schema_fallback=current_schema,
        )

    def _replay_cdf_lines(
        self,
        lines: list[str],
        *,
        starting_version: int | None,
        what: str,
        schema_fallback: Any = None,
        check_urls: bool = True,
    ) -> Any:
        """Write a delta-format ``changes`` response as a log and replay its CDF."""
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

        protocol: dict[str, Any] | None = None
        metadata: dict[int, dict[str, Any]] = {}
        actions: dict[int, list[dict[str, Any]]] = {}
        stamps: dict[int, int] = {}
        for obj in _parse_lines(lines, what):
            if "protocol" in obj:
                protocol = _unwrap(obj, "protocol", "deltaProtocol")
            elif "metaData" in obj:
                body = obj["metaData"]
                version = body.get("version")
                metadata[int(version) if version is not None else -1] = body.get(
                    "deltaMetadata", body
                )
            elif "file" in obj and "deltaSingleAction" in obj["file"]:
                entry = obj["file"]
                if check_urls:
                    self._check_action_urls(entry["deltaSingleAction"])
                version = int(entry["version"])
                actions.setdefault(version, []).append(entry["deltaSingleAction"])
                if entry.get("timestamp") is not None:
                    stamps[version] = int(entry["timestamp"])
        if protocol is None:
            raise UnreachableTableError(
                what, "the sharing server's delta response has no protocol line"
            )

        cdc_fields = [
            pa.field(CHANGE_TYPE, pa.string()),
            pa.field(COMMIT_VERSION, pa.int64()),
            pa.field(COMMIT_TIMESTAMP, pa.timestamp("ms", tz="UTC")),
        ]
        if not actions:
            known = [m for v, m in sorted(metadata.items())]
            schema_string = (
                known[-1]["schemaString"]
                if known
                else (schema_fallback() if schema_fallback is not None else None)
            )
            if schema_string is None:
                raise UnreachableTableError(what, "the response carries no table schema")
            data_schema = delta_schema_to_arrow(schema_string)
            return pa.schema([*data_schema, *cdc_fields]).empty_table()

        versions = set(actions) | {v for v in metadata if v >= 0}
        first = min(versions)
        if starting_version is not None:
            first = min(first, int(starting_version))
        last = max(versions)
        if first not in metadata:
            # The replay needs a metaData action at its first version.
            if not metadata:
                raise UnreachableTableError(what, "the response carries no table metadata")
            metadata[first] = metadata[min(metadata)]

        fake_checkpoint = sharing_module("delta_sharing.fake_checkpoint")
        with tempfile.TemporaryDirectory(prefix="deltaswamp-sharing-cdf-") as root:
            log = Path(root) / "_delta_log"
            log.mkdir()
            for version in range(first, last + 1):
                path = log / f"{version:020d}.json"
                with path.open("w") as out:
                    if version == first:
                        out.write(json.dumps({"protocol": protocol}) + "\n")
                    if version in metadata:
                        out.write(json.dumps({"metaData": metadata[version]}) + "\n")
                    for action in actions.get(version, ()):
                        out.write(json.dumps(action) + "\n")
                if version in stamps:
                    # The kernel takes _commit_timestamp from the commit file's
                    # modification time; set it to the millisecond.
                    ns = stamps[version] * 1_000_000
                    os.utime(path, ns=(ns, ns))
            if first > 0:
                # A checkpoint just before the range, so the kernel starts there
                # instead of looking for the commits the server did not send.
                payload = fake_checkpoint.get_fake_checkpoint_byte_array()
                (log / f"{first - 1:020d}.checkpoint.parquet").write_bytes(payload)
                (log / "_last_checkpoint").write_text(
                    json.dumps({"version": first - 1, "size": len(payload)})
                )
            uri = Path(root).as_uri()
            try:
                interface = kernel.PythonInterface(uri)
                scan = kernel.TableChangesScanBuilder(
                    kernel.Table(uri), interface, first, last
                ).build()
                result = scan.execute(interface)
                data = (
                    result.read_all()
                    if isinstance(result, pa.RecordBatchReader)
                    else pa.Table.from_batches(list(result))
                )
            except Exception as exc:
                raise UnreachableTableError(
                    what, f"replaying the shared table's change log failed ({exc})"
                ) from exc
        # Same types as the parquet path, so the two responses read alike.
        if COMMIT_TIMESTAMP in data.column_names:
            index = data.column_names.index(COMMIT_TIMESTAMP)
            data = data.set_column(
                index,
                cdc_fields[2],
                data.column(index).cast(pa.timestamp("ms", tz="UTC"), safe=False),
            )
        return data

    # --------------------------------------------------------------- metadata

    def snapshot(self, table: ResolvedTable, *, version: int | None = None) -> _SharedSnapshot:
        """A metadata-only view, so `Table.schema()` need not read the data."""
        return _SharedSnapshot(self, table, version)

    def detail(self, table: ResolvedTable, *, version: int | None = None) -> dict[str, Any]:
        """Version, protocol and metadata as the sharing server reports them."""
        protocol_module = sharing_module("delta_sharing.protocol")
        what = "describe the shared table"
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
                    protocol = metadata = None
                    for obj in _parse_lines(response.lines, what):
                        if "protocol" in obj:
                            protocol = protocol_module.Protocol.from_json(obj["protocol"])
                        elif "metaData" in obj:
                            metadata = protocol_module.Metadata.from_json(obj["metaData"])
                    if protocol is None or metadata is None:
                        raise UnreachableTableError(
                            what, "the sharing server's response has no protocol or metaData line"
                        )
        except request_error_types() as exc:
            raise self._refusal(what, exc, history=version is not None, client=client) from exc
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
