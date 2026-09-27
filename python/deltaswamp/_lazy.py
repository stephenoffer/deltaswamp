"""A pyarrow Dataset that reads a table lazily, with projection and filter pushdown.

`to_duckdb()`, `to_polars(lazy=True)`, `to_pyarrow_dataset()` and
`Connection.sql()` used to read the whole table into memory before the
caller's filter ran: `t.to_duckdb().filter("id < 10")` read ten million rows
to return ten. DuckDB and Polars both push projections and filters into a
pyarrow Dataset by calling its ``scanner()`` / ``to_batches()`` with a
pyarrow Expression; this Dataset answers those calls with a scan through the
table's own engine (so deletion vectors, column mapping and catalog-managed
tables read correctly), passing the columns and -- where it translates -- the
filter down for file skipping.

The filter is also applied exactly, by the consumer, to what the scan returns:
the translation to SQL only has to be no stronger than the expression, never
equal to it, and anything it cannot read is simply not pushed. "No stronger"
must hold on every value, NULL and NaN included, and the scan applies the
pushed predicate exactly too, so only what reads the same in both is pushed:
conjunctions (and disjunctions) of a column compared with a literal whose type
is the column's own kind -- integers, strings, booleans, dates, decimals --
plus IS [NOT] NULL and IN over such literals. Never NOT (Spark orders NaN above
every number, pyarrow compares it false, so ``NOT (x > 1)`` drops a NaN row
pyarrow keeps), never a float or double (the same NaN, and a float32 0.1 is
not the decimal 0.1), never a binary (its text form is hex). The literals'
types come from the expression itself, decoded from its serialized form; the
printed form cannot tell ``'6162'`` from ``b'ab'``.
"""

from __future__ import annotations

import datetime as _dt
import decimal as _decimal
import json
import re
from typing import Any

import pyarrow as pa
import pyarrow.dataset as pds

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_COMPARISONS = {
    "equal": "=",
    "not_equal": "<>",
    "less": "<",
    "less_equal": "<=",
    "greater": ">",
    "greater_equal": ">=",
}
#: The comparison seen from the other side: `1 < x` is `x > 1`.
_FLIPPED = {"=": "=", "<>": "<>", "<": ">", "<=": ">=", ">": "<", ">=": "<="}


class _Unsupported(Exception):
    """An expression form the translation does not push."""


# ----------------------------------------------------------- literals, typed


def _kind(arrow_type: Any) -> str | None:
    """The literal family a column (or literal) of `arrow_type` belongs to."""
    t = pa.types
    if t.is_integer(arrow_type):
        return "int"
    if t.is_string(arrow_type) or t.is_large_string(arrow_type) or _is_string_view(arrow_type):
        return "str"
    if t.is_boolean(arrow_type):
        return "bool"
    if t.is_date32(arrow_type):
        return "date"
    if t.is_decimal(arrow_type):
        return "decimal"
    return None  # floats (NaN), binary, timestamps, nested: never pushed


def _is_string_view(arrow_type: Any) -> bool:
    check = getattr(pa.types, "is_string_view", None)
    return bool(check is not None and check(arrow_type))


def _fits(column_type: Any, value: int) -> bool:
    """Whether the integer `value` is representable in the integer `column_type`."""
    bits = int(column_type.bit_width)
    if pa.types.is_signed_integer(column_type):
        return bool(-(1 << (bits - 1)) <= value < (1 << (bits - 1)))
    return bool(0 <= value < (1 << bits))


def _render(column_type: Any, kind: str, value: Any) -> str:
    """`value` (of literal family `kind`) as SQL that reads exactly as that value.

    Raises _Unsupported unless the literal is of the column's own family, so
    both sides compare the same values the same way.
    """
    if value is None or _kind(column_type) != kind:
        raise _Unsupported("literal type differs from the column's")
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int) or not _fits(column_type, value):
            raise _Unsupported("integer literal")
        return str(value)
    if kind == "str":
        if not isinstance(value, str):
            raise _Unsupported("string literal")
        # Spark SQL string syntax, which this library's parser reads: a
        # backslash escapes, and '' is two adjacent literals, not a quote.
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if kind == "bool":
        if not isinstance(value, bool):
            raise _Unsupported("boolean literal")
        return "TRUE" if value else "FALSE"
    if kind == "date":
        if not isinstance(value, _dt.date) or isinstance(value, _dt.datetime):
            raise _Unsupported("date literal")
        return f"DATE '{value.isoformat()}'"
    if kind == "decimal":
        if not isinstance(value, _decimal.Decimal) or not value.is_finite():
            raise _Unsupported("decimal literal")
        # Positional, never 1E+2: an unsuffixed number with a point is a
        # DECIMAL in SQL, compared exactly.
        return format(value, "f")
    raise _Unsupported(kind)


def _column(name: Any, schema: Any) -> tuple[str, Any]:
    """(quoted name, arrow type) for a plain top-level column of `schema`."""
    if not isinstance(name, str) or _IDENT.fullmatch(name) is None:
        raise _Unsupported("not a plain column")
    index = schema.get_field_index(name)
    if index < 0:
        raise _Unsupported("not a column")
    return f"`{name}`", schema.field(index).type


def _comparison(schema: Any, name: Any, op: str, kind: str | None, value: Any) -> str:
    quoted, column_type = _column(name, schema)
    if kind is None:
        raise _Unsupported("literal type")
    return f"({quoted} {op} {_render(column_type, kind, value)})"


def _conjoin(parts: list[str | None]) -> str | None:
    """AND of the parts that translated: dropping a conjunct only weakens it."""
    kept = [p for p in parts if p]
    if not kept:
        return None
    return kept[0] if len(kept) == 1 else "(" + " AND ".join(kept) + ")"


# ------------------------------------------------------- pyarrow Expressions


def _metadata_pairs(schema: Any) -> list[tuple[bytes, bytes]]:
    """Every (key, value) of `schema`'s metadata, in order, duplicates kept.

    `Schema.metadata` is a dict and keeps one value per key, where the
    serialized expression is a sequence of repeated keys. The C Data
    Interface export carries the whole sequence.
    """
    import ctypes

    class _ArrowSchema(ctypes.Structure):
        # The leading members of struct ArrowSchema; only these are read.
        _fields_ = (
            ("format", ctypes.c_char_p),
            ("name", ctypes.c_char_p),
            ("metadata", ctypes.c_void_p),
        )

    capsule = schema.__arrow_c_schema__()
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    pointer = get(capsule, b"arrow_schema")
    if not pointer:
        raise _Unsupported("no schema")
    base = _ArrowSchema.from_address(pointer).metadata
    if not base:
        return []

    def int32(at: int) -> int:
        return int(ctypes.c_int32.from_address(at).value)

    pairs, at = [], base + 4
    for _ in range(int32(base)):
        size = int32(at)
        key = ctypes.string_at(at + 4, size)
        at += 4 + size
        size = int32(at)
        value = ctypes.string_at(at + 4, size)
        at += 4 + size
        pairs.append((key, value))
    del capsule  # releases the exported schema, after the last read
    return pairs


def _decode(expression: Any) -> Any:
    """The pyarrow `expression` as a tree, with every literal as a typed scalar.

    Nodes: ("field", name), ("literal", scalar) and ("call", function, args,
    options). Read from Arrow's own serialization of the expression (what
    pickling it uses): a one-row batch holding each literal, and schema
    metadata spelling out the calls and field references in order.
    """
    buffer = expression.__reduce__()[1][0]
    reader = pa.ipc.open_file(buffer)
    batch = reader.get_batch(0) if reader.num_record_batches else None
    pairs = _metadata_pairs(reader.schema)

    def scalar(index: bytes) -> Any:
        if batch is None:
            raise _Unsupported("no literals")
        return batch.column(int(index))[0]

    def node(i: int) -> tuple[Any, int]:
        key, value = pairs[i]
        if key == b"literal":
            return ("literal", scalar(value)), i + 1
        if key == b"field_ref":
            return ("field", value.decode()), i + 1
        if key != b"call":
            raise _Unsupported(key.decode(errors="replace"))
        name, args, options, i = value.decode(), [], None, i + 1
        while pairs[i][0] != b"end":
            if pairs[i][0] == b"options":
                options, i = scalar(pairs[i][1]), i + 1
            else:
                child, i = node(i)
                args.append(child)
        return ("call", name, args, options), i + 1

    tree, used = node(0)
    if used != len(pairs):
        raise _Unsupported("trailing metadata")
    return tree


def _literal(scalar: Any) -> tuple[str | None, Any]:
    kind = _kind(scalar.type)
    if kind is None or not scalar.is_valid:
        return None, None
    return kind, scalar.as_py()


def _arrow_sql(node: Any, schema: Any) -> str:
    """SQL implied by the decoded pyarrow `node`, or _Unsupported."""
    tag = node[0]
    if tag == "field":
        _, column_type = _column(node[1], schema)
        if not pa.types.is_boolean(column_type):
            raise _Unsupported("bare non-boolean column")
        return _column(node[1], schema)[0]
    if tag != "call":
        raise _Unsupported("bare literal")
    _, name, args, options = node
    if name in ("and", "and_kleene"):
        out = _conjoin([_arrow_sql_or_none(a, schema) for a in args])
        if out is None:
            raise _Unsupported("no conjunct translates")
        return out
    if name in ("or", "or_kleene"):
        return "(" + " OR ".join(_arrow_sql(a, schema) for a in args) + ")"
    if name in _COMPARISONS and len(args) == 2:
        left, right = args
        op = _COMPARISONS[name]
        if left[0] == "literal" and right[0] == "field":
            left, right, op = right, left, _FLIPPED[op]
        if left[0] != "field" or right[0] != "literal":
            raise _Unsupported("not column-versus-literal")
        kind, value = _literal(right[1])
        return _comparison(schema, left[1], op, kind, value)
    if name in ("is_null", "is_valid") and len(args) == 1 and args[0][0] == "field":
        quoted, column_type = _column(args[0][1], schema)
        if name == "is_valid":
            return f"({quoted} IS NOT NULL)"
        nan_is_null = options is not None and bool(options.as_py().get("nan_is_null"))
        if nan_is_null and (pa.types.is_floating(column_type) or _kind(column_type) is None):
            raise _Unsupported("is_null matching NaN")
        return f"({quoted} IS NULL)"
    if name == "is_in" and len(args) == 1 and args[0][0] == "field" and options is not None:
        quoted, column_type = _column(args[0][1], schema)
        values = options["value_set"]
        kind = _kind(values.type.value_type)
        items = values.as_py() or []
        # With no null in the set every matching behaviour drops a null row,
        # as SQL IN does; a null in it is not pushed.
        if not items or kind is None or any(v is None for v in items):
            raise _Unsupported("value set")
        rendered = ", ".join(_render(column_type, kind, v) for v in items)
        return f"({quoted} IN ({rendered}))"
    # invert/not and every other function: never pushed.
    raise _Unsupported(name)


def _arrow_sql_or_none(node: Any, schema: Any) -> str | None:
    try:
        return _arrow_sql(node, schema)
    except (_Unsupported, KeyError, IndexError, TypeError, ValueError):
        return None


def expression_to_sql(expression: Any, schema: Any) -> str | None:
    """A SQL predicate implied by the pyarrow `expression` on `schema`, or None.

    Implied on every row, NULL and NaN included: whatever the expression
    keeps, the predicate keeps too (see the module docstring for what that
    allows). Conjuncts that do not translate are dropped -- a weaker
    predicate reads more, never less -- and anything else untranslatable
    drops the whole predicate.
    """
    if expression is None or not isinstance(schema, pa.Schema):
        return None
    try:
        tree = _decode(expression)
    except Exception:
        return None  # a form this pyarrow cannot serialize, or we cannot read
    return _arrow_sql_or_none(tree, schema)


# ------------------------------------------------------------ Polars filters

_POLARS_OPS = {"Eq": "=", "NotEq": "<>", "Lt": "<", "LtEq": "<=", "Gt": ">", "GtEq": ">="}
_POLARS_INTS = frozenset(
    {"Int8", "Int16", "Int32", "Int64", "Int128", "UInt8", "UInt16", "UInt32", "UInt64"}
)


def _polars_literal(node: Any) -> tuple[str | None, Any]:
    """(family, value) of a serialized Polars literal; (None, None) if not pushable."""
    literal = node.get("Literal") if isinstance(node, dict) else None
    if not isinstance(literal, dict) or len(literal) != 1:
        return None, None
    ((form, body),) = literal.items()
    if form not in ("Dyn", "Scalar") or not isinstance(body, dict) or len(body) != 1:
        return None, None
    ((dtype, value),) = body.items()
    if dtype == "Int" or dtype in _POLARS_INTS:
        return (
            ("int", value)
            if isinstance(value, int) and not isinstance(value, bool)
            else (
                None,
                None,
            )
        )
    if dtype in ("String", "Str") and isinstance(value, str):
        return "str", value
    if dtype == "Boolean" and isinstance(value, bool):
        return "bool", value
    if dtype == "Date" and isinstance(value, int) and not isinstance(value, bool):
        return "date", _dt.date(1970, 1, 1) + _dt.timedelta(days=value)
    return None, None


def _polars_sql(node: Any, schema: Any) -> str:
    if not isinstance(node, dict) or len(node) != 1:
        raise _Unsupported("node")
    ((tag, body),) = node.items()
    if tag == "Column":
        _, column_type = _column(body, schema)
        if not pa.types.is_boolean(column_type):
            raise _Unsupported("bare non-boolean column")
        return _column(body, schema)[0]
    if tag == "BinaryExpr":
        op, left, right = body.get("op"), body.get("left"), body.get("right")
        if op in ("And", "LogicalAnd"):
            out = _conjoin([_polars_sql_or_none(left, schema), _polars_sql_or_none(right, schema)])
            if out is None:
                raise _Unsupported("no conjunct translates")
            return out
        if op in ("Or", "LogicalOr"):
            return f"({_polars_sql(left, schema)} OR {_polars_sql(right, schema)})"
        if op in _POLARS_OPS:
            sql_op = _POLARS_OPS[op]
            if isinstance(left, dict) and "Literal" in left and "Column" in (right or {}):
                left, right, sql_op = right, left, _FLIPPED[sql_op]
            if not isinstance(left, dict) or "Column" not in left:
                raise _Unsupported("not column-versus-literal")
            kind, value = _polars_literal(right)
            return _comparison(schema, left["Column"], sql_op, kind, value)
        raise _Unsupported(str(op))
    if tag == "Function":
        function, args = body.get("function"), body.get("input") or []
        if function in ({"Boolean": "IsNull"}, {"Boolean": "IsNotNull"}) and len(args) == 1:
            if not isinstance(args[0], dict) or "Column" not in args[0]:
                raise _Unsupported("not a column")
            quoted, _ = _column(args[0]["Column"], schema)
            negated = function == {"Boolean": "IsNotNull"}
            return f"({quoted} IS {'NOT ' if negated else ''}NULL)"
    # Not and every other function: never pushed.
    raise _Unsupported(tag)


def _polars_sql_or_none(node: Any, schema: Any) -> str | None:
    try:
        return _polars_sql(node, schema)
    except (_Unsupported, KeyError, IndexError, TypeError, ValueError, AttributeError):
        return None


def polars_to_sql(predicate: Any, schema: Any) -> str | None:
    """A SQL predicate implied by the Polars `predicate` on `schema`, or None.

    Only used to skip files and rows the predicate cannot match: the frame
    applies `predicate` itself, with Polars' own semantics, afterwards.
    """
    if predicate is None:
        return None
    try:
        tree = json.loads(predicate.meta.serialize(format="json"))
    except Exception:
        return None
    return _polars_sql_or_none(tree, schema)


def polars_frame(dataset: TableDataset) -> Any:
    """A Polars LazyFrame over `dataset` that filters with Polars' semantics.

    `scan_pyarrow_dataset` translated a Polars filter into a pyarrow one and
    let pyarrow apply it, so ``x > 1`` dropped NaN rows Polars keeps (NaN is
    the largest float in Polars, and compares false in pyarrow). Here only the
    projection, the row limit and the safe part of the filter (for skipping)
    reach the scan; the filter itself runs in Polars, on what the scan read.
    """
    import polars as pl
    from polars.io.plugins import register_io_source

    arrow_schema = dataset.schema
    polars_schema = pl.from_arrow(arrow_schema.empty_table()).schema  # type: ignore[union-attr, unused-ignore]
    names = list(arrow_schema.names)

    def source(
        with_columns: list[str] | None,
        predicate: Any,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Any:
        pushed = polars_to_sql(predicate, arrow_schema)
        read = None
        if with_columns is not None:
            wanted = set(with_columns)
            if predicate is not None:
                wanted |= set(predicate.meta.root_names())
            read = [n for n in names if n in wanted] or names[:1]
        if n_rows is not None and n_rows <= 0:
            return
        remaining = n_rows
        for batch in dataset._read(read, pushed):
            frame = pl.from_arrow(batch)
            assert isinstance(frame, pl.DataFrame)
            if predicate is not None:
                frame = frame.filter(predicate)
            if with_columns is not None:
                frame = frame.select(with_columns)
            if remaining is not None:
                frame = frame.head(remaining)
                remaining -= frame.height
            yield frame
            if remaining is not None and remaining <= 0:
                return

    return register_io_source(source, schema=polars_schema)


def referenced_columns(expression: Any, names: list[str]) -> set[str]:
    """Every column name that appears in the expression's text (a superset)."""
    text = str(expression)
    return {
        n for n in names if re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", text) is not None
    }


class TableDataset(pds.InMemoryDataset):  # type: ignore[misc]
    """A pyarrow Dataset over a `Table` that scans it only when read.

    Every read is a fresh scan through the table's engine at the handle's
    version, so the same relation or LazyFrame can be executed repeatedly.
    Methods pyarrow would answer from the (empty) in-memory placeholder
    materialize the table first instead.
    """

    def __init__(
        self,
        table: Any,
        schema: Any,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        scan_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(schema.empty_table())
        self._table = table
        self._columns = columns
        self._predicate = predicate
        self._scan_options = dict(scan_options or {})
        self._schema_ = schema
        self._base_filter: Any = None

    @property
    def schema(self) -> Any:
        return self._schema_

    def _reader(self, columns: list[str] | None, filter: Any) -> Any:
        names = list(self._schema_.names)
        pushed = expression_to_sql(filter, self._schema_) if filter is not None else None
        read = None
        if columns is not None:
            wanted = set(columns) | (
                referenced_columns(filter, names) if filter is not None else set()
            )
            read = [n for n in names if n in wanted] or names[:1]
        return self._read(read, pushed)

    def _read(self, columns: list[str] | None, pushed: str | None) -> Any:
        """A scan of `columns` (None: all) with `pushed` SQL added to the predicate."""
        predicate = " AND ".join(f"({p})" for p in (self._predicate, pushed) if p) or None
        stream = self._table.scan(columns=columns, predicate=predicate, **self._scan_options)
        return pa.RecordBatchReader.from_stream(stream)

    def scanner(self, columns: Any = None, filter: Any = None, **kwargs: Any) -> Any:
        if self._base_filter is not None:
            filter = self._base_filter if filter is None else (self._base_filter & filter)
        project: Any
        if isinstance(columns, dict):
            # Computed projections: read every column and let pyarrow project.
            reader, project = self._reader(None, filter), columns
        else:
            project = list(columns) if columns is not None else None
            reader = self._reader(project, filter)
        return pds.Scanner.from_batches(
            reader,
            columns=project,
            filter=filter,
            **{k: v for k, v in kwargs.items() if k in ("batch_size", "use_threads")},
        )

    def to_batches(self, columns: Any = None, filter: Any = None, **kwargs: Any) -> Any:
        return self.scanner(columns=columns, filter=filter, **kwargs).to_batches()

    def to_table(self, columns: Any = None, filter: Any = None, **kwargs: Any) -> Any:
        return self.scanner(columns=columns, filter=filter, **kwargs).to_table()

    def head(self, num_rows: int, columns: Any = None, filter: Any = None, **kwargs: Any) -> Any:
        return self.scanner(columns=columns, filter=filter, **kwargs).head(num_rows)

    def count_rows(self, filter: Any = None, **kwargs: Any) -> int:
        if filter is None and self._base_filter is None and not self._scan_options:
            return int(self._table.count(predicate=self._predicate))
        return int(self.scanner(filter=filter, **kwargs).count_rows())

    def take(self, indices: Any, **kwargs: Any) -> Any:
        return self.to_table(**kwargs).take(indices)

    def filter(self, expression: Any) -> TableDataset:
        out = TableDataset(
            self._table,
            self._schema_,
            columns=self._columns,
            predicate=self._predicate,
            scan_options=self._scan_options,
        )
        base = self._base_filter
        out._base_filter = expression if base is None else (base & expression)
        return out

    def _materialized(self) -> Any:
        return pds.dataset(self.to_table())

    def sort_by(self, sorting: Any, **kwargs: Any) -> Any:
        return self._materialized().sort_by(sorting, **kwargs)

    def join(self, *args: Any, **kwargs: Any) -> Any:
        return self._materialized().join(*args, **kwargs)

    def join_asof(self, *args: Any, **kwargs: Any) -> Any:
        return self._materialized().join_asof(*args, **kwargs)

    def replace_schema(self, schema: Any) -> Any:
        return self._materialized().replace_schema(schema)

    def get_fragments(self, filter: Any = None) -> Any:
        return self._materialized().get_fragments(filter=filter)

    def __reduce__(self) -> Any:
        # The placeholder pickles as an empty table; ship the rows instead.
        return (pds.dataset, (self.to_table(),))
