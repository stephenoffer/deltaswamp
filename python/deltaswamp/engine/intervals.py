"""Databricks' ANSI interval columns, read as the warehouse gives them.

Delta has no interval type of its own, but Databricks stores ``INTERVAL DAY
TO SECOND`` (and its narrower day-time forms) as a Parquet INT64 of
microseconds and ``INTERVAL YEAR TO MONTH`` (and ``YEAR`` / ``MONTH``) as an
INT32 of months, with the interval spelled only in the log's schema. The
kernel reads both as bare integers, so a day-time value came back as
``93784500000`` where the warehouse sent ``1 day, 2:03:04.5`` -- and writing
that integer back through the warehouse stored it as *seconds*, a million
times too large, without an error.

So the direct engines' reads are retyped here from the log's schema, to what
the warehouse sends: a day-time interval is a microsecond ``duration``. A
year-month interval is text in Spark's own form (``INTERVAL '1-2' YEAR TO
MONTH``): the warehouse's Arrow type for it (``month_interval``) is one
pyarrow cannot convert to Python, pandas, polars or Parquet, and the text
casts back to the same column type on a write.
"""

from __future__ import annotations

import re
from typing import Any

from .._variant import ELEMENT, VALUE, Paths, _convert, _convert_type, _under
from ..errors import InvalidArgumentError

__all__ = [
    "DAY_TIME",
    "IntervalPaths",
    "interval_paths",
    "interval_schema",
    "interval_stream",
    "month_interval_text",
    "sql_identifiers",
    "storage_columns",
    "year_month_text",
]

#: The group of day-time intervals (read as ``duration[us]``). Every other
#: group is named by its year-month qualifier: ``YEAR TO MONTH``, ``YEAR`` or
#: ``MONTH``.
DAY_TIME = "DAY TO SECOND"

#: Interval positions of a schema, by group: each group's paths (field names,
#: with ELEMENT / VALUE steps into arrays and maps, as `_variant` has them).
IntervalPaths = dict[str, Paths]


def _group(kind: str) -> str | None:
    words = kind.lower().split()
    if not words or words[0] != "interval":
        return None
    if "year" in words or "month" in words:
        return " ".join(words[1:]).upper()
    return DAY_TIME


def interval_paths(schema: Any) -> IntervalPaths:
    """The interval columns of a Delta schema (the log's `schemaString`, parsed)."""
    out: dict[str, set[tuple[str, ...]]] = {}

    def walk_type(kind: Any, path: tuple[str, ...]) -> None:
        if isinstance(kind, str):
            group = _group(kind)
            if group is not None:
                out.setdefault(group, set()).add(path)
        elif isinstance(kind, dict):
            if kind.get("type") == "struct":
                walk(kind.get("fields"), path)
            elif kind.get("type") == "array":
                walk_type(kind.get("elementType"), (*path, ELEMENT))
            elif kind.get("type") == "map":
                walk_type(kind.get("valueType"), (*path, VALUE))

    def walk(fields: Any, path: tuple[str, ...]) -> None:
        for field in fields or ():
            if isinstance(field, dict) and isinstance(field.get("name"), str):
                walk_type(field.get("type"), (*path, field["name"]))

    if isinstance(schema, dict):
        walk(schema.get("fields"), ())
    return {group: frozenset(paths) for group, paths in out.items()}


def year_month_text(pa: Any, months: Any, qualifier: str) -> Any:
    """Months as Spark prints the interval: ``INTERVAL '-1-2' YEAR TO MONTH``."""
    import pyarrow.compute as pc

    wide = months.cast(pa.int64())
    negative = pc.less(wide, 0)
    count = pc.abs(wide)
    years = pc.divide(count, 12)
    if qualifier == "YEAR":
        body = years.cast(pa.string())
    elif qualifier == "MONTH":
        body = count.cast(pa.string())
    else:
        rest = pc.subtract(count, pc.multiply(years, 12))
        body = pc.binary_join_element_wise(years.cast(pa.string()), rest.cast(pa.string()), "-")
    signed = pc.if_else(negative, pc.binary_join_element_wise("-", body, ""), body)
    return pc.binary_join_element_wise("INTERVAL '", signed, f"' {qualifier}", "")


def _leaf_type(pa: Any, group: str) -> Any:
    wanted = pa.duration("us") if group == DAY_TIME else pa.string()

    def leaf(arrow_type: Any) -> Any:
        return wanted if pa.types.is_integer(arrow_type) else None

    return leaf


def _leaf(pa: Any, group: str) -> Any:
    def leaf(array: Any) -> Any:
        if not pa.types.is_integer(array.type):
            return array
        if group == DAY_TIME:
            return array.cast(pa.duration("us"))
        return year_month_text(pa, array, group)

    return leaf


def interval_schema(pa: Any, schema: Any, groups: IntervalPaths) -> Any:
    """`schema` with each interval column given the type reads give it."""
    for group, paths in groups.items():
        leaf = _leaf_type(pa, group)
        schema = pa.schema(
            [
                f.with_type(
                    _convert_type(pa, f.type, _under(paths, f.name), (f.name,) in paths, leaf)
                )
                for f in schema
            ],
            metadata=schema.metadata,
        )
    return schema


def _convert_column(pa: Any, array: Any, name: str, groups: IntervalPaths) -> Any:
    for group, paths in groups.items():
        array = _convert(
            pa,
            array,
            _under(paths, name),
            (name,) in paths,
            _leaf(pa, group),
            _leaf_type(pa, group),
        )
    return array


def interval_stream(stream: Any, groups: IntervalPaths) -> Any:
    """A direct engine's stream with its interval columns retyped (see the module docs)."""
    if not groups:
        return stream
    import pyarrow as pa

    from .base import TranslatingStream

    # A TranslatingStream is read as it is: through the C stream interface
    # its errors would arrive untyped.
    reader: Any = (
        stream.to_reader()
        if isinstance(stream, pa.Table)
        else stream
        if isinstance(stream, (pa.RecordBatchReader, TranslatingStream))
        else pa.RecordBatchReader.from_stream(stream)
    )
    schema = reader.schema
    target = interval_schema(pa, schema, groups)
    if target == schema:
        return reader

    def batches() -> Any:
        for batch in reader:
            arrays = [
                _convert_column(pa, batch.column(i), f.name, groups) for i, f in enumerate(schema)
            ]
            yield pa.RecordBatch.from_arrays(arrays, schema=target)

    return pa.RecordBatchReader.from_batches(target, batches())


def month_interval_text(
    pa: Any, schema: Any, batches: list[Any], qualifiers: dict[str, str]
) -> Any:
    """A warehouse result (`schema`, `batches`) as a table, ``month_interval`` columns as text.

    `qualifiers` gives each column's year-month qualifier (``YEAR``,
    ``MONTH``), from the result manifest; ``YEAR TO MONTH`` otherwise.
    pyarrow cannot even hand out a column of that type (a bare KeyError), so
    each batch is re-imported through the C data interface with the column
    typed as the INT32 of months it holds, then printed the way a direct
    read prints it.
    """
    wanted = [i for i, f in enumerate(schema) if str(f.type) == "month_interval"]
    months = pa.schema(
        [f.with_type(pa.int32()) if i in wanted else f for i, f in enumerate(schema)],
        metadata=schema.metadata,
    )
    rebuilt = []
    for batch in batches:
        _, array = batch.__arrow_c_array__()
        rebuilt.append(pa.RecordBatch._import_from_c_capsule(months.__arrow_c_schema__(), array))
    table = pa.Table.from_batches(rebuilt, schema=months)
    for index in wanted:
        field = months.field(index)
        qualifier = qualifiers.get(field.name, "YEAR TO MONTH")
        column = pa.chunked_array(
            [year_month_text(pa, chunk, qualifier) for chunk in table.column(index).chunks],
            type=pa.string(),
        )
        table = table.set_column(index, field.with_type(pa.string()), column)
    return table


_YEAR_MONTH = re.compile(
    r"\s*(?:interval\s*)?'?\s*([+-]?)(\d+)(?:-(\d+))?\s*'?\s*(year\s+to\s+month|year|month)?\s*",
    re.IGNORECASE,
)


def _months(text: str, qualifier: str) -> int:
    match = _YEAR_MONTH.fullmatch(text)
    if match is None:
        raise InvalidArgumentError(
            f"{text!r} is not a year-month interval such as \"INTERVAL '1-2' YEAR TO MONTH\""
        )
    sign, first, second, given = match.groups()
    unit = " ".join((given or qualifier).upper().split())
    if second is not None:
        if given is not None and unit != "YEAR TO MONTH":
            raise InvalidArgumentError(f"{text!r}: a 'years-months' value is YEAR TO MONTH")
        if int(second) > 11:
            # Spark: "month 13 outside range [0, 11]"; 1-13 is not 2-1.
            raise InvalidArgumentError(
                f"{text!r}: the month of a YEAR TO MONTH interval is 0 to 11, not {int(second)}"
            )
        value = int(first) * 12 + int(second)
    else:
        value = int(first) * (12 if unit == "YEAR" else 1)
    if qualifier == "YEAR":
        # An INTERVAL YEAR column holds whole years, as months. Spark casts a
        # YEAR TO MONTH value into one by dropping the months; stored as
        # given, '1-2' read back as '1' YEAR while arithmetic saw 14 months.
        value -= value % 12
    value = -value if sign == "-" else value
    if not -(2**31) <= value < 2**31:
        raise InvalidArgumentError(
            f"{text!r} is out of range for a year-month interval (at most 178956970 years "
            "7 months either way)"
        )
    return value


def _storage_leaf_type(pa: Any, group: str) -> Any:
    def leaf(arrow_type: Any) -> Any:
        if group == DAY_TIME:
            return pa.int64() if pa.types.is_duration(arrow_type) else None
        text = pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type)
        return pa.int32() if text else None

    return leaf


def _storage_leaf(pa: Any, group: str) -> Any:
    def leaf(array: Any) -> Any:
        if group == DAY_TIME:
            if not pa.types.is_duration(array.type):
                return array
            if array.type.unit != "us":
                array = array.cast(pa.duration("us"), safe=True)
            return array.cast(pa.int64())
        if not (pa.types.is_string(array.type) or pa.types.is_large_string(array.type)):
            return array
        return pa.array(
            [None if v is None else _months(v, group) for v in array.to_pylist()], pa.int32()
        )

    return leaf


def storage_columns(pa: Any, table: Any, groups: IntervalPaths) -> Any:
    """`table` with its interval columns as the integers Delta stores.

    The inverse of the read: a duration becomes microseconds (INT64) and
    year-month text months (INT32), which is what the kernel and delta-rs
    write for an interval column -- neither takes a duration.
    """
    for index, field in enumerate(table.schema):
        column = table.column(index)
        for group, paths in groups.items():
            column = _convert(
                pa,
                column,
                _under(paths, field.name),
                (field.name,) in paths,
                _storage_leaf(pa, group),
                _storage_leaf_type(pa, group),
            )
        if column is not table.column(index):
            table = table.set_column(index, field.with_type(column.type), column)
    return table


#: A string literal (skipped), a backquoted identifier, or a bare word.
_SQL_TOKEN = re.compile(
    r"""'(?:[^'\\]|\\.|'')*'|"(?:[^"\\]|\\.)*"|`((?:[^`]|``)+)`|([A-Za-z_][A-Za-z0-9_]*)"""
)


def sql_identifiers(text: str) -> set[str]:
    """Every identifier SQL `text` may name, lower-cased, string literals left out.

    Keywords and function names come along; the caller matches the result
    against column names, so a stray word can only make it cautious.
    """
    out = set()
    for match in _SQL_TOKEN.finditer(text):
        quoted, bare = match.groups()
        if quoted is not None:
            out.add(quoted.replace("``", "`").lower())
        elif bare is not None:
            out.add(bare.lower())
    return out
