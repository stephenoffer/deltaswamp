"""Rendering Databricks SQL text: identifiers, literals and types."""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..catalog import ResolvedTable
from ..errors import UnreachableTableError
from ..identity import RefKind, split_identifier

_PRIVILEGE = re.compile(r"^[A-Za-z][A-Za-z _]*$")

#: What a SQL type may be made of. An allowlist, not a list of things to
#: forbid: a blocklist has to anticipate every way out of the position, and
#: this one is narrow enough to state positively. It covers everything
#: `arrow_to_sql` produces, nesting included --
#: ``STRUCT<`n`: ARRAY<STRUCT<`x`: DECIMAL(5,2)>>>`` -- while a newline, a
#: semicolon, a quote or a comment marker is simply not in the set.
_SQL_TYPE = re.compile(r"^[A-Za-z0-9_ ,:()<>`]+$")

#: delta-rs `TableFeatures` members whose wire name is not just camelCase.
_FEATURE_ALIASES = {"TimestampWithoutTimezone": "timestampNtz"}


# ------------------------------------------------------------------ quoting


def quote(identifier: str) -> str:
    """Backtick-quote an identifier, doubling any embedded backtick."""
    return "`" + str(identifier).replace("`", "``") + "`"


def column(path: str | Sequence[str]) -> str:
    """Quote a column. A str is one identifier; a list is a nested field path."""
    if isinstance(path, str):
        return quote(path)
    parts = list(path)
    if not parts:
        raise ValueError("a column path needs at least one part")
    return ".".join(quote(p) for p in parts)


def columns(columns: Sequence[str | Sequence[str]]) -> str:
    return ", ".join(column(c) for c in columns)


def qualified(name: str) -> str:
    """Quote a dotted, possibly backtick-quoted, name part by part."""
    return ".".join(quote(p) for p in split_identifier(name))


def name(table: ResolvedTable) -> str:
    ref = table.ref
    if ref.kind is not RefKind.CATALOG:
        raise UnreachableTableError(
            "address the table by name",
            "a SQL warehouse addresses tables by name; this is a path reference",
        )
    return ".".join(quote(p) for p in (ref.catalog, ref.schema, ref.table) if p is not None)


def literal(value: str | None) -> str:
    """A string literal for grammar positions that refuse parameter markers.

    Databricks string literals honor backslash escapes, so both the backslash
    and the quote must be escaped or ``\\'`` would close the literal early.
    """
    if value is None:
        return "NULL"
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def sql_type(type_text: str) -> str:
    """Pass a SQL type through, refusing text that could end the statement."""
    text = str(type_text).strip()
    if not text or not _SQL_TYPE.match(text):
        raise ValueError(f"{type_text!r} is not a SQL type")
    return text


def principal(principal: str) -> str:
    return quote(principal)


def privileges(privileges: str | Sequence[str]) -> str:
    items = [privileges] if isinstance(privileges, str) else list(privileges)
    if not items:
        raise ValueError("at least one privilege is required")
    for item in items:
        if not _PRIVILEGE.match(item):
            raise ValueError(f"{item!r} is not a privilege name")
    return ", ".join(" ".join(i.upper().split()) for i in items)


def feature_name(feature: Any) -> str:
    """The wire name of a feature, from a string or a delta-rs `TableFeatures`."""
    if isinstance(feature, str):
        return feature
    text = str(getattr(feature, "value", feature))
    member = text.rsplit(".", 1)[-1]
    if member in _FEATURE_ALIASES:
        return _FEATURE_ALIASES[member]
    return member[:1].lower() + member[1:]


def timestamp_text(value: Any) -> str:
    if isinstance(value, _dt.datetime | _dt.date):
        return value.isoformat()
    return str(value)


def column_types(fields: Any) -> list[tuple[str, str]]:
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
        out.append((str(name), arrow_to_sql(arrow_type)))
    return out


def arrow_to_sql(arrow_type: Any) -> str:
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
        return f"ARRAY<{arrow_to_sql(arrow_type.value_type)}>"
    if t.is_map(arrow_type):
        return f"MAP<{arrow_to_sql(arrow_type.key_type)}, {arrow_to_sql(arrow_type.item_type)}>"
    if t.is_struct(arrow_type):
        inner = ", ".join(
            f"{quote(arrow_type.field(i).name)}: {arrow_to_sql(arrow_type.field(i).type)}"
            for i in range(arrow_type.num_fields)
        )
        return f"STRUCT<{inner}>"
    raise UnreachableTableError("add columns via SQL", f"no SQL type for Arrow type {arrow_type}")
