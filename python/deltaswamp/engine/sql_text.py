"""Rendering Databricks SQL text: identifiers, literals and types."""

from __future__ import annotations

import datetime as _dt
import numbers
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..catalog import ResolvedTable
from ..errors import InvalidArgumentError, UnreachableTableError
from ..identity import RefKind, split_identifier

_PRIVILEGE = re.compile(r"^[A-Za-z][A-Za-z _]*$")

#: What a SQL type may be made of. An allowlist, not a list of things to
#: forbid: a blocklist has to anticipate every way out of the position, and
#: this one is narrow enough to state positively. It covers everything
#: `arrow_to_sql` produces, nesting included --
#: ``STRUCT<`n`: ARRAY<STRUCT<`x`: DECIMAL(5,2)>>>`` -- while a newline, a
#: semicolon, a quote or a comment marker is simply not in the set.
#: Backtick-quoted field names are checked separately (`_QUOTED_IDENTIFIER`)
#: and may hold any character, so a struct field named ``a-b`` passes.
_SQL_TYPE = re.compile(r"^[A-Za-z0-9_ ,:()<>]+$")
_QUOTED_IDENTIFIER = re.compile(r"`(?:[^`]|``)*`")

#: delta-rs `TableFeatures` members whose wire name is not just camelCase.
_FEATURE_ALIASES = {
    "TimestampWithoutTimezone": "timestampNtz",
    # The protocol spells the preview features with a hyphen.
    "VariantTypePreview": "variantType-preview",
    "TypeWideningPreview": "typeWidening-preview",
}


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
        raise InvalidArgumentError("a column path needs at least one part")
    return ".".join(quote(p) for p in parts)


def columns(columns: str | Sequence[str | Sequence[str]]) -> str:
    # A bare string is one column. Iterating it would quote each character,
    # so zorder(t, "city") became ZORDER BY (`c`, `i`, `t`, `y`).
    if isinstance(columns, str):
        return column(columns)
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
    """Pass a SQL type through, refusing text that could end the statement.

    Backtick-quoted field names (``STRUCT<`a b`: INT>``, which is also what
    `arrow_to_sql` produces for a struct) are checked as balanced quoted
    identifiers and set aside; only the rest must match the allowlist.
    """
    text = str(type_text).strip()
    unquoted = _QUOTED_IDENTIFIER.sub("q", text)
    if not text or not _SQL_TYPE.match(unquoted):
        raise InvalidArgumentError(f"{type_text!r} is not a SQL type")
    return text


def principal(principal: str) -> str:
    return quote(principal)


def privileges(privileges: str | Sequence[str]) -> str:
    items = [privileges] if isinstance(privileges, str) else list(privileges)
    if not items:
        raise InvalidArgumentError("at least one privilege is required")
    for item in items:
        if not _PRIVILEGE.match(item):
            raise InvalidArgumentError(f"{item!r} is not a privilege name")
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


def property_value(key: Any, value: Any) -> str:
    """A TBLPROPERTIES value as the text Delta readers parse.

    `str(True)` is ``'True'``; delta-rs and delta-kernel parse table-property
    booleans case-sensitively, so ``delta.appendOnly = 'True'`` was a table
    they could no longer read the configuration of.
    """
    if value is None:
        raise InvalidArgumentError(f"property {key!r} has the value None; use unset_properties")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def properties(properties: Mapping[str, Any]) -> str:
    unset = sorted(str(k) for k, v in properties.items() if v is None)
    if unset:
        # str(None) would have stored the literal string 'None'.
        raise InvalidArgumentError(
            f"property values cannot be None ({', '.join(unset)}); use unset_properties"
        )
    return ", ".join(
        f"{literal(str(k))} = {literal(property_value(k, v))}" for k, v in properties.items()
    )


def predicate(predicate: str | None) -> str | None:
    """None means "no predicate"; an empty or blank string is refused.

    Truthiness turned ``delete("")`` into an unconditional DELETE of every
    row (and the same for UPDATE), which is the worst way to misread a typo.
    """
    if predicate is None:
        return None
    if not isinstance(predicate, str):
        raise TypeError(f"a predicate must be a SQL string, got {type(predicate).__name__}")
    if not predicate.strip():
        raise InvalidArgumentError(
            "the predicate is empty; pass None to mean every row, or a SQL condition"
        )
    return predicate


def expression(value: Any) -> str:
    """A caller's SQL expression. None is NULL (not the identifier `None`);
    anything else must already be SQL text -- `", ".join` on a non-str value
    was a bare TypeError."""
    if value is None:
        return "NULL"
    return str(value)


def timestamp_text(value: Any) -> str:
    """A time-travel timestamp as text with an explicit offset.

    A naive value means UTC everywhere else in deltaswamp (delta-rs reads it
    so). Sent without an offset, the warehouse read it in the *session* time
    zone instead, which a workspace can set to anything -- so the same call
    travelled to a different version depending on the engine. Epoch
    milliseconds (what `history()` reports) are UTC too.
    """
    if isinstance(value, bool):
        raise TypeError("a timestamp cannot be a bool")
    if isinstance(value, numbers.Real):
        moment = _dt.datetime(1970, 1, 1, tzinfo=_dt.UTC) + _dt.timedelta(milliseconds=float(value))
    elif isinstance(value, _dt.datetime):
        moment = value
    elif isinstance(value, _dt.date):
        moment = _dt.datetime(value.year, value.month, value.day)
    else:
        text = str(value).strip()
        try:
            moment = _dt.datetime.fromisoformat(text)
        except ValueError:
            return text  # let the warehouse say what it makes of it
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_dt.UTC)
    return moment.isoformat()


def type_text(value: Any) -> str:
    """A SQL type from SQL text, an Arrow DataType, or an Arrow Field.

    `str(pa.int64())` is ``'int64'``, which is not a Databricks type.
    """
    if (type(value).__module__ or "").startswith("pyarrow"):
        return arrow_to_sql(getattr(value, "type", value))
    return str(value)


def column_types(fields: Any) -> list[tuple[str, str]]:
    """``{name: sql_type}``, an Arrow schema, or a list of Arrow fields."""
    if isinstance(fields, Mapping):
        # A value may be an Arrow DataType ({"n": pa.int32()}); str() of that
        # is "int32", which is not a Databricks type.
        return [(str(k), type_text(v)) for k, v in fields.items()]
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
    import pyarrow as pa
    import pyarrow.types as t

    if isinstance(arrow_type, pa.BaseExtensionType):  # uuid, json, a tensor ...
        return arrow_to_sql(arrow_type.storage_type)
    if t.is_dictionary(arrow_type):  # e.g. a pandas categorical
        return arrow_to_sql(arrow_type.value_type)
    if t.is_boolean(arrow_type):
        return "BOOLEAN"
    # Databricks has no unsigned integers: widen to the next type that holds
    # every value, rather than refusing a uint column outright.
    if t.is_uint8(arrow_type):
        return "SMALLINT"
    if t.is_uint16(arrow_type):
        return "INT"
    if t.is_uint32(arrow_type):
        return "BIGINT"
    if t.is_uint64(arrow_type):
        return "DECIMAL(20,0)"
    if t.is_float16(arrow_type):
        return "FLOAT"
    if t.is_null(arrow_type):
        return "VOID"
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
        if arrow_type.precision > 38:
            raise UnreachableTableError(
                "add columns via SQL",
                f"{arrow_type} has more than the 38 digits of precision a Databricks DECIMAL holds",
            )
        return f"DECIMAL({arrow_type.precision},{arrow_type.scale})"
    if (
        t.is_string(arrow_type)
        or t.is_large_string(arrow_type)
        or getattr(t, "is_string_view", lambda _t: False)(arrow_type)
    ):
        return "STRING"
    if (
        t.is_binary(arrow_type)
        or t.is_large_binary(arrow_type)
        or t.is_fixed_size_binary(arrow_type)
        or getattr(t, "is_binary_view", lambda _t: False)(arrow_type)
    ):
        return "BINARY"
    if t.is_date(arrow_type):
        return "DATE"
    if t.is_timestamp(arrow_type):
        return "TIMESTAMP" if arrow_type.tz else "TIMESTAMP_NTZ"
    if t.is_list(arrow_type) or t.is_large_list(arrow_type) or t.is_fixed_size_list(arrow_type):
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
