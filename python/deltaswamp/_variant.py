"""VARIANT values as JSON text, the one representation every engine can give.

A VARIANT column reaches Python in two shapes. The kernel reads an unshredded
file as its binary encoding, ``struct<metadata: binary, value: binary>``; the
SQL warehouse sends the value as JSON text, and cannot send anything else. A
table read through both would have handed back two types for one column, and
writing the warehouse's text back stored every object as a string scalar.

So a read returns VARIANT columns as JSON text (Databricks' own rendering,
``to_json(v)``) on every engine, and a write takes JSON text: here it is
encoded into the binary form for the direct engines, and the warehouse runs it
through ``parse_json``. The binary encoding is the Parquet Variant spec
(parquet-format ``VariantEncoding.md``); shredded files are not decoded here.
"""

from __future__ import annotations

import base64
import datetime as _dt
import decimal
import json
import struct
import uuid
from typing import Any

__all__ = [
    "binary_columns",
    "datafusion_literal",
    "encode",
    "is_variant_struct",
    "json_column",
    "json_text_stream",
    "log_variant_columns",
    "string_variant",
    "text_schema",
    "to_json",
    "variant_column",
    "variant_paths",
]

_EPOCH = _dt.datetime(1970, 1, 1)
_EPOCH_DATE = _dt.date(1970, 1, 1)


def _int(data: bytes, start: int, size: int, *, signed: bool = False) -> int:
    return int.from_bytes(data[start : start + size], "little", signed=signed)


def _keys(metadata: bytes) -> list[str]:
    header = metadata[0]
    if header & 0x0F != 1:
        raise ValueError(f"unsupported variant metadata version {header & 0x0F}")
    size = ((header >> 6) & 0x03) + 1
    count = _int(metadata, 1, size)
    offsets = [_int(metadata, 1 + size * (i + 1), size) for i in range(count + 1)]
    base = 1 + size * (count + 2)
    return [metadata[base + offsets[i] : base + offsets[i + 1]].decode() for i in range(count)]


def _float32(value: float) -> str:
    """The shortest text that reads back as the same float32, as Spark prints it."""
    if value != value or value in (float("inf"), float("-inf")):
        return _number(value)
    packed = struct.pack("<f", value)
    for digits in range(1, 10):
        text = f"{value:.{digits}g}"
        if struct.pack("<f", float(text)) == packed:
            return _java_number(text)
    return _number(value)


def _number(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):
        # Not JSON numbers; Spark writes them as strings.
        return json.dumps(str(value).replace("inf", "Infinity").replace("nan", "NaN"))
    return _java_number(repr(value))


def _java_number(shortest: str) -> str:
    """A float's shortest round-trip digits, laid out as Java's toString does.

    Databricks' `to_json` prints a DOUBLE the Java way: plain notation with at
    least one fractional digit from 10^-3 up to 10^7 (`1.5`, `100.0`), else
    `1.0E20` / `1.0E-5`. Python's repr gave `1e+20` and `1e-05`, so the same
    table read back different text through the kernel and the warehouse.
    """
    sign, digits, exponent = decimal.Decimal(shortest).as_tuple()
    assert isinstance(exponent, int)
    lead = "-" if sign else ""
    text = "".join(map(str, digits)).lstrip("0")
    if not text:
        return lead + "0.0"
    # The value is 0.<text> x 10^point.
    point = len(text) + exponent
    text = text.rstrip("0")
    if -3 < point <= 7:
        if point <= 0:
            return f"{lead}0.{'0' * -point}{text}"
        whole, frac = text[:point].ljust(point, "0"), text[point:]
        return f"{lead}{whole}.{frac or '0'}"
    mantissa = text[0] + "." + (text[1:] or "0")
    return f"{lead}{mantissa}E{point - 1}"


def _decimal(unscaled: int, scale: int) -> str:
    # Databricks prints DECIMAL(10, 2) 1.50 as 1.5: trailing zeros go.
    value = decimal.Decimal(unscaled).scaleb(-scale).normalize()
    return format(value, "f")


def _micros(value: int, *, zone: bool) -> str:
    try:
        moment = _EPOCH + _dt.timedelta(microseconds=value)
    except OverflowError:
        return json.dumps(str(value))
    text = moment.isoformat(sep=" ")
    if "." in text:
        # Databricks prints the fraction without trailing zeros: .5, not .500000.
        text = text.rstrip("0").rstrip(".")
    return json.dumps(text + "+00:00" if zone else text)


def _render(value: bytes, pos: int, keys: list[str]) -> str:
    header = value[pos]
    basic, head = header & 0x03, header >> 2
    if basic == 1:  # short string
        return json.dumps(value[pos + 1 : pos + 1 + head].decode(), ensure_ascii=False)
    if basic in (2, 3):
        offset_size = (head & 0x03) + 1
        if basic == 2:
            id_size = ((head >> 2) & 0x03) + 1
            large = (head >> 4) & 0x01
        else:
            id_size = 0
            large = (head >> 2) & 0x01
        count_size = 4 if large else 1
        count = _int(value, pos + 1, count_size)
        ids_at = pos + 1 + count_size
        offsets_at = ids_at + count * id_size
        values_at = offsets_at + (count + 1) * offset_size
        offsets = [_int(value, offsets_at + i * offset_size, offset_size) for i in range(count)]
        items = [_render(value, values_at + o, keys) for o in offsets]
        if basic == 3:
            return "[" + ",".join(items) + "]"
        names = [keys[_int(value, ids_at + i * id_size, id_size)] for i in range(count)]
        return (
            "{"
            + ",".join(
                f"{json.dumps(n, ensure_ascii=False)}:{v}"
                for n, v in zip(names, items, strict=True)
            )
            + "}"
        )
    body = pos + 1
    if head == 0:
        return "null"
    if head in (1, 2):
        return "true" if head == 1 else "false"
    if head in (3, 4, 5, 6):
        return str(_int(value, body, 1 << (head - 3), signed=True))
    if head == 7:
        return _number(struct.unpack_from("<d", value, body)[0])
    if head in (8, 9, 10):
        size = {8: 4, 9: 8, 10: 16}[head]
        return _decimal(_int(value, body + 1, size, signed=True), value[body])
    if head == 11:
        days = _int(value, body, 4, signed=True)
        try:
            return json.dumps((_EPOCH_DATE + _dt.timedelta(days=days)).isoformat())
        except OverflowError:
            return json.dumps(str(days))
    if head in (12, 13):
        return _micros(_int(value, body, 8, signed=True), zone=head == 12)
    if head in (18, 19):
        nanos = _int(value, body, 8, signed=True)
        return _micros(nanos // 1000, zone=head == 18)
    if head == 14:
        return _float32(struct.unpack_from("<f", value, body)[0])
    if head in (15, 16):
        size = _int(value, body, 4)
        raw = value[body + 4 : body + 4 + size]
        text = base64.b64encode(raw).decode() if head == 15 else raw.decode()
        return json.dumps(text, ensure_ascii=False)
    if head == 17:
        micros = _int(value, body, 8, signed=True)
        return json.dumps((_EPOCH + _dt.timedelta(microseconds=micros)).time().isoformat())
    if head == 20:
        return json.dumps(str(uuid.UUID(bytes=bytes(value[body : body + 16]))))
    raise ValueError(f"unknown variant primitive type {head}")


def to_json(metadata: bytes, value: bytes) -> str:
    """One variant value, as the JSON text Databricks' ``to_json`` gives."""
    return _render(value, 0, _keys(metadata))


# ---------------------------------------------------------------- encoding


class _Float(str):
    """A JSON number with a fraction or exponent, kept as its text."""


def _size(n: int) -> int:
    return 1 if n < 1 << 8 else 2 if n < 1 << 16 else 3 if n < 1 << 24 else 4


def _decimal_bytes(unscaled: int, scale: int, digits: int) -> bytes:
    head, size = (8, 4) if digits <= 9 else (9, 8) if digits <= 18 else (10, 16)
    return bytes([head << 2, scale]) + unscaled.to_bytes(size, "little", signed=True)


def _encode(item: Any, ids: dict[str, int]) -> bytes:
    if item is None:
        return b"\x00"
    if item is True or item is False:
        return bytes([(1 if item else 2) << 2])
    if isinstance(item, _Float):
        exact = decimal.Decimal(item)
        _, digits, exponent = exact.as_tuple()
        scale = max(0, -exponent) if isinstance(exponent, int) else 0
        # DECIMAL(precision, scale) needs both within 38: `0.<37 zeros>12`
        # has 2 digits but scale 39, which Databricks types as a DOUBLE.
        precision = max(len(digits), scale)
        if "e" not in item.lower() and isinstance(exponent, int) and precision <= 38:
            return _decimal_bytes(int(exact.scaleb(scale)), scale, precision)
        return bytes([7 << 2]) + struct.pack("<d", float(item))
    if isinstance(item, int):
        for head, size in ((3, 1), (4, 2), (5, 4), (6, 8)):
            if -(1 << (8 * size - 1)) <= item < 1 << (8 * size - 1):
                return bytes([head << 2]) + item.to_bytes(size, "little", signed=True)
        if len(str(abs(item))) <= 38:
            return _decimal_bytes(item, 0, len(str(abs(item))))
        return bytes([7 << 2]) + struct.pack("<d", float(item))
    if isinstance(item, str):
        raw = item.encode()
        if len(raw) <= 63:
            return bytes([(len(raw) << 2) | 1]) + raw
        return bytes([16 << 2]) + len(raw).to_bytes(4, "little") + raw
    if isinstance(item, list):
        parts = [_encode(x, ids) for x in item]
        return _container(3, parts, None)
    if isinstance(item, dict):
        names = sorted(item, key=_java_order)
        parts = [_encode(item[n], ids) for n in names]
        return _container(2, parts, [ids[n] for n in names])
    raise ValueError(f"cannot encode {type(item).__name__} as a variant")


def _container(basic: int, parts: list[bytes], field_ids: list[int] | None) -> bytes:
    offsets = [0]
    for part in parts:
        offsets.append(offsets[-1] + len(part))
    offset_size = _size(offsets[-1])
    large = len(parts) > 255
    if field_ids is None:
        head = (offset_size - 1) | (large << 2)
        ids = b""
    else:
        id_size = _size(max(field_ids, default=0))
        head = (offset_size - 1) | ((id_size - 1) << 2) | (large << 4)
        ids = b"".join(i.to_bytes(id_size, "little") for i in field_ids)
    count = len(parts).to_bytes(4 if large else 1, "little")
    table = b"".join(o.to_bytes(offset_size, "little") for o in offsets)
    return bytes([basic | (head << 2)]) + count + ids + table + b"".join(parts)


def _names(item: Any, into: set[str]) -> None:
    if isinstance(item, dict):
        into.update(item)
        for v in item.values():
            _names(v, into)
    elif isinstance(item, list):
        for v in item:
            _names(v, into)


def _java_order(key: str) -> bytes:
    """Sort key giving Java's String order (by UTF-16 code unit), as Spark sorts.

    Python orders by code point, which puts U+FF01 before U+1F600; in UTF-16
    the emoji's surrogates come first. Spark looks object fields up by binary
    search in its own order, so an object sorted the Python way could miss keys.
    """
    return key.encode("utf-16-be")


def _no_constant(name: str) -> Any:
    # Python's json reads NaN and Infinity; they are not JSON, and
    # Databricks' parse_json refuses them (MALFORMED_RECORD_IN_PARSING).
    raise ValueError(f"{name} is not a JSON value")


def encode(text: str) -> tuple[bytes, bytes]:
    """JSON text as a variant's (metadata, value), typed the way ``parse_json`` does.

    Integers take the narrowest integer type, then DECIMAL(38); a number with a
    fraction is an exact DECIMAL when its precision and scale fit 38, else a
    DOUBLE, as is any number written with an exponent.
    """
    item = json.loads(text, parse_float=_Float, parse_constant=_no_constant)
    keys: set[str] = set()
    _names(item, keys)
    ordered = sorted(keys, key=_java_order)
    ids = {k: i for i, k in enumerate(ordered)}
    raw = [k.encode() for k in ordered]
    offsets = [0]
    for k in raw:
        offsets.append(offsets[-1] + len(k))
    size = _size(max(offsets[-1], len(raw)))
    # Version 1, sorted_strings set.
    header = bytes([1 | 0x10 | ((size - 1) << 6)])
    metadata = (
        header
        + len(raw).to_bytes(size, "little")
        + b"".join(o.to_bytes(size, "little") for o in offsets)
        + b"".join(raw)
    )
    return metadata, _encode(item, ids)


# ------------------------------------------------------------------ arrow


def is_variant_struct(pa: Any, arrow_type: Any) -> bool:
    """Whether an Arrow type is the variant binary encoding."""
    if not pa.types.is_struct(arrow_type) or arrow_type.num_fields != 2:
        return False
    names = {arrow_type.field(i).name: arrow_type.field(i).type for i in range(2)}
    return set(names) == {"metadata", "value"} and all(
        pa.types.is_binary(t) or pa.types.is_large_binary(t) for t in names.values()
    )


def json_column(pa: Any, column: Any) -> Any:
    """A variant binary column (array or chunked array) as JSON text."""
    values = column.to_pylist()
    return pa.array(
        [None if v is None else to_json(v["metadata"], v["value"]) for v in values], pa.string()
    )


#: The Arrow type the direct engines read and write VARIANT as.
def variant_type(pa: Any) -> Any:
    return pa.struct(
        [pa.field("metadata", pa.binary(), nullable=False), pa.field("value", pa.binary(), False)]
    )


def variant_column(pa: Any, column: Any) -> Any:
    """JSON text (array or chunked array) as the variant binary encoding."""
    texts = column.to_pylist()
    metadata: list[bytes] = []
    value: list[bytes] = []
    mask: list[bool] = []
    for text in texts:
        if text is None:
            metadata.append(b"")
            value.append(b"")
            mask.append(True)
            continue
        try:
            m, v = encode(text)
        except ValueError as exc:
            from .errors import InvalidArgumentError

            raise InvalidArgumentError(
                f"a VARIANT column takes JSON text, and {text[:80]!r} is not JSON ({exc})"
            ) from None
        metadata.append(m)
        value.append(v)
        mask.append(False)
    return pa.StructArray.from_arrays(
        [pa.array(metadata, pa.binary()), pa.array(value, pa.binary())],
        fields=list(variant_type(pa)),
        mask=pa.array(mask, pa.bool_()),
    )


# ------------------------------------------------------------- by schema

Paths = frozenset[tuple[str, ...]]

#: Path steps into an array's elements and a map's values. A Delta field name
#: cannot hold a NUL, so neither can be mistaken for a struct field.
ELEMENT = "\x00element"
VALUE = "\x00value"


def variant_paths(schema: Any) -> Paths:
    """The VARIANT columns of a Delta schema (the log's `schemaString`, parsed).

    Top-level columns and VARIANTs nested in structs, arrays and maps, each
    as its path: field names, with `ELEMENT` for an array's elements and
    `VALUE` for a map's values. The Arrow schema cannot tell: the kernel
    reads a VARIANT as a bare ``struct<metadata: binary, value: binary>``,
    and a real struct of that shape looked the same -- and was then decoded
    as a variant, and failed. A VARIANT in an array or a map was left binary
    while the warehouse sent it as text, so its type depended on the engine.
    """
    out: set[tuple[str, ...]] = set()

    def walk_type(kind: Any, path: tuple[str, ...]) -> None:
        if kind == "variant":
            out.add(path)
        elif isinstance(kind, dict):
            if kind.get("type") == "struct":
                walk(kind.get("fields"), path)
            elif kind.get("type") == "array":
                walk_type(kind.get("elementType"), (*path, ELEMENT))
            elif kind.get("type") == "map":
                walk_type(kind.get("valueType"), (*path, VALUE))

    def walk(fields: Any, prefix: tuple[str, ...]) -> None:
        for f in fields or []:
            if isinstance(f, dict):
                walk_type(f.get("type"), (*prefix, str(f.get("name"))))

    if isinstance(schema, dict):
        walk(schema.get("fields"), ())
    return frozenset(out)


def _under(paths: Paths, name: str) -> Paths:
    return frozenset(p[1:] for p in paths if p and p[0] == name and len(p) > 1)


def _is_list(pa: Any, arrow_type: Any) -> bool:
    return bool(pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type))


def _convert_type(pa: Any, arrow_type: Any, paths: Paths, here: bool, leaf: Any) -> Any:
    """`arrow_type` with each VARIANT at `paths` (or here) given `leaf`'s type.

    `leaf(type)` is the new type of a VARIANT position, or None to leave it.
    """
    if here:
        new = leaf(arrow_type)
        return arrow_type if new is None else new
    if not paths:
        return arrow_type
    if pa.types.is_struct(arrow_type):
        return pa.struct(
            [
                f.with_type(
                    _convert_type(pa, f.type, _under(paths, f.name), (f.name,) in paths, leaf)
                )
                for f in (arrow_type.field(i) for i in range(arrow_type.num_fields))
            ]
        )
    if _is_list(pa, arrow_type):
        field = arrow_type.value_field
        inner = _convert_type(pa, field.type, _under(paths, ELEMENT), (ELEMENT,) in paths, leaf)
        make = pa.large_list if pa.types.is_large_list(arrow_type) else pa.list_
        return make(field.with_type(inner))
    if pa.types.is_map(arrow_type):
        field = arrow_type.item_field
        inner = _convert_type(pa, field.type, _under(paths, VALUE), (VALUE,) in paths, leaf)
        return pa.map_(arrow_type.key_field, field.with_type(inner), arrow_type.keys_sorted)
    return arrow_type


def _convert(pa: Any, array: Any, paths: Paths, here: bool, leaf: Any, leaf_type: Any) -> Any:
    """`array` with each VARIANT at `paths` (or here) rewritten by `leaf`.

    `leaf(array)` returns the new array of a VARIANT position, or the array
    itself to leave it; `leaf_type` is the matching type rule, for
    `_convert_type`.
    """
    if here:
        return leaf(array)
    if not paths:
        return array
    kind = array.type
    if not (pa.types.is_struct(kind) or _is_list(pa, kind) or pa.types.is_map(kind)):
        return array
    if isinstance(array, pa.ChunkedArray):
        chunks = [_convert(pa, c, paths, False, leaf, leaf_type) for c in array.chunks]
        return pa.chunked_array(chunks, type=_convert_type(pa, kind, paths, False, leaf_type))
    mask = array.is_null() if array.null_count else None
    if pa.types.is_struct(kind):
        fields = [kind.field(i) for i in range(kind.num_fields)]
        # `flatten` carries the struct's own nulls into its children: under a
        # null row a VARIANT child holds placeholder bytes (its children are
        # non-nullable), which cannot be decoded.
        flat = array.flatten() if mask is not None else [array.field(i) for i in range(len(fields))]
        children = [
            _convert(pa, flat[i], _under(paths, f.name), (f.name,) in paths, leaf, leaf_type)
            for i, f in enumerate(fields)
        ]
        return pa.StructArray.from_arrays(
            children,
            fields=[f.with_type(c.type) for f, c in zip(fields, children, strict=True)],
            mask=mask,
        )
    if _is_list(pa, kind):
        values = _convert(
            pa, array.values, _under(paths, ELEMENT), (ELEMENT,) in paths, leaf, leaf_type
        )
        if values is array.values:
            return array
        new_type = _convert_type(pa, kind, paths, False, leaf_type)
        cls = pa.LargeListArray if pa.types.is_large_list(kind) else pa.ListArray
        return cls.from_arrays(array.offsets, values, type=new_type, mask=mask)
    items = _convert(pa, array.items, _under(paths, VALUE), (VALUE,) in paths, leaf, leaf_type)
    if items is array.items:
        return array
    new_type = _convert_type(pa, kind, paths, False, leaf_type)
    return pa.MapArray.from_arrays(array.offsets, array.keys, items, type=new_type, mask=mask)


def _text_leaf_type(pa: Any) -> Any:
    return lambda t: pa.string() if is_variant_struct(pa, t) else None


def _text_leaf(pa: Any) -> Any:
    return lambda a: json_column(pa, a) if is_variant_struct(pa, a.type) else a


def text_schema(pa: Any, schema: Any, paths: Paths) -> Any:
    """`schema` with each VARIANT (at `paths`) as a string, the type reads give it."""
    leaf = _text_leaf_type(pa)
    return pa.schema(
        [
            f.with_type(_convert_type(pa, f.type, _under(paths, f.name), (f.name,) in paths, leaf))
            for f in schema
        ],
        metadata=schema.metadata,
    )


def to_text(pa: Any, array: Any, paths: Paths, here: bool) -> Any:
    """`array` with the VARIANT binary at `paths` (or here) decoded to JSON text."""
    return _convert(pa, array, paths, here, _text_leaf(pa), _text_leaf_type(pa))


def json_text_stream(stream: Any, paths: Paths | None) -> Any:
    """A direct engine's stream with the VARIANT columns at `paths` as JSON text.

    With `paths` None (the log's schema was not to hand), every top-level
    column of the variant shape is taken as one.
    """
    import pyarrow as pa

    reader = (
        stream.to_reader()
        if isinstance(stream, pa.Table)
        else stream
        if isinstance(stream, pa.RecordBatchReader)
        else pa.RecordBatchReader.from_stream(stream)
    )
    schema = reader.schema
    if paths is None:
        paths = frozenset((f.name,) for f in schema if is_variant_struct(pa, f.type))
    target = text_schema(pa, schema, paths)
    if target == schema:
        return reader

    def batches() -> Any:
        for batch in reader:
            arrays = [
                to_text(pa, batch.column(i), _under(paths, f.name), (f.name,) in paths)
                for i, f in enumerate(schema)
            ]
            yield pa.RecordBatch.from_arrays(arrays, schema=target)

    return pa.RecordBatchReader.from_batches(target, batches())


def _is_text(pa: Any, arrow_type: Any) -> bool:
    t = pa.types
    return bool(t.is_string(arrow_type) or t.is_large_string(arrow_type) or t.is_null(arrow_type))


def _to_binary(pa: Any, array: Any, paths: Paths, here: bool) -> Any:
    def leaf(a: Any) -> Any:
        return variant_column(pa, a.cast(pa.string())) if _is_text(pa, a.type) else a

    def leaf_type(t: Any) -> Any:
        return variant_type(pa) if _is_text(pa, t) else None

    return _convert(pa, array, paths, here, leaf, leaf_type)


def binary_columns(pa: Any, data: Any, paths: Paths) -> Any:
    """A pyarrow Table with JSON text at the VARIANT `paths` encoded for a direct engine.

    Columns are matched case-insensitively, as a write matches them; data
    already in the binary shape passes through.
    """
    if not paths:
        return data
    by_lower = {p[0].lower(): p[0] for p in paths}
    for index, field in enumerate(data.schema):
        name = by_lower.get(field.name.lower())
        if name is None:
            continue
        column = _to_binary(pa, data.column(index), _under(paths, name), (name,) in paths)
        if column is not data.column(index):
            data = data.set_column(index, pa.field(field.name, column.type), column)
    return data


def log_variant_columns(metadata_json: Any) -> frozenset[str]:
    """Lower-cased top-level VARIANT columns, from a snapshot's metaData JSON."""
    try:
        metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        schema = json.loads(metadata.get("schemaString") or "{}")
    except (TypeError, ValueError, AttributeError):
        return frozenset()
    return frozenset(p[0].lower() for p in variant_paths(schema) if len(p) == 1)


def string_variant(text: str) -> str:
    """The JSON text of a VARIANT holding the string `text`.

    Spark stores a STRING assigned to a VARIANT column as a variant string
    (`SET v = '{"a":1}'` holds the text, not an object); `parse_json` is how
    SQL makes an object.
    """
    return json.dumps(text, ensure_ascii=False)


def datafusion_literal(column_sql: str, text: str) -> str:
    """DataFusion SQL for a VARIANT value (JSON `text`), typed as `column_sql`'s column.

    DataFusion builds a struct literal with nullable fields and cannot cast it
    to the column's non-null ones ("Unsupported CAST"); a CASE with the column
    itself in a branch that never applies gives the literal the column's type.
    """
    metadata, value = encode(text)
    literal = f"named_struct('metadata', X'{metadata.hex()}', 'value', X'{value.hex()}')"
    return (
        f"CASE WHEN ({column_sql} IS NULL) AND ({column_sql} IS NOT NULL) THEN {column_sql} "
        f"ELSE {literal} END"
    )
