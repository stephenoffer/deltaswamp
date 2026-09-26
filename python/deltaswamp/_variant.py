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

__all__ = ["encode", "is_variant_struct", "json_column", "to_json", "variant_column"]

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
    packed = struct.pack("<f", value)
    for digits in range(1, 10):
        text = f"{value:.{digits}g}"
        if struct.pack("<f", float(text)) == packed:
            return _number(float(text))
    return _number(value)


def _number(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):
        # Not JSON numbers; Spark writes them as strings.
        return json.dumps(str(value).replace("inf", "Infinity").replace("nan", "NaN"))
    return repr(value)


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
        if "e" not in item.lower() and isinstance(exponent, int) and len(digits) <= 38:
            scale = max(0, -exponent)
            return _decimal_bytes(int(exact.scaleb(scale)), scale, max(len(digits), scale))
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
        names = sorted(item)
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


def encode(text: str) -> tuple[bytes, bytes]:
    """JSON text as a variant's (metadata, value), typed the way ``parse_json`` does.

    Integers take the narrowest integer type, then DECIMAL(38); a number with a
    fraction is an exact DECIMAL when it fits 38 digits, else a DOUBLE, as is
    any number written with an exponent.
    """
    item = json.loads(text, parse_float=_Float, parse_constant=_Float)
    keys: set[str] = set()
    _names(item, keys)
    ordered = sorted(keys)
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
