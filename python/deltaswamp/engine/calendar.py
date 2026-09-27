"""Finding the data files a rewrite through delta-rs would corrupt.

Spark 2.x, and Spark 3 / Databricks writing with ``datetimeRebaseModeInWrite
= LEGACY`` (Photon does), store DATE and TIMESTAMP values in the hybrid
Julian/Gregorian calendar and say so in the Parquet footer. The kernel's reads
rebase them (``crates/native/src/rebase.rs``); delta-rs does not, and neither
does its INT96 decoding, which holds nanoseconds in an ``i64`` and so
overflows outside 1677-09-21 .. 2262-04-11: ``9999-12-31``, the usual SCD2
"open" sentinel, comes back as ``1816-03-30``. A read through delta-rs is
wrong, so the router sends neither reads nor rewrites of such a table there.
Its DELETE, UPDATE, MERGE, replaceWhere and OPTIMIZE are worse than wrong:
they copy the rows they did not change into new files, without the footer, so
the misread values become the table's data -- ``0001-01-01`` is
``0000-12-30`` from then on, for every reader, statistics included.

Only values before the last calendar switch (1582-10-15) move for a date;
for a timestamp, values before 1900-01-01T00:00:00Z (the bound of Spark's
per-zone rebase tables) and, stored as INT96, values past 2262-04-11, where
nanoseconds overflow. So the check reads a file's footer only when its
statistics cannot rule such a value out. Most tables never have one, and cost
nothing beyond the file listing.

The same values are what delta-rs must not *write*, whatever read them. Spark
reads a Parquet file whose footer names no Spark version in the mode
``spark.sql.parquet.datetimeRebaseModeInRead`` gives, and Databricks SQL
warehouses read those LEGACY: a proleptic ``0001-01-01`` written by delta-rs
(or pyarrow, or any arrow-rs writer) reads there as ``0001-01-03``, and
``1500-06-15`` as ``1500-06-05``, while every other reader, deltaswamp's
included, sees the value written. The kernel's writer names a Spark version
(``crates/native/src/writer.rs``); delta-rs 1.6.5 has no way to put a key in
the footer, so the router keeps such values -- in the rows being written
(`holds_early_datetimes`) or in the files a rewrite copies
(`early_datetime_files`) -- off it.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
from collections import OrderedDict
from typing import Any

__all__ = [
    "early_datetime_files",
    "has_datetime_columns",
    "holds_early_datetimes",
    "legacy_calendar_files",
]

#: Dates before this are rebased by Spark's legacy calendar.
_DATE_LIMIT = dt.date(1582, 10, 15)
#: Timestamps before this may be, depending on the writer's time zone.
_TIMESTAMP_LIMIT = dt.datetime(1900, 1, 1, tzinfo=dt.UTC)
#: Timestamps after this overflow an INT96 decoded as nanoseconds (the true
#: bound is 2262-04-11T23:47:16.854775807Z; statistics are truncated to
#: milliseconds, so the day before is the safe side of it).
_INT96_LIMIT = dt.datetime(2262, 4, 10, tzinfo=dt.UTC)

#: Results per (table root, version); a version's files never change.
_CACHE: OrderedDict[tuple[str, int], tuple[str, ...]] = OrderedDict()
_EARLY_CACHE: OrderedDict[tuple[str, int], tuple[str, ...]] = OrderedDict()
_CACHE_SIZE = 64
_CACHE_LOCK = threading.Lock()


def _is_datetime(dtype: Any) -> bool:
    # TIMESTAMP_NTZ is left out: Spark writes it without rebasing.
    return dtype in ("date", "timestamp")


def _contains_datetime(dtype: Any) -> bool:
    if isinstance(dtype, str):
        return _is_datetime(dtype)
    if not isinstance(dtype, dict):
        return False
    kind = dtype.get("type")
    if kind == "struct":
        return any(_contains_datetime(f.get("type")) for f in dtype.get("fields") or ())
    if kind == "array":
        return _contains_datetime(dtype.get("elementType"))
    if kind == "map":
        return _contains_datetime(dtype.get("keyType")) or _contains_datetime(
            dtype.get("valueType")
        )
    return False


def has_datetime_columns(schema: Any) -> bool:
    """Whether a Delta schema (as JSON) has a DATE or TIMESTAMP anywhere in it."""
    if not isinstance(schema, dict):
        return True  # unknown: assume it might
    return any(_contains_datetime(f.get("type")) for f in schema.get("fields") or ())


def _leaves(
    fields: Any, prefix: tuple[str, ...], mapped: bool
) -> tuple[list[tuple[tuple[str, ...], str]], bool]:
    """The DATE/TIMESTAMP leaves statistics can describe, and whether any cannot.

    Statistics are keyed by physical name under column mapping, and exist for
    struct fields but never inside an array or a map.
    """
    leaves: list[tuple[tuple[str, ...], str]] = []
    unstatted = False
    for f in fields or ():
        if not isinstance(f, dict):
            continue
        name = str(f.get("name"))
        if mapped:
            name = str((f.get("metadata") or {}).get("delta.columnMapping.physicalName") or name)
        dtype = f.get("type")
        path = (*prefix, name)
        if isinstance(dtype, str):
            if _is_datetime(dtype):
                leaves.append((path, dtype))
        elif isinstance(dtype, dict) and dtype.get("type") == "struct":
            inner, blind = _leaves(dtype.get("fields"), path, mapped)
            leaves += inner
            unstatted |= blind
        elif _contains_datetime(dtype):
            unstatted = True
    return leaves, unstatted


def _get(stats: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        if not isinstance(stats, dict):
            return None
        stats = stats.get(part)
    return stats


def _parse_stat(value: Any, kind: str) -> dt.date | dt.datetime | None:
    """A statistics bound, or None if it is missing or cannot be read."""
    if not isinstance(value, str):
        return None
    try:
        if kind == "date":
            return dt.date.fromisoformat(value)
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None  # a year Python cannot hold, or a form it cannot parse
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=dt.UTC)


def _before_limit(value: Any, kind: str) -> bool:
    """Whether a statistics minimum may be a value a legacy rebase moves."""
    parsed = _parse_stat(value, kind)
    if parsed is None:
        return True
    if kind == "date":
        return parsed < _DATE_LIMIT
    return parsed < _TIMESTAMP_LIMIT


def _after_limit(value: Any, kind: str) -> bool:
    """Whether a statistics maximum may be a timestamp INT96 decoding overflows.

    Whether the file stores the column as INT96 only its footer says, so any
    file that may hold such a value is a suspect: Photon writes INT96 by
    default, and a year-9999 sentinel is common.
    """
    if kind == "date":
        return False  # days since the epoch fit an i32 for every year Delta allows
    parsed = _parse_stat(value, kind)
    return parsed is None or parsed >= _INT96_LIMIT


def _suspects(
    files: Any, leaves: list[tuple[tuple[str, ...], str]], unstatted: bool
) -> list[tuple[str, int]]:
    """The files whose statistics do not rule out a value outside the limits."""
    import pyarrow as pa

    out: list[tuple[str, int]] = []
    for row in pa.table(files).select(["path", "size", "stats"]).to_pylist():
        path, size = row["path"], int(row.get("size") or 0)
        if unstatted:
            out.append((path, size))
            continue
        try:
            stats = json.loads(row.get("stats") or "null")
        except ValueError:
            stats = None
        if not isinstance(stats, dict):
            out.append((path, size))
            continue
        records = stats.get("numRecords")
        for leaf, kind in leaves:
            if records is not None and _get(stats.get("nullCount"), leaf) == records:
                continue  # every value null
            if _before_limit(_get(stats.get("minValues"), leaf), kind) or _after_limit(
                _get(stats.get("maxValues"), leaf), kind
            ):
                out.append((path, size))
                break
    return out


def legacy_calendar_files(snapshot: Any) -> tuple[str, ...]:
    """Live files of `snapshot` that delta-rs would misread and rewrite shifted.

    `snapshot` is a native kernel snapshot. Files are named as the log names
    them. Cached per table version.
    """
    key = (str(snapshot.table_root), int(snapshot.version))
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]

    metadata = json.loads(snapshot.metadata_json())
    schema = metadata.get("schemaString") or metadata.get("schema_string")
    if isinstance(schema, str):
        schema = json.loads(schema)
    found: tuple[str, ...] = ()
    if has_datetime_columns(schema):
        configuration = metadata.get("configuration") or {}
        mapped = str(configuration.get("delta.columnMapping.mode", "none")).lower() != "none"
        partitions = {str(c) for c in metadata.get("partitionColumns") or ()}
        fields = [
            f
            for f in (schema or {}).get("fields") or ()
            if isinstance(f, dict) and f.get("name") not in partitions
        ]
        leaves, unstatted = _leaves(fields, (), mapped)
        if leaves or unstatted:
            suspects = _suspects(snapshot.files(), leaves, unstatted)
            if suspects:
                found = tuple(snapshot.legacy_calendar_files(suspects))

    with _CACHE_LOCK:
        _CACHE[key] = found
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return found


def _early(files: Any, leaves: list[tuple[tuple[str, ...], str]], unstatted: bool) -> list[str]:
    """The files whose statistics do not rule out a value a legacy rebase moves."""
    import pyarrow as pa

    out: list[str] = []
    for row in pa.table(files).select(["path", "stats"]).to_pylist():
        path = row["path"]
        try:
            stats = json.loads(row.get("stats") or "null") if not unstatted else None
        except ValueError:
            stats = None
        if not isinstance(stats, dict):
            out.append(path)
            continue
        records = stats.get("numRecords")
        for leaf, kind in leaves:
            if records is not None and _get(stats.get("nullCount"), leaf) == records:
                continue
            if _before_limit(_get(stats.get("minValues"), leaf), kind):
                out.append(path)
                break
    return out


def early_datetime_files(snapshot: Any) -> tuple[str, ...]:
    """Live files of `snapshot` that may hold a DATE or TIMESTAMP a legacy rebase moves.

    Dates before 1582-10-15 and timestamps before 1900, whatever wrote them:
    a rewrite through delta-rs copies them into files without Spark's
    writer metadata, which Databricks then reads shifted (see the module
    docstring). By statistics alone; a file they cannot clear counts.
    Cached per table version.
    """
    key = (str(snapshot.table_root), int(snapshot.version))
    with _CACHE_LOCK:
        if key in _EARLY_CACHE:
            _EARLY_CACHE.move_to_end(key)
            return _EARLY_CACHE[key]

    metadata = json.loads(snapshot.metadata_json())
    schema = metadata.get("schemaString") or metadata.get("schema_string")
    if isinstance(schema, str):
        schema = json.loads(schema)
    found: tuple[str, ...] = ()
    if has_datetime_columns(schema):
        configuration = metadata.get("configuration") or {}
        mapped = str(configuration.get("delta.columnMapping.mode", "none")).lower() != "none"
        partitions = {str(c) for c in metadata.get("partitionColumns") or ()}
        # A partition value is in the log, not in the file, so its column
        # never reaches a footer.
        fields = [
            f
            for f in (schema or {}).get("fields") or ()
            if isinstance(f, dict) and f.get("name") not in partitions
        ]
        leaves, unstatted = _leaves(fields, (), mapped)
        if leaves or unstatted:
            found = tuple(_early(snapshot.files(), leaves, unstatted))

    with _CACHE_LOCK:
        _EARLY_CACHE[key] = found
        while len(_EARLY_CACHE) > _CACHE_SIZE:
            _EARLY_CACHE.popitem(last=False)
    return found


#: Epoch offsets of the limits, per Arrow unit.
_DATE_LIMIT_DAYS = -141427  # 1582-10-15
_TIMESTAMP_LIMIT_SECONDS = -2_208_988_800  # 1900-01-01T00:00:00Z
_PER_SECOND = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}


def _array_holds_early(array: Any) -> bool:
    import pyarrow as pa
    import pyarrow.compute as pc

    kind = array.type
    if isinstance(array, pa.ChunkedArray):
        return any(_array_holds_early(chunk) for chunk in array.chunks)
    if pa.types.is_dictionary(kind):
        return _array_holds_early(array.dictionary)
    if pa.types.is_struct(kind):
        return any(_array_holds_early(array.field(i)) for i in range(kind.num_fields))
    if pa.types.is_map(kind):
        return _array_holds_early(array.keys) or _array_holds_early(array.items)
    if (
        pa.types.is_list(kind)
        or pa.types.is_large_list(kind)
        or pa.types.is_fixed_size_list(kind)
        or getattr(pa.types, "is_list_view", lambda _: False)(kind)
        or getattr(pa.types, "is_large_list_view", lambda _: False)(kind)
    ):
        return _array_holds_early(array.flatten())
    if pa.types.is_date32(kind):
        limit = _DATE_LIMIT_DAYS
        raw = array.cast(pa.int32())
    elif pa.types.is_date64(kind):
        limit = _DATE_LIMIT_DAYS * 86_400_000
        raw = array.cast(pa.int64())
    elif pa.types.is_timestamp(kind):
        # Naive timestamps too: one written to a TIMESTAMP column is stored
        # zoned. (A TIMESTAMP_NTZ one is not rebased, so this errs safe.)
        limit = _TIMESTAMP_LIMIT_SECONDS * _PER_SECOND[kind.unit]
        raw = array.cast(pa.int64())
    else:
        return False
    if len(raw) == raw.null_count:
        return False
    low = pc.min(raw).as_py()
    return low is not None and low < limit


def holds_early_datetimes(data: Any) -> bool:
    """Whether in-memory `data` holds a date before 1582-10-15 or a timestamp before 1900.

    Those are the values Databricks reads shifted from a file whose footer
    names no Spark version (see the module docstring). A pyarrow Table or
    RecordBatch, or a pandas or polars DataFrame, is inspected; a stream
    cannot be without consuming it, and is taken not to.
    """
    try:
        import pyarrow as pa
    except ImportError:
        return False
    module = type(data).__module__ or ""
    if module.startswith(("pandas", "polars")) and type(data).__name__ == "DataFrame":
        try:
            data = pa.table(data)
        except Exception:
            return False
    if not isinstance(data, (pa.Table, pa.RecordBatch)):
        return False
    try:
        return any(_array_holds_early(column) for column in data.columns)
    except (pa.ArrowException, TypeError, ValueError):
        return True  # could not tell: the safe side is the writer that says so
