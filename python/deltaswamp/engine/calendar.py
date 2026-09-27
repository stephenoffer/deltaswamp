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
"""

from __future__ import annotations

import datetime as dt
import json
import threading
from collections import OrderedDict
from typing import Any

__all__ = ["has_datetime_columns", "legacy_calendar_files"]

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
