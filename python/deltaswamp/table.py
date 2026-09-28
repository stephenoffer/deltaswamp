"""`Table`: one table, every operation, routed per operation.

Protocol features are discovered in two stages. The catalog supplies enough to
make the decisions that must precede opening the log (is this a view, a shallow
clone, a table vending refuses). The full reader/writer feature lists exist only
in the log, so the first operation enriches the resolved table with them and
everything afterward routes on the complete picture.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import re
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from . import _results
from ._request import METHOD_OPERATIONS, NO_DATA, Request, derive, refusal
from ._request import strict_engine as _strict
from ._util import check_keywords, not_table_data, timestamp_ms
from .capability import FEATURE_SUPPORT, Capability, FeatureKind, Operation, feature_from_wire
from .capability import READ_OPERATIONS as _READ_OPERATIONS
from .capability import Engine as EngineKind
from .catalog import ResolvedTable, TableType
from .credentials import Operation as CredentialOperation
from .engine.base import (
    TranslatingStream,
    merge_clause,
    merge_clause_values,
    translating_stream,
)
from .engine.boundary import engine_cause
from .engine.deltars import DeltaRsEngine
from .engine.intervals import interval_paths, interval_schema, interval_stream, storage_columns
from .engine.kernel import KernelEngine
from .engine.metadata import cdf_clash_error, cdf_name_clash
from .errors import (
    SQL_FALLBACK_REMEDY,
    CorruptTableError,
    CredentialError,
    DeltaSwampError,
    EngineFallbackWarning,
    EngineLimitError,
    FallbackRequiredError,
    InvalidArgumentError,
    InvalidReferenceError,
    UnreachableTableError,
)
from .identity import RefKind, parse_ref

if TYPE_CHECKING:
    import datetime

    from .connection import Connection
    from .governance import ColumnLineage, Grant, Lineage, TableInfo

__all__ = ["Table"]


def _write_data(data: Any) -> Any:
    """Plain Python data (a column dict, a list of row dicts) as an Arrow table.

    The engines take Arrow-exporting objects and pandas only; a dict failed in
    delta-rs with "Expected object with __arrow_c_array__", after write_table
    had already created the table from its inferred schema.
    """
    if data is None:
        raise InvalidArgumentError("no data given to write")
    refusal = not_table_data(data)
    if refusal is not None:
        raise InvalidArgumentError(refusal)
    if isinstance(data, dict):
        pa = _require("pyarrow", "pyarrow")
        try:
            return pa.table(data)
        except (TypeError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError) as exc:
            # {"id": 1} (a row, not columns) raised a bare TypeError.
            raise InvalidArgumentError(
                f"a dict of data maps each column to its values, e.g. {{'id': [1, 2]}} ({exc})"
            ) from exc
    if isinstance(data, list) and data and all(isinstance(row, dict) for row in data):
        pa = _require("pyarrow", "pyarrow")
        try:
            return pa.Table.from_pylist(data)
        except (TypeError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError) as exc:
            raise InvalidArgumentError(f"the rows do not form a table ({exc})") from exc
    return data


def _engine_broke(exc: BaseException) -> bool:
    """Whether a read's failure is the engine's, so the next engine may serve it.

    The boundary's EngineError (a failure no rule recognised), or a storage
    failure it translated, which another engine's client may not share. Not
    the request's own mistake (InvalidArgumentError and the like): another
    engine would refuse it too, and the warning would bury the real message.
    """
    from .errors import EngineError, StorageError

    return isinstance(exc, EngineError) or (
        isinstance(exc, StorageError) and engine_cause(exc) is not exc
    )


#: Writes re-run after a concurrent schema or metadata change beat them.
_REALIGN_ATTEMPTS = 5


def _lost_to_metadata_change(exc: BaseException, data: Any) -> bool:
    """A delta-rs append conflict with a concurrent metadata commit, re-writable."""
    from .errors import CommitConflictError

    return (
        isinstance(exc, CommitConflictError)
        and "changed since last commit" in str(exc)
        and not _consumable(data)
    )


def _consumable(data: Any) -> bool:
    """Whether writing `data` consumes it (a stream), so it cannot be written twice."""
    return (
        hasattr(data, "read_next_batch")
        or hasattr(data, "__next__")
        or not hasattr(data, "__len__")
    )


def _plain_views(table: Any) -> Any:
    """Cast Arrow view types (string_view, binary_view) to their plain forms.

    delta-rs returns views, and several pyarrow kernels (sort, take) have no
    implementation for them yet.
    """
    import pyarrow as pa

    fields = []
    for field in table.schema:
        if pa.types.is_string_view(field.type):
            field = field.with_type(pa.string())
        elif pa.types.is_binary_view(field.type):
            field = field.with_type(pa.binary())
        fields.append(field)
    target = pa.schema(fields, metadata=table.schema.metadata)
    return table if target.equals(table.schema) else table.cast(target)


def _check_version(version: Any, what: str = "version") -> None:
    """A table version is a non-negative int; anything else fails in Rust later."""
    if version is None:
        return
    if isinstance(version, bool) or not isinstance(version, int):
        raise InvalidArgumentError(f"{what} must be an int, not {type(version).__name__}")
    if version < 0:
        raise InvalidArgumentError(f"{what} must be >= 0, got {version}")


def _check_count(value: Any, what: str) -> None:
    """A row or entry count: a non-negative int, or None for no bound."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentError(f"{what} must be an int, not {type(value).__name__}")
    if value < 0:
        raise InvalidArgumentError(f"{what} must be >= 0, got {value}")


def _columns_arg(columns: Any) -> list[str] | None:
    """Normalise a projection: one name becomes a list, an empty list is refused.

    An empty list aborts the whole process in the kernel (a non-unwinding Rust
    panic building a zero-field batch) and silently means ``SELECT *`` on the
    warehouse, so neither engine may see one.
    """
    if columns is None:
        return None
    if isinstance(columns, str):
        return [columns]
    names = list(columns)
    if not names:
        raise InvalidArgumentError(
            "columns=[] selects nothing; pass None for every column, or name at least one"
        )
    for name in names:
        if not isinstance(name, str):
            raise InvalidArgumentError(f"column names must be strings, got {name!r}")
    seen: dict[str, str] = {}
    for name in names:
        if name.lower() not in seen:
            seen[name.lower()] = name
            continue
        # Column names are case-insensitive, so `["id", "ID"]` names one column
        # twice. The kernel returned it once and the warehouse twice; an Arrow
        # table cannot hold both under one name anyway.
        first = seen[name.lower()]
        also = "twice" if first == name else f"and {name!r}, which are the same column"
        raise InvalidArgumentError(
            f"columns= names {first!r} {also} (names are case-insensitive); list each column once"
        )
    return names


def _column_path(column: Any, what: str) -> str:
    """The column an ALTER names, as one string: dotted for a nested field.

    A list is a nested path (``["s", "a"]``), as the SQL fallback always took
    it; drop_column refused one and the kernel failed on it. A part holding
    a dot is backtick-quoted so it stays one name.
    """
    if isinstance(column, str) and column:
        return column
    if (
        isinstance(column, (list, tuple))
        and column
        and all(isinstance(part, str) and part for part in column)
    ):
        return ".".join(
            "`" + part.replace("`", "``") + "`" if "." in part or "`" in part else part
            for part in column
        )
    raise InvalidArgumentError(
        f"{what} takes a column name or a nested path such as ['s', 'a'], not {column!r}"
    )


def _top_level(column: Any, names: Sequence[str]) -> str | None:
    """The top-level column a str names outright: exact spelling first, then any case."""
    if not isinstance(column, str):
        return None
    if column in names:
        return column
    folded = [n for n in names if n.lower() == column.lower()]
    return folded[0] if len(folded) == 1 else None


def _column_default(field: Any) -> str | None:
    """A column's DEFAULT expression, as Databricks records it in the field metadata."""
    raw = (field.metadata or {}).get(b"CURRENT_DEFAULT")
    return raw.decode() if raw is not None else None


def _default_column(pa: Any, field: Any, rows: int) -> Any:
    """The column's literal DEFAULT repeated `rows` times, or None if it is not one.

    Only literals whose value does not depend on the session are evaluated:
    strings, numbers, booleans, NULL and dates. A TIMESTAMP literal is read in
    the session time zone, and a function call is Databricks' to evaluate.
    """
    from .predicate import Literal, PredicateError, parse_value

    text = _column_default(field)
    if text is None:
        return None
    try:
        value = parse_value(text)
    except PredicateError:
        return None
    if not isinstance(value, Literal) or value.type not in (
        "string",
        "long",
        "decimal",
        "boolean",
        "null",
        "date",
    ):
        return None
    try:
        return pa.array([value.value] * rows).cast(field.type, safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError):
        return None


#: The features that let a table hold VARIANT columns.
_VARIANT_FEATURES: frozenset[str] = frozenset({"variantType", "variantType-preview"})

#: The engines that read a change feed across a schema change only by
#: splitting it there (`Table._feed_segments`).
_DIRECT_FEED_ENGINES: frozenset[EngineKind] = frozenset({EngineKind.KERNEL, EngineKind.DELTARS})


def _log_schema(engine: Any, table: Any, version: int | None) -> Any:
    """The table's schema as the log records it (`schemaString`, parsed), or None.

    Only the log says which columns are VARIANT: the direct engines read one
    as a plain ``struct<metadata, value>``.
    """
    try:
        if isinstance(engine, KernelEngine):
            metadata = engine.snapshot(table, version=version).metadata_json()
            return json.loads(json.loads(metadata)["schemaString"])
        if isinstance(engine, DeltaRsEngine):
            return json.loads(engine._open(table, version=version).schema().to_json())
    except Exception:
        return None
    return None


def _shredded_variant_error(exc: Exception) -> Exception | None:
    """The kernel's failure on a shredded VARIANT file, as a refusal naming the way out."""
    if "shredded" not in str(exc).lower():
        return None
    return EngineLimitError(
        "read a VARIANT column",
        "the file holds shredded VARIANT values, which neither direct engine decodes "
        f"({type(exc).__name__}: {str(exc).splitlines()[0][:160]})",
        "read the other columns (columns=[...]), or ds.connect(..., allow_sql_fallback=True) "
        "to read through a SQL warehouse",
    )


def _dotted_keys(targets: Any) -> Any:
    """UPDATE targets with a tuple path (``("s", "a")``) spelled as ``"s.a"``.

    A nested field was settable only as the dotted string; the tuple -- how
    `columns=` and the alter methods take a path -- was refused as "not a
    string". A part holding a dot or a backtick has no unambiguous dotted
    spelling, so it is refused.
    """
    if not isinstance(targets, Mapping) or not any(isinstance(k, tuple) for k in targets):
        return targets
    out: dict[Any, Any] = {}
    for key, value in targets.items():
        if isinstance(key, tuple):
            if not key or not all(isinstance(p, str) and p for p in key):
                raise InvalidArgumentError(f"an update path is a tuple of field names, got {key!r}")
            odd = [p for p in key if "." in p or "`" in p]
            if odd and len(key) > 1:
                raise InvalidArgumentError(
                    f"update path {key!r}: a nested field name holding '.' or '`' ({odd[0]!r}) "
                    "cannot be set by path"
                )
            key = ".".join(key) if len(key) > 1 else key[0]
        out[key] = value
    return out


def _data_column_names(data: Any) -> list[str]:
    """The column names of table data, where known without consuming it."""
    schema = getattr(data, "schema", None)
    names = getattr(schema, "names", None)
    if isinstance(names, list):
        return [str(n) for n in names]
    columns = getattr(data, "columns", None)
    module = type(data).__module__ or ""
    if module.startswith(("pandas", "polars")) and columns is not None:
        return [str(c) for c in columns]
    if isinstance(data, list) and data and isinstance(data[0], Mapping):
        return [str(k) for k in data[0]]
    return []


def _is_plain_primitive(arrow_type: Any) -> bool:
    """Not a struct, list or map: a type no collation can hide inside."""
    import pyarrow as pa

    t = pa.types
    return not (
        t.is_struct(arrow_type)
        or t.is_list(arrow_type)
        or t.is_large_list(arrow_type)
        or t.is_map(arrow_type)
        or getattr(t, "is_list_view", lambda _: False)(arrow_type)
        or t.is_fixed_size_list(arrow_type)
    )


def _check_predicate(predicate: Any, what: str) -> None:
    """Refuse a blank predicate rather than let an engine read it as 'every row'.

    The warehouse builds ``WHERE`` only for a truthy predicate, so ``""`` on a
    DELETE deleted the whole table, and on an overwrite it selected a full
    OVERWRITE instead of replaceWhere.
    """
    if predicate is None:
        return
    if not isinstance(predicate, str):
        raise InvalidArgumentError(
            f"{what}: predicate must be a SQL string, not {type(predicate).__name__}"
        )
    if not predicate.strip():
        raise InvalidArgumentError(f"{what}: the predicate is blank; pass None to mean every row")


def _check_sql_fragment(text: Any, what: str) -> None:
    """Refuse SQL text that is not exactly one expression, for every engine.

    One check (`dialect.check_expression`) for each fragment the API forwards:
    predicates, SET/INSERT values, MERGE ON and clause conditions,
    replaceWhere, CHECK constraints. Raised before routing, so no engine --
    delta-rs reading a prefix, DuckDB or the warehouse splicing it into a
    statement -- sees text that would mean something other than one
    expression.
    """
    from .engine.dialect import check_expression

    check_expression(text, what)


def _parses_as_spark(text: Any) -> bool:
    """Whether `text` is absent, or one expression of Spark's grammar in `dialect`."""
    from .engine.dialect import parses

    return not isinstance(text, str) or not text.strip() or parses(text)


def _check_merge_clause(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """`_check_sql_fragment` for a MERGE clause's condition and values."""
    for value in (*args, *kwargs.values()):
        if isinstance(value, str):
            _check_sql_fragment(value, "the MERGE clause condition")
        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, str):
                    _check_sql_fragment(item, "the MERGE value")


def _check_write_sizes(target_file_size: Any, max_commit_retries: Any) -> None:
    """Refuse negative sizes here: delta-rs raised a bare OverflowError for them."""
    _check_count(target_file_size, "target_file_size")
    _check_count(max_commit_retries, "max_commit_retries")


def _check_txn(txn: Any) -> None:
    """`txn` is ``(app_id, version)``: a non-empty string and a non-negative int."""
    if txn is None:
        return
    if not isinstance(txn, (tuple, list)) or len(txn) != 2:
        raise InvalidArgumentError(f"txn must be an (app_id, version) pair, not {txn!r}")
    app_id, version = txn
    if not isinstance(app_id, str) or not app_id:
        raise InvalidArgumentError(f"txn app_id must be a non-empty string, not {app_id!r}")
    _check_version(version, "txn version")


def _check_travel(version: Any, timestamp: Any) -> None:
    if version is not None and timestamp is not None:
        raise InvalidArgumentError("time travel takes a version or a timestamp, not both")
    _check_version(version)


def _timestamp_arg(value: Any, what: str = "timestamp") -> Any:
    """A time-travel timestamp, with epoch milliseconds read as UTC.

    `history()` reports commit times as epoch-millisecond ints. Handing one
    back worked for a scan on the kernel but raised TypeError on the change
    data feed, so an int is turned into the datetime it names for every engine.
    """
    import datetime as _dt

    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidArgumentError(f"{what} must be a timestamp, not bool")
    if isinstance(value, (int, float)):
        if value < 0:
            raise InvalidArgumentError(f"{what} must not be before 1970, got {value}")
        epoch = _dt.datetime(1970, 1, 1, tzinfo=_dt.UTC)
        return epoch + _dt.timedelta(milliseconds=value)
    if isinstance(value, str) and not value.strip():
        raise InvalidArgumentError(f"{what} is blank")
    if isinstance(value, str):
        # Malformed is the caller's mistake: the engines refused it as "cannot
        # time travel", a routing refusal. timestamp_ms raises
        # InvalidArgumentError for text that is not a time.
        timestamp_ms(value)
    elif not isinstance(value, _dt.date):
        raise InvalidArgumentError(
            f"{what} must be a datetime, a date, an ISO-8601 string or epoch milliseconds, "
            f"not {type(value).__name__}"
        )
    return value


def _names(what: str, value: Any) -> list[str]:
    """One name or a list of names, each a string.

    ``unset_properties(1)`` and ``cluster_by(1)`` raised a bare TypeError
    ("'int' object is not iterable") from deep inside an engine.
    """
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise InvalidArgumentError(
            f"{what} takes a name or a list of names, not {type(value).__name__}"
        )
    names = list(value)
    for name in names:
        if not isinstance(name, str):
            raise InvalidArgumentError(f"{what} names must be strings, got {name!r}")
    return names


def _check_comment(comment: Any) -> None:
    """A comment is text or None; the kernel stored 123 as '123', delta-rs raised."""
    if comment is not None and not isinstance(comment, str):
        raise InvalidArgumentError(
            f"a comment must be a string or None, not {type(comment).__name__}"
        )


def _looks_missing(error: str) -> bool:
    """Whether an open error says the table is not there at all."""
    text = error.lower()
    return any(
        marker in text
        for marker in (
            "tablenotfound",
            "not a delta table",
            "no such file or directory",
            "does not exist",
            "no files in log segment",
        )
    )


def _store_assignable(pa: Any, given: Any, wanted: Any) -> bool:
    """Whether a column of type `given` may be written into one of type `wanted`.

    What Delta's schema enforcement accepts without mergeSchema: the same
    type, or a widening that keeps every value (int to a wider int or to a
    double, float to double, a decimal to one holding all its digits). delta-rs
    cast the rest safely-but-silently -- 4.7 became 4 in a BIGINT column, '12'
    became 12 -- where Spark refuses the write. Nested types are left to the
    engine, which checks them field by field.
    """
    t = pa.types
    if given == wanted or t.is_null(given):
        return True
    if t.is_dictionary(given):
        return _store_assignable(pa, given.value_type, wanted)
    if t.is_nested(given) or t.is_nested(wanted):
        return True
    strings = (t.is_string, t.is_large_string, t.is_string_view)
    binaries = (t.is_binary, t.is_large_binary, t.is_binary_view)
    for family in (strings, binaries):
        if any(f(given) for f in family) and any(f(wanted) for f in family):
            return True
    if t.is_integer(given):
        if t.is_signed_integer(wanted):
            # Delta has no unsigned types: an unsigned column is cast safely
            # where it is aligned, which refuses a value out of range.
            return bool(given.bit_width <= wanted.bit_width)
        if t.is_float64(wanted):
            return bool(given.bit_width <= 32)
        if t.is_decimal(wanted):
            digits = {8: 3, 16: 5, 32: 10, 64: 19}[given.bit_width]
            return bool(wanted.precision - wanted.scale >= digits)
        return False
    if t.is_floating(given):
        return bool(t.is_floating(wanted) and given.bit_width <= wanted.bit_width)
    if t.is_decimal(given):
        return bool(
            t.is_decimal(wanted)
            and wanted.scale >= given.scale
            and wanted.precision - wanted.scale >= given.precision - given.scale
        )
    if t.is_timestamp(given):
        # A unit or zone difference is how Arrow spells the same instant (a
        # nanosecond is truncated to Delta's microsecond, as Spark does).
        return bool(t.is_timestamp(wanted))
    if t.is_date(given):
        # DATE widens to TIMESTAMP_NTZ (type widening), a date as its midnight;
        # not to a zoned TIMESTAMP, whose midnight depends on the session zone.
        return bool(t.is_date(wanted) or (t.is_timestamp(wanted) and wanted.tz is None))
    return False


def _refuse_lossy_types(
    pa: Any, schema: Any, by_name: dict[str, Any], canonical: Any, data: Any = None
) -> Any:
    """Raise InvalidArgumentError for a column Delta would not write as the table's type.

    A narrower numeric type (BIGINT data into an INT column, DOUBLE into
    FLOAT) is what Python ints and floats arrive as, and the warehouse's own
    INSERT takes it when every value fits. So with the rows to hand (`data`,
    an Arrow table with `schema`) such a column is cast when the cast keeps
    every value exactly, and only a value that would change is refused; the
    table is returned with those casts made. A stream cannot be checked
    without consuming it, so there the type decides.
    """
    bad = []
    for index, field in enumerate(schema):
        wanted = by_name.get(canonical(field.name))
        if wanted is None or _store_assignable(pa, field.type, wanted.type):
            continue
        if data is not None and _numeric(pa, field.type) and _numeric(pa, wanted.type):
            cast = _exact_cast(pa, data.column(index), wanted.type)
            if cast is not None:
                data = data.set_column(
                    index, pa.field(field.name, wanted.type, field.nullable), cast
                )
                continue
            bad.append(f"{wanted.name} ({field.type} into {wanted.type}: a value does not fit)")
            continue
        bad.append(f"{wanted.name} ({field.type} into {wanted.type})")
    if bad:
        raise InvalidArgumentError(
            "the data does not fit the table's column types: "
            f"{', '.join(bad)}. Delta refuses a write that would change a value or "
            "reinterpret a type; cast the data to the table's types first, or widen the "
            "column with alter_column_type()"
        )
    return data


def _numeric(pa: Any, arrow_type: Any) -> bool:
    t = pa.types
    return bool(t.is_integer(arrow_type) or t.is_floating(arrow_type) or t.is_decimal(arrow_type))


def _exact_cast(pa: Any, column: Any, wanted: Any) -> Any:
    """`column` cast to `wanted` if that keeps every value exactly, else None."""
    import pyarrow.compute as pc

    try:
        cast = column.cast(wanted, safe=True)
        back = cast.cast(column.type, safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, ValueError):
        return None
    same = pc.equal(back, column)
    if pa.types.is_floating(column.type):
        # NaN is NaN in either width, and equal() says it is not.
        same = pc.or_(same, pc.and_(pc.is_nan(back), pc.is_nan(column)))
    return cast if pc.all(same.fill_null(True)).as_py() is not False else None


def _fill_stream(pa: Any, reader: Any, target: Any, canonical: Any, partitions: set[str]) -> Any:
    """A stream with the nullable columns it leaves out added as nulls, lazily.

    The kernel fills them and delta-rs refused the same stream ("number of
    fields does not match"); in-memory data was already filled for both.
    Batches are extended as they are read, so nothing is materialised here.
    """
    names = [canonical(n) for n in reader.schema.names]
    if len(set(names)) != len(names):
        return reader
    computed = {
        f.name
        for f in target
        if f.metadata
        and any(
            key.startswith((b"delta.generationExpression", b"delta.identity."))
            for key in f.metadata
        )
    }
    missing = [f for f in target if f.name not in names and f.name not in computed]
    if not missing or any(not f.nullable or f.name in partitions for f in missing):
        return reader
    fields = [
        pa.field(n, f.type, f.nullable, f.metadata)
        for n, f in zip(names, reader.schema, strict=True)
    ]
    schema = pa.schema(fields + missing, metadata=reader.schema.metadata)

    def batches() -> Any:
        for batch in reader:
            columns = list(batch.columns) + [pa.nulls(batch.num_rows, f.type) for f in missing]
            yield pa.RecordBatch.from_arrays(columns, schema=schema)

    return pa.RecordBatchReader.from_batches(schema, batches())


def _has_map(pa: Any, wanted: Any) -> bool:
    """Whether a type contains a map anywhere."""
    if pa.types.is_map(wanted):
        return True
    if pa.types.is_struct(wanted):
        return any(_has_map(pa, wanted.field(i).type) for i in range(wanted.num_fields))
    if pa.types.is_list(wanted) or pa.types.is_large_list(wanted):
        return _has_map(pa, wanted.value_type)
    return False


def _maps_from_lists(pa: Any, array: Any, wanted: Any) -> Any:
    """Rebuild map columns that arrived as lists of key/value structs.

    Polars has no map type: a Delta map reads back as
    ``LargeList<Struct<key, value>>``, and delta-rs refuses to cast that to the
    table's map ("Cannot cast field m from LargeList(Struct(...)) to Map"), so
    a frame read from a table could not be appended to it. Nested maps (inside
    structs and lists) are rebuilt too; anything else is returned unchanged.
    """
    kind = array.type
    listy = pa.types.is_list(kind) or pa.types.is_large_list(kind)
    if pa.types.is_map(wanted) and listy:
        entry = kind.value_type
        if not (pa.types.is_struct(entry) and entry.num_fields == 2):
            return array
        if array.offset:
            array = pa.concat_arrays([array])  # zero the offset so offsets line up
        entries = array.values
        keys = _maps_from_lists(pa, entries.field(0), wanted.key_type).cast(wanted.key_type)
        items = _maps_from_lists(pa, entries.field(1), wanted.item_type).cast(wanted.item_type)
        offsets = array.offsets.cast(pa.int32())
        return pa.MapArray.from_arrays(
            offsets, keys, items, type=wanted, mask=array.is_null() if array.null_count else None
        )
    if pa.types.is_struct(wanted) and pa.types.is_struct(kind):
        names = [kind.field(i).name for i in range(kind.num_fields)]
        by_name = {wanted.field(i).name: wanted.field(i) for i in range(wanted.num_fields)}
        children = [array.field(i) for i in range(kind.num_fields)]
        rebuilt = [
            _maps_from_lists(pa, child, by_name[name].type) if name in by_name else child
            for name, child in zip(names, children, strict=True)
        ]
        if all(new is old for new, old in zip(rebuilt, children, strict=True)):
            return array
        fields = [
            pa.field(name, new.type, kind.field(i).nullable)
            for i, (name, new) in enumerate(zip(names, rebuilt, strict=True))
        ]
        return pa.StructArray.from_arrays(
            rebuilt, fields=fields, mask=array.is_null() if array.null_count else None
        )
    if listy and (pa.types.is_list(wanted) or pa.types.is_large_list(wanted)):
        if array.offset:
            array = pa.concat_arrays([array])
        values = _maps_from_lists(pa, array.values, wanted.value_type)
        if values is array.values:
            return array
        cls = pa.LargeListArray if pa.types.is_large_list(kind) else pa.ListArray
        return cls.from_arrays(
            array.offsets, values, mask=array.is_null() if array.null_count else None
        )
    return array


def _given(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop None-valued keyword arguments, so engines see only what was asked."""
    return {k: v for k, v in kwargs.items() if v is not None}


def _require(module: str, extra: str, what: str = "this conversion") -> Any:
    """Import an optional dependency, or say which extra provides it.

    Without this the failure surfaces as a ModuleNotFoundError raised from deep
    inside pyarrow, which does not tell you what to install.
    """

    try:
        # Through __import__ so an import hook (or a test blocking one) sees
        # the request even when the module is already loaded.
        __import__(module)
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} is needed for {what} but is not installed. "
            f"Install it with: pip install 'deltaswamp[{extra}]'"
        ) from exc


def _protocol_from_properties(resolved: ResolvedTable) -> dict[str, Any]:
    """The protocol as the catalog records it, for a log no engine could read.

    Unity Catalog mirrors `delta.minReaderVersion`, `delta.minWriterVersion` and
    one `delta.feature.<name>` per feature into the table's properties. That is
    enough to name the feature that makes the log unreadable -- which beats
    reporting an empty feature set.
    """
    if resolved.reader_features or resolved.writer_features:
        return {}
    props = resolved.properties
    names = {
        key.removeprefix("delta.feature.")
        for key, value in props.items()
        if key.startswith("delta.feature.") and value.lower() in ("supported", "enabled")
    }
    if not names:
        return {}

    def version(key: str) -> int | None:
        try:
            return int(props[key])
        except (KeyError, ValueError):
            return None

    reader_version = version("delta.minReaderVersion")
    readers = set()
    if reader_version is not None and reader_version >= 3:
        for name in names:
            feature = feature_from_wire(name)
            # Unknown names are kept as reader features: refusing too much is
            # recoverable, claiming a read that fails is not.
            if feature is None or FEATURE_SUPPORT[feature].kind is not FeatureKind.WRITER:
                readers.add(name)
    return {
        "min_reader_version": reader_version,
        "min_writer_version": version("delta.minWriterVersion"),
        "reader_features": frozenset(readers),
        "writer_features": frozenset(names),
    }


#: How often a read through a catalog handle re-checks that its name still
#: names the same table (seconds).
_NAME_RECHECK_SECONDS = 1.0


class Table:
    """One table: reads, writes, DDL and maintenance, routed per operation."""

    def __init__(
        self, connection: Connection, resolved: ResolvedTable, *, version: int | None = None
    ) -> None:
        _check_version(version)
        self._connection = connection
        self._resolved = resolved
        self._version = version
        self._enriched = False
        # What the catalog said, kept apart from what the log says: the log is
        # the truth for table properties, and re-enrichment after an ALTER must
        # not let a stale earlier read win.
        self._catalog_properties = dict(resolved.properties)

    # --------------------------------------------------------------- identity

    def __repr__(self) -> str:
        # A handle pinned to a past version reads (and refuses writes) as of
        # that version; printed without it, it looked like the live table.
        pinned = f", version={self._version}" if self._version is not None else ""
        return f"Table({self._resolved.ref}, location={self._resolved.location!r}{pinned})"

    @property
    def resolved(self) -> ResolvedTable:
        """The catalog's resolution of this table: location, type, protocol and credentials."""
        return self._resolved

    @property
    def location(self) -> str | None:
        """The table's storage root, or None where the catalog exposes none (a view, a share)."""
        return self._resolved.location

    @property
    def table_type(self) -> str | None:
        """The catalog's table type: ``"MANAGED"``, ``"EXTERNAL"``, ``"VIEW"``, ...

        None for a table opened by path.
        """
        t = self._resolved.table_type
        return t.value if t else None

    @property
    def securable_kind(self) -> str | None:
        """Unity Catalog's securable kind for the table, when the catalog reports one."""
        return self._resolved.securable_kind

    @property
    def is_catalog_managed(self) -> bool:
        """True when the catalog, not the log, ratifies commits (the ``catalogManaged`` feature)."""
        return self._resolved.is_catalog_managed

    # ------------------------------------------------------------- enrichment

    def _check_identity(
        self, metadata_id: str | None, properties: dict[str, str] | None = None
    ) -> None:
        """Refuse a table whose log identity does not match the catalog's.

        A table dropped and re-created under the same name keeps the name and
        gets a new id. Reading on with a cached id means reading a different
        table while believing it is the same one.

        A Unity Catalog managed table records the catalog's id in its
        ``io.unitycatalog.tableId`` property -- what the UC committer itself
        validates -- and its Metadata.id need not be the same UUID (a writer
        other than this library picks its own). Matching Metadata.id alone
        refused such a table as "dropped and re-created" on every open.
        """
        expected = self._resolved.table_uuid
        # Only a managed table's log carries the catalog's id; the catalog gives
        # a registered external table an id of its own.
        if self._resolved.table_type not in (TableType.MANAGED, None):
            return
        recorded = (properties or {}).get("io.unitycatalog.tableId")
        if expected and recorded and expected == recorded:
            return
        if expected and metadata_id and expected != metadata_id:
            raise CorruptTableError(
                f"{self._resolved.ref} resolves to a table whose log id is "
                f"{metadata_id!r}, but the catalog reports {expected!r}. The table was "
                "most likely dropped and re-created; re-open it to pick up the new one."
            )

    def _invalidate(self) -> None:
        """Forget cached protocol state after a commit.

        Enrichment caches the feature lists and properties read from the log.
        Any write or ALTER changes them, so a Table that kept the cache would
        keep reporting what was true before the call it just made.

        A catalog-managed table also carries the commit tail and ratified
        version captured at resolution. Left alone, the handle kept reading
        the version before its own write, and its next write re-committed that
        same version (a 409 from the catalog, every second append); txn=
        dedup also read the stale snapshot and let a replay through.
        """
        self._enriched = False
        # An enrichment already in flight on another thread read the log
        # before this commit; the generation stops it marking its stale
        # answer as current.
        self._generation = getattr(self, "_generation", 0) + 1
        if self._version is None and (
            self._resolved.is_catalog_managed or self._resolved.max_catalog_version is not None
        ):
            self._refresh_commit_tail()

    def _forget_snapshots(self) -> None:
        """Drop the kernel snapshots cached for this table and the enrichment.

        A snapshot cached before the log was damaged refreshes without
        re-reading `_last_checkpoint`, so routing kept answering from it
        (can("append") said delta-rs) while every engine failed the call.
        Resolved afresh, the log's damage reaches can() as it reaches the call.
        """
        kernel = self._connection.router.engines.get(EngineKind.KERNEL)
        forget = getattr(kernel, "forget", None)
        if forget is not None and self._resolved.location is not None:
            forget(self._resolved.location)
        self._invalidate()

    def _check_still_named(self) -> None:
        """Refuse a read through a handle whose catalog name no longer names its table.

        A write already re-resolves the name (and fails TableNotFoundError
        once the table is dropped), but a read went on with the location and
        credentials resolved at open: after DROP TABLE it returned the dropped
        table's rows, and after a re-create under the same name, still the old
        table's. A handle opened at a version reads that snapshot as asked.
        """
        if self._version is not None or self._resolved.ref.kind is not RefKind.CATALOG:
            return
        # One catalog round trip (~90 ms on Databricks) per read made a loop
        # of small reads slow; a drop is noticed within this window instead.
        # Writes check every time (_refresh_commit_tail(before_write=True)).
        now = time.monotonic()
        if now - getattr(self, "_named_checked_at", float("-inf")) < _NAME_RECHECK_SECONDS:
            return
        self._refresh_commit_tail(before_write=True)
        self._named_checked_at = now

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # A time.monotonic() reading, which means nothing on another host: a
        # handle shipped to a Ray worker whose clock started later skipped
        # the catalog re-check until the worker's uptime passed the driver's.
        state.pop("_named_checked_at", None)
        return state

    def _refresh_commit_tail(self, *, before_write: bool = False) -> None:
        """Re-read a catalog-managed table's ratified commits from the catalog.

        `before_write` makes a table that is gone, or that was dropped and
        re-created under this name, an error now -- before any data file is
        written -- rather than a raw 404/409 from the commit afterwards, with
        the files already orphaned in storage.
        """
        ref = self._resolved.ref
        if ref.kind is not RefKind.CATALOG:
            return
        try:
            fresh = self._connection._catalog_for(ref).resolve(ref)
        except InvalidReferenceError:
            if before_write:
                raise
            return
        except DeltaSwampError as exc:
            if getattr(exc, "denied", False) or isinstance(exc, CredentialError):
                # A revoked privilege or rejected credentials: reading on at
                # the location resolved earlier would serve a principal the
                # catalog now refuses.
                raise
            # The call that got us here succeeded; a re-open reads the tail.
            return
        recreated = any(
            old and new and old != new
            for old, new in (
                (self._resolved.table_uuid, fresh.table_uuid),
                (self._resolved.table_id, fresh.table_id),
            )
        )
        if recreated:
            # Keep this handle on the table it named, so its identity check
            # (and the catalog's uuid assertion) refuse rather than write on.
            if before_write:
                raise CorruptTableError(
                    f"{ref} was dropped and re-created since this handle opened it (the "
                    f"catalog's table id is now {fresh.table_id or fresh.table_uuid!r}); "
                    "re-open it with conn.table(...) to use the new table"
                )
            return
        self._resolved = dataclasses.replace(
            self._resolved,
            log_tail=fresh.log_tail,
            max_catalog_version=fresh.max_catalog_version,
            etag=fresh.etag or self._resolved.etag,
        )

    def _enrich(self) -> ResolvedTable:
        """Fill in the protocol feature lists by reading the log once.

        Does not go through the router: routing depends on these
        features, so asking the router first would be circular. We try kernel,
        then delta-rs, and fall back to catalog metadata alone if neither can
        open the table -- in which case routing proceeds on partial information
        and the engines themselves produce the refusal.
        """
        if self._enriched or self._resolved.location is None:
            return self._resolved

        last_error: str | None = None
        generation = getattr(self, "_generation", 0)
        for kind in (EngineKind.KERNEL, EngineKind.DELTARS):
            engine = self._connection.router.engines.get(kind)
            if engine is None or not engine.available():  # type: ignore[attr-defined]
                continue
            try:
                detail = engine.detail(self._resolved, version=self._version)  # type: ignore[attr-defined]
            except Exception as exc:
                # Losing this is how a vending failure turns into an empty
                # feature set and a confident, wrong "yes".
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            # Checked before anything is cached: raising after the cache was
            # filled let the very next call read the re-created table as if
            # it were the one the catalog described.
            self._check_identity(detail.get("metadata_id"), detail.get("properties"))
            self._resolved = dataclasses.replace(
                self._resolved,
                # A success clears an earlier failure; left in place it kept
                # the router refusing a table that now opens.
                open_error=None,
                min_reader_version=detail.get("min_reader_version"),
                min_writer_version=detail.get("min_writer_version"),
                reader_features=frozenset(detail.get("reader_features") or ()),
                writer_features=frozenset(detail.get("writer_features") or ()),
                properties={**self._catalog_properties, **(detail.get("properties") or {})},
                partition_columns=tuple(detail.get("partition_columns") or ()),
                has_invariants=bool(detail.get("has_invariants")),
                has_check_constraints=bool(detail.get("has_check_constraints")),
                has_generated_columns=bool(detail.get("has_generated_columns")),
                has_binary_partitions=bool(detail.get("has_binary_partitions")),
                has_datetime_columns=bool(detail.get("has_datetime_columns", True)),
            )
            self._enriched = getattr(self, "_generation", 0) == generation
            return self._resolved

        if last_error is not None and self._version is not None:
            self._check_pinned_version_exists(last_error)
        # A failure is not cached: a transient vending or network error used to
        # leave this handle refusing every operation for the rest of its life.
        if last_error is not None:
            self._resolved = dataclasses.replace(
                self._resolved, open_error=last_error, **_protocol_from_properties(self._resolved)
            )
        return self._resolved

    def _check_pinned_version_exists(self, error: str = "") -> None:
        """Say so plainly when the handle's version is past the latest one.

        Opening at a version that does not exist left every call (history
        included) refusing with "the Delta log could not be read" and advice
        to enable the SQL warehouse. A version whose commits were cleaned out
        of the log was reported as "there is no Delta table at this path".
        """
        for kind in (EngineKind.KERNEL, EngineKind.DELTARS):
            engine: Any = self._connection.router.engines.get(kind)
            if engine is None or not engine.available():
                continue
            try:
                latest = engine.detail(self._resolved, version=None).get("version")
            except Exception:
                continue
            if latest is not None and self._version is not None and self._version > latest:
                raise UnreachableTableError(
                    f"open {self._resolved.ref} at version {self._version}",
                    f"the table has no such version; the latest is {latest}",
                    "open it without version=, or at a committed version",
                )
            if (
                latest is not None
                and self._version is not None
                and self._version < latest
                and (_looks_missing(error) or "invalid table version" in error.lower())
            ):
                raise UnreachableTableError(
                    f"open {self._resolved.ref} at version {self._version}",
                    f"the table exists (latest version {latest}), but the commits needed to "
                    f"reconstruct version {self._version} are no longer in its log -- they "
                    "were removed by log retention or metadata cleanup",
                    "time travel to a version at or after the table's oldest checkpoint",
                )
            return

    # ------------------------------------------------------------ capabilities

    def capabilities(self) -> dict[Operation, Capability]:
        """What can and cannot be done with this table, and why.

        Every refusal names the blocker and, where one exists, the remedy.
        """
        return self._connection.router.capabilities(self._enrich())

    def can(self, operation: Operation | str, **shape: Any) -> Capability:
        """Whether an operation is possible, optionally for a specific request.

        `shape` takes the same arguments as the call itself -- the data too,
        as ``data=`` -- so you can preflight the write you actually intend::

            t.can("create", properties={"delta.enableRowTracking": "true"})
            t.can("append", schema_mode="merge", data=batch)
            t.can("plan_write", mode="overwrite")

        The answer is the router's verdict on the very request the call makes
        (see `deltaswamp._request`), so an ok here is the engine the call uses.
        A method name (``z_order``, ``compact_logs``, ``plan_write``, ...) is
        accepted in place of the operation it performs.
        """
        op, shape = _asked_operation(operation, shape)
        _check_can_options(op, shape)
        data = shape.pop("data", NO_DATA)
        if op is Operation.MERGE and "source" in shape:
            data = shape.pop("source")
        if data is not NO_DATA:
            data = _write_data(data)
            if op is not Operation.MERGE and not _consumable(data):
                # Lined up as the call lines it up: a left-out column with a
                # literal DEFAULT is filled in, and needs no SQL engine.
                data = self._align(data, shape.get("schema_mode"))
        request = self._request(op, shape, data)
        refused = refusal(self, request)
        if refused is not None:
            return refused
        verdict = self._connection.router.capability(
            request.operation, self._enrich(), needs=request.needs, **request.shape
        )
        if request.asked is Operation.INCREMENTAL:
            verdict = dataclasses.replace(verdict, operation=request.asked)
        return verdict

    def _request(self, operation: Operation, args: dict[str, Any], data: Any = NO_DATA) -> Request:
        """What a call of `operation` with these arguments routes on; see `_request`."""
        return derive(self, operation, args, data)

    def _route(self, request: Request, *, exclude: frozenset[EngineKind] = frozenset()) -> Any:
        """The engine that serves `request`: the one `can()` names for it."""
        return self._engine(request.operation, request.needs, exclude=exclude, **request.shape)

    def _predicate_needs(self, predicate: Any) -> set[str]:
        """``predicates``, and whether the predicate is clear of collated columns.

        On a table with collations the router keeps predicates off the direct
        engines, which compare bytes. That is only needed when the predicate
        reads a collated column, so one that provably does not (``id = 1``) is
        marked ``collation_free`` and may still be served directly. A column
        nested in a struct, list or map counts as collated: the collation of a
        nested field lives on its parent and does not reach the Arrow schema.
        """
        needs = {"predicates"}
        from .engine.dialect import warehouse_reason

        if warehouse_reason(predicate) is not None:
            needs.add("spark_sql")
        needs |= self._char_needs(predicate)
        table = self._enrich()
        if not table.features & {"collations", "collations-preview"}:
            return needs
        from .predicate import columns_of, parse

        try:
            paths = columns_of(parse(predicate))
            fields = {f.name.lower(): f for f in self.schema()}
        except Exception:
            # Unparseable here or no schema: keep the conservative answer, and
            # let the engine that serves the call report its own error.
            return needs
        for path in paths:
            field = fields.get(path[0].lower())
            if field is None or len(path) > 1 or not _is_plain_primitive(field.type):
                return needs
            if b"__COLLATIONS" in (field.metadata or {}):
                return needs
        needs.add("collation_free")
        return needs

    def _char_needs(self, predicate: Any) -> set[str]:
        """``char_padding`` when a predicate reads a CHAR(n) column.

        Spark compares a CHAR(n) value padded with spaces to n: in a CHAR(3)
        column holding 'a', ``c = 'a  '`` and ``c = 'a '`` match and
        ``c < 'a '`` does not. The direct engines compare the stored bytes, so
        such a read silently missed rows the warehouse returns (and a DELETE
        through the warehouse removed rows the same predicate did not read).
        Only the warehouse evaluates it.
        """
        if not isinstance(predicate, str) or not predicate.strip():
            return set()
        from .predicate import columns_of, parse

        try:
            schema = self.schema()
            chars = {
                f.name.lower()
                for f in schema
                if (f.metadata or {})
                .get(b"__CHAR_VARCHAR_TYPE_STRING", b"")
                .lower()
                .startswith(b"char(")
            }
            if not chars:
                return set()
            paths = columns_of(parse(predicate))
        except Exception:
            return set()
        return {"char_padding"} if any(p[0].lower() in chars for p in paths) else set()

    def _interval_columns(self) -> frozenset[str]:
        """The top-level columns (lower-cased) that are, or hold, an ANSI interval."""
        try:
            groups = interval_paths(self._raw_schema(with_log=True)[1])
        except Exception:
            return frozenset()
        return frozenset(p[0].lower() for paths in groups.values() for p in paths if p)

    def _interval_needs(self, *texts: Any) -> set[str]:
        """``interval_columns`` when SQL names a column Databricks stores an interval in.

        The log spells the interval; the files hold its INT32 months or INT64
        microseconds, and the direct engines see only those. delta-rs does not
        even resolve the column ("Schema error: No such field: ym") after
        can() had named it, and a kernel read compared the raw integers
        (``ym = -14`` matched ``INTERVAL '-1-2' YEAR TO MONTH``) where Spark
        compares intervals. Only the warehouse evaluates SQL on an interval
        as Spark does. `texts` are SQL strings, or mappings whose keys (the
        columns set) and values count.
        """
        names = self._interval_columns()
        if not names:
            return set()
        from .engine.intervals import sql_identifiers

        for text in texts:
            if isinstance(text, dict):
                keys = {str(k).strip("`").split(".")[-1].lower() for k in text}
                if keys & names:
                    return {"interval_columns"}
                values = [v for v in text.values() if isinstance(v, str)]
            else:
                values = [text] if isinstance(text, str) else []
            for value in values:
                if sql_identifiers(value) & names:
                    return {"interval_columns"}
        return set()

    def _variant_needs(self, columns: Any, predicate: Any) -> set[str]:
        """``variant_free`` when a read on a variant-shredding table skips every VARIANT column.

        Such a table is kept off the direct engines, which cannot decode a
        shredded file; a read of its other columns never opens one, so a
        ``count()`` or a projection still reads directly.
        """
        from .router import shreds_variants

        if not isinstance(columns, list | tuple) or not shreds_variants(self._enrich()):
            return set()
        try:
            variants = {p[0].lower() for p in self._variant_paths()}
            wanted = {str(c).lower() for c in columns}
            if predicate is not None:
                from .predicate import columns_of, parse

                wanted |= {path[0].lower() for path in columns_of(parse(predicate))}
        except Exception:
            return set()
        return set() if wanted & variants else {"variant_free"}

    def _variant_input(self, engine: Any, data: Any) -> Any:
        """JSON text bound for a VARIANT column, encoded for a direct engine.

        Reads give VARIANT as JSON text, so that is what a write takes back.
        The warehouse parses it itself (``parse_json``); the kernel and
        delta-rs write the binary encoding, which is built here. Data already
        in that binary shape passes through, and so does a stream.
        """
        if not isinstance(engine, (KernelEngine, DeltaRsEngine)):
            return data
        try:
            import pyarrow as pa
        except ImportError:
            return data
        if not isinstance(data, pa.Table):
            return data
        # A read gives an interval as a duration or as text; the direct
        # engines store the integers underneath (`engine.intervals`).
        groups = interval_paths(self._raw_schema(with_log=True)[1])
        if groups:
            data = storage_columns(pa, data, groups)
        if not self._enrich().features & _VARIANT_FEATURES:
            return data
        from ._variant import binary_columns

        return binary_columns(pa, data, self._variant_paths())

    def _engine(
        self,
        operation: Operation,
        needs: frozenset[str] = frozenset(),
        **shape: Any,
    ) -> Any:
        if operation not in _READ_OPERATIONS and self._version is None:
            # A write is routed on the protocol as it is now: another process
            # may have made the table append-only, added a constraint or
            # renamed a column since this handle last read the log, and the
            # stale answer sent the write to an engine that then failed with
            # a raw error instead of the router's refusal.
            self._enriched = False
        resolved = self._enrich()
        if (
            resolved.open_error is not None
            and resolved.ref.kind is RefKind.PATH
            and operation is not Operation.CREATE
            and _looks_missing(resolved.open_error)
        ):
            # The router's refusal suggests the SQL warehouse, which cannot
            # conjure a table at a path where there is none.
            raise UnreachableTableError(
                f"{operation.value} {resolved.location}",
                f"there is no Delta table at this path ({resolved.open_error})",
                "check the path, or create the table with create_table() / write_table()",
            )
        router = self._connection.router
        return _strict(
            router.engine_for(operation, resolved, needs=needs, **shape),
            operation,
            lambda kind: router.capability(
                operation,
                resolved,
                needs=needs,
                **{**shape, "exclude": frozenset(shape.get("exclude", ())) | {kind}},
            ),
        )

    def _read(
        self,
        request: Request | Operation,
        call: Callable[[Any], Any],
        *,
        exclude: frozenset[EngineKind] = frozenset(),
    ) -> Any:
        """Serve a read-only request, moving to the next engine if one breaks.

        Routing decides from the protocol, but an engine can still choke on a
        table it claims -- delta-rs cannot parse the file statistics Databricks
        writes for a CLONE, for one. A read changes nothing, so trying the next
        engine that claims it is safe.

        An engine can also refuse at read time what routing let through (the
        kernel reads a change feed across one schema only, which it learns only
        from the log). Such a limit (`EngineLimitError`) moves on to the next
        engine too; if every one refuses, the first refusal -- the preferred
        engine's -- is raised. Every other error this library raises (no such
        version, a timestamp before the history, a corrupt table, bad
        arguments) is about the request and propagates as it is.
        """
        if isinstance(request, Operation):
            request = self._request(request, {})
        self._check_still_named()
        operation, needs, shape = request.operation, request.needs, request.shape
        tried: set[EngineKind] = set(exclude)
        refusals: list[EngineLimitError] = []
        while True:
            try:
                engine: Any = _strict(
                    self._connection.router.engine_for(
                        operation, self._enrich(), needs=needs, exclude=frozenset(tried), **shape
                    ),
                    operation,
                    lambda kind: self._connection.router.capability(
                        operation,
                        self._resolved,
                        needs=needs,
                        exclude=frozenset(tried) | {kind},
                        **shape,
                    ),
                    self._connection.router.engines,
                )
            except DeltaSwampError:
                if refusals:
                    raise refusals[0] from refusals[0].__cause__
                raise
            try:
                return call(engine)
            except EngineLimitError as exc:
                tried.add(engine.kind)
                refusals.append(exc)
            except DeltaSwampError as exc:
                if not _engine_broke(exc):
                    raise
                tried.add(engine.kind)
                try:
                    self._connection.router.engine_for(
                        operation, self._resolved, needs=needs, exclude=frozenset(tried), **shape
                    )
                except DeltaSwampError:
                    if refusals:
                        raise refusals[0] from refusals[0].__cause__
                    raise exc from exc.__cause__
                cause = engine_cause(exc)
                warnings.warn(
                    f"{engine.kind.value} failed to serve {operation.value} "
                    f"({type(cause).__name__}: {str(cause)[:200]}); trying the next engine",
                    EngineFallbackWarning,
                    stacklevel=3,
                )

    # ------------------------------------------------------------------- read

    def scan(
        self,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: str | datetime.date | int | None = None,
        limit: int | None = None,
    ) -> Any:
        """Read the table as an Arrow stream (exports `__arrow_c_stream__`).

        `columns` projects, `predicate` is a Spark SQL boolean expression, and
        `version` or `timestamp` (a datetime, a date, an ISO-8601 string or
        epoch milliseconds; naive values are UTC) travels back in time.

        `limit` is a hint passed to the engine. Streaming engines ignore it and
        the caller simply stops reading; the SQL warehouse turns it into a real
        `LIMIT`, because it computes the result set before streaming any of it.
        """
        columns = _columns_arg(columns)
        _check_predicate(predicate, "scan")
        _check_count(limit, "limit")
        timestamp = _timestamp_arg(timestamp)
        version = self._travel_version(version, timestamp)
        # A handle opened at a version is time travel too, and routes as it
        # (see `_request`): an engine that serves SCAN but not TIME_TRAVEL
        # would read the latest. A predicate or a timestamp narrows which
        # engines can serve the call, so it is said up front instead of
        # letting one accept and then raise.
        request = self._request(
            Operation.SCAN,
            {
                "columns": columns,
                "predicate": predicate,
                "version": version,
                "timestamp": timestamp,
                "limit": limit,
            },
        )

        def scan(engine: Any) -> Any:
            if timestamp is not None and isinstance(engine, (KernelEngine, DeltaRsEngine)):
                self._refuse_future_timestamp(timestamp)
            stream = engine.scan(
                self._resolved,
                columns=columns,
                predicate=predicate,
                version=version,
                timestamp=timestamp,
                limit=limit,
            )
            if not isinstance(engine, (KernelEngine, DeltaRsEngine)):
                return stream
            return self._direct_stream(engine, stream, version, timestamp)

        return self._read(request, scan)

    def _direct_stream(self, engine: Any, stream: Any, version: Any, timestamp: Any) -> Any:
        """A direct engine's read, shown as the warehouse shows it, with errors named."""
        log = _log_schema(engine, self._resolved, version if version is not None else self._version)
        if self._enrich().features & _VARIANT_FEATURES:
            # The warehouse sends VARIANT as JSON text; so does this.
            from ._variant import json_text_stream, variant_paths

            stream = json_text_stream(stream, None if log is None else variant_paths(log))
        # And a day-time interval as a duration, not bare microseconds.
        stream = interval_stream(stream, interval_paths(log))
        # A file VACUUM (or a manual delete) removed fails only once reading
        # reaches it, as a bare OSError/ArrowInvalid; name it instead.
        where = self._resolved.location or str(self._resolved.ref)
        at = version if version is not None else timestamp
        context = f"{where}" + (f" at {at}" if at is not None else "")
        return translating_stream(stream, context, _shredded_variant_error)

    def added_since(
        self,
        version: int,
        *,
        until: int | None = None,
        columns: list[str] | None = None,
        predicate: str | None = None,
        only_appends: bool = False,
    ) -> Any:
        """The rows added after `version`, up to `until`, without the change data feed.

        Returns an Arrow stream of the rows in the data files committed in
        `(version, until]` (`until` defaults to the handle's version, else the
        latest) -- an incremental read of an append-only table, which needs no
        `delta.enableChangeDataFeed` (delta-rs#4554, delta-kernel-rs#1177).

        Only appends make that the rows added. A DELETE, UPDATE, MERGE,
        OPTIMIZE or overwrite in the range removes files and re-adds rows that
        were there before, so the call is refused if any file was removed;
        `only_appends=True` reads every file added anyway, rows a rewrite
        carried over included (for a consumer that dedupes on a key), and
        `cdf()` reads exactly what changed on a table with the feed on.
        """
        _check_version(version, "version")
        _check_version(until, "until")
        columns = _columns_arg(columns)
        _check_predicate(predicate, "added_since")
        if not isinstance(only_appends, bool):
            raise InvalidArgumentError(
                f"only_appends must be True or False, not {type(only_appends).__name__}"
            )
        end = until if until is not None else self._version
        if end is not None and end < version:
            raise InvalidArgumentError(f"until {end} is before version {version}")
        request = self._request(
            Operation.SCAN,
            {"incremental": True, "columns": columns, "predicate": predicate, "until": end},
        )

        def read(engine: Any) -> Any:
            stream = engine.added_since(
                self._resolved,
                version,
                until=end,
                columns=columns,
                predicate=predicate,
                only_appends=only_appends,
            )
            return self._direct_stream(engine, stream, end, None)

        return self._read(request, read)

    def _travel_version(self, version: int | None, timestamp: Any) -> int | None:
        """The version a read should use: the call's, else the handle's.

        An explicit timestamp overrides the handle's pinned version rather than
        being sent alongside it, which every engine refuses as "both".
        """
        _check_travel(version, timestamp)
        if version is not None or timestamp is not None:
            return version
        return self._version

    def _align(self, data: Any, schema_mode: str | None) -> Any:
        """Line an in-memory batch up with the table the way Delta does.

        Delta resolves column names case-insensitively and fills a nullable
        column the data leaves out with nulls. delta-rs does neither: "Field
        ID not found in schema", "number of fields does not match: 2 vs 3".
        pandas is converted with the table's types, so a map column (a list
        of pairs in pandas) no longer fails inference. A stream only has its
        left-out nullable columns added, batch by batch as it is read.
        """
        if schema_mode is not None:
            return data
        module = type(data).__module__ or ""
        is_pandas = module.startswith("pandas") and type(data).__name__ == "DataFrame"
        is_polars = module.startswith("polars") and type(data).__name__ == "DataFrame"
        try:
            import pyarrow as pa
        except ImportError:
            return data
        stream = isinstance(data, pa.RecordBatchReader)
        if not (is_pandas or is_polars or stream or isinstance(data, (pa.Table, pa.RecordBatch))):
            return data
        resolved = self._enrich()
        # A left-out generated or identity column is the engine's to compute.
        if resolved.writer_features & {"generatedColumns", "identityColumns"}:
            return data
        try:
            target = self.schema()
        except DeltaSwampError:
            return data
        by_name = {f.name: f for f in target}
        folded: dict[str, list[str]] = {}
        for name in by_name:
            folded.setdefault(name.lower(), []).append(name)

        def canonical(name: str) -> str:
            if name in by_name:
                return name
            matches = folded.get(name.lower(), [])
            return matches[0] if len(matches) == 1 else name

        if stream:
            _refuse_lossy_types(pa, data.schema, by_name, canonical)
            return _fill_stream(pa, data, target, canonical, set(resolved.partition_columns))
        if is_pandas:
            # A named index is data (even one that looks like a range, which
            # pyarrow would otherwise keep only as metadata); make it columns.
            unnamed = all(n is None for n in data.index.names)
            frame = data if unnamed else data.reset_index()
            columns = [canonical(str(c)) for c in frame.columns]
            if list(frame.columns) != columns:
                frame = frame.copy(deep=False)
                frame.columns = columns
            if len(set(columns)) == len(columns) and all(c in by_name for c in columns):
                try:
                    data = pa.Table.from_pandas(
                        frame,
                        schema=pa.schema([by_name[c] for c in columns]),
                        preserve_index=False,
                    )
                except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError):
                    data = pa.Table.from_pandas(frame, preserve_index=False)
            else:
                data = pa.Table.from_pandas(frame, preserve_index=False)
        elif is_polars:
            data = data.to_arrow()
        elif isinstance(data, pa.RecordBatch):
            data = pa.Table.from_batches([data])

        data = _refuse_lossy_types(pa, data.schema, by_name, canonical, data)
        names = [canonical(n) for n in data.column_names]
        if len(set(names)) != len(names):
            return data  # two columns fold to one name; let the engine refuse
        if names != data.column_names:
            data = data.rename_columns(names)
        for index, name in enumerate(names):
            column_type = data.schema.field(index).type
            wanted = by_name[name].type if name in by_name else None
            if pa.types.is_dictionary(column_type) and name in resolved.partition_columns:
                # delta-rs cannot read a partition value out of a dictionary
                # array ("failed to read partition column value as Scalar"),
                # which is what a polars Categorical arrives as.
                column_type = column_type.value_type
                data = data.set_column(
                    index,
                    pa.field(name, column_type, data.schema.field(index).nullable),
                    data.column(index).cast(column_type),
                )
            if (
                name in resolved.partition_columns
                and (pa.types.is_string(column_type) or pa.types.is_large_string(column_type))
                and data.num_rows
                and (name not in by_name or by_name[name].nullable)
            ):
                # Spark writes an empty-string partition value as null (a
                # directory cannot be named ""). delta-rs recorded "", which
                # DuckDB then read as null and this library as "".
                import pyarrow.compute as pc

                column = data.column(index)
                empty = pc.equal(column, "")
                if pc.any(empty).as_py():
                    data = data.set_column(
                        index,
                        data.schema.field(index),
                        pc.if_else(empty, pa.scalar(None, column_type), column),
                    )
            if wanted is not None and column_type != wanted and _has_map(pa, wanted):
                column = data.column(index)
                chunks = [_maps_from_lists(pa, chunk, wanted) for chunk in column.chunks]
                if any(new is not old for new, old in zip(chunks, column.chunks, strict=True)):
                    rebuilt = pa.chunked_array(chunks, type=chunks[0].type) if chunks else column
                    data = data.set_column(
                        index,
                        pa.field(name, rebuilt.type, data.schema.field(index).nullable),
                        rebuilt,
                    )
                    column_type = rebuilt.type
            if (
                wanted is not None
                and pa.types.is_unsigned_integer(column_type)
                and pa.types.is_signed_integer(wanted)
            ):
                # delta-rs reads uint8 as byte before looking at the table,
                # so 200 failed even against a short column. A safe cast to
                # the table's type keeps every value or raises here -- as a
                # library error: a uint64 above the BIGINT range escaped as a
                # bare ArrowInvalid.
                try:
                    cast = data.column(index).cast(wanted, safe=True)
                except pa.ArrowInvalid as exc:
                    raise InvalidArgumentError(
                        f"column {name!r} holds a value outside the range of the table's "
                        f"{wanted} ({exc}); widen the column (DECIMAL(20, 0) holds any uint64) "
                        "or filter the value out"
                    ) from exc
                data = data.set_column(index, by_name[name], cast)
        # Generated and identity columns are the engine's to compute. On a
        # legacy protocol (writer 4-6) no feature list names them, only the
        # field metadata does, and filling one with nulls failed delta-rs's
        # generation check ("rows failed validation check").
        computed = {
            f.name
            for f in target
            if f.metadata
            and any(
                key.startswith((b"delta.generationExpression", b"delta.identity."))
                for key in f.metadata
            )
        }
        missing = [f for f in target if f.name not in names and f.name not in computed]
        partitions = set(resolved.partition_columns)
        # A left-out column with a DEFAULT gets the default, as in Databricks,
        # not a null. A literal default is filled in here, for every engine; any
        # other (current_timestamp()) is left out for the warehouse to apply,
        # and `_default_needs` keeps such a write off the direct engines.
        fills: dict[str, Any] = {}
        for field in missing:
            if _column_default(field) is not None:
                fills[field.name] = _default_column(pa, field, data.num_rows)
        missing = [f for f in missing if _column_default(f) is None]
        for field in missing:
            # A left-out partition column is almost always a mistake (and a
            # dynamic overwrite derives its partitions from the data), so it
            # is never filled; nor is a required column. The engine refuses.
            if not field.nullable or field.name in partitions:
                return data
        for field in target:
            filled = fills.get(field.name)
            if filled is not None:
                data = data.append_column(field, filled)
        for field in missing:
            data = data.append_column(field, pa.nulls(data.num_rows, field.type))
        return data

    def _default_needs(self, data: Any) -> frozenset[str]:
        """``sql_column_defaults`` when the data leaves out a column only SQL can default.

        `_align` fills in a literal DEFAULT; one still missing after it is not
        a plain literal (``current_timestamp()``), or the batch could not be
        aligned, and has to be evaluated by Databricks. A direct engine would
        write a null in its place, so the write is refused there instead.
        """
        if not self._enrich().writer_features & {"allowColumnDefaults"}:
            return frozenset()
        names = getattr(data, "column_names", None)
        if names is None:
            return frozenset()
        try:
            target = self.schema()
        except DeltaSwampError:
            return frozenset()
        present = {str(n).lower() for n in names}
        if any(
            field.name.lower() not in present and _column_default(field) is not None
            for field in target
        ):
            return frozenset({"sql_column_defaults"})
        return frozenset()

    def _check_writable(self, what: str, *, pinned: bool = False) -> None:
        """Refuse a write through a handle opened at a past version.

        Every engine writes to the latest version, so a delete on
        ``conn.table(path, version=1)`` removed rows from the current table,
        not the version the handle shows. Delta refuses writes to a
        time-travelled table for the same reason. `pinned=True`: the call
        serves a pinned handle itself (and refuses through `_check_pinned`).
        """
        if self._version is not None and not pinned:
            raise InvalidArgumentError(
                f"cannot {what}: this handle is pinned to version {self._version}; "
                "open the table without version= to write to it"
            )
        if self._resolved.is_catalog_managed:
            # The catalog ratifies only latest+1, so a write from the tail
            # captured at resolution was a guaranteed 409 once anyone else had
            # committed -- and stayed one on every retry through this handle.
            self._refresh_commit_tail(before_write=True)

    def _check_pinned(self, request: Request) -> dict[str, Any]:
        """Refuse a pinned-handle DML `can()` refuses; else the engine's `read_version=`.

        DELETE, UPDATE and MERGE from a handle pinned to a past version read
        there and commit at the latest after Delta's conflict check against
        every commit since (delta-rs#4417): what the kernel's deletion-vector
        DML does when it loses a race to them. Where that cannot be done the
        call is refused, as before, saying why.
        """
        if self._version is None:
            return {}
        refused = refusal(self, request)
        if refused is not None:
            raise InvalidArgumentError(
                f"cannot {request.operation.value}: {refused.reason}; {refused.remedy}"
            )
        return {"read_version": self._version}

    def _pinned_write(self, write: Callable[[], Any]) -> Any:
        """`write()`, saying so when a pinned handle's commit lost to a later one.

        "Re-read the table and retry" never helps a handle pinned to the
        version it read: the same call conflicts again.
        """
        from .errors import CommitConflictError

        if self._version is None:
            return write()
        try:
            return write()
        except CommitConflictError as exc:
            exc.args = (
                f"{exc} This handle is pinned to version {self._version}, and a commit "
                "since then changed what it read: open a later version (or the latest) and "
                "run it from there.",
            )
            raise

    def to_arrow(self, **kwargs: Any) -> Any:
        """The table as a ``pyarrow.Table``, read eagerly.

        Takes `scan()`'s arguments (`columns`, `predicate`, `version`,
        `timestamp`, `limit`). Needs ``deltaswamp[pyarrow]``.
        """
        pa = _require("pyarrow", "pyarrow")
        check_keywords("to_arrow", self.scan, kwargs)
        limit = kwargs.pop("limit", None)
        if limit is not None:
            # The scan treats `limit` as a hint that streaming engines ignore,
            # so to_arrow(limit=1) returned the whole table. Stop at it here.
            return self.head(limit, **kwargs)
        stream = self.scan(**kwargs)
        # read_all() raises the typed error; pa.table() over the C stream
        # would flatten it to ArrowInvalid.
        return stream.read_all() if isinstance(stream, TranslatingStream) else pa.table(stream)

    def to_pandas(self, **kwargs: Any) -> Any:
        """The table as a pandas DataFrame, through `to_arrow()` and its arguments.

        Needs ``deltaswamp[pandas]``.
        """
        _require("pandas", "pandas")
        return self.to_arrow(**kwargs).to_pandas()

    def to_polars(self, *, lazy: bool = False, follow_latest: bool = False, **kwargs: Any) -> Any:
        """A Polars DataFrame, or with ``lazy=True`` a LazyFrame that reads on collect.

        The lazy form reads through this library, so it works on the tables
        `polars.scan_delta` cannot open (catalog-managed, row-tracked,
        vacuumProtocolCheck, ...). Its projection and the simple comparisons
        of its filters are pushed into the scan (columns read, files
        skipped); `columns=` and `predicate=` restrict it further. It reads
        the version current when it was made, at every collect; with
        ``follow_latest=True``, the latest at each collect.
        """
        pl = _require("polars", "polars")
        check_keywords("to_polars", self.scan, kwargs)
        if lazy and "limit" not in kwargs:
            from ._lazy import polars_frame

            # Not scan_pyarrow_dataset: that let pyarrow apply the filter, with
            # pyarrow's NaN semantics rather than Polars'.
            return polars_frame(self._lazy_dataset(follow_latest=follow_latest, **kwargs))
        frame = pl.DataFrame(self.to_arrow(**kwargs))
        return frame.lazy() if lazy else frame

    def _lazy_dataset(
        self, *, follow_latest: bool = False, duckdb_filters: bool = False, **kwargs: Any
    ) -> Any:
        """A pyarrow Dataset that scans this table when (and as far as) it is read.

        Every scan reads the version current when the dataset was made, as a
        pinned handle does: one DuckDB statement scans a relation once per
        reference (a self-join twice), and a LazyFrame once per collect, so
        re-resolving the latest each time mixed two snapshots in one answer,
        and a frame made before an overwrite yielded the new columns under the
        old declared schema. ``follow_latest=True`` reads the latest version
        at every scan instead; a scan whose schema is no longer the one the
        dataset declared then raises MetadataChangedError.

        ``duckdb_filters`` is for a dataset DuckDB scans: the filters it
        pushes are evaluated with DuckDB's semantics (see `_lazy`).
        """
        _require("pyarrow", "pyarrow")
        from ._lazy import TableDataset

        if not isinstance(follow_latest, bool):
            raise InvalidArgumentError(
                f"follow_latest must be True or False, got {follow_latest!r}"
            )
        columns = kwargs.pop("columns", None)
        predicate = kwargs.pop("predicate", None)
        source = self
        if (
            not follow_latest
            and self._version is None
            and kwargs.get("version") is None
            and kwargs.get("timestamp") is None
        ):
            pinned = self._current_version()
            if pinned is not None:
                # A handle, not a scan option: count_rows() still answers
                # from the log, at that version.
                source = Table(self._connection, self._resolved, version=pinned)
        # The stream's own schema, which is what every later scan yields.
        schema = source.head(0, columns=columns, predicate=predicate, **kwargs).schema
        # The hand-offs show a VARIANT as its JSON text, but the scan filters
        # the binary value: a pushed `v = '"x"'` compared a struct with a
        # string and failed. Nothing on such a column is pushed.
        opaque = frozenset(p[0] for p in self._variant_paths() if len(p) == 1)
        # An interval column is shown as a duration or Spark's text, but the
        # scan filters the stored integers (and refuses a predicate on them,
        # `_interval_needs`): a filter on one is evaluated on the frame.
        intervals = self._interval_columns()
        opaque |= frozenset(f.name for f in schema if f.name.lower() in intervals)
        return TableDataset(
            source,
            schema,
            columns=_columns_arg(columns),
            predicate=predicate,
            scan_options=kwargs,
            opaque=opaque,
            duckdb_filters=duckdb_filters,
        )

    def _current_version(self) -> int | None:
        """The latest version, to pin a lazy hand-off to; None where one cannot be read at it.

        A table that cannot be read at a version (time travel refused, say, by
        an access policy) is read at the latest by each scan, as before, with
        the declared schema still checked.
        """
        try:
            version = self.version
        except DeltaSwampError:
            return None
        if version is None or not self.can(Operation.SCAN, version=version).ok:
            return None
        return version

    def to_duckdb(
        self,
        connection: Any = None,
        *,
        name: str | None = None,
        follow_latest: bool = False,
        **kwargs: Any,
    ) -> Any:
        """A DuckDB relation over the table. With `name`, also a view of that name.

        DuckDB's own delta extension is C++ and knows nothing of Unity Catalog
        credentials or catalog-managed commits, so the relation reads through
        this library instead -- lazily, each time it runs, with DuckDB's
        projection and simple filters pushed into the scan. Every run reads
        the version current when the relation was made (so one statement sees
        one snapshot); ``follow_latest=True`` reads the latest at each scan.
        """
        duckdb = _require("duckdb", "duckdb")
        data = (
            self.to_arrow(**kwargs)
            if "limit" in kwargs
            else self._lazy_dataset(follow_latest=follow_latest, duckdb_filters=True, **kwargs)
        )
        con = connection
        if con is None and name is not None:
            # The view has to live where it can be queried by name. On a
            # private in-memory connection nothing but the returned relation
            # could reach it, so `duckdb.sql("... FROM <name>")` failed.
            default = duckdb.default_connection
            con = default() if callable(default) else default
        elif con is None:
            con = duckdb.connect()
        relation = con.from_arrow(data)
        if name is not None:
            relation.create_view(name, replace=True)
        return relation

    def plan_write(
        self,
        *,
        mode: str = "append",
        txn: tuple[str, int] | None = None,
        commit_metadata: dict[str, Any] | None = None,
        ship_catalog_auth: bool = False,
    ) -> Any:
        """Plan a distributed write, refusing now if the table will not accept it.

        A pickled plan carries the table's short-lived storage credential and
        no catalog credentials, so a worker whose credential expires must be
        given a fresh plan. ``ship_catalog_auth=True`` ships the catalog's
        credential provider (and so its token) instead, letting workers
        re-vend on their own.

        Returns a picklable `WritePlan`. Ship it to workers, call
        `plan.write(batch)` there, send the fragments back, and commit them all
        at once with `plan.commit(fragments)`. The write lands at a single
        version: a reader sees the whole job or none of it.

        The refusal happens here, on the driver, before any worker runs. That
        is the point of planning separately -- the usual way a distributed Delta
        write fails is to discover at commit time that the table rejects it,
        after the compute is spent, leaving orphaned Parquet behind. Anything
        `can()` reports as unavailable is raised here instead, with the reason.

        `mode` is ``append`` or ``overwrite``; overwrite removes every file
        visible in the planned snapshot in the same commit.
        """
        from .distributed import WritePlan, _commit_metadata_arg

        if mode not in ("append", "overwrite"):
            raise InvalidArgumentError(
                f"plan_write mode={mode!r}: a distributed write is 'append' or 'overwrite'"
            )
        # Settled here, not at commit: a pinned handle, a malformed txn or a
        # reserved commit_metadata key used to pass planning and fail only
        # when the job's fragments were committed -- after the compute.
        self._check_writable(f"plan a distributed {mode}")
        _check_txn(txn)
        # A list passes the check but the binding takes only a tuple, so
        # txn=["job", 1] failed with TypeError at commit, after the job ran.
        txn = (txn[0], int(txn[1])) if txn is not None else None
        commit_metadata = _commit_metadata_arg(commit_metadata)
        # can("plan_write", mode=...) asks about this very request.
        request = self._request(
            Operation.APPEND,
            {"mode": mode, "txn": txn, "commit_metadata": commit_metadata, "distributed": True},
        )
        if txn is not None and self._already_committed(txn):
            raise UnreachableTableError(
                f"plan an idempotent write for {txn[0]!r} at version {txn[1]}",
                "that transaction is already committed, so running the job would "
                "duplicate work whose result is already in the table",
                "raise the txn version, or drop txn= to write unconditionally",
            )
        engine = self._route(request)
        identity = None
        if isinstance(engine, KernelEngine) and self.version is not None:
            # Read from storage, not the cache: this is the identity every
            # worker's write and the commit are checked against.
            identity = str(
                engine.snapshot(self._resolved, version=self.version, fresh=True).metadata_id
            )
        return WritePlan(
            engine=engine,
            table=self._enrich(),
            variant_paths=tuple(sorted(self._variant_paths())),
            interval_paths=self._interval_plan(),
            table_identity=identity,
            mode=mode,
            version=self.version,
            txn=txn,
            commit_metadata=commit_metadata,
            ship_catalog_auth=bool(ship_catalog_auth),
            catalog=(
                self._connection._catalog_for(self._resolved.ref)
                if self._resolved.is_catalog_managed and self._resolved.ref.kind is RefKind.CATALOG
                else None
            ),
        )

    def plan_scan(
        self,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: Any = None,
        ship_catalog_auth: bool = False,
    ) -> Any:
        """Plan a distributed read: a picklable `ScanPlan` of per-file splits.

        As with `plan_write`, a pickled plan carries a short-lived storage
        credential and no catalog credentials unless ``ship_catalog_auth=True``.

        Ship the plan (or parts of it, via `plan.partitions(n)`) to workers and
        call `plan.read(splits)` there. Each worker re-resolves the same
        snapshot version and reads with the storage credential the driver
        vended when the plan was pickled, which it refuses within a minute of
        expiry; with ``ship_catalog_auth=True`` each worker vends its own.
        """
        from .distributed import ScanPlan

        columns = _columns_arg(columns)
        _check_predicate(predicate, "plan a scan")
        timestamp = _timestamp_arg(timestamp)
        version = self._travel_version(version, timestamp)
        request = self._request(
            Operation.SCAN,
            {
                "columns": columns,
                "predicate": predicate,
                "version": version,
                "timestamp": timestamp,
                "distributed": True,
            },
        )
        # Asked before routing: learning the schema routes a read of its own.
        intervals = self._interval_plan()
        engine = self._route(request)
        splits = engine.plan_scan(
            self._resolved,
            columns=columns,
            predicate=predicate,
            version=version,
            timestamp=timestamp,
        )
        # An empty snapshot yields no splits to carry its version, and reading
        # the plan then resolved the latest one, with its (possibly evolved)
        # schema. Pin it on the plan itself.
        planned = splits[0].commit_version if splits else version
        if planned is None and callable(getattr(engine, "snapshot", None)):
            try:
                planned = int(engine.snapshot(self._resolved, timestamp=timestamp).version)
            except Exception:
                planned = None
        return ScanPlan(
            engine=engine,
            table=self._resolved,
            splits=tuple(splits),
            columns=tuple(columns) if columns is not None else None,
            predicate=predicate,
            snapshot_version=planned,
            ship_catalog_auth=bool(ship_catalog_auth),
            variant_paths=tuple(sorted(self._variant_paths())),
            interval_paths=intervals,
        )

    def to_ray_dataset(self, *, override_num_blocks: int | None = None, **kwargs: Any) -> Any:
        """A Ray Dataset, read in parallel by Ray workers.

        The scan is planned on the driver and each read task reads a
        byte-balanced group of files. Where no engine can plan a distributed
        read, the table is read on the driver instead.
        """
        ray_data = _require("ray.data", "ray")
        from .distributed import DeltaSwampDatasource

        # A plan has no row limit; apply it to the dataset instead of passing
        # it to plan_scan(), which does not take one.
        limit = kwargs.pop("limit", None)
        _check_count(limit, "limit")
        if self.can(Operation.SCAN).engine is not None:
            try:
                plan = self.plan_scan(**kwargs)
            except UnreachableTableError:
                plan = None
            # With no files to read there are no read tasks, and Ray builds a
            # dataset with no schema at all; the driver read keeps the columns.
            if plan is not None and plan.splits:
                dataset = ray_data.read_datasource(
                    DeltaSwampDatasource(plan), override_num_blocks=override_num_blocks
                )
                return dataset.limit(limit) if limit is not None else dataset
        if limit is not None:
            kwargs["limit"] = limit
        table = self.to_arrow(**kwargs)
        if limit is not None:
            table = table.slice(0, limit)
        return ray_data.from_arrow(table)

    def to_daft(self, **kwargs: Any) -> Any:
        """A Daft DataFrame, read eagerly: pass `columns=` and `predicate=` to narrow it."""
        daft = _require("daft", "daft")
        return daft.from_arrow(self.to_arrow(**kwargs))

    def to_pyarrow_dataset(self, *, follow_latest: bool = False, **kwargs: Any) -> Any:
        """A pyarrow Dataset that scans the table when read, pushing columns and filters down.

        Every scan reads the version current when the dataset was made;
        ``follow_latest=True`` reads the latest at each scan.
        """
        _require("pyarrow.dataset", "pyarrow")
        if "limit" in kwargs:
            import pyarrow.dataset as dataset

            return dataset.dataset(self.to_arrow(**kwargs))
        return self._lazy_dataset(follow_latest=follow_latest, **kwargs)

    def head(self, n: int = 5, **kwargs: Any) -> Any:
        """The first `n` rows.

        Consumes the scan stream batch by batch and stops as soon as `n` rows
        are in hand, so this costs one batch on a table of any size.

        The stream is the engine's output, so deletion vectors, column mapping
        and partition values are already applied; stopping early here is not the
        limit pushdown the scan layer refuses, which would break
        the positional mapping a deletion vector depends on.
        """
        pa = _require("pyarrow", "pyarrow")
        # A negative n used to return an empty table silently.
        _check_count(n, "n")
        # The limit is a hint to the engine as well as a client-side stop: the
        # warehouse would otherwise compute the entire result set first.
        kwargs["limit"] = n
        stream = self.scan(**kwargs)
        reader = (
            stream
            if isinstance(stream, TranslatingStream)
            else pa.RecordBatchReader.from_stream(stream)
        )
        batches, taken = [], 0
        if n > 0:
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                batches.append(batch)
                taken += batch.num_rows
                if taken >= n:
                    break
        table = (
            pa.Table.from_batches(batches, reader.schema)
            if batches
            else reader.schema.empty_table()
        )
        return table.slice(0, n)

    def count(self, *, predicate: str | None = None) -> int:
        """Exact row count.

        On the kernel it comes from the log when it can: every file's
        `numRecords` less its deletion vector's cardinality, with a predicate
        on partition columns applied to each file's partition values. When a
        file has no `numRecords`, or the predicate reads data columns, the
        narrowest column is streamed instead of materializing the table. The
        engines' own statistics-based counts are approximate by their own
        documentation (a file without stats counts as zero rows), so they are
        not used.
        """
        _check_predicate(predicate, "count")
        self._check_still_named()
        exact = self._count_from_log(predicate)
        if exact is not None:
            return exact
        pa = _require("pyarrow", "pyarrow")
        schema = self.schema()
        names = list(getattr(schema, "names", None) or [f.name for f in schema])
        # Not a VARIANT column when there is another: that read is what the
        # direct engines cannot do on a shredded table (see can("count")).
        variants = {p[0] for p in self._variant_paths()}
        names = [n for n in names if n not in variants] or names
        narrow = [names[0]] if names and predicate is None else None
        total = 0
        stream = self.scan(columns=narrow, predicate=predicate)
        if not isinstance(stream, TranslatingStream):
            stream = pa.RecordBatchReader.from_stream(stream)
        for batch in stream:
            total += batch.num_rows
        return total

    def _count_from_log(self, predicate: str | None) -> int | None:
        """`count()` from the log's numRecords, or None to count by scanning.

        Anything unusual -- a table the kernel does not serve, VARIANT
        columns, a log that cannot be read -- goes to the scan, whose own
        routing and errors are the ones to report.
        """
        resolved = self._enrich()
        if resolved.features & _VARIANT_FEATURES or resolved.open_error is not None:
            return None
        request = self._request(Operation.SCAN, {"predicate": predicate})
        try:
            engine: Any = self._connection.router.engine_for(
                request.operation, resolved, needs=request.needs, **request.shape
            )
            if not isinstance(engine, KernelEngine):
                return None
            counted = engine.metadata_count(
                self._resolved, predicate=predicate, version=self._version
            )
        except Exception:
            return None
        return None if counted is None else int(counted)

    def files(self) -> Any:
        """The table's live data files: path, size, partition values and statistics.

        One row per file, in delta-rs's flattened layout whichever engine
        lists them: ``path``, ``size_bytes``, ``modification_time``,
        ``num_records``, then ``null_count.<col>``, ``min.<col>``,
        ``max.<col>`` and ``partition.<col>`` by logical column name.
        """
        served: list[Any] = []

        def files(engine: Any) -> Any:
            served.append(engine)
            return engine.files(self._resolved, version=self._version)

        result = self._read(Operation.FILES, files)
        try:
            import pyarrow as pa
        except ImportError:
            return result
        files_table = pa.table(result)
        if isinstance(served[-1], KernelEngine):
            # The kernel lists raw add actions (`size`, a partition map, the
            # stats JSON); the same call on a table delta-rs could open had
            # entirely different columns, so code written against one broke
            # on the other.
            files_table = _flat_files(pa, files_table, self.schema())
        return files_table

    def history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Commit history, newest first. `timestamp` is epoch milliseconds.

        Milliseconds is what the Delta log records and what delta-rs and the
        Iceberg engine return; the warehouse returns a datetime, so it is
        converted here and the type no longer depends on which engine served.
        """
        _check_count(limit, "limit")
        if limit == 0:
            # The warehouse treats a falsy limit as none and returned all of it.
            return []
        result: list[dict[str, Any]] = self._read(
            Operation.HISTORY, lambda engine: engine.history(self._resolved, limit=limit)
        )
        for entry in result:
            stamp: Any = entry.get("timestamp")
            if hasattr(stamp, "timestamp"):
                entry["timestamp"] = round(stamp.timestamp() * 1000)
        return result

    def detail(self) -> dict[str, Any]:
        """Version, location, protocol, properties and size, as DESCRIBE DETAIL reports them."""
        result: dict[str, Any] = self._read(
            Operation.DETAIL, lambda engine: engine.detail(self._resolved, version=self._version)
        )
        return result

    def cdf(self, **kwargs: Any) -> Any:
        """The change data feed, by version or timestamp range.

        Rows carry `_change_type`, `_commit_version` and `_commit_timestamp`.
        A range that ends before a column was added reads it as null, as
        Databricks' `table_changes()` does: without column mapping the feed
        has the table's current columns, whatever version it ends at (with
        column mapping, Delta reads a batch feed under its end version's
        schema, and so does this).
        """
        bounded = kwargs.get("columns") is None and (
            kwargs.get("ending_version") is not None or kwargs.get("ending_timestamp") is not None
        )
        result = self._cdf(kwargs, split=True)
        if not bounded:
            return result
        mode = str(self.properties().get("delta.columnMapping.mode", "none") or "none")
        return self._with_current_columns(result) if mode.lower() == "none" else result

    def _with_current_columns(self, feed: Any) -> Any:
        """`feed` with the columns the table has now and it lacks, as nulls.

        Only added columns: a feed whose columns were since dropped, renamed or
        retyped is returned as read (the direct engines refuse a range across
        such a change; a range ending before it keeps the older columns).
        """
        try:
            import pyarrow as pa

            current = self.schema()
            have = {n.lower() for n in feed.schema.names}
        except Exception:
            return feed
        if not isinstance(current, pa.Schema):
            return feed
        missing = [f for f in current if f.name.lower() not in have]
        if not missing:
            return feed
        added = {f.name for f in missing}
        order = [f.name for f in current if f.name.lower() in have or f.name in added]
        by_lower = {n.lower(): n for n in feed.schema.names}
        names = [by_lower.get(n.lower(), n) for n in order]
        names += [n for n in feed.schema.names if n not in names]
        fields = {f.name: f for f in feed.schema}
        fields.update({f.name: f.with_nullable(True) for f in missing})
        schema = pa.schema([fields[n] for n in names], metadata=feed.schema.metadata)

        def batches() -> Any:
            for batch in feed:
                columns = {n: batch.column(n) for n in batch.schema.names}
                yield pa.RecordBatch.from_arrays(
                    [
                        columns[n] if n in columns else pa.nulls(batch.num_rows, fields[n].type)
                        for n in names
                    ],
                    schema=schema,
                )

        return pa.RecordBatchReader.from_batches(schema, batches())

    def _cdf(self, kwargs: dict[str, Any], *, split: bool) -> Any:
        """`cdf()`; `split=False` for a range already known to cross no schema change."""
        unknown = sorted(set(kwargs) - _CDF_OPTIONS)
        if unknown:
            # A misspelt bound (start_version=) reached the engine as a bare
            # TypeError, or on some engines was accepted and meant "from 0".
            raise InvalidArgumentError(
                f"cdf() got unexpected option(s) {unknown}; it takes {sorted(_CDF_OPTIONS)}"
            )
        if "columns" in kwargs:
            kwargs["columns"] = _columns_arg(kwargs["columns"])
            if kwargs["columns"] is not None:
                # delta-rs and the warehouse projected the change metadata
                # away, so a projected feed could not be told apart by version.
                kwargs["columns"] += [c for c in _CDF_META if c not in kwargs["columns"]]
        _check_predicate(kwargs.get("predicate"), "read the change data feed")
        start, end = kwargs.get("starting_version"), kwargs.get("ending_version")
        _check_version(start, "starting_version")
        _check_version(end, "ending_version")
        if start is not None and end is not None and end < start:
            raise InvalidArgumentError(f"ending_version {end} is before starting_version {start}")
        for key in ("starting_timestamp", "ending_timestamp"):
            if key in kwargs:
                kwargs[key] = _timestamp_arg(kwargs[key], key)
        given = _given(kwargs)
        if not split or start is None:
            return self._cdf_read(given, start, end)
        # Splitting at schema changes is how the direct engines read across
        # one; the warehouse's table_changes() reads the range itself. Asked
        # first, as can() asks it: a table whose feed no engine can read was
        # refused with a schema-change error found on the way, not the
        # reason can() gave.
        engine = self._cdf_engine()
        if engine.kind not in _DIRECT_FEED_ENGINES:
            return self._cdf_read(given, start, end)
        segments = self._feed_segments(start, end)
        if segments is None or len(segments) == 1:
            return self._cdf_read(given, start, end)
        from .errors import ChangeFeedSchemaChangeError

        try:
            return self._stitched_cdf(given, segments)
        except ChangeFeedSchemaChangeError as exc:
            refusal = exc
        # A column renamed or dropped is where the direct engines stop; the
        # warehouse still reads such a range (a column-mapping table under its
        # latest schema), so it takes over where there is one.
        try:
            self._cdf_engine(exclude=_DIRECT_FEED_ENGINES)
        except DeltaSwampError:
            raise refusal from refusal.__cause__
        return self._cdf_read(given, start, end, exclude=_DIRECT_FEED_ENGINES)

    def _cdf_engine(self, exclude: frozenset[EngineKind] = frozenset()) -> Any:
        """The engine a change-feed read routes to (raises the refusal can() gives)."""
        request = self._request(Operation.CDF, {})
        return _strict(
            self._connection.router.engine_for(
                Operation.CDF, self._enrich(), needs=request.needs, exclude=exclude, **request.shape
            ),
            Operation.CDF,
            lambda kind: self._connection.router.capability(
                Operation.CDF,
                self._resolved,
                needs=request.needs,
                exclude=exclude | {kind},
                **request.shape,
            ),
        )

    def _cdf_read(
        self,
        given: dict[str, Any],
        start: int | None,
        end: int | None,
        *,
        exclude: frozenset[EngineKind] = frozenset(),
    ) -> Any:
        """One change-feed read, its failures translated."""
        # A file VACUUM removed, or rows written under a schema a later commit
        # replaced, failed as a raw ArrowInvalid (with a Python traceback
        # embedded in its message) where a scan names the missing file.
        where = self._resolved.location or str(self._resolved.ref)

        def translate(exc: BaseException) -> Exception | None:
            if "cannot skip miniblock" in str(exc):
                # arrow-rs skips DELTA_BINARY_PACKED values only in 32- or
                # 64-value miniblocks, and Photon writes 256.
                return EngineLimitError(
                    "read the change data feed",
                    "the kernel's Parquet reader cannot skip within a DELTA_BINARY_PACKED "
                    "page whose miniblocks hold 256 values, as Databricks writes them "
                    f"({(str(exc).splitlines() or [''])[0][:200]})",
                    "read a narrower version range, or ds.connect(..., "
                    "allow_sql_fallback=True) to read it with table_changes()",
                )
            off = re.search(r"feed is unsupported for the table at version (\d+)", str(exc))
            if off is not None:
                # Raised by the kernel mid-stream, after the rows before it,
                # as a raw ArrowInvalid that `except DeltaSwampError` missed.
                return _feed_gap_error(int(off.group(1)))
            if "DELTA_CHANGE_DATA_FEED_INCOMPATIBLE" in str(exc):
                # The warehouse's own refusal of the same feed, which reached
                # the caller as a raw SqlStatementError.
                return _warehouse_feed_schema_error(exc)
            return self._feed_schema_change(exc, start, end)

        served: dict[str, Any] = {}

        def call(engine: Any) -> Any:
            served["kind"] = getattr(engine, "kind", None)
            stream = engine.cdf(self._resolved, **given)
            if not isinstance(engine, (KernelEngine, DeltaRsEngine)):
                return stream
            # Intervals typed as the warehouse's table_changes() types them.
            log = _log_schema(engine, self._resolved, end)
            return interval_stream(stream, interval_paths(log))

        try:
            # Routed on the call's own options, as can("cdf", ...) routes them.
            stream = _cdf_types(
                self._read(self._request(Operation.CDF, given), call, exclude=exclude)
            )
        except Exception as exc:
            from .engine.base import missing_file_error

            translated = missing_file_error(exc, f"the change data feed of {where}")
            translated = translated or translate(exc)
            if translated is None:
                raise
            raise translated from exc
        result = translating_stream(stream, f"the change data feed of {where}", translate)
        if isinstance(result, TranslatingStream):
            # changes() streams a feed that arrives commit by commit (the
            # kernel's) instead of reading it whole.
            result.engine_kind = served.get("kind")  # type: ignore[attr-defined]
        return result

    def _schema_reader(self) -> Any:
        """`version -> [(name, type), ...]` from the log, or None without a snapshot engine."""
        import json

        snapshot = getattr(
            self._engine(Operation.TIME_TRAVEL, frozenset({"variant_free"})), "snapshot", None
        )
        if snapshot is None:
            return None

        def fields(version: int | None) -> tuple[int, list[tuple[str, str]]]:
            snap = snapshot(self._resolved, version=version)
            metadata = json.loads(snap.metadata_json())
            schema = json.loads(metadata.get("schemaString") or "{}")
            # Name and type only: a comment or a dropped NOT NULL changes
            # the metadata but not how an older row reads.
            return int(snap.version), [
                (f["name"], json.dumps(f["type"], sort_keys=True)) for f in schema.get("fields", [])
            ]

        return fields

    def _feed_segments(self, start: int, end: int | None) -> list[tuple[int, int]] | None:
        """`start..end` split where the table's schema changed, or None if unknown.

        The kernel reads a change feed under one schema only (and a range
        that crossed an ADD COLUMN failed with a Parquet decode error on
        files Photon wrote), so a range spanning a schema change is read as
        one range per schema. A range of up to `_FEED_SCHEMA_WALK` versions has
        every version's schema compared: a change later undone (ADD COLUMN,
        then RESTORE to before it) leaves the ends equal, and the kernel then
        failed mid-stream, which reached ``pa.table(t.cdf())`` as a bare
        ArrowInvalid. A longer range compares the ends and finds the change
        versions by bisection when they differ.
        """
        try:
            fields = self._schema_reader()
            if fields is None:
                return None
            high, last = fields(end)
            if start >= high:
                return None
            first = fields(start)[1]
            segments: list[tuple[int, int]] = []
            if high - start <= self._FEED_SCHEMA_WALK:
                low, current = start, first
                for version in range(start + 1, high + 1):
                    here = last if version == high else fields(version)[1]
                    if here != current:
                        segments.append((low, version - 1))
                        low, current = version, here
                segments.append((low, high))
                return segments
            if first == last:
                return [(start, high)]
            low, current = start, first
            while True:
                if fields(high)[1] == current:
                    segments.append((low, high))
                    return segments
                # The first version after `low` whose schema differs.
                lo, hi = low + 1, high
                while lo < hi:
                    mid = (lo + hi) // 2
                    if fields(mid)[1] == current:
                        lo = mid + 1
                    else:
                        hi = mid
                segments.append((low, lo - 1))
                low, current = lo, fields(lo)[1]
        except Exception:
            # Unknown, not wrong: the single read below reports what fails.
            return None

    def _stitched_cdf(self, given: dict[str, Any], segments: list[tuple[int, int]]) -> Any:
        """The change feed over several schemas, as Spark reads it: under the latest.

        Rows written before an ADD COLUMN read the new column as null. A change
        an older row cannot be read under (a column dropped, renamed or
        retyped) is refused, as Spark refuses it
        (DELTA_CHANGE_DATA_FEED_INCOMPATIBLE_SCHEMA_CHANGE), naming the version
        so a follower can resume after it.
        """
        from .errors import ChangeFeedSchemaChangeError

        pa = _require("pyarrow", "pyarrow")
        fields = self._schema_reader()
        assert fields is not None  # _feed_segments found the segments with it
        for low, _ in segments[1:]:
            earlier = dict(fields(low - 1)[1])
            later = dict(fields(low)[1])
            lost = [n for n, t in earlier.items() if later.get(n) != t]
            if lost:
                error = ChangeFeedSchemaChangeError(
                    "read the change data feed",
                    f"the table's schema changed at version {low} in a way rows written "
                    f"before it cannot be read under (column(s) {lost} dropped, renamed "
                    "or retyped)",
                    f"read the feed up to version {low - 1}, or from version {low} on",
                )
                error.version = low
                raise error
        parts = []
        for low, high in segments:
            part = self._cdf_read(
                {**given, "starting_version": low, "ending_version": high}, low, high
            )
            parts.append(part.read_all() if isinstance(part, TranslatingStream) else pa.table(part))
        tables = [_plain_views(t) for t in parts]
        stitched = pa.concat_tables(tables, promote_options="default")
        # New columns land where the latest schema has them, not at the end.
        order = [n for n in tables[-1].column_names if n in stitched.column_names]
        order += [n for n in stitched.column_names if n not in order]
        return pa.RecordBatchReader.from_batches(
            stitched.select(order).schema, stitched.select(order).to_batches()
        )

    #: How many versions back `_feed_schema_change` looks for the change.
    _FEED_SCHEMA_SEARCH = 200

    #: The longest range `_feed_segments` compares version by version.
    _FEED_SCHEMA_WALK = 200

    def _feed_schema_change(
        self, exc: BaseException, start: int | None, end: int | None
    ) -> Exception | None:
        """`exc` as a ChangeFeedSchemaChangeError when a schema change caused it, else None.

        Spark refuses a feed that spans an incompatible schema change
        (DELTA_CHANGE_DATA_FEED_INCOMPATIBLE_SCHEMA_CHANGE) and names the
        version, so a streaming consumer can restart after it. Only a failure
        that reads like a type or shape mismatch, in a range that really does
        contain a schema change, is translated.
        """
        import json
        import re

        from .errors import ChangeFeedSchemaChangeError

        # The first line, without the Python traceback the C stream embeds.
        detail = (str(exc).strip().splitlines() or [""])[0].split(" Detail: Python")[0][:200]
        # Only the error's own first line: a Parquet decode failure
        # ("cannot skip miniblock of size 256") matched on words in the
        # embedded traceback and was reported as a schema change.
        if (
            isinstance(exc, DeltaSwampError)
            or not re.search(
                r"cast|datatype|data type|number of fields|schema", detail, re.IGNORECASE
            )
            or re.search(r"parquet (argument )?error", detail, re.IGNORECASE)
        ):
            return None
        try:
            snapshot = getattr(
                self._engine(Operation.TIME_TRAVEL, frozenset({"variant_free"})), "snapshot", None
            )
            if snapshot is None:
                return None
            high = end if end is not None else int(snapshot(self._resolved).version)
            low = max(1, start if start is not None else high - self._FEED_SCHEMA_SEARCH)
            low = max(low, high - self._FEED_SCHEMA_SEARCH)

            def schema_at(version: int) -> Any:
                metadata = json.loads(snapshot(self._resolved, version=version).metadata_json())
                return json.loads(metadata.get("schemaString") or "{}")

            changed = None
            later = schema_at(high)
            for version in range(high, low - 1, -1):
                earlier = schema_at(version - 1)
                if earlier != later:
                    changed = version
                    break
                later = earlier
        except Exception:
            return None
        if changed is None:
            return None
        error = ChangeFeedSchemaChangeError(
            "read the change data feed",
            f"the table's schema changed at version {changed}, and rows the feed returns "
            f"were written under the schema before it, which they cannot be read as "
            f"({type(exc).__name__}: {detail})",
            f"read the feed from version {changed} on, or up to version {changed - 1}",
        )
        error.version = changed
        return error

    def changes(
        self,
        starting_version: int,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        poll_interval: float | None = None,
        include_snapshot: bool = False,
    ) -> Any:
        """Follow the change feed, one committed version at a time.

        Yields ``(version, pyarrow.Table)`` for every version from
        `starting_version` on, in order. With `poll_interval` (seconds) it keeps
        waiting for new commits, like a streaming read with a change-feed
        source; without, it stops at the latest version. Record the last
        version you processed and pass the next one to resume.

        `include_snapshot=True` bootstraps a consumer, as a streaming read
        with an initial snapshot does: the first yield is the whole table as
        of `starting_version`, every row an ``insert`` of that version, and
        the feed follows from the version after it. The feed need not have
        been on before `starting_version`.
        """
        import time

        _check_version(starting_version, "starting_version")
        if poll_interval is not None and (
            isinstance(poll_interval, bool) or not isinstance(poll_interval, (int, float))
        ):
            raise InvalidArgumentError(
                f"poll_interval must be a number of seconds, not {type(poll_interval).__name__}"
            )
        if poll_interval is not None and poll_interval < 0:
            raise InvalidArgumentError(f"poll_interval must be >= 0, got {poll_interval}")
        if not isinstance(include_snapshot, bool):
            raise InvalidArgumentError(
                f"include_snapshot must be True or False, not {type(include_snapshot).__name__}"
            )
        pa = _require("pyarrow", "pyarrow")
        columns = _columns_arg(columns)
        # Splitting by version needs `_commit_version`; a projection without
        # it crashed with "Invalid sort key column". Read it, then drop it.
        projection = columns
        if columns is not None and "_commit_version" not in columns:
            projection = [*columns, "_commit_version"]
        next_version = starting_version
        if include_snapshot:
            yield (
                starting_version,
                self._snapshot_as_inserts(pa, starting_version, columns, predicate),
            )
            next_version = starting_version + 1
        while True:
            current = self._connection._reresolve(self)
            latest = current.version
            if latest is not None and latest >= next_version:
                # One read per schema: each version is yielded under the
                # schema it was written with. Read as one range, a range
                # crossing a schema change failed on every poll, and the
                # follower never got past it.
                segments = current._feed_segments(next_version, latest) or [(next_version, latest)]

                for low, high in segments:
                    feed = current._cdf(
                        {
                            "starting_version": low,
                            "ending_version": high,
                            "columns": projection,
                            "predicate": predicate,
                        },
                        split=False,
                    )
                    if getattr(feed, "engine_kind", None) is EngineKind.KERNEL:
                        # The kernel produces the feed commit by commit, so
                        # each version is yielded as soon as it is complete:
                        # read whole, 5000 versions took 11 s and 1.8 GB
                        # before the first was yielded. A version the feed
                        # was off at fails the stream after the versions
                        # before it, which are yielded first.
                        yield from _stream_by_version(pa, feed, columns)
                    else:
                        # read_all() keeps the typed error pa.table() would flatten.
                        changes = (
                            feed.read_all()
                            if isinstance(feed, TranslatingStream)
                            else pa.table(feed)
                        )
                        yield from _by_version(pa, changes, columns)
                    next_version = high + 1
                next_version = latest + 1
            if poll_interval is None:
                return
            time.sleep(poll_interval)

    def _snapshot_as_inserts(
        self, pa: Any, version: int, columns: list[str] | None, predicate: str | None
    ) -> Any:
        """The table at `version` as change-feed rows: each an insert of that version."""
        current = self._connection._reresolve(self) if self._version is None else self
        pinned = Table(self._connection, current._resolved, version=version)
        wanted = None if columns is None else [c for c in columns if c not in _CDF_META]
        rows = _plain_views(pa.table(pinned.scan(columns=wanted, predicate=predicate)))
        stamp = pa.scalar(self._commit_timestamp(pinned, version), pa.timestamp("us", tz="UTC"))
        n = rows.num_rows
        rows = rows.append_column(
            pa.field("_change_type", pa.string(), nullable=False),
            pa.array(["insert"] * n, pa.string()),
        )
        rows = rows.append_column(
            pa.field("_commit_version", pa.int64(), nullable=False),
            pa.array([version] * n, pa.int64()),
        )
        rows = rows.append_column(
            pa.field("_commit_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.array([stamp.value] * n, pa.timestamp("us", tz="UTC")),
        )
        return rows if columns is None else rows.select(columns)

    def _commit_timestamp(self, pinned: Table, version: int) -> Any:
        """When `version` was committed (the in-commit timestamp where there is one)."""
        import datetime as dt

        kernel = self._connection.router.engines.get(EngineKind.KERNEL)
        if kernel is not None and hasattr(kernel, "snapshot"):
            with contextlib.suppress(Exception):
                ms = int(kernel.snapshot(pinned._resolved, version=version).timestamp())
                return dt.datetime.fromtimestamp(ms / 1000, tz=dt.UTC)
        for entry in pinned._connection._reresolve(pinned).history():
            if entry.get("version") == version and entry.get("timestamp") is not None:
                stamp = entry["timestamp"]
                if isinstance(stamp, (int, float)):
                    return dt.datetime.fromtimestamp(stamp / 1000, tz=dt.UTC)
                return stamp
        raise UnreachableTableError(
            f"read version {version} as a snapshot",
            "its commit timestamp could not be read from the history",
        )

    # -------------------------------------------------------------- metadata

    def schema(self) -> Any:
        """The table's Arrow schema, as a ``pyarrow.Schema`` when pyarrow is installed.

        A VARIANT column is ``string``: reads return it as JSON text on every
        engine, and writes take JSON text (see `deltaswamp._variant`).
        """
        schema, log = self._raw_schema(with_log=True)
        paths = self._variants_in(schema, log)
        try:
            import pyarrow as pa
        except ImportError:
            return schema
        if not isinstance(schema, pa.Schema):
            return schema
        if paths:
            from ._variant import text_schema

            schema = text_schema(pa, schema, paths)
        # A day-time interval reads as a duration and a year-month one as text.
        return interval_schema(pa, schema, interval_paths(log))

    def _interval_plan(self) -> tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]:
        """The table's interval columns (`engine.intervals`), in a form a plan can carry."""
        try:
            groups = interval_paths(self._raw_schema(with_log=True)[1])
        except DeltaSwampError:
            return ()
        return tuple(sorted((group, tuple(sorted(paths))) for group, paths in groups.items()))

    def _variant_paths(self) -> frozenset[tuple[str, ...]]:
        """The table's VARIANT columns (top level and nested in structs), as paths."""
        if not self._enrich().features & _VARIANT_FEATURES:
            return frozenset()
        return self._schema_and_variants()[1]

    def _schema_and_variants(self) -> tuple[Any, frozenset[tuple[str, ...]]]:
        """The raw schema, and where its VARIANT columns are.

        The kernel gives a VARIANT column as a bare ``struct<metadata, value>``
        with no marker of its own, and a real struct of that shape looks the
        same, so the log's schema -- where the type is ``variant`` -- decides.
        Where the log cannot be read here, a top-level column of that shape
        on a table with the variantType feature is taken as one.
        """
        schema, log = self._raw_schema(with_log=True)
        return schema, self._variants_in(schema, log)

    def _variants_in(self, schema: Any, log: Any) -> frozenset[tuple[str, ...]]:
        """Where `schema`'s VARIANT columns are, by the log's schema `log` if known."""
        if not self._enrich().features & _VARIANT_FEATURES:
            return frozenset()
        try:
            import pyarrow as pa
        except ImportError:
            return frozenset()
        from ._variant import is_variant_struct, variant_paths

        if log is not None:
            return variant_paths(log)
        if not isinstance(schema, pa.Schema):
            return frozenset()
        return frozenset((f.name,) for f in schema if is_variant_struct(pa, f.type))

    def _raw_schema(self, with_log: bool = False) -> Any:
        """The schema as the engines read and write it (VARIANT as its binary struct).

        With `with_log`, a pair: the schema and the log's own schema (parsed
        `schemaString`), or None where the serving engine does not expose it.
        """
        pinned = self._version is not None
        # Learning the schema reads the log, not a data file, so a table whose
        # VARIANT files are shredded still answers it directly.
        engine = self._engine(
            Operation.TIME_TRAVEL if pinned else Operation.SCAN, frozenset({"variant_free"})
        )
        snapshot = getattr(engine, "snapshot", None)
        if snapshot is not None:
            snap = snapshot(self._resolved, version=self._version)
            schema = snap.schema()
        else:
            # Open the stream and take its schema; reading the whole table to
            # learn its columns cost a full scan on every count().
            schema = self.scan(limit=0)
        log = _log_schema(engine, self._resolved, self._version) if with_log else None
        try:
            import pyarrow as pa
        except ImportError:
            return (schema, log) if with_log else schema
        # The kernel hands back an arro3 Schema; every other engine a pyarrow
        # one. Return one type, whichever engine served it.
        if not isinstance(schema, pa.Schema):
            if hasattr(schema, "__arrow_c_stream__"):
                schema = pa.RecordBatchReader.from_stream(schema).schema
            else:
                schema = pa.schema(schema)
        return (schema, log) if with_log else schema

    def _current(self) -> ResolvedTable:
        """Protocol state as the log has it now (as pinned, for a pinned handle).

        The cached enrichment is only refreshed by this handle's own commits,
        so properties() kept answering what was true before another writer's
        ALTER for the life of the handle -- schema() and version already read
        the log every time.
        """
        if self._version is None:
            self._enriched = False
        return self._enrich()

    def protocol(self) -> tuple[int | None, int | None]:
        """``(min_reader_version, min_writer_version)``."""
        r = self._current()
        return (r.min_reader_version, r.min_writer_version)

    def features(self) -> frozenset[str]:
        """Every table feature in force, including those a legacy protocol implies.

        A protocol below reader 3 / writer 7 names no features: its version
        number is the feature set. Databricks' DESCRIBE DETAIL reports the
        implied ones (a (1, 2) table lists appendOnly and invariants), and so
        does this, so the two agree on every table.
        """
        r = self._current()
        return r.effective_reader_features | r.effective_writer_features

    def properties(self) -> dict[str, str]:
        """The table's properties (``delta.*`` and any others), as the log holds them."""
        return dict(self._current().properties)

    @property
    def version(self) -> int | None:
        """The current version; None for an Iceberg table with no snapshot yet."""
        value = self.detail().get("version")
        return None if value is None else int(value)

    # ------------------------------------------------------------------ write

    @staticmethod
    def _write_needs(
        schema_mode: str | None,
        commit_metadata: dict[str, Any] | None,
        txn: tuple[str, int] | None,
        writer_properties: Any,
        partition_overwrite: str,
    ) -> frozenset[str]:
        """Translate write arguments into engine capabilities they require."""
        needs: set[str] = set()
        if schema_mode == "merge":
            needs.add("schema_merge")
        elif schema_mode == "overwrite":
            needs.add("schema_overwrite")
        if commit_metadata is not None:
            needs.add("commit_metadata")
        if txn is not None:
            needs.add("idempotent_txn")
        if writer_properties is not None:
            needs.add("writer_properties")
        if partition_overwrite == "dynamic":
            needs.add("dynamic_overwrite")
        return frozenset(needs)

    def _identity_needs(self, data: Any, clauses: list[str] | None = None) -> frozenset[str]:
        """``identity_insert`` / ``identity_update`` when the data sets an identity column.

        Delta refuses a value given for a GENERATED ALWAYS AS IDENTITY column
        (DELTA_IDENTITY_COLUMNS_EXPLICIT_INSERT_NOT_SUPPORTED), and a MERGE
        ``UPDATE SET *`` that would assign any identity column
        (DELTA_IDENTITY_COLUMNS_UPDATE_NOT_SUPPORTED). can() said yes to both
        and the warehouse then failed the statement. `clauses` are a MERGE's
        builder methods; without them the data is an append's or overwrite's.
        """
        names = _data_column_names(data)
        if not names or "identityColumns" not in self._enrich().writer_features:
            return frozenset()
        try:
            import pyarrow as pa

            schema = self.schema()
        except (ImportError, DeltaSwampError):
            return frozenset()
        if not isinstance(schema, pa.Schema):
            return frozenset()
        given = {n.lower() for n in names}
        always, identity = set(), set()
        for field in schema:
            metadata = field.metadata or {}
            if field.name.lower() not in given:
                continue
            if any(k.startswith(b"delta.identity.") for k in metadata):
                identity.add(field.name)
                if metadata.get(b"delta.identity.allowExplicitInsert", b"").lower() != b"true":
                    always.add(field.name)
        needs = set()
        inserts = clauses is None or "when_not_matched_insert_all" in clauses
        if always and inserts:
            needs.add("identity_insert")
        if identity and clauses is not None and "when_matched_update_all" in clauses:
            needs.add("identity_update")
        return frozenset(needs)

    def _data_needs(self, data: Any, partition_by: list[str] | None = None) -> frozenset[str]:
        """Engine capabilities the rows themselves require.

        delta-rs formats a negative decimal partition value with a fractional
        part as ``-1.-50`` (and -0.5 as ``0.-50``), commits that corrupt
        partition value, and only then fails re-reading it: the table is left
        unreadable. Such writes go to an engine that formats them correctly.
        A stream cannot be inspected without consuming it, so one aimed at a
        fractional-decimal partition column is treated as if it held one.
        """
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
        except ImportError:
            return frozenset()
        try:
            parts = list(partition_by or self._enrich().partition_columns)
        except DeltaSwampError:
            return frozenset()
        if not parts:
            return frozenset()
        schema = getattr(data, "schema", None)
        if not isinstance(schema, pa.Schema):
            try:
                schema = self.schema()
            except DeltaSwampError:
                return frozenset()
        wanted = {p.lower() for p in parts}
        risky = [
            f.name
            for f in schema
            if f.name.lower() in wanted and pa.types.is_decimal(f.type) and f.type.scale > 0
        ]
        if not risky:
            return frozenset()
        module = type(data).__module__ or ""
        if module.startswith(("pandas", "polars")) and type(data).__name__ == "DataFrame":
            try:
                data = pa.table(data)  # in memory, so inspecting it consumes nothing
            except Exception:
                return frozenset({"negative_decimal_partition_values"})
        if not isinstance(data, (pa.Table, pa.RecordBatch)):
            return frozenset({"negative_decimal_partition_values"})
        if not all(name in data.schema.names for name in risky):
            return frozenset({"negative_decimal_partition_values"})
        for name in risky:
            col = data.column(name)
            negative = pc.less(col, pa.scalar(0, col.type)).fill_null(False)
            if pc.any(negative).as_py():
                return frozenset({"negative_decimal_partition_values"})
        return frozenset()

    @staticmethod
    def _expression_needs(
        predicate: str | None, updates: dict[str, Any] | None = None
    ) -> frozenset[str]:
        """``sql_expressions`` when DML SQL goes beyond the kernel's grammar.

        The kernel evaluates DELETE/UPDATE/replaceWhere SQL itself and reads
        only comparisons, IN, BETWEEN, LIKE, IS NULL and AND/OR/NOT over columns
        and literals, with a SET value a literal or a column. Arithmetic or a
        function call (`id % 3 = 0`, `lower(s) = 'a'`, `x + 1`) goes to an engine
        that evaluates SQL instead of failing on the kernel mid-call.
        """
        from . import predicate as sqlpred
        from .engine.dialect import warehouse_reason

        texts = [predicate, *(updates.values() if isinstance(updates, dict) else ())]
        for text in texts:
            # Refused before any engine sees it: DataFusion reads up to a ';'
            # or a stray ')' and ignores the rest, so "id = 1) AND (s = 'x'"
            # deleted every id = 1 row and reported success.
            _check_sql_fragment(text, "the predicate" if text is predicate else "the SET value")
        if any(warehouse_reason(t) is not None for t in texts):
            # Spark SQL neither direct engine can be made to compute as
            # Databricks does (a 0-based array subscript, `split`).
            return frozenset({"sql_expressions", "spark_sql"})
        try:
            if isinstance(predicate, str) and predicate.strip():
                sqlpred.parse(predicate)
            if isinstance(updates, dict):
                for value in updates.values():
                    if not isinstance(value, str):
                        continue
                    parsed = sqlpred.parse_value(value)
                    if isinstance(parsed, sqlpred.Column) and len(parsed.path) > 1:
                        # The kernel's UPDATE reads a top-level column only:
                        # SET v = st.x was refused mid-call after can() had
                        # named the kernel. delta-rs evaluates it.
                        return frozenset({"sql_expressions"})
        except sqlpred.PredicateError as exc:
            if exc.beyond_grammar or all(_parses_as_spark(t) for t in texts):
                # Spark's grammar (`dialect`) reads it: SQL an engine that
                # evaluates SQL serves, though the kernel's parser does not.
                return frozenset({"sql_expressions"})
            # Malformed in any SQL (`id ===`): no engine serves it. Handing it
            # on as "no needs" let delta-rs read a prefix of it and act on
            # more rows; the kernel's parser names the mistake here instead.
            raise
        return frozenset()

    def _update_needs(self, targets: Any, literal: bool) -> frozenset[str]:
        """`_data_needs` for UPDATE's SET list.

        Setting a fractional-decimal partition column moves rows into a new
        partition, whose value delta-rs would format as ``-1.-50``. A literal is
        judged by its sign; a SQL expression could be anything, so it counts.
        """
        if not targets or not isinstance(targets, dict):
            return frozenset()
        try:
            import pyarrow as pa

            parts = {p.lower() for p in self._enrich().partition_columns}
            schema = self.schema() if parts else None
        except (ImportError, DeltaSwampError):
            return frozenset()
        if not parts or schema is None:
            return frozenset()
        risky = {
            f.name.lower()
            for f in schema
            if f.name.lower() in parts and pa.types.is_decimal(f.type) and f.type.scale > 0
        }
        for key, value in targets.items():
            if not isinstance(key, str):
                continue  # _update_targets refuses it
            bare = key[1:-1] if len(key) > 1 and key[0] == key[-1] == "`" else key
            if bare.lower() not in risky:
                continue
            if not literal:
                return frozenset({"negative_decimal_partition_values"})
            try:
                negative = value is not None and float(value) < 0
            except (TypeError, ValueError):
                negative = True
            if negative:
                return frozenset({"negative_decimal_partition_values"})
        return frozenset()

    def append(
        self,
        data: Any,
        *,
        schema_mode: str | None = None,
        partition_by: list[str] | None = None,
        target_file_size: int | None = None,
        writer_properties: Any = None,
        commit_metadata: dict[str, Any] | None = None,
        txn: tuple[str, int] | None = None,
        max_commit_retries: int | None = None,
    ) -> _results.OperationResult:
        """Append data. Returns an `OperationResult`: the `version` committed and
        the `num_files`, `num_rows` and `num_bytes` it added, plus `engine`.

        `schema_mode="merge"` widens the table schema to fit the data, and
        routes as MERGE_SCHEMA rather than a plain append.

        `txn=(app_id, version)` makes the write idempotent. Note that neither
        engine enforces this: delta-rs records the transaction identifier but
        happily appends the same one twice. So the check happens here, by
        comparing against the last committed version before writing. That
        removes the common replay case; it is not a substitute for engine-level
        enforcement, because a concurrent writer could still commit in between.
        """
        if self._version is not None and schema_mode is None:
            # A blind append reads nothing, so it means the same from any
            # handle: rows added at the latest version. Aligned to the latest
            # schema, not the pinned one.
            return Table(self._connection, self._resolved).append(
                data,
                partition_by=partition_by,
                target_file_size=target_file_size,
                writer_properties=writer_properties,
                commit_metadata=commit_metadata,
                txn=txn,
                max_commit_retries=max_commit_retries,
            )
        self._check_writable("append")
        _check_write_sizes(target_file_size, max_commit_retries)
        if schema_mode not in (None, "merge"):
            raise InvalidArgumentError(
                f"append takes schema_mode=None or 'merge', not {schema_mode!r}; replacing "
                "the schema is overwrite(..., schema_mode='overwrite')"
            )
        _check_txn(txn)
        data = _write_data(data)
        if txn is not None and self._already_committed(txn):
            return _results.nothing_written()
        raw = data
        self._check_cdf_columns(data, schema_mode, "append")
        outcome: dict[str, Any] = {}
        for attempt in range(_REALIGN_ATTEMPTS):
            data = self._align(raw, schema_mode)
            request = self._request(
                Operation.APPEND,
                {
                    "schema_mode": schema_mode,
                    "partition_by": partition_by,
                    "target_file_size": target_file_size,
                    "writer_properties": writer_properties,
                    "commit_metadata": commit_metadata,
                    "txn": txn,
                    "max_commit_retries": max_commit_retries,
                },
                data,
            )

            def write(data: Any = data, request: Request = request) -> None:
                engine = self._route(request)
                outcome["engine"] = getattr(engine, "kind", None)
                outcome["raw"] = engine.append(
                    self._resolved,
                    self._variant_input(engine, data),
                    schema_mode=schema_mode,
                    partition_by=partition_by,
                    target_file_size=target_file_size,
                    writer_properties=writer_properties,
                    commit_metadata=commit_metadata,
                    txn=txn,
                    max_commit_retries=max_commit_retries,
                )

            try:
                self._raced_append(write, data, txn, max_commit_retries)
                break
            except Exception as exc:
                # A column added by another writer between lining the batch up
                # and delta-rs opening the table fails with "number of fields
                # does not match": nothing was committed, so line it up with
                # the new schema and write again. So does a blind append that
                # delta-rs refused because a concurrent commit changed the
                # metadata: it does not rebase over one, where the kernel does.
                if attempt + 1 >= _REALIGN_ATTEMPTS or not (
                    _lost_to_metadata_change(exc, raw)
                    or self._schema_moved(exc, raw, data, schema_mode)
                ):
                    raise
        self._invalidate()
        return self._write_result(outcome)

    def _write_result(self, outcome: dict[str, Any]) -> _results.OperationResult:
        """The result of the append or overwrite `outcome` records (see `_results.write`)."""
        if "raw" not in outcome:
            return _results.nothing_written()  # a lost race won by this very txn
        raw = outcome["raw"]
        version = raw if isinstance(raw, int) and not isinstance(raw, bool) else None
        if isinstance(raw, dict):
            version = raw.get("version")
        engine = outcome.get("engine")
        totals = None if version is None else self._commit_totals(engine, version)
        return _results.write(raw, engine, totals)

    def _commit_totals(self, kind: Any, version: int) -> dict[str, Any] | None:
        """What commit `version` added and removed, read back from its log file.

        Neither engine reports it: delta-rs returns nothing, the kernel the
        version (delta-rs#3952). None when the commit cannot be read (a
        catalog's unpublished commit, an engine that cannot read one).
        """
        engine = self._connection.router.engines.get(kind) if kind is not None else None
        read = getattr(engine, "commit_text", None)
        if read is None or version < 1:
            return None
        try:
            text = read(self._resolved, int(version))
            actions = [json.loads(line) for line in text.splitlines() if line.strip()]
        except Exception:
            return None
        totals: dict[str, Any] = {"num_files": 0, "num_bytes": 0, "num_removed_files": 0}
        rows: int | None = 0
        for action in actions:
            add, remove = action.get("add"), action.get("remove")
            if add is not None and add.get("dataChange", True):
                totals["num_files"] += 1
                totals["num_bytes"] += int(add.get("size") or 0)
                try:
                    records = json.loads(add.get("stats") or "{}").get("numRecords")
                except (ValueError, AttributeError):
                    records = None
                rows = None if rows is None or records is None else rows + int(records)
            elif remove is not None and remove.get("dataChange", True):
                totals["num_removed_files"] += 1
        totals["num_rows"] = rows
        return totals

    def _check_cdf_columns(self, data: Any, schema_mode: str | None, what: str) -> None:
        """Refuse a schema-evolving write that adds a column the change feed reserves.

        add_column refused `_change_type` on a CDF table, but a write with
        schema_mode= added it, and every DML and feed read failed afterwards.
        """
        if schema_mode is None:
            return
        schema = getattr(data, "schema", None)
        names = getattr(schema, "names", None)
        if names is None or callable(names):
            return
        current = {n.lower() for n in self.schema().names} if schema_mode == "merge" else set()
        clash = cdf_name_clash(
            [n for n in names if n.lower() not in current], self._enrich().properties
        )
        if clash:
            raise cdf_clash_error(f"{what} with schema_mode={schema_mode!r}", clash)

    def _raced_append(
        self, write: Any, data: Any, txn: tuple[str, int] | None, retries: int | None
    ) -> None:
        """An append to a catalog-managed table, re-staged when it loses a race.

        A blind append commutes with any concurrent commit, and the engine
        re-stages one on a path table -- but it cannot re-read a catalog tail,
        so a catalog-managed append that lost the race by a millisecond failed
        outright. Here the tail is re-read and the append tried again.
        """
        from .errors import CommitConflictError

        attempts = 1 + max(0, 5 if retries is None else int(retries))
        for attempt in range(attempts):
            try:
                self._backfilled(write, data)
                return
            except CommitConflictError:
                if (
                    txn is not None
                    and not self._resolved.is_catalog_managed
                    and self._txn_landed(txn)
                ):
                    # The race was lost to this very batch: exactly-once is met,
                    # just as when the check before the write finds it.
                    return
                if (
                    not self._resolved.is_catalog_managed
                    or attempt + 1 >= attempts
                    or _consumable(data)
                ):
                    raise
                self._refresh_commit_tail(before_write=True)
                if txn is not None and self._already_committed(txn):
                    return  # the winner was this very batch

    def overwrite(
        self,
        data: Any,
        *,
        predicate: str | None = None,
        partition_overwrite: str = "static",
        schema_mode: str | None = None,
        target_file_size: int | None = None,
        writer_properties: Any = None,
        commit_metadata: dict[str, Any] | None = None,
        txn: tuple[str, int] | None = None,
        max_commit_retries: int | None = None,
    ) -> _results.OperationResult:
        """Replace data. Returns an `OperationResult`, as `append` does (with
        `num_removed_files` too).

        With `predicate`, replaces only matching rows (`replaceWhere`). With
        `partition_overwrite="dynamic"`, replaces exactly the partitions present
        in `data` and leaves the rest alone, which is Spark's
        `partitionOverwriteMode=dynamic`.
        """
        self._check_writable("overwrite")
        _check_write_sizes(target_file_size, max_commit_retries)
        if schema_mode not in (None, "merge", "overwrite"):
            raise InvalidArgumentError(
                f"schema_mode must be None, 'merge' or 'overwrite', not {schema_mode!r}"
            )
        if partition_overwrite not in ("static", "dynamic"):
            # It reached the router and came back as "no engine can overwrite
            # with partition_overwrite='Dynamic'", as if the table were at fault.
            raise InvalidArgumentError(
                f"partition_overwrite must be 'static' or 'dynamic', not {partition_overwrite!r}"
            )
        if partition_overwrite == "dynamic" and predicate is not None:
            # Contradictory arguments, which the engines refused as "cannot
            # be served" as if the table were at fault.
            raise InvalidArgumentError(
                "partition_overwrite='dynamic' derives its own predicate from the data, so "
                "it cannot be combined with predicate="
            )
        _check_predicate(predicate, "overwrite")
        _check_txn(txn)
        data = _write_data(data)
        if txn is not None and self._already_committed(txn):
            return _results.nothing_written()
        if (
            partition_overwrite == "dynamic"
            and predicate is None
            and getattr(data, "num_rows", None) == 0
            and hasattr(data, "column_names")
        ):
            # Spark's dynamic mode replaces the partitions present in the
            # data; an empty batch names none, so it changes nothing. It used
            # to raise, failing any pipeline whose batch happened to be empty.
            return _results.nothing_written()
        raw = data
        self._check_cdf_columns(data, schema_mode, "overwrite")
        data = self._align(raw, schema_mode)
        options = {
            "predicate": predicate,
            "partition_overwrite": partition_overwrite,
            "schema_mode": schema_mode,
            "target_file_size": target_file_size,
            "writer_properties": writer_properties,
            "commit_metadata": commit_metadata,
            "txn": txn,
            "max_commit_retries": max_commit_retries,
        }
        from .errors import CommitConflictError

        outcome: dict[str, Any] = {}
        for attempt in range(_REALIGN_ATTEMPTS):
            request = self._request(Operation.OVERWRITE, options, data)
            refused = refusal(self, request)
            if refused is not None:
                # Before anything is written, as can() refuses it.
                raise UnreachableTableError(
                    "replace the table's schema", refused.reason or "", refused.remedy
                )

            def write(data: Any = data, request: Request = request) -> None:
                engine = self._route(request)
                outcome["engine"] = getattr(engine, "kind", None)
                outcome["raw"] = engine.overwrite(
                    self._resolved,
                    self._variant_input(engine, data),
                    predicate=predicate,
                    partition_overwrite=partition_overwrite,
                    schema_mode=schema_mode,
                    target_file_size=target_file_size,
                    writer_properties=writer_properties,
                    commit_metadata=commit_metadata,
                    txn=txn,
                    max_commit_retries=max_commit_retries,
                )

            try:
                self._backfilled(write, data)
                break
            except CommitConflictError:
                # Lost to a writer that committed this very txn: already done.
                if txn is None or not self._txn_landed(txn):
                    raise
                outcome.pop("raw", None)
                break
            except Exception as exc:
                # A concurrent ADD COLUMN between aligning and writing: nothing
                # was committed, so align with the new schema and go again.
                if attempt + 1 >= _REALIGN_ATTEMPTS or not self._schema_moved(
                    exc, raw, data, schema_mode
                ):
                    raise
                data = self._align(raw, schema_mode)
        self._invalidate()
        return self._write_result(outcome)

    def _backfilled(self, write: Any, data: Any = None) -> Any:
        """Run a commit; on a catalog's backfill demand, publish and retry once.

        The 429 is not a rate limit: the catalog holds no more unpublished
        commits until someone publishes, and every write through this library
        failed there until the caller found `publish()` for themselves. The
        refused commit changed nothing, so publishing and committing again is
        safe -- when the data can be read a second time. A stream cannot, so
        the table is still published and the error says to write again.
        """
        from .errors import BackfillRequiredError

        try:
            return write()
        except CorruptTableError as exc:
            if getattr(exc, "stale_checkpoint_hint", False):
                self._forget_snapshots()
            raise
        except BackfillRequiredError as exc:
            if not self._resolved.is_catalog_managed:
                raise
            try:
                self._engine(Operation.PUBLISH).publish(self._resolved)
            except DeltaSwampError:
                raise exc from None
            self._refresh_commit_tail()
            if data is not None and _consumable(data):
                raise BackfillRequiredError(
                    f"{exc}. The table's commits have now been published; the data was a "
                    "stream this call has consumed, so write it again"
                ) from exc
            return write()

    def replace(self, data: Any, **kwargs: Any) -> _results.OperationResult:
        """Replace the table's contents and schema. REPLACE TABLE / RTAS."""
        _check_options("replace", kwargs, _REPLACE_OPTIONS)
        return self.overwrite(data, schema_mode="overwrite", **kwargs)

    def _already_committed(self, txn: tuple[str, int]) -> bool:
        """True if `txn` was already committed, so the write should be skipped.

        Neither engine deduplicates on its own: delta-rs 1.6.5 records the txn
        action and appends anyway.
        """
        app_id, version = txn
        # A missing capability (no engine can read transaction ids) raises
        # out of txn_version: treating it as "never committed" silently
        # dropped the dedup and let a replayed batch append twice.
        last = self.txn_version(app_id)
        return last is not None and version <= last

    def _schema_moved(self, exc: Exception, raw: Any, aligned: Any, schema_mode: Any) -> bool:
        """Whether `exc` is a schema mismatch caused by a concurrent schema change."""
        if type(engine_cause(exc)).__name__ != "SchemaMismatchError":
            return False
        before = getattr(aligned, "schema", None)
        if before is None:
            return False
        self._invalidate()
        try:
            again = self._align(raw, schema_mode)
        except Exception:
            return False
        return bool(getattr(again, "schema", None) != before)

    def _txn_landed(self, txn: tuple[str, int]) -> bool:
        """After a lost race: whether the winner committed this txn. False if unknown."""
        try:
            return self._already_committed(txn)
        except DeltaSwampError:
            return False

    def txn_version(self, app_id: str) -> int | None:
        """The last version committed under `app_id`, or None if never.

        Pair with `txn=` on a write to make a pipeline exactly-once::

            if t.txn_version("nightly-load") != batch_id:
                t.append(data, txn=("nightly-load", batch_id))
        """
        if not isinstance(app_id, str) or not app_id:
            # delta-rs raised a bare TypeError for None.
            raise InvalidArgumentError(f"app_id must be a non-empty string, not {app_id!r}")
        engine: Any = None
        try:
            engine = self._engine(Operation.APPEND, frozenset({"idempotent_txn"}))
        except DeltaSwampError:
            with contextlib.suppress(DeltaSwampError):
                engine = self._engine(Operation.APPEND)
        if callable(getattr(engine, "txn_version", None)):
            version: int | None = engine.txn_version(self._resolved, app_id)
            return version
        # The writing engine cannot read transaction ids (the warehouse), or
        # nothing here writes the table; reading them needs only the log. The
        # kernel first: it reads a catalog-managed table with the catalog's
        # commit tail, which delta-rs cannot open at all, so txn_version()
        # there failed.
        reason = (
            # engine.kind, not type(): that named the error boundary wrapping it.
            f"the {getattr(getattr(engine, 'kind', None), 'value', 'routed')} engine cannot "
            "read transaction identifiers"
            if engine is not None
            else "no engine here writes this table"
        )
        for kind in (EngineKind.KERNEL, EngineKind.DELTARS):
            fallback: Any = self._connection.router.engines.get(kind)
            if fallback is None or not callable(getattr(fallback, "txn_version", None)):
                continue
            available = getattr(fallback, "available", None)
            if available is not None and not available():
                continue
            try:
                version = fallback.txn_version(self._resolved, app_id)
                return version
            except Exception as exc:
                reason += f", and {kind.value} could not read this table's log ({exc})"
        raise UnreachableTableError(
            f"check transaction {app_id!r}",
            reason,
            "write without txn= and deduplicate yourself, or use a table the kernel or "
            "delta-rs can read",
        )

    def delete(self, predicate: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """DELETE rows matching a SQL predicate (every row when None)."""
        self._check_writable("delete", pinned=True)
        _check_options("delete", kwargs, _DML_OPTIONS)
        _check_predicate(predicate, "delete")
        request = self._request(Operation.DELETE, {"predicate": predicate, **kwargs})
        pinned = self._check_pinned(request)
        served: list[Any] = []

        def run() -> Any:
            engine = self._route(request)
            served.append(getattr(engine, "kind", None))
            return engine.delete(self._resolved, predicate, **_given(kwargs), **pinned)

        result = self._pinned_write(lambda: self._backfilled(run))
        self._invalidate()
        return _results.dml(result, served[-1] if served else None)

    def update(
        self,
        updates: dict[str, str] | None = None,
        *,
        new_values: dict[str, Any] | None = None,
        predicate: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """UPDATE. `updates` maps columns to SQL expressions; `new_values` to
        plain Python values, which need no quoting."""
        self._check_writable("update", pinned=True)
        _check_options("update", kwargs, _DML_OPTIONS | {"error_on_type_mismatch"})
        _check_predicate(predicate, "update")
        if updates is not None and new_values is not None:
            raise InvalidArgumentError("pass updates (SQL expressions) or new_values, not both")
        for given, name in ((updates, "updates"), (new_values, "new_values")):
            if given is not None and not isinstance(given, Mapping):
                raise InvalidArgumentError(
                    f"{name} maps each column to its new value, e.g. {{'v': \"'a'\"}}; "
                    f"got a {type(given).__name__}"
                )
        if not updates and not new_values:
            raise InvalidArgumentError("update needs at least one column to set")
        updates, new_values = _dotted_keys(updates), _dotted_keys(new_values)
        request = self._request(
            Operation.UPDATE,
            {"updates": updates, "new_values": new_values, "predicate": predicate, **kwargs},
        )
        pinned = self._check_pinned(request)
        engine = self._route(request)
        defaults = self._update_defaults(updates)[0]
        if defaults and isinstance(engine, (KernelEngine, DeltaRsEngine)):
            # The warehouse reads `DEFAULT` itself; a direct engine gets the
            # column's literal DEFAULT (or NULL) spelled out.
            updates = {k: defaults.get(k, v) for k, v in (updates or {}).items()}
        # delta-rs skips a SET target it cannot find -- an unknown name, a
        # different case, a nested field -- and still rewrites every matched
        # file, reporting the rows as updated while changing nothing.
        deltars = isinstance(engine, DeltaRsEngine)
        if updates is not None:
            updates = self._update_targets(updates, deltars)
        if new_values is not None:
            kwargs["new_values"] = self._update_targets(new_values, deltars)
            self._refuse_zoned_ntz(kwargs["new_values"])
        result: dict[str, Any] = self._pinned_write(
            lambda: self._backfilled(
                lambda: engine.update(
                    self._resolved, updates=updates, predicate=predicate, **_given(kwargs), **pinned
                )
            )
        )
        self._invalidate()
        return _results.dml(result, getattr(engine, "kind", None))

    def _update_defaults(self, updates: Any) -> tuple[dict[str, str], frozenset[str]]:
        """`SET c = DEFAULT`: each such column's DEFAULT as SQL, for a direct engine.

        Spark sets the column to its DEFAULT (NULL for a column without one).
        Read as SQL, `DEFAULT` was a column name, and the kernel refused "no
        column 'DEFAULT'". A literal DEFAULT is spelled out, as `append` fills
        one in; an expression (`current_timestamp()`) is Databricks' to
        evaluate, so the UPDATE needs ``sql_column_defaults``.
        """
        if not isinstance(updates, dict):
            return {}, frozenset()
        keys = [
            k
            for k, v in updates.items()
            if isinstance(k, str) and isinstance(v, str) and v.strip().upper() == "DEFAULT"
        ]
        if not keys:
            return {}, frozenset()
        try:
            import pyarrow as pa

            fields = {f.name.lower(): f for f in self.schema()}
        except (ImportError, DeltaSwampError):
            return {}, frozenset()
        if "default" in fields:
            # A column named `default` wins over the keyword, as on Databricks.
            return {}, frozenset()
        out: dict[str, str] = {}
        needs: set[str] = set()
        for key in keys:
            quoted = len(key) > 1 and key[0] == key[-1] == "`"
            field = fields.get((key[1:-1].replace("``", "`") if quoted else key).lower())
            if field is None:
                continue  # _update_targets names the unknown column
            text = _column_default(field)
            if text is None:
                out[key] = "NULL"
            elif _default_column(pa, field, 1) is not None:
                out[key] = text
            else:
                needs.add("sql_column_defaults")
        return out, frozenset(needs)

    def _refuse_zoned_ntz(self, values: dict[str, Any]) -> None:
        """Refuse a timezone-aware datetime for a TIMESTAMP_NTZ column.

        It was turned into its UTC wall time without a word: 07:08 at +02:00
        was stored as 05:08. A TIMESTAMP_NTZ holds no zone, so which wall time
        was meant is the caller's to say.
        """
        import datetime

        try:
            import pyarrow as pa

            fields = {f.name: f for f in self.schema()}
        except (ImportError, DeltaSwampError):
            return
        for name, value in values.items():
            field = fields.get(name.strip("`"))
            if (
                field is not None
                and isinstance(value, datetime.datetime)
                and value.tzinfo is not None
                and pa.types.is_timestamp(field.type)
                and field.type.tz is None
            ):
                raise InvalidArgumentError(
                    f"update {name}: the column is TIMESTAMP_NTZ, which holds no time zone, "
                    f"and {value.isoformat()} carries one; pass a naive datetime (for "
                    "example value.replace(tzinfo=None), or the value converted to the zone "
                    "you mean first)"
                )

    def _update_targets(self, targets: dict[str, Any], deltars: bool) -> dict[str, Any]:
        """Resolve UPDATE's SET targets against the table's columns.

        One rule for every engine: a key names a top-level column when one is
        spelled exactly so (`a.b` is the column called that, when there is
        one; backticks around the key are optional), else case-insensitively;
        only then is a dotted key a nested field path. A name that matches
        nothing is refused; so is a nested path on delta-rs, which cannot set
        struct fields.

        The kernel takes the column's name as it is. delta-rs parses a key as
        SQL, so a name that is not a plain identifier goes to it backticked:
        bare, `a.b` was read as field `b` of a struct `a`, which it skipped
        while reporting the rows updated.
        """
        strict = deltars
        if not isinstance(targets, dict):
            raise InvalidArgumentError(
                f"update targets must be a {{column: value}} mapping, not {type(targets).__name__}"
            )
        names = list(self.schema().names)
        folded: dict[str, list[str]] = {}
        for name in names:
            folded.setdefault(name.lower(), []).append(name)
        resolved: dict[str, Any] = {}
        for key, value in targets.items():
            if not isinstance(key, str):
                raise InvalidArgumentError(f"update column names must be strings, got {key!r}")
            quoted = len(key) > 1 and key[0] == key[-1] == "`"
            bare = key[1:-1].replace("``", "`") if quoted else key
            column: str | None = None
            if bare in names:
                column = bare
            elif len(folded.get(bare.lower(), ())) == 1:
                column = folded[bare.lower()][0]
            if column is not None:
                target = column
                if deltars and not _PLAIN_IDENTIFIER.fullmatch(column):
                    target = "`" + column.replace("`", "``") + "`"
            elif "." in bare and not quoted and not strict:
                target = key  # a struct field path; the warehouse resolves it
            else:
                hint = (
                    "; delta-rs cannot set a field inside a struct, set the whole column"
                    if "." in bare
                    else ""
                )
                raise InvalidArgumentError(
                    f"update: the table has no column {key!r}{hint}; columns are {names}"
                )
            if target in resolved:
                raise InvalidArgumentError(f"update sets the column {target!r} twice")
            resolved[target] = value
        return resolved

    def merge(self, source: Any, predicate: str, **kwargs: Any) -> Any:
        """MERGE INTO. Returns a builder with the delta-rs clause API
        (``when_matched_update_all()`` ... ``execute()``) whichever engine serves it."""
        self._check_writable("merge", pinned=True)
        _check_options("merge", kwargs, _MERGE_OPTIONS)
        if predicate is None:
            raise InvalidArgumentError("merge needs a join predicate")
        _check_predicate(predicate, "merge")
        _check_sql_fragment(predicate, "the MERGE ON condition")
        source = _write_data(source)
        request = self._request(Operation.MERGE, {"predicate": predicate, **kwargs}, source)
        pinned = self._check_pinned(request)

        routed = {"request": request}

        def build(exclude: frozenset[EngineKind]) -> tuple[Any, EngineKind | None]:
            engine = self._route(routed["request"], exclude=exclude)
            builder = engine.merge(
                self._resolved, self._variant_input(engine, source), predicate, **kwargs, **pinned
            )
            return builder, getattr(engine, "kind", None)

        def clauses_routed(clauses: list[tuple[Any, ...]]) -> EngineKind | None:
            # The clauses are known only at execute(), and they can add a need
            # (an UPDATE or DELETE clause removes rows, which an append-only
            # table forbids; a SET value may be an early date delta-rs must
            # not write). Routed again with them, before anything runs,
            # exactly as can("merge", ..., clauses=[...]) answers; the engine
            # that serves the clauses is returned for the builder to move to.
            aliases: dict[str, Any] = {
                k: kwargs[k] for k in ("source_alias", "target_alias") if k in kwargs
            }
            extra = self._request(Operation.MERGE, {**aliases, "clauses": clauses}, source).needs
            if extra <= request.needs:
                return None
            widened = dataclasses.replace(request, needs=request.needs | extra)
            engine = self._route(widened)
            routed["request"] = widened
            return getattr(engine, "kind", None)

        builder, kind = build(frozenset())
        # A consumed stream cannot be offered to a second engine.
        rebuild = None if _consumable(source) else build
        return _InvalidatingMerger(builder, self._invalidate, rebuild, kind, clauses_routed)

    # ------------------------------------------------------------ maintenance

    def optimize(
        self,
        *,
        zorder_by: list[str] | str | None = None,
        full: bool = False,
        predicate: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """OPTIMIZE: compaction, or Z-ordering with `zorder_by`.

        `full=True` is OPTIMIZE ... FULL, reclustering every file of a
        liquid-clustered table. `predicate` scopes the work (SQL fallback);
        delta-rs takes ``partition_filters=`` instead.
        """
        self._check_writable("optimize")
        _check_options("optimize", kwargs, _OPTIMIZE_OPTIONS)
        self._check_optimize_args(kwargs)
        if isinstance(zorder_by, str):
            zorder_by = [zorder_by]
        if zorder_by:
            self._check_zorder(list(zorder_by))
        sort_by = kwargs.get("sort_by")
        if isinstance(sort_by, str) and sort_by:
            self._check_zorder([sort_by])
        elif isinstance(sort_by, list | tuple) and sort_by:
            self._check_zorder(list(sort_by))
        engine = self._route(
            self._request(
                Operation.OPTIMIZE,
                {"zorder_by": zorder_by, "full": full, "predicate": predicate, **kwargs},
            )
        )
        result = engine.optimize(
            self._resolved, zorder_by=zorder_by, full=full, predicate=predicate, **kwargs
        )
        self._invalidate()
        return _results.optimize(result, getattr(engine, "kind", None))

    def z_order(self, columns: list[str] | str, **kwargs: Any) -> dict[str, Any]:
        """OPTIMIZE ZORDER BY `columns`; the same as ``optimize(zorder_by=columns)``."""
        self._check_writable("z-order")
        _check_options("z_order", kwargs, _OPTIMIZE_OPTIONS)
        self._check_optimize_args(kwargs)
        columns = [columns] if isinstance(columns, str) else list(columns or [])
        if not columns:
            raise InvalidArgumentError("z_order needs at least one column")
        self._check_zorder(columns)
        engine = self._route(self._request(Operation.ZORDER, {"zorder_by": columns, **kwargs}))
        result = engine.zorder(self._resolved, columns, **kwargs)
        self._invalidate()
        return _results.optimize(result, getattr(engine, "kind", None))

    def _check_optimize_args(self, kwargs: dict[str, Any]) -> None:
        """Refuse OPTIMIZE tuning values delta-rs mishandles, before any work starts.

        delta-rs waits forever for a task slot with max_concurrent_tasks=0,
        raises a bare OverflowError for -1 and ValueError for target_size=0,
        and treats a filter on a non-partition column as matching nothing:
        a "successful" OPTIMIZE that compacted 0 of 0 files.
        """
        for name in ("max_concurrent_tasks", "target_size"):
            value = kwargs.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise InvalidArgumentError(f"{name} must be a positive integer, got {value!r}")
        filters = kwargs.get("partition_filters")
        if filters is None:
            return
        if not isinstance(filters, list) or not all(
            isinstance(f, (list, tuple)) and len(f) == 3 and isinstance(f[0], str) for f in filters
        ):
            raise InvalidArgumentError(
                "partition_filters is a list of (column, op, value) tuples, e.g. "
                f"[('region', '=', 'eu')]; got {filters!r}"
            )
        partitions = list(self._enrich().partition_columns)
        stray = sorted({f[0] for f in filters} - set(partitions))
        if stray:
            raise InvalidArgumentError(
                f"partition_filters name {stray}, which are not partition columns "
                f"(the table is partitioned by {partitions}); delta-rs matches no file "
                "for them, so the OPTIMIZE would do nothing"
            )

    def _check_zorder(self, columns: list[str]) -> None:
        """Z-order keys must be data columns: partition columns are constant per file."""
        names = list(self.schema().names)
        partitions = set(self._enrich().partition_columns)
        for column in columns:
            if isinstance(column, str) and column not in names and "." in column:
                # cluster_by takes nested fields; delta-rs's Z-ORDER does not.
                raise InvalidArgumentError(
                    f"cannot z-order by {column!r}: Z-ORDER takes top-level columns only, "
                    "not nested fields (cluster_by= accepts nested fields)"
                )
            if not isinstance(column, str) or column not in names:
                raise InvalidArgumentError(
                    f"cannot z-order by {column!r}: the table has no such column; "
                    f"columns are {names}"
                )
            if column in partitions:
                raise InvalidArgumentError(
                    f"cannot z-order by {column!r}: it is a partition column, constant "
                    "within every file"
                )
            kind = self.schema().field(column).type
            if _nested_type(kind):
                # Refused for every engine, as Databricks refuses it: files
                # keep no min/max for a nested value, so ordering by one buys
                # no skipping, and the kernel's ranking cannot order them.
                raise InvalidArgumentError(
                    f"cannot z-order by {column!r}: it is a {kind} column, and Z-ORDER "
                    "takes columns with min/max statistics (numbers, strings, dates, "
                    "timestamps); order by one of its fields' values in a column of its own"
                )

    def vacuum(
        self,
        *,
        retention_hours: int | None = None,
        dry_run: bool = True,
        lite: bool = False,
        **kwargs: Any,
    ) -> Any:
        """VACUUM. A dry run by default, because the real thing deletes files.

        `lite=True` considers only files the log records as removed (VACUUM
        LITE), which is cheaper than listing storage for orphans.
        """
        _check_options("vacuum", kwargs, _VACUUM_OPTIONS)
        if retention_hours is not None and (
            isinstance(retention_hours, bool) or not isinstance(retention_hours, (int, float))
        ):
            raise InvalidArgumentError(
                f"retention_hours must be a number, not {type(retention_hours).__name__}"
            )
        if retention_hours is not None and retention_hours < 0:
            raise InvalidArgumentError(f"retention_hours must be >= 0, got {retention_hours}")
        request = self._request(
            Operation.VACUUM,
            {"retention_hours": retention_hours, "dry_run": dry_run, "lite": lite, **kwargs},
        )
        refused = refusal(self, request)
        if refused is not None:
            raise UnreachableTableError("vacuum", refused.reason or "", refused.remedy)
        result = self._route(request).vacuum(
            self._resolved, retention_hours=retention_hours, dry_run=dry_run, lite=lite, **kwargs
        )
        self._invalidate()
        return result

    def restore(self, target: Any, **kwargs: Any) -> dict[str, Any]:
        """RESTORE to a version (int) or a timestamp (datetime or string)."""
        self._check_writable("restore")
        _check_options("restore", kwargs, _RESTORE_OPTIONS)
        if target is None:
            raise InvalidArgumentError("restore needs a version or a timestamp")
        if str(self._enrich().properties.get("delta.appendOnly", "")).lower() == "true":
            # delta-rs committed the restore's remove actions anyway, deleting
            # rows from a table whose protocol promises they are never removed.
            raise UnreachableTableError(
                "restore the table",
                "delta.appendOnly is true, and a restore removes the files added since",
                "set delta.appendOnly to false first if the removal is intended",
            )
        target = target if isinstance(target, int) else _timestamp_arg(target, "restore target")
        if not isinstance(target, (bool, int)):
            # delta-rs resolved a timestamp before the first commit to version
            # 0 and restored the empty table, and ignored in-commit timestamps.
            # Resolve it the way a read does, so restore(ts) restores exactly
            # the table that to_arrow(timestamp=ts) returns.
            target = self._restore_version(target)
        # Routed before the no-op below: restore(current) returned success on
        # a table no engine may restore, while can(RESTORE) said no.
        engine = self._route(self._request(Operation.RESTORE, {"target": target, **kwargs}))
        if isinstance(target, (bool, int)):
            # -1 reached delta-rs as "either the version or datetime should
            # be provided"; True restored version 1.
            _check_version(target, "restore version")
            latest = self._connection._reresolve(self).version
            if latest is not None and target > latest:
                raise InvalidArgumentError(
                    f"cannot restore version {target}: the latest version is {latest}"
                )
            if latest is not None and target == latest:
                # The table already is that version. delta-rs raised a raw
                # "Version to restore 5 should be less then last available
                # version 5"; restoring to where you are changes nothing.
                return _results.restore(
                    {"numRemovedFile": 0, "numRestoredFile": 0}, getattr(engine, "kind", None)
                )
        result = engine.restore(self._resolved, target, **kwargs)
        self._invalidate()
        return _results.restore(result, getattr(engine, "kind", None))

    def _refuse_future_timestamp(self, timestamp: Any) -> None:
        """Refuse time travel to a time after the latest commit, as the warehouse does.

        The direct engines resolve such a timestamp to the latest version and
        read it; Databricks refuses it (DELTA_TIMESTAMP_GREATER_THAN_COMMIT),
        since no version exists at that time yet -- the same timestamp read
        different data once the next commit landed.
        """
        kernel = self._connection.router.engines.get(EngineKind.KERNEL)
        if not isinstance(kernel, KernelEngine) or self._resolved.location is None:
            return
        try:
            millis = timestamp_ms(timestamp)
            latest = kernel.snapshot(self._resolved)
            last = int(latest.timestamp())
        except Exception:
            return  # the read reports what is wrong
        if millis > last:
            raise InvalidArgumentError(
                f"cannot time travel to {timestamp}: it is after the latest commit (version "
                f"{int(latest.version)}), so no version of the table exists at that time "
                "(DELTA_TIMESTAMP_GREATER_THAN_COMMIT); read the latest version without a "
                "timestamp"
            )

    def _restore_version(self, timestamp: Any) -> Any:
        """The version a restore to `timestamp` means: the one a read at it sees.

        That is the latest commit at or before it (in-commit timestamps when
        enabled); one before the first recreatable commit is refused, as a read
        refuses it. Without a kernel that can open the table, the timestamp is
        passed on for the engine to resolve.
        """

        kernel = self._connection.router.engines.get(EngineKind.KERNEL)
        if not isinstance(kernel, KernelEngine) or self._resolved.location is None:
            return timestamp
        try:
            millis = timestamp_ms(timestamp)
        except UnreachableTableError as exc:
            raise InvalidArgumentError(f"restore target {timestamp!r}: {exc.reason}") from exc
        try:
            snapshot = kernel.snapshot(self._resolved, timestamp=millis)
        except UnreachableTableError as exc:
            if "earliest recreatable" in exc.reason or "out of range" in exc.reason:
                raise InvalidArgumentError(
                    f"cannot restore to {timestamp}: no version of the table exists at or "
                    "before it (it is before the first commit, or before the oldest one "
                    "log retention kept)"
                ) from exc
            return timestamp
        except Exception:
            return timestamp
        try:
            latest = kernel.snapshot(self._resolved)
            after_latest = int(snapshot.version) == int(latest.version) and millis > int(
                latest.timestamp()
            )
        except Exception:
            after_latest = False
        if after_latest:
            # It resolved to the latest version, so the restore was a silent
            # no-op. Spark refuses it (DELTA_TIMESTAMP_GREATER_THAN_COMMIT):
            # nothing is committed at that time yet.
            raise InvalidArgumentError(
                f"cannot restore to {timestamp}: it is after the latest commit (version "
                f"{int(latest.version)}), so there is no version of the table at that time"
            )
        return int(snapshot.version)

    def repair(self, **kwargs: Any) -> dict[str, Any]:
        """FSCK REPAIR TABLE: drop log entries for data files that are missing from storage.

        ``dry_run=True`` lists them without committing.
        """
        _check_options("repair", kwargs, _COMMIT_OPTIONS | {"dry_run"})
        result: dict[str, Any] = self._route(self._request(Operation.REPAIR, kwargs)).repair(
            self._resolved, **kwargs
        )
        self._invalidate()
        return result

    # ----------------------------------------------------------------- schema

    def add_column(self, fields: Any, **kwargs: Any) -> None:
        """Add columns: an Arrow schema, a list of Arrow or deltalake fields, or a
        ``{name: type}`` mapping of SQL, Delta or pyarrow type names. New columns
        must be nullable."""
        self._check_writable("add a column")
        _check_options("add_column", kwargs, _COMMIT_OPTIONS)
        if isinstance(fields, (str, bytes)) or fields is None:
            # A bare name has no type; the engines refused it as "cannot be
            # served", a routing refusal.
            raise InvalidArgumentError(
                "add_column takes pyarrow fields, deltalake Fields or a {name: type} mapping, "
                f"not {type(fields).__name__}"
            )
        if isinstance(fields, dict):
            new = list(fields)
        elif isinstance(fields, (list, tuple)):
            new = [getattr(f, "name", None) for f in fields]
        elif hasattr(fields, "names") and not hasattr(fields, "type"):
            new = list(fields.names)  # an Arrow schema
        else:
            new = [getattr(fields, "name", None)]
        if not new:
            raise InvalidArgumentError("add_column needs at least one field")
        existing = {name.lower(): name for name in self.schema().names}
        seen: set[str] = set()
        for name in new:
            if not isinstance(name, str) or not name:
                continue  # not a field this layer can read; the engine decides
            folded = name.lower()
            # delta-rs answered an existing name with "Cannot merge types long
            # and integer", or accepted it when the types matched.
            if folded in existing:
                raise InvalidArgumentError(
                    f"cannot add column {name!r}: the table already has {existing[folded]!r}"
                )
            if folded in seen:
                raise InvalidArgumentError(f"add_column names {name!r} twice")
            seen.add(folded)
        if isinstance(fields, dict):
            required = [
                n for n, t in fields.items() if isinstance(t, str) and "NOT NULL" in t.upper()
            ]
        else:
            if hasattr(fields, "names") and hasattr(fields, "field"):  # an Arrow schema
                items = [fields.field(i) for i in range(len(fields.names))]
            elif isinstance(getattr(fields, "fields", None), list):  # a deltalake Schema
                items = list(fields.fields)
            elif isinstance(fields, (list, tuple)):
                items = list(fields)
            else:
                items = [fields]
            required = [f.name for f in items if getattr(f, "nullable", True) is False]
        if required:
            # delta-rs and the kernel refused this, but the warehouse's ADD
            # COLUMNS was sent the column without NOT NULL: the call succeeded
            # and the column came back nullable.
            raise UnreachableTableError(
                f"add NOT NULL column(s) {', '.join(map(str, required))}",
                "existing rows have no value for a new column, so it must be nullable",
                "add the column as nullable, backfill it, then set_not_null()",
            )
        self._route(self._request(Operation.ADD_COLUMN, {"fields": fields, **kwargs})).add_columns(
            self._resolved, fields, **kwargs
        )
        self._invalidate()
        self._sync_catalog()

    def _alter_path(self, column: Any, what: str, names: Sequence[str] | None = None) -> str:
        """`_column_path`, with a str that is a top-level column's own name kept whole.

        A str is otherwise read as a dotted path, so an ALTER of a column
        named ``dot.name`` (or holding a backtick) went to a field ``name`` of
        a struct ``dot`` -- or failed on an unbalanced quote. The name as the
        table has it wins, then the dotted reading; a list is always a path.
        """
        path = _column_path(column, what)
        if isinstance(column, str) and ("." in column or "`" in column):
            top = _top_level(column, list(self.schema().names) if names is None else names)
            if top is not None:
                return "`" + top.replace("`", "``") + "`"
        return path

    def drop_column(self, column: str | list[str]) -> dict[str, Any]:
        """Drop a column; a dotted name or a list is a field inside a struct."""
        self._check_writable("drop a column")
        names = list(self.schema().names)
        top = _top_level(column, names)
        column = self._alter_path(column, "drop_column", names)
        partitions = set(self._enrich().partition_columns)
        rest = [name for name in names if name != top]
        if top is not None and rest and set(rest) <= partitions:
            # Delta needs a data column; the kernel panicked reading the
            # table this left behind.
            raise InvalidArgumentError(f"cannot drop {top!r}: it is the last non-partition column")
        if top is not None and not rest:
            raise InvalidArgumentError(f"cannot drop {top!r}: it is the table's only column")
        result = self._route(self._request(Operation.DROP_COLUMN, {"column": column})).drop_column(
            self._resolved, column
        )
        self._invalidate()
        self._sync_catalog()
        return _metrics(result)

    def rename_column(self, old: str | list[str], new: str) -> dict[str, Any]:
        """Rename a column; a dotted name or a list is a field inside a struct.

        `new` is the new last part (it may repeat the struct path).
        """
        self._check_writable("rename a column")
        old = self._alter_path(old, "rename_column")
        if not isinstance(new, str) or not new:
            raise InvalidArgumentError(f"rename_column needs a new name, not {new!r}")
        request = self._request(Operation.RENAME_COLUMN, {"old": old, "new": new})
        result = self._route(request).rename_column(self._resolved, old, new)
        self._invalidate()
        self._sync_catalog()
        return _metrics(result)

    def set_properties(self, properties: dict[str, str], **kwargs: Any) -> None:
        """ALTER TABLE ... SET TBLPROPERTIES.

        A value that implies a table feature (``delta.enableDeletionVectors``,
        ``delta.columnMapping.mode``, ...) raises the protocol with it.
        """
        self._check_writable("set properties")
        _check_options("set_properties", kwargs, _COMMIT_OPTIONS | {"raise_if_not_exists"})
        if properties is not None and not isinstance(properties, dict):
            raise InvalidArgumentError(
                f"set_properties takes a {{key: value}} dict, not {type(properties).__name__}"
            )
        for key in properties or {}:
            if not isinstance(key, str) or not key.strip():
                raise InvalidArgumentError(f"property keys must be non-empty strings, got {key!r}")
        if not properties:
            # Nothing to set; every engine would still commit an empty change.
            return
        request = self._request(Operation.SET_PROPERTIES, {"properties": properties, **kwargs})
        properties = request.shape["properties"]
        self._route(request).set_properties(self._resolved, properties, **kwargs)
        self._invalidate()
        self._sync_catalog()

    def add_feature(self, feature: Any, **kwargs: Any) -> None:
        """Add a table feature (or a list of them) by its protocol name, e.g. ``"deletionVectors"``.

        Features a feature depends on are added alongside; one that needs a
        backfill (row tracking) is refused.
        """
        self._check_writable("add a feature")
        _check_options(
            "add_feature", kwargs, _COMMIT_OPTIONS | {"allow_protocol_versions_increase"}
        )
        names = list(feature) if isinstance(feature, (list, tuple, set, frozenset)) else [feature]
        self._check_feature_names(names)
        request = self._request(Operation.ADD_FEATURE, {"feature": feature, **kwargs})
        self._route(request).add_feature(self._resolved, feature, **kwargs)
        self._invalidate()

    def _check_feature_names(self, names: list[Any]) -> None:
        """Refuse a feature name that is not text, or that no Delta protocol defines.

        Both were refused by every engine in turn, as "no engine can add
        teleportation" -- a routing refusal, for a typo. A name this library
        does not know may still be one Databricks does, so with the SQL
        fallback on the warehouse decides.
        """
        if not names:
            raise InvalidArgumentError("add_feature needs a feature name")
        for name in names:
            if not isinstance(name, str) or not name:
                raise InvalidArgumentError(
                    f"a table feature is named by a string such as 'deletionVectors', not {name!r}"
                )
        router = self._connection.router
        if router.allow_sql_fallback and router.engines.get(EngineKind.SQL) is not None:
            return
        unknown = [n for n in names if feature_from_wire(n) is None]
        if unknown:
            raise InvalidArgumentError(
                f"{unknown[0]!r} is not a Delta table feature; features are named as the "
                "protocol names them, e.g. 'deletionVectors', 'changeDataFeed', 'v2Checkpoint'"
            )

    def drop_feature(self, feature: str, **kwargs: Any) -> dict[str, Any]:
        """Drop a table feature. Databricks-only, so it needs the SQL fallback."""
        self._check_writable("drop a feature")
        _check_options("drop_feature", kwargs, frozenset({"truncate_history"}))
        request = self._request(Operation.DROP_FEATURE, {"feature": feature, **kwargs})
        result: dict[str, Any] = self._route(request).drop_feature(
            self._resolved, feature, **kwargs
        )
        self._invalidate()
        return result

    def add_constraint(self, constraints: dict[str, str], **kwargs: Any) -> None:
        """Add CHECK constraints, ``{name: SQL expression}``; existing rows are checked first."""
        self._check_writable("add a constraint")
        _check_options("add_constraint", kwargs, _COMMIT_OPTIONS)
        if not isinstance(constraints, dict) or not constraints:
            raise InvalidArgumentError("add_constraint needs at least one {name: expression}")
        for cname, expression in constraints.items():
            if not isinstance(cname, str) or not cname.strip():
                raise InvalidArgumentError(f"constraint names must be non-empty, got {cname!r}")
            _check_predicate(expression, f"add constraint {cname}")
            _check_sql_fragment(expression, f"constraint {cname!r}")
            if expression is None:
                raise InvalidArgumentError(f"constraint {cname!r} has no expression")
        request = self._request(Operation.ADD_CONSTRAINT, {"constraints": constraints, **kwargs})
        self._route(request).add_constraint(self._resolved, constraints, **kwargs)
        self._invalidate()

    def drop_constraint(self, name: str, *, if_exists: bool = False) -> None:
        """Drop a CHECK constraint; with `if_exists`, a missing one is not an error."""
        self._check_writable("drop a constraint")
        request = self._request(Operation.DROP_CONSTRAINT, {"name": name, "if_exists": if_exists})
        self._route(request).drop_constraint(self._resolved, name, if_exists=if_exists)
        self._invalidate()

    def unset_properties(self, keys: list[str] | str, *, if_exists: bool = True) -> None:
        """ALTER TABLE ... UNSET TBLPROPERTIES. delta-rs cannot remove a property."""
        self._check_writable("unset properties")
        names = _names("unset_properties", keys)
        if not names:
            # The kernel committed an empty metadata change; SQL raised.
            return
        request = self._request(Operation.UNSET_PROPERTIES, {"keys": names, "if_exists": if_exists})
        self._route(request).unset_properties(self._resolved, names, if_exists=if_exists)
        self._invalidate()
        self._sync_catalog(tuple(names))

    def set_comment(self, comment: str | None) -> None:
        """The table comment (the Metadata action's description)."""
        self._check_writable("set the comment")
        _check_comment(comment)
        request = self._request(Operation.SET_COMMENT, {"comment": comment})
        self._route(request).set_comment(self._resolved, comment)
        self._invalidate()
        self._sync_catalog()

    def set_column_comment(self, column: str | list[str], comment: str | None) -> None:
        """Set a column's comment (None clears it).

        A dotted name or a list is a field inside a struct.
        """
        self._check_writable("set a column comment")
        column = self._alter_path(column, "set_column_comment")
        _check_comment(comment)
        request = self._request(
            Operation.SET_COLUMN_COMMENT, {"column": column, "comment": comment}
        )
        self._route(request).set_column_comment(self._resolved, column, comment)
        self._invalidate()
        self._sync_catalog()

    def alter_column_type(self, column: str | list[str], new_type: str) -> None:
        """Widen a column's type without rewriting data (type widening).

        Allowed: byte->short->int->long, float->double, byte/short/int->double,
        date->timestamp_ntz, and decimals whose precision and scale do not
        shrink. The table needs ``delta.enableTypeWidening = true``.
        """
        self._check_writable("change a column type")
        column = self._alter_path(column, "alter_column_type")
        if self._has_type(column, new_type):
            # Nothing to change, as Spark treats it. Checked here because the
            # router refuses a type change on a table without type widening.
            return
        request = self._request(
            Operation.ALTER_COLUMN_TYPE, {"column": column, "new_type": new_type}
        )
        self._route(request).alter_column_type(self._resolved, column, new_type)
        self._invalidate()
        self._sync_catalog()

    def _has_type(self, column: str, new_type: str) -> bool:
        """Whether `column` already has `new_type` (``bigint`` for a long)."""
        import json

        from .engine.metadata import _find, _normalise_type, _split

        try:
            kernel: Any = self._connection.router.engines[EngineKind.KERNEL]
            metadata = json.loads(kernel.snapshot(self._enrich()).metadata_json())
            schema = json.loads(metadata["schemaString"])
            container, index = _find(schema, _split(column))
            return bool(container[index]["type"] == _normalise_type(new_type))
        except Exception:
            return False  # let the engine decide, and say why

    def set_not_null(self, column: str | list[str]) -> None:
        """Add a NOT NULL constraint, after checking no existing row is null."""
        self._check_writable("set NOT NULL")
        column = self._alter_path(column, "set_not_null")
        request = self._request(Operation.SET_NOT_NULL, {"column": column})
        self._route(request).set_not_null(self._resolved, column)
        self._invalidate()
        self._sync_catalog()

    def drop_not_null(self, column: str | list[str]) -> None:
        """Drop a NOT NULL constraint; a dotted name or a list is a field inside a struct."""
        self._check_writable("drop NOT NULL")
        column = self._alter_path(column, "drop_not_null")
        request = self._request(Operation.DROP_NOT_NULL, {"column": column})
        self._route(request).drop_not_null(self._resolved, column)
        self._invalidate()
        self._sync_catalog()

    def cluster_by(self, columns: list[str] | str | None) -> None:
        """Set the liquid-clustering keys (ALTER TABLE ... CLUSTER BY).

        ``None`` or ``[]`` is CLUSTER BY NONE. ``"auto"`` asks Databricks to
        choose keys, which only the SQL fallback can do. New keys apply to data
        written afterwards; existing files are reclustered by OPTIMIZE.
        """
        self._check_writable("change clustering")
        if columns is not None:
            _names("cluster_by", columns)
        request = self._request(Operation.CLUSTER_BY, {"columns": columns})
        self._route(request).cluster_by(self._resolved, columns)
        self._invalidate()

    # ------------------------------------------------------- log and layout

    def checkpoint(self) -> None:
        """Write a checkpoint at the current version.

        On a catalog-managed table staged commits are published first.
        """
        self._engine(Operation.CHECKPOINT).checkpoint(self._resolved)
        # The log changed shape under the cached state (a checkpoint, and on
        # a catalog-managed table a published tail), as after any write.
        self._invalidate()

    def compact_logs(self, start: int | None = None, end: int | None = None) -> Any:
        """Write a log compaction file for commits `start` to `end` (default: the whole log)."""
        request = self._request(Operation.LOG_COMPACTION, {"start": start, "end": end})
        result = self._route(request).compact_logs(self._resolved, start, end)
        self._invalidate()
        return result

    def cleanup_metadata(self) -> None:
        """Delete log files older than ``delta.logRetentionDuration``.

        This is what makes versions past log retention unreachable by time
        travel. It runs regardless of ``delta.enableExpiredLogCleanup``. It
        also happens implicitly, as in Spark: a delta-rs write that lands on
        a checkpoint interval deletes expired log files unless the table sets
        ``delta.enableExpiredLogCleanup=false`` (kernel writes never do).
        """
        self._engine(Operation.CLEANUP_METADATA).cleanup_metadata(self._resolved)
        self._invalidate()

    def analyze(self, *, columns: list[str] | None = None, delta_statistics: bool = False) -> Any:
        """ANALYZE TABLE. Databricks-only, so it needs the SQL fallback."""
        request = self._request(
            Operation.ANALYZE, {"columns": columns, "delta_statistics": delta_statistics}
        )
        return self._route(request).analyze(
            self._resolved, columns=columns, delta_statistics=delta_statistics
        )

    def sync_iceberg(self) -> Any:
        """Regenerate UniForm Iceberg metadata (MSCK REPAIR TABLE ... SYNC METADATA).

        Needed after anything other than Databricks writes to a table with
        Iceberg reads enabled. Databricks-only, so it needs the SQL fallback.
        """
        return self._engine(Operation.SYNC_ICEBERG).sync_iceberg_metadata(self._resolved)

    def refresh(self, *, full: bool = False) -> Any:
        """REFRESH a materialized view or streaming table. Needs the SQL fallback."""
        return self._route(self._request(Operation.REFRESH, {"full": full})).refresh(
            self._resolved, full=full
        )

    def generate(self) -> None:
        """Write symlink manifests, for engines that read those instead of the log."""
        self._engine(Operation.GENERATE).generate(self._resolved)

    def reorg(self, **kwargs: Any) -> dict[str, Any]:
        """REORG TABLE. Databricks-only, so it needs the SQL fallback."""
        self._check_writable("reorg")
        _check_options("reorg", kwargs, frozenset({"purge", "iceberg_compat_version", "predicate"}))
        result: dict[str, Any] = self._route(self._request(Operation.REORG, kwargs)).reorg(
            self._resolved, **kwargs
        )
        self._invalidate()
        return result

    def clone(self, target: str, **kwargs: Any) -> dict[str, Any]:
        """CLONE (`shallow=True` by default; `version=`/`timestamp=` clone a past one).

        A path table cloned to a storage path is written here, by the kernel
        (delta-rs#2456): version 0 of a new table with the source's protocol
        and metadata and a CLONE commit naming the source and its version. A
        shallow clone's adds name the source's files by absolute URL; a deep
        clone copies them first. Anything else -- a catalog name as the
        target, a source whose storage is reached with catalog-scoped
        credentials -- is Databricks' CLONE through the SQL fallback.
        """
        _check_options(
            "clone",
            kwargs,
            frozenset({"shallow", "replace", "if_not_exists", "version", "timestamp"}),
        )
        request = self._request(Operation.CLONE, {"target": target, **kwargs})
        result: dict[str, Any] = self._route(request).clone(self._resolved, target, **kwargs)
        self._invalidate()
        return result

    def publish(self) -> int:
        """Publish ratified-but-unpublished commits into `_delta_log/`.

        Required on catalog-managed tables, not optional housekeeping. The
        catalog caps how many unbackfilled commits it will hold and starts
        refusing writes past the limit, and checkpoints only run on published
        versions. `BackfillRequiredError` means do this.
        """
        version: int = self._engine(Operation.PUBLISH).publish(self._resolved)
        self._invalidate()
        return version

    # ------------------------------------------------------------- governance

    def _governance(self, what: str) -> Any:
        ref = self._resolved.ref
        # A table opened from another catalog (glue://, hms://) is governed
        # there, not by the catalog the connection is bound to.
        catalog = (
            self._connection._catalog_for(ref)
            if ref.kind is RefKind.CATALOG
            else self._connection.catalog
        )
        return _governed(catalog, "GovernedCatalog", what)

    def info(self) -> TableInfo:
        """The catalog's view of the table: owner, comment, columns, row filter,
        column masks, predictive optimization, audit timestamps."""
        cat = self._governance("read table info")
        info: TableInfo = _call("read table info", cat.table_info, self._resolved.ref)
        return info

    def grants(self, principal: str | None = None) -> list[Grant]:
        """Direct grants on the table, optionally for one principal. Unity Catalog only."""
        cat = self._governance("read grants")
        return list(_call("read grants", cat.grants, self._resolved.ref, principal))

    def effective_grants(self, principal: str | None = None) -> list[Grant]:
        """Grants including those inherited from the schema and catalog."""
        cat = self._governance("read effective grants")
        return list(
            _call("read effective grants", cat.effective_grants, self._resolved.ref, principal)
        )

    def grant(self, principal: str, privileges: list[str] | str) -> list[Grant]:
        """Grant privileges (``"SELECT"``, ``["SELECT", "MODIFY"]``) on the table to a principal."""
        names = [privileges] if isinstance(privileges, str) else list(privileges)
        cat = self._governance("grant")
        return list(_call("grant", cat.grant, self._resolved.ref, principal, names))

    def revoke(self, principal: str, privileges: list[str] | str) -> list[Grant]:
        """Revoke privileges on the table from a principal."""
        names = [privileges] if isinstance(privileges, str) else list(privileges)
        cat = self._governance("revoke")
        return list(_call("revoke", cat.revoke, self._resolved.ref, principal, names))

    def _tag_column(self, column: str | None, what: str) -> str | None:
        """`column` as the table spells it: Unity Catalog resolves column names in any case.

        The tag API takes the stored spelling only, so tags(column="EMAIL")
        said column `EMAIL` does not exist, and set_tags on a missing column
        failed as "cannot read tags".
        """
        if not isinstance(column, str):
            return column
        try:
            names = list(self.schema().names)
        except Exception:
            return column  # no schema to hand (a view, say): let the catalog answer
        if column in names:
            return column
        folded = [n for n in names if n.lower() == column.lower()]
        if len(folded) == 1:
            return str(folded[0])
        raise InvalidReferenceError(
            f"cannot {what}: {self._resolved.ref} has no column {column!r} "
            f"(its columns are {', '.join(names)})"
        )

    def tags(self, column: str | None = None) -> dict[str, str]:
        """The table's tags, or a column's with `column`. Databricks Unity Catalog only."""
        cat = self._governance("read tags")
        column = self._tag_column(column, "read tags")
        return dict(_call("read tags", cat.tags, self._resolved.ref, column))

    def set_tags(self, tags: dict[str, str], *, column: str | None = None) -> None:
        """Set tags ``{key: value}`` on the table, or on a column with `column`."""
        cat = self._governance("set tags")
        column = self._tag_column(column, "set tags")
        _call("set tags", cat.set_tags, self._resolved.ref, tags, column)

    def unset_tags(self, keys: list[str] | str, *, column: str | None = None) -> None:
        """Remove tags by key from the table, or from a column with `column`."""
        names = [keys] if isinstance(keys, str) else list(keys)
        cat = self._governance("unset tags")
        column = self._tag_column(column, "unset tags")
        _call("unset tags", cat.unset_tags, self._resolved.ref, names, column)

    def set_owner(self, principal: str) -> None:
        """Transfer ownership of the table to a user, group or service principal."""
        cat = self._governance("set owner")
        _call("set owner", cat.set_owner, self._resolved.ref, principal)

    def lineage(self, direction: str = "both") -> Lineage:
        """Upstream and downstream tables, notebooks, jobs and dashboards."""
        cat = self._governance("read lineage")
        lineage: Lineage = _call("read lineage", cat.lineage, self._resolved.ref, direction)
        return lineage

    def column_lineage(self, column: str, direction: str = "both") -> ColumnLineage:
        """Upstream and downstream lineage of one column.

        `direction` is ``"upstream"``, ``"downstream"`` or ``"both"``.
        """
        cat = self._governance("read column lineage")
        lineage: ColumnLineage = _call(
            "read column lineage", cat.column_lineage, self._resolved.ref, column, direction
        )
        return lineage

    def add_primary_key(self, name: str, columns: list[str], *, rely: bool = False) -> None:
        """An informational PRIMARY KEY constraint in Unity Catalog (not enforced)."""
        cat = self._governance("add a primary key")
        _call(
            "add a primary key", cat.add_primary_key, self._resolved.ref, name, columns, rely=rely
        )

    def add_foreign_key(
        self,
        name: str,
        columns: list[str],
        parent: str | Table,
        parent_columns: list[str],
        *,
        rely: bool = False,
    ) -> None:
        """An informational FOREIGN KEY constraint in Unity Catalog (not enforced)."""
        parent_ref = (
            parent.resolved.ref
            if isinstance(parent, Table)
            else parse_ref(
                parent,
                default_catalog=self._connection.default_catalog,
                default_schema=self._connection.default_schema,
            )
        )
        cat = self._governance("add a foreign key")
        _call(
            "add a foreign key",
            cat.add_foreign_key,
            self._resolved.ref,
            name,
            columns,
            parent_ref,
            parent_columns,
            rely=rely,
        )

    def drop_key_constraint(self, name: str, *, cascade: bool = False) -> None:
        """Drop an informational PRIMARY/FOREIGN KEY. CHECK constraints: `drop_constraint`."""
        cat = self._governance("drop a key constraint")
        _call(
            "drop a key constraint",
            cat.drop_table_constraint,
            self._resolved.ref,
            name,
            cascade=cascade,
        )

    def _warehouse(self, what: str) -> Any:
        engine = self._connection.router.engines.get(EngineKind.SQL)
        if not self._connection.router.warehouse_catalog:
            raise UnreachableTableError(
                what,
                "row filters and column masks are defined in SQL and enforced by Databricks, "
                "and this connection's catalog is not Databricks Unity Catalog",
            )
        if engine is None or not self._connection.router.allow_sql_fallback:
            raise FallbackRequiredError(
                what,
                "row filters and column masks are defined in SQL and enforced by Databricks",
                SQL_FALLBACK_REMEDY,
            )
        return engine

    def _sync_catalog(self, removed: tuple[str, ...] = ()) -> None:
        """Tell a catalog that keeps its own copy of an external table's metadata.

        Called after a schema, comment or property change committed to the log.
        OSS Unity Catalog does not re-read the log, so its column list and
        comment went stale. The change itself succeeded, so a failure here is a
        warning, not an error.
        """
        resolved = self._resolved
        if resolved.ref.kind is not RefKind.CATALOG or resolved.is_catalog_managed:
            return
        if resolved.table_type not in (TableType.EXTERNAL, None):
            return
        catalog = self._connection._catalog_for(resolved.ref)
        sync = getattr(catalog, "sync_external_metadata", None)
        if not callable(sync):
            return
        try:
            import json

            kernel: Any = self._connection.router.engines[EngineKind.KERNEL]
            metadata = json.loads(kernel.snapshot(self._enrich()).metadata_json())
            sync(
                resolved.ref,
                table_id=resolved.table_id,
                schema_string=metadata["schemaString"],
                description=metadata.get("description"),
                configuration=dict(metadata.get("configuration") or {}),
                removed=removed,
            )
            self._catalog_properties = {
                **{k: v for k, v in self._catalog_properties.items() if k not in removed},
                **dict(metadata.get("configuration") or {}),
            }
        except Exception as exc:
            warnings.warn(
                f"{resolved.ref} changed in its Delta log, but updating its entry in "
                f"{getattr(catalog, 'name', 'the catalog')} failed ({type(exc).__name__}: "
                f"{str(exc)[:200]}); the catalog's column list and comment may be stale "
                "until the table is registered again",
                UserWarning,
                stacklevel=3,
            )

    def _refresh_from_catalog(self) -> None:
        """Re-read the catalog's view of the table after a governance change.

        A row filter or column mask withdraws the table from credential
        vending (and dropping the last one restores it). The capability
        manifest captured at resolution still said otherwise, so the router
        kept sending reads to a direct engine that then failed on vending.
        """
        try:
            fresh = self._connection._reresolve(self).resolved
        except DeltaSwampError:
            # The change itself succeeded; do not report it as a failure.
            self._invalidate()
            return
        self._resolved = fresh
        self._catalog_properties = dict(fresh.properties)
        self._invalidate()

    def set_row_filter(self, function_name: str, columns: list[str]) -> None:
        """ALTER TABLE ... SET ROW FILTER `function_name` ON (`columns`). Needs the SQL fallback."""
        self._warehouse("set a row filter").set_row_filter(self._resolved, function_name, columns)
        self._refresh_from_catalog()

    def drop_row_filter(self) -> None:
        """ALTER TABLE ... DROP ROW FILTER. Needs the SQL fallback."""
        self._warehouse("drop a row filter").drop_row_filter(self._resolved)
        self._refresh_from_catalog()

    def set_column_mask(
        self, column: str, function_name: str, *, using_columns: list[str] | None = None
    ) -> None:
        """ALTER COLUMN ... SET MASK `function_name` USING COLUMNS (`using_columns`).

        Needs the SQL fallback.
        """
        self._warehouse("set a column mask").set_column_mask(
            self._resolved, column, function_name, using_columns
        )
        self._refresh_from_catalog()

    def drop_column_mask(self, column: str) -> None:
        """ALTER COLUMN ... DROP MASK. Needs the SQL fallback."""
        self._warehouse("drop a column mask").drop_column_mask(self._resolved, column)
        self._refresh_from_catalog()

    # ------------------------------------------------------------ credentials

    def credentials(self, *, write: bool = False) -> Any:
        """The vended credential currently in force, for debugging.

        Secrets are redacted in `repr`; call `.secrets` deliberately if you
        really need them.
        """
        provider = self._resolved.credential_provider
        if provider is None:
            raise UnreachableTableError(
                "vend credentials",
                "this table has no credential provider (it is not "
                "governed by a catalog that vends them)",
            )
        op = CredentialOperation.READ_WRITE if write else CredentialOperation.READ
        return provider.credentials(op)


def _flat_files(pa: Any, files: Any, schema: Any) -> Any:
    """The kernel's file listing in delta-rs's flattened, logical-name layout."""
    import decimal
    import json

    names = set(files.column_names)
    if not {"path", "size", "stats", "partition_values"} <= names:
        return files
    leaves: list[tuple[tuple[str, ...], str, Any]] = []  # physical path, logical name, type
    top: dict[str, tuple[str, Any]] = {}

    def walk(fields: Any, physical: tuple[str, ...], logical: tuple[str, ...]) -> None:
        for field in fields:
            meta = field.metadata or {}
            name = meta.get(b"delta.columnMapping.physicalName", field.name.encode()).decode()
            p, lg = (*physical, name), (*logical, field.name)
            if not physical:
                top[name] = (field.name, field.type)
            if pa.types.is_struct(field.type):
                walk(list(field.type), p, lg)
            else:
                leaves.append((p, ".".join(lg), field.type))

    walk(list(schema), (), ())
    # Numbers are kept exact: a DECIMAL(38, 9) bound read as a float comes back
    # rounded, and a 38-digit integer does not fit any Arrow integer at all.
    # `typed` narrows them to the column's own type.
    stats = [
        json.loads(s, parse_float=decimal.Decimal) if s else {}
        for s in files.column("stats").to_pylist()
    ]
    missing = object()

    def lookup(entry: Any, path: tuple[str, ...]) -> Any:
        for part in path:
            if not isinstance(entry, dict) or part not in entry:
                return missing
            entry = entry[part]
        return entry

    def typed(values: list[Any], wanted: Any) -> Any:
        if pa.types.is_decimal(wanted):
            try:
                exact = [
                    v if v is None or isinstance(v, str) else decimal.Decimal(v) for v in values
                ]
                return pa.array(exact, wanted)
            except (pa.ArrowInvalid, pa.ArrowTypeError, decimal.InvalidOperation, ValueError):
                pass
        else:
            values = [float(v) if isinstance(v, decimal.Decimal) else v for v in values]
        try:
            return pa.array(values).cast(wanted)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, OverflowError):
            try:
                return pa.array(values)
            except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError):
                return pa.array([None if v is None else str(v) for v in values], pa.string())

    columns: dict[str, Any] = {
        "path": files.column("path"),
        "size_bytes": files.column("size"),
        "modification_time": files.column("modification_time"),
        "num_records": files.column("num_records"),
    }
    for prefix, key in (("null_count", "nullCount"), ("min", "minValues"), ("max", "maxValues")):
        for physical, logical, wanted in leaves:
            found = [lookup(s.get(key), physical) for s in stats]
            if all(v is missing for v in found):
                continue
            values = [None if v is missing else v for v in found]
            columns[f"{prefix}.{logical}"] = typed(
                values, pa.int64() if prefix == "null_count" else wanted
            )
    partition_maps = [dict(m or ()) for m in files.column("partition_values").to_pylist()]
    order = list(top)
    keys = sorted(
        {k for m in partition_maps for k in m},
        key=lambda k: order.index(k) if k in top else len(order),
    )
    for key in keys:
        logical, wanted = top.get(key, (key, pa.string()))
        raw = [m.get(key) for m in partition_maps]
        columns[f"partition.{logical}"] = typed(raw, wanted)
    consumed = ("path", "size", "modification_time", "num_records", "stats", "partition_values")
    for extra in files.column_names:
        if extra not in consumed:
            columns[extra] = files.column(extra)
    return pa.table(columns)


#: The tuning options DELETE and UPDATE pass to the engine.
_DML_OPTIONS = frozenset({"commit_metadata", "writer_properties", "max_commit_retries"})


def _nested_type(kind: Any) -> bool:
    """Whether an Arrow type is a STRUCT, ARRAY or MAP (VARIANT included)."""
    import pyarrow as pa

    storage = getattr(kind, "storage_type", kind)
    return bool(
        pa.types.is_struct(storage)
        or pa.types.is_list(storage)
        or pa.types.is_large_list(storage)
        or pa.types.is_fixed_size_list(storage)
        or pa.types.is_map(storage)
    )


#: What every committing maintenance and ALTER call passes on: this library's
#: commit options and delta-rs's own. A misspelt one was silently ignored when
#: the kernel served the call and a TypeError naming a delta-rs signature
#: when delta-rs did.
_COMMIT_OPTIONS = frozenset(
    {"commit_metadata", "max_commit_retries", "commit_properties", "post_commithook_properties"}
)
_OPTIMIZE_OPTIONS = _COMMIT_OPTIONS | {
    "partition_filters",
    "target_size",
    "max_concurrent_tasks",
    "max_spill_size",
    "max_temp_directory_size",
    "min_commit_interval",
    "writer_properties",
    "min_file_size",
    "sort_by",
    "min_cube_size",
}
_VACUUM_OPTIONS = _COMMIT_OPTIONS | {"enforce_retention_duration", "keep_versions"}
_RESTORE_OPTIONS = _COMMIT_OPTIONS | {"ignore_missing_files", "protocol_downgrade_allowed"}
_MERGE_OPTIONS = _COMMIT_OPTIONS | {
    "source_alias",
    "target_alias",
    "merge_schema",
    "error_on_type_mismatch",
    "writer_properties",
    "streamed_exec",
    "max_spill_size",
    "max_temp_directory_size",
    "engine_info",
}
_REPLACE_OPTIONS = frozenset(
    {
        "predicate",
        "partition_overwrite",
        "target_file_size",
        "writer_properties",
        "commit_metadata",
        "txn",
        "max_commit_retries",
    }
)
#: A column name delta-rs reads as itself when not backticked.
_PLAIN_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _check_options(what: str, given: dict[str, Any], known: frozenset[str]) -> None:
    """Refuse an option no engine takes (a typo, usually).

    delta-rs raised a bare TypeError for one, while the kernel and the
    warehouse refused it as "cannot be served": three answers to one mistake.
    """
    unknown = sorted(set(given) - known)
    if unknown:
        raise InvalidArgumentError(
            f"{what}() got unexpected option(s) {unknown}; it takes {sorted(known)}"
        )


def _asked_operation(operation: Any, shape: dict[str, Any]) -> tuple[Operation, dict[str, Any]]:
    """The operation `can()` was asked about, from its name or a method's."""
    shape = dict(shape)
    if operation == "count":
        # count() reads no column but the ones its predicate names (or it
        # counts from the log), so a VARIANT column the direct engines cannot
        # decode does not stop it; can(SCAN) answers for reading every column.
        shape.setdefault("columns", [])
        return Operation.SCAN, shape
    alias = METHOD_OPERATIONS.get(operation) if isinstance(operation, str) else None
    if alias is not None:
        op, implied = alias
        if operation == "z_order" and "columns" in shape:
            shape["zorder_by"] = shape.pop("columns")
        shape.update(implied)
        return op, shape
    try:
        op = Operation(operation)
    except ValueError:
        names = sorted([o.value for o in Operation] + list(METHOD_OPERATIONS) + ["count"])
        raise InvalidArgumentError(
            f"{operation!r} is not an operation or a Table method; one of {names}"
        ) from None
    if op is Operation.ZORDER and "columns" in shape and "zorder_by" not in shape:
        # z_order(columns) is the call, so can(Operation.ZORDER, columns=...)
        # is how it is asked; only the string alias took that spelling.
        shape["zorder_by"] = shape.pop("columns")
    return op, shape


def _check_can_options(op: Operation, shape: dict[str, Any]) -> None:
    """Refuse an argument the call does not take.

    can("append", schema_mod="merge") answered for a plain append, so a typo
    preflighted fine and the call itself then did something else.
    """
    import inspect

    if "distributed" in shape:
        method = "plan_write" if op in (Operation.APPEND, Operation.OVERWRITE) else "plan_scan"
        known = {"distributed"}
    elif "incremental" in shape:
        method, known = "added_since", {"incremental"}
    else:
        method = _CALL_METHODS.get(op, op.value)
        known = set(_CALL_OPTIONS.get(op, ()))
    owner: Any = Table
    if op is Operation.CREATE:
        from .connection import Connection

        owner, method = Connection, "create_table"
    call = getattr(owner, method, None)
    if call is None:
        return
    params = inspect.signature(call).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()) and (
        op not in _CALL_OPTIONS
    ):
        return  # options passed through to an engine this layer does not list
    known |= set(params) - {"self", "kwargs"} - ({"name"} if op is Operation.CREATE else set())
    _check_options("can", shape, frozenset(known))


#: The Table method each operation is the call of, where it is not the
#: operation's own name.
_CALL_METHODS: dict[Operation, str] = {
    Operation.TIME_TRAVEL: "scan",
    Operation.MERGE_SCHEMA: "append",
    Operation.REPLACE_WHERE: "overwrite",
    Operation.INCREMENTAL: "changes",
    Operation.ZORDER: "optimize",
    Operation.LOG_COMPACTION: "compact_logs",
    Operation.CONVERT: "convert_to_delta",
}


#: What `Table.cdf` takes, across every engine that serves it.
_CDF_OPTIONS = frozenset(
    {
        "starting_version",
        "ending_version",
        "starting_timestamp",
        "ending_timestamp",
        "columns",
        "predicate",
        "allow_out_of_range",
    }
)

#: What each call takes through ``**kwargs``, which `can()` accepts too.
_CALL_OPTIONS: dict[Operation, frozenset[str]] = {
    Operation.CDF: _CDF_OPTIONS,
    Operation.DELETE: _DML_OPTIONS,
    Operation.UPDATE: _DML_OPTIONS | {"error_on_type_mismatch"},
    # The builder's clauses, by method name, which decide whether the MERGE
    # removes rows; and `data=`, the spelling every other write's can() takes.
    Operation.MERGE: _MERGE_OPTIONS | {"clauses", "data"},
    Operation.OPTIMIZE: _OPTIMIZE_OPTIONS,
    Operation.ZORDER: _OPTIMIZE_OPTIONS,
    Operation.VACUUM: _VACUUM_OPTIONS,
    Operation.RESTORE: _RESTORE_OPTIONS,
    Operation.REPAIR: _COMMIT_OPTIONS | {"dry_run"},
    Operation.ADD_COLUMN: _COMMIT_OPTIONS,
    Operation.SET_PROPERTIES: _COMMIT_OPTIONS | {"raise_if_not_exists"},
    # `features=` is the engines' spelling, which can() has always taken.
    Operation.ADD_FEATURE: _COMMIT_OPTIONS | {"allow_protocol_versions_increase", "features"},
    Operation.DROP_FEATURE: frozenset({"truncate_history"}),
    Operation.ADD_CONSTRAINT: _COMMIT_OPTIONS,
    Operation.REORG: frozenset({"purge", "iceberg_compat_version", "predicate"}),
    Operation.CLONE: frozenset({"shallow", "replace", "if_not_exists", "version", "timestamp"}),
}
_CDF_META = ("_change_type", "_commit_version", "_commit_timestamp")


def _feed_gap_error(version: int) -> UnreachableTableError:
    """The change feed was off at `version`; `.version` names it, as a schema change does."""
    error = UnreachableTableError(
        "read the change data feed",
        f"the change data feed was not enabled at version {version}, which the requested "
        "range includes",
        f"read up to version {version - 1}, or from the version the feed was enabled again",
    )
    error.version = version  # type: ignore[attr-defined]
    return error


def _warehouse_feed_schema_error(exc: BaseException) -> Exception:
    """The warehouse's DELTA_CHANGE_DATA_FEED_INCOMPATIBLE_* refusal, typed as the kernel's."""
    from .errors import ChangeFeedSchemaChangeError

    text = str(exc)
    found = re.search(r"(?:at|in) version (\d+)", text)
    changed = int(found.group(1)) if found is not None else None
    first = (text.strip().splitlines() or [""])[0][:300]
    error = ChangeFeedSchemaChangeError(
        "read the change data feed",
        "the table's schema changed"
        + (f" at version {changed}" if changed is not None else "")
        + f" in a way the rows written before it cannot be read under ({first})",
        f"read the feed up to version {changed - 1}, or from version {changed} on"
        if changed is not None
        else "read a range that does not span the schema change",
    )
    error.version = changed
    return error


def _by_version(pa: Any, changes: Any, columns: list[str] | None) -> Any:
    """`(version, rows)` for each commit in a change-feed table, in order."""
    if not changes.num_rows:
        return
    changes = _plain_views(changes).sort_by("_commit_version")
    counts = pa.compute.value_counts(changes.column("_commit_version")).to_pylist()
    offset = 0
    # Sorted, so each version is one contiguous slice: no pass over the
    # whole table per version.
    for entry in counts:
        chunk = changes.slice(offset, entry["counts"])
        offset += entry["counts"]
        yield int(entry["values"]), chunk.select(columns) if columns is not None else chunk


def _stream_by_version(pa: Any, feed: Any, columns: list[str] | None) -> Any:
    """`(version, rows)` from a feed whose batches arrive in commit order.

    A version is yielded once a batch of a later one arrives, or the feed
    ends. A batch that goes back to a version already yielded would split
    it, so that is refused rather than yielded twice.
    """
    buffered: list[Any] = []
    top = done = -1
    batches = iter(feed)
    while True:
        try:
            batch = next(batches)
        except StopIteration:
            break
        except UnreachableTableError as exc:
            # The feed was off at `version`, which the kernel reaches only
            # after every commit before it: those are complete, so yield
            # them before refusing, and a follower resumes from there.
            gap = getattr(exc, "version", None)
            if buffered and isinstance(gap, int) and gap > top:
                yield from _by_version(pa, pa.Table.from_batches(buffered), columns)
            raise
        if not batch.num_rows:
            continue
        versions = batch.column(batch.schema.get_field_index("_commit_version"))
        low = int(pa.compute.min(versions).as_py())
        if low <= done:
            raise CorruptTableError(
                f"the change data feed returned rows of version {low} after later versions; "
                "read it with cdf() instead"
            )
        if buffered and low > top:
            yield from _by_version(pa, pa.Table.from_batches(buffered), columns)
            buffered, done = [], top
        buffered.append(batch)
        top = max(top, int(pa.compute.max(versions).as_py()))
    if buffered:
        yield from _by_version(pa, pa.Table.from_batches(buffered), columns)


def _cdf_types(stream: Any) -> Any:
    """The change feed with its metadata columns typed the same on every engine.

    delta-rs reports ``_commit_version`` as uint64, ``_change_type`` as a
    string view and ``_commit_timestamp`` as naive milliseconds; the kernel
    as int64, string and UTC microseconds (Delta's long and timestamp). A
    consumer unioning feeds from two tables, or comparing a version with an
    int64 column, failed on one engine only.
    """
    try:
        import pyarrow as pa
    except ImportError:
        return stream
    wanted = {
        "_change_type": pa.string(),
        "_commit_version": pa.int64(),
        "_commit_timestamp": pa.timestamp("us", tz="UTC"),
    }
    reader = pa.RecordBatchReader.from_stream(stream)
    fields = [
        f.with_type(wanted[f.name]) if f.name in wanted and f.type != wanted[f.name] else f
        for f in reader.schema
    ]
    target = pa.schema(fields, metadata=reader.schema.metadata)
    return reader if target.equals(reader.schema) else reader.cast(target)


def _metrics(result: Any) -> dict[str, Any]:
    """An engine's result as the dict the Table API promises.

    The kernel's metadata commits return the committed version (an int) where
    the warehouse returns a status dict, so `rename_column()` handed back 22 on
    one table and ``{"status": "ok"}`` on another.
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, int) and not isinstance(result, bool):
        return {"version": result}
    return {} if result is None else {"result": result}


def _governed(catalog: Any, protocol_name: str, what: str) -> Any:
    """The catalog, if it implements `protocol_name`; else a refusal naming it."""
    from .catalog import base

    protocol = getattr(base, protocol_name)
    if not isinstance(catalog, protocol):
        raise UnreachableTableError(
            what,
            f"the {getattr(catalog, 'name', type(catalog).__name__)} catalog has no "
            "governance API for this",
            "Unity Catalog (Databricks or open source) provides it",
        )
    return catalog


def _call(what: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke a catalog method, turning 'this catalog lacks it' into a refusal."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        raise UnreachableTableError(what, str(exc) or "the catalog does not implement it") from exc


class _InvalidatingMerger:
    """Wraps a merge builder so the table forgets cached state once it executes.

    A MERGE runs at `execute()`, not when the builder is created, so
    invalidating any earlier would let a read in between re-cache the
    pre-merge protocol and properties.

    It also checks clause order for every engine, and records the clauses: an
    engine that finds at `execute()` that it cannot run them (an
    `EngineLimitError`, raised before anything is written) hands the MERGE to
    the next engine that serves the table, with the clauses replayed.
    """

    _OWN = (
        "_builder",
        "_invalidate",
        "_rebuild",
        "_kind",
        "_calls",
        "_unconditional",
        "_preflight",
    )

    def __init__(
        self,
        builder: Any,
        invalidate: Any,
        rebuild: Callable[[frozenset[EngineKind]], tuple[Any, EngineKind | None]] | None = None,
        kind: EngineKind | None = None,
        preflight: Callable[[list[tuple[Any, ...]]], EngineKind | None] | None = None,
    ) -> None:
        self._builder = builder
        self._preflight = preflight
        self._invalidate = invalidate
        self._rebuild = rebuild
        self._kind = kind
        self._calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self._unconditional: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not set in __init__. Before __init__ runs
        # (copy, pickle) `_builder` itself lands here, and looking it up on
        # itself recursed until RecursionError.
        if name.startswith("__") or name in self._OWN:
            raise AttributeError(name)
        attr = getattr(self._builder, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            if name == "execute":
                result = self._execute(attr, args, kwargs)
                self._invalidate()
                return _results.dml(result, self._kind)
            if name == "when_not_matched_insert" and "values" in kwargs:
                # An INSERT sets values, and the name is what users reach for;
                # the builders all call the parameter `updates` (delta-rs's
                # spelling), so values= was a bare TypeError.
                if "updates" in kwargs or args:
                    raise InvalidArgumentError(
                        "when_not_matched_insert() takes the columns once: updates= or values="
                    )
                kwargs = dict(kwargs)
                kwargs["updates"] = kwargs.pop("values")
            clause = merge_clause(name, args, kwargs)
            if clause is not None:
                _check_merge_clause(args, kwargs)
                kind, conditional = clause
                if kind in self._unconditional:
                    # Spark refuses this when it parses the MERGE; the engines
                    # here took the first such clause and silently dropped the
                    # rest.
                    raise InvalidArgumentError(
                        f"a {name}() clause follows an unconditional clause of the same kind, "
                        "which takes every row; only the last clause of a kind may omit "
                        "its condition"
                    )
                if not conditional:
                    self._unconditional.add(kind)
            result = attr(*args, **kwargs)
            if clause is not None:
                self._calls.append((name, args, kwargs))
            # Clause methods return the builder; keep the wrapper in the chain.
            if result is self._builder or type(result) is type(self._builder):
                self._builder = result
                return self
            return result

        return call

    def _move_to(self, kind: EngineKind) -> None:
        """Rebuild the MERGE on engine `kind`, which its clauses need, replaying them."""
        if self._rebuild is None:
            raise UnreachableTableError(
                "merge",
                f"its clauses need {kind.value} rather than "
                f"{self._kind.value if self._kind else 'the engine the builder was made on'}, "
                "and the source is a stream that cannot be handed to a second engine",
                "pass the source as a pyarrow Table rather than a stream",
            )
        builder, built = self._rebuild(frozenset(k for k in EngineKind if k is not kind))
        for name, call_args, call_kwargs in self._calls:
            result = getattr(builder, name)(*call_args, **call_kwargs)
            if result is not None and type(result) is type(builder):
                builder = result
        self._builder, self._kind = builder, built

    def _execute(self, execute: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if self._preflight is not None:
            clauses: list[tuple[Any, ...]] = []
            for name, call_args, call_kwargs in self._calls:
                clause = merge_clause(name, call_args, call_kwargs)
                values = merge_clause_values(name, call_args, call_kwargs)
                clauses.append((name, "condition" if clause and clause[1] else None, values))
            kind = self._preflight(clauses)
            if kind is not None and kind is not self._kind:
                self._move_to(kind)
                execute = self._builder.execute
        tried: set[EngineKind] = set()
        refusals: list[EngineLimitError] = []
        while True:
            try:
                return execute(*args, **kwargs)
            except EngineLimitError as exc:
                if self._rebuild is None or self._kind is None:
                    raise
                refusals.append(exc)
                tried.add(self._kind)
            try:
                builder, kind = self._rebuild(frozenset(tried))
            except DeltaSwampError:
                raise refusals[0] from refusals[0].__cause__
            warnings.warn(
                f"{self._kind.value if self._kind else 'the engine'} cannot run this MERGE "
                f"({refusals[-1].reason}); trying {kind.value if kind else 'the next engine'}",
                EngineFallbackWarning,
                stacklevel=4,
            )
            for name, call_args, call_kwargs in self._calls:
                result = getattr(builder, name)(*call_args, **call_kwargs)
                if result is not None and type(result) is type(builder):
                    builder = result
            self._builder, self._kind = builder, kind
            execute = builder.execute
