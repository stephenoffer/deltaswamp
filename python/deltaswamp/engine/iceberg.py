"""The Iceberg engine: tables behind a catalog's Iceberg REST endpoint, via PyIceberg.

Unity Catalog serves the Iceberg REST catalog protocol for its Iceberg tables
and for UniForm Delta tables (whose Iceberg metadata it generates from the Delta
log). Databricks serves it at ``{host}/api/2.1/unity-catalog/iceberg-rest`` and
open-source Unity Catalog at ``{base}/api/2.1/unity-catalog/iceberg``; either
way the warehouse is the UC catalog name and the Iceberg identifier is
``(schema, table)``. The catalog sets `ResolvedTable.iceberg_rest_uri`, and this
engine serves only tables that carry one.

What it serves:

* **native Iceberg tables** -- scan, time travel (a snapshot id passed as
  `version`, or a timestamp), history (the snapshot log), detail, append, and
  overwrite: whole-table, by predicate (``overwrite_filter``), or dynamic
  partition overwrite.
* **UniForm Delta tables** -- reads only. Their Iceberg metadata is derived from
  the Delta log, so a write through Iceberg would diverge from the Delta table;
  writes belong to the Delta engines.

Auth: the catalog token comes from ``credential_provider.workspace_auth()`` on
every catalog build. The SDK refreshes that token on its own clock, so a cached
catalog is rebuilt as soon as the token it was built with changes, rather than
holding one until it expires. Storage credentials are vended per table by the
REST server (``X-Iceberg-Access-Delegation: vended-credentials``).
"""

from __future__ import annotations

import importlib
import json
import operator
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from ..capability import Capability, Operation
from ..capability import Engine as EngineKind
from ..catalog import ResolvedTable
from ..errors import SQL_FALLBACK_REMEDY, InvalidArgumentError, UnreachableTableError
from .base import missing_method
from .sharing import filter_arrow_exact

__all__ = ["ACCESS_DELEGATION_HEADER", "IcebergEngine", "iceberg_rest_uri"]

ACCESS_DELEGATION_HEADER = "header.X-Iceberg-Access-Delegation"

_READS: frozenset[Operation] = frozenset(
    {Operation.SCAN, Operation.TIME_TRAVEL, Operation.HISTORY, Operation.DETAIL}
)
_WRITES: frozenset[Operation] = frozenset(
    {Operation.APPEND, Operation.OVERWRITE, Operation.REPLACE_WHERE}
)

#: The snapshot-summary key UniForm records the source Delta version under.
_UNIFORM_DELTA_VERSION = "delta-version"

CatalogFactory = Callable[[str, dict[str, str]], Any]

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def iceberg_rest_uri(base_url: str, *, databricks: bool) -> str:
    """The Iceberg REST endpoint of a Unity Catalog server.

    Databricks: ``{host}/api/2.1/unity-catalog/iceberg-rest``.
    Open-source Unity Catalog: ``{base}/api/2.1/unity-catalog/iceberg``.
    """
    suffix = "iceberg-rest" if databricks else "iceberg"
    return f"{base_url.rstrip('/')}/api/2.1/unity-catalog/{suffix}"


def _default_factory(name: str, properties: dict[str, str]) -> Any:
    # Looked up at call time, so `pyiceberg.catalog.load_catalog` can be patched.
    return importlib.import_module("pyiceberg.catalog").load_catalog(name, **properties)


def _timestamp_ms(timestamp: Any) -> int:
    if isinstance(timestamp, bool):
        raise UnreachableTableError(
            "time travel by timestamp",
            f"{timestamp!r} is not an ISO-8601 timestamp or epoch milliseconds",
        )
    if isinstance(timestamp, int):
        return timestamp
    if isinstance(timestamp, datetime):
        return _epoch_ms(timestamp)
    if isinstance(timestamp, date):
        return _epoch_ms(datetime(timestamp.year, timestamp.month, timestamp.day))
    text = str(timestamp).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError as exc:
        raise UnreachableTableError(
            "time travel by timestamp",
            f"{timestamp!r} is not an ISO-8601 timestamp or epoch milliseconds",
        ) from exc
    return _epoch_ms(moment)


def _epoch_ms(moment: datetime) -> int:
    """Epoch milliseconds, in integer arithmetic (naive means UTC).

    Floor division keeps pre-1970 instants and sub-millisecond parts exact,
    where ``int(moment.timestamp() * 1000)`` rounds through a float and
    truncates toward zero.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (moment - _EPOCH) // timedelta(milliseconds=1)


def _to_arrow_table(data: Any) -> Any:
    pa = importlib.import_module("pyarrow")
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatchReader):
        return data.read_all()
    return pa.table(data)  # PyCapsule streams, pandas, dict-of-columns


#: Snapshot-summary keys PyIceberg computes itself. Passing one as commit
#: metadata made `Summary(...)` raise "got multiple values for keyword
#: argument" at commit time -- after the data files had been written.
_RESERVED_SUMMARY_PREFIXES = ("added-", "deleted-", "removed-", "total-", "partitions.")
_RESERVED_SUMMARY_KEYS = frozenset(
    {"operation", "changed-partition-count", "partition-summaries-included"}
)


def _snapshot_properties(commit_metadata: dict[str, Any] | None) -> dict[str, str]:
    if not commit_metadata:
        return {}
    out = {
        str(k): v if isinstance(v, str) else json.dumps(v, default=str)
        for k, v in commit_metadata.items()
    }
    reserved = sorted(
        k for k in out if k in _RESERVED_SUMMARY_KEYS or k.startswith(_RESERVED_SUMMARY_PREFIXES)
    )
    if reserved:
        raise InvalidArgumentError(
            f"commit_metadata keys {reserved} are Iceberg snapshot-summary fields that "
            "PyIceberg computes itself; use other key names"
        )
    return out


def _conform_type(arrow_type: Any, pa: Any) -> Any:
    """`arrow_type` with the timestamp shapes PyIceberg refuses made writable.

    PyIceberg rejects ``timestamp[ns]`` (pandas' default) and any zone other
    than UTC. Iceberg stores microseconds, and Arrow stores every zoned
    timestamp as a UTC instant, so both cast without changing the instant.
    """
    types = pa.types
    if types.is_timestamp(arrow_type):
        unit = "us" if arrow_type.unit == "ns" else arrow_type.unit
        tz = arrow_type.tz
        if tz is not None and tz.upper() not in ("UTC", "+00:00", "ETC/UTC", "Z"):
            tz = "UTC"
        return pa.timestamp(unit, tz=tz)
    if types.is_struct(arrow_type):
        return pa.struct([f.with_type(_conform_type(f.type, pa)) for f in arrow_type])
    if types.is_large_list(arrow_type):
        return pa.large_list(
            arrow_type.value_field.with_type(_conform_type(arrow_type.value_type, pa))
        )
    if types.is_list(arrow_type):
        return pa.list_(arrow_type.value_field.with_type(_conform_type(arrow_type.value_type, pa)))
    if types.is_map(arrow_type):
        return pa.map_(
            arrow_type.key_field.with_type(_conform_type(arrow_type.key_type, pa)),
            arrow_type.item_field.with_type(_conform_type(arrow_type.item_type, pa)),
        )
    return arrow_type


def _uuid_bytes(column: Any, pa: Any) -> Any:
    """A string column of UUIDs as the 16-byte binary PyIceberg writes to uuid."""
    import uuid

    values: list[bytes | None] = []
    for value in column.to_pylist():
        if value is None:
            values.append(None)
            continue
        try:
            values.append(uuid.UUID(str(value)).bytes)
        except ValueError as exc:
            raise InvalidArgumentError(f"{value!r} is not a UUID") from exc
    return pa.array(values, pa.binary(16))


def _conform(rows: Any, schema: Any) -> Any:
    """Line an Arrow table up with the Iceberg schema it is written to.

    * column names that match the schema only case-insensitively are renamed
      to the schema's spelling, as SQL resolves them (PyIceberg reported the
      column as extra and refused);
    * ``timestamp[ns]`` and non-UTC zones are cast (see `_conform_type`);
    * UUID strings bound for a ``uuid`` column become 16-byte binary.

    Everything else -- real extra columns, lossy decimals -- is left for
    PyIceberg to refuse.
    """
    pa = importlib.import_module("pyarrow")
    fields = {f.name: f for f in schema.fields}
    lowered: dict[str, list[str]] = {}
    for name in fields:
        lowered.setdefault(name.lower(), []).append(name)
    names = []
    for name in rows.column_names:
        if name not in fields and len(lowered.get(name.lower(), [])) == 1:
            names.append(lowered[name.lower()][0])
        else:
            names.append(name)
    if names != rows.column_names and len(set(names)) == len(names):
        rows = rows.rename_columns(names)

    iceberg_types = importlib.import_module("pyiceberg.types")
    uuid_type = getattr(iceberg_types, "UUIDType", None)
    columns = []
    changed = False
    for name, column in zip(rows.column_names, rows.columns, strict=True):
        target = fields.get(name)
        if (
            target is not None
            and isinstance(target.field_type, iceberg_types.DecimalType)
            and pa.types.is_decimal(column.type)
            and (column.type.precision, column.type.scale)
            != (target.field_type.precision, target.field_type.scale)
            and column.type.scale <= target.field_type.scale
        ):
            # decimal(38, 2) values into decimal(10, 2): PyIceberg refused the
            # type outright even when every value fits. A safe cast checks
            # each value and fails on a real overflow.
            wanted_decimal = pa.decimal128(target.field_type.precision, target.field_type.scale)
            try:
                columns.append(column.cast(wanted_decimal, safe=True))
            except pa.lib.ArrowInvalid as exc:
                raise InvalidArgumentError(
                    f"column {name!r}: a value does not fit the table's "
                    f"decimal({target.field_type.precision}, {target.field_type.scale}): {exc}"
                ) from exc
            changed = True
            continue
        if (
            uuid_type is not None
            and target is not None
            and isinstance(target.field_type, uuid_type)
            and (pa.types.is_string(column.type) or pa.types.is_large_string(column.type))
        ):
            columns.append(_uuid_bytes(column, pa))
            changed = True
            continue
        wanted = _conform_type(column.type, pa)
        if wanted != column.type:
            # safe=False: ns -> us truncates sub-microsecond digits, which
            # Iceberg (and Delta) cannot store anyway.
            columns.append(column.cast(wanted, safe=False))
            changed = True
        else:
            columns.append(column)
    if changed:
        # Keep each field's nullability and metadata: a required Iceberg
        # field refuses a column that turned nullable on the way through.
        out_schema = pa.schema(
            [f.with_type(c.type) for f, c in zip(rows.schema, columns, strict=True)],
            metadata=rows.schema.metadata,
        )
        rows = pa.Table.from_arrays(columns, schema=out_schema)
    return rows


@contextmanager
def _schema_errors(what: str) -> Iterator[None]:
    """PyIceberg's schema-mismatch ValueError -> an error that names the remedy."""
    try:
        yield
    except ValueError as exc:
        text = str(exc)
        if "more columns" in text or "Mismatch in fields" in text:
            raise UnreachableTableError(
                what,
                f"the data does not match the table's Iceberg schema: {text}",
                "the Iceberg engine does not evolve schemas (no schema_mode='merge'); "
                "ALTER the table first, or select and cast the data to its schema",
            ) from exc
        raise


def _numeric(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _lossy_literal(expr: Any, schema: Any, case_sensitive: bool) -> bool:
    """True when binding would change a numeric literal's value.

    PyIceberg converts ``1.5`` to a long column's type by rounding (to 2), so
    ``id > 1.5`` would read as ``id > 2``. Those predicates are not pushed down.
    """
    expressions = importlib.import_module("pyiceberg.expressions")
    literals_mod = importlib.import_module("pyiceberg.expressions.literals")
    if isinstance(expr, expressions.And | expressions.Or):
        return _lossy_literal(expr.left, schema, case_sensitive) or _lossy_literal(
            expr.right, schema, case_sensitive
        )
    if isinstance(expr, expressions.Not):
        return _lossy_literal(expr.child, schema, case_sensitive)
    if isinstance(expr, expressions.LiteralPredicate):
        literals = [expr.literal]
    elif isinstance(expr, expressions.SetPredicate):
        literals = list(expr.literals)
    else:
        return False
    try:
        field_type = schema.find_field(expr.term.name, case_sensitive).field_type
    except Exception:
        return True
    for literal in literals:
        before = _numeric(literal.value)
        if before is None:
            continue
        converted = literal.to(field_type)
        if isinstance(converted, literals_mod.AboveMax | literals_mod.BelowMin):
            continue  # bind turns these into an exact AlwaysTrue / AlwaysFalse
        after = _numeric(converted.value)
        if after is not None and after != before:
            return True
    return False


def _null_exact(expr: Any) -> Any:
    """Conjoin each negated leaf with ``IS NOT NULL`` (input has no `Not` left).

    SQL keeps a row only where the predicate is TRUE, and ``NULL <> 1`` is not
    true; Iceberg's evaluators treat a null as satisfying a negation, which
    returns (and, in an overwrite filter, deletes) rows SQL would not.
    """
    expressions = importlib.import_module("pyiceberg.expressions")
    if isinstance(expr, expressions.And):
        return expressions.And(_null_exact(expr.left), _null_exact(expr.right))
    if isinstance(expr, expressions.Or):
        return expressions.Or(_null_exact(expr.left), _null_exact(expr.right))
    negations = tuple(
        getattr(expressions, n)
        for n in ("NotIn", "NotEqualTo", "NotStartsWith", "NotNaN")
        if hasattr(expressions, n)
    )
    if isinstance(expr, negations):
        return expressions.And(expr, expressions.NotNull(expr.term))
    return expr


def _version_of(table: ResolvedTable, snapshot_id: int, summary: Any) -> int | None:
    """The version `scan(version=...)` accepts for this snapshot."""
    if table.is_iceberg:
        return int(snapshot_id)
    raw = summary.get(_UNIFORM_DELTA_VERSION) if summary is not None else None
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _operation(summary: Any) -> str | None:
    operation = getattr(summary, "operation", None) if summary is not None else None
    return getattr(operation, "value", operation)


def _as_json(model: Any) -> Any:
    return json.loads(model.model_dump_json())


class IcebergEngine:
    """Reads and writes Iceberg tables through a catalog's Iceberg REST endpoint."""

    kind = EngineKind.ICEBERG
    supports_distributed_scan = False
    #: Pushed down when PyIceberg's parser accepts it, else applied exactly after.
    supports_predicates = True
    supports_timestamp_travel = True
    #: Recorded as Iceberg snapshot summary properties.
    supports_commit_metadata = True
    #: PyIceberg's dynamic_partition_overwrite.
    supports_dynamic_overwrite = True
    supports_schema_merge = False
    supports_schema_overwrite = False
    supports_idempotent_txn = False
    supports_writer_properties = False

    def __init__(
        self,
        *,
        properties: dict[str, str] | None = None,
        token: str | None = None,
        catalog_factory: CatalogFactory | None = None,
    ) -> None:
        """
        `properties` are extra PyIceberg REST catalog properties (TLS settings,
        ``rest.signing-*``, ...), applied over the defaults this engine sets.
        `token` is used only for tables whose credential provider offers no
        ``workspace_auth()``. `catalog_factory(name, properties)` replaces
        ``pyiceberg.catalog.load_catalog``.
        """
        self._properties = dict(properties or {})
        self._token = token
        self._factory: CatalogFactory = catalog_factory or _default_factory
        # (uri, warehouse) -> (token it was built with, catalog)
        self._catalogs: dict[tuple[str, str], tuple[str | None, Any]] = {}
        # (uri, warehouse) -> sequence number of the token fetch it was built from
        self._built_seq: dict[tuple[str, str], int] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_catalogs"] = {}
        state["_built_seq"] = {}
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # ----------------------------------------------------------- capabilities

    @staticmethod
    def available() -> bool:
        try:
            importlib.import_module("pyiceberg.catalog")
        except ImportError:
            return False
        return True

    def supports(self, operation: Operation, table: ResolvedTable, **shape: Any) -> Capability:
        if not self.available():
            return Capability(
                operation,
                ok=False,
                reason="the pyiceberg package is not installed",
                remedy="pip install 'deltaswamp[iceberg]'",
            )

        gap = missing_method(self, operation)
        if gap is not None:
            return gap

        if not table.iceberg_rest_uri:
            return Capability(
                operation,
                ok=False,
                reason="the table's catalog advertises no Iceberg REST endpoint",
                remedy="reach the table through a Unity Catalog connection (Databricks or "
                "open-source), which serves one",
            )

        uniform = not table.is_iceberg
        if uniform and not table.has_iceberg_compat:
            return Capability(
                operation,
                ok=False,
                reason="the table is Delta without UniForm Iceberg metadata, so the Iceberg "
                "REST endpoint does not serve it",
                remedy="the Delta engines serve it",
            )

        if table.external_read_supported is False:
            return Capability(
                operation,
                ok=False,
                reason="the catalog reports no external-engine read support for this table "
                "(usually a row filter or column mask)",
                remedy=SQL_FALLBACK_REMEDY,
            )

        if operation in _WRITES:
            if uniform:
                return Capability(
                    operation,
                    ok=False,
                    reason="the table is UniForm: its Iceberg metadata is generated from the "
                    "Delta log, so the Iceberg view is read-only and a write through it would "
                    "diverge from the Delta table",
                    remedy="write through the Delta engines",
                )
            if table.external_write_supported is False:
                return Capability(
                    operation,
                    ok=False,
                    reason="the catalog reports no external-engine write support for this table",
                    remedy=SQL_FALLBACK_REMEDY,
                )
        elif operation not in _READS:
            return Capability(
                operation,
                ok=False,
                reason=f"the Iceberg engine does not implement {operation.value}",
            )

        return Capability(operation, ok=True, engine=self.kind)

    # ---------------------------------------------------------------- catalog

    @staticmethod
    def _identifier(table: ResolvedTable) -> tuple[str, str, str]:
        ref = table.ref
        if not (ref.catalog and ref.schema and ref.table):
            raise UnreachableTableError(
                "open the table through Iceberg REST",
                f"{ref} is not a catalog.schema.table name, which the endpoint needs "
                "(warehouse = catalog, identifier = schema.table)",
            )
        return ref.catalog, ref.schema, ref.table

    def _fresh_token(self, table: ResolvedTable) -> str | None:
        """The catalog token, asked for anew on every catalog lookup."""
        auth = getattr(table.credential_provider, "workspace_auth", None)
        if auth is not None:
            _, token = auth()
            # `str(None)` is the truthy string "None", which would be sent as a
            # bearer token; an absent token means "no Authorization header".
            return str(token) if token else None
        return self._token or None

    def rest_properties(self, table: ResolvedTable) -> dict[str, str]:
        """The PyIceberg REST catalog properties for `table`, with a fresh token."""
        if not table.iceberg_rest_uri:
            raise UnreachableTableError(
                "open the table through Iceberg REST",
                "the table's catalog advertises no Iceberg REST endpoint",
            )
        warehouse, _, _ = self._identifier(table)
        properties = {
            "type": "rest",
            "uri": table.iceberg_rest_uri,
            "warehouse": warehouse,
            ACCESS_DELEGATION_HEADER: "vended-credentials",
        }
        token = self._fresh_token(table)
        if token:
            properties["token"] = token
        properties.update(self._properties)
        return properties

    def _catalog(self, table: ResolvedTable) -> Any:
        with self._lock:
            self._seq += 1
            seq = self._seq
        properties = self.rest_properties(table)
        key = (properties["uri"], properties["warehouse"])
        token = properties.get("token")
        with self._lock:
            cached = self._catalogs.get(key)
            if cached is not None and cached[0] == token:
                return cached[1]
        catalog = self._factory(f"deltaswamp-{key[1]}", properties)
        with self._lock:
            # Two threads racing a token refresh: the one that fetched its
            # token *earlier* can finish building later. Letting it overwrite
            # the cache put the older token back, so every later call saw a
            # mismatch and rebuilt (or kept using the stale token until it
            # expired). Only a build from a newer fetch replaces the entry.
            if seq >= self._built_seq.get(key, 0):
                self._catalogs[key] = (token, catalog)
                self._built_seq[key] = seq
        return catalog

    def _load(self, table: ResolvedTable) -> Any:
        _, schema, name = self._identifier(table)
        catalog = self._catalog(table)
        exceptions = importlib.import_module("pyiceberg.exceptions")
        missing = tuple(
            getattr(exceptions, n)
            for n in ("NoSuchTableError", "NoSuchNamespaceError", "NoSuchIdentifierError")
            if hasattr(exceptions, n)
        )
        try:
            return catalog.load_table((schema, name))
        except missing as exc:
            raise UnreachableTableError(
                "open the table through Iceberg REST",
                f"the Iceberg endpoint has no table {schema}.{name} in warehouse "
                f"{table.ref.catalog} ({exc})",
            ) from exc

    # ------------------------------------------------------------------- read

    @staticmethod
    def _row_filter(predicate: str | None, schema: Any = None) -> tuple[Any, str | None]:
        """PyIceberg's own parse of `predicate`, or no pushdown plus a residual.

        Returns ``(row_filter, residual)``: when PyIceberg cannot express the
        SQL exactly the scan reads everything and `residual` is applied exactly
        afterwards. Given the table `schema`, "exactly" also covers binding:

        * a predicate that does not bind (an unknown column, ``name = 1`` on a
          string column) is a residual, not a crash at scan time;
        * a literal PyIceberg would convert lossily is a residual -- it rounds
          ``id > 1.5`` on a long column to ``id > 2``, silently dropping 2;
        * negations (``!=``, ``NOT IN``, ``NOT LIKE``) are conjoined with
          ``IS NOT NULL``: Iceberg evaluates ``NULL NOT IN (...)`` as true
          (whole null-only files and partitions match), SQL as not-true.

        A column named in a different case from the schema binds
        case-insensitively, as SQL resolves it (`_pushdown` says which).
        """
        filt, residual, _ = IcebergEngine._pushdown(predicate, schema)
        return filt, residual

    @staticmethod
    def _pushdown(predicate: str | None, schema: Any = None) -> tuple[Any, str | None, bool]:
        """``(row_filter, residual, case_sensitive)``; see `_row_filter`."""
        expressions = importlib.import_module("pyiceberg.expressions")
        if not predicate or not predicate.strip():
            return expressions.AlwaysTrue(), None, True
        parser = importlib.import_module("pyiceberg.expressions.parser")
        try:
            parsed = parser.parse(predicate)
        except Exception:
            return expressions.AlwaysTrue(), predicate, True
        if schema is None:
            return parsed, None, True
        visitors = importlib.import_module("pyiceberg.expressions.visitors")
        for case_sensitive in (True, False):
            try:
                visitors.bind(schema, parsed, case_sensitive)
            except Exception:
                continue
            if _lossy_literal(parsed, schema, case_sensitive):
                break
            try:
                exact = _null_exact(visitors.rewrite_not(parsed))
            except Exception:
                break
            return exact, None, case_sensitive
        return expressions.AlwaysTrue(), predicate, True

    @staticmethod
    def _snapshot_id(
        iceberg: Any, table: ResolvedTable, version: int | None, timestamp: str | None
    ) -> int | None:
        if version is not None and timestamp is not None:
            # Used to read `version` and silently ignore the timestamp.
            raise InvalidArgumentError("time travel takes a version or a timestamp, not both")
        if version is not None:
            if isinstance(version, bool):
                raise TypeError("version must be an int, not bool")
            try:
                # numpy integers (a version read from a history frame) are fine.
                version = operator.index(version)
            except TypeError:
                raise TypeError(f"version must be an int, not {type(version).__name__}") from None
            if not table.is_iceberg:
                # UniForm: `version` is a Delta version. Map it through the
                # snapshot summary rather than misreading it as a snapshot id.
                for snapshot in iceberg.snapshots():
                    summary = snapshot.summary
                    if summary is not None and summary.get(_UNIFORM_DELTA_VERSION) == str(version):
                        return int(snapshot.snapshot_id)
                raise UnreachableTableError(
                    f"read Delta version {version} through the table's Iceberg metadata",
                    "no Iceberg snapshot records that Delta version (UniForm converts only "
                    "some commits, and expired snapshots are gone)",
                    "time travel through the Delta engines",
                )
            if iceberg.snapshot_by_id(version) is None:
                raise UnreachableTableError(
                    f"read snapshot {version}",
                    "the Iceberg table has no snapshot with that id. Iceberg time travel takes "
                    "a snapshot id, not a sequential version number",
                    "history() lists the snapshot ids",
                )
            return version
        if timestamp is not None:
            return IcebergEngine._snapshot_at(iceberg, timestamp)
        return None

    @staticmethod
    def _snapshot_at(iceberg: Any, timestamp: Any) -> int:
        """The snapshot that was current at `timestamp` (inclusive, in ms).

        Walks the snapshot log like PyIceberg's `snapshot_as_of_timestamp`,
        but (a) says so when the snapshot current at that time has expired --
        PyIceberg returned None, reported as "no snapshot before that time" --
        and (b) falls back to the current snapshot's ancestry when the metadata
        carries no snapshot log at all (some writers omit it), where PyIceberg
        found nothing even though snapshots exist.
        """
        target = _timestamp_ms(timestamp)
        log = list(iceberg.history())
        if log:
            for entry in reversed(log):
                if entry.timestamp_ms <= target:
                    if iceberg.snapshot_by_id(entry.snapshot_id) is None:
                        raise UnreachableTableError(
                            f"read the table as of {timestamp}",
                            f"snapshot {entry.snapshot_id}, current at that time, has expired",
                            "pick a later timestamp; history() lists the live snapshots",
                        )
                    return int(entry.snapshot_id)
        else:
            snapshot = iceberg.current_snapshot()
            seen: set[int] = set()
            while snapshot is not None and snapshot.snapshot_id not in seen:
                seen.add(snapshot.snapshot_id)
                if snapshot.timestamp_ms <= target:
                    return int(snapshot.snapshot_id)
                parent = snapshot.parent_snapshot_id
                snapshot = iceberg.snapshot_by_id(parent) if parent is not None else None
        raise UnreachableTableError(
            f"read the table as of {timestamp}",
            "the Iceberg table has no snapshot at or before that time",
            "history() lists the snapshot timestamps",
        )

    def scan(
        self,
        table: ResolvedTable,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: str | None = None,
        limit: int | None = None,
    ) -> Any:
        """Read as a `pyarrow.RecordBatchReader`. `version` is an Iceberg snapshot id."""
        pa = importlib.import_module("pyarrow")
        if limit is not None:
            if isinstance(limit, bool):
                raise TypeError("limit must be an int, not bool")
            try:
                limit = operator.index(limit)
            except TypeError:
                raise TypeError(f"limit must be an int, not {type(limit).__name__}") from None
        if limit is not None and limit < 0:
            raise InvalidArgumentError(f"limit must be >= 0, got {limit}")
        if columns is not None and len(columns) == 0:
            # No columns still has rows. PyIceberg's scan of `()` returned an
            # empty table with 0 rows; read the narrowest thing (one column)
            # and drop it, so the row count survives.
            probe = self._load(table)
            first = [probe.schema().fields[0].name] if probe.schema().fields else None
            inner = self.scan(
                table,
                columns=first,
                predicate=predicate,
                version=version,
                timestamp=timestamp,
                limit=limit,
            )
            return pa.RecordBatchReader.from_batches(
                pa.schema([]), (batch.select([]) for batch in inner)
            )
        iceberg = self._load(table)
        snapshot_id = self._snapshot_id(iceberg, table, version, timestamp)
        row_filter, residual, case_sensitive = self._pushdown(
            predicate, self._scan_schema(iceberg, snapshot_id)
        )
        # A residual may reference columns the caller did not select.
        selected = ("*",) if columns is None or residual else tuple(columns)
        reader = iceberg.scan(
            row_filter=row_filter,
            selected_fields=selected,
            case_sensitive=case_sensitive,
            snapshot_id=snapshot_id,
            # With a residual the limit applies to the rows it keeps, not the
            # rows PyIceberg reads before it.
            limit=limit if residual is None else None,
        ).to_arrow_batch_reader()
        if residual is None:
            return reader

        out_schema = (
            reader.schema
            if columns is None
            else pa.schema([reader.schema.field(c) for c in columns])
        )

        def batches() -> Iterator[Any]:
            remaining = limit
            if remaining == 0:
                return
            for batch in reader:
                rows = filter_arrow_exact(pa.Table.from_batches([batch]), residual)
                if columns is not None:
                    rows = rows.select(columns)
                if remaining is not None:
                    rows = rows.slice(0, remaining)
                    remaining -= rows.num_rows
                yield from rows.to_batches()
                if remaining is not None and remaining <= 0:
                    return

        return pa.RecordBatchReader.from_batches(out_schema, batches())

    @staticmethod
    def _scan_schema(iceberg: Any, snapshot_id: int | None) -> Any:
        """The schema a scan at `snapshot_id` binds its filter against."""
        if snapshot_id is not None:
            snapshot = iceberg.snapshot_by_id(snapshot_id)
            if snapshot is not None and snapshot.schema_id is not None:
                schema = iceberg.schemas().get(snapshot.schema_id)
                if schema is not None:
                    return schema
        return iceberg.schema()

    def history(self, table: ResolvedTable, *, limit: int | None = None) -> list[dict[str, Any]]:
        """The snapshot log, newest first.

        ``version`` is what `scan(version=...)` takes back: the snapshot id for
        a native Iceberg table, the source Delta version for a UniForm one (None
        where UniForm did not record it).
        """
        iceberg = self._load(table)
        out: list[dict[str, Any]] = []
        if limit is not None and limit <= 0:
            return out
        for entry in reversed(iceberg.history()):
            snapshot = iceberg.snapshot_by_id(entry.snapshot_id)
            summary = snapshot.summary if snapshot is not None else None
            out.append(
                {
                    "version": _version_of(table, entry.snapshot_id, summary),
                    "snapshot_id": entry.snapshot_id,
                    "timestamp": entry.timestamp_ms,
                    "parent_snapshot_id": snapshot.parent_snapshot_id if snapshot else None,
                    "sequence_number": snapshot.sequence_number if snapshot else None,
                    "operation": _operation(summary),
                    "summary": dict(summary.additional_properties) if summary is not None else {},
                    "schema_id": snapshot.schema_id if snapshot else None,
                }
            )
            if limit is not None and len(out) >= limit:
                break
        return out

    def detail(self, table: ResolvedTable, *, version: int | None = None) -> dict[str, Any]:
        """Format version, snapshot, schema, partition spec, properties, location.

        ``version`` is the current (or requested) snapshot id -- for a UniForm
        table the Delta version it was converted from, which is what `scan`
        takes -- and None for a table with no snapshots yet.
        """
        iceberg = self._load(table)
        metadata = iceberg.metadata
        snapshot_id = self._snapshot_id(iceberg, table, version, None)
        snapshot = (
            iceberg.snapshot_by_id(snapshot_id)
            if snapshot_id is not None
            else iceberg.current_snapshot()
        )
        schemas = iceberg.schemas()
        schema = (
            schemas.get(snapshot.schema_id, iceberg.schema())
            if snapshot is not None and snapshot.schema_id is not None
            else iceberg.schema()
        )
        spec = iceberg.spec()
        # Delta's partition columns are identity partitions; a bucket or day
        # transform of a column is not one (it is listed in partition_spec).
        partition_columns = [
            schema.find_column_name(f.source_id) or str(f.source_id)
            for f in spec.fields
            if str(f.transform) == "identity"
        ]
        summary = snapshot.summary if snapshot is not None else None
        return {
            "version": (
                _version_of(table, snapshot.snapshot_id, summary) if snapshot is not None else None
            ),
            "snapshot_id": snapshot.snapshot_id if snapshot is not None else None,
            "format": "iceberg",
            "format_version": metadata.format_version,
            "table_uuid": str(metadata.table_uuid),
            "location": iceberg.location(),
            "metadata_location": iceberg.metadata_location,
            "current_snapshot_id": metadata.current_snapshot_id,
            "snapshot_timestamp": snapshot.timestamp_ms if snapshot is not None else None,
            "sequence_number": snapshot.sequence_number if snapshot is not None else None,
            "snapshot_summary": dict(summary.additional_properties) if summary else {},
            "schema": _as_json(schema),
            "arrow_schema": schema.as_arrow(),
            "partition_spec": _as_json(spec),
            "partition_columns": partition_columns,
            "sort_order": _as_json(iceberg.sort_order()),
            "properties": dict(metadata.properties),
            "is_uniform": not table.is_iceberg,
        }

    # ------------------------------------------------------------------ write

    @staticmethod
    def _refuse_unsupported(what: str, **options: Any) -> None:
        given = sorted(k for k, v in options.items() if v is not None)
        if given:
            raise UnreachableTableError(
                what,
                f"the Iceberg engine does not support {', '.join(given)}",
                "drop the option, or write through a Delta engine if the table is Delta",
            )

    def append(
        self,
        table: ResolvedTable,
        data: Any,
        *,
        commit_metadata: dict[str, Any] | None = None,
        schema_mode: str | None = None,
        partition_by: list[str] | None = None,
        writer_properties: Any = None,
        txn: tuple[str, int] | None = None,
        **_: Any,
    ) -> None:
        """Append rows. `data` is anything Arrow-shaped (PyCapsule, pandas, ...)."""
        self._refuse_unsupported(
            "append to an Iceberg table",
            schema_mode=schema_mode,
            partition_by=partition_by,
            writer_properties=writer_properties,
            txn=txn,
        )
        properties = _snapshot_properties(commit_metadata)
        iceberg = self._load(table)
        rows = _conform(_to_arrow_table(data), iceberg.schema())
        if rows.num_rows == 0:
            # A no-op, as the Delta engines treat it. PyIceberg committed an
            # empty snapshot, which moved the version and grew the history.
            return
        with _schema_errors("append to an Iceberg table"):
            iceberg.append(rows, snapshot_properties=properties)

    def overwrite(
        self,
        table: ResolvedTable,
        data: Any,
        *,
        predicate: str | None = None,
        partition_overwrite: str = "static",
        commit_metadata: dict[str, Any] | None = None,
        schema_mode: str | None = None,
        writer_properties: Any = None,
        txn: tuple[str, int] | None = None,
        **_: Any,
    ) -> None:
        """Replace the whole table, the rows matching `predicate`, or (with
        ``partition_overwrite="dynamic"``) exactly the partitions in `data`."""
        self._refuse_unsupported(
            "overwrite an Iceberg table",
            schema_mode=schema_mode,
            writer_properties=writer_properties,
            txn=txn,
        )
        mode = str(partition_overwrite or "static").lower()
        if mode not in ("static", "dynamic"):
            # Anything else used to fall through to a whole-table overwrite.
            raise InvalidArgumentError(
                f"partition_overwrite must be 'static' or 'dynamic', got {partition_overwrite!r}"
            )
        properties = _snapshot_properties(commit_metadata)
        iceberg = self._load(table)
        rows = _conform(_to_arrow_table(data), iceberg.schema())

        if mode == "dynamic":
            if predicate:
                raise UnreachableTableError(
                    "overwrite an Iceberg table",
                    "a predicate and dynamic partition overwrite are mutually exclusive",
                )
            if iceberg.spec().is_unpartitioned():
                # Spark's partitionOverwriteMode=dynamic on an unpartitioned
                # table replaces the whole table; PyIceberg raised ValueError.
                iceberg.overwrite(rows, snapshot_properties=properties)
                return
            if rows.num_rows == 0:
                return  # no partitions in the data, so none to replace
            try:
                iceberg.dynamic_partition_overwrite(rows, snapshot_properties=properties)
            except ValueError as exc:
                raise UnreachableTableError(
                    "dynamically overwrite partitions of an Iceberg table",
                    str(exc),
                    "overwrite with a predicate over the partition source columns instead",
                ) from exc
            return

        if predicate and predicate.strip():
            row_filter, residual, case_sensitive = self._pushdown(predicate, iceberg.schema())
            if residual is not None:
                # An overwrite filter decides which rows are DELETED, so it
                # cannot be approximated the way a read filter can.
                raise UnreachableTableError(
                    "overwrite the rows matching a predicate",
                    f"PyIceberg cannot parse {predicate!r} into an exact Iceberg expression "
                    "over this table's schema (unparseable SQL, an unknown column, or a "
                    "literal that would be converted lossily to the column's type)",
                    "use comparisons, IN, IS [NOT] NULL, AND/OR/NOT, or LIKE 'prefix%', "
                    "with literals of the column's type",
                )
            # replaceWhere semantics: every written row must match the
            # predicate, or the next overwrite of that range would not see it.
            if rows.num_rows:
                kept = filter_arrow_exact(rows, predicate)
                if kept.num_rows != rows.num_rows:
                    raise UnreachableTableError(
                        "overwrite the rows matching a predicate",
                        f"{rows.num_rows - kept.num_rows} of the {rows.num_rows} rows written "
                        f"do not match {predicate!r}",
                        "filter the data to the predicate before writing",
                    )
            with _schema_errors("overwrite an Iceberg table"):
                iceberg.overwrite(
                    rows,
                    overwrite_filter=row_filter,
                    snapshot_properties=properties,
                    case_sensitive=case_sensitive,
                )
            return

        with _schema_errors("overwrite an Iceberg table"):
            iceberg.overwrite(rows, snapshot_properties=properties)

    # ------------------------------------------------------ distributed path

    def plan_scan(self, table: ResolvedTable, **kwargs: Any) -> list[Any]:
        raise NotImplementedError(
            "the Iceberg engine does not expose split planning; PyIceberg plans files "
            "per scan in-process"
        )

    def execute_scan(self, table: ResolvedTable, splits: list[Any], **kwargs: Any) -> Any:
        raise NotImplementedError("see plan_scan")
