"""`Table`: one table, every operation, routed per operation.

Protocol features are discovered in two stages. The catalog supplies enough to
make the decisions that must precede opening the log (is this a view, a shallow
clone, a table vending refuses). The full reader/writer feature lists exist only
in the log, so the first operation enriches the resolved table with them and
everything afterward routes on the complete picture.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .capability import FEATURE_SUPPORT, Capability, FeatureKind, Operation, feature_from_wire
from .capability import READ_OPERATIONS as _READ_OPERATIONS
from .capability import Engine as EngineKind
from .catalog import ResolvedTable, TableType
from .credentials import Operation as CredentialOperation
from .engine.base import TranslatingStream, translating_stream
from .engine.deltars import DeltaRsEngine
from .engine.kernel import KernelEngine
from .errors import (
    SQL_FALLBACK_REMEDY,
    CorruptTableError,
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
    from .connection import Connection

__all__ = ["Table"]


def _write_data(data: Any) -> Any:
    """Plain Python data (a column dict, a list of row dicts) as an Arrow table.

    The engines take Arrow-exporting objects and pandas only; a dict failed in
    delta-rs with "Expected object with __arrow_c_array__", after write_table
    had already created the table from its inferred schema.
    """
    if isinstance(data, dict):
        pa = _require("pyarrow", "pyarrow")
        return pa.table(data)
    if isinstance(data, list) and data and all(isinstance(row, dict) for row in data):
        pa = _require("pyarrow", "pyarrow")
        return pa.Table.from_pylist(data)
    if data is None:
        raise InvalidArgumentError("no data given to write")
    return data


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
    return names


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
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidArgumentError(f"{what} must be a timestamp, not bool")
    if isinstance(value, (int, float)):
        import datetime as _dt

        if value < 0:
            raise InvalidArgumentError(f"{what} must not be before 1970, got {value}")
        epoch = _dt.datetime(1970, 1, 1, tzinfo=_dt.UTC)
        return epoch + _dt.timedelta(milliseconds=value)
    if isinstance(value, str) and not value.strip():
        raise InvalidArgumentError(f"{what} is blank")
    return value


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


def _require(module: str, extra: str) -> Any:
    """Import an optional dependency, or say which extra provides it.

    Without this the failure surfaces as a ModuleNotFoundError raised from deep
    inside pyarrow, which does not tell you what to install.
    """
    import importlib

    try:
        # Through __import__ so an import hook (or a test blocking one) sees
        # the request even when the module is already loaded.
        __import__(module)
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} is needed for this conversion but is not installed. "
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
        return self._resolved

    @property
    def location(self) -> str | None:
        return self._resolved.location

    @property
    def table_type(self) -> str | None:
        t = self._resolved.table_type
        return t.value if t else None

    @property
    def securable_kind(self) -> str | None:
        return self._resolved.securable_kind

    @property
    def is_catalog_managed(self) -> bool:
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
            fresh = self._connection.catalog.resolve(ref)
        except InvalidReferenceError:
            if before_write:
                raise
            return
        except DeltaSwampError:
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
                    "re-open it with conn.table(...) to write to the new table"
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

        `shape` takes the same arguments as the call itself, so you can
        preflight the write you actually intend::

            t.can("create", properties={"delta.enableRowTracking": "true"})
            t.can("append", schema_mode="merge")
        """
        try:
            op = Operation(operation)
        except ValueError:
            raise InvalidArgumentError(
                f"{operation!r} is not an operation; one of {sorted(o.value for o in Operation)}"
            ) from None
        # Translate the call's arguments the way the call itself routes them.
        # Passing them through as a bare shape let `can("append",
        # schema_mode="merge")` answer for a plain APPEND (kernel: yes) while
        # the append itself routed as MERGE_SCHEMA and was refused.
        op, needs = self._call_route(op, shape)
        return self._connection.router.capability(op, self._enrich(), needs=needs, **shape)

    def _call_route(self, op: Operation, shape: dict[str, Any]) -> tuple[Operation, frozenset[str]]:
        """The operation and needs a call with these arguments routes on."""
        needs: set[str] = set()
        get = shape.get
        if op in (
            Operation.APPEND,
            Operation.OVERWRITE,
            Operation.REPLACE_WHERE,
            Operation.MERGE_SCHEMA,
        ):
            partition_overwrite = get("partition_overwrite") or "static"
            needs |= self._write_needs(
                get("schema_mode"),
                get("commit_metadata"),
                get("txn"),
                get("writer_properties"),
                partition_overwrite,
            )
            if op is Operation.APPEND and get("schema_mode") == "merge":
                op = Operation.MERGE_SCHEMA
            elif op is Operation.OVERWRITE and (
                get("predicate") is not None or partition_overwrite == "dynamic"
            ):
                op = Operation.REPLACE_WHERE
        elif op in (Operation.SCAN, Operation.TIME_TRAVEL):
            if get("predicate") is not None:
                needs.add("predicates")
            if get("timestamp") is not None:
                needs.add("timestamp_travel")
            if (
                get("version") is not None
                or get("timestamp") is not None
                or self._version is not None
            ):
                op = Operation.TIME_TRAVEL
        elif op in (Operation.OPTIMIZE, Operation.ZORDER):
            if get("zorder_by"):
                op = Operation.ZORDER
            if get("full"):
                needs.add("optimize_full")
            if get("predicate") is not None:
                needs.add("optimize_predicate")
        elif op is Operation.CLUSTER_BY:
            columns = get("columns")
            if isinstance(columns, str) and columns.lower() == "auto":
                needs.add("auto_clustering")
        return op, frozenset(needs)

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
        return self._connection.router.engine_for(operation, resolved, needs=needs, **shape)

    def _read(
        self,
        operation: Operation,
        call: Callable[[Any], Any],
        needs: frozenset[str] = frozenset(),
    ) -> Any:
        """Serve a read-only operation, moving to the next engine if one breaks.

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
        tried: set[EngineKind] = set()
        refusals: list[EngineLimitError] = []
        while True:
            try:
                engine: Any = self._connection.router.engine_for(
                    operation, self._enrich(), needs=needs, exclude=frozenset(tried)
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
            except DeltaSwampError:
                raise
            except Exception as exc:
                tried.add(engine.kind)
                try:
                    self._connection.router.engine_for(
                        operation, self._resolved, needs=needs, exclude=frozenset(tried)
                    )
                except DeltaSwampError:
                    if refusals:
                        raise refusals[0] from refusals[0].__cause__
                    raise exc from None
                warnings.warn(
                    f"{engine.kind.value} failed to serve {operation.value} "
                    f"({type(exc).__name__}: {str(exc)[:200]}); trying the next engine",
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
        timestamp: str | None = None,
        limit: int | None = None,
    ) -> Any:
        """Read the table as an Arrow stream (exports `__arrow_c_stream__`).

        `limit` is a hint passed to the engine. Streaming engines ignore it and
        the caller simply stops reading; the SQL warehouse turns it into a real
        `LIMIT`, because it computes the result set before streaming any of it.
        """
        columns = _columns_arg(columns)
        _check_predicate(predicate, "scan")
        _check_count(limit, "limit")
        timestamp = _timestamp_arg(timestamp)
        version = self._travel_version(version, timestamp)
        # `version or timestamp` would treat version 0 as no time travel. A
        # handle opened at a version is time travel too, and must route as it:
        # an engine that serves SCAN but not TIME_TRAVEL would read the latest.
        travelling = version is not None or timestamp is not None
        op = Operation.TIME_TRAVEL if travelling else Operation.SCAN
        # A predicate or a timestamp narrows which engines can serve the call,
        # so say so up front instead of letting one accept and then raise.
        needs = set()
        if predicate is not None:
            needs.add("predicates")
        if timestamp is not None:
            needs.add("timestamp_travel")

        def scan(engine: Any) -> Any:
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
            # A file VACUUM (or a manual delete) removed fails only once reading
            # reaches it, as a bare OSError/ArrowInvalid; name it instead.
            where = self._resolved.location or str(self._resolved.ref)
            at = version if version is not None else timestamp
            context = f"{where}" + (f" at {at}" if at is not None else "")
            return translating_stream(stream, context)

        return self._read(op, scan, frozenset(needs))

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
        of pairs in pandas) no longer fails inference. Streams are passed on
        untouched -- aligning them would mean reading them here.
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
        if not (is_pandas or is_polars or isinstance(data, (pa.Table, pa.RecordBatch))):
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

        names = [canonical(n) for n in data.column_names]
        if len(set(names)) != len(names):
            return data  # two columns fold to one name; let the engine refuse
        if names != data.column_names:
            data = data.rename_columns(names)
        for index, name in enumerate(names):
            column_type = data.schema.field(index).type
            wanted = by_name[name].type if name in by_name else None
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
                # the table's type keeps every value or raises here.
                data = data.set_column(
                    index, by_name[name], data.column(index).cast(wanted, safe=True)
                )
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
        for field in missing:
            # A left-out partition column is almost always a mistake (and a
            # dynamic overwrite derives its partitions from the data), so it
            # is never filled; nor is a required column. The engine refuses.
            if not field.nullable or field.name in partitions:
                return data
        for field in missing:
            data = data.append_column(field, pa.nulls(data.num_rows, field.type))
        return data

    def _check_writable(self, what: str) -> None:
        """Refuse a write through a handle opened at a past version.

        Every engine writes to the latest version, so a delete on
        ``conn.table(path, version=1)`` removed rows from the current table,
        not the version the handle shows. Delta refuses writes to a
        time-travelled table for the same reason.
        """
        if self._version is not None:
            raise InvalidArgumentError(
                f"cannot {what}: this handle is pinned to version {self._version}; "
                "open the table without version= to write to it"
            )
        if self._resolved.is_catalog_managed:
            # The catalog ratifies only latest+1, so a write from the tail
            # captured at resolution was a guaranteed 409 once anyone else had
            # committed -- and stayed one on every retry through this handle.
            self._refresh_commit_tail(before_write=True)

    def to_arrow(self, **kwargs: Any) -> Any:
        pa = _require("pyarrow", "pyarrow")
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
        _require("pandas", "pandas")
        return self.to_arrow(**kwargs).to_pandas()

    def to_polars(self, *, lazy: bool = False, **kwargs: Any) -> Any:
        """A Polars DataFrame, or a LazyFrame over it with ``lazy=True``.

        The lazy form still reads through this library, so it works on the
        tables `polars.scan_delta` cannot open (catalog-managed, row-tracked,
        vacuumProtocolCheck, ...).
        """
        pl = _require("polars", "polars")
        frame = pl.DataFrame(self.to_arrow(**kwargs))
        return frame.lazy() if lazy else frame

    def to_duckdb(self, connection: Any = None, *, name: str | None = None, **kwargs: Any) -> Any:
        """A DuckDB relation over the table. With `name`, also a view of that name.

        DuckDB's own delta extension is C++ and knows nothing of Unity Catalog
        credentials or catalog-managed commits; this hands it the rows instead.
        """
        duckdb = _require("duckdb", "duckdb")
        data = self.to_arrow(**kwargs)
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
        operation = Operation.OVERWRITE if mode == "overwrite" else Operation.APPEND
        needs = {"distributed_write"}
        if txn is not None:
            needs.add("idempotent_txn")
        if commit_metadata is not None:
            needs.add("commit_metadata")
        if txn is not None and self._already_committed(txn):
            raise UnreachableTableError(
                f"plan an idempotent write for {txn[0]!r} at version {txn[1]}",
                "that transaction is already committed, so running the job would "
                "duplicate work whose result is already in the table",
                "raise the txn version, or drop txn= to write unconditionally",
            )
        engine = self._engine(operation, frozenset(needs))
        return WritePlan(
            engine=engine,
            table=self._enrich(),
            mode=mode,
            version=self.version,
            txn=txn,
            commit_metadata=commit_metadata,
            ship_catalog_auth=bool(ship_catalog_auth),
            catalog=(
                self._connection.catalog
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
        snapshot version and vends its own credentials.
        """
        from .distributed import ScanPlan

        columns = _columns_arg(columns)
        _check_predicate(predicate, "plan a scan")
        timestamp = _timestamp_arg(timestamp)
        version = self._travel_version(version, timestamp)
        needs = {"distributed_scan"}
        if predicate is not None:
            needs.add("predicates")
        if timestamp is not None:
            needs.add("timestamp_travel")
        travelling = version is not None or timestamp is not None
        op = Operation.TIME_TRAVEL if travelling else Operation.SCAN
        engine = self._engine(op, frozenset(needs))
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
        """A Daft DataFrame."""
        daft = _require("daft", "daft")
        return daft.from_arrow(self.to_arrow(**kwargs))

    def to_pyarrow_dataset(self, **kwargs: Any) -> Any:
        dataset = _require("pyarrow.dataset", "pyarrow")
        return dataset.dataset(self.to_arrow(**kwargs))

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

        Streams the narrowest column rather than materializing the table; the
        engines' statistics-based counts are approximate by their own
        documentation (a file without stats counts as zero rows), so they are
        not used here.
        """
        pa = _require("pyarrow", "pyarrow")
        schema = self.schema()
        names = list(getattr(schema, "names", None) or [f.name for f in schema])
        narrow = [names[0]] if names and predicate is None else None
        total = 0
        stream = self.scan(columns=narrow, predicate=predicate)
        if not isinstance(stream, TranslatingStream):
            stream = pa.RecordBatchReader.from_stream(stream)
        for batch in stream:
            total += batch.num_rows
        return total

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
        result: dict[str, Any] = self._read(
            Operation.DETAIL, lambda engine: engine.detail(self._resolved, version=self._version)
        )
        return result

    def cdf(self, **kwargs: Any) -> Any:
        """The change data feed, by version or timestamp range.

        Rows carry `_change_type`, `_commit_version` and `_commit_timestamp`.
        """
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
        return _cdf_types(
            self._read(Operation.CDF, lambda engine: engine.cdf(self._resolved, **given))
        )

    def changes(
        self,
        starting_version: int,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        poll_interval: float | None = None,
    ) -> Any:
        """Follow the change feed, one committed version at a time.

        Yields ``(version, pyarrow.Table)`` for every version from
        `starting_version` on, in order. With `poll_interval` (seconds) it keeps
        waiting for new commits, like a streaming read with a change-feed
        source; without, it stops at the latest version. Record the last
        version you processed and pass the next one to resume.
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
        pa = _require("pyarrow", "pyarrow")
        columns = _columns_arg(columns)
        # Splitting by version needs `_commit_version`; a projection without
        # it crashed with "Invalid sort key column". Read it, then drop it.
        projection = columns
        if columns is not None and "_commit_version" not in columns:
            projection = [*columns, "_commit_version"]
        next_version = starting_version
        while True:
            current = self._connection._reresolve(self)
            latest = current.version
            if latest is not None and latest >= next_version:
                changes = pa.table(
                    current.cdf(
                        starting_version=next_version,
                        ending_version=latest,
                        columns=projection,
                        predicate=predicate,
                    )
                )
                if changes.num_rows:
                    changes = _plain_views(changes).sort_by("_commit_version")
                    versions = changes.column("_commit_version").to_pylist()
                    for version in sorted(set(versions)):
                        mask = pa.compute.equal(changes.column("_commit_version"), version)
                        chunk = changes.filter(mask)
                        if columns is not None:
                            chunk = chunk.select(columns)
                        yield int(version), chunk
                next_version = latest + 1
            if poll_interval is None:
                return
            time.sleep(poll_interval)

    # -------------------------------------------------------------- metadata

    def schema(self) -> Any:
        """The table's Arrow schema, as a ``pyarrow.Schema`` when pyarrow is installed."""
        pinned = self._version is not None
        engine = self._engine(Operation.TIME_TRAVEL if pinned else Operation.SCAN)
        snapshot = getattr(engine, "snapshot", None)
        if snapshot is not None:
            schema = snapshot(self._resolved, version=self._version).schema()
        else:
            # Open the stream and take its schema; reading the whole table to
            # learn its columns cost a full scan on every count().
            schema = self.scan(limit=0)
        try:
            import pyarrow as pa
        except ImportError:
            return schema
        # The kernel hands back an arro3 Schema; every other engine a pyarrow
        # one. Return one type, whichever engine served it.
        if isinstance(schema, pa.Schema):
            return schema
        if hasattr(schema, "__arrow_c_stream__"):
            return pa.RecordBatchReader.from_stream(schema).schema
        return pa.schema(schema)

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
    ) -> None:
        """Append data.

        `schema_mode="merge"` widens the table schema to fit the data, and
        routes as MERGE_SCHEMA rather than a plain append.

        `txn=(app_id, version)` makes the write idempotent. Note that neither
        engine enforces this: delta-rs records the transaction identifier but
        happily appends the same one twice. So the check happens here, by
        comparing against the last committed version before writing. That
        removes the common replay case; it is not a substitute for engine-level
        enforcement, because a concurrent writer could still commit in between.
        """
        self._check_writable("append")
        if schema_mode not in (None, "merge"):
            raise InvalidArgumentError(
                f"append takes schema_mode=None or 'merge', not {schema_mode!r}; replacing "
                "the schema is overwrite(..., schema_mode='overwrite')"
            )
        _check_txn(txn)
        data = _write_data(data)
        if txn is not None and self._already_committed(txn):
            return
        raw = data
        for attempt in range(_REALIGN_ATTEMPTS):
            data = self._align(raw, schema_mode)
            needs = self._write_needs(
                schema_mode, commit_metadata, txn, writer_properties, "static"
            )
            needs |= self._data_needs(data, partition_by)
            op = Operation.MERGE_SCHEMA if schema_mode == "merge" else Operation.APPEND
            try:
                self._raced_append(
                    lambda data=data, op=op, needs=needs: self._engine(op, needs).append(
                        self._resolved,
                        data,
                        schema_mode=schema_mode,
                        partition_by=partition_by,
                        target_file_size=target_file_size,
                        writer_properties=writer_properties,
                        commit_metadata=commit_metadata,
                        txn=txn,
                        max_commit_retries=max_commit_retries,
                    ),
                    data,
                    txn,
                    max_commit_retries,
                )
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
    ) -> None:
        """Replace data.

        With `predicate`, replaces only matching rows (`replaceWhere`). With
        `partition_overwrite="dynamic"`, replaces exactly the partitions present
        in `data` and leaves the rest alone, which is Spark's
        `partitionOverwriteMode=dynamic`.
        """
        self._check_writable("overwrite")
        if schema_mode not in (None, "merge", "overwrite"):
            raise InvalidArgumentError(
                f"schema_mode must be None, 'merge' or 'overwrite', not {schema_mode!r}"
            )
        _check_predicate(predicate, "overwrite")
        _check_txn(txn)
        data = _write_data(data)
        if txn is not None and self._already_committed(txn):
            return
        if (
            partition_overwrite == "dynamic"
            and predicate is None
            and getattr(data, "num_rows", None) == 0
            and hasattr(data, "column_names")
        ):
            # Spark's dynamic mode replaces the partitions present in the
            # data; an empty batch names none, so it changes nothing. It used
            # to raise, failing any pipeline whose batch happened to be empty.
            return
        raw = data
        data = self._align(raw, schema_mode)
        needs = self._write_needs(
            schema_mode, commit_metadata, txn, writer_properties, partition_overwrite
        )
        needs |= self._data_needs(data)
        op = (
            Operation.REPLACE_WHERE
            if (predicate is not None or partition_overwrite == "dynamic")
            else Operation.OVERWRITE
        )
        from .errors import CommitConflictError

        for attempt in range(_REALIGN_ATTEMPTS):
            try:
                self._backfilled(
                    lambda data=data: self._engine(op, needs).overwrite(
                        self._resolved,
                        data,
                        predicate=predicate,
                        partition_overwrite=partition_overwrite,
                        schema_mode=schema_mode,
                        target_file_size=target_file_size,
                        writer_properties=writer_properties,
                        commit_metadata=commit_metadata,
                        txn=txn,
                        max_commit_retries=max_commit_retries,
                    ),
                    data,
                )
                break
            except CommitConflictError:
                # Lost to a writer that committed this very txn: already done.
                if txn is None or not self._txn_landed(txn):
                    raise
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

    def replace(self, data: Any, **kwargs: Any) -> None:
        """Replace the table's contents and schema. REPLACE TABLE / RTAS."""
        self.overwrite(data, schema_mode="overwrite", **kwargs)

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
        if type(exc).__name__ != "SchemaMismatchError":
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
        try:
            engine = self._engine(Operation.APPEND, frozenset({"idempotent_txn"}))
        except DeltaSwampError:
            engine = self._engine(Operation.APPEND)
        if callable(getattr(engine, "txn_version", None)):
            version: int | None = engine.txn_version(self._resolved, app_id)
            return version
        # The writing engine cannot read transaction ids (the kernel, the
        # warehouse); ask delta-rs, which reads them from the same log.
        fallback: Any = self._connection.router.engines.get(EngineKind.DELTARS)
        reason = f"the {type(engine).__name__} engine cannot read transaction identifiers"
        if fallback is not None and callable(getattr(fallback, "txn_version", None)):
            available = getattr(fallback, "available", None)
            if available is None or available():
                try:
                    version = fallback.txn_version(self._resolved, app_id)
                    return version
                except Exception as exc:
                    reason += f", and delta-rs could not read this table's log ({exc})"
        raise UnreachableTableError(
            f"check transaction {app_id!r}",
            reason,
            "write without txn= and deduplicate yourself, or use a table delta-rs can read",
        )

    def delete(self, predicate: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """DELETE rows matching a SQL predicate (every row when None)."""
        self._check_writable("delete")
        _check_options("delete", kwargs, _DML_OPTIONS)
        _check_predicate(predicate, "delete")
        result: dict[str, Any] = self._backfilled(
            lambda: self._engine(Operation.DELETE).delete(
                self._resolved, predicate, **_given(kwargs)
            )
        )
        self._invalidate()
        return result

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
        self._check_writable("update")
        _check_options("update", kwargs, _DML_OPTIONS | {"error_on_type_mismatch"})
        _check_predicate(predicate, "update")
        if updates is not None and new_values is not None:
            raise InvalidArgumentError("pass updates (SQL expressions) or new_values, not both")
        if not updates and not new_values:
            raise InvalidArgumentError("update needs at least one column to set")
        needs = self._update_needs(updates, False) | self._update_needs(new_values, True)
        engine = self._engine(Operation.UPDATE, needs)
        # delta-rs skips a SET target it cannot find -- an unknown name, a
        # different case, a nested field -- and still rewrites every matched
        # file, reporting the rows as updated while changing nothing.
        strict = isinstance(engine, DeltaRsEngine)
        if updates is not None:
            updates = self._update_targets(updates, strict)
        if new_values is not None:
            kwargs["new_values"] = self._update_targets(new_values, strict)
        result: dict[str, Any] = self._backfilled(
            lambda: engine.update(
                self._resolved, updates=updates, predicate=predicate, **_given(kwargs)
            )
        )
        self._invalidate()
        return result

    def _update_targets(self, targets: dict[str, Any], strict: bool) -> dict[str, Any]:
        """Resolve UPDATE's SET targets against the table's columns.

        Delta column names are case-insensitive, so a case-only mismatch maps
        to the real column. A name that matches nothing is refused; so is a
        nested path when `strict` (delta-rs cannot set struct fields).
        """
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
            bare = key[1:-1] if len(key) > 1 and key[0] == key[-1] == "`" else key
            if bare in names:
                target = key
            elif len(folded.get(bare.lower(), ())) == 1:
                target = folded[bare.lower()][0]
            elif "." in bare and not strict:
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
        self._check_writable("merge")
        if predicate is None:
            raise InvalidArgumentError("merge needs a join predicate")
        _check_predicate(predicate, "merge")
        source = _write_data(source)
        needs = self._data_needs(source)
        builder = self._engine(Operation.MERGE, needs).merge(
            self._resolved, source, predicate, **kwargs
        )
        return _InvalidatingMerger(builder, self._invalidate)

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
        if isinstance(zorder_by, str):
            zorder_by = [zorder_by]
        if zorder_by:
            self._check_zorder(list(zorder_by))
        op = Operation.ZORDER if zorder_by else Operation.OPTIMIZE
        needs = set()
        if full:
            needs.add("optimize_full")
        if predicate is not None:
            needs.add("optimize_predicate")
        result: dict[str, Any] = self._engine(op, frozenset(needs)).optimize(
            self._resolved, zorder_by=zorder_by, full=full, predicate=predicate, **kwargs
        )
        self._invalidate()
        return result

    def z_order(self, columns: list[str] | str, **kwargs: Any) -> dict[str, Any]:
        self._check_writable("z-order")
        columns = [columns] if isinstance(columns, str) else list(columns or [])
        if not columns:
            raise InvalidArgumentError("z_order needs at least one column")
        self._check_zorder(columns)
        result: dict[str, Any] = self._engine(Operation.ZORDER).zorder(
            self._resolved, columns, **kwargs
        )
        self._invalidate()
        return result

    def _check_zorder(self, columns: list[str]) -> None:
        """Z-order keys must be data columns: partition columns are constant per file."""
        names = list(self.schema().names)
        partitions = set(self._enrich().partition_columns)
        for column in columns:
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
        if retention_hours is not None and (
            isinstance(retention_hours, bool) or not isinstance(retention_hours, (int, float))
        ):
            raise InvalidArgumentError(
                f"retention_hours must be a number, not {type(retention_hours).__name__}"
            )
        if retention_hours is not None and retention_hours < 0:
            raise InvalidArgumentError(f"retention_hours must be >= 0, got {retention_hours}")
        # The shape lets the router accept a dry run on tables a real VACUUM,
        # which commits, cannot touch.
        result = self._engine(Operation.VACUUM, dry_run=bool(dry_run)).vacuum(
            self._resolved, retention_hours=retention_hours, dry_run=dry_run, lite=lite, **kwargs
        )
        self._invalidate()
        return result

    def restore(self, target: Any, **kwargs: Any) -> dict[str, Any]:
        """RESTORE to a version (int) or a timestamp (datetime or string)."""
        self._check_writable("restore")
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
        if isinstance(target, (bool, int)):
            # -1 reached delta-rs as "either the version or datetime should
            # be provided"; True restored version 1.
            _check_version(target, "restore version")
            latest = self._connection._reresolve(self).version
            if latest is not None and target > latest:
                raise InvalidArgumentError(
                    f"cannot restore version {target}: the latest version is {latest}"
                )
        result: dict[str, Any] = self._engine(Operation.RESTORE).restore(
            self._resolved, target, **kwargs
        )
        self._invalidate()
        return result

    def repair(self, **kwargs: Any) -> dict[str, Any]:
        # A dry run commits nothing, so the router may accept it on tables a
        # real REPAIR (which commits removes) cannot touch.
        result: dict[str, Any] = self._engine(
            Operation.REPAIR, dry_run=kwargs.get("dry_run") is True
        ).repair(self._resolved, **kwargs)
        self._invalidate()
        return result

    # ----------------------------------------------------------------- schema

    def add_column(self, fields: Any, **kwargs: Any) -> None:
        """Add columns. `fields` is a list of Arrow/Delta fields, or a
        {name: sql_type} mapping when the SQL fallback serves it."""
        self._check_writable("add a column")
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
        self._engine(Operation.ADD_COLUMN).add_columns(self._resolved, fields, **kwargs)
        self._invalidate()

    def drop_column(self, column: str) -> dict[str, Any]:
        self._check_writable("drop a column")
        if not isinstance(column, str) or not column:
            raise InvalidArgumentError(f"drop_column takes one column name, not {column!r}")
        names = list(self.schema().names)
        partitions = set(self._enrich().partition_columns)
        rest = [name for name in names if name != column]
        if column in names and rest and set(rest) <= partitions:
            # Delta needs a data column; the kernel panicked reading the
            # table this left behind.
            raise InvalidArgumentError(
                f"cannot drop {column!r}: it is the last non-partition column"
            )
        if column in names and not rest:
            raise InvalidArgumentError(f"cannot drop {column!r}: it is the table's only column")
        result = self._engine(Operation.DROP_COLUMN).drop_column(self._resolved, column)
        self._invalidate()
        return _metrics(result)

    def rename_column(self, old: str, new: str) -> dict[str, Any]:
        self._check_writable("rename a column")
        result = self._engine(Operation.RENAME_COLUMN).rename_column(self._resolved, old, new)
        self._invalidate()
        return _metrics(result)

    def set_properties(self, properties: dict[str, str], **kwargs: Any) -> None:
        self._check_writable("set properties")
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
        self._engine(Operation.SET_PROPERTIES, properties=properties).set_properties(
            self._resolved, properties, **kwargs
        )
        self._invalidate()

    def add_feature(self, feature: Any, **kwargs: Any) -> None:
        self._check_writable("add a feature")
        names = list(feature) if isinstance(feature, (list, tuple, set, frozenset)) else [feature]
        self._engine(Operation.ADD_FEATURE, features=names).add_feature(
            self._resolved, feature, **kwargs
        )
        self._invalidate()

    def drop_feature(self, feature: str, **kwargs: Any) -> dict[str, Any]:
        """Drop a table feature. Databricks-only, so it needs the SQL fallback."""
        self._check_writable("drop a feature")
        result: dict[str, Any] = self._engine(Operation.DROP_FEATURE).drop_feature(
            self._resolved, feature, **kwargs
        )
        self._invalidate()
        return result

    def add_constraint(self, constraints: dict[str, str], **kwargs: Any) -> None:
        self._check_writable("add a constraint")
        if not isinstance(constraints, dict) or not constraints:
            raise InvalidArgumentError("add_constraint needs at least one {name: expression}")
        for cname, expression in constraints.items():
            if not isinstance(cname, str) or not cname.strip():
                raise InvalidArgumentError(f"constraint names must be non-empty, got {cname!r}")
            _check_predicate(expression, f"add constraint {cname}")
            if expression is None:
                raise InvalidArgumentError(f"constraint {cname!r} has no expression")
        self._engine(Operation.ADD_CONSTRAINT).add_constraint(self._resolved, constraints, **kwargs)
        self._invalidate()

    def drop_constraint(self, name: str, *, if_exists: bool = False) -> None:
        self._check_writable("drop a constraint")
        self._engine(Operation.DROP_CONSTRAINT).drop_constraint(
            self._resolved, name, if_exists=if_exists
        )
        self._invalidate()

    def unset_properties(self, keys: list[str] | str, *, if_exists: bool = True) -> None:
        """ALTER TABLE ... UNSET TBLPROPERTIES. delta-rs cannot remove a property."""
        self._check_writable("unset properties")
        names = [keys] if isinstance(keys, str) else list(keys)
        if not names:
            # The kernel committed an empty metadata change; SQL raised.
            return
        self._engine(Operation.UNSET_PROPERTIES).unset_properties(
            self._resolved, names, if_exists=if_exists
        )
        self._invalidate()

    def set_comment(self, comment: str | None) -> None:
        """The table comment (the Metadata action's description)."""
        self._check_writable("set the comment")
        _check_comment(comment)
        self._engine(Operation.SET_COMMENT).set_comment(self._resolved, comment)
        self._invalidate()

    def set_column_comment(self, column: str, comment: str | None) -> None:
        self._check_writable("set a column comment")
        _check_comment(comment)
        self._engine(Operation.SET_COLUMN_COMMENT).set_column_comment(
            self._resolved, column, comment
        )
        self._invalidate()

    def alter_column_type(self, column: str, new_type: str) -> None:
        """Widen a column's type without rewriting data (type widening).

        Allowed: byte->short->int->long, float->double, byte/short/int->double,
        date->timestamp_ntz, and decimals whose precision and scale do not
        shrink. The table needs ``delta.enableTypeWidening = true``.
        """
        self._check_writable("change a column type")
        self._engine(Operation.ALTER_COLUMN_TYPE).alter_column_type(
            self._resolved, column, new_type
        )
        self._invalidate()

    def set_not_null(self, column: str) -> None:
        """Add a NOT NULL constraint, after checking no existing row is null."""
        self._check_writable("set NOT NULL")
        self._engine(Operation.SET_NOT_NULL).set_not_null(self._resolved, column)
        self._invalidate()

    def drop_not_null(self, column: str) -> None:
        self._check_writable("drop NOT NULL")
        self._engine(Operation.DROP_NOT_NULL).drop_not_null(self._resolved, column)
        self._invalidate()

    def cluster_by(self, columns: list[str] | str | None) -> None:
        """Set the liquid-clustering keys (ALTER TABLE ... CLUSTER BY).

        ``None`` or ``[]`` is CLUSTER BY NONE. ``"auto"`` asks Databricks to
        choose keys, which only the SQL fallback can do. New keys apply to data
        written afterwards; existing files are reclustered by OPTIMIZE.
        """
        self._check_writable("change clustering")
        auto = isinstance(columns, str) and columns.lower() == "auto"
        needs = frozenset({"auto_clustering"}) if auto else frozenset()
        self._engine(Operation.CLUSTER_BY, needs).cluster_by(self._resolved, columns)
        self._invalidate()

    # ------------------------------------------------------- log and layout

    def checkpoint(self) -> None:
        self._engine(Operation.CHECKPOINT).checkpoint(self._resolved)

    def compact_logs(self, start: int | None = None, end: int | None = None) -> Any:
        return self._engine(Operation.LOG_COMPACTION).compact_logs(self._resolved, start, end)

    def cleanup_metadata(self) -> None:
        """Delete log files older than ``delta.logRetentionDuration``.

        This is what makes versions past log retention unreachable by time
        travel, so it is never done implicitly.
        """
        self._engine(Operation.CLEANUP_METADATA).cleanup_metadata(self._resolved)

    def analyze(self, *, columns: list[str] | None = None, delta_statistics: bool = False) -> Any:
        """ANALYZE TABLE. Databricks-only, so it needs the SQL fallback."""
        return self._engine(Operation.ANALYZE).analyze(
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
        return self._engine(Operation.REFRESH).refresh(self._resolved, full=full)

    def generate(self) -> None:
        """Write symlink manifests, for engines that read those instead of the log."""
        self._engine(Operation.GENERATE).generate(self._resolved)

    def reorg(self, **kwargs: Any) -> dict[str, Any]:
        """REORG TABLE. Databricks-only, so it needs the SQL fallback."""
        self._check_writable("reorg")
        result: dict[str, Any] = self._engine(Operation.REORG).reorg(self._resolved, **kwargs)
        self._invalidate()
        return result

    def clone(self, target: str, **kwargs: Any) -> dict[str, Any]:
        """CLONE. Databricks-only, so it needs the SQL fallback."""
        result: dict[str, Any] = self._engine(Operation.CLONE).clone(
            self._resolved, target, **kwargs
        )
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
        return _governed(self._connection.catalog, "GovernedCatalog", what)

    def info(self) -> Any:
        """The catalog's view of the table: owner, comment, columns, row filter,
        column masks, predictive optimization, audit timestamps."""
        cat = self._governance("read table info")
        return _call("read table info", cat.table_info, self._resolved.ref)

    def grants(self, principal: str | None = None) -> list[Any]:
        cat = self._governance("read grants")
        return list(_call("read grants", cat.grants, self._resolved.ref, principal))

    def effective_grants(self, principal: str | None = None) -> list[Any]:
        """Grants including those inherited from the schema and catalog."""
        cat = self._governance("read effective grants")
        return list(
            _call("read effective grants", cat.effective_grants, self._resolved.ref, principal)
        )

    def grant(self, principal: str, privileges: list[str] | str) -> list[Any]:
        names = [privileges] if isinstance(privileges, str) else list(privileges)
        cat = self._governance("grant")
        return list(_call("grant", cat.grant, self._resolved.ref, principal, names))

    def revoke(self, principal: str, privileges: list[str] | str) -> list[Any]:
        names = [privileges] if isinstance(privileges, str) else list(privileges)
        cat = self._governance("revoke")
        return list(_call("revoke", cat.revoke, self._resolved.ref, principal, names))

    def tags(self, column: str | None = None) -> dict[str, str]:
        cat = self._governance("read tags")
        return dict(_call("read tags", cat.tags, self._resolved.ref, column))

    def set_tags(self, tags: dict[str, str], *, column: str | None = None) -> None:
        cat = self._governance("set tags")
        _call("set tags", cat.set_tags, self._resolved.ref, tags, column)

    def unset_tags(self, keys: list[str] | str, *, column: str | None = None) -> None:
        names = [keys] if isinstance(keys, str) else list(keys)
        cat = self._governance("unset tags")
        _call("unset tags", cat.unset_tags, self._resolved.ref, names, column)

    def set_owner(self, principal: str) -> None:
        cat = self._governance("set owner")
        _call("set owner", cat.set_owner, self._resolved.ref, principal)

    def lineage(self, direction: str = "both") -> Any:
        """Upstream and downstream tables, notebooks, jobs and dashboards."""
        cat = self._governance("read lineage")
        return _call("read lineage", cat.lineage, self._resolved.ref, direction)

    def column_lineage(self, column: str, direction: str = "both") -> Any:
        cat = self._governance("read column lineage")
        return _call(
            "read column lineage", cat.column_lineage, self._resolved.ref, column, direction
        )

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
        if engine is None or not self._connection.router.allow_sql_fallback:
            raise FallbackRequiredError(
                what,
                "row filters and column masks are defined in SQL and enforced by Databricks",
                SQL_FALLBACK_REMEDY,
            )
        return engine

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
        self._warehouse("set a row filter").set_row_filter(self._resolved, function_name, columns)
        self._refresh_from_catalog()

    def drop_row_filter(self) -> None:
        self._warehouse("drop a row filter").drop_row_filter(self._resolved)
        self._refresh_from_catalog()

    def set_column_mask(
        self, column: str, function_name: str, *, using_columns: list[str] | None = None
    ) -> None:
        self._warehouse("set a column mask").set_column_mask(
            self._resolved, column, function_name, using_columns
        )
        self._refresh_from_catalog()

    def drop_column_mask(self, column: str) -> None:
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
    stats = [json.loads(s) if s else {} for s in files.column("stats").to_pylist()]
    missing = object()

    def lookup(entry: Any, path: tuple[str, ...]) -> Any:
        for part in path:
            if not isinstance(entry, dict) or part not in entry:
                return missing
            entry = entry[part]
        return entry

    def typed(values: list[Any], wanted: Any) -> Any:
        try:
            return pa.array(values).cast(wanted)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
            try:
                return pa.array(values)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
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
_CDF_META = ("_change_type", "_commit_version", "_commit_timestamp")


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
    """

    def __init__(self, builder: Any, invalidate: Any) -> None:
        self._builder = builder
        self._invalidate = invalidate

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not set in __init__. Before __init__ runs
        # (copy, pickle) `_builder` itself lands here, and looking it up on
        # itself recursed until RecursionError.
        if name.startswith("__") or name in ("_builder", "_invalidate"):
            raise AttributeError(name)
        attr = getattr(self._builder, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            if name == "execute":
                self._invalidate()
                return result
            # Clause methods return the builder; keep the wrapper in the chain.
            if result is self._builder or type(result) is type(self._builder):
                self._builder = result
                return self
            return result

        return call
