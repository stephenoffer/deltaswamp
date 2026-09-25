"""The public surface: `connect()`, `Connection`, and `Table`.

One object per table, reads and writes, regardless of what is behind it. Where
something cannot be done, `capabilities()` says so in advance and the operation
raises with the same reason rather than failing obscurely halfway through.

Protocol features are discovered in two stages. The catalog supplies enough to
make the decisions that must precede opening the log (is this a view, a shallow
clone, a table vending refuses). The full reader/writer feature lists only exist
in the log itself, so the first operation enriches the resolved table with them
and everything afterwards routes on the complete picture.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from .capability import Capability, Operation
from .capability import Engine as EngineKind
from .catalog import Catalog, ResolvedTable, TableType
from .catalog.filesystem import FilesystemCatalog
from .catalog.registry import catalog_for_uri
from .credentials import Operation as CredentialOperation
from .engine.deltars import DeltaRsEngine
from .engine.kernel import KernelEngine
from .errors import (
    CorruptTableError,
    DeltaSwampError,
    FallbackRequiredError,
    InvalidArgumentError,
    InvalidReferenceError,
    UnreachableTableError,
)
from .identity import RefKind, parse_ref
from .router import Router

__all__ = ["Connection", "InvalidArgumentError", "Table", "connect"]


def connect(
    uri: str | None = None,
    *,
    profile: str | None = None,
    host: str | None = None,
    token: str | None = None,
    config: Any = None,
    catalog: Catalog | None = None,
    allow_sql_fallback: bool = False,
    warehouse_id: str | None = None,
    staging_volume: str | None = None,
    storage_options: dict[str, str] | None = None,
    default_catalog: str | None = None,
    default_schema: str | None = None,
) -> Connection:
    """Open a connection.

    With no arguments, Databricks authentication is resolved by the SDK's usual
    precedence (explicit args, environment, `~/.databrickscfg`, cloud-native).
    We never reimplement that; `databricks.sdk.core.Config` already handles every
    permutation and this library just passes arguments through.

    Pass `token=` for an explicit personal access token, or leave it out and let
    the SDK find one. A PAT cannot be refreshed, so prefer OAuth M2M for anything
    that runs longer than the token's lifetime.

    `uri` may name an OSS Unity Catalog server (``uc://http://host:8080``) or a
    Hive metastore (``hms://thrift://host:9083``). Omit it for Databricks.

    Set `allow_sql_fallback=True` to permit routing through a SQL warehouse for
    operations no open-source engine implements. Without `warehouse_id` one is
    chosen automatically. `staging_volume="catalog.schema.volume"` lets the
    warehouse serve writes too: data is staged there as Parquet, loaded, and
    deleted. It is off by default because
    that reroute changes latency and cost by orders of magnitude, and a silent
    reroute is exactly the kind of surprise this library exists to avoid.
    """
    resolved_catalog = catalog or _default_catalog(
        uri, profile=profile, host=host, token=token, config=config
    )

    engines: dict[EngineKind, object] = {
        EngineKind.KERNEL: KernelEngine(storage_options=storage_options),
        EngineKind.DELTARS: DeltaRsEngine(storage_options=storage_options),
    }
    # Engines for tables that are not reached by storage location. Each serves
    # only its own kind of table and refuses the rest, so registering them
    # costs nothing for a connection that never meets one.
    from .engine.iceberg import IcebergEngine
    from .engine.sharing import SharingEngine

    if SharingEngine.available():
        engines[EngineKind.SHARING] = SharingEngine()
    if IcebergEngine.available():
        engines[EngineKind.ICEBERG] = IcebergEngine(token=token)

    if allow_sql_fallback:
        from .engine.sql import SqlEngine

        engines[EngineKind.SQL] = SqlEngine(
            profile=profile,
            host=host,
            token=token,
            config=config,
            warehouse_id=warehouse_id,
            staging_volume=staging_volume,
        )

    return Connection(
        catalog=resolved_catalog,
        router=Router(engines=engines, allow_sql_fallback=allow_sql_fallback),
        default_catalog=default_catalog,
        default_schema=default_schema,
        storage_options=dict(storage_options or {}),
    )


def _schema_of(data: Any) -> Any:
    """Infer an Arrow schema from whatever the caller passed."""
    data = _write_data(data)
    schema = getattr(data, "schema", None)
    # A Polars (or other) frame's `.schema` is its own type, not an Arrow
    # schema; only one that exports the Arrow C schema can create a table.
    if schema is not None and hasattr(schema, "__arrow_c_schema__"):
        return schema
    module = type(data).__module__ or ""
    if module.startswith("pandas") and type(data).__name__ == "DataFrame":
        pa = _require("pyarrow", "pyarrow")
        # The index is not data unless it is named (as the writers treat it);
        # inferring with it created an `__index_level_0__` column.
        unnamed = all(name is None for name in data.index.names)
        return pa.Schema.from_pandas(data, preserve_index=not unnamed)
    pa = _require("pyarrow", "pyarrow")
    return pa.table(data).schema


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


def _schema_arg(schema: Any) -> Any:
    """A create's schema: Arrow as given, else built with ``pyarrow.schema``.

    ``[("id", pa.int64())]`` and ``{"id": pa.int64()}`` are what pyarrow itself
    accepts, but only the managed create converted them; a path create failed
    with "Expected an object with dunder __arrow_c_schema__".
    """
    if schema is None:
        raise InvalidArgumentError("create_table needs a schema")
    try:
        import pyarrow as pa
    except ImportError:
        if hasattr(schema, "__arrow_c_schema__"):
            return schema
        raise
    if not isinstance(schema, pa.Schema):
        if hasattr(schema, "__arrow_c_schema__") and not isinstance(schema, (list, dict)):
            try:
                schema = pa.schema(schema)
            except Exception:
                return schema  # a Delta schema object; the engine reads it
        else:
            try:
                schema = pa.schema(schema)
            except (TypeError, ValueError, pa.ArrowInvalid) as exc:
                raise InvalidArgumentError(f"not a table schema: {schema!r} ({exc})") from exc
    return _widen_unsigned(schema)


def _widen_unsigned(schema: Any) -> Any:
    """Map unsigned integers to the signed type that holds all their values.

    Delta has no unsigned types, and the create mapped uint8 to byte: the
    table was created, then the first write failed with "Can't cast value 200
    to type Int8". uint64 has no wider Delta integer; it stays long, and a
    value above its range is refused by the write's safe cast.
    """
    import pyarrow as pa

    wider = {
        pa.uint8(): pa.int16(),
        pa.uint16(): pa.int32(),
        pa.uint32(): pa.int64(),
        pa.uint64(): pa.int64(),
    }

    def widen(t: Any, where: str) -> Any:
        if t in wider:
            return wider[t]
        if pa.types.is_struct(t):
            return pa.struct([f.with_type(widen(f.type, f"{where}.{f.name}")) for f in t])
        if pa.types.is_map(t):
            return pa.map_(widen(t.key_type, where), widen(t.item_type, where))
        if pa.types.is_list(t) or pa.types.is_large_list(t):
            item = t.value_field
            maker = pa.list_ if pa.types.is_list(t) else pa.large_list
            return maker(item.with_type(widen(item.type, where)))
        return t

    fields = [f.with_type(widen(f.type, f.name)) for f in schema]
    widened = pa.schema(fields, metadata=schema.metadata)
    return schema if widened.equals(schema) else widened


def _names_arg(value: Any, what: str) -> list[str] | None:
    """`partition_by` / `cluster_by`: one name becomes a list; names are strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    names = list(value)
    for name in names:
        if not isinstance(name, str):
            raise InvalidArgumentError(f"{what} names must be strings, got {name!r}")
    if len(set(names)) != len(names):
        raise InvalidArgumentError(f"{what} names a column twice: {names}")
    return names


def _schema_names(schema: Any) -> list[str] | None:
    """Top-level column names of an Arrow (or Arrow-exporting) schema."""
    names = getattr(schema, "names", None)
    if names is not None and not callable(names):
        return list(names)
    try:
        import pyarrow as pa

        return list(pa.schema(schema).names)
    except Exception:
        return None


def _check_layout(
    schema: Any, partition_by: list[str] | None, cluster_by: list[str] | None
) -> None:
    """Refuse partition/cluster columns the schema lacks, and all-partition tables.

    Missing names failed in the engines with "Generic delta kernel error"; a
    table whose every column is a partition column has no data column for a
    Parquet file to hold, which Delta refuses and the kernel panicked reading.
    """
    names = _schema_names(schema)
    if names is None:
        return
    for what, columns in (("partition_by", partition_by), ("cluster_by", cluster_by)):
        missing = [c for c in columns or () if c not in names]
        if missing:
            raise InvalidArgumentError(
                f"{what} names {missing}, which the schema does not have; columns are {names}"
            )
    if partition_by and names and set(names) <= set(partition_by):
        raise InvalidArgumentError(
            "every column is a partition column; a Delta table needs at least one "
            "non-partition column"
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


#: The save modes a create at a path takes.
_CREATE_MODES = frozenset({"error", "create", "ignore", "overwrite", "append"})


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


def _local_log_dir(name: str) -> str | None:
    """The `_delta_log` directory of a local-path table name, else None."""
    ref = parse_ref(name)
    if ref.kind is not RefKind.PATH:
        return None
    location = name
    if location.startswith("file://"):
        location = location[len("file://") :]
    elif "://" in location:
        return None
    return os.path.join(os.path.abspath(location), "_delta_log")


def _check_comment(comment: Any) -> None:
    """A comment is text or None; the kernel stored 123 as '123', delta-rs raised."""
    if comment is not None and not isinstance(comment, str):
        raise InvalidArgumentError(
            f"a comment must be a string or None, not {type(comment).__name__}"
        )


def _registered_anyway(catalog: Any, ref: Any, exc: BaseException) -> bool:
    """Whether a failed finalise had in fact registered the table."""
    text = str(exc)
    if " was created" in text and "reading it back failed" in text:
        return True
    exists = getattr(catalog, "table_exists", None)
    if not callable(exists):
        return False
    try:
        return bool(exists(ref))
    except Exception:
        return False


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


def _default_catalog(
    uri: str | None,
    *,
    profile: str | None,
    host: str | None,
    token: str | None,
    config: Any,
) -> Catalog:
    """Resolve the catalog for a connection URI via the plugin registry."""
    return catalog_for_uri(uri, profile=profile, host=host, token=token, config=config)


@dataclasses.dataclass
class Connection:
    """A bound catalog plus the engines available to it."""

    catalog: Catalog
    router: Router
    default_catalog: str | None = None
    default_schema: str | None = None
    storage_options: dict[str, str] = dataclasses.field(default_factory=dict)

    def table(self, name: str, *, version: int | None = None) -> Table:
        """Open a table by three-level name, or by path."""
        _check_version(version)
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
        catalog = FilesystemCatalog() if ref.kind is RefKind.PATH else self.catalog
        return Table(self, catalog.resolve(ref), version=version)

    def open_table(self, path: str, *, version: int | None = None) -> Table:
        """Open a table directly by storage path, bypassing the catalog."""
        _check_version(version)
        ref = parse_ref(path)
        if ref.kind is not RefKind.PATH:
            raise InvalidReferenceError(f"{path!r} is a catalog name; use .table() instead")
        return Table(self, FilesystemCatalog().resolve(ref), version=version)

    def _catalog_ref(self, name: str, what: str) -> Any:
        """Parse `name` as a catalog name, refusing a storage path.

        A path has no catalog, schema or name parts, so passing one on to the
        catalog sent ``None.None.None`` (or worse) to its API.
        """
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
        if ref.kind is RefKind.PATH:
            raise InvalidReferenceError(
                f"cannot {what} {name!r}: it is a storage path, and this needs a "
                "catalog.schema.table name"
            )
        return ref

    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]:
        return self.catalog.list_tables(catalog, schema)

    def create_table(
        self,
        name: str,
        schema: Any,
        *,
        location: str | None = None,
        partition_by: list[str] | None = None,
        mode: str = "error",
        properties: dict[str, str] | None = None,
        cluster_by: list[str] | None = None,
        comment: str | None = None,
    ) -> Table:
        """Create a table and return a handle to it.

        Three shapes:

        * a **path**: the Delta log is written there, and that is all.
        * a **catalog name with** `location`: an *external* table. The log is
          written with path credentials the catalog vends for creating tables,
          then registered, so the catalog and the log agree.
        * a **catalog name without** `location`: a *managed*, catalog-managed
          table. The catalog allocates the id and storage, version 0 is written
          there, and the catalog finalises the registration.

        The two catalog shapes need a catalog that supports table lifecycle
        (Databricks or open-source Unity Catalog).
        """
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
        schema = _schema_arg(schema)
        partition_by = _names_arg(partition_by, "partition_by")
        cluster_by = _names_arg(cluster_by, "cluster_by")
        if partition_by and cluster_by:
            raise InvalidArgumentError(
                "a table is either partitioned or liquid-clustered, not both"
            )
        _check_layout(schema, partition_by, cluster_by)
        if ref.kind is RefKind.PATH:
            if location is not None:
                # It was ignored: the table went to `name`, not `location`.
                raise InvalidArgumentError(
                    f"{name!r} is a path, so the table is created there; location= "
                    "applies only to a catalog name (an external table)"
                )
            if mode not in _CREATE_MODES:
                raise InvalidArgumentError(
                    f"create mode must be one of {sorted(_CREATE_MODES)}, not {mode!r}"
                )
            if mode in ("error", "create", "ignore") and self.table_exists(name):
                if mode == "ignore":
                    # Nothing is created, so nothing (the comment included)
                    # may be changed on the table that is already there.
                    return self.table(name)
                raise UnreachableTableError(
                    f"create {name}",
                    "a Delta table already exists there",
                    "pass mode='ignore' to keep it or mode='overwrite' to replace it",
                )
            return self._create_at(
                FilesystemCatalog().resolve(ref),
                schema,
                partition_by=partition_by,
                cluster_by=cluster_by,
                # Existence was settled above; the kernel knows no "ignore"
                # and delta-rs no "create".
                mode="error" if mode in ("create", "ignore") else mode,
                properties=properties,
                comment=comment,
            )

        lifecycle = self._lifecycle_catalog(f"create the catalog table {ref}")
        if mode == "ignore":
            # CREATE TABLE IF NOT EXISTS: it was refused as a "replace".
            if self.table_exists(name):
                return self.table(name)
            mode = "error"
        if mode not in ("error", "create"):
            raise UnreachableTableError(
                f"create {ref} with mode={mode!r}",
                "a catalog table is created once; replacing one is a different operation",
                "drop_table() first, or write to it with mode='append' / 'overwrite'",
            )
        if location is not None:
            return self._create_external(
                lifecycle, ref, location, schema, partition_by, cluster_by, properties, comment
            )
        return self._create_managed(
            lifecycle, ref, schema, partition_by, cluster_by, properties, comment
        )

    def _lifecycle_catalog(self, what: str) -> Any:
        from .catalog.base import TableLifecycleCatalog

        if not isinstance(self.catalog, TableLifecycleCatalog):
            raise UnreachableTableError(
                what,
                f"the {getattr(self.catalog, 'name', 'current')} catalog cannot register "
                "tables, and writing a Delta log alone would leave it orphaned while this "
                "call appeared to succeed",
                "create the table at a path with create_table('<path>', ...), then register "
                "it with the catalog's own tools",
            )
        return self.catalog

    def _create_at(
        self,
        resolved: ResolvedTable,
        schema: Any,
        *,
        partition_by: list[str] | None,
        cluster_by: list[str] | None,
        mode: str,
        properties: dict[str, str] | None,
        comment: str | None = None,
    ) -> Table:
        # Engines satisfy a structural protocol, so the router returns `object`.
        # Passing the properties lets a create delta-rs would reject fall
        # through to the kernel, which accepts most of the Delta spec.
        engine: Any = self.router.engine_for(
            Operation.CREATE, resolved, properties=properties, cluster_by=cluster_by, mode=mode
        )
        # delta-rs writes the comment into version 0; the kernel create takes
        # none, so there it is set by a follow-up commit rather than dropped.
        in_create = comment is not None and isinstance(engine, DeltaRsEngine)
        engine.create(
            resolved,
            schema,
            partition_by=partition_by,
            cluster_by=cluster_by,
            mode=mode,
            properties=properties,
            **({"description": comment} if in_create else {}),
        )
        table = Table(self, resolved)
        if comment is not None and not in_create:
            table.set_comment(comment)
        return table

    def _create_external(
        self,
        catalog: Any,
        ref: Any,
        location: str,
        schema: Any,
        partition_by: list[str] | None,
        cluster_by: list[str] | None,
        properties: dict[str, str] | None,
        comment: str | None,
    ) -> Table:
        import json

        from .credentials.base import StaticCredentialProvider

        credentials = catalog.path_credentials(location, "PATH_CREATE_TABLE")
        staged = ResolvedTable(
            ref=parse_ref(location),
            location=location,
            credential_provider=StaticCredentialProvider(credentials),
        )
        self._create_at(
            staged,
            schema,
            partition_by=partition_by,
            cluster_by=cluster_by,
            mode="error",
            properties=properties,
        )
        kernel: Any = self.router.engines[EngineKind.KERNEL]
        try:
            snapshot = kernel.snapshot(staged)
            metadata = json.loads(snapshot.metadata_json())
            resolved = catalog.register_table(
                ref,
                location,
                columns_schema_json=metadata["schemaString"],
                partition_columns=list(metadata.get("partitionColumns") or []),
                properties=dict(metadata.get("configuration") or {}),
                comment=comment,
            )
        except DeltaSwampError as exc:
            # The log is written by now, so repeating the create fails on it
            # ("already exists"); registering what is there is the way on.
            raise UnreachableTableError(
                f"register {ref}",
                f"the Delta log was written at {location}, but registering it failed: {exc}",
                f"conn.register_table({str(ref)!r}, {location!r}) once the cause is fixed",
            ) from exc
        return Table(self, resolved)

    def _create_managed(
        self,
        catalog: Any,
        ref: Any,
        schema: Any,
        partition_by: list[str] | None,
        cluster_by: list[str] | None,
        properties: dict[str, str] | None,
        comment: str | None,
    ) -> Table:
        """The catalog's staging-table flow: allocate, write version 0, finalise.

        If finalising fails the allocated location is left behind -- the
        catalog has no endpoint to release it -- and the error says where.
        """
        import json

        from . import __version__, _native
        from .engine.metadata import arrow_to_delta_schema, initial_actions

        staging = catalog.create_staging_table(ref)
        configuration: dict[str, str] = {}
        # What the kernel's own UC create flow writes, then what this catalog
        # says it requires, then what the caller asked for.
        configuration.update(_native.uc_required_properties(staging.table_id))
        configuration["delta.checkpointPolicy"] = "v2"
        for key, value in staging.required_properties.items():
            if value is None:
                value = (properties or {}).get(key) or staging.suggested_properties.get(key)
                if value is None:
                    raise UnreachableTableError(
                        f"create the managed table {ref}",
                        f"the catalog requires the property {key} and suggests no value",
                        f"pass properties={{{key!r}: ...}}",
                    )
            configuration[key] = value
        for key, value in (properties or {}).items():
            if key in staging.required_properties and staging.required_properties[key] not in (
                None,
                value,
            ):
                raise UnreachableTableError(
                    f"create the managed table {ref}",
                    f"the catalog requires {key}={staging.required_properties[key]!r}",
                )
            configuration[key] = value

        pa_schema = schema
        if not hasattr(pa_schema, "names"):
            pa = _require("pyarrow", "pyarrow")
            pa_schema = pa.schema(schema)
        actions = initial_actions(
            table_id=staging.table_id,
            schema=arrow_to_delta_schema(pa_schema),
            required_protocol=staging.required_protocol,
            configuration=configuration,
            partition_columns=partition_by,
            cluster_by=cluster_by,
            description=comment,
            engine_info=f"deltaswamp/{__version__}",
        )
        options = {**self.storage_options, **staging.storage_options}
        _native.commit_raw(staging.location, 0, actions, options=options or None)
        try:
            body = json.loads(
                _native.uc_create_table_request(staging.location, ref.table, options=options)
            )
            if comment:
                body["comment"] = comment
            resolved = catalog.finalize_managed_table(ref, body)
        except Exception as exc:
            if _registered_anyway(catalog, ref, exc):
                # The catalog took the registration and only the read-back
                # failed. Calling that "not accepted" sent users to retry,
                # which then failed with "already exists".
                raise UnreachableTableError(
                    f"open the managed table {ref} after creating it",
                    f"the catalog registered it (version 0 is at {staging.location}), but "
                    f"reading it back failed: {exc}",
                    f"do not repeat the create; open it with conn.table({str(ref)!r})",
                ) from exc
            raise UnreachableTableError(
                f"finalise the managed table {ref}",
                f"version 0 was written at {staging.location}, but the catalog did not "
                f"accept the registration: {exc}",
                "the staging location is not reclaimed automatically; retry the create, "
                "which allocates a fresh one",
            ) from exc
        return Table(self, resolved)

    def register_table(self, name: str, location: str, *, comment: str | None = None) -> Table:
        """Register an existing Delta table's location under a catalog name.

        The schema, partitioning and properties are taken from the log, so the
        catalog entry matches the table it points at.
        """
        import json

        ref = self._catalog_ref(name, "register")
        catalog = self._lifecycle_catalog(f"register {ref}")
        from .credentials.base import StaticCredentialProvider

        staged = ResolvedTable(
            ref=parse_ref(location),
            location=location,
            credential_provider=StaticCredentialProvider(
                catalog.path_credentials(location, "PATH_READ")
            ),
        )
        kernel: Any = self.router.engines[EngineKind.KERNEL]
        metadata = json.loads(kernel.snapshot(staged).metadata_json())
        resolved = catalog.register_table(
            ref,
            location,
            columns_schema_json=metadata["schemaString"],
            partition_columns=list(metadata.get("partitionColumns") or []),
            properties=dict(metadata.get("configuration") or {}),
            comment=comment,
        )
        return Table(self, resolved)

    def list_catalogs(self) -> list[str]:
        """Catalog names this connection can see."""
        names: list[str] = self.catalog.list_catalogs()
        return names

    def list_schemas(self, catalog: str | None = None) -> list[str]:
        """Schema names in `catalog`, defaulting to the connection's own."""
        target = catalog or self.default_catalog
        if target is None:
            raise InvalidReferenceError(
                "no catalog given and the connection has no default_catalog"
            )
        names: list[str] = self.catalog.list_schemas(target)
        return names

    def drop_table(self, name: str) -> None:
        """Remove a table from the catalog.

        For an EXTERNAL table the files remain; only the registration goes.
        """
        ref = self._catalog_ref(name, "drop")
        _call(f"drop {ref}", self.catalog.drop_table, ref)

    def table_exists(self, name: str) -> bool:
        """Whether a table is already there.

        For a path this reads the storage; for a catalog name it asks the
        catalog, so a name that resolves but has no Delta log counts as absent.
        """
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
        exists = getattr(self.catalog, "table_exists", None)
        if ref.kind is RefKind.CATALOG and callable(exists):
            try:
                return bool(exists(ref))
            except NotImplementedError:
                pass
        try:
            table = self.table(name)
        except DeltaSwampError:
            return False
        if table.location is None:
            return False
        # Open the log through the engines, which use the table's own vended
        # credentials. `DeltaTable.is_deltatable` with the connection's static
        # options cannot read a catalog table's storage, reported it absent,
        # and so sent write_table() off to create a table that was there.
        resolved = table._enrich()
        return resolved.open_error is None and resolved.min_reader_version is not None

    def write_table(
        self,
        name: str,
        data: Any,
        *,
        mode: str = "error",
        schema: Any = None,
        location: str | None = None,
        partition_by: list[str] | None = None,
        cluster_by: list[str] | None = None,
        properties: dict[str, str] | None = None,
        **write_options: Any,
    ) -> Table:
        """Write data, creating the table if it is not there yet.

        The modes follow Spark's save modes:

        ``error``
            refuse if the table already exists (the default)
        ``ignore``
            do nothing if it already exists
        ``append``
            create if absent, then append
        ``overwrite``
            create if absent, then replace the contents

        `schema` is only needed when creating; it is inferred from `data`
        otherwise.
        """
        if mode not in ("error", "ignore", "append", "overwrite"):
            raise UnreachableTableError(
                f"write with mode={mode!r}",
                "the modes are 'error', 'ignore', 'append' and 'overwrite'",
            )

        data = _write_data(data)
        exists = self.table_exists(name)
        if exists and mode == "error":
            raise UnreachableTableError(
                f"create {name}",
                "the table already exists",
                "pass mode='append', 'overwrite' or 'ignore'",
            )
        if exists and mode == "ignore":
            return self.table(name)

        if not exists and schema is not None:
            # Checked before creating: a mismatch used to fail the append
            # after an empty table with the wrong schema had been committed,
            # and the retry then refused because the table "already exists".
            wanted = {n.lower() for n in _schema_names(_schema_arg(schema)) or ()}
            given = _schema_names(_schema_of(data)) or []
            extra = [n for n in given if n.lower() not in wanted]
            if wanted and extra:
                raise InvalidArgumentError(
                    f"the data has columns {extra} that schema= does not; nothing was created"
                )
        if not exists:
            log_dir = _local_log_dir(name)
            log_existed = log_dir is not None and os.path.exists(log_dir)
            table = self.create_table(
                name,
                schema if schema is not None else _schema_of(data),
                location=location,
                partition_by=partition_by,
                cluster_by=cluster_by,
                properties=properties,
            )
            try:
                table.append(data, **write_options)
            except BaseException as exc:
                # Creating and writing are two commits. A write that failed
                # left an empty table behind, and the retry was then refused
                # because it "already exists". A local log this call created
                # is removed; elsewhere, say what is left.
                if log_dir is not None and not log_existed and os.path.isdir(log_dir):
                    import shutil

                    shutil.rmtree(log_dir, ignore_errors=True)
                elif isinstance(exc, Exception):
                    exc.add_note(
                        f"deltaswamp: the table {name} was created, but this first write "
                        "failed; it exists now and is empty"
                    )
                raise
            return table

        table = self.table(name)
        # Layout and properties apply when creating; on an existing table
        # they were silently dropped, so the caller believed them applied.
        wanted_parts = _names_arg(partition_by, "partition_by")
        current = table._enrich()
        if wanted_parts is not None and list(current.partition_columns) != wanted_parts:
            raise InvalidArgumentError(
                f"{name} is partitioned by {list(current.partition_columns)}, not "
                f"{wanted_parts}; partition_by= only applies when the table is created"
            )
        changed = {
            k: v for k, v in (properties or {}).items() if current.properties.get(k) != str(v)
        }
        if changed:
            raise InvalidArgumentError(
                f"{name} already exists, so properties= would not be applied: "
                f"{sorted(changed)}; call set_properties() on the table instead"
            )
        if mode == "overwrite":
            table.overwrite(data, **write_options)
        else:
            table.append(data, **write_options)
        return table

    def convert_to_delta(self, location: str, **kwargs: Any) -> Table:
        """Convert a directory of Parquet into a Delta table in place."""
        ref = parse_ref(location)
        resolved = FilesystemCatalog().resolve(ref)
        engine: Any = self.router.engine_for(Operation.CONVERT, resolved)
        engine.convert(location, **kwargs)
        return Table(self, resolved)

    def sql(
        self,
        query: str,
        *,
        tables: dict[str, Any] | None = None,
        engine: str = "duckdb",
    ) -> Any:
        """Run SQL over tables opened through this connection.

        `tables` maps the names used in `query` to table names, paths or
        `Table` objects; each is read through its own engine, so a query can
        join a catalog-managed table with a Glue table and a path. Returns a
        pyarrow Table.

        ``engine="duckdb"`` (the default) or ``"polars"`` run locally.
        ``engine="warehouse"`` sends `query` verbatim to the SQL fallback, where
        names resolve in Unity Catalog and `tables` is not used.
        """
        if engine == "warehouse":
            sql_engine: Any = self.router.engines.get(EngineKind.SQL)
            if sql_engine is None or not self.router.allow_sql_fallback:
                raise UnreachableTableError(
                    "run SQL on a warehouse",
                    "the SQL warehouse fallback is not enabled on this connection",
                    "ds.connect(..., allow_sql_fallback=True)",
                )
            return sql_engine.query(query)

        # Refuse an unknown engine before reading every table in `tables`.
        if engine not in ("duckdb", "polars"):
            raise UnreachableTableError(
                f"run SQL with engine={engine!r}",
                "the engines are 'duckdb', 'polars' and 'warehouse'",
            )
        pa = _require("pyarrow", "pyarrow")
        module = _require(engine, engine)
        frames = {
            alias: pa.table((ref if isinstance(ref, Table) else self.table(ref)).scan())
            for alias, ref in (tables or {}).items()
        }
        if engine == "duckdb":
            con = module.connect()
            try:
                for alias, frame in frames.items():
                    con.register(alias, frame)
                relation = con.sql(query)
                # A statement (CREATE, SET, ...) yields no relation to convert.
                return relation.to_arrow_table() if relation is not None else None
            finally:
                # An in-memory database per call; leaving it open leaked it
                # and every registered frame until garbage collection.
                con.close()
        ctx = module.SQLContext({alias: module.from_arrow(f) for alias, f in frames.items()})
        return ctx.execute(query, eager=True).to_arrow()

    # ------------------------------------------------------------- namespaces

    def _namespaces(self, what: str) -> Any:
        return _governed(self.catalog, "NamespaceCatalog", what)

    def create_catalog(
        self, name: str, *, comment: str | None = None, storage_root: str | None = None
    ) -> None:
        cat = self._namespaces(f"create catalog {name}")
        _call(f"create catalog {name}", cat.create_catalog, name, comment, storage_root)

    def drop_catalog(self, name: str, *, force: bool = False) -> None:
        cat = self._namespaces(f"drop catalog {name}")
        _call(f"drop catalog {name}", cat.drop_catalog, name, force)

    def create_schema(
        self,
        name: str,
        *,
        comment: str | None = None,
        storage_root: str | None = None,
    ) -> None:
        """Create `catalog.schema` (or `schema` under the default catalog)."""
        catalog, schema = self._schema_parts(name)
        cat = self._namespaces(f"create schema {name}")
        _call(f"create schema {name}", cat.create_schema, catalog, schema, comment, storage_root)

    def drop_schema(self, name: str, *, force: bool = False) -> None:
        catalog, schema = self._schema_parts(name)
        cat = self._namespaces(f"drop schema {name}")
        _call(f"drop schema {name}", cat.drop_schema, catalog, schema, force)

    def _schema_parts(self, name: str) -> tuple[str, str]:
        from .identity import split_identifier

        parts = split_identifier(name)
        if len(parts) == 2:
            return parts[0], parts[1]
        if len(parts) == 1 and self.default_catalog:
            return self.default_catalog, parts[0]
        raise InvalidReferenceError(f"{name!r} is not a catalog.schema name")

    def search_tables(
        self,
        catalog: str | None = None,
        *,
        schema_pattern: str | None = None,
        table_pattern: str | None = None,
    ) -> list[Any]:
        """Table summaries matching SQL LIKE patterns, in one listing call."""
        target = catalog or self.default_catalog
        if target is None:
            raise InvalidReferenceError("no catalog given and no default_catalog")
        cat = self._namespaces("search tables")
        return list(
            _call("search tables", cat.search_tables, target, schema_pattern, table_pattern)
        )

    def list_functions(self, schema: str) -> list[Any]:
        catalog, name = self._schema_parts(schema)
        cat = self._namespaces("list functions")
        return list(_call("list functions", cat.list_functions, catalog, name))

    def list_volumes(self, schema: str) -> list[Any]:
        catalog, name = self._schema_parts(schema)
        cat = self._namespaces("list volumes")
        return list(_call("list volumes", cat.list_volumes, catalog, name))

    def create_volume(
        self,
        name: str,
        *,
        volume_type: str = "MANAGED",
        storage_location: str | None = None,
        comment: str | None = None,
    ) -> Any:
        ref = self._catalog_ref(name, "create the volume")
        cat = self._namespaces(f"create volume {name}")
        return _call(
            f"create volume {name}",
            cat.create_volume,
            ref.catalog,
            ref.schema,
            ref.table,
            volume_type,
            storage_location,
            comment,
        )

    def drop_volume(self, name: str) -> None:
        ref = self._catalog_ref(name, "drop the volume")
        cat = self._namespaces(f"drop volume {name}")
        _call(f"drop volume {name}", cat.drop_volume, ref.catalog, ref.schema, ref.table)

    def volume(self, name: str) -> Any:
        """A Unity Catalog volume: list, read, write and delete its files."""
        ref = self._catalog_ref(name, "open the volume")
        cat = self._namespaces(f"open volume {name}")
        return _call(f"open volume {name}", cat.volume, ref)

    def grants(
        self, securable: str, *, securable_type: str = "SCHEMA", principal: str | None = None
    ) -> list[Any]:
        """Grants on a catalog, schema, volume or function. Tables: `Table.grants()`."""
        cat = _governed(self.catalog, "GovernedCatalog", f"read grants on {securable}")
        return list(
            _call(
                "read grants",
                cat.grants,
                securable,
                principal,
                securable_type=securable_type,
            )
        )

    def grant(
        self,
        securable: str,
        principal: str,
        privileges: list[str],
        *,
        securable_type: str = "SCHEMA",
    ) -> list[Any]:
        cat = _governed(self.catalog, "GovernedCatalog", f"grant on {securable}")
        return list(
            _call(
                "grant",
                cat.grant,
                securable,
                principal,
                privileges,
                securable_type=securable_type,
            )
        )

    def revoke(
        self,
        securable: str,
        principal: str,
        privileges: list[str],
        *,
        securable_type: str = "SCHEMA",
    ) -> list[Any]:
        cat = _governed(self.catalog, "GovernedCatalog", f"revoke on {securable}")
        return list(
            _call(
                "revoke",
                cat.revoke,
                securable,
                principal,
                privileges,
                securable_type=securable_type,
            )
        )

    def undrop_table(self, name: str) -> None:
        """UNDROP TABLE: restore a recently dropped managed table (SQL fallback)."""
        ref = self._catalog_ref(name, "undrop")
        sql_engine: Any = self.router.engines.get(EngineKind.SQL)
        if sql_engine is None or not self.router.allow_sql_fallback:
            raise FallbackRequiredError(
                f"undrop {ref}",
                "UNDROP is Databricks-only and runs on a SQL warehouse",
                "ds.connect(..., allow_sql_fallback=True)",
            )
        sql_engine.undrop(ref.full_name)

    def _reresolve(self, table: Table) -> Table:
        """A fresh handle on the same table: the latest version, and for a
        catalog-managed table a fresh commit tail from the catalog."""
        ref = table.resolved.ref
        catalog = FilesystemCatalog() if ref.kind is RefKind.PATH else self.catalog
        return Table(self, catalog.resolve(ref))

    def preflight(self) -> list[str]:
        """Check workspace prerequisites. Empty list means ready.

        Worth running once before anything else: the two settings it checks are
        off by default, grantable only by someone else, and account for most
        first-contact failures.
        """
        check = getattr(self.catalog, "preflight", None)
        return list(check()) if callable(check) else []


class Table:
    """One table. Reads, writes, and an honest account of what it cannot do."""

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
        return f"Table({self._resolved.ref}, location={self._resolved.location!r})"

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

    def _check_identity(self, metadata_id: str | None) -> None:
        """Refuse a table whose log identity does not match the catalog's.

        A table dropped and re-created under the same name keeps the name and
        gets a new id. Reading on with a cached id means reading a different
        table while believing it is the same one.
        """
        expected = self._resolved.table_uuid
        # Only a managed table's log carries the catalog's id; the catalog gives
        # a registered external table an id of its own.
        if self._resolved.table_type not in (TableType.MANAGED, None):
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
        """
        self._enriched = False

    def _enrich(self) -> ResolvedTable:
        """Fill in the protocol feature lists by reading the log once.

        Deliberately does not go through the router: routing depends on these
        features, so asking the router first would be circular. We try kernel,
        then delta-rs, and fall back to catalog metadata alone if neither can
        open the table -- in which case routing proceeds on partial information
        and the engines themselves produce the refusal.
        """
        if self._enriched or self._resolved.location is None:
            return self._resolved

        last_error: str | None = None
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
            self._check_identity(detail.get("metadata_id"))
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
            )
            self._enriched = True
            return self._resolved

        if last_error is not None and self._version is not None:
            self._check_pinned_version_exists()
        # A failure is not cached: a transient vending or network error used to
        # leave this handle refusing every operation for the rest of its life.
        if last_error is not None:
            self._resolved = dataclasses.replace(self._resolved, open_error=last_error)
        return self._resolved

    def _check_pinned_version_exists(self) -> None:
        """Say so plainly when the handle's version is past the latest one.

        Opening at a version that does not exist left every call (history
        included) refusing with "the Delta log could not be read" and advice
        to enable the SQL warehouse.
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
            return

    # ------------------------------------------------------------ capabilities

    def capabilities(self) -> dict[Operation, Capability]:
        """What can and cannot be done with this table, and why.

        The honesty surface. Every refusal names the blocker and, where one
        exists, the remedy.
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
        return self._connection.router.capability(op, self._enrich(), **shape)

    def _engine(
        self,
        operation: Operation,
        needs: frozenset[str] = frozenset(),
        **shape: Any,
    ) -> Any:
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
        engine = self._engine(op, frozenset(needs))
        return engine.scan(
            self._resolved,
            columns=columns,
            predicate=predicate,
            version=version,
            timestamp=timestamp,
            limit=limit,
        )

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

    def to_arrow(self, **kwargs: Any) -> Any:
        pa = _require("pyarrow", "pyarrow")
        limit = kwargs.pop("limit", None)
        if limit is not None:
            # The scan treats `limit` as a hint that streaming engines ignore,
            # so to_arrow(limit=1) returned the whole table. Stop at it here.
            return self.head(limit, **kwargs)
        return pa.table(self.scan(**kwargs))

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
        con = connection if connection is not None else duckdb.connect()
        relation = con.from_arrow(data)
        if name is not None:
            relation.create_view(name, replace=True)
        return relation

    def plan_scan(
        self,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: Any = None,
    ) -> Any:
        """Plan a distributed read: a picklable `ScanPlan` of per-file splits.

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
        return ScanPlan(
            engine=engine,
            table=self._resolved,
            splits=tuple(splits),
            columns=tuple(columns) if columns is not None else None,
            predicate=predicate,
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
        are in hand, so this costs one batch on a table of any size. It used to
        materialise the whole table and slice it, which on a multi-terabyte
        table never returned.

        The stream is the engine's output, so deletion vectors, column mapping
        and partition values are already applied; stopping early here is not the
        limit pushdown the scan layer deliberately refuses, which would break
        the positional mapping a deletion vector depends on.
        """
        pa = _require("pyarrow", "pyarrow")
        # A negative n used to return an empty table silently.
        _check_count(n, "n")
        # The limit is a hint to the engine as well as a client-side stop: the
        # warehouse would otherwise compute the entire result set first.
        kwargs["limit"] = n
        reader = pa.RecordBatchReader.from_stream(self.scan(**kwargs))
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

        Streams the narrowest column rather than materialising the table; the
        engines' statistics-based counts are approximate by their own
        documentation (a file without stats counts as zero rows), so they are
        not used here.
        """
        pa = _require("pyarrow", "pyarrow")
        schema = self.schema()
        names = list(getattr(schema, "names", None) or [f.name for f in schema])
        narrow = [names[0]] if names and predicate is None else None
        total = 0
        for batch in pa.RecordBatchReader.from_stream(
            self.scan(columns=narrow, predicate=predicate)
        ):
            total += batch.num_rows
        return total

    def files(self) -> Any:
        """The table's live data files: path, size, partition values and statistics."""
        return self._engine(Operation.FILES).files(self._resolved, version=self._version)

    def history(self, limit: int | None = None) -> list[dict[str, Any]]:
        _check_count(limit, "limit")
        if limit == 0:
            # The warehouse treats a falsy limit as none and returned all of it.
            return []
        result: list[dict[str, Any]] = self._engine(Operation.HISTORY).history(
            self._resolved, limit=limit
        )
        return result

    def detail(self) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.DETAIL).detail(
            self._resolved, version=self._version
        )
        return result

    def cdf(self, **kwargs: Any) -> Any:
        """The change data feed, by version or timestamp range.

        Rows carry `_change_type`, `_commit_version` and `_commit_timestamp`.
        """
        if "columns" in kwargs:
            kwargs["columns"] = _columns_arg(kwargs["columns"])
        _check_predicate(kwargs.get("predicate"), "read the change data feed")
        start, end = kwargs.get("starting_version"), kwargs.get("ending_version")
        _check_version(start, "starting_version")
        _check_version(end, "ending_version")
        if start is not None and end is not None and end < start:
            raise InvalidArgumentError(f"ending_version {end} is before starting_version {start}")
        for key in ("starting_timestamp", "ending_timestamp"):
            if key in kwargs:
                kwargs[key] = _timestamp_arg(kwargs[key], key)
        return self._engine(Operation.CDF).cdf(self._resolved, **_given(kwargs))

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

    def protocol(self) -> tuple[int | None, int | None]:
        r = self._enrich()
        return (r.min_reader_version, r.min_writer_version)

    def features(self) -> frozenset[str]:
        return self._enrich().features

    def properties(self) -> dict[str, str]:
        return dict(self._enrich().properties)

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
        data = self._align(data, schema_mode)
        needs = self._write_needs(schema_mode, commit_metadata, txn, writer_properties, "static")
        op = Operation.MERGE_SCHEMA if schema_mode == "merge" else Operation.APPEND
        self._engine(op, needs).append(
            self._resolved,
            data,
            schema_mode=schema_mode,
            partition_by=partition_by,
            target_file_size=target_file_size,
            writer_properties=writer_properties,
            commit_metadata=commit_metadata,
            txn=txn,
            max_commit_retries=max_commit_retries,
        )
        self._invalidate()

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
        data = self._align(data, schema_mode)
        needs = self._write_needs(
            schema_mode, commit_metadata, txn, writer_properties, partition_overwrite
        )
        op = (
            Operation.REPLACE_WHERE
            if (predicate is not None or partition_overwrite == "dynamic")
            else Operation.OVERWRITE
        )
        self._engine(op, needs).overwrite(
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
        )
        self._invalidate()

    def replace(self, data: Any, **kwargs: Any) -> None:
        """Replace the table's contents and schema. REPLACE TABLE / RTAS."""
        self.overwrite(data, schema_mode="overwrite", **kwargs)

    def _already_committed(self, txn: tuple[str, int]) -> bool:
        """True if `txn` was already committed, so the write should be skipped.

        Neither engine deduplicates on its own -- verified against delta-rs
        1.6.5, which records the txn action and appends anyway -- so the guard
        lives here.
        """
        app_id, version = txn
        # A missing capability (no engine can read transaction ids) raises
        # out of txn_version: treating it as "never committed" silently
        # dropped the dedup and let a replayed batch append twice.
        last = self.txn_version(app_id)
        return last is not None and version <= last

    def txn_version(self, app_id: str) -> int | None:
        """The last version committed under `app_id`, or None if never.

        Pair with `txn=` on a write to make a pipeline exactly-once::

            if t.txn_version("nightly-load") != batch_id:
                t.append(data, txn=("nightly-load", batch_id))
        """
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
        _check_predicate(predicate, "delete")
        result: dict[str, Any] = self._engine(Operation.DELETE).delete(
            self._resolved, predicate, **_given(kwargs)
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
        _check_predicate(predicate, "update")
        if updates is not None and new_values is not None:
            raise InvalidArgumentError("pass updates (SQL expressions) or new_values, not both")
        if not updates and not new_values:
            raise InvalidArgumentError("update needs at least one column to set")
        engine = self._engine(Operation.UPDATE)
        # delta-rs skips a SET target it cannot find -- an unknown name, a
        # different case, a nested field -- and still rewrites every matched
        # file, reporting the rows as updated while changing nothing.
        strict = isinstance(engine, DeltaRsEngine)
        if updates is not None:
            updates = self._update_targets(updates, strict)
        if new_values is not None:
            kwargs["new_values"] = self._update_targets(new_values, strict)
        result: dict[str, Any] = engine.update(
            self._resolved, updates=updates, predicate=predicate, **_given(kwargs)
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
        builder = self._engine(Operation.MERGE).merge(self._resolved, source, predicate, **kwargs)
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
        result: dict[str, Any] = self._engine(Operation.DROP_COLUMN).drop_column(
            self._resolved, column
        )
        self._invalidate()
        return result

    def rename_column(self, old: str, new: str) -> dict[str, Any]:
        self._check_writable("rename a column")
        result: dict[str, Any] = self._engine(Operation.RENAME_COLUMN).rename_column(
            self._resolved, old, new
        )
        self._invalidate()
        return result

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
        self._engine(Operation.ADD_FEATURE).add_feature(self._resolved, feature, **kwargs)
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
                "ds.connect(..., allow_sql_fallback=True)",
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
