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
from typing import Any

from .capability import Capability, Operation
from .capability import Engine as EngineKind
from .catalog import Catalog, ResolvedTable
from .catalog.filesystem import FilesystemCatalog
from .catalog.registry import catalog_for_uri
from .credentials import Operation as CredentialOperation
from .engine.deltars import DeltaRsEngine
from .engine.kernel import KernelEngine
from .errors import (
    CorruptTableError,
    DeltaSwampError,
    InvalidReferenceError,
    UnreachableTableError,
)
from .identity import RefKind, parse_ref
from .router import Router

__all__ = ["Connection", "Table", "connect"]


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
    operations no open-source engine implements. It is off by default because
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
    if allow_sql_fallback:
        from .engine.sql import SqlEngine

        engines[EngineKind.SQL] = SqlEngine(
            profile=profile,
            host=host,
            token=token,
            config=config,
            warehouse_id=warehouse_id,
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
    schema = getattr(data, "schema", None)
    if schema is not None:
        return schema
    pa = _require("pyarrow", "pyarrow")
    return pa.table(data).schema


def _require(module: str, extra: str) -> Any:
    """Import an optional dependency, or say which extra provides it.

    Without this the failure surfaces as a ModuleNotFoundError raised from deep
    inside pyarrow, which does not tell you what to install.
    """
    import importlib

    try:
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
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
        catalog = FilesystemCatalog() if ref.kind is RefKind.PATH else self.catalog
        return Table(self, catalog.resolve(ref), version=version)

    def open_table(self, path: str, *, version: int | None = None) -> Table:
        """Open a table directly by storage path, bypassing the catalog."""
        ref = parse_ref(path)
        if ref.kind is not RefKind.PATH:
            raise InvalidReferenceError(f"{path!r} is a catalog name; use .table() instead")
        return Table(self, FilesystemCatalog().resolve(ref), version=version)

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
    ) -> Table:
        """Create a table and return a handle to it.

        For a path reference the location is the path itself. For a catalog
        name, `location` is required: creating a *managed* table means asking
        the catalog to allocate storage and then finalising through its
        create-table API, which is a different flow and not yet implemented
        here -- so we say that rather than silently creating something in the
        wrong place.
        """
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
        if ref.kind is RefKind.PATH:
            resolved = FilesystemCatalog().resolve(ref)
        elif location is not None:
            # Writing a Delta log at `location` does NOT create a catalog table.
            # Returning a Table named `catalog.schema.name` here would say it
            # did, leaving an orphaned log that nothing in Unity Catalog knows
            # about. Registration needs PATH_CREATE_TABLE path credentials and a
            # tables.create call, which is not built yet, so refuse instead of
            # half-creating.
            raise UnreachableTableError(
                f"create the catalog table {ref}",
                "a Delta log at a path is not a registered catalog table, and "
                "registering one is not implemented yet. Creating the log alone would "
                "leave it orphaned while this call appeared to succeed",
                f"create the table at its path with open_table/create_table on "
                f"{location!r}, then register it from Databricks with "
                f"CREATE TABLE {ref} LOCATION {location!r}",
            )
        else:
            raise UnreachableTableError(
                "create a managed table",
                "creating a catalog-managed table requires the catalog to allocate "
                "storage via its staging-table API and to finalise the table afterwards; "
                "that flow is not implemented yet",
                "pass location= to create an external table, or CREATE TABLE from "
                "Databricks and then open it here",
            )

        table = Table(self, resolved)
        # Engines satisfy a structural protocol, so the router returns `object`.
        # Passing the properties lets a create delta-rs would reject fall
        # through to the kernel, which accepts most of the Delta spec.
        engine: Any = self.router.engine_for(
            Operation.CREATE, resolved, properties=properties, cluster_by=cluster_by
        )
        engine.create(
            resolved,
            schema,
            partition_by=partition_by,
            cluster_by=cluster_by,
            mode=mode,
            properties=properties,
        )
        return table

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
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
        self.catalog.drop_table(ref)

    def table_exists(self, name: str) -> bool:
        """Whether a table is already there.

        For a path this reads the storage; for a catalog name it asks the
        catalog, so a name that resolves but has no Delta log counts as absent.
        """
        try:
            table = self.table(name)
        except DeltaSwampError:
            return False
        if table.location is None:
            return False
        try:
            from deltalake import DeltaTable

            return bool(
                DeltaTable.is_deltatable(
                    table.location, storage_options=self.storage_options or None
                )
            )
        except Exception:
            return False

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

        exists = self.table_exists(name)
        if exists and mode == "error":
            raise UnreachableTableError(
                f"create {name}",
                "the table already exists",
                "pass mode='append', 'overwrite' or 'ignore'",
            )
        if exists and mode == "ignore":
            return self.table(name)

        if not exists:
            table = self.create_table(
                name,
                schema if schema is not None else _schema_of(data),
                location=location,
                partition_by=partition_by,
                cluster_by=cluster_by,
                properties=properties,
            )
        else:
            table = self.table(name)

        if mode == "overwrite" and exists:
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
        self._connection = connection
        self._resolved = resolved
        self._version = version
        self._enriched = False

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

        self._enriched = True
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
            self._resolved = dataclasses.replace(
                self._resolved,
                min_reader_version=detail.get("min_reader_version"),
                min_writer_version=detail.get("min_writer_version"),
                reader_features=frozenset(detail.get("reader_features") or ()),
                writer_features=frozenset(detail.get("writer_features") or ()),
                properties={**(detail.get("properties") or {}), **self._resolved.properties},
                partition_columns=tuple(detail.get("partition_columns") or ()),
            )
            self._check_identity(detail.get("metadata_id"))
            return self._resolved

        if last_error is not None:
            self._resolved = dataclasses.replace(self._resolved, open_error=last_error)
        return self._resolved

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
        op = Operation(operation) if isinstance(operation, str) else operation
        return self._connection.router.capability(op, self._enrich(), **shape)

    def _engine(
        self,
        operation: Operation,
        needs: frozenset[str] = frozenset(),
        **shape: Any,
    ) -> Any:
        return self._connection.router.engine_for(operation, self._enrich(), needs=needs, **shape)

    # ------------------------------------------------------------------- read

    def scan(
        self,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: str | None = None,
    ) -> Any:
        """Read the table as an Arrow stream (exports `__arrow_c_stream__`)."""
        # `version or timestamp` would treat version 0 as no time travel.
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
            version=version if version is not None else self._version,
            timestamp=timestamp,
        )

    def to_arrow(self, **kwargs: Any) -> Any:
        pa = _require("pyarrow", "pyarrow")
        return pa.table(self.scan(**kwargs))

    def to_pandas(self, **kwargs: Any) -> Any:
        _require("pandas", "pandas")
        return self.to_arrow(**kwargs).to_pandas()

    def to_polars(self, **kwargs: Any) -> Any:
        pl = _require("polars", "polars")
        return pl.DataFrame(self.scan(**kwargs))

    def to_pyarrow_dataset(self, **kwargs: Any) -> Any:
        dataset = _require("pyarrow.dataset", "pyarrow")
        return dataset.dataset(self.to_arrow(**kwargs))

    def head(self, n: int = 5, **kwargs: Any) -> Any:
        return self.to_arrow(**kwargs).slice(0, n)

    def count(self) -> int:
        return int(self.to_arrow().num_rows)

    def history(self, limit: int | None = None) -> list[dict[str, Any]]:
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
        return self._engine(Operation.CDF).cdf(self._resolved, **kwargs)

    # -------------------------------------------------------------- metadata

    def schema(self) -> Any:
        engine = self._engine(Operation.SCAN)
        snapshot = getattr(engine, "snapshot", None)
        if snapshot is not None:
            return snapshot(self._resolved, version=self._version).schema()
        return self.to_arrow().schema

    def protocol(self) -> tuple[int | None, int | None]:
        r = self._enrich()
        return (r.min_reader_version, r.min_writer_version)

    def features(self) -> frozenset[str]:
        return self._enrich().features

    def properties(self) -> dict[str, str]:
        return dict(self._enrich().properties)

    @property
    def version(self) -> int:
        return int(self.detail()["version"])

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
        if txn is not None and self._already_committed(txn):
            return
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
        if txn is not None and self._already_committed(txn):
            return
        needs = self._write_needs(
            schema_mode, commit_metadata, txn, writer_properties, partition_overwrite
        )
        op = (
            Operation.REPLACE_WHERE
            if (predicate or partition_overwrite == "dynamic")
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
        try:
            last = self.txn_version(app_id)
        except Exception:
            # No transaction log to read yet, or the engine cannot report it.
            # Better to write than to silently drop data.
            return False
        return last is not None and version <= last

    def txn_version(self, app_id: str) -> int | None:
        """The last version committed under `app_id`, or None if never.

        Pair with `txn=` on a write to make a pipeline exactly-once::

            if t.txn_version("nightly-load") != batch_id:
                t.append(data, txn=("nightly-load", batch_id))
        """
        version: int | None = self._engine(Operation.APPEND).txn_version(self._resolved, app_id)
        return version

    def delete(self, predicate: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.DELETE).delete(self._resolved, predicate)
        self._invalidate()
        return result

    def update(
        self, updates: dict[str, str] | None = None, *, predicate: str | None = None
    ) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.UPDATE).update(
            self._resolved, updates=updates, predicate=predicate
        )
        self._invalidate()
        return result

    def merge(self, source: Any, predicate: str, **kwargs: Any) -> Any:
        self._invalidate()
        return self._engine(Operation.MERGE).merge(self._resolved, source, predicate, **kwargs)

    # ------------------------------------------------------------ maintenance

    def optimize(self, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.OPTIMIZE).optimize(self._resolved, **kwargs)
        self._invalidate()
        return result

    def z_order(self, columns: list[str], **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.ZORDER).zorder(
            self._resolved, columns, **kwargs
        )
        self._invalidate()
        return result

    def vacuum(self, **kwargs: Any) -> list[str]:
        result: list[str] = self._engine(Operation.VACUUM).vacuum(self._resolved, **kwargs)
        self._invalidate()
        return result

    def restore(self, target: Any, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.RESTORE).restore(
            self._resolved, target, **kwargs
        )
        self._invalidate()
        return result

    def repair(self, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.REPAIR).repair(self._resolved, **kwargs)
        self._invalidate()
        return result

    # ----------------------------------------------------------------- schema

    def add_column(self, fields: Any, **kwargs: Any) -> None:
        """Add columns. `fields` is a list of Arrow/Delta fields, or a
        {name: sql_type} mapping when the SQL fallback serves it."""
        self._engine(Operation.ADD_COLUMN).add_columns(self._resolved, fields, **kwargs)
        self._invalidate()

    def drop_column(self, column: str) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.DROP_COLUMN).drop_column(
            self._resolved, column
        )
        self._invalidate()
        return result

    def rename_column(self, old: str, new: str) -> dict[str, Any]:
        result: dict[str, Any] = self._engine(Operation.RENAME_COLUMN).rename_column(
            self._resolved, old, new
        )
        self._invalidate()
        return result

    def set_properties(self, properties: dict[str, str], **kwargs: Any) -> None:
        self._engine(Operation.SET_PROPERTIES, properties=properties).set_properties(
            self._resolved, properties, **kwargs
        )
        self._invalidate()

    def add_feature(self, feature: Any, **kwargs: Any) -> None:
        self._engine(Operation.ADD_FEATURE).add_feature(self._resolved, feature, **kwargs)
        self._invalidate()

    def drop_feature(self, feature: str, **kwargs: Any) -> dict[str, Any]:
        """Drop a table feature. Databricks-only, so it needs the SQL fallback."""
        result: dict[str, Any] = self._engine(Operation.DROP_FEATURE).drop_feature(
            self._resolved, feature, **kwargs
        )
        self._invalidate()
        return result

    def add_constraint(self, constraints: dict[str, str], **kwargs: Any) -> None:
        self._engine(Operation.ADD_CONSTRAINT).add_constraint(self._resolved, constraints, **kwargs)
        self._invalidate()

    # ------------------------------------------------------- log and layout

    def checkpoint(self) -> None:
        self._engine(Operation.CHECKPOINT).checkpoint(self._resolved)

    def compact_logs(self, start: int | None = None, end: int | None = None) -> Any:
        return self._engine(Operation.LOG_COMPACTION).compact_logs(self._resolved, start, end)

    def generate(self) -> None:
        """Write symlink manifests, for engines that read those instead of the log."""
        self._engine(Operation.GENERATE).generate(self._resolved)

    def reorg(self, **kwargs: Any) -> dict[str, Any]:
        """REORG TABLE. Databricks-only, so it needs the SQL fallback."""
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
