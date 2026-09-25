"""`connect()` and `Connection`: the entry point to every catalog."""

from __future__ import annotations

import dataclasses
from typing import Any

from .capability import Engine as EngineKind
from .capability import Operation
from .catalog import Catalog, ResolvedTable
from .catalog.filesystem import FilesystemCatalog
from .catalog.registry import catalog_for_uri
from .engine.deltars import DeltaRsEngine
from .engine.kernel import KernelEngine
from .errors import (
    SQL_FALLBACK_REMEDY,
    DeltaSwampError,
    FallbackRequiredError,
    InvalidReferenceError,
    UnreachableTableError,
)
from .identity import RefKind, parse_ref
from .router import Router
from .table import Table, _call, _governed, _require

__all__ = ["Connection", "connect"]


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
    schema = getattr(data, "schema", None)
    if schema is not None:
        return schema
    pa = _require("pyarrow", "pyarrow")
    return pa.table(data).schema


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
          there, and the catalog finalizes the registration.

        The two catalog shapes need a catalog that supports table lifecycle
        (Databricks or open-source Unity Catalog).
        """
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
        if ref.kind is RefKind.PATH:
            return self._create_at(
                FilesystemCatalog().resolve(ref),
                schema,
                partition_by=partition_by,
                cluster_by=cluster_by,
                mode=mode,
                properties=properties,
            )

        lifecycle = self._lifecycle_catalog(f"create the catalog table {ref}")
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
    ) -> Table:
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
        return Table(self, resolved)

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
        """The catalog's staging-table flow: allocate, write version 0, finalize.

        If finalizing fails the allocated location is left behind -- the
        catalog has no endpoint to release it -- and the error says where.
        """
        import json

        from . import __version__, _native
        from .engine.metadata import arrow_to_delta_schema, initial_actions

        try:
            staging = catalog.create_staging_table(ref)
        except DeltaSwampError:
            # Nothing has been written yet, so the warehouse can take over
            # cleanly -- if the caller allowed it.
            sql: Any = self.router.engines.get(EngineKind.SQL)
            if not self.router.allow_sql_fallback or sql is None:
                raise
            sql.create_managed(
                ref.full_name,
                schema,
                partition_by=partition_by,
                cluster_by=cluster_by,
                properties=properties,
                comment=comment,
            )
            return self.table(ref.full_name)
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
            raise UnreachableTableError(
                f"finalize the managed table {ref}",
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

        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
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
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
        self.catalog.drop_table(ref)

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
        from deltalake import DeltaTable

        return bool(
            DeltaTable.is_deltatable(table.location, storage_options=self.storage_options or None)
        )

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
                    SQL_FALLBACK_REMEDY,
                )
            return sql_engine.query(query)

        pa = _require("pyarrow", "pyarrow")
        frames = {
            alias: pa.table((ref if isinstance(ref, Table) else self.table(ref)).scan())
            for alias, ref in (tables or {}).items()
        }
        if engine == "duckdb":
            duckdb = _require("duckdb", "duckdb")
            con = duckdb.connect()
            for alias, frame in frames.items():
                con.register(alias, frame)
            return con.sql(query).to_arrow_table()
        if engine == "polars":
            pl = _require("polars", "polars")
            ctx = pl.SQLContext({alias: pl.from_arrow(f) for alias, f in frames.items()})
            return ctx.execute(query, eager=True).to_arrow()
        raise UnreachableTableError(
            f"run SQL with engine={engine!r}",
            "the engines are 'duckdb', 'polars' and 'warehouse'",
        )

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
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
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
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
        cat = self._namespaces(f"drop volume {name}")
        _call(f"drop volume {name}", cat.drop_volume, ref.catalog, ref.schema, ref.table)

    def volume(self, name: str) -> Any:
        """A Unity Catalog volume: list, read, write and delete its files."""
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
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
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
        sql_engine: Any = self.router.engines.get(EngineKind.SQL)
        if sql_engine is None or not self.router.allow_sql_fallback:
            raise FallbackRequiredError(
                f"undrop {ref}",
                "UNDROP is Databricks-only and runs on a SQL warehouse",
                SQL_FALLBACK_REMEDY,
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
