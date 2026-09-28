"""`connect()` and `Connection`: the entry point to every catalog."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import re
from collections.abc import Iterator, Mapping
from typing import Any

from ._util import check_keywords
from .capability import Engine as EngineKind
from .capability import Operation
from .catalog import Catalog, ResolvedTable
from .catalog.filesystem import FilesystemCatalog
from .catalog.registry import catalog_for_uri
from .engine.deltars import DeltaRsEngine
from .engine.kernel import KernelEngine
from .engine.metadata import cdf_clash_error, cdf_name_clash
from .errors import (
    SQL_FALLBACK_REMEDY,
    DeltaSwampError,
    EngineError,
    FallbackRequiredError,
    InvalidArgumentError,
    InvalidReferenceError,
    TableNotFoundError,
    UnreachableTableError,
)
from .identity import RefKind, TableRef, parse_ref
from .properties import with_checkpoint_stats
from .router import Router
from .table import Table, _call, _check_version, _governed, _require, _write_data

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
    iceberg_properties: dict[str, str] | None = None,
    ship_credentials: bool = False,
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

    `storage_options` reach every engine: the kernel and delta-rs take them as
    object_store options (see docs/storage.md for how they merge with vended
    credentials); Iceberg tables get their PyIceberg FileIO equivalents
    (``s3.endpoint``, ``s3.region``, ``s3.proxy-uri``, ...), and Delta Sharing
    downloads their ``proxy_url`` and ``timeout``. `iceberg_properties` are
    PyIceberg catalog and FileIO properties passed verbatim, over those.

    A pickled connection, table or credential provider carries no literal
    secret (a token, a client secret): a worker re-derives Databricks auth
    from its own environment. `ship_credentials=True` pickles them too, for
    workers that have no auth of their own.
    """
    if not isinstance(allow_sql_fallback, bool):
        # The fallback costs money, so it is on only when asked for exactly.
        # A string from an environment variable -- "false", "0" -- is truthy
        # and switched it on.
        raise InvalidArgumentError(
            f"allow_sql_fallback must be True or False, not {allow_sql_fallback!r}"
        )
    if catalog is not None and uri is not None and str(uri).strip():
        # The URI was silently ignored: ds.connect("hms://...", catalog=cat)
        # talked to `cat` while the caller believed it reached the metastore.
        raise InvalidArgumentError(
            f"pass a connection URI or catalog=, not both (got {uri!r} and a "
            f"{type(catalog).__name__})"
        )
    if storage_options is not None and not isinstance(storage_options, Mapping):
        # A list of pairs or a string reached the engines, which failed with a
        # raw AttributeError ('list' object has no attribute 'items').
        raise InvalidArgumentError(
            f"storage_options is a {{key: value}} dict, not {type(storage_options).__name__}"
        )
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
    from ._storage import canonical_options, http_settings, iceberg_fileio_properties
    from .engine.iceberg import IcebergEngine
    from .engine.sharing import SharingEngine

    # Canonical up front, so a dict naming one setting twice is refused here
    # rather than at the first table.
    for cloud_scheme in ("s3://", "abfss://", "gs://"):
        canonical_options(storage_options, cloud_scheme)
    if SharingEngine.available():
        http = http_settings(storage_options)
        engines[EngineKind.SHARING] = SharingEngine(
            proxy_url=http.get("proxy_url"),
            **({"request_timeout": http["timeout"]} if "timeout" in http else {}),
        )
    if IcebergEngine.available():
        # connect()'s storage_options never reached PyIceberg: the proxy, CA,
        # endpoint and region the other engines honour were silently ignored.
        properties = {**iceberg_fileio_properties(storage_options), **(iceberg_properties or {})}
        engines[EngineKind.ICEBERG] = IcebergEngine(token=token, properties=properties or None)
        engines[EngineKind.ICEBERG]._ship_secrets = bool(ship_credentials)  # type: ignore[attr-defined]
    if ship_credentials and hasattr(resolved_catalog, "_ship_secrets"):
        resolved_catalog._ship_secrets = True

    # The warehouse addresses tables by Unity Catalog name in its own
    # workspace. Any other catalog -- an OSS Unity Catalog server above all,
    # whose names look exactly the same -- has tables the warehouse cannot
    # see, and a same-named Databricks table was read and written instead.
    warehouse_catalog = bool(getattr(resolved_catalog, "sql_warehouse", False))
    if allow_sql_fallback and not warehouse_catalog:
        raise InvalidArgumentError(
            "allow_sql_fallback=True routes operations to a Databricks SQL warehouse, which "
            "serves only the tables of its own workspace's Unity Catalog; this connection's "
            f"catalog is {getattr(resolved_catalog, 'name', type(resolved_catalog).__name__)!r}. "
            "Leave the fallback off here, or open the table through ds.connect() to the "
            "Databricks workspace that holds it"
        )
    if allow_sql_fallback:
        from .engine.sql import SqlEngine

        if catalog is not None and not any(v is not None for v in (profile, host, token, config)):
            # A catalog passed in carries its own authentication; the engine
            # built from none resolved the SDK defaults, which may be another
            # workspace than the one whose tables it was sent.
            profile = getattr(catalog, "_profile", None)
            host = getattr(catalog, "_host", None)
            token = getattr(catalog, "_token", None)
            config = getattr(catalog, "_explicit_config", None)
        engines[EngineKind.SQL] = SqlEngine(
            profile=profile,
            host=host,
            token=token,
            config=config,
            warehouse_id=warehouse_id,
            staging_volume=staging_volume,
        )

    if default_catalog is None:
        # A metastore has two levels: `db.table` on an hms:// or glue://
        # connection was refused as "a two-part name ... no default catalog".
        default_catalog = getattr(resolved_catalog, "default_catalog_name", None)
    return Connection(
        catalog=resolved_catalog,
        router=Router(
            engines=engines,
            allow_sql_fallback=allow_sql_fallback,
            warehouse_catalog=warehouse_catalog,
        ),
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


def _only_the_create_commit(log_dir: str) -> bool:
    """Whether a local _delta_log holds nothing but version 0.

    write_table removes the log of a table it created when the first write
    fails. Another writer can append to that table in between (it exists from
    the create on), and removing the log then deleted that writer's committed
    data along with the empty table.
    """
    try:
        names = os.listdir(log_dir)
    except OSError:
        return False
    versions = {n.split(".", 1)[0] for n in names if n[:1].isdigit()}
    return versions <= {"0" * 20}


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
        # A list or dict schema is built with pyarrow.schema().
        _require("pyarrow", "pyarrow", "a list or dict schema (pass an Arrow schema object)")
        raise
    if not isinstance(schema, pa.Schema):
        if hasattr(schema, "__arrow_c_schema__") and not isinstance(schema, (list, dict)):
            try:
                schema = pa.schema(schema)
            except Exception:
                return schema  # a Delta schema object; the engine reads it
        else:
            try:
                schema = pa.schema(_sql_typed(schema))
            except (TypeError, ValueError, pa.ArrowInvalid) as exc:
                raise InvalidArgumentError(f"not a table schema: {schema!r} ({exc})") from exc
    if len(schema) == 0:
        # Databricks created a zero-column table from it, which nothing can
        # write to; Delta needs at least one column.
        raise InvalidArgumentError("create_table needs a schema with at least one column")
    return _widen_unsigned(schema)


def _sql_typed(schema: Any) -> Any:
    """A `{name: type}` or `[(name, type)]` schema with SQL type names read as Delta reads them.

    pyarrow knows only its own aliases, so ``{"id": "bigint"}``, ``"long"``,
    ``"timestamp"`` or ``"decimal(10,2)"`` -- the names Spark, Databricks and
    the Delta log use -- were refused. A name neither reads is left for
    pyarrow, which names what it cannot read.
    """
    from .engine.metadata import sql_type_to_delta
    from .engine.sharing import _delta_type_to_arrow

    def arrow(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return _delta_type_to_arrow(sql_type_to_delta(value))
        except (ValueError, DeltaSwampError):
            return value

    if isinstance(schema, dict):
        return {name: arrow(value) for name, value in schema.items()}
    if isinstance(schema, list):
        return [
            (item[0], arrow(item[1]), *item[2:])
            if isinstance(item, tuple) and len(item) >= 2
            else item
            for item in schema
        ]
    return schema


def _widen_unsigned(schema: Any) -> Any:
    """Map unsigned integers to the signed type that holds all their values,
    nanosecond timestamps to microseconds and zoned timestamps to UTC.

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
        if pa.types.is_timestamp(t) and t.unit == "ns":
            # pandas' datetime64[ns] (and pa.timestamp("ns")) created a
            # timestamp_nanos column behind delta-rs's non-standard
            # timestampNanos feature, which DuckDB, Spark and Databricks
            # cannot read. A Delta timestamp is microseconds, as Spark
            # writes it; the data is cast on the way in, as appends cast it.
            t = pa.timestamp("us", t.tz)
        if pa.types.is_timestamp(t) and t.tz is not None and t.tz.upper() not in ("UTC", "+00:00"):
            # A Delta timestamp is an instant stored in UTC; the zone is only
            # how the caller's frame displays it. Appends already convert, and
            # a create with America/New_York failed with a bare Exception.
            return pa.timestamp(t.unit, "UTC")
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


#: The local SQL engines' exceptions for a query that is wrong: it does not
#: parse, or names a table, column or function that is not there.
_BAD_QUERY = {
    "duckdb": ("ParserException", "CatalogException", "BinderException", "SyntaxException"),
    "polars": (
        "SQLSyntaxError",
        "SQLInterfaceError",
        "ColumnNotFoundError",
        "SchemaFieldNotFoundError",
    ),
}


@contextlib.contextmanager
def _local_sql_errors(engine: str, query: str) -> Iterator[None]:
    """duckdb's or Polars' exceptions from running `query`, as this library's.

    They escaped raw, so `except DeltaSwampError` around conn.sql() caught
    nothing. A query that does not parse or names what is not there is the
    caller's mistake; any other failure is the local engine's.
    """
    from .errors import engine_error

    try:
        yield
    except DeltaSwampError:
        raise
    except Exception as exc:
        names = {cls.__name__ for cls in type(exc).__mro__}
        if names & set(_BAD_QUERY[engine]):
            raise InvalidArgumentError(
                f"sql(): {engine} refused the query: {str(exc).strip()[:400]}"
            ) from exc
        raise engine_error(
            f"sql(): {engine} failed running the query: {type(exc).__name__}: {exc}",
            engine=engine,
            operation="sql",
            original=exc,
        ) from exc


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


_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")


def _check_local_create_path(name: str) -> None:
    """Refuse a local path neither engine can read back once it is created.

    Both write version 0 at the literal path, then read the log through a URL
    that decodes `%41` to `A` and treats `\\` as a separator: the create failed
    with "File not found" after the table was written, and nothing could open
    it afterwards.
    """
    if "://" in name:
        # A URL, file:// included, whose escapes are meant as escapes. But a
        # URL reads `?` and `#` as a query and a fragment: the table went to
        # `.../x` for "file://.../x#y", and every write through the returned
        # handle landed there too, while opening the URI was refused.
        if "?" in name or "#" in name:
            raise InvalidArgumentError(
                f"the table URI {name!r} contains '?' or '#', which a URL reads as a query "
                "or fragment, so the table would be created at a different path; nothing "
                "was created. Percent-encode them (%3F, %23) or choose a name without them"
            )
        return
    bad = _PERCENT_ESCAPE.search(name)
    if bad is not None or (os.sep == "/" and "\\" in name):
        what = repr(bad.group(0)) if bad is not None else "a backslash"
        raise InvalidArgumentError(
            f"the local path {name!r} contains {what}, which the engines read back as a "
            "different path; nothing was created. Choose a directory name without it"
        )


def _refuse_foreign_files(location: str, options: Mapping[str, str] | None = None) -> None:
    """Refuse to create a table where files that are not a Delta table already are.

    Spark refuses a create at a non-empty location. Here the create succeeded,
    and the table's first standard VACUUM then deleted every file already in
    the directory -- documents, another dataset -- as unreferenced. What a
    Delta table leaves behind is allowed (a log directory, Parquet and
    deletion-vector files, ``col=value`` partition directories, hidden
    entries): a table recreated where one was, or an interrupted create.
    """
    from urllib.parse import unquote, urlparse

    from .engine.deltars import _delta_file_name

    parsed = urlparse(location)
    listed: list[tuple[str, bool]] = []
    if parsed.scheme.lower() in ("", "file"):
        path = unquote(parsed.path) if parsed.scheme else location
        if not os.path.isdir(path):
            return
        listed = [(e.name, e.is_dir()) for e in os.scandir(path)]
    else:
        try:
            import pyarrow.fs as pafs
            from deltalake._internal import DeltaFileSystemHandler

            handler = DeltaFileSystemHandler(location, dict(options or {}))
            infos = handler.get_file_info_selector("", True, False)
        except Exception:
            # Not listable with these credentials (or not at all): the create
            # itself decides, as before.
            return
        listed = [
            (str(i.path).rstrip("/").rsplit("/", 1)[-1], i.type == pafs.FileType.Directory)
            for i in infos
        ]
    entries = [
        name
        for name, is_dir in listed
        if name
        and not name.startswith(("_", "."))
        and not (is_dir and "=" in name)
        and not (not is_dir and _delta_file_name(name))
    ]
    if entries:
        shown = ", ".join(sorted(entries)[:3]) + (", ..." if len(entries) > 3 else "")
        raise UnreachableTableError(
            f"create a table at {location}",
            f"the location already holds {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} "
            f"that are not a Delta table ({shown}); a VACUUM of the new table would delete "
            "them as unreferenced files",
            "create the table in an empty directory, or convert_to_delta() a directory of "
            "Parquet files",
        )


def _refuse_local_catalog_location(catalog: Any, location: str) -> None:
    """Refuse a local location a remote catalog chose for a table.

    A catalog server that answers a staging or create request with
    ``file:///home/<user>/...`` had version 0 written into that directory of
    this machine, and a VACUUM later deleted what else was there. Only a
    catalog on this machine (the OSS UC quickstart) may place tables on its
    filesystem.
    """
    from urllib.parse import urlparse

    scheme = urlparse(location).scheme.lower()
    if scheme not in ("", "file"):
        return
    endpoint = str(
        getattr(catalog, "_base_url", "")
        or getattr(catalog, "base_url", "")
        or getattr(catalog, "_host", "")
        or ""
    )
    host = (urlparse(endpoint).hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        return
    raise UnreachableTableError(
        f"create a table at {location}",
        f"the catalog{f' at {endpoint}' if endpoint else ''} placed the table on this "
        "machine's filesystem, which only a catalog running on this machine may do",
        "check the catalog's storage root; a remote catalog must use cloud storage",
    )


def _local_log_dir(ref: TableRef, name: str) -> str | None:
    """The `_delta_log` directory of a local-path table name, else None.

    Takes the ref the caller parsed with the connection's defaults: parsing
    `name` again without them refused every bare `orders`, so write_table's
    create-if-absent path failed on a connection with default_catalog/schema.
    """
    if ref.kind is not RefKind.PATH:
        return None
    location = name
    if location.startswith("file://"):
        location = location[len("file://") :]
    elif "://" in location:
        return None
    return os.path.join(os.path.abspath(location), "_delta_log")


def _registered_anyway(catalog: Any, ref: Any, exc: BaseException) -> bool:
    """Whether a failed finalize had in fact registered the table."""
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


def _ref_scheme(ref: TableRef) -> str | None:
    """The catalog a reference names by its scheme, or None for a bare name.

    A bare ``catalog.schema.table`` parses with the ``uc`` scheme, but it
    names no catalog of its own: it belongs to whichever one the connection
    is bound to. Only a written-out ``uc://`` / ``unity://`` names Unity
    Catalog.
    """
    scheme = ref.scheme
    if scheme in ("uc", "unity"):
        raw = (ref.raw or "").lstrip().lower()
        return "uc" if raw.startswith(("uc://", "unity://")) else None
    if scheme == "hive":
        return "hms"
    return scheme


#: What each reference scheme names, for refusals.
_SCHEME_LABELS = {
    "uc": "Unity Catalog",
    "hms": "Hive Metastore",
    "glue": "AWS Glue",
    "deltasharing": "Delta Sharing",
}


#: Properties by which a create chooses whether a table is catalog-managed.
_CATALOG_MANAGED_KEYS = frozenset(
    {"delta.feature.catalogmanaged", "delta.feature.catalogowned-preview"}
)


#: The save modes a create at a path takes.
_CREATE_MODES = frozenset({"error", "create", "ignore", "overwrite", "append"})


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
    # Not in repr: these carry object-store secrets (access keys, SAS tokens),
    # and a Connection printed in a log or a traceback leaked them verbatim.
    storage_options: dict[str, str] = dataclasses.field(default_factory=dict, repr=False)
    # Catalogs built for references that name another catalog (glue://,
    # hms://host), keyed by the URI they were built from.
    _foreign: dict[str, Any] = dataclasses.field(default_factory=dict, repr=False, compare=False)

    def _catalog_for(self, ref: TableRef) -> Any:
        """The catalog that serves `ref`.

        A reference whose scheme names a catalog goes to that catalog.
        Sending every one to the bound catalog read the wrong table without
        a word: on a Unity Catalog connection ``glue://crm.customers`` opened
        UC's ``glue.crm.customers``, and ``hms://other:9083/db/t`` Databricks'
        own ``hive_metastore.db.t``. Glue and a Hive Metastore are reachable
        from any connection (the reference carries the catalog id or the
        metastore endpoint), so a catalog for them is built on first use and
        kept. Unity Catalog and Delta Sharing need credentials a reference
        does not carry, so a connection bound elsewhere refuses them.
        """
        if ref.kind is RefKind.PATH:
            return FilesystemCatalog()
        scheme = _ref_scheme(ref)
        # A plugin catalog that declares nothing keeps serving every name.
        served = getattr(self.catalog, "ref_schemes", None)
        if scheme is None or served is None or scheme in served:
            return self.catalog
        bound = getattr(self.catalog, "name", type(self.catalog).__name__)
        label = _SCHEME_LABELS.get(scheme, scheme)
        if scheme == "glue":
            uri = f"glue://{ref.endpoint or ''}"
        elif scheme == "hms" and ref.endpoint:
            uri = f"hms://{ref.endpoint}"
        elif scheme == "hms":
            raise InvalidReferenceError(
                f"{ref.raw or ref!r} names no metastore endpoint, and this connection is "
                f"bound to the {bound} catalog, not a Hive Metastore. Name the metastore "
                "(hms://host:9083/db/table), or connect to it with "
                "ds.connect('hms://host:9083')."
            )
        else:
            raise InvalidReferenceError(
                f"{ref.raw or ref!r} names a {label} table, but this connection is bound to "
                f"the {bound} catalog, which does not serve it. Open it through a "
                f"connection to {label}."
            )
        catalog = self._foreign.get(uri)
        if catalog is None:
            catalog = self._foreign.setdefault(uri, catalog_for_uri(uri))
        return catalog

    def _bound_ref(self, name: str, what: str) -> TableRef:
        """A catalog name for an operation only the bound catalog performs."""
        try:
            ref: TableRef = self._catalog_ref(name, what)
        except InvalidReferenceError as exc:
            text = str(exc)
            if "volume" not in what or not ("bare table name" in text or "two-part" in text):
                raise
            # parse_ref speaks of tables; this names a volume.
            raise InvalidReferenceError(
                f"cannot {what} {name!r}: a volume is named catalog.schema.volume, and "
                "the connection has no default catalog/schema to complete it"
            ) from exc
        if self._catalog_for(ref) is not self.catalog:
            raise InvalidReferenceError(
                f"cannot {what} {name!r}: it names another catalog than the one this "
                f"connection is bound to ({getattr(self.catalog, 'name', 'current')})"
            )
        return ref

    def table(self, name: str, *, version: int | None = None) -> Table:
        """Open a table by three-level name, or by path."""
        _check_version(version)
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
        return Table(self, self._catalog_for(ref).resolve(ref), version=version)

    def open_table(self, path: str, *, version: int | None = None) -> Table:
        """Open a table directly by storage path, bypassing the catalog."""
        _check_version(version)
        ref = parse_ref(path)
        if ref.kind is not RefKind.PATH:
            raise InvalidReferenceError(
                f"{path!r} is a catalog table name (catalog.schema.table), not a storage "
                "path; open it with .table() instead"
            )
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

    def _catalog_call(self, method: str, what: str, *args: Any, catalog: Any = None) -> Any:
        """Call an optional catalog method, refusing cleanly when it is absent.

        The plugin contract is `resolve` and `list_tables`; the built-in path
        catalog raises NotImplementedError for the rest. Both escaped raw
        (AttributeError, NotImplementedError) instead of a DeltaSwampError.
        """
        catalog = self.catalog if catalog is None else catalog
        fn = getattr(catalog, method, None)
        if not callable(fn):
            raise UnreachableTableError(
                what,
                f"the {getattr(catalog, 'name', type(catalog).__name__)} catalog "
                f"does not implement {method}()",
            )
        return _call(what, fn, *args)

    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]:
        what = f"list tables in {catalog}.{schema}"
        return list(self._catalog_call("list_tables", what, catalog, schema))

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
        schema = _schema_arg(schema)
        if _schema_names(schema) == []:
            # Databricks creates a zero-column table; the kernel refuses one
            # only after the catalog has allocated it.
            raise InvalidArgumentError("a table needs at least one column")
        if mode not in _CREATE_MODES:
            # Checked for catalog names too, which blamed any unknown mode on
            # "replacing a catalog table".
            raise InvalidArgumentError(
                f"create mode must be one of {sorted(_CREATE_MODES)}, not {mode!r}"
            )
        if properties is not None and not isinstance(properties, Mapping):
            # A list failed with a bare AttributeError, or ValueError from dict().
            raise InvalidArgumentError(
                f"properties= maps property names to values, e.g. "
                f"{{'delta.appendOnly': 'true'}}; got a {type(properties).__name__}"
            )
        partition_by = _names_arg(partition_by, "partition_by")
        cluster_by = _names_arg(cluster_by, "cluster_by")
        if partition_by and cluster_by:
            raise InvalidArgumentError(
                "a table is either partitioned or liquid-clustered, not both"
            )
        _check_layout(schema, partition_by, cluster_by)
        # JSON stats off with no word on struct stats: record Spark's default
        # (struct on), or this library's checkpoints would keep no stats.
        properties = with_checkpoint_stats(properties)
        clash = cdf_name_clash(_schema_names(schema) or (), properties)
        if clash:
            # The kernel's create refused this; delta-rs's created the table,
            # and every DML and change-feed read on it then failed.
            raise cdf_clash_error(f"create {name}", clash)
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
            _check_local_create_path(name)
            present = self.table_exists(name)
            if mode in ("error", "create", "ignore") and present:
                if mode == "ignore":
                    # Nothing is created, so nothing (the comment included)
                    # may be changed on the table that is already there.
                    return self.table(name)
                raise UnreachableTableError(
                    f"create {name}",
                    "a Delta table already exists there",
                    "pass mode='ignore' to keep it or mode='overwrite' to replace it",
                )
            if not present:
                _refuse_foreign_files(name, self.storage_options)
            try:
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
            except Exception as exc:
                # Another writer created it between the check and this create:
                # delta-rs answered with a raw DeltaError, and mode="ignore"
                # failed on exactly the case it exists for.
                if (
                    mode not in ("error", "create", "ignore")
                    or "exist" not in str(exc).lower()
                    or not self.table_exists(name)
                ):
                    raise
                if mode == "ignore":
                    return self.table(name)
                raise UnreachableTableError(
                    f"create {name}",
                    "a Delta table already exists there (another writer created it first)",
                    "pass mode='ignore' to keep it or mode='overwrite' to replace it",
                ) from exc

        lifecycle = self._lifecycle_catalog(f"create the catalog table {ref}", ref)
        exists = None
        if mode == "ignore":
            # CREATE TABLE IF NOT EXISTS: it was refused as a "replace".
            exists = self.table_exists(name)
            if exists:
                return self.table(name)
            mode = "error"
        if mode not in ("error", "create"):
            raise UnreachableTableError(
                f"create {ref} with mode={mode!r}",
                "a catalog table is created once; replacing one is a different operation",
                "drop_table() first, or write to it with mode='append' / 'overwrite'",
            )
        if exists is None and self.table_exists(name):
            # The warehouse's raw TABLE_OR_VIEW_ALREADY_EXISTS otherwise, or a
            # staging table allocated for nothing.
            raise UnreachableTableError(
                f"create {ref}",
                "a table of that name already exists",
                "pass mode='ignore' to keep it, or write to it with conn.table(...)",
            )
        if location is not None:
            return self._create_external(
                lifecycle, ref, location, schema, partition_by, cluster_by, properties, comment
            )
        return self._create_managed(
            lifecycle, ref, schema, partition_by, cluster_by, properties, comment
        )

    def _lifecycle_catalog(self, what: str, ref: TableRef) -> Any:
        from .catalog.base import TableLifecycleCatalog

        catalog = self._catalog_for(ref)
        if not isinstance(catalog, TableLifecycleCatalog):
            raise UnreachableTableError(
                what,
                f"the {getattr(catalog, 'name', 'current')} catalog cannot register "
                "tables, and writing a Delta log alone would leave it orphaned while this "
                "call appeared to succeed",
                "create the table at a path with create_table('<path>', ...), then register "
                "it with the catalog's own tools",
            )
        return catalog

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
        # The schema too: its column metadata and types decide which engine
        # can declare the features the table needs.
        engine: Any = self.router.engine_for(
            Operation.CREATE,
            resolved,
            properties=properties,
            cluster_by=cluster_by,
            mode=mode,
            schema=schema,
            partition_by=partition_by,
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
        from ._storage import engine_options, store_options

        _refuse_foreign_files(
            location,
            store_options(
                engine_options(
                    self.storage_options, getattr(credentials, "secrets", None), location
                )
            ),
        )
        staged = ResolvedTable(
            ref=parse_ref(location),
            location=location,
            credential_provider=StaticCredentialProvider(credentials),
        )
        try:
            self._create_at(
                staged,
                schema,
                partition_by=partition_by,
                cluster_by=cluster_by,
                mode="error",
                properties=properties,
            )
        except EngineError as exc:
            # delta-rs says only in its message (which the engine boundary's
            # EngineError carries) that a table is already at the location:
            # the commonest failure of all, and it wants registering.
            if "already exists" not in str(exc).lower():
                raise
            raise UnreachableTableError(
                f"create {ref} at {location}",
                f"a Delta table already exists at that location ({exc})",
                f"conn.register_table({str(ref)!r}, {location!r}) registers it as it is",
            ) from exc
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
            if (
                not self.router.allow_sql_fallback
                or sql is None
                or not self.router.warehouse_catalog
            ):
                raise
            # The staging flow makes a catalog-managed table, as documented;
            # a plain CREATE TABLE made an ordinary managed one, which Unity
            # Catalog then refuses external writes to, so every later write
            # needed the warehouse too. Unless the caller chose otherwise.
            fallback_properties = dict(properties or {})
            if not any(key.lower() in _CATALOG_MANAGED_KEYS for key in fallback_properties):
                fallback_properties["delta.feature.catalogManaged"] = "supported"
            sql.create_managed(
                ref.full_name,
                schema,
                partition_by=partition_by,
                cluster_by=cluster_by,
                properties=fallback_properties,
                comment=comment,
            )
            return self.table(ref.full_name)
        _refuse_local_catalog_location(catalog, staging.location)
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
        from ._storage import engine_options, store_options

        # The engines' merge rule: a plain dict merge kept alias spellings side
        # by side, and object_store chose between them at random.
        options = store_options(
            engine_options(self.storage_options, staging.storage_options, staging.location)
        )
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

        ref = self._catalog_ref(name, "register")
        catalog = self._lifecycle_catalog(f"register {ref}", ref)
        from .credentials.base import StaticCredentialProvider

        staged = ResolvedTable(
            ref=parse_ref(location),
            location=location,
            credential_provider=StaticCredentialProvider(
                catalog.path_credentials(location, "PATH_READ")
            ),
        )
        kernel: Any = self.router.engines[EngineKind.KERNEL]
        try:
            metadata = json.loads(kernel.snapshot(staged).metadata_json())
        except EngineError as exc:
            text = str(exc).lower()
            if any(
                m in text
                for m in (
                    "no files in log segment",
                    "not found",
                    "no such file",
                    # The kernel's words for a local path that is not there.
                    "path does not exist",
                    "invalid table location",
                )
            ):
                raise UnreachableTableError(
                    f"register {ref}",
                    f"there is no Delta log at {location} ({str(exc).splitlines()[0][:200]})",
                    "check the location, or create the table with create_table(location=...)",
                ) from exc
            raise
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
        return list(self._catalog_call("list_catalogs", "list catalogs"))

    def list_schemas(self, catalog: str | None = None) -> list[str]:
        """Schema names in `catalog`, defaulting to the connection's own."""
        target = catalog or self.default_catalog
        if target is None:
            raise InvalidReferenceError(
                "no catalog given and the connection has no default_catalog"
            )
        return list(self._catalog_call("list_schemas", f"list schemas in {target}", target))

    def drop_table(self, name: str, *, if_exists: bool = False) -> None:
        """Remove a table from the catalog.

        For an EXTERNAL table the files remain; only the registration goes.
        With `if_exists`, a table that is not there is not an error (DROP
        TABLE IF EXISTS).
        """
        ref = self._catalog_ref(name, "drop")
        try:
            self._catalog_call("drop_table", f"drop {ref}", ref, catalog=self._catalog_for(ref))
        except TableNotFoundError:
            if not if_exists:
                raise

    def table_exists(self, name: str) -> bool:
        """Whether a table is already there.

        For a path this reads the storage; for a catalog name it asks the
        catalog, so a name that resolves but has no Delta log counts as absent.
        """
        ref = parse_ref(
            name, default_catalog=self.default_catalog, default_schema=self.default_schema
        )
        exists = getattr(self._catalog_for(ref), "table_exists", None)
        if ref.kind is RefKind.CATALOG and callable(exists):
            try:
                return bool(exists(ref))
            except NotImplementedError:
                pass
        try:
            table = self.table(name)
        except DeltaSwampError:
            return False
        if table.resolved.is_shared:
            # A shared table has no storage location by design -- the share
            # serves presigned URLs -- and resolving it already asked the
            # server. The location test below reported every one absent.
            return True
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
            # A malformed argument, as create_table(mode=...) reports it; not
            # "no engine can serve this table".
            raise InvalidArgumentError(
                f"write_table mode={mode!r}: the modes are 'error', 'ignore', 'append' "
                "and 'overwrite'"
            )

        data = _write_data(data)
        # Checked before anything is created: a misspelt option failed the
        # first write, as a TypeError naming Table.append, after the create.
        check_keywords(
            "write_table", Table.overwrite if mode == "overwrite" else Table.append, write_options
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
            log_dir = _local_log_dir(
                parse_ref(
                    name,
                    default_catalog=self.default_catalog,
                    default_schema=self.default_schema,
                ),
                name,
            )
            log_existed = log_dir is not None and os.path.exists(log_dir)
            try:
                table = self.create_table(
                    name,
                    schema if schema is not None else _schema_of(data),
                    location=location,
                    partition_by=partition_by,
                    cluster_by=cluster_by,
                    properties=properties,
                )
            except UnreachableTableError:
                # Created by another writer since the existence check: the
                # mode decides, exactly as if it had been there all along.
                if mode == "error" or not self.table_exists(name):
                    raise
                if mode == "ignore":
                    return self.table(name)
                return self._write_existing(
                    name, data, mode, partition_by, properties, write_options
                )
            try:
                table.append(data, **write_options)
            except BaseException as exc:
                # Creating and writing are two commits. A write that failed
                # left an empty table behind, and the retry was then refused
                # because it "already exists". A local log this call created
                # is removed; elsewhere, say what is left.
                if (
                    log_dir is not None
                    and not log_existed
                    and os.path.isdir(log_dir)
                    and _only_the_create_commit(log_dir)
                ):
                    import shutil

                    shutil.rmtree(log_dir, ignore_errors=True)
                elif isinstance(exc, Exception):
                    exc.add_note(
                        f"deltaswamp: the table {name} was created, but this first write "
                        "failed; it exists now and is empty"
                    )
                raise
            return table

        return self._write_existing(name, data, mode, partition_by, properties, write_options)

    def _write_existing(
        self,
        name: str,
        data: Any,
        mode: str,
        partition_by: list[str] | None,
        properties: dict[str, str] | None,
        write_options: dict[str, Any],
    ) -> Table:
        """write_table's append/overwrite into a table that is already there."""
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
        if not isinstance(query, str):
            # duckdb refused it with its own InvalidInputException.
            raise InvalidArgumentError(
                f"sql() takes the query as a string, not {type(query).__name__}"
            )
        if tables is not None and not isinstance(tables, Mapping):
            # A list failed with a bare AttributeError ('list' has no 'items').
            raise InvalidArgumentError(
                f"tables= maps the names the query uses to tables, e.g. {{'t': 'cat.sch.t'}}; "
                f"got a {type(tables).__name__}"
            )
        if engine == "warehouse":
            sql_engine: Any = self.router.engines.get(EngineKind.SQL)
            if not self.router.warehouse_catalog:
                raise UnreachableTableError(
                    "run SQL on a warehouse",
                    "a SQL warehouse serves only a Databricks workspace's own tables, and "
                    "this connection's catalog is not Databricks Unity Catalog",
                    "run it locally (engine='duckdb' or 'polars')",
                )
            if sql_engine is None or not self.router.allow_sql_fallback:
                raise UnreachableTableError(
                    "run SQL on a warehouse",
                    "the SQL warehouse fallback is not enabled on this connection",
                    SQL_FALLBACK_REMEDY,
                )
            return sql_engine.query(query)

        # Refuse an unknown engine before reading every table in `tables`.
        if engine not in ("duckdb", "polars"):
            raise InvalidArgumentError(
                f"sql engine={engine!r}: the engines are 'duckdb', 'polars' and 'warehouse'"
            )
        _require("pyarrow", "pyarrow")
        module = _require(engine, engine)
        # Lazy datasets, not tables read in full: each engine pushes the
        # query's projection and simple WHERE comparisons into the scan. Each
        # is pinned to the version current now, so one statement reads one
        # snapshot of each table however many times it scans it.
        frames = {
            alias: (ref if isinstance(ref, Table) else self.table(ref))._lazy_dataset(
                duckdb_filters=engine == "duckdb"
            )
            for alias, ref in (tables or {}).items()
        }
        if engine == "duckdb":
            con = module.connect()
            try:
                for alias, frame in frames.items():
                    con.register(alias, frame)
                with _local_sql_errors("duckdb", query):
                    relation = con.sql(query)
                    # A statement (CREATE, SET, ...) yields no relation to convert.
                    return relation.to_arrow_table() if relation is not None else None
            finally:
                # An in-memory database per call; leaving it open leaked it
                # and every registered frame until garbage collection.
                con.close()
        from ._lazy import polars_frame

        ctx = module.SQLContext({alias: polars_frame(f) for alias, f in frames.items()})
        with _local_sql_errors("polars", query):
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
        ref = self._bound_ref(name, "create the volume")
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
        ref = self._bound_ref(name, "drop the volume")
        cat = self._namespaces(f"drop volume {name}")
        _call(f"drop volume {name}", cat.drop_volume, ref.catalog, ref.schema, ref.table)

    def volume(self, name: str) -> Any:
        """A Unity Catalog volume: list, read, write and delete its files."""
        ref = self._bound_ref(name, "open the volume")
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
        ref = self._bound_ref(name, "undrop")
        sql_engine: Any = self.router.engines.get(EngineKind.SQL)
        if not self.router.warehouse_catalog:
            raise UnreachableTableError(
                f"undrop {ref}",
                "UNDROP is Databricks-only, and this connection's catalog is not "
                "Databricks Unity Catalog",
            )
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
        return Table(self, self._catalog_for(ref).resolve(ref))

    def preflight(self) -> list[str]:
        """Check workspace prerequisites. Empty list means ready.

        Worth running once before anything else. On Databricks it checks
        external data access on the metastore and, when the connection has a
        default catalog and schema, EXTERNAL USE SCHEMA on that schema: both
        are off by default, grantable only by someone else, and account for
        most first-contact failures. With the SQL fallback on, it also checks
        that the warehouse exists, and says when no staging volume is set, so
        writes the direct engines cannot serve would be refused.

        It cannot see per-table limits: Unity Catalog allows external writes
        only to some table kinds, which `Table.can()` reports per table.
        """
        check = getattr(self.catalog, "preflight", None)
        problems: list[str] = []
        if callable(check):
            schema = (
                f"{self.default_catalog}.{self.default_schema}"
                if self.default_catalog and self.default_schema
                else None
            )
            import inspect

            try:
                takes_schema = "schema" in inspect.signature(check).parameters
            except (TypeError, ValueError):
                takes_schema = False
            # A plugin catalog's preflight may take no arguments.
            problems = list(check(schema=schema) if takes_schema else check())
        sql_engine: Any = self.router.engines.get(EngineKind.SQL)
        if self.router.allow_sql_fallback and sql_engine is not None:
            resolve = getattr(sql_engine, "_resolve_warehouse", None)
            if callable(resolve) and resolve() is None:
                problems.append(
                    "the SQL fallback is on, but no usable warehouse was found: "
                    + (
                        getattr(sql_engine, "_selection_error", None)
                        or getattr(sql_engine, "_last_listing_error", None)
                        or "no SQL warehouse configured"
                    )
                )
            if getattr(sql_engine, "_staging_volume", True) is None:
                problems.append(
                    "the SQL fallback has no staging volume, so it cannot serve writes "
                    "(append, overwrite, MERGE) to tables the direct engines cannot write; "
                    "pass ds.connect(..., staging_volume='catalog.schema.volume')"
                )
            else:
                # A staging volume that does not exist (or cannot be read)
                # passed preflight, and the first write then failed.
                probe = getattr(sql_engine, "staging_volume_problem", None)
                problem = probe() if callable(probe) else None
                if problem is not None:
                    problems.append(problem)
        return problems
