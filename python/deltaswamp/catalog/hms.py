"""Hive Metastore.

A legacy surface, supported because plenty of on-premise and non-Databricks
deployments still run one. Note it is legacy on the Databricks side too: from
30 September 2026 new workspaces are provisioned without a Hive metastore, DBFS
root or DBFS mounts. Databricks' own `hive_metastore` catalog is *not* a Unity
Catalog securable, so it cannot be credential-vended and is reachable only
through SQL.

HMS holds a stale, partial view of a Delta table's schema. We take only the
storage location from it and treat `_delta_log` as the truth -- which is what
every correct Delta-on-HMS reader does.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..errors import InvalidReferenceError, PreflightError
from ..identity import RefKind, TableRef
from .base import ResolvedTable, TableType
from .glue import normalise_location

__all__ = ["HiveMetastoreCatalog"]

# How a Delta table announces itself in HMS table parameters.
_DELTA_MARKERS = (
    ("spark.sql.sources.provider", "delta"),
    ("table_type", "DELTA"),
)

#: Spark's HiveExternalCatalog records a non-Hive-compatible data source table
#: (every Delta table) with this suffix on the location and the real path in
#: the SerDe ``path`` parameter.
_PLACEHOLDER = "-__PLACEHOLDER__"

_VIEW_TYPES = ("VIRTUAL_VIEW", "MATERIALIZED_VIEW")


def _parse_endpoint(text: str, default_port: int = 9083) -> tuple[str, int]:
    """``host``, ``host:port``, ``[v6]:port`` or ``thrift://host:port`` -> (host, port)."""
    body = text.split("://", 1)[1] if "://" in text else text
    body = body.removeprefix("thrift://").split("/", 1)[0].split("?", 1)[0]
    if body.startswith("["):
        host, _, rest = body[1:].partition("]")
        port_part = rest.removeprefix(":")
    elif body.count(":") > 1:
        host, port_part = body, ""  # a bare IPv6 address, no port
    else:
        host, _, port_part = body.partition(":")
    if not port_part:
        return host, default_port
    try:
        port = int(port_part)
    except ValueError:
        raise InvalidReferenceError(
            f"{text!r}: the Hive Metastore port {port_part!r} is not a number"
        ) from None
    if not 0 < port < 65536:
        raise InvalidReferenceError(f"{text!r}: the Hive Metastore port {port} is out of range")
    return host, port


def _hms_class() -> Any:
    # pymetastore >= 0.4 ships `pymetastore.metastore`; there is no
    # `pymetastore.hms` module, so importing that always failed.
    for module in ("pymetastore.metastore", "pymetastore"):
        try:
            found = getattr(importlib.import_module(module), "HMS", None)
        except ImportError:
            continue
        if found is not None:
            return found
    raise PreflightError(
        "Hive Metastore support needs the pymetastore package; "
        "install 'deltaswamp[hms]'. (delta-rs has no HMS catalog crate at all, "
        "so there is no built-in alternative.)"
    )


class HiveMetastoreCatalog:
    """Resolves Delta tables registered in a Hive Metastore."""

    name = "hive"

    @classmethod
    def from_uri(cls, uri: str | None, **kwargs: Any) -> HiveMetastoreCatalog:
        """Build from ``hms://host:9083`` or ``hms://thrift://host:9083``."""
        # `host` and `token` are the Databricks options `connect()` always
        # passes (usually None); the metastore's host comes from the URI.
        host = kwargs.pop("host", None)
        for unused in ("profile", "config", "warehouse_id", "token"):
            kwargs.pop(unused, None)
        if uri and "://" not in uri and uri.lower() in ("hms", "hive"):
            uri = None  # a bare scheme names the catalog kind, not a host
        if uri and "://" in uri and not uri.split("://", 1)[1].strip("/"):
            uri = None  # `hms://` with no endpoint
        if uri is None and host:
            return cls(host=host, **kwargs)
        return cls(uri=uri, **kwargs)

    def __init__(self, uri: str | None = None, *, host: str | None = None, port: int = 9083):
        if uri:
            host_part, port = _parse_endpoint(uri, port)
            host = host or host_part
        if not host:
            raise InvalidReferenceError("a Hive Metastore host is required")
        self._host = host
        self._port = int(port)

    def _check_endpoint(self, ref: TableRef) -> None:
        """Refuse a reference that names a different metastore.

        ``hms://other-host:9083/db/t`` used to be read from this connection's
        metastore, silently returning a same-named table from the wrong one.
        Connecting to whatever host a table name carries would be the other
        option; refusing is safer. The comparison ignores host case, a
        ``thrift://`` prefix and an omitted default port (9083).
        """
        if not ref.endpoint:
            return
        host, port = _parse_endpoint(ref.endpoint, 9083)
        if host and (host.lower(), port) != (self._host.lower(), self._port):
            raise InvalidReferenceError(
                f"{ref.raw or ref} names the Hive Metastore at {host}:{port}, but this "
                f"connection is to {self._host}:{self._port}. Connect to that metastore "
                f"(ds.connect('hms://{host}:{port}')) to read it."
            )

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        """One Thrift connection, closed on exit.

        Opened per operation, not cached: pymetastore's connection object
        keeps a single transport, so two threads sharing it overwrote each
        other's socket and one's close() cut the other off mid-call.
        """
        host, port = self._host, self._port
        cls = _hms_class()
        try:
            connection = cls.create(host=host, port=port)
            hms = connection.__enter__()
        except PreflightError:
            raise
        except Exception as exc:
            raise PreflightError(
                f"could not connect to the Hive Metastore at {host}:{port}: {exc}"
            ) from exc
        try:
            yield hms
        except BaseException as exc:
            if not connection.__exit__(type(exc), exc, exc.__traceback__):
                raise
        else:
            connection.__exit__(None, None, None)

    @staticmethod
    def _get_table(hms: Any, schema: str, table: str) -> dict[str, Any]:
        """The fields we use, from the raw Thrift table where possible.

        pymetastore's own `get_table` parses every column type and raises
        TypeError on shapes it does not expect (a table without SerDe
        parameters, a type its parser does not know). We only need the
        parameters and the location, so the raw Thrift struct is safer.
        """
        try:
            client = getattr(hms, "client", None)
            if client is not None and hasattr(client, "get_table"):
                raw = client.get_table(schema, table)
                sd = getattr(raw, "sd", None)
                serde = getattr(sd, "serdeInfo", None)
                return {
                    "parameters": getattr(raw, "parameters", None),
                    "location": getattr(sd, "location", None),
                    "serde_parameters": getattr(serde, "parameters", None),
                    "table_type": getattr(raw, "tableType", None),
                    "view_text": getattr(raw, "viewOriginalText", None),
                }
            parsed = hms.get_table(schema, table)
        except Exception as exc:
            if type(exc).__name__ in ("NoSuchObjectException", "UnknownTableException"):
                raise InvalidReferenceError(
                    f"{schema}.{table} does not exist in the Hive Metastore: "
                    f"{getattr(exc, 'message', None) or exc}"
                ) from exc
            raise
        storage = getattr(parsed, "storage", None)
        return {
            "parameters": getattr(parsed, "parameters", None),
            "location": getattr(storage, "location", None),
            "serde_parameters": getattr(storage, "serde_parameters", None),
            "table_type": getattr(parsed, "table_type", None),
            "view_text": getattr(parsed, "view_original_text", None),
        }

    def resolve(self, ref: TableRef) -> ResolvedTable:
        if ref.kind is not RefKind.CATALOG or not ref.schema or not ref.table:
            raise InvalidReferenceError(f"{ref} does not name a database and table")

        self._check_endpoint(ref)
        with self._connect() as hms:
            return self._resolved(ref, self._get_table(hms, ref.schema, ref.table))

    def _resolved(self, ref: TableRef, table: dict[str, Any]) -> ResolvedTable:
        params = {str(k): str(v) for k, v in (table.get("parameters") or {}).items()}
        raw_type = table.get("table_type")
        type_name = str(getattr(raw_type, "value", None) or raw_type or "").upper()
        if type_name in _VIEW_TYPES or table.get("view_text"):
            raise InvalidReferenceError(
                f"{ref.schema}.{ref.table} is a Hive Metastore view, not a Delta table"
            )
        if not self._looks_like_delta(params):
            raise InvalidReferenceError(
                f"{ref.schema}.{ref.table} is registered in the Hive Metastore but does not "
                f"advertise itself as Delta (parameters: {sorted(params)}). "
                "Reading it as Delta would be a guess."
            )

        location = table.get("location") or None
        if not location or str(location).rstrip("/").endswith(_PLACEHOLDER):
            serde = table.get("serde_parameters") or {}
            location = serde.get("path") or params.get("path") or None
        location = normalise_location(location)
        if not location:
            raise InvalidReferenceError(
                f"{ref.schema}.{ref.table} has no storage location in the Hive Metastore"
            )

        table_type = self._table_type(raw_type)
        if params.get("EXTERNAL", "").upper() == "TRUE":
            table_type = TableType.EXTERNAL
        if ref.scheme != "hms":
            # `hive_metastore.db.t` parses with the default (uc) scheme, which
            # the router takes for Databricks' legacy hive_metastore catalog and
            # refuses. This catalog resolved it, so it is a real HMS table.
            ref = dataclasses.replace(ref, scheme="hms")
        return ResolvedTable(
            ref=ref,
            location=location,
            # HMS distinguishes MANAGED/EXTERNAL; anything else we leave unset
            # rather than mis-declare.
            table_type=table_type,
            data_source_format="DELTA",
            properties=params,
        )

    @staticmethod
    def _looks_like_delta(params: dict[str, str]) -> bool:
        # Keys and values case-insensitively (Trino writes provider=DELTA), plus
        # the Delta Hive connector's storage handler.
        lowered = {str(k).lower(): str(v).strip().lower() for k, v in params.items()}
        if "io.delta.hive" in lowered.get("storage_handler", ""):
            return True
        return any(lowered.get(key.lower()) == value.lower() for key, value in _DELTA_MARKERS)

    @staticmethod
    def _table_type(raw: Any) -> TableType | None:
        value = getattr(raw, "value", None) or (str(raw) if raw else None)
        if not value:
            return None
        upper = str(value).upper()
        if upper in ("EXTERNAL_TABLE", "EXTERNAL"):
            return TableType.EXTERNAL
        if upper in ("MANAGED_TABLE", "MANAGED"):
            return TableType.MANAGED
        return None

    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]:
        out = []
        # One connection for the listing and every lookup, instead of one per table.
        with self._connect() as hms:
            try:
                names = hms.list_tables(schema)
            except Exception as exc:
                if type(exc).__name__ in ("NoSuchObjectException", "UnknownDBException"):
                    raise InvalidReferenceError(
                        f"database {schema} does not exist in the Hive Metastore"
                    ) from exc
                raise
            for name in names:
                ref = TableRef(
                    kind=RefKind.CATALOG,
                    catalog="hive_metastore",
                    schema=schema,
                    table=name,
                    scheme="hms",
                    raw=f"hive_metastore.{schema}.{name}",
                )
                try:
                    out.append(self._resolved(ref, self._get_table(hms, schema, name)))
                except InvalidReferenceError:
                    continue  # not a Delta table (or dropped meanwhile); skip it
        return out

    def list_catalogs(self) -> list[str]:
        raise NotImplementedError("HiveMetastoreCatalog has no catalog namespace to enumerate")

    def list_schemas(self, catalog: str) -> list[str]:
        raise NotImplementedError("HiveMetastoreCatalog does not implement schema discovery")

    def drop_table(self, ref: TableRef) -> None:
        raise NotImplementedError("HiveMetastoreCatalog does not implement dropping tables")
