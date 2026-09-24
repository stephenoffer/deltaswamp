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

from typing import Any

from ..errors import InvalidReferenceError, PreflightError
from ..identity import RefKind, TableRef
from .base import ResolvedTable, TableType

__all__ = ["HiveMetastoreCatalog"]

# How a Delta table announces itself in HMS table parameters.
_DELTA_MARKERS = (
    ("spark.sql.sources.provider", "delta"),
    ("table_type", "DELTA"),
)


class HiveMetastoreCatalog:
    """Resolves Delta tables registered in a Hive Metastore."""

    name = "hive"

    @classmethod
    def from_uri(cls, uri: str | None, **kwargs: Any) -> HiveMetastoreCatalog:
        """Build from ``hms://host:9083`` or ``hms://thrift://host:9083``."""
        for unused in ("profile", "host", "config", "warehouse_id"):
            kwargs.pop(unused, None)
        return cls(uri=uri, **kwargs)

    def __init__(self, uri: str | None = None, *, host: str | None = None, port: int = 9083):
        if uri:
            # hms://thrift://host:9083 or hms://host:9083
            body = uri.split("://", 1)[1] if "://" in uri else uri
            body = body.removeprefix("thrift://")
            body = body.split("/", 1)[0]
            host_part, _, port_part = body.partition(":")
            host = host or host_part
            port = int(port_part) if port_part else port
        if not host:
            raise InvalidReferenceError("a Hive Metastore host is required")
        self._host = host
        self._port = port
        self._client: Any = None

    def _hms(self) -> Any:
        if self._client is None:
            try:
                from pymetastore.hms import HMS
            except ImportError as exc:
                raise PreflightError(
                    "Hive Metastore support needs the pymetastore package; "
                    "install 'deltaswamp[hms]'. (delta-rs has no HMS catalog crate at all, "
                    "so there is no built-in alternative.)"
                ) from exc
            self._client = HMS.create(host=self._host, port=self._port)
        return self._client

    def resolve(self, ref: TableRef) -> ResolvedTable:
        if ref.kind is not RefKind.CATALOG or not ref.schema or not ref.table:
            raise InvalidReferenceError(f"{ref} does not name a database and table")

        with self._hms() as hms:
            table = hms.get_table(ref.schema, ref.table)

        params = dict(getattr(table, "parameters", None) or {})
        if not self._looks_like_delta(params):
            raise InvalidReferenceError(
                f"{ref.schema}.{ref.table} is registered in the Hive Metastore but does not "
                f"advertise itself as Delta (parameters: {sorted(params)}). "
                "Reading it as Delta would be a guess."
            )

        location = getattr(getattr(table, "storage", None), "location", None)
        return ResolvedTable(
            ref=ref,
            location=location,
            # HMS distinguishes MANAGED/EXTERNAL; anything else we leave unset
            # rather than mis-declare.
            table_type=self._table_type(getattr(table, "table_type", None)),
            data_source_format="DELTA",
            properties=params,
        )

    @staticmethod
    def _looks_like_delta(params: dict[str, str]) -> bool:
        return any(params.get(key, "").lower() == value.lower() for key, value in _DELTA_MARKERS)

    @staticmethod
    def _table_type(raw: Any) -> TableType | None:
        value = getattr(raw, "value", None) or (str(raw) if raw else None)
        if not value:
            return None
        upper = value.upper()
        if "EXTERNAL" in upper:
            return TableType.EXTERNAL
        if "MANAGED" in upper:
            return TableType.MANAGED
        return None

    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]:
        with self._hms() as hms:
            names = hms.list_tables(schema)
        out = []
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
                out.append(self.resolve(ref))
            except InvalidReferenceError:
                continue  # not a Delta table; skip rather than fail the listing
        return out

    def list_catalogs(self) -> list[str]:
        raise NotImplementedError("HiveMetastoreCatalog has no catalog namespace to enumerate")

    def list_schemas(self, catalog: str) -> list[str]:
        raise NotImplementedError("HiveMetastoreCatalog does not implement schema discovery")

    def drop_table(self, ref: TableRef) -> None:
        raise NotImplementedError("HiveMetastoreCatalog does not implement dropping tables")
