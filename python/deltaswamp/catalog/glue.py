"""AWS Glue Data Catalog.

delta-rs compiles Glue support into its wheel but never exposes it: the public
`DeltaTable.from_data_catalog` wrapper was removed at 1.0 and only a private,
undocumented `RawDeltaTable.get_table_uri_from_data_catalog` survives. Rather
than depend on a private symbol, we read the location from Glue directly, which
is a handful of lines and stays working across delta-rs releases.
"""

from __future__ import annotations

import threading
import urllib.parse
from typing import Any

from ..errors import InvalidReferenceError, PreflightError
from ..identity import RefKind, TableRef
from .base import ResolvedTable, TableType

__all__ = ["GlueCatalog"]

#: Spark's HiveExternalCatalog stores a data source table that is not
#: Hive-compatible (every Delta table) with this suffix on
#: ``StorageDescriptor.Location`` and the real path in the SerDe ``path``.
_PLACEHOLDER = "-__PLACEHOLDER__"

#: A resource link is followed at most this many hops, so a cycle cannot loop.
_MAX_LINK_HOPS = 4


def normalise_location(location: str | None) -> str | None:
    """``s3a://`` / ``s3n://`` (Hadoop connector schemes) -> ``s3://``.

    Metastores fed by Spark on EMR or Hadoop record the Hadoop scheme; the
    object-store engines address the same bucket as ``s3://`` (and do not
    know ``s3n`` at all).
    """
    if not location:
        return None
    scheme, sep, rest = location.partition("://")
    if sep and scheme.lower() in ("s3a", "s3n"):
        return f"s3://{rest}"
    return location


def _storage_location(table: dict[str, Any]) -> str | None:
    """The table's real storage path, seeing through Spark's placeholder."""
    sd = table.get("StorageDescriptor") or {}
    location = sd.get("Location") or None
    serde_path = ((sd.get("SerdeInfo") or {}).get("Parameters") or {}).get("path")
    params = table.get("Parameters") or {}
    if not location or location.rstrip("/").endswith(_PLACEHOLDER):
        location = serde_path or params.get("path") or None
    return normalise_location(location)


_DELTA_MARKER_KEYS = ("spark.sql.sources.provider", "table_type", "classification")


def _is_delta(params: dict[str, str], table: dict[str, Any] | None = None) -> bool:
    """Whether any writer's Delta marker is present.

    Spark/Databricks/EMR set ``spark.sql.sources.provider=delta``, Athena
    ``table_type=DELTA``, the Glue crawler ``classification=delta`` (in the
    table's or the storage descriptor's parameters), and the Delta Hive
    connector an ``io.delta.hive`` input format / storage handler. Keys and
    values are compared case-insensitively: writers disagree on both.
    """
    sd = (table or {}).get("StorageDescriptor") or {}
    sources = [
        params,
        sd.get("Parameters") or {},
        (sd.get("SerdeInfo") or {}).get("Parameters") or {},
    ]
    for source in sources:
        lowered = {str(k).lower(): str(v).strip().lower() for k, v in source.items()}
        if any(lowered.get(key) == "delta" for key in _DELTA_MARKER_KEYS):
            return True
        if "io.delta.hive" in lowered.get("storage_handler", ""):
            return True
    return "io.delta.hive" in str(sd.get("InputFormat") or "").lower()


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str((response.get("Error") or {}).get("Code") or "")
    return ""


class GlueCatalog:
    """Resolves Delta tables registered in AWS Glue."""

    name = "glue"

    @classmethod
    def from_uri(cls, uri: str | None, **kwargs: Any) -> GlueCatalog:
        """Build from ``glue://``, ``glue://<catalog-id>`` or
        ``glue://<catalog-id>?region=<region>``."""
        catalog_id = kwargs.pop("catalog_id", None)
        region_name = kwargs.pop("region_name", None) or kwargs.pop("region", None)
        if uri and uri.lower().startswith("glue://"):
            body, _, query = uri[len("glue://") :].partition("?")
            catalog_id = catalog_id or (body.strip("/") or None)
            regions = urllib.parse.parse_qs(query).get("region") or []
            region_name = region_name or (regions[0] if regions else None)
        # Connection-wide options that belong to other catalogs. `token` is a
        # Databricks token that `connect()` always passes; Glue auth is boto3's.
        for unused in ("profile", "host", "config", "warehouse_id", "token"):
            kwargs.pop(unused, None)
        return cls(catalog_id=catalog_id, region_name=region_name, **kwargs)

    def __init__(self, *, region_name: str | None = None, catalog_id: str | None = None) -> None:
        self._region_name = region_name
        self._catalog_id = catalog_id
        # region -> boto3 client (a resource link may point at another region)
        self._clients: dict[str | None, Any] = {}
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        # A boto3 client does not pickle; each process builds its own.
        state = self.__dict__.copy()
        state["_clients"] = {}
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _glue(self, region_name: str | None = None) -> Any:
        region = region_name or self._region_name
        with self._lock:
            client = self._clients.get(region)
            if client is not None:
                return client
            try:
                import boto3
            except ImportError as exc:
                raise PreflightError(
                    "Glue support needs boto3; install 'deltaswamp[glue]'"
                ) from exc
            try:
                client = boto3.client("glue", region_name=region)
            except Exception as exc:
                if type(exc).__name__ == "NoRegionError":
                    raise PreflightError(
                        "no AWS region is configured for Glue: set AWS_REGION (or "
                        "AWS_DEFAULT_REGION), a region in ~/.aws/config, or connect with "
                        "glue://<catalog-id>?region=<region>"
                    ) from exc
                raise
            self._clients[region] = client
            return client

    def _kwargs(self, extra: dict[str, Any], catalog_id: str | None = None) -> dict[str, Any]:
        catalog_id = catalog_id or self._catalog_id
        if catalog_id:
            extra["CatalogId"] = catalog_id
        return extra

    def _catalog_id_for(self, ref: TableRef) -> str | None:
        if ref.endpoint:
            return ref.endpoint
        if self._catalog_id:
            return self._catalog_id
        # `123456789012.db.table`: a three-level name whose catalog is an AWS
        # account id names that account's Glue catalog.
        if ref.catalog and ref.catalog.isdigit() and len(ref.catalog) == 12:
            return ref.catalog
        return None

    @staticmethod
    def _call(what: str, fn: Any, **kwargs: Any) -> Any:
        """Call Glue, turning its errors into ones that say what to fix."""
        try:
            return fn(**kwargs)
        except Exception as exc:
            code = _error_code(exc)
            name = type(exc).__name__
            if "EntityNotFoundException" in (code, name):
                where = f" {kwargs['CatalogId']}" if kwargs.get("CatalogId") else ""
                raise InvalidReferenceError(
                    f"{what} does not exist in the Glue Data Catalog{where}: {exc}"
                ) from exc
            if code in ("AccessDeniedException", "AccessDenied") or name == (
                "AccessDeniedException"
            ):
                raise PreflightError(f"Glue denied access to {what}: {exc}") from exc
            if name in ("NoCredentialsError", "PartialCredentialsError"):
                raise PreflightError(
                    f"no AWS credentials were found to read {what} from Glue: {exc}"
                ) from exc
            raise

    def resolve(self, ref: TableRef) -> ResolvedTable:
        if ref.kind is not RefKind.CATALOG or not ref.schema or not ref.table:
            raise InvalidReferenceError(f"{ref} does not name a database and table")

        kwargs: dict[str, Any] = {"DatabaseName": ref.schema, "Name": ref.table}
        catalog_id = self._catalog_id_for(ref)
        if catalog_id:
            kwargs["CatalogId"] = catalog_id
        client = self._glue()
        table = self._call(f"Glue table {ref.schema}.{ref.table}", client.get_table, **kwargs)
        return self._resolved(ref, table["Table"])

    def _follow_link(self, ref: TableRef, table: dict[str, Any]) -> dict[str, Any]:
        """A Lake Formation resource link -> the table it points at."""
        for _ in range(_MAX_LINK_HOPS):
            target = table.get("TargetTable")
            if not target:
                return table
            kwargs: dict[str, Any] = {
                "DatabaseName": target["DatabaseName"],
                "Name": target["Name"],
            }
            if target.get("CatalogId"):
                kwargs["CatalogId"] = target["CatalogId"]
            client = self._glue(target.get("Region"))
            table = self._call(
                f"Glue table {target['DatabaseName']}.{target['Name']} "
                f"(the target of resource link {ref.schema}.{ref.table})",
                client.get_table,
                **kwargs,
            )["Table"]
        raise InvalidReferenceError(
            f"Glue table {ref.schema}.{ref.table} is a chain of more than "
            f"{_MAX_LINK_HOPS} resource links"
        )

    def _resolved(self, ref: TableRef, table: dict[str, Any]) -> ResolvedTable:
        table = self._follow_link(ref, table)
        params = {str(k): str(v) for k, v in (table.get("Parameters") or {}).items()}
        name = f"Glue table {ref.schema}.{ref.table}"
        glue_type = str(table.get("TableType") or "").upper()

        if glue_type == "VIRTUAL_VIEW" or table.get("ViewOriginalText"):
            raise InvalidReferenceError(f"{name} is a view, not a Delta table")
        if not _is_delta(params, table):
            fmt = (
                params.get("table_type")
                or params.get("spark.sql.sources.provider")
                or params.get("classification")
            )
            said = f"declares format {fmt!r}, not Delta" if fmt else "does not declare a format"
            raise InvalidReferenceError(
                f"{name} {said} (expected spark.sql.sources.provider=delta, table_type=DELTA "
                "or classification=delta in its parameters). Reading it as Delta would be "
                "a guess."
            )

        location = _storage_location(table)
        if not location:
            raise InvalidReferenceError(f"{name} has no storage location in Glue")

        table_type: TableType | None
        if glue_type == "EXTERNAL_TABLE":
            table_type = TableType.EXTERNAL
        elif glue_type == "MANAGED_TABLE":
            table_type = TableType.MANAGED
        else:
            table_type = None  # GOVERNED, or unset: leave it rather than mis-declare

        return ResolvedTable(
            ref=ref,
            location=location,
            table_type=table_type,
            data_source_format="DELTA",
            properties=params,
        )

    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]:
        probe = TableRef(kind=RefKind.CATALOG, catalog=catalog, schema=schema, table="_")
        catalog_id = self._catalog_id_for(probe)
        paginator = self._glue().get_paginator("get_tables")
        pages = self._call(
            f"Glue database {schema}",
            lambda: list(paginator.paginate(**self._kwargs({"DatabaseName": schema}, catalog_id))),
        )
        out: list[ResolvedTable] = []
        for page in pages:
            for table in page.get("TableList", []):
                ref = TableRef(
                    kind=RefKind.CATALOG,
                    catalog=catalog or "glue",
                    schema=schema,
                    table=table["Name"],
                    scheme="glue",
                    endpoint=catalog_id,
                    raw=f"{schema}.{table['Name']}",
                )
                try:
                    # GetTables already returned each table in full; resolving
                    # each again cost one GetTable call per table.
                    out.append(self._resolved(ref, table))
                except InvalidReferenceError:
                    continue
        return out

    def list_catalogs(self) -> list[str]:
        raise NotImplementedError("GlueCatalog has no catalog namespace to enumerate")

    def list_schemas(self, catalog: str) -> list[str]:
        raise NotImplementedError("GlueCatalog does not implement schema discovery")

    def drop_table(self, ref: TableRef) -> None:
        raise NotImplementedError("GlueCatalog does not implement dropping tables")
