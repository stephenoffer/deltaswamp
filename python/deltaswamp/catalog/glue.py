"""AWS Glue Data Catalog.

delta-rs compiles Glue support into its wheel but never exposes it: the public
`DeltaTable.from_data_catalog` wrapper was removed at 1.0 and only a private,
undocumented `RawDeltaTable.get_table_uri_from_data_catalog` survives. Rather
than depend on a private symbol, we read the location from Glue directly, which
is a handful of lines and stays working across delta-rs releases.
"""

from __future__ import annotations

from typing import Any

from ..errors import InvalidReferenceError, PreflightError
from ..identity import RefKind, TableRef
from .base import ResolvedTable, TableType

__all__ = ["GlueCatalog"]


class GlueCatalog:
    """Resolves Delta tables registered in AWS Glue."""

    name = "glue"

    @classmethod
    def from_uri(cls, uri: str | None, **kwargs: Any) -> GlueCatalog:
        """Build from ``glue://`` or ``glue://<catalog-id>``."""
        catalog_id = kwargs.pop("catalog_id", None)
        if uri and uri.startswith("glue://"):
            catalog_id = catalog_id or (uri[len("glue://") :] or None)
        for unused in ("profile", "host", "config", "warehouse_id"):
            kwargs.pop(unused, None)
        return cls(catalog_id=catalog_id, **kwargs)

    def __init__(self, *, region_name: str | None = None, catalog_id: str | None = None) -> None:
        self._region_name = region_name
        self._catalog_id = catalog_id
        self._client: Any = None

    def _glue(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise PreflightError(
                    "Glue support needs boto3; install 'deltaswamp[glue]'"
                ) from exc
            self._client = boto3.client("glue", region_name=self._region_name)
        return self._client

    def _kwargs(self, extra: dict[str, Any]) -> dict[str, Any]:
        catalog_id = self._catalog_id
        if catalog_id:
            extra["CatalogId"] = catalog_id
        return extra

    def resolve(self, ref: TableRef) -> ResolvedTable:
        if ref.kind is not RefKind.CATALOG or not ref.schema or not ref.table:
            raise InvalidReferenceError(f"{ref} does not name a database and table")

        catalog_id = ref.endpoint or self._catalog_id
        kwargs: dict[str, Any] = {"DatabaseName": ref.schema, "Name": ref.table}
        if catalog_id:
            kwargs["CatalogId"] = catalog_id

        table = self._glue().get_table(**kwargs)["Table"]
        params = dict(table.get("Parameters") or {})
        location = (table.get("StorageDescriptor") or {}).get("Location")

        provider = params.get("spark.sql.sources.provider", "").lower()
        if provider and provider != "delta" and params.get("table_type", "").upper() != "DELTA":
            raise InvalidReferenceError(
                f"Glue table {ref.schema}.{ref.table} declares provider {provider!r}, not Delta"
            )

        return ResolvedTable(
            ref=ref,
            location=location,
            table_type=(
                TableType.EXTERNAL
                if str(table.get("TableType", "")).upper() == "EXTERNAL_TABLE"
                else TableType.MANAGED
            ),
            data_source_format="DELTA",
            properties=params,
        )

    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]:
        paginator = self._glue().get_paginator("get_tables")
        out: list[ResolvedTable] = []
        for page in paginator.paginate(**self._kwargs({"DatabaseName": schema})):
            for table in page.get("TableList", []):
                ref = TableRef(
                    kind=RefKind.CATALOG,
                    catalog=catalog or "glue",
                    schema=schema,
                    table=table["Name"],
                    scheme="glue",
                    raw=f"{schema}.{table['Name']}",
                )
                try:
                    out.append(self.resolve(ref))
                except InvalidReferenceError:
                    continue
        return out

    def list_catalogs(self) -> list[str]:
        raise NotImplementedError("GlueCatalog has no catalog namespace to enumerate")

    def list_schemas(self, catalog: str) -> list[str]:
        raise NotImplementedError("GlueCatalog does not implement schema discovery")

    def drop_table(self, ref: TableRef) -> None:
        raise NotImplementedError("GlueCatalog does not implement dropping tables")
