"""Open-source Unity Catalog (unitycatalog.io).

Worth supporting on its own merits, but the decisive reason is testing: OSS UC
0.5+ implements the same `/delta/v1` commit API that delta-kernel-rs targets, so
the catalog-managed read and commit paths can be exercised in a container
without a Databricks account. It is the only way to put the headline feature
under CI.

Uses stdlib HTTP so the base install needs no extra dependency.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NoReturn, TypeVar

from ..credentials.base import Cloud, Credentials, Operation
from ..errors import CredentialError, InvalidReferenceError, PreflightError
from ..governance import (
    ColumnLineage,
    FunctionSummary,
    Grant,
    Lineage,
    StagingTable,
    TableInfo,
    TableSummary,
    Volume,
    VolumeSummary,
    create_table_body,
    delta_schema_to_columns,
    normalize_privilege,
    path_operation,
)
from ..identity import RefKind, TableRef
from .base import ResolvedTable, TableType, parse_commit_tail

__all__ = ["OSSUnityCatalog", "OSSUnityCredentialProvider", "UnityCatalogHTTPError"]

UC_API = "/api/2.1/unity-catalog"
UC_DELTA_API = f"{UC_API}/delta/v1"

_T = TypeVar("_T")

#: Governance methods whose OSS Unity Catalog endpoint does not exist, and why.
#: Each raises NotImplementedError with this text.
_UNSUPPORTED = {
    "effective_grants": "OSS Unity Catalog has no effective-permissions endpoint; "
    "grants() lists only direct grants",
    "tags": "OSS Unity Catalog has no tag (entity-tag-assignments) API",
    "set_tags": "OSS Unity Catalog has no tag (entity-tag-assignments) API",
    "unset_tags": "OSS Unity Catalog has no tag (entity-tag-assignments) API",
    "set_owner": "OSS Unity Catalog has no table update endpoint, so ownership cannot change",
    "lineage": "OSS Unity Catalog does not track lineage",
    "column_lineage": "OSS Unity Catalog does not track lineage",
    "add_primary_key": "OSS Unity Catalog has no table-constraints API",
    "add_foreign_key": "OSS Unity Catalog has no table-constraints API",
    "drop_table_constraint": "OSS Unity Catalog has no table-constraints API",
    "volume": "OSS Unity Catalog has no Files API; read a volume's storage_location with "
    "credentials from its temporary-volume-credentials endpoint instead",
}

_NEEDS = {
    "read": "Reading needs USE CATALOG and USE SCHEMA on the parents and SELECT or ownership.",
    "grant": "Changing grants needs ownership of the securable.",
    "catalog": "Creating a catalog needs CREATE CATALOG on the metastore.",
    "schema": "Creating a schema needs CREATE SCHEMA and USE CATALOG.",
    "volume": "Creating a volume needs CREATE VOLUME, USE SCHEMA and USE CATALOG.",
    "register": "Registering a table needs CREATE TABLE and USE SCHEMA on the schema, "
    "USE CATALOG, and EXTERNAL USE SCHEMA where the server enforces it.",
    "path": "Path credentials need access to an external location covering the path.",
    "staging": "Creating a managed table needs CREATE TABLE and USE SCHEMA on the schema "
    "and USE CATALOG; the server must be UC 0.5+ with managed tables enabled.",
}


def _dotted(ref: TableRef) -> str:
    if ref.kind is not RefKind.CATALOG or not (ref.catalog and ref.schema and ref.table):
        raise InvalidReferenceError(f"{ref} is not a catalog.schema.table reference")
    return f"{ref.catalog}.{ref.schema}.{ref.table}"


def _q(part: str) -> str:
    return urllib.parse.quote(part, safe="")


def _like(pattern: str | None) -> re.Pattern[str] | None:
    """SQL LIKE (``%``, ``_``) as a regex, matching how Databricks filters names."""
    if pattern is None:
        return None
    body = "".join(".*" if ch == "%" else "." if ch == "_" else re.escape(ch) for ch in pattern)
    return re.compile(f"^{body}$", re.IGNORECASE)


def _unsupported(method: str) -> NoReturn:
    raise NotImplementedError(f"{method}: {_UNSUPPORTED[method]}")


class UnityCatalogHTTPError(PreflightError):
    """A non-2xx answer from Unity Catalog, carrying the status code."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def _request(
    base_url: str,
    path: str,
    token: str | None,
    *,
    method: str = "GET",
    body: Any = None,
) -> Any:
    """One JSON request. `body`, when given, is sent as JSON."""
    url = base_url.rstrip("/") + path
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if not url.startswith(("http://", "https://")):
        raise InvalidReferenceError(f"refusing non-HTTP catalog URL {url!r}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise UnityCatalogHTTPError(
            f"{method} {url} failed with HTTP {exc.code}: {detail}", exc.code
        ) from exc
    except urllib.error.URLError as exc:
        raise PreflightError(f"{method} {url} failed: {exc.reason}") from exc
    return json.loads(text) if text else {}


def _parse_vended(body: dict[str, Any]) -> Credentials:
    """Parse a UC temporary-credentials response (table or path)."""
    url = body.get("url") or ""
    expiry = body.get("expiration_time") or body.get("expirationTime")
    expires_at = float(expiry) / 1000.0 if expiry else None

    if "aws_temp_credentials" in body:
        c = body["aws_temp_credentials"]
        return Credentials(
            cloud=Cloud.AWS,
            url=url,
            expires_at=expires_at,
            secrets={
                "aws_access_key_id": c["access_key_id"],
                "aws_secret_access_key": c["secret_access_key"],
                "aws_session_token": c["session_token"],
            },
            scope_prefix=url or None,
        )
    if "gcp_oauth_token" in body:
        return Credentials(
            cloud=Cloud.GCP,
            url=url,
            expires_at=expires_at,
            secrets={"google_bearer_token": body["gcp_oauth_token"]["oauth_token"]},
            scope_prefix=url or None,
        )
    if "azure_user_delegation_sas" in body:
        from ..credentials.databricks import azure_endpoint_for

        endpoint = azure_endpoint_for(url)
        secrets = {"azure_storage_sas_key": body["azure_user_delegation_sas"]["sas_token"]}
        if endpoint:
            secrets["azure_endpoint"] = endpoint
        return Credentials(
            cloud=Cloud.AZURE,
            url=url,
            expires_at=expires_at,
            secrets=secrets,
            scope_prefix=url or None,
        )
    raise CredentialError(
        f"OSS Unity Catalog returned no recognised credential block: {sorted(body)}"
    )


class OSSUnityCredentialProvider:
    """Vends credentials from the OSS UC Delta API.

    OSS UC omits `expirationTime` from its load-table response and has no
    `loadCredentials` endpoint (unitycatalog#1885), so there is often no expiry
    to honour. We treat a missing expiry as "no stated deadline" rather than
    inventing one, and re-vend on demand via `invalidate()`.
    """

    def __init__(
        self,
        base_url: str,
        ref: TableRef,
        token: str | None = None,
        table_id: str | None = None,
    ) -> None:
        self._table_id = table_id
        self._base_url = base_url
        self._catalog = ref.catalog
        self._schema = ref.schema
        self._table = ref.table
        self._token = token
        self._cache: dict[Operation, Credentials] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    @property
    def table_id(self) -> str | None:
        return self._table_id

    def workspace_auth(self) -> tuple[str, str]:
        """`(base_url, token)` for the commit API.

        Without this, every catalog-managed write against an open-source Unity
        Catalog server failed: the kernel committer asks the provider for the
        endpoint and bearer token, and this class had no way to answer.
        """
        return self._base_url.rstrip("/"), self._token or ""

    def invalidate(self) -> None:
        self._cache.clear()

    def credentials(self, operation: Operation = Operation.READ) -> Credentials:
        cached = self._cache.get(operation)
        if cached is not None and not cached.expires_within():
            return cached
        path = (
            f"{UC_DELTA_API}/catalogs/{self._catalog}/schemas/{self._schema}"
            f"/tables/{self._table}/credentials?operation={operation.value}"
        )
        body = _request(self._base_url, path, self._token)
        creds = self._parse(body)
        self._cache[operation] = creds
        return creds

    def _parse(self, body: dict[str, Any]) -> Credentials:
        return _parse_vended(body)


class OSSUnityCatalog:
    """Resolves tables in an open-source Unity Catalog server."""

    name = "unity"

    @classmethod
    def from_uri(cls, uri: str | None, **kwargs: Any) -> OSSUnityCatalog:
        """Build from ``uc://http://host:8080`` or ``unity://host:8080``."""
        if not uri:
            raise InvalidReferenceError("the unity catalog needs a server URL")
        base_url = uri.split("://", 1)[1]
        # Drop kwargs meant for Databricks auth; they are meaningless here.
        for unused in ("profile", "host", "config"):
            kwargs.pop(unused, None)
        return cls(base_url=base_url, **kwargs)

    def __init__(self, base_url: str, token: str | None = None) -> None:
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        self._base_url = base_url
        self._token = token

    def resolve(self, ref: TableRef) -> ResolvedTable:
        if ref.kind is not RefKind.CATALOG:
            raise InvalidReferenceError(f"{ref} is a path; use the filesystem catalog")
        assert ref.catalog and ref.schema and ref.table

        full = urllib.parse.quote(f"{ref.catalog}.{ref.schema}.{ref.table}", safe="")
        info = _request(self._base_url, f"{UC_API}/tables/{full}", self._token)

        properties = dict(info.get("properties") or {})
        resolved = ResolvedTable(
            ref=ref,
            location=info.get("storage_location"),
            table_type=self._table_type(info.get("table_type")),
            data_source_format=info.get("data_source_format"),
            table_id=info.get("table_id"),
            properties=properties,
            credential_provider=OSSUnityCredentialProvider(
                self._base_url, ref, self._token, info.get("table_id")
            ),
            table_uuid=info.get("table_id"),
        )

        # Every Unity Catalog table is reachable through the catalog's Iceberg
        # REST endpoint when it has Iceberg metadata (managed Iceberg, foreign
        # Iceberg, UniForm); the Iceberg engine decides whether it applies.
        import dataclasses as _dc

        from ..engine.iceberg import iceberg_rest_uri

        resolved = _dc.replace(
            resolved, iceberg_rest_uri=iceberg_rest_uri(self._base_url, databricks=False)
        )

        if resolved.is_catalog_managed:
            resolved = self._with_catalog_commits(resolved, ref)
        return resolved

    def _with_catalog_commits(self, resolved: ResolvedTable, ref: TableRef) -> ResolvedTable:
        import dataclasses

        path = f"{UC_DELTA_API}/catalogs/{ref.catalog}/schemas/{ref.schema}/tables/{ref.table}"
        body = _request(self._base_url, path, self._token)
        entries, latest, location = parse_commit_tail(body, resolved.location)
        return dataclasses.replace(
            resolved,
            log_tail=entries,
            max_catalog_version=latest,
            location=location,
        )

    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]:
        query = urllib.parse.urlencode({"catalog_name": catalog, "schema_name": schema})
        body = _request(self._base_url, f"{UC_API}/tables?{query}", self._token)
        out = []
        for info in body.get("tables", []):
            ref = TableRef(
                kind=RefKind.CATALOG,
                catalog=catalog,
                schema=schema,
                table=info["name"],
                scheme="uc",
                raw=f"{catalog}.{schema}.{info['name']}",
            )
            out.append(
                ResolvedTable(
                    ref=ref,
                    location=info.get("storage_location"),
                    table_type=self._table_type(info.get("table_type")),
                    data_source_format=info.get("data_source_format"),
                    table_id=info.get("table_id"),
                    properties=dict(info.get("properties") or {}),
                    credential_provider=OSSUnityCredentialProvider(
                        self._base_url, ref, self._token
                    ),
                )
            )
        return out

    def list_catalogs(self) -> list[str]:
        body = _request(self._base_url, f"{UC_API}/catalogs", self._token)
        return [c["name"] for c in body.get("catalogs", []) if c.get("name")]

    def list_schemas(self, catalog: str) -> list[str]:
        query = urllib.parse.urlencode({"catalog_name": catalog})
        body = _request(self._base_url, f"{UC_API}/schemas?{query}", self._token)
        return [s["name"] for s in body.get("schemas", []) if s.get("name")]

    def drop_table(self, ref: TableRef) -> None:
        # _dotted, not an f-string: it rejects a reference that is not
        # catalog.schema.table. Interpolating directly turned a path reference
        # into a DELETE of a table literally named "None.None.None".
        full = _q(_dotted(ref))
        _request(self._base_url, f"{UC_API}/tables/{full}", self._token, method="DELETE")

    @staticmethod
    def _table_type(raw: str | None) -> TableType | None:
        if not raw:
            return None
        try:
            return TableType(raw)
        except ValueError:
            return None

    def preflight(self) -> list[str]:
        try:
            _request(self._base_url, f"{UC_DELTA_API}/config", self._token)
        except PreflightError as exc:
            return [
                f"the UC Delta API is not reachable at {self._base_url}{UC_DELTA_API}/config "
                f"({exc}). Catalog-managed tables need it; UC 0.5+ is required."
            ]
        return []

    # =============================================================== governance

    #: Methods that raise NotImplementedError against this server.
    unsupported_operations: frozenset[str] = frozenset(_UNSUPPORTED)

    def _get(self, path: str) -> Any:
        return _request(self._base_url, path, self._token)

    def _send(self, method: str, path: str, body: Any = None) -> Any:
        return _request(self._base_url, path, self._token, method=method, body=body)

    def _call(self, action: str, name: str, needs: str, fn: Callable[[], _T]) -> _T:
        """Run a request, naming the missing privilege on a 403."""
        try:
            return fn()
        except UnityCatalogHTTPError as exc:
            if exc.status == 404:
                raise InvalidReferenceError(
                    f"cannot {action}: {name} does not exist in Unity Catalog. "
                    f"Underlying error: {exc}"
                ) from exc
            if exc.status in (401, 403):
                raise PreflightError(
                    f"cannot {action}: access to {name} was denied. {_NEEDS[needs]} "
                    f"Underlying error: {exc}"
                ) from exc
            raise

    def table_info(self, ref: TableRef) -> TableInfo:
        name = _dotted(ref)
        return TableInfo.from_api(
            self._call(
                "read table metadata",
                name,
                "read",
                lambda: self._get(f"{UC_API}/tables/{_q(name)}"),
            )
        )

    # ------------------------------------------------------------ permissions

    @staticmethod
    def _permissions_path(target: TableRef | str, securable_type: str) -> tuple[str, str]:
        name = _dotted(target) if isinstance(target, TableRef) else str(target)
        # OSS spells securable types in lower case in the path.
        return name, f"{UC_API}/permissions/{_q(securable_type.lower())}/{_q(name)}"

    @staticmethod
    def _from_wire(body: Any) -> list[Grant]:
        # Grant.list_from_api normalises OSS's "USE SCHEMA" to "USE_SCHEMA".
        return Grant.list_from_api(body)

    def grants(
        self,
        target: TableRef | str,
        principal: str | None = None,
        *,
        securable_type: str = "TABLE",
    ) -> list[Grant]:
        name, path = self._permissions_path(target, securable_type)
        if principal:
            path += "?" + urllib.parse.urlencode({"principal": principal})
        return self._from_wire(self._call("read grants", name, "read", lambda: self._get(path)))

    def effective_grants(
        self,
        target: TableRef | str,
        principal: str | None = None,
        *,
        securable_type: str = "TABLE",
    ) -> list[Grant]:
        _unsupported("effective_grants")

    def _change(
        self,
        target: TableRef | str,
        principal: str,
        privileges: Iterable[str],
        securable_type: str,
        *,
        add: bool,
    ) -> list[Grant]:
        # OSS spells privileges with spaces: "USE SCHEMA", not "USE_SCHEMA".
        wire = [normalize_privilege(p).replace("_", " ") for p in privileges]
        if not wire:
            raise InvalidReferenceError("grant/revoke needs at least one privilege")
        name, path = self._permissions_path(target, securable_type)
        change = {"principal": principal, "add" if add else "remove": wire}
        body = self._call(
            "grant privileges" if add else "revoke privileges",
            name,
            "grant",
            lambda: self._send("PATCH", path, {"changes": [change]}),
        )
        return self._from_wire(body)

    def grant(
        self,
        target: TableRef | str,
        principal: str,
        privileges: Iterable[str],
        *,
        securable_type: str = "TABLE",
    ) -> list[Grant]:
        return self._change(target, principal, privileges, securable_type, add=True)

    def revoke(
        self,
        target: TableRef | str,
        principal: str,
        privileges: Iterable[str],
        *,
        securable_type: str = "TABLE",
    ) -> list[Grant]:
        return self._change(target, principal, privileges, securable_type, add=False)

    # --------------------------------------------- what the OSS server lacks

    def tags(self, ref: TableRef, column: str | None = None) -> dict[str, str]:
        _unsupported("tags")

    def set_tags(self, ref: TableRef, tags: Mapping[str, str], column: str | None = None) -> None:
        _unsupported("set_tags")

    def unset_tags(self, ref: TableRef, keys: Iterable[str], column: str | None = None) -> None:
        _unsupported("unset_tags")

    def set_owner(self, ref: TableRef, principal: str) -> None:
        _unsupported("set_owner")

    def lineage(self, ref: TableRef, direction: str = "both") -> Lineage:
        _unsupported("lineage")

    def column_lineage(self, ref: TableRef, column: str, direction: str = "both") -> ColumnLineage:
        _unsupported("column_lineage")

    def add_primary_key(
        self, ref: TableRef, name: str, columns: Iterable[str], *, rely: bool = False
    ) -> None:
        _unsupported("add_primary_key")

    def add_foreign_key(
        self,
        ref: TableRef,
        name: str,
        columns: Iterable[str],
        parent_ref: TableRef,
        parent_columns: Iterable[str],
        *,
        rely: bool = False,
    ) -> None:
        _unsupported("add_foreign_key")

    def drop_table_constraint(self, ref: TableRef, name: str, *, cascade: bool = False) -> None:
        _unsupported("drop_table_constraint")

    def volume(self, ref: TableRef | str) -> Volume:
        _unsupported("volume")

    # ============================================================= namespaces

    def create_catalog(
        self, name: str, comment: str | None = None, storage_root: str | None = None
    ) -> None:
        body: dict[str, Any] = {"name": name}
        if comment is not None:
            body["comment"] = comment
        if storage_root is not None:
            body["storage_root"] = storage_root
        self._call(
            "create a catalog",
            name,
            "catalog",
            lambda: self._send("POST", f"{UC_API}/catalogs", body),
        )

    def drop_catalog(self, name: str, force: bool = False) -> None:
        query = "?force=true" if force else ""
        self._call(
            "drop a catalog",
            name,
            "catalog",
            lambda: self._send("DELETE", f"{UC_API}/catalogs/{_q(name)}{query}"),
        )

    def create_schema(
        self,
        catalog: str,
        name: str,
        comment: str | None = None,
        storage_root: str | None = None,
    ) -> None:
        body: dict[str, Any] = {"name": name, "catalog_name": catalog}
        if comment is not None:
            body["comment"] = comment
        if storage_root is not None:
            body["storage_root"] = storage_root
        self._call(
            "create a schema",
            f"{catalog}.{name}",
            "schema",
            lambda: self._send("POST", f"{UC_API}/schemas", body),
        )

    def drop_schema(self, catalog: str, name: str, force: bool = False) -> None:
        query = "?force=true" if force else ""
        full = f"{catalog}.{name}"
        self._call(
            "drop a schema",
            full,
            "schema",
            lambda: self._send("DELETE", f"{UC_API}/schemas/{_q(full)}{query}"),
        )

    def table_exists(self, ref: TableRef) -> bool:
        name = _dotted(ref)
        try:
            self._get(f"{UC_API}/tables/{_q(name)}")
        except UnityCatalogHTTPError as exc:
            if exc.status == 404:
                return False
            raise
        return True

    def _paged(self, path: str, key: str, params: dict[str, str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        token: str | None = None
        while True:
            query = dict(params)
            if token:
                query["page_token"] = token
            body = self._get(f"{path}?{urllib.parse.urlencode(query)}")
            out.extend(body.get(key) or [])
            token = body.get("next_page_token")
            if not token:
                return out

    def search_tables(
        self,
        catalog: str,
        schema_pattern: str | None = None,
        table_pattern: str | None = None,
    ) -> list[TableSummary]:
        """Emulated: OSS has no summaries endpoint, so this lists per schema."""
        schemas, tables = _like(schema_pattern), _like(table_pattern)
        out: list[TableSummary] = []
        for schema in self.list_schemas(catalog):
            if schemas is not None and not schemas.match(schema):
                continue
            for info in self._paged(
                f"{UC_API}/tables", "tables", {"catalog_name": catalog, "schema_name": schema}
            ):
                if tables is None or tables.match(str(info.get("name") or "")):
                    out.append(TableSummary.from_api({"catalog_name": catalog, **info}))
        return out

    def list_functions(self, catalog: str, schema: str) -> list[FunctionSummary]:
        return [
            FunctionSummary.from_api(f)
            for f in self._call(
                "list functions",
                f"{catalog}.{schema}",
                "read",
                lambda: self._paged(
                    f"{UC_API}/functions",
                    "functions",
                    {"catalog_name": catalog, "schema_name": schema},
                ),
            )
        ]

    def list_volumes(self, catalog: str, schema: str) -> list[VolumeSummary]:
        return [
            VolumeSummary.from_api(v)
            for v in self._call(
                "list volumes",
                f"{catalog}.{schema}",
                "read",
                lambda: self._paged(
                    f"{UC_API}/volumes", "volumes", {"catalog_name": catalog, "schema_name": schema}
                ),
            )
        ]

    def create_volume(
        self,
        catalog: str,
        schema: str,
        name: str,
        volume_type: str = "MANAGED",
        storage_location: str | None = None,
        comment: str | None = None,
    ) -> VolumeSummary:
        kind = volume_type.upper()
        if kind == "EXTERNAL" and not storage_location:
            raise InvalidReferenceError("an EXTERNAL volume needs a storage_location")
        body: dict[str, Any] = {
            "catalog_name": catalog,
            "schema_name": schema,
            "name": name,
            "volume_type": kind,
        }
        if storage_location is not None:
            body["storage_location"] = storage_location
        if comment is not None:
            body["comment"] = comment
        info = self._call(
            "create a volume",
            f"{catalog}.{schema}.{name}",
            "volume",
            lambda: self._send("POST", f"{UC_API}/volumes", body),
        )
        return VolumeSummary.from_api(info)

    def drop_volume(self, catalog: str, schema: str, name: str) -> None:
        full = f"{catalog}.{schema}.{name}"
        self._call(
            "drop a volume",
            full,
            "volume",
            lambda: self._send("DELETE", f"{UC_API}/volumes/{_q(full)}"),
        )

    # ============================================================== lifecycle

    def register_table(
        self,
        ref: TableRef,
        location: str,
        *,
        columns_schema_json: str | Mapping[str, Any] | None = None,
        partition_columns: Iterable[str] | None = None,
        properties: Mapping[str, str] | None = None,
        comment: str | None = None,
    ) -> ResolvedTable:
        """Register an existing Delta log at `location` as an EXTERNAL table."""
        name = _dotted(ref)
        partitions = list(partition_columns or ())
        if partitions and columns_schema_json is None:
            raise InvalidReferenceError(
                "partition columns are recorded on the column list, so registering a "
                "partitioned table needs columns_schema_json"
            )
        body: dict[str, Any] = {
            "name": ref.table,
            "catalog_name": ref.catalog,
            "schema_name": ref.schema,
            "table_type": "EXTERNAL",
            "data_source_format": "DELTA",
            "storage_location": location,
            "columns": (
                delta_schema_to_columns(columns_schema_json, partitions)
                if columns_schema_json is not None
                else []
            ),
            "properties": dict(properties or {}),
        }
        if comment is not None:
            body["comment"] = comment
        self._call(
            "register an external table",
            name,
            "register",
            lambda: self._send("POST", f"{UC_API}/tables", body),
        )
        return self.resolve(ref)

    def path_credentials(self, url: str, operation: str = "PATH_READ") -> Credentials:
        op = path_operation(operation)
        body = self._call(
            f"vend {op} credentials",
            url,
            "path",
            lambda: self._send(
                "POST", f"{UC_API}/temporary-path-credentials", {"url": url, "operation": op}
            ),
        )
        body.setdefault("url", url)
        return _parse_vended(body)

    def _delta_tables_path(self, ref: TableRef, leaf: str) -> str:
        assert ref.catalog and ref.schema
        return f"{UC_DELTA_API}/catalogs/{_q(ref.catalog)}/schemas/{_q(ref.schema)}/{leaf}"

    def create_staging_table(self, ref: TableRef) -> StagingTable:
        name = _dotted(ref)
        body = self._call(
            "create a staging table",
            name,
            "staging",
            lambda: self._send(
                "POST", self._delta_tables_path(ref, "staging-tables"), {"name": ref.table}
            ),
        )
        return StagingTable.from_api(name, body)

    def finalize_managed_table(
        self, ref: TableRef, request_body: Mapping[str, Any]
    ) -> ResolvedTable:
        name = _dotted(ref)
        body = create_table_body(ref, request_body)
        self._call(
            "finalize a managed table",
            name,
            "staging",
            lambda: self._send("POST", self._delta_tables_path(ref, "tables"), body),
        )
        return self.resolve(ref)
