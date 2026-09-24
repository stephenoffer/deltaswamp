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
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..credentials.base import Cloud, Credentials, Operation
from ..errors import CredentialError, InvalidReferenceError, PreflightError
from ..identity import RefKind, TableRef
from .base import LogTailEntry, ResolvedTable, TableType

__all__ = ["OSSUnityCatalog", "OSSUnityCredentialProvider"]

UC_API = "/api/2.1/unity-catalog"
UC_DELTA_API = f"{UC_API}/delta/v1"


def _request(base_url: str, path: str, token: str | None, *, method: str = "GET") -> Any:
    url = base_url.rstrip("/") + path
    request = urllib.request.Request(url, method=method)
    request.add_header("Accept", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if not url.startswith(("http://", "https://")):
        raise InvalidReferenceError(f"refusing non-HTTP catalog URL {url!r}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise PreflightError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise PreflightError(f"{method} {url} failed: {exc.reason}") from exc
    return json.loads(body) if body else {}


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

        if resolved.is_catalog_managed:
            resolved = self._with_catalog_commits(resolved, ref)
        return resolved

    def _with_catalog_commits(self, resolved: ResolvedTable, ref: TableRef) -> ResolvedTable:
        import dataclasses

        path = f"{UC_DELTA_API}/catalogs/{ref.catalog}/schemas/{ref.schema}/tables/{ref.table}"
        body = _request(self._base_url, path, self._token)
        commits = body.get("commits") or []
        entries = tuple(
            LogTailEntry(
                version=int(c["version"]),
                path=c.get("file_name") or c.get("fileName") or "",
                size=int(c.get("file_size") or c.get("fileSize") or 0),
                timestamp=int(c["timestamp"]) if c.get("timestamp") is not None else None,
            )
            for c in commits
        )
        latest = body.get("latest_table_version", body.get("latestTableVersion"))
        return dataclasses.replace(
            resolved,
            log_tail=entries,
            max_catalog_version=int(latest) if latest is not None else None,
            location=body.get("location") or resolved.location,
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
        full = urllib.parse.quote(f"{ref.catalog}.{ref.schema}.{ref.table}", safe="")
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
