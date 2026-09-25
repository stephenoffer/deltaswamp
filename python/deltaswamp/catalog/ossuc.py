"""Open-source Unity Catalog (unitycatalog.io).

Worth supporting on its own merits, but the decisive reason is testing: OSS UC
0.5+ implements the same `/delta/v1` commit API that delta-kernel-rs targets, so
the catalog-managed read and commit paths can be exercised in a container
without a Databricks account. It is the only way to put the headline feature
under CI.

Uses stdlib HTTP so the base install needs no extra dependency.
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NoReturn, TypeVar

from ..credentials.base import DEFAULT_REFRESH_MARGIN_SECONDS, Cloud, Credentials, Operation
from ..credentials.databricks import _expires_at_seconds, credentials_from_response
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
    "drop": "Dropping a table needs ownership of it, plus USE SCHEMA and USE CATALOG.",
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


def _prop(value: Any) -> str:
    """A table property value as UC stores it: a string, booleans lower-cased."""
    return str(value).lower() if isinstance(value, bool) else str(value)


def _unsupported(method: str) -> NoReturn:
    raise NotImplementedError(f"{method}: {_UNSUPPORTED[method]}")


class _Recreated(Exception):
    """The Delta API described a different table id than the tables API."""


class UnityCatalogHTTPError(PreflightError):
    """A non-2xx answer from Unity Catalog, carrying the status code."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status

    def __reduce__(self) -> tuple[object, ...]:
        # The default rebuilds from `args` alone and this __init__ also needs
        # `status`: raised while a Ray worker vended credentials, the error
        # failed to unpickle on the driver and became an unrelated TypeError.
        return (type(self), (str(self), self.status), self.__dict__)


def _request(
    base_url: str,
    path: str,
    token: str | None,
    *,
    method: str = "GET",
    body: Any = None,
    timeout: float = 30.0,
) -> Any:
    """One JSON request. `body`, when given, is sent as JSON."""
    url = base_url.rstrip("/") + path
    # Checked before anything is built: urllib would otherwise happily open a
    # file:// URL, or fail with an unhelpful "unknown url type".
    if not url.startswith(("http://", "https://")):
        raise InvalidReferenceError(f"refusing non-HTTP catalog URL {url!r}")
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        except Exception:  # an HTTPError need not carry a readable body
            detail = str(exc.reason)
        raise UnityCatalogHTTPError(
            f"{method} {url} failed with HTTP {exc.code}: {detail}", exc.code
        ) from exc
    except urllib.error.URLError as exc:
        raise PreflightError(f"{method} {url} failed: {exc.reason}") from exc
    except (TimeoutError, OSError, http.client.HTTPException) as exc:
        # A read timeout, a reset connection or a truncated body surfaces as
        # one of these, not URLError, and used to escape as a bare socket error.
        raise PreflightError(f"{method} {url} failed: {exc!r}") from exc
    if not text.strip():
        return {}
    try:
        return json.loads(text)
    except ValueError as exc:
        # A proxy's HTML error page or a truncated body: say what came back
        # instead of raising a JSONDecodeError with no URL in it.
        raise PreflightError(
            f"{method} {url} returned a non-JSON response: {text[:200]!r}"
        ) from exc


def _cloud_for(url: str) -> Cloud:
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme in ("s3", "s3a", "s3n"):
        return Cloud.AWS
    if scheme in ("abfs", "abfss", "wasb", "wasbs", "az", "adl"):
        return Cloud.AZURE
    if scheme in ("gs", "gcs"):
        return Cloud.GCP
    if scheme in ("", "file"):
        return Cloud.LOCAL
    raise CredentialError(f"cannot tell which cloud {url!r} is on")


_VENDED_BLOCKS = (
    "aws_temp_credentials",
    "r2_temp_credentials",
    "azure_user_delegation_sas",
    "azure_aad",
    "gcp_oauth_token",
)


def _parse_vended(
    body: dict[str, Any],
    fallback_url: str | None = None,
    table_id: str | None = None,
    operation: Operation | None = None,
) -> Credentials:
    """Parse a UC temporary-credentials response (table or path).

    The per-cloud blocks share `credentials_from_response` with Databricks.
    `fallback_url` is the table location, used when the response names no
    URL: without it the credential had no scope, so every table's credential
    shared one registry key and an Azure SAS scoped to one table path was
    reused for another.
    """
    if not isinstance(body, Mapping):
        raise CredentialError(
            f"OSS Unity Catalog returned a non-object credentials response: {body!r:.200}"
        )
    url = body.get("url") or fallback_url or ""
    expiry = body.get("expiration_time")
    if expiry is None:
        expiry = body.get("expirationTime")
    expires_at = _expires_at_seconds(expiry)
    if expiry not in (None, "") and expires_at is None:
        raise CredentialError(f"unparseable credential expiration_time {expiry!r}")

    # A server serializing every optional block explicitly sends the ones that
    # do not apply as null; those are skipped, not taken as the answer.
    if any(isinstance(body.get(name), Mapping) and body.get(name) for name in _VENDED_BLOCKS):
        # What it was vended for: StaticCredentialProvider refuses to hand a
        # read-only path credential to a write only when this says READ.
        return credentials_from_response(
            {**body, "url": url}, url=url, table_id=table_id, operation=operation
        )
    storage = body.get("storage-credentials") or body.get("storage_credentials")
    if isinstance(storage, Mapping):
        storage = [storage]
    if storage:
        # The /delta/v1 shape (as staging-tables answers): a list of
        # {prefix, operation, config, expiration-time-ms}.
        from ..governance import staging_storage_options

        location = url or str((storage[0] or {}).get("prefix") or "")
        options, stated = staging_storage_options(location, storage)
        return Credentials(
            cloud=_cloud_for(location),
            url=location,
            expires_at=stated if stated is not None else expires_at,
            secrets=options,
            scope_prefix=location or None,
            table_id=table_id,
            operation=operation,
        )
    raise CredentialError(
        f"OSS Unity Catalog returned no recognized credential block: {sorted(body)}"
    )


class OSSUnityCredentialProvider:
    """Vends credentials from the OSS UC Delta API.

    OSS UC omits `expirationTime` from its load-table response and has no
    `loadCredentials` endpoint (unitycatalog#1885), so there is often no expiry
    to honor. A missing expiry means no stated deadline; `invalidate()` forces a
    re-vend.
    """

    def __init__(
        self,
        base_url: str,
        ref: TableRef,
        token: str | None = None,
        table_id: str | None = None,
        location: str | None = None,
    ) -> None:
        if ref.kind is not RefKind.CATALOG or not (ref.catalog and ref.schema and ref.table):
            raise InvalidReferenceError(f"{ref} is not a catalog.schema.table reference")
        self._table_id = table_id
        self._location = location
        self._base_url = base_url
        self._catalog: str = ref.catalog
        self._schema: str = ref.schema
        self._table: str = ref.table
        self._token = token
        self._cache: dict[Operation, Credentials] = {}
        self._vended_at: dict[Operation, float] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = {}
        state["_vended_at"] = {}
        return state

    @property
    def table_id(self) -> str | None:
        return self._table_id

    def workspace_auth(self) -> tuple[str, str]:
        """`(base_url, token)` for the commit API."""
        return self._base_url.rstrip("/"), self._token or ""

    def invalidate(self) -> None:
        self._cache.clear()
        getattr(self, "_vended_at", {}).clear()

    def credentials(self, operation: Operation = Operation.READ) -> Credentials:
        try:
            # A plain "READ" string used to die on `.value`.
            operation = Operation(str(getattr(operation, "value", operation)).upper())
        except ValueError:
            raise CredentialError(
                f"credential operation must be READ or READ_WRITE, not {operation!r}"
            ) from None
        cached = self._cache.get(operation)
        if cached is not None and not cached.expires_within(self._margin(operation, cached)):
            return cached
        # Percent-encoded: a name with a space, '#', '?' or '/' otherwise
        # built a different URL (or a different table) entirely.
        query = urllib.parse.urlencode({"operation": operation.value})
        path = (
            f"{UC_DELTA_API}/catalogs/{_q(self._catalog)}/schemas/{_q(self._schema)}"
            f"/tables/{_q(self._table)}/credentials?{query}"
        )
        try:
            body = _request(self._base_url, path, self._token)
        except UnityCatalogHTTPError as exc:
            # Surfaced as a bare HTTP error (a PreflightError), so `except
            # CredentialError` missed it and nothing said what to do.
            table = f"{self._catalog}.{self._schema}.{self._table}"
            hint = {
                404: "the table no longer exists under this name (dropped, renamed, or "
                "re-created with a new id); re-resolve it",
                401: "the catalog token was rejected (expired or invalid)",
                403: f"the principal may not {operation.value} this table's storage (SELECT "
                "for READ, MODIFY for READ_WRITE, plus USE SCHEMA and USE CATALOG)",
            }.get(exc.status, "the server refused the request")
            raise CredentialError(
                f"credential vending failed for {table} ({operation.value}): {hint}. "
                f"Underlying error: {exc}"
            ) from exc
        creds = self._parse(body, operation)
        self._cache[operation] = creds
        vended = getattr(self, "_vended_at", None)
        if vended is None:
            vended = self._vended_at = {}
        vended[operation] = time.time()
        return creds

    def _margin(self, operation: Operation, cached: Credentials) -> float:
        """The refresh margin, capped at half the credential's real lifetime.

        A credential issued with less life than the 300s default margin was
        "expiring" the moment it arrived, so every call re-vended it -- two or
        three vends per append on a server with short-lived credentials.
        """
        vended_at = (getattr(self, "_vended_at", None) or {}).get(operation)
        if cached.expires_at is None or vended_at is None:
            return DEFAULT_REFRESH_MARGIN_SECONDS
        lifetime = max(0.0, float(cached.expires_at) - float(vended_at))
        return min(DEFAULT_REFRESH_MARGIN_SECONDS, lifetime / 2.0)

    def _parse(self, body: dict[str, Any], operation: Operation | None = None) -> Credentials:
        return _parse_vended(
            body,
            fallback_url=getattr(self, "_location", None),
            table_id=self._table_id,
            operation=operation,
        )


class OSSUnityCatalog:
    """Resolves tables in an open-source Unity Catalog server."""

    name = "unity"

    @classmethod
    def from_uri(cls, uri: str | None, **kwargs: Any) -> OSSUnityCatalog:
        """Build from ``uc://http://host:8080`` or ``unity://host:8080``."""
        if not uri:
            raise InvalidReferenceError("the unity catalog needs a server URL")
        # "unity://host:8080" -> "host:8080"; a bare "host:8080" (no scheme at
        # all) used to die with an IndexError.
        base_url = uri.split("://", 1)[1] if "://" in uri else uri
        if uri.lower().startswith(("http://", "https://")):
            # Already the server URL: stripping its scheme and re-adding
            # "http://" silently downgraded an https server to plain HTTP,
            # sending the bearer token in the clear.
            base_url = uri
        if not base_url.strip("/"):
            raise InvalidReferenceError(f"the unity catalog URL {uri!r} names no server")
        # Drop kwargs meant for Databricks auth; they are meaningless here.
        for unused in ("profile", "host", "config"):
            kwargs.pop(unused, None)
        return cls(base_url=base_url, **kwargs)

    def __init__(self, base_url: str, token: str | None = None) -> None:
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        self._base_url = base_url
        self._token = token

    def resolve(self, ref: TableRef, *, _retried: bool = False) -> ResolvedTable:
        if ref.kind is not RefKind.CATALOG:
            raise InvalidReferenceError(f"{ref} is a path; use the filesystem catalog")
        # _dotted, not an assert: under `python -O` the assert vanished and a
        # two-part name was looked up as "cat.schema.None".
        name = _dotted(ref)
        info = self._call(
            "resolve the table",
            name,
            "read",
            lambda: _request(self._base_url, f"{UC_API}/tables/{_q(name)}", self._token),
        )
        resolved = self._resolved_from_info(ref, info)

        # Every Unity Catalog table is reachable through the catalog's Iceberg
        # REST endpoint when it has Iceberg metadata (managed Iceberg, foreign
        # Iceberg, UniForm); the Iceberg engine decides whether it applies.
        from ..engine.iceberg import iceberg_rest_uri

        resolved = dataclasses.replace(
            resolved, iceberg_rest_uri=iceberg_rest_uri(self._base_url, databricks=False)
        )

        if resolved.is_catalog_managed:
            try:
                resolved = self._with_catalog_commits(resolved, ref)
            except _Recreated as exc:
                if _retried:
                    raise PreflightError(
                        f"{name} keeps changing identity while being resolved (the tables "
                        f"API and the Delta API name different table ids, now {exc}); "
                        "retry once it has settled"
                    ) from None
                return self.resolve(ref, _retried=True)
        return resolved

    def _resolved_from_info(self, ref: TableRef, info: Mapping[str, Any]) -> ResolvedTable:
        """One table-info response as a ResolvedTable (shared by resolve and list)."""
        table_id = info.get("table_id")
        location = info.get("storage_location")
        return ResolvedTable(
            ref=ref,
            location=location,
            table_type=self._table_type(info.get("table_type")),
            data_source_format=info.get("data_source_format"),
            table_id=table_id,
            # Values stringified: object_store / delta-rs take str -> str only.
            properties={
                str(k): str(v) for k, v in (info.get("properties") or {}).items() if v is not None
            },
            credential_provider=OSSUnityCredentialProvider(
                self._base_url, ref, self._token, table_id, location
            ),
            table_uuid=table_id,
        )

    def _with_catalog_commits(self, resolved: ResolvedTable, ref: TableRef) -> ResolvedTable:
        path = self._delta_tables_path(ref, f"tables/{_q(ref.table or '')}")
        try:
            body = self._call(
                "read the catalog commit tail",
                _dotted(ref),
                "read",
                lambda: _request(self._base_url, path, self._token),
            )
        except InvalidReferenceError as exc:
            # The table was found a moment ago through the tables API, so a
            # 404 here means the server has no UC Delta API, not that the
            # table is missing.
            raise PreflightError(
                f"{_dotted(ref)} is catalog-managed, but {self._base_url}{UC_DELTA_API} "
                "did not serve its commit tail; catalog-managed tables need Unity Catalog "
                f"0.5+ with the Delta API enabled ({exc})"
            ) from exc
        if not isinstance(body, Mapping):
            raise PreflightError(
                f"the UC Delta API returned a non-object table response for {_dotted(ref)}: "
                f"{body!r:.200}"
            )
        metadata = body.get("metadata")
        tail_id = (
            (metadata.get("table-uuid") or metadata.get("table_uuid"))
            if isinstance(metadata, Mapping)
            else None
        )
        if tail_id and resolved.table_id and str(tail_id) != str(resolved.table_id):
            # Two calls, two tables: dropped and re-created between the tables
            # API lookup and this one. Pairing one table's id and credentials
            # with the other's commit tail and location read (and committed
            # to) the wrong table.
            raise _Recreated(str(tail_id))
        entries, latest, location = parse_commit_tail(body, resolved.location)
        return dataclasses.replace(
            resolved,
            log_tail=entries,
            max_catalog_version=latest,
            location=location,
        )

    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]:
        # Paged: the server returns one page (100 by default) plus a
        # next_page_token, and reading only the first silently dropped the rest.
        out = []
        # Under _call: a missing schema or a denial is named like every other
        # catalog call instead of surfacing as a bare HTTP error.
        infos = self._call(
            "list tables",
            f"{catalog}.{schema}",
            "read",
            lambda: self._paged(
                f"{UC_API}/tables", "tables", {"catalog_name": catalog, "schema_name": schema}
            ),
        )
        for info in infos:
            if not info.get("name"):
                continue
            name = str(info["name"])
            ref = TableRef(
                kind=RefKind.CATALOG,
                catalog=catalog,
                schema=schema,
                table=name,
                scheme="uc",
                raw=f"{catalog}.{schema}.{name}",
            )
            # The table id reaches the provider and table_uuid too; listing used
            # to drop both, so listed tables skipped the stale-id check.
            out.append(self._resolved_from_info(ref, info))
        return out

    def list_catalogs(self) -> list[str]:
        return [
            str(c["name"])
            for c in self._call(
                "list catalogs",
                "the metastore",
                "read",
                lambda: self._paged(f"{UC_API}/catalogs", "catalogs", {}),
            )
            if c.get("name")
        ]

    def list_schemas(self, catalog: str) -> list[str]:
        return [
            str(s["name"])
            for s in self._call(
                "list schemas",
                catalog,
                "read",
                lambda: self._paged(f"{UC_API}/schemas", "schemas", {"catalog_name": catalog}),
            )
            if s.get("name")
        ]

    def drop_table(self, ref: TableRef) -> None:
        # _dotted, not an f-string: it rejects a reference that is not
        # catalog.schema.table. Interpolating directly turned a path reference
        # into a DELETE of a table literally named "None.None.None".
        name = _dotted(ref)
        # Under _call: dropping a missing table is an InvalidReferenceError,
        # as it is on Databricks, not a raw HTTP 404.
        self._call(
            "drop the table",
            name,
            "drop",
            lambda: _request(
                self._base_url, f"{UC_API}/tables/{_q(name)}", self._token, method="DELETE"
            ),
        )

    @staticmethod
    def _table_type(raw: str | None) -> TableType | None:
        if not raw:
            return None
        try:
            return TableType(str(raw).upper())
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
            if exc.status == 401:
                # Not a privilege problem: naming grants sent admins after the
                # wrong fix for what is an expired or wrong token.
                raise PreflightError(
                    f"cannot {action}: Unity Catalog rejected the credentials (HTTP 401: "
                    "the token is missing, expired or invalid for this server). "
                    f"Underlying error: {exc}"
                ) from exc
            if exc.status == 403:
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
        name = _dotted(target) if isinstance(target, TableRef) else str(target).strip()
        if not name:
            raise InvalidReferenceError("a permissions call needs a securable name")
        # An SDK SecurableType enum stringifies as "SecurableType.TABLE".
        kind = str(getattr(securable_type, "value", securable_type)).strip()
        # OSS spells securable types in lower case in the path.
        return name, f"{UC_API}/permissions/{_q(kind.lower())}/{_q(name)}"

    @staticmethod
    def _from_wire(body: Any) -> list[Grant]:
        # Grant.list_from_api normalizes OSS's "USE SCHEMA" to "USE_SCHEMA".
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
        if not principal or not str(principal).strip():
            raise InvalidReferenceError("grant/revoke needs a principal")
        # A bare string is one privilege: iterating "SELECT" sent the
        # privileges "S", "E", "L", "E", "C", "T".
        if isinstance(privileges, str):
            privileges = [privileges]
        # OSS spells privileges with spaces: "USE SCHEMA", not "USE_SCHEMA".
        wire = list(
            dict.fromkeys(
                normalize_privilege(p).replace("_", " ") for p in privileges if str(p).strip()
            )
        )
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
        seen: set[str] = set()
        while True:
            query = dict(params)
            if token:
                query["page_token"] = token
            encoded = urllib.parse.urlencode(query)
            body = self._get(f"{path}?{encoded}" if encoded else path)
            if not isinstance(body, Mapping):
                raise PreflightError(
                    f"{self._base_url}{path} returned {type(body).__name__}, not a JSON object"
                )
            out.extend(dict(item) for item in (body.get(key) or []) if isinstance(item, Mapping))
            token = body.get("next_page_token") or None
            if not token:
                return out
            if token in seen:
                # A server that hands back the same token forever used to spin
                # here indefinitely, growing `out` without bound.
                raise PreflightError(
                    f"{self._base_url}{path} repeated page token {token!r}; refusing to loop"
                )
            seen.add(token)

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
            try:
                infos = self._paged(
                    f"{UC_API}/tables", "tables", {"catalog_name": catalog, "schema_name": schema}
                )
            except UnityCatalogHTTPError as exc:
                if exc.status == 404:
                    # Dropped between listing the schemas and listing its
                    # tables: it has no tables to report, not a failed search.
                    continue
                raise
            for info in infos:
                if tables is None or tables.match(str(info.get("name") or "")):
                    # The listed schema fills in a name the entry leaves out, which
                    # otherwise came back as "catalog.table".
                    out.append(
                        TableSummary.from_api(
                            {"catalog_name": catalog, "schema_name": schema, **info}
                        )
                    )
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
        kind = str(getattr(volume_type, "value", volume_type)).strip().upper()
        if kind not in ("MANAGED", "EXTERNAL"):
            raise InvalidReferenceError(
                f"volume_type must be MANAGED or EXTERNAL, not {volume_type!r}"
            )
        if kind == "EXTERNAL" and not storage_location:
            raise InvalidReferenceError("an EXTERNAL volume needs a storage_location")
        if kind == "MANAGED" and storage_location:
            raise InvalidReferenceError(
                "a MANAGED volume's location is chosen by the catalog; pass "
                "volume_type='EXTERNAL' to use storage_location"
            )
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
        if not location:
            raise InvalidReferenceError(f"registering {name} needs a storage location")
        # A bare string is one column, not one column per character.
        if isinstance(partition_columns, str):
            partition_columns = [partition_columns]
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
            # UC properties are string -> string; a bool or int was sent as a
            # JSON literal and rejected (or stored as "True").
            "properties": {str(k): _prop(v) for k, v in dict(properties or {}).items()},
        }
        if comment is not None:
            body["comment"] = comment
        self._call(
            "register an external table",
            name,
            "register",
            lambda: self._send("POST", f"{UC_API}/tables", body),
        )
        return self._resolve_after(ref, "registered")

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
        if not isinstance(body, dict):
            raise CredentialError(f"unexpected path-credentials response for {url}: {body!r:.200}")
        # `or`, not setdefault: a present-but-empty url left the credential
        # with no scope and, on Azure, no endpoint.
        body["url"] = body.get("url") or url
        return _parse_vended(
            body, operation=Operation.READ if op == "PATH_READ" else Operation.READ_WRITE
        )

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
        return self._resolve_after(ref, "created")

    def _resolve_after(self, ref: TableRef, done: str) -> ResolvedTable:
        """Resolve a table this catalog has just registered.

        The registration already succeeded, so a failure here must say so: a
        bare "does not exist" or network error read as if nothing had happened,
        and retrying the create then failed with "already exists".
        """
        try:
            return self.resolve(ref)
        except (PreflightError, InvalidReferenceError, CredentialError) as exc:
            raise PreflightError(
                f"{_dotted(ref)} was {done} in Unity Catalog, but reading it back failed: "
                f"{exc}. The table exists; open it again with conn.table({_dotted(ref)!r}) "
                "rather than repeating the create"
            ) from exc
