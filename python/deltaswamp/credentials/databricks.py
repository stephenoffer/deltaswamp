"""Databricks Unity Catalog credential vending.

- Authentication is `databricks.sdk.core.Config`, which already covers PAT,
  OAuth U2M and M2M, Azure CLI, MSI and service principals, GCP service
  accounts, GitHub OIDC and in-cluster runtimes. Prefer OAuth M2M for long
  jobs, since a PAT cannot be refreshed.
- The provider is picklable and the credential is not: `__getstate__` drops the
  live client and the cached secret, so a Ray worker gets configuration and
  vends its own credential.
- Azure always gets an explicit endpoint. Account-name inference only works on
  `*.blob.core.windows.net`.
- AWS always gets an explicit region. UC vends keys without one, and
  object_store would otherwise assume us-east-1 and fail on any other bucket.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .._sdk import workspace_client
from ..errors import CredentialError
from .base import DEFAULT_REFRESH_MARGIN_SECONDS, Cloud, Credentials, Operation

if TYPE_CHECKING:  # pragma: no cover
    from databricks.sdk.core import Config

__all__ = ["DatabricksCredentialProvider", "azure_endpoint_for"]


def azure_endpoint_for(url: str) -> str | None:
    """Derive an explicit Azure storage endpoint from a table URL.

    Returns e.g. ``https://acct.dfs.core.windows.net`` for
    ``abfss://container@acct.dfs.core.windows.net/path``. Returning the host
    verbatim (rather than rewriting dfs<->blob) is what keeps this correct on
    private-link and sovereign-cloud hosts, where the suffix is not
    ``core.windows.net`` at all.
    """
    parsed = urlparse(url)
    host = parsed.netloc.rpartition("@")[2]
    if not host:
        return None
    return f"https://{host}"


class DatabricksCredentialProvider:
    """Vends per-table credentials from the UC temporary-credentials API.

    Scoped to exactly one table: the API takes a single `table_id` and has no
    batch form, so N tables means N calls. Results are cached until they are
    within the refresh margin of their stated `expiration_time` -- Databricks
    does not publish a TTL, so that field is the only authority.
    """

    def __init__(
        self,
        table_id: str,
        *,
        table_url: str | None = None,
        region: str | None = None,
        config: Config | None = None,
        profile: str | None = None,
        host: str | None = None,
        token: str | None = None,
        refresh_margin: float = DEFAULT_REFRESH_MARGIN_SECONDS,
        **config_kwargs: Any,
    ) -> None:
        self._table_id = table_id
        self._table_url = table_url
        self._region = region
        self._refresh_margin = refresh_margin
        # Configuration is kept as plain data so this object stays picklable.
        self._profile = profile
        self._host = host
        self._token = token
        self._config_kwargs = config_kwargs
        self._explicit_config = config

        self._client: Any = None
        self._cache: dict[Operation, Credentials] = {}
        self._lock = threading.Lock()

    # --- picklability: ship configuration, never a live client or a token ---

    def _aws_region(self) -> str | None:
        """The region for an S3 bucket: the catalog's, else the environment's.

        The catalog passes its metastore region, which is authoritative for a
        UC-managed location. The environment fallback covers a caller who
        resolved a table some other way.
        """
        return self._region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        state["_cache"] = {}
        # A lock cannot be pickled; __setstate__ installs a fresh one.
        del state["_lock"]
        # A Config may hold non-picklable auth state; workers rebuild it.
        state["_explicit_config"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    @property
    def table_id(self) -> str | None:
        return self._table_id

    def _workspace(self) -> Any:
        if self._client is None:
            self._client = workspace_client(
                config=self._explicit_config,
                profile=self._profile,
                host=self._host,
                token=self._token,
                **self._config_kwargs,
            )
        return self._client

    def credentials(self, operation: Operation = Operation.READ) -> Credentials:
        # __setstate__ always restores a real lock, so this is never None.
        with self._lock:
            cached = self._cache.get(operation)
            if cached is not None and not cached.expires_within(self._refresh_margin):
                return cached
            fresh = self._vend(operation)
            self._cache[operation] = fresh
            return fresh

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    def _vend(self, operation: Operation) -> Credentials:
        from databricks.sdk.service.catalog import TableOperation

        op = TableOperation.READ_WRITE if operation is Operation.READ_WRITE else TableOperation.READ
        try:
            resp = (
                self._workspace().temporary_table_credentials.generate_temporary_table_credentials(
                    table_id=self._table_id, operation=op
                )
            )
        except Exception as exc:
            raise CredentialError(
                f"credential vending failed for table_id={self._table_id} ({operation.value}). "
                "Usually one of: external data access is off on the metastore (an "
                "account-admin setting); the principal lacks EXTERNAL USE SCHEMA; or the "
                f"table has a row filter or column mask. Underlying error: {exc}"
            ) from exc

        return self._to_credentials(resp, operation)

    def _to_credentials(self, resp: Any, operation: Operation) -> Credentials:
        return credentials_from_response(
            resp, url=self._table_url or "", table_id=self._table_id, aws_region=self._aws_region()
        )

    def workspace_auth(self) -> tuple[str, str]:
        """`(workspace_url, bearer_token)` for the UC commit API.

        Note this is the *catalog* credential, on a different clock from the
        vended storage credential -- the SDK refreshes this one, we re-vend the
        other. Conflating them is the usual cause of a job that dies after an
        hour with healthy-looking auth.
        """
        config = self._workspace().config
        headers = config.authenticate()
        authorization = headers.get("Authorization", "")
        if not authorization.lower().startswith("bearer "):
            raise CredentialError(
                "the configured Databricks authentication did not yield a bearer token "
                f"(got {authorization.split(' ')[0] if authorization else 'nothing'!r}). "
                "The Unity Catalog commit API requires one; OAuth M2M or a PAT will work."
            )
        return config.host.rstrip("/"), authorization[len("bearer ") :]

    def __repr__(self) -> str:
        return f"DatabricksCredentialProvider(table_id={self._table_id!r})"


def _get(obj: Any, name: str) -> Any:
    """A field of an SDK response object or of the same response as JSON."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def credentials_from_response(
    resp: Any,
    *,
    url: str = "",
    table_id: str | None = None,
    aws_region: str | None = None,
) -> Credentials:
    """Map a Unity Catalog temporary-credentials response to object-store options.

    `resp` is the Databricks SDK response or the same response as JSON from
    open-source Unity Catalog; table and path vending share the shape.
    """
    url = _get(resp, "url") or url
    expiry = _get(resp, "expiration_time") or _get(resp, "expirationTime")
    expires_at = float(expiry) / 1000.0 if expiry else None
    cloud: Cloud

    if aws := _get(resp, "aws_temp_credentials"):
        cloud = Cloud.AWS
        secrets = {
            "aws_access_key_id": _get(aws, "access_key_id"),
            "aws_secret_access_key": _get(aws, "secret_access_key"),
            "aws_session_token": _get(aws, "session_token"),
        }
        # UC vends keys but never a region, and object_store would assume
        # us-east-1; any other bucket then fails with an opaque redirect.
        region = aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if region:
            secrets["aws_region"] = region
        if access_point := _get(aws, "access_point"):
            secrets["aws_endpoint_url"] = access_point
    elif r2 := _get(resp, "r2_temp_credentials"):
        cloud = Cloud.R2
        secrets = {
            "aws_access_key_id": _get(r2, "access_key_id"),
            "aws_secret_access_key": _get(r2, "secret_access_key"),
            "aws_session_token": _get(r2, "session_token"),
        }
    elif sas := _get(resp, "azure_user_delegation_sas"):
        cloud = Cloud.AZURE
        endpoint = azure_endpoint_for(url)
        if endpoint is None:
            raise CredentialError(
                f"cannot derive an Azure endpoint from {url!r}; object_store would "
                "fall back to account-name inference, which breaks Azurite, "
                "private-link DNS and sovereign clouds"
            )
        secrets = {"azure_storage_sas_key": _get(sas, "sas_token"), "azure_endpoint": endpoint}
    elif _get(resp, "azure_aad"):
        raise CredentialError(
            "Unity Catalog returned an Azure AAD token, but the object-store layer "
            "supports only user-delegation SAS for Azure. Ask Databricks support to "
            "enable SAS vending for this metastore, or use the SQL fallback."
        )
    elif gcp := _get(resp, "gcp_oauth_token"):
        cloud = Cloud.GCP
        # Consumed by the Rust store layer. delta-rs maps a token onto
        # google_application_credentials, which object_store reads as a file path.
        secrets = {"google_bearer_token": _get(gcp, "oauth_token")}
    else:
        fields = resp if isinstance(resp, dict) else vars(resp)
        raise CredentialError(
            f"Unity Catalog returned no recognized credential block (table_id={table_id}). "
            f"Response fields: {sorted(k for k in fields if not k.startswith('_'))}"
        )

    return Credentials(
        cloud=cloud,
        url=url,
        expires_at=expires_at,
        secrets=secrets,
        # Azure SAS is path-scoped, so the store registry must key on this.
        scope_prefix=url or None,
        table_id=table_id,
    )
