"""Databricks Unity Catalog credential vending.

Deliberate choices, each tied to a failure mode observed in the wild:

* **We never reimplement Databricks auth.** `databricks.sdk.core.Config` is the
  only thing in the ecosystem that already handles the full permutation --
  PAT, OAuth U2M/M2M, Azure CLI/MSI/SP, GCP SA, GitHub OIDC, in-cluster runtime.
  We take a `Config` (or the arguments to build one) and let it resolve.
* **OAuth M2M is the right default for long jobs; PAT has no refresh story.**
* **The provider is picklable, the credential is not.** `__getstate__` drops the
  live client and the cached secret, so shipping this to a Ray worker sends
  configuration rather than a token, and each worker vends its own.
* **Azure gets an explicit endpoint.** Account-name inference happens to work on
  `*.blob.core.windows.net` and silently breaks everywhere else.
* **AWS gets an explicit region.** UC vends keys but no region, and object_store
  then defaults to us-east-1, so every bucket outside it answers a redirect with
  no Location header. The catalog supplies its metastore's region.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

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
            try:
                from .._sdk import workspace_client
            except ImportError as exc:  # pragma: no cover
                raise CredentialError(
                    "the databricks-sdk package is required for Unity Catalog access; "
                    "install deltaswamp's base dependencies"
                ) from exc
            if self._explicit_config is not None:
                self._client = workspace_client(config=self._explicit_config)
            else:
                kwargs = dict(self._config_kwargs)
                if self._profile:
                    kwargs["profile"] = self._profile
                if self._host:
                    kwargs["host"] = self._host
                if self._token:
                    kwargs["token"] = self._token
                self._client = workspace_client(**kwargs)
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
                "Common causes, in order of likelihood: the metastore does not have "
                "external data access enabled (an account-admin setting, off by default); "
                "the principal lacks EXTERNAL USE SCHEMA on the schema (grantable only by "
                "the catalog owner); or the table has row filters or column masks, which "
                f"vending refuses outright. Underlying error: {exc}"
            ) from exc

        return self._to_credentials(resp, operation)

    def _to_credentials(self, resp: Any, operation: Operation) -> Credentials:
        url = getattr(resp, "url", None) or self._table_url or ""
        expires_at = None
        expiration_ms = getattr(resp, "expiration_time", None)
        if expiration_ms:
            expires_at = float(expiration_ms) / 1000.0

        secrets: dict[str, str] = {}
        cloud: Cloud

        if getattr(resp, "aws_temp_credentials", None):
            c = resp.aws_temp_credentials
            cloud = Cloud.AWS
            secrets = {
                "aws_access_key_id": c.access_key_id,
                "aws_secret_access_key": c.secret_access_key,
                "aws_session_token": c.session_token,
            }
            # UC vends keys but never a region. Without one object_store falls
            # back to us-east-1 and any other bucket answers a redirect carrying
            # no Location header, which surfaces as an opaque "Generic S3 error".
            region = self._aws_region()
            if region:
                secrets["aws_region"] = region
            if getattr(c, "access_point", None):
                secrets["aws_endpoint_url"] = c.access_point
        elif getattr(resp, "r2_temp_credentials", None):
            c = resp.r2_temp_credentials
            cloud = Cloud.R2
            secrets = {
                "aws_access_key_id": c.access_key_id,
                "aws_secret_access_key": c.secret_access_key,
                "aws_session_token": c.session_token,
            }
        elif getattr(resp, "azure_user_delegation_sas", None):
            cloud = Cloud.AZURE
            secrets = {"azure_storage_sas_key": resp.azure_user_delegation_sas.sas_token}
            endpoint = azure_endpoint_for(url)
            if endpoint is None:
                raise CredentialError(
                    f"cannot derive an Azure endpoint from {url!r}; object_store would "
                    "fall back to account-name inference, which breaks Azurite, "
                    "private-link DNS and sovereign clouds"
                )
            secrets["azure_endpoint"] = endpoint
        elif getattr(resp, "azure_aad", None):
            # The SDK surfaces this, but object_store has no bearer path for
            # Azure -- only SAS. Say so rather than producing a broken store.
            raise CredentialError(
                "Unity Catalog returned an Azure AAD token, but the object-store layer "
                "supports only user-delegation SAS for Azure. Ask Databricks support to "
                "enable SAS vending for this metastore, or use the SQL fallback."
            )
        elif getattr(resp, "gcp_oauth_token", None):
            cloud = Cloud.GCP
            # Our own key, consumed by the Rust store layer. Note delta-rs maps
            # this onto google_application_credentials, which object_store reads
            # as a *file path* -- so that path is broken for vended tokens.
            secrets = {"google_bearer_token": resp.gcp_oauth_token.oauth_token}
        else:
            raise CredentialError(
                "Unity Catalog returned no recognised credential block for "
                f"table_id={self._table_id}. Response fields: "
                f"{sorted(k for k in vars(resp) if not k.startswith('_'))}"
            )

        return Credentials(
            cloud=cloud,
            url=url,
            expires_at=expires_at,
            secrets=secrets,
            # Azure SAS is path-scoped, so the store registry must key on this.
            scope_prefix=url or None,
            table_id=self._table_id,
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
