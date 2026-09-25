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
import time
from datetime import UTC
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .._sdk import workspace_client
from ..errors import CredentialError
from .base import DEFAULT_REFRESH_MARGIN_SECONDS, Cloud, Credentials, Operation

if TYPE_CHECKING:  # pragma: no cover
    from databricks.sdk.core import Config

__all__ = ["DatabricksCredentialProvider", "azure_endpoint_for", "r2_endpoint_for"]

#: How long a credential whose response carried no usable expiration is served
#: before it is re-vended. Vended credentials live about an hour.
UNKNOWN_EXPIRY_MAX_AGE_SECONDS = 900.0
#: How long a credential that arrives already "expired" by the local clock (a
#: clock running ahead of the server) is served before it is re-vended.
CLOCK_SKEW_GRACE_SECONDS = 60.0
#: Pause between refresh-ahead attempts after one fails, while the cached
#: credential is still valid.
REFRESH_RETRY_BACKOFF_SECONDS = 30.0


def _is_local_host(host: str) -> bool:
    """An emulator host (Azurite): an IP literal or localhost, path-style addressed."""
    import ipaddress

    name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    name = name.strip("[]")
    if name.lower() == "localhost":
        return True
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def _blob_host(host: str) -> str:
    """``acct.dfs.<suffix>`` -> ``acct.blob.<suffix>``; anything else verbatim.

    object_store speaks only the *Blob* REST API (``?restype=container&comp=list``
    to list, ``?comp=block`` / ``?comp=blocklist`` to upload), and treats the
    endpoint it is given as the Blob service root. The ADLS Gen2 ``dfs``
    endpoint serves the Data Lake API instead -- listing and block uploads
    against it fail -- which is why object_store itself maps an
    ``abfss://c@acct.dfs.core.windows.net`` URL to ``acct.blob.core.windows.net``
    when no endpoint is set. Only the service label is swapped, so the
    sovereign-cloud and Fabric suffixes are preserved.
    """
    account, dot, rest = host.partition(".")
    if dot and rest.lower().startswith("dfs."):
        return f"{account}.blob.{rest[4:]}"
    return host


def azure_endpoint_for(url: str) -> str | None:
    """Derive an explicit Azure *Blob* endpoint from a table URL.

    Returns e.g. ``https://acct.blob.core.windows.net`` for
    ``abfss://container@acct.dfs.core.windows.net/path``. The endpoint is the
    one object_store talks to, and it speaks only the Blob API, so a ``dfs``
    host is mapped to its ``blob`` sibling (see `_blob_host`). Beyond that
    label the host is kept verbatim, which is what keeps this correct on
    private-link and sovereign-cloud hosts, where the suffix is not
    ``core.windows.net`` at all.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    _container, at, host = parsed.netloc.rpartition("@")
    if not at:
        # Only an http(s) URL names the account host without a container
        # prefix. `az://container/path` and `abfss://container/path` carry the
        # *container* in the netloc, and turning that into
        # ``https://container`` produced an endpoint that resolves nowhere.
        if scheme not in ("http", "https"):
            return None
        host = parsed.netloc
    if not host:
        return None
    # Azurite and other local emulators answer plain http; keep the scheme the
    # URL itself asked for rather than forcing TLS onto them.
    prefix = "http" if scheme == "http" else "https"
    if scheme in ("http", "https") and _is_local_host(host):
        # Path-style addressing: the account is the first path segment, and
        # object_store appends only ``/<container>/<path>`` to the endpoint --
        # dropping the account sent every request to a path Azurite rejects.
        account = next((seg for seg in parsed.path.split("/") if seg), "")
        return f"{prefix}://{host}/{account}" if account else f"{prefix}://{host}"
    return f"{prefix}://{_blob_host(host)}"


def _expires_at_seconds(raw: Any) -> float | None:
    """`expiration_time` as epoch seconds.

    The Databricks API documents epoch *milliseconds*. Some Unity Catalog
    servers answer in seconds; read as milliseconds that lands in January 1970,
    so every credential looked expired and was re-vended on every request (and
    a path credential was refused outright). Anything below 1e11 cannot be a
    millisecond timestamp after 1973, so it is taken as seconds.
    """
    if raw is None or raw == "" or isinstance(raw, bool):
        return None
    if hasattr(raw, "timestamp") and callable(raw.timestamp):
        # A datetime (an SDK or server that already parsed the field). A naive
        # one is UTC, as every Databricks timestamp is.
        if getattr(raw, "tzinfo", 1) is None:
            raw = raw.replace(tzinfo=UTC)
        return float(raw.timestamp())
    try:
        value = float(raw)
    except (TypeError, ValueError):
        # An ISO-8601 string: unparsed, it read as "never expires", so the
        # credential was cached for the life of the process and died at the
        # storage layer an hour later.
        from datetime import datetime

        try:
            parsed = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    if value <= 0:
        return None
    return value if value < 1e11 else value / 1000.0


def _require(values: dict[str, Any], what: str) -> dict[str, str]:
    """Drop absent fields and insist on the ones a store cannot work without.

    storage_options are string-valued only; a None slipping through fails in
    delta-rs with an opaque TypeError far from the vending call.
    """
    missing = [k for k, v in values.items() if not v]
    if missing:
        raise CredentialError(
            f"Unity Catalog returned {what} credentials without {', '.join(sorted(missing))}"
        )
    return {k: str(v) for k, v in values.items()}


def _s3_access_point_options(access_point: str) -> dict[str, str]:
    """storage_options addressing an S3 access point.

    UC reports the access point as an ARN. object_store takes an endpoint
    *URL*, so passing the ARN through as `aws_endpoint_url` broke every request
    against such a table. The access point is served at its own virtual host,
    which object_store uses verbatim when virtual-hosted-style is on.
    """
    if "://" in access_point:
        return {"aws_endpoint_url": access_point, "aws_virtual_hosted_style_request": "true"}
    parts = access_point.split(":", 5)
    if len(parts) == 6 and parts[0] == "arn" and parts[2] == "s3":
        partition, region, account, resource = parts[1], parts[3], parts[4], parts[5]
        name = resource.split("/", 1)[1] if "/" in resource else ""
        if name and region and account:
            suffix = "amazonaws.com.cn" if partition == "aws-cn" else "amazonaws.com"
            return {
                "aws_endpoint_url": f"https://{name}-{account}.s3-accesspoint.{region}.{suffix}",
                "aws_virtual_hosted_style_request": "true",
                "aws_region": region,
            }
    raise CredentialError(
        f"Unity Catalog vended credentials for S3 access point {access_point!r}, which is "
        "not an access-point ARN this layer can turn into an endpoint"
    )


def r2_endpoint_for(url: str) -> str | None:
    """The S3-compatible endpoint for an ``r2://bucket@account.r2.cloudflarestorage.com`` URL."""
    parsed = urlparse(url)
    host = parsed.netloc.rpartition("@")[2]
    if not host or "." not in host:
        return None
    return f"https://{host}"


def _response_fields(resp: Any) -> list[str]:
    if isinstance(resp, dict):
        return sorted(str(k) for k in resp if not str(k).startswith("_"))
    as_dict = getattr(resp, "as_dict", None)
    if callable(as_dict):
        try:
            return sorted(as_dict())
        except Exception:  # pragma: no cover - diagnostic only
            pass
    try:
        return sorted(k for k in vars(resp) if not k.startswith("_"))
    except TypeError:
        return [type(resp).__name__]


_TRANSIENT_ERRORS = frozenset(
    {
        "TooManyRequests",
        "ResourceExhausted",
        "RequestLimitExceeded",
        "TemporarilyUnavailable",
        "DeadlineExceeded",
        "TimeoutError",
    }
)
_TRANSIENT_CODES = frozenset(
    {"RESOURCE_EXHAUSTED", "REQUEST_LIMIT_EXCEEDED", "TEMPORARILY_UNAVAILABLE", "TOO_MANY_REQUESTS"}
)


def _error_kind(exc: BaseException) -> str | None:
    """ "not_found" / "denied" / "unauthenticated" / "transient" for an SDK error, else None.

    Unity Catalog reports a missing securable under several codes: the typed
    404s, ``*_DOES_NOT_EXIST``, and -- as a *400* ``BadRequest`` --
    ``TABLE_OR_VIEW_NOT_FOUND`` / ``SCHEMA_NOT_FOUND`` / ``CATALOG_NOT_FOUND``.
    Keying only on ``DOES_NOT_EXIST`` sent those down the privilege path.
    Throttling and unavailability (after the SDK's own retries gave up) are
    "transient": they say nothing about privileges or metastore settings.
    """
    names = {c.__name__ for c in type(exc).__mro__}
    code = str(getattr(exc, "error_code", "") or "").upper()
    if (
        names & {"NotFound", "ResourceDoesNotExist"}
        or code.endswith("DOES_NOT_EXIST")
        or code == "NOT_FOUND"
        or code.endswith("_NOT_FOUND")
    ):
        return "not_found"
    if "Unauthenticated" in names or code == "UNAUTHENTICATED":
        return "unauthenticated"
    if "PermissionDenied" in names or code == "PERMISSION_DENIED":
        return "denied"
    if names & _TRANSIENT_ERRORS or code in _TRANSIENT_CODES:
        return "transient"
    return None


def _config_attributes(config: Any) -> dict[str, Any]:
    """The plain, picklable attributes a `Config` was built from."""
    try:
        raw = dict(config.as_dict())
    except Exception:
        raw = {}
    out = {k: v for k, v in raw.items() if isinstance(v, (str, int, float, bool)) and v is not None}
    host = getattr(config, "host", None)
    if isinstance(host, str) and host:
        out.setdefault("host", host)
    if out.get("host") and out.get("auth_type"):
        # The profile's values are already carried, resolved. Naming the
        # profile too made a worker whose ~/.databrickscfg lacks it (or has a
        # different one of that name) fail with "has no <profile> profile
        # configured" -- or silently merge someone else's settings.
        out.pop("profile", None)
        out.pop("config_file", None)
    return out


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
        # Local wall-clock time each cached credential was vended at, so the
        # refresh margin can be capped to a fraction of its real lifetime.
        self._vended_at: dict[Operation, float] = {}
        # After a failed refresh-ahead, when the next attempt is allowed.
        self._retry_after: dict[Operation, float] = {}
        # Operations whose cached credential arrived already expired by the
        # local clock (local clock ahead of the server's).
        self._skewed: set[Operation] = set()
        # One in-flight vend per operation; see credentials().
        self._vend_locks: dict[Operation, threading.Lock] = {}
        # Re-entrant: `credentials()` holds it across `_vend()`, which builds
        # the client under the same lock.
        self._lock = threading.RLock()

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
        state["_vended_at"] = {}
        state["_retry_after"] = {}
        state["_skewed"] = set()
        # Locks cannot be pickled; __setstate__ installs fresh ones.
        del state["_lock"]
        state.pop("_vend_locks", None)
        # A Config may hold non-picklable auth state; workers rebuild it. Its
        # plain attributes (host, auth type, client id, ...) are carried over:
        # dropping the Config outright left a worker with no host at all, so it
        # fell back to whatever default profile the worker happened to have --
        # a different workspace, or none.
        config = state["_explicit_config"]
        state["_explicit_config"] = None
        if config is not None:
            state["_config_kwargs"] = {**_config_attributes(config), **self._config_kwargs}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("_vended_at", {})
        state.setdefault("_retry_after", {})
        state.setdefault("_skewed", set())
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._vend_locks = {}

    @property
    def table_id(self) -> str | None:
        return self._table_id

    def _workspace(self) -> Any:
        with self._lock:
            return self._workspace_locked()

    def _workspace_locked(self) -> Any:
        if self._client is None:
            self._client = workspace_client(
                config=self._explicit_config,
                profile=self._profile,
                host=self._host,
                token=self._token,
                **self._config_kwargs,
            )
        return self._client

    def _margin_for(self, operation: Operation, cached: Credentials) -> float:
        """The refresh margin, capped at half the credential's real lifetime.

        A credential vended with less life than the margin (short-TTL policies,
        or a path credential near its end) was otherwise "expiring" the moment
        it arrived, so every storage request re-vended it.
        """
        vended_at = self._vended_at.get(operation)
        if cached.expires_at is None or vended_at is None:
            return self._refresh_margin
        lifetime = max(0.0, cached.expires_at - vended_at)
        return min(self._refresh_margin, lifetime / 2.0)

    def _is_fresh(self, operation: Operation, cached: Credentials, now: float) -> bool:
        """Whether a cached credential can be served without re-vending."""
        vended_at = self._vended_at.get(operation)
        if cached.expires_at is None:
            # No stated expiry is not "valid forever": the server merely did
            # not say. Caching it for the life of the process served a dead
            # token once the real (typically one-hour) TTL passed.
            return vended_at is None or now - vended_at < UNKNOWN_EXPIRY_MAX_AGE_SECONDS
        if vended_at is not None and operation in self._skewed:
            # Expired on arrival by the local clock, yet just issued: the local
            # clock is ahead of the server's. Re-vending cannot fix a clock, so
            # every request re-vended; serve it for a short grace instead.
            return now - vended_at < CLOCK_SKEW_GRACE_SECONDS
        return not cached.expires_within(self._margin_for(operation, cached))

    def _vend_lock(self, operation: Operation) -> threading.Lock:
        with self._lock:
            lock = self._vend_locks.get(operation)
            if lock is None:
                lock = self._vend_locks[operation] = threading.Lock()
            return lock

    def credentials(self, operation: Operation = Operation.READ) -> Credentials:
        # A plain "READ_WRITE" string compares equal to the enum but is not
        # identical to it, and an identity test downstream vended *read-only*
        # credentials for a write. Normalise once, here.
        try:
            operation = Operation(operation)
        except ValueError as exc:
            raise CredentialError(
                f"unknown credential operation {operation!r}; expected one of "
                f"{[o.value for o in Operation]}"
            ) from exc
        with self._lock:
            cached = self._cache.get(operation)
            now = time.time()
            if cached is not None and self._is_fresh(operation, cached, now):
                return cached
            usable = cached is not None and not cached.is_expired
            if cached is not None and usable and now < self._retry_after.get(operation, 0.0):
                # A refresh-ahead just failed; do not hammer a throttled
                # workspace on every request while the credential still works.
                return cached
        vend_lock = self._vend_lock(operation)
        # The network call runs outside `_lock`. Holding it across the vend
        # (which the SDK may retry for minutes on a 429) stalled every caller,
        # including those a still-valid cached credential could have served.
        if cached is not None and usable:
            if not vend_lock.acquire(blocking=False):
                return cached  # another thread is refreshing it
        else:
            vend_lock.acquire()  # wait for an in-flight vend rather than duplicate it
        try:
            with self._lock:
                current = self._cache.get(operation)
                if (
                    current is not None
                    and current is not cached
                    and self._is_fresh(operation, current, time.time())
                ):
                    return current  # refreshed by the thread we waited on
            try:
                fresh = self._vend(operation)
            except CredentialError:
                # A refresh-ahead that fails (a 503, a throttled workspace) must
                # not fail a request the cached credential can still serve: it
                # is inside the margin, not past its expiry.
                if cached is not None and not cached.is_expired:
                    with self._lock:
                        remaining = (cached.expires_at or time.time()) - time.time()
                        self._retry_after[operation] = time.time() + min(
                            REFRESH_RETRY_BACKOFF_SECONDS, max(1.0, remaining / 4.0)
                        )
                    return cached
                raise
            with self._lock:
                vended_at = time.time()
                self._cache[operation] = fresh
                self._vended_at[operation] = vended_at
                self._retry_after.pop(operation, None)
                if fresh.expires_at is not None and fresh.expires_at <= vended_at:
                    self._skewed.add(operation)
                else:
                    self._skewed.discard(operation)
            return fresh
        finally:
            vend_lock.release()

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()
            self._vended_at.clear()
            self._retry_after.clear()
            self._skewed.clear()

    def _vend(self, operation: Operation) -> Credentials:
        from databricks.sdk.service.catalog import TableOperation

        op = TableOperation.READ_WRITE if operation is Operation.READ_WRITE else TableOperation.READ
        # Built outside the vend's error handling: a Config that cannot resolve
        # any authentication raises here, and that was reported as "external
        # data access is not enabled / missing EXTERNAL USE SCHEMA".
        try:
            workspace = self._workspace()
        except CredentialError:
            raise
        except Exception as exc:
            raise CredentialError(
                f"credential vending failed for table_id={self._table_id} "
                f"({operation.value}): could not configure Databricks authentication "
                "(no usable host/token/OAuth settings, or an unknown profile). "
                f"Underlying error: {exc}"
            ) from exc
        try:
            resp = workspace.temporary_table_credentials.generate_temporary_table_credentials(
                table_id=self._table_id, operation=op
            )
        except Exception as exc:
            kind = _error_kind(exc)
            if kind == "transient":
                raise CredentialError(
                    f"credential vending failed for table_id={self._table_id} "
                    f"({operation.value}): the workspace is throttling or temporarily "
                    "unavailable and the SDK's retries were exhausted; retry later. "
                    f"Underlying error: {exc}"
                ) from exc
            if kind == "not_found":
                raise CredentialError(
                    f"credential vending failed for table_id={self._table_id} "
                    f"({operation.value}): Unity Catalog has no table with that id. It was "
                    "most likely dropped (and perhaps re-created under the same name, which "
                    f"assigns a new id); re-resolve the table. Underlying error: {exc}"
                ) from exc
            if kind == "unauthenticated":
                raise CredentialError(
                    f"credential vending failed for table_id={self._table_id} "
                    f"({operation.value}): the Databricks credentials were rejected "
                    "(expired or invalid token, or wrong workspace host). "
                    f"Underlying error: {exc}"
                ) from exc
            raise CredentialError(
                f"credential vending failed for table_id={self._table_id} ({operation.value}). "
                "Usually one of: external data access is off on the metastore (an "
                "account-admin setting); the principal lacks EXTERNAL USE SCHEMA; or the "
                f"table has a row filter or column mask. Underlying error: {exc}"
            ) from exc

        return self._to_credentials(resp, operation)

    def _to_credentials(self, resp: Any, operation: Operation) -> Credentials:
        return credentials_from_response(
            resp,
            url=self._table_url or "",
            table_id=self._table_id,
            aws_region=self._aws_region(),
            operation=operation,
        )

    def workspace_auth(self) -> tuple[str, str]:
        """`(workspace_url, bearer_token)` for the UC commit API.

        Note this is the *catalog* credential, on a different clock from the
        vended storage credential -- the SDK refreshes this one, we re-vend the
        other. Conflating them is the usual cause of a job that dies after an
        hour with healthy-looking auth.
        """
        config = self._workspace().config
        try:
            headers = config.authenticate()
        except CredentialError:
            raise
        except Exception as exc:
            # An OAuth refresh that fails (revoked secret, expired U2M session)
            # surfaced as a raw SDK/requests error from deep in the commit path.
            raise CredentialError(
                "could not obtain a Databricks token for the Unity Catalog commit API "
                f"({type(exc).__name__}): {exc}"
            ) from exc
        authorization = next(
            (v for k, v in (headers or {}).items() if k.lower() == "authorization"), ""
        )
        if not authorization.lower().startswith("bearer "):
            # Name only the scheme. A header with no scheme is the credential
            # itself, and must not reach an exception message.
            scheme = authorization.split(" ", 1)[0] if " " in authorization else None
            shown = scheme or ("an unrecognised header" if authorization else "nothing")
            raise CredentialError(
                "the configured Databricks authentication did not yield a bearer token "
                f"(got {shown!r}). "
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
    operation: Operation | None = None,
) -> Credentials:
    """Map a Unity Catalog temporary-credentials response to object-store options.

    `resp` is the Databricks SDK response or the same response as JSON from
    open-source Unity Catalog; table and path vending share the shape.
    """
    url = _get(resp, "url") or url
    expiry = _get(resp, "expiration_time")
    if expiry is None:
        expiry = _get(resp, "expirationTime")
    expires_at = _expires_at_seconds(expiry)
    cloud: Cloud

    if aws := _get(resp, "aws_temp_credentials"):
        cloud = Cloud.AWS
        secrets = _require(
            {
                "aws_access_key_id": _get(aws, "access_key_id"),
                "aws_secret_access_key": _get(aws, "secret_access_key"),
            },
            "AWS",
        )
        if token := _get(aws, "session_token"):
            secrets["aws_session_token"] = str(token)
        # UC vends keys but never a region, and object_store would assume
        # us-east-1; any other bucket then fails with an opaque redirect.
        region = aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if region:
            secrets["aws_region"] = region
        if access_point := _get(aws, "access_point"):
            # The ARN names the access point's own region, which wins.
            secrets.update(_s3_access_point_options(str(access_point)))
    elif r2 := _get(resp, "r2_temp_credentials"):
        cloud = Cloud.R2
        secrets = _require(
            {
                "aws_access_key_id": _get(r2, "access_key_id"),
                "aws_secret_access_key": _get(r2, "secret_access_key"),
            },
            "R2",
        )
        if token := _get(r2, "session_token"):
            secrets["aws_session_token"] = str(token)
        # R2 speaks S3 at the account's own endpoint. Without it these keys
        # went to AWS S3 in us-east-1, which rejects them.
        endpoint = r2_endpoint_for(url)
        if endpoint is None:
            raise CredentialError(
                f"cannot derive a Cloudflare R2 endpoint from {url!r}; expected "
                "r2://bucket@<account>.r2.cloudflarestorage.com/..."
            )
        secrets["aws_endpoint_url"] = endpoint
        secrets["aws_region"] = "auto"
    elif sas_block := _get(resp, "azure_user_delegation_sas"):
        cloud = Cloud.AZURE
        sas = _get(sas_block, "sas_token")
        if not sas:
            raise CredentialError("Unity Catalog returned an Azure SAS block with no token")
        endpoint = azure_endpoint_for(url)
        if endpoint is None:
            raise CredentialError(
                f"cannot derive an Azure endpoint from {url!r}; object_store would "
                "fall back to account-name inference, which breaks Azurite, "
                "private-link DNS and sovereign clouds"
            )
        # The SAS is a query string; a leading "?" is not part of it.
        secrets = {"azure_storage_sas_key": str(sas).lstrip("?"), "azure_endpoint": endpoint}
        if endpoint.startswith("http://"):
            # object_store refuses plain-http endpoints unless told to allow
            # them, so an emulator endpoint alone failed every request with
            # "URL scheme is not allowed".
            secrets["azure_allow_http"] = "true"
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
        token = _get(gcp, "oauth_token")
        if not token:
            raise CredentialError("Unity Catalog returned a GCP block with no OAuth token")
        secrets = {"google_bearer_token": str(token)}
    else:
        raise CredentialError(
            f"Unity Catalog returned no recognized credential block (table_id={table_id}). "
            f"Response fields: {_response_fields(resp)}"
        )

    return Credentials(
        cloud=cloud,
        url=url,
        expires_at=expires_at,
        secrets=secrets,
        # Azure SAS is path-scoped, so the store registry must key on this.
        scope_prefix=url or None,
        table_id=table_id,
        operation=operation,
    )
