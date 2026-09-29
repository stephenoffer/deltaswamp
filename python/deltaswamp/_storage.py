"""Storage options: one canonical spelling, one precedence, every engine.

A connection's `storage_options` and a catalog's vended credentials are merged
here, and nowhere else, so the kernel and delta-rs see the same settings.

**Canonical keys.** object_store accepts each setting under several aliases,
case-insensitively (``AWS_REGION``, ``aws_region`` and ``region`` are one
setting). Merging two dicts that spell a setting differently kept both
spellings, and object_store then applied them in HashMap order: which value won
changed from one process to the next. Keys are therefore rewritten to
object_store's own canonical name, per cloud (``endpoint`` means the S3
endpoint on ``s3://`` and the Azure one on ``abfss://``), before merging. One
dict that names a setting twice with different values is refused.

**Precedence.** From strongest to weakest:

1. Vended *credentials* (keys, secrets, session tokens, SAS, bearer tokens):
   when the catalog vends any, every credential key the caller passed for that
   cloud is dropped. Mixing the two paired one principal's key with another's
   secret.
2. An S3 endpoint the catalog vended (an access point, an R2 account): it wins
   together with the region and addressing style that belong to it.
3. Everything else the caller passed: region, endpoint, proxy, timeouts,
   retry and TLS settings. A vended ``aws_region`` is the metastore's region,
   a guess for an external bucket in another region, and a vended Azure
   endpoint is derived from the URL's host; the caller's value is the one that
   knows about cross-region buckets, private link and emulators. (A region in
   another AWS partition than the vended one -- us-east-1 against a
   cn-north-1 credential -- cannot be meant for it, and is ignored.)
4. Vended non-credential settings.

After merging, an S3 store with keys and a region but no endpoint gets an
explicit regional endpoint (``amazonaws.com.cn`` in the China regions, which
object_store's default misses).
"""

from __future__ import annotations

import threading
from typing import Any
from urllib.parse import urlparse

from .errors import InvalidArgumentError

__all__ = [
    "OPT_OUT_PROBE",
    "azure_store_location",
    "canonical_options",
    "cloud_of",
    "commit_refusal",
    "engine_options",
    "environment_options",
    "iceberg_fileio_properties",
    "merge_options",
    "pin_s3_endpoint",
]

#: The storage option that skips the put-if-absent probe for an S3-compatible
#: endpoint (see `commit_refusal`). deltaswamp's own; never sent to a store.
OPT_OUT_PROBE = "deltaswamp_skip_put_if_absent_probe"
_OWN_KEYS = frozenset({OPT_OUT_PROBE})

_S3_SCHEMES = frozenset({"s3", "s3a", "s3n", "r2"})
_AZURE_SCHEMES = frozenset({"az", "abfs", "abfss", "adl", "azure", "wasb", "wasbs"})
_GCS_SCHEMES = frozenset({"gs", "gcs"})

# object_store 0.13's aliases (aws/builder.rs, azure/builder.rs,
# gcp/builder.rs), lower-case alias -> canonical name.
_S3_ALIASES: dict[str, tuple[str, ...]] = {
    "aws_access_key_id": ("access_key_id",),
    "aws_secret_access_key": ("secret_access_key",),
    "aws_session_token": ("aws_token", "session_token", "token"),
    "aws_region": ("region",),
    "aws_default_region": ("default_region",),
    "aws_bucket": ("aws_bucket_name", "bucket_name", "bucket"),
    "aws_endpoint": ("aws_endpoint_url", "endpoint_url", "endpoint"),
    "aws_virtual_hosted_style_request": ("virtual_hosted_style_request",),
    "aws_s3_express": ("s3_express",),
    "aws_imdsv1_fallback": ("imdsv1_fallback",),
    "aws_metadata_endpoint": ("metadata_endpoint",),
    "aws_unsigned_payload": ("unsigned_payload",),
    "aws_checksum_algorithm": ("checksum_algorithm",),
    "aws_role_arn": ("role_arn",),
    "aws_role_session_name": ("role_session_name",),
    "aws_web_identity_token_file": ("web_identity_token_file",),
    "aws_endpoint_url_sts": ("endpoint_url_sts",),
    "aws_skip_signature": ("skip_signature",),
    "aws_copy_if_not_exists": ("copy_if_not_exists",),
    "aws_conditional_put": ("conditional_put",),
    "aws_disable_tagging": ("disable_tagging",),
    "aws_request_payer": ("request_payer",),
    "aws_server_side_encryption": ("server_side_encryption",),
    "aws_sse_kms_key_id": ("sse_kms_key_id",),
    "aws_sse_bucket_key_enabled": ("sse_bucket_key_enabled",),
    "allow_http": ("aws_allow_http",),
}
_AZURE_ALIASES: dict[str, tuple[str, ...]] = {
    "azure_storage_account_key": (
        "azure_storage_access_key",
        "azure_storage_master_key",
        "master_key",
        "account_key",
        "access_key",
    ),
    "azure_storage_account_name": ("account_name",),
    "azure_storage_client_id": ("azure_client_id", "client_id"),
    "azure_storage_client_secret": ("azure_client_secret", "client_secret"),
    "azure_storage_tenant_id": (
        "azure_storage_authority_id",
        "azure_tenant_id",
        "azure_authority_id",
        "tenant_id",
        "authority_id",
    ),
    "azure_storage_sas_key": ("azure_storage_sas_token", "sas_key", "sas_token"),
    "azure_storage_token": ("bearer_token", "token"),
    "azure_storage_use_emulator": ("use_emulator",),
    "azure_storage_endpoint": ("azure_endpoint", "endpoint"),
    "azure_msi_endpoint": ("azure_identity_endpoint", "identity_endpoint", "msi_endpoint"),
    "azure_federated_token_file": ("federated_token_file",),
    "azure_use_azure_cli": ("use_azure_cli",),
    "azure_skip_signature": ("skip_signature",),
    "azure_container_name": ("container_name",),
    "azure_disable_tagging": ("disable_tagging",),
    "azure_use_fabric_endpoint": ("use_fabric_endpoint",),
    "allow_http": ("azure_allow_http",),
}
_GCS_ALIASES: dict[str, tuple[str, ...]] = {
    "google_service_account": (
        "service_account",
        "google_service_account_path",
        "service_account_path",
    ),
    "google_service_account_key": ("service_account_key",),
    "google_application_credentials": ("application_credentials",),
    "google_bucket": ("google_bucket_name", "bucket", "bucket_name"),
    "google_base_url": ("base_url",),
    "google_skip_signature": ("skip_signature",),
    # Not object_store keys: the native store takes a raw OAuth token under
    # these (crates/native/src/store.rs GCS_BEARER_KEYS), and Unity Catalog
    # names it gcp_oauth_token.
    "google_bearer_token": ("gcp_oauth_token", "bearer_token"),
}
# ClientConfigKey (client/mod.rs); each cloud also accepts its own prefix.
_CLIENT_KEYS = frozenset(
    {
        "allow_http",
        "allow_invalid_certificates",
        "connect_timeout",
        "default_content_type",
        "http1_only",
        "http2_only",
        "http2_keep_alive_interval",
        "http2_keep_alive_timeout",
        "http2_keep_alive_while_idle",
        "http2_max_frame_size",
        "pool_idle_timeout",
        "pool_max_idle_per_host",
        "proxy_url",
        "proxy_ca_certificate",
        "proxy_excludes",
        "randomize_addresses",
        "timeout",
        "user_agent",
    }
)
_CLIENT_PREFIX = {"s3": "aws_", "azure": "azure_", "gcs": "google_"}


def _alias_table(aliases: dict[str, tuple[str, ...]]) -> dict[str, str]:
    table = {canonical: canonical for canonical in aliases}
    for canonical, names in aliases.items():
        for name in names:
            table[name] = canonical
    return table


_ALIASES = {
    "s3": _alias_table(_S3_ALIASES),
    "azure": _alias_table(_AZURE_ALIASES),
    "gcs": _alias_table(_GCS_ALIASES),
}

#: Keys that authenticate, per cloud: a vended credential replaces all of them.
_CREDENTIAL_KEYS = {
    "s3": frozenset(
        {
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_role_arn",
            "aws_role_session_name",
            "aws_web_identity_token_file",
            "aws_skip_signature",
        }
    ),
    "azure": frozenset(
        {
            "azure_storage_account_key",
            "azure_storage_client_id",
            "azure_storage_client_secret",
            "azure_storage_tenant_id",
            "azure_storage_sas_key",
            "azure_storage_token",
            "azure_federated_token_file",
            "azure_use_azure_cli",
            "azure_skip_signature",
        }
    ),
    "gcs": frozenset(
        {
            "google_service_account",
            "google_service_account_key",
            "google_application_credentials",
            "google_bearer_token",
            "google_skip_signature",
        }
    ),
}
#: With a vended S3 endpoint, the settings that belong to it.
_S3_ENDPOINT_UNIT = frozenset({"aws_endpoint", "aws_region", "aws_virtual_hosted_style_request"})


def cloud_of(location: str | None) -> str | None:
    """``"s3"``, ``"azure"``, ``"gcs"`` or None for a table location."""
    scheme = (location or "").split("://", 1)[0].lower() if "://" in (location or "") else ""
    if scheme in _S3_SCHEMES:
        return "s3"
    if scheme in _AZURE_SCHEMES:
        return "azure"
    if scheme in _GCS_SCHEMES:
        return "gcs"
    return None


def canonical_key(key: str, cloud: str | None) -> str:
    """object_store's canonical name for `key` on `cloud`; unknown keys verbatim."""
    lowered = key.lower()
    if lowered in _OWN_KEYS:
        return lowered
    if cloud is None:
        return key
    known = _ALIASES[cloud].get(lowered)
    if known is not None:
        return known
    prefix = _CLIENT_PREFIX[cloud]
    bare = lowered[len(prefix) :] if lowered.startswith(prefix) else lowered
    if bare in _CLIENT_KEYS:
        return bare
    # delta-rs's own keys (AWS_S3_LOCKING_PROVIDER, ...) and anything else
    # pass through as written; delta-rs reads some of them upper-case.
    return key


def canonical_options(
    options: dict[str, str] | None, location: str | None, *, what: str = "storage_options"
) -> dict[str, str]:
    """`options` with every known key under its canonical name.

    Two spellings of one setting with the same value collapse; with different
    values the dict is refused, since no answer would be the caller's.
    """
    cloud = cloud_of(location)
    out: dict[str, str] = {}
    spelled: dict[str, str] = {}
    for key, value in (options or {}).items():
        name = canonical_key(str(key), cloud)
        text = str(value)
        if name in out and out[name] != text:
            raise InvalidArgumentError(
                f"{what} sets {name} twice, as {spelled[name]!r} and {key!r}, with different "
                "values; object_store would apply one of them at random. Pass one."
            )
        out[name] = text
        spelled.setdefault(name, str(key))
    return out


def merge_options(
    base: dict[str, str] | None, vended: dict[str, str] | None, location: str | None
) -> dict[str, str]:
    """The caller's options and a vended credential's, merged (module docstring)."""
    cloud = cloud_of(location)
    user = canonical_options(base, location)
    given = canonical_options(vended, location, what="the vended credential")
    if not given:
        return user
    credential_keys = _CREDENTIAL_KEYS.get(cloud or "", frozenset())
    vends_credential = any(k in credential_keys for k in given)
    vended_endpoint = cloud == "s3" and bool(given.get("aws_endpoint"))
    merged = dict(given)
    for key, value in user.items():
        if vends_credential and key in credential_keys:
            continue
        if vended_endpoint and key in _S3_ENDPOINT_UNIT:
            continue
        if (
            key == "aws_region"
            and vends_credential
            and given.get("aws_region")
            and _partition(value) != _partition(given["aws_region"])
        ):
            # Vended keys belong to one AWS partition: a connection-wide
            # us-east-1 cannot take a cn-north-1 credential anywhere it works.
            continue
        merged[key] = value
    return merged


def _partition(region: str) -> str:
    region = region.strip().lower()
    if region.startswith("cn-"):
        return "aws-cn"
    if region.startswith("us-gov-"):
        return "aws-us-gov"
    return "aws"


def pin_s3_endpoint(options: dict[str, str], location: str | None, *, vended: bool) -> None:
    """Give S3 keys an explicit regional endpoint, in place.

    object_store's default endpoint is ``s3.<region>.amazonaws.com``, which does
    not exist for the China regions (``amazonaws.com.cn``): the kernel sent
    every request there. delta-rs, with no endpoint, treats the store as real
    AWS and builds an AWS SDK config via `aws_config::from_env()`, whose region
    chain ignores the `aws_region` we pass and ends at EC2 instance metadata --
    three one-second connect timeouts on the first open in every process, for
    credentials that are already static. So vended keys always get one, and any
    keys in a China region do.
    """
    if cloud_of(location) != "s3":
        return
    region = options.get("aws_region")
    if not region or "aws_access_key_id" not in options or options.get("aws_endpoint"):
        return
    china = region.lower().startswith("cn-")
    if not (vended or china):
        return
    suffix = "amazonaws.com.cn" if china else "amazonaws.com"
    options["aws_endpoint"] = f"https://s3.{region}.{suffix}"


def environment_options(location: str | None, present: dict[str, str]) -> dict[str, str]:
    """The standard ``AWS_*`` environment settings `present` does not already make.

    delta-rs builds its S3 store from the environment (object_store's
    ``from_env``) and then applies the options; the kernel's store took the
    options alone, so ``AWS_ENDPOINT_URL`` or keys exported in the shell
    reached one engine and not the other -- the kernel then read real AWS, or
    nothing. The weakest layer, below everything the caller or a catalog
    passed, and credential keys only when neither passed a credential: a
    session token from the environment paired with the caller's own key is
    another principal's.
    """
    import os

    if cloud_of(location) != "s3":
        return {}
    credential_keys = _CREDENTIAL_KEYS["s3"]
    has_credential = any(k in credential_keys for k in present)
    known = set(_S3_ALIASES) | _CLIENT_KEYS
    out: dict[str, str] = {}
    # Sorted so that two spellings of one setting resolve the same way in
    # every process, as `canonical_options` insists for a caller's dict.
    for name in sorted(os.environ):
        value = os.environ[name]
        if not name.upper().startswith("AWS_") or not value:
            continue
        key = canonical_key(name, "s3")
        if key not in known or key in present or key in out:
            continue
        if has_credential and key in credential_keys:
            continue
        out[key] = value
    return out


def engine_options(
    base: dict[str, str] | None, vended: dict[str, str] | None, location: str | None
) -> dict[str, str]:
    """What an engine hands its object store for `location`."""
    options = merge_options(base, vended, location)
    environment = environment_options(location, options)
    if vended and any(str(k).lower() in _CREDENTIAL_KEYS["s3"] for k in vended):
        # Vended keys belong to the catalog's storage: an AWS_ENDPOINT_URL
        # exported for some other store sent them there instead.
        for key in ("aws_endpoint", "aws_virtual_hosted_style_request"):
            environment.pop(key, None)
    options.update(environment)
    pin_s3_endpoint(options, location, vended=bool(vended))
    return options


def store_options(options: dict[str, str]) -> dict[str, str]:
    """`options` without deltaswamp's own keys, which no store knows."""
    return {k: v for k, v in options.items() if k.lower() not in _OWN_KEYS}


# ------------------------------------------------------------------ Azure URLs

WASB_REMEDY = (
    "wasb:// and wasbs:// are the legacy Hadoop WASB driver's schemes, which neither "
    "engine's object store supports; address the same data as "
    "abfss://<container>@<account>.dfs.<suffix>/<path> (the blob host's dfs sibling)"
)


def location_refusal(location: str | None) -> str | None:
    """Why no engine can open `location` at all, or None."""
    scheme = (location or "").split("://", 1)[0].lower() if "://" in (location or "") else ""
    if scheme in ("wasb", "wasbs"):
        return f"the table's location is {scheme}://; " + WASB_REMEDY
    return None


#: Host suffixes object_store's Azure URL parser recognises in
#: ``abfss://container@account.<suffix>/``. Any other (the sovereign clouds,
#: private-link aliases) fails with "URL did not match any known pattern".
_PARSEABLE_AZURE_SUFFIXES = (
    "dfs.core.windows.net",
    "blob.core.windows.net",
    "dfs.fabric.microsoft.com",
    "blob.fabric.microsoft.com",
)


def azure_store_location(location: str, options: dict[str, str]) -> tuple[str, dict[str, str]]:
    """A location and options delta-rs's Azure store can open.

    object_store parses only the public-cloud and Fabric hosts out of an
    ``abfss://container@account.<host>/path`` URL, so a table on
    ``*.core.chinacloudapi.cn`` or ``*.core.usgovcloudapi.net`` could not be
    opened at all, even with an explicit endpoint. Such a URL is handed to the
    store as ``az://container/path`` with the account named and the endpoint
    set (derived from the host unless the options name one), which addresses
    the same objects. The native store does the same (store.rs).
    """
    parsed = urlparse(location)
    if parsed.scheme.lower() not in ("abfs", "abfss", "az"):
        return location, options
    container, at, host = parsed.netloc.rpartition("@")
    if not at or not container or "." not in host:
        return location, options
    _check_azure_host(location, host, options)
    if host.lower().endswith(_PARSEABLE_AZURE_SUFFIXES):
        return location, options
    from .credentials.databricks import azure_endpoint_for

    out = dict(options)
    out.setdefault("azure_storage_account_name", host.split(".", 1)[0])
    if not _names_azure_endpoint(out):
        endpoint = azure_endpoint_for(location)
        if endpoint is not None:
            out["azure_storage_endpoint"] = endpoint
    return f"az://{container}{parsed.path}", out


def _check_azure_host(location: str, host: str, options: dict[str, str]) -> None:
    """Refuse a location whose host would receive credentials it should not.

    Without an explicit endpoint the store talks to the host the location
    names, with the connection's Azure secret. The location comes from a
    catalog entry or a table log, which others may write, so a host outside
    Azure Storage's domains (`is_azure_storage_host`) -- or another account
    than the one the options name -- is refused unless the caller named the
    endpoint. The native store refuses the same (store.rs).
    """
    from .credentials.databricks import is_azure_storage_host

    if _names_azure_endpoint(options):
        return
    if not is_azure_storage_host(host):
        raise InvalidArgumentError(
            f"{location!r} names the host {host!r}, which is not an Azure Storage domain "
            "(*.core.windows.net, the sovereign clouds, Fabric); storage credentials are "
            "sent there only when you pass it as azure_storage_endpoint"
        )
    lowered = {k.lower(): str(v).strip() for k, v in options.items()}
    named = lowered.get("azure_storage_account_name") or lowered.get("account_name")
    account = host.split(".", 1)[0].lower()
    if named and not host.lower().endswith("fabric.microsoft.com") and named.lower() != account:
        raise InvalidArgumentError(
            f"{location!r} is in the storage account {account!r}, but the connection's "
            f"credentials are for {named!r}; refused rather than send them to another account"
        )


def _names_azure_endpoint(options: dict[str, str]) -> bool:
    # Every alias object_store accepts, case-insensitively (store.rs does the same).
    lowered = {str(k).lower(): v for k, v in options.items()}
    for key in ("azure_storage_endpoint", "azure_endpoint", "endpoint"):
        if str(lowered.get(key) or "").strip():
            return True
    for key in ("azure_storage_use_emulator", "use_emulator"):
        if str(lowered.get(key) or "").strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False


# ------------------------------------------------------------- commit safety

_PROBED: dict[tuple[str, str], bool] = {}
_PROBE_LOCK = threading.Lock()
#: Endpoints known to honour If-None-Match on PUT, so never probed.
_CONDITIONAL_HOSTS = ("amazonaws.com", "amazonaws.com.cn", "r2.cloudflarestorage.com")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def commit_refusal(location: str | None, options: dict[str, str]) -> tuple[str, str] | None:
    """Why no commit may be written to `location` with `options`, as (reason, remedy).

    Every commit, on both engines, is a put-if-absent of ``N.json``; a store
    that ignores the condition lets two writers both "win" version N, and one
    commit is silently lost. On S3 that is object_store's ``If-None-Match: *``,
    which AWS and R2 honour but some S3-compatible stores (older MinIO, Ceph
    RGW, appliances) ignore. So a custom endpoint is probed once per process
    (two put-if-absent calls on a sentinel under ``_delta_log/``), and a store
    that accepts the second put is refused.
    """
    if cloud_of(location) != "s3" or location is None:
        return None
    lowered = {k.lower(): v for k, v in options.items()}
    if lowered.get("aws_s3_locking_provider", "").strip().lower() == "dynamodb":
        return (
            "AWS_S3_LOCKING_PROVIDER=dynamodb is set, but neither engine coordinates "
            "through DynamoDB (delta-rs 1.x dropped S3DynamoDBLogStore and the kernel never "
            "had it); commits here use S3 put-if-absent instead, which a Spark writer on "
            "S3DynamoDBLogStore does not respect, so commits could be lost",
            "remove AWS_S3_LOCKING_PROVIDER when every writer of the table uses conditional "
            "puts (Spark with S3 put-if-absent, delta-rs 1.x, deltaswamp)",
        )
    if lowered.get("aws_conditional_put", "").strip().lower() == "disabled":
        return (
            "aws_conditional_put=disabled turns off S3's put-if-absent, and a Delta commit "
            "without it could silently overwrite another writer's",
            "drop aws_conditional_put (the default, etag, sends If-None-Match)",
        )
    endpoint = options.get("aws_endpoint") or ""
    if not endpoint or _truthy(options.get(OPT_OUT_PROBE)):
        return None
    host = (urlparse(endpoint).hostname or "").lower()
    if host.endswith(_CONDITIONAL_HOSTS):
        return None
    honoured = _probe(location, endpoint, options)
    if honoured is False:
        return (
            f"the S3-compatible store at {endpoint} ignores If-None-Match (a put-if-absent "
            "probe under _delta_log/ succeeded twice), so concurrent commits would silently "
            "overwrite each other",
            "upgrade the store to one that honours conditional PUT (MinIO 2024+, recent Ceph "
            f"RGW), or, if this process is the table's only writer, pass "
            f"storage_options={{'{OPT_OUT_PROBE}': 'true'}}",
        )
    return None


def _probe(location: str, endpoint: str, options: dict[str, str]) -> bool | None:
    """Whether the store honours put-if-absent; None if the probe could not tell."""
    bucket = urlparse(location).netloc
    key = (endpoint.rstrip("/"), bucket)
    with _PROBE_LOCK:
        if key in _PROBED:
            return _PROBED[key]
    try:
        from . import _native
    except ImportError:
        return None
    probe = getattr(_native, "probe_put_if_absent", None)
    if probe is None:
        return None
    try:
        honoured = bool(probe(location, options=store_options(options)))
    except Exception:
        # A credential that cannot write, an unreachable endpoint: the write
        # itself will fail and say why. Unknown is not "unsafe".
        return None
    with _PROBE_LOCK:
        _PROBED[key] = honoured
    return honoured


# --------------------------------------------------------------- other engines

#: object_store option -> PyIceberg FileIO property (pyiceberg.io).
_ICEBERG_FILEIO = {
    "s3": {
        "aws_endpoint": "s3.endpoint",
        "aws_region": "s3.region",
        "aws_access_key_id": "s3.access-key-id",
        "aws_secret_access_key": "s3.secret-access-key",
        "aws_session_token": "s3.session-token",
        "proxy_url": "s3.proxy-uri",
        "connect_timeout": "s3.connect-timeout",
        "timeout": "s3.request-timeout",
    },
    "azure": {
        "azure_storage_account_name": "adls.account-name",
        "azure_storage_account_key": "adls.account-key",
        "azure_storage_sas_key": "adls.sas-token",
        "azure_storage_tenant_id": "adls.tenant-id",
        "azure_storage_client_id": "adls.client-id",
        "azure_storage_client_secret": "adls.client-secret",
    },
    "gcs": {
        "google_base_url": "gcs.service.host",
        "google_bearer_token": "gcs.oauth2.token",
    },
}


def iceberg_fileio_properties(options: dict[str, str] | None) -> dict[str, str]:
    """The PyIceberg FileIO properties that carry `options`' meaning.

    The cloud is not known until a table is loaded, so each key is tried
    against every cloud; the names do not overlap. Durations are object_store
    strings (``"5s"``); PyIceberg takes seconds, so only a bare number or an
    ``s`` suffix is translated, and anything else is left out rather than
    guessed.
    """
    out: dict[str, str] = {}
    for cloud, table in _ICEBERG_FILEIO.items():
        for key, value in canonical_options(options, f"{_SCHEME_OF[cloud]}://x").items():
            name = table.get(key)
            if name is None:
                continue
            if key in ("connect_timeout", "timeout"):
                seconds = _seconds(value)
                if seconds is None:
                    continue
                value = seconds
            out[name] = value
    return out


_SCHEME_OF = {"s3": "s3", "azure": "abfss", "gcs": "gs"}


def _seconds(value: str) -> str | None:
    text = value.strip().lower()
    if text.endswith("s") and not text.endswith("ms"):
        text = text[:-1].strip()
    try:
        return str(float(text))
    except ValueError:
        return None


def http_settings(options: dict[str, str] | None) -> dict[str, Any]:
    """The proxy and timeout among `options`, for engines that fetch over plain HTTP."""
    out: dict[str, Any] = {}
    lowered = {str(k).lower(): str(v) for k, v in (options or {}).items()}
    for prefix in ("", "aws_", "azure_", "google_"):
        proxy = lowered.get(f"{prefix}proxy_url")
        if proxy and "proxy_url" not in out:
            out["proxy_url"] = proxy
        timeout = lowered.get(f"{prefix}timeout")
        if timeout and "timeout" not in out:
            seconds = _seconds(timeout)
            if seconds is not None:
                out["timeout"] = float(seconds)
    return out


def write_refusal(
    location: str | None, base: dict[str, str] | None, provider: Any
) -> tuple[str, str] | None:
    """`commit_refusal` for a capability check, which must not vend.

    Uses the connection's options and a held (static) credential only; a
    catalog that vends on demand points at AWS, R2, Azure or GCS anyway.
    """
    if cloud_of(location) != "s3":
        return None
    vended = None
    peek = getattr(provider, "peek", None)
    if callable(peek):
        try:
            vended = dict(peek().secrets)
        except Exception:
            vended = None
    try:
        options = engine_options(base, vended, location)
    except InvalidArgumentError:
        return None  # the operation itself reports the conflicting keys
    return commit_refusal(location, options)


# ---------------------------------------------------------- S3 bucket regions

#: Buckets' regions as S3 reported them, per process.
_BUCKET_REGIONS: dict[str, str | None] = {}
_BUCKET_REGIONS_LOCK = threading.Lock()

#: Set to 0 to never ask S3 where a bucket is (air-gapped hosts, tests).
BUCKET_REGION_PROBE_ENV = "DELTASWAMP_S3_REGION_PROBE"

#: Where the region is asked: S3's global endpoint answers for every bucket.
S3_REGION_ENDPOINT = "https://s3.amazonaws.com"


def s3_bucket_region(location: str | None, *, timeout: float = 3.0) -> str | None:
    """The AWS region of the bucket `location` is in, or None if S3 will not say.

    Unity Catalog vends S3 keys without a region, and the catalog's region is
    the *metastore's*: an external table in a bucket elsewhere got the wrong
    one, and every request failed with a redirect that names no region
    object_store can follow. S3 answers an anonymous ``HEAD`` of any bucket
    with its region in ``x-amz-bucket-region`` (even when it refuses access),
    so one request per bucket per process settles it.
    """
    import os

    if os.environ.get(BUCKET_REGION_PROBE_ENV, "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    parsed = urlparse(location or "")
    if parsed.scheme.lower() not in ("s3", "s3a", "s3n") or not parsed.netloc:
        return None
    bucket = parsed.netloc.split("@")[-1]
    with _BUCKET_REGIONS_LOCK:
        if bucket in _BUCKET_REGIONS:
            return _BUCKET_REGIONS[bucket]
    import urllib.error
    import urllib.parse
    import urllib.request

    region: str | None = None
    request = urllib.request.Request(
        f"{S3_REGION_ENDPOINT}/{urllib.parse.quote(bucket, safe='')}", method="HEAD"
    )

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args: Any, **kwargs: Any) -> None:
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            region = response.headers.get("x-amz-bucket-region")
    except urllib.error.HTTPError as exc:
        region = exc.headers.get("x-amz-bucket-region") if exc.headers else None
    except Exception:
        return None  # unreachable: not remembered, asked again next time
    region = region.strip() if region else None
    with _BUCKET_REGIONS_LOCK:
        _BUCKET_REGIONS[bucket] = region or None
    return region or None
