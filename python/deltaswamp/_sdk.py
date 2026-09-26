"""Databricks SDK clients that identify this library in their User-Agent.

The Unity Catalog Delta API rejects clients whose User-Agent names no
application (the SDK default is `unknown/0.0.0`). The product is set per
`Config`, not through the global `useragent.with_product`, so an application
embedding deltaswamp keeps its own identity.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["PRODUCT", "product_kwargs", "sdk_version", "workspace_client"]

#: The application name sent to Databricks. Alphanumeric: the SDK validates it.
PRODUCT = "deltaswamp"


def sdk_version() -> str:
    """This library's version, as the SDK's semver validator wants it."""
    from . import __version__

    return __version__


def product_kwargs() -> dict[str, str]:
    """`product`/`product_version` for a `Config` or `WorkspaceClient`."""
    return {"product": PRODUCT, "product_version": sdk_version()}


def _has_product(config: Any) -> bool:
    """Whether an explicit Config already names a product worth keeping."""
    info = getattr(config, "_product_info", None)
    if not info:
        return False
    name = info[0] if isinstance(info, (tuple, list)) and info else None
    return bool(name) and name != "unknown"


#: JSON keys whose values are secrets in the credential-vending responses
#: (and in OAuth token responses) that databricks-sdk logs at DEBUG.
_SECRET_KEYS = (
    "secret_access_key",
    "session_token",
    "access_key_id",
    "sas_token",
    "oauth_token",
    "access_token",
    "refresh_token",
    "id_token",
    "client_secret",
)


class _RedactSecrets:
    """A `logging.Filter` that blanks vended secrets in the SDK's debug log.

    databricks-sdk logs every response body at DEBUG, masking only a few
    field names of its own. The temporary-credentials responses carry the
    storage secret key and session token in fields it does not mask, so
    turning on DEBUG to chase a routing problem wrote them to the log.
    """

    def __init__(self) -> None:
        import re

        keys = "|".join(_SECRET_KEYS)
        self._pattern = re.compile(rf'("(?:{keys})"\s*:\s*)"[^"]*"')

    def filter(self, record: Any) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = self._pattern.sub(r'\1"**REDACTED**"', message)
        if redacted != message:
            record.msg, record.args = redacted, None
        return True


def _install_log_redaction() -> None:
    """Attach `_RedactSecrets` to the SDK's logger, once per process."""
    import logging

    logger = logging.getLogger("databricks.sdk")
    if not any(isinstance(f, _RedactSecrets) for f in logger.filters):
        logger.addFilter(_RedactSecrets())


_PROXY_VARIABLES = ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")


def _check_resolvable(host: Any) -> None:
    """Refuse a workspace host whose name does not resolve, before the SDK tries.

    The SDK retries a failed connection for its whole retry budget (five
    minutes by default), starting in the Config constructor, and then
    reports "Timed out after 0:05:00", which deltaswamp read as throttling.
    A DNS lookup answers the common case -- a mistyped host -- in
    milliseconds. Behind a proxy the local resolver may not know the name,
    so the check is skipped there.
    """
    if not isinstance(host, str) or not host.strip():
        return
    if any(os.environ.get(v) for v in _PROXY_VARIABLES):
        return
    import socket
    import urllib.parse

    text = host.strip()
    parsed = urllib.parse.urlparse(text if "://" in text else f"https://{text}")
    name = parsed.hostname
    if not name or "." not in name:
        # A single-label name (localhost, a test double's host) is left to
        # the SDK; every workspace host is a dotted DNS name.
        return
    try:
        socket.getaddrinfo(name, parsed.port or 443)
    except socket.gaierror as exc:
        from .errors import PreflightError

        raise PreflightError(
            f"cannot reach the Databricks host {text}: its name does not resolve ({exc}). "
            "Check the host (host=, DATABRICKS_HOST or the profile) and the network."
        ) from exc
    except (OSError, UnicodeError):
        return


def workspace_client(
    *,
    config: Any = None,
    profile: str | None = None,
    host: str | None = None,
    token: str | None = None,
    **kwargs: Any,
) -> Any:
    """A `WorkspaceClient` carrying this library's product identity.

    `config` is an explicit `databricks.sdk.core.Config`. Otherwise `profile`,
    `host`, `token` and any other connection arguments are passed through,
    skipping the ones left as None.
    """
    from databricks.sdk import WorkspaceClient

    _install_log_redaction()
    if config is not None:
        if not _has_product(config):
            # Stamping the object rather than rebuilding it keeps whatever auth
            # state the caller already resolved.
            config._product_info = (PRODUCT, sdk_version())
        return WorkspaceClient(config=config)

    given = {"profile": profile, "host": host, "token": token}
    kwargs.update({k: v for k, v in given.items() if v})
    # An explicit `product=None` (a caller forwarding optional arguments) must
    # not erase the stamp: the SDK would send `unknown/0.0.0` and every UC
    # Delta API call would 400.
    kwargs = {
        k: v for k, v in kwargs.items() if not (k in ("product", "product_version") and v is None)
    }
    kwargs = {**product_kwargs(), **kwargs}
    _check_resolvable(
        kwargs.get("host") or (None if kwargs.get("profile") else os.environ.get("DATABRICKS_HOST"))
    )
    import inspect

    accepted = inspect.signature(WorkspaceClient.__init__).parameters
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values())
    if not takes_any and any(k not in accepted for k in kwargs):
        # A pickled catalog or provider carries every plain attribute of the
        # driver's resolved Config, some of which (discovery_url,
        # retry_timeout_seconds, ...) WorkspaceClient() does not take as an
        # argument, though Config() does.
        from databricks.sdk.core import Config

        return WorkspaceClient(config=Config(**kwargs))
    return WorkspaceClient(**kwargs)
