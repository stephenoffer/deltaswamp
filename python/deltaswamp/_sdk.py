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


#: Name parts that make a field's value a secret, whatever the separators:
#: ``secret_access_key``, ``s3.secret-access-key``, ``adls.sas-token.<host>``,
#: ``gcs.oauth2.token``, ``aad_token``, ``client_secret``, ``password``.
_SECRET_PARTS = frozenset(
    {"secret", "token", "password", "passwd", "pwd", "signature", "sig", "sas", "authorization"}
)
#: ``<qualifier>_key`` names that hold key material, not a key's name.
_KEY_QUALIFIERS = frozenset({"access", "account", "private", "secret", "sas", "storage", "api"})
#: Named explicitly: an identifier that, with the secret, is a credential.
_SECRET_NAMES = frozenset({"access_key_id", "accesskeyid"})


def _is_secret_key(name: str) -> bool:
    import re

    lowered = name.strip().lower()
    parts = [p for p in re.split(r"[^a-z0-9]+", lowered) if p]
    if lowered.replace("-", "_").replace(".", "_") in _SECRET_NAMES or "access_key_id" in lowered:
        return True
    if any(p in _SECRET_PARTS or p.endswith(("token", "secret", "password")) for p in parts):
        return True
    return any(
        p == "key" and i > 0 and parts[i - 1] in _KEY_QUALIFIERS for i, p in enumerate(parts)
    )


class _RedactSecrets:
    """A `logging.Filter` that blanks secrets in the SDK's debug log.

    databricks-sdk logs every request and response at DEBUG, masking only a
    few field names of its own. The credential responses carry storage
    secrets in fields it does not mask -- hyphenated in the UC Delta API
    (``s3.secret-access-key``, ``adls.sas-token.<host>``), ``aad_token`` in
    an Azure answer -- and ``debug_headers=True`` logs the Authorization
    header. So any field whose name marks a secret (`_is_secret_key`), any
    Authorization / Cookie header line, and presigned-URL signatures are
    blanked.
    """

    def __init__(self) -> None:
        import re

        self._fields = re.compile(
            r"""(["'])([^"'\n]{1,200})\1(\s*:\s*)(["'])((?:\\.|(?!\4).)*)\4"""
        )
        self._headers = re.compile(
            r"(?im)^((?:[^\S\n]|[<>*])*(?:proxy-)?(?:authorization|cookie|set-cookie|"
            r"x-databricks-[a-z-]*token)[^\S\n]*:[^\S\n]*).+$"
        )
        self._query = re.compile(
            r"(?i)([?&](?:sig|signature|x-amz-signature|x-amz-security-token|"
            r"x-goog-signature|x-ms-signature)=)[^&\s\"']+"
        )

    def _field(self, match: Any) -> str:
        quote, name, sep, vquote, _value = match.groups()
        if not _is_secret_key(name):
            return str(match.group(0))
        return f"{quote}{name}{quote}{sep}{vquote}**REDACTED**{vquote}"

    def redact(self, message: str) -> str:
        message = self._fields.sub(self._field, message)
        message = self._headers.sub(r"\1**REDACTED**", message)
        return self._query.sub(r"\1**REDACTED**", message)

    def filter(self, record: Any) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = self.redact(message)
        if redacted != message:
            record.msg, record.args = redacted, None
        return True


_NAMESPACE = "databricks.sdk"


def _install_log_redaction() -> None:
    """Redact every record of the SDK's loggers, once per process.

    A filter on the ``databricks.sdk`` logger does not see records of its
    children (``databricks.sdk.oauth``, ...): logging runs only the emitting
    logger's filters. So the filter also runs from the log-record factory,
    for records of the SDK's namespace only, and wraps whatever factory is
    installed.
    """
    import logging

    logger = logging.getLogger(_NAMESPACE)
    redactor = next((f for f in logger.filters if isinstance(f, _RedactSecrets)), None)
    if redactor is None:
        redactor = _RedactSecrets()
        logger.addFilter(redactor)
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_deltaswamp_redacts", False):
        return

    def factory(*args: Any, **kwargs: Any) -> Any:
        record = previous(*args, **kwargs)
        name = getattr(record, "name", "")
        if name == _NAMESPACE or str(name).startswith(_NAMESPACE + "."):
            redactor.filter(record)
        return record

    factory._deltaswamp_redacts = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


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
