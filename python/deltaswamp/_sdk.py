"""Databricks SDK clients that identify this library in their User-Agent.

The Unity Catalog Delta API rejects clients whose User-Agent names no
application (the SDK default is `unknown/0.0.0`). The product is set per
`Config`, not through the global `useragent.with_product`, so an application
embedding deltaswamp keeps its own identity.
"""

from __future__ import annotations

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


def workspace_client(*, config: Any = None, **kwargs: Any) -> Any:
    """A `WorkspaceClient` carrying this library's product identity.

    `config` is an explicit `databricks.sdk.core.Config`; anything else is
    passed through as connection keyword arguments.
    """
    from databricks.sdk import WorkspaceClient

    if config is not None:
        if not _has_product(config):
            # Stamping the object rather than rebuilding it keeps whatever auth
            # state the caller already resolved.
            config._product_info = (PRODUCT, sdk_version())
        return WorkspaceClient(config=config)

    return WorkspaceClient(**{**product_kwargs(), **kwargs})
