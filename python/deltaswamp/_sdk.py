"""Building Databricks SDK clients that identify this library.

The Unity Catalog Delta API refuses any request whose User-Agent does not name
the calling application::

    The UC Delta API requires clients to identify the calling application in the
    User-Agent header. The provided User-Agent 'unknown/0.0.0 databricks-sdk-py/...'
    is insufficient.

That is a 400 on `create_staging_table` and on every `/delta/v1` commit, so
without this stamp managed-table creation and every catalog-managed write fail
against a real workspace -- the two things this library exists to do. The SDK
sends `unknown/0.0.0` unless a product is set, and nothing else in the stack
sets one.

The stamp goes on the `Config`, never through `useragent.with_product`, which is
global process state: an application embedding deltaswamp keeps its own product
identity, and only the clients built here are relabelled. A caller who passes an
explicit `Config` with a product already set keeps it.
"""

from __future__ import annotations

import contextlib
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
            with contextlib.suppress(Exception):  # a Config that refuses the attribute
                config._product_info = (PRODUCT, sdk_version())
        return WorkspaceClient(config=config)

    # An explicit `product=None` (a caller forwarding optional arguments) must
    # not erase the stamp: the SDK would send `unknown/0.0.0` and every UC
    # Delta API call would 400.
    given = {
        k: v for k, v in kwargs.items() if not (k in ("product", "product_version") and v is None)
    }
    return WorkspaceClient(**{**product_kwargs(), **given})
