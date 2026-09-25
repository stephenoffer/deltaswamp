"""Credential vending, refresh, and rendering.

Two things here are load-bearing and easy to get wrong.

**There are two independent clocks.** The catalog token (OAuth M2M against
``/oidc/v1/token``) and the vended *storage* credential (~1h, carries its own
``expiration_time``) expire separately. Refreshing the first does not refresh the
second; that is the mechanism behind delta-rs#4628, where a `uc://` table dies
after about an hour with ``The provided token has expired.`` even though the
catalog session is perfectly healthy.

**Providers travel to workers; credentials do not.** A `Credentials` object is a
snapshot frozen at submission time -- shipping one to a Ray worker puts a
plaintext token in task payloads and logs, and guarantees mid-job expiry on a
long read. So `CredentialProvider` is the picklable unit, and each worker vends
its own. Databricks' `storage_options` being `dict[str, str]` means a token is
the *only* thing you could otherwise send, which is exactly the trap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ..errors import CredentialError

__all__ = [
    "Cloud",
    "CredentialProvider",
    "Credentials",
    "Operation",
]

# Refresh this far ahead of stated expiry. A vended credential that expires
# mid-request fails the request, so we never run to the wire.
DEFAULT_REFRESH_MARGIN_SECONDS = 300.0


class Cloud(StrEnum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    R2 = "r2"


class Operation(StrEnum):
    """The `operation` field of the UC temporary-credentials request."""

    READ = "READ"
    READ_WRITE = "READ_WRITE"


@dataclass(frozen=True, slots=True)
class Credentials:
    """One vended credential set, scoped to one table.

    `expires_at` is an absolute wall-clock deadline (epoch seconds). Databricks
    never publishes a TTL for these -- the response's `expiration_time` is the
    only authority -- so nothing here assumes one hour.
    """

    cloud: Cloud
    url: str
    expires_at: float | None
    secrets: dict[str, str] = field(default_factory=dict)
    # Azure user-delegation SAS is scoped to a *path*, not a container, so the
    # store registry must key on this. Reusing one table's SAS for a sibling
    # table in the same container yields 403 AuthenticationFailed (delta-rs#4425).
    scope_prefix: str | None = None
    table_id: str | None = None

    def expires_within(self, seconds: float = DEFAULT_REFRESH_MARGIN_SECONDS) -> bool:
        """True if this credential is gone, or will be within `seconds`."""
        if self.expires_at is None:
            return False
        return time.time() + seconds >= self.expires_at

    @property
    def is_expired(self) -> bool:
        return self.expires_within(0.0)

    def registry_key(self) -> tuple[str, str]:
        """The key a store registry must use.

        Keyed by (table identity, path prefix) rather than (scheme, bucket):
        see the note on `scope_prefix`.
        """
        return (self.table_id or "", self.scope_prefix or self.url)

    def as_storage_options(self) -> dict[str, str]:
        """Render for delta-rs `storage_options` (string-valued only)."""
        opts = dict(self.secrets)
        # Never rely on account-name inference for the Azure endpoint: it
        # silently breaks Azurite, private-link DNS, and sovereign clouds
        # (.chinacloudapi.cn, .usgovcloudapi.net). See ClickHouse#115098.
        if self.cloud is Cloud.AZURE and "azure_endpoint" not in opts:
            raise CredentialError(
                "Azure credentials must carry an explicit azure_endpoint; "
                "account-name inference breaks Azurite, private-link and "
                "sovereign-cloud endpoints"
            )
        return opts

    def redacted(self) -> dict[str, str]:
        """Safe for logs and error messages."""
        return {k: "***" for k in self.secrets}

    def __repr__(self) -> str:  # never let a token reach a traceback
        exp = "never" if self.expires_at is None else f"{self.expires_at:.0f}"
        return (
            f"Credentials(cloud={self.cloud.value}, url={self.url!r}, "
            f"expires_at={exp}, keys={sorted(self.secrets)})"
        )


@runtime_checkable
class CredentialProvider(Protocol):
    """Mints short-lived credentials for one table, on demand.

    Implementations MUST be picklable: this object, not its output, is what
    travels to distributed workers. Concretely that means holding configuration
    (workspace URL, auth mode, table id) and constructing any HTTP client lazily
    on first use, rather than capturing one at build time. A `__getstate__` that
    drops live clients is the usual way.
    """

    @property
    def table_id(self) -> str | None:
        """The UC table id these credentials are vended against, if known."""
        ...

    def credentials(self, operation: Operation = Operation.READ) -> Credentials:
        """Return valid credentials, vending or re-vending as needed.

        Must be cheap on the happy path -- callers invoke it per request and
        rely on the implementation to cache until `expires_within()`.
        """
        ...

    def invalidate(self) -> None:
        """Discard any cached credential, forcing a fresh vend.

        Called when storage returns 403/ExpiredToken despite a cached credential
        that still looks valid, which happens when clocks disagree.
        """
        ...


class StaticCredentialProvider:
    """Serves one already-vended credential, for a single short operation.

    Used where the catalog vends a credential for a *path* rather than a table
    -- creating an external table, or reading a log before registering it --
    so there is no table identity to re-vend against. It is not
    refreshable: past `expires_at` it raises rather than handing out a dead
    credential that fails later with a storage 403.
    """

    def __init__(self, credentials: Credentials) -> None:
        self._credentials = credentials

    @property
    def table_id(self) -> str | None:
        return self._credentials.table_id

    def credentials(self, operation: Operation = Operation.READ) -> Credentials:
        if self._credentials.is_expired:
            raise CredentialError(
                "the path credential has expired, and a path credential cannot be re-vended "
                "from here; retry the operation to obtain a fresh one"
            )
        return self._credentials

    def invalidate(self) -> None:
        """Nothing to refresh; the next call re-checks expiry."""

    def peek(self) -> Credentials:
        """The held credential, without the expiry check."""
        return self._credentials

    def __repr__(self) -> str:
        return f"StaticCredentialProvider({self._credentials!r})"
