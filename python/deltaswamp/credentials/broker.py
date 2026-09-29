"""Driver-side credential vending for distributed workers.

A plan pickled for workers carries the storage credential vended on the driver
and no catalog credentials (`distributed.ShippedCredentials`), so a job that
outlives that credential (about an hour) failed on its workers. Shipping the
catalog's own credentials (``ship_catalog_auth=True``) fixes that by putting a
PAT or client secret in every task payload.

A `CredentialBroker` is the third way: it keeps the catalog credentials where
they are and vends on request. Give a plan a *credential source* -- any
picklable callable ``source(table_id, operation) -> dict`` that reaches a
broker -- and workers ask it for a fresh storage credential ahead of expiry.
Workers in one process share each answer, so a job asks once per refresh per
process, not once per task.

deltaswamp does not depend on Ray. With Ray, host the broker in an actor and
make the source a small picklable wrapper around the actor handle::

    Broker = ray.remote(CredentialBroker)
    broker = Broker.remote()
    ray.get(broker.add.remote(CredentialBroker.portable(table)))

    class Source:
        def __init__(self, broker): self.broker = broker
        def __call__(self, table_id, operation):
            return ray.get(self.broker.vend.remote(table_id, operation))

    plan = table.plan_scan(credential_source=Source(broker))

In one process (threads, tests) the broker itself is the source.
"""

from __future__ import annotations

import threading
from typing import Any

from ..errors import CredentialError
from .base import Credentials, Operation

__all__ = ["CredentialBroker"]


def _provider_of(table: Any) -> Any:
    """The credential provider of a Table, a ResolvedTable, a plan or a provider.

    A plan's is its table's: for a write that creates its table (a
    `Connection.plan_write` of a name not there yet), the only place the
    provider of the table-to-be lives.
    """
    planned = getattr(table, "table", None)
    if hasattr(planned, "credential_provider"):
        table = planned
    resolved = getattr(table, "_resolved", table)
    return getattr(resolved, "credential_provider", resolved)


def _key_of(provider: Any) -> str | None:
    """What workers ask for `provider`'s table by: its id, else its own key."""
    key = getattr(provider, "table_id", None) or getattr(provider, "credential_key", None)
    return str(key) if key else None


class CredentialBroker:
    """Vends storage credentials for registered tables, by UC table id."""

    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}
        self._lock = threading.Lock()

    @staticmethod
    def portable(table: Any) -> Any:
        """`table`'s credential provider, picklable *with* its catalog secrets.

        For registering with a broker in another process (an actor); a
        provider pickles no literal secret by default.
        """
        from .databricks import shipping

        provider = _provider_of(table)
        if provider is None:
            raise CredentialError("this table has no credential provider: nothing to broker")
        portable = getattr(provider, "portable", None)
        if callable(portable):
            return portable()
        return shipping(provider)

    def add(self, table: Any) -> str:
        """Serve `table`'s credentials (a Table, a ResolvedTable, a plan or a provider).

        Returns the key workers ask by: the UC table id, or for a table a
        planned write creates, the key its provider names (its location).
        """
        provider = _provider_of(table)
        key = _key_of(provider) if provider is not None else None
        if provider is None or not key:
            raise CredentialError(
                "only a catalog table with a table id (or a table a planned write "
                "creates) can be brokered; this one has no credential provider or no id"
            )
        with self._lock:
            self._providers[key] = provider
        return key

    def vend(self, table_id: str, operation: str = Operation.READ.value) -> dict[str, Any]:
        """A fresh storage credential for `table_id`, as plain data."""
        with self._lock:
            provider = self._providers.get(str(table_id))
        if provider is None:
            raise CredentialError(
                f"this credential broker serves no table {table_id!r}; add() it on the driver"
            )
        credentials: Credentials = provider.credentials(Operation(str(operation).upper()))
        return credentials._state()

    __call__ = vend

    def __reduce__(self) -> Any:
        # A broker holds live catalog credentials; it is meant to stay put
        # (the driver, or an actor) and be reached through a source.
        raise TypeError(
            "a CredentialBroker is not picklable: keep it on the driver or in an actor "
            "and give plans a credential source that reaches it"
        )
