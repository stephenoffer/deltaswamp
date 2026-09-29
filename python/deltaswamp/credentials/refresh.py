"""Vended credentials that outlive the object store built with them.

The native object store was built from static options, so a snapshot (cached
for a whole job), a long read or a long write held the credential it was built
with, and failed partway through with a storage 403 once the vended token
(about an hour) expired.

Now a store built for a vended credential names a *credential slot*
(``SLOT_KEY`` in its options). The slot lives in the native extension, one
per credential identity and operation, and the store reads its credential
from it on every request. This module keeps each slot fresh: a daemon thread
asks the provider for a new credential ahead of expiry and publishes it. The
store itself never calls into Python, so a request never waits on the GIL.

A slot's identity is the provider's ``credential_identity`` when it has one --
the same principal and table from every unpickled copy of a provider -- so N
read tasks in one worker process share one slot, one refresh schedule and one
cached snapshot, instead of each vending its own.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any

from .base import Credentials, Operation

__all__ = ["SLOT_KEY", "slot_identity", "slot_options"]

#: The storage option naming a slot. Only the kernel's store reads it.
SLOT_KEY = "deltaswamp_credential_slot"

#: Refresh this far ahead of expiry (capped at half the credential's life).
REFRESH_AHEAD_SECONDS = 300.0

#: A slot nobody has asked for in this long stops being refreshed: a provider
#: kept alive by a cache in a long-lived process otherwise vended every hour
#: forever. A read or write that runs longer than this is not expected.
IDLE_SECONDS = 12 * 3600.0

#: Retry a failed refresh after this long (at most), while the credential
#: still has life left.
RETRY_SECONDS = 30.0


def slot_identity(provider: Any, operation: Operation) -> str:
    """The slot a provider's credential for `operation` is published in."""
    identity = None
    get = getattr(provider, "credential_identity", None)
    if callable(get):
        try:
            identity = get()
        except Exception:
            identity = None
    if not identity:
        identity = f"object-{id(provider):x}"
    return f"{identity}:{operation.value}:{os.getpid()}"


@dataclass
class _Entry:
    provider: Any  # a weak reference, or the provider when it cannot be one
    operation: Operation
    expires_at: float | None
    published_at: float
    last_used: float
    due: float
    options: dict[str, str]

    def live(self) -> Any:
        return self.provider() if isinstance(self.provider, weakref.ref) else self.provider


class _Refresher:
    """The slots this process publishes, and the thread that keeps them fresh."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._entries: dict[str, _Entry] = {}
        self._thread: threading.Thread | None = None
        self._pid = os.getpid()

    def _native(self) -> Any:
        from .. import _native

        return _native

    def available(self) -> bool:
        try:
            return "credential_slots" in getattr(self._native(), "FEATURES", ())
        except Exception:
            return False

    def track(
        self, provider: Any, operation: Operation, credentials: Credentials, options: dict[str, str]
    ) -> str:
        """Publish `options` in the provider's slot and keep it fresh; its id."""
        now = time.time()
        slot = slot_identity(provider, operation)
        with self._cond:
            if self._pid != os.getpid():
                # A forked child: the parent's thread did not come along.
                self._entries.clear()
                self._thread = None
                self._pid = os.getpid()
            entry = self._entries.get(slot)
            if entry is None or entry.live() is None:
                try:
                    ref: Any = weakref.ref(provider)
                except TypeError:
                    ref = provider
                entry = self._entries[slot] = _Entry(
                    provider=ref,
                    operation=operation,
                    expires_at=None,
                    published_at=now,
                    last_used=now,
                    due=now,
                    options={},
                )
            entry.last_used = now
            if options != entry.options:
                self._publish(slot, entry, credentials, options, now)
            self._start()
            self._cond.notify_all()
        return slot

    def _publish(
        self,
        slot: str,
        entry: _Entry,
        credentials: Credentials,
        options: dict[str, str],
        now: float,
    ) -> None:
        self._native().set_credential_slot(slot, dict(options), credentials.expires_at)
        entry.options = dict(options)
        entry.expires_at = credentials.expires_at
        entry.published_at = now
        entry.due = self._due(entry, now)

    @staticmethod
    def _due(entry: _Entry, now: float) -> float:
        if entry.expires_at is None:
            return float("inf")
        life = max(0.0, entry.expires_at - entry.published_at)
        ahead = min(REFRESH_AHEAD_SECONDS, life / 2.0)
        return max(now + 1.0, entry.expires_at - ahead)

    def _start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="deltaswamp-credential-refresh", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            with self._cond:
                if self._pid != os.getpid():
                    return
                now = time.time()
                due = [(slot, e) for slot, e in self._entries.items() if e.due <= now]
                if not due:
                    upcoming = min((e.due for e in self._entries.values()), default=float("inf"))
                    self._cond.wait(timeout=min(60.0, max(0.5, upcoming - now)))
                    continue
            for slot, entry in due:
                self._refresh(slot, entry)

    def _refresh(self, slot: str, entry: _Entry) -> None:
        from .base import Credentials as _Credentials

        now = time.time()
        provider = entry.live()
        if provider is None or now - entry.last_used > IDLE_SECONDS:
            self._drop(slot)
            return
        try:
            fresh = provider.credentials(entry.operation)
            if not isinstance(fresh, _Credentials):
                raise TypeError(type(fresh).__name__)
            options = fresh.as_storage_options()
        except Exception:
            # Vending failed (throttled, revoked): try again while the current
            # credential still has life; the store keeps serving it.
            with self._cond:
                left = (entry.expires_at or now) - now
                entry.due = now + min(RETRY_SECONDS, max(1.0, left / 4.0))
            return
        with self._cond:
            if options != entry.options:
                self._publish(slot, entry, fresh, options, now)
            else:
                # The provider still serves the same one (its own margin is
                # smaller than ours): ask again shortly.
                left = (entry.expires_at or now) - now
                entry.due = now + min(RETRY_SECONDS, max(1.0, left / 4.0))

    def _drop(self, slot: str) -> None:
        with self._cond:
            self._entries.pop(slot, None)
        with contextlib.suppress(Exception):
            self._native().remove_credential_slot(slot)

    def entries(self) -> dict[str, _Entry]:
        """The tracked slots (for tests and diagnostics)."""
        with self._cond:
            return dict(self._entries)


_REFRESHER = _Refresher()


def slot_options(
    provider: Any, operation: Operation, credentials: Credentials, options: dict[str, str]
) -> dict[str, str]:
    """`options` (a vended credential's) naming a slot the store refreshes from.

    Unchanged when the credential never expires, or the extension predates
    slots: there is nothing to refresh, or nothing to refresh it through.
    """
    if credentials.expires_at is None or not _REFRESHER.available():
        return options
    slot = _REFRESHER.track(provider, operation, credentials, options)
    return {**options, SLOT_KEY: slot}
