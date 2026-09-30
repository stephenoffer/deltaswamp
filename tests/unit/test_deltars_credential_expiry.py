"""delta-rs holds a vended credential for a whole operation: say so when it is short."""

from __future__ import annotations

import time
import warnings
from typing import Any

import pytest
from deltaswamp.credentials import Credentials
from deltaswamp.credentials.base import Cloud
from deltaswamp.errors import CredentialExpiryWarning

pytest.importorskip("deltalake")


def _credentials(expires_in: float | None) -> Credentials:
    return Credentials(
        cloud=Cloud.LOCAL,
        url="file:///x",
        expires_at=None if expires_in is None else time.time() + expires_in,
        secrets={"token": "t"},
    )


def _warned(engine: Any, credentials: Credentials) -> list[Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine._warn_if_short_lived(credentials)
    return [w for w in caught if issubclass(w.category, CredentialExpiryWarning)]


def test_a_short_lived_credential_warns_and_a_long_one_does_not() -> None:
    from deltaswamp.engine.deltars import DeltaRsEngine

    engine = DeltaRsEngine()
    assert _warned(engine, _credentials(60.0))
    assert not _warned(engine, _credentials(3600.0))
    assert not _warned(engine, _credentials(None))


def test_the_threshold_is_configurable() -> None:
    from deltaswamp.engine.deltars import DeltaRsEngine

    engine = DeltaRsEngine()
    engine.expiry_warning_seconds = 30.0
    assert not _warned(engine, _credentials(60.0))
