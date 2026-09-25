"""Fixtures shared by the integration tests."""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers import direct_router


@pytest.fixture
def conn() -> Any:
    """A path-table connection over the real kernel and delta-rs engines."""
    from deltaswamp import Connection
    from deltaswamp.catalog.filesystem import FilesystemCatalog

    return Connection(catalog=FilesystemCatalog(), router=direct_router())
