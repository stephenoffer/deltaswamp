"""Fixtures and environment gating for the live Databricks suite.

The live tests authenticate with a Databricks personal access token. They are
opt-in, they create and drop tables in a schema you nominate, and they cost
money to run, so nothing here fires by accident.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass(frozen=True)
class LiveConfig:
    """What the live suite needs to reach a real workspace."""

    host: str
    token: str
    catalog: str
    schema: str
    warehouse_id: str | None
    external_location: str | None

    @property
    def prefix(self) -> str:
        return f"{self.catalog}.{self.schema}"

    def __repr__(self) -> str:
        # Never let a token reach a traceback or a failure report.
        return (
            f"LiveConfig(host={self.host!r}, catalog={self.catalog!r}, "
            f"schema={self.schema!r}, token=***)"
        )


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


@pytest.fixture(scope="session")
def live_config() -> LiveConfig:
    """Skip the whole live suite unless a PAT and a target schema are given."""
    if not _truthy("DELTASWAMP_TEST_DATABRICKS"):
        pytest.skip(
            "live tests are opt-in: set DELTASWAMP_TEST_DATABRICKS=1 along with "
            "DATABRICKS_HOST, DATABRICKS_TOKEN, DELTASWAMP_TEST_CATALOG and "
            "DELTASWAMP_TEST_SCHEMA (see docs/testing.md)"
        )

    host = os.environ.get("DATABRICKS_HOST", "").strip()
    token = os.environ.get("DATABRICKS_TOKEN", "").strip()
    catalog = os.environ.get("DELTASWAMP_TEST_CATALOG", "").strip()
    schema = os.environ.get("DELTASWAMP_TEST_SCHEMA", "").strip()

    missing = [
        name
        for name, value in [
            ("DATABRICKS_HOST", host),
            ("DATABRICKS_TOKEN", token),
            ("DELTASWAMP_TEST_CATALOG", catalog),
            ("DELTASWAMP_TEST_SCHEMA", schema),
        ]
        if not value
    ]
    if missing:
        pytest.skip(f"missing environment: {', '.join(missing)}")

    if not token.startswith("dapi") and not _truthy("DELTASWAMP_TEST_ALLOW_ANY_TOKEN"):
        # OAuth tokens work too, so this is a nudge rather than a rule. It
        # catches the common case of a pasted value carrying a stray quote or
        # newline, which otherwise fails much later and far less clearly.
        pytest.skip(
            "DATABRICKS_TOKEN does not look like a personal access token (it should "
            "start with 'dapi'). Set DELTASWAMP_TEST_ALLOW_ANY_TOKEN=1 to use it anyway."
        )

    return LiveConfig(
        host=host.rstrip("/"),
        token=token,
        catalog=catalog,
        schema=schema,
        warehouse_id=os.environ.get("DELTASWAMP_TEST_WAREHOUSE_ID") or None,
        external_location=os.environ.get("DELTASWAMP_TEST_EXTERNAL_LOCATION") or None,
    )


@pytest.fixture(scope="session")
def live_connection(live_config: LiveConfig) -> Any:
    """A connection authenticated with the PAT, nothing inferred."""
    import deltaswamp as ds

    return ds.connect(
        host=live_config.host,
        token=live_config.token,
        allow_sql_fallback=bool(live_config.warehouse_id),
        warehouse_id=live_config.warehouse_id,
        default_catalog=live_config.catalog,
        default_schema=live_config.schema,
    )


@pytest.fixture(scope="session")
def live_workspace(live_config: LiveConfig) -> Any:
    """A raw SDK client, for setup and teardown the library does not do."""
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient(host=live_config.host, token=live_config.token)


@pytest.fixture(scope="session")
def live_tables(live_connection: Any, live_config: LiveConfig) -> list[Any]:
    """Every table in the target schema, resolved once."""
    tables: list[Any] = live_connection.list_tables(live_config.catalog, live_config.schema)
    return tables


@pytest.fixture
def scratch_name(live_config: LiveConfig) -> str:
    """A unique table name inside the target schema.

    Randomised so a crashed run never collides with the next one.
    """
    return f"{live_config.prefix}.dsw_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def scratch_sql(live_connection: Any, live_config: LiveConfig, scratch_name: str) -> Any:
    """Create a managed Delta table via SQL, and drop it afterwards.

    Managed-table creation is not something this library implements, so the
    fixture uses the warehouse. Teardown runs even when the test fails, because
    leaving tables behind in someone's metastore is rude.
    """
    if not live_config.warehouse_id:
        pytest.skip("DELTASWAMP_TEST_WAREHOUSE_ID is required to create fixtures")

    from deltaswamp.capability import Operation
    from deltaswamp.engine.sql import SqlEngine

    engine = SqlEngine(
        host=live_config.host,
        token=live_config.token,
        warehouse_id=live_config.warehouse_id,
        warn_on_use=False,
    )

    def run(statement: str) -> None:
        engine.execute(Operation.SCAN, statement, fetch=False)

    run(f"CREATE TABLE {scratch_name} (id BIGINT, city STRING) USING DELTA")
    run(f"INSERT INTO {scratch_name} VALUES (1, 'oslo'), (2, 'lima'), (3, 'cairo')")
    try:
        yield scratch_name, run
    finally:
        with contextlib.suppress(Exception):
            run(f"DROP TABLE IF EXISTS {scratch_name}")
