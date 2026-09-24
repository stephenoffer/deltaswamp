"""Check a live-test setup before running the suite.

Run with `python -m tests.live_preflight`. It reports the three things that
usually go wrong -- a token that cannot authenticate, a metastore without
external data access, and a missing schema grant -- so you learn which one you
have in seconds rather than inferring it from a failure partway through a run.

Prints no secrets.
"""

from __future__ import annotations

import os
import sys


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def main() -> int:
    host = os.environ.get("DATABRICKS_HOST", "").strip().rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "").strip()
    catalog = os.environ.get("DELTASWAMP_TEST_CATALOG", "").strip()
    schema = os.environ.get("DELTASWAMP_TEST_SCHEMA", "").strip()

    print("environment")
    print(f"  DATABRICKS_HOST            {host or '(unset)'}")
    print(f"  DATABRICKS_TOKEN           {_mask(token) if token else '(unset)'}")
    print(f"  DELTASWAMP_TEST_CATALOG    {catalog or '(unset)'}")
    print(f"  DELTASWAMP_TEST_SCHEMA     {schema or '(unset)'}")
    warehouse = os.environ.get("DELTASWAMP_TEST_WAREHOUSE_ID") or "(unset)"
    print(f"  WAREHOUSE_ID               {warehouse}")
    print()

    if not (host and token and catalog and schema):
        print("FAIL  set the four required variables first (see docs/testing.md)")
        return 1

    import deltaswamp as ds

    conn = ds.connect(host=host, token=token, default_catalog=catalog, default_schema=schema)

    print("checks")
    try:
        catalogs = conn.list_catalogs()
    except Exception as exc:
        print(f"  FAIL  the token could not list catalogs: {exc}")
        return 1
    print(f"  ok    token authenticates ({len(catalogs)} catalogs visible)")

    if catalog not in catalogs:
        print(f"  FAIL  catalog {catalog!r} is not visible to this principal")
        return 1
    print(f"  ok    catalog {catalog!r} is visible")

    try:
        schemas = conn.list_schemas(catalog)
    except Exception as exc:
        print(f"  FAIL  could not list schemas in {catalog!r}: {exc}")
        return 1
    if schema not in schemas:
        print(f"  FAIL  schema {schema!r} not found; create it first")
        return 1
    print(f"  ok    schema {schema!r} exists")

    problems = conn.preflight()
    if problems:
        for problem in problems:
            print(f"  FAIL  {problem}")
        return 1
    print("  ok    metastore allows external data access")

    tables = conn.list_tables(catalog, schema)
    print(f"  ok    {len(tables)} tables in {catalog}.{schema}")

    vendable = [t for t in tables if t.external_read_supported]
    if tables and not vendable:
        print(
            "  warn  no table reports external read support. The usual cause is a "
            "missing EXTERNAL USE SCHEMA grant, which only the catalog owner can give."
        )
    elif vendable:
        print(f"  ok    {len(vendable)} tables are eligible for credential vending")

    print()
    print("ready: pytest tests/integration/test_live_databricks.py -v")
    return 0


if __name__ == "__main__":
    sys.exit(main())
