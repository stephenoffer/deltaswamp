"""A missing table is a TableNotFoundError, still an InvalidReferenceError."""

from __future__ import annotations

import pytest

ds = pytest.importorskip("deltaswamp")

from tests.fake_uc import FakeUnityCatalog  # noqa: E402


def test_missing_catalog_table_is_table_not_found() -> None:
    with FakeUnityCatalog() as uc:
        uc.add_schema("main", "s")
        conn = ds.connect(f"uc://{uc.url}")
        with pytest.raises(ds.TableNotFoundError, match="does not exist"):
            conn.table("main.s.nope")
        with pytest.raises(ds.InvalidReferenceError):
            conn.table("main.s.nope")


def test_drop_table_if_exists() -> None:
    with FakeUnityCatalog() as uc:
        uc.add_schema("main", "s")
        conn = ds.connect(f"uc://{uc.url}")
        with pytest.raises(ds.TableNotFoundError):
            conn.drop_table("main.s.nope")
        conn.drop_table("main.s.nope", if_exists=True)
