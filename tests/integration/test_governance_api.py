"""Governance through the public API, against the fake Unity Catalog.

The catalog-level behaviour is covered in tests/unit/test_governance.py; this
checks the `Connection` and `Table` surface delegates to it, and that what a
catalog lacks comes back as a refusal naming the gap rather than an
AttributeError or NotImplementedError.
"""

from __future__ import annotations

from typing import Any

import deltaswamp as ds
import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

from tests.fake_uc import FakeUnityCatalog  # noqa: E402


@pytest.fixture
def conn(tmp_path: Any) -> Any:
    from deltaswamp.catalog.ossuc import OSSUnityCatalog

    with FakeUnityCatalog(staging_root=tmp_path / "managed") as uc:
        c = ds.connect(catalog=OSSUnityCatalog(uc.url), default_catalog="main")
        c.create_catalog("main")
        c.create_schema("sales")
        yield c


@pytest.fixture
def table(conn: Any) -> Any:
    return conn.create_table("main.sales.orders", pa.schema([("id", pa.int64())]))


class TestConnection:
    def test_table_exists_asks_the_catalog(self, conn: Any, table: Any) -> None:
        assert conn.table_exists("main.sales.orders")
        assert not conn.table_exists("main.sales.missing")

    def test_search_and_listings(self, conn: Any, table: Any) -> None:
        found = conn.search_tables(table_pattern="ord%")
        assert [t.full_name for t in found] == ["main.sales.orders"]
        assert conn.list_volumes("sales") == []

    def test_schema_grants(self, conn: Any) -> None:
        conn.grant("main.sales", "bob", ["USE_SCHEMA"])
        assert any(g.principal == "bob" for g in conn.grants("main.sales"))

    def test_drop_schema(self, conn: Any, table: Any) -> None:
        conn.drop_schema("sales", force=True)
        assert not conn.table_exists("main.sales.orders")

    def test_undrop_needs_the_warehouse(self, conn: Any) -> None:
        with pytest.raises(ds.FallbackRequiredError, match="UNDROP"):
            conn.undrop_table("main.sales.orders")


class TestTable:
    def test_info(self, table: Any) -> None:
        info = table.info()
        assert info.table_type == "MANAGED"
        assert [c.name for c in info.columns] == ["id"]

    def test_grant_and_revoke(self, table: Any) -> None:
        table.grant("alice", "SELECT")
        assert [g.principal for g in table.grants()] == ["alice"]
        table.revoke("alice", ["SELECT"])
        assert table.grants() == []

    def test_open_source_gaps_are_refusals(self, table: Any) -> None:
        with pytest.raises(ds.UnreachableTableError, match="tag"):
            table.tags()
        with pytest.raises(ds.UnreachableTableError, match="lineage"):
            table.lineage()

    def test_row_filters_need_the_warehouse(self, table: Any) -> None:
        with pytest.raises(ds.FallbackRequiredError, match="allow_sql_fallback"):
            table.set_row_filter("main.sales.only_eu", ["id"])

    def test_path_catalog_has_no_governance(self, tmp_path: Any) -> None:
        from deltalake import write_deltalake
        from deltaswamp.catalog.filesystem import FilesystemCatalog

        path = str(tmp_path / "t")
        write_deltalake(path, pa.table({"id": [1]}))
        t = ds.connect(catalog=FilesystemCatalog()).open_table(path)
        with pytest.raises(ds.UnreachableTableError, match="no governance API"):
            t.grants()
