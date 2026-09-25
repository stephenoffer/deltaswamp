"""Bringing tables into Unity Catalog, end to end, against the fake server.

Two flows, each finishing with a read through the public `Connection`:

* External: write a Delta log at a path, then register it. The catalog records
  what it is told and never reads the log, so the read afterwards is the check
  that what was registered is really a table.
* Managed (catalog-managed): reserve a staging table, write version 0 at the
  location the catalog chose carrying every property it required, then
  finalise. The fake refuses a finalise that skips any of those steps, as the
  real server does.

Version 0 is written here with delta-rs and patched to declare
`catalogManaged`, the same way `test_catalog_managed.py` builds its fixture.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any
from urllib.parse import urlparse

import deltaswamp as ds
import pytest
from deltaswamp.identity import parse_ref

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

from tests.fake_uc import FakeUnityCatalog  # noqa: E402
from tests.helpers import direct_router  # noqa: E402


@pytest.fixture
def uc(tmp_path: Any) -> Any:
    with FakeUnityCatalog(staging_root=tmp_path / "managed") as server:
        yield server


@pytest.fixture
def catalog(uc: FakeUnityCatalog) -> Any:
    from deltaswamp.catalog.ossuc import OSSUnityCatalog

    cat = OSSUnityCatalog(uc.url)
    cat.create_catalog("main")
    cat.create_schema("main", "sales")
    return cat


@pytest.fixture
def conn(catalog: Any) -> Any:
    return ds.Connection(catalog=catalog, router=direct_router())


def _log_actions(table_root: pathlib.Path) -> list[dict[str, Any]]:
    first = table_root / "_delta_log" / f"{0:020d}.json"
    return [json.loads(line) for line in first.read_text().splitlines() if line.strip()]


class TestRegisterExternal:
    @pytest.fixture
    def registered(self, tmp_path: Any, catalog: Any) -> str:
        from deltalake import DeltaTable, write_deltalake

        path = str(tmp_path / "ext")
        write_deltalake(
            path,
            pa.table({"id": [1, 2, 3], "region": ["eu", "us", "eu"]}),
            partition_by=["region"],
        )
        resolved = catalog.register_table(
            parse_ref("main.sales.ext"),
            path,
            columns_schema_json=DeltaTable(path).schema().to_json(),
            partition_columns=["region"],
            comment="registered from a path",
        )
        assert resolved.location == path
        return path

    def test_catalog_records_what_it_was_told(self, catalog: Any, registered: str) -> None:
        info = catalog.table_info(parse_ref("main.sales.ext"))
        assert info.table_type == "EXTERNAL"
        assert info.storage_location == registered
        assert info.partition_columns == ("region",)
        assert info.comment == "registered from a path"
        assert [c.type_text for c in info.columns] == ["bigint", "string"]

    def test_read_through_the_connection(self, conn: Any, registered: str) -> None:
        assert conn.table("main.sales.ext").to_arrow().num_rows == 3


class TestCreateManaged:
    def _write_version_zero(self, staged: Any) -> pathlib.Path:
        """Write v0 at the staged location, honouring what staging demanded."""
        from deltalake import write_deltalake

        root = pathlib.Path(urlparse(staged.location).path)
        write_deltalake(str(root), pa.table({"id": [1, 2], "city": ["oslo", "lima"]}))

        protocol = staged.required_protocol
        actions = _log_actions(root)
        for action in actions:
            if "protocol" in action:
                action["protocol"] = {
                    "minReaderVersion": protocol["min-reader-version"],
                    "minWriterVersion": protocol["min-writer-version"],
                    "readerFeatures": protocol["reader-features"],
                    "writerFeatures": protocol["writer-features"],
                }
            if "metaData" in action:
                # The catalog's table id and the log's identity must agree.
                action["metaData"]["id"] = staged.table_id
                configuration = action["metaData"].setdefault("configuration", {})
                for key, value in staged.required_properties.items():
                    # delta.feature.* is carried by the protocol, never metadata.
                    if not key.startswith("delta.feature.") and value is not None:
                        configuration[key] = value
        (root / "_delta_log" / f"{0:020d}.json").write_text(
            "\n".join(json.dumps(a) for a in actions) + "\n"
        )
        return root

    def _create_table_request(self, root: pathlib.Path, name: str, location: str) -> dict[str, Any]:
        """The UC Delta API CreateTableRequest, built from the v0 log."""
        actions = _log_actions(root)
        protocol = next(a["protocol"] for a in actions if "protocol" in a)
        metadata = next(a["metaData"] for a in actions if "metaData" in a)
        commit_info: dict[str, Any] = next(
            (a["commitInfo"] for a in actions if "commitInfo" in a), {}
        )
        return {
            "name": name,
            "location": location,
            "table-type": "MANAGED",
            "columns": json.loads(metadata["schemaString"]),
            "partition-columns": metadata.get("partitionColumns", []),
            "protocol": {
                "min-reader-version": protocol["minReaderVersion"],
                "min-writer-version": protocol["minWriterVersion"],
                "reader-features": protocol.get("readerFeatures", []),
                "writer-features": protocol.get("writerFeatures", []),
            },
            "properties": metadata.get("configuration", {}),
            "last-commit-timestamp-ms": commit_info.get("timestamp", 0),
        }

    def test_stage_write_finalize_read(self, catalog: Any, conn: Any) -> None:
        ref = parse_ref("main.sales.fresh")
        staged = catalog.create_staging_table(ref)
        assert not catalog.table_exists(ref)

        root = self._write_version_zero(staged)
        resolved = catalog.finalize_managed_table(
            ref, self._create_table_request(root, "fresh", staged.location)
        )

        assert resolved.is_catalog_managed
        assert resolved.table_id == staged.table_id
        assert catalog.table_exists(ref)

        table = conn.table("main.sales.fresh")
        assert table.is_catalog_managed
        assert table.to_arrow().num_rows == 2

    def test_finalize_refuses_a_log_missing_required_properties(self, catalog: Any) -> None:
        from deltalake import write_deltalake
        from deltaswamp.errors import PreflightError

        ref = parse_ref("main.sales.sloppy")
        staged = catalog.create_staging_table(ref)
        root = pathlib.Path(urlparse(staged.location).path)
        write_deltalake(str(root), pa.table({"id": [1]}))

        with pytest.raises(PreflightError, match=r"catalogManaged|required"):
            catalog.finalize_managed_table(
                ref, self._create_table_request(root, "sloppy", staged.location)
            )
        assert not catalog.table_exists(ref)


class TestThroughTheConnection:
    """The same flows through `Connection.create_table`, with the library
    writing version 0 itself -- no hand-built log."""

    def test_create_managed_then_write_and_read(self, conn: Any, uc: Any) -> None:
        schema = pa.schema([("id", pa.int64()), ("city", pa.string())])
        table = conn.create_table("main.sales.orders", schema, comment="orders")
        assert table.is_catalog_managed or "catalogManaged" in table.features()

        info = conn.catalog.table_info(table.resolved.ref)
        assert info.table_type == "MANAGED"

        # The log's identity is the catalog's id, as finalise demanded.
        root = pathlib.Path(urlparse(table.location).path)
        meta = next(a["metaData"] for a in _log_actions(root) if "metaData" in a)
        assert meta["id"] == table.resolved.table_id
        assert meta["description"] == "orders"

        table.append(pa.table({"id": [1, 2], "city": ["oslo", "lima"]}))
        fresh = conn.table("main.sales.orders")
        assert fresh.count() == 2

    def test_create_managed_clustered(self, conn: Any) -> None:
        schema = pa.schema([("id", pa.int64()), ("city", pa.string())])
        table = conn.create_table("main.sales.clustered", schema, cluster_by=["city"])
        assert "clustering" in table.features()

    def test_create_external_registers_what_it_wrote(self, conn: Any, tmp_path: Any) -> None:
        location = f"file://{tmp_path / 'ext2'}"
        schema = pa.schema([("id", pa.int64()), ("region", pa.string())])
        conn.create_table("main.sales.ext2", schema, location=location, partition_by=["region"])
        info = conn.catalog.table_info(parse_ref("main.sales.ext2"))
        assert info.table_type == "EXTERNAL"
        assert info.partition_columns == ("region",)
        table = conn.table("main.sales.ext2")
        table.append(pa.table({"id": [1], "region": ["eu"]}))
        assert conn.table("main.sales.ext2").count() == 1

    def test_register_an_existing_path(self, conn: Any, tmp_path: Any) -> None:
        from deltalake import write_deltalake

        path = str(tmp_path / "legacy")
        write_deltalake(path, pa.table({"id": [1, 2]}))
        table = conn.register_table("main.sales.legacy", path)
        assert table.count() == 2

    def test_replacing_a_catalog_table_is_refused(self, conn: Any) -> None:
        with pytest.raises(ds.UnreachableTableError, match="created once"):
            conn.create_table("main.sales.x", pa.schema([("id", pa.int64())]), mode="overwrite")


class TestCatalogManagedWithoutDatabricks:
    """DML, predicate overwrite and checkpointing on a catalog-managed table,
    each committed through the catalog, with no warehouse involved."""

    @pytest.fixture
    def managed(self, conn: Any) -> Any:
        schema = pa.schema([("id", pa.int64()), ("c", pa.string())])
        conn.create_table("main.sales.dml", schema)
        conn.table("main.sales.dml").append(
            pa.table({"id": [1, 2, 3, None], "c": ["a", "b", "c", "d"]})
        )
        return conn

    def rows(self, conn: Any) -> list[tuple[Any, Any]]:
        table = conn.table("main.sales.dml").to_arrow()
        return sorted(
            zip(table.column("id").to_pylist(), table.column("c").to_pylist(), strict=True),
            key=str,
        )

    def test_routes_to_the_kernel(self, managed: Any) -> None:
        from deltaswamp.capability import Engine

        table = managed.table("main.sales.dml")
        assert table.is_catalog_managed
        for op in ("delete", "update", "replace_where", "checkpoint"):
            assert table.can(op).engine is Engine.KERNEL, op

    def test_delete_keeps_null_results(self, managed: Any) -> None:
        assert managed.table("main.sales.dml").delete("id = 2")["num_deleted_rows"] == 1
        assert self.rows(managed) == [(1, "a"), (3, "c"), (None, "d")]

    def test_dml_is_written_as_deletion_vectors_through_the_catalog(self, managed: Any) -> None:
        """A managed table enables deletion vectors, so DML marks rows deleted
        (staged and ratified by the catalog) instead of rewriting the table."""
        pytest.importorskip("duckdb")
        table = managed.table("main.sales.dml")
        root = pathlib.Path(urlparse(table.location).path)
        data_files = set(root.rglob("*.parquet")) - set(root.rglob("_delta_log/**/*"))
        table.delete("id = 2")
        managed.table("main.sales.dml").update(new_values={"c": "z"}, predicate="id = 3")
        (
            managed.table("main.sales.dml")
            .merge(
                pa.table({"id": [1, 7], "c": ["m", "n"]}),
                "t.id = s.id",
                source_alias="s",
                target_alias="t",
            )
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute()
        )
        assert self.rows(managed) == [(1, "m"), (3, "z"), (7, "n"), (None, "d")]
        assert list(root.rglob("deletion_vector_*.bin")), "vectors were written"
        still = set(root.rglob("*.parquet"))
        assert data_files <= still, "no original data file was rewritten away"
        staged = list((root / "_delta_log" / "_staged_commits").glob("*.json"))
        text = "".join(p.read_text() for p in staged)
        assert '"deletionVector"' in text, "the vectors went through catalog-staged commits"

    def test_update_with_plain_values(self, managed: Any) -> None:
        managed.table("main.sales.dml").update(new_values={"c": "z"}, predicate="id > 1")
        assert self.rows(managed) == [(1, "a"), (2, "z"), (3, "z"), (None, "d")]

    def test_update_with_sql_literals_and_columns(self, managed: Any) -> None:
        managed.table("main.sales.dml").update({"c": "'q'"}, predicate="id = 1")
        assert (1, "q") in self.rows(managed)

    def test_arbitrary_expressions_are_refused(self, managed: Any) -> None:
        with pytest.raises(Exception, match="only a literal or a column"):
            managed.table("main.sales.dml").update({"id": "id + 1"})

    def test_replace_where(self, managed: Any) -> None:
        managed.table("main.sales.dml").overwrite(
            pa.table({"id": [9], "c": ["n"]}), predicate="id IS NULL OR id = 9"
        )
        assert self.rows(managed) == [(1, "a"), (2, "b"), (3, "c"), (9, "n")]
        # A new row outside the predicate is refused, as Databricks does.
        with pytest.raises(Exception, match="do not satisfy the predicate"):
            managed.table("main.sales.dml").overwrite(
                pa.table({"id": [8], "c": ["m"]}), predicate="id IS NULL"
            )
        assert self.rows(managed) == [(1, "a"), (2, "b"), (3, "c"), (9, "n")]

    def test_checkpoint_publishes_first(self, managed: Any) -> None:
        table = managed.table("main.sales.dml")
        table.checkpoint()
        root = pathlib.Path(urlparse(table.location).path) / "_delta_log"
        assert list(root.glob("*.checkpoint*.parquet")) or list(root.glob("_last_checkpoint"))

    def test_deletion_vector_dml_ignores_the_rewrite_size_limit(self, managed: Any) -> None:
        # The managed table enables deletion vectors, so DML marks rows deleted
        # rather than rewriting the table, and the rewrite's bound is moot.
        from deltaswamp.engine.kernel import KernelEngine

        engine = managed.router.engines[__import__("deltaswamp").Engine.KERNEL]
        engine.rewrite_max_bytes = 1
        try:
            table = managed.table("main.sales.dml")
            assert table.properties().get("delta.enableDeletionVectors") == "true"
            verdict = table.can("delete")
            assert verdict.ok, verdict.reason
        finally:
            engine.rewrite_max_bytes = KernelEngine.rewrite_max_bytes
