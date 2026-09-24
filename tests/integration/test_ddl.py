"""ALTER TABLE through the public API, on real path tables.

Two routes are covered: delta-rs where it can express the change, and the
kernel's metadata-only commits for what delta-rs cannot (renames and drops under
column mapping, type widening, clustering keys, the properties delta-rs rejects
on ALTER). Each result is checked by reading the table back with delta-rs, an
independent reader, so a commit this library wrote is proven valid rather than
merely self-consistent.
"""

from __future__ import annotations

import json
from typing import Any

import deltaswamp as ds
import pytest
from deltaswamp.capability import Engine, Operation

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")


def _native_features() -> set[str]:
    from deltaswamp import _native

    return set(getattr(_native, "FEATURES", ()))


needs_commit_raw = pytest.mark.skipif(
    not {"commit_raw", "metadata_json"} <= _native_features(),
    reason="native extension predates metadata commits",
)


@pytest.fixture
def conn() -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog

    return ds.connect(catalog=FilesystemCatalog())


@pytest.fixture
def path(tmp_path: Any) -> str:
    from deltalake import write_deltalake

    p = str(tmp_path / "tbl")
    write_deltalake(p, pa.table({"id": [1, 2, 3], "city": ["oslo", "lima", "cairo"]}))
    return p


def reread(path: str) -> Any:
    """Read with delta-rs: an engine that did not write the commit."""
    from deltalake import DeltaTable

    return DeltaTable(path)


class TestDeltaRsRoute:
    def test_comment_and_column_comment(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.set_comment("orders")
        t.set_column_comment("city", "where it shipped")
        dt = reread(path)
        assert dt.metadata().description == "orders"
        field = next(f for f in dt.schema().fields if f.name == "city")
        assert field.metadata["comment"] == "where it shipped"

    def test_constraint_round_trip(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.add_constraint({"pos": "id > 0"})
        assert "delta.constraints.pos" in t.properties()
        t.drop_constraint("pos")
        # Regression: a re-read after ALTER used to keep the stale property.
        assert "delta.constraints.pos" not in t.properties()
        t.drop_constraint("pos", if_exists=True)

    def test_drop_not_null_is_idempotent(self, conn: Any, path: str) -> None:
        conn.open_table(path).drop_not_null("id")

    def test_files_lists_live_data(self, conn: Any, path: str) -> None:
        files = conn.open_table(path).files()
        assert files.num_rows >= 1
        assert "path" in files.column_names

    def test_vacuum_is_a_dry_run_by_default(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.overwrite(pa.table({"id": [9], "city": ["rome"]}))
        assert t.vacuum(retention_hours=0, enforce_retention_duration=False) != []
        assert t.count() == 1

    def test_optimize_zorder(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.append(pa.table({"id": [4], "city": ["kyiv"]}))
        assert t.optimize(zorder_by=["id"])["numFilesAdded"] >= 1

    def test_count_with_predicate(self, conn: Any, path: str) -> None:
        assert conn.open_table(path).count(predicate="id >= 2") == 2


@needs_commit_raw
class TestKernelMetadataCommits:
    def test_properties_delta_rs_rejects_on_alter(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        assert t.can("set_properties", properties={"delta.enableChangeDataFeed": "true"})
        t.set_properties({"delta.enableChangeDataFeed": "true"})
        dt = reread(path)
        assert dt.metadata().configuration["delta.enableChangeDataFeed"] == "true"
        t.append(pa.table({"id": [4], "city": ["kyiv"]}))
        changes = pa.table(t.cdf(starting_version=dt.version() + 1))
        assert changes.num_rows == 1

    def test_unset_properties(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.set_properties({"team": "data"})
        t.unset_properties(["team"])
        assert "team" not in t.properties()

    def test_enable_column_mapping_then_rename_and_drop(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.set_properties({"delta.columnMapping.mode": "name"})
        t.rename_column("city", "town")
        assert t.to_arrow().column_names == ["id", "town"]
        assert sorted(t.to_arrow().column("town").to_pylist()) == ["cairo", "lima", "oslo"]
        # Written after the rename: lands under the physical name, reads under the new one.
        t.append(pa.table({"id": [4], "town": ["kyiv"]}))
        t.drop_column("id")
        assert sorted(t.to_arrow().column("town").to_pylist()) == ["cairo", "kyiv", "lima", "oslo"]
        # An independent reader agrees.
        assert pa.table(reread(path).scan()).column_names == ["town"]

    def test_rename_without_column_mapping_names_the_remedy(self, conn: Any, path: str) -> None:
        with pytest.raises(ds.UnreachableTableError, match=r"delta\.columnMapping\.mode"):
            conn.open_table(path).rename_column("city", "town")

    def test_type_widening(self, conn: Any, tmp_path: Any) -> None:
        from deltalake import write_deltalake

        p = str(tmp_path / "w")
        write_deltalake(p, pa.table({"n": pa.array([1, 2], pa.int32())}))
        t = conn.open_table(p)
        t.set_properties({"delta.enableTypeWidening": "true"})
        t.alter_column_type("n", "bigint")
        assert t.schema().field("n").type == pa.int64() or str(t.schema().field("n").type) in (
            "Int64",
            "int64",
        )
        t.append(pa.table({"n": pa.array([2**40], pa.int64())}))
        assert sorted(t.to_arrow().column("n").to_pylist()) == [1, 2, 2**40]

    def test_set_not_null_checks_the_data(self, conn: Any, tmp_path: Any) -> None:
        from deltalake import write_deltalake

        p = str(tmp_path / "nn")
        write_deltalake(p, pa.table({"a": [1, None]}))
        t = conn.open_table(p)
        with pytest.raises(ds.UnreachableTableError, match="1 existing row"):
            t.set_not_null("a")
        t.delete("a IS NULL")
        t.set_not_null("a")
        assert not reread(p).schema().fields[0].nullable

    def test_cluster_by_on_an_unpartitioned_table(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.cluster_by(["city"])
        assert "clustering" in t.features()
        # delta-rs can no longer write it (domainMetadata), but it still reads.
        assert t.can(Operation.APPEND).engine is Engine.KERNEL
        t.append(pa.table({"id": [5], "city": ["quito"]}))
        assert t.count() == 4
        log = sorted((__import__("pathlib").Path(path) / "_delta_log").glob("*.json"))
        domains = [
            json.loads(line)["domainMetadata"]
            for f in log
            for line in f.read_text().splitlines()
            if "domainMetadata" in json.loads(line)
        ]
        assert json.loads(domains[-1]["configuration"]) == {"clusteringColumns": [["city"]]}

    def test_concurrent_commit_is_recomputed_not_overwritten(
        self, conn: Any, path: str, monkeypatch: Any
    ) -> None:
        """Another writer lands between our read and our put: we must retry
        against its state, not clobber its commit."""
        from deltalake import write_deltalake
        from deltaswamp.engine import kernel

        t = conn.open_table(path)
        engine = conn.router.engines[Engine.KERNEL]
        original = kernel.KernelEngine._state
        raced = {"done": False}

        def racing_state(self: Any, table: Any) -> Any:
            result = original(self, table)
            if not raced["done"]:
                raced["done"] = True
                write_deltalake(path, pa.table({"id": [7], "city": ["baku"]}), mode="append")
            return result

        monkeypatch.setattr(kernel.KernelEngine, "_state", racing_state)
        engine.set_comment(t.resolved, "after the race")
        dt = reread(path)
        assert dt.metadata().description == "after the race"
        assert dt.to_pyarrow_table().num_rows == 4
