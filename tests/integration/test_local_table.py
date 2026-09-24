"""The whole stack against a real on-disk table.

No catalog, no credentials, no network: a path-based Delta table exercises
resolution, routing, both engines and the public API together.
"""

from __future__ import annotations

from typing import Any

import deltaswamp as ds
import pytest
from deltaswamp.capability import Engine, Operation

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")


@pytest.fixture
def conn() -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router
    from deltaswamp.table import Connection

    return Connection(
        catalog=FilesystemCatalog(),
        router=Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()}),
    )


@pytest.fixture
def path(tmp_path: Any) -> str:
    from deltalake import write_deltalake

    p = str(tmp_path / "tbl")
    write_deltalake(p, pa.table({"id": [1, 2, 3], "city": ["oslo", "lima", "cairo"]}))
    return p


class TestRead:
    def test_to_arrow(self, conn: Any, path: str) -> None:
        assert conn.open_table(path).to_arrow().num_rows == 3

    def test_count(self, conn: Any, path: str) -> None:
        assert conn.open_table(path).count() == 3

    def test_scan_exports_arrow_capsule(self, conn: Any, path: str) -> None:
        assert hasattr(conn.open_table(path).scan(), "__arrow_c_stream__")

    def test_projection(self, conn: Any, path: str) -> None:
        assert conn.open_table(path).to_arrow(columns=["city"]).column_names == ["city"]

    def test_reads_route_to_kernel(self, conn: Any, path: str) -> None:
        assert conn.open_table(path).can(Operation.SCAN).engine is Engine.KERNEL


class TestWrite:
    def test_append(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.append(pa.table({"id": [4], "city": ["dakar"]}))
        assert t.count() == 4

    def test_delete(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.delete("id = 1")
        assert set(t.to_arrow().to_pydict()["city"]) == {"lima", "cairo"}

    def test_overwrite(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.overwrite(pa.table({"id": [9], "city": ["quito"]}))
        assert t.to_arrow().to_pydict() == {"id": [9], "city": ["quito"]}

    def test_writes_route_to_deltars(self, conn: Any, path: str) -> None:
        assert conn.open_table(path).can(Operation.MERGE).engine is Engine.DELTARS

    def test_history_grows_with_each_commit(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        before = len(t.history())
        t.append(pa.table({"id": [5], "city": ["tunis"]}))
        assert len(t.history()) == before + 1


class TestCrossEngineAgreement:
    """Both engines must see the same table; divergence here is a real bug."""

    def test_version_agrees(self, conn: Any, path: str) -> None:
        from deltalake import DeltaTable

        assert conn.open_table(path).version == DeltaTable(path).version()

    def test_row_count_agrees(self, conn: Any, path: str) -> None:
        from deltalake import DeltaTable

        ours = conn.open_table(path).count()
        theirs = DeltaTable(path).to_pyarrow_table().num_rows
        assert ours == theirs

    def test_round_trip_through_both_engines(self, conn: Any, path: str) -> None:
        """delta-rs writes, kernel reads -- the interop that matters most."""
        t = conn.open_table(path)
        t.append(pa.table({"id": [7, 8], "city": ["riga", "sofia"]}))
        assert set(t.to_arrow().to_pydict()["id"]) == {1, 2, 3, 7, 8}


class TestTimeTravel:
    def test_reads_prior_version(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.append(pa.table({"id": [4], "city": ["dakar"]}))
        assert t.to_arrow(version=0).num_rows == 3
        assert t.to_arrow().num_rows == 4


class TestCapabilities:
    def test_report_covers_every_operation(self, conn: Any, path: str) -> None:
        assert set(conn.open_table(path).capabilities()) == set(Operation)

    def test_databricks_only_operations_are_refused_with_a_remedy(
        self, conn: Any, path: str
    ) -> None:
        cap = conn.open_table(path).can(Operation.CLONE)
        assert not cap.ok
        assert "Databricks" in cap.reason
        assert "allow_sql_fallback" in cap.remedy


class TestCreate:
    def test_creates_a_table_at_a_path(self, conn: Any, tmp_path: Any) -> None:
        schema = pa.schema([("id", pa.int64()), ("city", pa.string())])
        target = str(tmp_path / "created")
        table = conn.create_table(target, schema)
        table.append(pa.table({"id": [1], "city": ["oslo"]}))
        assert table.count() == 1

    def test_catalog_create_without_a_lifecycle_catalog_is_refused(self, conn: Any) -> None:
        """A catalog name needs a catalog that can register it; writing only a
        log would orphan it while appearing to succeed."""
        from deltaswamp.errors import UnreachableTableError

        conn.default_catalog, conn.default_schema = "main", "sales"
        schema = pa.schema([("id", pa.int64())])
        with pytest.raises(UnreachableTableError, match="orphaned"):
            conn.create_table("main.sales.newthing", schema)


class TestRequestShapeRouting:
    """Predicates and timestamps now reach the kernel.

    The kernel skips files with a structured predicate and never drops a row,
    so the exact filter is applied afterwards. These check the kernel's answer
    against delta-rs, which evaluates the same SQL with DataFusion.
    """

    def test_plain_scan_uses_kernel(self, conn: Any, path: str) -> None:
        assert conn.open_table(path).can(Operation.SCAN).engine is Engine.KERNEL

    def test_predicate_scan_uses_kernel(self, conn: Any, path: str) -> None:
        table = conn.open_table(path)
        capability = conn.router.capability(
            Operation.SCAN, table.resolved, needs=frozenset({"predicates"})
        )
        assert capability.engine is Engine.KERNEL

    @pytest.mark.parametrize(
        "predicate",
        ["id > 1", "city = 'lima' OR id = 1", "NOT (id = 2)", "city LIKE '%o'", "id IN (1, 3)"],
    )
    def test_kernel_and_deltars_agree(self, conn: Any, path: str, predicate: str) -> None:
        from deltalake import DeltaTable

        ours = pa.table(conn.open_table(path).scan(predicate=predicate)).to_pylist()
        theirs = pa.table(DeltaTable(path).scan(predicate=predicate)).to_pylist()
        key = lambda row: row["id"]  # noqa: E731
        assert sorted(ours, key=key) == sorted(theirs, key=key)

    def test_projection_excludes_predicate_only_columns(self, conn: Any, path: str) -> None:
        got = pa.table(conn.open_table(path).scan(columns=["city"], predicate="id > 1"))
        assert got.column_names == ["city"]
        assert sorted(got.column("city").to_pylist()) == ["cairo", "lima"]

    def test_timestamp_travel_on_the_kernel(self, conn: Any, path: str) -> None:
        import datetime as dt
        import time

        table = conn.open_table(path)
        time.sleep(0.01)
        between = dt.datetime.now(dt.UTC)
        time.sleep(0.01)
        table.append(pa.table({"id": [4], "city": ["kyiv"]}))
        assert table.can(Operation.TIME_TRAVEL).engine is Engine.KERNEL
        assert pa.table(table.scan(timestamp=between.isoformat())).num_rows == 3
        assert pa.table(table.scan()).num_rows == 4

    def test_timestamp_before_history_is_named(self, conn: Any, path: str) -> None:
        with pytest.raises(ds.UnreachableTableError, match="recreatable"):
            conn.open_table(path).scan(timestamp="2000-01-01T00:00:00Z")


class TestOptionalDependencyErrors:
    """A missing extra should name itself, not surface as a ModuleNotFoundError
    from somewhere inside pyarrow."""

    def test_missing_extra_names_the_install_command(
        self, conn: Any, path: str, monkeypatch: Any
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "polars":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(ImportError, match=r"deltaswamp\[polars\]"):
            conn.open_table(path).to_polars()


class TestMetadataFreshness:
    """A Table that just altered itself must not report the old metadata.

    Enrichment caches protocol state read from the log, so without explicit
    invalidation `properties()` keeps answering with what was true before the
    call that just changed it.
    """

    def test_set_properties_is_visible_immediately(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.properties()  # prime the cache
        t.set_properties({"delta.deletedFileRetentionDuration": "interval 30 days"})
        assert t.properties()["delta.deletedFileRetentionDuration"] == "interval 30 days"

    def test_add_constraint_is_visible_immediately(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.properties()
        t.add_constraint({"id_positive": "id > 0"})
        assert any("constraint" in key for key in t.properties())

    def test_append_updates_the_reported_version(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        before = t.version
        t.append(pa.table({"id": [99], "city": ["riga"]}))
        assert t.version == before + 1


class TestNewOperations:
    def test_add_column(self, conn: Any, path: str) -> None:
        from deltalake import Schema

        t = conn.open_table(path)
        field = Schema.from_arrow(pa.schema([("region", pa.string())])).fields[0]
        t.add_column([field])
        assert "region" in [f.name for f in t.to_arrow().schema]

    def test_checkpoint_and_generate(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.checkpoint()
        t.generate()

    def test_compact_logs_defaults_to_the_whole_log(self, conn: Any, path: str) -> None:
        """delta-rs needs concrete versions, so None must not reach it."""
        conn.open_table(path).compact_logs()

    def test_publish_on_a_path_table_returns_the_version(self, conn: Any, path: str) -> None:
        assert conn.open_table(path).publish() == 0

    @pytest.mark.parametrize(
        ("name", "args"),
        [
            ("drop_column", ("city",)),
            ("rename_column", ("city", "town")),
            ("drop_feature", ("deletionVectors",)),
            ("reorg", ()),
            ("clone", ("main.s.copy",)),
        ],
    )
    def test_databricks_only_operations_refuse_without_the_fallback(
        self, conn: Any, path: str, name: str, args: tuple[Any, ...]
    ) -> None:
        from deltaswamp.errors import DeltaSwampError

        with pytest.raises(DeltaSwampError):
            getattr(conn.open_table(path), name)(*args)

    def test_convert_to_delta(self, conn: Any, tmp_path: Any) -> None:
        import pyarrow.parquet as papq

        raw = tmp_path / "raw"
        raw.mkdir()
        papq.write_table(pa.table({"id": [9]}), str(raw / "part-0.parquet"))
        assert conn.convert_to_delta(str(raw)).count() == 1


class TestCreateRoutingAndProperties:
    """Creation routes on the properties requested, and nothing is dropped.

    delta-rs rejects roughly half the Delta property surface with one opaque
    message and panics on `delta.minReaderVersion`. The kernel accepts nearly
    all of it, so a create it cannot serve must fall through rather than fail.
    """

    def _schema(self) -> Any:
        return pa.schema([("id", pa.int64()), ("city", pa.string())])

    def test_plain_create_uses_deltars(self, conn: Any, tmp_path: Any) -> None:
        t = conn.create_table(str(tmp_path / "plain"), self._schema())
        assert t.can(Operation.CREATE).engine is Engine.DELTARS

    @pytest.mark.parametrize(
        "properties",
        [
            {"delta.enableRowTracking": "true"},
            {"delta.enableInCommitTimestamps": "true"},
            {"delta.enableTypeWidening": "true"},
            {"delta.feature.deletionVectors": "supported"},
            {"custom.owner": "analytics"},
        ],
    )
    def test_kernel_only_properties_route_to_kernel_and_work(
        self, conn: Any, tmp_path: Any, properties: dict[str, str]
    ) -> None:
        target = str(tmp_path / "k" / next(iter(properties)).replace(".", "_"))
        t = conn.create_table(target, self._schema(), properties=properties)
        t.append(pa.table({"id": [1], "city": ["oslo"]}))
        assert t.count() == 1

    def test_cluster_by_reaches_the_engine(self, conn: Any, tmp_path: Any) -> None:
        """The router picks kernel for a clustered create; the argument must
        then actually arrive, or the table is silently unclustered."""
        t = conn.create_table(str(tmp_path / "clustered"), self._schema(), cluster_by=["id"])
        assert "clustering" in t.features(), (
            f"cluster_by was dropped: features are {sorted(t.features())}"
        )
        assert "domainMetadata" in t.features()

    def test_partitioned_create(self, conn: Any, tmp_path: Any) -> None:
        t = conn.create_table(str(tmp_path / "parts"), self._schema(), partition_by=["city"])
        t.append(pa.table({"id": [1], "city": ["oslo"]}))
        assert t.count() == 1

    def test_minreaderversion_is_refused_not_a_panic(self, conn: Any, tmp_path: Any) -> None:
        """delta-rs panics on this, and a panic is not catchable as Exception."""
        from deltaswamp.errors import DeltaSwampError

        with pytest.raises(DeltaSwampError):
            conn.create_table(
                str(tmp_path / "boom"),
                self._schema(),
                properties={"delta.minReaderVersion": "3"},
            )

    def test_inert_property_warns(self, conn: Any, tmp_path: Any) -> None:
        from deltaswamp.errors import IgnoredPropertyWarning

        with pytest.warns(IgnoredPropertyWarning, match="checkpointPolicy"):
            conn.create_table(
                str(tmp_path / "inert"),
                self._schema(),
                properties={"delta.checkpointPolicy": "v2"},
            )


class TestTableIdentity:
    """A cached catalog id must not outlive the table it named."""

    def test_mismatched_log_id_is_refused(self, conn: Any, path: str) -> None:
        """Dropping and re-creating a table keeps the name and changes the id.
        Reading on regardless means reading a different table."""
        import dataclasses

        from deltaswamp.errors import CorruptTableError
        from deltaswamp.table import Table

        resolved = conn.open_table(path).resolved
        stale = dataclasses.replace(resolved, table_uuid="00000000-dead-beef-0000-000000000000")
        with pytest.raises(CorruptTableError, match="dropped and re-created"):
            Table(conn, stale).features()

    def test_matching_id_passes(self, conn: Any, path: str) -> None:
        import dataclasses

        from deltaswamp._native import Snapshot
        from deltaswamp.table import Table

        resolved = conn.open_table(path).resolved
        actual = Snapshot.resolve(path).metadata_id
        matched = dataclasses.replace(resolved, table_uuid=actual)
        assert Table(conn, matched).features() is not None

    def test_no_catalog_uuid_means_no_check(self, conn: Any, path: str) -> None:
        """Path-based tables have no catalog identity, and that is fine."""
        assert conn.open_table(path).features() is not None
