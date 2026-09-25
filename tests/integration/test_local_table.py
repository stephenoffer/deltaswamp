"""The whole stack against a real on-disk table.

No catalog, no credentials, no network: a path-based Delta table exercises
resolution, routing, both engines and the public API together.
"""

from __future__ import annotations

from typing import Any

import deltaswamp as ds
import pytest
from deltaswamp.capability import Engine, Operation
from deltaswamp.errors import UnreachableTableError

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

    def test_head_returns_the_first_n_rows(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        assert t.head(2).num_rows == 2
        assert t.head(0).num_rows == 0
        # More than the table holds is not an error, just the whole table.
        assert t.head(99).num_rows == 3
        assert t.head(2, columns=["city"]).column_names == ["city"]

    def test_head_stops_reading_once_it_has_enough(self, conn: Any, path: str) -> None:
        """head() must not materialise the table to slice it.

        It used to be `to_arrow().slice(0, n)`, which on a real multi-terabyte
        table never returned -- it hung the live suite on a 10 TiB table until
        this was changed to consume the stream and stop.
        """
        t = conn.open_table(path)
        consumed: list[int] = []
        original = t.scan

        def counting_scan(**kwargs: Any) -> Any:
            import pyarrow as pa

            reader = pa.RecordBatchReader.from_stream(original(**kwargs))

            def batches() -> Any:
                for batch in reader:
                    consumed.append(batch.num_rows)
                    yield batch

            return pa.RecordBatchReader.from_batches(reader.schema, batches())

        t.scan = counting_scan
        assert t.head(1).num_rows == 1
        assert len(consumed) == 1, (
            f"head(1) pulled {len(consumed)} batches; it must stop at the first"
        )


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


class TestUnmodelledWriterFeatures:
    """Tables carrying a writer feature neither engine understands.

    The Delta protocol is explicit: a writer must not write to a table whose
    protocol lists a writer feature it does not support. Reads are a different
    question -- a writer-only feature never blocks one -- and that asymmetry is
    what lets the kernel out-read delta-rs.
    """

    @staticmethod
    def _with_writer_feature(conn: Any, path: str, feature: str) -> Any:
        """Append a protocol action adding `feature`, as a newer writer would."""
        import glob
        import json
        import os

        t = conn.open_table(path)
        t.to_arrow()
        logs = sorted(glob.glob(os.path.join(path, "_delta_log", "*.json")))
        nxt = int(os.path.basename(logs[-1]).split(".")[0]) + 1
        with open(os.path.join(path, "_delta_log", f"{nxt:020d}.json"), "w") as fh:
            fh.write(
                json.dumps(
                    {
                        "protocol": {
                            "minReaderVersion": 3,
                            "minWriterVersion": 7,
                            "readerFeatures": ["deletionVectors"],
                            "writerFeatures": ["deletionVectors", feature],
                        }
                    }
                )
                + "\n"
            )
        return conn.open_table(path)

    def test_an_unmodelled_feature_still_reads(self, conn: Any, path: str) -> None:
        t = self._with_writer_feature(conn, path, "checkpointProtection")
        assert t.to_arrow().num_rows == 3

    def test_an_unmodelled_feature_blocks_even_a_metadata_commit(
        self, conn: Any, path: str
    ) -> None:
        """checkpointProtection governs which checkpoints may be removed.

        The kernel has no variant for it at all, so it cannot know what the
        feature requires of a commit. A metadata-only commit was allowed through
        because it writes no data, which is the wrong test: writing blind to a
        table whose rules you cannot read risks corrupting history.
        """
        t = self._with_writer_feature(conn, path, "checkpointProtection")
        for op in (Operation.APPEND, Operation.ADD_COLUMN, Operation.SET_PROPERTIES):
            verdict = t.can(op)
            assert not verdict.ok, f"{op.value} should be refused"
            assert "checkpointProtection" in verdict.reason

    def test_a_modelled_feature_still_allows_a_metadata_commit(self, conn: Any, path: str) -> None:
        """identityColumns is understood; the kernel just cannot write data for it.

        A metadata-only commit writes no rows, so it stays available. Keeping
        this distinction is the point: the blanket rule would be correct but
        needlessly refuse half the DDL surface.
        """
        t = self._with_writer_feature(conn, path, "identityColumns")
        assert not t.can(Operation.APPEND).ok
        assert t.can(Operation.ADD_COLUMN).ok


class TestDistributedWrite:
    """Plan on the driver, write on workers, commit once.

    The failure this shape exists to prevent is the usual one in distributed
    Delta writers: discovering at commit time that the table refuses the write,
    after the compute is spent, leaving orphaned Parquet behind.
    """

    def test_fragments_from_several_workers_land_as_one_version(self, conn: Any, path: str) -> None:
        import pickle

        table = conn.open_table(path)
        before = table.version
        plan = conn.open_table(path).plan_write()
        fragments = [
            pickle.loads(pickle.dumps(plan)).write(pa.table({"id": [10], "city": ["a"]})),
            pickle.loads(pickle.dumps(plan)).write(pa.table({"id": [11], "city": ["b"]})),
            pickle.loads(pickle.dumps(plan)).write(pa.table({"id": [12], "city": ["c"]})),
        ]
        assert conn.open_table(path).version == before, "nothing commits before commit()"

        version = plan.commit(fragments)
        assert version == before + 1, "three fragments, one version"
        assert conn.open_table(path).to_arrow().num_rows == 6

    def test_the_plan_pickles_without_a_credential(self, conn: Any, path: str) -> None:
        import pickle

        payload = pickle.dumps(conn.open_table(path).plan_write())
        for shape in (b"dapi", b"AKIA", b"aws_secret_access_key"):
            assert shape not in payload

    def test_overwrite_removes_the_old_files_in_the_same_commit(self, conn: Any, path: str) -> None:
        plan = conn.open_table(path).plan_write(mode="overwrite")
        plan.commit([plan.write(pa.table({"id": [99], "city": ["z"]}))])
        assert conn.open_table(path).to_arrow().to_pydict()["id"] == [99]

    def test_an_unknown_mode_is_refused(self, conn: Any, path: str) -> None:
        with pytest.raises(UnreachableTableError, match="append"):
            conn.open_table(path).plan_write(mode="upsert")

    def test_a_partitioned_table_writes_partition_directories(self, conn: Any) -> None:
        import glob
        import os
        import tempfile

        location = os.path.join(tempfile.mkdtemp(), "t")
        table = conn.create_table(
            location,
            pa.schema([("id", pa.int64()), ("region", pa.string())]),
            partition_by=["region"],
        )
        plan = table.plan_write()
        plan.commit(
            [
                plan.write(pa.table({"id": [1, 2], "region": ["us", "eu"]})),
                plan.write(pa.table({"id": [3], "region": ["us"]})),
            ]
        )
        written = {
            os.path.basename(os.path.dirname(f))
            for f in glob.glob(os.path.join(location, "*", "*.parquet"))
        }
        assert written == {"region=us", "region=eu"}
        assert conn.open_table(location).to_arrow().num_rows == 3

    def test_an_idempotent_plan_refuses_a_replay_before_the_job(self, conn: Any, path: str) -> None:
        """The whole point of planning: catch it before the compute, not after."""
        plan = conn.open_table(path).plan_write(txn=("nightly", 1))
        plan.commit([plan.write(pa.table({"id": [7], "city": ["x"]}))])

        with pytest.raises(UnreachableTableError, match="already committed"):
            conn.open_table(path).plan_write(txn=("nightly", 1))

    def test_an_append_lands_on_top_of_a_concurrent_writer(self, conn: Any, path: str) -> None:
        """An append means "add these rows", so a racing writer is not a conflict."""
        plan = conn.open_table(path).plan_write()
        fragment = plan.write(pa.table({"id": [10], "city": ["a"]}))
        conn.open_table(path).append(pa.table({"id": [99], "city": ["z"]}))

        plan.commit([fragment])
        got = set(conn.open_table(path).to_arrow().to_pydict()["id"])
        assert {10, 99} <= got, "both writers' rows must survive"

    def test_a_raced_overwrite_is_refused_rather_than_losing_the_winner(
        self, conn: Any, path: str
    ) -> None:
        """An overwrite removes what it finds, so a racing writer would vanish.

        Nothing in the log would record that a writer was lost, which is why
        this is refused rather than resolved silently.
        """
        plan = conn.open_table(path).plan_write(mode="overwrite")
        fragment = plan.write(pa.table({"id": [10], "city": ["a"]}))
        conn.open_table(path).append(pa.table({"id": [99], "city": ["z"]}))

        with pytest.raises(UnreachableTableError, match="discard"):
            plan.commit([fragment])
        assert 99 in conn.open_table(path).to_arrow().to_pydict()["id"]

    def test_a_raced_overwrite_can_be_forced(self, conn: Any, path: str) -> None:
        plan = conn.open_table(path).plan_write(mode="overwrite")
        fragment = plan.write(pa.table({"id": [10], "city": ["a"]}))
        conn.open_table(path).append(pa.table({"id": [99], "city": ["z"]}))

        plan.commit([fragment], allow_concurrent_overwrite=True)
        assert conn.open_table(path).to_arrow().to_pydict()["id"] == [10]

    def test_an_unraced_overwrite_needs_no_opt_in(self, conn: Any, path: str) -> None:
        plan = conn.open_table(path).plan_write(mode="overwrite")
        plan.commit([plan.write(pa.table({"id": [10], "city": ["a"]}))])
        assert conn.open_table(path).to_arrow().to_pydict()["id"] == [10]

    def test_a_refused_write_writes_no_files(self, conn: Any, path: str) -> None:
        """A table the kernel cannot write must be refused before any file lands."""
        import glob
        import json
        import os

        conn.open_table(path).to_arrow()
        logs = sorted(glob.glob(os.path.join(path, "_delta_log", "*.json")))
        nxt = int(os.path.basename(logs[-1]).split(".")[0]) + 1
        with open(os.path.join(path, "_delta_log", f"{nxt:020d}.json"), "w") as fh:
            fh.write(
                json.dumps(
                    {
                        "protocol": {
                            "minReaderVersion": 3,
                            "minWriterVersion": 7,
                            "readerFeatures": [],
                            "writerFeatures": ["checkpointProtection"],
                        }
                    }
                )
                + "\n"
            )
        before = len(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
        with pytest.raises(UnreachableTableError, match="checkpointProtection"):
            conn.open_table(path).plan_write()
        after = len(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
        assert after == before, "a refused plan must not have written anything"


class TestFragmentIdentity:
    """A fragment names its files relative to the table it was written under.

    Committing one into a different table therefore writes an add action
    pointing at a file that is not there. The commit succeeds and the table is
    unreadable from then on, with nothing to say which write broke it -- so the
    fragment carries the identity of its table and the commit checks it.
    """

    @staticmethod
    def _table(conn: Any, schema: Any = None) -> str:
        import os
        import tempfile

        location = os.path.join(tempfile.mkdtemp(), "t")
        conn.create_table(
            location, schema or pa.schema([("id", pa.int64()), ("region", pa.string())])
        )
        return location

    def test_a_fragment_cannot_be_committed_to_another_table(self, conn: Any) -> None:
        a, b = self._table(conn), self._table(conn)
        plan_a = conn.open_table(a).plan_write()
        fragment = plan_a.write(pa.table({"id": [1], "region": ["us"]}))

        plan_b = conn.open_table(b).plan_write()
        with pytest.raises(ValueError, match="written for a different table"):
            plan_b.commit([fragment])
        assert conn.open_table(b).to_arrow().num_rows == 0, "b must be untouched"
        assert conn.open_table(b).to_arrow().num_rows == 0

    def test_a_fragment_cannot_survive_the_table_being_recreated(self, conn: Any) -> None:
        """Same path, new table: the metadata id is what distinguishes them."""
        import shutil

        location = self._table(conn)
        plan = conn.open_table(location).plan_write()
        fragment = plan.write(pa.table({"id": [1], "region": ["us"]}))

        shutil.rmtree(location)
        conn.create_table(location, pa.schema([("id", pa.int64()), ("region", pa.string())]))
        with pytest.raises(ValueError, match="written for a different table"):
            plan.commit([fragment])

    def test_a_fragment_that_is_not_ours_is_refused(self, conn: Any) -> None:
        location = self._table(conn)
        plan = conn.open_table(location).plan_write()
        for junk in (b"not-arrow-ipc", b"\x00\x01\x02"):
            with pytest.raises(ValueError):
                plan.commit([junk])

    def test_an_empty_fragment_is_a_no_op(self, conn: Any) -> None:
        """A worker that received no rows still returns something committable."""
        location = self._table(conn)
        plan = conn.open_table(location).plan_write()
        empty = plan.write(
            pa.table({"id": pa.array([], pa.int64()), "region": pa.array([], pa.string())})
        )
        plan.commit([empty])
        assert conn.open_table(location).to_arrow().num_rows == 0

    def test_a_column_added_mid_flight_reads_null(self, conn: Any) -> None:
        """The fragment predates the column, which is ordinary Delta behaviour."""
        location = self._table(conn)
        plan = conn.open_table(location).plan_write()
        fragment = plan.write(pa.table({"id": [1], "region": ["us"]}))
        conn.open_table(location).add_column(pa.field("amt", pa.float64()))

        plan.commit([fragment])
        assert conn.open_table(location).to_arrow().to_pydict()["amt"] == [None]


class TestAddColumnAcceptsTheSameFormsOnEveryEngine:
    """`add_column` took pyarrow fields on the kernel path and only delta-rs
    `Field`s on the delta-rs path, so the identical call worked or raised
    depending on which engine the router picked -- a difference the caller
    cannot see."""

    @staticmethod
    def _table(conn: Any, properties: dict[str, str]) -> str:
        import os
        import tempfile

        location = os.path.join(tempfile.mkdtemp(), "t")
        conn.create_table(location, pa.schema([("id", pa.int64())]), properties=properties)
        return location

    @pytest.mark.parametrize(
        "properties",
        [{}, {"delta.enableTypeWidening": "true"}],
        ids=["deltars", "kernel"],
    )
    @pytest.mark.parametrize(
        "definition",
        [
            pa.field("a", pa.float64()),
            [pa.field("a", pa.float64())],
            {"a": "double"},
            pa.schema([("a", pa.float64())]),
        ],
        ids=["field", "list", "mapping", "schema"],
    )
    def test_every_form_is_accepted(
        self, conn: Any, properties: dict[str, str], definition: Any
    ) -> None:
        location = self._table(conn, properties)
        conn.open_table(location).add_column(definition)
        assert "a" in [f.name for f in conn.open_table(location).schema()]
