"""Every write mode, on a real table.

The spec has more write modes than either engine implements, and the ones that
are missing used to be silently dropped kwargs. Each mode gets a test that
checks the data afterwards, not just that the call returned.
"""

from __future__ import annotations

from typing import Any

import deltaswamp as ds
import pytest
from deltaswamp.capability import Engine, Operation
from deltaswamp.errors import DeltaSwampError, UnreachableTableError

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
def partitioned(conn: Any, tmp_path: Any) -> Any:
    """Three regions, one row each."""
    from deltalake import write_deltalake

    path = str(tmp_path / "parts")
    write_deltalake(
        path,
        pa.table({"id": [1, 2, 3], "region": ["eu", "us", "apac"]}),
        partition_by=["region"],
    )
    return conn.open_table(path)


@pytest.fixture
def plain(conn: Any, tmp_path: Any) -> Any:
    from deltalake import write_deltalake

    path = str(tmp_path / "plain")
    write_deltalake(path, pa.table({"id": [1, 2], "city": ["oslo", "lima"]}))
    return conn.open_table(path)


class TestBasicModes:
    def test_append(self, plain: Any) -> None:
        plain.append(pa.table({"id": [3], "city": ["cairo"]}))
        assert plain.count() == 3

    def test_full_overwrite(self, plain: Any) -> None:
        plain.overwrite(pa.table({"id": [9], "city": ["quito"]}))
        assert plain.to_arrow().to_pydict() == {"id": [9], "city": ["quito"]}

    def test_replace_where(self, plain: Any) -> None:
        plain.overwrite(pa.table({"id": [1], "city": ["bergen"]}), predicate="id = 1")
        got = dict(zip(*[plain.to_arrow().to_pydict()[k] for k in ("id", "city")], strict=True))
        assert got[1] == "bergen"
        assert got[2] == "lima"


class TestDynamicPartitionOverwrite:
    """Spark's partitionOverwriteMode=dynamic, emulated with a replaceWhere.

    delta-rs has no such mode, so the predicate is built from the partition
    values actually present in the incoming data.
    """

    def test_replaces_only_the_partitions_present(self, partitioned: Any) -> None:
        partitioned.overwrite(
            pa.table({"id": [20, 21], "region": ["us", "us"]}),
            partition_overwrite="dynamic",
        )
        got = partitioned.to_arrow().to_pydict()
        by_region = dict(zip(got["region"], got["id"], strict=True))
        assert sorted(got["region"]) == ["apac", "eu", "us", "us"]
        assert by_region["eu"] == 1  # untouched
        assert by_region["apac"] == 3  # untouched

    def test_unpartitioned_table_is_refused(self, plain: Any) -> None:
        with pytest.raises(UnreachableTableError, match="not partitioned"):
            plain.overwrite(pa.table({"id": [1], "city": ["x"]}), partition_overwrite="dynamic")

    def test_predicate_and_dynamic_are_mutually_exclusive(self, partitioned: Any) -> None:
        with pytest.raises(UnreachableTableError, match="cannot be combined"):
            partitioned.overwrite(
                pa.table({"id": [1], "region": ["eu"]}),
                predicate="region = 'eu'",
                partition_overwrite="dynamic",
            )

    def test_missing_partition_column_is_refused(self, partitioned: Any) -> None:
        with pytest.raises(UnreachableTableError, match="no region column"):
            partitioned.overwrite(pa.table({"id": [1]}), partition_overwrite="dynamic")

    def test_too_many_partitions_is_refused(self, partitioned: Any, monkeypatch: Any) -> None:
        from deltaswamp.engine.deltars import DeltaRsEngine

        monkeypatch.setattr(DeltaRsEngine, "max_dynamic_partitions", 1)
        with pytest.raises(UnreachableTableError, match="above the"):
            partitioned.overwrite(
                pa.table({"id": [1, 2], "region": ["eu", "us"]}),
                partition_overwrite="dynamic",
            )

    def test_unknown_mode_is_refused(self, partitioned: Any) -> None:
        with pytest.raises(UnreachableTableError, match="'static' and 'dynamic'"):
            partitioned.overwrite(
                pa.table({"id": [1], "region": ["eu"]}), partition_overwrite="sideways"
            )


class TestSchemaEvolution:
    def test_merge_widens_the_schema(self, plain: Any) -> None:
        plain.append(pa.table({"id": [3], "city": ["cairo"], "note": ["new"]}), schema_mode="merge")
        assert "note" in plain.to_arrow().column_names

    def test_merge_routes_as_merge_schema_not_append(self, plain: Any) -> None:
        """It is a distinct operation, and routing it as a plain append is how
        the request shape used to get lost."""
        assert plain.can(Operation.MERGE_SCHEMA).ok

    def test_replace_swaps_the_schema(self, plain: Any) -> None:
        plain.replace(pa.table({"other": ["a", "b"]}))
        assert plain.to_arrow().column_names == ["other"]


class TestIdempotentWrites:
    """txnAppId / txnVersion. Committing the same pair twice must be a no-op."""

    def test_replaying_the_same_txn_is_a_no_op(self, plain: Any) -> None:
        plain.append(pa.table({"id": [3], "city": ["cairo"]}), txn=("loader", 7))
        after_first = plain.count()
        plain.append(pa.table({"id": [3], "city": ["cairo"]}), txn=("loader", 7))
        assert plain.count() == after_first

    def test_a_later_version_does_commit(self, plain: Any) -> None:
        plain.append(pa.table({"id": [3], "city": ["cairo"]}), txn=("loader", 7))
        plain.append(pa.table({"id": [4], "city": ["dakar"]}), txn=("loader", 8))
        assert plain.count() == 4

    def test_txn_version_reads_back_what_was_committed(self, plain: Any) -> None:
        assert plain.txn_version("loader") is None
        plain.append(pa.table({"id": [3], "city": ["cairo"]}), txn=("loader", 7))
        assert plain.txn_version("loader") == 7


class TestCommitMetadata:
    def test_custom_metadata_reaches_the_history(self, plain: Any) -> None:
        plain.append(
            pa.table({"id": [3], "city": ["cairo"]}),
            commit_metadata={"pipeline": "nightly", "run": "42"},
        )
        entry = plain.history(1)[0]
        assert entry.get("pipeline") == "nightly"
        assert entry.get("run") == "42"


class TestConfigurationOnWriteIsRefused:
    """delta-rs accepts configuration= on a write and then discards it."""

    def test_refused_rather_than_silently_dropped(self, plain: Any) -> None:
        from deltaswamp.engine.deltars import DeltaRsEngine

        with pytest.raises(UnreachableTableError, match="silently ignores"):
            DeltaRsEngine()._write(
                plain.resolved,
                pa.table({"id": [3], "city": ["x"]}),
                mode="append",
                configuration={"delta.appendOnly": "true"},
            )


class TestKernelRefusesWhatItCannotHonor:
    """The kernel append path used to swallow every unknown kwarg."""

    def test_unsupported_options_raise_rather_than_no_op(self, plain: Any) -> None:
        from deltaswamp.engine.kernel import KernelEngine

        with pytest.raises(DeltaSwampError, match="does not implement"):
            KernelEngine().append(
                plain.resolved, pa.table({"id": [3], "city": ["x"]}), schema_mode="merge"
            )


class TestIdempotencyIsEnforcedHere:
    """Neither engine deduplicates, so the library must.

    Verified against delta-rs 1.6.5: passing the same (app_id, version) twice
    records the txn action and appends the rows again. The guard compares against
    the last committed version before writing.
    """

    def test_delta_rs_alone_does_not_deduplicate(self, tmp_path: Any) -> None:
        """Pins the engine behavior the guard exists to compensate for."""
        from deltalake import CommitProperties, DeltaTable, Transaction, write_deltalake

        path = str(tmp_path / "raw")
        write_deltalake(path, pa.table({"id": [1]}))
        props = CommitProperties(app_transactions=[Transaction(app_id="a", version=1)])
        write_deltalake(path, pa.table({"id": [2]}), mode="append", commit_properties=props)
        write_deltalake(path, pa.table({"id": [2]}), mode="append", commit_properties=props)
        assert DeltaTable(path).to_pyarrow_table().num_rows == 3, (
            "delta-rs deduplicated after all; the guard in Table can be simplified"
        )

    def test_overwrite_replay_is_also_skipped(self, plain: Any) -> None:
        plain.overwrite(pa.table({"id": [9], "city": ["quito"]}), txn=("loader", 3))
        rows = plain.count()
        plain.overwrite(pa.table({"id": [9], "city": ["quito"]}), txn=("loader", 3))
        assert plain.count() == rows

    def test_an_earlier_version_is_also_skipped(self, plain: Any) -> None:
        plain.append(pa.table({"id": [3], "city": ["cairo"]}), txn=("loader", 10))
        rows = plain.count()
        plain.append(pa.table({"id": [4], "city": ["dakar"]}), txn=("loader", 9))
        assert plain.count() == rows


class TestWriteTableSaveModes:
    """Spark's save modes, including the create-if-missing cases the Table API
    could not express because it required an existing table."""

    def test_creates_when_absent(self, conn: Any, tmp_path: Any) -> None:
        t = conn.write_table(str(tmp_path / "new"), pa.table({"id": [1, 2]}))
        assert t.count() == 2

    def test_error_mode_refuses_an_existing_table(self, conn: Any, plain: Any) -> None:
        with pytest.raises(UnreachableTableError, match="already exists"):
            conn.write_table(plain.location, pa.table({"id": [3], "city": ["x"]}))

    def test_ignore_mode_is_a_no_op_when_present(self, conn: Any, plain: Any) -> None:
        before = plain.count()
        conn.write_table(plain.location, pa.table({"id": [3], "city": ["x"]}), mode="ignore")
        assert conn.open_table(plain.location).count() == before

    def test_append_mode_creates_then_appends(self, conn: Any, tmp_path: Any) -> None:
        target = str(tmp_path / "grow")
        conn.write_table(target, pa.table({"id": [1]}), mode="append")
        conn.write_table(target, pa.table({"id": [2]}), mode="append")
        assert conn.open_table(target).count() == 2

    def test_overwrite_mode_replaces(self, conn: Any, plain: Any) -> None:
        conn.write_table(plain.location, pa.table({"id": [9], "city": ["quito"]}), mode="overwrite")
        assert conn.open_table(plain.location).to_arrow().to_pydict()["id"] == [9]

    def test_unknown_mode_is_refused(self, conn: Any, tmp_path: Any) -> None:
        with pytest.raises(UnreachableTableError, match="'error', 'ignore'"):
            conn.write_table(str(tmp_path / "x"), pa.table({"id": [1]}), mode="sideways")

    def test_schema_is_inferred_from_the_data(self, conn: Any, tmp_path: Any) -> None:
        t = conn.write_table(str(tmp_path / "inferred"), pa.table({"a": [1], "b": ["x"]}))
        assert t.to_arrow().column_names == ["a", "b"]

    def test_table_exists(self, conn: Any, plain: Any, tmp_path: Any) -> None:
        assert conn.table_exists(plain.location) is True
        assert conn.table_exists(str(tmp_path / "nothing-here")) is False


class TestKernelOverwrite:
    """The kernel path removes every file in the snapshot in the same commit.

    It is the only route to overwriting a catalog-managed table, since delta-rs
    cannot open one at all.
    """

    def test_overwrite_replaces_everything(self, plain: Any) -> None:
        from deltaswamp.engine.kernel import KernelEngine

        KernelEngine().overwrite(plain.resolved, pa.table({"id": [9], "city": ["quito"]}))
        assert plain.to_arrow().to_pydict() == {"id": [9], "city": ["quito"]}

    def test_delta_rs_agrees_after_a_kernel_overwrite(self, plain: Any) -> None:
        from deltalake import DeltaTable
        from deltaswamp.engine.kernel import KernelEngine

        KernelEngine().overwrite(plain.resolved, pa.table({"id": [9], "city": ["quito"]}))
        assert DeltaTable(plain.location).to_pyarrow_table().num_rows == 1

    def test_it_is_a_single_commit(self, plain: Any) -> None:
        """Removes and adds land together, so no reader sees an empty table."""
        from deltaswamp.engine.kernel import KernelEngine

        before = len(plain.history())
        KernelEngine().overwrite(plain.resolved, pa.table({"id": [9], "city": ["quito"]}))
        assert len(plain.history()) == before + 1

    def test_predicate_overwrite_rewrites_only_matching_rows(self, plain: Any) -> None:
        """A predicate overwrite on the kernel is a whole-table rewrite: rows
        matching the predicate are replaced, every other row survives."""
        from deltalake import DeltaTable
        from deltaswamp.engine.kernel import KernelEngine

        KernelEngine().overwrite(
            plain.resolved, pa.table({"id": [9], "city": ["q"]}), predicate="id = 1"
        )
        got = DeltaTable(plain.location).to_pyarrow_table().to_pylist()
        assert {r["id"] for r in got} == {2, 9}

    def test_kernel_txn_and_commit_metadata(self, plain: Any) -> None:
        from deltaswamp.engine.kernel import KernelEngine

        KernelEngine().append(
            plain.resolved,
            pa.table({"id": [3], "city": ["cairo"]}),
            txn=("loader", 4),
            commit_metadata={"run": "7"},
        )
        assert plain.history(1)[0].get("run") == "7"
        assert plain.txn_version("loader") == 4
