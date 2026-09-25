"""The kernel commit path.

Exercised here against a path-based table with `FileSystemCommitter`, which is
the same `Transaction` machinery a catalog-managed commit uses -- only the
committer differs. The UC-specific half runs against `tests/fake_uc.py` in
`test_catalog_managed.py`.
"""

from __future__ import annotations

from typing import Any

import deltaswamp as ds
import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

from tests import helpers  # noqa: E402


@pytest.fixture
def path(tmp_path: Any) -> str:
    from deltalake import write_deltalake

    p = str(tmp_path / "tbl")
    write_deltalake(p, pa.table({"id": [1, 2, 3], "city": ["oslo", "lima", "cairo"]}))
    return p


class TestAppend:
    def test_returns_the_committed_version(self, path: str) -> None:
        version = helpers.snapshot(path).append(
            pa.table({"id": [4], "city": ["dakar"]}).to_reader()
        )
        assert version == 1

    def test_rows_are_visible_afterwards(self, path: str) -> None:
        helpers.snapshot(path).append(pa.table({"id": [4], "city": ["dakar"]}).to_reader())
        assert pa.table(helpers.snapshot(path).scan()).num_rows == 4

    def test_delta_rs_agrees(self, path: str) -> None:
        """The commit must be readable by an independent implementation, not
        just by the one that wrote it."""
        from deltalake import DeltaTable

        helpers.snapshot(path).append(pa.table({"id": [4], "city": ["dakar"]}).to_reader())
        dt = DeltaTable(path)
        assert dt.version() == 1
        assert set(dt.to_pyarrow_table().to_pydict()["city"]) == {
            "oslo",
            "lima",
            "cairo",
            "dakar",
        }

    def test_operation_and_engine_info_reach_the_log(self, path: str) -> None:
        from deltalake import DeltaTable

        helpers.snapshot(path).append(
            pa.table({"id": [4], "city": ["dakar"]}).to_reader(),
            engine_info="test-engine",
            operation="CUSTOM_WRITE",
        )
        entry = DeltaTable(path).history(1)[0]
        assert entry.get("operation") == "CUSTOM_WRITE"
        assert entry.get("engineInfo") == "test-engine"

    def test_successive_appends_increment_version(self, path: str) -> None:
        assert helpers.snapshot(path).append(pa.table({"id": [4], "city": ["a"]}).to_reader()) == 1
        assert helpers.snapshot(path).append(pa.table({"id": [5], "city": ["b"]}).to_reader()) == 2


class TestPartitionGuard:
    def test_partition_columns_are_reported(self, tmp_path: Any) -> None:
        from deltalake import write_deltalake

        p = str(tmp_path / "part")
        write_deltalake(p, pa.table({"id": [1], "region": ["eu"]}), partition_by=["region"])
        assert helpers.snapshot(p).partition_columns == ["region"]

    def test_unpartitioned_table_reports_none(self, path: str) -> None:
        assert helpers.snapshot(path).partition_columns == []

    def test_engine_appends_to_a_partitioned_table(self, tmp_path: Any) -> None:
        """Rows must land in their partition with the right values, which an
        independent reader (delta-rs) confirms -- the failure mode here is wrong
        data, not an error."""
        from deltalake import DeltaTable, write_deltalake
        from deltaswamp.catalog import ResolvedTable
        from deltaswamp.engine.kernel import KernelEngine
        from deltaswamp.identity import parse_ref

        p = str(tmp_path / "part")
        write_deltalake(p, pa.table({"id": [1], "region": ["eu"]}), partition_by=["region"])
        table = ResolvedTable(ref=parse_ref(p), location=p, partition_columns=("region",))
        KernelEngine().append(table, pa.table({"id": [2, 3], "region": ["us", None]}))
        got = DeltaTable(p).to_pyarrow_table().sort_by("id").to_pydict()
        assert got == {"id": [1, 2, 3], "region": ["eu", "us", None]}
        partitions = {tuple(sorted(d.items())) for d in DeltaTable(p).partitions()}
        assert (("region", "us"),) in partitions


class TestPublish:
    def test_publish_is_a_no_op_on_a_path_based_table(self, path: str) -> None:
        """A path-based table has no staged commits, so publish should return
        the current version rather than failing."""
        assert helpers.snapshot(path).publish() == 0


class TestExceptionsAreDistinguishable:
    """409 and 429 mean different things and must not be conflated."""

    def test_native_exposes_distinct_exception_types(self) -> None:
        from deltaswamp import _native

        # Distinct types, so `except CommitConflictError` cannot accidentally
        # swallow a backfill demand -- which needs a publish, not a retry.
        assert issubclass(_native.CommitConflictError, RuntimeError)
        assert issubclass(_native.BackfillRequiredError, RuntimeError)
        assert not issubclass(_native.CommitConflictError, _native.BackfillRequiredError)
        assert not issubclass(_native.BackfillRequiredError, _native.CommitConflictError)
