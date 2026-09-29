"""`Connection.plan_write` of a table that is not there yet: created by the commit.

What a Ray ``write_delta("cat.schema.new_table")`` leans on: nothing visible
(and nothing registered) until the job's data can land, the table and its
data appearing together, and nothing left behind -- no files, no empty table,
no catalog entry -- when the job fails or is aborted.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
from typing import Any
from urllib.parse import urlparse

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

from deltaswamp.errors import (  # noqa: E402
    InvalidArgumentError,
    MetadataChangedError,
    TransientCommitError,
    UnreachableTableError,
)

from tests.fake_uc import FakeUnityCatalog  # noqa: E402
from tests.helpers import direct_router  # noqa: E402

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

SCHEMA = pa.schema([("id", pa.int64()), ("region", pa.string())])


def _rows(ids: list[int], region: str = "eu") -> Any:
    return pa.table({"id": pa.array(ids, pa.int64()), "region": [region] * len(ids)})


def _local(location: str) -> str:
    return urlparse(location).path if location.startswith("file://") else location


def _parquet(location: str) -> set[str]:
    root = _local(location)
    return {
        os.path.relpath(p, root)
        for p in glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
        if "_delta_log" not in p
    }


def _log(location: str) -> list[str]:
    log_dir = os.path.join(_local(location), "_delta_log")
    return (
        sorted(n for n in os.listdir(log_dir) if n.endswith(".json"))
        if os.path.isdir(log_dir)
        else []
    )


def _pending(location: str) -> list[str]:
    root = os.path.join(_local(location), "_deltaswamp_pending")
    return sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))


def _ids(table: Any) -> list[int]:
    return sorted(table.to_arrow().column("id").to_pylist())


def _worker_write(plan: Any, *batches: Any) -> list[bytes]:
    """Write `batches` as a worker would: through a pickled copy of the plan."""
    worker = pickle.loads(pickle.dumps(plan))
    return [worker.write(b) for b in batches]


# ------------------------------------------------------------------- paths


class TestPathTable:
    def test_the_table_appears_with_its_data_at_commit(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        plan = conn.plan_write(path, schema=SCHEMA, partition_by=["region"], mode="error")
        assert not conn.table_exists(path)
        assert _log(path) == []

        fragments = _worker_write(plan, _rows([1, 2]), _rows([3], "us"))
        # Written under the real root, in the real layout.
        assert {p.split("/")[0] for p in _parquet(path)} == {"region=eu", "region=us"}
        assert not conn.table_exists(path)

        assert plan.commit(fragments) == 1
        table = conn.table(path)
        assert _ids(table) == [1, 2, 3]
        assert list(table._enrich().partition_columns) == ["region"]
        assert _log(path) == [f"{0:020}.json", f"{1:020}.json"]
        # The template is gone once the table is there.
        assert _pending(path) == []
        assert not os.path.exists(os.path.join(path, "_deltaswamp_pending"))

    def test_version_zero_is_what_create_table_writes(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "cm")
        properties = {"delta.columnMapping.mode": "name", "delta.enableDeletionVectors": "true"}
        plan = conn.plan_write(path, schema=SCHEMA, properties=properties, comment="orders")
        plan.commit(_worker_write(plan, _rows([1, 2])))

        table = conn.table(path)
        assert _ids(table) == [1, 2]
        assert table.properties()["delta.columnMapping.mode"] == "name"
        with open(os.path.join(path, "_delta_log", f"{0:020}.json")) as f:
            v0 = [json.loads(line) for line in f if line.strip()]
        meta = next(a["metaData"] for a in v0 if "metaData" in a)
        assert meta["description"] == "orders"
        assert (
            "columnMapping" in next(a["protocol"] for a in v0 if "protocol" in a)["writerFeatures"]
        )
        # The files were written with the physical names version 0 assigned.
        assert table.count() == 2

    def test_no_data_still_creates_the_table(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "empty")
        plan = conn.plan_write(path, schema=SCHEMA)
        assert plan.commit([]) == 0
        assert conn.table(path).count() == 0

    def test_committing_again_returns_the_version(self, conn: Any, tmp_path: Any) -> None:
        """A driver that lost the answer commits the same fragments again."""
        path = str(tmp_path / "t")
        plan = conn.plan_write(path, schema=SCHEMA)
        fragments = _worker_write(plan, _rows([1, 2]))
        assert plan.commit(fragments) == 1
        assert plan.commit(fragments) == 1
        assert _ids(conn.table(path)) == [1, 2]

    def test_a_version_zero_that_landed_is_not_published_again(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """Version 0 landed, then the data commit's outcome was lost: commit again."""
        path = str(tmp_path / "t")
        plan = conn.plan_write(path, schema=SCHEMA)
        fragments = _worker_write(plan, _rows([1, 2]))

        def unknown(*_: Any, **__: Any) -> int:
            raise TransientCommitError("the put timed out")

        monkeypatch.setattr(plan.engine, "commit_files", unknown)
        with pytest.raises(TransientCommitError):
            plan.commit(fragments, retries=0)
        # Nothing deleted, as the outcome was unknown.
        assert _parquet(path)
        monkeypatch.undo()
        assert plan.commit(fragments) == 1
        assert _ids(conn.table(path)) == [1, 2]

    def test_a_worker_copy_cannot_commit_or_abort(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        plan = conn.plan_write(path, schema=SCHEMA)
        worker = pickle.loads(pickle.dumps(plan))
        assert worker.create is None
        fragment = worker.write(_rows([1]))
        with pytest.raises(UnreachableTableError, match="driver"):
            worker.commit([fragment])
        with pytest.raises(UnreachableTableError, match="driver"):
            worker.abort([fragment])
        assert _log(path) == []

    def test_the_pickled_plan_carries_no_creator(self, conn: Any, tmp_path: Any) -> None:
        plan = conn.plan_write(str(tmp_path / "t"), schema=SCHEMA)
        assert b"_Creator" not in pickle.dumps(plan)


class TestNothingLeftBehind:
    def test_abort_deletes_the_files_and_the_template(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        plan = conn.plan_write(path, schema=SCHEMA)
        fragments = _worker_write(plan, _rows([1, 2]), _rows([3], "us"))
        assert _parquet(path)
        assert plan.abort(fragments) == 2
        assert _parquet(path) == set()
        assert _log(path) == []
        assert _pending(path) == []
        assert not conn.table_exists(path)
        # And the name is free for the retry.
        again = conn.plan_write(path, schema=SCHEMA, mode="error")
        assert again.commit(_worker_write(again, _rows([7]))) == 1

    def test_a_certain_failure_undoes_the_create(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        path = str(tmp_path / "t")
        plan = conn.plan_write(path, schema=SCHEMA)
        fragments = _worker_write(plan, _rows([1, 2]))

        def changed(*_: Any, **__: Any) -> int:
            raise MetadataChangedError(0, "the schema changed")

        monkeypatch.setattr(plan.engine, "commit_files", changed)
        with pytest.raises(MetadataChangedError):
            plan.commit(fragments)
        assert _parquet(path) == set()
        assert _log(path) == []
        assert _pending(path) == []
        monkeypatch.undo()
        assert not conn.table_exists(path)

    def test_abort_after_the_data_landed_is_refused(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        plan = conn.plan_write(path, schema=SCHEMA)
        fragments = _worker_write(plan, _rows([1]))
        plan.commit(fragments)
        with pytest.raises(UnreachableTableError, match="committed"):
            plan.abort(fragments)
        assert _ids(conn.table(path)) == [1]

    def test_a_refused_plan_leaves_nothing(self, conn: Any, tmp_path: Any) -> None:
        """Refused before any worker runs, and the template it wrote is gone again."""
        path = str(tmp_path / "t")
        with pytest.raises(ds.DeltaSwampError):
            conn.plan_write(path, schema=SCHEMA, txn=("job", -1))
        assert not os.path.exists(os.path.join(path, "_deltaswamp_pending"))
        assert not conn.table_exists(path)

    def test_a_table_the_engines_cannot_create_is_refused_at_planning(
        self, conn: Any, tmp_path: Any
    ) -> None:
        """Composing version 0 is the create's own check, before anything is written."""
        path = str(tmp_path / "uniform")
        properties = {
            "delta.columnMapping.mode": "name",
            "delta.enableIcebergCompatV2": "true",
            "delta.universalFormat.enabledFormats": "iceberg",
        }
        with pytest.raises(UnreachableTableError, match="cannot create"):
            conn.plan_write(path, schema=SCHEMA, properties=properties)
        assert not os.path.exists(path)


class TestAnotherCreator:
    """Another writer creates the table while the job runs."""

    def _race(self, conn: Any, tmp_path: Any, mode: str, schema: Any = SCHEMA) -> Any:
        path = str(tmp_path / "t")
        plan = conn.plan_write(path, schema=SCHEMA, mode=mode)
        fragments = _worker_write(plan, _rows([1, 2]))
        mine = _parquet(path)
        theirs = conn.create_table(path, schema)
        first = 100 if schema.field("id").type == pa.int64() else "x"
        theirs.append(pa.table({"id": pa.array([first], schema.field("id").type), "region": ["x"]}))
        return path, plan, fragments, mine

    def test_error_fails_and_deletes_the_files(self, conn: Any, tmp_path: Any) -> None:
        path, plan, fragments, mine = self._race(conn, tmp_path, "error")
        with pytest.raises(UnreachableTableError, match="another writer created"):
            plan.commit(fragments)
        assert not mine & _parquet(path)
        assert _ids(conn.table(path)) == [100]
        assert _pending(path) == []

    def test_ignore_keeps_theirs(self, conn: Any, tmp_path: Any) -> None:
        path, plan, fragments, mine = self._race(conn, tmp_path, "ignore")
        assert plan.commit(fragments) == 1
        assert not mine & _parquet(path)
        assert _ids(conn.table(path)) == [100]

    def test_append_joins_a_table_of_the_same_layout(self, conn: Any, tmp_path: Any) -> None:
        path, plan, fragments, _ = self._race(conn, tmp_path, "append")
        assert plan.commit(fragments) == 2
        assert _ids(conn.table(path)) == [1, 2, 100]

    def test_two_creating_jobs_both_land(self, conn: Any, tmp_path: Any) -> None:
        """Both planned before either committed: the second appends to the first's table."""
        path = str(tmp_path / "t")
        first = conn.plan_write(path, schema=SCHEMA, mode="append")
        second = conn.plan_write(path, schema=SCHEMA, mode="append")
        a = _worker_write(first, _rows([1, 2]))
        b = _worker_write(second, _rows([3]))
        assert first.commit(a) == 1
        assert second.commit(b) == 2
        assert _ids(conn.table(path)) == [1, 2, 3]
        assert _pending(path) == []

    def test_append_refuses_another_layout(self, conn: Any, tmp_path: Any) -> None:
        other = pa.schema([("id", pa.string()), ("region", pa.string())])
        path, plan, fragments, mine = self._race(conn, tmp_path, "append", other)
        with pytest.raises(UnreachableTableError, match="layout differs"):
            plan.commit(fragments)
        assert not mine & _parquet(path)


class TestExistingTable:
    def test_modes(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        conn.create_table(path, SCHEMA, partition_by=["region"])
        with pytest.raises(UnreachableTableError, match="already exists"):
            conn.plan_write(path, schema=SCHEMA, mode="error")
        assert conn.plan_write(path, mode="ignore") is None

        plan = conn.plan_write(path, mode="append")
        assert plan.create is None
        assert plan.commit(_worker_write(plan, _rows([1]))) == 1
        plan = conn.plan_write(path, mode="overwrite")
        assert plan.commit(_worker_write(plan, _rows([2]))) == 2
        assert _ids(conn.table(path)) == [2]

    def test_create_time_layout_must_agree(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        conn.create_table(path, SCHEMA)
        with pytest.raises(InvalidArgumentError, match="partition_by"):
            conn.plan_write(path, mode="append", partition_by=["region"])

    def test_a_new_table_needs_a_schema(self, conn: Any, tmp_path: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="schema="):
            conn.plan_write(str(tmp_path / "t"))

    def test_an_unknown_mode_is_refused(self, conn: Any, tmp_path: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="mode"):
            conn.plan_write(str(tmp_path / "t"), schema=SCHEMA, mode="upsert")


# ----------------------------------------------------------- Unity Catalog


@pytest.fixture
def uc(tmp_path: Any) -> Any:
    with FakeUnityCatalog(staging_root=tmp_path / "managed") as server:
        yield server


@pytest.fixture
def uc_conn(uc: Any) -> Any:
    from deltaswamp.catalog.ossuc import OSSUnityCatalog

    catalog = OSSUnityCatalog(uc.url)
    catalog.create_catalog("main")
    catalog.create_schema("main", "sales")
    return ds.Connection(catalog=catalog, router=direct_router())


class TestManaged:
    def test_registered_by_the_commit_with_its_data(self, uc_conn: Any, uc: Any) -> None:
        plan = uc_conn.plan_write("main.sales.orders", schema=SCHEMA, comment="orders")
        # Allocated, but not registered: nothing to see.
        assert not uc_conn.table_exists("main.sales.orders")
        assert "main.sales.orders" not in uc.tables
        fragments = _worker_write(plan, _rows([1, 2]))
        assert plan.commit(fragments) == 1

        table = uc_conn.table("main.sales.orders")
        assert table.is_catalog_managed
        assert _ids(table) == [1, 2]
        # Version 1 was ratified by the catalog, not written to the log.
        assert len(uc.commit_log) == 1
        published = os.path.join(_local(table.location), "_delta_log", f"{1:020}.json")
        assert not os.path.exists(published)
        info = uc_conn.catalog.table_info(table.resolved.ref)
        assert info.table_type == "MANAGED"
        assert _pending(table.location) == []

    def test_abort_leaves_no_table(self, uc_conn: Any, uc: Any) -> None:
        plan = uc_conn.plan_write("main.sales.orders", schema=SCHEMA)
        fragments = _worker_write(plan, _rows([1, 2]))
        location = plan.table.location
        assert _parquet(location)
        plan.abort(fragments)
        assert _parquet(location) == set()
        assert not uc_conn.table_exists("main.sales.orders")

    def test_a_certain_failure_after_registering_drops_the_table(
        self, uc_conn: Any, uc: Any, monkeypatch: Any
    ) -> None:
        plan = uc_conn.plan_write("main.sales.orders", schema=SCHEMA)
        fragments = _worker_write(plan, _rows([1, 2]))
        location = plan.table.location

        def changed(*_: Any, **__: Any) -> int:
            raise MetadataChangedError(0, "the schema changed")

        monkeypatch.setattr(plan.engine, "commit_files", changed)
        with pytest.raises(MetadataChangedError):
            plan.commit(fragments)
        assert _parquet(location) == set()
        assert not uc_conn.table_exists("main.sales.orders")

    def test_committing_again_after_a_lost_answer(self, uc_conn: Any, uc: Any) -> None:
        plan = uc_conn.plan_write("main.sales.orders", schema=SCHEMA)
        fragments = _worker_write(plan, _rows([1, 2]))
        assert plan.commit(fragments) == 1
        assert plan.commit(fragments) == 1
        assert _ids(uc_conn.table("main.sales.orders")) == [1, 2]

    def test_another_creator_wins_the_name(self, uc_conn: Any, uc: Any) -> None:
        plan = uc_conn.plan_write("main.sales.orders", schema=SCHEMA, mode="append")
        fragments = _worker_write(plan, _rows([1, 2]))
        location = plan.table.location
        uc_conn.create_table("main.sales.orders", SCHEMA)
        with pytest.raises(UnreachableTableError, match="another writer created"):
            plan.commit(fragments)
        assert _parquet(location) == set()
        assert uc_conn.table("main.sales.orders").count() == 0

    def test_workers_carry_no_catalog_token(self, uc_conn: Any, uc: Any) -> None:
        plan = uc_conn.plan_write("main.sales.orders", schema=SCHEMA)
        worker = pickle.loads(pickle.dumps(plan))
        assert worker.catalog is None and worker.create is None
        # The storage credential vended on the driver, not a way to vend more.
        assert type(worker.table.credential_provider).__name__ == "ShippedCredentials"

    def test_ship_catalog_auth_is_refused(self, uc_conn: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="ship_catalog_auth"):
            uc_conn.plan_write("main.sales.orders", schema=SCHEMA, ship_catalog_auth=True)


class TestExternal:
    def test_registered_after_its_data_is_in(self, uc_conn: Any, uc: Any, tmp_path: Any) -> None:
        location = f"file://{tmp_path / 'ext'}"
        plan = uc_conn.plan_write(
            "main.sales.ext", schema=SCHEMA, location=location, partition_by=["region"]
        )
        assert "main.sales.ext" not in uc.tables
        fragments = _worker_write(plan, _rows([1, 2]), _rows([3], "us"))
        assert "main.sales.ext" not in uc.tables
        assert plan.commit(fragments) == 1

        table = uc_conn.table("main.sales.ext")
        assert _ids(table) == [1, 2, 3]
        info = uc_conn.catalog.table_info(table.resolved.ref)
        assert info.table_type == "EXTERNAL"
        assert info.partition_columns == ("region",)
        # Committing again finds both the data and the registration there.
        assert plan.commit(fragments) == 1

    def test_a_failed_registration_names_the_location(
        self, uc_conn: Any, uc: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        location = f"file://{tmp_path / 'ext'}"
        plan = uc_conn.plan_write("main.sales.ext", schema=SCHEMA, location=location)
        fragments = _worker_write(plan, _rows([1]))

        def refused(*_: Any, **__: Any) -> Any:
            raise ds.DeltaSwampError("permission denied")

        monkeypatch.setattr(uc_conn.catalog, "register_table", refused)
        with pytest.raises(UnreachableTableError, match="register_table") as caught:
            plan.commit(fragments)
        assert location in str(caught.value)
        monkeypatch.undo()
        # The data is committed; registering is all that is left, and repeats.
        registered = uc_conn.register_table("main.sales.ext", location)
        assert _ids(registered) == [1]
        assert uc_conn.register_table("main.sales.ext", location).count() == 1

    def test_abort_leaves_nothing(self, uc_conn: Any, uc: Any, tmp_path: Any) -> None:
        location = f"file://{tmp_path / 'ext'}"
        plan = uc_conn.plan_write("main.sales.ext", schema=SCHEMA, location=location)
        fragments = _worker_write(plan, _rows([1]))
        plan.abort(fragments)
        assert _parquet(location) == set()
        assert _log(location) == []
        assert "main.sales.ext" not in uc.tables

    def test_a_non_empty_location_is_refused_before_the_job(
        self, uc_conn: Any, tmp_path: Any
    ) -> None:
        folder = tmp_path / "ext"
        folder.mkdir()
        (folder / "notes.txt").write_text("not a table")
        with pytest.raises(UnreachableTableError, match="not a Delta table"):
            uc_conn.plan_write("main.sales.ext", schema=SCHEMA, location=f"file://{folder}")
        assert not (folder / "_deltaswamp_pending").exists()
