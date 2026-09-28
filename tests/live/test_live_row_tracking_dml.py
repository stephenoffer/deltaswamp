"""Kernel DML on row-tracked tables, and copy-on-write MERGE, read by Databricks.

Each test builds a table locally, runs the DML through deltaswamp (the kernel
serves it; delta-rs cannot write these tables), uploads it to a scratch UC
volume and asks the warehouse for the rows by path -- on a row-tracked table
with `_metadata.row_id` and `_metadata.row_commit_version`, which must be the
ids and versions the kernel reads, and the ones the rows had before the DML
wherever the protocol keeps them.

Opt-in on top of the live suite (it needs DELTASWAMP_TEST_WAREHOUSE_ID):

    DELTASWAMP_TEST_DATABRICKS=1 pytest tests/live/test_live_row_tracking_dml.py -v

The volume (named a8rt_...) is created in the target schema and dropped
afterwards.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

import deltaswamp as ds  # noqa: E402
from deltaswamp.engine.kernel import KernelEngine  # noqa: E402

from .test_live_interop_stats import _Volume  # noqa: E402

pytestmark = pytest.mark.databricks


class _RowTrackingVolume(_Volume):
    def __init__(self, workspace: Any, config: Any) -> None:
        self.w, self.config = workspace, config
        self.name = f"a8rt_{uuid.uuid4().hex[:10]}"
        self.root = f"/Volumes/{config.catalog}/{config.schema}/{self.name}"
        self.sql(f"CREATE VOLUME {config.prefix}.{self.name}")


@pytest.fixture(scope="module")
def volume(live_config: Any, live_workspace: Any) -> Any:
    if not live_config.warehouse_id:
        pytest.skip("DELTASWAMP_TEST_WAREHOUSE_ID is required to query files by path")
    vol = _RowTrackingVolume(live_workspace, live_config)
    try:
        yield vol
    finally:
        vol.drop()


def _kernel_row_tracking(conn: Any, path: str) -> dict[int, tuple[str, int, int]]:
    """Each row's (v, row id, row commit version), as the kernel reads them."""
    snapshot = KernelEngine().snapshot(conn.open_table(path).resolved)
    rows = pa.table(snapshot.scan(row_positions=True, row_tracking=True))
    return {
        i: (v, rid, cv)
        for i, v, rid, cv in zip(
            rows.column("id").to_pylist(),
            rows.column("v").to_pylist(),
            rows.column("__deltaswamp_row_id").to_pylist(),
            rows.column("__deltaswamp_row_commit_version").to_pylist(),
            strict=True,
        )
    }


def _databricks_row_tracking(volume: Any, ref: str) -> dict[int, tuple[str, int, int]]:
    rows = volume.sql(f"SELECT id, v, _metadata.row_id, _metadata.row_commit_version FROM {ref}")
    return {int(i): (v, int(rid), int(cv)) for i, v, rid, cv in rows}


@pytest.mark.parametrize(
    "properties",
    [
        {"delta.enableRowTracking": "true"},
        {"delta.enableRowTracking": "true", "delta.enableDeletionVectors": "true"},
    ],
    ids=["copy_on_write", "deletion_vectors"],
)
def test_row_ids_databricks_reads_after_kernel_dml(
    volume: Any, tmp_path: Any, properties: dict[str, str]
) -> None:
    conn = ds.connect()
    path = str(tmp_path / "rt")
    conn.create_table(
        path, pa.schema([("id", pa.int64()), ("v", pa.string())]), properties=properties
    )
    table = conn.open_table(path)
    table.append(pa.table({"id": pa.array([1, 2, 3], pa.int64()), "v": ["a", "b", "c"]}))
    table.append(pa.table({"id": pa.array([4, 5, 6], pa.int64()), "v": ["d", "e", "f"]}))
    start = _kernel_row_tracking(conn, path)

    table = conn.open_table(path)
    assert table.update({"v": "'B'"}, predicate="id = 2").engine == "kernel"
    assert conn.open_table(path).delete("id = 5").engine == "kernel"
    source = pa.table({"id": pa.array([3, 9], pa.int64()), "v": ["C", "i"]})
    merged = (
        conn.open_table(path)
        .merge(source=source, predicate="t.id = s.id", source_alias="s", target_alias="t")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute()
    )
    assert merged.engine == "kernel"
    conn.open_table(path).overwrite(
        pa.table({"id": pa.array([6], pa.int64()), "v": ["F"]}), predicate="id = 6"
    )
    local = _kernel_row_tracking(conn, path)
    # Rows the DML left alone keep id and version; updated ones keep their id.
    for i in (1, 4):
        assert local[i] == start[i]
    for i in (2, 3):
        assert local[i][1] == start[i][1] and local[i][2] > start[i][2]

    ref = volume.upload(path, "rt_" + "_".join(sorted(properties)).replace(".", "_"))
    assert _databricks_row_tracking(volume, ref) == local
    # Databricks writes on top, and the ids it keeps are the ones written here.
    volume.sql(f"UPDATE {ref} SET v = 'dbx' WHERE id = 1")
    after = _databricks_row_tracking(volume, ref)
    assert after[1][1] == start[1][1]
    assert {i: r for i, r in after.items() if i != 1} == {i: r for i, r in local.items() if i != 1}


def test_copy_on_write_merge_databricks_reads(volume: Any, tmp_path: Any) -> None:
    """A clustered, in-commit-timestamp table without deletion vectors: MERGE rewrites files."""
    conn = ds.connect()
    path = str(tmp_path / "cow")
    conn.create_table(
        path,
        pa.schema([("id", pa.int64()), ("v", pa.string())]),
        properties={"delta.enableInCommitTimestamps": "true"},
        cluster_by=["id"],
    )
    table = conn.open_table(path)
    table.append(pa.table({"id": pa.array([1, 2, 3], pa.int64()), "v": ["a", "b", "c"]}))
    table.append(pa.table({"id": pa.array([4, 5, 6], pa.int64()), "v": ["d", "e", "f"]}))
    source = pa.table({"id": pa.array([2, 5, 9], pa.int64()), "v": ["B", "gone", "i"]})
    result = (
        conn.open_table(path)
        .merge(source=source, predicate="t.id = s.id", source_alias="s", target_alias="t")
        .when_matched_delete("s.v = 'gone'")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute()
    )
    assert result.engine == "kernel"
    ref = volume.upload(path, "cow")
    rows = sorted((int(i), v) for i, v in volume.sql(f"SELECT id, v FROM {ref}"))
    assert rows == [(1, "a"), (2, "B"), (3, "c"), (4, "d"), (6, "f"), (9, "i")]
    history = volume.sql(f"SELECT operation FROM (DESCRIBE HISTORY {ref}) ORDER BY version DESC")
    assert history[0][0] == "MERGE"
    volume.sql(f"INSERT INTO {ref} VALUES (10, 'j')")
    assert int(volume.sql(f"SELECT count(*) FROM {ref}")[0][0]) == 7
