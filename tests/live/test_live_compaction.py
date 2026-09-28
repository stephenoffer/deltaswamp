"""Kernel compactions of the tables delta-rs cannot optimize, read by path on Databricks.

Row-tracked tables (every row keeps `_metadata.row_id` and
`_metadata.row_commit_version`), liquid-clustered ones (clustered by their
keys, the keys intact), column-mapped ones (name and id mode, partitioned),
and an overwrite of a row-tracked table. Each table is written locally,
uploaded to a scratch UC volume before and after, and queried there.

Opt-in on top of the live suite (it needs DELTASWAMP_TEST_WAREHOUSE_ID):

    DELTASWAMP_TEST_DATABRICKS=1 pytest tests/live/test_live_compaction.py -v

The volume is created in the target schema and dropped afterwards.
"""

from __future__ import annotations

import json
import random
import shutil
import uuid
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

import deltaswamp as ds  # noqa: E402

from .test_live_interop_stats import _Volume  # noqa: E402

pytestmark = pytest.mark.databricks

TRACKED = "SELECT id, v, _metadata.row_id, _metadata.row_commit_version FROM {} ORDER BY id"


@pytest.fixture(scope="module")
def volume(live_config: Any, live_workspace: Any) -> Any:
    if not live_config.warehouse_id:
        pytest.skip("DELTASWAMP_TEST_WAREHOUSE_ID is required to query files by path")
    vol = _Volume(live_workspace, live_config)
    try:
        yield vol
    finally:
        vol.drop()


def _upload(volume: Any, path: str, name: str) -> str:
    # A fresh name for every upload: the warehouse caches a path's log.
    return str(volume.upload(path, f"{name}_{uuid.uuid4().hex[:8]}"))


def _rows(start: int, n: int) -> Any:
    ids = range(start, start + n)
    return pa.table({"id": pa.array(ids, pa.int64()), "v": [f"r{i}" for i in ids]})


def _tracked(tmp_path: Any, name: str, dv: bool) -> Any:
    conn = ds.connect("file://")
    properties = {"delta.enableRowTracking": "true"}
    if dv:
        properties["delta.enableDeletionVectors"] = "true"
    path = str(tmp_path / name)
    schema = pa.schema([("id", pa.int64()), ("v", pa.string())])
    table = conn.create_table(path, schema, properties=properties)
    for i in range(4):
        table.append(_rows(i * 5, 5))
    table = conn.open_table(path)
    if dv:
        table.delete("id = 3")
        table.update(predicate="id = 7", updates={"v": "'upd'"})
    return conn.open_table(path)


@pytest.mark.parametrize("dv", [False, True], ids=["plain", "deletion_vectors"])
def test_optimize_keeps_row_ids_and_commit_versions(volume: Any, tmp_path: Any, dv: bool) -> None:
    table = _tracked(tmp_path, f"rt_{dv}", dv)
    before = volume.sql(TRACKED.format(_upload(volume, table.location, "rt_before")))
    assert table.optimize()["numFilesAdded"] == 1
    ref = _upload(volume, table.location, "rt_after")
    assert volume.sql(TRACKED.format(ref)) == before
    # Databricks writes on after it: fresh ids above the high-water mark.
    volume.sql(f"INSERT INTO {ref} VALUES (1000, 'db')")
    count, distinct = volume.sql(f"SELECT count(*), count(DISTINCT _metadata.row_id) FROM {ref}")[0]
    assert count == distinct == str(len(before) + 1)


@pytest.mark.parametrize("dv", [False, True], ids=["plain", "deletion_vectors"])
def test_overwrite_of_a_row_tracked_table(volume: Any, tmp_path: Any, dv: bool) -> None:
    table = _tracked(tmp_path, f"rtow_{dv}", dv)
    table.optimize()
    ds.connect("file://").open_table(table.location).overwrite(_rows(100, 2))
    rows = volume.sql(TRACKED.format(_upload(volume, table.location, "rt_ow")))
    assert [r[:2] for r in rows] == [["100", "r100"], ["101", "r101"]]
    # New rows, new ids: past every id the replaced rows had.
    assert sorted(int(r[2]) for r in rows) == [40, 41]


def test_optimize_clusters_a_liquid_table_by_its_keys(volume: Any, tmp_path: Any) -> None:
    rng = random.Random(7)
    conn = ds.connect("file://")
    path = str(tmp_path / "cl")
    schema = pa.schema([("id", pa.int64()), ("g", pa.int32()), ("h", pa.int32())])
    table = conn.create_table(
        path,
        schema,
        cluster_by=["g", "h"],
        properties={"delta.enableRowTracking": "true", "delta.enableDeletionVectors": "true"},
    )
    for i in range(12):
        table.append(
            pa.table(
                {
                    "id": pa.array(range(i * 200, i * 200 + 200), pa.int64()),
                    "g": pa.array([rng.randrange(100) for _ in range(200)], pa.int32()),
                    "h": pa.array([rng.randrange(100) for _ in range(200)], pa.int32()),
                }
            )
        )
    table = conn.open_table(path)
    predicates = ["g = 17", "h = 42", "g = 3 AND h = 90"]
    query = "SELECT id, g, h, _metadata.row_id, _metadata.row_commit_version FROM {} ORDER BY id"

    def answers(ref: str) -> dict[str, Any]:
        return {
            p: (
                volume.sql(f"SELECT id FROM {ref} WHERE {p} ORDER BY id"),
                int(
                    volume.sql(f"SELECT count(DISTINCT _metadata.file_path) FROM {ref} WHERE {p}")[
                        0
                    ][0]
                ),
            )
            for p in predicates
        }

    ref = _upload(volume, path, "cl_before")
    before, before_answers = volume.sql(query.format(ref)), answers(ref)
    verdict = table.can("optimize")
    assert verdict.ok and "clustering keys" in verdict.reason
    assert table.optimize(target_size=8000)["numFilesAdded"] > 1
    ref = _upload(volume, path, "cl_after")
    assert volume.sql(query.format(ref)) == before
    after_answers = answers(ref)
    for p in predicates:
        assert after_answers[p][0] == before_answers[p][0], p
    for p in predicates[:2]:
        assert after_answers[p][1] < before_answers[p][1], (p, before_answers[p], after_answers[p])
    detail = volume.sql(f"DESCRIBE DETAIL {ref}")[0]
    assert json.loads(detail[8]) == ["g", "h"]
    # Databricks reclusters on after it.
    volume.sql(f"OPTIMIZE {ref}")
    assert volume.sql(f"SELECT count(*) FROM {ref}") == [["2400"]]
    assert conn.open_table(path).optimize(target_size=8000)["numFilesAdded"] == 0


@pytest.mark.parametrize("mode", ["name", "id"])
def test_optimize_of_a_column_mapped_partitioned_table(
    volume: Any, tmp_path: Any, mode: str
) -> None:
    conn = ds.connect("file://")
    path = str(tmp_path / f"cm_{mode}")
    table = conn.create_table(
        path,
        pa.schema([("id", pa.int64()), ("v", pa.string()), ("part", pa.string())]),
        partition_by=["part"],
        properties={"delta.columnMapping.mode": mode},
    )
    for i in range(3):
        table.append(
            pa.table(
                {"id": pa.array([i, i + 10], pa.int64()), "v": [f"a{i}", "b"], "part": ["x", "y"]}
            )
        )
    conn.open_table(path).rename_column("v", "value")
    query = "SELECT id, value, part FROM {} ORDER BY id"
    before = volume.sql(query.format(_upload(volume, path, f"cm_{mode}_before")))
    assert conn.open_table(path).optimize()["numFilesAdded"] == 2
    ref = _upload(volume, path, f"cm_{mode}_after")
    assert volume.sql(query.format(ref)) == before
    assert volume.sql(f"SELECT count(*) FROM {ref} WHERE part = 'x'") == [["3"]]


def test_databricks_takes_the_kernels_z_order_cubes(volume: Any, tmp_path: Any) -> None:
    conn = ds.connect("file://")
    path = str(tmp_path / "zc")
    shutil.rmtree(path, ignore_errors=True)
    table = conn.create_table(path, pa.schema([("id", pa.int64()), ("g", pa.int32())]))
    for i in range(4):
        ids = range(i * 10, i * 10 + 10)
        table.append(
            pa.table(
                {"id": pa.array(ids, pa.int64()), "g": pa.array([j % 7 for j in ids], pa.int32())}
            )
        )
    conn.open_table(path).z_order(["g"])
    assert conn.open_table(path).z_order(["g"])["numFilesAdded"] == 0
    ref = _upload(volume, path, "zc")
    stats = json.loads(volume.sql(f"OPTIMIZE {ref} ZORDER BY (g)")[0][1])["zOrderStats"]
    assert stats["inputCubeFiles"]["num"] == "1"
    assert volume.sql(f"SELECT count(*), sum(id) FROM {ref}") == [["40", "780"]]
