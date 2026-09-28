"""The small gaps, checked against Databricks reading the files by path.

Each test builds a table locally, uploads it to a scratch UC volume, and asks
the warehouse about it (or has Databricks write to it and reads the result
back): version checksums Databricks agrees with, rows added since a version
as Databricks' history says, clones Databricks reads as the source, and DML
from a pinned handle.

Opt-in on top of the live suite (it needs DELTASWAMP_TEST_WAREHOUSE_ID):

    DELTASWAMP_TEST_DATABRICKS=1 pytest tests/live/test_live_gaps_small.py -v

The volume (named f7gs_...) is created in the target schema and dropped
afterwards.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

import deltaswamp as ds  # noqa: E402

from .test_live_interop_stats import _Volume  # noqa: E402

pytestmark = pytest.mark.databricks


class _GapVolume(_Volume):
    def __init__(self, workspace: Any, config: Any) -> None:
        self.w, self.config = workspace, config
        self.name = f"f7gs_{uuid.uuid4().hex[:10]}"
        self.root = f"/Volumes/{config.catalog}/{config.schema}/{self.name}"
        self.sql(f"CREATE VOLUME {config.prefix}.{self.name}")


@pytest.fixture(scope="module")
def volume(live_config: Any, live_workspace: Any) -> Any:
    if not live_config.warehouse_id:
        pytest.skip("DELTASWAMP_TEST_WAREHOUSE_ID is required to query files by path")
    vol = _GapVolume(live_workspace, live_config)
    try:
        yield vol
    finally:
        vol.drop()


def _ids(volume: Any, ref: str) -> list[int]:
    return sorted(int(r[0]) for r in volume.sql(f"SELECT id FROM {ref}"))


def _crc(path: str, version: int) -> dict[str, Any]:
    with open(os.path.join(path, "_delta_log", f"{version:020}.crc")) as f:
        crc: dict[str, Any] = json.load(f)
    return crc


def test_checksums_agree_with_describe_detail(volume: Any, tmp_path: Any) -> None:
    """#8: every version has a .crc, and the last matches Databricks' own counts."""
    path = str(tmp_path / "crc")
    conn = ds.connect()
    conn.write_table(
        path, pa.table({"id": [1, 2]}), properties={"delta.enableDeletionVectors": "true"}
    )
    t = conn.table(path)
    t.append(pa.table({"id": [3]}))
    t.set_properties({"f7gs.k": "v"})
    t.delete("id = 1")
    t.append(pa.table({"id": [4, 5]}))
    latest = conn.table(path).version
    names = os.listdir(os.path.join(path, "_delta_log"))
    assert all(f"{v:020}.crc" in names for v in range(latest + 1))
    ref = volume.upload(path, "crc")
    assert _ids(volume, ref) == [2, 3, 4, 5]
    detail = volume.sql(f"DESCRIBE DETAIL {ref}")[0]
    columns = [
        c.name
        for c in volume.w.statement_execution.execute_statement(
            warehouse_id=volume.config.warehouse_id,
            statement=f"DESCRIBE DETAIL {ref}",
            wait_timeout="50s",
        ).manifest.schema.columns
    ]
    row = dict(zip(columns, detail, strict=True))
    crc = _crc(path, latest)
    assert int(row["numFiles"]) == crc["numFiles"]
    assert int(row["sizeInBytes"]) == crc["tableSizeBytes"]
    # Databricks appends on top of the chain, and the table stays readable.
    volume.sql(f"INSERT INTO {ref} VALUES (6)")
    assert _ids(volume, ref) == [2, 3, 4, 5, 6]


def test_added_since_matches_what_databricks_appended(volume: Any, tmp_path: Any) -> None:
    """#10: Databricks appends; the rows added since a version are those appends."""
    path = str(tmp_path / "inc")
    ds.connect().write_table(path, pa.table({"id": pa.array([1, 2], pa.int64())}))
    ref = volume.upload(path, "inc")
    base = int(volume.sql(f"DESCRIBE HISTORY {ref} LIMIT 1")[0][0])
    volume.sql(f"INSERT INTO {ref} VALUES (10), (11)")
    volume.sql(f"INSERT INTO {ref} VALUES (12)")
    history = volume.sql(f"SELECT version, operation FROM (DESCRIBE HISTORY {ref})")
    appends = [int(v) for v, op in history if int(v) > base]
    assert len(appends) == 2 and all(op == "WRITE" for v, op in history if int(v) > base)
    shutil.rmtree(path)
    volume.download("inc", path)
    t = ds.connect().table(path)
    assert sorted(pa.table(t.added_since(base)).column("id").to_pylist()) == [10, 11, 12]
    assert sorted(pa.table(t.added_since(base, until=min(appends))).column("id").to_pylist()) == [
        10,
        11,
    ]


def test_clones_read_on_databricks_as_the_source(volume: Any, tmp_path: Any) -> None:
    """#12: a deep clone, and a shallow one over the uploaded source's files."""
    conn = ds.connect()
    src = str(tmp_path / "src")
    conn.write_table(
        src,
        pa.table({"id": [1, 2, 3], "p": ["a", "b", "a"]}),
        partition_by=["p"],
        properties={"delta.enableDeletionVectors": "true"},
    )
    conn.table(src).append(pa.table({"id": [4], "p": ["b"]}))
    conn.table(src).delete("id = 1")
    source_ref = volume.upload(src, "src")
    want = _ids(volume, source_ref)
    assert want == [2, 3, 4]

    deep = str(tmp_path / "deep")
    conn.table(src).clone(deep, shallow=False)
    assert _ids(volume, volume.upload(deep, "deep")) == want

    shallow = str(tmp_path / "shallow")
    conn.table(src).clone(shallow)
    # Its adds name the local source; point them at the uploaded copy, the
    # one place the test differs from a clone made where the source lives.
    log = os.path.join(shallow, "_delta_log", f"{0:020}.json")
    local = "file://" + os.path.realpath(src) + "/"
    remote = f"dbfs:{volume.root}/src/"
    with open(log) as f:
        text = f.read()
    assert local in text
    with open(log, "w") as f:
        f.write(text.replace(local, remote))
    os.remove(os.path.join(shallow, "_delta_log", f"{0:020}.crc"))
    shallow_ref = volume.upload(shallow, "shallow")
    assert _ids(volume, shallow_ref) == want
    op = volume.sql(f"SELECT operation FROM (DESCRIBE HISTORY {shallow_ref}) WHERE version = 0")
    assert op[0][0] == "CLONE"


def test_pinned_dml_reads_on_databricks(volume: Any, tmp_path: Any) -> None:
    """#13: DML from a pinned handle over a later append, then read by Databricks."""
    conn = ds.connect()
    path = str(tmp_path / "pin")
    conn.write_table(
        path, pa.table({"id": [1], "v": [0]}), properties={"delta.enableDeletionVectors": "true"}
    )
    conn.table(path).append(pa.table({"id": [2], "v": [0]}))
    pinned = conn.table(path, version=conn.table(path).version)
    conn.table(path).append(pa.table({"id": [3], "v": [0]}))
    pinned.delete("id = 1")
    pinned.update(new_values={"v": 7}, predicate="id = 2")
    ref = volume.upload(path, "pin")
    rows = sorted((int(a), int(b)) for a, b in volume.sql(f"SELECT id, v FROM {ref}"))
    assert rows == [(2, 7), (3, 0)]
