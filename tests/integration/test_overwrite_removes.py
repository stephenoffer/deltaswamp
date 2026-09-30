"""An overwrite's removes, produced as its commit is written rather than held.

They must be the actions kernel would have staged -- every live file, with
its partition values, size and deletion vector -- and the version checksum
written after the commit must count them.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltaswamp._native")

DV = {"delta.enableDeletionVectors": "true"}


def _actions(path: str, version: int) -> list[dict[str, Any]]:
    with open(os.path.join(path, "_delta_log", f"{version:020d}.json")) as f:
        return [json.loads(line) for line in f if line.strip()]


def _table(conn: Any, path: str) -> str:
    conn.create_table(
        path,
        pa.schema([("id", pa.int64()), ("region", pa.string())]),
        partition_by=["region"],
        properties=DV,
    )
    for region in ("eu", "us"):
        conn.open_table(path).append(
            pa.table({"id": pa.array([1, 2, 3], pa.int64()), "region": [region] * 3})
        )
    conn.open_table(path).delete("id = 2")  # deletion vectors on live files
    return path


@pytest.mark.parametrize("distributed", [False, True], ids=["local", "distributed"])
def test_every_live_file_is_removed_as_kernel_would(
    conn: Any, tmp_path: Any, distributed: bool
) -> None:
    path = _table(conn, str(tmp_path / "t"))
    table = conn.open_table(path)
    live = set(table.files().column("path").to_pylist())
    data = pa.table({"id": pa.array([9], pa.int64()), "region": ["eu"]})
    if distributed:
        plan = table.plan_write(mode="overwrite")
        version = plan.commit([plan.write(data)])
    else:
        from deltaswamp.capability import Engine

        kernel = conn.router.engines[Engine.KERNEL]
        kernel.append(table._enrich(), data, overwrite=True)
        version = conn.open_table(path).version
    removes = [a["remove"] for a in _actions(path, version) if "remove" in a]
    assert {r["path"] for r in removes} == live
    for remove in removes:
        assert remove["dataChange"] is True
        assert remove["extendedFileMetadata"] is True
        assert remove["partitionValues"]["region"] in ("eu", "us")
        assert remove["size"] > 0
    assert sum(1 for r in removes if r.get("deletionVector")) == 2
    assert conn.open_table(path).to_arrow().column("id").to_pylist() == [9]


def test_the_checksum_counts_the_removes(conn: Any, tmp_path: Any) -> None:
    path = _table(conn, str(tmp_path / "t"))
    plan = conn.open_table(path).plan_write(mode="overwrite")
    version = plan.commit(
        [plan.write(pa.table({"id": pa.array([9], pa.int64()), "region": ["eu"]}))]
    )
    crc = os.path.join(path, "_delta_log", f"{version:020d}.crc")
    if not os.path.exists(crc):
        pytest.skip("no checksum written for this version")
    with open(crc) as f:
        checksum = json.load(f)
    assert checksum["numFiles"] == conn.open_table(path).files().num_rows == 1
