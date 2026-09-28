"""FSCK REPAIR (`repair()`) of the tables only the kernel can write.

Round-8 maintenance gaps: repair was refused on every table delta-rs cannot
commit to (clustering, row tracking, in-commit timestamps, type widening,
column defaults). The kernel removes the missing files as delta-rs does.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any
from urllib.parse import unquote

import pytest

pa = pytest.importorskip("pyarrow")
deltalake = pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

from deltaswamp.capability import Engine  # noqa: E402
from deltaswamp.errors import UnreachableTableError  # noqa: E402

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

KERNEL_ONLY = [
    pytest.param({"properties": {"delta.enableInCommitTimestamps": "true"}}, id="ict"),
    pytest.param({"cluster_by": ["id"]}, id="clustered"),
    pytest.param({"properties": {"delta.enableTypeWidening": "true"}}, id="type_widening"),
    pytest.param(
        {
            "properties": {
                "delta.enableRowTracking": "true",
                "delta.enableDeletionVectors": "true",
            }
        },
        id="row_tracking_dv",
    ),
]


def _table(conn: Any, tmp_path: Any, kw: dict[str, Any]) -> str:
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": pa.array([1, 2], pa.int64())}), **kw)
    conn.open_table(path).append(pa.table({"id": pa.array([3], pa.int64())}))
    conn.open_table(path).append(pa.table({"id": pa.array([4, 5], pa.int64())}))
    return path


def _file_of(conn: Any, path: str, key: int) -> str:
    """The logged path of the file holding row `key`."""
    files = conn.open_table(path).files()
    paths, lows = files.column("path").to_pylist(), files.column("min.id").to_pylist()
    for logged, low in zip(paths, lows, strict=True):
        if low == key:
            return str(logged)
    raise AssertionError(key)


def _last_commit(path: str) -> list[dict[str, Any]]:
    log = sorted(pathlib.Path(path, "_delta_log").glob("[0-9]*.json"))[-1]
    return [json.loads(line) for line in log.read_text().splitlines() if line]


@pytest.mark.parametrize("kw", KERNEL_ONLY)
def test_missing_files_are_removed(conn: Any, tmp_path: Any, kw: dict[str, Any]) -> None:
    path = _table(conn, tmp_path, kw)
    gone = _file_of(conn, path, 3)
    pathlib.Path(path, unquote(gone)).unlink()
    t = conn.open_table(path)
    assert t.can("repair").engine is Engine.KERNEL
    version = t.version
    assert t.repair(dry_run=True) == {"dry_run": True, "files_removed": [gone]}
    assert conn.open_table(path).version == version
    result = conn.open_table(path).repair()
    assert result["files_removed"] == [gone] and result["dry_run"] is False
    assert result["version"] == version + 1
    actions = _last_commit(path)
    info = actions[0]["commitInfo"]
    assert info["operation"] == "FSCK"
    (remove,) = [a["remove"] for a in actions if "remove" in a]
    assert remove["path"] == gone and remove["dataChange"] is True
    if "delta.enableRowTracking" in kw.get("properties", {}):
        assert remove["baseRowId"] is not None
    assert sorted(conn.open_table(path).to_arrow().column("id").to_pylist()) == [1, 2, 4, 5]
    # Nothing left to repair.
    assert conn.open_table(path).repair() == {"dry_run": False, "files_removed": []}


def test_a_healthy_table_commits_nothing(conn: Any, tmp_path: Any) -> None:
    path = _table(conn, tmp_path, KERNEL_ONLY[0].values[0])
    version = conn.open_table(path).version
    assert conn.open_table(path).repair() == {"dry_run": False, "files_removed": []}
    assert conn.open_table(path).version == version


def test_an_append_only_kernel_only_table_repairs_only_dry(conn: Any, tmp_path: Any) -> None:
    kw = {"properties": {"delta.enableInCommitTimestamps": "true", "delta.appendOnly": "true"}}
    path = _table(conn, tmp_path, kw)
    t = conn.open_table(path)
    assert t.can("repair", dry_run=True).ok
    verdict = t.can("repair")
    assert not verdict.ok and "append-only" in verdict.reason
    with pytest.raises(UnreachableTableError, match="append-only"):
        t.repair()
