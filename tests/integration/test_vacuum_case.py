"""VACUUM on a case-insensitive filesystem keeps a live file listed in another case.

macOS and Windows filesystems are case-insensitive by default: `nt/` and `nT/`
are one directory, which lists under the spelling that created it. Spark and
the kernel pick random mixed-case directory prefixes, so a live file the log
names `nt/x.parquet` could list as `nT/x.parquet`, and a VACUUM matching paths
exactly deleted it -- the table then failed to read.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("deltaswamp")

from deltaswamp.capability import Engine  # noqa: E402


def _case_insensitive(path: pathlib.Path) -> bool:
    probe = path / "CaseProbe"
    probe.write_text("x")
    try:
        return (path / "caseprobe").exists()
    finally:
        probe.unlink()


def _live_paths(path: pathlib.Path) -> list[str]:
    live: dict[str, bool] = {}
    for commit in sorted((path / "_delta_log").glob("*.json")):
        for line in commit.read_text().splitlines():
            action = json.loads(line)
            if "add" in action:
                live[action["add"]["path"]] = True
            if "remove" in action:
                live.pop(action["remove"]["path"], None)
    return list(live)


def _relist_in_other_case(path: pathlib.Path) -> str:
    """Rename a live file's prefix directory so storage lists it in upper case."""
    prefixed = [
        p
        for p in _live_paths(path)
        if "/" in p and p.split("/", 1)[0] != p.split("/", 1)[0].upper()
    ]
    assert prefixed, "the table has no file under a random prefix with a letter in it"
    prefix = prefixed[0].split("/", 1)[0]
    os.rename(path / prefix, path / f"{prefix}.tmp")
    os.rename(path / f"{prefix}.tmp", path / prefix.upper())
    return prefixed[0]


@pytest.mark.parametrize(
    ("properties", "engine"),
    [
        ({"delta.columnMapping.mode": "name"}, Engine.DELTARS),
        (
            {"delta.columnMapping.mode": "name", "delta.enableDeletionVectors": "true"},
            Engine.KERNEL,
        ),
    ],
    ids=["deltars", "kernel"],
)
def test_vacuum_keeps_a_live_file_listed_in_another_case(
    tmp_path: Any, properties: dict[str, str], engine: Engine
) -> None:
    if not _case_insensitive(tmp_path):
        pytest.skip("the filesystem is case-sensitive")
    path = tmp_path / "t"
    conn = ds.connect()
    t = conn.create_table(str(path), pa.schema([("id", pa.int64())]), properties=properties)
    for i in range(8):
        t.append(pa.table({"id": [i]}))
    moved = _relist_in_other_case(path)
    t = conn.table(str(path))
    assert t.can("vacuum").engine is engine
    t.vacuum(retention_hours=0, dry_run=False, enforce_retention_duration=False)
    assert (path / moved).exists(), f"VACUUM deleted the live file {moved}"
    assert sorted(conn.table(str(path)).to_arrow()["id"].to_pylist()) == list(range(8))


def test_new_files_get_lowercase_prefixes(tmp_path: Any) -> None:
    path = tmp_path / "t"
    conn = ds.connect()
    t = conn.create_table(
        str(path),
        pa.schema([("id", pa.int64())]),
        # In-commit timestamps: only the kernel writes the table.
        properties={
            "delta.columnMapping.mode": "name",
            "delta.enableInCommitTimestamps": "true",
        },
    )
    assert t.can("append").engine is Engine.KERNEL
    for i in range(20):
        t.append(pa.table({"id": [i]}))
    prefixes = {p.split("/", 1)[0] for p in _live_paths(path) if "/" in p}
    assert prefixes
    assert all(p == p.lower() for p in prefixes), prefixes


@pytest.mark.parametrize(
    ("properties", "engine"),
    [
        ({"delta.columnMapping.mode": "name"}, Engine.DELTARS),
        (
            {"delta.columnMapping.mode": "name", "delta.enableInCommitTimestamps": "true"},
            Engine.KERNEL,
        ),
    ],
    ids=["deltars", "kernel"],
)
def test_repair_keeps_a_live_file_listed_in_another_case(
    tmp_path: Any, properties: dict[str, str], engine: Engine
) -> None:
    if not _case_insensitive(tmp_path):
        pytest.skip("the filesystem is case-sensitive")
    path = tmp_path / "t"
    conn = ds.connect()
    t = conn.create_table(str(path), pa.schema([("id", pa.int64())]), properties=properties)
    for i in range(8):
        t.append(pa.table({"id": [i]}))
    moved = _relist_in_other_case(path)
    gone = next(p for p in _live_paths(path) if p != moved)
    (path / gone).unlink()
    t = conn.table(str(path))
    assert t.can("repair").engine is engine
    assert t.repair(dry_run=True)["files_removed"] == [gone]
    t.repair()
    assert moved in _live_paths(path)
    assert gone not in _live_paths(path)
    assert conn.table(str(path)).to_arrow().num_rows == 7
