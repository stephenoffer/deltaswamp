"""Expired log cleanup (`cleanup_metadata`) served by the kernel.

Round-8 maintenance gaps: cleanup was refused on every table with in-commit
timestamps (delta-rs cannot open one for writing), and elsewhere delta-rs
went by file modification times and knew nothing of v2 checkpoints' sidecars.
The kernel's cleanup (crates/native/src/logclean.rs) deletes only below the
newest checkpoint committed before the retention boundary, so every retained
version still reads -- checked here through this library and delta-rs.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
import uuid
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
deltalake = pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

from deltaswamp.capability import Engine  # noqa: E402
from deltaswamp.errors import UnreachableTableError  # noqa: E402

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

DAY = 86400.0


def _rows(n: int, off: int = 0) -> Any:
    return pa.table({"id": pa.array(range(off, off + n), pa.int64())})


def _log(path: str) -> pathlib.Path:
    return pathlib.Path(path, "_delta_log")


def _names(path: str) -> set[str]:
    return {f.name for f in _log(path).iterdir() if f.is_file()}


def _versions_of(path: str, suffix: str = ".json") -> list[int]:
    return sorted(
        int(f.name[:20])
        for f in _log(path).iterdir()
        if f.name[:20].isdigit() and f.name[20:] == suffix
    )


def _age(path: str, versions: range, days: float) -> None:
    """Make every log file of `versions` look `days` old."""
    stamp = time.time() - days * DAY
    for f in _log(path).iterdir():
        if f.is_file() and f.name[:20].isdigit() and int(f.name[:20]) in versions:
            os.utime(f, (stamp, stamp))


def _table(
    conn: Any, tmp_path: Any, commits: int, checkpoints: tuple[int, ...] = (), **kw: Any
) -> str:
    """A table of `commits` versions (0 is the create), checkpointed at `checkpoints`."""
    path = str(tmp_path / "t")
    conn.create_table(path, _rows(0).schema, **kw)
    for v in range(1, commits):
        conn.open_table(path).append(_rows(1, v))
        if v in checkpoints:
            conn.open_table(path).checkpoint()
    return path


def _ids(conn: Any, path: str, version: int) -> list[int]:
    return sorted(conn.open_table(path).to_arrow(version=version).column("id").to_pylist())


def _deltars_ids(path: str, version: int) -> list[int]:
    dt = deltalake.DeltaTable(path, version=version)
    return sorted(pa.table(dt.to_pyarrow_table()).column("id").to_pylist())


def _every_version_reads(conn: Any, path: str, versions: range, *, deltars: bool = True) -> None:
    for v in versions:
        assert _ids(conn, path, v) == list(range(1, v + 1)), v
        if deltars:
            assert _deltars_ids(path, v) == list(range(1, v + 1)), v


def test_cleanup_keeps_the_newest_checkpoint_before_the_boundary(conn: Any, tmp_path: Any) -> None:
    path = _table(conn, tmp_path, commits=10, checkpoints=(3, 7))
    # Versions 0-5 are past the 30-day retention, 6-9 within it: the newest
    # checkpoint committed before the boundary is 3's (7's is too recent).
    _age(path, range(0, 6), 40)
    _age(path, range(6, 10), 1)
    t = conn.open_table(path)
    assert t.can("cleanup_metadata").engine is Engine.KERNEL
    t.cleanup_metadata()
    assert _versions_of(path) == list(range(3, 10))
    assert _versions_of(path, ".crc") == list(range(3, 10))
    assert _versions_of(path, ".checkpoint.parquet") == [3, 7]
    assert "_last_checkpoint" in _names(path)
    # Every version from the kept checkpoint on still reads, time travel
    # included; the ones before it are gone.
    _every_version_reads(conn, path, range(3, 10))
    with pytest.raises(UnreachableTableError, match="log retention"):
        conn.open_table(path).to_arrow(version=2)
    assert [h["version"] for h in conn.open_table(path).history()][-1] == 3
    # A second run finds nothing more to delete.
    before = _names(path)
    conn.open_table(path).cleanup_metadata()
    assert _names(path) == before


def test_nothing_goes_without_a_checkpoint_before_the_boundary(conn: Any, tmp_path: Any) -> None:
    path = _table(conn, tmp_path, commits=6, checkpoints=(4,))
    _age(path, range(0, 4), 40)  # the checkpoint's own version (4) is recent
    before = _names(path)
    conn.open_table(path).cleanup_metadata()
    assert _names(path) == before
    _every_version_reads(conn, path, range(0, 6))


def test_a_table_within_retention_keeps_its_whole_log(conn: Any, tmp_path: Any) -> None:
    path = _table(conn, tmp_path, commits=5, checkpoints=(2, 3))
    before = _names(path)
    conn.open_table(path).cleanup_metadata()
    assert _names(path) == before


def test_log_retention_duration_is_honored(conn: Any, tmp_path: Any) -> None:
    path = _table(conn, tmp_path, commits=6, checkpoints=(4,))
    conn.open_table(path).set_properties({"delta.logRetentionDuration": "interval 2 days"})
    conn.open_table(path).checkpoint()  # version 6
    _age(path, range(0, 5), 3)  # older than 2 days, younger than 30
    conn.open_table(path).cleanup_metadata()
    assert _versions_of(path) == [4, 5, 6]
    _every_version_reads(conn, path, range(4, 6))


def test_non_monotonic_modification_times_do_not_expire_newer_versions(
    conn: Any, tmp_path: Any
) -> None:
    path = _table(conn, tmp_path, commits=8, checkpoints=(2, 5))
    # Version 3 looks recent, 4-7 old again: times are made monotonic, as
    # time travel reads them, so 3 and everything after it are retained and
    # checkpoint 2 is the newest one before the boundary.
    _age(path, range(0, 8), 40)
    _age(path, range(3, 4), 1)
    conn.open_table(path).cleanup_metadata()
    assert _versions_of(path) == list(range(2, 8))
    _every_version_reads(conn, path, range(2, 8))


def _shift_ict(path: str, days: float) -> None:
    """Move every in-commit timestamp `days` into the past (commits and checksums)."""
    shift = int(days * DAY * 1000)
    for f in sorted(_log(path).iterdir()):
        if f.name.endswith(".json") and f.name[:20].isdigit():
            actions = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            for action in actions:
                info = action.get("commitInfo")
                if info and "inCommitTimestamp" in info:
                    info["inCommitTimestamp"] -= shift
                meta = action.get("metaData")
                config = (meta or {}).get("configuration") or {}
                if "delta.inCommitTimestampEnablementTimestamp" in config:
                    config["delta.inCommitTimestampEnablementTimestamp"] = str(
                        int(config["delta.inCommitTimestampEnablementTimestamp"]) - shift
                    )
            f.write_text("\n".join(json.dumps(a) for a in actions) + "\n")
        elif f.name.endswith(".crc"):
            crc = json.loads(f.read_text())
            if crc.get("inCommitTimestampOpt") is not None:
                crc["inCommitTimestampOpt"] -= shift
            f.write_text(json.dumps(crc))


ICT = {"properties": {"delta.enableInCommitTimestamps": "true"}}


def test_in_commit_timestamps_are_the_commit_times(conn: Any, tmp_path: Any) -> None:
    """A copied ICT table: every file is new, every commit old. delta-rs refused."""
    path = _table(conn, tmp_path, commits=8, checkpoints=(3, 6), **ICT)
    _shift_ict(path, 40)
    for f in _log(path).iterdir():
        os.utime(f)  # just copied
    t = conn.open_table(path)
    assert t.can("cleanup_metadata").engine is Engine.KERNEL
    t.cleanup_metadata()
    assert _versions_of(path) == list(range(6, 8))
    assert _versions_of(path, ".checkpoint.parquet") == [6]
    _every_version_reads(conn, path, range(6, 8), deltars=False)
    assert [h["version"] for h in conn.open_table(path).history()] == [7, 6]


def test_old_files_of_recent_ict_commits_are_kept(conn: Any, tmp_path: Any) -> None:
    path = _table(conn, tmp_path, commits=6, checkpoints=(3,), **ICT)
    _age(path, range(0, 6), 40)  # the files look old, but the commits are recent
    before = _names(path)
    conn.open_table(path).cleanup_metadata()
    assert _names(path) == before


def test_compacted_files_below_the_checkpoint_go(conn: Any, tmp_path: Any) -> None:
    path = _table(conn, tmp_path, commits=9, checkpoints=(6,))
    conn.open_table(path).compact_logs(1, 3)
    conn.open_table(path).compact_logs(4, 7)
    _age(path, range(0, 9), 40)
    conn.open_table(path).cleanup_metadata()
    names = _names(path)
    assert f"{1:020d}.{3:020d}.compacted.json" not in names
    # Straddling the kept checkpoint: kept.
    assert f"{4:020d}.{7:020d}.compacted.json" in names
    _every_version_reads(conn, path, range(6, 9))


def test_multi_part_checkpoints_below_the_kept_one_go(conn: Any, tmp_path: Any) -> None:
    path = _table(conn, tmp_path, commits=8, checkpoints=(5,))
    log = _log(path)
    # A (fake, older) two-part checkpoint at version 2, and an incomplete
    # one at 6, which never counts as the checkpoint to keep.
    for part in (1, 2):
        (log / f"{2:020d}.checkpoint.{part:010d}.{2:010d}.parquet").write_bytes(b"x")
    (log / f"{6:020d}.checkpoint.{1:010d}.{2:010d}.parquet").write_bytes(b"x")
    _age(path, range(0, 8), 40)
    conn.open_table(path).cleanup_metadata()
    names = _names(path)
    assert not any(n.startswith(f"{2:020d}.") for n in names)
    assert f"{6:020d}.checkpoint.{1:010d}.{2:010d}.parquet" in names
    assert _versions_of(path) == list(range(5, 8))


def _v2_with_sidecar(path: str, version: int) -> str:
    """Turn the classic checkpoint at `version` into a UUID-named v2 one with a sidecar."""
    log = _log(path)
    classic = log / f"{version:020d}.checkpoint.parquet"
    rows = pq.read_table(classic)
    adds = rows.filter(rows.column("add").is_valid()).select(["add"])
    (log / "_sidecars").mkdir(exist_ok=True)
    sidecar = f"{uuid.uuid4()}.parquet"
    pq.write_table(adds, log / "_sidecars" / sidecar)
    first = [json.loads(line) for line in (log / f"{0:020d}.json").read_text().splitlines() if line]
    top = [a for a in first if "protocol" in a or "metaData" in a]
    top.append({"checkpointMetadata": {"version": version}})
    stat = (log / "_sidecars" / sidecar).stat()
    top.append(
        {
            "sidecar": {
                "path": sidecar,
                "sizeInBytes": stat.st_size,
                "modificationTime": int(stat.st_mtime * 1000),
            }
        }
    )
    name = f"{version:020d}.checkpoint.{uuid.uuid4()}.json"
    (log / name).write_text("\n".join(json.dumps(a) for a in top) + "\n")
    classic.unlink()
    (log / "_last_checkpoint").unlink(missing_ok=True)
    return sidecar


V2 = {"properties": {"delta.checkpointPolicy": "v2"}}


def test_sidecars_nothing_retained_references_go(conn: Any, tmp_path: Any) -> None:
    path = _table(conn, tmp_path, commits=9, checkpoints=(3, 6), **V2)
    old = _v2_with_sidecar(path, 3)
    kept = _v2_with_sidecar(path, 6)
    orphan = _log(path) / "_sidecars" / f"{uuid.uuid4()}.parquet"
    orphan.write_bytes(b"x")
    fresh = _log(path) / "_sidecars" / f"{uuid.uuid4()}.parquet"
    fresh.write_bytes(b"x")  # e.g. a checkpoint being written right now
    _every_version_reads(conn, path, range(3, 9), deltars=False)
    _age(path, range(0, 9), 40)
    stamp = time.time() - 40 * DAY
    for f in (_log(path) / "_sidecars").iterdir():
        if f != fresh:
            os.utime(f, (stamp, stamp))
    conn.open_table(path).cleanup_metadata()
    sidecars = {f.name for f in (_log(path) / "_sidecars").iterdir()}
    assert sidecars == {kept, fresh.name}
    assert old not in sidecars
    assert _versions_of(path) == list(range(6, 9))
    _every_version_reads(conn, path, range(6, 9), deltars=False)


def test_a_v2_checkpoint_missing_its_sidecar_is_not_kept(conn: Any, tmp_path: Any) -> None:
    path = _table(conn, tmp_path, commits=8, checkpoints=(2, 5), **V2)
    sidecar = _v2_with_sidecar(path, 5)
    (_log(path) / "_sidecars" / sidecar).unlink()
    _age(path, range(0, 8), 40)
    conn.open_table(path).cleanup_metadata()
    # Checkpoint 5 is unusable, so 2 is the one kept and 2-7 remain.
    assert _versions_of(path) == list(range(2, 8))


def test_staged_commits_and_strangers_are_never_touched(conn: Any, tmp_path: Any) -> None:
    path = _table(conn, tmp_path, commits=6, checkpoints=(4,))
    log = _log(path)
    (log / "_staged_commits").mkdir()
    staged = log / "_staged_commits" / f"{1:020d}.{uuid.uuid4()}.json"
    staged.write_text("{}\n")
    stranger = log / "notes.txt"
    stranger.write_text("mine")
    temp = log / f".{2:020d}.json.tmp"
    temp.write_text("{}")
    _age(path, range(0, 6), 40)
    stamp = time.time() - 40 * DAY
    for f in (staged, stranger, temp):
        os.utime(f, (stamp, stamp))
    conn.open_table(path).cleanup_metadata()
    assert staged.exists() and stranger.exists() and temp.exists()
    assert _versions_of(path) == [4, 5]


def test_vacuum_and_checkpoint_after_cleanup(conn: Any, tmp_path: Any) -> None:
    """VACUUM reads the retained segment and deletes nothing a retained version needs."""
    path = _table(conn, tmp_path, commits=8, checkpoints=(4,))
    conn.open_table(path).delete("id = 1")  # version 8 removes a file
    _age(path, range(0, 6), 40)
    conn.open_table(path).cleanup_metadata()
    assert _versions_of(path)[0] == 4
    t = conn.open_table(path)
    assert t.vacuum(dry_run=False) == []  # the remove is within file retention
    t.checkpoint()
    for v in range(4, 8):
        assert _ids(conn, path, v) == list(range(1, v + 1))
        assert _deltars_ids(path, v) == list(range(1, v + 1))
    assert _ids(conn, path, 8) == list(range(2, 8))


def test_catalog_managed_and_protected_tables_are_refused(conn: Any, tmp_path: Any) -> None:
    from deltaswamp.engine.kernel import KernelEngine

    path = _table(conn, tmp_path, commits=3)
    table = conn.open_table(path)._enrich()
    import dataclasses

    protected = dataclasses.replace(
        table, writer_features=table.writer_features | {"checkpointProtection"}
    )
    verdict = KernelEngine().supports(ds.capability.Operation.CLEANUP_METADATA, protected)
    assert not verdict.ok and "checkpointProtection" in verdict.reason
    with pytest.raises(UnreachableTableError, match="checkpointProtection"):
        KernelEngine().cleanup_metadata(protected)
