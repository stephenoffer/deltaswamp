"""The kernel's log cleanup and FSCK, checked against Databricks reading by path.

Each test builds a table locally, cleans up its log (or repairs it) with the
kernel, uploads it to a scratch UC volume, and has the warehouse read it:
the latest version and every retained one by time travel, and DESCRIBE
HISTORY from the kept checkpoint on.

Opt-in on top of the live suite (it needs DELTASWAMP_TEST_WAREHOUSE_ID):

    DELTASWAMP_TEST_DATABRICKS=1 pytest tests/live/test_live_maintenance.py -v

The volume (named a8mt_...) is created in the target schema and dropped
afterwards.
"""

from __future__ import annotations

import pathlib
import uuid
from typing import Any
from urllib.parse import unquote

import pytest

pa = pytest.importorskip("pyarrow")

import deltaswamp as ds  # noqa: E402
from deltaswamp.capability import Engine  # noqa: E402

from tests.integration.test_audit_logclean import (  # noqa: E402
    _age,
    _shift_ict,
    _table,
    _v2_with_sidecar,
    _versions_of,
)

from .test_live_interop_stats import _Volume  # noqa: E402

pytestmark = pytest.mark.databricks


class _MaintVolume(_Volume):
    def __init__(self, workspace: Any, config: Any) -> None:
        self.w, self.config = workspace, config
        self.name = f"a8mt_{uuid.uuid4().hex[:10]}"
        self.root = f"/Volumes/{config.catalog}/{config.schema}/{self.name}"
        self.sql(f"CREATE VOLUME {config.prefix}.{self.name}")


@pytest.fixture(scope="module")
def volume(live_config: Any, live_workspace: Any) -> Any:
    if not live_config.warehouse_id:
        pytest.skip("DELTASWAMP_TEST_WAREHOUSE_ID is required to query files by path")
    vol = _MaintVolume(live_workspace, live_config)
    try:
        yield vol
    finally:
        vol.drop()


@pytest.fixture
def conn() -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog

    from tests.helpers import direct_router

    return ds.Connection(catalog=FilesystemCatalog(), router=direct_router())


def _ids(volume: Any, ref: str, version: int | None = None) -> list[int]:
    travel = "" if version is None else f" VERSION AS OF {version}"
    return sorted(int(r[0]) for r in volume.sql(f"SELECT id FROM {ref}{travel}"))


def _history(volume: Any, ref: str) -> list[int]:
    return sorted(int(r[0]) for r in volume.sql(f"SELECT version FROM (DESCRIBE HISTORY {ref})"))


def _check_retained(volume: Any, ref: str, retained: range) -> None:
    """Databricks reads every retained version, and none before them."""
    for v in retained:
        assert _ids(volume, ref, v) == list(range(1, v + 1)), v
    assert _ids(volume, ref) == list(range(1, retained[-1] + 1))
    assert _history(volume, ref) == list(retained)
    with pytest.raises(AssertionError):
        _ids(volume, ref, retained[0] - 1)


def test_in_commit_timestamp_cleanup_reads_on_databricks(
    volume: Any, conn: Any, tmp_path: Any
) -> None:
    """delta-rs refused this table; the kernel cleans it up by its commit times."""
    path = _table(
        conn,
        tmp_path,
        commits=8,
        checkpoints=(3, 6),
        # Databricks refuses time travel past the deleted-file retention,
        # and these commits are 40 days old.
        properties={
            "delta.enableInCommitTimestamps": "true",
            "delta.deletedFileRetentionDuration": "interval 60 days",
        },
    )
    _shift_ict(path, 40)
    t = conn.open_table(path)
    assert t.can("cleanup_metadata").engine is Engine.KERNEL
    t.cleanup_metadata()
    assert _versions_of(path) == [6, 7]
    _check_retained(volume, volume.upload(path, "ict"), range(6, 8))


def test_clustered_cleanup_by_modification_time_reads_on_databricks(
    volume: Any, conn: Any, tmp_path: Any
) -> None:
    path = _table(conn, tmp_path, commits=9, checkpoints=(3, 7), cluster_by=["id"])
    _age(path, range(0, 6), 40)
    conn.open_table(path).cleanup_metadata()
    assert _versions_of(path) == list(range(3, 9))
    _check_retained(volume, volume.upload(path, "clustered"), range(3, 9))


def test_v2_checkpoint_with_a_sidecar_reads_on_databricks(
    volume: Any, conn: Any, tmp_path: Any
) -> None:
    path = _table(
        conn,
        tmp_path,
        commits=8,
        checkpoints=(2, 5),
        properties={"delta.checkpointPolicy": "v2"},
    )
    old = _v2_with_sidecar(path, 2)
    kept = _v2_with_sidecar(path, 5)
    _age(path, range(0, 8), 40)
    import os
    import time

    stamp = time.time() - 40 * 86400
    for f in pathlib.Path(path, "_delta_log", "_sidecars").iterdir():
        os.utime(f, (stamp, stamp))
    conn.open_table(path).cleanup_metadata()
    sidecars = {f.name for f in pathlib.Path(path, "_delta_log", "_sidecars").iterdir()}
    assert sidecars == {kept} and old not in sidecars
    _check_retained(volume, volume.upload(path, "v2"), range(5, 8))


def test_fsck_of_an_in_commit_timestamp_table_reads_on_databricks(
    volume: Any, conn: Any, tmp_path: Any
) -> None:
    path = _table(conn, tmp_path, commits=4, properties={"delta.enableInCommitTimestamps": "true"})
    files = conn.open_table(path).files()
    paths, lows = files.column("path").to_pylist(), files.column("min.id").to_pylist()
    gone = next(p for p, low in zip(paths, lows, strict=True) if low == 2)
    pathlib.Path(path, unquote(str(gone))).unlink()
    t = conn.open_table(path)
    assert t.can("repair").engine is Engine.KERNEL
    assert t.repair()["files_removed"] == [gone]
    ref = volume.upload(path, "fsck")
    assert _ids(volume, ref) == [1, 3]
    top = volume.sql(f"SELECT operation FROM (DESCRIBE HISTORY {ref}) ORDER BY version DESC")
    assert top[0][0] == "FSCK"
