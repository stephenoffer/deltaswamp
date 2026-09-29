"""Kernel checkpoints of constrained kernel-only tables, read by Databricks by path.

A table with in-commit timestamps and a CHECK constraint had no engine that
could checkpoint it. Each test builds one locally, checkpoints it with the
kernel (classic or v2), uploads it -- whole, and as a copy whose commits
below the checkpoint are deleted, so the warehouse can only start from the
checkpoint -- and has the warehouse read it: the rows, time travel, DESCRIBE
HISTORY, the table's properties, and an INSERT violating the constraint,
which must still fail.

Opt-in on top of the live suite (it needs DELTASWAMP_TEST_WAREHOUSE_ID):

    DELTASWAMP_TEST_DATABRICKS=1 pytest tests/live/test_live_checkpoint.py -v

The volume (named a8ck_...) is created in the target schema and dropped
afterwards.
"""

from __future__ import annotations

import json
import shutil
import uuid
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

import deltaswamp as ds  # noqa: E402
from deltaswamp.capability import Engine  # noqa: E402

from tests.integration.test_audit_ckpt import (  # noqa: E402
    _add_generated_column,
    _assert_checkpoint_is_the_log,
    _assert_checksums,
    _constrained,
    _log,
    _versions_of,
)

from .test_live_interop_stats import _Volume  # noqa: E402

pytestmark = pytest.mark.databricks


class _CheckpointVolume(_Volume):
    def __init__(self, workspace: Any, config: Any) -> None:
        self.w, self.config = workspace, config
        self.name = f"a8ck_{uuid.uuid4().hex[:10]}"
        self.root = f"/Volumes/{config.catalog}/{config.schema}/{self.name}"
        self.sql(f"CREATE VOLUME {config.prefix}.{self.name}")


@pytest.fixture(scope="module")
def volume(live_config: Any, live_workspace: Any) -> Any:
    if not live_config.warehouse_id:
        pytest.skip("DELTASWAMP_TEST_WAREHOUSE_ID is required to query files by path")
    vol = _CheckpointVolume(live_workspace, live_config)
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


def _properties(volume: Any, ref: str) -> dict[str, str]:
    return {str(r[0]): str(r[1]) for r in volume.sql(f"SHOW TBLPROPERTIES {ref}")}


def _from_checkpoint_only(path: str, dest: str, version: int) -> str:
    """A copy of `path` without the commits (and checksums) below `version`."""
    shutil.copytree(path, dest)
    for f in _log(dest).iterdir():
        if f.name[:20].isdigit() and int(f.name[:20]) < version:
            f.unlink()
    return dest


def _constraint_still_binds(volume: Any, ref: str, before: list[int]) -> None:
    with pytest.raises(AssertionError, match=r"(?i)check|constraint|pos"):
        volume.sql(f"INSERT INTO {ref} VALUES (-1)")
    assert _ids(volume, ref) == before
    volume.sql(f"INSERT INTO {ref} VALUES (100)")
    assert _ids(volume, ref) == [*before, 100]


@pytest.mark.parametrize("policy", ["classic", "v2"])
def test_a_constrained_kernel_checkpoint_reads_on_databricks(
    volume: Any, conn: Any, tmp_path: Any, policy: str
) -> None:
    path = _constrained(conn, tmp_path, **{"delta.checkpointPolicy": policy})
    t = conn.open_table(path)
    assert t.can("checkpoint").engine is Engine.KERNEL
    t.checkpoint()
    _assert_checkpoint_is_the_log(path, 4)
    _assert_checksums(path, range(0, 5))

    whole = volume.upload(path, f"whole_{policy}")
    for v, ids in [(1, []), (2, [0, 1]), (3, list(range(4))), (4, list(range(6)))]:
        assert _ids(volume, whole, v) == ids, v
    assert _history(volume, whole) == list(range(5))

    only = _from_checkpoint_only(path, str(tmp_path / "only"), 4)
    ref = volume.upload(only, f"only_{policy}")
    assert _ids(volume, ref) == list(range(6))
    assert _ids(volume, ref, 4) == list(range(6))
    assert _history(volume, ref) == [4]
    props = _properties(volume, ref)
    assert props.get("delta.constraints.pos") == "id >= 0"
    assert props.get("delta.enableInCommitTimestamps") == "true"
    assert props.get("delta.feature.checkConstraints") == "supported"
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
    crc = json.loads((_log(only) / f"{4:020d}.crc").read_text())
    assert int(row["numFiles"]) == crc["numFiles"] == 3
    assert int(row["sizeInBytes"]) == crc["tableSizeBytes"]
    _constraint_still_binds(volume, ref, list(range(6)))


def test_automatic_checkpoints_read_on_databricks(volume: Any, conn: Any, tmp_path: Any) -> None:
    """Checkpoints at delta.checkpointInterval, which kernel writes on such a
    table used to skip silently."""
    path = _constrained(conn, tmp_path, appends=6, **{"delta.checkpointInterval": "3"})
    assert _versions_of(path, ".checkpoint.parquet") == [3, 6]
    only = _from_checkpoint_only(path, str(tmp_path / "only"), 6)
    ref = volume.upload(only, "interval")
    assert _ids(volume, ref) == list(range(12))
    assert _history(volume, ref) == [6, 7]
    _constraint_still_binds(volume, ref, list(range(12)))


def test_a_generated_column_checkpoint_reads_on_databricks(
    volume: Any, conn: Any, tmp_path: Any
) -> None:
    path = _constrained(conn, tmp_path, appends=2)
    version = _add_generated_column(path)
    conn.open_table(path).checkpoint()
    _assert_checkpoint_is_the_log(path, version)
    only = _from_checkpoint_only(path, str(tmp_path / "only"), version)
    ref = volume.upload(only, "generated")
    assert _ids(volume, ref) == list(range(4))
    # Databricks computes the generated column from the checkpoint's schema.
    volume.sql(f"INSERT INTO {ref} (id) VALUES (7)")
    got = volume.sql(f"SELECT g FROM {ref} WHERE id = 7")
    assert [int(r[0]) for r in got] == [14]
    with pytest.raises(AssertionError, match=r"(?i)check|constraint|pos"):
        volume.sql(f"INSERT INTO {ref} (id) VALUES (-3)")
