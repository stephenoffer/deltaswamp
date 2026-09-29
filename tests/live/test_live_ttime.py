"""Timestamp -> version on logs whose commit files are out of time order, against Databricks.

Each test builds a table locally and uploads its commit files to a scratch UC
volume in a scrambled order, a few seconds apart, so the files' modification
times on the volume are out of version order. It then reads those times back
from the volume, sets them on the local copy, and checks that time travel,
the change feed's bounds and commit times, and DESCRIBE HISTORY answer on
every direct engine what the warehouse answers by path.

Opt-in on top of the live suite (it needs DELTASWAMP_TEST_WAREHOUSE_ID):

    DELTASWAMP_TEST_DATABRICKS=1 pytest tests/live/test_live_ttime.py -v

The volume (named a8tt_...) is created in the target schema and dropped
afterwards.
"""

from __future__ import annotations

import datetime as dt
import io
import os
import pathlib
import re
import time
import uuid
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

import deltaswamp as ds  # noqa: E402

from tests.integration.test_audit_ttime import (  # noqa: E402
    _connection,
    at,
    build,
    in_commit_timestamps,
)

from .test_live_interop_stats import _Volume  # noqa: E402

pytestmark = [
    pytest.mark.databricks,
    pytest.mark.skipif(not ds.has_native(), reason="native extension not built"),
]


class _TravelVolume(_Volume):
    def __init__(self, workspace: Any, config: Any) -> None:
        self.w, self.config = workspace, config
        self.name = f"a8tt_{uuid.uuid4().hex[:10]}"
        self.root = f"/Volumes/{config.catalog}/{config.schema}/{self.name}"
        self.sql(f"CREATE VOLUME {config.prefix}.{self.name}")

    def answer(self, statement: str) -> Any:
        """Rows, or the error class the warehouse refused with."""
        try:
            return [[int(c) for c in row] for row in self.sql(statement)]
        except AssertionError as exc:
            code = re.search(r"\[([A-Z_]+)\]", str(exc))
            return "refused" if code is None else code.group(1)

    def upload_scrambled(self, local: str, rel: str, order: list[int]) -> dict[int, int]:
        """Upload `local`, its commits in `order`; the volume's commit times (ms)."""
        root = pathlib.Path(local)
        files = sorted(f for f in root.rglob("*") if f.is_file())
        commits = {
            int(f.name[:20]): f
            for f in files
            if f.parent.name == "_delta_log" and re.fullmatch(r"\d{20}\.json", f.name)
        }
        for f in (f for f in files if f not in commits.values()):
            self._put(f, root, rel)
        for version in order:
            self._put(commits[version], root, rel)
            time.sleep(2.2)  # the volume reports whole seconds
        return {
            int(e.name[:20]): int(e.last_modified)
            for e in self.w.files.list_directory_contents(f"{self.root}/{rel}/_delta_log")
            if e.name and re.fullmatch(r"\d{20}\.json", e.name)
        }

    def _put(self, f: pathlib.Path, root: pathlib.Path, rel: str) -> None:
        self.w.files.upload(
            f"{self.root}/{rel}/{f.relative_to(root)}", io.BytesIO(f.read_bytes()), overwrite=True
        )


@pytest.fixture(scope="module")
def volume(live_config: Any, live_workspace: Any) -> Any:
    if not live_config.warehouse_id:
        pytest.skip("DELTASWAMP_TEST_WAREHOUSE_ID is required to query files by path")
    vol = _TravelVolume(live_workspace, live_config)
    try:
        yield vol
    finally:
        vol.drop()


def _local_answers(t: Any, stamp: int) -> dict[str, Any]:
    def attempt(call: Any) -> Any:
        try:
            return call()
        except ds.DeltaSwampError as exc:
            return "refused" if "after the latest" not in str(exc) else "after"

    when = at(stamp)
    feed = lambda **kw: sorted(  # noqa: E731
        set(t.cdf(**kw).read_all().column("_commit_version").to_pylist())
    )
    return {
        "read": attempt(lambda: t.to_arrow(timestamp=when).num_rows),
        "start": attempt(lambda: feed(starting_timestamp=when)),
        "end": attempt(lambda: feed(starting_version=0, ending_timestamp=when)),
    }


def _remote_answers(volume: Any, ref: str, stamp: int) -> dict[str, Any]:
    ts = f"timestamp_millis({stamp})"

    def norm(answer: Any, rows: bool = False) -> Any:
        if answer == "DELTA_TIMESTAMP_GREATER_THAN_COMMIT":
            return "after"
        if isinstance(answer, str):
            return "refused"
        return answer[0][0] if rows else sorted(r[0] for r in answer)

    feed = f"SELECT DISTINCT _commit_version FROM table_changes('{ref}', "
    return {
        "read": norm(volume.answer(f"SELECT count(*) FROM {ref} TIMESTAMP AS OF {ts}"), True),
        "start": norm(volume.answer(f"{feed}{ts})")),
        "end": norm(volume.answer(f"{feed}0, {ts})")),
    }


def _check(volume: Any, local: str, rel: str, order: list[int]) -> None:
    mtimes = volume.upload_scrambled(local, rel, order)
    for version, millis in mtimes.items():
        f = pathlib.Path(local, "_delta_log", f"{version:020}.json")
        os.utime(f, ns=(millis * 1_000_000, millis * 1_000_000))
    ref = f"delta.`{volume.root}/{rel}`"
    history = {
        v: s
        for v, s in volume.answer(
            f"SELECT version, CAST(unix_millis(timestamp) AS BIGINT) FROM (DESCRIBE HISTORY {ref})"
        )
    }
    stamps = sorted(set(history.values()) | set(in_commit_timestamps(local).values()))
    probes = sorted({s + d for s in stamps for d in (-1, 0, 1)})
    remote = {p: _remote_answers(volume, ref, p) for p in probes}
    remote_times = dict(
        volume.answer(
            f"SELECT DISTINCT _commit_version, unix_millis(_commit_timestamp) "
            f"FROM table_changes('{ref}', 0)"
        )
    )
    for kind in ("kernel", "deltars"):
        t = _connection(kind).open_table(local)
        feed = t.cdf(starting_version=0).read_all()
        times = {
            v: s.value // 1000
            for v, s in zip(
                feed.column("_commit_version").to_pylist(),
                feed.column("_commit_timestamp"),
                strict=True,
            )
        }
        assert times == remote_times, kind
        if kind == "deltars":
            assert {h["version"]: h["timestamp"] for h in t.history()} == history
        for p in probes:
            assert _local_answers(t, p) == remote[p], (kind, p, dt.datetime.fromtimestamp(p / 1000))


def test_file_times_out_of_order_resolve_as_on_databricks(volume: Any, tmp_path: Any) -> None:
    # v1's file lands last: v2 and v3 look older than it.
    path = build(_connection("both"), tmp_path / "t", 4)
    _check(volume, path, "scrambled", [0, 2, 3, 1])


def test_in_commit_timestamps_enabled_later_resolve_as_on_databricks(
    volume: Any, tmp_path: Any
) -> None:
    # Enabled at v2; the files before it land after every in-commit timestamp.
    path = build(_connection("both"), tmp_path / "t", 5, ict_at=2)
    _check(volume, path, "ict_later", [0, 1, 3, 4, 2])
