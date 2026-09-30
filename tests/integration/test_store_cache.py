"""Object stores are built once per process, not once per call.

Building one costs 100-250 ms on a cloud (object_store's HTTP clients load
the system's root certificates). Without the cache an append built eight, a
DELETE seven, and every `WritePlan.write()` one.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import textwrap
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
native = pytest.importorskip("deltaswamp._native")

pytestmark = pytest.mark.skipif(
    "store_cache" not in getattr(native, "FEATURES", ()), reason="stale native build"
)


def _ids(*values: int) -> Any:
    return pa.table({"id": pa.array(list(values), pa.int64())})


@pytest.fixture
def path(conn: Any, tmp_path: Any) -> str:
    path = str(tmp_path / "t")
    conn.write_table(path, _ids(0))
    return path


def test_repeated_operations_on_the_driver_build_no_more_stores(conn: Any, path: str) -> None:
    conn.open_table(path).append(_ids(1))
    before = native.store_builds()
    for i in range(5):
        t = conn.open_table(path)
        t.append(_ids(i + 2))
        t.delete("id = 99")
        t.to_arrow()
    assert native.store_builds() == before
    assert sorted(conn.open_table(path).to_arrow().column("id").to_pylist()) == list(range(7))


_WORKER = textwrap.dedent(
    """
    import pickle, sys
    import pyarrow as pa
    from deltaswamp import _native
    plan = pickle.load(open(sys.argv[1], "rb"))
    before = _native.store_builds()
    fragments = [plan.write(pa.table({"id": pa.array([i], pa.int64())})) for i in range(6)]
    print(_native.store_builds() - before)
    with open(sys.argv[2], "wb") as f:
        pickle.dump(fragments, f)
    """
)


def _worker_builds(conn: Any, path: str, tmp_path: Any, env: dict[str, str]) -> int:
    plan = conn.open_table(path).plan_write()
    job, out = tmp_path / "plan.pkl", tmp_path / "fragments.pkl"
    job.write_bytes(pickle.dumps(plan))
    ran = subprocess.run(
        [sys.executable, "-c", _WORKER, str(job), str(out)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    plan.commit(pickle.loads(out.read_bytes()))
    return int(ran.stdout.strip())


def test_a_worker_builds_one_store_for_all_its_writes(conn: Any, path: str, tmp_path: Any) -> None:
    assert _worker_builds(conn, path, tmp_path, os.environ.copy()) == 1
    assert conn.open_table(path).count() == 7


def test_the_cache_can_be_turned_off(conn: Any, path: str, tmp_path: Any) -> None:
    env = {**os.environ, "DELTASWAMP_STORE_CACHE": "0"}
    assert _worker_builds(conn, path, tmp_path, env) == 6
