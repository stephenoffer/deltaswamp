"""Concurrent writers on one path table (audit r5he: HB-1, HB-2, HB-4, HB-5).

The races run in spawned processes, as separate writers do; each test keeps
to a few rounds so the file stays well under a minute.
"""

from __future__ import annotations

import multiprocessing as mp
import warnings
from typing import Any

import deltalake
import deltaswamp as ds
import pyarrow as pa
import pytest
from deltaswamp.capability import Engine
from deltaswamp.engine.boundary import as_commit_conflict
from deltaswamp.engine.kernel import KernelEngine
from deltaswamp.errors import CommitConflictError, MetadataChangedError

pytestmark = pytest.mark.skipif(not KernelEngine.available(), reason="needs the native extension")

DV = {"delta.enableDeletionVectors": "true"}


def _optimize(path: str, barrier: Any, q: Any, zorder: bool) -> None:
    warnings.filterwarnings("ignore")
    t = ds.connect().open_table(path)
    barrier.wait()
    try:
        r = t.z_order(["id"]) if zorder else t.optimize()
        q.put(("ok", r["num_files_removed"]))
    except CommitConflictError as exc:
        q.put(("conflict", str(exc)))
    except Exception as exc:  # pragma: no cover - reported by the test
        q.put(("error", f"{type(exc).__name__}: {exc}"))


def _append(path: str, barrier: Any, q: Any, first: int) -> None:
    warnings.filterwarnings("ignore")
    t = ds.connect().open_table(path)
    barrier.wait()
    for i in range(first, first + 5):
        t.append(pa.table({"id": pa.array([i], pa.int64()), "p": [f"p{i % 2}"]}))
    q.put(("ok", 0))


def _merge(path: str, barrier: Any, q: Any) -> None:
    warnings.filterwarnings("ignore")
    t = ds.connect().open_table(path)
    t.schema()
    barrier.wait()
    src = pa.table({"id": pa.array([999], pa.int64()), "v": pa.array([0], pa.int64())})
    try:
        (
            t.merge(src, "t.id = s.id", source_alias="s", target_alias="t")
            .when_matched_update({"v": "t.v + 1"})
            .when_not_matched_insert({"id": "s.id", "v": "s.v"})
            .execute()
        )
        q.put(("ok", 0))
    except CommitConflictError as exc:
        q.put(("conflict", str(exc)))
    except Exception as exc:  # pragma: no cover - reported by the test
        q.put(("error", f"{type(exc).__name__}: {exc}"))


def _race(target: Any, args: list[tuple[Any, ...]]) -> list[tuple[str, Any]]:
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(len(args)), ctx.Queue()
    procs = [ctx.Process(target=target, args=(a[0], barrier, q, *a[1:])) for a in args]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    return [q.get(timeout=5) for _ in procs]


def _small_files(tmp_path: Any, name: str, n: int, properties: dict[str, str]) -> str:
    path = str(tmp_path / name)
    ds.connect().create_table(
        path,
        schema=pa.schema([("id", pa.int64()), ("p", pa.string())]),
        partition_by=["p"],
        properties=properties,
    )
    for i in range(n):
        deltalake.write_deltalake(
            path, pa.table({"id": pa.array([i], pa.int64()), "p": [f"p{i % 2}"]}), mode="append"
        )
    return path


def _ids(path: str) -> list[int]:
    return sorted(ds.connect().open_table(path).to_arrow().column("id").to_pylist())


@pytest.mark.parametrize("properties", [{}, DV], ids=["plain", "dv"])
@pytest.mark.parametrize("zorder", [False, True], ids=["compact", "zorder"])
def test_concurrent_optimize_never_duplicates_rows(
    tmp_path: Any, properties: dict[str, str], zorder: bool
) -> None:
    """HB-1: delta-rs committed every racing OPTIMIZE (150 rows became 450)."""
    for round_ in range(2):
        path = _small_files(tmp_path, f"t{round_}", 30, properties)
        outcomes = _race(_optimize, [(path, zorder)] * 3)
        assert all(kind in ("ok", "conflict") for kind, _ in outcomes), outcomes
        assert _ids(path) == list(range(30)), outcomes
        # Every file was compacted exactly once, whoever did it.
        if not zorder:
            assert sum(n for kind, n in outcomes if kind == "ok") == 30, outcomes


def test_optimize_racing_appends_loses_nothing(tmp_path: Any) -> None:
    path = _small_files(tmp_path, "t", 20, DV)
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(3), ctx.Queue()
    procs = [
        ctx.Process(target=_optimize, args=(path, barrier, q, False)),
        ctx.Process(target=_append, args=(path, barrier, q, 100)),
        ctx.Process(target=_append, args=(path, barrier, q, 200)),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    results = [q.get(timeout=5) for _ in procs]
    assert all(kind in ("ok", "conflict") for kind, _ in results), results
    assert _ids(path) == [*range(20), *range(100, 105), *range(200, 205)]


def test_optimize_commits_as_a_compaction(tmp_path: Any) -> None:
    path = _small_files(tmp_path, "t", 6, {"delta.enableChangeDataFeed": "true"})
    t = ds.connect().open_table(path)
    result = t.optimize(commit_metadata={"job": "nightly"})
    assert result["num_files_removed"] == 6 and result["num_files_added"] == 2
    entry = deltalake.DeltaTable(path).history(1)[0]
    assert entry["operation"] == "OPTIMIZE" and entry.get("job") == "nightly"
    log = deltalake.DeltaTable(path)
    assert pa.table(log.get_add_actions(flatten=True)).num_rows == 2
    assert _ids(path) == list(range(6))
    # A second run finds nothing left to do.
    assert t.optimize()["num_files_removed"] == 0


def test_stale_optimize_replans_instead_of_compacting_twice(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """A compaction that lost to another over the same files re-plans from the new snapshot."""
    path = _small_files(tmp_path, "t", 6, {})
    real = KernelEngine.snapshot
    state = {"raced": False}

    def racing(self: Any, table: Any, **kw: Any) -> Any:
        snap = real(self, table, **kw)
        if kw.get("write") and not state["raced"]:
            state["raced"] = True
            deltalake.DeltaTable(path).optimize.compact()  # the other writer wins
        return snap

    monkeypatch.setattr(KernelEngine, "snapshot", racing)
    ds.connect().open_table(path).optimize()
    monkeypatch.undo()
    assert state["raced"]
    assert _ids(path) == list(range(6))


def test_concurrent_kernel_merges_insert_a_new_key_once(tmp_path: Any) -> None:
    """HB-2: every MERGE rebased over the others' inserts, each adding id 999."""
    for round_ in range(3):
        path = str(tmp_path / f"t{round_}")
        t = ds.connect().create_table(
            path, schema=pa.schema([("id", pa.int64()), ("v", pa.int64())]), properties=DV
        )
        for i in range(5):
            t.append(pa.table({"id": pa.array([i], pa.int64()), "v": pa.array([0], pa.int64())}))
        outcomes = _race(_merge, [(path,)] * 4)
        assert all(kind in ("ok", "conflict") for kind, _ in outcomes), outcomes
        a = ds.connect().open_table(path).to_arrow()
        rows = [
            (i, v)
            for i, v in zip(a.column("id").to_pylist(), a.column("v").to_pylist(), strict=True)
            if i == 999
        ]
        oks = sum(kind == "ok" for kind, _ in outcomes)
        assert rows == [(999, oks - 1)], outcomes


def _kernel_only() -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.router import Router

    return Connection(
        catalog=FilesystemCatalog(), router=Router(engines={Engine.KERNEL: KernelEngine()})
    )


def _with_a_race(monkeypatch: Any, act: Any) -> dict[str, bool]:
    """Run `act` right after the next write snapshot is read: a writer that won."""
    real = KernelEngine.snapshot
    state = {"raced": False}

    def racing(self: Any, table: Any, **kw: Any) -> Any:
        snap = real(self, table, **kw)
        if kw.get("write") and not state["raced"]:
            state["raced"] = True
            act()
        return snap

    monkeypatch.setattr(KernelEngine, "snapshot", racing)
    return state


def _two_columns(tmp_path: Any, properties: dict[str, str]) -> str:
    path = str(tmp_path / "t")
    t = ds.connect().create_table(
        path, schema=pa.schema([("id", pa.int64()), ("v", pa.int64())]), properties=properties
    )
    for i in range(4):
        t.append(pa.table({"id": pa.array([i], pa.int64()), "v": pa.array([0], pa.int64())}))
    return path


def _append_row(path: str, key: int) -> Any:
    return lambda: (
        _kernel_only()
        .open_table(path)
        .append(pa.table({"id": pa.array([key], pa.int64()), "v": pa.array([0], pa.int64())}))
    )


@pytest.mark.parametrize(
    ("properties", "expected"),
    [
        # The copy-on-write DELETE rewrites only the files it read and is
        # committed as read, as the vector DELETE is: WriteSerializable
        # orders it before the append, whose rows it never read.
        ({}, [0, 1, 7]),
        # The vector DELETE is committed as read: WriteSerializable orders it
        # before the append, whose rows it never read (Spark's rule).
        (DV, [0, 1, 7]),
    ],
    ids=["rewrite", "dv"],
)
def test_kernel_delete_survives_a_blind_append(
    tmp_path: Any, monkeypatch: Any, properties: dict[str, str], expected: list[int]
) -> None:
    """HB-4: without deletion vectors the kernel DELETE failed on any concurrent append."""
    path = _two_columns(tmp_path, properties)
    state = _with_a_race(monkeypatch, _append_row(path, 7))
    _kernel_only().open_table(path).delete("id >= 2")
    monkeypatch.undo()
    assert state["raced"]
    assert _ids(path) == expected


def test_kernel_merge_conflicts_with_a_concurrent_merge_insert(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """HB-2, one process: the winner's MERGE added the key this MERGE's read covers."""
    path = _two_columns(tmp_path, DV)
    src = pa.table({"id": pa.array([999], pa.int64()), "v": pa.array([0], pa.int64())})

    def upsert(conn: Any) -> Any:
        return (
            conn.open_table(path)
            .merge(src, "t.id = s.id", source_alias="s", target_alias="t")
            .when_matched_update({"v": "t.v + 1"})
            .when_not_matched_insert({"id": "s.id", "v": "s.v"})
            .execute()
        )

    state = _with_a_race(monkeypatch, lambda: upsert(ds.connect()))
    with pytest.raises(CommitConflictError, match="not a blind append"):
        upsert(ds.connect())
    monkeypatch.undo()
    assert state["raced"]
    assert [i for i in _ids(path) if i == 999] == [999]


def test_kernel_merge_rebases_over_a_blind_append_of_other_keys(
    tmp_path: Any, monkeypatch: Any
) -> None:
    path = _two_columns(tmp_path, DV)
    src = pa.table({"id": pa.array([1], pa.int64()), "v": pa.array([5], pa.int64())})
    state = _with_a_race(monkeypatch, _append_row(path, 50))
    (
        ds.connect()
        .open_table(path)
        .merge(src, "t.id = s.id", source_alias="s", target_alias="t")
        .when_matched_update({"v": "s.v"})
        .execute()
    )
    monkeypatch.undo()
    assert state["raced"]
    assert _ids(path) == [0, 1, 2, 3, 50]


def test_delete_conflicts_with_a_concurrent_update_of_matching_rows(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """A concurrent UPDATE is not a blind append: its new rows are what the DELETE reads.

    The UPDATE leaves the file the DELETE touches alone, but writes a row
    (id 10) the DELETE's predicate matches.
    """
    path = _two_columns(tmp_path, DV)
    state = _with_a_race(
        monkeypatch, lambda: ds.connect().open_table(path).update({"id": "10"}, predicate="id = 0")
    )
    with pytest.raises(CommitConflictError, match="not a blind append"):
        ds.connect().open_table(path).delete("id >= 3")
    monkeypatch.undo()
    assert state["raced"]


def test_deltars_metadata_race_is_metadata_changed_error() -> None:
    """HB-5: the kernel raised MetadataChangedError for this race, delta-rs the base class."""

    class CommitFailedError(Exception):
        pass

    error = as_commit_conflict(
        CommitFailedError("Failed to commit transaction: Metadata changed since last commit.")
    )
    assert isinstance(error, MetadataChangedError)
    plain = as_commit_conflict(CommitFailedError("Failed to commit transaction: 0"))
    assert type(plain) is CommitConflictError
