"""Concurrency, multi-writer and lifecycle defects, pinned deterministically.

Each race is reproduced by forcing the losing interleaving (a stale snapshot,
a check that ran before the winner committed) rather than by hoping threads
collide, so every test is seeded by construction and runs in well under a
second.
"""

from __future__ import annotations

import collections
import threading
from typing import Any

import deltaswamp as ds
import pytest
from deltaswamp.capability import Engine, Operation
from deltaswamp.errors import CommitConflictError, UnreachableTableError

pa = pytest.importorskip("pyarrow")
deltalake = pytest.importorskip("deltalake")

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")


def _conn(*kinds: Engine) -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router
    from deltaswamp.table import Connection

    available = {Engine.KERNEL: KernelEngine, Engine.DELTARS: DeltaRsEngine}
    engines = {k: available[k]() for k in (kinds or (Engine.KERNEL, Engine.DELTARS))}
    return Connection(catalog=FilesystemCatalog(), router=Router(engines=engines))


def _rows(n: int = 3, start: int = 0) -> Any:
    ids = list(range(start, start + n))
    return pa.table({"id": pa.array(ids, pa.int64()), "s": [str(i) for i in ids]})


@pytest.fixture
def path(tmp_path: Any) -> str:
    p = str(tmp_path / "t")
    deltalake.write_deltalake(p, _rows())
    return p


def _ids(conn: Any, path: str) -> list[int]:
    return sorted(conn.open_table(path).to_arrow().column("id").to_pylist())


# ----------------------------------------------------------- exactly-once txn


@pytest.mark.parametrize("route", [Engine.DELTARS, Engine.KERNEL])
def test_txn_check_that_raced_the_winner_does_not_append_twice(
    path: str, route: Engine, monkeypatch: Any
) -> None:
    """Two writers of one (app_id, version): the one whose check ran first must not
    commit a second copy once the other has committed."""
    conn = _conn(route)
    conn.open_table(path).append(_rows(1, 100), txn=("job", 7))
    late = conn.open_table(path)
    # The late writer's txn check ran before the winner committed.
    real = type(late)._already_committed
    calls: list[int] = []

    def stale_once(self: Any, txn: Any) -> bool:
        calls.append(1)
        return False if len(calls) == 1 else real(self, txn)

    monkeypatch.setattr(type(late), "_already_committed", stale_once)
    late.append(_rows(1, 100), txn=("job", 7))
    monkeypatch.undo()
    assert collections.Counter(_ids(conn, path))[100] == 1


def test_deltars_txn_race_inside_the_commit_is_caught(path: str, monkeypatch: Any) -> None:
    """The winner commits between delta-rs loading its snapshot and committing:
    delta-rs's retry rebased over it and appended the batch again."""
    from deltaswamp.engine import deltars

    conn = _conn(Engine.DELTARS)
    t = conn.open_table(path)
    real = deltalake.write_deltalake
    fired = []

    def racing(target: Any, data: Any, **kwargs: Any) -> Any:
        if not fired:
            fired.append(1)
            real(
                path,
                _rows(1, 100),
                mode="append",
                commit_properties=deltalake.CommitProperties(
                    app_transactions=[deltalake.Transaction("job", 7)]
                ),
            )
        return real(target, data, **kwargs)

    monkeypatch.setattr(deltalake, "write_deltalake", racing)
    monkeypatch.setattr(deltars, "write_deltalake", racing, raising=False)
    t.append(_rows(1, 100), txn=("job", 7))
    monkeypatch.undo()
    assert collections.Counter(_ids(conn, path))[100] == 1


def test_deltars_txn_append_retries_a_plain_lost_race(path: str, monkeypatch: Any) -> None:
    """A lost race to an unrelated writer is re-staged, not surfaced."""
    conn = _conn(Engine.DELTARS)
    t = conn.open_table(path)
    real = deltalake.write_deltalake
    fired = []

    def racing(target: Any, data: Any, **kwargs: Any) -> Any:
        if not fired:
            fired.append(1)
            real(path, _rows(1, 50), mode="append")
        return real(target, data, **kwargs)

    monkeypatch.setattr(deltalake, "write_deltalake", racing)
    t.append(_rows(1, 100), txn=("job", 1))
    monkeypatch.undo()
    assert 50 in _ids(conn, path) and collections.Counter(_ids(conn, path))[100] == 1
    assert conn.open_table(path).txn_version("job") == 1


# ------------------------------------------------------ conflict error types


def test_deltars_lost_race_is_a_commit_conflict_error(path: str, monkeypatch: Any) -> None:
    from deltaswamp.engine.deltars import DeltaRsEngine

    conn = _conn(Engine.DELTARS)
    t = conn.open_table(path)
    stale = deltalake.DeltaTable(path)
    conn.open_table(path).delete("id = 1")  # the winner removes the same file
    real = DeltaRsEngine._open
    monkeypatch.setattr(
        DeltaRsEngine,
        "_open",
        lambda self, table, *, version=None, write=False: stale
        if write
        else real(self, table, version=version, write=write),
    )
    with pytest.raises(CommitConflictError):
        t.delete("id = 2")


def test_non_conflict_commit_failure_is_not_relabelled() -> None:
    from deltaswamp.engine.deltars import _as_commit_conflict

    class CommitFailedError(Exception):
        pass

    append_only = CommitFailedError(
        "The transaction includes Remove action with data change but Delta table is append-only"
    )
    assert _as_commit_conflict(append_only) is None
    assert isinstance(
        _as_commit_conflict(CommitFailedError("Failed to commit transaction: 0")),
        CommitConflictError,
    )


def test_kernel_metadata_commit_that_keeps_losing_is_a_conflict(
    path: str, monkeypatch: Any
) -> None:
    from deltaswamp import _native

    conn = _conn(Engine.KERNEL)
    t = conn.open_table(path)

    def lose(*args: Any, **kwargs: Any) -> int:
        raise _native.CommitConflictError("version 9 already exists")

    monkeypatch.setattr(_native, "commit_raw", lose)
    with pytest.raises(CommitConflictError):
        t.set_properties({"x.k": "v"})


# -------------------------------------------------- compaction conflict hole


def _stale_write_open(monkeypatch: Any, stale: Any) -> None:
    from deltaswamp.engine.deltars import DeltaRsEngine

    real = DeltaRsEngine._open
    served = []

    def fake(self: Any, table: Any, *, version: Any = None, write: bool = False) -> Any:
        if write and version is None and not served:
            served.append(1)
            return stale
        return real(self, table, version=version, write=write)

    monkeypatch.setattr(DeltaRsEngine, "_open", fake)


def test_optimize_over_a_concurrent_optimize_does_not_duplicate_rows(
    tmp_path: Any, monkeypatch: Any
) -> None:
    p = str(tmp_path / "t")
    for i in range(4):
        deltalake.write_deltalake(p, _rows(1, i), mode="append")
    conn = _conn(Engine.DELTARS)
    t = conn.open_table(p)
    stale = deltalake.DeltaTable(p)
    deltalake.DeltaTable(p).optimize.compact()  # the other process wins
    _stale_write_open(monkeypatch, stale)
    with pytest.raises(CommitConflictError, match="rolled back"):
        t.optimize()
    monkeypatch.undo()
    assert _ids(conn, p) == [0, 1, 2, 3]


def test_overwrite_after_a_concurrent_optimize_conflicts(tmp_path: Any, monkeypatch: Any) -> None:
    """delta-rs rebased the overwrite over the compaction and left the compacted
    file live: the "replaced" rows stayed next to the new ones."""
    p = str(tmp_path / "t")
    for i in range(3):
        deltalake.write_deltalake(p, _rows(1, i), mode="append")
    conn = _conn(Engine.DELTARS)
    t = conn.open_table(p)
    stale = deltalake.DeltaTable(p)
    deltalake.DeltaTable(p).optimize.compact()
    real = deltalake.write_deltalake
    monkeypatch.setattr(
        deltalake,
        "write_deltalake",
        lambda target, data, **kw: real(stale if isinstance(target, str) else target, data, **kw),
    )
    with pytest.raises(CommitConflictError):
        t.overwrite(_rows(1, 9))
    monkeypatch.undo()
    assert _ids(conn, p) == [0, 1, 2]


def test_compactions_from_threads_are_serialised(tmp_path: Any) -> None:
    p = str(tmp_path / "t")
    for i in range(6):
        deltalake.write_deltalake(p, _rows(1, i), mode="append")
    conn = _conn(Engine.DELTARS)
    t = conn.open_table(p)
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def run() -> None:
        barrier.wait()
        try:
            t.optimize()
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(30)
    assert not errors
    assert _ids(conn, p) == list(range(6))


# ------------------------------------------------------------- fork safety


def test_forked_delta_rs_runtime_is_a_clear_error_and_rerouted(path: str, monkeypatch: Any) -> None:
    from deltaswamp.engine import deltars

    class PanicException(BaseException):
        pass

    def forked(*args: Any, **kwargs: Any) -> Any:
        raise PanicException("Forked process detected - current PID is 2 but the tokio runtime...")

    monkeypatch.setattr(deltalake, "DeltaTable", forked)
    try:
        engine = deltars.DeltaRsEngine()
        from deltaswamp.catalog.filesystem import FilesystemCatalog
        from deltaswamp.identity import parse_ref

        resolved = FilesystemCatalog().resolve(parse_ref(path))
        with pytest.raises(UnreachableTableError, match="spawn"):
            engine.history(resolved)
        cap = engine.supports(Operation.APPEND, resolved)
        assert not cap.ok and "forked" in cap.reason
    finally:
        deltars._FORKED_RUNTIME.clear()


# ---------------------------------------------------- stale handle state


def test_enrichment_that_raced_a_commit_is_not_cached(path: str, monkeypatch: Any) -> None:
    from deltaswamp.engine.kernel import KernelEngine

    conn = _conn()
    t = conn.open_table(path)
    real = KernelEngine.detail

    def detail_while_a_commit_lands(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = real(self, *args, **kwargs)
        t._invalidate()  # another thread committed through this handle meanwhile
        return result

    monkeypatch.setattr(KernelEngine, "detail", detail_while_a_commit_lands)
    t._enrich()
    assert t._enriched is False


def test_write_through_stale_handle_gets_the_routers_refusal(path: str) -> None:
    conn = _conn()
    stale = conn.open_table(path)
    stale.count()
    conn.open_table(path).set_properties({"delta.appendOnly": "true"})
    with pytest.raises(UnreachableTableError, match="append-only"):
        stale.delete("id = 1")


def test_properties_see_another_writers_alter(path: str) -> None:
    conn = _conn()
    t = conn.open_table(path)
    assert "x.k" not in t.properties()
    conn.open_table(path).set_properties({"x.k": "v"})
    assert t.properties().get("x.k") == "v"


def test_append_realigns_after_a_concurrent_add_column(path: str, monkeypatch: Any) -> None:
    from deltaswamp.engine.deltars import DeltaRsEngine

    conn = _conn()
    t = conn.open_table(path)
    real = DeltaRsEngine.append
    fired = []

    def column_lands_first(self: Any, table: Any, data: Any, **kwargs: Any) -> Any:
        if not fired:
            fired.append(1)
            conn.open_table(path).add_column(pa.field("extra", pa.string()))
        return real(self, table, data, **kwargs)

    monkeypatch.setattr(DeltaRsEngine, "append", column_lands_first)
    t.append(_rows(1, 50))
    assert 50 in _ids(conn, path)


def test_overwrite_realigns_after_a_concurrent_add_column(path: str, monkeypatch: Any) -> None:
    from deltaswamp.engine.deltars import DeltaRsEngine

    conn = _conn()
    t = conn.open_table(path)
    real = DeltaRsEngine.overwrite
    fired = []

    def column_lands_first(self: Any, table: Any, data: Any, **kwargs: Any) -> Any:
        if not fired:
            fired.append(1)
            conn.open_table(path).add_column(pa.field("extra", pa.string()))
        return real(self, table, data, **kwargs)

    monkeypatch.setattr(DeltaRsEngine, "overwrite", column_lands_first)
    t.overwrite(pa.table({"id": pa.array([0], pa.int64()), "s": ["new"]}), predicate="id = 0")
    rows = conn.open_table(path).to_arrow()
    assert "extra" in rows.column_names
    assert sorted(
        zip(rows.column("id").to_pylist(), rows.column("s").to_pylist(), strict=True)
    ) == [
        (0, "new"),
        (1, "1"),
        (2, "2"),
    ]


# ------------------------------------------------------------------ pickling


def test_native_exceptions_pickle() -> None:
    import pickle

    from deltaswamp import _native

    for name in ("CommitConflictError", "InvalidInputError", "RetryableError"):
        cls = getattr(_native, name)
        again = pickle.loads(pickle.dumps(cls("boom")))
        assert type(again) is cls and str(again) == "boom"


def test_sql_engine_pickles_after_use_and_with_a_config() -> None:
    import pickle

    from deltaswamp.engine.sql import SqlEngine

    class Config:
        host = "https://example.cloud.databricks.com"
        unpicklable = threading.Lock()

        def as_dict(self) -> dict[str, Any]:
            return {"host": self.host, "auth_type": "pat", "token": "dapi-x"}

    engine = SqlEngine(config=Config(), warehouse_id="w1")
    engine._client = threading.Lock()  # a live client: not picklable
    again = pickle.loads(pickle.dumps(engine))
    assert again._client is None and again._config is None
    assert again._config_kwargs["host"] == Config.host
    assert again._warehouse_id == "w1"


# ------------------------------------------- metadata races through delta-rs


def test_deltars_alter_is_recomputed_after_a_concurrent_metadata_change(
    path: str, monkeypatch: Any
) -> None:
    conn = _conn(Engine.DELTARS)
    t = conn.open_table(path)
    stale = deltalake.DeltaTable(path)
    deltalake.DeltaTable(path).alter.set_table_description("someone else")
    _stale_write_open(monkeypatch, stale)
    t.add_column(pa.field("extra", pa.string()))
    monkeypatch.undo()
    assert "extra" in conn.open_table(path).schema().names


def test_blind_append_is_rewritten_after_a_concurrent_metadata_change(
    path: str, monkeypatch: Any
) -> None:
    conn = _conn(Engine.DELTARS)
    t = conn.open_table(path)
    real = deltalake.write_deltalake
    fired = []

    def metadata_lands_first(target: Any, data: Any, **kwargs: Any) -> Any:
        if not fired:
            fired.append(1)
            stale = deltalake.DeltaTable(path)
            deltalake.DeltaTable(path).alter.set_table_description("someone else")
            return real(stale, data, **{k: v for k, v in kwargs.items() if k != "storage_options"})
        return real(target, data, **kwargs)

    monkeypatch.setattr(deltalake, "write_deltalake", metadata_lands_first)
    t.append(_rows(1, 50))
    monkeypatch.undo()
    assert collections.Counter(_ids(conn, path))[50] == 1


# ------------------------------------------------------- concurrent creation


def _stale_exists(monkeypatch: Any, misses: int = 2) -> None:
    """table_exists() answers "no" `misses` times: the checks ran before the winner."""
    from deltaswamp.table import Connection

    real = Connection.table_exists
    calls: list[int] = []

    def stale(self: Any, name: str) -> bool:
        calls.append(1)
        return False if len(calls) <= misses else real(self, name)

    monkeypatch.setattr(Connection, "table_exists", stale)


@pytest.mark.parametrize(
    ("mode", "expected"), [("append", [1, 2]), ("overwrite", [2]), ("ignore", [1])]
)
def test_write_table_that_lost_the_create_race_follows_its_mode(
    tmp_path: Any, monkeypatch: Any, mode: str, expected: list[int]
) -> None:
    conn = _conn()
    p = str(tmp_path / "t")
    conn.write_table(p, pa.table({"id": pa.array([1], pa.int64())}))  # the winner
    _stale_exists(monkeypatch)
    conn.write_table(p, pa.table({"id": pa.array([2], pa.int64())}), mode=mode)
    monkeypatch.undo()
    assert _ids(conn, p) == expected


def test_create_that_lost_the_race_is_a_library_error(tmp_path: Any, monkeypatch: Any) -> None:
    conn = _conn()
    p = str(tmp_path / "t")
    conn.write_table(p, pa.table({"id": pa.array([1], pa.int64())}))
    _stale_exists(monkeypatch, misses=1)
    with pytest.raises(UnreachableTableError, match="already exists"):
        conn.create_table(p, pa.schema([("id", pa.int64())]))
    _stale_exists(monkeypatch, misses=1)
    assert conn.create_table(p, pa.schema([("id", pa.int64())]), mode="ignore").count() == 1


def test_failed_first_write_keeps_another_writers_commits(tmp_path: Any, monkeypatch: Any) -> None:
    """write_table removes the log of the table it created when its first write
    fails -- which deleted rows another writer had appended in between."""
    from deltaswamp.table import Table

    conn = _conn()
    p = str(tmp_path / "t")
    real = Table.append

    def someone_appends_then_we_fail(self: Any, data: Any, **kwargs: Any) -> None:
        real(conn.open_table(p), pa.table({"id": pa.array([7], pa.int64())}))
        raise RuntimeError("our write failed")

    monkeypatch.setattr(Table, "append", someone_appends_then_we_fail)
    with pytest.raises(RuntimeError):
        conn.write_table(p, pa.table({"id": pa.array([1], pa.int64())}))
    monkeypatch.undo()
    assert _ids(conn, p) == [7]


def test_delete_all_after_a_concurrent_optimize_conflicts(tmp_path: Any, monkeypatch: Any) -> None:
    """A delete of every row that lost to an OPTIMIZE reported success and left
    every row in the table."""
    p = str(tmp_path / "t")
    for i in range(3):
        deltalake.write_deltalake(p, _rows(1, i), mode="append")
    conn = _conn(Engine.DELTARS)
    t = conn.open_table(p)
    stale = deltalake.DeltaTable(p)
    deltalake.DeltaTable(p).optimize.compact()
    _stale_write_open(monkeypatch, stale)
    with pytest.raises(CommitConflictError):
        t.delete()
    monkeypatch.undo()
    assert _ids(conn, p) == [0, 1, 2]
    t.delete()  # uncontended, it still deletes everything
    assert _ids(conn, p) == []


def test_kernel_in_a_macos_fork_child_refuses_instead_of_crashing(
    path: str, monkeypatch: Any
) -> None:
    """The first native call in such a child killed it with SIGTRAP."""
    import os
    import sys

    from deltaswamp.engine import kernel

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(kernel, "_RUNTIME_OWNER", [os.getpid() + 1])  # "the parent"
    conn = _conn(Engine.KERNEL)
    t = conn.open_table(path)
    assert not t.can(Operation.SCAN).ok
    with pytest.raises(UnreachableTableError, match="spawn"):
        conn.router.engines[Engine.KERNEL].snapshot(t.resolved)
