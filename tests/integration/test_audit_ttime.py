"""Timestamp -> version resolution on logs whose commit files are out of time order.

A commit's timestamp is its in-commit timestamp, and before those are enabled
its commit file's modification time made monotonic: a commit is at least a
millisecond after the one before it (Spark's DeltaHistoryManager, and what
Databricks answered for these exact file times, see
tests/live/test_live_ttime.py). The kernel's history manager compared a
timestamp with the latest commit's *raw* file time first, so any time after
it read the latest version; delta-rs compared raw file times throughout, and
history and the change feed reported them.

Every test here failed before the fix.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
from typing import Any

import pytest
from deltaswamp.errors import InvalidArgumentError, UnreachableTableError

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

UTC = dt.UTC
SCHEMA = pa.schema([("id", pa.int64())])
CDF = {"delta.enableChangeDataFeed": "true"}
#: Whole seconds, as object stores report them, in epoch ms.
T = 1_700_000_000_000
S = 1000


def build(
    conn: Any,
    path: pathlib.Path,
    commits: int,
    *,
    properties: dict[str, str] | None = None,
    ict_at: int | None = None,
) -> str:
    """create (v0), then one row per commit (id = version); ICT turned on at `ict_at`."""
    del conn  # built where every property can be set; read through the one under test
    t = _connection("both").create_table(
        str(path), SCHEMA, properties={**CDF, **(properties or {})}
    )
    for k in range(1, commits):
        if k == ict_at:
            t.set_properties({"delta.enableInCommitTimestamps": "true"})
        else:
            t.append(pa.table({"id": [k]}, schema=SCHEMA))
    return str(path)


def set_mtimes(path: str, millis: dict[int, int]) -> None:
    for version, when in millis.items():
        f = pathlib.Path(path, "_delta_log", f"{version:020}.json")
        os.utime(f, ns=(when * 1_000_000, when * 1_000_000))


def in_commit_timestamps(path: str) -> dict[int, int]:
    out = {}
    for f in sorted(pathlib.Path(path, "_delta_log").glob("[0-9]*.json")):
        for line in f.read_text().splitlines():
            info = json.loads(line).get("commitInfo") if line.strip() else None
            if info and "inCommitTimestamp" in info:
                out[int(f.name[:20])] = int(info["inCommitTimestamp"])
    return out


def at(millis: int) -> dt.datetime:
    return dt.datetime(1970, 1, 1, tzinfo=UTC) + dt.timedelta(milliseconds=millis)


def _connection(kind: str) -> Any:
    from deltaswamp import Connection
    from deltaswamp.capability import Engine
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router

    engines: dict[Any, Any] = {}
    if kind in ("kernel", "both"):
        engines[Engine.KERNEL] = KernelEngine()
    if kind in ("deltars", "both"):
        engines[Engine.DELTARS] = DeltaRsEngine()
    return Connection(catalog=FilesystemCatalog(), router=Router(engines=engines))


@pytest.fixture(params=["kernel", "deltars", "both"])
def conn(request: Any) -> Any:
    return _connection(request.param)


def _feed(t: Any, **kwargs: Any) -> list[int]:
    got = t.cdf(**kwargs).read_all().column("_commit_version").to_pylist()
    return sorted(set(got))


def _feed_times(t: Any) -> dict[int, int]:
    feed = t.cdf(starting_version=0).read_all()
    return {
        v: s.value // 1000
        for v, s in zip(
            feed.column("_commit_version").to_pylist(),
            feed.column("_commit_timestamp"),
            strict=True,
        )
    }


def _history(t: Any) -> dict[int, int]:
    return {h["version"]: h["timestamp"] for h in t.history()}


class TestFileTimesOutOfOrder:
    """v1's file is the newest; v2 and v3 look older (Databricks' table `a`)."""

    @pytest.fixture
    def table(self, conn: Any, tmp_path: Any) -> Any:
        path = build(conn, tmp_path / "t", 4)
        # Made monotonic: v0 T, v1 T+30, v2 T+30.001, v3 T+30.002.
        set_mtimes(path, {0: T, 1: T + 30 * S, 2: T + 10 * S, 3: T + 20 * S})
        return conn.open_table(path)

    def test_time_travel(self, table: Any) -> None:
        # Raw file times put v2 at T+10 and v3 at T+20.
        assert table.to_arrow(timestamp=at(T + 15 * S)).num_rows == 0
        assert table.to_arrow(timestamp=at(T + 25 * S)).num_rows == 0
        assert table.to_arrow(timestamp=at(T + 30 * S)).num_rows == 1
        assert table.to_arrow(timestamp=at(T + 30 * S + 1)).num_rows == 2
        assert pa.table(table.scan(timestamp=T + 30 * S + 2)).num_rows == 3
        with pytest.raises(InvalidArgumentError, match="after the latest commit"):
            table.to_arrow(timestamp=at(T + 30 * S + 3))
        with pytest.raises(UnreachableTableError):
            table.to_arrow(timestamp=at(T - S))

    def test_change_feed_bounds(self, table: Any) -> None:
        assert _feed(table, starting_timestamp=at(T + 25 * S)) == [1, 2, 3]
        assert _feed(table, starting_timestamp=at(T + 30 * S + 1)) == [2, 3]
        assert _feed(table, starting_timestamp=at(T - S)) == [1, 2, 3]
        assert _feed(table, starting_version=0, ending_timestamp=at(T + 25 * S)) == []
        assert _feed(table, starting_version=0, ending_timestamp=at(T + 30 * S + 1)) == [
            1,
            2,
        ]
        for bound in ("starting_timestamp", "ending_timestamp"):
            with pytest.raises(InvalidArgumentError, match="after the latest commit"):
                table.cdf(**{bound: at(T + 30 * S + 3)}).read_all()
        with pytest.raises(InvalidArgumentError):
            table.cdf(starting_version=0, ending_timestamp=at(T - S)).read_all()

    def test_change_feed_out_of_range_allowed(self, conn: Any, table: Any) -> None:
        if not conn.router.engines.get(ds.capability.Engine.DELTARS):
            pytest.skip("allow_out_of_range is read by delta-rs")
        after = at(T + 30 * S + 3)
        assert table.cdf(starting_timestamp=after, allow_out_of_range=True).read_all().num_rows == 0
        assert _feed(
            table,
            starting_timestamp=at(T + 25 * S),
            ending_timestamp=after,
            allow_out_of_range=True,
        ) == [1, 2, 3]

    def test_change_feed_commit_times(self, table: Any) -> None:
        base = T + 30 * S
        assert _feed_times(table) == {1: base, 2: base + 1, 3: base + 2}

    def test_history(self, conn: Any, table: Any) -> None:
        if not conn.router.engines.get(ds.capability.Engine.DELTARS):
            pytest.skip("history is served by delta-rs")
        base = T + 30 * S
        assert _history(table) == {0: T, 1: base, 2: base + 1, 3: base + 2}

    def test_restore(self, table: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="after the latest commit"):
            table.restore(at(T + 30 * S + 3))
        table.restore(at(T + 25 * S))
        assert table.to_arrow().num_rows == 0


class TestLatestFileLooksOldest:
    """The latest commit's file is older than every other (Databricks' table `b`)."""

    @pytest.fixture
    def table(self, conn: Any, tmp_path: Any) -> Any:
        path = build(conn, tmp_path / "t", 4)
        # Made monotonic: v3 is T+20.001.
        set_mtimes(path, {0: T, 1: T + 10 * S, 2: T + 20 * S, 3: T - 100 * S})
        return conn.open_table(path)

    def test_time_travel(self, table: Any) -> None:
        # Every time after T-100 read the latest version.
        assert table.to_arrow(timestamp=at(T + 5 * S)).num_rows == 0
        assert table.to_arrow(timestamp=at(T + 15 * S)).num_rows == 1
        assert table.to_arrow(timestamp=at(T + 20 * S)).num_rows == 2
        assert table.to_arrow(timestamp=at(T + 20 * S + 1)).num_rows == 3
        with pytest.raises(InvalidArgumentError, match="after the latest commit"):
            table.to_arrow(timestamp=at(T + 21 * S))

    def test_change_feed(self, table: Any) -> None:
        # "No commit at or after" any time past T-100.
        assert _feed(table, starting_timestamp=at(T + 15 * S)) == [2, 3]
        assert _feed(table, starting_version=1, ending_timestamp=at(T + 15 * S)) == [1]
        assert _feed_times(table)[3] == T + 20 * S + 1


class TestInCommitTimestamps:
    """In-commit timestamps decide, whatever the files' times."""

    def test_enabled_from_creation(self, conn: Any, tmp_path: Any) -> None:
        path = build(conn, tmp_path / "t", 4, properties={"delta.enableInCommitTimestamps": "true"})
        set_mtimes(path, {0: T + 30 * S, 1: T, 2: T + 10 * S, 3: T - 50 * S})
        icts = in_commit_timestamps(path)
        t = conn.open_table(path)
        assert t.to_arrow(timestamp=at(icts[2])).num_rows == 2
        assert t.to_arrow(timestamp=at(icts[3])).num_rows == 3
        with pytest.raises(InvalidArgumentError, match="after the latest commit"):
            t.to_arrow(timestamp=at(icts[3] + 1))
        assert _feed_times(t) == {v: icts[v] for v in (1, 2, 3)}

    def test_enabled_later_on_a_copied_log(self, conn: Any, tmp_path: Any) -> None:
        # v2 enables them; the files before it were copied after it was
        # written, so their times are later than every in-commit timestamp
        # (Databricks' table `mid`).
        path = build(conn, tmp_path / "t", 5, ict_at=2)
        icts = in_commit_timestamps(path)
        enabled = icts[2]
        later = enabled + 100 * S
        set_mtimes(
            path, {0: later, 1: later + 5 * S, 2: later + 3 * S, 3: later + S, 4: later + 2 * S}
        )
        t = conn.open_table(path)
        with pytest.raises(UnreachableTableError):
            t.to_arrow(timestamp=at(enabled - 1))
        assert t.to_arrow(timestamp=at(enabled)).num_rows == 1
        assert t.to_arrow(timestamp=at(icts[4])).num_rows == 3
        with pytest.raises(InvalidArgumentError, match="after the latest commit"):
            t.to_arrow(timestamp=at(later))
        # Before in-commit timestamps: the first file-timed commit is after it.
        assert _feed(t, starting_timestamp=at(enabled - 1)) == [1, 3, 4]
        assert _feed(t, starting_timestamp=at(enabled)) == [3, 4]
        assert _feed_times(t) == {1: later + 5 * S, 3: icts[3], 4: icts[4]}

    def test_enabled_later_with_file_times_out_of_order_before(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = build(conn, tmp_path / "t", 5, ict_at=3)
        icts = in_commit_timestamps(path)
        e = icts[3]
        # Made monotonic: v0 e-30s, v1 e-20s, v2 a millisecond after v1.
        set_mtimes(
            path, {0: e - 30 * S, 1: e - 20 * S, 2: e - 40 * S, 3: e + 50 * S, 4: e + 60 * S}
        )
        t = conn.open_table(path)
        assert t.to_arrow(timestamp=at(e - 25 * S)).num_rows == 0
        assert t.to_arrow(timestamp=at(e - 20 * S)).num_rows == 1
        assert t.to_arrow(timestamp=at(e - 20 * S + 1)).num_rows == 2
        assert t.to_arrow(timestamp=at(icts[3])).num_rows == 2
        assert t.to_arrow(timestamp=at(icts[4])).num_rows == 3
        assert _feed(t, starting_timestamp=at(e - 25 * S)) == [1, 2, 4]
        assert _feed(t, starting_timestamp=at(e - 20 * S + 1)) == [2, 4]
        assert _feed_times(t)[2] == e - 20 * S + 1


class TestNative:
    """The extension's own commit times: what every lookup resolves against."""

    def test_commit_times(self, tmp_path: Any) -> None:
        from tests.helpers import snapshot

        path = build(_connection("kernel"), tmp_path / "t", 4)
        set_mtimes(path, {0: T, 1: T + 30 * S, 2: T + 10 * S, 3: T + 20 * S})
        snap = snapshot(path)
        base = T + 30 * S
        assert snap.file_commit_timestamps() == [
            (0, T),
            (1, base),
            (2, base + 1),
            (3, base + 2),
        ]
        assert snap.commit_timestamp() == base + 2
        assert snap.timestamp() == T + 20 * S  # the file's own
        assert snap.version_at(T + 25 * S) == (0, T)
        assert snap.version_at(T + 25 * S, at_or_after=True) == (1, base)
        with pytest.raises(ValueError, match="out of range"):
            snap.version_at(T - S)
        with pytest.raises(ValueError, match="out of range"):
            snap.version_at(base + 3, at_or_after=True)
        assert snapshot(path, timestamp_ms=base + 1).version == 2
        with pytest.raises(ValueError, match="after the latest commit"):
            snapshot(path, timestamp_ms=base + 3)
