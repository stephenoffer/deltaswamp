"""Machinery for the Databricks interop suite (test_live_interop.py).

Each case builds a table locally through a deltaswamp operation history,
uploads the table directory to a Unity Catalog volume, and then asks a SQL
warehouse -- ground truth for what Databricks makes of a Delta table -- the
same questions deltaswamp answers. Databricks then writes on top, and the
result comes back for deltaswamp to read and write again.

Nothing here asserts. `run_case` gathers evidence (row differences, detail
mismatches, predicate counts, ...) per check and the test module asserts on
it, so a case's warehouse work can run in a thread pool while the tests stay
one assertion each. Every name the warehouse sees lives under one volume the
session creates with a random name and drops at the end.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import datetime as dt
import decimal
import hashlib
import io
import json
import math
import re
import shutil
import time
import traceback
import urllib.parse
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import deltaswamp as ds
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from deltaswamp.capability import Engine
from deltaswamp.errors import (
    ChangeFeedSchemaChangeError,
    InvalidArgumentError,
    UnreachableTableError,
)

UTC = dt.UTC
D = decimal.Decimal

#: What an honest refusal looks like: typed, raised before anything is written.
REFUSALS: tuple[type[BaseException], ...] = (UnreachableTableError, InvalidArgumentError)


# ======================================================================= data

ANCIENT = [
    dt.date(1, 1, 1),
    dt.date(1000, 2, 28),
    dt.date(1582, 10, 4),
    dt.date(1582, 10, 15),
    dt.date(1850, 3, 1),
    dt.date(1899, 12, 31),
    dt.date(1970, 1, 1),
    dt.date(2024, 2, 29),
    dt.date(9999, 12, 31),
    None,
]
MODERN = [
    dt.date(1970, 1, 1),
    dt.date(1999, 12, 31),
    dt.date(2000, 2, 29),
    dt.date(2001, 3, 1),
    dt.date(2010, 6, 15),
    dt.date(2020, 1, 1),
    dt.date(2021, 7, 4),
    dt.date(2024, 2, 29),
    dt.date(9999, 12, 31),
    None,
]
STRS = ["a", "", None, "x/y=z", "sp ace", "pct%20", "é漢字", "hash#q?", "O'Brien", "tab\tnl\n"]
DECS = [
    D("12345678901234567890.123456789012345678"),
    D("-0.000000000000000001"),
    D("0"),
    None,
    D("99999999999999999999.999999999999999999"),
    D("-99999999999999999999.999999999999999999"),
    D("1.5"),
    D("-2.25"),
    D("3"),
    D("0.1"),
]
FLOATS = [1.5, -0.0, float("nan"), float("inf"), float("-inf"), None, 3.25, -7.0, 1e300, 2.0]


def _ts(d: dt.date | None, i: int) -> dt.datetime | None:
    if d is None:
        return None
    return dt.datetime(d.year, d.month, d.day, (i * 7) % 24, i % 60, 56, 123456 + i)


def _utc(d: dt.date | None, i: int) -> dt.datetime | None:
    t = _ts(d, i)
    return None if t is None else t.replace(tzinfo=UTC)


def rows(
    base: int,
    n: int = 10,
    *,
    ancient: bool = False,
    ntz: bool = False,
    long: bool = False,
    nested: bool = True,
    part: bool = False,
    nested_ts: bool = False,
    extra: Mapping[str, Callable[[list[int], list[int], list[Any]], Any]] | None = None,
) -> pa.Table:
    """Rows base..base+n-1 across a broad set of column types.

    Deterministic, so a rerun builds the same files. `ancient` swaps in dates
    before 1582 and 1900 (the calendar-rebase edge cases); `long` adds 20-40 KB
    strings, well past the 32-character stats truncation.
    """
    dates = ANCIENT if ancient else MODERN
    ids = list(range(base, base + n))
    k = [i % 10 for i in ids]
    cols: dict[str, Any] = {
        "id": pa.array(ids, pa.int64()),
        "i": pa.array(
            [None if x == 3 else (i * 37) % 1000 - 500 for i, x in zip(ids, k, strict=True)],
            pa.int32(),
        ),
        "s": pa.array(
            [STRS[x] if x != 0 else f"s{i}" for i, x in zip(ids, k, strict=True)], pa.string()
        ),
        "d": pa.array([dates[x] for x in k], pa.date32()),
        "ts": pa.array(
            [_utc(dates[x], i) for i, x in zip(ids, k, strict=True)],
            pa.timestamp("us", "UTC"),
        ),
        "dec": pa.array([DECS[x] for x in k], pa.decimal128(38, 18)),
        "f": pa.array([FLOATS[x] for x in k], pa.float64()),
        "fl": pa.array(
            [None if x == 4 else float(i % 7) / 4 for i, x in zip(ids, k, strict=True)],
            pa.float32(),
        ),
        "b": pa.array(
            [None if x == 5 else i % 2 == 0 for i, x in zip(ids, k, strict=True)], pa.bool_()
        ),
        "bin": pa.array(
            [None if x == 6 else bytes([i % 256, 0, 255]) for i, x in zip(ids, k, strict=True)],
            pa.binary(),
        ),
        "sm": pa.array([(i % 60000) - 30000 for i in ids], pa.int16()),
        "by": pa.array([(i % 250) - 125 for i in ids], pa.int8()),
    }
    if ntz:
        cols["n"] = pa.array(
            [_ts(dates[x], i) for i, x in zip(ids, k, strict=True)], pa.timestamp("us")
        )
    if nested:
        cols["st"] = pa.array(
            [
                None if x == 7 else {"a": i, "b": STRS[x], "c": {"dd": dates[x]}}
                for i, x in zip(ids, k, strict=True)
            ],
            pa.struct(
                [
                    ("a", pa.int64()),
                    ("b", pa.string()),
                    ("c", pa.struct([("dd", pa.date32())])),
                ]
            ),
        )
        cols["arr"] = pa.array(
            [
                None if x == 8 else [i, None, i * 2] if x % 2 else []
                for i, x in zip(ids, k, strict=True)
            ],
            pa.list_(pa.int64()),
        )
        cols["m"] = pa.array(
            [
                None if x == 9 else [(f"k{i}", i), ("z", None)] if x % 3 else []
                for i, x in zip(ids, k, strict=True)
            ],
            pa.map_(pa.string(), pa.int64()),
        )
    if nested_ts:
        cols["aos"] = pa.array(
            [
                None if x == 2 else [{"x": i, "y": [_utc(dates[x] or dt.date(2000, 1, 1), i)]}]
                for i, x in zip(ids, k, strict=True)
            ],
            pa.list_(pa.struct([("x", pa.int64()), ("y", pa.list_(pa.timestamp("us", "UTC")))])),
        )
        cols["ats"] = pa.array(
            [[_utc(dates[x] or dt.date(2000, 1, 1), i)] for i, x in zip(ids, k, strict=True)],
            pa.list_(pa.timestamp("us", "UTC")),
        )
    if long:
        cols["ls"] = pa.array(
            [f"L{i}-" * (20000 + i) if x != 1 else None for i, x in zip(ids, k, strict=True)],
            pa.string(),
        )
    if part:
        cols["ps"] = pa.array([STRS[(x * 3) % 10] for x in k], pa.string())
        cols["pd"] = pa.array([dates[(x * 7) % 10] for x in k], pa.date32())
        cols["pi"] = pa.array([None if x == 4 else x % 3 for x in k], pa.int32())
        cols["pts"] = pa.array(
            [
                None
                if dates[(x * 3) % 10] is None
                else dt.datetime(dates[(x * 3) % 10].year, 1, 2, 3, 4, 5, 6, tzinfo=UTC)  # type: ignore[union-attr]
                for x in k
            ],
            pa.timestamp("us", "UTC"),
        )
    for name, fn in (extra or {}).items():
        cols[name] = fn(ids, k, dates)
    return pa.table(cols)


# ================================================================= warehouse


class Warehouse:
    """Statements on a SQL warehouse: ground truth for what Databricks sees."""

    def __init__(self, workspace: Any, warehouse_id: str) -> None:
        self.workspace = workspace
        self.warehouse_id = warehouse_id

    def run(self, sql: str) -> Any:
        api = self.workspace.statement_execution
        r = api.execute_statement(warehouse_id=self.warehouse_id, statement=sql, wait_timeout="50s")
        while r.status.state.value in ("PENDING", "RUNNING"):
            time.sleep(1)
            r = api.get_statement(r.statement_id)
        if r.status.error:
            raise RuntimeError(f"{_short(sql, 200)!r}: {r.status.error.message}")
        if r.status.state.value != "SUCCEEDED":
            raise RuntimeError(f"{_short(sql, 200)!r}: statement {r.status.state.value}")
        return r

    def rows(self, sql: str) -> list[list[Any]]:
        r = self.run(sql)
        out = list(r.result.data_array or []) if r.result else []
        chunk = r.result.next_chunk_index if r.result else None
        while chunk is not None:
            part = self.workspace.statement_execution.get_statement_result_chunk_n(
                r.statement_id, chunk
            )
            out += list(part.data_array or [])
            chunk = part.next_chunk_index
        return out

    def record(self, sql: str) -> dict[str, Any]:
        """The first row of a result, by column name."""
        r = self.run(sql)
        names = [c.name for c in r.manifest.schema.columns]
        data = (r.result.data_array or []) if r.result else []
        return dict(zip(names, data[0], strict=True)) if data else {}


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


# ==================================================================== staging


class Staging:
    """Table directories on a UC volume, through deltaswamp's own Volume API.

    The Files API has no directory upload, so a directory goes up as parallel
    single-file writes -- no archive, nothing to unpack on the far side.
    """

    def __init__(self, volume: Any, *, workers: int = 8) -> None:
        self.volume = volume
        self.root: str = volume.root
        self.workers = workers

    def uri(self, rel: str) -> str:
        return f"{self.root}/{rel}"

    def put(self, rel: str, data: bytes) -> None:
        _retry(lambda: self.volume.write(rel, data, overwrite=True))

    def upload_dir(self, local: Path, rel: str, *, only: Iterable[Path] | None = None) -> int:
        files = sorted(only) if only is not None else [p for p in local.rglob("*") if p.is_file()]
        spelled = _log_spellings(local)

        def up(path: Path) -> None:
            sub = path.relative_to(local).as_posix()
            # A case-insensitive local filesystem (macOS) folds the random
            # column-mapping prefixes `AY/` and `aY/` into one directory; the
            # volume is case-sensitive, so upload under the log's spelling.
            sub = spelled.get(sub.lower(), sub)
            self.put(f"{rel}/{sub}", path.read_bytes())

        with concurrent.futures.ThreadPoolExecutor(self.workers) as pool:
            list(pool.map(up, files))
        return len(files)

    def listing(self, rel: str) -> list[str]:
        """Every file under `rel`, as volume-relative paths."""
        out: list[str] = []
        pending = [rel]
        while pending:
            here = pending.pop()
            for entry in _retry(lambda h=here: self.volume.list(h)):  # type: ignore[misc]
                (pending if entry.is_directory else out).append(entry.path)
        return out

    def download_dir(self, rel: str, local: Path) -> set[Path]:
        files = self.listing(rel)
        prefix = self.volume.path(rel)[len(self.root) :].strip("/")

        def down(path: str) -> Path:
            dest = local / path[len(prefix) :].lstrip("/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_retry(lambda: self.volume.read(path)))
            return dest

        with concurrent.futures.ThreadPoolExecutor(self.workers) as pool:
            return set(pool.map(down, files))


def _retry(fn: Callable[[], Any], attempts: int = 4) -> Any:
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 * (i + 1))
    return None


def _log_spellings(local: Path) -> dict[str, str]:
    """Lower-cased relative path -> the spelling the log uses for it."""
    out: dict[str, str] = {}
    log = local / "_delta_log"
    for f in log.glob("*.json"):
        for m in re.finditer(r'"path":"([^"]+)"', f.read_text(errors="ignore")):
            p = urllib.parse.unquote(m.group(1))
            out[p.lower()] = p
    for f in [*log.glob("*.checkpoint*.parquet"), *(log / "_sidecars").glob("*.parquet")]:
        with contextlib.suppress(Exception):
            names = pq.read_schema(f).names
            t = pq.read_table(f, columns=[c for c in ("add", "remove") if c in names])
            for c in t.column_names:
                for x in t.column(c).to_pylist():
                    if x and x.get("path"):
                        p = urllib.parse.unquote(x["path"])
                        out[p.lower()] = p
    return out


def _digests(root: Path) -> dict[Path, str]:
    return {p: hashlib.sha1(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


# ==================================================================== history


@dataclass
class Step:
    name: str
    outcome: str  # "ok", "refused" or "error"
    before: int | None
    after: int | None
    expect: str | None
    detail: str = ""


def _single_engine(kind: Engine) -> Any:
    """A path-table connection that can only route to one engine."""
    from deltaswamp import Connection
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.router import Router

    if kind is Engine.KERNEL:
        from deltaswamp.engine.kernel import KernelEngine

        return Connection(
            catalog=FilesystemCatalog(), router=Router(engines={kind: KernelEngine()})
        )
    from deltaswamp.engine.deltars import DeltaRsEngine

    return Connection(catalog=FilesystemCatalog(), router=Router(engines={kind: DeltaRsEngine()}))


class Builder:
    """Runs an operation history against a local table and records each step.

    A step either commits, or is refused with a typed error that leaves the
    table untouched; anything else is recorded as an error for the history
    test to report. `via` routes a step to one engine, so a history can write
    through the kernel and delta-rs alike.
    """

    def __init__(self, root: Path, ctx: Context | None) -> None:
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        self.root = root
        self.path = str(root / "work")
        self.ctx = ctx
        self.conns = {
            "default": ds.connect("file://"),
            "kernel": _single_engine(Engine.KERNEL),
            "deltars": _single_engine(Engine.DELTARS),
        }
        self.steps: list[Step] = []
        #: Versions another writer (plain deltalake, Databricks) committed.
        self.external: set[int] = set()
        self.marks: dict[str, int] = {}
        self.gen: Callable[..., pa.Table] = rows

    @property
    def conn(self) -> Any:
        return self.conns["default"]

    def version(self) -> int | None:
        try:
            v = self.conn.open_table(self.path).version
            return int(v) if v is not None else None
        except Exception:
            return None

    def step(
        self,
        name: str,
        fn: Callable[[Any], Any],
        *,
        via: str = "default",
        expect: str | None = None,
        external: bool = False,
    ) -> Step:
        before = self.version()
        table = self.conns[via].open_table(self.path) if before is not None else None
        outcome, detail = "ok", ""
        try:
            fn(table)
        except REFUSALS as exc:
            outcome, detail = "refused", f"{type(exc).__name__}: {_short(str(exc), 400)}"
        except Exception as exc:
            outcome = "error"
            detail = (
                f"{type(exc).__name__}: {_short(str(exc), 600)}\n{traceback.format_exc()[-1500:]}"
            )
        after = self.version()
        if external and outcome == "ok" and after is not None:
            self.external.update(range((before if before is not None else -1) + 1, after + 1))
        rec = Step(name, outcome, before, after, expect, detail)
        self.steps.append(rec)
        return rec

    def mark(self, name: str) -> None:
        v = self.version()
        if v is not None:
            self.marks[name] = v

    def snapshot(self, tag: str) -> Path:
        dest = self.root / tag
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(self.path, dest)
        return dest


def _ends(d: int) -> str:
    return ", ".join(str(b + d) for b in (0, 100, 200, 300, 400, 500, 600, 700, 800, 1200))


def _upsert(src: pa.Table, set_: Mapping[str, str] | None = None) -> Callable[[Any], Any]:
    def go(t: Any) -> Any:
        m = t.merge(src, "target.id = source.id")
        m = m.when_matched_update_all() if set_ is None else m.when_matched_update(dict(set_))
        return m.when_not_matched_insert_all().execute()

    return go


def _distributed(parts: list[pa.Table]) -> Callable[[Any], Any]:
    def go(t: Any) -> Any:
        plan = t.plan_write()
        return plan.commit([plan.write(p) for p in parts])

    return go


def _with_newc(b: Builder, base: int, n: int) -> pa.Table:
    return b.gen(base, n).append_column("newc", pa.array([f"n{base + i}" for i in range(n)]))


def maintenance(
    b: Builder,
    *,
    zorder: tuple[str, ...] | None = ("id",),
    vacuum: bool = True,
    restore_to: int | None = None,
    compact: bool = True,
) -> None:
    """The tail every history ends with: compaction, checkpoints, cleanup.

    Snapshots `pre` (full history, for time travel) before VACUUM and `final`
    after VACUUM and metadata cleanup with zero retention.
    """
    b.mark("stats")  # the most fragmented version: files from every writer
    if zorder:
        b.step("zorder", lambda t: t.z_order(list(zorder)))
    b.step("optimize", lambda t: t.optimize())
    b.step("checkpoint", lambda t: t.checkpoint())
    if restore_to is not None:
        b.step("restore", lambda t: t.restore(restore_to))
        b.step("append_after_restore", lambda t: t.append(b.gen(9000, 5)))
    if compact:
        b.step("compact_logs", lambda t: t.compact_logs())
    b.snapshot("pre")
    if vacuum:
        b.step(
            "vacuum_lite",
            lambda t: t.vacuum(
                retention_hours=0, dry_run=False, lite=True, enforce_retention_duration=False
            ),
        )
        b.step(
            "vacuum_full",
            lambda t: t.vacuum(retention_hours=0, dry_run=False, enforce_retention_duration=False),
        )
    b.step(
        "zero_log_retention",
        lambda t: t.set_properties(
            {
                "delta.logRetentionDuration": "interval 0 seconds",
                "delta.checkpointRetentionDuration": "interval 0 seconds",
            }
        ),
    )
    b.step("checkpoint_final", lambda t: t.checkpoint())
    b.step("cleanup_metadata", lambda t: t.cleanup_metadata())
    b.snapshot("final")


def full_history(
    b: Builder,
    *,
    props: Mapping[str, str] | None = None,
    partition_by: list[str] | None = None,
    cluster_by: list[str] | None = None,
    dml: bool = True,
    **gen: Any,
) -> None:
    """Appends through every write path, then DML through every engine, then DDL."""
    b.gen = lambda base, n=10: rows(base, n, **gen)
    extra = {"cluster_by": cluster_by} if cluster_by else {}
    b.step(
        "create",
        lambda t: b.conn.write_table(
            b.path, b.gen(0), properties=props, partition_by=partition_by, **extra
        ),
        expect="ok",
    )
    b.step("append", lambda t: t.append(b.gen(100)), expect="ok")
    b.step("append_kernel", lambda t: t.append(b.gen(200)), via="kernel")
    b.step("append_deltars", lambda t: t.append(b.gen(300)), via="deltars")
    b.step("txn_append", lambda t: t.append(b.gen(400, 5), txn=("app1", 1)))
    b.step("txn_append_replay", lambda t: t.append(b.gen(400, 5), txn=("app1", 1)))
    b.step("distributed_write", _distributed([b.gen(500, 6), b.gen(600, 6)]))
    if dml:
        b.step("delete", lambda t: t.delete(f"id IN ({_ends(3)})"))
        b.step(
            "update", lambda t: t.update({"s": "'upd'", "i": "i"}, predicate=f"id IN ({_ends(4)})")
        )
        b.step("update_arith", lambda t: t.update({"i": "i + 1"}, predicate="id % 10 = 4"))
        b.step(
            "update_kernel",
            lambda t: t.update({"s": "'updk'"}, predicate=f"id IN ({_ends(5)})"),
            via="kernel",
        )
        b.step("delete_kernel", lambda t: t.delete("id = 201"), via="kernel")
        b.step("merge", _upsert(b.gen(205, 10)))
        b.step("merge_kernel", _upsert(b.gen(100, 3), {"s": "'mk'"}), via="kernel")
        b.step("merge_deltars", _upsert(b.gen(300, 3), {"s": "'md'"}), via="deltars")
        b.step(
            "replace_where",
            lambda t: t.overwrite(b.gen(600, 4), predicate="id >= 600 AND id < 610"),
        )
    b.step("add_column", lambda t: t.add_column([pa.field("newc", pa.string())]))
    b.mark("add_column")
    b.step(
        "append_newc",
        lambda t: t.append(
            b.gen(700, 5).append_column("newc", pa.array(["n1", None, "", "n4", "n5"]))
        ),
    )
    b.step(
        "set_properties",
        lambda t: t.set_properties({"delta.checkpointInterval": "2", "custom.k": "v"}),
    )
    b.step(
        "append_after_interval",
        lambda t: t.append(b.gen(800, 3).append_column("newc", pa.array(["a", "b", "c"]))),
    )
    b.step(
        "append_after_interval_kernel",
        lambda t: t.append(b.gen(810, 3).append_column("newc", pa.array(["a", "b", "c"]))),
        via="kernel",
    )


# ---------------------------------------------------------------- histories


def h_plain(b: Builder) -> None:
    import deltalake

    full_history(b, long=True)
    b.step(
        "external_deltalake_append",
        lambda t: deltalake.write_deltalake(b.path, _with_newc(b, 900, 4), mode="append"),
        external=True,
    )
    b.step("overwrite_full", lambda t: t.overwrite(t.to_arrow()))
    maintenance(b, restore_to=10)


def h_plain_ancient(b: Builder) -> None:
    full_history(b, ancient=True, long=True)
    maintenance(b, restore_to=10)


def h_partitioned(b: Builder) -> None:
    full_history(b, partition_by=["ps", "pd", "pi", "pts"], part=True)
    b.step(
        "dynamic_partition_overwrite",
        lambda t: t.overwrite(_with_newc(b, 1000, 3), partition_overwrite="dynamic"),
    )
    b.step(
        "replace_where_partition",
        lambda t: t.overwrite(
            _with_newc(b, 1100, 10).filter(pc.equal(pc.field("pi"), 1)), predicate="pi = 1"
        ),
    )
    b.step("delete_partition", lambda t: t.delete("ps = 'x/y=z'"))
    b.step("delete_null_partition", lambda t: t.delete("pd IS NULL AND id > 500"))
    maintenance(b, restore_to=12)


def h_dv(b: Builder) -> None:
    full_history(b, props={"delta.enableDeletionVectors": "true"}, ancient=True)
    maintenance(b, restore_to=9)


def h_dv_partitioned(b: Builder) -> None:
    full_history(
        b,
        props={"delta.enableDeletionVectors": "true"},
        partition_by=["ps", "pi"],
        part=True,
        ancient=True,
    )
    maintenance(b, restore_to=9)


def h_cdf(b: Builder) -> None:
    full_history(b, props={"delta.enableChangeDataFeed": "true"})
    maintenance(b, vacuum=False)


def h_dv_cdf(b: Builder) -> None:
    full_history(
        b, props={"delta.enableChangeDataFeed": "true", "delta.enableDeletionVectors": "true"}
    )
    maintenance(b, vacuum=False)


def _column_mapping(b: Builder, mode: str) -> None:
    full_history(b, props={"delta.columnMapping.mode": mode}, partition_by=["pi"], part=True)
    b.step("rename_column", lambda t: t.rename_column("s", "s renamed"))
    b.step("rename_partition_column", lambda t: t.rename_column("pi", "pi2"))
    b.step("rename_nested", lambda t: t.rename_column(["st", "b"], "b2"))
    b.step("drop_column", lambda t: t.drop_column("dec"))
    b.step("drop_nested", lambda t: t.drop_column(["st", "c"]))
    b.step("readd_column", lambda t: t.add_column([pa.field("dec", pa.string())]))
    b.step(
        "add_column_odd_name",
        lambda t: t.add_column([pa.field("col with space,;{}()", pa.int32())]),
    )

    def append_current(t: Any) -> Any:
        data = t.to_arrow().slice(0, 5)
        data = data.set_column(
            data.schema.get_field_index("id"), "id", pa.array(range(1200, 1205), pa.int64())
        )
        data = data.set_column(
            data.schema.get_field_index("dec"), "dec", pa.array(["re", None, "x", "y", "z"])
        )
        return t.append(data)

    b.step("append_after_schema_change", append_current)
    b.step("append_after_schema_change_kernel", append_current, via="kernel")
    b.step(
        "update_renamed",
        lambda t: t.update({"`s renamed`": "'r'"}, predicate=f"id IN ({_ends(1)})"),
    )
    b.step("delete_readded", lambda t: t.delete("dec = 're'"))
    maintenance(b)


def h_cm_name(b: Builder) -> None:
    _column_mapping(b, "name")


def h_cm_id(b: Builder) -> None:
    _column_mapping(b, "id")


def h_row_tracking(b: Builder) -> None:
    full_history(
        b,
        props={"delta.enableRowTracking": "true", "delta.enableDeletionVectors": "true"},
        ancient=True,
    )
    maintenance(b, restore_to=9)


def h_ict(b: Builder) -> None:
    full_history(
        b,
        props={"delta.enableInCommitTimestamps": "true", "delta.enableChangeDataFeed": "true"},
    )
    maintenance(b, restore_to=8)


def _widen(column: str, to: str) -> Callable[[Any], Any]:
    return lambda t: t.alter_column_type(column, to)


def h_type_widening(b: Builder) -> None:
    extra: dict[str, Callable[[list[int], list[int], list[Any]], Any]] = {
        "w_i": lambda ids, k, dates: pa.array([i % 100 for i in ids], pa.int8()),
        "w_s": lambda ids, k, dates: pa.array(ids, pa.int16()),
        "w_int": lambda ids, k, dates: pa.array([i * 1000 for i in ids], pa.int32()),
        "w_f": lambda ids, k, dates: pa.array([i / 3 for i in ids], pa.float32()),
        "w_dec": lambda ids, k, dates: pa.array([D(i) / 4 for i in ids], pa.decimal128(10, 2)),
        "w_d": lambda ids, k, dates: pa.array([dates[x] for x in k], pa.date32()),
        "w_i2d": lambda ids, k, dates: pa.array(ids, pa.int32()),
    }
    b.gen = lambda base, n=10: rows(base, n, nested=False, extra=extra)
    b.step(
        "create",
        lambda t: b.conn.write_table(
            b.path, b.gen(0), properties={"delta.enableTypeWidening": "true"}
        ),
        expect="ok",
    )
    b.step("append", lambda t: t.append(b.gen(100)), expect="ok")
    for col, ty in [
        ("w_i", "smallint"),
        ("w_s", "int"),
        ("w_int", "bigint"),
        ("w_f", "double"),
        ("w_dec", "decimal(20,4)"),
        ("w_d", "timestamp_ntz"),
        ("w_i2d", "double"),
        ("w_i", "bigint"),
    ]:
        b.step(f"widen_{col}_{ty}", _widen(col, ty))

    def wide(base: int, n: int = 10) -> pa.Table:
        target = b.conn.open_table(b.path).schema()
        data = b.gen(base, n)
        arrays = []
        for name in data.column_names:
            col = data.column(name)
            ty = target.field(name).type if name in target.names else col.type
            if pa.types.is_timestamp(ty) and pa.types.is_date(col.type):
                col = pc.cast(pc.cast(col, pa.timestamp("s")), ty)
            else:
                col = col.cast(ty)
            arrays.append(col)
        return pa.table(arrays, names=data.column_names)

    b.step("append_wide", lambda t: t.append(wide(200)))
    b.step("append_wide_kernel", lambda t: t.append(wide(300)), via="kernel")
    b.step("append_wide_deltars", lambda t: t.append(wide(400)), via="deltars")
    b.step("update_wide", lambda t: t.update({"w_int": "6000000000"}, predicate="id < 5"))
    b.step("merge_wide", _upsert(wide(405, 10)))
    b.step("delete_wide", lambda t: t.delete("w_d < TIMESTAMP_NTZ'1600-01-01 00:00:00'"))
    b.step("distributed_wide", _distributed([wide(500, 4)]))
    maintenance(b)


def h_ntz(b: Builder) -> None:
    full_history(
        b,
        ntz=True,
        partition_by=["n"],
        nested=False,
        ancient=True,
        props={"delta.enableDeletionVectors": "true"},
    )
    maintenance(b, restore_to=5)


def h_types(b: Builder) -> None:
    full_history(
        b,
        long=True,
        ancient=True,
        props={"delta.dataSkippingNumIndexedCols": "40", "delta.enableDeletionVectors": "true"},
    )
    maintenance(b)


def h_generated(b: Builder) -> None:
    b.gen = lambda base, n=10: rows(base, n, nested=False).select(["id", "s", "ts", "d", "dec"])
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("s", pa.string()),
            pa.field("ts", pa.timestamp("us", "UTC")),
            pa.field("d", pa.date32()),
            pa.field("dec", pa.decimal128(38, 18)),
            pa.field("g", pa.int64(), metadata={"delta.generationExpression": "id * 2"}),
            pa.field(
                "gd", pa.date32(), metadata={"delta.generationExpression": "CAST(ts AS DATE)"}
            ),
        ]
    )
    b.step(
        "create", lambda t: b.conn.create_table(b.path, schema, partition_by=["gd"]), expect="ok"
    )
    b.step("append", lambda t: t.append(b.gen(0)), expect="ok")
    b.step("append_kernel", lambda t: t.append(b.gen(100)), via="kernel")
    b.step("append_deltars", lambda t: t.append(b.gen(200)), via="deltars")
    b.step("update", lambda t: t.update({"id": "id + 1000"}, predicate="id % 10 = 4"))
    b.step("delete", lambda t: t.delete("gd < DATE'1600-01-01'"))
    b.step("merge", _upsert(b.gen(205, 10)))
    b.step(
        "replace_where", lambda t: t.overwrite(b.gen(300, 3), predicate="id >= 300 AND id < 303")
    )
    b.step("add_constraint", lambda t: t.add_constraint({"idpos": "id >= 0"}))
    b.step("violating_append", lambda t: t.append(b.gen(-5, 2)), expect="refused")
    b.step("distributed_write", _distributed([b.gen(400, 3)]))
    maintenance(b, restore_to=3)


def h_constraints(b: Builder) -> None:
    b.gen = lambda base, n=10: rows(base, n)
    b.step("create", lambda t: b.conn.write_table(b.path, b.gen(0)), expect="ok")
    b.step(
        "add_constraint",
        lambda t: t.add_constraint(
            {"idpos": "id >= 0", "ilim": "i IS NULL OR i BETWEEN -1000 AND 1000"}
        ),
        expect="ok",
    )
    b.step("set_not_null", lambda t: t.set_not_null("id"), expect="ok")
    b.step("append", lambda t: t.append(b.gen(100)), expect="ok")
    b.step("append_kernel", lambda t: t.append(b.gen(200)), via="kernel")
    b.step("violating_append", lambda t: t.append(b.gen(-5, 2)), expect="refused")
    b.step(
        "violating_update", lambda t: t.update({"i": "5000"}, predicate="id = 1"), expect="refused"
    )
    b.step("update", lambda t: t.update({"i": "7"}, predicate="id % 10 = 2"))
    b.step("merge", _upsert(b.gen(205, 10)))
    b.step("delete", lambda t: t.delete("id % 10 = 1"))
    maintenance(b, restore_to=4)


def h_v2_checkpoints(b: Builder) -> None:
    full_history(
        b,
        props={
            "delta.checkpointPolicy": "v2",
            "delta.checkpointInterval": "2",
            "delta.enableDeletionVectors": "true",
            "delta.enableRowTracking": "true",
        },
        ancient=True,
    )
    maintenance(b)


def h_liquid(b: Builder) -> None:
    full_history(b, cluster_by=["id", "d"])
    b.step("cluster_by_change", lambda t: t.cluster_by(["s"]))
    b.step("append_after_cluster_change", lambda t: t.append(_with_newc(b, 1300, 5)))
    maintenance(b, zorder=None)


def h_log_compaction(b: Builder) -> None:
    full_history(
        b,
        props={"delta.checkpointInterval": "2", "delta.enableDeletionVectors": "true"},
        ancient=True,
    )
    b.step("compact_3_6", lambda t: t.compact_logs(3, 6))
    b.step("compact_7_12", lambda t: t.compact_logs(7, 12))
    maintenance(b, compact=False)


def h_legacy_upgrade(b: Builder) -> None:
    """A (1,2) table plain delta-rs created (no Spark footer), upgraded by deltaswamp."""
    import deltalake

    b.gen = lambda base, n=10: rows(base, n)
    b.step("deltalake_create", lambda t: deltalake.write_deltalake(b.path, b.gen(0)), external=True)
    b.step(
        "deltalake_append",
        lambda t: deltalake.write_deltalake(b.path, b.gen(100), mode="append"),
        external=True,
    )
    b.step("append", lambda t: t.append(b.gen(200)), expect="ok")
    b.step("append_kernel", lambda t: t.append(b.gen(300)), via="kernel")
    b.step("delete", lambda t: t.delete("id % 10 = 3"))
    b.step("update", lambda t: t.update({"s": "'u'"}, predicate="id % 10 = 4"))
    b.step("merge", _upsert(b.gen(305, 10)))
    b.step("add_feature_dv", lambda t: t.add_feature("deletionVectors"))
    b.step("enable_dv", lambda t: t.set_properties({"delta.enableDeletionVectors": "true"}))
    b.step("delete_dv", lambda t: t.delete("id % 10 = 5"))
    b.step("add_feature_ntz", lambda t: t.add_feature("timestampNtz"))
    maintenance(b, restore_to=2)


def h_legacy_cdf(b: Builder) -> None:
    """(1,4): the change feed enabled on a legacy table through set_properties."""
    b.gen = lambda base, n=10: rows(base, n)
    b.step("create", lambda t: b.conn.write_table(b.path, b.gen(0)), expect="ok")
    b.step(
        "enable_cdf",
        lambda t: t.set_properties({"delta.enableChangeDataFeed": "true"}),
        expect="ok",
    )
    b.step("append", lambda t: t.append(b.gen(100)), expect="ok")
    b.step("delete", lambda t: t.delete("id % 10 = 3"))
    b.step("update", lambda t: t.update({"s": "'u'"}, predicate="id % 10 = 4"))
    b.step("merge", _upsert(b.gen(105, 10)))
    b.step("append_kernel", lambda t: t.append(b.gen(300)), via="kernel")
    b.step(
        "update_kernel", lambda t: t.update({"s": "'uk'"}, predicate="id % 10 = 6"), via="kernel"
    )
    maintenance(b, vacuum=False)


def h_everything(b: Builder) -> None:
    full_history(
        b,
        props={
            "delta.enableDeletionVectors": "true",
            "delta.enableChangeDataFeed": "true",
            "delta.columnMapping.mode": "name",
            "delta.enableRowTracking": "true",
            "delta.enableInCommitTimestamps": "true",
            "delta.checkpointPolicy": "v2",
            "delta.enableTypeWidening": "true",
        },
        partition_by=["ps", "pd"],
        part=True,
        ntz=True,
        ancient=True,
    )
    b.step("rename", lambda t: t.rename_column("s", "s2"))
    b.step("widen", lambda t: t.alter_column_type("i", "bigint"))
    b.step("append_after", lambda t: t.append(t.to_arrow().slice(0, 4)))
    maintenance(b)


def _variant_rows(base: int, n: int = 10, *, null_elements: bool = True) -> pa.Table:
    ids = list(range(base, base + n))

    def js(i: int) -> str | None:
        if i % 10 == 3:
            return None
        return json.dumps(
            {
                "k": i,
                "s": "x" * (i % 5),
                "arr": [i, None, {"z": 1.5}],
                "d": "2020-01-01",
                "big": 12345678901234567890 if i % 2 else -1,
            }
        )

    def elems(i: int) -> list[str | None]:
        return [js(i), None, "[1,2]"] if null_elements else [js(i) or "null", "[1,2]"]

    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "v": pa.array([js(i) for i in ids], pa.string()),
            "sv": pa.array(
                [None if i % 10 == 4 else {"x": js(i + 1), "n": i} for i in ids],
                pa.struct([("x", pa.string()), ("n", pa.int32())]),
            ),
            "av": pa.array([None if i % 10 == 5 else elems(i) for i in ids], pa.list_(pa.string())),
            "mv": pa.array(
                [None if i % 10 == 6 else [("a", js(i)), ("b", "true")] for i in ids],
                pa.map_(pa.string(), pa.string()),
            ),
            "s": pa.array([f"s{i}" for i in ids], pa.string()),
        }
    )


def h_variant(b: Builder) -> None:
    """A VARIANT table Databricks creates (deltaswamp cannot), then deltaswamp's history."""
    b.gen = _variant_rows
    assert b.ctx is not None, "the variant history needs the warehouse to create its table"
    ctx = b.ctx
    rel = f"{ctx.case_dir(b)}/databricks_base"

    def create(t: Any) -> Any:
        ref = f"delta.`{ctx.staging.uri(rel)}`"
        ctx.warehouse.run(
            f"CREATE TABLE {ref} (id BIGINT, v VARIANT, sv STRUCT<x: VARIANT, n: INT>, "
            "av ARRAY<VARIANT>, mv MAP<STRING, VARIANT>, s STRING) TBLPROPERTIES "
            "('delta.enableDeletionVectors' = 'true', 'delta.enableVariantShredding' = 'false')"
        )
        ctx.warehouse.run(
            f"INSERT INTO {ref} SELECT id, parse_json(to_json(named_struct('k', id, 'q', 'dbx'))), "
            "named_struct('x', parse_json('[1,2]'), 'n', 1), array(parse_json('1'), NULL), "
            "map('a', parse_json('{\"z\": null}')), 'dbx' FROM range(-5, 0)"
        )
        shutil.rmtree(b.path, ignore_errors=True)
        ctx.staging.download_dir(rel, Path(b.path))

    b.step("databricks_create", create, expect="ok", external=True)
    b.step("append", lambda t: t.append(_variant_rows(0)), expect="ok")
    b.step("append_kernel", lambda t: t.append(_variant_rows(100)), via="kernel")
    b.step("append_deltars", lambda t: t.append(_variant_rows(200)), via="deltars")
    b.step("txn_append", lambda t: t.append(_variant_rows(400, 5), txn=("app1", 1)))
    b.step("distributed_write", _distributed([_variant_rows(500, 6), _variant_rows(600, 6)]))
    b.step("delete", lambda t: t.delete("id % 10 = 1"))
    b.step("update", lambda t: t.update({"s": "'upd'"}, predicate="id % 10 = 2"))
    # Null array elements in a MERGE source are a known failure with its own
    # test (test_variant_merge_null_element); the history merges without them.
    b.step("merge", _upsert(_variant_rows(205, 10, null_elements=False)))
    b.step(
        "replace_where",
        lambda t: t.overwrite(_variant_rows(600, 4), predicate="id >= 600 AND id < 610"),
    )
    b.step("add_column", lambda t: t.add_column([pa.field("newc", pa.string())]))
    b.step("set_properties", lambda t: t.set_properties({"delta.checkpointInterval": "2"}))
    b.step(
        "append_after",
        lambda t: t.append(_variant_rows(800, 3).append_column("newc", pa.array(["a", None, "c"]))),
    )
    maintenance(b, restore_to=3)


def h_dv_decimal_stats(b: Builder) -> None:
    """Kernel DV DML on a kernel-written file whose decimal bounds need all 38 digits."""
    data = pa.table(
        {
            "id": pa.array(range(1, 7), pa.int64()),
            "big": pa.array(
                [9007199254740993, 9007199254740995, -9007199254740995, 1, 2, 3], pa.int64()
            ),
            "dec": pa.array(
                [
                    D("12345678901234567890.123456789012345678"),
                    D("0.1"),
                    D("-12345678901234567890.123456789012345678"),
                    D("1"),
                    D("2"),
                    D("3"),
                ],
                pa.decimal128(38, 18),
            ),
            "s": pa.array(["a", "b", "c", "d", "e", "f"]),
        }
    )
    b.gen = lambda base, n=10: data
    b.step(
        "create_empty",
        lambda t: b.conn.write_table(
            b.path, data.slice(0, 0), properties={"delta.enableDeletionVectors": "true"}
        ),
        expect="ok",
    )
    b.step("append_kernel", lambda t: t.append(data), via="kernel", expect="ok")
    b.step("delete", lambda t: t.delete("id = 2"), expect="ok")
    b.step("update", lambda t: t.update({"s": "'u'"}, predicate="id = 4"), expect="ok")
    b.step(
        "merge_delete",
        lambda t: (
            t.merge(pa.table({"id": pa.array([5], pa.int64())}), "target.id = source.id")
            .when_matched_delete()
            .execute()
        ),
        expect="ok",
    )
    b.mark("stats")
    b.snapshot("pre")
    b.snapshot("final")


def h_nested_timestamps(b: Builder) -> None:
    """ARRAY<TIMESTAMP>, alone and inside a struct: Databricks writes these as INT96."""
    b.gen = lambda base, n=10: rows(base, n, nested_ts=True, nested=False)
    b.step("create", lambda t: b.conn.write_table(b.path, b.gen(0)), expect="ok")
    b.step("append", lambda t: t.append(b.gen(100)), expect="ok")
    b.step("append_kernel", lambda t: t.append(b.gen(200)), via="kernel")
    b.step("delete", lambda t: t.delete("id % 10 = 3"))
    maintenance(b, zorder=None, vacuum=False)


def h_early_write_paths(b: Builder) -> None:
    """Early dates and timestamps through every write path (the footer round's harness)."""
    days = [
        dt.date(1, 1, 1),
        dt.date(1000, 2, 28),
        dt.date(1500, 6, 15),
        dt.date(1582, 10, 4),
        dt.date(1582, 10, 15),
        dt.date(1700, 3, 1),
        dt.date(1850, 3, 1),
        dt.date(1899, 12, 31),
        dt.date(1900, 1, 1),
        dt.date(2024, 1, 1),
    ]

    def gen(base: int, n: int = 10, shift: int = 0) -> pa.Table:
        d = days[:n]
        ts = [dt.datetime(x.year, x.month, x.day, 12, 34, 56, 123456) for x in d]
        ids = [base + i for i in range(len(d))]
        return pa.table(
            {
                "id": pa.array(ids, pa.int64()),
                "d": pa.array(d, pa.date32()),
                "ts": pa.array([t.replace(tzinfo=UTC) for t in ts], pa.timestamp("us", "UTC")),
                "ntz": pa.array(ts, pa.timestamp("us")),
                "st": pa.array([{"d": x} for x in d], pa.struct([("d", pa.date32())])),
                "arr": pa.array([[x, None] for x in d], pa.list_(pa.date32())),
                "s": pa.array([f"v{i + shift}" for i in ids]),
            }
        )

    b.gen = gen
    b.step("create", lambda t: b.conn.write_table(b.path, gen(0)), expect="ok")
    b.step("append", lambda t: t.append(gen(100)), expect="ok")
    b.step("append_late_only", lambda t: t.append(gen(200).slice(9)), expect="ok")
    b.step("replace_where", lambda t: t.overwrite(gen(0, 5, shift=7), predicate="id < 5"))
    b.step("delete", lambda t: t.delete("id IN (1, 102)"))
    b.step("update", lambda t: t.update({"s": "'upd'"}, predicate="id IN (2, 103)"))
    b.step("merge", _upsert(gen(500).slice(0, 4)))
    b.step("distributed_write", _distributed([gen(600), gen(700)]))
    b.step("add_feature_dv", lambda t: t.set_properties({"delta.enableDeletionVectors": "true"}))
    b.step("delete_dv", lambda t: t.delete("id IN (3, 104)"))
    b.step("update_dv", lambda t: t.update({"s": "'upd2'"}, predicate="id IN (4, 105)"))
    b.step("merge_dv", _upsert(gen(0).slice(0, 3)))
    maintenance(b, restore_to=None)


# ------------------------------------------ Ray Data write paths (docs/ray-data.md)

#: The flat columns the Ray-path histories write: every stats group but the
#: float, binary and nested ones, which those histories do not exercise.
_FLAT = ["id", "s", "d", "ts", "dec", "i"]
_FLAT_STATS = ("integer", "decimal", "string", "date", "timestamp")
_FLAT_SQL = "id BIGINT, s STRING, d DATE, ts TIMESTAMP, dec DECIMAL(38,18), i INT"


def _flat(base: int, n: int = 10) -> pa.Table:
    return rows(base, n, nested=False).select(_FLAT)


def _as_worker(plan: Any) -> Any:
    """The plan as a Ray worker receives it: pickled, with nothing of the driver."""
    import pickle

    return pickle.loads(pickle.dumps(plan))


def _distributed_merged(parts: list[pa.Table], **plan: Any) -> Callable[[Any], Any]:
    """Workers write `parts` (one task each), the first two fragments are merged
    on a worker, and the driver commits what arrives."""
    from deltaswamp.distributed import merge_fragments

    def go(t: Any) -> Any:
        p = t.plan_write(**plan)
        indexed = "identity_tasks" in plan
        frags = [
            _as_worker(p).write(part, **({"task_index": i} if indexed else {}))
            for i, part in enumerate(parts)
        ]
        return p.commit([merge_fragments(frags[:2]), *frags[2:]])

    return go


def _databricks_base(b: Builder, columns: str, props: str, statements: list[str]) -> None:
    """Databricks creates the table (a shape deltaswamp does not create) under
    the case's volume directory, runs `statements` on it, and the history
    starts from a local copy."""
    assert b.ctx is not None, "this history needs the warehouse to create its table"
    ctx = b.ctx
    rel = f"{ctx.case_dir(b)}/databricks_base"

    def create(t: Any) -> Any:
        ref = f"delta.`{ctx.staging.uri(rel)}`"
        ctx.warehouse.run(f"CREATE TABLE {ref} ({columns}) TBLPROPERTIES ({props})")
        for sql in statements:
            ctx.warehouse.run(sql.format(ref=ref))
        shutil.rmtree(b.path, ignore_errors=True)
        ctx.staging.download_dir(rel, Path(b.path))

    b.step("databricks_create", create, expect="ok", external=True)


def _duplicates(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    dup = []
    for v in values:
        if v in seen:
            dup.append(v)
        seen.add(v)
    return dup


def h_identity(b: Builder) -> None:
    """Identity columns Databricks declares, written locally and by a distributed
    plan that reserves a block per task; Databricks then inserts on top."""
    b.gen = _flat
    _databricks_base(
        b,
        "idn BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1), "
        "idd BIGINT GENERATED BY DEFAULT AS IDENTITY (START WITH 1000 INCREMENT BY 10), "
        + _FLAT_SQL,
        "'delta.enableDeletionVectors' = 'true'",
        [
            "INSERT INTO {ref} (" + ", ".join(_FLAT) + ") SELECT id, 'dbx', DATE'2020-01-01', "
            "TIMESTAMP'2020-01-01 00:00:00', 1.5, CAST(id AS INT) FROM range(-5, 0)"
        ],
    )
    b.step("append", lambda t: t.append(b.gen(0)), expect="ok")
    b.step("append_kernel", lambda t: t.append(b.gen(100)), via="kernel", expect="ok")
    # BY DEFAULT keeps a given value; negative, so no generated value meets it.
    b.step(
        "append_by_default_given",
        lambda t: t.append(
            b.gen(50, 3).append_column("idd", pa.array([-11, -12, -13], pa.int64()))
        ),
        expect="ok",
    )
    b.step(
        "given_always_value",
        lambda t: t.append(b.gen(60, 1).append_column("idn", pa.array([5], pa.int64()))),
        expect="refused",
    )
    b.step(
        "distributed_identity",
        _distributed_merged(
            [b.gen(500, 6), b.gen(600, 6), b.gen(700, 4)],
            identity_tasks=3,
            identity_rows_per_task=100,
        ),
        expect="ok",
    )
    b.step("delete", lambda t: t.delete("id % 10 = 1"))
    b.step("update", lambda t: t.update({"s": "'upd'"}, predicate="id % 10 = 2"))
    b.step("append_after_dml", lambda t: t.append(b.gen(800, 5)), expect="ok")
    maintenance(b, restore_to=None)


def _identity_problems(values: dict[str, list[Any]], where: str) -> list[str]:
    out = []
    for col, got in values.items():
        if any(v is None for v in got):
            out.append(f"{where}: {col} has NULLs")
        dup = _duplicates(v for v in got if v is not None)
        if dup:
            out.append(f"{where}: {col} repeats {sorted(set(dup))[:10]}")
    return out


def c_identity(ctx: Context, b: Builder, rel: str, uri: str) -> list[str]:
    """Identity values stay unique across deltaswamp's writes and Databricks',
    including a distributed write on what Databricks left and an INSERT by
    Databricks after it (which only a covering high-water mark keeps apart)."""
    cols = ["idn", "idd"]
    problems = _identity_problems(
        {c: _read(b.root / "pre").column(c).to_pylist() for c in cols}, "deltaswamp's table"
    )
    local = b.root / "identity_check"
    shutil.rmtree(local, ignore_errors=True)
    ctx.staging.download_dir(rel, local)
    before = _digests(local)
    t = _open(local)
    plan = t.plan_write(identity_tasks=2, identity_rows_per_task=10)
    plan.commit([_as_worker(plan).write(_flat(9100 + 10 * i, 5), task_index=i) for i in range(2)])
    ctx.staging.upload_dir(
        local, rel, only=[p for p, h in _digests(local).items() if before.get(p) != h]
    )
    ref = dref(uri)
    sel = ", ".join(_FLAT)
    ctx.warehouse.run(
        f"INSERT INTO {ref} ({sel}) "
        f"SELECT id + 20000, s, d, ts, dec, i FROM {ref} WHERE id % 10 = 3"
    )
    n, dn, dd, nulls = (
        int(x)
        for x in ctx.warehouse.rows(
            f"SELECT count(*), count(DISTINCT idn), count(DISTINCT idd), "
            f"count_if(idn IS NULL OR idd IS NULL) FROM {ref}"
        )[0]
    )
    if not (n == dn == dd) or nulls:
        problems.append(
            f"Databricks sees {n} rows, {dn} distinct idn, {dd} distinct idd, {nulls} NULLs"
        )
    shutil.rmtree(local, ignore_errors=True)
    ctx.staging.download_dir(rel, local)
    problems += _identity_problems(
        {c: _read(local).column(c).to_pylist() for c in cols}, "deltaswamp reading Databricks'"
    )
    diff = compare_many(ctx, BY_ID["identity-databricks_created"], [("x", ref, _read(local), ())])
    if not diff["x"].ok:
        problems.append(diff["x"].explain())
    return problems


def h_checkpoint_protection(b: Builder) -> None:
    """A table Databricks left with checkpointProtection, as dropping a feature
    leaves it (Delta 4.0), written by every deltaswamp path."""
    b.gen = _flat
    _databricks_base(
        b,
        _FLAT_SQL,
        "'delta.enableDeletionVectors' = 'true'",
        [
            "INSERT INTO {ref} SELECT id, 'dbx', DATE'2020-01-01', TIMESTAMP'2020-01-01 00:00:00', "
            "1.5, CAST(id AS INT) FROM range(-10, 0)",
            "DELETE FROM {ref} WHERE id = -3",
            "ALTER TABLE {ref} DROP FEATURE deletionVectors",
        ],
    )

    def has_it(t: Any) -> None:
        if "checkpointProtection" not in t.features():
            raise AssertionError(f"Databricks left no checkpointProtection: {t.features()}")

    b.step("has_checkpoint_protection", has_it, expect="ok")
    b.step("append", lambda t: t.append(b.gen(0)), expect="ok")
    b.step("append_kernel", lambda t: t.append(b.gen(100)), via="kernel", expect="ok")
    b.step(
        "distributed_write",
        _distributed_merged([b.gen(500, 6), b.gen(600, 6), b.gen(700, 3)]),
        expect="ok",
    )
    b.step("add_column", lambda t: t.add_column([pa.field("newc", pa.string())]), expect="ok")
    b.step("delete", lambda t: t.delete("id % 10 = 1"))
    b.step("update", lambda t: t.update({"s": "'upd'"}, predicate="id % 10 = 2"))
    b.step("merge", _upsert(b.gen(205, 10)))
    b.step("checkpoint_mid", lambda t: t.checkpoint(), expect="ok")
    b.step(
        "overwrite_distributed",
        _distributed_merged([b.gen(900, 4), b.gen(950, 4)], mode="overwrite"),
        expect="ok",
    )
    b.step("append_after", lambda t: t.append(b.gen(1000, 5)), expect="ok")
    maintenance(b, restore_to=None)


def c_checkpoint_protection(ctx: Context, b: Builder, rel: str, uri: str) -> list[str]:
    """Log cleanup is refused (the feature's one rule); every other step committed."""
    steps = {s.name: s for s in b.steps}
    problems = []
    cleanup = steps["cleanup_metadata"]
    if cleanup.outcome != "refused":
        problems.append(f"cleanup_metadata was {cleanup.outcome}, not refused: {cleanup.detail}")
    final = b.root / "final"
    if not (final / "_delta_log" / f"{0:020d}.json").exists():
        problems.append("version 0's commit file is gone from the final table")
    if "checkpointProtection" not in _open(final).features():
        problems.append("the final table lost checkpointProtection")
    return problems


def h_domain_metadata(b: Builder) -> None:
    """User domain metadata set by distributed commits."""
    b.gen = _flat
    b.step(
        "create",
        lambda t: b.conn.create_table(
            b.path,
            b.gen(0, 1).schema,
            properties={
                "delta.feature.domainMetadata": "supported",
                "delta.enableDeletionVectors": "true",
            },
        ),
        expect="ok",
    )
    b.step("append", lambda t: t.append(b.gen(0)), expect="ok")
    b.step(
        "distributed_domain",
        _distributed_merged(
            [b.gen(100, 5), b.gen(200, 5)], domain_metadata={"myapp.x": '{"run": 1}'}
        ),
        expect="ok",
    )
    b.step("append_kernel", lambda t: t.append(b.gen(300)), via="kernel", expect="ok")
    b.step(
        "distributed_domain_again",
        _distributed_merged(
            [b.gen(400, 5), b.gen(500, 5), b.gen(600, 5)],
            domain_metadata={"myapp.x": '{"run": 2}', "myapp.y": "y"},
        ),
        expect="ok",
    )
    b.step(
        "system_domain",
        lambda t: t.plan_write(domain_metadata={"delta.rowTracking": "{}"}),
        expect="refused",
    )
    b.step("delete", lambda t: t.delete("id % 10 = 1"))
    b.step("update", lambda t: t.update({"s": "'upd'"}, predicate="id % 10 = 2"))
    maintenance(b)


def c_domain_metadata(ctx: Context, b: Builder, rel: str, uri: str) -> list[str]:
    """The domains survive checkpoints, metadata cleanup and Databricks' writes."""
    from deltaswamp.engine.kernel import KernelEngine

    want = {"myapp.x": '{"run": 2}', "myapp.y": "y"}
    problems = []
    for tag in ("pre", "final", "databricks"):
        snapshot = KernelEngine().snapshot(_open(b.root / tag)._enrich())
        got = {d: snapshot.domain_metadata(d) for d in want}
        if got != want:
            problems.append(f"{tag}: domains {got}, expected {want}")
    return problems


def h_cdf_overwrite(b: Builder) -> None:
    """Distributed overwrites of a change-feed table, read back as changes."""
    b.gen = _flat
    b.step(
        "create",
        lambda t: b.conn.create_table(
            b.path, b.gen(0, 1).schema, properties={"delta.enableChangeDataFeed": "true"}
        ),
        expect="ok",
    )
    b.step("append", lambda t: t.append(b.gen(0)), expect="ok")
    b.step("append_kernel", lambda t: t.append(b.gen(100)), via="kernel", expect="ok")
    b.step(
        "overwrite_distributed",
        _distributed_merged([b.gen(200, 5), b.gen(300, 5), b.gen(400, 5)], mode="overwrite"),
        expect="ok",
    )
    b.step("append_distributed", _distributed_merged([b.gen(500, 5), b.gen(600, 5)]), expect="ok")
    b.step("delete", lambda t: t.delete("id % 10 = 1"))
    b.step(
        "overwrite_distributed_again",
        _distributed_merged([b.gen(700, 6), b.gen(800, 6)], mode="overwrite"),
        expect="ok",
    )
    b.step("append_after", lambda t: t.append(b.gen(900, 5)), expect="ok")
    maintenance(b, vacuum=False)


def c_planned_changes(ctx: Context, b: Builder, rel: str, uri: str) -> list[str]:
    """plan_changes read on workers is the feed cdf() returns (which the cdf-*
    checks hold to Databricks' table_changes)."""
    pre = b.root / "pre"
    t = _open(pre)
    first = next(v for v in range(int(t.version) + 1) if _cdf_on(pre, v))
    plan = t.plan_changes(first)
    worker = _as_worker(plan)
    parts = [worker.read(p) for p in plan.partitions(3)]
    got = pa.concat_tables([p for p in parts if p.num_rows]) if parts else None
    want = t.cdf(starting_version=first)
    want = want if isinstance(want, pa.Table) else pa.table(want)

    def key(r: dict[str, Any]) -> tuple[Any, ...]:
        return (r["_commit_version"], r["_change_type"], r["id"], r["s"])

    g = sorted(map(key, got.to_pylist())) if got is not None else []
    w = sorted(map(key, want.to_pylist()))
    return [] if g == w else [f"plan_changes read {len(g)} changes, cdf() {len(w)}"]


def h_create_by_plan(b: Builder) -> None:
    """A table a distributed write creates: nothing exists until its commit."""
    b.gen = _flat

    def create(t: Any) -> Any:
        from deltaswamp.distributed import merge_fragments

        plan = b.conn.plan_write(
            b.path,
            schema=b.gen(0, 1).schema,
            mode="error",
            properties={"delta.enableDeletionVectors": "true"},
        )
        frags = [_as_worker(plan).write(b.gen(base, 6)) for base in (0, 100, 200)]
        if Path(b.path, "_delta_log").exists() and any(Path(b.path, "_delta_log").iterdir()):
            raise AssertionError("the table's log exists before the commit")
        return plan.commit([merge_fragments(frags[:2]), frags[2]])

    b.step("create_by_plan", create, expect="ok")

    def append_by_plan(t: Any) -> Any:
        plan = b.conn.plan_write(b.path, schema=b.gen(0, 1).schema, mode="append")
        return plan.commit([_as_worker(plan).write(b.gen(300, 5))])

    b.step("append_by_plan", append_by_plan, expect="ok")
    b.step(
        "error_mode_on_existing",
        lambda t: b.conn.plan_write(b.path, schema=b.gen(0, 1).schema, mode="error"),
        expect="refused",
    )
    b.step("append", lambda t: t.append(b.gen(400)), expect="ok")
    b.step("delete", lambda t: t.delete("id % 10 = 1"))
    b.step("update", lambda t: t.update({"s": "'upd'"}, predicate="id % 10 = 2"))
    b.step("merge", _upsert(b.gen(205, 10)))
    maintenance(b, restore_to=2)


def _write_log(path: str, protocol: dict[str, Any], fields: list[Any], **config: str) -> None:
    """Version 0 of a table as another writer leaves it (a handwritten log)."""
    log = Path(path, "_delta_log")
    log.mkdir(parents=True)
    actions = [
        {"commitInfo": {"timestamp": 1, "operation": "CREATE TABLE"}},
        {"protocol": protocol},
        {
            "metaData": {
                "id": "4d7f1a60-0000-4000-8000-00000000c0de",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps({"type": "struct", "fields": fields}),
                "partitionColumns": [],
                "configuration": config,
                "createdTime": 1,
            }
        },
    ]
    (log / f"{0:020d}.json").write_text("\n".join(json.dumps(a) for a in actions) + "\n")


def h_generated_defaults(b: Builder) -> None:
    """Generated columns, an invariant, a CHECK constraint and a column
    DEFAULT, computed and checked by kernel workers of a distributed write."""
    b.gen = _flat

    def create(t: Any) -> Any:
        def f(name: str, ty: str, **meta: Any) -> dict[str, Any]:
            return {"name": name, "type": ty, "nullable": True, "metadata": meta}

        _write_log(
            b.path,
            {
                "minReaderVersion": 1,
                "minWriterVersion": 7,
                "writerFeatures": [
                    "invariants",
                    "checkConstraints",
                    "generatedColumns",
                    "allowColumnDefaults",
                ],
            },
            [
                f(
                    "id",
                    "long",
                    **{"delta.invariants": json.dumps({"expression": {"expression": "id >= 0"}})},
                ),
                f("s", "string"),
                f("d", "date"),
                f("ts", "timestamp"),
                f("dec", "decimal(38,18)"),
                f("i", "integer"),
                f("g", "long", **{"delta.generationExpression": "id * 2"}),
                f("gs", "string", **{"delta.generationExpression": "concat('g', s)"}),
                f("dflt", "string", CURRENT_DEFAULT="'dflt'"),
            ],
            **{"delta.constraints.idlim": "id < 100000"},
        )

    b.step("create", create, expect="ok")
    parts = [b.gen(0, 6), b.gen(100, 6), b.gen(200, 6)]
    b.step("distributed_computed", _distributed_merged(parts), expect="ok")
    given = b.gen(300, 4)
    given = given.append_column("g", pa.array([i * 2 for i in range(300, 304)], pa.int64()))
    given = given.append_column("dflt", pa.array(["mine"] * 4))
    b.step(
        "distributed_given", _distributed_merged([given, b.gen(400, 3), b.gen(500, 3)]), expect="ok"
    )
    b.step(
        "distributed_invariant_violated",
        _distributed_merged([b.gen(600, 2), b.gen(-5, 2)]),
        expect="refused",
    )
    b.step(
        "distributed_generated_wrong",
        _distributed_merged(
            [b.gen(700, 2).append_column("g", pa.array([1, 2], pa.int64())), b.gen(710, 2)]
        ),
        expect="refused",
    )
    b.step("append_kernel", lambda t: t.append(b.gen(800, 5)), via="kernel", expect="ok")
    b.step("append", lambda t: t.append(b.gen(900, 5)))
    b.step("delete", lambda t: t.delete("id % 10 = 1"))
    b.step("update", lambda t: t.update({"s": "'upd'"}, predicate="id % 10 = 2"))
    maintenance(b, restore_to=None)


def c_generated_defaults(ctx: Context, b: Builder, rel: str, uri: str) -> list[str]:
    """Every row's generated values match their expressions and every default
    was filled, as deltaswamp and as Databricks (after its writes) see them."""
    problems = []
    for tag in ("pre", "databricks"):
        for r in _read(b.root / tag).to_pylist():
            if r["g"] != r["id"] * 2 or r["gs"] != (None if r["s"] is None else "g" + r["s"]):
                problems.append(f"{tag}: row {r['id']} has g={r['g']} gs={r['gs']!r}")
            if r["dflt"] is None:
                problems.append(f"{tag}: row {r['id']} has no default")
    bad = int(
        ctx.warehouse.rows(
            f"SELECT count_if(g <> id * 2 OR dflt IS NULL OR id < 0) FROM {dref(uri)}"
        )[0][0]
    )
    if bad:
        problems.append(f"Databricks sees {bad} rows with a wrong generated value or default")
    return problems[:20]


# ======================================================================= cases

#: Predicate groups for the stats battery; see `_group`.
STATS_GROUPS = (
    "integer",
    "decimal",
    "float",
    "nan",
    "string",
    "long-string",
    "binary-bool",
    "date",
    "timestamp",
    "nested",
)


def groups(
    *, nested: bool = True, long: bool = False, nan: bool = True, decimal: bool = True
) -> tuple[str, ...]:
    """The stats groups a `rows()` table covers, for a case declaration."""
    out = ["integer", "float", "string", "binary-bool", "date", "timestamp"]
    if decimal:
        out.append("decimal")
    if nan:
        out.append("nan")
    if long:
        out.append("long-string")
    if nested:
        out.append("nested")
    return tuple(g for g in STATS_GROUPS if g in out)


@dataclass(frozen=True)
class Case:
    """One table shape under one operation history."""

    shape: str
    history: str
    build: Callable[[Builder], None]
    #: Stats groups the table must cover (the battery fails on a missing one).
    stats: tuple[str, ...]
    cdf: bool = False
    #: The history adds a column after the change feed is on.
    adds_column: bool = True
    #: How the expected side of a comparison rebuilds a VARIANT column: a
    #: SQL template over `{c}`, the column as JSON text.
    variant: Mapping[str, str] = field(default_factory=dict)
    #: VARIANT leaves (read back as JSON text), which SQL cannot order or compare.
    unordered: frozenset[tuple[str, ...]] = frozenset()
    #: Columns Databricks computes, left out of its INSERT statements.
    generated: frozenset[str] = frozenset()
    #: check key -> reason: open bugs, marked xfail(strict=True) so the
    #: suite flips when they are fixed.
    known: Mapping[str, str] = field(default_factory=dict)
    #: Columns left out of the rows deltaswamp appends to Databricks' result
    #: (identity values it must generate itself, not copy).
    drop_on_append: frozenset[str] = frozenset()
    #: check key -> what the case checks beyond the common pipeline, run
    #: last: ``fn(ctx, builder, rel, uri)`` returns the problems found.
    checks: Mapping[str, Callable[..., list[str]]] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.shape}-{self.history}"


CDF_SCHEMA = (
    "table_changes() over a range that ends before a later ADD COLUMN returns the "
    "current schema on Databricks; deltaswamp cdf() returns the end version's"
)
DV_DECIMAL = (
    "kernel DV DML re-serialises the re-added file's stats through f64, so a "
    "DECIMAL(38,18) max falls below the real value and Databricks skips the row"
)
INT96_NESTED = (
    "deltaswamp cannot read ARRAY<TIMESTAMP> that Databricks wrote as INT96 "
    "(Cannot cast LIST to non-list data type Timestamp)"
)
VARIANT_NULL = (
    "MERGE into a VARIANT table fails when an ARRAY<VARIANT> source value has a NULL "
    "element (the null element keeps null metadata/value children)"
)

_VARIANT_TEMPLATES = {
    "v": "parse_json({c})",
    "sv": (
        "CASE WHEN {c} IS NULL THEN NULL ELSE named_struct('x', parse_json({c}.x), 'n', {c}.n) END"
    ),
    "av": "transform({c}, e -> parse_json(e))",
    "mv": "transform_values({c}, (k, x) -> parse_json(x))",
}


def _nan(**more: str) -> dict[str, str]:
    """Open bugs a case still has; empty once they are fixed (audit7/ixfix)."""
    return {}


CASES: list[Case] = [
    Case("plain", "full", h_plain, groups(long=True), known=_nan()),
    Case("plain_ancient", "full", h_plain_ancient, groups(long=True), known=_nan()),
    # Partitioned tables with NaN read right: each file there holds one
    # partition's rows, so the footer bounds that leave NaN out prune nothing.
    Case("partitioned", "partition_ops", h_partitioned, groups()),
    Case("dv", "full", h_dv, groups(), known=_nan()),
    Case("dv_partitioned", "full", h_dv_partitioned, groups()),
    Case(
        "cdf",
        "full",
        h_cdf,
        groups(),
        cdf=True,
        known=_nan(**{"cdf-before-add-column": CDF_SCHEMA}),
    ),
    Case(
        "dv_cdf",
        "full",
        h_dv_cdf,
        groups(),
        cdf=True,
        known=_nan(**{"cdf-before-add-column": CDF_SCHEMA}),
    ),
    Case("cm_name", "rename_drop", h_cm_name, groups(decimal=False), known=_nan()),
    Case("cm_id", "rename_drop", h_cm_id, groups(decimal=False), known=_nan()),
    Case("row_tracking", "full", h_row_tracking, groups(), known=_nan()),
    Case(
        "ict",
        "full",
        h_ict,
        groups(),
        cdf=True,
        known=_nan(**{"cdf-before-add-column": CDF_SCHEMA}),
    ),
    Case("type_widening", "widen", h_type_widening, groups(nested=False), known=_nan()),
    Case("ntz", "full", h_ntz, groups(nested=False)),
    Case("types", "full", h_types, groups(long=True), known=_nan()),
    Case(
        "generated",
        "constraints",
        h_generated,
        ("integer", "decimal", "string", "date", "timestamp"),
        generated=frozenset({"g", "gd"}),
    ),
    Case("constraints", "violations", h_constraints, groups(), known=_nan()),
    Case("v2_checkpoints", "full", h_v2_checkpoints, groups(), known=_nan()),
    Case("liquid", "recluster", h_liquid, groups(), known=_nan()),
    Case("checkpoint_interval", "log_compaction", h_log_compaction, groups(), known=_nan()),
    Case("legacy_deltalake", "upgrade", h_legacy_upgrade, groups(), known=_nan()),
    Case(
        "legacy_cdf",
        "enable_cdf",
        h_legacy_cdf,
        groups(),
        cdf=True,
        adds_column=False,
        known=_nan(),
    ),
    Case("everything", "full", h_everything, groups(), cdf=True),
    Case(
        "variant",
        "databricks_created",
        h_variant,
        ("integer", "string", "nested"),
        variant=_VARIANT_TEMPLATES,
        unordered=frozenset({("v",), ("sv", "x")}),
        known={},
    ),
    Case(
        "dv_decimal",
        "kernel_dv_dml",
        h_dv_decimal_stats,
        ("integer", "decimal", "string"),
        known={},
    ),
    Case(
        "nested_timestamps",
        "append",
        h_nested_timestamps,
        groups(nested=False),
        known=_nan(**{"roundtrip-read": INT96_NESTED, "roundtrip-append": INT96_NESTED}),
    ),
    Case(
        "early_dates",
        "write_paths",
        h_early_write_paths,
        ("integer", "string", "date", "timestamp", "nested"),
    ),
    # The write paths a Ray Data datasink uses (docs/ray-data.md).
    Case(
        "identity",
        "databricks_created",
        h_identity,
        _FLAT_STATS,
        generated=frozenset({"idn", "idd"}),
        drop_on_append=frozenset({"idn", "idd"}),
        checks={"identity-unique": c_identity},
    ),
    Case(
        "checkpoint_protection",
        "databricks_created",
        h_checkpoint_protection,
        _FLAT_STATS,
        checks={"checkpoint-protection": c_checkpoint_protection},
    ),
    Case(
        "domain_metadata",
        "distributed",
        h_domain_metadata,
        _FLAT_STATS,
        checks={"domain-metadata": c_domain_metadata},
    ),
    Case(
        "cdf_overwrite",
        "distributed",
        h_cdf_overwrite,
        _FLAT_STATS,
        cdf=True,
        adds_column=False,
        checks={"planned-changes": c_planned_changes},
    ),
    Case("create_by_plan", "distributed", h_create_by_plan, _FLAT_STATS),
    Case(
        "generated_defaults",
        "distributed",
        h_generated_defaults,
        _FLAT_STATS,
        generated=frozenset({"g", "gs"}),
        checks={"generated-values": c_generated_defaults},
    ),
]
BY_ID = {c.id: c for c in CASES}


# ================================================================== evidence


@dataclass
class Skip:
    reason: str


@dataclass
class Diff:
    """Rows one side has and the other lacks (EXCEPT ALL, both ways)."""

    label: str
    relation: str
    ds_rows: int
    db_rows: int
    db_only: int
    ds_only: int
    db_columns: list[str] = field(default_factory=list)
    ds_columns: list[str] = field(default_factory=list)
    samples: dict[str, list[str]] = field(default_factory=dict)
    note: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.db_only == 0
            and self.ds_only == 0
            and self.db_rows == self.ds_rows
            and self.db_columns == self.ds_columns
        )

    def explain(self) -> str:
        lines = [
            f"{self.label}: Databricks {self.db_rows} rows, deltaswamp {self.ds_rows}; "
            f"only on Databricks {self.db_only}, only in deltaswamp {self.ds_only}",
            f"  relation: {_short(self.relation, 300)}",
        ]
        if self.db_columns != self.ds_columns:
            lines.append(f"  columns: Databricks {self.db_columns}, deltaswamp {self.ds_columns}")
        for side, rs in self.samples.items():
            lines += [f"  {side}: {_short(r, 400)}" for r in rs]
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)


@dataclass
class Refused:
    """Both sides refused (a change-feed range across a rename, say)."""

    deltaswamp: str
    databricks: str


@dataclass
class Predicates:
    group: str
    checked: int
    wrong: list[tuple[str, int | None, int]]  # (predicate, Databricks count, expected)


@dataclass
class Roundtrip:
    steps: list[Step]
    diff: Diff | None


@dataclass
class CaseResult:
    case: Case
    evidence: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    @contextlib.contextmanager
    def gather(self, *keys: str) -> Iterator[None]:
        """Record an exception while gathering as the failure of every key."""
        try:
            yield
        except Exception:
            text = traceback.format_exc()[-3000:]
            for k in keys:
                self.errors.setdefault(k, text)


# ============================================================== comparisons


def _q(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _s(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def dref(uri: str, version: int | None = None) -> str:
    ref = f"delta.`{uri}`"
    return ref if version is None else f"{ref} VERSION AS OF {version}"


def expected_file(ctx: Context, key: str, table: pa.Table) -> str:
    """Stage deltaswamp's answer as a Parquet file Databricks can read.

    Columns are renamed c0..cN (a plain Parquet read refuses the characters
    column mapping allows), and the footer names a Spark version so the
    warehouse reads early dates proleptically, as written.
    """
    t = table.rename_columns([f"c{i}" for i in range(table.num_columns)])
    t = t.replace_schema_metadata({"org.apache.spark.version": "3.5.0"})
    buf = io.BytesIO()
    pq.write_table(t, buf)
    rel = f"expected/{key}.parquet"
    ctx.staging.put(rel, buf.getvalue())
    return ctx.staging.uri(rel)


#: (label, Databricks relation, deltaswamp's rows, columns to leave out)
Item = tuple[str, str, pa.Table, tuple[str, ...]]


def compare_many(
    ctx: Context,
    case: Case,
    items: list[Item],
) -> dict[str, Diff]:
    """EXCEPT ALL both ways for each (label, relation, deltaswamp table, drop).

    Both sides become one JSON document per row (`to_json` of the row's
    struct), so every type, nested value and column name compares the same
    way. `to_json` leaves NULL fields out, so a column only one side has
    could hide there; the column lists are compared separately. All row
    counts go in one statement.
    """
    if not items:
        return {}
    parts = []
    sides = {}
    for label, relation, table, drop in items:
        table = table.select([c for c in table.column_names if c not in drop])
        uri = expected_file(ctx, f"{case.id}/{label}", table)
        inner = (
            f"(SELECT * EXCEPT ({', '.join(_q(d) for d in drop)}) FROM {relation})"
            if drop
            else relation
        )
        a = f"(SELECT to_json(struct(*)) j FROM {inner})"
        fields = ", ".join(
            f"{_s(n)}, {case.variant.get(n, '{c}').format(c=f'c{i}')}"
            for i, n in enumerate(table.column_names)
        )
        b = f"(SELECT to_json(named_struct({fields})) j FROM parquet.`{uri}`)"
        db_columns = [
            c.name
            for c in ctx.warehouse.run(f"SELECT * FROM {inner} LIMIT 0").manifest.schema.columns
        ]
        sides[label] = (a, b, relation, table.num_rows, db_columns, table.column_names)
        parts += [
            f"(SELECT count(*) FROM {a} x)",
            f"(SELECT count(*) FROM (SELECT * FROM {a} x EXCEPT ALL SELECT * FROM {b} y))",
            f"(SELECT count(*) FROM (SELECT * FROM {b} y EXCEPT ALL SELECT * FROM {a} x))",
        ]
    counts = ctx.warehouse.rows("SELECT " + ", ".join(parts))[0]
    out: dict[str, Diff] = {}
    for n, (label, (a, b, relation, ds_rows, db_cols, ds_cols)) in enumerate(sides.items()):
        db_rows, db_only, ds_only = (int(x) for x in counts[3 * n : 3 * n + 3])
        diff = Diff(label, relation, ds_rows, db_rows, db_only, ds_only, db_cols, list(ds_cols))
        if not diff.ok:
            with contextlib.suppress(Exception):
                diff.samples["only on Databricks"] = [
                    r[0]
                    for r in ctx.warehouse.rows(
                        f"SELECT * FROM {a} x EXCEPT ALL SELECT * FROM {b} y LIMIT 3"
                    )
                ]
                diff.samples["only in deltaswamp"] = [
                    r[0]
                    for r in ctx.warehouse.rows(
                        f"SELECT * FROM {b} y EXCEPT ALL SELECT * FROM {a} x LIMIT 3"
                    )
                ]
        out[label] = diff
    return out


def _open(path: Path | str, version: int | None = None) -> Any:
    return ds.connect("file://").open_table(str(path), version=version)


def _read(path: Path | str, version: int | None = None) -> pa.Table:
    return pa.table(_open(path, version).to_arrow())


def detail_mismatches(ctx: Context, uri: str, local: Path) -> dict[str, Any]:
    """Where DESCRIBE DETAIL and deltaswamp disagree about the table."""
    row = ctx.warehouse.record(f"DESCRIBE DETAIL {dref(uri)}")
    t = _open(local)
    theirs: dict[str, Any] = {
        "features": sorted(json.loads(row.get("tableFeatures") or "[]")),
        "properties": json.loads(row.get("properties") or "{}"),
        "protocol": [int(row["minReaderVersion"]), int(row["minWriterVersion"])],
        "partitionColumns": json.loads(row.get("partitionColumns") or "[]"),
        "numFiles": int(row["numFiles"]),
    }
    ours: dict[str, Any] = {
        "features": sorted(t.features()),
        "properties": dict(t.properties()),
        "protocol": list(t.protocol()),
        "partitionColumns": list(t.detail().get("partition_columns") or []),
        "numFiles": pa.table(t.files()).num_rows,
    }
    return {
        k: {"databricks": theirs[k], "deltaswamp": ours[k]} for k in theirs if theirs[k] != ours[k]
    }


# ------------------------------------------------------------ stats battery


def _literal(v: Any, ty: pa.DataType) -> str | None:
    if v is None:
        return "NULL"
    if pa.types.is_boolean(ty):
        return "true" if v else "false"
    if pa.types.is_integer(ty):
        sql = {8: "TINYINT", 16: "SMALLINT", 32: "INT", 64: "BIGINT"}[ty.bit_width]
        return f"CAST({v} AS {sql})"
    if pa.types.is_floating(ty):
        if math.isnan(v):
            text = "NaN"
        elif math.isinf(v):
            text = "Infinity" if v > 0 else "-Infinity"
        else:
            text = repr(float(v))
        return f"CAST('{text}' AS {'FLOAT' if ty.bit_width == 32 else 'DOUBLE'})"
    if pa.types.is_decimal(ty):
        return f"CAST('{format(v, 'f')}' AS DECIMAL({ty.precision},{ty.scale}))"
    if pa.types.is_string(ty) or pa.types.is_large_string(ty):
        return _s(v).replace("\n", "\\n").replace("\t", "\\t")
    if pa.types.is_binary(ty):
        return "X'" + bytes(v).hex() + "'"
    if pa.types.is_date(ty):
        return f"DATE'{v.year:04d}-{v.month:02d}-{v.day:02d}'"
    if pa.types.is_timestamp(ty):
        text = (
            f"{v.year:04d}-{v.month:02d}-{v.day:02d} "
            f"{v.hour:02d}:{v.minute:02d}:{v.second:02d}.{v.microsecond:06d}"
        )
        return f"TIMESTAMP_NTZ'{text}'" if ty.tz is None else f"TIMESTAMP'{text}+00:00'"
    return None


def _key(v: Any) -> tuple[int, Any]:
    # Spark orders NaN above every other value, +Infinity included.
    if isinstance(v, float) and math.isnan(v):
        return (1, 0.0)
    return (0, v)


def _leaves(schema: pa.Schema) -> list[tuple[tuple[str, ...], pa.DataType]]:
    out: list[tuple[tuple[str, ...], pa.DataType]] = []

    def walk(prefix: tuple[str, ...], f: pa.Field) -> None:
        if pa.types.is_struct(f.type):
            for child in f.type:
                walk((*prefix, f.name), child)
        elif not pa.types.is_nested(f.type):
            out.append(((*prefix, f.name), f.type))

    for f in schema:
        walk((), f)
    return out


def _get(row: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    v: Any = row
    for p in path:
        if v is None:
            return None
        v = v.get(p)
    return v


def _group(path: tuple[str, ...], ty: pa.DataType, values: list[Any]) -> str | None:
    if len(path) > 1:
        return "nested"
    if pa.types.is_integer(ty):
        return "integer"
    if pa.types.is_decimal(ty):
        return "decimal"
    if pa.types.is_floating(ty):
        return "nan" if any(isinstance(v, float) and math.isnan(v) for v in values) else "float"
    if pa.types.is_string(ty) or pa.types.is_large_string(ty):
        return "long-string" if any(v is not None and len(v) > 1000 for v in values) else "string"
    if pa.types.is_binary(ty) or pa.types.is_boolean(ty):
        return "binary-bool"
    if pa.types.is_date(ty):
        return "date"
    if pa.types.is_timestamp(ty):
        return "timestamp"
    return None


def predicate_battery(
    table: pa.Table, skip: frozenset[tuple[str, ...]] = frozenset()
) -> dict[str, list[tuple[str, int]]]:
    """Selective predicates per leaf column, with the row count each must return.

    For the smallest, median and largest value of every leaf (struct fields
    included): =, <, >, >= and IS NULL -- the shapes data skipping prunes on,
    so a wrong min/max in the log or a footer drops rows here.
    """
    data = table.to_pylist()
    out: dict[str, list[tuple[str, int]]] = {}
    for path, ty in _leaves(table.schema):
        if path in skip:
            continue
        values = [_get(r, path) for r in data]
        group = _group(path, ty, values)
        if group is None:
            continue
        col = ".".join(_q(p) for p in path)
        preds = out.setdefault(group, [])
        preds.append((f"{col} IS NULL", sum(v is None for v in values)))
        present = [v for v in values if v is not None]
        distinct = sorted({_key(v) for v in present})
        if not distinct:
            continue
        finite = [k for k in distinct if k[0] == 0]
        picks = [distinct[0], distinct[-1]]
        if finite:
            picks += [finite[len(finite) // 2], finite[-1]]
        if pa.types.is_binary(ty) or pa.types.is_boolean(ty):
            picks = picks[:2]
        for pick in dict.fromkeys(picks):
            value = float("nan") if pick[0] == 1 else pick[1]
            lit = _literal(value, ty)
            if lit is None:
                break
            keys = [_key(v) for v in present]
            eq = sum(k == pick for k in keys)
            lt = sum(k < pick for k in keys)
            gt = sum(k > pick for k in keys)
            preds += [
                (f"{col} = {lit}", eq),
                (f"{col} < {lit}", lt),
                (f"{col} > {lit}", gt),
                (f"{col} >= {lit}", eq + gt),
            ]
    return out


def run_predicates(
    ctx: Context, relation: str, preds: list[tuple[str, int]], chunk: int
) -> list[tuple[str, int | None, int]]:
    wrong: list[tuple[str, int | None, int]] = []
    for start in range(0, len(preds), chunk):
        part = preds[start : start + chunk]
        sql = " UNION ALL ".join(
            f"SELECT {j} k, count(*) n FROM {relation} WHERE {p}" for j, (p, _) in enumerate(part)
        )
        got = {int(k): int(n) for k, n in ctx.warehouse.rows(sql)}
        wrong += [(p, got.get(j), n) for j, (p, n) in enumerate(part) if got.get(j) != n]
    return wrong


# ------------------------------------------------------------------ footers


@dataclass
class FooterAudit:
    """The Parquet footers of the data files deltaswamp committed."""

    #: Files a kernel commit added, and those among them without the key.
    kernel: int
    kernel_unmarked: list[str]
    #: Files a delta-rs commit added. delta-rs 1.6.5 cannot write the key, so
    #: these go without it by design -- and must then hold no early value.
    deltars: int
    early_in_unmarked: list[str]


def _early_value(column: Any) -> bool:
    """Whether an Arrow column holds a date before 1582-10-15 or a timestamp before 1900."""
    chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
    for arr in chunks:
        ty = arr.type
        if pa.types.is_struct(ty):
            if any(_early_value(arr.field(i)) for i in range(ty.num_fields)):
                return True
        elif pa.types.is_map(ty):
            if _early_value(arr.keys) or _early_value(arr.items):
                return True
        elif pa.types.is_list(ty) or pa.types.is_large_list(ty):
            if _early_value(arr.flatten()):
                return True
        elif pa.types.is_date(ty) or pa.types.is_timestamp(ty):
            low = pc.min(arr).as_py()
            if low is None:
                continue
            if isinstance(low, dt.datetime):
                if low.replace(tzinfo=None) < dt.datetime(1900, 1, 1):
                    return True
            elif low < dt.date(1582, 10, 15):
                return True
    return False


def footer_audit(local: Path, external: set[int]) -> FooterAudit:
    """Which data files deltaswamp added carry org.apache.spark.version.

    A file without it is read by Databricks with the legacy calendar rebase,
    shifting early dates. Every kernel-written file must carry it; a
    delta-rs-written one cannot, so it must hold no value the rebase moves.
    Files another writer committed (`external` versions) are left out.
    """
    added: dict[str, str] = {}
    foreign: set[str] = set()
    for f in sorted((local / "_delta_log").glob("*.json")):
        m = re.match(r"^(\d{20})\.json$", f.name)
        if not m:
            continue
        actions = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        info: dict[str, Any] = next((a["commitInfo"] for a in actions if "commitInfo" in a), {})
        writer = "kernel" if str(info.get("engineInfo", "")).startswith("deltaswamp") else "deltars"
        for a in actions:
            for kind in ("add", "cdc"):
                if kind in a:
                    rel = urllib.parse.unquote(a[kind]["path"])
                    if int(m.group(1)) in external:
                        foreign.add(rel)
                    else:
                        added.setdefault(rel, writer)
    audit = FooterAudit(0, [], 0, [])
    for rel, writer in sorted(added.items()):
        path = local / rel
        if rel in foreign or not rel.endswith(".parquet") or not path.exists():
            continue
        marked = b"org.apache.spark.version" in (pq.ParquetFile(path).metadata.metadata or {})
        if writer == "kernel":
            audit.kernel += 1
            if not marked:
                audit.kernel_unmarked.append(rel)
        else:
            audit.deltars += 1
        if not marked and any(_early_value(c) for c in pq.read_table(path).columns):
            audit.early_in_unmarked.append(rel)
    return audit


# ================================================================== pipeline


@dataclass
class Context:
    """What a case needs from the session: warehouse, volume, scratch space."""

    warehouse: Warehouse
    staging: Staging
    local: Path

    def case_dir(self, b: Builder) -> str:
        return f"tables/{b.root.name}"


DATABRICKS_KEYS = ("databricks-dml", "roundtrip-read", "roundtrip-append")
READ_KEYS = ("read-latest", "read-early", "read-middle", "read-late", "read-vacuumed")


def check_keys(case: Case) -> list[str]:
    """Every evidence key the pipeline gathers for `case`."""
    keys = ["history", "footers", *READ_KEYS, "detail-pre", "detail-final"]
    keys += [f"stats-{g}" for g in case.stats] + ["stats-coverage"]
    if case.cdf:
        keys += ["cdf-full", "cdf-window", "cdf-databricks-writes"]
        if case.adds_column:
            keys.append("cdf-before-add-column")
    keys += list(DATABRICKS_KEYS)
    if case.variant:
        keys.append("variant-merge-null-element")
    keys += list(case.checks)
    return keys


def run_case(case: Case, ctx: Context) -> CaseResult:
    started = time.monotonic()
    res = CaseResult(case)
    b = Builder(ctx.local / case.id, ctx)
    with res.gather(*check_keys(case)):
        case.build(b)
    res.evidence["history"] = b.steps
    if res.errors:
        return res  # the history itself crashed; nothing else is meaningful
    remote = f"tables/{case.id}"
    pre, final = b.root / "pre", b.root / "final"

    with res.gather("footers"):
        res.evidence["footers"] = footer_audit(pre, b.external)

    with res.gather(*[k for k in check_keys(case) if k not in ("history", "footers")]):
        ctx.staging.upload_dir(pre, f"{remote}/pre")
        ctx.staging.upload_dir(final, f"{remote}/final")
    if set(res.errors) - {"footers"}:
        return res
    pre_uri, final_uri = ctx.staging.uri(f"{remote}/pre"), ctx.staging.uri(f"{remote}/final")

    # (a) the same rows at the latest version, at earlier ones, after VACUUM
    with res.gather(*READ_KEYS):
        latest = int(_open(pre).version)
        points = {
            "read-early": min(2, latest),
            "read-middle": latest // 2,
            "read-late": max(latest - 2, 0),
        }
        items: list[Item] = [("read-latest", dref(pre_uri), _read(pre), ())]
        items += [(k, dref(pre_uri, v), _read(pre, v), ()) for k, v in points.items()]
        items.append(("read-vacuumed", dref(final_uri), _read(final), ()))
        res.evidence.update(compare_many(ctx, case, items))

    # (b) features, properties, protocol, partitioning, file count
    for tag, uri, local in (("pre", pre_uri, pre), ("final", final_uri, final)):
        with res.gather(f"detail-{tag}"):
            res.evidence[f"detail-{tag}"] = detail_mismatches(ctx, uri, local)

    # (d) the change feed
    if case.cdf:
        _change_feed(res, ctx, case, b, pre, pre_uri, latest)

    # (e) data skipping: selective predicates on the most fragmented version
    with res.gather(*[f"stats-{g}" for g in case.stats], "stats-coverage"):
        version = b.marks.get("stats", latest)
        battery = predicate_battery(_read(pre, version), case.unordered)
        res.evidence["stats-coverage"] = (sorted(case.stats), sorted(battery))
        for g in case.stats:
            with res.gather(f"stats-{g}"):
                preds = battery.get(g)
                if not preds:
                    res.evidence[f"stats-{g}"] = Predicates(g, 0, [])
                    continue
                chunk = 6 if g == "long-string" else 60
                wrong = run_predicates(ctx, dref(pre_uri, version), preds, chunk)
                res.evidence[f"stats-{g}"] = Predicates(g, len(preds), wrong)

    # (c) Databricks writes on top; deltaswamp reads and writes the result
    _databricks_writes(res, ctx, case, b, final, f"{remote}/final", final_uri)

    if case.variant:
        with res.gather("variant-merge-null-element"):
            res.evidence["variant-merge-null-element"] = _variant_merge(b, final)
    for key, check in case.checks.items():
        with res.gather(key):
            res.evidence[key] = check(ctx, b, f"{remote}/final", final_uri)
    res.seconds = time.monotonic() - started
    return res


def _cdf_on(path: Path, v: int) -> bool:
    try:
        props = _open(path, v).properties()
    except Exception:
        return False
    return str(props.get("delta.enableChangeDataFeed", "")).lower() == "true"


def _cdf_items(
    ctx: Context,
    case: Case,
    local: Path,
    uri: str,
    ranges: dict[str, tuple[int, int]],
    res: CaseResult,
) -> list[tuple[str, str, pa.Table, tuple[str, ...]]]:
    t = _open(local)
    ict = str(t.properties().get("delta.enableInCommitTimestamps", "")).lower() == "true"
    # Without in-commit timestamps the commit time is the file's mtime, which
    # the upload rewrites.
    drop = () if ict else ("_commit_timestamp",)
    items: list[Item] = []
    for key, (start, end) in ranges.items():
        relation = f"table_changes('delta.`{uri}`', {start}, {end})"
        try:
            got = t.cdf(starting_version=start, ending_version=end)
            got = got if isinstance(got, pa.Table) else pa.table(got)
        except ChangeFeedSchemaChangeError as exc:
            try:
                ctx.warehouse.rows(f"SELECT count(*) FROM {relation}")
            except Exception as theirs:
                res.evidence[key] = Refused(str(exc)[:300], str(theirs)[:300])
                continue
            raise
        items.append((key, relation, got, drop))
    return items


def _change_feed(
    res: CaseResult, ctx: Context, case: Case, b: Builder, pre: Path, uri: str, latest: int
) -> None:
    keys = ["cdf-full", "cdf-window"] + (["cdf-before-add-column"] if case.adds_column else [])
    with res.gather(*keys):
        first = next(v for v in range(latest + 1) if _cdf_on(pre, v))
        ranges = {"cdf-full": (first, latest), "cdf-window": (max(first, latest - 3), latest)}
        if case.adds_column:
            end = b.marks["add_column"] - 1
            ranges["cdf-before-add-column"] = (first, end)
        res.evidence.update(compare_many(ctx, case, _cdf_items(ctx, case, pre, uri, ranges, res)))


def databricks_statements(case: Case, ref: str, table: Any) -> list[str]:
    names = table.schema().names
    scol = next((n for n in ("s", "s renamed", "s2") if n in names), None)
    cols = [n for n in names if n not in case.generated]
    sel = ", ".join(_q(n) for n in cols)
    out = []
    if scol:
        out.append(f"UPDATE {ref} SET {_q(scol)} = 'dbx_u' WHERE id % 10 = 2")
    out.append(f"DELETE FROM {ref} WHERE id % 10 = 6")
    if scol:
        out.append(
            f"MERGE INTO {ref} t USING (SELECT DISTINCT id FROM {ref} WHERE id % 10 = 7) s "
            f"ON t.id = s.id WHEN MATCHED THEN UPDATE SET t.{_q(scol)} = 'dbx_m'"
        )
    out.append(
        f"MERGE INTO {ref} t USING (SELECT {sel} FROM {ref} WHERE id % 10 = 8) s ON false "
        f"WHEN NOT MATCHED THEN INSERT ({sel}) VALUES ({', '.join('s.' + _q(n) for n in cols)})"
    )
    out.append(f"INSERT INTO {ref} ({sel}) SELECT {sel} FROM {ref} WHERE id % 10 = 9")
    out.append(f"OPTIMIZE {ref}")
    out.append(f"VACUUM {ref} DRY RUN")
    if "deletionVectors" in table.features():
        out.append(f"REORG TABLE {ref} APPLY (PURGE)")
    return out


def _databricks_writes(
    res: CaseResult, ctx: Context, case: Case, b: Builder, final: Path, rel: str, uri: str
) -> None:
    keys = [*DATABRICKS_KEYS] + (["cdf-databricks-writes"] if case.cdf else [])
    local = b.root / "databricks"
    with res.gather(*keys):
        v0 = int(_open(final).version)
        results: list[tuple[str, str | None]] = []
        for sql in databricks_statements(case, dref(uri), _open(final)):
            try:
                ctx.warehouse.run(sql)
                results.append((sql, None))
            except Exception as exc:
                results.append((sql, str(exc)[:500]))
        res.evidence["databricks-dml"] = results

        shutil.rmtree(local, ignore_errors=True)
        ctx.staging.download_dir(rel, local)
    if set(keys) & set(res.errors):
        return

    with res.gather("roundtrip-read"):
        lv = int(_open(local).version)
        items: list[Item] = [
            (f"databricks-v{v}", dref(uri, v), _read(local, v), ()) for v in range(v0 + 1, lv + 1)
        ]
        diffs = compare_many(ctx, case, items)
        bad = [d for d in diffs.values() if not d.ok]
        res.evidence["roundtrip-read"] = bad[0] if bad else next(iter(diffs.values()))

    if case.cdf:
        with res.gather("cdf-databricks-writes"):
            ranges = {"cdf-databricks-writes": (v0 + 1, int(_open(local).version))}
            res.evidence.update(
                compare_many(ctx, case, _cdf_items(ctx, case, local, uri, ranges, res))
            )

    with res.gather("roundtrip-append"):
        before = _digests(local)
        # The rows come from deltaswamp's own copy, so an unreadable
        # Databricks file does not stop the append from being tried.
        extra = _read(final).slice(0, 3)
        extra = extra.drop_columns([c for c in case.drop_on_append if c in extra.column_names])
        steps = []
        ops: list[tuple[str, Callable[[Any], Any]]] = [
            ("append", lambda t: t.append(extra)),
            ("delete", lambda t: t.delete(f"id = {extra.column('id')[0].as_py()}")),
            ("checkpoint", lambda t: t.checkpoint()),
        ]
        for name, fn in ops:
            vb = _open(local).version
            outcome, detail = "ok", ""
            try:
                fn(_open(local))
            except REFUSALS as exc:
                outcome, detail = "refused", f"{type(exc).__name__}: {_short(str(exc), 300)}"
            except Exception as exc:
                outcome, detail = "error", f"{type(exc).__name__}: {_short(str(exc), 500)}"
            steps.append(
                Step(
                    name,
                    outcome,
                    vb,
                    _open(local).version,
                    "ok" if name == "append" else None,
                    detail,
                )
            )
        changed = [p for p, h in _digests(local).items() if before.get(p) != h]
        ctx.staging.upload_dir(local, rel, only=changed)
        diff = compare_many(ctx, case, [("after-append", dref(uri), _read(local), ())])[
            "after-append"
        ]
        res.evidence["roundtrip-append"] = Roundtrip(steps, diff)


def _variant_merge(b: Builder, final: Path) -> Any:
    """MERGE a row whose ARRAY<VARIANT> holds a NULL element; return the merged row."""
    path = b.root / "variant_merge"
    shutil.rmtree(path, ignore_errors=True)
    shutil.copytree(final, path)
    t = _open(path)
    src = _variant_rows(7003, 1).append_column("newc", pa.array([None], pa.string()))
    t.merge(
        src, "target.id = source.id"
    ).when_matched_update_all().when_not_matched_insert_all().execute()
    got = _read(path).filter(pc.equal(pc.field("id"), 7003)).select(["id", "av"]).to_pylist()
    return got


# =================================================================== runner


class Runner:
    """Runs every selected case in a thread pool; tests wait on their case."""

    def __init__(self, cases: list[Case], ctx: Context, workers: int) -> None:
        self.ctx = ctx
        self.pool = concurrent.futures.ThreadPoolExecutor(workers, thread_name_prefix="interop")
        self.futures = {c.id: self.pool.submit(run_case, c, ctx) for c in cases}

    def result(self, case_id: str) -> CaseResult:
        return self.futures[case_id].result()

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
