"""Compaction on the kernel, and the conflict rule for blind appends (audit r6cp).

CP-1: a replaceWhere with only adds read as a blind append, so concurrent
loads of one range each committed. CP-2: OPTIMIZE fell back to delta-rs's
own commit (duplicating rows under a concurrent compaction) on CDF, CHECK
and generated-column tables and for delta-rs-only options. CP-3/CP-4: the
kernel compaction replayed the log per bin and held whole steps in memory.
CP-6..CP-11: partition filters, typed conflicts, commit metadata, Z-order
keys and sizing, and the kernel's own OPTIMIZE capability.

The races run in spawned processes, as separate writers do, a few rounds each.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import multiprocessing as mp
import os
import warnings
from typing import Any

import deltalake
import deltaswamp as ds
import pyarrow as pa
import pytest
from deltaswamp.capability import Engine
from deltaswamp.engine.kernel import KernelEngine
from deltaswamp.errors import (
    CommitConflictError,
    DeltaSwampError,
    InvalidArgumentError,
    UnreachableTableError,
)

pytestmark = pytest.mark.skipif(not KernelEngine.available(), reason="needs the native extension")

DV = {"delta.enableDeletionVectors": "true"}


def _commits(path: str) -> list[list[dict[str, Any]]]:
    out = []
    for name in sorted(glob.glob(os.path.join(path, "_delta_log", "*.json"))):
        with open(name) as f:
            out.append([json.loads(line) for line in f if line.strip()])
    return out


def _info(path: str, version: int = -1) -> dict[str, Any]:
    actions = _commits(path)[version]
    return next(a["commitInfo"] for a in actions if "commitInfo" in a)


def _ids(path: str) -> list[int]:
    return sorted(ds.connect().open_table(path).to_arrow().column("id").to_pylist())


def _race(target: Any, args: list[tuple[Any, ...]]) -> list[tuple[str, Any]]:
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(len(args)), ctx.Queue()
    procs = [ctx.Process(target=target, args=(a[0], barrier, q, *a[1:])) for a in args]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    return [q.get(timeout=5) for _ in procs]


# ------------------------------------------------------------ CP-1


def _replace_where(path: str, barrier: Any, q: Any, i: int, merge: bool) -> None:
    warnings.filterwarnings("ignore")
    t = ds.connect().open_table(path)
    data = pa.table(
        {
            "id": pa.array([999, 1000], pa.int64()),
            "day": ["2026-09-27"] * 2,
            "v": pa.array([i, i], pa.int64()),
        }
    )
    barrier.wait()
    try:
        if merge:
            (
                t.merge(data, "t.id = s.id", source_alias="s", target_alias="t")
                .when_matched_update({"v": "s.v"})
                .when_not_matched_insert({"id": "s.id", "day": "s.day", "v": "s.v"})
                .execute()
            )
        else:
            t.overwrite(data, predicate="day = '2026-09-27'")
        q.put(("ok", i))
    except CommitConflictError as exc:
        q.put(("conflict", str(exc)))
    except Exception as exc:  # pragma: no cover - reported by the test
        q.put(("error", f"{type(exc).__name__}: {exc}"))


def _dv_table(tmp_path: Any, name: str) -> str:
    path = str(tmp_path / name)
    ds.connect().write_table(
        path,
        pa.table(
            {
                "id": pa.array(range(10), pa.int64()),
                "day": ["2026-09-26"] * 10,
                "v": pa.array([0] * 10, pa.int64()),
            }
        ),
        properties=DV,
    )
    return path


@pytest.mark.parametrize("merge", [False, True], ids=["replacewhere", "vs-merge"])
def test_concurrent_replace_where_loads_one_copy(tmp_path: Any, merge: bool) -> None:
    for r in range(2):
        path = _dv_table(tmp_path, f"t{r}")
        outcomes = _race(_replace_where, [(path, i, merge and i % 2 == 1) for i in range(4)])
        assert all(kind in ("ok", "conflict") for kind, _ in outcomes), outcomes
        day = ds.connect().open_table(path).to_arrow(predicate="day = '2026-09-27'")
        # Every serial order leaves exactly one row per key.
        assert sorted(day.column("id").to_pylist()) == [999, 1000], outcomes


def test_kernel_commits_say_which_are_blind_appends(tmp_path: Any) -> None:
    path = _dv_table(tmp_path, "t")
    t = ds.connect().open_table(path)
    t.overwrite(
        pa.table({"id": pa.array([1], pa.int64()), "day": ["x"], "v": pa.array([1], pa.int64())}),
        predicate="day = 'x'",
    )
    info = _info(path)
    assert info["operation"] == "WRITE" and info["isBlindAppend"] is False
    assert info["operationParameters"]["mode"] == "Overwrite"
    assert json.loads(info["operationParameters"]["predicate"]) == ["day = 'x'"]
    t.delete("id = 1")
    info = _info(path)
    assert info["isBlindAppend"] is False
    assert json.loads(info["operationParameters"]["predicate"]) == ["id = 1"]

    _kernel_only().open_table(path).append(
        pa.table({"id": pa.array([5], pa.int64()), "day": ["y"], "v": pa.array([0], pa.int64())})
    )
    info = _info(path)
    assert info["isBlindAppend"] is True
    assert info["operationParameters"]["mode"] == "Append"


def test_only_a_flag_or_append_mode_makes_a_blind_append() -> None:
    from deltaswamp.engine.kernel import _blind_append

    # Spark's INSERT ... SELECT: a WRITE with only adds that read the table.
    assert not _blind_append(
        {"operation": "WRITE", "operationParameters": {"mode": "Append"}, "isBlindAppend": False}
    )
    # A replaceWhere with only adds, as the kernel used to write it.
    assert not _blind_append({"operation": "WRITE", "operationParameters": {}})
    assert not _blind_append(
        {"operation": "WRITE", "operationParameters": {"mode": "Overwrite", "predicate": "x"}}
    )
    # delta-rs never writes the flag; its appends say mode Append.
    assert _blind_append({"operation": "WRITE", "operationParameters": {"mode": "Append"}})
    assert _blind_append({"operation": "STREAMING UPDATE", "isBlindAppend": True})


# ------------------------------------------------------------ CP-2


def _optimize(path: str, barrier: Any, q: Any) -> None:
    warnings.filterwarnings("ignore")
    t = ds.connect().open_table(path)
    barrier.wait()
    try:
        r = t.optimize()
        q.put(("ok", (r["num_files_removed"], getattr(r, "engine", None))))
    except CommitConflictError as exc:
        q.put(("conflict", str(exc)))
    except Exception as exc:  # pragma: no cover - reported by the test
        q.put(("error", f"{type(exc).__name__}: {exc}"))


def _fallback_table(tmp_path: Any, name: str, shape: str) -> str:
    path = str(tmp_path / name)
    first = pa.table({"id": pa.array([-1], pa.int64())})
    if shape == "cdf4":
        # Writer version 4, which implies checkConstraints and generatedColumns.
        deltalake.write_deltalake(path, first, configuration={"delta.enableChangeDataFeed": "true"})
    else:
        deltalake.write_deltalake(path, first)
        ds.connect().open_table(path).add_constraint({"pos": "id > -10"})
    for i in range(40):
        deltalake.write_deltalake(path, pa.table({"id": pa.array([i], pa.int64())}), mode="append")
    return path


@pytest.mark.parametrize("shape", ["cdf4", "check"])
def test_concurrent_optimize_of_constrained_tables_never_duplicates(
    tmp_path: Any, shape: str
) -> None:
    for r in range(2):
        path = _fallback_table(tmp_path, f"t{r}", shape)
        protocol = deltalake.DeltaTable(path).protocol()
        outcomes = _race(_optimize, [(path,)] * 3)
        assert all(kind in ("ok", "conflict") for kind, _ in outcomes), outcomes
        assert _ids(path) == list(range(-1, 40)), outcomes
        assert [v[1] for kind, v in outcomes if kind == "ok"] == ["kernel"] * sum(
            kind == "ok" for kind, _ in outcomes
        )
        after = deltalake.DeltaTable(path).protocol()
        # The table's own protocol stays in force: nothing was set aside for good.
        assert (after.min_reader_version, after.min_writer_version) == (
            protocol.min_reader_version,
            protocol.min_writer_version,
        )
        assert not any("protocol" in a for a in _commits(path)[-1])
    if shape == "check":
        with pytest.raises(DeltaSwampError):
            ds.connect().open_table(path).append(pa.table({"id": pa.array([-50], pa.int64())}))


def test_generated_columns_table_is_compacted_by_the_kernel(tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    schema = deltalake.Schema(
        [
            deltalake.Field("id", deltalake.schema.PrimitiveType("long")),
            deltalake.Field(
                "twice",
                deltalake.schema.PrimitiveType("long"),
                metadata={"delta.generationExpression": "id * 2"},
            ),
        ]
    )
    deltalake.DeltaTable.create(path, schema=schema)
    for i in range(5):
        deltalake.write_deltalake(path, pa.table({"id": pa.array([i], pa.int64())}), mode="append")
    t = ds.connect().open_table(path)
    assert t.can("optimize").engine == Engine.KERNEL
    result = t.optimize()
    assert result["num_files_removed"] == 5 and getattr(result, "engine", None) == Engine.KERNEL
    rows = ds.connect().open_table(path).to_arrow().sort_by("id").to_pylist()
    assert rows == [{"id": i, "twice": 2 * i} for i in range(5)]


@pytest.mark.parametrize(
    "option",
    [
        {"writer_properties": {"compression": "ZSTD"}},
        {"min_commit_interval": dt.timedelta(seconds=5)},
    ],
    ids=["writer_properties", "min_commit_interval"],
)
def test_options_only_delta_rs_takes_are_refused_not_run_unsafely(
    tmp_path: Any, option: dict[str, Any]
) -> None:
    path = _fallback_table(tmp_path, "t", "cdf4")
    t = ds.connect().open_table(path)
    name = next(iter(option))
    verdict = t.can("optimize", **option)
    assert not verdict.ok and name in verdict.reason
    with pytest.raises(UnreachableTableError, match=name):
        t.optimize(**option)
    assert len(_commits(path)) == 41  # nothing committed


def test_app_transactions_on_optimize_are_refused(tmp_path: Any) -> None:
    path = _fallback_table(tmp_path, "t", "check")
    props = deltalake.CommitProperties(
        app_transactions=[deltalake.Transaction(app_id="job", version=1)]
    )
    with pytest.raises(UnreachableTableError, match="app transactions"):
        ds.connect().open_table(path).optimize(commit_properties=props)


# ------------------------------------------------------------ CP-3 / CP-4


class _Counting:
    """A kernel snapshot that counts its scans."""

    scans = 0

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def scan(self, *args: Any, **kwargs: Any) -> Any:
        _Counting.scans += 1
        return self._inner.scan(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _partitioned(tmp_path: Any, partitions: int, files: int) -> str:
    path = str(tmp_path / "t")
    for f in range(files):
        deltalake.write_deltalake(
            path,
            pa.table(
                {
                    "id": pa.array(range(f * partitions, (f + 1) * partitions), pa.int64()),
                    "p": [f"p{i}" for i in range(partitions)],
                }
            ),
            partition_by=["p"],
            mode="append",
        )
    return path


def test_optimize_reads_each_step_with_one_scan(tmp_path: Any, monkeypatch: Any) -> None:
    """Each bin's own scan replayed the log: 200 bins, 200 replays (16 s vs 2 s)."""
    path = _partitioned(tmp_path, 20, 3)
    real = KernelEngine.snapshot

    def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _Counting(real(self, *args, **kwargs))

    monkeypatch.setattr(KernelEngine, "snapshot", counting)
    _Counting.scans = 0
    result = ds.connect().open_table(path).optimize()
    monkeypatch.undo()
    assert result["num_files_removed"] == 60 and result["num_files_added"] == 20
    assert _Counting.scans == 1
    assert _ids(path) == list(range(60))


def test_output_files_are_bounded_in_memory(tmp_path: Any, monkeypatch: Any) -> None:
    """One output file used to hold its whole bin; a Z-order, its whole partition."""
    path = str(tmp_path / "t")
    for f in range(8):
        deltalake.write_deltalake(
            path,
            pa.table({"id": pa.array(range(f * 20_000, (f + 1) * 20_000), pa.int64())}),
            mode="append",
        )
    # 160,000 rows of 8 bytes; at most ~400 KB decoded per output file.
    monkeypatch.setattr(KernelEngine, "compaction_max_file_bytes", 400_000)
    result = ds.connect().open_table(path).optimize()
    assert result["num_files_removed"] == 8 and 3 <= result["num_files_added"] <= 5
    assert _ids(path) == list(range(160_000))
    # Those files are as big as the bound allows: compacting them again would
    # write as many, so they are left alone.
    assert ds.connect().open_table(path).optimize()["num_files_removed"] == 0
    z = ds.connect().open_table(path).z_order(["id"])
    assert z["num_files_added"] >= 3
    assert _ids(path) == list(range(160_000))


# ------------------------------------------------------------ CP-6


def _timestamp_partitions(tmp_path: Any) -> str:
    path = str(tmp_path / "t")
    stamps = [
        dt.datetime(2024, 1, 1, 0, 30, 15, 123456, tzinfo=dt.UTC),
        dt.datetime(2024, 1, 1, 1, 30, 15, 123456, tzinfo=dt.UTC),
    ]
    for k in range(3):
        deltalake.write_deltalake(
            path,
            pa.table(
                {
                    "id": pa.array([10 * k + i for i in range(4)], pa.int64()),
                    "pt": pa.array(stamps * 2, pa.timestamp("us", tz="UTC")),
                    "pi": pa.array([1, 2, None, 0], pa.int32()),
                    "ps": pa.array(["a", None, "b", "c"]),
                }
            ),
            partition_by=["pt", "pi", "ps"],
            mode="append",
        )
    return path


def _removed_values(path: str, column: str) -> set[Any]:
    return {a["remove"]["partitionValues"][column] for a in _commits(path)[-1] if "remove" in a}


@pytest.mark.parametrize(
    ("filters", "column", "expected"),
    [
        ([("pt", "=", "2024-01-01T01:30:15.123456Z")], "pt", {"2024-01-01 01:30:15.123456"}),
        ([("pt", ">", "2024-01-01T00:30:15.123456Z")], "pt", {"2024-01-01 01:30:15.123456"}),
        ([("pt", "=", "2024-01-01T02:30:15.123456+01:00")], "pt", {"2024-01-01 01:30:15.123456"}),
        ([("pi", "!=", "1")], "pi", {"0", "2"}),
        ([("pi", "=", "")], "pi", {None}),
        ([("ps", "!=", "")], "ps", {"a", "b", "c"}),
        ([("ps", "not in", ["", "a"])], "ps", set()),
    ],
)
def test_partition_filters_match_as_delta_rs_does(
    tmp_path: Any, filters: list[Any], column: str, expected: set[Any]
) -> None:
    path = _timestamp_partitions(tmp_path)
    ds.connect().open_table(path).optimize(partition_filters=filters)
    if expected:
        assert _removed_values(path, column) == expected
    else:
        assert len(_commits(path)) == 3  # nothing matched, nothing committed


def test_a_partition_filter_value_of_the_wrong_type_is_the_callers_mistake(
    tmp_path: Any,
) -> None:
    path = _timestamp_partitions(tmp_path)
    with pytest.raises(InvalidArgumentError, match=r"not a .*timestamp"):
        ds.connect().open_table(path).optimize(partition_filters=[("pt", ">", "yesterday")])


# ------------------------------------------------------------ CP-7 / CP-8


def test_optimize_losing_to_a_protocol_change_it_cannot_write_is_a_conflict(
    tmp_path: Any, monkeypatch: Any
) -> None:
    path = str(tmp_path / "t")
    for i in range(4):
        deltalake.write_deltalake(path, pa.table({"id": pa.array([i], pa.int64())}), mode="append")
    real = KernelEngine._commit_compaction
    fired: list[int] = []

    def row_tracking_wins(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not fired:
            fired.append(1)
            deltalake.DeltaTable(path).alter.add_feature(
                deltalake.TableFeatures.RowTracking, allow_protocol_versions_increase=True
            )
        return real(self, *args, **kwargs)

    monkeypatch.setattr(KernelEngine, "_commit_compaction", row_tracking_wins)
    with pytest.raises(CommitConflictError, match="row ids"):
        ds.connect().open_table(path).optimize()
    assert _ids(path) == [0, 1, 2, 3]


def test_optimize_records_what_it_did(tmp_path: Any) -> None:
    path = _partitioned(tmp_path, 2, 3)
    result = (
        ds.connect()
        .open_table(path)
        .optimize(zorder_by=["id"], partition_filters=[("p", "=", "p0")])
    )
    assert getattr(result, "engine", None) == Engine.KERNEL
    info = _info(path)
    assert info["operation"] == "OPTIMIZE" and info["isBlindAppend"] is False
    parameters = info["operationParameters"]
    assert json.loads(parameters["zOrderBy"]) == ["id"]
    assert json.loads(parameters["predicate"]) == ["p = 'p0'"]


# ------------------------------------------------------------ CP-9 / CP-10


def test_z_order_by_a_nested_column_is_refused_typed(tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    rows = pa.table(
        {
            "id": pa.array([1, 2], pa.int64()),
            "st": pa.array([{"a": 1}, {"a": 2}]),
            "l": pa.array([[1], [2]]),
        }
    )
    deltalake.write_deltalake(path, rows)
    deltalake.write_deltalake(path, rows, mode="append")
    t = ds.connect().open_table(path)
    for column in ("st", "l"):
        with pytest.raises(InvalidArgumentError, match="min/max statistics"):
            t.z_order([column])


def test_z_order_files_come_out_near_the_target(tmp_path: Any) -> None:
    import random

    random.seed(1)
    path = str(tmp_path / "t")
    n = 100_000
    xs = [random.randrange(10_000) for _ in range(n)]
    ys = [random.randrange(10_000) for _ in range(n)]
    for k in range(10):
        part = slice(k * n // 10, (k + 1) * n // 10)
        deltalake.write_deltalake(
            path,
            pa.table({"x": pa.array(xs[part], pa.int64()), "y": pa.array(ys[part], pa.int64())}),
            mode="append",
        )
    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, "*.parquet")))
    target = size // 8
    ds.connect().open_table(path).optimize(zorder_by=["x", "y"], target_size=target)
    sizes = [a["add"]["size"] for a in _commits(path)[-1] if "add" in a]
    # Sorted rows compress worse than the input did (files were 40% over),
    # and the last slice was a sliver next to full files.
    assert max(sizes) <= target * 1.2, (sizes, target)
    assert min(sizes) >= target * 0.5, (sizes, target)


# ------------------------------------------------------------ CP-11


def _kernel_only() -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.router import Router

    return Connection(
        catalog=FilesystemCatalog(), router=Router(engines={Engine.KERNEL: KernelEngine()})
    )


def test_kernel_only_connections_optimize(tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    for i in range(3):
        deltalake.write_deltalake(path, pa.table({"id": pa.array([i], pa.int64())}), mode="append")
    t = _kernel_only().open_table(path)
    assert t.can("optimize").engine == Engine.KERNEL
    assert t.optimize()["num_files_removed"] == 3
    assert _ids(path) == [0, 1, 2]


def test_tables_delta_rs_cannot_open_are_compacted(tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    conn = ds.connect()
    conn.create_table(
        path,
        schema=pa.schema([("id", pa.int64())]),
        properties={"delta.enableInCommitTimestamps": "true"},
    )
    for i in range(3):
        conn.open_table(path).append(pa.table({"id": pa.array([i], pa.int64())}))
    t = conn.open_table(path)
    assert t.can("optimize").engine == Engine.KERNEL
    assert t.optimize()["num_files_removed"] == 3
    assert "inCommitTimestamp" in _info(path)
    assert _ids(path) == [0, 1, 2]
