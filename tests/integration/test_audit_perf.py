"""Regressions for the optimizations & scale audit (OP-*), each on a real table.

Every test here failed before its fix. The scale numbers themselves are
measured outside the suite; these pin the behavior that produced them (files
skipped, rows not read, snapshots not re-resolved).
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import threading
from typing import Any

import pytest
from deltaswamp.errors import InvalidArgumentError

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
deltalake = pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")


def _many_files(conn: Any, path: str, n: int = 20, props: dict[str, str] | None = None) -> Any:
    t = conn.create_table(
        path,
        pa.schema([("id", pa.int64()), ("v", pa.float64()), ("s", pa.string())]),
        properties=props,
    )
    for i in range(n):
        t.append(
            pa.table(
                {
                    "id": pa.array([i * 10 + j for j in range(10)], pa.int64()),
                    "v": [i + 0.5] * 10,
                    "s": [f"k{i:02d}"] * 10,
                }
            )
        )
    return t


class _Counting:
    """Counts native scans (the files a read touches) on one table."""

    def __init__(self, monkeypatch: Any) -> None:
        import deltaswamp._native as native

        self.resolves = 0
        original = native.Snapshot.resolve
        counter = self

        def resolve(*a: Any, **k: Any) -> Any:
            counter.resolves += 1
            return original(*a, **k)

        monkeypatch.setattr(native.Snapshot, "resolve", staticmethod(resolve))


class TestOptimizeArguments:
    """OP-1, OP-18: values delta-rs hung or crashed on are refused first."""

    @pytest.fixture
    def table(self, conn: Any, tmp_path: Any) -> Any:
        t = conn.create_table(
            str(tmp_path / "t"),
            pa.schema([("id", pa.int64()), ("g", pa.string())]),
            partition_by=["g"],
        )
        for i in range(3):
            t.append(pa.table({"id": pa.array([i], pa.int64()), "g": ["a"]}))
        return t

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_concurrent_tasks": 0},  # waited forever
            {"max_concurrent_tasks": -1},  # OverflowError
            {"target_size": 0},  # raw ValueError
            {"partition_filters": ("g", "=", "a")},  # a bare tuple: raw ValueError
            {"partition_filters": [("id", "=", "1")]},  # a silent no-op
        ],
    )
    def test_refused(self, table: Any, kwargs: dict[str, Any]) -> None:
        with pytest.raises(InvalidArgumentError):
            table.optimize(**kwargs)
        with pytest.raises(InvalidArgumentError):
            table.z_order(["id"], **kwargs)
        assert table.count() == 3

    def test_valid_filters_still_work(self, table: Any) -> None:
        result = table.optimize(partition_filters=[("g", "=", "a")], max_concurrent_tasks=2)
        assert result["numFilesRemoved"] == 3

    def test_nested_zorder_says_nested_fields_are_unsupported(
        self, conn: Any, tmp_path: Any
    ) -> None:
        # OP-21: "the table has no such column" for a field cluster_by accepts.
        t = conn.create_table(
            str(tmp_path / "n"),
            pa.schema([("id", pa.int64()), ("s", pa.struct([("a", pa.int64())]))]),
        )
        with pytest.raises(InvalidArgumentError, match="nested fields"):
            t.z_order(["s.a"])


class TestCreateProperties:
    """OP-2: keys delta-kernel refuses at CREATE are applied as version 1."""

    @pytest.mark.parametrize(
        ("props", "cluster"),
        [
            (
                {
                    "delta.enableDeletionVectors": "true",
                    "delta.autoOptimize.optimizeWrite": "true",
                    "delta.autoOptimize.autoCompact": "true",
                },
                None,
            ),
            (
                {"delta.targetFileSize": "134217728", "delta.isolationLevel": "WriteSerializable"},
                ["id"],
            ),
            ({"delta.enableRowTracking": "true", "delta.tuneFileSizesForRewrites": "true"}, None),
        ],
    )
    def test_kernel_create_stores_them(
        self, conn: Any, tmp_path: Any, props: dict[str, str], cluster: Any
    ) -> None:
        t = conn.create_table(
            str(tmp_path / "t"),
            pa.schema([("id", pa.int64())]),
            properties=props,
            cluster_by=cluster,
        )
        stored = t.properties()
        for key, value in props.items():
            assert stored[key] == value
        t.append(pa.table({"id": pa.array([1], pa.int64())}))
        assert t.count() == 1


class TestTargetFileSize:
    """OP-15: "2mb" was accepted and then ignored by OPTIMIZE."""

    def test_byte_string_is_honored(self, conn: Any, tmp_path: Any) -> None:
        np = pytest.importorskip("numpy")
        t = conn.create_table(
            str(tmp_path / "t"),
            pa.schema([("id", pa.int64()), ("v", pa.float64())]),
            properties={"delta.targetFileSize": "1mb"},
        )
        for i in range(30):
            t.append(
                pa.table({"id": np.arange(i * 20000, (i + 1) * 20000), "v": np.random.rand(20000)})
            )
        t.optimize()
        assert t.files().num_rows > 1  # delta-rs's 100 MB default made one file

    def test_nonsense_is_refused(self, conn: Any, tmp_path: Any) -> None:
        with pytest.raises(ds.errors.PropertyNotSupportedError, match="size in bytes"):
            conn.create_table(
                str(tmp_path / "t"),
                pa.schema([("id", pa.int64())]),
                properties={"delta.targetFileSize": "big"},
            )


class TestSkipping:
    """OP-7, OP-8: decimal literals and prefix LIKE skip files."""

    @pytest.fixture
    def table(self, conn: Any, tmp_path: Any) -> Any:
        return _many_files(conn, str(tmp_path / "t"))

    @pytest.mark.parametrize(
        ("predicate", "files", "rows"),
        [
            ("id < 30.0", 3, 30),
            ("id < 25.5", 3, 26),
            ("id > 184.5", 2, 15),
            ("id IN (1, 12.0)", 2, 2),
            ("v < 2.5", 2, 20),
            ("s LIKE 'k0%'", 10, 100),
            ("s LIKE 'k01'", 1, 10),
            ("s LIKE 'k1_'", 10, 100),
        ],
    )
    def test_prunes_and_stays_exact(
        self, table: Any, predicate: str, files: int, rows: int
    ) -> None:
        assert len(table.plan_scan(predicate=predicate).splits) == files
        assert table.count(predicate=predicate) == rows
        assert table.to_arrow(predicate=predicate).num_rows == rows

    def test_negated_like_is_not_weakened(self, table: Any) -> None:
        # The range is weaker than the LIKE; beneath a NOT it would skip matches.
        assert table.count(predicate="NOT s LIKE 'k0%'") == 100
        assert table.count(predicate="NOT (id < 25.5)") == 174


class TestCount:
    """OP-9: count() answers from numRecords without opening data files."""

    def test_count_does_not_scan(self, conn: Any, tmp_path: Any, monkeypatch: Any) -> None:
        t = _many_files(conn, str(tmp_path / "t"), props={"delta.enableDeletionVectors": "true"})
        t.delete("id < 5")
        from deltaswamp.engine.kernel import KernelEngine

        def no_scan(*a: Any, **k: Any) -> Any:
            raise AssertionError("count() scanned the table")

        from deltaswamp.engine.deltars import DeltaRsEngine

        monkeypatch.setattr(KernelEngine, "scan", no_scan)
        monkeypatch.setattr(DeltaRsEngine, "scan", no_scan)
        assert t.count() == 195

    def test_partition_predicate_is_exact(self, conn: Any, tmp_path: Any) -> None:
        t = conn.create_table(
            str(tmp_path / "p"),
            pa.schema([("id", pa.int64()), ("p", pa.string())]),
            partition_by=["p"],
        )
        t.append(
            pa.table({"id": pa.array(range(6), pa.int64()), "p": ["a", "a", "b", None, "c", "c"]})
        )
        assert t.count(predicate="p = 'a'") == 2
        assert t.count(predicate="p IS NULL OR p = 'c'") == 3
        assert t.count(predicate="p <> 'a'") == 3  # NULL is not <> 'a'
        assert t.count(predicate="id > 3") == 2  # a data column: scanned


class TestSnapshotReuse:
    """OP-10, OP-5: metadata calls and plan writes reuse the resolved snapshot."""

    def test_metadata_calls_do_not_re_resolve(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        _many_files(conn, str(tmp_path / "t"), n=3)
        counting = _Counting(monkeypatch)
        t2 = conn.table(str(tmp_path / "t"))
        for _ in range(3):
            t2.version, t2.schema(), t2.properties(), t2.protocol(), t2.detail()
        assert counting.resolves <= 1

    def test_cache_sees_other_writers(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = _many_files(conn, path, n=2)
        before = t.count()
        deltalake.write_deltalake(
            path,
            pa.table({"id": pa.array([999], pa.int64()), "v": [1.0], "s": ["z"]}),
            mode="append",
        )
        fresh = conn.table(path)
        assert fresh.count() == before + 1
        assert fresh.version == 3

    def test_cache_sees_a_recreated_table(self, conn: Any, tmp_path: Any) -> None:
        import shutil

        path = str(tmp_path / "t")
        t = _many_files(conn, path, n=2)
        assert t.count() == 20
        shutil.rmtree(path)
        again = conn.create_table(path, pa.schema([("x", pa.string())]))
        for _ in range(3):
            again.append(pa.table({"x": ["a"]}))
        assert conn.table(path).schema().names == ["x"]
        assert conn.table(path).count() == 3

    def test_plan_writes_resolve_once(self, conn: Any, tmp_path: Any, monkeypatch: Any) -> None:
        t = _many_files(conn, str(tmp_path / "t"), n=2)
        plan = t.plan_write()
        counting = _Counting(monkeypatch)
        blob = pickle.dumps(plan)
        fragments = [
            pickle.loads(blob).write(
                pa.table({"id": pa.array([i], pa.int64()), "v": [1.0], "s": ["w"]})
            )
            for i in range(10)
        ]
        assert counting.resolves <= 1
        plan.commit(fragments)
        assert t.count() == 30


class TestLazyHandOffs:
    """OP-6: to_duckdb/to_polars(lazy)/to_pyarrow_dataset/sql push filters down."""

    @pytest.fixture
    def table(self, conn: Any, tmp_path: Any, monkeypatch: Any) -> Any:
        t = _many_files(conn, str(tmp_path / "t"), props={"delta.enableDeletionVectors": "true"})
        t.delete("id = 3")
        self.scans: list[tuple[Any, Any]] = []
        original = type(t).scan
        seen = self.scans

        def scan(self: Any, *a: Any, **k: Any) -> Any:
            seen.append((k.get("columns"), k.get("predicate")))
            return original(self, *a, **k)

        monkeypatch.setattr(type(t), "scan", scan)
        return t

    def test_duckdb_filter_is_pushed(self, table: Any) -> None:
        pytest.importorskip("duckdb")
        rel = table.to_duckdb()
        self.scans.clear()
        assert rel.filter("id < 10").count("*").fetchone() == (9,)
        assert any(p and "id" in p for _, p in self.scans)
        # Re-executable: each run is a fresh scan.
        assert rel.filter("s = 'k01'").project("id").fetchall()[0] == (10,)

    def test_polars_lazy_is_pushed(self, table: Any) -> None:
        pl = pytest.importorskip("polars")
        lf = table.to_polars(lazy=True)
        self.scans.clear()
        out = lf.filter(pl.col("id") < 10).select("s").collect()
        assert out.height == 9
        assert any(p and "id" in p for _, p in self.scans)

    def test_dataset_and_sql(self, conn: Any, table: Any) -> None:
        pytest.importorskip("duckdb")
        pds = pytest.importorskip("pyarrow.dataset")
        d = table.to_pyarrow_dataset()
        assert d.to_table(filter=pds.field("id") < 10).num_rows == 9
        assert d.count_rows() == 199
        for engine in ("duckdb", "polars"):
            pytest.importorskip(engine)
            out = conn.sql(
                "select count(*) as n from t where id < 10 or s = 'k19'",
                tables={"t": table},
                engine=engine,
            )
            assert out.column(0).to_pylist() == [19]

    def test_strings_with_quotes_round_trip(self, conn: Any, tmp_path: Any) -> None:
        duckdb = pytest.importorskip("duckdb")
        t = conn.create_table(str(tmp_path / "q"), pa.schema([("s", pa.string())]))
        values = ["it's", 'say "hi"', "a\\b", "line\nbreak", "tab\there", "x"]
        t.append(pa.table({"s": values}))
        con = duckdb.connect()
        rel = t.to_duckdb(con)
        for value in values:
            lit = "'" + value.replace("'", "''") + "'"
            assert rel.filter(f"s = {lit}").count("*").fetchone() == (1,), value
            assert rel.filter(f"s <> {lit}").count("*").fetchone() == (len(values) - 1,), value


class TestMergeBounds:
    """OP-3, OP-4: MERGE reads only the target files the source keys can match."""

    def test_deltars_merge_scans_one_file(self, conn: Any, tmp_path: Any) -> None:
        t = _many_files(conn, str(tmp_path / "t"))
        assert t.can("merge").engine.value == "deltars"
        src = pa.table({"id": pa.array([5, 7], pa.int64()), "v": [-1.0, -2.0]})
        r = t.merge(src, "target.id = source.id").when_matched_update({"v": "source.v"}).execute()
        assert r["num_target_rows_updated"] == 2
        assert r["num_target_files_scanned"] == 1
        assert sorted(t.to_arrow(predicate="v < 0").column("id").to_pylist()) == [5, 7]

    def test_deltars_merge_by_source_still_sees_every_row(self, conn: Any, tmp_path: Any) -> None:
        t = _many_files(conn, str(tmp_path / "t"))
        src = pa.table({"id": pa.array([5], pa.int64()), "v": [0.0]})
        (
            t.merge(src, "target.id = source.id")
            .when_matched_update({"v": "source.v"})
            .when_not_matched_by_source_delete("target.id >= 190")
            .execute()
        )
        assert t.count() == 190

    def test_kernel_merge_past_the_in_list_limit(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from deltaswamp.engine import kernel_merge

        monkeypatch.setattr(kernel_merge, "_SKIP_VALUES_LIMIT", 3)
        t = _many_files(conn, str(tmp_path / "t"), props={"delta.enableDeletionVectors": "true"})
        assert t.can("merge").engine.value == "kernel"
        src = pa.table({"id": pa.array([20, 21, 22, 23, 24], pa.int64()), "v": [-1.0] * 5})
        skipping = []
        original = kernel_merge.KernelMerger._skipping

        def spy(self: Any, schema: Any) -> Any:
            out = original(self, schema)
            skipping.append(out)
            return out

        monkeypatch.setattr(kernel_merge.KernelMerger, "_skipping", spy)
        t.merge(src, "target.id = source.id").when_matched_update({"v": "source.v"}).execute()
        assert skipping and skipping[0] is not None
        assert t.count(predicate="v < 0") == 5
        assert t.count() == 200

    def test_kernel_merge_by_source_skips_by_its_condition(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from deltaswamp.engine import kernel_merge

        t = _many_files(conn, str(tmp_path / "t"), props={"delta.enableDeletionVectors": "true"})
        skipping = []
        original = kernel_merge.KernelMerger._skipping

        def spy(self: Any, schema: Any) -> Any:
            out = original(self, schema)
            skipping.append(out)
            return out

        monkeypatch.setattr(kernel_merge.KernelMerger, "_skipping", spy)
        src = pa.table({"id": pa.array([5], pa.int64()), "v": [0.0]})
        (
            t.merge(src, "target.id = source.id")
            .when_matched_update({"v": "source.v"})
            .when_not_matched_by_source_delete("target.id >= 190")
            .execute()
        )
        assert skipping and skipping[0] is not None
        assert t.count() == 190
        assert t.count(predicate="id = 5 AND v = 0") == 1


class TestCheckpointStats:
    """OP-12, OP-13: checkpoints keep file statistics, and files() reads them."""

    def test_json_off_records_struct_on(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = _many_files(conn, path, props={"delta.checkpoint.writeStatsAsJson": "false"})
        assert t.properties()["delta.checkpoint.writeStatsAsStruct"] == "true"
        t.checkpoint()
        assert len(conn.table(path).plan_scan(predicate="id < 10").splits) == 1
        files = conn.table(path).files()
        assert files.column("num_records").null_count == 0
        assert "min.id" in files.column_names

    def test_checkpoint_that_would_drop_stats_is_refused(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        _many_files(conn, path, n=3)
        # Another writer sets JSON stats off without struct stats.
        dt = deltalake.DeltaTable(path)
        dt.alter.set_table_properties({"delta.checkpoint.writeStatsAsJson": "false"})
        t = conn.table(path)
        assert not t.can("checkpoint").ok
        with pytest.raises(ds.errors.DeltaSwampError, match="writeStatsAsStruct"):
            t.checkpoint()
        assert not glob.glob(path + "/_delta_log/*.checkpoint.parquet")

    def test_files_reads_stats_parsed(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = conn.create_table(
            path,
            pa.schema([("id", pa.int64())]),
            properties={
                "delta.checkpoint.writeStatsAsJson": "false",
                "delta.checkpoint.writeStatsAsStruct": "true",
                "delta.enableDeletionVectors": "true",
            },
        )
        for i in range(3):
            t.append(pa.table({"id": pa.array([i * 10, i * 10 + 5], pa.int64())}))
        t.checkpoint()
        files = conn.table(path).files()
        assert files.column("num_records").to_pylist() == [2, 2, 2]
        assert sorted(files.column("min.id").to_pylist()) == [0, 10, 20]


class TestStatsColumns:
    """OP-16: nested dataSkippingStatsColumns get statistics."""

    def test_nested_names_get_stats(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        schema = pa.schema([("id", pa.int64()), ("s", pa.struct([("a", pa.int64())]))])
        t = conn.create_table(path, schema, properties={"delta.dataSkippingStatsColumns": "id,s.a"})
        t.append(pa.table({"id": pa.array([1], pa.int64()), "s": [{"a": 7}]}, schema=schema))
        last = sorted(glob.glob(path + "/_delta_log/*.json"))[-1]
        with open(last) as log:
            adds = [json.loads(line)["add"] for line in log if '"add"' in line]
        stats = json.loads(adds[0]["stats"])
        assert stats["minValues"] == {"id": 1, "s": {"a": 7}}


class TestSmall:
    def test_compact_logs_reports_what_it_wrote(self, conn: Any, tmp_path: Any) -> None:
        # OP-22
        path = str(tmp_path / "t")
        t = _many_files(conn, path, n=5)
        deltalake.DeltaTable(path, version=3).create_checkpoint()
        out = t.compact_logs()
        assert out["start"] == 4 and out["end"] == 5
        assert os.path.exists(os.path.join(path, out["path"]))

    def test_balance_is_greedy_and_fast(self) -> None:
        # OP-19: the heap keeps the old assignment.
        from deltaswamp.distributed import balance
        from deltaswamp.engine.base import ScanSplit

        splits = [ScanSplit(path=f"f{i}", size=(i * 37) % 101 + 1) for i in range(20000)]
        bins = balance(splits, 20000)
        assert len(bins) == 20000
        few = balance(splits[:10], 3)
        loads = sorted(sum(s.size for s in b) for b in few)
        assert loads[-1] - loads[0] <= max(s.size for s in splits[:10])

    def test_concurrent_snapshot_cache(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        _many_files(conn, path, n=3)
        errors: list[BaseException] = []

        def read() -> None:
            try:
                for _ in range(5):
                    assert conn.table(path).count() == 30
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=read) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errors
