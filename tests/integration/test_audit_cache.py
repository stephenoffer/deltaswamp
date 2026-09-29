"""Round-4 review: the snapshot cache, lazy pushdown, and kernel MERGE skipping.

Every test here failed before its fix. The S3 tests need two moto servers
(127.0.0.1:5071 and :5072, each with a bucket ``bkt``) and skip otherwise.
"""

from __future__ import annotations

import os
import shutil
import socket
import uuid
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pds = pytest.importorskip("pyarrow.dataset")
ds = pytest.importorskip("deltaswamp")

from deltaswamp.catalog.filesystem import FilesystemCatalog  # noqa: E402
from deltaswamp.engine.kernel import KernelEngine  # noqa: E402
from deltaswamp.errors import DeltaSwampError, MetadataChangedError  # noqa: E402


def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


class _Counting:
    """Counts `Snapshot.resolve` calls: reads of the log from storage."""

    def __init__(self, monkeypatch: Any) -> None:
        import deltaswamp._native as native

        self.resolves = 0
        original = native.Snapshot.resolve
        counter = self

        def resolve(*a: Any, **k: Any) -> Any:
            counter.resolves += 1
            return original(*a, **k)

        monkeypatch.setattr(native.Snapshot, "resolve", staticmethod(resolve))


# --------------------------------------------------------------- the cache


def _pad_to(commit: str, size: int) -> None:
    """Pad a commit file with JSON whitespace (before its last newline) to `size` bytes."""
    with open(commit, "rb") as f:
        body = f.read()
    missing = size - len(body)
    assert missing >= 0, (len(body), size)
    stripped = body.rstrip(b"\n")
    with open(commit, "wb") as f:
        f.write(stripped + b" " * missing + body[len(stripped) :])


class TestRecreatedAtTheSamePath:
    def test_pinned_version_reads_the_new_table(self, conn: Any, tmp_path: Any) -> None:
        # r4 #1: open_table(p, version=1) on the re-created table returned the
        # old schema, applied to the new table's files ([{'a': None}]).
        path = str(tmp_path / "t")
        conn.create_table(path, pa.schema([("a", pa.int64())]))
        conn.open_table(path).append(pa.table({"a": [1, 2, 3]}))
        assert conn.open_table(path, version=1).schema().names == ["a"]
        shutil.rmtree(path)
        conn.create_table(path, pa.schema([("b", pa.string())]))
        conn.open_table(path).append(pa.table({"b": ["x"]}))
        again = ds.connect().open_table(path, version=1)
        assert again.schema().names == ["b"]
        assert again.to_arrow().to_pylist() == [{"b": "x"}]

    def test_latest_read_sees_a_same_size_same_mtime_commit(self, conn: Any, tmp_path: Any) -> None:
        # r4 #3 (latest): the staleness check compared size and Last-Modified,
        # which S3 reports in whole seconds; a same-size commit written in the
        # same second passed it. Simulated locally by restoring the mtime.
        path = str(tmp_path / "t")
        conn.create_table(path, pa.schema([("a", pa.int64())]))
        conn.open_table(path).append(pa.table({"a": [1, 2, 3]}))
        commit = os.path.join(path, "_delta_log", "00000000000000000001.json")
        # Two commits of the same size, made so: the writer's own commits vary
        # by a byte or two (commitInfo's execution_time_ms), which made the
        # sizes differ in most runs and the check this tests go unexercised.
        # JSON allows the whitespace padding either one out to the size.
        size = os.stat(commit).st_size + 256
        _pad_to(commit, size)
        assert conn.open_table(path).count() == 3
        before = os.stat(commit)
        shutil.rmtree(os.path.join(path, "_delta_log"))
        other = ds.connect()
        other.create_table(path, pa.schema([("b", pa.int64())]))
        other.open_table(path).append(pa.table({"b": [7, 8, 9]}))
        _pad_to(commit, size)
        assert os.stat(commit).st_size == before.st_size
        os.utime(commit, ns=(before.st_atime_ns, before.st_mtime_ns))
        fresh = conn.open_table(path)
        assert fresh.schema().names == ["b"]
        assert sorted(r["b"] for r in fresh.to_arrow().to_pylist()) == [7, 8, 9]

    def _recreate(self, path: str) -> None:
        shutil.rmtree(path)
        other = ds.connect()
        other.create_table(path, pa.schema([("b", pa.string())]))
        other.open_table(path).append(pa.table({"b": ["keep-me"]}))

    def test_plan_write_after_a_recreate_is_refused(self, conn: Any, tmp_path: Any) -> None:
        # r4 #2: a guarded overwrite planned at v1 committed at v2 of the
        # re-created table from the cached snapshot: 'keep-me' removed, and a
        # file with column `a` added under schema b.
        path = str(tmp_path / "t")
        conn.create_table(path, pa.schema([("a", pa.int64())]))
        conn.open_table(path).append(pa.table({"a": [1, 2, 3]}))
        plan = conn.open_table(path).plan_write(mode="overwrite")
        plan.write(pa.table({"a": [0]}))  # warms this process's cache
        self._recreate(path)
        with pytest.raises(MetadataChangedError, match="re-created"):
            plan.write(pa.table({"a": [10]}))
        assert ds.connect().open_table(path).to_arrow().to_pylist() == [{"b": "keep-me"}]

    def test_commit_after_a_recreate_is_refused(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        conn.create_table(path, pa.schema([("a", pa.int64())]))
        conn.open_table(path).append(pa.table({"a": [1, 2, 3]}))
        for mode in ("overwrite", "append"):
            if mode == "append":
                shutil.rmtree(path)
                conn.create_table(path, pa.schema([("a", pa.int64())]))
                conn.open_table(path).append(pa.table({"a": [1, 2, 3]}))
            plan = conn.open_table(path).plan_write(mode=mode)
            fragments = [plan.write(pa.table({"a": [10]}))]
            self._recreate(path)
            with pytest.raises(DeltaSwampError):
                plan.commit(fragments, retries=0)
            assert ds.connect().open_table(path).to_arrow().to_pylist() == [{"b": "keep-me"}]

    def test_same_schema_recreate_still_refuses_old_fragments(
        self, conn: Any, tmp_path: Any
    ) -> None:
        # Same layout, different table: only the metaData id tells them apart.
        path = str(tmp_path / "t")
        conn.create_table(path, pa.schema([("a", pa.int64())]))
        conn.open_table(path).append(pa.table({"a": [1]}))
        plan = conn.open_table(path).plan_write()
        fragments = [plan.write(pa.table({"a": [10]}))]
        shutil.rmtree(path)
        other = ds.connect()
        other.create_table(path, pa.schema([("a", pa.int64())]))
        other.open_table(path).append(pa.table({"a": [2]}))
        with pytest.raises(DeltaSwampError, match="different table"):
            plan.commit(fragments, retries=0)
        assert ds.connect().open_table(path).to_arrow().to_pylist() == [{"a": 2}]

    def test_empty_overwrite_after_a_recreate_is_refused(self, conn: Any, tmp_path: Any) -> None:
        # No fragment carries a file, so only the plan's table id can tell.
        path = str(tmp_path / "t")
        conn.create_table(path, pa.schema([("a", pa.int64())]))
        conn.open_table(path).append(pa.table({"a": [1]}))
        plan = conn.open_table(path).plan_write(mode="overwrite")
        shutil.rmtree(path)
        other = ds.connect()
        other.create_table(path, pa.schema([("a", pa.int64())]))
        other.open_table(path).append(pa.table({"a": [2]}))
        with pytest.raises(DeltaSwampError, match="different table"):
            plan.commit([], allow_empty_overwrite=True, allow_concurrent_overwrite=True)
        assert ds.connect().open_table(path).to_arrow().to_pylist() == [{"a": 2}]


class TestCacheReuse:
    def test_commits_never_reuse_a_cached_snapshot(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        path = str(tmp_path / "t")
        t = conn.create_table(path, pa.schema([("a", pa.int64())]))
        t.append(pa.table({"a": [1]}))
        engine = KernelEngine()
        resolved = conn.open_table(path)._resolved
        engine.snapshot(resolved)  # cached
        counting = _Counting(monkeypatch)
        engine.snapshot(resolved)
        assert counting.resolves == 0  # reads do reuse it
        engine.snapshot(resolved, write=True)
        assert counting.resolves == 1  # a commit's snapshot is read afresh

    def test_pinned_writes_still_reuse(self, conn: Any, tmp_path: Any, monkeypatch: Any) -> None:
        # OP-5 kept: every WritePlan.write() after the first revalidates the
        # cached snapshot instead of replaying the log.
        path = str(tmp_path / "t")
        t = conn.create_table(path, pa.schema([("a", pa.int64())]))
        t.append(pa.table({"a": [1]}))
        plan = conn.open_table(path).plan_write()
        counting = _Counting(monkeypatch)
        fragments = [plan.write(pa.table({"a": [i]})) for i in range(5)]
        assert counting.resolves <= 1
        plan.commit(fragments)
        assert conn.open_table(path).count() == 6

    def test_store_options_are_part_of_the_key(self, conn: Any, tmp_path: Any) -> None:
        # r4 #3: the key was (location, version), so two connections to one
        # URL on different stores shared entries.
        path = str(tmp_path / "t")
        conn.create_table(path, pa.schema([("a", pa.int64())]))
        conn.open_table(path).append(pa.table({"a": [1]}))
        with KernelEngine._snapshots_lock:
            KernelEngine._snapshots.clear()
        for options in ({"timeout": "30s"}, {"timeout": "31s"}):
            engine = KernelEngine(storage_options=options)
            engine.snapshot(conn.open_table(path)._resolved, version=1)
        keys = [k for k in KernelEngine._snapshots if k[0] == path and k[1] == 1]
        assert len(keys) == 2
        assert all("30s" not in repr(k) and "31s" not in repr(k) for k in keys)

    def test_no_secret_in_the_key(self) -> None:
        from deltaswamp.engine.kernel import _store_fingerprint

        a = _store_fingerprint({"aws_secret_access_key": "hunter2", "aws_endpoint": "x"})
        b = _store_fingerprint({"aws_secret_access_key": "hunter3", "aws_endpoint": "x"})
        assert a != b and "hunter" not in a


_S3 = pytest.mark.skipif(
    not (_listening(5071) and _listening(5072)),
    reason="moto_server is not running on :5071 and :5072",
)


def _s3(port: int) -> dict[str, str]:
    return {
        "aws_endpoint": f"http://127.0.0.1:{port}",
        "aws_access_key_id": "k",
        "aws_secret_access_key": "s",
        "aws_region": "us-east-1",
        "aws_allow_http": "true",
    }


@_S3
class TestTwoEndpoints:
    def test_pinned_reads_do_not_cross_endpoints(self) -> None:
        url = f"s3://bkt/t_{uuid.uuid4().hex[:6]}"
        a = ds.connect(catalog=FilesystemCatalog(), storage_options=_s3(5071))
        b = ds.connect(catalog=FilesystemCatalog(), storage_options=_s3(5072))
        a.create_table(url, pa.schema([("secret_a", pa.string())]))
        a.open_table(url).append(pa.table({"secret_a": ["A-only-row"]}))
        b.create_table(url, pa.schema([("b", pa.int64())]))
        b.open_table(url).append(pa.table({"b": [1]}))
        assert a.open_table(url, version=1).to_arrow().to_pylist() == [{"secret_a": "A-only-row"}]
        assert b.open_table(url, version=1).to_arrow().to_pylist() == [{"b": 1}]
        dead = ds.connect(catalog=FilesystemCatalog(), storage_options=_s3(5099))
        with pytest.raises(Exception):  # noqa: B017 - any failure, not A's metadata
            dead.open_table(url, version=1).schema()

    def test_latest_reads_do_not_cross_endpoints(self) -> None:
        for _ in range(3):
            url = f"s3://bkt/l_{uuid.uuid4().hex[:6]}"
            a = ds.connect(catalog=FilesystemCatalog(), storage_options=_s3(5071))
            b = ds.connect(catalog=FilesystemCatalog(), storage_options=_s3(5072))
            a.create_table(url, pa.schema([("ssn", pa.string())]))
            b.create_table(url, pa.schema([("zip", pa.string())]))
            a.open_table(url).append(pa.table({"ssn": ["111"]}))
            b.open_table(url).append(pa.table({"zip": ["222"]}))
            a.open_table(url).schema()
            assert b.open_table(url).to_arrow().to_pylist() == [{"zip": "222"}]

    def test_kernel_honours_aws_environment(self, monkeypatch: Any) -> None:
        # The kernel's store took the options alone; delta-rs also reads the
        # AWS_* environment, so the two engines read different stores.
        for key, value in {
            "AWS_ENDPOINT_URL": "http://127.0.0.1:5071",
            "AWS_ACCESS_KEY_ID": "k",
            "AWS_SECRET_ACCESS_KEY": "s",
            "AWS_REGION": "us-east-1",
            "AWS_ALLOW_HTTP": "true",
        }.items():
            monkeypatch.setenv(key, value)
        url = f"s3://bkt/env_{uuid.uuid4().hex[:6]}"
        c = ds.connect(catalog=FilesystemCatalog())
        c.create_table(url, pa.schema([("a", pa.int64())]))
        c.open_table(url).append(pa.table({"a": [1]}))
        resolved = c.open_table(url)._resolved
        assert int(KernelEngine().snapshot(resolved).version) == 1


class TestEnvironmentOptions:
    def test_env_fills_gaps_but_never_mixes_credentials(self, monkeypatch: Any) -> None:
        from deltaswamp._storage import engine_options

        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "env-token")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
        both = engine_options(
            {"aws_access_key_id": "mine", "aws_secret_access_key": "x"}, None, "s3://b/t"
        )
        assert both["aws_endpoint"] == "http://127.0.0.1:9"
        assert both["aws_access_key_id"] == "mine"
        assert "aws_session_token" not in both
        bare = engine_options(None, None, "s3://b/t")
        assert bare["aws_access_key_id"] == "env-key"
        assert bare["aws_session_token"] == "env-token"
        assert (
            engine_options({"aws_endpoint": "http://mine"}, None, "s3://b/t")["aws_endpoint"]
            == "http://mine"
        )
        assert "aws_endpoint" not in engine_options(None, None, "/local/path")


# ------------------------------------------------------------ lazy pushdown


@pytest.fixture
def lazy_table(conn: Any, tmp_path: Any) -> Any:
    data = pa.table(
        {
            "id": pa.array([1, 2], pa.int64()),
            "h": pa.array([b"\x01\x02", b"ab"]),
            "f": pa.array([0.1, 2.0], pa.float32()),
            "x": [float("nan"), 0.0],
        }
    )
    t = conn.create_table(str(tmp_path / "t"), data.schema)
    t.append(data)
    return conn.open_table(str(tmp_path / "t"))


class TestLazyPushdown:
    def test_binary_literal(self, conn: Any, lazy_table: Any) -> None:
        # r4 #4: b'ab' printed as "6162" and was pushed as the string '6162'.
        pytest.importorskip("duckdb")
        for q, want in [
            ("SELECT id FROM t WHERE h = 'ab'::BLOB", [2]),
            ("SELECT id FROM t WHERE h = '\\x01\\x02'::BLOB", [1]),
            ("SELECT id FROM t WHERE h IN ('ab'::BLOB)", [2]),
        ]:
            got = conn.sql(q, tables={"t": lazy_table}).column("id").to_pylist()
            assert got == want, q
        d = lazy_table.to_pyarrow_dataset()
        assert d.to_table(filter=pds.field("h") == pa.scalar(b"ab")).num_rows == 1

    def test_float32_literal(self, lazy_table: Any) -> None:
        # r4 #5: float32(0.1) was pushed as the decimal 0.1.
        f = pds.field("f") == pa.scalar(0.1, pa.float32())
        assert lazy_table.to_pyarrow_dataset().to_table(filter=f).num_rows == 1

    def test_negation_keeps_nan(self, lazy_table: Any) -> None:
        # r4 #6: NOT (x > 1) is false for NaN in Spark, true in pyarrow.
        e = ~(pds.field("x") > 1)
        assert lazy_table.to_pyarrow_dataset().to_table(filter=e).num_rows == 2

    def test_polars_lazy_uses_polars_semantics(self, lazy_table: Any) -> None:
        # r4 #7: NaN is the largest float in Polars; the lazy frame dropped it.
        pl = pytest.importorskip("polars")
        lazy = lazy_table.to_polars(lazy=True).filter(pl.col("x") > 1).collect()
        eager = lazy_table.to_polars().filter(pl.col("x") > 1)
        assert lazy.height == eager.height == 1

    def test_polars_sql_uses_polars_semantics(self, conn: Any, lazy_table: Any) -> None:
        pytest.importorskip("polars")
        out = conn.sql("SELECT id FROM t WHERE x > 1", tables={"t": lazy_table}, engine="polars")
        assert out.column("id").to_pylist() == [1]

    def test_what_is_pushed(self) -> None:
        import datetime
        import decimal

        from deltaswamp._lazy import expression_to_sql

        f = pds.field
        schema = pa.schema(
            [
                ("i", pa.int32()),
                ("s", pa.string()),
                ("d", pa.date32()),
                ("m", pa.decimal128(5, 2)),
                ("f", pa.float64()),
                ("h", pa.binary()),
            ]
        )
        assert expression_to_sql(f("i") < 10, schema) == "(`i` < 10)"
        assert expression_to_sql((f("i") < 10) & (f("f") > 1), schema) == "(`i` < 10)"
        assert expression_to_sql((f("i") < 10) | (f("f") > 1), schema) is None
        assert expression_to_sql(f("s").isin(["a", "b"]), schema) == "(`s` IN ('a', 'b'))"
        d = pa.scalar(datetime.date(2020, 1, 2))
        assert expression_to_sql(f("d") >= d, schema) == "(`d` >= DATE '2020-01-02')"
        m = pa.scalar(decimal.Decimal("1.50"))
        assert expression_to_sql(f("m") == m, schema) == "(`m` = 1.50)"
        for never in (
            ~(f("i") > 1),
            f("f") > 1,
            f("h") == b"ab",
            f("i") == pa.scalar(1.0),
            f("i") < 2**40,
            f("f").is_null(nan_is_null=True),
        ):
            assert expression_to_sql(never, schema) is None, never

    def test_polars_pushes_only_safe_conjuncts(self) -> None:
        pl = pytest.importorskip("polars")
        from deltaswamp._lazy import polars_to_sql

        schema = pa.schema([("i", pa.int64()), ("f", pa.float64()), ("s", pa.string())])
        both = (pl.col("i") < 10) & (pl.col("f") > 1)
        assert polars_to_sql(both, schema) == "(`i` < 10)"
        assert polars_to_sql(~(pl.col("i") > 1), schema) is None
        assert polars_to_sql(pl.col("f") > 0.5, schema) is None
        assert polars_to_sql(pl.col("s") == "x", schema) == "(`s` = 'x')"


# ------------------------------------------------------------- kernel MERGE


def test_kernel_merge_skipping_follows_on_coercion(conn: Any, tmp_path: Any) -> None:
    # r4 #8: the bound cast source key 1 to '1', skipping the file with '01',
    # which the ON clause matches ('01' = 1): a duplicate was inserted.
    path = str(tmp_path / "t")
    conn.create_table(
        path,
        pa.schema([("k", pa.string()), ("v", pa.string())]),
        properties={"delta.enableDeletionVectors": "true"},
    )
    conn.open_table(path).append(pa.table({"k": ["01", "02"], "v": ["a", "b"]}))
    conn.open_table(path).append(pa.table({"k": ["5"], "v": ["c"]}))
    t = conn.open_table(path)
    assert t.can("merge").engine.value == "kernel"
    result = (
        t.merge(
            pa.table({"k": [1, 5], "v": ["A", "C"]}),
            "t.k = s.k",
            source_alias="s",
            target_alias="t",
        )
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute()
    )
    assert result["num_target_rows_inserted"] == 0
    assert conn.open_table(path).count() == 3
