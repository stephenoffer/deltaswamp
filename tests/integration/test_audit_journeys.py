"""Regressions for defects found by walking whole user journeys.

Every test here failed before its fix: a follower that never advanced, a
table one ALTER left unwritable, data no other engine could read.
"""

from __future__ import annotations

from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

from deltaswamp.errors import (  # noqa: E402
    ChangeFeedSchemaChangeError,
    DeltaSwampError,
    EngineLimitError,
)


@pytest.fixture
def conn() -> Any:
    return ds.connect("file://")


def _cdf_table(conn: Any, path: str, **properties: str) -> Any:
    props = {"delta.enableChangeDataFeed": "true", **properties}
    return conn.create_table(path, pa.schema([("id", pa.int64())]), properties=props)


class TestChangeFeedAcrossASchemaChange:
    """changes() read start..latest as one range, and a range crossing an ADD
    COLUMN failed on every poll: the follower never got past it."""

    def test_follower_advances_past_add_column(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = _cdf_table(conn, path, **{"delta.columnMapping.mode": "name"})
        t.append(pa.table({"id": [1, 2]}))
        t.add_column(pa.schema([("x", pa.string())]))
        t.append(pa.table({"id": [3], "x": ["a"]}))
        got = list(conn.open_table(path).changes(0))
        assert [(v, b.num_rows) for v, b in got] == [(1, 2), (3, 1)]
        assert "x" not in got[0][1].column_names
        assert got[1][1].column("x").to_pylist() == ["a"]

    def test_cdf_reads_older_rows_under_the_new_schema(self, conn: Any, tmp_path: Any) -> None:
        # Spark reads a batch feed across an additive change, older rows null.
        path = str(tmp_path / "t")
        t = _cdf_table(conn, path, **{"delta.columnMapping.mode": "name"})
        t.append(pa.table({"id": [1, 2]}))
        t.add_column(pa.schema([("x", pa.string())]))
        t.append(pa.table({"id": [3], "x": ["a"]}))
        feed = conn.open_table(path).cdf(starting_version=0).read_all()
        assert feed.column_names[:2] == ["id", "x"]
        rows = sorted(feed.select(["id", "x"]).to_pylist(), key=lambda r: r["id"])
        assert rows == [{"id": 1, "x": None}, {"id": 2, "x": None}, {"id": 3, "x": "a"}]

    def test_dropped_column_names_the_version(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = conn.create_table(
            path,
            pa.schema([("id", pa.int64()), ("x", pa.string())]),
            properties={"delta.enableChangeDataFeed": "true", "delta.columnMapping.mode": "name"},
        )
        t.append(pa.table({"id": [1], "x": ["a"]}))
        t.drop_column("x")
        t.append(pa.table({"id": [2]}))
        with pytest.raises(ChangeFeedSchemaChangeError) as caught:
            conn.open_table(path).cdf(starting_version=0)
        assert caught.value.version == 2
        assert "from version 2 on" in str(caught.value)
        # A follower yields each version under the schema it was written with.
        got = list(conn.open_table(path).changes(0))
        assert [v for v, _ in got] == [1, 3]

    def test_parquet_decode_error_is_not_a_schema_change(self, conn: Any, tmp_path: Any) -> None:
        # arrow-rs's "cannot skip miniblock of size 256" on a Photon file was
        # reported as a schema change: words in the traceback the C stream
        # embeds matched.
        path = str(tmp_path / "t")
        t = _cdf_table(conn, path, **{"delta.columnMapping.mode": "name"})
        t.append(pa.table({"id": [1]}))
        t.add_column(pa.schema([("x", pa.string())]))
        exc = OSError(
            "Invalid: C Data interface error: External error: Arrow error: Parquet argument "
            "error: Parquet error: cannot skip miniblock of size 256. Detail: Python "
            "exception: Traceback ... schema ... cast"
        )
        assert conn.open_table(path)._feed_schema_change(exc, 0, None) is None


class TestMiniblockErrorIsTyped:
    def test_miniblock_error_is_an_engine_limit(self, conn: Any, tmp_path: Any) -> None:
        from deltaswamp.engine import base

        path = str(tmp_path / "t")
        t = _cdf_table(conn, path)
        t.append(pa.table({"id": [1]}))
        captured: dict[str, Any] = {}
        real = base.translating_stream

        def spy(source: Any, context: str, translate: Any = None) -> Any:
            captured["translate"] = translate
            return real(source, context, translate)

        import deltaswamp.table as table_module

        table_module.translating_stream = spy
        try:
            conn.open_table(path).cdf(starting_version=0, ending_version=1)
        finally:
            table_module.translating_stream = real
        error = captured["translate"](OSError("Parquet error: cannot skip miniblock of size 256"))
        assert isinstance(error, EngineLimitError)
        assert isinstance(error, DeltaSwampError)
        assert "allow_sql_fallback" in str(error)


def _dv_table(conn: Any, path: str, **properties: str) -> Any:
    props = {"delta.enableDeletionVectors": "true", **properties}
    t = conn.create_table(path, pa.schema([("id", pa.int64())]), properties=props)
    t.append(pa.table({"id": [1, 2, 3]}))
    return t


def _race(monkeypatch: Any, concurrent: Any) -> None:
    """Run `concurrent` once, just after the next DV DML has read its snapshot."""
    from deltaswamp.engine.kernel import KernelEngine

    real = KernelEngine._dv_dml
    state = {"done": False}

    def racing_snapshot(self: Any, table: Any, **kwargs: Any) -> Any:
        snap = real_snapshot(self, table, **kwargs)
        if kwargs.get("write") and not state["done"]:
            state["done"] = True
            concurrent()
        return snap

    real_snapshot = KernelEngine.snapshot

    def dml(self: Any, *args: Any, **kwargs: Any) -> Any:
        monkeypatch.setattr(KernelEngine, "snapshot", racing_snapshot)
        try:
            return real(self, *args, **kwargs)
        finally:
            monkeypatch.setattr(KernelEngine, "snapshot", real_snapshot)

    monkeypatch.setattr(KernelEngine, "_dv_dml", dml)


class TestDeletionVectorDmlSurvivesBlindAppends:
    """A kernel DV DELETE/UPDATE failed CommitConflictError on any concurrent
    append (23 of 24 deletes, with eight appenders); delta-rs retried."""

    def test_delete_rebases_over_a_blind_append(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        path = str(tmp_path / "t")
        t = _dv_table(conn, path)
        assert "kernel" in str(t.can("delete"))
        other = ds.connect("file://")
        _race(monkeypatch, lambda: other.open_table(path).append(pa.table({"id": [1, 9]})))
        result = conn.open_table(path).delete("id = 1")
        assert result["num_deleted_rows"] == 1
        # WriteSerializable: the appended id 1 was never read, so it stays.
        ids = sorted(conn.open_table(path).to_arrow().column("id").to_pylist())
        assert ids == [1, 2, 3, 9]

    def test_update_rebases_over_a_blind_append(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        path = str(tmp_path / "t")
        _dv_table(conn, path)
        other = ds.connect("file://")
        _race(monkeypatch, lambda: other.open_table(path).append(pa.table({"id": [7]})))
        conn.open_table(path).update(new_values={"id": 20}, predicate="id = 2")
        ids = sorted(conn.open_table(path).to_arrow().column("id").to_pylist())
        assert ids == [1, 3, 7, 20]

    def test_concurrent_delete_of_the_same_file_conflicts(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from deltaswamp.errors import CommitConflictError

        path = str(tmp_path / "t")
        _dv_table(conn, path)
        other = ds.connect("file://")
        _race(monkeypatch, lambda: other.open_table(path).delete("id = 3"))
        with pytest.raises(CommitConflictError) as caught:
            conn.open_table(path).delete("id = 1")
        assert "staged file" not in str(caught.value)
        assert "txnId" not in str(caught.value)
        ids = sorted(conn.open_table(path).to_arrow().column("id").to_pylist())
        assert ids == [1, 2]

    def test_serializable_isolation_conflicts_on_an_append(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from deltaswamp.errors import CommitConflictError

        path = str(tmp_path / "t")
        _dv_table(conn, path).set_properties({"delta.isolationLevel": "Serializable"})
        other = ds.connect("file://")
        _race(monkeypatch, lambda: other.open_table(path).append(pa.table({"id": [1]})))
        with pytest.raises(CommitConflictError):
            conn.open_table(path).delete("id = 1")


def _legacy_cdf_table(conn: Any, path: str) -> Any:
    t = conn.create_table(path, pa.schema([("id", pa.int64()), ("city", pa.string())]))
    t.append(pa.table({"id": [1], "city": ["a"]}))
    t.set_properties({"delta.enableChangeDataFeed": "true"})
    t = conn.open_table(path)
    assert t.protocol() == (1, 4)
    return t


class TestNoAlterStrandsALegacyTable:
    """One ALTER moved a writer-4 change-feed table to writer 7 listing
    checkConstraints and generatedColumns (as Databricks lists them), which the
    kernel refuses, plus a feature delta-rs cannot write: no local engine could
    append to it again."""

    @pytest.mark.parametrize(
        "alter, shape",
        [
            ("cluster_by", {}),
            ("set_properties", {"properties": {"delta.enableTypeWidening": "true"}}),
            ("add_feature", {"features": ["typeWidening"]}),
        ],
    )
    def test_refused_before_anything_is_committed(
        self, conn: Any, tmp_path: Any, alter: str, shape: dict[str, Any]
    ) -> None:
        from deltaswamp.errors import UnreachableTableError

        path = str(tmp_path / "t")
        t = _legacy_cdf_table(conn, path)
        verdict = t.can(alter, **shape)
        assert not verdict.ok
        assert "no local engine could write" in verdict.reason
        with pytest.raises(UnreachableTableError, match="no local engine could write"):
            if alter == "cluster_by":
                t.cluster_by(["city"])
            elif alter == "set_properties":
                t.set_properties(shape["properties"])
            else:
                t.add_feature(shape["features"])
        t = conn.open_table(path)
        assert t.protocol() == (1, 4)
        t.append(pa.table({"id": [2], "city": ["b"]}))
        assert t.count() == 2

    def test_a_feature_delta_rs_writes_is_still_allowed(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = _legacy_cdf_table(conn, path)
        assert t.can("add_feature", features=["deletionVectors"]).ok
        t.add_feature("deletionVectors")
        conn.open_table(path).append(pa.table({"id": [2], "city": ["b"]}))


class TestAddFeatureCapabilityAgrees:
    """can("add_feature", features=["generatedColumns"]) said "via kernel", and
    the call refused -- even where the legacy protocol already implied it."""

    def test_implied_feature_is_a_no_op(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = _legacy_cdf_table(conn, path)
        before = t.version
        assert t.can("add_feature", features=["generatedColumns"]).ok
        t.add_feature("generatedColumns")
        t = conn.open_table(path)
        assert (t.version, t.protocol()) == (before, (1, 4))

    @pytest.mark.parametrize("feature", ["generatedColumns", "columnMapping"])
    def test_unaddable_feature_is_refused_by_can(
        self, conn: Any, tmp_path: Any, feature: str
    ) -> None:
        from deltaswamp.errors import UnreachableTableError

        path = str(tmp_path / "t")
        t = conn.create_table(path, pa.schema([("id", pa.int64())]))
        verdict = t.can("add_feature", features=[feature])
        assert not verdict.ok
        with pytest.raises(UnreachableTableError):
            t.add_feature(feature)


class TestNanosecondTimestampsAreMicroseconds:
    """pandas datetime64[ns] created timestamp_nanos columns behind delta-rs's
    non-standard timestampNanos feature, which DuckDB and Databricks cannot read."""

    def test_write_table_from_pandas(self, conn: Any, tmp_path: Any) -> None:
        pd = pytest.importorskip("pandas")
        path = str(tmp_path / "t")
        frame = pd.DataFrame(
            {"ts": pd.date_range("2026-01-01", periods=3, freq="h").astype("datetime64[ns]")}
        )
        t = conn.write_table(path, frame, mode="overwrite")
        assert "timestampNanos" not in t.features()
        assert t.schema().field("ts").type == pa.timestamp("us")
        assert t.count() == 3

    def test_create_table_with_ns_schema(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = conn.create_table(
            path, pa.schema([("a", pa.timestamp("ns")), ("b", pa.timestamp("ns", "UTC"))])
        )
        assert "timestampNanos" not in t.features()
        assert t.schema().field("a").type == pa.timestamp("us")
        assert t.schema().field("b").type == pa.timestamp("us", "UTC")


class TestMergeSchemaRouting:
    def test_can_agrees_on_a_deletion_vector_table(self, conn: Any, tmp_path: Any) -> None:
        # can("merge", merge_schema=True) said "via kernel"; the call raised
        # "cannot merge with merge_schema on the kernel".
        path = str(tmp_path / "t")
        t = _dv_table(conn, path)
        verdict = t.can("merge", merge_schema=True)
        assert "kernel" not in str(verdict.engine)
        source = pa.table({"id": [3, 4], "extra": ["x", "y"]})
        (
            conn.open_table(path)
            .merge(source, "t.id = s.id", source_alias="s", target_alias="t", merge_schema=True)
            .when_not_matched_insert_all()
            .execute()
        )
        t = conn.open_table(path)
        assert "extra" in t.schema().names
        assert t.count() == 4

    def test_target_only_assignment_is_refused_before_writing(
        self, conn: Any, tmp_path: Any
    ) -> None:
        # delta-rs fails "No field named flag" on a schema-evolving MERGE that
        # sets a column only the target has -- the SCD2 close-out.
        from deltaswamp.errors import UnreachableTableError

        path = str(tmp_path / "t")
        before = conn.write_table(path, pa.table({"id": [1, 2], "flag": [True, True]})).version
        source = pa.table({"id": [1, 3], "tier": ["g", "s"]})
        merge = (
            conn.open_table(path)
            .merge(source, "t.id = s.id", source_alias="s", target_alias="t", merge_schema=True)
            .when_matched_update(updates={"flag": "false"})
        )
        with pytest.raises(UnreachableTableError, match="flag"):
            merge.execute()
        assert conn.open_table(path).version == before


class TestTxnOnCatalogManagedTables:
    """txn_version() on a catalog-managed table the kernel does not write for
    this principal (the warehouse writes it) fell back to delta-rs, which cannot
    open such a table at all."""

    def test_txn_version_reads_through_the_kernel(self, tmp_path: Any, monkeypatch: Any) -> None:
        if not ds.has_native():
            pytest.skip("native extension not built")
        from deltaswamp import Connection
        from deltaswamp.capability import Capability, Operation
        from deltaswamp.catalog.ossuc import OSSUnityCatalog
        from deltaswamp.engine.kernel import KernelEngine
        from deltaswamp.errors import UnreachableTableError

        from tests import helpers
        from tests.fake_uc import FakeUnityCatalog

        with FakeUnityCatalog(staging_root=str(tmp_path)) as uc:
            conn = Connection(catalog=OSSUnityCatalog(uc.url), router=helpers.direct_router())
            conn.create_catalog("main")
            conn.create_schema("main.sales")
            conn.create_table("main.sales.cm", pa.schema([("id", pa.int64())]))
            conn.table("main.sales.cm").append(pa.table({"id": [1]}), txn=("nightly", 3))

            real = KernelEngine.supports

            def no_writes(self: Any, operation: Any, table: Any, **shape: Any) -> Any:
                if operation is Operation.APPEND:
                    return Capability(operation, ok=False, reason="writes withheld")
                return real(self, operation, table, **shape)

            monkeypatch.setattr(KernelEngine, "supports", no_writes)
            t = conn.table("main.sales.cm")
            assert t.txn_version("nightly") == 3
            assert t.txn_version("other") is None
            # The write itself is refused as can() refuses it, not by the check.
            assert not t.can("append", txn=("nightly", 4)).ok
            with pytest.raises(UnreachableTableError, match="cannot append"):
                t.append(pa.table({"id": [2]}), txn=("nightly", 4))
            # A replay of a committed batch is skipped, as Spark skips it.
            t.append(pa.table({"id": [1]}), txn=("nightly", 3))


class TestAppendsRefuseLossyCasts:
    """An append cast 4.7 to 4 in a BIGINT column and '12' to 12; Spark refuses."""

    @pytest.mark.parametrize(
        "column",
        [
            pa.array([4.7]),
            pa.array(["12"]),
            pa.array([True]),
        ],
    )
    def test_refused(self, conn: Any, tmp_path: Any, column: Any) -> None:
        from deltaswamp.errors import InvalidArgumentError

        path = str(tmp_path / "t")
        t = conn.write_table(path, pa.table({"id": pa.array([1], pa.int64())}))
        with pytest.raises(InvalidArgumentError, match="id"):
            t.append(pa.table({"id": column}))
        assert conn.open_table(path).to_arrow().column("id").to_pylist() == [1]

    def test_decimal_that_would_lose_digits(self, conn: Any, tmp_path: Any) -> None:
        # Surfaced as delta-rs's own SchemaMismatchError.
        import decimal

        from deltaswamp.errors import InvalidArgumentError

        path = str(tmp_path / "t")
        t = conn.write_table(path, pa.table({"amt": pa.array([None], pa.decimal128(10, 2))}))
        data = pa.table({"amt": pa.array([decimal.Decimal("1.234")], pa.decimal128(15, 3))})
        with pytest.raises(InvalidArgumentError):
            t.append(data)

    def test_safe_widening_still_writes(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = conn.write_table(
            path,
            pa.table(
                {"i": pa.array([1], pa.int64()), "f": pa.array([1.0]), "s": pa.array(["a"])}
            ),
        )
        t.append(
            pa.table(
                {
                    "i": pa.array([2], pa.int32()),
                    "f": pa.array([2.5], pa.float32()),
                    "s": pa.array(["b"], pa.large_string()),
                }
            )
        )
        assert conn.open_table(path).count() == 2


class TestRawEngineErrorsAreTyped:
    def test_extra_column_without_merge(self, conn: Any, tmp_path: Any) -> None:
        # delta-rs's SchemaMismatchError, which `except DeltaSwampError` missed.
        path = str(tmp_path / "t")
        t = conn.write_table(path, pa.table({"id": [1]}))
        with pytest.raises(DeltaSwampError):
            t.append(pa.table({"id": [2], "extra": ["x"]}))

    def test_update_to_an_unknown_column_reference(self, conn: Any, tmp_path: Any) -> None:
        # A DeltaError: updates= takes SQL, so "zzz" is a column that is not there.
        from deltaswamp.errors import InvalidArgumentError

        path = str(tmp_path / "t")
        t = conn.write_table(path, pa.table({"id": [1], "name": ["a"]}))
        with pytest.raises(InvalidArgumentError, match="zzz"):
            t.update({"name": "zzz"}, predicate="id = 1")

    def test_storage_failure_is_a_storage_error(self) -> None:
        from deltaswamp.errors import StorageError
        from deltaswamp.table import _library_error

        error = _library_error(OSError("Generic S3 error: 503 Slow Down"), "append")
        assert isinstance(error, StorageError)
        assert isinstance(error, OSError) and isinstance(error, DeltaSwampError)
        assert _library_error(FileNotFoundError("x"), "append") is None


class TestSchemaArgumentsTakeSqlTypeNames:
    def test_dict_and_list_schemas(self, conn: Any, tmp_path: Any) -> None:
        t = conn.create_table(
            str(tmp_path / "a"),
            {
                "id": "bigint",
                "n": "long",
                "amt": "decimal(10,2)",
                "ts": "timestamp",
                "tn": "timestamp_ntz",
                "tags": "array<string>",
                "x": "int64",  # a pyarrow alias still works
            },
        )
        schema = t.schema()
        assert schema.field("id").type == pa.int64() == schema.field("n").type
        assert schema.field("amt").type == pa.decimal128(10, 2)
        assert schema.field("ts").type == pa.timestamp("us", "UTC")
        assert schema.field("tn").type == pa.timestamp("us")
        assert schema.field("tags").type.value_type == pa.string()
        t = conn.create_table(str(tmp_path / "b"), [("id", "long"), ("s", "string")])
        assert t.schema().names == ["id", "s"]


class TestSmallJourneyFixes:
    def test_alter_column_type_to_its_own_type_is_a_no_op(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "t")
        t = conn.create_table(path, pa.schema([("id", pa.int64())]))
        before = t.version
        t.alter_column_type("id", "bigint")
        assert conn.open_table(path).version == before

    def test_replace_where_violation_is_the_same_error_on_both_engines(
        self, conn: Any, tmp_path: Any
    ) -> None:
        from deltaswamp.errors import InvalidArgumentError

        plain = conn.write_table(str(tmp_path / "p"), pa.table({"id": [1, 2]}))
        dv = _dv_table(conn, str(tmp_path / "d"))
        assert "kernel" in str(dv.can("overwrite", predicate="id = 1"))
        for t in (plain, dv):
            with pytest.raises(InvalidArgumentError):
                t.overwrite(pa.table({"id": [5]}), predicate="id = 1")

    def test_zoned_datetime_into_timestamp_ntz_is_refused(
        self, conn: Any, tmp_path: Any
    ) -> None:
        import datetime as dt

        from deltaswamp.errors import InvalidArgumentError

        path = str(tmp_path / "t")
        t = conn.write_table(
            path,
            pa.table({"id": [1], "ts": pa.array([dt.datetime(2026, 1, 1)], pa.timestamp("us"))}),
        )
        zoned = dt.datetime(2027, 5, 6, 7, 8, 9, tzinfo=dt.timezone(dt.timedelta(hours=2)))
        with pytest.raises(InvalidArgumentError, match="TIMESTAMP_NTZ"):
            t.update(new_values={"ts": zoned}, predicate="id = 1")
        assert conn.open_table(path).to_arrow().column("ts").to_pylist() == [
            dt.datetime(2026, 1, 1)
        ]


class TestLazyPolarsPushesDown:
    """to_polars(lazy=True) read the whole table before returning the frame."""

    def test_projection_and_limit_reach_the_scan(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        pl = pytest.importorskip("polars")
        from deltaswamp.table import Table

        path = str(tmp_path / "t")
        conn.write_table(path, pa.table({"id": list(range(50)), "s": [str(i) for i in range(50)]}))
        seen: list[dict[str, Any]] = []
        real = Table.scan

        def spy(self: Any, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return real(self, **kwargs)

        monkeypatch.setattr(Table, "scan", spy)
        lazy = conn.open_table(path).to_polars(lazy=True)
        assert all(k.get("limit") == 0 for k in seen)  # only the schema so far
        seen.clear()
        got = lazy.select("id").filter(pl.col("id") > 45).collect()
        assert got["id"].to_list() == [46, 47, 48, 49]
        assert [k.get("columns") for k in seen] == [["id"]]
        assert lazy.filter(pl.col("id") > 40).head(2).collect()["id"].to_list() == [41, 42]


class TestCreateAtAUriWithAFragment:
    """create_table("file://.../x#y") created the table at .../x, and writes
    through the returned handle landed there, while opening the URI refused."""

    @pytest.mark.parametrize("name", ["x#y", "q?r"])
    def test_refused_before_anything_is_created(
        self, conn: Any, tmp_path: Any, name: str
    ) -> None:
        import os

        from deltaswamp.errors import InvalidArgumentError

        uri = "file://" + str(tmp_path / name)
        with pytest.raises(InvalidArgumentError, match="query or fragment"):
            conn.create_table(uri, pa.schema([("id", pa.int64())]))
        with pytest.raises(InvalidArgumentError):
            conn.write_table(uri, pa.table({"id": [1]}))
        assert os.listdir(tmp_path) == []


class TestRestoreToAFutureTimestamp:
    def test_refused_like_spark(self, conn: Any, tmp_path: Any) -> None:
        # It resolved to the latest version: a silent no-op.
        import datetime as dt

        from deltaswamp.errors import InvalidArgumentError

        path = str(tmp_path / "t")
        t = conn.write_table(path, pa.table({"id": [1]}))
        t.append(pa.table({"id": [2]}))
        with pytest.raises(InvalidArgumentError, match="after the latest commit"):
            t.restore(dt.datetime.now(dt.UTC) + dt.timedelta(days=1))
        assert conn.open_table(path).version == 2
