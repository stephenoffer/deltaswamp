"""Regressions for the calendar, INT96 and lazy hand-off defects of audit round 5 (HE-*).

Every test here failed before its fix.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import warnings
from typing import Any

import pytest
from deltaswamp.errors import EngineFallbackWarning, MetadataChangedError
from deltaswamp.predicate import PredicateError

from tests.integration.test_audit_dialect import _field, _protocol, _variant_table, _write_log
from tests.integration.test_audit_read import _spark_table
from tests.integration.test_audit_rebase import _ts_table

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

UTC = dt.UTC
EXPECTED_DATES = [dt.date(1, 1, 1), dt.date(1500, 6, 15), dt.date(2024, 1, 1)]


class TestReadsOfMisreadTables:
    """HE-1: a predicate outside the kernel's grammar moved the read to delta-rs, which misreads."""

    @pytest.mark.parametrize(
        ("predicate", "expected"),
        [
            ("length(cast(rid AS STRING)) > 0", [1, 2, 3]),
            ("year(dt) = 1500", [2]),
            ("dt = DATE '1500-06-15' OR abs(rid) < 0", [2]),
            ("rid % 2 = 1", [1, 3]),
        ],
    )
    def test_the_kernel_reads_and_duckdb_filters(
        self, conn: Any, tmp_path: Any, predicate: str, expected: list[int]
    ) -> None:
        pytest.importorskip("duckdb")
        t = conn.open_table(_spark_table(tmp_path / "t"))
        assert t.can("scan", predicate=predicate).engine is ds.Engine.KERNEL
        with warnings.catch_warnings():
            warnings.simplefilter("error", EngineFallbackWarning)
            rows = t.to_arrow(predicate=predicate).sort_by("rid").to_pylist()
        assert [r["rid"] for r in rows] == expected
        by_rid = dict(zip([1, 2, 3], EXPECTED_DATES, strict=True))
        assert [r["dt"] for r in rows] == [by_rid[r] for r in expected]
        projected = t.to_arrow(columns=["dt"], predicate=predicate)
        assert projected.column_names == ["dt"]
        assert t.count(predicate=predicate) == len(expected)

    def test_delta_rs_never_reads_such_a_table(self, conn: Any, tmp_path: Any) -> None:
        t = conn.open_table(_spark_table(tmp_path / "t"))
        verdict = conn.router.capability(
            ds.Operation.SCAN,
            t._enrich(),
            needs=frozenset({"predicates", "sql_expressions"}),
            exclude=frozenset({ds.Engine.KERNEL}),
            predicate="abs(rid) > 0",
        )
        assert not verdict.ok
        assert "deltars" in verdict.reason and "shifted values" in verdict.reason

    def test_proleptic_tables_still_filter_on_delta_rs(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "p")
        conn.write_table(path, pa.table({"rid": pa.array([1, 2, 3], pa.int64())}))
        t = conn.open_table(path)
        assert t.can("scan", predicate="abs(rid) > 1").engine is ds.Engine.DELTARS


def _with_stats(path: str, stats: dict[str, Any]) -> None:
    log = pathlib.Path(path) / "_delta_log" / f"{0:020d}.json"
    actions = [json.loads(line) for line in log.read_text().splitlines()]
    for action in actions:
        if "add" in action:
            action["add"]["stats"] = json.dumps(stats)
    log.write_text("\n".join(json.dumps(a) for a in actions) + "\n")


class TestInt96PastNanoseconds:
    """HE-2: INT96 values after 2262 overflowed on delta-rs, and its rewrites stored the garbage."""

    def _table(self, tmp_path: Any) -> tuple[str, list[int]]:
        values = [253402300799000000, 10413792000000000, 1704067200000000]  # 9999, 2300, 2024
        path = _ts_table(tmp_path / "t", values, {"org.apache.spark.version": "3.5.0"}, int96=True)
        _with_stats(
            path,
            {
                "numRecords": 3,
                "minValues": {"rid": 0, "ts": "2024-01-01T00:00:00.000Z"},
                "maxValues": {"rid": 2, "ts": "9999-12-31T23:59:59.000Z"},
                "nullCount": {"rid": 0, "ts": 0},
            },
        )
        return path, values

    def test_rewrites_and_reads_avoid_delta_rs(self, conn: Any, tmp_path: Any) -> None:
        path, values = self._table(tmp_path)
        t = conn.open_table(path)
        for operation in ("optimize", "update", "delete"):
            assert t.can(operation).engine is not ds.Engine.DELTARS, operation
        assert "2262-04-11" in t.can("optimize").reason
        t.delete("rid = 2")
        read = conn.open_table(path).to_arrow().sort_by("rid")
        assert read.column("ts").cast(pa.int64()).to_pylist() == values[:2]

    def test_modern_int96_is_still_compacted(self, conn: Any, tmp_path: Any) -> None:
        values = [1704067200000000]
        path = _ts_table(tmp_path / "t", values, {"org.apache.spark.version": "3.5.0"}, int96=True)
        _with_stats(
            path,
            {
                "numRecords": 1,
                "minValues": {"rid": 0, "ts": "2024-01-01T00:00:00.000Z"},
                "maxValues": {"rid": 0, "ts": "2024-01-01T00:00:00.000Z"},
                "nullCount": {"rid": 0, "ts": 0},
            },
        )
        # The kernel commits every compaction now (see test_audit_compact.py);
        # modern INT96 values are not refused.
        assert conn.open_table(path).can("optimize").engine is ds.Engine.KERNEL


class TestLazyHandOffsPinTheirVersion:
    """HE-3: every scan of one frame or relation re-resolved the latest version."""

    def _table(self, conn: Any, tmp_path: Any) -> Any:
        path = str(tmp_path / "t")
        conn.write_table(path, pa.table({"id": pa.array([1, 2], pa.int64()), "s": ["a", "b"]}))
        return conn.open_table(path)

    def _overwrite(self, conn: Any, t: Any) -> None:
        conn.open_table(t.location).overwrite(
            pa.table({"id": pa.array(["x"]), "q": pa.array([1.5])}), schema_mode="overwrite"
        )

    def test_frames_read_the_version_they_were_made_at(self, conn: Any, tmp_path: Any) -> None:
        pl = pytest.importorskip("polars")
        duckdb = pytest.importorskip("duckdb")
        t = self._table(conn, tmp_path)
        lf = t.to_polars(lazy=True)
        dataset = t.to_pyarrow_dataset()
        relation = t.to_duckdb(duckdb.connect())
        conn.open_table(t.location).append(pa.table({"id": pa.array([3], pa.int64()), "s": ["c"]}))
        self._overwrite(conn, t)
        assert sorted(lf.select(pl.col("id") + 1).collect()["id"].to_list()) == [2, 3]
        assert dataset.to_table().column_names == ["id", "s"]
        assert dataset.count_rows() == 2
        assert sorted(r[0] for r in relation.project("id").fetchall()) == [1, 2]

    def test_following_the_latest_raises_on_a_new_schema(self, conn: Any, tmp_path: Any) -> None:
        t = self._table(conn, tmp_path)
        dataset = t.to_pyarrow_dataset(follow_latest=True)
        conn.open_table(t.location).append(pa.table({"id": pa.array([3], pa.int64()), "s": ["c"]}))
        assert dataset.count_rows() == 3
        self._overwrite(conn, t)
        with pytest.raises(MetadataChangedError, match="schema changed"):
            dataset.to_table()


class TestVariantFilters:
    """HE-5: a filter on a VARIANT column (JSON text) was pushed against its binary struct."""

    def test_lazy_filters_on_variants_are_not_pushed(self, conn: Any, tmp_path: Any) -> None:
        import pyarrow.compute as pc

        t = conn.open_table(_variant_table(tmp_path, dv=False))
        table = t.to_pyarrow_dataset().to_table(filter=pc.field("v") == '"s"')
        assert table.column("id").to_pylist() == [2]
        pl = pytest.importorskip("polars")
        frame = t.to_polars(lazy=True).filter(pl.col("v") == '"s"').collect()
        assert frame["id"].to_list() == [2]

    def test_a_predicate_comparing_a_variant_is_a_predicate_error(
        self, conn: Any, tmp_path: Any
    ) -> None:
        t = conn.open_table(_variant_table(tmp_path, dv=False))
        with pytest.raises(PredicateError, match="VARIANT column cannot be compared"):
            t.to_arrow(predicate="v = '\"s\"'")


class TestDuckDBFilterSemantics:
    """HE-6: DuckDB's pushed filters were applied by pyarrow: NaN and -0.0 answered wrongly."""

    @pytest.mark.parametrize(
        ("where", "expected"),
        [("x > 1", [2, 3]), ("x >= 0", [1, 2, 3, 4]), ("x IN (0, 1.5)", [1, 4])],
    )
    def test_duckdb_semantics(
        self, conn: Any, tmp_path: Any, where: str, expected: list[int]
    ) -> None:
        duckdb = pytest.importorskip("duckdb")
        path = str(tmp_path / "t")
        data = pa.table(
            {
                "id": pa.array([1, 2, 3, 4], pa.int64()),
                "x": pa.array([0.0, float("nan"), 2.0, -0.0], pa.float64()),
            }
        )
        conn.write_table(path, data)
        t = conn.open_table(path)
        con = duckdb.connect()
        # A DuckDB table, not an Arrow scan: DuckDB pushes into pyarrow there too.
        con.register("src", data)
        con.execute("CREATE TABLE ref AS SELECT * FROM src")
        want = sorted(r[0] for r in con.sql(f"SELECT id FROM ref WHERE {where}").fetchall())
        assert want == expected
        got = sorted(r[0] for r in t.to_duckdb(con).filter(where).project("id").fetchall())
        assert got == expected
        result = conn.sql(f"SELECT id FROM t WHERE {where}", tables={"t": t})
        assert sorted(result.column("id").to_pylist()) == expected


class TestCountThroughLazyHandOffs:
    """r5lv #2: Polars asks for no columns for count(*), and got a frame of no rows."""

    def test_count_star_counts_every_row(self, conn: Any, tmp_path: Any) -> None:
        pl = pytest.importorskip("polars")
        duckdb = pytest.importorskip("duckdb")
        path = str(tmp_path / "t")
        conn.write_table(
            path, pa.table({"id": pa.array([1, 2, None], pa.int64()), "s": ["a", "b", "c"]})
        )
        t = conn.open_table(path)
        for engine in ("polars", "duckdb"):
            for query, want in (
                ("SELECT count(*) AS n FROM m", 3),
                ("SELECT count(1) AS n FROM m", 3),
                ("SELECT count(id) AS n FROM m", 2),
                ("SELECT count(*) AS n FROM m WHERE s = 'a'", 1),
            ):
                result = conn.sql(query, tables={"m": t}, engine=engine)
                assert result.column("n").to_pylist() == [want], (engine, query)
        lf = t.to_polars(lazy=True)
        assert lf.select(pl.len()).collect().item() == 3
        context = pl.SQLContext(m=lf)
        assert context.execute("SELECT count(*) AS n FROM m").collect()["n"].to_list() == [3]
        assert t.to_duckdb(duckdb.connect()).count("*").fetchone()[0] == 3


class TestMapOfVariants:
    """HE-4: a NULL MAP<STRING, VARIANT> failed the kernel read mid-stream.

    Spark stores a VARIANT as ``value, metadata``; the kernel asks for
    ``metadata, value``, and its map reorder made the column non-nullable.
    """

    @pytest.mark.parametrize("disk_order", [("value", "metadata"), ("metadata", "value")])
    def test_a_null_map_reads_as_null(
        self, conn: Any, tmp_path: Any, disk_order: tuple[str, str]
    ) -> None:
        from deltaswamp._variant import encode

        metadata, value = encode("1")
        parts = {"value": value, "metadata": metadata}
        variant = pa.struct([pa.field(n, pa.binary(), nullable=False) for n in disk_order])
        maps = pa.array(
            [[("k", {n: parts[n] for n in disk_order})], None, [("n", None)], []],
            pa.map_(pa.string(), variant),
        )
        data = pa.table({"id": pa.array([0, 1, 2, 3], pa.int64()), "mv": maps})
        kind = {
            "type": "map",
            "keyType": "string",
            "valueType": "variant",
            "valueContainsNull": True,
        }
        schema = {"type": "struct", "fields": [_field("id", "long"), _field("mv", kind)]}
        path = _write_log(
            str(tmp_path / "t"), schema, _protocol(["variantType"], ["variantType"]), data
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", EngineFallbackWarning)
            rows = conn.open_table(path).to_arrow().sort_by("id").to_pylist()
        assert [r["mv"] for r in rows] == [[("k", "1")], None, [("n", None)], []]


def test_a_lazy_read_of_a_vacuumed_version_names_the_missing_file(conn: Any, tmp_path: Any) -> None:
    """HB-3: the lazy hand-offs raised ArrowInvalid where to_arrow() raised MissingDataFileError."""
    from deltaswamp.errors import MissingDataFileError

    pl = pytest.importorskip("polars")
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": pa.array([1, 2], pa.int64())}))
    conn.open_table(path).overwrite(pa.table({"id": pa.array([3], pa.int64())}))
    pinned = conn.open_table(path, version=1)
    frame, dataset = pinned.to_polars(lazy=True), pinned.to_pyarrow_dataset()
    log = pathlib.Path(path) / "_delta_log" / f"{1:020d}.json"
    for line in log.read_text().splitlines():
        if "add" in (action := json.loads(line)):
            (pathlib.Path(path) / action["add"]["path"]).unlink()
    with pytest.raises(MissingDataFileError):
        dataset.to_table()
    # Polars wraps whatever an IO source raises; the message keeps the type.
    with pytest.raises(pl.exceptions.ComputeError, match="MissingDataFileError"):
        frame.collect()
