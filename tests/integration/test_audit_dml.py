"""Regressions for the DML defects the deep audit found, each on a real table.

Every test here failed before its fix. Where the expected result is Spark's,
it was checked against a Databricks SQL warehouse: DuckDB-, DataFusion- and
Arrow-isms that differ from it are the bugs.
"""

from __future__ import annotations

import datetime as dt
import decimal
import glob
import json
import os
import pathlib
import warnings
from typing import Any, ClassVar

import pytest
from deltaswamp.capability import Engine, Operation
from deltaswamp.errors import (
    InvalidArgumentError,
    MissingDataFileError,
    UnreachableTableError,
)

pa = pytest.importorskip("pyarrow")
deltalake = pytest.importorskip("deltalake")
pytest.importorskip("duckdb")
native = pytest.importorskip("deltaswamp._native")

pytestmark = pytest.mark.skipif(
    "deletion_vector_dml" not in native.FEATURES, reason="native build predates DV DML"
)

DV = {"delta.enableDeletionVectors": "true"}
CDF = {"delta.enableChangeDataFeed": "true"}


def _rows(conn: Any, path: str, *columns: str) -> list[tuple[Any, ...]]:
    table = conn.open_table(path).to_arrow()
    names = list(columns) or table.column_names
    return sorted(
        (tuple(r[c] for c in names) for r in table.to_pylist()),
        key=lambda row: tuple((v is None, str(v)) for v in row),
    )


def _deltars_rows(path: str, column: str) -> list[Any]:
    # QueryBuilder applies deletion vectors; to_pyarrow_dataset does not.
    from deltalake import QueryBuilder

    result = (
        QueryBuilder().register("t", deltalake.DeltaTable(path)).execute(f"SELECT {column} FROM t")
    )
    return sorted(pa.table(result.read_all()).column(column).to_pylist())


def _last_commit(path: str) -> list[dict[str, Any]]:
    logs = sorted(pathlib.Path(path, "_delta_log").glob("*.json"))
    return [json.loads(line) for line in logs[-1].read_text().splitlines()]


def _table(
    conn: Any,
    tmp_path: Any,
    data: Any,
    properties: dict[str, str] | None = None,
    name: str = "t",
) -> str:
    path = str(tmp_path / name)
    conn.create_table(path, data.schema, properties=properties or {})
    conn.open_table(path).append(data)
    return path


def _ids(n: int = 3) -> Any:
    return pa.table(
        {
            "id": pa.array(range(1, n + 1), pa.int64()),
            "s": [chr(ord("x") + i % 3) for i in range(n)],
        }
    )


# --------------------------------------------------------------- DM-01


class TestChangeFeedMergeWithConditionalInsert:
    SOURCE = pa.table({"id": pa.array([5, 6, 7], pa.int64()), "s": ["abc", "b", "ab"]})

    def test_deltars_still_writes_null_rows(self, tmp_path: Any) -> None:
        """Probe: when this fails, delta-rs is fixed and the refusal can go."""
        path = str(tmp_path / "t")
        deltalake.write_deltalake(path, _ids(2), configuration=CDF)
        deltalake.DeltaTable(path).merge(
            self.SOURCE, "t.id = s.id", source_alias="s", target_alias="t"
        ).when_not_matched_insert_all(predicate="s.s LIKE 'a%'").execute()
        ids = deltalake.DeltaTable(path).to_pyarrow_table().column("id").to_pylist()
        assert None in ids, (
            "delta-rs no longer writes an all-NULL row per rejected source row on a CDF "
            "table; _CheckedMerger.execute can stop refusing conditional NOT MATCHED clauses"
        )

    def test_refused_before_anything_is_written(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path, _ids(2), CDF)
        t = conn.open_table(path)
        assert t.can(Operation.MERGE).engine is Engine.DELTARS
        assert not t.can(
            Operation.MERGE, clauses=[("when_not_matched_insert_all", "s.s LIKE 'a%'")]
        ).ok
        # Refused by the router as the clauses arrive, as can() refuses it.
        with pytest.raises(UnreachableTableError, match="all-NULL row"):
            t.merge(
                self.SOURCE, "t.id = s.id", source_alias="s", target_alias="t"
            ).when_not_matched_insert_all(predicate="s.s LIKE 'a%'").execute()
        assert _rows(conn, path, "id") == [(1,), (2,)]
        assert conn.open_table(path).version == 1

    def test_an_unconditional_last_clause_is_still_served(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path, _ids(2), CDF)
        conn.open_table(path).merge(
            self.SOURCE, "t.id = s.id", source_alias="s", target_alias="t"
        ).when_not_matched_insert_all(predicate="s.s = 'b'").when_not_matched_insert(
            {"id": "s.id", "s": "'other'"}
        ).execute()
        assert _rows(conn, path) == [
            (1, "x"),
            (2, "y"),
            (5, "other"),
            (6, "b"),
            (7, "other"),
        ]


# --------------------------------------------------------------- DM-07


class TestNotMatchedBySourceWithDuplicateMatches:
    TARGET = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "k": pa.array([1, 2, 3], pa.int64()),
            "v": list("abc"),
        }
    )
    SOURCE = pa.table({"k": pa.array([1, 1, 9], pa.int64())})

    def test_deltars_still_duplicates_the_row(self, tmp_path: Any) -> None:
        """Probe: when this fails, the no-op MATCHED clause workaround can go."""
        path = str(tmp_path / "t")
        deltalake.write_deltalake(path, self.TARGET)
        deltalake.DeltaTable(path).merge(
            self.SOURCE, "t.k = s.k", source_alias="s", target_alias="t"
        ).when_not_matched_by_source_delete().execute()
        assert deltalake.DeltaTable(path).to_pyarrow_table().num_rows == 2, (
            "delta-rs no longer writes a target row once per matching source row; "
            "_CheckedMerger.execute can stop adding its no-op MATCHED clause"
        )

    @pytest.mark.parametrize("properties", [{}, DV], ids=["deltars", "kernel"])
    def test_one_copy_is_kept(self, conn: Any, tmp_path: Any, properties: Any) -> None:
        # Databricks keeps exactly one copy of (1, 1, a).
        path = _table(conn, tmp_path, self.TARGET, properties)
        conn.open_table(path).merge(
            self.SOURCE, "t.k = s.k", source_alias="s", target_alias="t"
        ).when_not_matched_by_source_delete().execute()
        assert _rows(conn, path) == [(1, 1, "a")]

    def test_update_by_source_keeps_one_copy(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path, self.TARGET)
        conn.open_table(path).merge(
            self.SOURCE, "t.k = s.k", source_alias="s", target_alias="t"
        ).when_not_matched_by_source_update({"v": "'z'"}, predicate="t.id = 3").execute()
        assert _rows(conn, path) == [(1, 1, "a"), (2, 2, "b"), (3, 3, "z")]


# --------------------------------------------------------------- DM-08


class TestDottedColumnUpdate:
    PROPS: ClassVar[dict[str, str]] = {"delta.columnMapping.mode": "name"}

    def _path(self, conn: Any, tmp_path: Any, dv: bool) -> str:
        props = {**self.PROPS, **(DV if dv else {})}
        data = pa.table({"Id": pa.array([1, 2], pa.int64()), "a.b": pa.array([10, 20], pa.int64())})
        return _table(conn, tmp_path, data, props)

    @pytest.mark.parametrize("dv", [False, True], ids=["deltars", "kernel"])
    @pytest.mark.parametrize(
        "call",
        [
            lambda t: t.update(new_values={"a.b": 5}, predicate="Id = 1"),
            lambda t: t.update({"a.b": "5"}, predicate="Id = 1"),
            lambda t: t.update({"`a.b`": "5"}, predicate="Id = 1"),
            lambda t: t.update(new_values={"A.B": 5}, predicate="Id = 1"),
        ],
        ids=["new_values", "updates", "backticked", "other case"],
    )
    def test_every_spelling_updates_the_column(
        self, conn: Any, tmp_path: Any, dv: bool, call: Any
    ) -> None:
        path = self._path(conn, tmp_path, dv)
        t = conn.open_table(path)
        assert t.can(Operation.UPDATE).engine is (Engine.KERNEL if dv else Engine.DELTARS)
        result = call(t)
        assert result["num_updated_rows"] == 1
        assert _rows(conn, path) == [(1, 5), (2, 20)]


# --------------------------------------------------------------- DM-10


class TestCheckpointWithoutJsonStats:
    """Databricks writes checkpoints with delta.checkpoint.writeStatsAsJson=false."""

    @pytest.mark.parametrize("row_tracking", [False, True])
    def test_dml_writes_deletion_vectors(
        self, conn: Any, tmp_path: Any, row_tracking: bool
    ) -> None:
        props = {
            **DV,
            "delta.checkpoint.writeStatsAsJson": "false",
            "delta.checkpoint.writeStatsAsStruct": "true",
        }
        if row_tracking:
            props["delta.enableRowTracking"] = "true"
        path = _table(conn, tmp_path, pa.table({"id": pa.array([1, 2, 3, 4], pa.int64())}), props)
        conn.open_table(path).checkpoint()
        kernel = conn.router.engines[Engine.KERNEL]
        files = pa.table(kernel.snapshot(conn.open_table(path).resolved).files())
        # stats_parsed only in the checkpoint; the listing re-serializes it (OP-13).
        assert files.column("num_records").to_pylist() == [4]

        t = conn.open_table(path)
        assert t.can(Operation.DELETE).engine is Engine.KERNEL
        assert t.delete("id = 2")["num_deleted_rows"] == 1
        adds = [a["add"] for a in _last_commit(path) if "add" in a]
        assert [a["deletionVector"]["cardinality"] for a in adds] == [1]
        # The count came from the Parquet footer; the new add records it.
        assert json.loads(adds[0]["stats"])["numRecords"] == 4

        result = conn.open_table(path).update(new_values={"id": 30}, predicate="id = 3")
        assert result.items() >= {"num_updated_rows": 1, "version": 3}.items()
        merged = (
            conn.open_table(path)
            .merge(
                pa.table({"id": pa.array([4], pa.int64())}),
                "t.id = s.id",
                source_alias="s",
                target_alias="t",
            )
            .when_matched_delete()
            .execute()
        )
        assert merged["num_target_rows_deleted"] == 1
        assert _rows(conn, path) == [(1,), (30,)]
        assert _deltars_rows(path, "id") == [1, 30]


# --------------------------------------------------------------- DM-02


class TestDeletionVectorDmlBeyondTheKernelGrammar:
    def _path(self, conn: Any, tmp_path: Any) -> str:
        data = pa.table(
            {
                "id": pa.array([1, 2, 3, 4], pa.int64()),
                "x": pa.array([10, 20, 30, 40], pa.int64()),
                "status": ["a", "B", "c", "d"],
            }
        )
        return _table(conn, tmp_path, data, DV)

    def test_arithmetic_delete_goes_to_deltars(self, conn: Any, tmp_path: Any) -> None:
        path = self._path(conn, tmp_path)
        t = conn.open_table(path)
        assert t.can(Operation.DELETE).engine is Engine.KERNEL
        assert t.can(Operation.DELETE, predicate="id % 2 = 0").engine is Engine.DELTARS
        assert t.delete("id % 2 = 0")["num_deleted_rows"] == 2
        assert _rows(conn, path, "id") == [(1,), (3,)]

    def test_functions_in_update(self, conn: Any, tmp_path: Any) -> None:
        path = self._path(conn, tmp_path)
        t = conn.open_table(path)
        assert t.can(Operation.UPDATE, updates={"x": "x + 1"}).engine is Engine.DELTARS
        t.update({"x": "x + 1", "status": "upper(status)"}, predicate="lower(status) = 'b'")
        assert _rows(conn, path, "id", "x", "status") == [
            (1, 10, "a"),
            (2, 21, "B"),
            (3, 30, "c"),
            (4, 40, "d"),
        ]

    def test_replace_where_with_arithmetic(self, conn: Any, tmp_path: Any) -> None:
        path = self._path(conn, tmp_path)
        new = pa.table(
            {"id": pa.array([2], pa.int64()), "x": pa.array([99], pa.int64()), "status": ["n"]}
        )
        t = conn.open_table(path)
        assert t.can(Operation.OVERWRITE, predicate="id % 2 = 0").engine is Engine.DELTARS
        t.overwrite(new, predicate="id % 2 = 0")
        assert _rows(conn, path, "id", "x") == [(1, 10), (2, 99), (3, 30)]

    def test_documented_literal_update_stays_on_the_kernel(self, conn: Any, tmp_path: Any) -> None:
        # docs/usage.md: t.update({"status": "'archived'"}, predicate=...)
        path = self._path(conn, tmp_path)
        t = conn.open_table(path)
        assert (
            t.can(Operation.UPDATE, updates={"status": "'archived'"}, predicate="id = 1").engine
            is Engine.KERNEL
        )
        t.update({"status": "'archived'"}, predicate="id = 1")
        assert _rows(conn, path, "id", "status")[0] == (1, "archived")
        assert any("deletionVector" in a.get("add", {}) for a in _last_commit(path))


# --------------------------------------------------------------- DM-09


class TestKernelMergeSpeaksSparkSql:
    def _path(self, conn: Any, tmp_path: Any, rows: Any = None) -> str:
        data = rows or pa.table(
            {
                "id": pa.array([1, 2, 3], pa.int64()),
                "v": ["a\\b", "a\\\\b", None],
                "k": pa.array([3, 2, None], pa.int32()),
            }
        )
        return _table(conn, tmp_path, data, DV)

    def _merge(self, conn: Any, path: str, source: Any, on: str = "t.id = s.id") -> Any:
        t = conn.open_table(path)
        assert t.can(Operation.MERGE).engine is Engine.KERNEL
        return t.merge(source, on, source_alias="s", target_alias="t")

    def test_backslash_escapes_match_the_delete_predicate(self, conn: Any, tmp_path: Any) -> None:
        # Spark reads 'a\\b' as a\b: Databricks' MERGE deletes row 1, as DELETE does.
        path = self._path(conn, tmp_path)
        source = pa.table({"id": pa.array([1, 2], pa.int64())})
        self._merge(conn, path, source).when_matched_delete(predicate="t.v = 'a\\\\b'").execute()
        assert _rows(conn, path, "id") == [(2,), (3,)]

    def test_spark_spellings(self, conn: Any, tmp_path: Any) -> None:
        path = self._path(conn, tmp_path)
        source = pa.table({"id": pa.array([1, 3], pa.int64()), "my col": ["O'", None]})
        self._merge(conn, path, source, on="t.id <=> s.id").when_matched_update(
            {"v": "concat(nvl(s.`my col`, \"none\"), 'Brien\\'s')"}
        ).execute()
        assert _rows(conn, path, "id", "v") == [(1, "O'Brien's"), (2, "a\\\\b"), (3, "noneBrien's")]

    def test_concat_with_null_is_null(self, conn: Any, tmp_path: Any) -> None:
        path = self._path(conn, tmp_path)
        source = pa.table({"id": pa.array([3], pa.int64())})
        self._merge(conn, path, source).when_matched_update({"v": "concat(t.v, '!')"}).execute()
        assert _rows(conn, path, "id", "v")[2] == (3, None)

    def test_null_safe_join(self, conn: Any, tmp_path: Any) -> None:
        path = self._path(conn, tmp_path)
        source = pa.table({"k": pa.array([None], pa.int32())})
        self._merge(conn, path, source, on="t.k <=> s.k").when_matched_delete().execute()
        assert _rows(conn, path, "id") == [(1,), (2,)]

    def test_fraction_into_an_integer_column_truncates(self, conn: Any, tmp_path: Any) -> None:
        # Databricks stores 3 / 2 into an INT column as 1.
        path = self._path(conn, tmp_path)
        source = pa.table({"id": pa.array([1], pa.int64())})
        self._merge(conn, path, source).when_matched_update({"k": "t.k / 2"}).execute()
        assert _rows(conn, path, "id", "k")[0] == (1, 1)

    def test_sql_duckdb_cannot_run_goes_to_deltars(self, conn: Any, tmp_path: Any) -> None:
        path = self._path(conn, tmp_path)
        source = pa.table({"id": pa.array([1], pa.int64()), "w": ["hello world"]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._merge(conn, path, source).when_matched_update({"v": "initcap(s.w)"}).execute()
        assert _rows(conn, path, "id", "v")[0] == (1, "Hello World")
        assert any("trying deltars" in str(w.message) for w in caught)


# --------------------------------------------------------------- DM-03


class TestKernelMergeTypes:
    def _path(self, conn: Any, tmp_path: Any) -> str:
        data = pa.table(
            {
                "id": pa.array([1, 2], pa.int64()),
                "ts": pa.array([dt.datetime(2020, 1, 1)] * 2, pa.timestamp("us", tz="UTC")),
                "st": pa.array([{"a": 1}, {"a": 2}], pa.struct([("a", pa.int64())])),
                "b": pa.array([b"x", b"y"], pa.binary()),
                "d": pa.array([decimal.Decimal("1.00")] * 2, pa.decimal128(10, 2)),
            }
        )
        return _table(conn, tmp_path, data, DV)

    @pytest.mark.parametrize("column", ["ts", "st", "b", "d"])
    def test_set_null(self, conn: Any, tmp_path: Any, column: str) -> None:
        path = self._path(conn, tmp_path)
        conn.open_table(path).merge(
            pa.table({"id": pa.array([1], pa.int64())}),
            "t.id = s.id",
            source_alias="s",
            target_alias="t",
        ).when_matched_update({column: "NULL"}).when_not_matched_by_source_update(
            {column: "NULL"}, predicate="t.id = 99"
        ).execute()
        values = {r["id"]: r[column] for r in conn.open_table(path).to_arrow().to_pylist()}
        assert values[1] is None and values[2] is not None

    def test_decimal_rounds_half_up(self, conn: Any, tmp_path: Any) -> None:
        # Databricks stores 12.345 into DECIMAL(10,2) as 12.35, by MERGE and UPDATE.
        path = self._path(conn, tmp_path)
        conn.open_table(path).merge(
            pa.table({"id": pa.array([1], pa.int64())}),
            "t.id = s.id",
            source_alias="s",
            target_alias="t",
        ).when_matched_update({"d": "12.345"}).execute()
        conn.open_table(path).update({"d": "2.355"}, predicate="id = 2")
        assert _rows(conn, path, "id", "d") == [
            (1, decimal.Decimal("12.35")),
            (2, decimal.Decimal("2.36")),
        ]

    @pytest.mark.parametrize("properties", [DV, {}], ids=["kernel", "deltars"])
    @pytest.mark.parametrize("clause", ["insert_all", "update_all"])
    def test_star_clauses_refuse_a_missing_source_column(
        self, conn: Any, tmp_path: Any, properties: Any, clause: str
    ) -> None:
        # Databricks: [DELTA_MERGE_UNRESOLVED_EXPRESSION] Cannot resolve v in ... clause.
        data = pa.table({"id": pa.array([1], pa.int64()), "v": ["a"]})
        path = _table(conn, tmp_path, data, properties)
        merger = conn.open_table(path).merge(
            pa.table({"id": pa.array([1, 2], pa.int64())}),
            "t.id = s.id",
            source_alias="s",
            target_alias="t",
        )
        with pytest.raises(InvalidArgumentError, match="lacks v"):
            if clause == "insert_all":
                merger.when_not_matched_insert_all().execute()
            else:
                merger.when_matched_update_all().execute()
        assert _rows(conn, path) == [(1, "a")]


# ----------------------------------------------------------- DM-04/05/06


class TestKernelMergeEdges:
    @pytest.mark.parametrize("name", ["path", "row_index"])
    def test_target_columns_named_like_positions(self, conn: Any, tmp_path: Any, name: str) -> None:
        data = pa.table({"id": pa.array([1, 2], pa.int64()), name: ["a", "b"]})
        path = _table(conn, tmp_path, data, DV)
        conn.open_table(path).merge(
            pa.table({"id": pa.array([2], pa.int64()), name: ["B"]}),
            "t.id = s.id",
            source_alias="s",
            target_alias="t",
        ).when_matched_update_all().execute()
        assert _rows(conn, path) == [(1, "a"), (2, "B")]

    def test_duplicate_matches_of_a_delete_count_once(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path, _ids(3), DV)
        result = (
            conn.open_table(path)
            .merge(
                pa.table({"id": pa.array([1, 1, 1, 2], pa.int64())}),
                "t.id = s.id",
                source_alias="s",
                target_alias="t",
            )
            .when_matched_delete()
            .execute()
        )
        assert result["num_target_rows_deleted"] == 2
        assert _rows(conn, path, "id") == [(3,)]

    @pytest.mark.parametrize("insert", [False, True])
    def test_duplicates_no_clause_acts_on_are_accepted(
        self, conn: Any, tmp_path: Any, insert: bool
    ) -> None:
        # Databricks accepts this: neither source row for id 1 satisfies a
        # MATCHED condition, so which one "wins" does not arise.
        path = _table(conn, tmp_path, _ids(3), DV)
        source = pa.table(
            {
                "id": pa.array([1, 1, 2], pa.int64()),
                "v": ["p", "q", "z"],
                "flag": [False, False, True],
            }
        )
        merger = (
            conn.open_table(path)
            .merge(source, "t.id = s.id", source_alias="s", target_alias="t")
            .when_matched_update({"s": "s.v"}, predicate="s.flag")
        )
        if insert:
            merger = merger.when_not_matched_insert({"id": "s.id", "s": "s.v"})
        merger.execute()
        assert _rows(conn, path) == [(1, "x"), (2, "z"), (3, "z")]

    def test_duplicates_a_clause_acts_on_are_refused(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path, _ids(3), DV)
        source = pa.table({"id": pa.array([1, 1], pa.int64()), "flag": [True, True]})
        with pytest.raises(InvalidArgumentError, match="more than one source row"):
            conn.open_table(path).merge(
                source, "t.id = s.id", source_alias="s", target_alias="t"
            ).when_matched_update({"s": "'n'"}, predicate="s.flag").execute()


# --------------------------------------------------------- DM-11/12


class TestMergeClauseRules:
    @pytest.mark.parametrize("properties", [DV, {}], ids=["kernel", "deltars"])
    def test_a_clause_after_an_unconditional_one_is_refused(
        self, conn: Any, tmp_path: Any, properties: Any
    ) -> None:
        path = _table(conn, tmp_path, _ids(2), properties)
        merger = conn.open_table(path).merge(
            _ids(3), "t.id = s.id", source_alias="s", target_alias="t"
        )
        merger = merger.when_matched_update({"s": "'first'"})
        with pytest.raises(InvalidArgumentError, match="only the last clause"):
            merger.when_matched_update({"s": "'second'"})
        with pytest.raises(InvalidArgumentError, match="only the last clause"):
            merger.when_not_matched_insert_all().when_not_matched_insert_all(predicate="s.id = 3")

    @pytest.mark.parametrize("properties", [DV, {}], ids=["kernel", "deltars"])
    def test_default_aliases(self, conn: Any, tmp_path: Any, properties: Any) -> None:
        path = _table(conn, tmp_path, _ids(2), properties)
        source = pa.table({"id": pa.array([2], pa.int64()), "s": ["new"]})
        conn.open_table(path).merge(
            source, "target.id = source.id"
        ).when_matched_update_all().execute()
        assert _rows(conn, path) == [(1, "x"), (2, "new")]


# --------------------------------------------------------------- DM-14


class TestDeletionVectorDmlErrors:
    def test_null_into_not_null(self, conn: Any, tmp_path: Any) -> None:
        schema = pa.schema([pa.field("id", pa.int64()), pa.field("v", pa.int64(), nullable=False)])
        path = str(tmp_path / "t")
        conn.create_table(path, schema, properties=DV)
        conn.open_table(path).append(
            pa.table(
                {"id": pa.array([1], pa.int64()), "v": pa.array([1], pa.int64())}, schema=schema
            )
        )
        with pytest.raises(InvalidArgumentError, match="NOT NULL"):
            conn.open_table(path).update(new_values={"v": None}, predicate="id = 1")
        assert _rows(conn, path) == [(1, 1)]

    def test_missing_deletion_vector_file(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path, _ids(3), DV)
        conn.open_table(path).delete("id = 1")
        for file in glob.glob(os.path.join(path, "deletion_vector_*.bin")):
            os.remove(file)
        with pytest.raises(MissingDataFileError, match="deletion_vector_"):
            conn.open_table(path).to_arrow()
