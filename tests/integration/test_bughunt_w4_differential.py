"""Regression tests from differential testing: kernel vs delta-rs vs a SQL oracle.

Every case here returned different rows (or a DELETE removed different rows)
depending on which engine served the call. Each engine is forced by giving
the connection only that engine.
"""

from __future__ import annotations

import decimal
import math
import shutil
from typing import Any

import deltaswamp as ds
import pytest
from deltaswamp.capability import Engine

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

D = decimal.Decimal
NAN = float("nan")


def connection(kind: str) -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router
    from deltaswamp.table import Connection

    engine = {"kernel": (Engine.KERNEL, KernelEngine), "deltars": (Engine.DELTARS, DeltaRsEngine)}
    key, cls = engine[kind]
    return Connection(catalog=FilesystemCatalog(), router=Router(engines={key: cls()}))


ENGINES = ["kernel", "deltars"]


def ids(conn: Any, path: str, predicate: str) -> list[int]:
    table = pa.table(conn.open_table(path).scan(predicate=predicate))
    return sorted(table.column("id").to_pylist())


@pytest.fixture
def typed(tmp_path: Any) -> str:
    """One file per row, so every file's min equals its max."""
    from deltalake import write_deltalake

    path = str(tmp_path / "typed")
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("Name", pa.string()),
            ("i8", pa.int8()),
            ("f", pa.float64()),
            ("wide", pa.decimal128(18, 4)),
            ("d", pa.date32()),
            ("ntz", pa.timestamp("us")),
            ("st", pa.struct([("a", pa.int32())])),
        ]
    )
    import datetime as dt

    rows = [
        (1, "a\\b", 1, NAN, D("12345678901234.5678"), dt.date(2024, 1, 1), dt.datetime(2024, 1, 1)),
        (2, "it's", 5, 3.25, D("1.5000"), dt.date(9999, 12, 31), dt.datetime(2024, 6, 1)),
        (3, "x", None, -0.0, D("12345678901234.5678"), dt.date(1970, 1, 1), None),
        (4, None, 127, None, None, None, dt.datetime(2024, 1, 1)),
    ]
    for rid, name, i8, f, wide, d, ntz in rows:
        write_deltalake(
            path,
            pa.table(
                {
                    "id": [rid],
                    "Name": [name],
                    "i8": [i8],
                    "f": [f],
                    "wide": [wide],
                    "d": [d],
                    "ntz": [ntz],
                    "st": [{"a": rid}],
                },
                schema=schema,
            ),
            mode="append",
        )
    return path


@pytest.mark.parametrize("kind", ENGINES)
class TestSameRowsOnEveryEngine:
    def test_column_names_are_case_insensitive(self, typed: str, kind: str) -> None:
        assert ids(connection(kind), typed, "NAME = 'x'") == [3]
        assert ids(connection(kind), typed, "ST.A = 2") == [2]

    def test_suffixed_numeric_literals(self, typed: str, kind: str) -> None:
        assert ids(connection(kind), typed, "i8 = 5L") == [2]
        assert ids(connection(kind), typed, "f = 3.25D") == [2]
        assert ids(connection(kind), typed, "-1D < i8") == [1, 2, 4]

    def test_timestamp_ntz_literal(self, typed: str, kind: str) -> None:
        assert ids(connection(kind), typed, "ntz = TIMESTAMP_NTZ '2024-01-01 00:00:00'") == [1, 4]

    def test_decimal_literal_is_exact(self, typed: str, kind: str) -> None:
        # DataFusion read 12345678901234.5678 as a DOUBLE and matched nothing.
        assert ids(connection(kind), typed, "wide = 12345678901234.5678") == [1, 3]
        assert ids(connection(kind), typed, "wide <= 12345678901234.5678") == [1, 2, 3]

    def test_not_in_with_null_matches_nothing(self, typed: str, kind: str) -> None:
        assert ids(connection(kind), typed, "id NOT IN (1, NULL)") == []
        assert ids(connection(kind), typed, "NOT (id IN (1, NULL))") == []

    def test_nan_rows_are_not_pruned(self, typed: str, kind: str) -> None:
        assert ids(connection(kind), typed, "f = 'NaN'") == [1]
        assert ids(connection(kind), typed, "f > 3") == [1, 2]

    def test_negative_zero_equals_zero(self, typed: str, kind: str) -> None:
        assert ids(connection(kind), typed, "f = 0") == [3]
        assert ids(connection(kind), typed, "f IN (0.0, 7)") == [3]
        assert ids(connection(kind), typed, "f < 0") == []

    def test_date_against_timestamp_beyond_nanosecond_range(self, typed: str, kind: str) -> None:
        # delta-rs cast DATE 9999-12-31 to Timestamp(ns) and failed.
        assert ids(connection(kind), typed, "d >= TIMESTAMP '2024-01-01 12:00:00'") == [2]

    def test_double_quoted_text_is_a_string(self, typed: str, kind: str) -> None:
        assert ids(connection(kind), typed, 'Name = "x"') == [3]

    def test_like_on_an_integer_column(self, typed: str, kind: str) -> None:
        assert ids(connection(kind), typed, "i8 LIKE '12%'") == [4]

    def test_backslash_and_quote_escapes(self, typed: str, kind: str) -> None:
        assert ids(connection(kind), typed, r"Name = 'a\\b'") == [1]
        assert ids(connection(kind), typed, r"Name = 'it\'s'") == [2]

    def test_literal_wider_than_the_column(self, typed: str, kind: str) -> None:
        conn = connection(kind)
        assert ids(conn, typed, "wide < 1234567890123456789012345678.0123456789") == [1, 2, 3]
        assert ids(conn, typed, "i8 >= 1000") == []

    def test_count_and_head_agree(self, typed: str, kind: str) -> None:
        table = connection(kind).open_table(typed)
        assert table.count(predicate="f = 'NaN' OR NAME = 'x'") == 2
        assert table.head(5, predicate="f = 'NaN' OR NAME = 'x'").num_rows == 2


class TestDeltaRsReadsValuesAsWritten:
    def test_filtered_column_keeps_its_value(self, typed: str) -> None:
        # delta-rs replaced a column whose file min == max with the statistic:
        # NaN (absent from stats) and a wide decimal (stats are doubles).
        table = connection("deltars").open_table(typed)
        got = pa.table(table.scan(predicate="f IS NOT NULL OR f IS NULL")).to_pylist()
        nan_row = next(r for r in got if r["id"] == 1)
        assert math.isnan(nan_row["f"])
        got = pa.table(table.scan(predicate="wide IS NOT NULL")).to_pylist()
        assert {r["wide"] for r in got} == {D("12345678901234.5678"), D("1.5000")}

    def test_scan_uses_plain_string_types(self, typed: str) -> None:
        reader = pa.RecordBatchReader.from_stream(connection("deltars").open_table(typed).scan())
        assert reader.schema.field("Name").type == pa.string()
        table = reader.read_all()
        # `take` has no string_view kernel; this failed before.
        assert table.filter(pa.compute.equal(table["id"], 3))["Name"].to_pylist() == ["x"]

    def test_projection_is_case_insensitive(self, typed: str) -> None:
        table = pa.table(connection("deltars").open_table(typed).scan(columns=["ID", "name"]))
        assert table.column_names == ["id", "Name"]


class TestDeltaRsDml:
    def test_delete_with_not_in_null_deletes_nothing(self, typed: str) -> None:
        table = connection("deltars").open_table(typed)
        table.delete("id NOT IN (1, NULL)")
        assert table.count() == 4

    def test_delete_keeps_wide_decimal_rows_it_does_not_match(self, typed: str) -> None:
        table = connection("deltars").open_table(typed)
        table.delete("wide <> 12345678901234.5678")
        assert sorted(pa.table(table.scan()).column("id").to_pylist()) == [1, 3, 4]

    def test_delete_with_case_varied_names_and_nan(self, typed: str) -> None:
        table = connection("deltars").open_table(typed)
        table.delete("F = 'NaN' OR NAME = 'x'")
        assert sorted(pa.table(table.scan()).column("id").to_pylist()) == [2, 4]

    def test_delete_with_an_out_of_range_literal(self, typed: str) -> None:
        table = connection("deltars").open_table(typed)
        table.delete("i8 >= 1000")
        assert table.count() == 4

    def test_update_assignments_mean_what_spark_means(self, typed: str) -> None:
        table = connection("deltars").open_table(typed)
        table.update(
            updates={
                "NAME": r"'c:\\tmp'",
                "wide": "12345678901234.5678",
                "ntz": "TIMESTAMP_NTZ '2020-02-02 00:00:00'",
            },
            predicate="id = 2",
        )
        row = next(r for r in pa.table(table.scan()).to_pylist() if r["id"] == 2)
        assert row["Name"] == "c:\\tmp"
        assert row["wide"] == D("12345678901234.5678")
        assert row["ntz"].isoformat() == "2020-02-02T00:00:00"

    def test_update_double_quoted_value_is_a_string(self, typed: str) -> None:
        table = connection("deltars").open_table(typed)
        table.update(updates={"Name": '"dq"'}, predicate="id = 3")
        row = next(r for r in pa.table(table.scan()).to_pylist() if r["id"] == 3)
        assert row["Name"] == "dq"


class TestKernelSkipping:
    def test_file_holding_nan_is_not_skipped(self, typed: str) -> None:
        # The NaN file's statistics have no max, but a file that also holds
        # 1.0 has max 1.0 and was skipped by `f > 2`.
        from deltalake import write_deltalake

        write_deltalake(
            typed,
            pa.table({"id": [5, 6], "f": [1.0, NAN]}),
            mode="append",
            schema_mode="merge",
        )
        assert ids(connection("kernel"), typed, "f > 3") == [1, 2, 6]
        assert ids(connection("kernel"), typed, "NOT (f < 2)") == [1, 2, 6]

    def test_wide_decimal_file_is_not_skipped(self, typed: str) -> None:
        # Stats stored 12345678901234.5678 as ...568, above the literal.
        assert ids(connection("kernel"), typed, "wide <= 12345678901234.5678") == [1, 2, 3]
        assert ids(connection("kernel"), typed, "wide < 12345678901234.5679") == [1, 2, 3]


class TestChangeDataFeed:
    def test_cdf_projection_is_case_insensitive_and_plain_typed(self, tmp_path: Any) -> None:
        from deltalake import write_deltalake

        path = str(tmp_path / "cdf")
        write_deltalake(
            path,
            pa.table({"id": [1, 2], "Name": ["a", "b"]}),
            configuration={"delta.enableChangeDataFeed": "true"},
        )
        for kind in ENGINES:
            feed = pa.table(
                connection(kind)
                .open_table(path)
                .cdf(starting_version=0, columns=["NAME"], predicate="ID = 2")
            )
            assert feed.column("Name").to_pylist() == ["b"], kind
            assert feed.schema.field("Name").type == pa.string(), kind


def test_engines_agree_after_copy(typed: str, tmp_path: Any) -> None:
    """A broad sweep over one table: both engines return the same ids."""
    copy = str(tmp_path / "copy")
    shutil.copytree(typed, copy)
    predicates = [
        "f >= -0.0",
        "f <> 3.25",
        "wide BETWEEN 1.5 AND 12345678901234.5678",
        "d IN (DATE '9999-12-31', NULL)",
        "st.a <=> 3",
        "Name IS DISTINCT FROM 'x'",
        "i8 NOT BETWEEN 2 AND 1000",
    ]
    for text in predicates:
        assert ids(connection("kernel"), copy, text) == ids(connection("deltars"), copy, text), text


class TestPartitionColumns:
    @pytest.fixture
    def partitioned(self, tmp_path: Any) -> str:
        from deltalake import write_deltalake

        path = str(tmp_path / "part")
        write_deltalake(
            path,
            pa.table({"id": [1, 2, 3, 4], "p": pa.array([1, 1, 2, 3], pa.int8())}),
            partition_by=["p"],
        )
        return path

    @pytest.mark.parametrize("kind", ENGINES)
    def test_in_list_with_null_on_a_partition_column(self, partitioned: str, kind: str) -> None:
        # delta-rs pruned partitions wrongly on the NULL-preserving form and
        # returned every row.
        assert ids(connection(kind), partitioned, "p IN (1, NULL)") == [1, 2]
        assert ids(connection(kind), partitioned, "p NOT IN (1, NULL)") == []

    def test_delete_by_in_list_with_null_on_a_partition_column(self, partitioned: str) -> None:
        table = connection("deltars").open_table(partitioned)
        table.delete("p IN (1, NULL)")
        assert sorted(pa.table(table.scan()).column("id").to_pylist()) == [3, 4]


@pytest.mark.parametrize("op", ["delete", "update"])
def test_dml_on_a_whole_struct_column_is_refused_clearly(typed: str, op: str) -> None:
    table = connection("deltars").open_table(typed)
    with pytest.raises(ds.InvalidArgumentError, match="whole struct"):
        if op == "delete":
            table.delete("ST IS NULL")
        else:
            table.update(new_values={"Name": "x"}, predicate="st IS NOT NULL")
    assert table.count() == 4


WIDE = "12345678901234.5678"


class TestDeltaRsStatsSubstitutionThroughTheApi:
    """delta-rs replaces a predicate column whose file min == max with the
    statistic; for NaN and wide decimals that is not the value."""

    def test_delete_by_a_function_predicate_keeps_the_row(self, typed: str) -> None:
        table = connection("deltars").open_table(typed)
        table.delete(f"wide <> CAST('{WIDE}' AS DECIMAL(18,4)) AND abs(id) > 0")
        assert sorted(pa.table(table.scan()).column("id").to_pylist()) == [1, 3, 4]

    def test_update_by_a_function_predicate_does_not_rewrite_the_value(self, typed: str) -> None:
        table = connection("deltars").open_table(typed)
        table.update(
            new_values={"Name": "U"},
            predicate=f"wide = CAST('{WIDE}' AS DECIMAL(18,4)) AND abs(id) > 0",
        )
        rows = {r["id"]: r for r in pa.table(table.scan()).to_pylist()}
        assert rows[1]["wide"] == D(WIDE) and rows[1]["Name"] == "U"

    def test_function_predicate_on_nan_keeps_the_nan(self, typed: str) -> None:
        table = connection("deltars").open_table(typed)
        table.update(new_values={"Name": "U"}, predicate="abs(id) = 1 AND f IS NOT NULL")
        row = next(r for r in pa.table(table.scan()).to_pylist() if r["id"] == 1)
        assert math.isnan(row["f"])

    def test_long_decimal_literal_in_passthrough_text_is_exact(self, typed: str) -> None:
        # DataFusion reads it as a DOUBLE, and the DELETE removed rows 1 and 3.
        table = connection("deltars").open_table(typed)
        table.delete(f"abs(wide) <> {WIDE}")
        assert sorted(pa.table(table.scan()).column("id").to_pylist()) == [1, 3, 4]

    def test_merge_clause_with_a_long_decimal_literal(self, typed: str) -> None:
        table = connection("deltars").open_table(typed)
        src = pa.table({"id": [2]})
        (
            table.merge(src, "t.id = s.id", source_alias="s", target_alias="t")
            .when_not_matched_by_source_delete(f"t.wide <> {WIDE}")
            .execute()
        )
        assert sorted(pa.table(table.scan()).column("id").to_pylist()) == [1, 2, 3, 4]

    def test_replace_where_by_a_wide_decimal(self, typed: str) -> None:
        table = connection("deltars").open_table(typed)
        full = pa.table(table.scan())
        first = full.filter(pa.compute.equal(full["id"], 1))
        table.overwrite(first, predicate=f"wide = {WIDE} AND id = 1")
        assert sorted(pa.table(table.scan()).column("id").to_pylist()) == [1, 2, 3, 4]

    @pytest.mark.parametrize(
        "predicate",
        [
            "i8 <> 5 AND abs(id) > 0",
            "Name <> 'x' AND abs(id) > 0",
            "d <> DATE '2024-01-01' AND abs(id) > 0",
        ],
    )
    def test_exact_statistics_columns_are_unaffected(self, typed: str, predicate: str) -> None:
        before = pa.table(connection("kernel").open_table(typed).scan())
        want = {r["id"] for r in before.to_pylist()} - set(
            ids(connection("kernel"), typed, predicate.split(" AND ")[0])
        )
        table = connection("deltars").open_table(typed)
        table.delete(predicate)
        assert set(pa.table(table.scan()).column("id").to_pylist()) == want


class TestDeltaRsNullLiterals:
    @pytest.mark.parametrize(
        "predicate",
        [
            "id = NULL",
            "NOT (id = NULL)",
            "id <> NULL",
            "id IN (NULL)",
            "NOT (id IN (NULL))",
            "1 = NULL",
            "i8 BETWEEN NULL AND 5",
            "(id > 1) AND (id = NULL)",
        ],
    )
    def test_a_null_comparison_deletes_nothing(self, typed: str, predicate: str) -> None:
        # delta-rs deleted every row: DataFusion folds the predicate to a
        # NULL literal, which delta-rs's DELETE treats as no predicate.
        table = connection("deltars").open_table(typed)
        table.delete(predicate)
        assert table.count() == 4

    def test_null_safe_comparison_with_null(self, typed: str) -> None:
        assert ids(connection("deltars"), typed, "i8 <=> NULL") == [3]
        assert ids(connection("deltars"), typed, "NOT (i8 <=> NULL)") == [1, 2, 4]

    @pytest.mark.parametrize(
        "predicate",
        [
            "abs(id) = NULL",
            "abs(i8) NOT IN (1, NULL)",
            "CASE WHEN i8 = 1 THEN TRUE ELSE NULL END",
            "NULLIF(TRUE, TRUE)",
        ],
    )
    def test_unparseable_text_with_a_null_literal_is_refused(
        self, typed: str, predicate: str
    ) -> None:
        table = connection("deltars").open_table(typed)
        with pytest.raises(ds.InvalidArgumentError, match="NULL literal"):
            table.delete(predicate)
        with pytest.raises(ds.InvalidArgumentError, match="NULL literal"):
            table.scan(predicate=predicate)
        assert table.count() == 4

    def test_is_null_in_unparseable_text_still_works(self, typed: str) -> None:
        table = connection("deltars").open_table(typed)
        table.delete("abs(id) > 3 OR Name IS NULL")
        assert table.count() == 3


class TestFloatPartitionValues:
    @pytest.mark.parametrize("value", [1e-300, 1e300, 5e-324])
    def test_delta_rs_refuses_a_value_it_would_spell_too_long(
        self, tmp_path: Any, value: float
    ) -> None:
        conn = connection("deltars")
        path = str(tmp_path / "fp")
        with pytest.raises(ds.InvalidArgumentError, match="directory"):
            conn.write_table(path, pa.table({"id": [1], "f": [value]}), partition_by=["f"])

    def test_ordinary_float_partition_values_still_write(self, tmp_path: Any) -> None:
        conn = connection("deltars")
        path = str(tmp_path / "fp")
        conn.write_table(path, pa.table({"id": [1, 2], "f": [1.5, 1e-5]}), partition_by=["f"])
        table = conn.open_table(path)
        table.append(pa.table({"id": [3], "f": [1.5]}))
        assert table.count() == 3
        with pytest.raises(ds.InvalidArgumentError, match="directory"):
            table.append(pa.table({"id": [4], "f": [1e-300]}))
        assert table.count() == 3
