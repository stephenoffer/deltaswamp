"""Spark SQL as the direct engines evaluate it, and VARIANT text as Databricks renders it.

Every expected value here is what a Databricks SQL warehouse (ANSI mode)
returned for the same expression; DuckDB's and DataFusion's own answers for
the untranslated text differed, and each case failed before the fix.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from deltaswamp import _variant
from deltaswamp import predicate as P
from deltaswamp.engine import dialect as D
from deltaswamp.errors import EngineLimitError

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")
deltalake = pytest.importorskip("deltalake")

ERROR = object()


def _duckdb(text: str) -> Any:
    con = duckdb.connect()
    D.install_duckdb_macros(con)
    try:
        return con.sql(f"SELECT ({D.to_duckdb(text)})").fetchone()[0]
    except duckdb.Error:
        return ERROR
    finally:
        con.close()


def _datafusion(text: str) -> Any:
    from deltalake import QueryBuilder

    try:
        result = QueryBuilder().execute(f"SELECT ({D.to_datafusion(text)}) AS x").read_all()
    except Exception:
        return ERROR
    return result.column(0)[0].as_py()


def _same(got: Any, want: Any) -> bool:
    if want is ERROR or got is ERROR:
        return got is want
    if isinstance(want, float) or isinstance(got, float):
        return got is not None and want is not None and abs(float(got) - float(want)) < 1e-12
    if isinstance(want, bool) or isinstance(got, bool):
        return bool(got) == bool(want) and type(got) is type(want)
    return str(got) == str(want)


#: (Spark SQL, what the warehouse returned). ERROR: the warehouse raised.
CASES = [
    # CR-1: a double-quoted literal is a string, as are adjacent ones.
    ('"ab"', "ab"),
    ("'a' \"b\"", "ab"),
    ("'it''s'", "its"),
    ("\"ab\" = 'ab'", True),
    # CR-3: `/` divides into a DOUBLE; DIV truncates toward zero.
    ("5 / 2", 2.5),
    ("-7 / 2", -3.5),
    ("10 / 4 * 2", 5.0),
    ("1 + 7 / 2", 4.5),
    ("-7 DIV 2", -3),
    ("7 DIV -2", -3),
    ("7 DIV 0", ERROR),
    ("7 / 0", ERROR),
    ("7 % 0", ERROR),
    # CR-3: a fraction CAST to an integral type truncates.
    ("CAST(1.9D AS BIGINT)", 1),
    ("CAST(-1.9D AS INT)", -1),
    ("CAST(2.5 AS INT)", 2),
    ("CAST(-2.5 AS SMALLINT)", -2),
    ("CAST(2.7D AS LONG)", 2),
    ("2.7D::INT", 2),
    ("CAST('1.9' AS INT)", ERROR),
    ("CAST(' 7 ' AS INT)", 7),
    ("TRY_CAST('1.9' AS INT)", None),
    ("CAST(2.99 AS DECIMAL)", 3),
    # CR-3: substring counts position 0 as 1, and negatives from the end.
    ("substring('abc', 0, 2)", "ab"),
    ("substring('abcdef', -3, 2)", "de"),
    ("substring('abcdef', -10, 8)", "abcd"),
    ("substring('abcdef', 2, -1)", ""),
    ("substr('abcdef', -2, 5)", "ef"),
    ("substring('abcdef', 1 - 1, 1 + 1)", "ab"),
    ("left('abcdef', -1)", ""),
    ("right('abcdef', -1)", ""),
    ("right('abcdef', 2)", "ef"),
    # CR-3: other functions whose meaning differs.
    ("concat('a', CAST(NULL AS STRING))", None),
    ("log(100)", 4.605170185988092),
    ("trim('xy', 'xyaxy')", "a"),
    ("ltrim('xy', 'xyaxy')", "axy"),
    ("regexp_replace('abcabc', 'b', 'X')", "aXcaXc"),
    ("pmod(-7, 3)", 2),
    ("5 ^ 3", 6),
    ("if(1 = 2, 'y', 'n')", "n"),
    ("nvl2(1, 'a', 'b')", "a"),
    # CR-5: syntax neither engine parses.
    ("'abc' RLIKE 'b.'", True),
    ("'abc' NOT RLIKE 'b.'", False),
    ("'abc' REGEXP '^a'", True),
    ("1.5D + 1", 2.5),
    ("7L + 1", 8),
    ("10S + 1", 11),
    ("1.5BD + 1", "2.5"),
    ("1 <=> NULL", False),
    ("NULL <=> NULL", True),
    ("'a_b' LIKE 'a\\\\_b'", True),
    ("'axb' LIKE 'a\\\\_b'", False),
    ("'axb' LIKE 'a!_b' ESCAPE '!'", False),
    ("'a_b' LIKE 'a!_b' ESCAPE '!'", True),
]


@pytest.mark.parametrize(("spark", "want"), CASES)
def test_duckdb_computes_what_spark_does(spark: str, want: Any) -> None:
    assert _same(_duckdb(spark), want)


@pytest.mark.parametrize(("spark", "want"), CASES)
def test_datafusion_computes_what_spark_does(spark: str, want: Any) -> None:
    if spark in ("7 / 0",) or "1.5BD" in spark:
        pytest.skip("DataFusion cannot raise on a DOUBLE quotient, or print a DECIMAL sum so")
    assert _same(_datafusion(spark), want)


def test_datafusion_divides_integers_into_a_double_only_for_integers() -> None:
    # A DECIMAL quotient stays DECIMAL, as in Spark.
    kinds = {"d": "decimal", "n": "int", "f": "float"}
    assert "Float64" not in D.to_datafusion("d / 2", kinds)
    assert "Float64" not in D.to_datafusion("f / n", kinds)
    assert "arrow_cast(n, 'Float64')" in D.to_datafusion("n / 2", kinds)


def test_in_list_with_a_double_compares_as_double() -> None:
    # Spark: one DOUBLE makes the list DOUBLE; DataFusion's pruning refused
    # BIGINT IN (BIGINT, DOUBLE) outright.
    text = D.to_datafusion("l IN (9007199254740992, 7D)", {"l": "int"})
    assert text.count("Float64") == 3


@pytest.mark.parametrize(
    "spark",
    [
        "arr[0] = 1",
        "split(s, ',')[0] = 'a'",
        "date_format(d, 'yyyy-MM') = '2024-01'",
        "to_date(s, 'yyyyMMdd') > DATE '2024-01-01'",
        "regexp_replace(s, '(b)', '$1$1') = 'x'",
        "hash(s) = 1",
        "x -> x DIV 2",
    ],
)
def test_sql_no_direct_engine_can_match_is_refused(spark: str) -> None:
    assert D.warehouse_reason(spark) is not None
    with pytest.raises(EngineLimitError):
        D.to_datafusion(spark)
    with pytest.raises(EngineLimitError):
        D.to_duckdb(spark)


@pytest.mark.parametrize(
    "spark", ["m['k'] = 1", "lower(s) = 'a'", "element_at(arr, 1) = 1", "trim(BOTH 'x' FROM s)"]
)
def test_sql_the_engines_agree_on_is_not_refused(spark: str) -> None:
    assert D.warehouse_reason(spark) is None


def test_text_outside_the_grammar_still_has_its_literals_respelled() -> None:
    # A lambda is passed through; its string literals still read as Spark's.
    out = D.to_datafusion('exists(arr, x -> x = "a")')
    assert "'a'" in out and '"a"' not in out


def test_standard_string_literals_reads_double_quotes_as_a_string() -> None:
    assert P.standard_string_literals("concat(s, '') = \"ab\"") == "concat(s, '') = 'ab'"


def test_duckdb_output_has_no_word_the_sharing_screen_refuses() -> None:
    from deltaswamp.engine.sharing import _screen_expression

    _screen_expression(D.to_duckdb("a <=> b AND a IS DISTINCT FROM 1"))


def test_sharing_filter_reads_a_double_quoted_literal_as_a_string() -> None:
    from deltaswamp.engine.sharing import filter_arrow_exact

    table = pa.table({"s": ["ab", "zz"], "ab": ["x", "zz"]})
    out = filter_arrow_exact(table, "concat(s, '') = \"ab\" AND length(s) = 2")
    assert out.column("s").to_pylist() == ["ab"]


# ------------------------------------------------------------------ VARIANT


def _roundtrip(text: str) -> str:
    return _variant.to_json(*_variant.encode(text))


@pytest.mark.parametrize(
    ("json_text", "databricks"),
    [
        ("1e20", "1.0E20"),
        ("1.0E-5", "1.0E-5"),
        ("1E2", "100.0"),
        # V2: precision 2 but scale 39 -- past DECIMAL(38), so a DOUBLE.
        ("0." + "0" * 37 + "12", "1.2E-38"),
        ("0." + "0" * 36 + "12", "0." + "0" * 36 + "12"),
        ("123456789012345678901234567890123456789", "1.2345678901234568E38"),
        # V3: object keys in Java's (UTF-16) order, as Spark sorts them.
        ('{"\uff01":1,"\U0001f600":2}', '{"\U0001f600":2,"\uff01":1}'),
        ('{"b":1,"a":2,"B":3}', '{"B":3,"a":2,"b":1}'),
    ],
)
def test_variant_text_is_databricks_to_json(json_text: str, databricks: str) -> None:
    assert _roundtrip(json_text) == databricks


def test_variant_timestamps_drop_trailing_fraction_zeros() -> None:
    # V4: Databricks prints .5, not .500000.
    micros = int(
        (dt.datetime(2024, 1, 1, 0, 0, 0, 500000) - dt.datetime(1970, 1, 1)).total_seconds()
    )
    micros = micros * 1_000_000 + 500000
    assert _variant._micros(micros, zone=True) == '"2024-01-01 00:00:00.5+00:00"'
    assert _variant._micros(micros - 500000, zone=False) == '"2024-01-01 00:00:00"'


@pytest.mark.parametrize("text", ["NaN", "Infinity", '{"a": -Infinity}'])
def test_variant_refuses_what_parse_json_refuses(text: str) -> None:
    # Databricks' parse_json raises MALFORMED_RECORD_IN_PARSING for these.
    from deltaswamp.errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError):
        _variant.variant_column(pa, pa.array([text]))


@pytest.mark.parametrize(
    ("text", "want"),
    [
        # A SQL warehouse (ANSI mode): an integer literal is an INT, so a
        # TINYINT plus 1 is an INT, and 127 + 1 - 1 is 127. DuckDB narrowed the
        # literal to TINYINT and overflowed.
        ("CAST(127 AS TINYINT) + 1 - 1", 127),
        ("CAST(32767 AS SMALLINT) * 2", 65534),
        ("CAST(2147483647 AS INT) + 1", ERROR),  # ARITHMETIC_OVERFLOW
        ("CAST(2147483647 AS INT) + 1L", 2147483648),
        ("2147483648 + 1", 2147483649),  # a BIGINT literal
        ("CAST(10 AS INT) / 0", ERROR),  # DIVIDE_BY_ZERO
        ("CAST(10 AS INT) % 0", ERROR),  # REMAINDER_BY_ZERO
    ],
)
def test_integer_literals_are_typed_as_spark_types_them(text: str, want: Any) -> None:
    con = duckdb.connect(config={"disabled_optimizers": "expression_rewriter"})
    D.install_duckdb_macros(con)
    try:
        got = con.sql(f"SELECT ({D.to_duckdb(text)})").fetchone()[0]
    except duckdb.Error:
        got = ERROR
    finally:
        con.close()
    assert _same(got, want), (text, got, want)
