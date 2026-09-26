"""Regressions for predicate-semantics defects found by the read-path audit.

Each case was compared against a Databricks SQL warehouse, and each failed
before its fix.
"""

from __future__ import annotations

from typing import Any

import pytest
from deltaswamp import predicate as P
from deltaswamp.predicate import PredicateError

pa = pytest.importorskip("pyarrow")


def ids(table: Any, predicate: str) -> list[Any]:
    return sorted(P.filter_table(table, predicate).column("id").to_pylist())


class TestAdjacentLiterals:
    """Spark reads `'it''s'` as `'it' 's'` (`its`); the kernel read `it's`."""

    def test_doubled_quote_concatenates(self) -> None:
        assert P.parse("s = 'it''s'").args[1].value == "its"
        assert P.parse("s = 'a' \"b\"  'c'").args[1].value == "abc"
        assert P.parse(r"s = 'it\'s'").args[1].value == "it's"

    def test_empty_adjacent_literal(self) -> None:
        assert P.parse("s = ''''").args[1].value == ""

    @pytest.mark.parametrize(
        ("spark", "ansi"),
        [
            ("s = 'it''s'", "s = 'its'"),
            (r"s = 'it\'s'", "s = 'it''s'"),
            (r"f(s) = 'a\\b' AND g = 'x' 'y'", "f(s) = 'a\\b' AND g = 'xy'"),
            ("\"ident\" = 'v'", "\"ident\" = 'v'"),
            ("`it's` = 'v'", "`it's` = 'v'"),
            ("s = 'unterminated", "s = 'unterminated"),
            ("no literals", "no literals"),
        ],
    )
    def test_passthrough_text_is_respelled_for_ansi_engines(self, spark: str, ansi: str) -> None:
        assert P.standard_string_literals(spark) == ansi

    def test_datafusion_rendering_carries_the_spark_value(self) -> None:
        schema = pa.schema([("s", pa.string())])
        assert "'its'" in (P.to_datafusion(P.parse("s = 'it''s'"), schema) or "")


class TestStringNumberComparisons:
    """Spark casts a STRING column to the number's type; Arrow compared as text."""

    @pytest.fixture
    def table(self) -> Any:
        return pa.table({"id": [1, 2, 3], "s": ["1", "01", " 1"], "b": [True, False, None]})

    @pytest.mark.parametrize("predicate", ["s = 1", "s = 1.0", "s IN (1, 10)", "s > 5", "1 = s"])
    def test_refused_with_a_remedy(self, table: Any, predicate: str) -> None:
        with pytest.raises(PredicateError, match="Quote the literal"):
            P.filter_table(table, predicate)

    def test_quoted_literal_compares_as_text(self, table: Any) -> None:
        assert ids(table, "s = '1'") == [1]

    @pytest.mark.parametrize("predicate", ["id = true", "id IN (1, false)", "b = 1"])
    def test_boolean_and_number_do_not_compare(self, table: Any, predicate: str) -> None:
        with pytest.raises(PredicateError, match="type mismatch"):
            P.filter_table(table, predicate)

    def test_non_integral_string_against_an_integer_column(self, table: Any) -> None:
        # The warehouse: CAST_INVALID_INPUT ('7.0' is not a BIGINT).
        with pytest.raises(PredicateError, match="not a valid value"):
            P.filter_table(table, "id = '1.0'")
        assert ids(table, "id = '1'") == [1]


class TestMixedInList:
    """Spark compares an IN list in its items' common type."""

    def test_one_double_makes_the_whole_list_double(self) -> None:
        big = 9007199254740993
        table = pa.table({"id": [1, 2, 3], "l": [big, big - 1, 7]})
        # The warehouse returns all three: 9007199254740993 equals
        # 9007199254740992 once both are DOUBLE.
        assert ids(table, "l IN (9007199254740992, 7D)") == [1, 2, 3]
        assert ids(table, "l IN (9007199254740992, 7)") == [2, 3]

    def test_skipping_stays_conservative(self) -> None:
        node = P.parse("l IN (9007199254740992, 7D)")
        items = node.args[1:]
        assert all(isinstance(i.value, float) for i in items)
