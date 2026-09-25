"""Regression tests from differential testing of the predicate evaluator.

Each case was found by comparing the kernel path's exact Arrow filter (and its
skipping JSON) against an independent SQL oracle with Spark's semantics.
"""

from __future__ import annotations

import decimal
import json
from typing import Any, ClassVar

import pytest

pa = pytest.importorskip("pyarrow")

from deltaswamp import predicate as P  # noqa: E402

D = decimal.Decimal
NAN = float("nan")


def rows(table: Any, text: str) -> list[Any]:
    values: list[Any] = P.filter_table(table, text).column(0).to_pylist()
    return values


class TestNumericCoercion:
    def test_integer_literal_against_decimal_column(self) -> None:
        t = pa.table({"d": pa.array([D("0.00"), D("1.50")], pa.decimal128(5, 2))})
        assert rows(t, "d = 0") == [D("0.00")]
        assert rows(t, "d < 2") == [D("0.00"), D("1.50")]

    def test_decimal_literal_wider_than_the_column(self) -> None:
        t = pa.table({"d": pa.array([D("1.50"), D("-999.99")], pa.decimal128(5, 2))})
        assert rows(t, "d < 1234567890123456789012345678.0123456789") == [D("1.50"), D("-999.99")]
        big = pa.table({"i": pa.array([2**63 - 1], pa.int64())})
        assert rows(big, "i < 12345678901234567890123.5") == [2**63 - 1]

    def test_double_literal_against_bigint_compares_as_double(self) -> None:
        # Spark widens the BIGINT to DOUBLE: 2**53 + 1 becomes 2**53.
        t = pa.table({"i": pa.array([2**53 + 1], pa.int64())})
        assert rows(t, "i <= 9007199254740992D") == [2**53 + 1]

    def test_integer_literal_against_float_column_beyond_float_range(self) -> None:
        t = pa.table({"f": pa.array([1.0], pa.float32())})
        assert rows(t, "f < 2147483647") == [1.0]
        assert rows(t, "f IN (9223372036854775807, 1)") == [1.0]

    def test_bigint_column_against_double_column(self) -> None:
        t = pa.table({"i": pa.array([2**63 - 1], pa.int64()), "f": [1.0]})
        assert rows(t, "i > f") == [2**63 - 1]

    def test_out_of_range_literal_widens_a_narrow_integer_column(self) -> None:
        t = pa.table({"i": pa.array([1, 127], pa.int8())})
        assert rows(t, "i < 1000") == [1, 127]
        assert rows(t, "i >= 1000") == []


class TestFloatingPointSemantics:
    def table(self) -> Any:
        return pa.table({"f": pa.array([NAN, 1.0, None, -0.0, 0.0], pa.float64())})

    def test_nan_equals_nan(self) -> None:
        got = rows(self.table(), "f = 'NaN'")
        assert len(got) == 1 and got[0] != got[0]

    def test_nan_sorts_above_everything(self) -> None:
        assert len(rows(self.table(), "f > 1e308")) == 1
        assert rows(self.table(), "f < 'NaN'") == [1.0, -0.0, 0.0]
        # NOT (NaN > 1) is FALSE, so only 1.0 and the zeros survive.
        assert rows(self.table(), "NOT (f > 1)") == [1.0, -0.0, 0.0]

    def test_between_includes_nan_at_the_top(self) -> None:
        assert len(rows(self.table(), "f BETWEEN 1 AND 'NaN'")) == 2

    def test_in_list_matches_negative_zero(self) -> None:
        assert rows(self.table(), "f IN (0, 5)") == [-0.0, 0.0]
        assert rows(self.table(), "f IN (-0.0, 5)") == [-0.0, 0.0]

    def test_float32_column_with_nan(self) -> None:
        t = pa.table({"f": pa.array([NAN, 3.25], pa.float32())})
        assert len(rows(t, "f >= 3.25")) == 2


class TestSkippingSafety:
    def schema(self) -> Any:
        return pa.schema(
            [
                ("f", pa.float64()),
                ("wide", pa.decimal128(18, 4)),
                ("narrow", pa.decimal128(10, 2)),
                ("s", pa.struct([("g", pa.float32())])),
            ]
        )

    def skip(self, text: str) -> Any:
        rendered = P.to_kernel_json(P.parse(text), self.schema())
        return None if rendered is None else json.loads(rendered)

    def test_comparisons_nan_satisfies_are_not_used_for_skipping(self) -> None:
        # Statistics leave NaN out of min/max, so a file whose max is 1.0 may
        # still hold a NaN that `f > 2` matches.
        assert self.skip("f > 2") is None
        assert self.skip("f >= 2") is None
        assert self.skip("f != 2") is None
        assert self.skip("f = 'NaN'") is None
        assert self.skip("NOT (f < 2)") is None
        assert self.skip("s.g > 1") is None

    def test_comparisons_nan_cannot_satisfy_still_skip(self) -> None:
        assert self.skip("f < 2") is not None
        assert self.skip("f = 2") is not None
        assert self.skip("2 > f") is not None
        assert self.skip("f IN (1, 2)") is not None
        assert self.skip("f IS NULL") is not None

    def test_wide_decimals_are_not_used_for_skipping(self) -> None:
        # delta-rs writes these statistics as JSON doubles.
        assert self.skip("wide <= 12345678901234.5678") is None
        assert self.skip("narrow <= 1.5") is not None

    def test_without_a_schema_nothing_changes(self) -> None:
        assert P.to_kernel_json(P.parse("f > 2")) is not None


class TestStringLiterals:
    def value(self, text: str) -> Any:
        return P.parse(text).args[1].value

    def test_backslash_escapes_follow_spark(self) -> None:
        assert self.value(r"s = 'a\\b'") == "a\\b"
        assert self.value(r"s = 'tab\there'") == "tab\there"
        assert self.value(r"s = 'é'") == "é"

    def test_escaped_quote_is_part_of_the_string(self) -> None:
        assert self.value(r"s = 'it\'s'") == "it's"
        assert self.value("s = 'it''s'") == "it's"
        assert self.value(r's = "say \"hi\""') == 'say "hi"'

    def test_like_keeps_escaped_wildcards_literal(self) -> None:
        t = pa.table({"s": ["a_b", "axb", "a\\xb"]})
        assert rows(t, r"s LIKE 'a\_b'") == ["a_b"]
        assert rows(t, r"s LIKE 'a\\_b'") == ["a_b"]

    def test_filter_matches_a_single_backslash(self) -> None:
        t = pa.table({"s": ["a\\b", "a\\\\b"]})
        assert rows(t, r"s = 'a\\b'") == ["a\\b"]


class TestDataFusionRendering:
    def schema(self) -> Any:
        return pa.schema(
            [
                ("Id", pa.int64()),
                ("i8", pa.int8()),
                ("d", pa.decimal128(18, 4)),
                ("ntz", pa.timestamp("us")),
                ("st", pa.struct([("A", pa.int32())])),
            ]
        )

    def render(self, text: str) -> str:
        out = P.to_datafusion(P.parse(text), self.schema())
        assert out is not None
        return out

    def test_columns_are_spelled_as_the_schema_spells_them(self) -> None:
        assert '"Id"' in self.render("ID = 1")
        assert "\"st\"['A']" in self.render("ST.a = 1")

    def test_spark_literal_forms_are_rendered(self) -> None:
        assert "5L" not in self.render("id = 5L")
        assert "Decimal128(18, 4)" in self.render("d = 12345678901234.5678")
        assert "Timestamp(Microsecond, None)" in self.render(
            "ntz = TIMESTAMP_NTZ '2024-01-01 00:00:00'"
        )

    def test_not_in_with_null_keeps_sql_semantics(self) -> None:
        assert "NULL" in self.render("id NOT IN (1, NULL)")

    def test_out_of_range_literal_widens_the_column(self) -> None:
        assert "arrow_cast(\"i8\", 'Int64')" in self.render("i8 >= 1000")

    def test_unknown_column_is_a_predicate_error(self) -> None:
        with pytest.raises(P.PredicateError):
            P.to_datafusion(P.parse("nope = 1"), self.schema())


class TestSharingHints:
    SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "struct",
        "fields": [
            {"name": "f", "type": "double", "nullable": True, "metadata": {}},
            {"name": "s", "type": "string", "nullable": True, "metadata": {}},
        ],
    }

    def hint(self, text: str) -> Any:
        from deltaswamp.engine.sharing import json_predicate_hints

        out = json_predicate_hints(text, self.SCHEMA)
        return None if out is None else json.loads(out)

    def test_no_hint_that_nan_rows_satisfy(self) -> None:
        assert self.hint("f > 2") is None
        assert self.hint("f >= 2") is None
        assert self.hint("f != 2") is None
        assert self.hint("f < 2") is not None

    def test_string_hint_matches_the_exact_filter(self) -> None:
        hint = self.hint(r"s = 'a\\b'")
        assert hint["children"][1]["value"] == "a\\b"
