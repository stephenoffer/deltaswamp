"""SQL predicates: parsing, exact filtering, and conservative skipping."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from deltaswamp import predicate as P

pa = pytest.importorskip("pyarrow")


@pytest.fixture
def table() -> object:
    return pa.table(
        {
            "id": [1, 2, 3, None],
            "city": ["ams", "ber", None, "den"],
            "day": [dt.date(2024, 1, 1), dt.date(2024, 6, 1), None, dt.date(2025, 1, 1)],
            "addr": [{"zip": 1}, {"zip": 5}, {"zip": None}, {"zip": 9}],
        }
    )


def ids(table: object, predicate: str) -> list[object]:
    result = P.filter_table(table, predicate)
    return result.column("id").to_pylist()  # type: ignore[no-any-return]


class TestExactFiltering:
    """A row survives only when the predicate is TRUE, as in a WHERE clause."""

    @pytest.mark.parametrize(
        ("predicate", "expected"),
        [
            ("id > 1", [2, 3]),
            ("id >= 2 AND id < 3", [2]),
            ("id = 1 OR city = 'den'", [1, None]),
            ("NOT (id = 2)", [1, 3]),  # NULL id stays excluded: NOT NULL is NULL
            ("id IN (1, 3)", [1, 3]),
            ("id NOT IN (1, 3)", [2]),
            ("city IS NULL", [3]),
            ("city IS NOT NULL", [1, 2, None]),
            ("id BETWEEN 2 AND 3", [2, 3]),
            ("id NOT BETWEEN 2 AND 3", [1]),
            ("city LIKE 'b%'", [2]),
            ("city NOT LIKE 'b%'", [1, None]),
            ("day >= DATE '2024-05-01'", [2, None]),
            ("day >= '2024-05-01'", [2, None]),  # literal coerced to the column type
            ("addr.zip > 4", [2, None]),
            ("`city` = 'ams'", [1]),
            ("id <=> NULL", [None]),
            ("id = NULL", []),
            ("TRUE", [1, 2, 3, None]),
            ("id = 1.0", [1]),
            ("city = 'it''s'", []),
        ],
    )
    def test_where_semantics(self, table: object, predicate: str, expected: list[object]) -> None:
        assert ids(table, predicate) == expected

    def test_stream_filter_projects_after_filtering(self, table: object) -> None:
        reader = P.filter_stream(table, "id > 1", keep=["city"])
        assert reader.read_all().to_pydict() == {"city": ["ber", None]}

    def test_unknown_column_is_named(self, table: object) -> None:
        with pytest.raises(P.PredicateError, match="unknown column 'nope'"):
            P.filter_table(table, "nope = 1")


class TestRefusals:
    @pytest.mark.parametrize(
        "predicate",
        ["id + 1 > 2", "lower(city) = 'a'", "id >", "", "id = 1 garbage", "(id = 1"],
    )
    def test_unsupported_sql_raises(self, predicate: str) -> None:
        with pytest.raises(P.PredicateError):
            P.parse(predicate)


class TestSkipping:
    """The kernel only skips files. A skipping predicate may be weaker than the
    real one, never stronger."""

    def skip(self, predicate: str) -> object:
        rendered = P.to_kernel_json(P.parse(predicate))
        return None if rendered is None else json.loads(rendered)

    def test_simple_comparison(self) -> None:
        assert self.skip("id > 1") == {
            "op": "gt",
            "args": [{"column": ["id"]}, {"literal": 1, "type": "long"}],
        }

    def test_unexpressible_conjunct_is_dropped(self) -> None:
        assert self.skip("id > 1 AND city LIKE 'a%'") == self.skip("id > 1")

    def test_unexpressible_disjunct_abandons_the_or(self) -> None:
        assert self.skip("id > 1 OR city LIKE 'a%'") is None

    def test_weakening_under_not_is_refused(self) -> None:
        """NOT(a AND b) with b dropped would be NOT(a) -- stronger, and wrong."""
        assert self.skip("NOT (id > 1 AND city LIKE 'a%')") is None

    def test_exact_under_not_is_kept(self) -> None:
        assert self.skip("NOT (id > 1 AND city = 'x')") is not None

    def test_in_expands_to_equalities(self) -> None:
        rendered = self.skip("id IN (1, 2)")
        assert isinstance(rendered, dict)
        assert rendered["op"] == "or"
        assert len(rendered["args"]) == 2

    def test_typed_literals(self) -> None:
        rendered = self.skip("day = DATE '2024-01-02'")
        assert isinstance(rendered, dict)
        assert rendered["args"][1] == {"literal": "2024-01-02", "type": "date"}

    def test_columns_of(self) -> None:
        assert P.columns_of(P.parse("a.b > 1 AND c IS NULL")) == {("a", "b"), ("c",)}
