"""Identity value generation, against Delta Spark's rules (IdentityColumn.scala)."""

from __future__ import annotations

import pytest
from deltaswamp.engine.values import IdentitySpec, round_to_next


def _spec(start: int, step: int, mark: int | None) -> IdentitySpec:
    return IdentitySpec("id", start, step, mark, allow_explicit=True)


@pytest.mark.parametrize(
    ("start", "step", "value", "want"),
    [
        (1, 1, 5, 5),
        (10, 3, 13, 13),
        (10, 3, 11, 13),  # rounded up to the sequence
        (10, 3, 9, 10),  # before the start: rounded towards it, never past
        (10, 3, -20, 10 + 3 * -10),  # Java truncation: -30/3 = -10, no +1
        (-1, -2, -4, -5),  # a negative step rounds down
        (-1, -2, -5, -5),
        (-1, -2, 3, -1 + -2 * -2),
        (0, 5, -7, -5),
    ],
)
def test_round_to_next_matches_spark(start: int, step: int, value: int, want: int) -> None:
    assert round_to_next(start, step, value) == want


def test_the_first_value_follows_the_mark() -> None:
    assert _spec(1, 1, None).first_free() == 1
    assert _spec(1, 1, 7).first_free() == 8
    assert _spec(100, -10, 70).first_free() == 60


def test_a_mark_off_the_sequence_does_not_put_values_off_it() -> None:
    # start 10, step 3: values 10, 13, 16, ... A mark of 14 (another writer's)
    # stands for 16, as Spark rounds a mark up to the sequence when it stores
    # one; the next value is 19, not 17.
    assert _spec(10, 3, 14).first_free() == 19
    assert _spec(-1, -2, -4).first_free() == -7


def test_a_mark_before_the_start_counts_for_nothing() -> None:
    assert _spec(10, 3, 4).first_free() == 10
    assert _spec(-10, -1, 5).first_free() == -10
