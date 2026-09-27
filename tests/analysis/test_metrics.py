"""Hand-computable metric checks, independent of any model or training loss."""

import math

import numpy as np
import pytest

from metanoia_analysis.load import Item
from metanoia_analysis.metrics import (
    correctness,
    expected_calibration_error,
    per_example,
    reliability,
)
from metanoia_analysis.statistics import group_bootstrap_ci, reversal_rates


def item(probs: list[float], label: int, group: str = "g") -> Item:
    return Item(
        example_id=f"e{id(probs)}",
        group_id=group,
        split="test",
        budget_id="b0",
        repeat_id=0,
        candidate_ids=tuple(f"c{i}" for i in range(len(probs))),
        label_index=label,
        probabilities=np.asarray(probs),
        logits=None,
        failed=False,
        annotations={},
    )


def test_accuracy_nll_brier_by_hand() -> None:
    items = [item([0.7, 0.2, 0.1], 0), item([0.5, 0.5], 1), item([0.1, 0.9], 0)]
    assert per_example(items, "accuracy", 1e-15).tolist() == [1.0, 0.5, 0.0]
    assert per_example(items, "nll", 1e-15) == pytest.approx(
        [-math.log(0.7), -math.log(0.5), -math.log(0.1)]
    )
    assert per_example(items, "brier", 1e-15) == pytest.approx(
        [0.09 + 0.04 + 0.01, 0.25 + 0.25, 0.81 + 0.81]
    )


def test_ties_are_not_broken_by_candidate_order() -> None:
    assert correctness(item([0.25] * 4, 0)) == correctness(item([0.25] * 4, 3)) == 0.25


def test_nll_floor_is_applied() -> None:
    assert per_example([item([1.0, 0.0], 1)], "nll", 1e-15)[0] == pytest.approx(-math.log(1e-15))


def test_ece_and_reliability() -> None:
    # Two bins used: confidence 0.9 (one right, one wrong) and 0.6 (right).
    items = [item([0.9, 0.1], 0), item([0.9, 0.1], 1), item([0.6, 0.4], 0)]
    table = reliability(items, bins=10)
    assert sum(row["count"] for row in table) == 3
    expected = (2 / 3) * abs(0.5 - 0.9) + (1 / 3) * abs(1.0 - 0.6)
    assert expected_calibration_error(items, bins=10) == pytest.approx(expected)
    # A perfectly calibrated toy set.
    perfect = [item([0.5, 0.5], 0), item([0.5, 0.5], 1)]
    assert expected_calibration_error(perfect, bins=10) == pytest.approx(0.0)


def test_group_bootstrap_interval() -> None:
    values = np.array([1.0, 1.0, 0.0, 0.0])
    groups = ["a", "a", "b", "b"]
    low, high = group_bootstrap_ci(values, groups, 500, seed=0, confidence=0.9)
    assert low == 0.0 and high == 1.0  # only two independent groups
    assert group_bootstrap_ci(values, ["a"] * 4, 500, 0, 0.9) is None


def test_reversals_in_both_directions() -> None:
    rates = reversal_rates(np.array([0, 0, 1, 1]), np.array([1, 0, 0, 1]))
    assert rates == {"wrong_to_right": 0.25, "right_to_wrong": 0.25}
