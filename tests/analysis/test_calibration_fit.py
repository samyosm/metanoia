import numpy as np
import pytest

from metanoia_analysis.calibration import (
    apply_temperature,
    at_search_bound,
    fit_temperature,
    softmax,
)
from metanoia_analysis.load import Item


def items_with_temperature(true_t: float, n: int = 4000, seed: int = 0) -> list[Item]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        logits = rng.normal(0, 3, size=4)
        label = int(rng.choice(4, p=softmax(logits, true_t)))
        out.append(
            Item(
                example_id=str(i),
                group_id=str(i),
                split="calibration",
                budget_id="b0",
                repeat_id=0,
                candidate_ids=("a", "b", "c", "d"),
                label_index=label,
                probabilities=softmax(logits, 1.0),
                logits=logits,
                failed=False,
                annotations={},
            )
        )
    return out


def test_fit_recovers_generating_temperature() -> None:
    fitted = fit_temperature(items_with_temperature(2.5))
    assert fitted == pytest.approx(2.5, rel=0.1)
    assert not at_search_bound(fitted)


def test_temperature_preserves_ranking() -> None:
    items = items_with_temperature(2.0, n=50)
    for before, after in zip(items, apply_temperature(items, 3.0), strict=True):
        assert before.probabilities.argmax() == after.probabilities.argmax()
