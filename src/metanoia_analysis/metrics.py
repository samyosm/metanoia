"""Official metric definitions (SCIENTIFIC_FOUNDATIONS §3.3, §13.5).

Per-example values are returned so the same definitions feed point estimates and
resampled intervals. Accuracy is tie-aware: if the label shares the top probability
with k-1 other candidates it scores 1/k, so candidate order is never a tie-breaker.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from metanoia_analysis.load import Item


def correctness(item: Item) -> float:
    top = item.probabilities.max()
    tied = item.probabilities == top
    return float(tied[item.label_index]) / float(tied.sum())


def strictly_correct(item: Item) -> bool:
    """The label is the unique most probable candidate. Used for reversal counts."""
    return correctness(item) == 1.0


def per_example(items: Sequence[Item], metric: str, nll_floor: float) -> np.ndarray:
    if metric == "accuracy":
        return np.array([correctness(i) for i in items])
    if metric == "nll":
        return np.array(
            [-np.log(max(float(i.probabilities[i.label_index]), nll_floor)) for i in items]
        )
    if metric == "brier":
        values = []
        for i in items:
            onehot = np.zeros_like(i.probabilities)
            onehot[i.label_index] = 1.0
            values.append(float(((i.probabilities - onehot) ** 2).sum()))
        return np.array(values)
    raise ValueError(f"{metric} is not a per-example metric")


def reliability(items: Sequence[Item], bins: int) -> list[dict[str, Any]]:
    """Top-label reliability over equal-width confidence bins on [0, 1]."""
    confidence = np.array([i.probabilities.max() for i in items])
    correct = np.array([correctness(i) for i in items])
    # Right-closed bins; confidence 0 falls into the first bin.
    index = np.clip(np.ceil(confidence * bins).astype(int) - 1, 0, bins - 1)
    table = []
    for b in range(bins):
        members = index == b
        count = int(members.sum())
        table.append(
            {
                "bin": b,
                "low": b / bins,
                "high": (b + 1) / bins,
                "count": count,
                "mean_confidence": float(confidence[members].mean()) if count else None,
                "accuracy": float(correct[members].mean()) if count else None,
            }
        )
    return table


def expected_calibration_error(items: Sequence[Item], bins: int) -> float:
    """Top-label ECE. Depends on binning and sample size; never the sole criterion."""
    total = len(items)
    return float(
        sum(
            row["count"] / total * abs(row["accuracy"] - row["mean_confidence"])
            for row in reliability(items, bins)
            if row["count"]
        )
    )
