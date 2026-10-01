"""Selective-prediction metrics: answer only when confident (PLAN_V3 A4).

For a set of predictions, confidence is the top probability and a prediction is correct as
in :func:`metanoia_analysis.metrics.correctness`. Reported:

- ``auroc``: probability that a correct prediction is more confident than a wrong one
  (ties count one half). 0.5 means confidence carries no signal.
- ``coverage_at_precision``: the share of questions that can be answered while at least the
  target share of answered ones are right. Two versions: ``oracle`` picks the threshold on
  the evaluated set itself (optimistic), ``transfer`` picks it on a reference set (the dev
  split) and reports what it achieves here (honest).
- ``tertiles``: accuracy and share of the least, middle and most confident thirds, with
  edges fixed on the reference set so the thirds mean the same thing in every split.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata

from metanoia_analysis.load import Item
from metanoia_analysis.metrics import correctness


def confidence_and_correct(items: Sequence[Item]) -> tuple[np.ndarray, np.ndarray]:
    confidence = np.array([i.probabilities.max() for i in items], dtype=float)
    correct = np.array([correctness(i) for i in items], dtype=float)
    return confidence, correct


def auroc(confidence: np.ndarray, correct: np.ndarray) -> float | None:
    right = correct >= 0.5
    n_right, n_wrong = int(right.sum()), int((~right).sum())
    if n_right == 0 or n_wrong == 0:
        return None
    ranks = rankdata(confidence)  # average ranks: ties count one half
    return float((ranks[right].sum() - n_right * (n_right + 1) / 2) / (n_right * n_wrong))


def _threshold_for(confidence: np.ndarray, correct: np.ndarray, target: float) -> float | None:
    """Smallest confidence threshold whose answered set (confidence >= it) has precision
    >= target, taking the largest such answered set. None if no set qualifies."""
    order = np.argsort(-confidence, kind="stable")
    conf, corr = confidence[order], correct[order]
    precision = np.cumsum(corr) / np.arange(1, len(corr) + 1)
    # A threshold is only usable between distinct confidences: take the last index of each
    # run of equal confidence.
    last_of_run = np.r_[conf[1:] != conf[:-1], True]
    ok = np.flatnonzero((precision >= target) & last_of_run)
    return float(conf[ok[-1]]) if len(ok) else None


def coverage_at_precision(
    confidence: np.ndarray,
    correct: np.ndarray,
    target: float,
    reference: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Coverage and achieved precision at the threshold for ``target``.

    The threshold is fitted on ``reference`` (confidence, correct) if given, else on the
    evaluated set itself.
    """
    threshold = _threshold_for(*(reference or (confidence, correct)), target)
    if threshold is None:
        return {"target": target, "threshold": None, "coverage": 0.0, "precision": None}
    answered = confidence >= threshold
    return {
        "target": target,
        "threshold": threshold,
        "coverage": float(answered.mean()),
        "precision": float(correct[answered].mean()) if answered.any() else None,
    }


def tertiles(
    confidence: np.ndarray,
    correct: np.ndarray,
    edges: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    low, high = edges if edges is not None else tuple(np.quantile(confidence, [1 / 3, 2 / 3]))
    parts = {
        "least_confident": confidence <= low,
        "middle": (confidence > low) & (confidence <= high),
        "most_confident": confidence > high,
    }
    return [
        {
            "tertile": name,
            "share": float(mask.mean()),
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
            "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
        }
        for name, mask in parts.items()
    ]


def selective_row(
    name: str,
    items: Sequence[Item],
    reference: Sequence[Item],
    precisions: Sequence[float],
) -> dict[str, Any]:
    confidence, correct = confidence_and_correct(items)
    ref = confidence_and_correct(reference) if reference else None
    edges = None
    if ref is not None and len(ref[0]) >= 3:
        q = np.quantile(ref[0], [1 / 3, 2 / 3])
        edges = (float(q[0]), float(q[1]))
    return {
        "slice": name,
        "n": len(items),
        "accuracy": float(correct.mean()),
        "auroc": auroc(confidence, correct),
        "coverage_at_precision": {
            f"{p:g}": {
                "oracle": coverage_at_precision(confidence, correct, p),
                "transfer": coverage_at_precision(confidence, correct, p, ref),
            }
            for p in precisions
        },
        "tertiles": tertiles(confidence, correct, edges),
    }
