"""Uncertainty intervals that resample independent groups (e.g. generated worlds).

Examples from one world share entities and templates, so resampling examples as if
independent would understate uncertainty (SCIENTIFIC_FOUNDATIONS §13.5).
"""

from collections.abc import Sequence

import numpy as np


def group_bootstrap_ci(
    values: np.ndarray,
    groups: Sequence[str],
    resamples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float] | None:
    """Percentile interval for the example-weighted mean. None if not computable."""
    if resamples == 0 or len(values) == 0:
        return None
    names, index = np.unique(np.asarray(groups), return_inverse=True)
    if len(names) < 2:
        return None
    sums = np.bincount(index, weights=values, minlength=len(names))
    counts = np.bincount(index, minlength=len(names)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(resamples, len(names)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def reversal_rates(correct_before: np.ndarray, correct_after: np.ndarray) -> dict[str, float]:
    """Both directions (eq. 46-47): accuracy change = wrong_to_right - right_to_wrong."""
    before = correct_before.astype(bool)
    after = correct_after.astype(bool)
    return {
        "wrong_to_right": float(np.mean(~before & after)),
        "right_to_wrong": float(np.mean(before & ~after)),
    }
