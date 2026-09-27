"""Fit a positive scalar temperature on calibration-split predictions only.

Temperature scaling cannot change the top-ranked candidate, so it cannot explain an
accuracy gain (SCIENTIFIC_FOUNDATIONS §3.3). Fitting uses saved raw logits; applying
the resulting artifact in ``metanoia.calibration`` must agree numerically with
:func:`softmax`.
"""

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from metanoia_analysis.load import Item

LOG_TEMPERATURE_BOUNDS = (-5.0, 5.0)


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    exps = np.exp(scaled - scaled.max())
    return exps / exps.sum()


def mean_nll(items: Sequence[Item], temperature: float) -> float:
    total = 0.0
    for item in items:
        assert item.logits is not None
        scaled = item.logits / temperature
        top = scaled.max()
        total += top + np.log(np.exp(scaled - top).sum()) - scaled[item.label_index]
    return total / len(items)


def fit_temperature(items: Sequence[Item]) -> float:
    usable = [i for i in items if not i.failed]
    if not usable:
        raise ValueError("no successful predictions to fit calibration on")
    result = minimize_scalar(
        lambda log_t: mean_nll(usable, float(np.exp(log_t))),
        bounds=LOG_TEMPERATURE_BOUNDS,
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"temperature fit failed: {result.message}")
    return float(np.exp(result.x))


def at_search_bound(temperature: float, tolerance: float = 1e-3) -> bool:
    """A fit on the bound means the optimum may lie outside it (e.g. near-uniform output)."""
    log_t = float(np.log(temperature))
    return any(abs(log_t - bound) < tolerance for bound in LOG_TEMPERATURE_BOUNDS)


def apply_temperature(items: Sequence[Item], temperature: float) -> list[Item]:
    """Re-score from raw logits. Failed items keep their policy-assigned distribution."""
    return [
        item if item.logits is None else item.with_probabilities(softmax(item.logits, temperature))
        for item in items
    ]
