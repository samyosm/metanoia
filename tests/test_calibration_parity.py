"""Core application of a fitted temperature must match offline analysis (PLAN.md §7.5)."""

import numpy as np

from metanoia.calibration import softmax as core_softmax
from metanoia_analysis.calibration import softmax as analysis_softmax


def test_core_and_analysis_softmax_agree() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        logits = rng.normal(0, 10, size=rng.integers(2, 8))
        temperature = float(np.exp(rng.uniform(-3, 3)))
        np.testing.assert_allclose(
            core_softmax(logits.tolist(), temperature),
            analysis_softmax(logits, temperature),
            rtol=0,
            atol=1e-12,
        )
