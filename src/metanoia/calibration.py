"""Apply an already fitted calibration artifact. Fitting belongs to metanoia_analysis.

Pure Python in float64 so that core and offline application agree numerically.
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path

from metanoia_contracts import Budget, CalibrationInfo, CalibrationStatus
from metanoia_contracts.artifacts import CalibrationArtifact
from metanoia_contracts.base import load_strict


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = math.fsum(exps)
    return [e / total for e in exps]


class Calibrator:
    """Identity temperature unless constructed from a fitted artifact."""

    def __init__(self, artifact: CalibrationArtifact | None = None) -> None:
        self.artifact = artifact

    @classmethod
    def load(cls, path: Path, predictor_id: str) -> "Calibrator":
        artifact = load_strict(CalibrationArtifact, json.loads(path.read_text()))
        if artifact.compatible_predictor_id != predictor_id:
            raise ValueError(
                f"calibration {artifact.calibrator_id} was fitted for "
                f"{artifact.compatible_predictor_id}, not {predictor_id}"
            )
        return cls(artifact)

    @property
    def info(self) -> CalibrationInfo:
        if self.artifact is None:
            return CalibrationInfo()
        return CalibrationInfo(
            status=CalibrationStatus.FITTED,
            calibrator_id=self.artifact.calibrator_id,
            temperature=self.artifact.temperature,
        )

    def check_budget(self, budget: Budget) -> None:
        if self.artifact is not None and budget != self.artifact.budget:
            raise ValueError(
                f"calibration {self.artifact.calibrator_id} was fitted for budget "
                f"{self.artifact.budget_id}; refusing to apply it to a different budget"
            )

    def probabilities(self, logits: Sequence[float]) -> list[float]:
        return softmax(logits, self.info.temperature)
