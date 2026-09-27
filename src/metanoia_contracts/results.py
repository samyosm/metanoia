"""Decision results: scores, probabilities, calibration identity, costs, and status."""

import math
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from metanoia_contracts.base import StrictModel
from metanoia_contracts.requests import Budget
from metanoia_contracts.traces import TraceEvent

PROBABILITY_SUM_TOLERANCE = 1e-6


class CalibrationStatus(StrEnum):
    # A plain softmax of the logits. Schema validity says nothing about empirical calibration.
    UNCALIBRATED = "uncalibrated"
    # Probabilities use a calibration artifact fitted on a declared calibration split.
    FITTED = "fitted"


class CalibrationInfo(StrictModel):
    status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
    calibrator_id: str | None = None
    temperature: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _identity_matches_status(self) -> "CalibrationInfo":
        if self.status is CalibrationStatus.FITTED and self.calibrator_id is None:
            raise ValueError("a fitted calibration must name its calibrator_id")
        if self.status is CalibrationStatus.UNCALIBRATED and (
            self.calibrator_id is not None or self.temperature != 1.0
        ):
            raise ValueError("uncalibrated output cannot carry a calibrator or temperature")
        return self


class ResourceUsage(StrictModel):
    """Consumed work, counted by the engine. Mirrors :class:`Budget`."""

    nodes: int = Field(default=0, ge=0)
    queries: int = Field(default=0, ge=0)
    proposals: int = Field(default=0, ge=0)
    pairs: int = Field(default=0, ge=0)
    edges: int = Field(default=0, ge=0)
    composites: int = Field(default=0, ge=0)
    rounds: int = Field(default=0, ge=0)
    depth: int = Field(default=0, ge=0)


class Termination(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    POLICY_STOP = "policy_stop"


class DecisionResult(StrictModel):
    """Scores over the original candidates, in the order of ``candidate_ids``.

    ``logits`` are raw (pre-calibration) scores; ``probabilities`` are
    ``softmax(logits / calibration.temperature)``.
    """

    contract_version: Literal[1] = 1
    request_id: str | None = None
    predictor_id: str = Field(min_length=1)
    candidate_ids: tuple[str, ...] = Field(min_length=2)
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    calibration: CalibrationInfo = Field(default_factory=CalibrationInfo)
    budget: Budget
    usage: ResourceUsage = Field(default_factory=ResourceUsage)
    termination: Termination = Termination.COMPLETED
    trace: tuple[TraceEvent, ...] | None = None

    @model_validator(mode="after")
    def _consistent_scores(self) -> "DecisionResult":
        n = len(self.candidate_ids)
        if len(set(self.candidate_ids)) != n:
            raise ValueError("duplicate candidate ids in result")
        if len(self.logits) != n or len(self.probabilities) != n:
            raise ValueError(
                f"expected {n} logits and probabilities, got "
                f"{len(self.logits)} and {len(self.probabilities)}"
            )
        if not all(math.isfinite(x) for x in self.logits):
            raise ValueError("logits must be finite")
        if not all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in self.probabilities):
            raise ValueError("probabilities must lie in [0, 1]")
        if abs(math.fsum(self.probabilities) - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError("probabilities must sum to 1")
        return self

    def usage_within_budget(self) -> bool:
        b, u = self.budget, self.usage
        return (
            u.nodes <= b.max_nodes
            and u.queries <= b.max_queries
            and u.proposals <= b.max_proposals
            and u.pairs <= b.max_pairs
            and u.edges <= b.max_edges
            and u.composites <= b.max_composites
            and u.rounds <= b.max_rounds
            and u.depth <= b.max_depth
        )
