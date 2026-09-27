"""Schemas for the artifacts that connect data preparation, collection, and analysis.

Formats (PLAN.md §6.4): JSON manifests, JSONL cases and traces, Parquet tables for
labels and predictions. Tables are validated row by row against the models below;
the Parquet reader and writer live in the areas that need them.
"""

import math
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from metanoia_contracts.base import StrictModel
from metanoia_contracts.requests import Budget, DecisionRequest
from metanoia_contracts.results import CalibrationStatus, ResourceUsage, Termination

AnnotationValue = str | int | float | bool


class FileRef(StrictModel):
    """A file inside an artifact directory, identified by a content hash."""

    path: str
    sha256: str
    rows: int = Field(ge=0)


# --- Dataset bundle ---------------------------------------------------------------


class CaseRecord(StrictModel):
    """One line of ``cases.jsonl``: the only data a predictor may see, plus routing keys."""

    example_id: str
    group_id: str
    split: str
    request: DecisionRequest


# ``labels.parquet`` stores these columns; every other column is an annotation.
LABEL_CORE_COLUMNS = ("example_id", "group_id", "split", "label_id")


class LabelRecord(StrictModel):
    """One row of ``labels.parquet``. Annotations are benchmark-only, never model input."""

    example_id: str
    group_id: str
    split: str
    label_id: str
    annotations: dict[str, AnnotationValue] = Field(default_factory=dict)


class DatasetManifest(StrictModel):
    dataset_version: Literal[1] = 1
    dataset_id: str
    task: Literal["exclusive_choice"] = "exclusive_choice"
    generator: dict[str, Any]
    splits: dict[str, int]
    cases: FileRef
    labels: FileRef
    proofs: FileRef | None = None
    created_at: str


# --- Predictor bundle -------------------------------------------------------------


class PredictorBundleManifest(StrictModel):
    bundle_version: Literal[1] = 1
    predictor_id: str
    contract_version: Literal[1] = 1
    resolved_config: dict[str, Any]
    weights: FileRef
    memory: FileRef | None = None
    training: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    source: dict[str, Any] = Field(default_factory=dict)
    created_at: str


# --- Collection run ---------------------------------------------------------------


class PredictionStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"


class PredictionRecord(StrictModel):
    """One row of ``predictions.parquet``, keyed by (run, example, budget, repeat)."""

    run_id: str
    example_id: str
    budget_id: str
    repeat_id: int = Field(ge=0)
    status: PredictionStatus
    error: str | None = None
    candidate_ids: list[str]
    logits: list[float]
    probabilities: list[float]
    calibration_status: CalibrationStatus | None = None
    calibrator_id: str | None = None
    temperature: float | None = None
    termination: Termination | None = None
    usage_nodes: int = 0
    usage_queries: int = 0
    usage_proposals: int = 0
    usage_pairs: int = 0
    usage_edges: int = 0
    usage_composites: int = 0
    usage_rounds: int = 0
    usage_depth: int = 0
    wall_time_ms: float = Field(ge=0)
    batch_index: int = Field(ge=0)

    @model_validator(mode="after")
    def _status_consistent(self) -> "PredictionRecord":
        n = len(self.candidate_ids)
        if len(set(self.candidate_ids)) != n:
            raise ValueError("duplicate candidate ids")
        if self.status is PredictionStatus.OK:
            if len(self.logits) != n or len(self.probabilities) != n:
                raise ValueError("score arrays must match candidate_ids")
            if not all(math.isfinite(x) for x in [*self.logits, *self.probabilities]):
                raise ValueError("scores must be finite")
            if abs(math.fsum(self.probabilities) - 1.0) > 1e-6:
                raise ValueError("probabilities must sum to 1")
            if self.termination is None or self.calibration_status is None:
                raise ValueError("successful predictions need termination and calibration")
        elif self.logits or self.probabilities or not self.error:
            raise ValueError("failed predictions carry an error and no scores")
        return self

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.example_id, self.budget_id, self.repeat_id)

    @staticmethod
    def usage_columns(usage: ResourceUsage) -> dict[str, int]:
        return {f"usage_{name}": value for name, value in usage.model_dump().items()}


class RunCounts(StrictModel):
    expected: int = Field(ge=0)
    attempted: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)


class RunManifest(StrictModel):
    run_version: Literal[1] = 1
    run_id: str
    status: Literal["complete"] = "complete"
    suite: dict[str, Any]
    protocol_sha256: str
    protocol: dict[str, Any]
    dataset_id: str
    cases_sha256: str
    labels_sha256: str
    predictor: dict[str, Any]
    runner_revision: str
    environment: dict[str, Any]
    source: dict[str, Any]
    counts: RunCounts
    predictions: FileRef
    traces: FileRef | None = None
    started_at: str
    finished_at: str


# --- Calibration ------------------------------------------------------------------


class CalibrationArtifact(StrictModel):
    """A fitted positive scalar temperature (SCIENTIFIC_FOUNDATIONS §3.3)."""

    calibration_version: Literal[1] = 1
    calibrator_id: str
    method: Literal["temperature"] = "temperature"
    temperature: float = Field(gt=0)
    compatible_predictor_id: str
    # Calibration is valid only for the budget/stopping policy it was fitted under.
    budget_id: str
    budget: Budget
    fitted_on: dict[str, Any]
    analysis_revision: str
    created_at: str


# --- Analysis ---------------------------------------------------------------------


class MetricValue(StrictModel):
    split: str
    budget_id: str
    # "raw" scores stored probabilities; "calibrated" re-scores raw logits offline.
    variant: str = "raw"
    # "all", or "<annotation>=<value>" for a declared slice.
    slice: str = "all"
    metric: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n: int = Field(ge=0)


class AnalysisResult(StrictModel):
    analysis_version: Literal[1] = 1
    analysis_id: str
    kind: Literal["analyze", "compare"]
    inputs: dict[str, Any]
    specification: dict[str, Any]
    analysis_revision: str
    environment: dict[str, Any]
    counts: dict[str, dict[str, int]]
    metrics: list[MetricValue]
    tables: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    limitations: list[str]
    created_at: str


# --- Checkpoint selection ---------------------------------------------------------


class SelectionCandidate(StrictModel):
    run_id: str
    predictor_id: str
    predictor_path: str | None
    predictions_sha256: str
    value: float
    n: int = Field(ge=1)


class SelectionArtifact(StrictModel):
    """Which predictor (e.g. training checkpoint) a rule chose from saved predictions.

    Written by the analysis process; training and collection consume only the chosen
    identity, never the metric implementation (PLAN.md §7.2).
    """

    selection_version: Literal[1] = 1
    selection_id: str
    split: str
    budget_id: str
    metric: str
    rule: Literal["min", "max"]
    labels_sha256: str
    candidates: tuple[SelectionCandidate, ...] = Field(min_length=1)
    chosen: SelectionCandidate
    analysis_revision: str
    created_at: str
