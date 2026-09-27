"""Evaluation protocols: the fixed rules a collection run follows (PLAN.md §8.1)."""

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from metanoia_contracts.base import StrictModel
from metanoia_contracts.requests import Budget, TraceMode


class FailurePolicy(StrEnum):
    # Failed predictions are scored as a uniform distribution and always counted.
    UNIFORM = "uniform"
    # Any failed prediction makes the run unusable for official scoring.
    ERROR = "error"


class NamedBudget(StrictModel):
    id: str = Field(min_length=1)
    budget: Budget = Field(default_factory=Budget)


class DatasetPin(StrictModel):
    """Immutable content references for the cases and labels a suite evaluates."""

    dataset_id: str
    cases_sha256: str
    labels_sha256: str


class EvaluationRules(StrictModel):
    splits: tuple[str, ...] = Field(min_length=1)
    batch_size: int = Field(default=1, ge=1)
    repeats: int = Field(default=1, ge=1)
    base_seed: int = 0
    trace: TraceMode = TraceMode.OFF
    failure_policy: FailurePolicy = FailurePolicy.UNIFORM
    timing_scope: Literal["decide_call"] = "decide_call"


class SuiteProtocol(StrictModel):
    protocol_version: Literal[1] = 1
    suite_id: str
    revision: int = Field(ge=1)
    task: Literal["exclusive_choice"] = "exclusive_choice"
    question: str
    dataset: DatasetPin
    evaluation: EvaluationRules
    budgets: tuple[NamedBudget, ...] = Field(min_length=1)
    analysis_specification: str

    @model_validator(mode="after")
    def _unique_budget_ids(self) -> "SuiteProtocol":
        ids = [b.id for b in self.budgets]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate budget ids")
        return self
