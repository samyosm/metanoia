"""Decision requests: inputs, candidates, budgets, and options.

A request carries only inference inputs. Gold labels, proofs, and benchmark-only
annotations are never part of it (PLAN.md §6.1).
"""

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from metanoia_contracts.base import StrictModel

CONTRACT_VERSION = 1

ObservationValue = str | int | float | bool


class Candidate(StrictModel):
    """One outcome of an exclusive-choice decision."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Budget(StrictModel):
    """Explicit operation limits (SCIENTIFIC_FOUNDATIONS §7.1, PLAN.md §6.3).

    All zero means the direct scorer without graph work. Candidate-state capacity is
    reserved separately and is not counted in ``max_nodes``. Depth is derivation
    ancestry, not the number of message-passing rounds.
    """

    max_nodes: int = Field(default=0, ge=0)
    max_queries: int = Field(default=0, ge=0)
    max_proposals: int = Field(default=0, ge=0)
    max_pairs: int = Field(default=0, ge=0)
    max_edges: int = Field(default=0, ge=0)
    max_composites: int = Field(default=0, ge=0)
    max_rounds: int = Field(default=0, ge=0)
    max_depth: int = Field(default=0, ge=0)

    @property
    def is_direct(self) -> bool:
        return all(value == 0 for value in self.model_dump().values())


class TraceMode(StrEnum):
    OFF = "off"
    EVENTS = "events"


class DecisionRequest(StrictModel):
    """An exclusive-choice decision: exactly one candidate is correct."""

    contract_version: Literal[1] = 1
    request_id: str | None = None
    context: str
    question: str = Field(min_length=1)
    candidates: tuple[Candidate, ...] = Field(min_length=2)
    observations: dict[str, ObservationValue] = Field(default_factory=dict)
    budget: Budget = Field(default_factory=Budget)
    trace: TraceMode = TraceMode.OFF
    seed: int | None = None

    @model_validator(mode="after")
    def _unique_candidate_ids(self) -> "DecisionRequest":
        ids = [candidate.id for candidate in self.candidates]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate candidate ids: {duplicates}")
        return self

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.id for candidate in self.candidates)
