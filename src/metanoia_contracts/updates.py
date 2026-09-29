"""Progress of one decision while it is computed (streaming ``serve``).

A stream is a sequence of updates followed by the ordinary ``DecisionResult``, which
is identical to a non-streamed decision. Updates describe stages that were actually
computed, in order; they are for display and carry no guarantee beyond that. Their
probabilities are an uncalibrated softmax: a calibration applies only to the final
result, at the budget it was fitted for.
"""

from typing import Literal

from pydantic import Field

from metanoia_contracts.base import StrictModel


class NodeView(StrictModel):
    """One memory node in a candidate's branch."""

    id: str
    concept: str
    text: str
    similarity: float


class DecisionUpdate(StrictModel):
    seq: int = Field(ge=0)
    # render: the input text; score: the decision after reading ``width`` nodes per
    # candidate (0 = the direct answer); expand: one candidate's proposed nodes;
    # contribution: one candidate's per-node leave-one-out contributions.
    stage: Literal["render", "score", "expand", "contribution"]
    elapsed_ms: float = Field(ge=0)
    text: str | None = None
    width: int | None = Field(default=None, ge=0)
    candidate_ids: tuple[str, ...] = ()
    logits: tuple[float, ...] | None = None
    probabilities: tuple[float, ...] | None = None
    candidate: str | None = None
    nodes: tuple[NodeView, ...] = ()
    contributions: tuple[float, ...] | None = None
