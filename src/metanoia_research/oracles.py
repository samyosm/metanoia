"""Controlled symbolic reference predictors.

The micro-world oracle sees only the request (never labels or proofs): it parses the
rendered context with the generator's exact templates and forward-chains the rules.
Comparing it with learned predictors separates language-grounding error from
inference error (SCIENTIFIC_FOUNDATIONS §4.5, §13.4 item 9).
"""

from collections.abc import Sequence
from typing import Any

from metanoia.calibration import softmax
from metanoia_contracts import DecisionRequest, DecisionResult, TraceEvent, TraceMode
from metanoia_research.data.microworld import (
    GENERATOR_REVISION,
    forward_chain,
    parse_context,
    parse_question,
)


class MicroworldOracle:
    """Logit ``margin`` for candidates that provably hold the target, 0 otherwise.

    The margin is a declared constant, so its probabilities are not calibrated beliefs.
    """

    def __init__(self, margin: float = 20.0) -> None:
        self.margin = margin

    @property
    def predictor_id(self) -> str:
        return f"microworld_oracle@{GENERATOR_REVISION}/margin={self.margin:g}"

    def describe(self) -> dict[str, Any]:
        return {"predictor_id": self.predictor_id, "kind": "oracle", "margin": self.margin}

    def decide(self, request: DecisionRequest) -> DecisionResult:
        world = parse_context(request.context)
        target = parse_question(request.question)
        known = forward_chain(world.facts, world.rules)
        logits = [self.margin if (c.text, target) in known else 0.0 for c in request.candidates]
        trace = None
        if request.trace is TraceMode.EVENTS:
            trace = (
                TraceEvent(
                    seq=0,
                    op="forward_chain",
                    details={"facts": len(world.facts), "rules": len(world.rules)},
                ),
            )
        return DecisionResult(
            request_id=request.request_id,
            predictor_id=self.predictor_id,
            candidate_ids=request.candidate_ids,
            logits=tuple(logits),
            probabilities=tuple(softmax(logits)),
            budget=request.budget,
            trace=trace,
        )

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResult]:
        return [self.decide(r) for r in requests]
