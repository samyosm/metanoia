"""A deterministic, parameter-free fixture predictor.

It counts how often each candidate's words occur in the context and question. It is a
tiny known reference for exercising the collection/analysis pipeline, not a claim
about reasoning.
"""

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from metanoia.calibration import softmax
from metanoia_contracts import DecisionRequest, DecisionResult, TraceEvent, TraceMode

_WORD = re.compile(r"\w+")


class LexicalOverlapPredictor:
    predictor_id = "lexical_overlap@1"

    def describe(self) -> dict[str, Any]:
        return {"predictor_id": self.predictor_id, "kind": "lexical_overlap"}

    def decide(self, request: DecisionRequest) -> DecisionResult:
        words = Counter(_WORD.findall(f"{request.context} {request.question}".lower()))
        logits = [
            float(sum(words[w] for w in _WORD.findall(c.text.lower()))) for c in request.candidates
        ]
        trace = None
        if request.trace is TraceMode.EVENTS:
            trace = (TraceEvent(seq=0, op="lexical_count", details={"logits": logits}),)
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
