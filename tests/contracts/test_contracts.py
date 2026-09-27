import math

import pytest
from pydantic import ValidationError

from metanoia_contracts import (
    Budget,
    CalibrationInfo,
    CalibrationStatus,
    Candidate,
    DecisionRequest,
    DecisionResult,
)
from metanoia_contracts.artifacts import PredictionRecord, PredictionStatus
from metanoia_contracts.base import load_strict
from metanoia_contracts.provenance import content_sha256

CANDIDATES = (Candidate(id="a", text="A"), Candidate(id="b", text="B"))


def request(**overrides: object) -> DecisionRequest:
    data: dict = {
        "context": "c",
        "question": "q?",
        "candidates": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}],
    }
    data.update(overrides)
    return load_strict(DecisionRequest, data)


def test_request_round_trip() -> None:
    r = request(observations={"units": "V", "reading": 3.5})
    assert DecisionRequest.model_validate_json(r.model_dump_json()) == r
    assert r.budget.is_direct


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidates": [{"id": "a", "text": "A"}, {"id": "a", "text": "B"}]},  # duplicate ids
        {"candidates": [{"id": "a", "text": "A"}]},  # a decision needs >= 2 outcomes
        {"label": "a"},  # gold answers are not request fields
        {"budget": {"max_nodes": -1}},
        {"budget": {"max_nodez": 3}},  # unknown keys are rejected
        {"question": 3},  # no coercion
    ],
)
def test_invalid_requests_rejected(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        request(**overrides)


def result(**overrides: object) -> DecisionResult:
    data: dict = {
        "predictor_id": "p",
        "candidate_ids": ("a", "b"),
        "logits": (1.0, 0.0),
        "probabilities": (0.7310585786300049, 0.2689414213699951),
        "budget": Budget(),
    }
    data.update(overrides)
    return DecisionResult(**data)


def test_result_is_uncalibrated_by_default() -> None:
    assert result().calibration.status is CalibrationStatus.UNCALIBRATED


@pytest.mark.parametrize(
    "overrides",
    [
        {"logits": (1.0,)},
        {"logits": (math.nan, 0.0)},
        {"probabilities": (0.5, 0.6)},
        {"candidate_ids": ("a", "a")},
    ],
)
def test_invalid_results_rejected(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        result(**overrides)


def test_calibration_identity_required() -> None:
    with pytest.raises(ValidationError):
        CalibrationInfo(status=CalibrationStatus.FITTED)
    with pytest.raises(ValidationError):
        CalibrationInfo(temperature=2.0)


def test_failed_prediction_records_keep_error_and_no_scores() -> None:
    base = {
        "run_id": "r",
        "example_id": "e",
        "budget_id": "b0",
        "repeat_id": 0,
        "candidate_ids": ["a", "b"],
        "wall_time_ms": 1.0,
        "batch_index": 0,
    }
    PredictionRecord(
        **base, status=PredictionStatus.FAILED, error="boom", logits=[], probabilities=[]
    )
    with pytest.raises(ValidationError):
        PredictionRecord(**base, status=PredictionStatus.FAILED, logits=[], probabilities=[])
    with pytest.raises(ValidationError):
        PredictionRecord(**base, status=PredictionStatus.OK, logits=[0.0], probabilities=[1.0])


def test_non_finite_rows_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        load_strict(Budget, {"max_nodes": math.inf})


def test_content_hash_ignores_key_order() -> None:
    assert content_sha256([{"a": 1, "b": 2}]) == content_sha256([{"b": 2, "a": 1}])
    assert content_sha256([{"a": 1}]) != content_sha256([{"a": 2}])
