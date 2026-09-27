"""Joint candidate encoding keeps the Stage-1 gates; the decide CLI builds valid requests."""

import argparse

import pytest
import torch

from metanoia.build import build_model
from metanoia.cli import build_request
from metanoia.config import DirectScorerConfig, HashingEncoderConfig, ModelConfig
from metanoia_contracts import Candidate, DecisionRequest

JOINT = ModelConfig(
    name="joint",
    encoder=HashingEncoderConfig(buckets=256, dim=32, layers=1),
    scorer=DirectScorerConfig(candidate_encoding="joint"),
)


def make(names: list[str]) -> DecisionRequest:
    return DecisionRequest(
        context="Alpha is bright. Beta is dim.",
        question="Which one is bright?",
        candidates=tuple(Candidate(id=n.lower(), text=n) for n in names),
    )


def test_joint_encoding_is_permutation_equivariant() -> None:
    model = build_model(JOINT, seed=0).eval()
    names = ["Alpha", "Beta", "Gamma"]
    base = model([make(names)]).logits[0]
    order = [2, 0, 1]
    permuted = model([make([names[i] for i in order])]).logits[0]
    torch.testing.assert_close(permuted, base[order], rtol=1e-5, atol=1e-5)


def test_joint_differs_from_separate() -> None:
    separate = build_model(JOINT.model_copy(update={"scorer": DirectScorerConfig()}), seed=0)
    joint = build_model(JOINT, seed=0)
    request = make(["Alpha", "Beta"])
    assert not torch.allclose(separate.eval()([request]).logits, joint.eval()([request]).logits)


def args(**kw) -> argparse.Namespace:
    base = dict(request=None, question=None, candidate=None, context=None, context_file=None)
    return argparse.Namespace(**{**base, "trace": False, **kw})


def test_cli_builds_request_with_unique_ids() -> None:
    request = build_request(
        args(question="Capital of France?", candidate=["Paris", "paris!", "Rome"], context="")
    )
    assert request.candidate_ids == ("paris", "paris_2", "rome")
    assert [c.text for c in request.candidates] == ["Paris", "paris!", "Rome"]


def test_cli_requires_two_candidates() -> None:
    with pytest.raises(ValueError, match="two"):
        build_request(args(question="Q?", candidate=["only"]))
