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


def test_trace_records_the_direct_path() -> None:
    from metanoia.api import TorchPredictor
    from metanoia_contracts import TraceMode

    model = build_model(JOINT, seed=0).eval()
    request = make(["Alpha", "Beta"]).model_copy(update={"trace": TraceMode.EVENTS})
    result = TorchPredictor(model, "t@x", {}).decide(request)
    assert result.trace is not None
    assert [e.op for e in result.trace] == ["render", "encode", "direct_score", "calibrate"]
    encode = result.trace[1].details
    assert encode["candidate_encoding"] == "joint" and encode["truncated"] is False
    assert result.trace[2].details["logits"] == list(result.logits)
    assert result.trace[3].details["probabilities"] == list(result.probabilities)


def test_serve_answers_json_lines(tmp_path, monkeypatch, capsys) -> None:
    import io
    import json

    import metanoia.cli as cli
    from metanoia.api import TorchPredictor

    model = build_model(JOINT, seed=0).eval()
    monkeypatch.setattr(cli, "_load", lambda args: TorchPredictor(model, "t@x", {}))
    good = {"id": 1, "request": make(["Alpha", "Beta"]).model_dump(mode="json")}
    bad = {"id": 2, "request": {"question": "q"}}
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{json.dumps(good)}\n\n{json.dumps(bad)}\n"))
    cli.main(["serve", "--bundle", "unused"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["ready"] is True
    assert lines[1]["id"] == 1 and len(lines[1]["result"]["probabilities"]) == 2
    assert lines[2]["id"] == 2 and "error" in lines[2]
