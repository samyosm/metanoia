"""Stage-1 gates: candidate symmetry, padding, gradients, tiny-task learning, reload."""

import json
from pathlib import Path

import pytest
import torch

from metanoia.api import TorchPredictor, load_predictor
from metanoia.build import build_model
from metanoia.config import HashingEncoderConfig, ModelConfig
from metanoia_contracts import Candidate, DecisionRequest

CONFIG = ModelConfig(name="test", encoder=HashingEncoderConfig(buckets=256, dim=32, layers=1))


def make(context: str, names: list[str]) -> DecisionRequest:
    return DecisionRequest(
        context=context,
        question="Which one is bright?",
        candidates=tuple(Candidate(id=n.lower(), text=n) for n in names),
    )


@pytest.fixture
def model() -> torch.nn.Module:
    return build_model(CONFIG, seed=0).eval()


def test_permutation_equivariance(model) -> None:
    names = ["Alpha", "Beta", "Gamma", "Delta"]
    base = model([make("Alpha is bright. Beta is dim.", names)]).logits[0]
    order = [2, 0, 3, 1]
    permuted = model([make("Alpha is bright. Beta is dim.", [names[i] for i in order])]).logits[0]
    torch.testing.assert_close(permuted, base[order], rtol=1e-5, atol=1e-5)


def test_padding_does_not_change_scores(model) -> None:
    short = make("Alpha is bright.", ["Alpha", "Beta"])
    long = make("Gamma is dim. " * 20, ["Gamma", "Delta", "Alpha", "Beta"])
    alone = model([short]).logits[0]
    batched = model([short, long])
    torch.testing.assert_close(batched.logits[0, :2], alone, rtol=1e-5, atol=1e-5)
    assert torch.isinf(batched.logits[0, 2:]).all()
    assert batched.candidate_mask.tolist() == [[True, True, False, False], [True] * 4]


def test_gradients_reach_encoder_and_scorer(model) -> None:
    model.train()
    output = model([make("Alpha is bright.", ["Alpha", "Beta"])])
    torch.nn.functional.cross_entropy(output.logits, torch.tensor([0])).backward()
    for part in (model.encoder, model.scorer):
        norms = [p.grad.norm() for p in part.parameters() if p.grad is not None]
        assert norms and float(torch.stack(norms).sum()) > 0


@pytest.mark.slow
def test_tiny_task_is_learned() -> None:
    """The context names who is bright; candidate order is shuffled across examples."""
    torch.manual_seed(0)
    model = build_model(CONFIG, seed=0)
    names = ["Alpha", "Beta", "Gamma", "Delta"]
    examples = []
    for i in range(16):
        answer = names[i % 4]
        order = names[i % 3 :] + names[: i % 3]
        others = [n for n in names if n != answer]
        context = f"{answer} is bright. " + " ".join(f"{n} is dim." for n in others)
        examples.append((make(context, order), order.index(answer)))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    targets = torch.tensor([t for _, t in examples])
    for _ in range(150):
        loss = torch.nn.functional.cross_entropy(model([r for r, _ in examples]).logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1


def test_bundle_round_trip(tmp_path: Path, model) -> None:
    from metanoia_contracts.artifacts import FileRef, PredictorBundleManifest
    from metanoia_contracts.provenance import sha256_file, utc_now

    torch.save(model.persistent_state_dict(), tmp_path / "weights.pt")
    manifest = PredictorBundleManifest(
        predictor_id="test@x",
        resolved_config={
            "model": CONFIG.model_dump(mode="json"),
            "encoder_identity": model.encoder.identity(),
        },
        weights=FileRef(path="weights.pt", sha256=sha256_file(tmp_path / "weights.pt"), rows=0),
        created_at=utc_now(),
    )
    (tmp_path / "bundle.json").write_text(json.dumps(manifest.model_dump(mode="json")))
    request = make("Alpha is bright.", ["Alpha", "Beta", "Gamma"])
    before = TorchPredictor(model, "test@x", {}).decide(request)
    after = load_predictor(tmp_path).decide(request)
    assert after.logits == pytest.approx(before.logits, abs=1e-6)

    (tmp_path / "weights.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        load_predictor(tmp_path)


def test_trace_capture_does_not_change_prediction(model) -> None:
    from metanoia_contracts import TraceMode

    request = make("Alpha is bright.", ["Alpha", "Beta"])
    predictor = TorchPredictor(model, "test@x", {})
    plain = predictor.decide(request)
    traced = predictor.decide(request.model_copy(update={"trace": TraceMode.EVENTS}))
    assert traced.trace and plain.trace is None
    assert traced.logits == plain.logits and traced.probabilities == plain.probabilities
