"""EXPAND v1 invariants (decision 0011), offline: tiny memory, random concepts."""

import numpy as np
import pytest
import torch

from metanoia.api import TorchPredictor
from metanoia.build import build_model
from metanoia.config import DirectScorerConfig, ExpansionConfig, HashingEncoderConfig, ModelConfig
from metanoia_contracts import Budget, Candidate, DecisionRequest, Termination, TraceMode
from tests.core.test_memory import write_memory

EXPAND = Budget(max_depth=1, max_queries=8, max_proposals=64, max_nodes=64)


def config(memory, concepts="random", proposals=3) -> ModelConfig:
    return ModelConfig(
        name="ex",
        encoder=HashingEncoderConfig(buckets=256, dim=32, layers=1),
        scorer=DirectScorerConfig(latent_dim=32, hidden_dim=32, candidate_encoding="joint"),
        expansion=ExpansionConfig(
            memory=str(memory), concepts=concepts, proposals=proposals, dim=16, hidden_dim=16
        ),
    )


@pytest.fixture
def memory(tmp_path):
    write_memory(tmp_path / "mem", np.random.default_rng(0).normal(size=(12, 8)))
    return tmp_path / "mem"


def make(names: list[str], budget: Budget = EXPAND) -> DecisionRequest:
    return DecisionRequest(
        context="",
        question="Which one conducts electricity?",
        candidates=tuple(Candidate(id=n.lower(), text=n) for n in names),
        budget=budget,
    )


def test_budget_zero_is_exactly_the_direct_scorer(memory) -> None:
    model = build_model(config(memory), seed=0).eval()
    request = make(["copper", "wood"], Budget())
    out = model([request])
    direct = super(type(model), model).forward([request]).logits
    torch.testing.assert_close(out.logits, direct)
    assert out.expanded == [0] and out.evidence is None


def test_expansion_uses_the_budget_and_stays_equivariant(memory) -> None:
    model = build_model(config(memory), seed=0).eval()
    names = ["copper", "wood", "glass"]
    budget = EXPAND.model_copy(update={"max_proposals": 6, "max_nodes": 6})  # 2 per candidate
    out = model([make(names, budget)])
    assert out.expanded == [2] and all(len(c) == 2 for c in out.concepts[0])
    assert not torch.allclose(out.logits, out.initial_logits)
    order = [2, 0, 1]
    permuted = model([make([names[i] for i in order], budget)]).logits[0]
    torch.testing.assert_close(permuted, out.logits[0][order], rtol=1e-4, atol=1e-4)


def test_insufficient_budget_is_reported(memory) -> None:
    model = build_model(config(memory), seed=0).eval()
    starved = EXPAND.model_copy(update={"max_queries": 1})  # fewer queries than candidates
    out = model([make(["copper", "wood"], starved)])
    assert out.expanded == [0] and out.termination == [Termination.BUDGET_EXHAUSTED]


def test_predictor_reports_usage_and_the_expansion_tree(memory) -> None:
    model = build_model(config(memory), seed=0).eval()
    request = make(["copper", "wood"]).model_copy(update={"trace": TraceMode.EVENTS})
    result = TorchPredictor(model, "ex@x", {"model": {"expansion": {}}}).decide(request)
    # The budget allows 32 concepts per candidate; the 12-concept memory caps it.
    assert result.usage.queries == 2 and result.usage.proposals == 24 and result.usage.depth == 1
    assert result.usage_within_budget()
    ops = [e.op for e in result.trace]
    assert ops == ["render", "encode", "direct_score", "expand", "expand", "aggregate", "calibrate"]
    expand = result.trace[3].details
    assert len(expand["concepts"]) == 12 and expand["candidate"] == "copper"
    aggregate = result.trace[5].details
    initial = result.trace[2].details["logits"]
    assert [a + e for a, e in zip(initial, aggregate["evidence"], strict=True)] == pytest.approx(
        list(result.logits), abs=1e-4
    )


def test_none_control_encodes_no_concepts(memory) -> None:
    model = build_model(config(memory, concepts="none"), seed=0).eval()
    seen: list[str] = []
    encode = model.encoder.encode
    model.encoder.encode = lambda texts: (seen.extend(texts), encode(texts))[1]
    out = model([make(["copper", "wood"])], force=True)
    assert not any(t.startswith("concept:") for t in seen)
    assert out.concepts == [[[], []]]


def test_direct_path_is_frozen(memory) -> None:
    model = build_model(config(memory), seed=0)
    trainable = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"expansion"}
