"""Stage-3a workspace invariants (decision 0008)."""

import pytest
import torch

from metanoia.api import TorchPredictor
from metanoia.build import build_model
from metanoia.config import DirectScorerConfig, HashingEncoderConfig, ModelConfig, WorkspaceConfig
from metanoia.reasoning.workspace import sentence_spans, statement_nodes
from metanoia_contracts import Budget, Candidate, DecisionRequest, Termination, TraceMode

BASE = dict(
    name="ws",
    encoder=HashingEncoderConfig(buckets=256, dim=32, layers=1),
    scorer=DirectScorerConfig(latent_dim=32, hidden_dim=32, candidate_encoding="joint"),
)
STATEMENTS = ModelConfig(**BASE, workspace=WorkspaceConfig(dim=32, hidden_dim=32))
LATENT = ModelConfig(**BASE, workspace=WorkspaceConfig(dim=32, hidden_dim=32, mode="latent"))
ROOMY = Budget(max_nodes=64, max_pairs=10**7, max_rounds=3)


def make(names: list[str], budget: Budget = ROOMY) -> DecisionRequest:
    return DecisionRequest(
        context="Alpha is dax. Everything that is dax is also wug. Beta is fep.",
        question="Which one is wug?",
        candidates=tuple(Candidate(id=n.lower(), text=n) for n in names),
        budget=budget,
    )


def perturbed(config: ModelConfig) -> torch.nn.Module:
    model = build_model(config, seed=0).eval()
    torch.nn.init.normal_(model.workspace.out.weight, std=0.5)  # leave the zero start
    return model


def test_sentence_spans_and_pooling() -> None:
    text = "question: Which one is wug?\ncontext: Alpha is dax. Beta is fep."
    spans = sentence_spans(text)
    assert [text[a:b] for a, b in spans] == [
        "question: Which one is wug?",
        "context: Alpha is dax.",
        "Beta is fep.",
    ]
    model = build_model(STATEMENTS, seed=0).eval()
    encoded = model.encoder.encode([text])
    nodes = statement_nodes([text], encoded.states, encoded.mask, encoded.offsets)
    assert nodes.mask.tolist() == [[True, True, True]]
    # "Beta is fep." covers the last four tokens (Beta, is, fep, .).
    torch.testing.assert_close(nodes.states[0, 2], encoded.states[0, -4:].mean(0))


@pytest.mark.parametrize("config", [STATEMENTS, LATENT])
def test_budget_zero_is_exactly_the_direct_scorer(config) -> None:
    model = perturbed(config)
    request = make(["Alpha", "Beta"], Budget())
    output = model([request])
    direct = super(type(model), model).forward([request]).logits
    torch.testing.assert_close(output.logits, direct)
    assert output.rounds == [0] and output.pairs == [0]


def test_fresh_workspace_starts_at_the_direct_scorer() -> None:
    model = build_model(STATEMENTS, seed=0).eval()
    direct = model([make(["Alpha", "Beta"], Budget())]).logits
    torch.testing.assert_close(model([make(["Alpha", "Beta"])]).logits, direct)


@pytest.mark.parametrize("config", [STATEMENTS, LATENT])
def test_rounds_are_permutation_equivariant(config) -> None:
    model = perturbed(config)
    names = ["Alpha", "Beta", "Gamma"]
    base = model([make(names)]).logits[0]
    order = [2, 0, 1]
    permuted = model([make([names[i] for i in order])]).logits[0]
    torch.testing.assert_close(permuted, base[order], rtol=1e-4, atol=1e-4)
    assert not torch.allclose(base, model([make(names, Budget())]).logits[0])


def test_caps_limit_rounds_with_honest_termination() -> None:
    model = perturbed(STATEMENTS)
    full = model([make(["Alpha", "Beta"])])
    per_round = full.pairs[0] // 3
    squeezed = model([make(["Alpha", "Beta"], ROOMY.model_copy(update={"max_pairs": per_round}))])
    assert squeezed.rounds == [1]
    assert squeezed.termination == [Termination.BUDGET_EXHAUSTED]
    few_nodes = model([make(["Alpha", "Beta"], ROOMY.model_copy(update={"max_nodes": 1}))])
    assert few_nodes.rounds == [0] and few_nodes.termination == [Termination.BUDGET_EXHAUSTED]
    capped = build_model(
        STATEMENTS.model_copy(
            update={"workspace": WorkspaceConfig(dim=32, hidden_dim=32, max_rounds=2)}
        ),
        seed=0,
    ).eval()
    out = capped([make(["Alpha", "Beta"])])
    assert out.rounds == [2] and out.termination == [Termination.POLICY_STOP]


def test_predictor_reports_usage_and_round_trace() -> None:
    model = perturbed(STATEMENTS)
    request = make(["Alpha", "Beta"]).model_copy(update={"trace": TraceMode.EVENTS})
    result = TorchPredictor(model, "ws@x", {"model": STATEMENTS.model_dump()}).decide(request)
    assert result.usage.rounds == 3 and result.usage.nodes == 4  # question + 3 sentences
    assert result.usage_within_budget() and result.termination is Termination.COMPLETED
    ops = [e.op for e in result.trace or ()]
    assert ops == [
        "render",
        "encode",
        "direct_score",
        "workspace",
        "aggregate",
        "aggregate",
        "aggregate",
        "calibrate",
    ]
    last_round = result.trace[6].details
    assert last_round["logits"] == pytest.approx(list(result.logits))
    assert any("Everything that is dax" in a for a in last_round["attended"])


def test_forced_rounds_return_every_round_for_deep_supervision() -> None:
    model = perturbed(STATEMENTS)
    out = model([make(["Alpha", "Beta"], Budget())], rounds=4)
    assert len(out.round_logits) == 4 and out.rounds == [4]
    torch.testing.assert_close(out.logits, out.round_logits[-1])


ISOLATED = ModelConfig(
    **BASE,
    workspace=WorkspaceConfig(dim=32, hidden_dim=32, encoding="isolated", readout="workspace"),
)


def test_isolated_encoding_requires_the_workspace_readout() -> None:
    with pytest.raises(ValueError, match="isolated"):
        WorkspaceConfig(encoding="isolated")


def test_isolated_encoder_never_sees_two_sentences() -> None:
    model = build_model(ISOLATED, seed=0).eval()
    seen: list[str] = []
    encode = model.encoder.encode
    model.encoder.encode = lambda texts: (seen.extend(texts), encode(texts))[1]
    model([make(["Alpha", "Beta"])])
    assert seen and all(len(sentence_spans(t)) <= 1 for t in seen)
    assert "Alpha" in seen and "Everything that is dax is also wug." in seen


def test_isolated_rounds_change_the_decision_and_stay_equivariant() -> None:
    model = build_model(ISOLATED, seed=0).eval()
    names = ["Alpha", "Beta", "Gamma"]
    zero = model([make(names, Budget())])
    assert zero.has_direct_path is False and zero.rounds == [0]
    base = model([make(names)]).logits[0]
    assert not torch.allclose(base, zero.logits[0])
    order = [2, 0, 1]
    permuted = model([make([names[i] for i in order])]).logits[0]
    torch.testing.assert_close(permuted, base[order], rtol=1e-4, atol=1e-4)


def test_trace_first_score_is_the_zero_round_decision() -> None:
    residual = perturbed(STATEMENTS)
    request = make(["Alpha", "Beta"]).model_copy(update={"trace": TraceMode.EVENTS})
    result = TorchPredictor(residual, "ws@x", {}).decide(request)
    direct = super(type(residual), residual).forward([request]).logits[0].tolist()
    assert result.trace[2].op == "direct_score"
    assert result.trace[2].details["logits"] == pytest.approx(direct, abs=1e-5)
    assert result.trace[2].details["logits"] != pytest.approx(list(result.logits), abs=1e-5)
    isolated = build_model(ISOLATED, seed=0).eval()
    result = TorchPredictor(isolated, "iso@x", {}).decide(request)
    assert result.trace[2].op == "initial_readout"
