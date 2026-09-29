"""EXPAND v2 invariants (decision 0012), offline: tiny memory, hashing encoder."""

import numpy as np
import pytest
import torch

from metanoia.api import TorchPredictor
from metanoia.build import build_model
from metanoia.config import BranchConfig, DirectScorerConfig, HashingEncoderConfig, ModelConfig
from metanoia.model import DirectDecisionModel
from metanoia.reasoning.branch import branch_context
from metanoia_contracts import Budget, Candidate, DecisionRequest, Termination, TraceMode
from tests.core.test_memory import write_memory

EXPAND = Budget(max_depth=1, max_queries=8, max_proposals=64, max_nodes=64)


def config(memory, nodes="random", max_width=4) -> ModelConfig:
    return ModelConfig(
        name="br",
        encoder=HashingEncoderConfig(buckets=256, dim=32, layers=1),
        scorer=DirectScorerConfig(latent_dim=32, hidden_dim=32, candidate_encoding="joint"),
        branch=BranchConfig(
            memory=str(memory), nodes=nodes, train_widths=(0, 2), max_width=max_width
        ),
    )


@pytest.fixture
def memory(tmp_path):
    write_memory(tmp_path / "mem", np.random.default_rng(0).normal(size=(12, 8)))
    return tmp_path / "mem"


def make(names: list[str], budget: Budget = EXPAND, context: str = "") -> DecisionRequest:
    return DecisionRequest(
        context=context,
        question="Which one conducts electricity?",
        candidates=tuple(Candidate(id=n.lower(), text=n) for n in names),
        budget=budget,
    )


def direct_twin(model) -> DirectDecisionModel:
    return DirectDecisionModel(model.encoder, model.scorer, "joint").eval()


def test_budget_zero_is_exactly_the_direct_model(memory) -> None:
    model = build_model(config(memory), seed=0).eval()
    request = make(["copper", "wood"], Budget(), context="Metals are shiny.")
    out = model([request])
    torch.testing.assert_close(out.logits, direct_twin(model)([request]).logits)
    assert out.width == [0] and out.nodes == [[[], []]]


def test_each_candidate_reads_only_its_own_branch(memory) -> None:
    model = build_model(config(memory), seed=0).eval()
    seen: list[str] = []
    encode = model.encoder.encode
    model.encoder.encode = lambda texts: (seen.extend(texts), encode(texts))[1]
    out = model([make(["copper", "wood"], context="Given text.")], width=2)
    memory_texts = [c.text for c in model.memory().concepts]
    candidate_texts = [t for t in seen if t.startswith("answer:")]
    assert len(candidate_texts) == 2
    for text, own in zip(candidate_texts, out.nodes[0], strict=True):
        read = {m for m in memory_texts if m in text}
        assert read == {memory_texts[c] for c in own}
        assert text.endswith("Given text.")  # nodes come before the given context
    assert branch_context(["a", "b"], "") == "[1] a [2] b"


def test_budget_sets_width_and_stays_equivariant(memory) -> None:
    model = build_model(config(memory), seed=0).eval()
    names = ["copper", "wood", "glass"]
    budget = EXPAND.model_copy(update={"max_proposals": 6, "max_nodes": 6})  # 2 per candidate
    out = model([make(names, budget)])
    assert out.width == [2] and all(len(c) == 2 for c in out.nodes[0])
    order = [2, 0, 1]
    permuted = model([make([names[i] for i in order], budget)]).logits[0]
    torch.testing.assert_close(permuted, out.logits[0][order], rtol=1e-4, atol=1e-4)


def test_width_is_capped_and_starved_budgets_are_reported(memory) -> None:
    model = build_model(config(memory, max_width=3), seed=0).eval()
    out = model([make(["copper", "wood"])])  # the budget allows 32 per candidate
    assert out.width == [3] and out.termination == [Termination.POLICY_STOP]
    starved = EXPAND.model_copy(update={"max_queries": 1})
    out = model([make(["copper", "wood"], starved)])
    assert out.width == [0] and out.termination == [Termination.BUDGET_EXHAUSTED]


def test_none_control_reads_empty_branches(memory) -> None:
    model = build_model(config(memory, nodes="none"), seed=0).eval()
    request = make(["copper", "wood"])
    out = model([request], width=2)
    assert out.width == [0]
    torch.testing.assert_close(out.logits, direct_twin(model)([request]).logits)


def test_trace_shows_nodes_and_their_contributions(memory) -> None:
    model = build_model(config(memory), seed=0).eval()
    budget = EXPAND.model_copy(update={"max_proposals": 4, "max_nodes": 4})  # 2 per candidate
    request = make(["copper", "wood"], budget).model_copy(update={"trace": TraceMode.EVENTS})
    with torch.no_grad():
        result = TorchPredictor(model, "br@x", {"model": {"branch": {}}}).decide(request)
    assert result.usage.queries == 2 and result.usage.nodes == 4 and result.usage.depth == 1
    ops = [e.op for e in result.trace]
    assert ops == ["render", "encode", "direct_score", "expand", "expand", "aggregate", "calibrate"]
    # The direct score is the empty-branch decision; evidence brings it to the final one.
    direct = direct_twin(model)([request]).logits[0].tolist()
    assert result.trace[2].details["logits"] == pytest.approx(direct, abs=1e-4)
    evidence = result.trace[5].details["evidence"]
    assert [d + e for d, e in zip(direct, evidence, strict=True)] == pytest.approx(
        list(result.logits), abs=1e-4
    )
    # Each node's relevance is its leave-one-out contribution: recompute one.
    with torch.no_grad():
        out = model([request], explain=True)
    assert result.trace[3].details["relevance"] == pytest.approx(
        [round(c, 4) for c in out.contributions[0][0]], abs=1e-4
    )


def test_branch_config_requires_joint_encoding(memory) -> None:
    with pytest.raises(ValueError, match="joint"):
        ModelConfig(
            name="br",
            encoder=HashingEncoderConfig(buckets=256, dim=32, layers=1),
            scorer=DirectScorerConfig(latent_dim=32, hidden_dim=32, candidate_encoding="separate"),
            branch=BranchConfig(memory=str(memory)),
        )


def test_memory_follows_the_model_device(memory) -> None:
    model = build_model(config(memory), seed=0)
    model.memory()  # loaded while the model is on the CPU
    meta = torch.device("meta")
    model.to(meta)
    assert model.memory().keys.device == meta


def test_focused_proposals_put_statements_about_the_candidate_first(memory) -> None:
    from metanoia.memory.store import Concept
    from metanoia.reasoning.branch import _stems

    cfg = config(memory, nodes="retrieved")
    branch = cfg.branch.model_copy(update={"proposals": "focused"})
    model = build_model(cfg.model_copy(update={"branch": branch}), seed=0).eval()
    texts = ["Heat moves by conduction.", "Copper wires conduct heat.", "Wood burns.", "Coppers."]
    concepts = [Concept(f"c{i}", "x", ("x",), "n", t) for i, t in enumerate(texts)]
    model.memory().concepts[:4] = concepts
    order = model._focus("What conducts heat best?", "copper", [0, 1, 2, 3])
    assert order == [1, 3, 0, 2]  # mentions of "copper" first, retrieval order kept
    assert _stems("Batteries and wires") == {"battery", "wire"}


def fused_model(memory, cache=None):
    from metanoia.config import RerankerConfig

    cfg = config(memory, nodes="retrieved")
    branch = cfg.branch.model_copy(
        update={
            "proposals": "fused",
            "reranker": RerankerConfig(model="stub", revision="r"),
            "proposal_cache": cache,
        }
    )
    model = build_model(cfg.model_copy(update={"branch": branch}), seed=0).eval()

    class Stub:  # prefers later pool positions, the reverse of retrieval order
        def score_pairs(self, pairs):
            return [float(int(t.split()[1].rstrip(":"))) for _, t in pairs]

    # Retrieval order 0, 1, ..., 11 with similarity 1, 0.9, ...
    model.propose = lambda pairs, k, source: (
        [list(range(min(k, 12))) for _ in pairs],
        [[1 - 0.1 * m for m in range(min(k, 12))] for _ in pairs],
    )
    model.reranker = lambda: Stub()
    return model


def test_fused_proposals_are_reciprocal_rank_fusion(memory) -> None:
    from metanoia.memory.rerank import reciprocal_rank_fusion

    model = fused_model(memory)
    nodes, sims = model.fused([("q?", "copper")], 4)
    expected = reciprocal_rank_fusion([list(range(12)), list(range(11, -1, -1))])[:4]
    assert nodes == [expected]
    assert sims == [[1 - 0.1 * m for m in expected]]
    assert model.expansion_identity()["proposals"]["method"] == "fused"


def test_proposal_cache_is_read_checked_and_equivalent(memory, tmp_path) -> None:
    from metanoia.memory.proposals import proposal_key, write_cache

    live = fused_model(memory)
    nodes, sims = live.fused([("q?", "copper"), ("q?", "wood")], 8)
    rows = {proposal_key("q?", "copper"): (nodes[0], sims[0])}  # "wood" left out
    (tmp_path / "cache").mkdir()
    write_cache(tmp_path / "cache", live.proposal_identity(), 8, rows, {})
    cached = fused_model(memory, cache=str(tmp_path / "cache"))
    cached.reranker = lambda: pytest.fail("a cached pair must not be reranked")
    assert cached.fused([("q?", "copper")], 8) == ([nodes[0]], [pytest.approx(sims[0])])
    # A miss is ranked live, identically.
    cached.reranker = live.reranker
    assert cached.fused([("q?", "wood")], 8)[0] == [nodes[1]]
    assert cached.expansion_identity()["proposal_cache"] == cached.proposal_cache().cache_id()
    # A cache built for another identity is refused.
    (tmp_path / "other").mkdir()
    write_cache(tmp_path / "other", {**live.proposal_identity(), "pool": 5}, 8, rows, {})
    with pytest.raises(ValueError, match="was built for"):
        fused_model(memory, cache=str(tmp_path / "other")).proposal_cache()
