"""The Stage-3a reasoning loop: a budgeted workspace on top of the direct scorer.

Decision 0008. The engine owns the workspace state, the round loop and resource
accounting; the block (``reasoning.aggregate``) only computes updates. Budget zero, or
a request with ``max_rounds = 0``, returns exactly the direct scorer's logits.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from metanoia.config import WorkspaceConfig
from metanoia.decision import CandidateScorer
from metanoia.interfaces import TextEncoder
from metanoia.model import DecisionOutput, DirectDecisionModel
from metanoia.reasoning.aggregate import WorkspaceBlock
from metanoia.reasoning.workspace import statement_nodes
from metanoia_contracts import DecisionRequest, Termination

# Node types: the question line, other sentences, candidates, latent slots.
_QUESTION, _STATEMENT, _CANDIDATE, _LATENT = range(4)


@dataclass(frozen=True)
class WorkspaceOutput(DecisionOutput):
    """``logits`` are each example's logits after its own executed rounds."""

    mode: str  # "statements" or "latent"
    round_logits: list[torch.Tensor]  # logits after rounds 1..R_run, [B, N] each
    rounds: list[int]  # rounds executed per example
    nodes: list[int]  # statement (or latent) nodes used per example
    pairs: list[int]  # attention pairs scored per example, all rounds
    termination: list[Termination]
    statements: list[list[str]]  # sentence texts per example
    attention: list[torch.Tensor]  # per round: candidate -> sentence weights [B, N, S]


class WorkspaceModules(nn.Module):
    """The workspace's learned parts (parameter names ``workspace.*``)."""

    def __init__(self, input_dim: int, config: WorkspaceConfig) -> None:
        super().__init__()
        d = config.dim
        self.statement_proj = nn.Linear(input_dim, d)
        self.candidate_proj = nn.Linear(input_dim, d)
        self.types = nn.Embedding(4, d)
        self.block = WorkspaceBlock(
            d, config.heads, config.hidden_dim, config.dropout, cross=config.mode == "latent"
        )
        self.readout = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.out = nn.Linear(d, 1)
        # Zero-initialized output: a warm-started model starts identical to the direct
        # scorer at every budget, and learns only what the rounds add.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.slots = (
            nn.Parameter(torch.randn(config.latent_slots, d) * 0.02)
            if config.mode == "latent"
            else None
        )

    def head(self, candidates: torch.Tensor) -> torch.Tensor:
        return self.out(self.readout(candidates))

    def latent_slots(self, batch: int) -> torch.Tensor:
        if self.slots is None:
            raise ValueError("latent slots exist only in latent mode")
        return self.slots.expand(batch, -1, -1)


class WorkspaceDecisionModel(DirectDecisionModel):
    def __init__(
        self,
        encoder: TextEncoder,
        scorer: CandidateScorer,
        candidate_encoding: Literal["separate", "joint"],
        config: WorkspaceConfig,
    ) -> None:
        super().__init__(encoder, scorer, candidate_encoding)
        self.workspace_config = config
        self.workspace = WorkspaceModules(encoder.output_dim, config)

    def forward(  # type: ignore[override]
        self, requests: Sequence[DecisionRequest], rounds: int | None = None
    ) -> WorkspaceOutput:
        """``rounds`` (training) runs that many rounds on every example and ignores the
        budgets; ``None`` (inference) runs what each request's budget allows."""
        inputs = self.encode_inputs(requests)
        direct = self.scorer(inputs.context, inputs.candidates, inputs.candidate_mask)
        if inputs.context.offsets is None:
            raise ValueError("the workspace needs token offsets from the encoder")
        sentences = statement_nodes(
            inputs.rendered, inputs.context.states, inputs.context.mask, inputs.context.offsets
        )
        cfg, ws = self.workspace_config, self.workspace
        batch, n_statements = sentences.mask.shape
        types = torch.full_like(sentences.mask, _STATEMENT, dtype=torch.long)
        types[:, 0] = _QUESTION  # the rendered input starts with the question line
        statements = ws.statement_proj(sentences.states) + ws.types(types)
        candidates = ws.candidate_proj(inputs.candidates) + ws.types.weight[_CANDIDATE]
        memory = memory_mask = None
        if cfg.mode == "statements":
            first = statements
            first_mask = sentences.mask
        else:
            first = ws.latent_slots(batch) + ws.types.weight[_LATENT]
            first_mask = torch.ones(batch, cfg.latent_slots, dtype=torch.bool, device=direct.device)
            memory, memory_mask = statements, sentences.mask
        nodes = torch.cat([first, candidates], dim=1)
        mask = torch.cat([first_mask, inputs.candidate_mask], dim=1)
        offset = first.shape[1]

        plan = [
            self._allowed(
                r,
                int(first_mask[i].sum()),
                int(sentences.mask[i].sum()),
                int(inputs.candidate_mask[i].sum()),
                rounds,
            )
            for i, r in enumerate(requests)
        ]
        run = max(p[0] for p in plan)
        round_logits, attention = [], []
        for _ in range(run):
            nodes, weights = ws.block(nodes, mask, memory, memory_mask)
            delta = ws.head(nodes[:, offset:]).squeeze(-1)
            round_logits.append(direct + delta.masked_fill(~inputs.candidate_mask, 0.0))
            columns = weights[:, offset:, :n_statements] if memory is None else weights[:, offset:]
            attention.append(columns.detach())
        final = torch.stack(
            [direct[i] if k == 0 else round_logits[k - 1][i] for i, (k, *_) in enumerate(plan)]
        )
        return WorkspaceOutput(
            mode=cfg.mode,
            logits=final,
            candidate_mask=inputs.candidate_mask,
            round_logits=round_logits,
            rounds=[p[0] for p in plan],
            nodes=[p[1] for p in plan],
            pairs=[p[2] for p in plan],
            termination=[p[3] for p in plan],
            statements=sentences.texts,
            attention=attention,
        )

    def _allowed(
        self,
        request: DecisionRequest,
        first_nodes: int,
        statements: int,
        candidates: int,
        forced: int | None,
    ) -> tuple[int, int, int, Termination]:
        """(rounds, nodes, pairs, termination) for one example. A round runs only if it
        fits every relevant cap; candidate capacity is reserved outside ``max_nodes``."""
        cfg = self.workspace_config
        total = first_nodes + candidates
        per_round = total * total + (total * statements if cfg.mode == "latent" else 0)
        if forced is not None:
            return forced, first_nodes, forced * per_round, Termination.COMPLETED
        budget = request.budget
        wanted = budget.max_rounds
        if wanted == 0:
            return 0, 0, 0, Termination.COMPLETED
        if first_nodes > budget.max_nodes:
            return 0, 0, 0, Termination.BUDGET_EXHAUSTED
        affordable = min(wanted, budget.max_pairs // per_round)
        rounds = min(affordable, cfg.max_rounds)
        if affordable < wanted:
            termination = Termination.BUDGET_EXHAUSTED
        elif rounds < wanted:
            termination = Termination.POLICY_STOP
        else:
            termination = Termination.COMPLETED
        nodes = first_nodes if rounds else 0
        return rounds, nodes, rounds * per_round, termination
