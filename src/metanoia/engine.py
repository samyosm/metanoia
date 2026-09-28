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
from metanoia.encoding.inputs import render_input
from metanoia.interfaces import TextEncoder
from metanoia.model import DecisionOutput, DirectDecisionModel
from metanoia.reasoning.aggregate import WorkspaceBlock
from metanoia.reasoning.workspace import StatementNodes, sentence_spans, statement_nodes
from metanoia_contracts import DecisionRequest, Termination

# Node types: the question line, other sentences, candidates, latent slots.
_QUESTION, _STATEMENT, _CANDIDATE, _LATENT = range(4)


@dataclass(frozen=True)
class WorkspaceOutput(DecisionOutput):
    """``logits`` are each example's logits after its own executed rounds."""

    mode: str  # "statements" or "latent"
    initial_logits: torch.Tensor  # the zero-round decision [B, N]
    has_direct_path: bool  # True when initial_logits are the direct scorer's
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
        if config.readout == "residual":
            # Zero-initialized output: a warm-started model starts identical to the
            # direct scorer at every budget, and learns only what the rounds add.
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
        cfg, ws = self.workspace_config, self.workspace
        direct: torch.Tensor | None = None
        if cfg.encoding == "context":
            inputs = self.encode_inputs(requests)
            if inputs.context.offsets is None:
                raise ValueError("the workspace needs token offsets from the encoder")
            sentences = statement_nodes(
                inputs.rendered, inputs.context.states, inputs.context.mask, inputs.context.offsets
            )
            candidate_states, candidate_mask = inputs.candidates, inputs.candidate_mask
            if cfg.readout == "residual":
                direct = self.scorer(inputs.context, inputs.candidates, inputs.candidate_mask)
        else:
            sentences, candidate_states, candidate_mask = self._encode_isolated(requests)
        batch, n_statements = sentences.mask.shape
        types = torch.full_like(sentences.mask, _STATEMENT, dtype=torch.long)
        types[:, 0] = _QUESTION  # the rendered input starts with the question line
        statements = ws.statement_proj(sentences.states) + ws.types(types)
        candidates = ws.candidate_proj(candidate_states) + ws.types.weight[_CANDIDATE]
        memory = memory_mask = None
        if cfg.mode == "statements":
            first = statements
            first_mask = sentences.mask
        else:
            first = ws.latent_slots(batch) + ws.types.weight[_LATENT]
            first_mask = torch.ones(
                batch, cfg.latent_slots, dtype=torch.bool, device=candidate_mask.device
            )
            memory, memory_mask = statements, sentences.mask
        nodes = torch.cat([first, candidates], dim=1)
        mask = torch.cat([first_mask, candidate_mask], dim=1)
        offset = first.shape[1]

        plan = [
            self._allowed(
                r,
                int(first_mask[i].sum()),
                int(sentences.mask[i].sum()),
                int(candidate_mask[i].sum()),
                rounds,
            )
            for i, r in enumerate(requests)
        ]

        def readout(states: torch.Tensor) -> torch.Tensor:
            scores = ws.head(states[:, offset:]).squeeze(-1)
            if direct is not None:
                return direct + scores.masked_fill(~candidate_mask, 0.0)
            return scores.masked_fill(~candidate_mask, float("-inf"))

        initial = direct if direct is not None else readout(nodes)
        run = max(p[0] for p in plan)
        round_logits, attention = [], []
        for _ in range(run):
            nodes, weights = ws.block(nodes, mask, memory, memory_mask)
            round_logits.append(readout(nodes))
            columns = weights[:, offset:, :n_statements] if memory is None else weights[:, offset:]
            attention.append(columns.detach())
        final = torch.stack(
            [initial[i] if k == 0 else round_logits[k - 1][i] for i, (k, *_) in enumerate(plan)]
        )
        return WorkspaceOutput(
            mode=cfg.mode,
            initial_logits=initial,
            has_direct_path=direct is not None,
            logits=final,
            candidate_mask=candidate_mask,
            round_logits=round_logits,
            rounds=[p[0] for p in plan],
            nodes=[p[1] for p in plan],
            pairs=[p[2] for p in plan],
            termination=[p[3] for p in plan],
            statements=sentences.texts,
            attention=attention,
        )

    def _encode_isolated(
        self, requests: Sequence[DecisionRequest]
    ) -> tuple[StatementNodes, torch.Tensor, torch.Tensor]:
        """Encode every sentence and every candidate on its own (decision 0009): no
        sequence the encoder reads contains two premises."""
        rendered = [render_input(r) for r in requests]
        texts = [[t[a:b] for a, b in sentence_spans(t)] for t in rendered]
        pooled = self.encoder.encode([s for example in texts for s in example]).pooled
        width = max(len(example) for example in texts)
        states = pooled.new_zeros(len(requests), width, pooled.shape[-1])
        mask = torch.zeros(len(requests), width, dtype=torch.bool, device=pooled.device)
        start = 0
        for i, example in enumerate(texts):
            states[i, : len(example)] = pooled[start : start + len(example)]
            mask[i, : len(example)] = True
            start += len(example)
        counts = [len(r.candidates) for r in requests]
        flat = self.encoder.encode([c.text for r in requests for c in r.candidates]).pooled
        candidates = flat.new_zeros(len(requests), max(counts), flat.shape[-1])
        candidate_mask = torch.zeros(
            len(requests), max(counts), dtype=torch.bool, device=flat.device
        )
        start = 0
        for i, count in enumerate(counts):
            candidates[i, :count] = flat[start : start + count]
            candidate_mask[i, :count] = True
            start += count
        return StatementNodes(states=states, mask=mask, texts=texts), candidates, candidate_mask

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
