"""EXPAND v1: each candidate expands into concepts; evidence aggregates back (decision 0011).

The direct scorer gives the depth-zero decision. For each candidate, the frozen retrieval
encoder turns "question + candidate" into a query, and the concept memory proposes k
concepts. The decision model's reader encodes each concept together with the rendered
input, a relevance score weights it, and every candidate attends over the concepts of
*all* branches to collect evidence that adjusts its score. Budget zero is exactly the
direct scorer.
"""

import contextlib
import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from metanoia.config import ExpansionConfig
from metanoia.decision import CandidateScorer
from metanoia.interfaces import TextEncoder
from metanoia.memory.encoder import RetrievalEncoder
from metanoia.memory.store import ConceptMemory
from metanoia.model import DecisionOutput, DirectDecisionModel
from metanoia_contracts import DecisionRequest, Termination

# Hard cap on concepts per candidate, whatever the budget allows (memory safety).
MAX_PER_CANDIDATE = 64
# Concept texts encoded per encoder call.
ENCODE_CHUNK = 64


@dataclass(frozen=True)
class ExpansionOutput(DecisionOutput):
    initial_logits: torch.Tensor  # the direct scorer's logits [B, N]
    expanded: list[int]  # concepts per candidate actually used (0 = no expansion)
    concepts: list[list[list[int]]]  # per example, per candidate: memory indices
    relevance: torch.Tensor | None  # [B, N, K] relevance scores u (None if not expanded)
    attention: torch.Tensor | None  # [B, N, N*K] candidate -> concept weights
    evidence: torch.Tensor | None  # [B, N] the added evidence (s - s0)
    termination: list[Termination]


class ExpansionModules(nn.Module):
    """The trainable expansion parts (parameter names ``expansion.*``)."""

    def __init__(self, input_dim: int, config: ExpansionConfig) -> None:
        super().__init__()
        d = config.dim
        self.concept_proj = nn.Linear(input_dim, d)
        self.candidate_proj = nn.Linear(input_dim, d)
        self.context_proj = nn.Linear(input_dim, d)
        self.relevance = nn.Sequential(
            nn.Linear(4 * d, config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, 1)
        )
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.same_branch = nn.Parameter(torch.zeros(()))
        self.out = nn.Sequential(
            nn.Linear(3 * d, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )


class ExpansionDecisionModel(DirectDecisionModel):
    def __init__(
        self,
        encoder: TextEncoder,
        scorer: CandidateScorer,
        candidate_encoding: Literal["separate", "joint"],
        config: ExpansionConfig,
    ) -> None:
        super().__init__(encoder, scorer, candidate_encoding)
        self.expansion_config = config
        self.expansion = ExpansionModules(encoder.output_dim, config)
        if config.freeze_direct:
            self.encoder.requires_grad_(False)
            self.scorer.requires_grad_(False)
        # The memory and its retrieval encoder are frozen external resources, identified
        # in the bundle, not stored in it (like pretrained encoder weights). They are kept
        # outside nn.Module registration so they never enter the state dict.
        object.__setattr__(self, "_memory", None)
        object.__setattr__(self, "_retriever", None)

    def train(self, mode: bool = True) -> "ExpansionDecisionModel":
        super().train(mode)
        if self.expansion_config.freeze_direct:
            self.encoder.eval()  # frozen parts run deterministically (no dropout)
            self.scorer.eval()
        return self

    def memory(self) -> ConceptMemory:
        memory: ConceptMemory | None = self.__dict__.get("_memory")
        if memory is None:
            memory = ConceptMemory(Path(self.expansion_config.memory), device=str(self.device))
            object.__setattr__(self, "_memory", memory)
        return memory

    def retriever(self) -> RetrievalEncoder:
        existing: RetrievalEncoder | None = self.__dict__.get("_retriever")
        if existing is None:
            spec = self.memory().manifest.encoder
            retriever = RetrievalEncoder(
                spec["model"],
                spec["requested_revision"],
                spec["pooling"],
                spec["max_length"],
                spec["query_prefix"],
            ).to(self.device)
            if retriever.identity() != spec:
                raise ValueError("retrieval encoder differs from the memory's encoder")
            object.__setattr__(self, "_retriever", retriever)
            existing = retriever
        return existing

    def expansion_identity(self) -> dict[str, object]:
        manifest = self.memory().manifest
        return {
            "memory_id": manifest.memory_id,
            "concepts_sha256": manifest.concepts.sha256,
            "vectors_sha256": manifest.vectors.sha256,
            "retrieval_encoder": manifest.encoder,
        }

    def forward(  # type: ignore[override]
        self, requests: Sequence[DecisionRequest], force: bool = False
    ) -> ExpansionOutput:
        """``force`` (training, dev evaluation): expand every example with the configured
        number of concepts; otherwise each request's budget decides."""
        cfg = self.expansion_config
        # A frozen direct path needs no activations stored: run it without autograd.
        frozen = torch.no_grad() if cfg.freeze_direct else contextlib.nullcontext()
        with frozen:
            inputs = self.encode_inputs(requests)
            direct = self.scorer(inputs.context, inputs.candidates, inputs.candidate_mask)
        counts = [len(r.candidates) for r in requests]
        plan = [self._allowed(r, n, force) for r, n in zip(requests, counts, strict=True)]
        if any(p[0] for p in plan):
            # A memory smaller than the requested width caps it; usage reports what ran.
            size = len(self.memory())
            plan = [(min(k_b, size), term) for k_b, term in plan]
        k = max(p[0] for p in plan)
        if k == 0:
            return ExpansionOutput(
                logits=direct,
                candidate_mask=inputs.candidate_mask,
                initial_logits=direct,
                expanded=[0] * len(requests),
                concepts=[[[] for _ in range(n)] for n in counts],
                relevance=None,
                attention=None,
                evidence=None,
                termination=[p[1] for p in plan],
            )
        chosen = self._propose(requests, k) if cfg.concepts != "none" else None
        ex = self.expansion
        batch, n_max = inputs.candidate_mask.shape
        memory = self.memory()
        if chosen is None:  # the "none" control: same head, no concept content
            z = direct.new_zeros(batch, n_max, k, ex.concept_proj.in_features)
        else:
            texts = [
                f"concept: {memory.concepts[c].text}\n{rendered}"
                for request, rendered, per_candidate in zip(
                    requests, inputs.rendered, chosen, strict=True
                )
                for i in range(n_max)
                for c in (per_candidate[i] if i < len(request.candidates) else per_candidate[0])
            ]
            with frozen:
                pooled = [
                    self.encoder.encode(texts[i : i + ENCODE_CHUNK]).pooled
                    for i in range(0, len(texts), ENCODE_CHUNK)
                ]
            z = torch.cat(pooled).view(batch, n_max, k, -1)
        concepts = ex.concept_proj(z.float())  # [B, N, K, d]
        candidates = ex.candidate_proj(inputs.candidates.float())  # [B, N, d]
        context = ex.context_proj(inputs.context.pooled.float())  # [B, d]
        d = concepts.shape[-1]
        relevance = ex.relevance(
            torch.cat(
                [
                    concepts,
                    candidates[:, :, None].expand_as(concepts),
                    context[:, None, None].expand_as(concepts),
                    concepts * candidates[:, :, None],
                ],
                dim=-1,
            )
        ).squeeze(-1)  # [B, N, K]
        # Each example uses only its own k' concepts per candidate, and only real candidates.
        valid = torch.zeros(batch, n_max, k, dtype=torch.bool, device=direct.device)
        for b, (k_b, _) in enumerate(plan):
            valid[b, : counts[b], :k_b] = True
        flat_valid = valid.view(batch, n_max * k)
        nodes = concepts.view(batch, n_max * k, d)
        scores = ex.query(candidates) @ ex.key(nodes).transpose(1, 2) / math.sqrt(d)
        branch = torch.arange(n_max, device=direct.device).repeat_interleave(k)
        same = (torch.arange(n_max, device=direct.device)[:, None] == branch[None, :]).float()
        scores = scores + relevance.view(batch, 1, n_max * k) + ex.same_branch * same
        scores = scores.masked_fill(~flat_valid[:, None, :], float("-inf"))
        attention = torch.softmax(scores, dim=-1)  # [B, N, N*K]
        attention = torch.nan_to_num(attention)  # examples without expansion
        read = attention @ ex.value(nodes)  # [B, N, d]
        evidence = ex.out(torch.cat([candidates, read, candidates * read], dim=-1)).squeeze(-1)
        expanded_rows = torch.tensor([p[0] > 0 for p in plan], device=direct.device)
        evidence = evidence.masked_fill(~inputs.candidate_mask | ~expanded_rows[:, None], 0.0)
        logits = direct + evidence
        return ExpansionOutput(
            logits=logits,
            candidate_mask=inputs.candidate_mask,
            initial_logits=direct,
            expanded=[p[0] for p in plan],
            concepts=[
                [per_candidate[i][: plan[b][0]] for i in range(counts[b])]
                for b, per_candidate in enumerate(chosen)
            ]
            if chosen is not None
            else [[[] for _ in range(n)] for n in counts],
            relevance=relevance.masked_fill(~valid, float("-inf")).detach(),
            attention=attention.detach(),
            evidence=evidence.detach(),
            termination=[p[1] for p in plan],
        )

    def _propose(self, requests: Sequence[DecisionRequest], k: int) -> list[list[list[int]]]:
        """Memory indices [example][candidate][k]: retrieved, or random for the control."""
        cfg = self.expansion_config
        memory = self.memory()
        n_max = max(len(r.candidates) for r in requests)
        queries = [
            (r.question, (r.candidates[i] if i < len(r.candidates) else r.candidates[0]).text)
            for r in requests
            for i in range(n_max)
        ]
        if cfg.concepts == "retrieved":
            vectors = self.retriever().embed([f"{q} {c}" for q, c in queries], query=True)
            flat = memory.search(vectors, k)[0].tolist()
        else:
            # Deterministic per (question, candidate), so collection runs are reproducible.
            flat = []
            for q, c in queries:
                seed = int.from_bytes(hashlib.sha256(f"{q}\x1f{c}".encode()).digest()[:8], "big")
                flat.append(random.Random(seed).sample(range(len(memory)), k))
        return [flat[b * n_max : (b + 1) * n_max] for b in range(len(requests))]

    def _allowed(self, request: DecisionRequest, n: int, force: bool) -> tuple[int, Termination]:
        """(concepts per candidate, termination). Expansion needs depth, one query per
        candidate, and at least one proposal per candidate; it uses what the budget
        allows, up to the configured number."""
        cfg = self.expansion_config
        if force:
            return cfg.proposals, Termination.COMPLETED
        budget = request.budget
        if budget.max_depth == 0 and budget.max_proposals == 0:
            return 0, Termination.COMPLETED  # a direct request
        per_candidate = min(budget.max_proposals, budget.max_nodes) // n
        if budget.max_depth < 1 or budget.max_queries < n or per_candidate < 1:
            return 0, Termination.BUDGET_EXHAUSTED
        if per_candidate > MAX_PER_CANDIDATE:
            return MAX_PER_CANDIDATE, Termination.POLICY_STOP
        return per_candidate, Termination.COMPLETED
