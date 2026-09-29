"""EXPAND v2: the reader reads each candidate's branch (decision 0012).

Each candidate retrieves k nodes (statements about concepts) from the memory with the
query "question candidate". The nodes are rendered into that candidate's own joint
input, ahead of any given context:

    answer: {candidate}
    question: {question}
    context: [1] {node} [2] {node} … {context}

The trainable reader encodes it, and the candidate-conditioned scorer compares the
candidates, so evidence in one branch also moves the others' probabilities. Budget zero
gives empty branches: exactly the direct model's input.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from metanoia.config import BranchConfig
from metanoia.decision import CandidateScorer
from metanoia.encoding.inputs import render_candidate_joint, render_input
from metanoia.interfaces import EncodedText, TextEncoder
from metanoia.memory.access import MemoryAccess
from metanoia.memory.proposals import ProposalCache
from metanoia.memory.rerank import Reranker, reciprocal_rank_fusion, reranker_query
from metanoia.model import DecisionOutput, DirectDecisionModel
from metanoia_contracts import DecisionRequest, Termination


@dataclass(frozen=True)
class BranchOutput(DecisionOutput):
    width: list[int]  # nodes per candidate actually read (0 = no expansion)
    nodes: list[list[list[int]]]  # per example, per candidate: memory indices
    similarity: list[list[list[float]]]  # retrieval similarity of each node
    termination: list[Termination]
    # Only with ``explain``: the empty-branch (direct) logits [B, N], and each node's
    # contribution to its own candidate's logit (full branch minus the branch without it).
    initial_logits: torch.Tensor | None = None
    contributions: list[list[list[float]]] | None = None
    # Only with ``keep``: what leave-one-out needs, to explain candidates one at a time
    # afterwards (see ``BranchDecisionModel.explain_candidate``).
    encoded: Any = None


@dataclass(frozen=True)
class _Encoded:
    requests: Sequence[DecisionRequest]
    context: EncodedText
    candidates: torch.Tensor
    mask: torch.Tensor
    statements: Any  # (b, i, drop) -> the statements candidate i of example b reads


def knowledge_text(statements: Sequence[str]) -> str:
    return " ".join(f"[{m + 1}] {text}" for m, text in enumerate(statements))


def branch_context(statements: Sequence[str], context: str) -> str:
    """The context a candidate reads: its nodes first, so truncation of a long context
    never removes them."""
    knowledge = knowledge_text(statements)
    if not knowledge:
        return context
    return f"{knowledge}\n{context}" if context else knowledge


class BranchDecisionModel(MemoryAccess, DirectDecisionModel):
    def __init__(
        self,
        encoder: TextEncoder,
        scorer: CandidateScorer,
        candidate_encoding: Literal["separate", "joint"],
        config: BranchConfig,
    ) -> None:
        if candidate_encoding != "joint":
            raise ValueError("branch reading needs joint candidate encoding")
        super().__init__(encoder, scorer, candidate_encoding)
        self.branch_config = config
        self._init_memory(config.memory)
        # Frozen external judges and caches, kept out of the state dict (like the memory).
        object.__setattr__(self, "_reranker", None)
        object.__setattr__(self, "_proposal_cache", None)

    def proposal_identity(self) -> dict[str, Any]:
        """Everything a fused ordering depends on (what a proposal cache must match)."""
        cfg = self.branch_config
        assert cfg.reranker is not None
        manifest = self.memory().manifest
        return {
            "method": "fused",
            "pool": FUSED_POOL,
            "rrf_k": RRF_K,
            "memory_id": manifest.memory_id,
            "vectors_sha256": manifest.vectors.sha256,
            "retrieval_encoder": manifest.encoder,
            "reranker": {
                "model": cfg.reranker.model,
                "requested_revision": cfg.reranker.revision,
                "max_length": cfg.reranker.max_length,
                "query_format": "qa",
            },
        }

    def expansion_identity(self) -> dict[str, object]:
        identity = super().expansion_identity()
        if self.branch_config.proposals == "fused":
            cache = self.proposal_cache()
            identity["proposals"] = self.proposal_identity()
            identity["proposal_cache"] = cache.cache_id() if cache is not None else None
        return identity

    def reranker(self) -> Reranker:
        existing: Reranker | None = self.__dict__.get("_reranker")
        cfg = self.branch_config.reranker
        assert cfg is not None
        if existing is None:
            existing = Reranker(cfg.model, cfg.revision, cfg.max_length).to(self.device)
            object.__setattr__(self, "_reranker", existing)
        elif next(existing.parameters()).device != self.device:
            existing.to(self.device)
        return existing

    def proposal_cache(self) -> ProposalCache | None:
        path = self.branch_config.proposal_cache
        if path is None:
            return None
        cache: ProposalCache | None = self.__dict__.get("_proposal_cache")
        if cache is None:
            cache = ProposalCache(Path(path), self.proposal_identity())
            object.__setattr__(self, "_proposal_cache", cache)
        return cache

    def fused(
        self, pairs: Sequence[tuple[str, str]], k: int
    ) -> tuple[list[list[int]], list[list[float]]]:
        """The first k nodes of the fused ordering for each (question, candidate): the top
        ``FUSED_POOL`` by retrieval similarity, reordered by reciprocal-rank fusion of
        similarity and reranker scores. Cached pairs are read; others are ranked live."""
        cache = self.proposal_cache()
        found: dict[int, tuple[list[int], list[float]]] = {}
        if cache is not None and k <= cache.width:
            for i, (q, c) in enumerate(pairs):
                hit = cache.get(q, c)
                if hit is not None:
                    found[i] = hit
        missing = [i for i in range(len(pairs)) if i not in found]
        if missing:
            pools, sims = self.propose([pairs[i] for i in missing], FUSED_POOL, "retrieved")
            memory = self.memory()
            judged = [
                (reranker_query(*pairs[i]), memory.concepts[n].text)
                for i, pool in zip(missing, pools, strict=True)
                for n in pool
            ]
            flat = self.reranker().score_pairs(judged)
            start = 0
            for i, pool, pool_sims in zip(missing, pools, sims, strict=True):
                scores = flat[start : start + len(pool)]
                start += len(pool)
                by_reranker = sorted(range(len(pool)), key=lambda m: -scores[m])
                order = reciprocal_rank_fusion([list(range(len(pool))), by_reranker], RRF_K)
                found[i] = ([pool[m] for m in order], [pool_sims[m] for m in order])
        return [found[i][0][:k] for i in range(len(pairs))], [
            found[i][1][:k] for i in range(len(pairs))
        ]

    def forward(  # type: ignore[override]
        self,
        requests: Sequence[DecisionRequest],
        width: int | None = None,
        explain: bool = False,
        proposals: tuple[list[list[list[int]]], list[list[list[float]]]] | None = None,
        keep: bool = False,
    ) -> BranchOutput:
        """``width`` (training, dev evaluation): read that many nodes per candidate for
        every example; otherwise each request's budget decides. ``explain`` adds the
        empty-branch logits and per-node leave-one-out contributions (for traces).
        ``proposals`` (nodes, similarities per example and candidate) reuses nodes
        proposed earlier, e.g. by :meth:`propose_nodes` in a streamed decision; each
        branch reads its first ``width`` of them."""
        if not requests:
            raise ValueError("empty batch")
        counts = [len(r.candidates) for r in requests]
        plan = [
            (width, Termination.COMPLETED) if width is not None else self.width_for(r)
            for r in requests
        ]
        if self.branch_config.nodes == "none":
            plan = [(0, term) for _, term in plan]
        if any(k for k, _ in plan):
            size = len(self.memory())
            plan = [(min(k, size), term) for k, term in plan]
        if proposals is None:
            nodes, sims = self._nodes(requests, [k for k, _ in plan])
        else:
            nodes = [[b[:k] for b in per] for per, (k, _) in zip(proposals[0], plan, strict=True)]
            sims = [[b[:k] for b in per] for per, (k, _) in zip(proposals[1], plan, strict=True)]
        memory = self.memory() if any(k for k, _ in plan) else None

        def statements(b: int, i: int, drop: int | None = None) -> list[str]:
            if memory is None:
                return []
            return [memory.concepts[c].text for m, c in enumerate(nodes[b][i]) if m != drop]

        rendered = [render_input(r) for r in requests]
        context = self.encoder.encode(rendered)
        texts = [
            self._candidate_text(r, i, statements(b, i))
            for b, r in enumerate(requests)
            for i in range(len(r.candidates))
        ]
        candidates, mask = self._pack(self.encoder.encode(texts).pooled, counts)
        logits = self.scorer(context, candidates, mask)
        encoded = _Encoded(requests, context, candidates, mask, statements)
        initial, contributions = None, None
        if explain:
            initial = self._initial(encoded)
            contributions = [
                [self._leave_one_out(encoded, logits, b, i, len(nodes[b][i])) for i in range(n)]
                for b, n in enumerate(counts)
            ]
        return BranchOutput(
            logits=logits,
            candidate_mask=mask,
            width=[k for k, _ in plan],
            nodes=nodes,
            similarity=sims,
            termination=[t for _, t in plan],
            initial_logits=initial,
            contributions=contributions,
            encoded=encoded if keep else None,
        )

    def width_for(self, request: DecisionRequest) -> tuple[int, Termination]:
        """Nodes per candidate the request's budget allows (0 for the "none" control)."""
        k, termination = self._allowed(request, len(request.candidates))
        if self.branch_config.nodes == "none":
            return 0, termination
        if k:
            k = min(k, len(self.memory()))
        return k, termination

    def propose_nodes(
        self, request: DecisionRequest, k: int
    ) -> tuple[list[list[list[int]]], list[list[list[float]]]]:
        """The first k proposals of each candidate, in the shape ``forward`` reuses."""
        return self._nodes([request], [k])

    def _candidate_text(self, request: DecisionRequest, i: int, statements: list[str]) -> str:
        own = request.model_copy(update={"context": branch_context(statements, request.context)})
        return render_candidate_joint(request.candidates[i].text, render_input(own))

    def _pack(self, flat: torch.Tensor, counts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = flat.new_zeros(len(counts), max(counts), flat.shape[-1])
        mask = torch.zeros(len(counts), max(counts), dtype=torch.bool, device=flat.device)
        start = 0
        for b, n in enumerate(counts):
            candidates[b, :n] = flat[start : start + n]
            mask[b, :n] = True
            start += n
        return candidates, mask

    def _nodes(
        self, requests: Sequence[DecisionRequest], widths: list[int]
    ) -> tuple[list[list[list[int]]], list[list[list[float]]]]:
        """Each candidate's first k_b proposals (retrieved or random). With focused
        proposals, a pool of ``FOCUS_POOL`` retrieved statements is reordered so those
        mentioning the candidate come first (decision 0013)."""
        kmax = max(widths)
        if kmax == 0:
            return [[[] for _ in r.candidates] for r in requests], [
                [[] for _ in r.candidates] for r in requests
            ]
        cfg = self.branch_config
        pairs = [(r.question, c.text) for r in requests for c in r.candidates]
        source: Literal["retrieved", "random"] = (
            "retrieved" if cfg.nodes == "retrieved" else "random"
        )
        focused = source == "retrieved" and cfg.proposals == "focused"
        if source == "retrieved" and cfg.proposals == "fused":
            flat, flat_sims = self.fused(pairs, kmax)
        else:
            pool = max(kmax, FOCUS_POOL) if focused else kmax
            flat, flat_sims = self.propose(pairs, pool, source)
        nodes, sims, row = [], [], 0
        for r, k in zip(requests, widths, strict=True):
            n = len(r.candidates)
            per_nodes, per_sims = [], []
            for i in range(n):
                order = list(range(len(flat[row + i])))
                if focused:
                    order = self._focus(r.question, r.candidates[i].text, flat[row + i])
                per_nodes.append([flat[row + i][m] for m in order[:k]])
                per_sims.append([flat_sims[row + i][m] for m in order[:k]])
            nodes.append(per_nodes)
            sims.append(per_sims)
            row += n
        return nodes, sims

    def _focus(self, question: str, candidate: str, pool: list[int]) -> list[int]:
        """Pool positions, statements mentioning the candidate's own words first (words
        not in the question); retrieval order is kept within each group."""
        own = _stems(candidate) - _stems(question)
        memory = self.memory()
        mentions = [bool(own & _stems(memory.concepts[c].text)) for c in pool]
        return [m for m in range(len(pool)) if mentions[m]] + [
            m for m in range(len(pool)) if not mentions[m]
        ]

    def explain_candidate(self, output: BranchOutput, b: int, i: int) -> list[float]:
        """Leave-one-out contributions of candidate i's nodes (example b), from an output
        computed with ``keep=True``: one candidate at a time, for streamed traces."""
        if output.encoded is None:
            raise ValueError("explain_candidate needs an output computed with keep=True")
        return self._leave_one_out(output.encoded, output.logits, b, i, len(output.nodes[b][i]))

    def direct_logits(self, output: BranchOutput) -> torch.Tensor:
        """The empty-branch (direct) logits for an output computed with ``keep=True``."""
        if output.encoded is None:
            raise ValueError("direct_logits needs an output computed with keep=True")
        return self._initial(output.encoded)

    def _initial(self, encoded: _Encoded) -> torch.Tensor:
        """Empty-branch logits: the direct decision, with the same question encoding."""
        requests = encoded.requests
        counts = [len(r.candidates) for r in requests]
        empty = [self._candidate_text(r, i, []) for r in requests for i in range(len(r.candidates))]
        base, _ = self._pack(self.encoder.encode(empty).pooled, counts)
        return self.scorer(encoded.context, base, encoded.mask)

    def _leave_one_out(
        self, encoded: _Encoded, logits: torch.Tensor, b: int, i: int, count: int
    ) -> list[float]:
        """For each of candidate i's nodes, the drop in its own logit when only that node
        is removed from its branch (other branches unchanged)."""
        if count == 0:
            return []
        request = encoded.requests[b]
        texts = [
            self._candidate_text(request, i, encoded.statements(b, i, m)) for m in range(count)
        ]
        pooled = self.encoder.encode(texts).pooled
        variants = encoded.candidates[b : b + 1].repeat(count, 1, 1)
        variants[:, i] = pooled
        context = _repeat(_row(encoded.context, b), count)
        without = self.scorer(context, variants, encoded.mask[b : b + 1].repeat(count, 1))
        return [float(logits[b, i] - without[m, i]) for m in range(count)]

    def _allowed(self, request: DecisionRequest, n: int) -> tuple[int, Termination]:
        """(nodes per candidate, termination). Expansion needs depth, one query per
        candidate, and at least one node per candidate; up to ``max_width``."""
        budget = request.budget
        if budget.max_depth == 0 and budget.max_proposals == 0:
            return 0, Termination.COMPLETED  # a direct request
        per_candidate = min(budget.max_proposals, budget.max_nodes) // n
        if budget.max_depth < 1 or budget.max_queries < n or per_candidate < 1:
            return 0, Termination.BUDGET_EXHAUSTED
        if per_candidate > self.branch_config.max_width:
            return self.branch_config.max_width, Termination.POLICY_STOP
        return per_candidate, Termination.COMPLETED


_WORD = re.compile(r"[a-z]{3,}")
# Retrieved statements considered per candidate before focused reordering.
FOCUS_POOL = 64
# Fused proposals (R025): pool size, and the reciprocal-rank-fusion constant.
FUSED_POOL = 32
RRF_K = 60


def _stems(text: str) -> set[str]:
    """Content-word stems: lower-case words of 3+ letters, plurals made singular."""
    stems = set()
    for word in _WORD.findall(text.lower()):
        if word in _STOP:
            continue
        stems.add(_singular(word))
    return stems


def _singular(word: str) -> str:
    """batteries → battery, boxes → box, wires → wire; "glass" stays."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


_STOP = frozenset(
    [
        "the",
        "and",
        "for",
        "from",
        "with",
        "that",
        "this",
        "these",
        "those",
        "are",
        "was",
        "were",
        "been",
        "being",
        "has",
        "have",
        "had",
        "not",
        "but",
        "its",
        "into",
        "than",
        "then",
        "there",
        "their",
        "them",
        "they",
        "which",
        "what",
        "who",
        "how",
        "why",
        "when",
        "can",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "most",
        "more",
        "less",
        "some",
        "any",
        "all",
        "each",
        "other",
        "such",
        "only",
        "also",
        "very",
        "much",
        "many",
        "one",
        "two",
        "use",
        "used",
    ]
)


def _repeat(encoded: EncodedText, count: int) -> EncodedText:
    """One encoded example repeated ``count`` times along the batch."""
    return EncodedText(
        states=encoded.states.repeat(count, 1, 1),
        mask=encoded.mask.repeat(count, 1),
        pooled=encoded.pooled.repeat(count, 1),
        offsets=None if encoded.offsets is None else encoded.offsets.repeat(count, 1, 1),
    )


def _row(encoded: EncodedText, b: int) -> EncodedText:
    """One example of an encoded batch."""
    return EncodedText(
        states=encoded.states[b : b + 1],
        mask=encoded.mask[b : b + 1],
        pooled=encoded.pooled[b : b + 1],
        offsets=None if encoded.offsets is None else encoded.offsets[b : b + 1],
    )
