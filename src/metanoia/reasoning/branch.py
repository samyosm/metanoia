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
from typing import Literal

import torch

from metanoia.config import BranchConfig
from metanoia.decision import CandidateScorer
from metanoia.encoding.inputs import render_candidate_joint, render_input
from metanoia.interfaces import EncodedText, TextEncoder
from metanoia.memory.access import MemoryAccess
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

    def forward(  # type: ignore[override]
        self,
        requests: Sequence[DecisionRequest],
        width: int | None = None,
        explain: bool = False,
    ) -> BranchOutput:
        """``width`` (training, dev evaluation): read that many nodes per candidate for
        every example; otherwise each request's budget decides. ``explain`` adds the
        empty-branch logits and per-node leave-one-out contributions (for traces)."""
        if not requests:
            raise ValueError("empty batch")
        counts = [len(r.candidates) for r in requests]
        plan = [
            (width, Termination.COMPLETED) if width is not None else self._allowed(r, n)
            for r, n in zip(requests, counts, strict=True)
        ]
        if self.branch_config.nodes == "none":
            plan = [(0, term) for _, term in plan]
        if any(k for k, _ in plan):
            size = len(self.memory())
            plan = [(min(k, size), term) for k, term in plan]
        nodes, sims = self._nodes(requests, [k for k, _ in plan])
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
        initial, contributions = None, None
        if explain:
            initial, contributions = self._explain(
                requests, context, candidates, mask, logits, nodes, statements
            )
        return BranchOutput(
            logits=logits,
            candidate_mask=mask,
            width=[k for k, _ in plan],
            nodes=nodes,
            similarity=sims,
            termination=[t for _, t in plan],
            initial_logits=initial,
            contributions=contributions,
        )

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

    def _explain(
        self,
        requests: Sequence[DecisionRequest],
        context: EncodedText,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        logits: torch.Tensor,
        nodes: list[list[list[int]]],
        statements,  # type: ignore[no-untyped-def]
    ) -> tuple[torch.Tensor, list[list[list[float]]]]:
        """Empty-branch logits, and for each node the drop in its own candidate's logit
        when only that node is removed from that branch (other branches unchanged)."""
        counts = [len(r.candidates) for r in requests]
        empty = [self._candidate_text(r, i, []) for r in requests for i in range(len(r.candidates))]
        base, _ = self._pack(self.encoder.encode(empty).pooled, counts)
        initial = self.scorer(context, base, mask)
        variants = [
            (b, i, m)
            for b in range(len(requests))
            for i in range(counts[b])
            for m in range(len(nodes[b][i]))
        ]
        contributions: list[list[list[float]]] = [[[] for _ in range(n)] for n in counts]
        if not variants:
            return initial, contributions
        texts = [self._candidate_text(requests[b], i, statements(b, i, m)) for b, i, m in variants]
        pooled = self.encoder.encode(texts).pooled
        for (b, i, _), vector in zip(variants, pooled, strict=True):
            one = candidates[b : b + 1].clone()
            one[0, i] = vector
            without = self.scorer(_row(context, b), one, mask[b : b + 1])
            contributions[b][i].append(float(logits[b, i] - without[0, i]))
        return initial, contributions

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


def _row(encoded: EncodedText, b: int) -> EncodedText:
    """One example of an encoded batch."""
    return EncodedText(
        states=encoded.states[b : b + 1],
        mask=encoded.mask[b : b + 1],
        pooled=encoded.pooled[b : b + 1],
        offsets=None if encoded.offsets is None else encoded.offsets[b : b + 1],
    )
