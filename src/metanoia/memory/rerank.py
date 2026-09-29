"""A frozen cross-encoder that judges (question, candidate, statement) triples.

Decision 0014: a proposal judge that reads the candidate and the statement together.
Like the retrieval encoder, it is a pinned external model: identified in the bundle,
never trained, never stored in it.
"""

from collections.abc import Sequence

import torch
from torch import nn


def reranker_query(question: str, candidate: str) -> str:
    """The query format screened in R025 ("qa")."""
    return f"Question: {question} Answer: {candidate}"


class Reranker(nn.Module):
    def __init__(self, model: str, revision: str, max_length: int = 192) -> None:
        super().__init__()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name, self.revision, self.max_length = model, revision, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, revision=revision)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "Reranker":
        super().train(False)  # frozen: always deterministic
        return self

    def identity(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "requested_revision": self.revision,
            "max_length": self.max_length,
            "query_format": "qa",
        }

    def score(self, query: str, texts: Sequence[str], batch_size: int = 256) -> list[float]:
        """Relevance logits of each text for the query (higher is more relevant)."""
        return self.score_pairs([(query, t) for t in texts], batch_size)

    @torch.no_grad()
    def score_pairs(self, pairs: Sequence[tuple[str, str]], batch_size: int = 256) -> list[float]:
        """Relevance logits of (query, text) pairs, batched across queries."""
        device = next(self.model.parameters()).device
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            batch = self.tokenizer(
                [q for q, _ in chunk],
                [t for _, t in chunk],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                scores += self.model(**batch).logits.view(-1).float().tolist()
        return scores


def reciprocal_rank_fusion(orderings: Sequence[Sequence[int]], k: int = 60) -> list[int]:
    """Items ranked by the sum of 1 / (k + rank) over the orderings (ties: first seen)."""
    score: dict[int, float] = {}
    for order in orderings:
        for rank, item in enumerate(order):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda item: -score[item])
