"""A frozen sentence-embedding encoder for memory keys and retrieval queries.

Separate from the decision model's encoder, so fine-tuning the decision model never
makes the memory keys stale (PLAN §7.1, option 3; decision 0010).
"""

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn


class RetrievalEncoder(nn.Module):
    def __init__(
        self,
        model: str,
        revision: str,
        pooling: Literal["cls", "mean"],
        max_length: int = 128,
        query_prefix: str = "",
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_name, self.requested_revision = model, revision
        self.pooling, self.max_length, self.query_prefix = pooling, max_length, query_prefix
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModel.from_pretrained(model, revision=revision)
        self.model.requires_grad_(False)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    def train(self, mode: bool = True) -> "RetrievalEncoder":
        super().train(mode)
        self.model.eval()  # always frozen and deterministic
        return self

    @torch.no_grad()
    def embed(
        self, texts: Sequence[str], query: bool = False, batch_size: int = 256
    ) -> torch.Tensor:
        """L2-normalized embeddings [N, dim] (float32, on the model's device)."""
        device = next(self.model.parameters()).device
        prefix = self.query_prefix if query else ""
        out = []
        for start in range(0, len(texts), batch_size):
            chunk = [prefix + t for t in texts[start : start + batch_size]]
            batch = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            states = self.model(**batch).last_hidden_state
            if self.pooling == "cls":
                pooled = states[:, 0]
            else:
                mask = batch["attention_mask"].unsqueeze(-1).to(states.dtype)
                pooled = (states * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            out.append(torch.nn.functional.normalize(pooled.float(), dim=-1))
        return torch.cat(out) if out else torch.zeros(0, self.dim, device=device)

    def identity(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "requested_revision": self.requested_revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "pooling": self.pooling,
            "max_length": self.max_length,
            "query_prefix": self.query_prefix,
            "normalized": True,
        }
