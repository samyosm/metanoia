"""Frozen Hugging Face encoder adapter (requires the ``hf`` optional dependencies)."""

from collections.abc import Sequence
from typing import Literal

from torch import nn

from metanoia.encoding.hashing import masked_mean
from metanoia.interfaces import EncodedText


class HFEncoder(nn.Module):
    # Pretrained weights are identified by model name and resolved revision, not copied
    # into predictor bundles.
    external_prefixes: tuple[str, ...] = ("model.",)

    def __init__(
        self, model: str, revision: str, max_length: int, pooling: Literal["mean", "cls"]
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_name, self.requested_revision = model, revision
        self.max_length, self.pooling = max_length, pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModel.from_pretrained(model, revision=revision)
        self.model.requires_grad_(False)
        self.model.eval()
        self.output_dim = int(self.model.config.hidden_size)

    def train(self, mode: bool = True) -> "HFEncoder":
        # The frozen encoder always runs deterministically (no dropout).
        super().train(mode)
        self.model.eval()
        return self

    def encode(self, texts: Sequence[str]) -> EncodedText:
        device = next(self.model.parameters()).device
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        states = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].bool()
        pooled = states[:, 0] if self.pooling == "cls" else masked_mean(states, mask)
        return EncodedText(states=states, mask=mask, pooled=pooled)

    def identity(self) -> dict[str, object]:
        return {
            "kind": "hf",
            "model": self.model_name,
            "requested_revision": self.requested_revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "pooling": self.pooling,
            "max_length": self.max_length,
        }
