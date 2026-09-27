"""Hugging Face encoder adapter (requires the ``hf`` optional dependencies).

Frozen by default; ``frozen=False`` fine-tunes the pretrained weights.
"""

from collections.abc import Sequence
from typing import Literal

from torch import nn

from metanoia.encoding.hashing import masked_mean
from metanoia.interfaces import EncodedText


class HFEncoder(nn.Module):
    def __init__(
        self,
        model: str,
        revision: str,
        max_length: int,
        pooling: Literal["mean", "cls"],
        frozen: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_name, self.requested_revision = model, revision
        self.max_length, self.pooling = max_length, pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModel.from_pretrained(model, revision=revision)
        self.frozen = frozen
        # Frozen pretrained weights are identified by model name and resolved revision,
        # not copied into predictor bundles; fine-tuned weights are stored.
        self.external_prefixes: tuple[str, ...] = ("model.",) if frozen else ()
        if frozen:
            self.model.requires_grad_(False)
            self.model.eval()
        elif gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.output_dim = int(self.model.config.hidden_size)

    def train(self, mode: bool = True) -> "HFEncoder":
        # A frozen encoder always runs deterministically (no dropout).
        super().train(mode)
        if self.frozen:
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
        identity: dict[str, object] = {
            "kind": "hf",
            "model": self.model_name,
            "requested_revision": self.requested_revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "pooling": self.pooling,
            "max_length": self.max_length,
        }
        # Recorded only when fine-tuned, so identities of frozen bundles stay unchanged.
        if not self.frozen:
            identity["frozen"] = False
        return identity
