"""Hugging Face encoder adapter (requires the ``hf`` optional dependencies).

Frozen by default; ``frozen=False`` fine-tunes the pretrained weights.
"""

from collections.abc import Sequence
from typing import Literal

import torch
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

    def token_count(self, text: str) -> int:
        """Tokens (with special tokens) before truncation to ``max_length``."""
        return len(self.tokenizer(text, truncation=False)["input_ids"])

    def encode(self, texts: Sequence[str]) -> EncodedText:
        device = next(self.model.parameters()).device
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(device)
        offsets = batch.pop("offset_mapping").long()
        states = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].bool()
        pooled = states[:, 0] if self.pooling == "cls" else masked_mean(states, mask)
        return EncodedText(states=states, mask=mask, pooled=pooled, offsets=offsets)

    def encode_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Token states for pre-built id sequences (the marker format, decision 0015)."""
        kwargs = {} if position_ids is None else {"position_ids": position_ids}
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        ).last_hidden_state

    def _resolved_revision(self) -> str | None:
        """The checkpoint commit. Newer transformers releases may not record it on the
        config; a full commit hash in ``revision`` is itself the resolved revision."""
        recorded = getattr(self.model.config, "_commit_hash", None)
        if recorded:
            return str(recorded)
        pinned = self.requested_revision
        return (
            pinned if len(pinned) == 40 and all(c in "0123456789abcdef" for c in pinned) else None
        )

    def identity(self) -> dict[str, object]:
        identity: dict[str, object] = {
            "kind": "hf",
            "model": self.model_name,
            "requested_revision": self.requested_revision,
            "resolved_revision": self._resolved_revision(),
            "pooling": self.pooling,
            "max_length": self.max_length,
        }
        # Recorded only when fine-tuned, so identities of frozen bundles stay unchanged.
        if not self.frozen:
            identity["frozen"] = False
        return identity
