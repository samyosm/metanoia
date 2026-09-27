"""A trainable hashed-token encoder that needs no download.

It is a control and a smoke-test encoder, not a substitute for pretrained language
understanding. Token hashing uses BLAKE2b so ids are stable across processes.
"""

import hashlib
import re
from collections.abc import Sequence

import torch
from torch import nn

from metanoia.interfaces import EncodedText

_TOKEN = re.compile(r"\w+|[^\w\s]")
TOKENIZER_REVISION = "regex-lower-blake2b/1"


class HashingEncoder(nn.Module):
    external_prefixes: tuple[str, ...] = ()

    def __init__(self, buckets: int, dim: int, layers: int, heads: int, max_length: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"encoder dim {dim} is not divisible by heads {heads}")
        self.buckets, self.max_length, self.output_dim = buckets, max_length, dim
        self.embedding = nn.Embedding(buckets + 1, dim, padding_idx=0)
        self.position = nn.Embedding(max_length, dim)
        self.layers: nn.TransformerEncoder | None = None
        if layers:
            layer = nn.TransformerEncoderLayer(
                dim, heads, dim_feedforward=2 * dim, dropout=0.0, batch_first=True
            )
            self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def token_ids(self, text: str) -> list[int]:
        """Id 0 is padding; truncates to ``max_length`` tokens."""
        tokens = _TOKEN.findall(text.lower())[: self.max_length]
        return [
            1 + int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest()) % self.buckets
            for t in tokens
        ] or [1]

    def token_count(self, text: str) -> int:
        """Tokens before truncation to ``max_length`` (for trace diagnostics)."""
        return len(_TOKEN.findall(text.lower()))

    def encode(self, texts: Sequence[str]) -> EncodedText:
        device = self.embedding.weight.device
        ids = [self.token_ids(text) for text in texts]
        length = max(len(row) for row in ids)
        padded = torch.zeros(len(ids), length, dtype=torch.long, device=device)
        for i, row in enumerate(ids):
            padded[i, : len(row)] = torch.tensor(row, device=device)
        mask = padded != 0
        positions = torch.arange(length, device=device)
        states = self.embedding(padded) + self.position(positions)[None]
        if self.layers is not None:
            states = self.layers(states, src_key_padding_mask=~mask)
        return EncodedText(states=states, mask=mask, pooled=masked_mean(states, mask))

    def identity(self) -> dict[str, object]:
        return {"kind": "hashing", "tokenizer": TOKENIZER_REVISION}


def masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(states.dtype)
    return (states * weights).sum(1) / weights.sum(1).clamp_min(1.0)
