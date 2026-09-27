"""Internal replaceable-component contracts (PLAN.md §5.2).

These may use PyTorch tensors. Shapes: B batch, L tokens, N candidates, D features.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class EncodedText:
    """Token states are kept so later operations can re-attend to the source text."""

    states: torch.Tensor  # [B, L, D] float
    mask: torch.Tensor  # [B, L] bool, True for real tokens
    pooled: torch.Tensor  # [B, D] float


class TextEncoder(Protocol):
    """Text -> token states, masks, and pooled states. Implementations are nn.Modules."""

    output_dim: int
    # State-dict prefixes whose weights come from an external pinned source and are
    # therefore not saved in predictor bundles.
    external_prefixes: tuple[str, ...]

    def encode(self, texts: Sequence[str]) -> EncodedText: ...

    def identity(self) -> dict[str, object]:
        """Resolved identity (checkpoint revision, tokenizer, pooling) for bundles."""
        ...
