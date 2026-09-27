"""The learned decision model: one forward path shared by training and inference.

Inference callers select eval mode and disable gradients at the boundary
(:mod:`metanoia.api`); nothing here suppresses gradients unconditionally.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from metanoia.decision import CandidateScorer
from metanoia.encoding.inputs import render_candidate_joint, render_input
from metanoia.interfaces import TextEncoder
from metanoia_contracts import DecisionRequest


@dataclass(frozen=True)
class DecisionOutput:
    logits: torch.Tensor  # [B, N_max]; -inf at padded positions
    candidate_mask: torch.Tensor  # [B, N_max] bool


class DirectDecisionModel(nn.Module):
    """Depth-zero decision: encode once, score every original candidate."""

    def __init__(
        self,
        encoder: TextEncoder,
        scorer: CandidateScorer,
        candidate_encoding: Literal["separate", "joint"] = "separate",
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoders must be torch.nn.Module instances")
        self.encoder = encoder
        self.scorer = scorer
        self.candidate_encoding = candidate_encoding

    @property
    def device(self) -> torch.device:
        return next(self.scorer.parameters()).device

    def forward(self, requests: Sequence[DecisionRequest]) -> DecisionOutput:
        if not requests:
            raise ValueError("empty batch")
        rendered = [render_input(r) for r in requests]
        context = self.encoder.encode(rendered)
        counts = [len(r.candidates) for r in requests]
        if self.candidate_encoding == "joint":
            texts = [
                render_candidate_joint(c.text, text)
                for r, text in zip(requests, rendered, strict=True)
                for c in r.candidates
            ]
        else:
            texts = [c.text for r in requests for c in r.candidates]
        flat = self.encoder.encode(texts).pooled
        candidates = flat.new_zeros(len(requests), max(counts), flat.shape[-1])
        mask = torch.zeros(len(requests), max(counts), dtype=torch.bool, device=flat.device)
        start = 0
        for i, count in enumerate(counts):
            candidates[i, :count] = flat[start : start + count]
            mask[i, :count] = True
            start += count
        return DecisionOutput(logits=self.scorer(context, candidates, mask), candidate_mask=mask)

    def persistent_state_dict(self) -> dict[str, torch.Tensor]:
        """Weights to store in a bundle, excluding externally pinned encoder weights."""
        external = tuple(f"encoder.{p}" for p in self.encoder.external_prefixes)
        return {k: v for k, v in self.state_dict().items() if not k.startswith(external)}

    def load_persistent_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        external = tuple(f"encoder.{p}" for p in self.encoder.external_prefixes)
        result = self.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith(external)]
        if missing or result.unexpected_keys:
            raise ValueError(
                f"incompatible weights: missing {missing}, unexpected {result.unexpected_keys}"
            )
