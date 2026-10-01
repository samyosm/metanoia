"""Single-sequence decision model (decision 0015): all options in one marked sequence.

One encoder pass per request scores every option, and every option attends to the question,
the context and every other option at token level. The marker format reuses the masked-
language-model encoder's own "what fills this slot" behaviour. Unlike the joint model, it is
**position-sensitive**; order robustness comes from random option order in training, rotation
averaging at inference, and (optionally) position resets (``SequenceConfig``).
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from metanoia.config import SequenceConfig
from metanoia.encoding.sequence import BuiltSequences, build_sequences
from metanoia.model import DecisionOutput, DirectDecisionModel
from metanoia_contracts import Candidate, DecisionRequest


class SequenceHead(nn.Module):
    """Transformer layers over the whole sequence, then an MLP on each option's marker."""

    def __init__(self, dim: int, config: SequenceConfig) -> None:
        super().__init__()
        if dim % config.head_heads:
            raise ValueError(f"encoder dim {dim} is not divisible by head_heads")
        layer = nn.TransformerEncoderLayer(
            dim,
            config.head_heads,
            dim_feedforward=config.head_ff_mult * dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = (
            nn.TransformerEncoder(layer, config.head_layers, enable_nested_tensor=False)
            if config.head_layers
            else None
        )
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(
        self,
        states: torch.Tensor,
        attention_mask: torch.Tensor,
        markers: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.layers is not None:
            states = self.layers(states, src_key_padding_mask=~attention_mask.bool())
        gathered = states.gather(1, markers.unsqueeze(-1).expand(-1, -1, states.shape[-1]))
        logits = self.mlp(self.norm(gathered)).squeeze(-1)
        return logits.masked_fill(~option_mask, float("-inf"))


class SequenceDecisionModel(DirectDecisionModel):
    candidate_encoding = "sequence"  # type: ignore[assignment]

    def __init__(self, encoder: Any, config: SequenceConfig) -> None:
        head = SequenceHead(encoder.output_dim, config)
        super().__init__(encoder, head, candidate_encoding="separate")  # type: ignore[arg-type]
        self.candidate_encoding = "sequence"
        self.config = config

    def build(self, requests: Sequence[DecisionRequest]) -> BuiltSequences:
        return build_sequences(
            self.encoder.tokenizer,
            requests,
            max_length=self.encoder.max_length,
            head_max_len=self.config.head_max_len,
            option_max_tokens=self.config.option_max_tokens,
            position_reset=self.config.position_reset,
            position_hash=self.config.position_hash,
            position_hash_range=self.config.position_hash_range,
        )

    def _score(self, requests: Sequence[DecisionRequest]) -> torch.Tensor:
        built = self.build(requests)
        device = self.device
        ids = built.input_ids.to(device)
        mask = built.attention_mask.to(device)
        positions = None if built.position_ids is None else built.position_ids.to(device)
        encoder: Any = self.encoder
        states = encoder.encode_ids(ids, mask, positions)
        head: Any = self.scorer
        return head(states, mask, built.markers.to(device), built.option_mask.to(device))

    def forward(self, requests: Sequence[DecisionRequest]) -> DecisionOutput:
        if not requests:
            raise ValueError("empty batch")
        counts = [len(r.candidates) for r in requests]
        n_max = max(counts)
        if self.training and self.config.shuffle_options:
            orders = [torch.randperm(n).tolist() for n in counts]
            logits = self._score_in_orders(requests, orders)
        elif self.config.rotations > 1:
            logits = self._rotation_average(requests)
        else:
            logits = self._score(requests)
        mask = torch.zeros(len(requests), n_max, dtype=torch.bool, device=logits.device)
        for i, count in enumerate(counts):
            mask[i, :count] = True
        return DecisionOutput(logits=logits, candidate_mask=mask)

    def _score_in_orders(
        self, requests: Sequence[DecisionRequest], orders: Sequence[Sequence[int]]
    ) -> torch.Tensor:
        """Score with options presented in ``orders``; logits come back in the original order."""
        permuted = [_permuted(r, order) for r, order in zip(requests, orders, strict=True)]
        scored = self._score(permuted)
        out = scored.new_full(scored.shape, float("-inf"))
        for b, order in enumerate(orders):
            # presented slot j holds original option order[j]
            out[b, torch.tensor(order, device=out.device)] = scored[b, : len(order)]
        return out

    def _rotation_average(self, requests: Sequence[DecisionRequest]) -> torch.Tensor:
        """Mean log-probability over cyclic rotations of the option order, one batch."""
        k = self.config.rotations
        flat_requests, flat_orders, owner = [], [], []
        for b, r in enumerate(requests):
            n = len(r.candidates)
            for shift in range(min(k, n)):
                flat_requests.append(r)
                flat_orders.append([(i + shift) % n for i in range(n)])
                owner.append(b)
        scored = self._score_in_orders(flat_requests, flat_orders)
        log_probs = F.log_softmax(scored.float(), dim=-1)
        n_max = scored.shape[1]
        total = scored.new_zeros(len(requests), n_max, dtype=torch.float32)
        counts = torch.zeros(len(requests), device=scored.device)
        for row, b in enumerate(owner):
            total[b] += torch.nan_to_num(log_probs[row], neginf=0.0)
            counts[b] += 1
        mean = total / counts.unsqueeze(-1)
        mask = torch.zeros_like(mean, dtype=torch.bool)
        for b, r in enumerate(requests):
            mask[b, : len(r.candidates)] = True
        return mean.masked_fill(~mask, float("-inf"))


def _permuted(request: DecisionRequest, order: Sequence[int]) -> DecisionRequest:
    candidates: tuple[Candidate, ...] = tuple(request.candidates[i] for i in order)
    return request.model_copy(update={"candidates": candidates})
