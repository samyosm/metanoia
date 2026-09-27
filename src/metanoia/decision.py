"""Shared candidate-conditioned scorer (SCIENTIFIC_FOUNDATIONS eq. 7).

    s_i = g([h_x; a_i; mean_j phi(a_j); r_i; a_i * h_x])

``r_i`` is candidate-specific evidence obtained by attending from the candidate to the
retained context token states. The set term is a mean (not a sum) so its scale does
not depend on the number of candidates. Every per-candidate map is shared, so the
scorer is permutation-equivariant by construction and has no fixed label-ID head.
"""

import torch
from torch import nn

from metanoia.interfaces import EncodedText


class CandidateScorer(nn.Module):
    def __init__(
        self, input_dim: int, latent_dim: int, heads: int, hidden_dim: int, dropout: float
    ) -> None:
        super().__init__()
        if latent_dim % heads:
            raise ValueError(f"latent_dim {latent_dim} is not divisible by heads {heads}")
        self.context_proj = nn.Linear(input_dim, latent_dim)
        self.candidate_proj = nn.Linear(input_dim, latent_dim)
        self.set_encoder = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU())
        self.evidence = nn.MultiheadAttention(latent_dim, heads, dropout=dropout, batch_first=True)
        self.readout = nn.Sequential(
            nn.Linear(5 * latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, context: EncodedText, candidates: torch.Tensor, candidate_mask: torch.Tensor
    ) -> torch.Tensor:
        """``candidates`` [B, N, D_in], ``candidate_mask`` [B, N] -> logits [B, N].

        Padded candidate positions receive ``-inf``.
        """
        tokens = self.context_proj(context.states)  # [B, L, d]
        pooled = self.context_proj(context.pooled)  # [B, d]
        a = self.candidate_proj(candidates)  # [B, N, d]

        weights = candidate_mask.unsqueeze(-1).to(a.dtype)
        set_term = (self.set_encoder(a) * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        evidence, _ = self.evidence(a, tokens, tokens, key_padding_mask=~context.mask)

        n = a.shape[1]
        features = torch.cat(
            [
                pooled.unsqueeze(1).expand(-1, n, -1),
                a,
                set_term.unsqueeze(1).expand(-1, n, -1),
                evidence,
                a * pooled.unsqueeze(1),
            ],
            dim=-1,
        )
        logits = self.readout(features).squeeze(-1)
        return logits.masked_fill(~candidate_mask, float("-inf"))
