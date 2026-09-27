"""One round of CONNECT + AGGREGATE over the workspace (decision 0008).

A pre-norm transformer block whose weights are shared across rounds. Its attention
weights are that round's question-conditioned, weighted edges between nodes; the
MLP updates each node from its aggregated messages. With ``memory`` (the latent
control), nodes additionally read a fixed memory by cross-attention.
"""

import torch
from torch import nn


class WorkspaceBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden_dim: int, dropout: float, cross: bool):
        super().__init__()
        if dim % heads:
            raise ValueError(f"workspace dim {dim} is not divisible by heads {heads}")
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_norm: nn.LayerNorm | None = None
        self.cross: nn.MultiheadAttention | None = None
        if cross:
            self.cross_norm = nn.LayerNorm(dim)
            self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim)
        )

    def forward(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``nodes`` [B, N, d], ``mask`` [B, N] -> updated nodes and the attention weights
        [B, N, K] of the round (K = N, or the memory size for the latent control's read)."""
        h = self.norm(nodes)
        messages, weights = self.attention(
            h, h, h, key_padding_mask=~mask, need_weights=True, average_attn_weights=True
        )
        nodes = nodes + messages
        if self.cross is not None and self.cross_norm is not None:
            if memory is None or memory_mask is None:
                raise ValueError("a cross-attending block needs a memory")
            h = self.cross_norm(nodes)
            read, weights = self.cross(
                h,
                memory,
                memory,
                key_padding_mask=~memory_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            nodes = nodes + read
        nodes = nodes + self.mlp(self.mlp_norm(nodes))
        return nodes, weights
