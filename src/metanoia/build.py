"""Explicit factories from validated configuration to components (PLAN.md §5.2).

No plugin discovery: every supported name is listed here, and unknown names are
rejected by the configuration schema before reaching this module.
"""

from typing import Any

import torch

from metanoia.config import HashingEncoderConfig, HFEncoderConfig, ModelConfig
from metanoia.decision import CandidateScorer
from metanoia.interfaces import TextEncoder
from metanoia.model import DirectDecisionModel


def build_encoder(config: HashingEncoderConfig | HFEncoderConfig) -> TextEncoder:
    if isinstance(config, HashingEncoderConfig):
        from metanoia.encoding.hashing import HashingEncoder

        return HashingEncoder(
            config.buckets, config.dim, config.layers, config.heads, config.max_length
        )
    from metanoia.encoding.hf import HFEncoder

    return HFEncoder(
        config.model,
        config.revision,
        config.max_length,
        config.pooling,
        frozen=config.frozen,
        gradient_checkpointing=config.gradient_checkpointing,
    )


def build_model(config: ModelConfig, seed: int | None = None) -> DirectDecisionModel:
    """Build a freshly initialized model. ``seed`` makes initialization reproducible."""
    if seed is not None:
        torch.manual_seed(seed)
    encoder = build_encoder(config.encoder)
    scorer = CandidateScorer(
        input_dim=encoder.output_dim,
        latent_dim=config.scorer.latent_dim,
        heads=config.scorer.heads,
        hidden_dim=config.scorer.hidden_dim,
        dropout=config.scorer.dropout,
    )
    return DirectDecisionModel(encoder, scorer, config.scorer.candidate_encoding)


def check_encoder_identity(model: DirectDecisionModel, recorded: dict[str, Any]) -> None:
    """Refuse weights trained against a different encoder checkpoint or tokenizer."""
    current = model.encoder.identity()
    if current != recorded:
        raise ValueError(f"encoder identity mismatch: bundle {recorded}, loaded {current}")
