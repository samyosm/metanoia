"""Validated model configuration. Names map to implementations in :mod:`metanoia.build`.

The meaning of an established configuration must never change: add a new ``kind``
or field with an explicit default instead of redefining an old one.
"""

from typing import Annotated, Literal

from pydantic import Field

from metanoia_contracts.base import StrictModel


class HashingEncoderConfig(StrictModel):
    """Trainable hashed-token encoder: offline, CPU-friendly, for smoke tests and controls."""

    kind: Literal["hashing"] = "hashing"
    buckets: int = Field(default=4096, ge=16)
    dim: int = Field(default=64, ge=8)
    layers: int = Field(default=1, ge=0)
    heads: int = Field(default=4, ge=1)
    max_length: int = Field(default=512, ge=8)


class HFEncoderConfig(StrictModel):
    """Pretrained Hugging Face encoder. Pin ``revision`` to a commit hash.

    ``frozen = false`` fine-tunes the encoder; its weights are then stored in the
    predictor bundle instead of being referenced by revision only.
    """

    kind: Literal["hf"] = "hf"
    model: str
    revision: str
    max_length: int = Field(default=512, ge=8)
    pooling: Literal["mean", "cls"] = "mean"
    frozen: bool = True
    gradient_checkpointing: bool = False


EncoderConfig = Annotated[HashingEncoderConfig | HFEncoderConfig, Field(discriminator="kind")]


class DirectScorerConfig(StrictModel):
    kind: Literal["direct"] = "direct"
    latent_dim: int = Field(default=64, ge=8)
    heads: int = Field(default=4, ge=1)
    hidden_dim: int = Field(default=128, ge=8)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    # ``separate``: each candidate text is encoded alone (a_i ignores the input).
    # ``joint``: each candidate is encoded together with the rendered input, so a_i is
    # contextual (N encoder passes per request instead of one).
    candidate_encoding: Literal["separate", "joint"] = "separate"


class ModelConfig(StrictModel):
    config_version: Literal[1] = 1
    name: str = Field(min_length=1)
    encoder: EncoderConfig
    scorer: DirectScorerConfig = Field(default_factory=DirectScorerConfig)
