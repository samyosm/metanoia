"""Validated model configuration. Names map to implementations in :mod:`metanoia.build`.

The meaning of an established configuration must never change: add a new ``kind``
or field with an explicit default instead of redefining an old one.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

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


class WorkspaceConfig(StrictModel):
    """Stage-3a budgeted workspace (decision 0008).

    ``statements``: nodes are the input's sentences plus the candidates, all updated each
    round. ``latent``: the equal-cost control, where ``latent_slots`` learned nodes plus the
    candidates are updated and the sentence embeddings are read-only memory.
    """

    kind: Literal["workspace"] = "workspace"
    mode: Literal["statements", "latent"] = "statements"
    dim: int = Field(default=256, ge=8)
    heads: int = Field(default=4, ge=1)
    hidden_dim: int = Field(default=512, ge=8)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    latent_slots: int = Field(default=16, ge=1)
    # Hard cap on rounds, whatever the request's budget allows.
    max_rounds: int = Field(default=32, ge=1)
    # Decision 0009. "context": sentence nodes pooled from the full-context encoding (0008).
    # "isolated": each sentence and each candidate encoded on its own, so only the
    # workspace can combine premises.
    encoding: Literal["context", "isolated"] = "context"
    # "residual": direct logits + workspace delta (0008). "workspace": the candidate
    # nodes' readout only, with no direct-path bypass (0009).
    readout: Literal["residual", "workspace"] = "residual"

    @model_validator(mode="after")
    def _no_bypass(self) -> "WorkspaceConfig":
        if self.encoding == "isolated" and self.readout != "workspace":
            raise ValueError('encoding "isolated" requires readout "workspace"')
        return self


class ExpansionConfig(StrictModel):
    """Stage-2 EXPAND v1 (decision 0011): candidates expand into concepts from a memory.

    ``concepts``: "retrieved" (the idea), "random" (equal-cost control: random concepts),
    or "none" (the same trainable head with no concepts, a capacity control).
    """

    kind: Literal["expansion"] = "expansion"
    memory: str  # path to a memory bundle (artifacts/memories/<id>)
    concepts: Literal["retrieved", "random", "none"] = "retrieved"
    proposals: int = Field(default=16, ge=1)  # concepts per candidate in training
    dim: int = Field(default=256, ge=8)
    hidden_dim: int = Field(default=512, ge=8)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    # v1: the encoder and direct scorer stay frozen; only the expansion trains.
    freeze_direct: bool = True


class ModelConfig(StrictModel):
    config_version: Literal[1] = 1
    name: str = Field(min_length=1)
    encoder: EncoderConfig
    scorer: DirectScorerConfig = Field(default_factory=DirectScorerConfig)
    # None: the direct scorer only (Stage 1). Set: the Stage-3a workspace on top of it.
    workspace: WorkspaceConfig | None = None
    # Set: EXPAND v1 on top of the direct scorer (decision 0011).
    expansion: ExpansionConfig | None = None

    @model_validator(mode="after")
    def _one_mechanism(self) -> "ModelConfig":
        if self.workspace is not None and self.expansion is not None:
            raise ValueError("a model has either a workspace or an expansion, not both")
        return self
