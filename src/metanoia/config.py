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


class RerankerConfig(StrictModel):
    """A pinned cross-encoder judge (decision 0014)."""

    model: str
    revision: str
    max_length: int = Field(default=192, ge=16)


class BranchConfig(StrictModel):
    """Stage-2 EXPAND v2 (decision 0012): each candidate's branch is read by the reader.

    Each candidate retrieves ``width`` nodes (statements about concepts) from the memory;
    they are rendered into that candidate's own joint input, and the trainable reader and
    candidate-conditioned scorer judge the candidates with their branches. No new weights.

    ``nodes``: "retrieved" (the idea), "random" (content control), or "none" (empty
    branches: the direct model trained on the same schedule).
    """

    kind: Literal["branch"] = "branch"
    memory: str  # path to a memory bundle (artifacts/memories/<id>)
    nodes: Literal["retrieved", "random", "none"] = "retrieved"
    # How retrieved nodes are chosen: "similarity" (top-k for "question candidate",
    # decision 0012) or "focused" (a larger retrieved pool, statements mentioning the
    # candidate first; decision 0013).
    # "fused" (decision 0014, R025): the top 32 for "question candidate", ordered by
    # reciprocal-rank fusion of retrieval similarity and a cross-encoder reranker.
    proposals: Literal["similarity", "focused", "fused"] = "similarity"
    reranker: RerankerConfig | None = None  # required by "fused"
    # Optional precomputed orderings (a proposal-cache artifact) for speed; a request not
    # in the cache is ranked live, identically.
    proposal_cache: str | None = None
    # Widths (nodes per candidate) sampled per training batch; dev loss uses the largest.
    train_widths: tuple[int, ...] = (0, 1, 2, 4, 8)
    max_width: int = Field(default=8, ge=1)  # the most nodes per candidate at inference

    @model_validator(mode="after")
    def _widths(self) -> "BranchConfig":
        if not self.train_widths or min(self.train_widths) < 0:
            raise ValueError("train_widths must be non-negative and non-empty")
        if max(self.train_widths) > self.max_width:
            raise ValueError("train_widths cannot exceed max_width")
        if (self.proposals == "fused") != (self.reranker is not None):
            raise ValueError("a reranker is set exactly when proposals are fused")
        if self.proposal_cache is not None and self.proposals != "fused":
            raise ValueError("proposal caches exist only for fused proposals")
        return self


class ModelConfig(StrictModel):
    config_version: Literal[1] = 1
    name: str = Field(min_length=1)
    encoder: EncoderConfig
    scorer: DirectScorerConfig = Field(default_factory=DirectScorerConfig)
    # None: the direct scorer only (Stage 1). Set: the Stage-3a workspace on top of it.
    workspace: WorkspaceConfig | None = None
    # Set: EXPAND v1 on top of the direct scorer (decision 0011).
    expansion: ExpansionConfig | None = None
    # Set: EXPAND v2, branch reading (decision 0012).
    branch: BranchConfig | None = None

    @model_validator(mode="after")
    def _one_mechanism(self) -> "ModelConfig":
        mechanisms = [m for m in (self.workspace, self.expansion, self.branch) if m is not None]
        if len(mechanisms) > 1:
            raise ValueError("a model has at most one of workspace, expansion and branch")
        if self.branch is not None and self.scorer.candidate_encoding != "joint":
            raise ValueError("branch reading needs joint candidate encoding")
        return self
