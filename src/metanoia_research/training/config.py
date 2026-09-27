"""Validated training configuration."""

from typing import Literal

from pydantic import Field

from metanoia_contracts.base import StrictModel


class TrainingConfig(StrictModel):
    config_version: Literal[1] = 1
    name: str = Field(min_length=1)
    model_config_path: str
    dataset: str
    train_split: str = "train"
    dev_split: str = "dev"
    epochs: int = Field(ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    grad_clip: float | None = Field(default=1.0, gt=0)
    seed: int = 0
    device: str = "cpu"
    max_train_examples: int | None = Field(default=None, ge=1)
    max_dev_examples: int | None = Field(default=None, ge=1)
    # How ``max_*_examples`` selects: the first rows, or a seeded random sample.
    subset: Literal["head", "random"] = "head"
    # Defaults reproduce the original loop: constant LR, fp32, no accumulation.
    encoder_learning_rate: float | None = Field(default=None, gt=0)
    warmup_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    schedule: Literal["constant", "linear"] = "constant"
    precision: Literal["fp32", "fp16-mixed"] = "fp32"
    # Examples per forward pass; gradients accumulate to ``batch_size``.
    micro_batch_size: int | None = Field(default=None, ge=1)
    log_every_steps: int | None = Field(default=None, ge=1)
