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
