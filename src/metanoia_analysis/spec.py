"""Versioned analysis specifications: which metrics, slices, and intervals to report."""

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from metanoia_contracts.base import StrictModel, load_strict

Metric = Literal["accuracy", "nll", "brier", "ece"]


class AnalysisSpec(StrictModel):
    spec_version: Literal[1] = 1
    spec_id: str
    revision: int = Field(ge=1)
    splits: tuple[str, ...] = Field(min_length=1)
    metrics: tuple[Metric, ...] = ("accuracy", "nll", "brier", "ece")
    ece_bins: int = Field(default=15, ge=1)
    slice_by: tuple[str, ...] = ()
    bootstrap_resamples: int = Field(default=1000, ge=0)
    bootstrap_seed: int = 0
    confidence: float = Field(default=0.95, gt=0, lt=1)
    # log(0) is undefined; probabilities are floored here before NLL and declared.
    nll_floor: float = Field(default=1e-15, gt=0, lt=1e-3)


def load_spec(path: Path) -> tuple[AnalysisSpec, dict[str, Any]]:
    raw = tomllib.loads(path.read_text())
    return load_strict(AnalysisSpec, raw), raw
