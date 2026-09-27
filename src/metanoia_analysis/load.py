"""Load and validate saved runs and labels, then align them by stable example IDs.

Every check fails loudly: hash mismatches, duplicate or missing keys, labels absent
from the candidate set. Failed predictions are kept and handled by the protocol's
declared failure policy, never silently dropped.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from metanoia_contracts.artifacts import (
    LABEL_CORE_COLUMNS,
    DatasetManifest,
    LabelRecord,
    PredictionRecord,
    PredictionStatus,
    RunManifest,
)
from metanoia_contracts.base import load_strict
from metanoia_contracts.provenance import content_sha256
from metanoia_contracts.suites import FailurePolicy, SuiteProtocol


@dataclass(frozen=True)
class LoadedRun:
    path: Path
    manifest: RunManifest
    records: list[PredictionRecord]

    @property
    def protocol(self) -> SuiteProtocol:
        return load_strict(SuiteProtocol, self.manifest.protocol)


@dataclass(frozen=True)
class Item:
    """One scored prediction joined with its label."""

    example_id: str
    group_id: str
    split: str
    budget_id: str
    repeat_id: int
    candidate_ids: tuple[str, ...]
    label_index: int
    probabilities: np.ndarray
    logits: np.ndarray | None  # None for failed predictions
    failed: bool
    annotations: dict[str, Any]

    def with_probabilities(self, probabilities: np.ndarray) -> "Item":
        return replace(self, probabilities=probabilities)


def load_run(run_dir: Path) -> LoadedRun:
    manifest = load_strict(RunManifest, json.loads((run_dir / "run.json").read_text()))
    rows = pq.read_table(run_dir / manifest.predictions.path).to_pylist()
    if content_sha256(rows) != manifest.predictions.sha256:
        raise ValueError(f"predictions in {run_dir} do not match their recorded hash")
    records = [load_strict(PredictionRecord, row) for row in rows]
    keys = [r.key for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate prediction keys in {run_dir}")
    if len(records) != manifest.counts.expected:
        raise ValueError(
            f"{run_dir} has {len(records)} predictions; its protocol expects "
            f"{manifest.counts.expected}"
        )
    return LoadedRun(run_dir, manifest, records)


def load_labels(dataset_dir: Path, expected_sha256: str) -> dict[str, LabelRecord]:
    manifest = load_strict(DatasetManifest, json.loads((dataset_dir / "dataset.json").read_text()))
    rows = pq.read_table(dataset_dir / manifest.labels.path).to_pylist()
    labels = [
        LabelRecord(
            **{k: row[k] for k in LABEL_CORE_COLUMNS},
            annotations={k: v for k, v in row.items() if k not in LABEL_CORE_COLUMNS},
        )
        for row in rows
    ]
    if content_sha256(label.model_dump(mode="json") for label in labels) != expected_sha256:
        raise ValueError(f"labels in {dataset_dir} are not the labels this run was collected for")
    return {label.example_id: label for label in labels}


def align(run: LoadedRun, labels: dict[str, LabelRecord]) -> list[Item]:
    policy = run.protocol.evaluation.failure_policy
    items = []
    for record in run.records:
        label = labels.get(record.example_id)
        if label is None:
            raise ValueError(f"no label for {record.example_id}")
        if label.label_id not in record.candidate_ids:
            raise ValueError(f"label of {record.example_id} is not among its candidates")
        failed = record.status is PredictionStatus.FAILED
        if failed and policy is FailurePolicy.ERROR:
            raise ValueError(f"{record.key} failed and the protocol forbids failures")
        n = len(record.candidate_ids)
        items.append(
            Item(
                example_id=record.example_id,
                group_id=label.group_id,
                split=label.split,
                budget_id=record.budget_id,
                repeat_id=record.repeat_id,
                candidate_ids=tuple(record.candidate_ids),
                label_index=record.candidate_ids.index(label.label_id),
                # FailurePolicy.UNIFORM: a failure is scored as a uniform distribution.
                probabilities=np.full(n, 1.0 / n) if failed else np.asarray(record.probabilities),
                logits=None if failed else np.asarray(record.logits, dtype=np.float64),
                failed=failed,
                annotations=label.annotations,
            )
        )
    return items
