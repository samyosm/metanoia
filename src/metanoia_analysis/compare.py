"""Paired comparison of two runs collected under the same protocol.

Runs must cover exactly the same keys; the intersection of successful examples is
never taken silently. A cross-protocol comparison must name each accepted mismatch.
"""

from pathlib import Path
from typing import Any

import numpy as np

from metanoia_analysis import ANALYSIS_REVISION
from metanoia_analysis.load import Item, LoadedRun, align, load_labels, load_run
from metanoia_analysis.metrics import per_example, strictly_correct
from metanoia_analysis.report import STANDING_LIMITATIONS, partition
from metanoia_analysis.spec import AnalysisSpec
from metanoia_analysis.statistics import group_bootstrap_ci, reversal_rates
from metanoia_contracts.artifacts import AnalysisResult, MetricValue
from metanoia_contracts.provenance import environment_snapshot, utc_now

COMPATIBILITY_FIELDS = ("suite", "protocol_sha256", "dataset_id", "cases_sha256", "labels_sha256")


def mismatches(a: LoadedRun, b: LoadedRun) -> list[str]:
    fields = []
    for name in COMPATIBILITY_FIELDS:
        left, right = getattr(a.manifest, name), getattr(b.manifest, name)
        if name == "suite":
            left = (left["suite_id"], left["revision"])
            right = (right["suite_id"], right["revision"])
        if left != right:
            fields.append(name)
    return fields


def compare(
    run_a: Path,
    run_b: Path,
    dataset_dir: Path,
    spec: AnalysisSpec,
    spec_raw: dict[str, Any],
    accept_mismatches: tuple[str, ...] = (),
) -> AnalysisResult:
    """Differences are ``b - a``: a positive accuracy difference means ``b`` is better."""
    a, b = load_run(run_a), load_run(run_b)
    found = mismatches(a, b)
    unaccepted = [f for f in found if f not in accept_mismatches]
    if unaccepted:
        raise ValueError(f"runs are not comparable; mismatched: {unaccepted}")
    items_a = align(a, load_labels(dataset_dir, a.manifest.labels_sha256))
    items_b = align(b, load_labels(dataset_dir, b.manifest.labels_sha256))
    budget_ids = [x.id for x in a.protocol.budgets]

    metrics: list[MetricValue] = []
    by_key_b = {(i.example_id, i.budget_id, i.repeat_id): i for i in items_b}
    for split, budget_id, left in partition(items_a, spec, budget_ids):
        try:
            right = [by_key_b[(i.example_id, i.budget_id, i.repeat_id)] for i in left]
        except KeyError as missing:
            raise ValueError(f"run b lacks prediction {missing}") from None
        if len(right) != sum(1 for i in items_b if i.split == split and i.budget_id == budget_id):
            raise ValueError("run b has predictions that run a lacks")
        metrics += _paired(left, right, spec, split, budget_id)

    limitations = list(STANDING_LIMITATIONS)
    if found:
        limitations.append(f"Accepted protocol mismatches: {found}.")
    return AnalysisResult(
        analysis_id=f"{a.manifest.run_id}__vs__{b.manifest.run_id}__{spec.spec_id}-r{spec.revision}",
        kind="compare",
        inputs={
            "a": {
                "run_id": a.manifest.run_id,
                "predictor_id": a.manifest.predictor["predictor_id"],
                "predictions_sha256": a.manifest.predictions.sha256,
            },
            "b": {
                "run_id": b.manifest.run_id,
                "predictor_id": b.manifest.predictor["predictor_id"],
                "predictions_sha256": b.manifest.predictions.sha256,
            },
            "direction": "b - a",
            "accepted_mismatches": list(found),
        },
        specification=spec_raw,
        analysis_revision=ANALYSIS_REVISION,
        environment=environment_snapshot(),
        counts={
            "a": a.manifest.counts.model_dump(),
            "b": b.manifest.counts.model_dump(),
        },
        metrics=metrics,
        limitations=limitations,
        created_at=utc_now(),
    )


def _paired(
    left: list[Item], right: list[Item], spec: AnalysisSpec, split: str, budget_id: str
) -> list[MetricValue]:
    groups = [i.group_id for i in left]
    values = []
    for metric in ("accuracy", "nll", "brier"):
        if metric not in spec.metrics:
            continue
        diff = per_example(right, metric, spec.nll_floor) - per_example(
            left, metric, spec.nll_floor
        )
        ci = group_bootstrap_ci(
            diff, groups, spec.bootstrap_resamples, spec.bootstrap_seed, spec.confidence
        )
        values.append(
            MetricValue(
                split=split,
                budget_id=budget_id,
                metric=f"{metric}_diff",
                value=float(diff.mean()),
                ci_low=ci[0] if ci else None,
                ci_high=ci[1] if ci else None,
                n=len(left),
            )
        )
    rates = reversal_rates(
        np.array([strictly_correct(i) for i in left]),
        np.array([strictly_correct(i) for i in right]),
    )
    values += [
        MetricValue(split=split, budget_id=budget_id, metric=name, value=value, n=len(left))
        for name, value in rates.items()
    ]
    return values
