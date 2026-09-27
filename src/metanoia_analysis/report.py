"""Score one saved run against its labels under a versioned analysis specification."""

import json
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from metanoia_analysis import ANALYSIS_REVISION
from metanoia_analysis.calibration import apply_temperature
from metanoia_analysis.load import Item, LoadedRun, align, load_labels, load_run
from metanoia_analysis.metrics import expected_calibration_error, per_example, reliability
from metanoia_analysis.spec import AnalysisSpec
from metanoia_analysis.statistics import group_bootstrap_ci
from metanoia_contracts.artifacts import AnalysisResult, CalibrationArtifact, MetricValue
from metanoia_contracts.provenance import environment_snapshot, utc_now

STANDING_LIMITATIONS = [
    "ECE uses top-label confidence with equal-width bins; it depends on binning and sample size.",
    "Intervals are percentile bootstraps over groups (worlds), not over individual examples.",
    "Calibration quality is established only for the evaluated distribution and budgets.",
]


def slices(items: Sequence[Item], slice_by: Sequence[str]) -> Iterator[tuple[str, list[Item]]]:
    yield "all", list(items)
    for key in slice_by:
        buckets: dict[str, list[Item]] = defaultdict(list)
        for item in items:
            if key not in item.annotations:
                raise ValueError(f"slice annotation {key!r} is missing for {item.example_id}")
            buckets[f"{key}={item.annotations[key]}"].append(item)
        yield from sorted(buckets.items())


def metric_values(
    items: Sequence[Item], spec: AnalysisSpec, split: str, budget_id: str, variant: str, name: str
) -> list[MetricValue]:
    values = []
    groups = [i.group_id for i in items]
    for metric in spec.metrics:
        if metric == "ece":
            values.append(
                MetricValue(
                    split=split,
                    budget_id=budget_id,
                    variant=variant,
                    slice=name,
                    metric=metric,
                    value=expected_calibration_error(items, spec.ece_bins),
                    n=len(items),
                )
            )
            continue
        per = per_example(items, metric, spec.nll_floor)
        ci = group_bootstrap_ci(
            per, groups, spec.bootstrap_resamples, spec.bootstrap_seed, spec.confidence
        )
        values.append(
            MetricValue(
                split=split,
                budget_id=budget_id,
                variant=variant,
                slice=name,
                metric=metric,
                value=float(np.mean(per)),
                n=len(items),
                ci_low=ci[0] if ci else None,
                ci_high=ci[1] if ci else None,
            )
        )
    return values


def partition(
    items: Sequence[Item], spec: AnalysisSpec, budget_ids: Sequence[str]
) -> Iterator[tuple[str, str, list[Item]]]:
    for split in spec.splits:
        for budget_id in budget_ids:
            selected = [i for i in items if i.split == split and i.budget_id == budget_id]
            if not selected:
                raise ValueError(f"no predictions for split {split!r}, budget {budget_id!r}")
            yield split, budget_id, selected


def counts(
    run: LoadedRun, items: Sequence[Item], splits: Sequence[str]
) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for split in splits:
        chosen = [i for i in items if i.split == split]
        failed = sum(i.failed for i in chosen)
        table[split] = {
            "attempted": len(chosen),
            "completed": len(chosen) - failed,
            "failed": failed,
        }
    table["run"] = run.manifest.counts.model_dump()
    return table


def analyze(
    run_dir: Path,
    dataset_dir: Path,
    spec: AnalysisSpec,
    spec_raw: dict[str, Any],
    calibration: CalibrationArtifact | None = None,
) -> AnalysisResult:
    run = load_run(run_dir)
    items = align(run, load_labels(dataset_dir, run.manifest.labels_sha256))
    budget_ids = [b.id for b in run.protocol.budgets]
    predictor_id = run.manifest.predictor["predictor_id"]

    variants: list[tuple[str, Callable[[list[Item]], list[Item]]]] = [("raw", lambda x: x)]
    limitations = list(STANDING_LIMITATIONS)
    if calibration is not None:
        if calibration.compatible_predictor_id != predictor_id:
            raise ValueError(
                f"calibration {calibration.calibrator_id} is for "
                f"{calibration.compatible_predictor_id}, not {predictor_id}"
            )
        temperature = calibration.temperature
        variants.append(("calibrated", lambda x: apply_temperature(x, temperature)))
        budget_ids_for_cal = [calibration.budget_id]
        limitations.append(
            f"Calibrated metrics apply temperature {temperature:.4f} fitted on budget "
            f"{calibration.budget_id} only."
        )
    else:
        budget_ids_for_cal = []

    metrics: list[MetricValue] = []
    tables: dict[str, list[dict[str, Any]]] = {}
    for split, budget_id, selected in partition(items, spec, budget_ids):
        for variant, transform in variants:
            if variant == "calibrated" and budget_id not in budget_ids_for_cal:
                continue
            scored = transform(selected)
            for name, subset in slices(scored, spec.slice_by):
                metrics += metric_values(subset, spec, split, budget_id, variant, name)
            tables[f"reliability/{split}/{budget_id}/{variant}"] = reliability(
                scored, spec.ece_bins
            )

    failed = sum(i.failed for i in items)
    if failed:
        limitations.append(f"{failed} failed predictions were scored as uniform distributions.")

    analysis_id = f"{run.manifest.run_id}__{spec.spec_id}-r{spec.revision}"
    if calibration is not None:
        analysis_id += f"__{calibration.calibrator_id}"
    return AnalysisResult(
        analysis_id=analysis_id,
        kind="analyze",
        inputs={
            "run_id": run.manifest.run_id,
            "run_predictions_sha256": run.manifest.predictions.sha256,
            "predictor_id": predictor_id,
            "suite": run.manifest.suite,
            "protocol_sha256": run.manifest.protocol_sha256,
            "labels_sha256": run.manifest.labels_sha256,
            "calibration": calibration.model_dump(mode="json") if calibration else None,
        },
        specification=spec_raw,
        analysis_revision=ANALYSIS_REVISION,
        environment=environment_snapshot(),
        counts=counts(run, items, spec.splits),
        metrics=metrics,
        tables=tables,
        limitations=limitations,
        created_at=utc_now(),
    )


def write_result(result: AnalysisResult, out_root: Path) -> Path:
    """Analyses are new artifacts: an existing analysis directory is never overwritten."""
    out = out_root / result.analysis_id
    out.mkdir(parents=True, exist_ok=False)
    path = out / "analysis.json"
    path.write_text(json.dumps(result.model_dump(mode="json"), indent=2) + "\n")
    return path


def format_table(result: AnalysisResult) -> str:
    lines = [f"{result.analysis_id}"]
    for m in result.metrics:
        ci = f" [{m.ci_low:.4f}, {m.ci_high:.4f}]" if m.ci_low is not None else ""
        lines.append(
            f"  {m.split:<12} {m.budget_id:<6} {m.variant:<10} {m.slice:<22} "
            f"{m.metric:<16} {m.value:.4f}{ci}  n={m.n}"
        )
    return "\n".join(lines)
