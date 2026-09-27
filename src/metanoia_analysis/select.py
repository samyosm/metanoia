"""Choose one predictor (typically a training checkpoint) from saved predictions.

Selection reads only the named split, which must not be a final-evaluation split. All
candidate runs must cover exactly the same examples under the same labels, so the
choice is a paired comparison of point estimates. Ties keep the earliest candidate in
the given order.
"""

from collections.abc import Sequence
from pathlib import Path

from metanoia_analysis import ANALYSIS_REVISION
from metanoia_analysis.load import align, load_labels, load_run
from metanoia_analysis.metrics import per_example
from metanoia_contracts.artifacts import SelectionArtifact, SelectionCandidate
from metanoia_contracts.provenance import sha256_text, utc_now

RULES = {"nll": "min", "brier": "min", "accuracy": "max"}
FORBIDDEN_SPLITS = ("test", "calibration")


def select(
    run_dirs: Sequence[Path],
    dataset_dir: Path,
    split: str,
    budget_id: str,
    metric: str,
    nll_floor: float = 1e-15,
) -> SelectionArtifact:
    if split in FORBIDDEN_SPLITS:
        raise ValueError(f"selection on the {split!r} split is not allowed")
    if metric not in RULES:
        raise ValueError(f"unsupported selection metric {metric!r}; use one of {sorted(RULES)}")
    if not run_dirs:
        raise ValueError("no runs to select from")
    candidates, example_sets, labels_sha = [], [], None
    for run_dir in run_dirs:
        run = load_run(run_dir)
        if labels_sha is None:
            labels_sha = run.manifest.labels_sha256
        elif run.manifest.labels_sha256 != labels_sha:
            raise ValueError(f"{run_dir} was collected against different labels")
        items = [
            i
            for i in align(run, load_labels(dataset_dir, run.manifest.labels_sha256))
            if i.split == split and i.budget_id == budget_id
        ]
        if not items:
            raise ValueError(f"{run_dir} has no predictions for {split!r}/{budget_id!r}")
        example_sets.append(sorted((i.example_id, i.repeat_id) for i in items))
        values = per_example(items, metric, nll_floor)
        spec = run.manifest.predictor.get("spec", {})
        candidates.append(
            SelectionCandidate(
                run_id=run.manifest.run_id,
                predictor_id=run.manifest.predictor["predictor_id"],
                predictor_path=spec.get("path"),
                predictions_sha256=run.manifest.predictions.sha256,
                value=float(values.mean()),
                n=len(items),
            )
        )
    if any(s != example_sets[0] for s in example_sets[1:]):
        raise ValueError("candidate runs do not cover the same examples")
    rule = RULES[metric]
    best = candidates[0]
    for candidate in candidates[1:]:
        better = candidate.value < best.value if rule == "min" else candidate.value > best.value
        if better:
            best = candidate
    source = ":".join([split, budget_id, metric, *(c.predictions_sha256 for c in candidates)])
    assert labels_sha is not None
    return SelectionArtifact(
        selection_id=f"select-{sha256_text(source)[:12]}",
        split=split,
        budget_id=budget_id,
        metric=metric,
        rule=rule,  # type: ignore[arg-type]
        labels_sha256=labels_sha,
        candidates=tuple(candidates),
        chosen=best,
        analysis_revision=ANALYSIS_REVISION,
        created_at=utc_now(),
    )
