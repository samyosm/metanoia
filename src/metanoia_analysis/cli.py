"""``metanoia-analysis``: calibrate -> analyze -> compare, from saved artifacts only."""

import argparse
import json
import sys
from pathlib import Path

from metanoia_contracts.artifacts import CalibrationArtifact
from metanoia_contracts.base import load_strict


def calibrate(args: argparse.Namespace) -> None:
    from metanoia_analysis import ANALYSIS_REVISION
    from metanoia_analysis.calibration import at_search_bound, fit_temperature, mean_nll
    from metanoia_analysis.load import align, load_labels, load_run
    from metanoia_contracts.provenance import sha256_text, utc_now

    run = load_run(Path(args.run))
    items = [
        i
        for i in align(run, load_labels(Path(args.dataset), run.manifest.labels_sha256))
        if i.split == args.split and i.budget_id == args.budget and not i.failed
    ]
    if not items:
        raise ValueError(
            f"no successful predictions for split {args.split!r}, budget {args.budget!r}"
        )
    budget = next(b.budget for b in run.protocol.budgets if b.id == args.budget)
    temperature = fit_temperature(items)
    source = f"{run.manifest.predictions.sha256}:{args.split}:{args.budget}"
    artifact = CalibrationArtifact(
        calibrator_id=f"temperature-{sha256_text(source)[:12]}",
        temperature=temperature,
        compatible_predictor_id=run.manifest.predictor["predictor_id"],
        budget_id=args.budget,
        budget=budget,
        fitted_on={
            "run_id": run.manifest.run_id,
            "predictions_sha256": run.manifest.predictions.sha256,
            "split": args.split,
            "n_examples": len(items),
            "nll_at_temperature_1": mean_nll(items, 1.0),
            "nll_at_fitted_temperature": mean_nll(items, temperature),
            "at_search_bound": at_search_bound(temperature),
        },
        analysis_revision=ANALYSIS_REVISION,
        created_at=utc_now(),
    )
    out = Path(args.out or f"artifacts/calibrations/{artifact.calibrator_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    out.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2) + "\n")
    print(f"wrote {out}: temperature {temperature:.4f} on {len(items)} examples")
    if at_search_bound(temperature):
        print("warning: the fitted temperature lies on its search bound", file=sys.stderr)


def analyze(args: argparse.Namespace) -> None:
    from metanoia_analysis.report import analyze as run_analysis
    from metanoia_analysis.report import format_table, write_result
    from metanoia_analysis.spec import load_spec

    spec, raw = load_spec(Path(args.spec))
    calibration = None
    if args.calibration:
        calibration = load_strict(
            CalibrationArtifact, json.loads(Path(args.calibration).read_text())
        )
    result = run_analysis(Path(args.run), Path(args.dataset), spec, raw, calibration)
    path = write_result(result, Path(args.out_root))
    if args.figures:
        from metanoia_analysis.plots import reliability_diagram

        for name, table in result.tables.items():
            if name.startswith("reliability/"):
                figure = path.parent / (name.replace("/", "_") + ".png")
                reliability_diagram(table, name.removeprefix("reliability/"), figure)
    print(format_table(result))
    print(f"wrote {path}")


def compare(args: argparse.Namespace) -> None:
    from metanoia_analysis.compare import compare as run_compare
    from metanoia_analysis.report import format_table, write_result
    from metanoia_analysis.spec import load_spec

    spec, raw = load_spec(Path(args.spec))
    result = run_compare(
        Path(args.run_a),
        Path(args.run_b),
        Path(args.dataset),
        spec,
        raw,
        accept_mismatches=tuple(args.accept_mismatch),
    )
    path = write_result(result, Path(args.out_root))
    print(format_table(result))
    print(f"wrote {path}")


def select(args: argparse.Namespace) -> None:
    from metanoia_analysis.select import select as run_select

    artifact = run_select(
        [Path(r) for r in args.run], Path(args.dataset), args.split, args.budget, args.metric
    )
    out = Path(args.out or f"artifacts/selections/{artifact.selection_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    out.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2) + "\n")
    for c in artifact.candidates:
        mark = "*" if c == artifact.chosen else " "
        print(f"{mark} {c.run_id:<60} {artifact.metric} {c.value:.4f}  n={c.n}")
    print(f"chose {artifact.chosen.predictor_id} ({artifact.chosen.predictor_path})")
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="metanoia-analysis", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("calibrate", help="fit a temperature on calibration predictions")
    p.add_argument("--run", required=True)
    p.add_argument("--dataset", required=True, help="dataset bundle holding the labels")
    p.add_argument("--split", default="calibration")
    p.add_argument("--budget", required=True)
    p.add_argument("--out")
    p.set_defaults(func=calibrate)

    p = commands.add_parser("analyze", help="score a run against its labels")
    p.add_argument("--run", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--calibration")
    p.add_argument("--out-root", default="artifacts/analyses")
    p.add_argument("--figures", action="store_true", help="also write reliability diagrams")
    p.set_defaults(func=analyze)

    p = commands.add_parser("compare", help="paired comparison of two compatible runs")
    p.add_argument("--run-a", required=True)
    p.add_argument("--run-b", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--accept-mismatch", action="append", default=[])
    p.add_argument("--out-root", default="artifacts/analyses")
    p.set_defaults(func=compare)

    p = commands.add_parser("select", help="choose a checkpoint from saved dev predictions")
    p.add_argument("--run", action="append", required=True, help="repeat once per candidate")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--budget", required=True)
    p.add_argument("--metric", default="nll", choices=["nll", "brier", "accuracy"])
    p.add_argument("--out")
    p.set_defaults(func=select)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
