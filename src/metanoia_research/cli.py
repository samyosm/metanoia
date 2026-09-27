"""``metanoia-research``: prepare -> train -> collect.

Thin wrappers around library functions. Paths in configuration files are relative to
the current directory; run commands from the project root.
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

from metanoia_contracts.base import load_strict


def _toml(path: str) -> dict:
    return tomllib.loads(Path(path).read_text())


def prepare(args: argparse.Namespace) -> None:
    from metanoia_research.data.bundle import write_dataset
    from metanoia_research.data.microworld import GENERATOR_REVISION, MicroworldConfig, generate

    config = load_strict(MicroworldConfig, _toml(args.config))
    cases, labels, proofs = generate(config)
    out = Path(args.out or f"artifacts/datasets/{config.dataset_id}")
    manifest = write_dataset(
        out,
        config.dataset_id,
        {"revision": GENERATOR_REVISION, "config": config.model_dump(mode="json")},
        cases,
        labels,
        proofs,
    )
    print(f"wrote {out}: {manifest.splits}")
    print(
        json.dumps(
            {
                "dataset_id": manifest.dataset_id,
                "cases_sha256": manifest.cases.sha256,
                "labels_sha256": manifest.labels.sha256,
            },
            indent=2,
        )
    )


def train(args: argparse.Namespace) -> None:
    from metanoia.config import ModelConfig
    from metanoia_research.training.config import TrainingConfig
    from metanoia_research.training.loop import train as run_training

    config = load_strict(TrainingConfig, _toml(args.config))
    model_config = load_strict(ModelConfig, _toml(config.model_config_path))
    out = Path(args.out or f"artifacts/predictors/{config.name}")
    manifest = run_training(
        config, model_config, out, on_epoch=lambda record: print(json.dumps(record), flush=True)
    )
    print(f"wrote {out}: {manifest.predictor_id}")


def collect(args: argparse.Namespace) -> None:
    from metanoia_research.runner import CollectionConfig
    from metanoia_research.runner import collect as run_collection

    raw = _toml(args.config)
    if args.run_id:
        raw["run_id"] = args.run_id
    run_dir = run_collection(load_strict(CollectionConfig, raw))
    manifest = json.loads((run_dir / "run.json").read_text())
    print(f"wrote {run_dir}: {manifest['counts']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="metanoia-research", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("prepare", help="generate a versioned dataset bundle")
    p.add_argument("--config", required=True)
    p.add_argument("--out", help="default: artifacts/datasets/<dataset_id>")
    p.set_defaults(func=prepare)

    p = commands.add_parser("train", help="train a predictor and write a bundle")
    p.add_argument("--config", required=True)
    p.add_argument("--out", help="default: artifacts/predictors/<name>")
    p.set_defaults(func=train)

    p = commands.add_parser("collect", help="run a suite with a predictor and save predictions")
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", help="override the run_id in the configuration")
    p.set_defaults(func=collect)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
