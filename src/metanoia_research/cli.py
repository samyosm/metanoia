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

    raw = _toml(args.config)
    generator = raw.get("generator", "microworld")
    if generator == "microworld":
        from metanoia_research.data import microworld

        config = load_strict(microworld.MicroworldConfig, raw)
        cases, labels, proofs = microworld.generate(config)
        info = {
            "revision": microworld.generator_revision(config),
            "config": config.model_dump(mode="json"),
        }
    elif generator == "mcqa":
        from metanoia_research.data import mcqa

        mc_config = load_strict(mcqa.McqaConfig, raw)
        cases, labels, skipped, proofs = mcqa.generate(mc_config)
        info = {
            "revision": mcqa.GENERATOR_REVISION,
            "config": mc_config.model_dump(mode="json"),
            "sources": mcqa.source_revisions(mc_config),
            "skipped": skipped,
        }
        config = mc_config  # type: ignore[assignment]
    else:
        raise ValueError(f"unknown generator {generator!r}")
    out = Path(args.out or f"artifacts/datasets/{config.dataset_id}")
    manifest = write_dataset(out, config.dataset_id, info, cases, labels, proofs)
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
        config,
        model_config,
        out,
        on_epoch=lambda record: print(json.dumps(record), flush=True),
        resume=args.resume,
    )
    print(f"wrote {out}: {manifest.predictor_id}")


def collect(args: argparse.Namespace) -> None:
    from metanoia_research.runner import CollectionConfig
    from metanoia_research.runner import collect as run_collection

    raw = _toml(args.config)
    if args.run_id:
        raw["run_id"] = args.run_id
    if args.from_selection:
        from metanoia_contracts.artifacts import SelectionArtifact

        selection = load_strict(
            SelectionArtifact, json.loads(Path(args.from_selection).read_text())
        )
        if selection.chosen.predictor_path is None:
            raise ValueError(f"{args.from_selection} names no predictor path")
        raw["predictor"]["path"] = selection.chosen.predictor_path
        print(f"collecting {selection.chosen.predictor_id} chosen by {selection.selection_id}")
    run_dir = run_collection(load_strict(CollectionConfig, raw))
    manifest = json.loads((run_dir / "run.json").read_text())
    print(f"wrote {run_dir}: {manifest['counts']}")


def collect_checkpoints(args: argparse.Namespace) -> None:
    """Collect every checkpoint of a bundle (and the final weights) on a few splits."""
    from metanoia_research.runner import CollectionConfig, PredictorSpec
    from metanoia_research.runner import collect as run_collection

    bundle = Path(args.bundle)
    targets = sorted((bundle / "checkpoints").glob("step_*"))
    targets = [t for t in targets if not t.name.endswith(".incomplete")] + [bundle]
    parts = [*args.splits, *(args.budgets or [])]
    prefix = args.run_prefix or f"{bundle.name}__{'-'.join(parts)}"
    for target in targets:
        label = target.name if target != bundle else "final"
        config = CollectionConfig(
            run_id=f"{prefix}__{label}",
            suite=args.suite,
            dataset=args.dataset,
            predictor=PredictorSpec(
                kind="bundle", path=str(target), device=args.device, precision=args.precision
            ),
            output_root=args.output_root,
            splits=tuple(args.splits),
            budgets=tuple(args.budgets) if args.budgets else None,
        )
        if (Path(config.output_root) / config.run_id).exists():
            print(f"skip {config.run_id}: already collected")
            continue
        run_dir = run_collection(config)
        print(f"wrote {run_dir}", flush=True)


def build_memory(args: argparse.Namespace) -> None:
    from metanoia_research.memory_build import MemoryConfig
    from metanoia_research.memory_build import build_memory as run_build

    config = load_strict(MemoryConfig, _toml(args.config))
    out = Path(args.out or f"artifacts/memories/{config.memory_id}")
    manifest = run_build(config, out)
    print(f"wrote {out}: {manifest.concepts.rows} concepts, dim {manifest.dim}")


def build_proposals(args: argparse.Namespace) -> None:
    from metanoia.config import ModelConfig
    from metanoia_research.proposals_build import build_proposals as run_build

    model_config = load_strict(ModelConfig, _toml(args.model_config))
    branch = model_config.branch
    out = Path(args.out or (branch.proposal_cache if branch and branch.proposal_cache else ""))
    if not str(out):
        raise ValueError("give --out, or set proposal_cache in the model config")
    manifest = run_build(model_config, Path(args.dataset), tuple(args.splits), out, args.device)
    print(f"wrote {out}: {manifest['entries']['rows']} orderings in {manifest['seconds']} s")


def memory_recall(args: argparse.Namespace) -> None:
    from metanoia.memory.encoder import RetrievalEncoder
    from metanoia.memory.store import ConceptMemory
    from metanoia_contracts.provenance import utc_now
    from metanoia_research.memory_eval import access_recall

    memory = ConceptMemory(Path(args.memory), device=args.device)
    spec = memory.manifest.encoder
    encoder = RetrievalEncoder(
        spec["model"],
        spec["requested_revision"],
        spec["pooling"],
        spec["max_length"],
        spec["query_prefix"],
    ).to(args.device)
    if encoder.identity() != spec:
        raise ValueError(f"retrieval encoder {encoder.identity()} differs from the memory's {spec}")
    report = access_recall(
        Path(args.dataset),
        memory,
        encoder,
        tuple(args.k),
        tuple(args.splits),
        args.seed,
        match=args.match,
        nltk_data=args.nltk_data,
    )
    report["created_at"] = utc_now()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for key, values in report["recall"].items():
        print(f"{key:>10}: retrieved {values['retrieved']:.3f}  random {values['random']:.3f}")
    print(f"wrote {out}")


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
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run from <out>.incomplete/checkpoints/latest",
    )
    p.set_defaults(func=train)

    p = commands.add_parser("collect", help="run a suite with a predictor and save predictions")
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", help="override the run_id in the configuration")
    p.add_argument(
        "--from-selection",
        help="a selection artifact (metanoia-analysis select); collect its chosen predictor",
    )
    p.set_defaults(func=collect)

    p = commands.add_parser(
        "collect-checkpoints", help="collect each checkpoint of a bundle, e.g. on dev only"
    )
    p.add_argument("--bundle", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--splits", nargs="+", default=["dev"])
    p.add_argument("--budgets", nargs="+", help="a subset of the protocol's budget ids")
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--run-prefix")
    p.add_argument("--output-root", default="artifacts/runs")
    p.set_defaults(func=collect_checkpoints)

    p = commands.add_parser("build-memory", help="build a versioned concept memory bundle")
    p.add_argument("--config", required=True)
    p.add_argument("--out", help="default: artifacts/memories/<memory_id>")
    p.set_defaults(func=build_memory)

    p = commands.add_parser("build-proposals", help="precompute fused proposal orderings")
    p.add_argument("--model-config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "dev", "calibration", "test"])
    p.add_argument("--out", help="default: the model config's proposal_cache")
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=build_proposals)

    p = commands.add_parser("memory-recall", help="access recall of a memory against gold facts")
    p.add_argument("--memory", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "dev"])
    p.add_argument("--k", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--match", choices=["names", "text"], default="names")
    p.add_argument("--nltk-data", default="artifacts/nltk_data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    p.set_defaults(func=memory_recall)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
