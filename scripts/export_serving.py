"""Copy only what is needed to *serve* trained models into ``serving/artifacts``.

Training bundles hold optimizer state and every checkpoint (1-5 GB each). Serving needs the
weights, the manifest, matching calibrations and, for expansion models, the memory.

    python scripts/export_serving.py                      # the two default models
    python scripts/export_serving.py NAME [NAME ...]      # bundles under artifacts/predictors
    python scripts/export_serving.py --out DIR NAME ...

Standard library only. Run from the project root.
"""

import argparse
import json
import shutil
import tomllib
from pathlib import Path

DEFAULT = ["mbft_joint_mcqa_v0", "branch_v2_focused_knowledge_v0"]


def served_dir(artifacts: Path, name: str, cards: dict) -> Path:
    """The directory the website loads: the card's pinned checkpoint, else the bundle."""
    base = artifacts / "predictors" / name
    pinned = cards.get(name, {}).get("checkpoint")
    return base / "checkpoints" / pinned if pinned else base


def copy_bundle(source: Path, target: Path) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "bundle.json").read_text())
    shutil.copy2(source / "bundle.json", target / "bundle.json")
    shutil.copy2(source / manifest["weights"]["path"], target / manifest["weights"]["path"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="*", default=DEFAULT)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="serving/artifacts")
    args = parser.parse_args()
    artifacts, out = Path(args.artifacts), Path(args.out)
    cards = tomllib.loads(Path("model_cards.toml").read_text())["models"]

    predictor_ids: set[str] = set()
    memories: set[str] = set()
    for name in args.names:
        source = served_dir(artifacts, name, cards)
        relative = source.relative_to(artifacts / "predictors")
        manifest = copy_bundle(source, out / "predictors" / relative)
        predictor_ids.add(manifest["predictor_id"])
        model = manifest["resolved_config"]["model"]
        for mechanism in ("branch", "expansion"):
            if model.get(mechanism):
                memories.add(model[mechanism]["memory"])
        print(f"{name}: {manifest['predictor_id']}")

    for file in sorted((artifacts / "calibrations").glob("*.json")):
        if json.loads(file.read_text()).get("compatible_predictor_id") in predictor_ids:
            (out / "calibrations").mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, out / "calibrations" / file.name)
            print(f"calibration: {file.name}")
    for memory in sorted(memories):  # paths are relative to the project root
        target = out / Path(memory).relative_to("artifacts")
        shutil.copytree(memory, target, dirs_exist_ok=True)
        print(f"memory: {memory}")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out} ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
