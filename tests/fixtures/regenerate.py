"""Rebuild the stored fixtures: a tiny dataset, suite, and two saved runs.

Analysis tests read these files without importing model code, so a change to a
predictor cannot silently redefine what a metric test means. Regenerate only on a
deliberate contract change:  uv run python tests/fixtures/regenerate.py
"""

import shutil
from pathlib import Path

from metanoia_research.data.bundle import write_dataset
from metanoia_research.data.microworld import GENERATOR_REVISION, MicroworldConfig, generate
from metanoia_research.runner import CollectionConfig, PredictorSpec, collect

# Relative, so recorded paths do not embed a machine-specific location. Run from the root.
HERE = Path("tests/fixtures")
CONFIG = MicroworldConfig(
    dataset_id="microworld_tiny",
    seed=7,
    n_worlds=10,
    examples_per_world=3,
    depths=(1, 2),
    heldout_depths=(3,),
    conjunction_probability=0.4,
    splits={"train": 0.4, "dev": 0.2, "calibration": 0.2, "test": 0.2},
)

PROTOCOL = """suite_id = "tiny"
revision = 1
task = "exclusive_choice"
question = "Fixture suite for tests."
analysis_specification = "tests/fixtures/spec.toml"

[dataset]
dataset_id = "microworld_tiny"
cases_sha256 = "{cases}"
labels_sha256 = "{labels}"

[evaluation]
splits = ["dev", "calibration", "test"]
batch_size = 4
repeats = 1
base_seed = 0
trace = "off"
failure_policy = "uniform"

[[budgets]]
id = "b0"
"""


def main() -> None:
    for name in ("dataset", "suite", "runs"):
        shutil.rmtree(HERE / name, ignore_errors=True)
    cases, labels, proofs = generate(CONFIG)
    manifest = write_dataset(
        HERE / "dataset",
        CONFIG.dataset_id,
        {"revision": GENERATOR_REVISION, "config": CONFIG.model_dump(mode="json")},
        cases,
        labels,
        proofs,
    )
    (HERE / "suite").mkdir()
    (HERE / "suite" / "protocol.toml").write_text(
        PROTOCOL.format(cases=manifest.cases.sha256, labels=manifest.labels.sha256)
    )
    for kind in ("oracle", "lexical"):
        collect(
            CollectionConfig(
                run_id=kind,
                suite=str(HERE / "suite"),
                dataset=str(HERE / "dataset"),
                predictor=PredictorSpec(kind=kind),
                output_root=str(HERE / "runs"),
            )
        )


if __name__ == "__main__":
    main()
