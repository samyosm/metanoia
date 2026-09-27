"""Analysis of stored fixture runs: no model code is involved."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from metanoia_analysis.compare import compare
from metanoia_analysis.load import load_labels, load_run
from metanoia_analysis.report import analyze
from metanoia_analysis.spec import load_spec


def metric(result, split: str, name: str, slice_: str = "all") -> float:
    [value] = [
        m
        for m in result.metrics
        if m.split == split and m.metric == name and m.slice == slice_ and m.variant == "raw"
    ]
    return value.value


def test_oracle_and_lexical_scores(fixtures: Path) -> None:
    spec, raw = load_spec(fixtures / "spec.toml")
    oracle = analyze(fixtures / "runs/oracle", fixtures / "dataset", spec, raw)
    lexical = analyze(fixtures / "runs/lexical", fixtures / "dataset", spec, raw)
    for split in spec.splits:
        assert metric(oracle, split, "accuracy") == 1.0
        # Every candidate is mentioned equally often, so the fixture ties everywhere.
        assert metric(lexical, split, "accuracy") == pytest.approx(0.25)
        assert metric(lexical, split, "nll") == pytest.approx(1.3862943611198906)
    assert oracle.counts["test"]["failed"] == 0


def test_comparison_is_paired_and_directional(fixtures: Path) -> None:
    spec, raw = load_spec(fixtures / "spec.toml")
    result = compare(
        fixtures / "runs/lexical", fixtures / "runs/oracle", fixtures / "dataset", spec, raw
    )
    diffs = {(m.split, m.metric): m.value for m in result.metrics}
    assert diffs[("test", "accuracy_diff")] == pytest.approx(0.75)
    assert diffs[("test", "wrong_to_right")] == 1.0  # lexical is never strictly correct
    assert diffs[("test", "right_to_wrong")] == 0.0


def test_tampered_predictions_are_rejected(fixtures: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    shutil.copytree(fixtures / "runs/oracle", run)
    manifest = json.loads((run / "run.json").read_text())
    manifest["predictions"]["sha256"] = "0" * 64
    (run / "run.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash"):
        load_run(run)


def test_labels_must_belong_to_the_run(fixtures: Path) -> None:
    with pytest.raises(ValueError, match="not the labels"):
        load_labels(fixtures / "dataset", "0" * 64)


def test_incompatible_runs_are_not_compared(fixtures: Path, tmp_path: Path) -> None:
    run = tmp_path / "other"
    shutil.copytree(fixtures / "runs/lexical", run)
    manifest = json.loads((run / "run.json").read_text())
    manifest["protocol_sha256"] = "f" * 64
    (run / "run.json").write_text(json.dumps(manifest))
    spec, raw = load_spec(fixtures / "spec.toml")
    with pytest.raises(ValueError, match="not comparable"):
        compare(fixtures / "runs/oracle", run, fixtures / "dataset", spec, raw)


BLOCK_ML = """
import sys
BLOCKED = {'torch', 'transformers', 'faiss', 'metanoia', 'metanoia_research'}
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise ImportError(f'blocked: {name}')
sys.meta_path.insert(0, Block())
from metanoia_analysis.cli import main
main(sys.argv[1:])
"""


def test_analysis_cli_runs_without_ml_dependencies(fixtures: Path, tmp_path: Path) -> None:
    args = [
        "analyze",
        "--run",
        str(fixtures / "runs/oracle"),
        "--dataset",
        str(fixtures / "dataset"),
        "--spec",
        str(fixtures / "spec.toml"),
        "--out-root",
        str(tmp_path),
    ]
    out = subprocess.run([sys.executable, "-c", BLOCK_ML, *args], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert list(tmp_path.glob("*/analysis.json"))
