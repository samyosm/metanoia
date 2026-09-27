"""Collection runner behavior and a train -> collect smoke run on a tiny dataset."""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest

from metanoia_contracts import DecisionRequest, DecisionResult
from metanoia_research.runner import CollectionConfig, PredictorSpec, collect


def config(
    fixtures: Path, tmp_path: Path, run_id: str, kind: str = "lexical", **spec
) -> CollectionConfig:
    return CollectionConfig(
        run_id=run_id,
        suite=str(fixtures / "suite"),
        dataset=str(fixtures / "dataset"),
        predictor=PredictorSpec(kind=kind, **spec),
        output_root=str(tmp_path),
    )


def test_run_is_complete_and_never_overwritten(fixtures: Path, tmp_path: Path) -> None:
    run_dir = collect(config(fixtures, tmp_path, "r1"))
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["counts"] == {"expected": 18, "attempted": 18, "completed": 18, "failed": 0}
    with pytest.raises(FileExistsError):
        collect(config(fixtures, tmp_path, "r1"))
    assert not list(tmp_path.glob("*.incomplete"))


def test_dataset_must_match_suite_pin(fixtures: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    shutil.copytree(fixtures / "dataset", dataset)
    text = (dataset / "cases.jsonl").read_text().replace("Which", "What", 1)
    (dataset / "cases.jsonl").write_text(text)
    bad = config(fixtures, tmp_path, "r2").model_copy(update={"dataset": str(dataset)})
    with pytest.raises(ValueError, match="hash"):
        collect(bad)


class FlakyPredictor:
    """Fails on every other batch; returns the lexical fixture's answers otherwise."""

    predictor_id = "flaky@1"

    def __init__(self) -> None:
        self.calls = 0

    def describe(self) -> dict:
        return {"predictor_id": self.predictor_id}

    def decide(self, request: DecisionRequest) -> DecisionResult:
        return self.decide_batch([request])[0]

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResult]:
        from metanoia.baselines.lexical import LexicalOverlapPredictor

        self.calls += 1
        if self.calls % 2 == 0:
            raise RuntimeError("simulated failure")
        return [
            r.model_copy(update={"predictor_id": self.predictor_id})
            for r in LexicalOverlapPredictor().decide_batch(requests)
        ]


def test_failures_are_retained_not_dropped(fixtures: Path, tmp_path: Path) -> None:
    run_dir = collect(config(fixtures, tmp_path, "flaky"), predictor=FlakyPredictor())
    counts = json.loads((run_dir / "run.json").read_text())["counts"]
    assert counts["attempted"] == counts["expected"] == 18
    assert counts["failed"] > 0 and counts["completed"] + counts["failed"] == 18


def test_runner_never_passes_labels(fixtures: Path, tmp_path: Path) -> None:
    seen: list[DecisionRequest] = []

    class Spy(FlakyPredictor):
        def decide_batch(self, requests):
            seen.extend(requests)
            self.calls = 0
            return super().decide_batch(requests)

    collect(config(fixtures, tmp_path, "spy"), predictor=Spy())
    assert seen and all(set(r.model_dump()) == set(DecisionRequest.model_fields) for r in seen)
    assert all(r.request_id and r.seed is not None for r in seen)


@pytest.mark.slow
def test_train_then_collect(fixtures: Path, tmp_path: Path) -> None:
    from metanoia.config import HashingEncoderConfig, ModelConfig
    from metanoia_research.training.config import TrainingConfig
    from metanoia_research.training.loop import train

    model_config = ModelConfig(name="tiny", encoder=HashingEncoderConfig(buckets=128, dim=16))
    training = TrainingConfig(
        name="tiny",
        model_config_path="unused",
        dataset=str(fixtures / "dataset"),
        epochs=2,
        batch_size=4,
        seed=0,
    )
    manifest = train(training, model_config, tmp_path / "bundle")
    log = [
        json.loads(line)
        for line in (tmp_path / "bundle" / "training_log.jsonl").read_text().splitlines()
    ]
    assert len(log) == 2 and log[0]["grad_norm_scorer"] > 0
    run_dir = collect(
        config(fixtures, tmp_path, "trained", kind="bundle", path=str(tmp_path / "bundle"))
    )
    run = json.loads((run_dir / "run.json").read_text())
    assert run["predictor"]["predictor_id"] == manifest.predictor_id
    assert run["counts"]["failed"] == 0
