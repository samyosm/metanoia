"""Checkpoints are loadable bundles; an interrupted run resumes to the same weights."""

import json
from pathlib import Path

import pytest

from metanoia.api import load_predictor
from metanoia.config import HashingEncoderConfig, ModelConfig
from metanoia_research.training.config import TrainingConfig
from metanoia_research.training.loop import train

MODEL = ModelConfig(name="tiny", encoder=HashingEncoderConfig(buckets=128, dim=16))


def config(fixtures: Path, **overrides) -> TrainingConfig:
    values = {
        "name": "tiny",
        "model_config_path": "unused",
        "dataset": str(fixtures / "dataset"),
        "epochs": 2,
        "batch_size": 4,  # 12 train examples: 3 steps per epoch
        "seed": 0,
        "checkpoint_every_steps": 2,
    }
    return TrainingConfig(**{**values, **overrides})


class Interrupt(Exception):
    pass


def test_checkpoints_are_bundles_and_only_latest_keeps_state(fixtures, tmp_path) -> None:
    manifest = train(config(fixtures), MODEL, tmp_path / "b")
    root = tmp_path / "b" / "checkpoints"
    steps = sorted(p.name for p in root.glob("step_*"))
    assert steps == ["step_0000002", "step_0000004"]
    assert (root / "latest").read_text().strip() == "step_0000004"
    assert not (root / "step_0000002" / "train_state.pt").exists()
    assert (root / "step_0000004" / "train_state.pt").exists()
    first = load_predictor(root / "step_0000002")
    assert first.predictor_id != manifest.predictor_id
    recorded = json.loads((root / "step_0000002" / "bundle.json").read_text())["training"]
    assert recorded["checkpoint"] == {"step": 2, "epoch": 0, "position": 8}
    assert manifest.training["steps"] == 6 and manifest.training["wall_time_seconds"] >= 0


def test_interrupted_run_resumes_to_identical_weights(fixtures, tmp_path) -> None:
    straight = train(config(fixtures), MODEL, tmp_path / "straight")

    def stop_after_first_epoch(record: dict) -> None:
        if record["epoch"] == 0:
            raise Interrupt

    with pytest.raises(Interrupt):
        train(config(fixtures), MODEL, tmp_path / "r", on_epoch=stop_after_first_epoch)
    assert (tmp_path / "r.incomplete" / "checkpoints" / "latest").exists()
    resumed = train(config(fixtures), MODEL, tmp_path / "r", resume=True)
    assert resumed.predictor_id == straight.predictor_id
    log = (tmp_path / "r" / "training_log.jsonl").read_text().splitlines()
    assert [json.loads(line)["epoch"] for line in log] == [0, 1]


def test_resume_refuses_a_changed_config(fixtures, tmp_path) -> None:
    def stop(record: dict) -> None:
        raise Interrupt

    with pytest.raises(Interrupt):
        train(config(fixtures), MODEL, tmp_path / "r", on_epoch=stop)
    with pytest.raises(ValueError, match="training configuration"):
        train(config(fixtures, learning_rate=0.5), MODEL, tmp_path / "r", resume=True)


def test_failure_without_checkpoint_discards_work(fixtures, tmp_path) -> None:
    def stop(record: dict) -> None:
        raise Interrupt

    with pytest.raises(Interrupt):
        train(config(fixtures, checkpoint_every_steps=None), MODEL, tmp_path / "x", on_epoch=stop)
    assert not (tmp_path / "x.incomplete").exists()


def test_dev_only_collection_and_selection(fixtures, tmp_path) -> None:
    from metanoia_analysis.select import select
    from metanoia_research.cli import main

    train(config(fixtures), MODEL, tmp_path / "b")
    runs = tmp_path / "runs"
    main(
        [
            "collect-checkpoints",
            "--bundle",
            str(tmp_path / "b"),
            "--suite",
            str(fixtures / "suite"),
            "--dataset",
            str(fixtures / "dataset"),
            "--output-root",
            str(runs),
        ]
    )
    names = sorted(p.name for p in runs.iterdir())
    assert names == ["b__dev__final", "b__dev__step_0000002", "b__dev__step_0000004"]
    manifest = json.loads((runs / "b__dev__final" / "run.json").read_text())
    assert manifest["suite"]["splits"] == ["dev"] and manifest["counts"]["expected"] == 6

    artifact = select([runs / n for n in names], fixtures / "dataset", "dev", "b0", "nll")
    assert artifact.chosen.value == min(c.value for c in artifact.candidates)
    assert artifact.chosen.predictor_path is not None
    load_predictor(Path(artifact.chosen.predictor_path))
    with pytest.raises(ValueError, match="not allowed"):
        select([runs / names[0]], fixtures / "dataset", "test", "b0", "nll")


def test_init_from_starts_at_the_source_weights(fixtures, tmp_path) -> None:
    from metanoia_contracts import Candidate, DecisionRequest

    train(config(fixtures), MODEL, tmp_path / "a")
    warm = config(fixtures, init_from=str(tmp_path / "a"), learning_rate=1e-12, epochs=1)
    manifest = train(warm, MODEL, tmp_path / "b")
    assert manifest.training["initialized_from"] == load_predictor(tmp_path / "a").predictor_id
    request = DecisionRequest(
        context="Alpha is bright.",
        question="Which one is bright?",
        candidates=(Candidate(id="a", text="Alpha"), Candidate(id="b", text="Beta")),
    )
    before = load_predictor(tmp_path / "a").decide(request).logits
    after = load_predictor(tmp_path / "b").decide(request).logits
    assert after == pytest.approx(before, abs=1e-6)

    other = ModelConfig(name="tiny", encoder=HashingEncoderConfig(buckets=64, dim=16))
    with pytest.raises(ValueError, match="model configuration"):
        train(warm, other, tmp_path / "c")
