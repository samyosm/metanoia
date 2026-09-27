"""Stage-1 training: the direct candidate scorer with a proper prediction loss.

Only the negative log-likelihood of the correct candidate is optimized. The dev loss
logged here is a debugging aid; official checkpoint comparisons use collected dev
predictions scored by ``metanoia_analysis`` (PLAN.md §7.2).
"""

import json
import math
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from metanoia.api import BUNDLE_MANIFEST
from metanoia.build import build_model
from metanoia.config import ModelConfig
from metanoia.encoding.inputs import RENDERING_REVISION
from metanoia.model import DirectDecisionModel
from metanoia_contracts import DecisionRequest
from metanoia_contracts.artifacts import DatasetManifest, FileRef, PredictorBundleManifest
from metanoia_contracts.provenance import (
    canonical_json,
    environment_snapshot,
    sha256_file,
    source_revision,
    utc_now,
)
from metanoia_research.artifacts import (
    accelerator_snapshot,
    discard_incomplete,
    finalize_directory,
    new_incomplete_directory,
)
from metanoia_research.data.bundle import read_cases, read_labels
from metanoia_research.training.config import TrainingConfig

Example = tuple[DecisionRequest, int]


def load_examples(
    dataset_dir: Path,
    split: str,
    limit: int | None = None,
    subset: str = "head",
    seed: int = 0,
) -> list[Example]:
    _, cases = read_cases(dataset_dir)
    labels = {label.example_id: label.label_id for label in read_labels(dataset_dir)}
    examples = [
        (case.request, case.request.candidate_ids.index(labels[case.example_id]))
        for case in cases
        if case.split == split
    ]
    if not examples:
        raise ValueError(f"split {split!r} is empty in {dataset_dir}")
    if limit and subset == "random":
        return random.Random(f"{seed}:{split}").sample(examples, min(limit, len(examples)))
    return examples[:limit] if limit else examples


def batch_loss(
    model: DirectDecisionModel, batch: Sequence[Example], precision: str = "fp32"
) -> torch.Tensor:
    device_type = model.device.type
    with torch.autocast(device_type, dtype=torch.float16, enabled=precision == "fp16-mixed"):
        output = model([request for request, _ in batch])
    targets = torch.tensor([target for _, target in batch], device=output.logits.device)
    # The loss is computed in fp32 regardless of autocast.
    return F.cross_entropy(output.logits.float(), targets)


def mean_loss(
    model: DirectDecisionModel,
    examples: Sequence[Example],
    batch_size: int,
    precision: str = "fp32",
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            total += batch_loss(model, batch, precision).item() * len(batch)
    return total / len(examples)


def grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    norms = [p.grad.norm() for p in parameters if p.grad is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0


def _finite_mean(values: list[float]) -> float:
    """Mean over finite values: fp16 steps skipped by the grad scaler have inf norms."""
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def _lr_factor(config: TrainingConfig, step: int, total: int) -> float:
    warmup = int(config.warmup_fraction * total)
    if step < warmup:
        return (step + 1) / warmup
    if config.schedule == "linear":
        return max(0.0, (total - step) / max(1, total - warmup))
    return 1.0


def fit(
    model: DirectDecisionModel,
    train: Sequence[Example],
    dev: Sequence[Example],
    config: TrainingConfig,
    on_epoch: Callable[[dict[str, Any]], None] = lambda record: None,
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    rng = random.Random(config.seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    scorer_params = [p for p in model.scorer.parameters() if p.requires_grad]
    groups = [{"params": scorer_params, "lr": config.learning_rate}]
    if encoder_params:
        groups.append(
            {"params": encoder_params, "lr": config.encoder_learning_rate or config.learning_rate}
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler(model.device.type, enabled=config.precision == "fp16-mixed")
    micro = config.micro_batch_size or config.batch_size
    steps_per_epoch = -(-len(train) // config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    step = 0
    log = []
    for epoch in range(config.epochs):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses, encoder_norms, scorer_norms = [], [], []
        for start in range(0, len(order), config.batch_size):
            batch = [train[i] for i in order[start : start + config.batch_size]]
            for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base * _lr_factor(config, step, total_steps)
            optimizer.zero_grad()
            batch_total = 0.0
            for m in range(0, len(batch), micro):
                part = batch[m : m + micro]
                loss = batch_loss(model, part, config.precision) * (len(part) / len(batch))
                scaler.scale(loss).backward()
                batch_total += loss.item()
            scaler.unscale_(optimizer)
            encoder_norms.append(grad_norm(encoder_params))
            scorer_norms.append(grad_norm(scorer_params))
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(batch_total)
            step += 1
            if config.log_every_steps and step % config.log_every_steps == 0:
                recent = losses[-config.log_every_steps :]
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "of": total_steps,
                            "train_loss": sum(recent) / len(recent),
                        }
                    ),
                    flush=True,
                )
        record = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "dev_loss": mean_loss(model, dev, micro, config.precision),
            "grad_norm_encoder": _finite_mean(encoder_norms),
            "grad_norm_scorer": _finite_mean(scorer_norms),
            "nonfinite_grad_steps": sum(not math.isfinite(v) for v in scorer_norms + encoder_norms),
        }
        log.append(record)
        on_epoch(record)
    return log, optimizer


def train(
    config: TrainingConfig,
    model_config: ModelConfig,
    out_dir: Path,
    on_epoch: Callable[[dict[str, Any]], None] = lambda record: None,
) -> PredictorBundleManifest:
    """Train and write a predictor bundle. Refuses to overwrite ``out_dir``."""
    dataset_dir = Path(config.dataset)
    dataset_manifest, _ = read_cases(dataset_dir)
    train_examples = load_examples(
        dataset_dir, config.train_split, config.max_train_examples, config.subset, config.seed
    )
    dev_examples = load_examples(
        dataset_dir, config.dev_split, config.max_dev_examples, config.subset, config.seed
    )

    model = build_model(model_config, seed=config.seed).to(config.device)
    work = new_incomplete_directory(out_dir)
    try:
        manifest = _train_into(
            work,
            model,
            model_config,
            config,
            dataset_manifest,
            train_examples,
            dev_examples,
            on_epoch,
        )
    except BaseException:
        discard_incomplete(work)
        raise
    finalize_directory(work, out_dir)
    return manifest


def _train_into(
    work: Path,
    model: DirectDecisionModel,
    model_config: ModelConfig,
    config: TrainingConfig,
    dataset_manifest: DatasetManifest,
    train_examples: list[Example],
    dev_examples: list[Example],
    on_epoch: Callable[[dict[str, Any]], None],
) -> PredictorBundleManifest:
    log, optimizer = fit(model, train_examples, dev_examples, config, on_epoch)

    weights = work / "weights.pt"
    torch.save(model.persistent_state_dict(), weights)
    weights_sha = sha256_file(weights)
    # Resume state (not yet consumed by a resume command): optimizer, epoch, RNG.
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "epochs_completed": config.epochs,
            "torch_rng": torch.get_rng_state(),
        },
        work / "train_state.pt",
    )
    (work / "training_log.jsonl").write_text("".join(canonical_json(r) + "\n" for r in log))

    manifest = PredictorBundleManifest(
        predictor_id=f"{model_config.name}@{weights_sha[:12]}",
        resolved_config={
            "model": model_config.model_dump(mode="json"),
            "encoder_identity": model.encoder.identity(),
            "input_rendering": RENDERING_REVISION,
        },
        weights=FileRef(path="weights.pt", sha256=weights_sha, rows=0),
        training={
            "config": config.model_dump(mode="json"),
            "dataset_id": dataset_manifest.dataset_id,
            "cases_sha256": dataset_manifest.cases.sha256,
            "labels_sha256": dataset_manifest.labels.sha256,
            "train_examples": len(train_examples),
            "dev_examples": len(dev_examples),
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "final_epoch": log[-1],
        },
        environment=environment_snapshot(accelerator_snapshot(config.device)),
        source=source_revision(),
        created_at=utc_now(),
    )
    (work / BUNDLE_MANIFEST).write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n"
    )
    return manifest
