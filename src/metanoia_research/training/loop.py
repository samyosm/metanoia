"""Stage-1 training: the direct candidate scorer with a proper prediction loss.

Only the negative log-likelihood of the correct candidate is optimized. The dev loss
logged here is a debugging aid; official checkpoint comparisons use collected dev
predictions scored by ``metanoia_analysis`` (PLAN.md §7.2).
"""

import json
import math
import random
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from metanoia.api import BUNDLE_MANIFEST
from metanoia.build import build_model, check_encoder_identity
from metanoia.config import ModelConfig
from metanoia.encoding.inputs import RENDERING_REVISION
from metanoia.model import DirectDecisionModel
from metanoia_contracts import DecisionRequest
from metanoia_contracts.artifacts import DatasetManifest, FileRef, PredictorBundleManifest
from metanoia_contracts.base import load_strict
from metanoia_contracts.provenance import (
    canonical_json,
    environment_snapshot,
    sha256_file,
    source_revision,
    utc_now,
)
from metanoia_research.artifacts import (
    INCOMPLETE_SUFFIX,
    accelerator_snapshot,
    discard_incomplete,
    finalize_directory,
    new_incomplete_directory,
)
from metanoia_research.data.bundle import read_cases, read_labels
from metanoia_research.training.config import TrainingConfig

Example = tuple[DecisionRequest, int]


def _selected(
    dataset_dir: Path, split: str, limit: int | None, subset: str, seed: int
) -> list[tuple[DecisionRequest, int, str]]:
    _, cases = read_cases(dataset_dir)
    labels = {label.example_id: label for label in read_labels(dataset_dir)}
    examples = [
        (
            case.request,
            case.request.candidate_ids.index(labels[case.example_id].label_id),
            str(labels[case.example_id].annotations.get("source", "")),
        )
        for case in cases
        if case.split == split
    ]
    if not examples:
        raise ValueError(f"split {split!r} is empty in {dataset_dir}")
    if limit and subset == "random":
        return random.Random(f"{seed}:{split}").sample(examples, min(limit, len(examples)))
    return examples[:limit] if limit else examples


def load_examples(
    dataset_dir: Path,
    split: str,
    limit: int | None = None,
    subset: str = "head",
    seed: int = 0,
) -> list[Example]:
    return [(r, t) for r, t, _ in _selected(dataset_dir, split, limit, subset, seed)]


def load_sources(
    dataset_dir: Path,
    split: str,
    limit: int | None = None,
    subset: str = "head",
    seed: int = 0,
) -> list[str]:
    """The source of each example ``load_examples`` returns with the same arguments."""
    return [s for _, _, s in _selected(dataset_dir, split, limit, subset, seed)]


def epoch_order(
    size: int, sources: Sequence[str] | None, config: TrainingConfig, rng: random.Random
) -> list[int]:
    """One epoch's example order. Plain: a shuffled pass. Mixed: see ``mix_alpha``."""
    if config.mix_alpha is None:
        order = list(range(size))
        rng.shuffle(order)
        return order
    if sources is None or len(sources) != size:
        raise ValueError("mix_alpha needs the source of every training example")
    by_source: dict[str, list[int]] = {}
    for index, source in enumerate(sources):
        by_source.setdefault(source, []).append(index)
    total = config.epoch_examples or size
    weights = {s: len(ix) ** config.mix_alpha for s, ix in by_source.items()}
    norm = sum(weights.values())
    order = []
    for source, indices in sorted(by_source.items()):
        share = max(1, round(total * weights[source] / norm))
        pool: list[int] = []
        while len(pool) < share:
            cycle = indices[:]
            rng.shuffle(cycle)
            pool += cycle
        order += pool[:share]
    rng.shuffle(order)
    return order


def batch_loss(
    model: DirectDecisionModel,
    batch: Sequence[Example],
    precision: str = "fp32",
    rounds: int | None = None,
) -> torch.Tensor:
    """Cross-entropy of the correct candidate. With ``rounds`` (workspace models), the
    mean over every round's readout: deep supervision, so each budget is trained. For a
    branch model (decision 0012), ``rounds`` is the number of nodes per candidate."""
    output = _forward(model, batch, precision, rounds)
    targets = torch.tensor([target for _, target in batch], device=output.logits.device)
    # The loss is computed in fp32 regardless of autocast.
    stages = getattr(output, "round_logits", None) if rounds else None
    if stages:
        return torch.stack([F.cross_entropy(s.float(), targets) for s in stages]).mean()
    return F.cross_entropy(output.logits.float(), targets)


def _forward(
    model: DirectDecisionModel, batch: Sequence[Example], precision: str, rounds: int | None
) -> Any:
    requests = [request for request, _ in batch]
    with torch.autocast(model.device.type, dtype=torch.float16, enabled=precision == "fp16-mixed"):
        if hasattr(model, "branch_config"):
            return model(requests, width=rounds or 0)  # type: ignore[call-arg]
        if hasattr(model, "expansion_config"):
            # Training and dev evaluation always expand with the configured width.
            return model(requests, force=True)  # type: ignore[call-arg]
        if rounds is None:
            return model(requests)
        return model(requests, rounds=rounds)  # type: ignore[call-arg]


def evaluate(
    model: DirectDecisionModel,
    examples: Sequence[Example],
    batch_size: int,
    precision: str = "fp32",
    rounds: int | None = None,
) -> tuple[float, float]:
    """Mean loss and mean per-example standard deviation of candidate logits.

    A logit spread near 0 with a loss near ln N means the model has collapsed to
    uniform output, which is different from slow learning (PLAN.md §13, lesson 8).
    """
    model.eval()
    total, spread = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            output = _forward(model, batch, precision, rounds)
            logits = output.logits.float()
            targets = torch.tensor([t for _, t in batch], device=logits.device)
            total += F.cross_entropy(logits, targets).item() * len(batch)
            for row, mask in zip(logits, output.candidate_mask, strict=True):
                spread += row[mask].std().item()
    return total / len(examples), spread / len(examples)


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


@dataclass
class Progress:
    """Everything, besides weights and optimizer state, needed to continue training.

    ``order`` is the current epoch's example order (``None`` between epochs) and
    ``position`` the index of the next batch in it. ``python_rng`` is the shuffling RNG
    state after the current epoch's shuffle.
    """

    epoch: int = 0
    position: int = 0
    step: int = 0
    order: list[int] | None = None
    python_rng: Any = None
    losses: list[float] = field(default_factory=list)
    encoder_norms: list[float] = field(default_factory=list)
    scorer_norms: list[float] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    rounds_rng: Any = None  # state of the workspace-rounds RNG (workspace models only)
    workspace_norms: list[float] = field(default_factory=list)


# ``checkpoint(progress, optimizer, scaler)`` is called after a step or an epoch.
CheckpointFn = Callable[[Progress, torch.optim.Optimizer, torch.amp.GradScaler], None]


def fit(
    model: DirectDecisionModel,
    train: Sequence[Example],
    dev: Sequence[Example],
    config: TrainingConfig,
    on_epoch: Callable[[dict[str, Any]], None] = lambda record: None,
    resume: dict[str, Any] | None = None,
    checkpoint: CheckpointFn | None = None,
    train_sources: Sequence[str] | None = None,
) -> tuple[Progress, torch.optim.Optimizer, torch.amp.GradScaler]:
    """Train in place. ``resume`` is a saved train state (see :func:`train_state`)."""
    rng = random.Random(config.seed)
    # A separate RNG for workspace rounds, so the data order matches the direct baseline's.
    rounds_rng = random.Random(f"{config.seed}:rounds")
    eval_rounds = config.eval_rounds
    if config.train_rounds is not None and eval_rounds is None:
        eval_rounds = config.train_rounds[1]
    # A branch model samples its width per batch; dev loss uses the widest (decision 0012).
    branch = getattr(model, "branch_config", None)
    if branch is not None and eval_rounds is None:
        eval_rounds = max(branch.train_widths)
    trainable = [p for p in model.parameters() if p.requires_grad]
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    scorer_params = [p for p in model.scorer.parameters() if p.requires_grad]
    # Every trainable parameter outside the encoder (the scorer, and the workspace of a
    # Stage-3a model) uses the head learning rate. For a direct model this is exactly
    # the scorer's parameters.
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [p for p in trainable if id(p) not in encoder_ids]
    workspace_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and name.startswith("workspace.")
    ]
    groups = [{"params": head_params, "lr": config.learning_rate}]
    if encoder_params:
        groups.append(
            {"params": encoder_params, "lr": config.encoder_learning_rate or config.learning_rate}
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler(model.device.type, enabled=config.precision == "fp16-mixed")
    progress = Progress()
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        progress = Progress(**resume["progress"])
        rng.setstate(_as_rng_state(progress.python_rng))
        if progress.rounds_rng is not None:
            rounds_rng.setstate(_as_rng_state(progress.rounds_rng))
        torch.set_rng_state(resume["torch_rng"])
        if resume.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
    micro = config.micro_batch_size or config.batch_size
    epoch_size = len(train) if config.mix_alpha is None else (config.epoch_examples or len(train))
    steps_per_epoch = -(-epoch_size // config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    started = time.perf_counter() - progress.elapsed_seconds

    while progress.epoch < config.epochs:
        if progress.order is None:
            order = epoch_order(len(train), train_sources, config, rng)
            progress.order, progress.python_rng, progress.position = order, rng.getstate(), 0
        model.train()
        while progress.position < len(progress.order):
            indices = progress.order[progress.position : progress.position + config.batch_size]
            batch = [train[i] for i in indices]
            for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base * _lr_factor(config, progress.step, total_steps)
            optimizer.zero_grad()
            batch_total = 0.0
            rounds = None
            if config.train_rounds is not None:
                rounds = rounds_rng.randint(*config.train_rounds)
            elif branch is not None:
                rounds = rounds_rng.choice(branch.train_widths)
            for m in range(0, len(batch), micro):
                part = batch[m : m + micro]
                loss = batch_loss(model, part, config.precision, rounds) * (len(part) / len(batch))
                scaler.scale(loss).backward()
                batch_total += loss.item()
            scaler.unscale_(optimizer)
            progress.encoder_norms.append(grad_norm(encoder_params))
            progress.scorer_norms.append(grad_norm(scorer_params))
            if workspace_params:
                progress.workspace_norms.append(grad_norm(workspace_params))
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            progress.losses.append(batch_total)
            progress.rounds_rng = rounds_rng.getstate()
            progress.step += 1
            progress.position += config.batch_size
            if config.log_every_steps and progress.step % config.log_every_steps == 0:
                recent = progress.losses[-config.log_every_steps :]
                print(
                    json.dumps(
                        {
                            "epoch": progress.epoch,
                            "step": progress.step,
                            "of": total_steps,
                            "train_loss": sum(recent) / len(recent),
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
            mid_epoch = progress.position < len(progress.order)
            every = config.checkpoint_every_steps
            if checkpoint and mid_epoch and every and progress.step % every == 0:
                progress.elapsed_seconds = time.perf_counter() - started
                checkpoint(progress, optimizer, scaler)
        norms = progress.scorer_norms + progress.encoder_norms
        dev_loss, dev_spread = evaluate(model, dev, micro, config.precision, eval_rounds)
        record = {
            "epoch": progress.epoch,
            "step": progress.step,
            "train_loss": sum(progress.losses) / len(progress.losses),
            "dev_loss": dev_loss,
            "dev_logit_spread": dev_spread,
            "grad_norm_encoder": _finite_mean(progress.encoder_norms),
            "grad_norm_scorer": _finite_mean(progress.scorer_norms),
            **(
                {"grad_norm_workspace": _finite_mean(progress.workspace_norms)}
                if workspace_params
                else {}
            ),
            "nonfinite_grad_steps": sum(not math.isfinite(v) for v in norms),
            "elapsed_seconds": round(time.perf_counter() - started, 1),
        }
        progress.log.append(record)
        on_epoch(record)
        progress.epoch += 1
        progress.order, progress.position = None, 0
        progress.losses, progress.encoder_norms, progress.scorer_norms = [], [], []
        progress.workspace_norms = []
        progress.elapsed_seconds = time.perf_counter() - started
        wanted = config.checkpoint_each_epoch or (
            config.checkpoint_every_steps is not None
            and progress.step % config.checkpoint_every_steps == 0
        )
        if checkpoint and wanted and progress.epoch < config.epochs:
            checkpoint(progress, optimizer, scaler)
    progress.elapsed_seconds = time.perf_counter() - started
    return progress, optimizer, scaler


def _as_rng_state(state: Any) -> tuple[Any, ...]:
    """``random.Random`` states are nested tuples; serialization may turn them into lists."""
    return (state[0], tuple(state[1]), state[2])


def train_state(
    progress: Progress, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler
) -> dict[str, Any]:
    return {
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "progress": asdict(progress),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


CHECKPOINTS = "checkpoints"
LATEST = "latest"


def train(
    config: TrainingConfig,
    model_config: ModelConfig,
    out_dir: Path,
    on_epoch: Callable[[dict[str, Any]], None] = lambda record: None,
    resume: bool = False,
) -> PredictorBundleManifest:
    """Train and write a predictor bundle. Refuses to overwrite ``out_dir``.

    With checkpointing enabled, an interrupted run keeps ``<out_dir>.incomplete/`` and
    continues from its latest checkpoint when called again with ``resume=True``.
    """
    dataset_dir = Path(config.dataset)
    dataset_manifest, _ = read_cases(dataset_dir)
    train_examples = load_examples(
        dataset_dir, config.train_split, config.max_train_examples, config.subset, config.seed
    )
    train_sources = (
        load_sources(
            dataset_dir, config.train_split, config.max_train_examples, config.subset, config.seed
        )
        if config.mix_alpha is not None
        else None
    )
    dev_dir = Path(config.dev_dataset) if config.dev_dataset else dataset_dir
    dev_examples = load_examples(
        dev_dir, config.dev_split, config.max_dev_examples, config.subset, config.seed
    )

    if (model_config.workspace is None) != (config.train_rounds is None):
        raise ValueError("train_rounds must be set exactly when the model has a workspace")
    model = build_model(model_config, seed=config.seed)
    init_predictor = _load_initial_weights(model, model_config, config.init_from)
    model = model.to(config.device)
    state = None
    if resume:
        work, state = _load_resume_state(out_dir, model, model_config, config, dataset_manifest)
    else:
        work = new_incomplete_directory(out_dir)
    context = _BundleContext(
        model,
        model_config,
        config,
        dataset_manifest,
        len(train_examples),
        len(dev_examples),
        init_predictor,
    )

    def checkpoint(
        progress: Progress, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler
    ) -> None:
        _write_checkpoint(work, context, progress, optimizer, scaler)

    try:
        progress, optimizer, scaler = fit(
            model, train_examples, dev_examples, config, on_epoch, state, checkpoint, train_sources
        )
        manifest = context.write_bundle(work, progress, train_state(progress, optimizer, scaler))
    except BaseException:
        if (work / CHECKPOINTS / LATEST).exists():
            print(
                f"training interrupted; {work} is kept. Continue with --resume.",
                file=sys.stderr,
            )
        else:
            discard_incomplete(work)
        raise
    finalize_directory(work, out_dir)
    return manifest


def _weight_layout(model_config: dict[str, Any]) -> dict[str, Any]:
    """The parts of a model configuration that determine its weights.

    The name and gradient checkpointing never do. A pretrained Hugging Face encoder's
    ``max_length`` is only a truncation setting, whereas the hashing encoder sizes its
    position table by it, so there it stays part of the layout.
    """
    layout = json.loads(json.dumps(model_config))
    layout.pop("name", None)
    # Bundles written before workspaces or expansions existed lack those keys entirely.
    for optional in ("workspace", "expansion"):
        if layout.get(optional) is None:
            layout.pop(optional, None)
    # Branch reading adds no weights: it only changes the reader's input (decision 0012).
    layout.pop("branch", None)
    encoder = layout.get("encoder", {})
    encoder.pop("gradient_checkpointing", None)
    if encoder.get("kind") == "hf":
        encoder.pop("max_length", None)
    return layout


def _load_initial_weights(
    model: DirectDecisionModel, model_config: ModelConfig, init_from: str | None
) -> str | None:
    """Load a bundle's weights as the starting point; returns its predictor id."""
    if init_from is None:
        return None
    directory = Path(init_from)
    manifest = load_strict(
        PredictorBundleManifest, json.loads((directory / BUNDLE_MANIFEST).read_text())
    )
    source_layout = _weight_layout(manifest.resolved_config["model"])
    target_layout = _weight_layout(model_config.model_dump(mode="json"))
    # A workspace model may start from a direct bundle: its encoder and scorer come from
    # the bundle and only the workspace starts fresh (decision 0008).
    # A reasoning mechanism absent from the source starts fresh; everything else loads.
    fresh = [m for m in ("workspace", "expansion") if m not in source_layout and m in target_layout]
    for mechanism in fresh:
        del target_layout[mechanism]
    fresh_workspace = bool(fresh)
    if source_layout != target_layout:
        raise ValueError(f"model configuration differs from the one in {directory}")
    recorded = dict(manifest.resolved_config["encoder_identity"])
    current = dict(model.encoder.identity())
    if recorded.get("kind") == "hf":
        recorded.pop("max_length", None)
        current.pop("max_length", None)
    if recorded != current:
        raise ValueError(f"encoder identity mismatch: bundle {recorded}, loaded {current}")
    weights = directory / manifest.weights.path
    if sha256_file(weights) != manifest.weights.sha256:
        raise ValueError(f"weights hash mismatch in {directory}")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if fresh_workspace:
        result = model.load_state_dict(state, strict=False)
        external = tuple(f"encoder.{p}" for p in model.encoder.external_prefixes)
        fresh_prefixes = tuple(f"{m}." for m in fresh)
        missing = [k for k in result.missing_keys if not k.startswith((*fresh_prefixes, *external))]
        if missing or result.unexpected_keys:
            raise ValueError(
                f"incompatible weights: missing {missing}, unexpected {result.unexpected_keys}"
            )
    else:
        model.load_persistent_state_dict(state)
    return manifest.predictor_id


def _load_resume_state(
    out_dir: Path,
    model: DirectDecisionModel,
    model_config: ModelConfig,
    config: TrainingConfig,
    dataset_manifest: DatasetManifest,
) -> tuple[Path, dict[str, Any]]:
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} is already complete")
    work = out_dir.with_name(out_dir.name + INCOMPLETE_SUFFIX)
    latest = work / CHECKPOINTS / LATEST
    if not latest.exists():
        raise FileNotFoundError(f"no checkpoint to resume from in {work}")
    directory = work / CHECKPOINTS / latest.read_text().strip()
    manifest = load_strict(
        PredictorBundleManifest, json.loads((directory / BUNDLE_MANIFEST).read_text())
    )
    recorded = manifest.training
    if recorded["config"] != config.model_dump(mode="json"):
        raise ValueError("training configuration differs from the interrupted run")
    if manifest.resolved_config["model"] != model_config.model_dump(mode="json"):
        raise ValueError("model configuration differs from the interrupted run")
    if (recorded["cases_sha256"], recorded["labels_sha256"]) != (
        dataset_manifest.cases.sha256,
        dataset_manifest.labels.sha256,
    ):
        raise ValueError("dataset differs from the interrupted run")
    check_encoder_identity(model, manifest.resolved_config["encoder_identity"])
    weights = directory / manifest.weights.path
    if sha256_file(weights) != manifest.weights.sha256:
        raise ValueError(f"checkpoint weights hash mismatch in {directory}")
    model.load_persistent_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    state = torch.load(directory / TRAIN_STATE, map_location="cpu", weights_only=False)
    print(f"resuming from {directory} (step {state['progress']['step']})", file=sys.stderr)
    return work, state


TRAIN_STATE = "train_state.pt"


def _expansion_identity(model: DirectDecisionModel) -> dict[str, Any]:
    """The concept memory an expansion model depends on (checked at load time)."""
    identity = getattr(model, "expansion_identity", None)
    return {"expansion_identity": identity()} if callable(identity) else {}


@dataclass
class _BundleContext:
    model: DirectDecisionModel
    model_config: ModelConfig
    config: TrainingConfig
    dataset_manifest: DatasetManifest
    n_train: int
    n_dev: int
    init_predictor: str | None = None

    def write_bundle(
        self,
        directory: Path,
        progress: Progress,
        state: dict[str, Any],
        checkpoint: dict[str, Any] | None = None,
    ) -> PredictorBundleManifest:
        """Weights, train state, log and manifest: a loadable predictor bundle."""
        weights = directory / "weights.pt"
        torch.save(self.model.persistent_state_dict(), weights)
        weights_sha = sha256_file(weights)
        torch.save(state, directory / TRAIN_STATE)
        (directory / "training_log.jsonl").write_text(
            "".join(canonical_json(r) + "\n" for r in progress.log)
        )
        # Examples processed so far (the last batch of an epoch may be smaller).
        seen = progress.epoch * self.n_train + (
            min(progress.position, self.n_train) if progress.order is not None else 0
        )
        training: dict[str, Any] = {
            "config": self.config.model_dump(mode="json"),
            "dataset_id": self.dataset_manifest.dataset_id,
            "cases_sha256": self.dataset_manifest.cases.sha256,
            "labels_sha256": self.dataset_manifest.labels.sha256,
            "train_examples": self.n_train,
            "dev_examples": self.n_dev,
            "parameters_total": sum(p.numel() for p in self.model.parameters()),
            "parameters_trainable": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "steps": progress.step,
            "final_epoch": progress.log[-1] if progress.log else None,
            "wall_time_seconds": round(progress.elapsed_seconds, 1),
            "train_examples_per_second": round(seen / max(progress.elapsed_seconds, 1e-9), 2),
        }
        if self.init_predictor is not None:
            training["initialized_from"] = self.init_predictor
        if checkpoint is not None:
            training["checkpoint"] = checkpoint
        manifest = PredictorBundleManifest(
            predictor_id=f"{self.model_config.name}@{weights_sha[:12]}",
            resolved_config={
                "model": self.model_config.model_dump(mode="json"),
                "encoder_identity": self.model.encoder.identity(),
                **_expansion_identity(self.model),
                "input_rendering": RENDERING_REVISION,
            },
            weights=FileRef(path="weights.pt", sha256=weights_sha, rows=0),
            training=training,
            environment=environment_snapshot(accelerator_snapshot(self.config.device)),
            source=source_revision(),
            created_at=utc_now(),
        )
        (directory / BUNDLE_MANIFEST).write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n"
        )
        return manifest


def _write_checkpoint(
    work: Path,
    context: _BundleContext,
    progress: Progress,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> None:
    """Write ``checkpoints/step_N`` atomically, then drop older checkpoints' train state."""
    root = work / CHECKPOINTS
    name = f"step_{progress.step:07d}"
    final = root / name
    if final.exists():
        return
    temporary = root / (name + INCOMPLETE_SUFFIX)
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    context.write_bundle(
        temporary,
        progress,
        train_state(progress, optimizer, scaler),
        checkpoint={"step": progress.step, "epoch": progress.epoch, "position": progress.position},
    )
    temporary.rename(final)
    pointer = root / (LATEST + ".tmp")
    pointer.write_text(name + "\n")
    pointer.replace(root / LATEST)
    for older in root.glob("step_*"):
        if older.name != name and (older / TRAIN_STATE).exists():
            (older / TRAIN_STATE).unlink()
    print(f"checkpoint {final}", file=sys.stderr, flush=True)
