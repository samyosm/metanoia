"""Public decision interface: a validated request in, a validated result out.

Every predictor (learned model, baseline, fixture, oracle) implements :class:`Predictor`.
Single-request and batch entry points share semantics.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from metanoia_contracts import (
    DecisionRequest,
    DecisionResult,
    ResourceUsage,
    Termination,
    TraceEvent,
    TraceMode,
)
from metanoia_contracts.artifacts import PredictorBundleManifest
from metanoia_contracts.base import load_strict
from metanoia_contracts.provenance import sha256_file


@runtime_checkable
class Predictor(Protocol):
    @property
    def predictor_id(self) -> str: ...

    def describe(self) -> dict[str, Any]:
        """Identity and resolved configuration, recorded in run manifests."""
        ...

    def decide(self, request: DecisionRequest) -> DecisionResult: ...

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResult]: ...


class TorchPredictor:
    """Wraps a trained :class:`~metanoia.model.DirectDecisionModel` and a calibrator."""

    def __init__(
        self,
        model: Any,
        predictor_id: str,
        config: dict[str, Any],
        calibrator: Any = None,
        precision: str = "fp32",
    ):
        from metanoia.calibration import Calibrator

        if precision not in ("fp32", "fp16"):
            raise ValueError(f"unsupported precision {precision!r}")
        self.model = model.eval()
        self.precision = precision
        self._predictor_id = predictor_id
        self._config = config
        self.calibrator = calibrator or Calibrator()

    @property
    def predictor_id(self) -> str:
        return self._predictor_id

    def describe(self) -> dict[str, Any]:
        return {
            "predictor_id": self.predictor_id,
            "kind": _predictor_kind(self._config.get("model", {})),
            "model_config": self._config,
            "encoder": self.model.encoder.identity(),
            "calibration": self.calibrator.info.model_dump(mode="json"),
            "precision": self.precision,
        }

    def decide(self, request: DecisionRequest) -> DecisionResult:
        return self.decide_batch([request])[0]

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResult]:
        import torch

        for request in requests:
            self.calibrator.check_budget(request.budget)
        # Inference disables gradients here, at the boundary, not inside the model.
        fp16 = self.precision == "fp16"
        with (
            torch.inference_mode(),
            torch.autocast(self.model.device.type, dtype=torch.float16, enabled=fp16),
        ):
            output = self.model(requests)
            if output.logits.is_cuda:
                torch.cuda.synchronize()
            rows = output.logits.double().cpu().tolist()
        results = []
        for i, (request, row) in enumerate(zip(requests, rows, strict=True)):
            logits = row[: len(request.candidates)]
            record = _workspace_record(output, i) or _expansion_record(self.model, output, i)
            results.append(self._result(request, logits, record))
        return results

    def _trace(
        self,
        request: DecisionRequest,
        logits: list[float],
        probabilities: tuple[float, ...],
        workspace: "WorkspaceRecord | ExpansionRecord | None" = None,
    ) -> tuple[TraceEvent, ...]:
        """What the direct model computed, as detached summaries. Tracing never feeds
        back into the prediction (the logits above were computed without it)."""
        from metanoia.encoding.inputs import RENDERING_REVISION, render_input

        encoder = self.model.encoder
        rendered = render_input(request)
        tokens = encoder.token_count(rendered)
        info = self.calibrator.info
        events = [
            TraceEvent(
                seq=0,
                op="render",
                inputs=(
                    "question",
                    "context",
                    *(("observations",) if request.observations else ()),
                ),
                outputs=("input",),
                details={"rendering": RENDERING_REVISION, "text": rendered},
            ),
            TraceEvent(
                seq=1,
                op="encode",
                inputs=("input", *request.candidate_ids),
                outputs=("context", *(f"a:{cid}" for cid in request.candidate_ids)),
                details={
                    "encoder": str(encoder.identity().get("model", encoder.identity()["kind"])),
                    "input_tokens": tokens,
                    "max_length": encoder.max_length,
                    "truncated": tokens > encoder.max_length,
                    # joint: each candidate is encoded together with the input.
                    "candidate_encoding": self.model.candidate_encoding,
                },
            ),
            _first_score(request, logits, workspace),
        ]
        if isinstance(workspace, WorkspaceRecord):
            events += _workspace_events(request, workspace, len(events))
        elif isinstance(workspace, ExpansionRecord):
            events += _expansion_events(request, workspace, len(events))
        events.append(
            TraceEvent(
                seq=len(events),
                op="calibrate",
                inputs=request.candidate_ids,
                outputs=request.candidate_ids,
                details={
                    "status": info.status.value,
                    "calibrator_id": info.calibrator_id,
                    "temperature": info.temperature,
                    "probabilities": list(probabilities),
                },
            ),
        )
        return tuple(events)

    def _result(
        self,
        request: DecisionRequest,
        logits: list[float],
        workspace: "WorkspaceRecord | ExpansionRecord | None" = None,
    ) -> DecisionResult:
        # The direct model performs no graph work, so it consumes none of the reasoning
        # budget; the budget is a cap, not a quota.
        probabilities = tuple(self.calibrator.probabilities(logits))
        trace = (
            self._trace(request, logits, probabilities, workspace)
            if request.trace is TraceMode.EVENTS
            else None
        )
        usage = ResourceUsage()
        termination = Termination.COMPLETED
        if isinstance(workspace, WorkspaceRecord):
            usage = ResourceUsage(
                nodes=workspace.nodes, pairs=workspace.pairs, rounds=workspace.rounds
            )
            termination = workspace.termination
        elif isinstance(workspace, ExpansionRecord):
            n = len(request.candidates) if workspace.per_candidate else 0
            usage = ResourceUsage(
                queries=n,
                proposals=n * workspace.per_candidate,
                nodes=n * workspace.per_candidate,
                depth=1 if workspace.per_candidate else 0,
            )
            termination = workspace.termination
        return DecisionResult(
            request_id=request.request_id,
            predictor_id=self.predictor_id,
            candidate_ids=request.candidate_ids,
            logits=tuple(logits),
            probabilities=probabilities,
            calibration=self.calibrator.info,
            budget=request.budget,
            usage=usage,
            termination=termination,
            trace=trace,
        )


@dataclass(frozen=True)
class WorkspaceRecord:
    """One example's detached workspace summary, for results and traces."""

    mode: str
    rounds: int
    nodes: int
    pairs: int
    termination: Termination
    statements: list[str]
    initial_logits: list[float]  # the zero-round decision
    has_direct_path: bool  # initial_logits are the direct scorer's
    round_logits: list[list[float]]  # after rounds 1..rounds
    attention: list[list[list[float]]]  # per round: candidate -> sentence weights


def _workspace_record(output: Any, i: int) -> WorkspaceRecord | None:
    from metanoia.engine import WorkspaceOutput

    if not isinstance(output, WorkspaceOutput):
        return None
    k = output.rounds[i]
    n = int(output.candidate_mask[i].sum())
    return WorkspaceRecord(
        mode=str(output.mode),
        rounds=k,
        nodes=output.nodes[i],
        pairs=output.pairs[i],
        termination=output.termination[i],
        statements=output.statements[i],
        initial_logits=output.initial_logits[i, :n].double().cpu().tolist(),
        has_direct_path=bool(output.has_direct_path),
        round_logits=[r[i, :n].double().cpu().tolist() for r in output.round_logits[:k]],
        attention=[a[i, :n].double().cpu().tolist() for a in output.attention[:k]],
    )


def _first_score(
    request: DecisionRequest,
    logits: list[float],
    workspace: "WorkspaceRecord | ExpansionRecord | None",
) -> TraceEvent:
    """The decision before any workspace round: the direct scorer's logits, or, for a
    workspace-only readout (decision 0009), the readout of the initial nodes."""
    inputs = ("context", *(f"a:{cid}" for cid in request.candidate_ids))
    if workspace is None:
        return TraceEvent(
            seq=2,
            op="direct_score",
            inputs=inputs,
            outputs=request.candidate_ids,
            details={"logits": list(logits)},
        )
    return TraceEvent(
        seq=2,
        op="direct_score" if workspace.has_direct_path else "initial_readout",
        inputs=inputs,
        outputs=request.candidate_ids,
        details={"logits": workspace.initial_logits},
    )


@dataclass(frozen=True)
class ExpansionRecord:
    """One example's detached expansion summary (decision 0011), for results and traces."""

    per_candidate: int  # concepts each candidate expanded into (0: no expansion)
    termination: Termination
    initial_logits: list[float]  # the direct scorer's logits
    has_direct_path: bool
    # Per candidate: its concepts as (concept id, display text, relevance).
    concepts: list[list[tuple[str, str, float]]]
    # Per candidate: its most-attended concepts as (source candidate id, concept, weight).
    attended: list[list[tuple[str, str, float]]]
    evidence: list[float]  # added to each candidate's direct logit


_TOP_EVIDENCE = 4


def _expansion_record(model: Any, output: Any, i: int) -> "ExpansionRecord | None":
    from metanoia.reasoning.expand import ExpansionOutput

    if not isinstance(output, ExpansionOutput):
        return None
    n = int(output.candidate_mask[i].sum())
    initial = output.initial_logits[i, :n].double().cpu().tolist()
    k = output.expanded[i]
    if k == 0 or output.relevance is None or output.attention is None:
        return ExpansionRecord(0, output.termination[i], initial, True, [], [], [0.0] * n)
    memory = model.memory()
    ids = output.concepts[i]
    concepts = [
        [
            (memory.concepts[c].id, memory.concepts[c].text, float(output.relevance[i, j, m]))
            for m, c in enumerate(ids[j])
        ]
        for j in range(len(ids))
    ]
    width = output.relevance.shape[-1]
    attended = []
    for j in range(n):
        row = output.attention[i, j].double().cpu().tolist()
        ranked = sorted(range(len(row)), key=lambda v: -row[v])[:_TOP_EVIDENCE]
        entries = []
        for v in ranked:
            branch, slot = divmod(v, width)
            if branch < len(ids) and slot < len(ids[branch]) and row[v] > 0:
                entries.append((str(branch), memory.concepts[ids[branch][slot]].text, row[v]))
        attended.append(entries)
    evidence = [] if output.evidence is None else output.evidence[i, :n].double().cpu().tolist()
    return ExpansionRecord(k, output.termination[i], initial, True, concepts, attended, evidence)


def _expansion_events(
    request: DecisionRequest, record: ExpansionRecord, start: int
) -> list[TraceEvent]:
    """An ``expand`` event per candidate (its concepts and their relevance), then one
    ``aggregate`` event: each candidate's most-attended concepts, from any branch, and
    the evidence added to its score. Weights describe computation, not proof."""
    ids = request.candidate_ids
    events: list[TraceEvent] = []
    for j, cid in enumerate(ids):
        if j >= len(record.concepts):
            break
        events.append(
            TraceEvent(
                seq=start + len(events),
                op="expand",
                inputs=(f"a:{cid}",),
                outputs=tuple(concept for concept, _, _ in record.concepts[j]),
                details={
                    "candidate": cid,
                    "concepts": [text for _, text, _ in record.concepts[j]],
                    "relevance": [round(r, 4) for _, _, r in record.concepts[j]],
                },
            )
        )
    if record.per_candidate:
        attended = [
            f"{cid}|{ids[int(branch)]}|{weight:.3f}|{text[:120]}"
            for cid, entries in zip(ids, record.attended, strict=False)
            for branch, text, weight in entries
        ]
        events.append(
            TraceEvent(
                seq=start + len(events),
                op="aggregate",
                inputs=tuple(f"a:{cid}" for cid in ids),
                outputs=ids,
                details={"evidence": record.evidence, "attended": attended},
            )
        )
    return events


_TOP_STATEMENTS = 3


def _workspace_events(
    request: DecisionRequest, workspace: WorkspaceRecord, start: int
) -> list[TraceEvent]:
    """A ``workspace`` event, then one ``aggregate`` event per executed round, each with
    the candidates' logits after the round and their most-attended sentences. Attention
    weights describe the computation; they are not an explanation (decision 0008)."""
    candidates = [f"a:{cid}" for cid in request.candidate_ids]
    events = [
        TraceEvent(
            seq=start,
            op="workspace",
            inputs=("input", *candidates),
            outputs=("workspace",),
            details={
                "mode": workspace.mode,
                "statement_nodes": len(workspace.statements),
                "rounds_requested": request.budget.max_rounds,
                "rounds": workspace.rounds,
                "termination": workspace.termination.value,
            },
        )
    ]
    for r, (logits, weights) in enumerate(
        zip(workspace.round_logits, workspace.attention, strict=True), start=1
    ):
        attended = []
        for cid, row in zip(request.candidate_ids, weights, strict=True):
            ranked = sorted(range(len(row)), key=lambda j: -row[j])[:_TOP_STATEMENTS]
            for j in ranked:
                text = workspace.statements[j] if j < len(workspace.statements) else f"#{j}"
                attended.append(f"{cid}|{j}|{row[j]:.3f}|{text[:120]}")
        events.append(
            TraceEvent(
                seq=start + r,
                op="aggregate",
                inputs=("workspace",),
                outputs=("workspace", *request.candidate_ids),
                details={"round": r, "logits": logits, "attended": attended},
            )
        )
    return events


def _predictor_kind(model_config: dict[str, Any]) -> str:
    if model_config.get("expansion"):
        return "torch_expansion"
    if model_config.get("workspace"):
        return "torch_workspace"
    return "torch_direct"


BUNDLE_MANIFEST = "bundle.json"


def load_predictor(
    bundle_dir: Path, calibration: Path | None = None, device: str = "cpu", precision: str = "fp32"
) -> TorchPredictor:
    """Load a predictor bundle, verify weight hashes and encoder identity."""
    import torch

    from metanoia.build import build_model, check_encoder_identity
    from metanoia.calibration import Calibrator
    from metanoia.config import ModelConfig

    manifest = load_strict(
        PredictorBundleManifest, json.loads((bundle_dir / BUNDLE_MANIFEST).read_text())
    )
    weights_path = bundle_dir / manifest.weights.path
    if sha256_file(weights_path) != manifest.weights.sha256:
        raise ValueError(f"weights hash mismatch for {weights_path}")
    config = load_strict(ModelConfig, manifest.resolved_config["model"])
    model = build_model(config)
    check_encoder_identity(model, manifest.resolved_config["encoder_identity"])
    recorded_memory = manifest.resolved_config.get("expansion_identity")
    if recorded_memory is not None:
        found = model.expansion_identity()  # type: ignore[operator]
        if found != recorded_memory:
            raise ValueError(f"concept memory mismatch: bundle {recorded_memory}, found {found}")
    model.load_persistent_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.to(device)
    calibrator = Calibrator.load(calibration, manifest.predictor_id) if calibration else None
    return TorchPredictor(
        model,
        manifest.predictor_id,
        manifest.resolved_config,
        calibrator=calibrator,
        precision=precision,
    )
