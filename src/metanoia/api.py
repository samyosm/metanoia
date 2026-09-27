"""Public decision interface: a validated request in, a validated result out.

Every predictor (learned model, baseline, fixture, oracle) implements :class:`Predictor`.
Single-request and batch entry points share semantics.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from metanoia_contracts import DecisionRequest, DecisionResult, TraceEvent, TraceMode
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
            "kind": "torch_direct",
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
        for request, row in zip(requests, rows, strict=True):
            logits = row[: len(request.candidates)]
            results.append(self._result(request, logits))
        return results

    def _result(self, request: DecisionRequest, logits: list[float]) -> DecisionResult:
        # The direct model performs no graph work, so it consumes none of the reasoning
        # budget; the budget is a cap, not a quota.
        trace = None
        if request.trace is TraceMode.EVENTS:
            trace = (
                TraceEvent(seq=0, op="encode", outputs=("context", "candidates")),
                TraceEvent(
                    seq=1,
                    op="direct_score",
                    inputs=("context", *request.candidate_ids),
                    details={"logits": list(logits)},
                ),
            )
        return DecisionResult(
            request_id=request.request_id,
            predictor_id=self.predictor_id,
            candidate_ids=request.candidate_ids,
            logits=tuple(logits),
            probabilities=tuple(self.calibrator.probabilities(logits)),
            calibration=self.calibrator.info,
            budget=request.budget,
            trace=trace,
        )


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
