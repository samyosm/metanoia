"""Fixed-suite collection: run a predictor on a suite's cases and persist raw outputs.

The runner passes only inference inputs to the predictor and never reads labels. It
does not compute metrics; ``metanoia_analysis`` scores the saved run (PLAN.md §7.4).
"""

import hashlib
import json
import time
import tomllib
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

import metanoia_research
from metanoia.api import Predictor
from metanoia_contracts import DecisionRequest, DecisionResult
from metanoia_contracts.artifacts import (
    CaseRecord,
    FileRef,
    PredictionRecord,
    PredictionStatus,
    RunCounts,
    RunManifest,
)
from metanoia_contracts.base import StrictModel, load_strict
from metanoia_contracts.provenance import (
    canonical_json,
    content_sha256,
    environment_snapshot,
    sha256_text,
    source_revision,
    utc_now,
)
from metanoia_contracts.suites import NamedBudget, SuiteProtocol
from metanoia_research.artifacts import (
    accelerator_snapshot,
    discard_incomplete,
    finalize_directory,
    new_incomplete_directory,
)
from metanoia_research.data.bundle import read_cases

RUNNER_REVISION = "runner/1"


class PredictorSpec(StrictModel):
    kind: Literal["bundle", "lexical", "oracle"]
    path: str | None = None
    calibration: str | None = None
    device: str = "cpu"
    # Inference precision for bundle predictors; part of the predictor's identity.
    precision: Literal["fp32", "fp16"] = "fp32"
    # Marker-encoding bundles: average over this many option rotations (identity changes).
    rotations: int = Field(default=1, ge=1)


class CollectionConfig(StrictModel):
    config_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    suite: str
    dataset: str
    predictor: PredictorSpec
    output_root: str = "artifacts/runs"
    # A subset of the protocol's splits, e.g. ``["dev"]`` for checkpoint selection. The
    # subset is recorded in the run's suite identity, so such a run is never compared
    # with a full-protocol run by accident.
    splits: tuple[str, ...] | None = None
    # A subset of the protocol's budget ids (e.g. one budget for checkpoint selection),
    # recorded in the suite identity like ``splits``.
    budgets: tuple[str, ...] | None = None


def load_protocol(suite_dir: Path) -> tuple[SuiteProtocol, dict[str, Any], str]:
    raw = tomllib.loads((suite_dir / "protocol.toml").read_text())
    return load_strict(SuiteProtocol, raw), raw, sha256_text(canonical_json(raw))


def build_predictor(spec: PredictorSpec) -> Predictor:
    if spec.kind == "lexical":
        from metanoia.baselines.lexical import LexicalOverlapPredictor

        return LexicalOverlapPredictor()
    if spec.kind == "oracle":
        from metanoia_research.oracles import MicroworldOracle

        return MicroworldOracle()
    if spec.path is None:
        raise ValueError("a bundle predictor needs a path")
    from metanoia.api import load_predictor

    calibration = Path(spec.calibration) if spec.calibration else None
    return load_predictor(
        Path(spec.path),
        calibration=calibration,
        device=spec.device,
        precision=spec.precision,
        rotations=spec.rotations,
    )


def derive_seed(base_seed: int, example_id: str, budget_id: str, repeat: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}:{budget_id}:{repeat}".encode()).digest()
    return int.from_bytes(digest[:4])


_PREDICTION_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("example_id", pa.string()),
        ("budget_id", pa.string()),
        ("repeat_id", pa.int64()),
        ("status", pa.string()),
        ("error", pa.string()),
        ("candidate_ids", pa.list_(pa.string())),
        ("logits", pa.list_(pa.float64())),
        ("probabilities", pa.list_(pa.float64())),
        ("calibration_status", pa.string()),
        ("calibrator_id", pa.string()),
        ("temperature", pa.float64()),
        ("termination", pa.string()),
        *[
            (f"usage_{n}", pa.int64())
            for n in (
                "nodes",
                "queries",
                "proposals",
                "pairs",
                "edges",
                "composites",
                "rounds",
                "depth",
            )
        ],
        ("wall_time_ms", pa.float64()),
        ("batch_index", pa.int64()),
    ]
)


def _check_result(predictor: Predictor, request: DecisionRequest, result: DecisionResult) -> None:
    if result.candidate_ids != request.candidate_ids:
        raise ValueError("result candidate order differs from the request")
    if result.predictor_id != predictor.predictor_id:
        raise ValueError(f"unexpected predictor id {result.predictor_id}")
    if result.budget != request.budget or not result.usage_within_budget():
        raise ValueError("result exceeds or misreports the requested budget")


def _run_batch(
    predictor: Predictor,
    run_id: str,
    batch: Sequence[CaseRecord],
    requests: list[DecisionRequest],
    named: NamedBudget,
    repeat: int,
    batch_index: int,
) -> tuple[list[PredictionRecord], list[dict[str, Any]]]:
    keys = {
        "run_id": run_id,
        "budget_id": named.id,
        "repeat_id": repeat,
        "batch_index": batch_index,
    }
    start = time.perf_counter()
    try:
        results = predictor.decide_batch(requests)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if len(results) != len(requests):
            raise ValueError("predictor returned the wrong number of results")
        for request, result in zip(requests, results, strict=True):
            _check_result(predictor, request, result)
    except Exception as error:  # noqa: BLE001 - every failure is retained as a record
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        message = f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}"
        return [
            PredictionRecord.model_validate(
                {
                    **keys,
                    "example_id": case.example_id,
                    "status": PredictionStatus.FAILED,
                    "error": message,
                    "candidate_ids": list(request.candidate_ids),
                    "logits": [],
                    "probabilities": [],
                    "wall_time_ms": elapsed_ms / len(batch),
                }
            )
            for case, request in zip(batch, requests, strict=True)
        ], []

    records, traces = [], []
    for case, result in zip(batch, results, strict=True):
        records.append(
            PredictionRecord.model_validate(
                {
                    **keys,
                    "example_id": case.example_id,
                    "status": PredictionStatus.OK,
                    "candidate_ids": list(result.candidate_ids),
                    "logits": list(result.logits),
                    "probabilities": list(result.probabilities),
                    "calibration_status": result.calibration.status,
                    "calibrator_id": result.calibration.calibrator_id,
                    "temperature": result.calibration.temperature,
                    "termination": result.termination,
                    **PredictionRecord.usage_columns(result.usage),
                    # Batch wall time shared equally; batch_size is part of the protocol.
                    "wall_time_ms": elapsed_ms / len(batch),
                }
            )
        )
        if result.trace is not None:
            traces.append(
                {
                    "example_id": case.example_id,
                    "budget_id": named.id,
                    "repeat_id": repeat,
                    "events": [e.model_dump(mode="json") for e in result.trace],
                }
            )
    return records, traces


def collect(config: CollectionConfig, predictor: Predictor | None = None) -> Path:
    """Execute the suite and write a finalized run directory. Returns its path."""
    protocol, protocol_raw, protocol_sha = load_protocol(Path(config.suite))
    dataset_manifest, cases = read_cases(Path(config.dataset))
    pin = protocol.dataset
    if (
        dataset_manifest.dataset_id,
        dataset_manifest.cases.sha256,
        dataset_manifest.labels.sha256,
    ) != (
        pin.dataset_id,
        pin.cases_sha256,
        pin.labels_sha256,
    ):
        raise ValueError(
            f"dataset {dataset_manifest.dataset_id} at {config.dataset} does not match the "
            f"content pinned by suite {protocol.suite_id} r{protocol.revision}"
        )
    rules = protocol.evaluation
    splits = config.splits or rules.splits
    unknown = set(splits) - set(rules.splits)
    if unknown:
        raise ValueError(f"splits {sorted(unknown)} are not in the protocol {rules.splits}")
    selected = [c for split in splits for c in cases if c.split == split]
    if not selected:
        raise ValueError(f"no cases in splits {splits}")
    budgets = list(protocol.budgets)
    if config.budgets:
        known = {b.id for b in protocol.budgets}
        unknown = set(config.budgets) - known
        if unknown:
            raise ValueError(f"budgets {sorted(unknown)} are not in the protocol")
        budgets = [b for b in protocol.budgets if b.id in config.budgets]

    predictor = predictor or build_predictor(config.predictor)
    run_dir = Path(config.output_root) / config.run_id
    work = new_incomplete_directory(run_dir)
    started = utc_now()
    try:
        records: list[PredictionRecord] = []
        traces: list[dict[str, Any]] = []
        batch_index = 0
        for named in budgets:
            for repeat in range(rules.repeats):
                for start in range(0, len(selected), rules.batch_size):
                    batch = selected[start : start + rules.batch_size]
                    requests = [
                        load_strict(
                            DecisionRequest,
                            {
                                **case.request.model_dump(mode="json"),
                                "request_id": case.example_id,
                                "budget": named.budget.model_dump(mode="json"),
                                "trace": rules.trace.value,
                                "seed": derive_seed(
                                    rules.base_seed, case.example_id, named.id, repeat
                                ),
                            },
                        )
                        for case in batch
                    ]
                    new_records, new_traces = _run_batch(
                        predictor, config.run_id, batch, requests, named, repeat, batch_index
                    )
                    records += new_records
                    traces += new_traces
                    batch_index += 1

        rows = [r.model_dump(mode="json") for r in records]
        pq.write_table(
            pa.Table.from_pylist(rows, schema=_PREDICTION_SCHEMA), work / "predictions.parquet"
        )
        trace_ref = None
        if traces:
            (work / "traces.jsonl").write_text("".join(canonical_json(t) + "\n" for t in traces))
            trace_ref = FileRef(
                path="traces.jsonl", sha256=content_sha256(traces), rows=len(traces)
            )

        expected = {
            (c.example_id, b.id, r) for c in selected for b in budgets for r in range(rules.repeats)
        }
        _validate_written(work / "predictions.parquet", expected)
        failed = sum(r.status is PredictionStatus.FAILED for r in records)
        manifest = RunManifest(
            run_id=config.run_id,
            suite={
                "suite_id": protocol.suite_id,
                "revision": protocol.revision,
                "path": config.suite,
                **({"splits": list(config.splits)} if config.splits else {}),
                **({"budgets": list(config.budgets)} if config.budgets else {}),
            },
            protocol_sha256=protocol_sha,
            protocol=protocol_raw,
            dataset_id=dataset_manifest.dataset_id,
            cases_sha256=dataset_manifest.cases.sha256,
            labels_sha256=dataset_manifest.labels.sha256,
            predictor={**predictor.describe(), "spec": config.predictor.model_dump(mode="json")},
            runner_revision=f"{RUNNER_REVISION}+metanoia_research-{metanoia_research.__version__}",
            environment=environment_snapshot(accelerator_snapshot(config.predictor.device)),
            source=source_revision(),
            counts=RunCounts(
                expected=len(expected),
                attempted=len(records),
                completed=len(records) - failed,
                failed=failed,
            ),
            predictions=FileRef(
                path="predictions.parquet", sha256=content_sha256(rows), rows=len(rows)
            ),
            traces=trace_ref,
            started_at=started,
            finished_at=utc_now(),
        )
        (work / "run.json").write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n"
        )
    except BaseException:
        discard_incomplete(work)
        raise
    finalize_directory(work, run_dir)
    return run_dir


def _validate_written(path: Path, expected: set[tuple[str, str, int]]) -> None:
    seen: set[tuple[str, str, int]] = set()
    for row in pq.read_table(path).to_pylist():
        record = load_strict(PredictionRecord, row)
        if record.key in seen:
            raise ValueError(f"duplicate prediction key {record.key}")
        seen.add(record.key)
    if seen != expected:
        raise ValueError(
            f"prediction keys differ from the protocol: {len(expected - seen)} missing, "
            f"{len(seen - expected)} unexpected"
        )
