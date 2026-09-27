"""Dataset bundle IO: ``cases.jsonl`` (inputs), ``labels.parquet``, ``proofs.jsonl``.

Labels and proofs are stored separately from inputs so that a runner can pass cases to
a predictor without ever loading answers.
"""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from metanoia_contracts.artifacts import (
    LABEL_CORE_COLUMNS,
    CaseRecord,
    DatasetManifest,
    FileRef,
    LabelRecord,
)
from metanoia_contracts.base import load_strict
from metanoia_contracts.provenance import canonical_json, content_sha256, utc_now
from metanoia_research.artifacts import finalize_directory, new_incomplete_directory

MANIFEST = "dataset.json"


def write_dataset(
    out_dir: Path,
    dataset_id: str,
    generator: dict[str, Any],
    cases: list[CaseRecord],
    labels: list[LabelRecord],
    proofs: list[dict[str, Any]],
) -> DatasetManifest:
    if [c.example_id for c in cases] != [label.example_id for label in labels]:
        raise ValueError("cases and labels must list the same examples in the same order")
    if len({c.example_id for c in cases}) != len(cases):
        raise ValueError("duplicate example ids")

    work = new_incomplete_directory(out_dir)
    case_rows = [c.model_dump(mode="json") for c in cases]
    (work / "cases.jsonl").write_text("".join(canonical_json(r) + "\n" for r in case_rows))

    label_rows = [label.model_dump(mode="json") for label in labels]
    flat = [{**{k: r[k] for k in LABEL_CORE_COLUMNS}, **r["annotations"]} for r in label_rows]
    pq.write_table(pa.Table.from_pylist(flat), work / "labels.parquet")
    (work / "proofs.jsonl").write_text("".join(canonical_json(p) + "\n" for p in proofs))

    splits: dict[str, int] = {}
    for case in cases:
        splits[case.split] = splits.get(case.split, 0) + 1
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        generator=generator,
        splits=dict(sorted(splits.items())),
        cases=FileRef(path="cases.jsonl", sha256=content_sha256(case_rows), rows=len(cases)),
        labels=FileRef(path="labels.parquet", sha256=content_sha256(label_rows), rows=len(labels)),
        proofs=FileRef(path="proofs.jsonl", sha256=content_sha256(proofs), rows=len(proofs)),
        created_at=utc_now(),
    )
    (work / MANIFEST).write_text(manifest.model_dump_json(indent=2) + "\n")
    finalize_directory(work, out_dir)
    return manifest


def read_manifest(dataset_dir: Path) -> DatasetManifest:
    return load_strict(DatasetManifest, json.loads((dataset_dir / MANIFEST).read_text()))


def read_cases(dataset_dir: Path) -> tuple[DatasetManifest, list[CaseRecord]]:
    manifest = read_manifest(dataset_dir)
    rows = [
        json.loads(line)
        for line in (dataset_dir / manifest.cases.path).read_text().splitlines()
        if line
    ]
    if content_sha256(rows) != manifest.cases.sha256:
        raise ValueError(f"cases content hash mismatch in {dataset_dir}")
    return manifest, [load_strict(CaseRecord, row) for row in rows]


def read_labels(dataset_dir: Path) -> list[LabelRecord]:
    """Used by training only. Collection runs never read labels."""
    manifest = read_manifest(dataset_dir)
    rows = pq.read_table(dataset_dir / manifest.labels.path).to_pylist()
    labels = [
        LabelRecord(
            **{k: row[k] for k in LABEL_CORE_COLUMNS},
            annotations={k: v for k, v in row.items() if k not in LABEL_CORE_COLUMNS},
        )
        for row in rows
    ]
    if content_sha256(label.model_dump(mode="json") for label in labels) != manifest.labels.sha256:
        raise ValueError(f"labels content hash mismatch in {dataset_dir}")
    return labels
