"""Concept memory bundles: hash-checked loading and exact search."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from metanoia.memory.store import MEMORY_MANIFEST, ConceptMemory
from metanoia_contracts.artifacts import FileRef, MemoryManifest
from metanoia_contracts.provenance import sha256_file, utc_now


def write_memory(directory: Path, vectors: np.ndarray) -> None:
    directory.mkdir()
    rows = [
        {
            "id": f"c{i}",
            "name": f"concept {i}",
            "names": [f"concept {i}"],
            "pos": "n",
            "text": f"concept {i}: gloss",
        }
        for i in range(len(vectors))
    ]
    (directory / "concepts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.save(directory / "vectors.npy", vectors.astype(np.float16))
    manifest = MemoryManifest(
        memory_id="tiny",
        source={"kind": "test"},
        text_template="{name}: {gloss}",
        encoder={"model": "none"},
        concepts=FileRef(
            path="concepts.jsonl", sha256=sha256_file(directory / "concepts.jsonl"), rows=len(rows)
        ),
        vectors=FileRef(
            path="vectors.npy", sha256=sha256_file(directory / "vectors.npy"), rows=len(rows)
        ),
        dim=vectors.shape[1],
        environment={},
        code_source={},
        created_at=utc_now(),
    )
    (directory / MEMORY_MANIFEST).write_text(manifest.model_dump_json())


def test_exact_search_returns_nearest_concepts(tmp_path) -> None:
    keys = np.eye(4)
    write_memory(tmp_path / "m", keys)
    memory = ConceptMemory(tmp_path / "m")
    assert len(memory) == 4 and memory.concepts[2].name == "concept 2"
    query = torch.tensor([[0.0, 0.8, 0.6, 0.0]])
    indices, sims = memory.search(query, k=2)
    assert indices.tolist() == [[1, 2]]
    assert sims[0].tolist() == pytest.approx([0.8, 0.6], abs=1e-3)


def test_tampered_memory_is_refused(tmp_path) -> None:
    write_memory(tmp_path / "m", np.eye(3))
    (tmp_path / "m" / "vectors.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        ConceptMemory(tmp_path / "m")
