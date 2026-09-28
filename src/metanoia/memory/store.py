"""Load a concept memory bundle and search it exactly (decision 0010).

Exact search (a matrix product plus top-k) is the reference; an ANN index is a later,
separate adapter whose recall is measured against this one (PLAN §3).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from metanoia_contracts.artifacts import MemoryManifest
from metanoia_contracts.base import load_strict
from metanoia_contracts.provenance import sha256_file

MEMORY_MANIFEST = "memory.json"


@dataclass(frozen=True)
class Concept:
    id: str  # stable identity, e.g. a WordNet synset name
    name: str  # display name
    names: tuple[str, ...]  # all surface names (lemmas), lower-case
    pos: str
    text: str  # the text whose embedding is the key


class ConceptMemory:
    def __init__(self, directory: Path, device: str = "cpu") -> None:
        self.directory = directory
        self.manifest = load_strict(
            MemoryManifest, json.loads((directory / MEMORY_MANIFEST).read_text())
        )
        for ref in (self.manifest.concepts, self.manifest.vectors):
            if sha256_file(directory / ref.path) != ref.sha256:
                raise ValueError(f"{directory / ref.path} does not match its recorded hash")
        rows = [json.loads(line) for line in (directory / self.manifest.concepts.path).open()]
        self.concepts = [
            Concept(r["id"], r["name"], tuple(r["names"]), r["pos"], r["text"]) for r in rows
        ]
        vectors = np.load(directory / self.manifest.vectors.path)
        if vectors.shape != (len(self.concepts), self.manifest.dim):
            raise ValueError(f"vectors {vectors.shape} do not match {len(self.concepts)} concepts")
        self.keys = torch.from_numpy(vectors).to(device)

    @property
    def memory_id(self) -> str:
        return self.manifest.memory_id

    def __len__(self) -> int:
        return len(self.concepts)

    def search(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact top-k by cosine similarity: ``queries`` [Q, dim], normalized.
        Returns (indices [Q, k], similarities [Q, k])."""
        scores = queries.to(self.keys.device, self.keys.dtype) @ self.keys.T
        top = torch.topk(scores, k=min(k, len(self.concepts)), dim=-1)
        return top.indices, top.values.float()
