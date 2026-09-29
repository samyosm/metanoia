"""Lazy access to a frozen memory for decision models that expand candidates.

The memory and its retrieval encoder are frozen external resources, identified in the
bundle, not stored in it (like pretrained encoder weights). They are kept outside
``nn.Module`` registration so they never enter the state dict.
"""

import hashlib
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch

from metanoia.memory.encoder import RetrievalEncoder
from metanoia.memory.store import ConceptMemory


class MemoryAccess:
    """Mixin for ``nn.Module`` decision models; call :meth:`_init_memory` in ``__init__``."""

    device: torch.device

    def _init_memory(self, path: str) -> None:
        object.__setattr__(self, "_memory_path", path)
        object.__setattr__(self, "_memory", None)
        object.__setattr__(self, "_retriever", None)

    def memory(self) -> ConceptMemory:
        memory: ConceptMemory | None = self.__dict__.get("_memory")
        if memory is None:
            memory = ConceptMemory(Path(self.__dict__["_memory_path"]), device=str(self.device))
            object.__setattr__(self, "_memory", memory)
        elif memory.keys.device != self.device:
            # Loaded before the model moved (e.g. the identity check at load time).
            memory.keys = memory.keys.to(self.device)
        return memory

    def retriever(self) -> RetrievalEncoder:
        existing: RetrievalEncoder | None = self.__dict__.get("_retriever")
        if existing is None:
            spec = self.memory().manifest.encoder
            retriever = RetrievalEncoder(
                spec["model"],
                spec["requested_revision"],
                spec["pooling"],
                spec["max_length"],
                spec["query_prefix"],
            ).to(self.device)
            if retriever.identity() != spec:
                raise ValueError("retrieval encoder differs from the memory's encoder")
            object.__setattr__(self, "_retriever", retriever)
            existing = retriever
        return existing

    def expansion_identity(self) -> dict[str, object]:
        manifest = self.memory().manifest
        return {
            "memory_id": manifest.memory_id,
            "concepts_sha256": manifest.concepts.sha256,
            "vectors_sha256": manifest.vectors.sha256,
            "retrieval_encoder": manifest.encoder,
        }

    def propose(
        self,
        pairs: Sequence[tuple[str, str]],
        k: int,
        source: Literal["retrieved", "random"],
    ) -> tuple[list[list[int]], list[list[float]]]:
        """k memory indices per (question, candidate) pair, with their similarity.

        "retrieved": exact search from "question candidate" through the frozen retrieval
        encoder. "random": a draw that is deterministic per (question, candidate), so
        collection runs are reproducible; its similarity is reported as 0.
        """
        memory = self.memory()
        if source == "retrieved":
            with torch.no_grad():
                vectors = self.retriever().embed([f"{q} {c}" for q, c in pairs], query=True)
                indices, sims = memory.search(vectors, k)
            return indices.tolist(), sims.tolist()
        drawn = []
        for q, c in pairs:
            seed = int.from_bytes(hashlib.sha256(f"{q}\x1f{c}".encode()).digest()[:8], "big")
            drawn.append(random.Random(seed).sample(range(len(memory)), k))
        return drawn, [[0.0] * k for _ in drawn]
