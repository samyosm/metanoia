"""Precomputed proposal orderings (decision 0014): a speed cache, never a behavior change.

Fused proposals rerank a pool with a cross-encoder, which is too slow to run inside
training. A proposal cache stores, for each (question, candidate), the first
``width`` memory nodes of the fused ordering and their retrieval similarities. It is
an artifact like a memory: an entries file, a manifest with its hash, and the identity
of everything the ordering depends on. A model refuses a cache whose identity differs
from its own, and ranks any (question, candidate) missing from the cache live.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from metanoia_contracts.provenance import sha256_file

CACHE_MANIFEST = "proposals.json"
ENTRIES = "entries.jsonl"


def proposal_key(question: str, candidate: str) -> str:
    return hashlib.sha256(f"{question}\x1f{candidate}".encode()).hexdigest()[:32]


class ProposalCache:
    def __init__(self, directory: Path, identity: dict[str, Any]) -> None:
        manifest = json.loads((directory / CACHE_MANIFEST).read_text())
        if manifest["identity"] != identity:
            raise ValueError(
                f"proposal cache {directory} was built for {manifest['identity']}, not {identity}"
            )
        entries = directory / manifest["entries"]["path"]
        if sha256_file(entries) != manifest["entries"]["sha256"]:
            raise ValueError(f"{entries} does not match its recorded hash")
        self.manifest = manifest
        self.width: int = manifest["width"]
        self.entries: dict[str, tuple[list[int], list[float]]] = {}
        with entries.open() as handle:
            for line in handle:
                row = json.loads(line)
                self.entries[row["key"]] = (row["nodes"], row["similarity"])

    def get(self, question: str, candidate: str) -> tuple[list[int], list[float]] | None:
        return self.entries.get(proposal_key(question, candidate))

    def cache_id(self) -> str:
        return str(self.manifest["entries"]["sha256"])


def write_cache(
    directory: Path,
    identity: dict[str, Any],
    width: int,
    rows: dict[str, tuple[list[int], list[float]]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Write entries and manifest into ``directory`` (which must be new and empty)."""
    entries = directory / ENTRIES
    with entries.open("w") as handle:
        for key in sorted(rows):
            nodes, sims = rows[key]
            record = {"key": key, "nodes": nodes, "similarity": [round(x, 6) for x in sims]}
            handle.write(json.dumps(record) + "\n")
    manifest = {
        "kind": "proposal_cache",
        "identity": identity,
        "width": width,
        "entries": {"path": ENTRIES, "sha256": sha256_file(entries), "rows": len(rows)},
        **extra,
    }
    (directory / CACHE_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
