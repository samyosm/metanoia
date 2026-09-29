"""Build a proposal cache (decision 0014): fused orderings for every (question,
candidate) of a dataset, so training and collection need not run the reranker.

The cache changes speed, not behavior: a model ranks cache misses live with the same
code (``BranchDecisionModel.fused``), and a cache is refused unless its identity matches
the model's proposal identity.
"""

import time
from pathlib import Path

from metanoia.build import build_model
from metanoia.config import ModelConfig
from metanoia.memory.proposals import proposal_key, write_cache
from metanoia_contracts.provenance import environment_snapshot, source_revision, utc_now
from metanoia_research.artifacts import (
    accelerator_snapshot,
    discard_incomplete,
    finalize_directory,
    new_incomplete_directory,
)
from metanoia_research.data.bundle import read_cases

CHUNK = 64  # (question, candidate) pairs per call


def build_proposals(
    model_config: ModelConfig,
    dataset_dir: Path,
    splits: tuple[str, ...],
    out_dir: Path,
    device: str = "cuda",
) -> dict:
    branch = model_config.branch
    if branch is None or branch.proposals != "fused":
        raise ValueError("proposal caches are built for branch models with fused proposals")
    # Rank everything live: the cache being built must not be read.
    live = model_config.model_copy(
        update={"branch": branch.model_copy(update={"proposal_cache": None})}
    )
    model = build_model(live).to(device).eval()
    manifest, cases = read_cases(dataset_dir)
    pairs = list(
        dict.fromkeys(
            (case.request.question, candidate.text)
            for case in cases
            if case.split in splits
            for candidate in case.request.candidates
        )
    )
    width = branch.max_width
    work = new_incomplete_directory(out_dir)
    try:
        rows: dict[str, tuple[list[int], list[float]]] = {}
        started = time.perf_counter()
        for start in range(0, len(pairs), CHUNK):
            chunk = pairs[start : start + CHUNK]
            nodes, sims = model.fused(chunk, width)  # type: ignore[operator]
            for (q, c), n, s in zip(chunk, nodes, sims, strict=True):
                rows[proposal_key(q, c)] = (n, s)
            if (start // CHUNK) % 50 == 0:
                elapsed = time.perf_counter() - started
                print(f"{start + len(chunk)}/{len(pairs)} pairs, {elapsed:.0f} s", flush=True)
        extra = {
            "dataset_id": manifest.dataset_id,
            "cases_sha256": manifest.cases.sha256,
            "splits": list(splits),
            "pairs": len(pairs),
            "seconds": round(time.perf_counter() - started, 1),
            "environment": environment_snapshot(accelerator_snapshot(device)),
            "code_source": source_revision(),
            "created_at": utc_now(),
        }
        cache_manifest = write_cache(
            work,
            model.proposal_identity(),  # type: ignore[operator]
            width,
            rows,
            extra,
        )
    except BaseException:
        discard_incomplete(work)
        raise
    finalize_directory(work, out_dir)
    return cache_manifest
