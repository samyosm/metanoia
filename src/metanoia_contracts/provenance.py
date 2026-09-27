"""Hashing, canonical serialization, and environment records for artifact provenance.

Standard library only, so every area (including analysis without ML dependencies)
records provenance the same way.
"""

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, no NaN."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash of row content independent of the container format (JSONL, Parquet, ...)."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(dict(row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# Packages whose versions can change a result. Recorded when installed.
_RELEVANT_PACKAGES = (
    "metanoia",
    "pydantic",
    "torch",
    "transformers",
    "numpy",
    "scipy",
    "pyarrow",
    "faiss-cpu",
    "faiss-gpu",
)


def environment_snapshot(accelerator: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``accelerator`` is supplied by callers that own the numerical backend (contracts
    never import torch); ``None`` records that no accelerator description was given."""
    packages: dict[str, str] = {}
    for name in _RELEVANT_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "accelerator": dict(accelerator) if accelerator is not None else None,
    }


def source_revision(path: Path | None = None) -> dict[str, Any]:
    """Git revision and dirty diff hash, or ``{"vcs": None}`` outside a repository."""
    cwd = path or Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        diff = git("diff", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return {"vcs": None}
    return {
        "vcs": "git",
        "commit": commit,
        "dirty": bool(diff),
        "diff_sha256": sha256_text(diff) if diff else None,
    }
