"""Artifact writing discipline: write to an incomplete directory, validate, then finalize.

Schemas come from ``metanoia_contracts``; this module owns only the file mechanics.
A completed artifact directory is never overwritten.
"""

import shutil
from pathlib import Path

INCOMPLETE_SUFFIX = ".incomplete"


def new_incomplete_directory(final: Path) -> Path:
    if final.exists():
        raise FileExistsError(f"{final} already exists; completed artifacts are never overwritten")
    work = final.with_name(final.name + INCOMPLETE_SUFFIX)
    if work.exists():
        raise FileExistsError(
            f"{work} is left over from an interrupted write; inspect and remove it first"
        )
    work.mkdir(parents=True)
    return work


def finalize_directory(work: Path, final: Path) -> None:
    """Atomic on a single filesystem: the final path appears only when complete."""
    if final.exists():
        raise FileExistsError(final)
    work.rename(final)


def discard_incomplete(work: Path) -> None:
    if work.name.endswith(INCOMPLETE_SUFFIX) and work.exists():
        shutil.rmtree(work)
