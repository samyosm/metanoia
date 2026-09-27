"""Artifact writing discipline: write to an incomplete directory, validate, then finalize.

Schemas come from ``metanoia_contracts``; this module owns only the file mechanics.
A completed artifact directory is never overwritten.
"""

import shutil
from pathlib import Path
from typing import Any

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


def accelerator_snapshot(device: str) -> dict[str, Any]:
    """Accelerator and numerical-determinism details for manifests (PLAN.md §6.4)."""
    import torch

    info: dict[str, Any] = {
        "requested_device": device,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    if torch.cuda.is_available() and device.startswith("cuda"):
        index = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(index)
        info.update(
            gpu_name=props.name,
            gpu_capability=f"{props.major}.{props.minor}",
            gpu_memory_bytes=props.total_memory,
            driver_version=_nvidia_driver_version(),
        )
    return info


def _nvidia_driver_version() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip().splitlines()[0] if out.strip() else None
