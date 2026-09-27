# 0004 — CUDA 12.8 PyTorch wheels (supersedes the CPU-only index of 0002)

**Choice.** `pyproject.toml` pulls torch from the `pytorch-cu128` index. The lock
resolved torch 2.11.0+cu128 (the newest on that index), transformers 5.17.0, Python
3.12. First GPU machine: RTX 2060 6 GB (sm_75), driver 595.71.05, Arch Linux.

**Reason.** Stage 1 needs pretrained encoders; fine-tuning them is impractical on CPU.
cu128 wheels still include sm_75 kernels and run on CPU-only machines (CUDA is simply
unavailable), so no second lock is needed yet.

**Checks.** `uv run pytest` passes; `metanoia-research prepare` reproduces the pinned
`reasoning_v0` hashes (cases `50f21588…`, labels `251aabbf…`) under the new
environment. Run and bundle manifests now record `environment.accelerator` (GPU name,
capability, memory, driver, torch CUDA, cuDNN and determinism flags), supplied by
`metanoia_research.artifacts.accelerator_snapshot` so contracts stay torch-free.

**Compatibility.** torch moved from 2.14.0+cpu to 2.11.0+cu128. CPU-era numerical
results are not expected to be bitwise reproducible; data generation is unaffected.

**Revisit when** a machine needs a GPU arch outside cu128, or CPU-only installs must
avoid the CUDA wheel size (then use per-platform markers or a separate extra).
