# 0002 — CPU-first environment and an offline hashed-token encoder

**Choice.** Python 3.12. PyTorch comes from the CPU wheel index (`[tool.uv.sources]` in
`pyproject.toml`). Stage 1 ships two encoders:
- `hashing`: trainable, needs no download, used for smoke runs and as a control;
- `hf`: a frozen Hugging Face adapter, with ModernBERT-base configured as the first
  candidate.

**Reason.** The first development machine has no GPU (WSL2, 7 GB RAM). Smoke runs
must work offline in under a minute. The `hf` extra is optional, so the default
environment stays small.

**Status.** The `hf` adapter is implemented but has not run yet: `transformers` is not
installed and no checkpoint has been downloaded. Before any promoted experiment:
- pin its `revision` to a commit hash (the resolved revision is recorded in bundles
  anyway);
- compare it with a BERT-family baseline (PLAN.md §3).

**Alternatives.** CUDA wheels from PyPI, which are much larger and useless without a
GPU.

**Revisit when** a GPU is available: switch the torch source per platform and record
accelerator and driver details in run manifests.
