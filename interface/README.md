# Metanoia interface

A local workbench for trained Metanoia predictors. You enter a context, a question and
candidates, and it shows each candidate's probability and the path the predictor took:
render → encode → score → calibrate.

It only uses the public decision contract, through `metanoia serve` (see
`../src/metanoia/cli.py`). For deployment, see the Docker setup in the project README.

## Run

```sh
pnpm install
pnpm dev          # http://localhost:3000
```

**Requirements.** The Python environment must be set up in the parent folder
(`uv sync --extra research --extra analysis --extra hf`), and at least one trained
bundle must exist under `../artifacts/predictors/`.

**How models are found.**
- Models are read from the artifacts directory each time the page loads.
- A calibration is applied automatically when one was fitted for exactly that
  predictor (matched on `compatible_predictor_id`).

**Model cards.** Each model shows its card from `../model_cards.toml`: status,
headline result, strengths, weaknesses and what makes it different, grounded in
`../RESULT_LOG.md`. A card appears only if its `predictor_id` matches the bundle, so a
retrained model never inherits an old description.

**Timing.** The first request loads the model, which takes about 10 s on CPU. Later
requests take about 1–2 s on CPU.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `METANOIA_ROOT` | `..` | The Python project |
| `METANOIA_BIN` | `<root>/.venv/bin/metanoia` | The `metanoia` command |
| `METANOIA_DEVICE` | `cpu` | `cpu`, `cuda` or `auto` |
| `METANOIA_PRECISION` | `fp32` | `fp32` or `fp16` (use `fp16` with `cuda`) |
| `METANOIA_DEFAULT_MODEL` | `mbft_joint_mcqa_v0` | Model listed first |
| `METANOIA_TIMEOUT_MS` | `120000` | Per-request timeout |
| `METANOIA_MODEL_CARDS` | `../model_cards.toml` | Model card file |

The device defaults to `cpu` so the interface never competes with a training run for
the 6 GB GPU. When no training is running, `METANOIA_DEVICE=cuda
METANOIA_PRECISION=fp16 pnpm dev` is much faster.

## How it works

- **`src/app/page.tsx`** is a Server Component. It lists models from
  `artifacts/predictors` and passes only names and metadata to the client, never
  filesystem paths.
- **`src/app/api/decide/route.ts`** is a Route Handler.
  - It validates input with zod.
  - It accepts only listed model names.
  - It builds a `DecisionRequest` with candidate IDs `a`, `b`, `c`, … and trace events
    enabled.
- **`src/lib/metanoia/worker.ts`** (server-only) keeps one `metanoia serve` process
  warm. The process speaks JSON lines over stdin and stdout; its replies are validated
  against the contract schemas in `contract.ts`.
  - Switching model replaces the process.
  - A crash surfaces its stderr as the error message.
- **`src/components/`** holds the client UI, styled after IBM Carbon (Plex fonts,
  Carbon White and Gray 100 tokens). The Path view renders known trace operations
  explicitly, and any other operation as raw details, so traces from later reasoning
  stages display without changes.

## Checks

```sh
pnpm lint            # Biome (format + lint)
pnpm exec tsc --noEmit
pnpm build
```
