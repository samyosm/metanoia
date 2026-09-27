# Metanoia (code)

A decision model that does not generate text. It takes context, a question, arbitrary
candidate answers and a compute budget. It returns candidate probabilities and an
optional trace. An evaluation instrument separate from the model judges its saved
predictions.

This folder holds the code. The design documents live one level up:
- [`../PLAN.md`](../PLAN.md): the implementation plan, boundaries and milestones;
- [`../SCIENTIFIC_FOUNDATIONS.pdf`](../SCIENTIFIC_FOUNDATIONS.pdf): the scientific
  constraints;
- [`../OPTIONS.md`](../OPTIONS.md) and [`../idea.docx`](../idea.docx): background.

Local decisions are recorded in [`docs/decisions/`](docs/decisions/).

## Status

- **Stage 0 is implemented.** It includes:
  - contracts;
  - a deterministic fixture predictor;
  - a minimal suite;
  - an independent artifact scorer;
  - import-boundary checks;
  - analysis that runs without ML dependencies.
- **Stage 1 is mostly implemented.** It includes:
  - a shared, permutation-equivariant candidate scorer;
  - hashed-token, frozen and fine-tunable Hugging Face encoders, with `separate` or
    `joint` (candidate + input) candidate encoding;
  - an explicit training loop (fp16 mixed precision, accumulation, warmup/linear LR);
  - the controlled micro-world suite `reasoning_v0` and the realistic RACE + SciQ suite
    `mcqa_v0`;
  - a calibration split and temperature fitting;
  - `metanoia decide`, a CLI that returns candidate probabilities for one request.
- **Results so far** (one seed, untuned; see [`../RESULT_LOG.md`](../RESULT_LOG.md)):
  every frozen or hashed direct scorer stays at chance on `reasoning_v0` (R000–R001).
  On `mcqa_v0`, fine-tuned ModernBERT with joint candidate encoding reaches 0.768 test
  accuracy and NLL 0.608, with calibration error 0.016 (R005); the lexical fixture gets
  0.399.
- **Missing from Stage 1:** multiple training seeds. Checkpointing, resume and dev-based
  checkpoint selection exist (see below). Every result is logged in
  [`../RESULT_LOG.md`](../RESULT_LOG.md).
- **Not started:** memory and retrieval, the reasoning operations (expand, connect,
  compose, aggregate) and budget sweeps.

## Setup

```sh
uv sync --extra research --extra analysis --extra hf   # CUDA 12.8 PyTorch (decision 0004)
```

Run every command from this folder. Paths in the configurations are relative to it.

## Shortest working example

```python
from metanoia.baselines.lexical import LexicalOverlapPredictor
from metanoia_contracts import Candidate, DecisionRequest

request = DecisionRequest(
    context="Kalo is blicket. Everything that is blicket is also dax. Venmi is wug.",
    question="Which one is dax?",
    candidates=(Candidate(id="kalo", text="Kalo"), Candidate(id="venmi", text="Venmi")),
)
print(LexicalOverlapPredictor().decide(request).probabilities)
```

A trained bundle loads with `metanoia.api.load_predictor(path)`, which offers the same
`decide` and `decide_batch` methods. From the shell:

```sh
.venv/bin/metanoia decide --bundle artifacts/predictors/mbft_joint_sciq_quick \
    --calibration artifacts/calibrations/mbft_joint_sciq_quick_b0.json \
    --context "Photosynthesis takes place in the chloroplasts of plant cells." \
    --question "Where does photosynthesis happen?" \
    --candidate mitochondria --candidate chloroplasts --candidate nucleus
# --context-file PATH, --request request.json (or - for stdin), --json, --trace,
# --device cpu|cuda|auto, --precision fp32|fp16
```

Probabilities are marked `uncalibrated` unless `--calibration` names a temperature
fitted for that predictor on its calibration split.

`metanoia serve --bundle … [--calibration …] [--device …]` keeps a bundle loaded and
answers JSON lines on stdin and stdout: `{"id", "request"}` in, `{"id", "result"}` or
`{"id", "error"}` out, after a `{"ready"}` line. The web workbench in
[`../interface`](../interface/) uses it to show each candidate's probability and the
trace (render → encode → score → calibrate).

## Workflow: prepare → train → collect → calibrate → analyze → compare

```sh
R=.venv/bin/metanoia-research; A=.venv/bin/metanoia-analysis
D=artifacts/datasets/microworld_v0; S=analysis/specifications/core_v0.toml

$R prepare --config configs/data/microworld_v0.toml          # must reproduce the suite's pinned hashes
$R train   --config configs/training/direct_hashing_v0.toml  # ~1 min on CPU
$R collect --config configs/collection/direct_hashing_v0_reasoning_v0.toml
$R collect --config configs/collection/lexical_reasoning_v0.toml
$R collect --config configs/collection/oracle_reasoning_v0.toml

$A calibrate --run artifacts/runs/direct_hashing_v0_reasoning_v0 --dataset $D --budget b0 \
             --out artifacts/calibrations/direct_hashing_v0_b0.json
$A analyze --run artifacts/runs/direct_hashing_v0_reasoning_v0 --dataset $D --spec $S \
           --calibration artifacts/calibrations/direct_hashing_v0_b0.json --figures
$A compare --run-a artifacts/runs/lexical_reasoning_v0 \
           --run-b artifacts/runs/direct_hashing_v0_reasoning_v0 --dataset $D --spec $S
```

- Every output directory is new. Existing runs and analyses are never overwritten.
- Runs are written to `<run>.incomplete/`, checked against the protocol's expected keys,
  and then renamed.
- The runner passes only request fields to the predictor. It never reads labels.

### Realistic suite (`mcqa_v0`) with a fine-tuned encoder

```sh
D=artifacts/datasets/mcqa_v0; S=analysis/specifications/mcqa_v0.toml
$R prepare --config configs/data/mcqa_v0.toml       # downloads pinned RACE + SciQ
$R prepare --config configs/data/mcqa_sciq_v0.toml  # SciQ only, for the quick predictor
$R train   --config configs/training/mbft_joint_sciq_quick.toml   # ~8 min, RTX 2060
$R train   --config configs/training/mbft_joint_mcqa_v0.toml      # ~5 h, full RACE + SciQ
$R collect --config configs/collection/mbft_joint_mcqa_v0_mcqa_v0.toml   # ~10 min
$A calibrate --run artifacts/runs/mbft_joint_mcqa_v0_mcqa_v0 --dataset $D --budget b0 \
             --out artifacts/calibrations/mbft_joint_mcqa_v0_b0.json
$A analyze --run artifacts/runs/mbft_joint_mcqa_v0_mcqa_v0 --dataset $D --spec $S \
           --calibration artifacts/calibrations/mbft_joint_mcqa_v0_b0.json
```

### Checkpoints, resume and selection

Set `checkpoint_every_steps = N` or `checkpoint_each_epoch = true` in a training config.
Each checkpoint `<bundle>/checkpoints/step_NNNNNNN/` is a loadable predictor bundle; only
the latest keeps optimizer state. An interrupted run keeps `<bundle>.incomplete/`:

```sh
$R train --config CONFIG --resume      # same config and dataset, checked
$R collect-checkpoints --bundle artifacts/predictors/NAME --suite suites/SUITE \
    --dataset $D --splits dev --device cuda --precision fp16
$A select --run artifacts/runs/NAME__dev__step_0000100 --run artifacts/runs/NAME__dev__final \
    --dataset $D --split dev --budget b0 --metric nll     # writes artifacts/selections/*.json
```

Then collect the chosen checkpoint on the full suite with an ordinary collection config
whose `path` is the chosen checkpoint directory. `select` refuses the `test` and
`calibration` splits. Dev-only runs record `suite.splits`, so `compare` never mixes them
with full-protocol runs.

### Stage 3a: the budgeted workspace (decision 0008)

A model config with a `[workspace]` table adds a reasoning workspace on top of the
direct scorer. Its nodes are the input's sentences plus the candidates, updated for R
rounds by one shared attention block.
- `mode = "latent"` is the equal-cost control: learned slots instead of sentences.
- The request budget's `max_rounds` sets R. Budget 0 is exactly the direct scorer.
- Training samples R from `train_rounds` with deep supervision.
- Traces contain one `aggregate` event per round: candidate logits and most-attended
  sentences.

```sh
scripts/stage3a_reasoning_v1.sh     # ~8 h: baseline + statement workspace + latent control,
                                    # each trained, dev-selected and collected on the
                                    # reasoning_v1 revision-2 budget grid (0-16 rounds)
.venv/bin/metanoia decide --bundle artifacts/predictors/NAME --rounds 8 ...
```

Collection options for this workflow:
- `collect --from-selection <artifact>` collects exactly the checkpoint a selection
  chose.
- `collect-checkpoints --budgets r4` collects on a subset of the protocol's budgets
  (recorded in the run's suite identity).

## Layout and boundaries

| Package | Owns | May import |
|---|---|---|
| `metanoia_contracts` | requests, results, traces, artifact schemas, suite protocols, provenance | Pydantic, stdlib |
| `metanoia` | encoders, candidate scorer, model, calibration application, public API, baselines | contracts, PyTorch |
| `metanoia_research` | micro-world data, bundle IO, training, collection runner, oracle | core, contracts |
| `metanoia_analysis` | loading and validation, metrics, statistics, calibration fitting, comparison, plots | contracts, NumPy/SciPy/PyArrow |

Two tests enforce these boundaries:
- `tests/test_boundaries.py` scans the import statements;
- a subprocess test runs the analysis CLI with `torch`, `transformers` and `metanoia`
  blocked.

Other locations:
- `configs/` holds data, model, training and collection settings;
- `suites/` holds immutable evaluation protocols;
- `analysis/specifications/` holds versioned metric specifications;
- `tests/fixtures/` holds stored runs that analysis tests read without any model code.

## Checks

```sh
uv run pytest            # 68 tests, ~10 s (slow-marked learning smoke tests included)
uv run pytest -m "not slow"
uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy
```
