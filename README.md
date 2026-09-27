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
- **Stage 1 is partly implemented.** It includes:
  - a shared, permutation-equivariant candidate scorer;
  - a hashed-token encoder and a frozen Hugging Face adapter (not yet run);
  - an explicit training loop;
  - the controlled micro-world dataset;
  - a calibration split and temperature fitting.
- **Stage 1 has not produced a result yet.** Nothing has been trained with a pretrained
  encoder, and resuming training is not wired up. On `reasoning_v0`, the hashed-token
  direct scorer stays at chance on held-out worlds (see decision 0003).
- **Not started:** memory and retrieval, the reasoning operations (expand, connect,
  compose, aggregate) and budget sweeps.

## Setup

```sh
uv sync --extra research --extra analysis   # CPU PyTorch, pyarrow, scipy, dev tools
uv sync --extra research --extra analysis --extra hf   # + transformers for the hf encoder
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
`decide` and `decide_batch` methods.

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
uv run pytest            # 56 tests, ~5 s (slow-marked learning smoke tests included)
uv run pytest -m "not slow"
uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy
```
