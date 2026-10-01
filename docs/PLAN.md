# Metanoia implementation plan

**Status (2026-09-30, after R020–R027):**
- **Idea.** Given context, a question and candidates, each candidate **expands** into related concepts retrieved from a concept memory. Those expand further, **connect** and **compose**, and their evidence **aggregates** back into each candidate's probability. More budget buys wider and deeper expansion.
- **Stage 0 (the measuring instrument)** is implemented.
- **Stage 1 (the direct response)** works. On realistic reading comprehension (`mcqa_v0`), the fine-tuned direct scorer reaches 0.768 test accuracy, well calibrated (R005). `metanoia decide` and the `interface/` workbench serve it. Seeds are still owed.
- **Stage 2 (concept memory, retrieval, EXPAND)** is **the current work**, on a knowledge suite of science questions without a passage (§10.1).
  - **EXPAND v1 failed** (R016): candidates expanded into WordNet concepts, but the model ignored what the concepts said. Retrieved and random concepts gave identical decisions, flat from 8 to 128 concepts, and 6 points below a fine-tuned direct model (0.581).
  - **Why, and what works** (R017): v1 saw each concept only as a pooled vector dominated by the question. The same reader, given one relevant fact *in its input next to the candidate*, gains 23 points on OpenBookQA with no training. WordNet definitions are weak knowledge for these questions.
  - **EXPAND v2 works, with caveats** (R020, decision 0012). Each candidate reads its own branch of retrieved GenericsKB statements in the trainable reader's input.
    - **The result:** the first budget curve in the project, 0.537 → 0.611 test accuracy as branches grow to about 8 statements per candidate. It beats random statements (+5.5) and the matched direct model (+2.9, NLL −0.045), with intervals clear of zero.
    - **The caveats:** the whole gain is on ARC-Easy (+6.4); OBQA and ARC-Challenge gain nothing; without statements the model is 4.5 points worse than the direct one.
  - **What limits it is the selection of statements, not their reading** (R021, R022).
    - Half the branches never mention their own candidate, and 55–60% of statements are shared across candidates. It is question-topic knowledge more than expansion of the candidate.
    - The same trained model reaches 0.674 on OBQA when its branches hold the gold fact, against 0.608 with retrieval.
    - Anchoring branches on the candidate's own concept makes them specific but less relevant (accuracy −5).
  - **Focused proposals are the best model so far** (R024): statements mentioning the candidate come first. Test 0.652 at 8 per candidate, against 0.581 for the matched direct model (+7.0 [5.6, 8.7], NLL −0.136) and 0.611 for similarity. ARC-Challenge now gains (+4.5); OBQA still does not.
  - **Verdict after round 2** (R026, R027).
    - **The idea works on this suite.** Three independently retrained variants beat the matched direct model; the two with candidate-relevant proposals do so by about 7 points (focused +7.0, fused +6.6 to +7.1), with budget curves that rise to 4–8 statements per candidate.
    - **One-hop proposal tuning on GenericsKB has plateaued.** Fused ties focused, and OBQA has not moved with any method: only about half of its gold facts are in the memory, retrieval finds about half of those, and its questions often need two facts combined.
    - More re-ranking of one-hop proposals would be forcing it; that line stops here (decision 0014, round 3 on one hop is dropped).
  - **Proposed next** (owner's go needed):
    1. **Confirm the result with seeds:** focused and direct, two more seeds each (about 1 h), to make the +7 a promotable claim (H1 on `knowledge_v0`).
    2. **Test recursion where the knowledge exists:** QASC (CC-BY-4.0, 8,134 train questions), where every question has two gold facts that compose. This is the owner's "expand, then expand again, and chain back", with a memory that contains both hops by design.
  - **Earlier plan** (§10.1, step 3c, decision 0014): a portfolio of proposal methods, screened together and composed rather than one forced to work. It includes the owner's relationship-based retrieval (concept paths), rerankers, follow-up search, and a scorer learned from the reader. R023 verified the ingredients: paths and rerankers each need composing.
- **Stage 3** (CONNECT, COMPOSE, AGGREGATE over the expansion tree) follows.
- **A parked track.** A detour (decisions 0007–0009) tested a sentence-attention substitute on synthetic logic puzzles. Its rounds went unused (R012), and its redesign failed feasibility (R013). Those results do **not** test the idea. The track is parked with its assets, for later use with COMPOSE (§10.2).
- **Records.** The code lives in [the project root](../). Every result is recorded in [RESULT_LOG.md](../RESULT_LOG.md), with its files in [`result_logs/`](../result_logs/), and every model has a card in [`model_cards.toml`](../model_cards.toml) (§11, rule 11). Lessons that changed the plan are in §13.

- **Status (2026-10-01): paused.** The newest work is the marker-sequence encoding and a broad multiple-choice mix (R028–R035, decision 0015). The project is not under active development; see the README.

**Main decision:** use Python and PyTorch for the first research implementation, with a clear separation between **the system that makes predictions**, **the experiment runner**, and **the analysis that judges saved predictions**. Use small, explicit interfaces at genuine points of scientific variation. Keep the reference implementation understandable before optimizing it.

## 1. Purpose, priorities, and source documents

Build our own decision model. Given context, a question, arbitrary candidate meanings, optional observations, and an inference budget, it returns candidate probabilities and an optional structured computation trace. Additional computation can use **expand, connect, compose, and aggregate**. Whether those operations improve decisions is an experimental question. Jev is motivation, not a dependency or a training specification.

Preserve the five objectives in their stated order:

| Priority | Practical consequence |
|---|---|
| **1. Code maintainability** | One repository, one initial distribution, clear ownership of responsibilities, limited dependencies, readable control flow. |
| **2. Developer experience** | A reproducible environment, CPU smoke runs, short feedback loops, useful errors, straightforward debugging, inspectable artifacts. |
| **3. Ease of change** | Replace an algorithm behind a documented interface; change an experiment through explicit configuration; avoid edits across unrelated layers. |
| **4. Ease of integration** | A small Python decision interface and versioned, language-neutral result formats; add HTTP only when needed. |
| **5. Scientific result** | Fixed comparison protocols, independent scoring, retained predictions, meaningful baselines, calibrated outputs, and reproducible positive or negative findings. |

Scientific validity is a requirement, not something to trade away because it appears fifth. Prefer the simpler design that still answers the scientific question.

The original concept report, `OPTIONS.md` and the scientific notes are not part of this repository; this plan, `RESULT_LOG.md` and the decision records are.

This plan refines the illustrative layout in `OPTIONS.md`: evaluation and analysis move **outside the inference package**. It also incorporates the earlier proposal's workspace, explicit inference loop, heuristic-first scheduling, staged training, and first-class trace. It replaces single-parent tree assumptions with graph/hypergraph provenance and treats native code as a later profiling decision.

## 2. Separate the model from the measuring instrument

Putting notebooks in another directory is not sufficient. Independence needs **import boundaries, stable data contracts, and versioned experiment protocols**.

### 2.1 Four areas, one initial distribution

| Area | Owns | Must not own |
|---|---|---|
| `metanoia_contracts` | Public requests/results, artifact schemas, budget definitions, manifest validation | Neural implementations, metrics, notebook behavior |
| `metanoia` | Encoder adapters, concept access, workspace, reasoning operations, scheduling, candidate scoring, application of a fitted calibrator | Dataset-specific answers, benchmark definitions, evaluation metrics, report generation |
| `metanoia_research` | Dataset preparation, training, experiment execution, reference-oracle adapters, writing prediction artifacts | A second copy of model equations, official metric definitions, presentation logic |
| `metanoia_analysis` | Artifact loading, metric calculations, calibration fitting, paired comparisons, statistics, tables and figures | Loading model checkpoints, invoking inference, mutating source runs, accessing private workspace tensors |

These are importable Python packages within **one** project, not four services or separately released products. Keep their top-level imports lightweight. Splitting distributions is a later response to a real installation or release need.

Allowed project dependencies:

```text
metanoia                 -> metanoia_contracts
metanoia_research        -> metanoia + metanoia_contracts
metanoia_analysis        -> metanoia_contracts
analysis notebooks      -> metanoia_analysis

metanoia_research -- saved artifacts --> metanoia_analysis
```

`metanoia_contracts` imports none of the other areas. Core inference must not import research or analysis, even indirectly. Analysis must work in an environment without PyTorch, Transformers, or Faiss. An automated import-boundary check and a lightweight analysis smoke run will enforce this.

### 2.2 Three independently changing things

1. **Predictor:** the implementation, trained weights, memory, and inference configuration.
2. **Evaluation protocol:** the exact cases, splits, budgets, batching/timing rules, intervention definitions, expected outputs, and runner behavior.
3. **Analysis:** metric implementations, statistical comparisons, plots, and narrative.

For a model comparison, hold the protocol and scoring revision fixed and change the predictor. For a new analysis, hold prediction artifacts fixed and change the scoring/report revision. A new dataset, intervention, budget policy, or timing procedure is a **new protocol**, not a cosmetic analysis change.

Persist results between collection and scoring. A report should be reproducible from artifacts alone, with no model download, network query, GPU, or hidden notebook state.

### 2.3 What independence can and cannot guarantee

A common schema makes implementations comparable; it does not prove the experiment is fair. Archive the suite, runner revision, dependencies, and scoring revision as well as the model revision. Changing the runner while leaving its configuration unchanged can still change the test.

Initially use one lockfile with optional dependency sets. Record the resolved installed environment for each process. For historical comparisons that require conflicting environments, run each predictor in its recorded checkout/environment and score their exported artifacts with the **same** analysis revision. Do not silently upgrade all old runs to the newest environment.

## 3. Tools selected for the first implementation

Versions are pinned in `metanoia/uv.lock`. They were verified on the first GPU machine: an RTX 2060 with 6 GB, running Arch Linux, with Python 3.12, torch 2.11.0+cu128 and transformers 5.17.0 (decision 0004). The CUDA wheels also run on CPU-only machines. This plan does not assume a particular performance result, but §13 records what fits on a 6 GB GPU.

| Responsibility | Choice | Reason and boundary |
|---|---|---|
| Language/environment | Python, `uv`, `pyproject.toml`, committed `uv.lock` | One development language and a reproducible dependency definition. |
| Neural computation | PyTorch, initially eager execution | Use normal tensor operations and `nn.Module`; keep gradients and debugging visible. |
| Pretrained language understanding | Hugging Face Transformers adapter | Frozen or fine-tuned (`frozen = false`). ModernBERT-base and BERT-base-uncased are both pinned by commit. On realistic data, frozen encoders were insufficient, while fine-tuned ModernBERT with joint candidate encoding works (R005). A frozen encoder is still the right choice wherever a memory index must share its geometry (§7.1). |
| Concept retrieval | Exact normalized-vector search first; Faiss adapter later | Establish reference correctness before adding approximation. Choose index and device from measured recall, memory, and latency. |
| Configuration | TOML plus strict Pydantic validation | Small, readable experiment descriptions; reject unknown keys, invalid values, and unsupported combinations. |
| Interfaces | Python type hints and `typing.Protocol` where useful | Structural contracts without deep inheritance trees. Neural implementations remain ordinary PyTorch modules. |
| Tests/static checks | pytest, Ruff, mypy | One test runner, one formatter/linter, one type checker. |
| Analysis | NumPy, SciPy, PyArrow | Numerical metrics/statistics, temperature fitting, and columnar artifact access. Add another dataframe library only for a demonstrated need. |
| Figures | Matplotlib | Publication/export figures derived from the same recorded numerical results as tables. |
| Interactive analysis | marimo | Reviewable Python notebooks that call the analysis library and load saved artifacts. |
| User entry points | Small CLIs using `argparse` | Thin wrappers around library functions, without a workflow framework initially. |
| Run tracking | Local immutable directories and JSON manifests | No mandatory tracking service; add MLflow or another tracker when shared discovery warrants it. |
| Versioning | Git for source/configuration/small fixtures; hashed manifests for large assets | Trace changes without committing model weights and large indexes into ordinary Git history. |

Use optional dependency sets for `model`, `research`, `analysis`, and later `ann`/`serve`, plus a development group. The contracts layer stays free of GPU dependencies. Public JSON uses strict validation; hot-path tensor states do not pass through Pydantic on every operation.

The selected tools support the intended boundaries: uv distinguishes optional and development dependencies; Pydantic supplies strict validation; Python protocols describe structural interfaces; PyArrow reads and writes Parquet; marimo supplies reactive Python notebooks. See the official [uv](https://docs.astral.sh/uv/concepts/projects/dependencies/), [Pydantic](https://docs.pydantic.dev/latest/concepts/strict_mode/), [typing](https://docs.python.org/3/library/typing.html#typing.Protocol), [PyArrow](https://arrow.apache.org/docs/python/parquet.html), and [marimo](https://docs.marimo.io/) documentation.

**Deferred:** JAX or another neural backend, a web workbench, distributed training, graph databases, general-purpose plugin discovery, learned scheduling, and custom native kernels. Faiss already provides native similarity-search implementations; using Python does not imply that its numerical work runs in Python. [Faiss repository](https://github.com/facebookresearch/faiss).

If profiling later justifies native work, choose C++/CUDA for a suitable tensor/kernel bottleneck or Rust/C++ for stable CPU bookkeeping according to the actual integration need and team expertise. Preserve the Python reference and verify numerical and behavioral parity. Do not create an empty native subsystem now.

## 4. File structure philosophy and proposed layout

Organize by **responsibility and dependency direction**, then by scientific component. Avoid both a monolithic experiment script and a forest of one-method abstractions. Keep one source of truth for model computation and a separate source of truth for official evaluation.

The following is the intended layout as the prototype grows. Files are illustrative responsibility locations, not a requirement to scaffold every entry in the first change. Existing documents stay at their current paths.

**Deviation (decision 0001):** the code tree below (`pyproject.toml`, `src/`, `configs/`, `suites/`, `analysis/`, `tests/`, `docs/decisions/`, `artifacts/`) lives in the `metanoia/` subfolder, separate from these context documents.

```text
metanoia/
├── PLAN.md
├── CLAUDE.md                     # standing instructions for agents (results must be logged)
├── RESULT_LOG.md                 # every experimental result, newest first, never rewritten
├── result_logs/                  # per-entry small files: manifests, predictions, analyses, configs
├── model_cards.toml              # per-bundle capabilities, each claim citing RESULT_LOG entries
├── handoff*.md                   # session handoffs
├── interface/                    # local Next.js workbench over `metanoia serve` (decision 0006)
├── OPTIONS.md
├── SCIENTIFIC_FOUNDATIONS.tex / .pdf
├── SCIENTIFIC_OVERVIEW.tex / .pdf
├── idea.docx
├── README.md                     # setup, shortest working example, document map
├── pyproject.toml
├── uv.lock
├── .python-version
├── src/
│   ├── metanoia_contracts/
│   │   ├── requests.py           # inputs, candidates, budgets, options
│   │   ├── results.py            # predictions, costs, termination status
│   │   ├── artifacts.py          # schemas and manifest validation
│   │   └── traces.py             # versioned event definitions
│   ├── metanoia/
│   │   ├── api.py                # load predictor, decide, decide_batch
│   │   ├── interfaces.py         # internal replaceable-component contracts
│   │   ├── build.py              # explicit factories and compatibility checks
│   │   ├── model.py              # registered learned modules and shared forward
│   │   ├── engine.py             # visible operation scheduling/execution loop
│   │   ├── encoding/             # encoder adapter, pooling, projections
│   │   ├── memory/               # item store and exact/ANN retrieval adapters
│   │   ├── reasoning/
│   │   │   ├── workspace.py      # bounded tensor state and provenance references
│   │   │   ├── expand.py
│   │   │   ├── connect.py
│   │   │   ├── compose.py
│   │   │   └── aggregate.py
│   │   ├── scheduling/           # fixed, random-control, heuristic policies
│   │   ├── decision.py           # shared candidate scorer and readout
│   │   ├── calibration.py        # apply an already fitted calibration artifact
│   │   ├── baselines/            # direct, retrieval pooling, recurrent predictors
│   │   └── diagnostics.py       # supported interventions and trace capture
│   ├── metanoia_research/
│   │   ├── cli.py
│   │   ├── data/                # synthetic worlds, rendering, dataset adapters
│   │   ├── memory_build.py       # versioned vocabulary/vector/index construction
│   │   ├── training/            # explicit loop, losses, curriculum, checkpoints
│   │   ├── runner.py            # fixed-suite execution and result collection
│   │   ├── artifacts.py         # artifact writers; schemas come from contracts
│   │   └── oracles.py           # controlled symbolic reference predictors
│   └── metanoia_analysis/
│       ├── cli.py
│       ├── load.py              # validate artifacts and align example IDs
│       ├── metrics.py           # accuracy, NLL, Brier, calibration summaries
│       ├── statistics.py        # paired differences and uncertainty intervals
│       ├── calibration.py       # fit using calibration-split predictions only
│       ├── compare.py           # compatibility checks and comparison policies
│       ├── traces.py            # analyze saved events; no live workspace access
│       └── plots.py             # reusable figures and report tables
├── configs/
│   ├── models/                  # direct, expansion, full, alternative components
│   ├── training/                # optimizer, stages, seeds, training budgets
│   └── collection/              # predictor/run selections, output location
├── suites/
│   └── reasoning_v1/
│       ├── protocol.toml        # exact evaluation and timing rules
│       ├── data_manifest.json   # immutable case/split content references
│       └── README.md            # hypothesis, labels, limitations, main endpoint
├── analysis/
│   ├── specifications/          # versioned metric/report configurations
│   └── notebooks/              # compare runs, calibration, trace inspection
├── tests/
│   ├── contracts/
│   ├── core/
│   ├── research/
│   ├── analysis/
│   └── fixtures/                # tiny known cases and stored prediction bundles
├── docs/decisions/              # short decisions and reasons to revisit them
├── data/manifests/              # provenance for training and memory assets
└── artifacts/                  # ignored bulk datasets, models, runs, analyses
```

Put reproducible analysis logic in `metanoia_analysis`, not duplicated notebook cells. Notebooks select artifacts, explain questions, and display results. Put exploratory model ideas into small named implementations or temporary research experiments; do not turn a notebook into an unrecorded alternative inference path.

Do not move functions into `utils.py` just because two modules use them. Identify their responsibility and owner. Do not duplicate training forward equations in the runtime to make deployment look cleaner.

## 5. Abstraction where it enables a real experiment

### 5.1 The public boundary

Expose a small predictor interface: **a validated decision request goes in; a validated decision result comes out**. Single-request and batch entry points share semantics. All model baselines implement this boundary, including an oracle on supported controlled tasks.

Public requests/results contain text, IDs, ordinary numbers, and structured metadata—not PyTorch tensors, Faiss handles, or live graph objects. Version this contract. It is the future boundary for a Python client, CLI, HTTP adapter, or a predictor running in another process.

### 5.2 Internal replacement points

| Interface | Responsibility | Initial implementation / useful alternative |
|---|---|---|
| Semantic encoder | Text/observations → token states, masks, pooled states | Frozen pretrained adapter / another checkpoint or pooling method |
| Concept store and retriever | Query keys → proposed IDs, vectors, similarity values | Exact search / Faiss ANN; identity/provenance separated from access method |
| Expander | Proposed concepts + context → selected refinements and states | Small learned contextual scorer / similarity-only or random control |
| Connector | Bounded proposed pairs + context → typed weighted links | Learned pair scorer / fixed or shuffled links |
| Composer | Ordered or typed parent states + context → reusable composite | Small MLP / role-binding or other explicitly tested construction |
| Aggregator | Workspace + context → updated states and evidence for candidates | Bounded message passing / pooled-evidence baseline |
| Scheduler | State summary + remaining resources → next bounded actions | Fixed schedule with exploration / random or heuristic allocation |
| Decision head | Original candidates + gathered evidence → logits | Shared candidate-conditioned scorer, not a fixed label-ID head |
| Calibrator | Logits + fitted artifact → probabilities | Identity temperature / fitted positive scalar temperature |
| Trace sink | Structured events → optional stored diagnostic record | No-op sink / buffered event capture |

Internal interfaces **may use PyTorch tensors** and declare dimensions, masks, dtype/device expectations, gradient behavior, and ordering semantics. Avoid an abstract tensor system intended to support every framework. The workspace should not know which encoder checkpoint or ANN library produced its inputs, but it may naturally own PyTorch tensor state.

Use composition and dependency injection through ordinary constructors. A small explicit factory maps validated configuration names to implementations. Reject unknown names and incompatible dimensions early. Do not start with dynamic plugin loading, arbitrary import paths in configuration, deep class hierarchies, or a dependency-injection framework.

An abstraction earns its place if it isolates an external dependency, supports a named comparison, or protects a stable public contract. A trivial mathematical helper can remain a function. Add capabilities when needed; an unsupported diagnostic operation must fail explicitly, not silently become a no-op.

### 5.3 Component changes have explicit compatibility costs

Changing a composer while preserving its contract should leave the runner and analysis unchanged. It may still require retraining; interchangeable interfaces do not make weights interchangeable. Changing latent dimension, tokenizer, relation vocabulary, or embedding geometry can invalidate checkpoints and memory artifacts. The builder validates these dependencies and refuses an incompatible bundle.

Keep the previous implementation selectable for comparison. Do not overwrite the meaning of an established configuration name. Record defaults in the fully resolved run configuration; mutable library defaults must not redefine an old experiment.

## 6. State, output, and artifact contracts

### 6.1 Decision semantics for v0

Start with an explicitly **exclusive-choice task**: one correct outcome among a declared complete candidate set. Each candidate has a stable ID and descriptive text. The protocol decides how to represent unknown/other outcomes and rejects duplicate IDs; semantic duplicate handling is documented rather than guessed. Multi-label decisions and numeric scores are later, distinct contracts.

A request carries context, question, candidates, optional declared observation fields, budget, trace mode, and reproducibility options. It does not contain the gold label, gold proof, or benchmark-only difficulty annotation. Those remain outside the model input.

A result carries candidate IDs in a documented order, raw logits, probabilities, calibration status/identity, requested and consumed resources, termination reason, predictor identity, and an optional trace reference. The default uncalibrated softmax is clearly marked **uncalibrated**. Schema validity is not a promise of empirical calibration.

### 6.2 Workspace invariants

- Distinguish persistent concept identity from a contextual active instance. Distinguish concepts, assertions, candidates, and composites with explicit types.
- Retain original candidates; pruning expansion work must not delete a candidate from the answer distribution.
- Store multi-parent composition inputs and their roles. A live interaction graph may have cycles; versioned provenance events form an acyclic history of actual computations.
- Keep retrieval similarity, relation/attention weights, candidate logits, and probabilities separate. An active node has no generic “truth probability” unless a specific event and training target justify it.
- Candidate support may cross discovery branches. Track common evidence sources so duplication can be tested rather than mistaken for independent support.
- The engine owns mutation and resource accounting. Neural operations compute updates; state changes must preserve autograd when training. Avoid arbitrary component side effects on unrelated state.
- Keep tensors in the computation path and detached summaries in traces. Trace capture must not change the prediction or random draws.

### 6.3 Budgets and stopping

Use explicit limits for active non-candidate nodes, retrieval queries, proposals scored, pairs scored, retained edges, composites, reasoning rounds, and composition/expansion depth. Reserve candidate-state capacity separately. Define depth as derivation ancestry, not number of recurrent message updates; otherwise lateral cycles make “height” ambiguous.

Depth/budget zero returns the model's zero-round decision without graph work. For a residual workspace, that is exactly the direct scorer. For a model whose reasoning lives in the workspace (decision 0009), it is its readout before any round, which may be near chance: that is an honest statement of what the model knows without deliberating. Every selected action must fit the remaining relevant caps before executing. Charge rejected proposals and pair scoring. Record exhaustion, policy stop, completion, and failure distinctly. Serving deadlines are an optional additional mechanism; initial scientific sweeps use deterministic operation limits plus measured wall time.

### 6.4 Versioned artifacts are the bridge to analysis

| Artifact | Minimum contents |
|---|---|
| Dataset bundle | Input cases, separately stored labels/proofs, stable example IDs, group IDs, split membership, generator/source revision, provenance and content hashes |
| Memory bundle | Item IDs/types/descriptions/sources, vector content hash, encoder/tokenizer/projection/pooling/normalization revisions, dtype and index build metadata |
| Predictor bundle | Component names and resolved configuration, weights, memory reference, supported contract version, optional calibrator reference |
| Collection run | Manifest, expected case/budget/repeat keys, per-example predictions and statuses, costs/timings, optional traces, completion status |
| Calibration bundle | Fitting algorithm/revision, fitted parameters, calibration dataset and predictions, compatible predictor and budget/stopping policy |
| Analysis result | Input run hashes, label/split references, analysis code/configuration versions, selection rules, computed metrics/statistics, figures, explicit limitations |

Use JSON for manifests and small calibration parameters; JSONL for bounded trace events; Parquet for predictions, labels, and cost tables. Keep weights and large vector arrays in appropriate tensor/array formats rather than serializing them into JSON. Do not use pickled live model objects as the scientific exchange format.

A prediction record is keyed by `(run_id, example_id, budget_id, repeat_id)` with explicit candidate IDs and score arrays. Schema checks catch wrong lengths, invalid numbers, duplicate/missing cases, and candidate mismatches. Failure records are retained. The protocol specifies their treatment, and reports always state attempted, completed, and failed counts; do not improve metrics by silently dropping hard cases.

Write runs to a temporary/incomplete state, validate them, then finalize atomically. Never overwrite a completed run. Resume only when its configuration and artifact identities match, without duplicating keys. Trace detail levels and sampling rules are recorded; normal prediction artifacts must remain useful without full tensor dumps.

Record source revision and dirty diff, relevant component revisions/hashes, resolved environment, seeds, accelerator/driver details, and all artifact hashes. A seed alone does not promise identical output across devices or numerical libraries. Define tolerances and distinguish reproducibility of conclusions from bitwise identity.

## 7. Basic execution flow

### 7.1 Prepare data and memory

1. Specify outcome semantics and a controlled task family with independently checkable answers.
2. Split by underlying worlds, rule combinations, and template families before generating derivative variants. Reserve training, model-selection, calibration, and final-test partitions.
3. Produce versioned input and label bundles. Gold derivations can supervise training or oracle diagnostics but are not routine predictor inputs.
4. Build a small concept memory from permitted sources. Encode descriptions with a recorded projection/pooling recipe and validate exact retrieval.
5. Freeze dataset and memory manifests for the first comparison. Begin small; reach 100k–1M concepts only after usefulness is established.

Memory keys and query/composite projections must inhabit a compatible learned space. A dimension match alone is insufficient. Initially freeze the memory encoder; later updates require an explicit refresh/rebuild policy with new artifact identities.

This conflicts with the Stage-1 finding that the encoder must be fine-tuned to be useful (R001 vs R005). Stage 2 must choose, and record in a decision, one of three options:
- a frozen memory encoder with a learned query projection;
- rebuilding memory keys with the fine-tuned encoder, which gives the memory a new identity for each predictor;
- a separate retrieval encoder.

Do not attach a memory index built with one encoder to a predictor fine-tuned from another.

### 7.2 Train

1. Load a validated training configuration, declared training data, and memory bundle.
2. Build the same model components and computation path used at inference.
3. Start with the direct candidate scorer, then add retrieval/expansion, connections, composition, and recursive budgets in separate experimental stages.
4. Train using proper prediction loss and justified auxiliary supervision. Use a transparent scheduler first; state which parameters receive gradients.
5. Evaluate development checkpoints on the model-selection split. Save resolved configuration, optimizer/scheduler state, seeds/RNG state, and weights needed to resume.
6. Select the checkpoint without final-test labels. Produce calibration-split predictions and fit calibration separately before final evaluation.
7. Record the training wall-clock time and throughput in the bundle. R005 took about 5 h, but that time was never recorded.

Practical lessons from Stage 1 (§13):
- **Smoke runs.** Use an easy source, such as SciQ, to check that the pipeline learns at all. 150 steps on RACE could not tell a broken setup from a slow start (R003), although the full run learned (R005).
- **Initialization and curriculum are part of the training protocol.** From plain ModernBERT, the direct scorer collapses to uniform output on `reasoning_v0`, even with 20k examples or a depth-1-only curriculum (R007, R009). Warm-started from the reading-comprehension model R005, it learns within about 150 steps (R008). Record the initialization (`init_from`, stored as `training.initialized_from`). Give every model compared on a suite the same initialization and curriculum, or report the difference (foundations §13.4).
- **Train to the plateau before comparing.** Longer, padded inputs delay learning. R011 started learning around step 150, against step 75 in R008, and was cut off while still improving. Give a reference baseline enough steps and check the convergence gate (§10) before comparing anything with it.
- **Watch for collapse, not just loss.** A training loss flat at ln N together with near-identical candidate logits (per-example logit standard deviation near 0) means the model has collapsed to uniform. That is different from slow learning. The epoch log records the dev logit spread for this reason.
- **Throughput.** Fine-tuning with joint candidate encoding costs 1 + N encoder passes per request. The training options exist for this: fp16 mixed precision, gradient accumulation, gradient checkpointing, warmup with linear decay, and a separate encoder learning rate. All of them default to the original behaviour.

Hard ANN and top-k choices are not silently “end-to-end differentiable.” Train downstream selected tensor operations initially; proposals or selection policies need separate supervision/relaxation/policy learning if they are to learn. Do not decorate the shared forward path with unconditional inference-only gradient suppression; inference callers select evaluation mode and disable gradients at the boundary.

The trainer may track optimization and validation losses for debugging. Official checkpoint comparisons use saved development predictions and the independent analysis process. Feed the chosen checkpoint ID back as an explicit selection artifact; do not add analysis imports or a second official metric implementation to the trainer.

### 7.3 Decide on one input

```text
validate request + predictor/memory compatibility
  -> encode context/question/observations and candidate texts
  -> keep token states and masks; compute direct candidate logits
  -> initialize bounded workspace with original candidate nodes
  -> repeat while policy continues and resources remain:
       choose bounded expansion targets, retaining some exploration
       retrieve proposals and score contextual refinements       [EXPAND]
       propose bounded pairs and score typed interactions        [CONNECT]
       select parent groups and create reusable composite states [COMPOSE]
       propagate messages and update candidate evidence/logits   [AGGREGATE]
       charge actual work and emit optional provenance events
  -> apply the declared fitted calibration, or identity temperature
  -> return original candidate IDs, scores/probabilities, costs, and status
```

This is the initial explicit schedule, not a requirement that every later algorithm use the same internal graph. Alternative predictors can implement the public decision contract. Preserve the reference loop for diagnosis.

Re-attending to cached context avoids another encoder pass but still costs computation. Batch tensor work where practical; do not score every node pair merely to discard most pairs afterward. No heuristic may inspect ground truth. Random policies use explicit per-example/repeat randomness, with tie behavior defined so candidate ordering does not become an accidental feature.

### 7.4 Collect a fixed-suite run

1. Resolve the suite version, input case hashes, predictor bundle, requested budgets, and runner revision before starting.
2. Validate capabilities and compatibility. The collection runner passes **only inference inputs**, not gold answers, to the predictor.
3. Execute the same declared cases, batching, repeat/seed policy, and timing procedure for each variant. Persist raw outputs and actual resource use.
4. For fixed-budget curves, independently run the specified budgets by default. Intermediate snapshots from one larger run are diagnostic unless their equivalence to separate-budget executions has been established; a scheduler may act differently when it knows its total budget.
5. Validate expected keys and finalize the artifact. Do not calculate official model rankings inside the runtime.

Standardize end-to-end timing in the runner, including candidate encoding and device synchronization. Keep diagnostic full tracing separate from throughput measurements, with trace overhead reported. Count graph operations inside the engine. Record cold and warm behavior separately and disclose cache policies.

### 7.5 Fit calibration and analyze

The analysis process reads saved predictions, then joins labels by stable IDs. It validates protocol compatibility, computes metrics and paired differences, and writes a **new** analysis artifact. It does not need the live model.

Calibration fitting reads only the calibration partition. The first method is a positive scalar temperature. Save raw logits so alternative calibration can be compared without rerunning inference. Applying the fitted parameters in core must agree numerically with offline application.

Always report calibration **per slice** as well as overall. A single global temperature can help the whole set and harm an easy slice. In R004, calibrated ECE on SciQ rose from 0.020 to 0.135. Slice-conditional calibration, such as a temperature per source, belongs to a new analysis revision. It is legitimate only if the slice key is available at inference time. A benchmark-only annotation cannot drive it.

A temperature fit at its search bound means the model has no usable signal (R000–R001). Report it as such; it is not a successful calibration.

An analysis-only calibration change can re-score fixed predictions. If calibrated confidence influences stopping or action selection, changing it changes the predictor policy and requires a new collection run. Offline reweighting cannot reconstruct an inference trajectory that was never executed.

Notebooks import these analysis functions. They must not redefine the official metrics or launch hidden training/inference as a side effect of loading a report.

## 8. Freeze comparisons without freezing research

### 8.1 What makes a suite stable

The suite records task semantics; input/label content; split and group identities; candidate and duplicate policy; budget grid; repeats/seeds; batching and cache rules; failure handling; trace settings; supported interventions; timing scope; and the primary scientific question. It points to a versioned analysis specification defining metrics, calibration policy, aggregation, uncertainty intervals, and the planned main comparison.

Promoted suites are immutable. Correct a discovered bug in a new revision and document which conclusions it affects. Do not preserve a known-bad metric merely to maintain a version number. Re-score **all** relevant saved runs with the corrected analysis, or recollect them if the protocol or missing data demands it.

### 8.2 Three common change workflows

| Change | Edit | Reuse | Recompute |
|---|---|---|---|
| Test a new composer | Composer implementation + new model/training configuration | Suite, labels, runner protocol, analysis revision | Training as needed; predictions; same analysis on old and new runs |
| Add a calibration plot or metric | Analysis code + new analysis specification | Existing prediction/logit artifacts and labels | Analysis only, provided all required information was saved |
| Add a harder dataset, graph intervention, or new budget policy | New suite revision; adapter/diagnostic capability only if necessary | Compatible predictors and old artifacts as historical records | New collections for the new cases/policy; all compared variants |

The comparison tool rejects mismatched cases, label semantics, or budgets by default. An intentional cross-protocol comparison must name the mismatch and its interpretation. Never silently take the intersection of successful examples. Model-specific preprocessing is part of the predictor identity; benchmark-specific label handling is part of the protocol.

A stored trace supports inspection of what happened. It is not automatically sufficient to rerun counterfactual graph interventions; those normally require the archived predictor and a new diagnostic execution through an explicit supported interface.

### 8.3 Required scientific comparisons

Preserve direct scoring, retrieval pooling, and a recurrent latent baseline before drawing conclusions about composition. On realistic suites, always include the lexical-overlap fixture as a shortcut baseline, with results per source. It solves 82% of SciQ with no learning (R002), so accuracy on easy sources is weak evidence on its own. The current reference points for later stages are:
- on realistic data, the fine-tuned direct scorer R005;
- on rule reasoning, the warm-started direct scorer R008.

A reasoning mechanism must beat them at matched compute, training data, initialization and curriculum, not merely beat chance. On controlled suites, report accuracy and calibration **per held-out depth**: a mechanism's claimed benefit is better extrapolation to longer derivations, and a pooled score would hide it. Calibration fitted on in-distribution depths can worsen held-out depths (R008). Report both raw and calibrated values there. Add random/similarity-only expansion, fixed/shuffled connections, no-composition, and equal-cost extra-message-passing controls. Include an oracle for controlled tasks and later multi-hop memory/contextual graph baselines as described in the foundations report.

Compare both native end-to-end retrieval and an explicitly evidence-matched condition. Distinguish disabling an operation in a trained model from retraining a model without it. Match or disclose training data, auxiliary supervision, parameter count, tuning effort, and compute. Repeating the same deterministic classifier is not a stronger baseline.

Report accuracy, NLL, Brier score, reliability diagrams/ECE with declared bins, risk–coverage, both directions of decision reversal, latency distributions, throughput, and memory. Use multiple training seeds and paired uncertainty estimates; resample independent worlds/groups when examples are correlated. Predeclare the main metric and practically relevant improvement before final testing. These policies come from the scientific notes (not included), §13.

## 9. Verification and daily developer workflow

### 9.1 Keep three forms of verification distinct

| Kind | Examples | When |
|---|---|---|
| Software correctness | Shapes/masks, budgets, candidate-ID alignment, schema migration, import boundaries, serialization | Fast local checks and normal code-change CI |
| Learning/integration smoke checks | A tiny overfit task, expected gradients, train/infer compatibility, saved-run round trip, analysis without ML dependencies | Relevant model/training/artifact changes |
| Scientific evaluation | Held-out compositions, matched baselines, multiple seeds, calibration and performance curves | Explicit experiment milestones; not every edit |

Metric tests use hand-computable probabilities and known counterexamples, independently of the training loss implementation. Test duplicate evidence, role reversal, candidate permutations, padding, empty retrieval, exhausted budgets, invalid inputs, trace-on/off equivalence, and stale memory rejection. Test the analysis loader with saved fixtures so model changes cannot redefine what a metric test means.

For model changes, verify meaningful gradients and tiny-task learning; shape checks alone are insufficient. Numerical tolerances and deterministic test modes are explicit. Do not write broad golden-output tests that freeze every floating-point detail across devices.

### 9.2 User workflow

The command vocabulary makes the separation obvious: **prepare → train → collect → calibrate → analyze → compare**. These commands are implemented; `metanoia/README.md` has working examples for both suites.

- `metanoia-research prepare` builds versioned task and memory artifacts.
- `metanoia-research train` trains a configured predictor and saves a bundle.
- `metanoia-research collect` runs a named suite with that bundle and writes predictions.
- `metanoia-analysis calibrate` fits from calibration predictions and writes parameters.
- `metanoia-analysis analyze` scores an existing collection against its labels/specification.
- `metanoia-analysis compare` checks compatibility and compares named analyses/runs.
- `metanoia decide` asks a trained bundle for candidate probabilities for a single request, given as flags or JSON. It can apply a fitted calibration. It is the user-facing entry point and belongs to the core package. `metanoia serve` keeps a bundle loaded and answers the same requests as JSON lines on stdin and stdout. The local workbench in `interface/` uses it to show each candidate's probability and the recorded path.

After every analysis or comparison that produces a result, add an entry to `RESULT_LOG.md` and copy the small files to `result_logs/` (§11, rule 11).

Every command takes explicit configuration/artifact identifiers and writes its resolved inputs. Do not depend on a mutable “latest” checkpoint in a promoted experiment. Cheap smoke commands should run locally after environment setup; substantive downloads/training remain separate and visible.

On the owner's machine:
- An agent may run a command expected to finish within about 5–10 minutes itself.
- A longer command (full training, large sweeps) is handed to the owner as exact commands, with an estimated duration and a stop criterion. For example: "stop if the step loss is still ≈ ln 4 after about 1,000 steps".
- Never run two GPU jobs at once on the 6 GB GPU.
- Commit before a run that is meant to be recorded, so its manifests name a clean source revision. R005 was recorded against a dirty tree.

## 10. Implementation milestones and completion gates

Stage 0 is **implemented**.

Stage 1's listed gate is **met**:
- tiny-task learning, meaningful gradients and candidate symmetry are covered by tests;
- bundles reload with verified hashes;
- raw probability metrics and a calibration split exist;
- one direct baseline is genuinely useful: R005 on `mcqa_v0`.

Before any Stage-1 result is promoted, one item remains: at least three training seeds.
Two earlier items are done:
- dev-based checkpoint selection: `collect-checkpoints` → `metanoia-analysis select`, which writes a selection artifact;
- training resume: `train --resume`, which reproduces identical weights on CPU.

Later stages are **not started**.

The Stage-1 deliverable changed from "frozen encoder adapter" to "frozen **or fine-tuned** encoder adapter" (decision 0005, §13).

### 10.1 Current path: candidate expansion (decision 0010)

Stage 2 first, then Stage 3, in this order. Each step is a vertical slice with a gate.

1. **Knowledge suite `knowledge_v0`.** **Done** (R014): the lexical fixture scores 0.251, so there is no shortcut. OpenBookQA and ARC (Easy and Challenge): science multiple choice without a passage.
   - The direct model is knowledge-limited here, so retrieved concepts can add evidence. This follows lesson 13: the extra computation must be needed.
   - Validation: the lexical fixture, the direct baseline (the R005 recipe), and an oracle-free check that labels are sound.
   - OpenBookQA's gold core fact is kept as a proof annotation: it never enters inference, and it is used to measure retrieval recall.
2. **Concept memory v0.** **Done** (R015): novel gold-fact words are recalled at 19% and 26% with 16 and 32 concepts per candidate, against 2% and 4% at random. WordNet synsets (name plus gloss, about 117k).
   - Keys come from a **separate frozen sentence-embedding encoder**; exact search first (PLAN §7.1 option 3).
   - It is versioned as a memory bundle: IDs, texts, a vector hash, and the encoder revision.
   - **Gate: access recall.** Retrieval from question plus candidate must surface the gold core fact's concepts at a useful rate before any expansion is trained.
3. **EXPAND, one level, v1.** **Done, negative** (R016, decision 0011).
   - The design: a frozen R005 reader, each concept read with the question into a pooled vector, and a trained head aggregating across branches.
   - The outcome: retrieved = random = no concepts (evidence correlated at 0.99); flat over budgets; below the fine-tuned direct model (0.519 against 0.581).
   - R017 locates the failure in the *reading*, not the idea. Pooled "concept + input" vectors barely depend on the concept, while reading a relevant fact jointly lifts OpenBookQA by 23 points. WordNet glosses are weak knowledge (+2 points at best).
   - v1 models stay available in the interface, with "not useful" cards.
3b. **EXPAND, one level, v2** (decision 0012): **done, positive with caveats** (R020, R021).
   - **Result:** test accuracy 0.537 → 0.611 from 0 to 32 statements (NLL 1.099 → 0.953). Against the matched direct model: +2.9 points [1.4, 4.6], NLL −0.045. Against random statements: +5.5.
   - **Caveats:**
     - the gain is all ARC-Easy (+6.4); OBQA −0.8, ARC-Challenge −2.3;
     - at budget 0 it is 4.5 points worse than the direct model;
     - one seed.
   - **The model uses per-candidate content** (R021): swapping branches removes the gain.
   - **Selection is weak** (R021): at 2 per candidate, half the branches never mention their candidate, and 55–60% of statements are shared between branches.
   - **Interface:** it serves `branch_v2_retrieved_knowledge_v0` as the recommended model for passage-free science questions, with a calibration at 16 statements.
   - **Memory `generics_v0`:** **built** (R018). GenericsKB-Best, 1,015,610 unique statements, each keyed by its concept term (CC-BY-4.0), with the same frozen retrieval encoder.
   - **Gate (a), access recall:** **a tie** with `wordnet_v0` (R018). Novel fact words at 4 per candidate: 31% against 30%. Word coverage cannot tell a definition from a needed relation.
   - **Gate (b), reading probe:** **passed** (R018). In EXPAND v2's own input format, the untrained R005 reader does better with retrieved statements than with random ones (+10–13 points) or none (up to +4), and better than with WordNet (+4–5).
   - **Weakness seen in R018:** retrieval follows the question's topic, so a question's branches often hold the same statements. If v2 works, candidate-focused proposals are the next lever.
   - **Model:** branch reading. Each candidate's nodes are rendered into its own reader input; the reader is trainable (from R005); the width is sampled per batch from {0, 1, 2, 4, 8}.
   - **Matched controls:** random statements; empty branches (the direct model on the same schedule).
   - **Budget grid:** 0, 1, 2, 4, 8 nodes per candidate, in `knowledge_v0` revision 3.
   - **Trace:** per candidate, its nodes; per node, its leave-one-out contribution.
3c. **Proposals that relate the candidate to the question** (decisions 0013, 0014): **current step.**
   - **Why:** reading is not the bottleneck; choosing what each candidate expands into is (R021: gold fact 0.674 against 0.608 retrieved on OBQA).
   - **Tried without retraining** (R022):
     - anchoring on the candidate's own concept makes branches specific (92% mention their candidate, overlap 0.09) but less relevant: accuracy 0.556 against 0.610, and OBQA gold recall 0.19 against 0.25;
     - a naive second hop does not help;
     - **focused** reordering of the "question + candidate" pool (statements mentioning the candidate first) keeps relevance: 85% on-candidate, overlap 0.11–0.17, gold-fact recall equal or higher, accuracy within noise;
     - removing statements shared across branches hurts: topic knowledge carries signal.
   - **Focused, retrained: done** (R024). Test 0.652 at p32: +7.0 over direct, +4.1 over similarity; ARC-Challenge +4.5 over direct; OBQA flat. It is the reference the portfolio must beat (G1 pool + J1 judge), and the second trained reader for screening.
   - **Plan: a portfolio of proposal methods, composed and screened together** (decision 0014). No single method is backed alone.
     - **Generators:**
       - G1 similarity;
       - G2 anchored on the candidate's concept;
       - G3 **concept paths** from candidate to question or context concepts, one or two hops (the owner's relationship-based retrieval);
       - G4 follow-up search conditioned on the first statement;
       - G5 context sentences as graph nodes (passage suites);
       - G6 generated bridges (a later comparison).
     - **Judges:**
       - J0 similarity;
       - J1 focused;
       - J2 reranker (bge-reranker-v2-m3);
       - J3 the trained reader's leave-one-out effect (a teacher);
       - J4 a scorer learned from J3 (decision 0013);
       - rank fusions of judges.
     - **Selection:** top-k per candidate; statements shared by all candidates move to a shared question context; paths stop when they reach the question.
   - **Verified before building** (R023):
     - the concept graph is cheap (19 s index), but path existence barely separates gold from distractors (one hop: 48% against 43%), so paths generate and do not judge;
     - common concepts ("energy", "water") must be down-weighted, not capped;
     - off-the-shelf rerankers are not better than similarity at the top of the list (gold fact in the top 2: 10–25% against 33%), though v2-m3 is candidate-sensitive and may add to similarity through fusion.
   - **Screening protocol**, fixed in decision 0014:
     - dev + calibration only;
     - per configuration at 2 and 8 per candidate: OBQA gold-fact recall, on-candidate rate, overlap, accuracy and NLL per source under *both* trained v2 readers, and cost;
     - promote at most two per round to a matched retraining, if gold recall rises by at least 0.03 without losing more than 2 points, or OBQA + ARC-Challenge accuracy rises by at least 2 points;
     - test only for promoted methods.
   - **Round 1: done** (R025).
     - **Promoted:** similarity + reranker fusion, G1/RRF(J0, J2): +2.5 points on OBQA + ARC-Challenge under the focused reader.
     - **Concept paths** (G3) gave the most candidate-specific branches (95% on-candidate) but no accuracy or recall gain: lexical concepts confuse senses, and path existence does not judge (R023).
     - Follow-up search (G4) adds nothing.
     - Implemented as `proposals = "fused"`, with a pinned reranker and a verified proposal cache (`build-proposals`).
   - **Round 2** (owner): `scripts/stage2_expand_v2_fused.sh`, about 2 h (cache about 80 min, then training and collection), matched to R020 and R024.
   - **Round 2: done** (R026). Fused **ties** focused on test (p32 −0.4 [−1.6, +0.9]); both are about +7 over direct. The R025 screening gain did not replicate.
   - **Stopped** (R026, R027): further one-hop proposal tuning on GenericsKB. OBQA's limit is memory coverage plus two-fact composition, which one-hop re-ranking cannot fix.
   - **Still open, now aimed at recursion:** sense-aware concept paths and a judge trained on the reader's leave-one-out labels. They return as the way to choose each hop in step 4, not as one-hop re-rankers. Also open: `knowledge_v0` r4 budget accounting.
   - **Rounds:**
     1. screen G1, G3, G4 × J0, J1, J2 and fusions;
     2. promote and retrain;
     3. J3 labels and J4 over the union pool;
     4. recursion with stopping (step 4);
     5. G6 once.
   - **Engineering:**
     - the concept graph becomes a versioned memory artifact;
     - the reranker is pinned like the retrieval encoder;
     - a suite revision `knowledge_v0` r4 separates pool size (`max_proposals`), statements read (`max_nodes`) and hops (`max_depth`), because today the focused pool of 64 is not counted in usage;
     - traces show each branch as its path and the judge that chose it.
4. **Recursive EXPAND** (depth ≥ 2: a node's concept becomes a new query, so a branch grows into a chain), with **adaptive stopping** (expand deeper only while the decision is uncertain), then **CONNECT** (links across branches) and **COMPOSE** (combined concepts). Each is added as a selectable, ablatable operation, with budget sweeps over width and depth.
5. **The central H1 test.** On held-out questions, accuracy and NLL improve with the expansion budget, beat the random-concept control at equal cost, and beat the direct model.

### 10.2 Parked track: controlled logic puzzles (decisions 0007–0009)

Kept for later, not abandoned. It will be used as a controlled test of COMPOSE and derivations once expansion exists.
- **Assets:**
  - suites `reasoning_v0`, `reasoning_v1` and `reasoning_v1` revision 2 (a budget grid);
  - the generator (revision 2);
  - the converged direct baseline (R012);
  - the workspace code (0008: residual; 0009: isolated encoding), all selectable.
- **Findings:**
  - A cold start collapses, and a warm start from R005 learns (R007–R009).
  - The direct model solves the trained depths and fails to extrapolate while staying overconfident (R011, R012).
  - A residual workspace stays unused when the direct path fits the training data (R012).
  - Small modules trained from scratch cannot learn transferable matching over nonce symbols (R013).
- **Open fork, when this track resumes:**
  - (a) iterated single-step inference with the pretrained reader (derived statements as nodes);
  - (b) token-level reads with a curriculum;
  - (c) a deeper-chain curriculum for the direct model.

Complete useful vertical slices, not all directories at once. Each stage leaves a runnable, reviewable result and records its remaining limitations.

| Stage | Deliverable | Completion gate |
|---|---|---|
| **0. Contracts and comparison skeleton** | Environment, import areas, schemas, tiny deterministic predictor fixture, minimal suite, independent artifact scorer | Produce and re-analyze a saved run; changing the predictor fixture leaves analysis unchanged; analysis runs without ML dependencies |
| **1. Direct decision baseline** | Frozen or fine-tuned encoder adapter, shared arbitrary-candidate scorer, explicit training loop, controlled and realistic datasets | Tiny-task learning, meaningful gradients, candidate symmetry, checkpoint reload, raw probability metrics and calibration split |
| **2. Memory and retrieval** (current, §10.1) | Small versioned concept store (WordNet v0) with a separate frozen retrieval encoder, exact retriever, contextual EXPAND from candidates, pooled-retrieval and random-concept baselines, on the knowledge suite | Relevant-item recall measured; alignment/cache checks; same suite works unchanged; retrieval benefit separated from graph benefit |
| **3. Explicit reasoning** over the expansion tree (a detour built a sentence workspace early; it is parked, §10.2) | Workspace, aggregation, then connections and bounded composition, first-class trace | Each operation selectable and ablatable; budget accounting includes rejected work; multi-parent provenance and cross-candidate support verified |
| **4. Frozen-model compute study** | Budget sweeps, recurrent/memory/graph controls as appropriate, held-out compositions, explanation interventions | Reproducible comparisons, both reversal directions, uncertainty intervals, calibrated/uncalibrated results, documented positive or negative outcome |
| **5. Scale and allocation** | ANN adapter, larger memory, batching/selected compilation; learned scheduling only as a separate study | Approximation and optimization preserve acceptable quality; gains survive real resource accounting; scheduler beats simpler allocation if claimed |
| **6. Integration and targeted optimization** | Thin HTTP boundary or trace workbench if useful (a local workbench exists early, in `interface/`, over the stdin/stdout `metanoia serve` worker; decision 0006); measured native optimization if needed | Core and analysis stay independent; public compatibility and parity tests pass; end-to-end benefit demonstrated |

Scaling and product polish should follow evidence. If composition does not beat the matched alternatives, revise the mechanism or retain the simpler model; do not manufacture a positive result by changing the test suite.

## 11. Working rules for humans and LLM agents

1. **Read this plan and the relevant scientific sections before editing.** Identify whether the task concerns core behavior, data/protocol, training, or analysis. Follow the user's current scope; this document is not permission to implement later stages unprompted.
2. **State the hypothesis and affected boundary.** For algorithmic work, name the baseline and what remains fixed. For analysis work, name the input artifacts and analysis revision.
3. **Make the smallest coherent change.** Add a selectable implementation for a real experiment; avoid unrelated refactors and speculative frameworks.
4. **Respect dependency direction.** No core-to-analysis imports, no private-tensor access from notebooks, no benchmark answers in inference inputs, no duplicated forward implementation.
5. **Preserve history.** New model variants, protocols, and analysis revisions get distinct identities. Never edit completed run outputs or silently relax a failing evaluation to favor a model.
6. **Update contracts deliberately.** Breaking schema changes need a new version and an explicit reader/migration strategy. Preserve old raw artifacts. Do not pretend an interface-compatible model is checkpoint-compatible.
7. **Verify at the relevant level.** Run affected correctness and smoke checks; run expensive scientific studies when the task requires them. Report what was and was not measured.
8. **Keep claims tied to artifacts.** A trace is not automatically a proof, a normalized score is not automatically calibrated, and more compute is not automatically better reasoning.
9. **Record material decisions.** A short note states the choice, reason, alternatives, compatibility impact, and evidence that would justify revisiting it. Keep `PLAN.md` status consistent with actual completed work.
10. **Finish with a reviewable result.** State changed responsibilities, tests, artifact IDs, observed outcomes, and unresolved limitations. Do not label a milestone complete while its completion gate is unmet.
11. **Record every result in `RESULT_LOG.md`, including negative and aborted ones.**
    - **Entries.** Give each entry the next `Rnnn` ID and list it in the summary table. An entry states:
      - the question;
      - the predictor ID, configuration, data and hardware;
      - the numbers, with intervals;
      - its interpretation;
      - its limitations.
    - **Files.** Copy the small files behind each result into `result_logs/Rnnn_<date>_<slug>/`, with a `SHA256SUMS` file: bundle manifest, training log, run manifest, predictions, calibration, analysis and configs. Weights and datasets stay in `metanoia/artifacts/`.
    - **Corrections.** Never rewrite a past entry. Add a new entry that supersedes it.
    - **Timing.** Log each result as soon as it exists, not at the end of a session.
    - **Unarchived observations.** An observation without an artifact, such as an ad-hoc diagnostic script, is still logged and marked "not archived".

## 12. Open choices that do not block this plan

Resolve these during the relevant implementation milestone, with measurements rather than additional speculative abstractions.

**Resolved in Stage 1** (decisions 0002–0005):
- **Environment:** RTX 2060 6 GB, CUDA 12.8 wheels, Python 3.12.
- **First controlled task:** `reasoning_v0` Horn-rule micro-worlds.
- **Realistic dataset:** `mcqa_v0`, RACE + SciQ.
- **Working encoder:** fine-tuned ModernBERT-base, mean pooling, joint candidate encoding, 384 tokens.

**Still open:**
- **Retrieval geometry:** resolved for now. A separate frozen retrieval encoder (bge-small) keys the memory (decision 0010), and the decision model's reader is free to be fine-tuned.
- **What a memory node holds:** definitions (WordNet) proved weak (R017). Decision 0012 moves to statements about concepts. Whether a node should keep both, and how recursion picks the next concept, is open.
- **How nodes are read:** pooled per-node vectors from a frozen reader failed (R016). Branch reading (decision 0012) costs one longer encoder pass per candidate. Per-node joint reading, which is costlier but gives a cleaner tree, remains an option.
- **`reasoning_v0` difficulty:** resolved. A warm-started direct scorer solves the trained depths (R008), so the study moves to held-out depths in `reasoning_v1`. Binary relations, role binding and negation (for H2) remain for a later revision.
- **Cold-start training:** why ModernBERT plus the scorer collapses on shortcut-free data. A depth-0 curriculum, a different learning rate, CLS or span pooling, or a warm start are candidate explanations. Decide whether the warm start stays the standard protocol, or a cold-start recipe is found.
- **Calibration under difficulty shift (H5):** a model perfect in-distribution gets a sharpening temperature (T = 0.07) that is catastrophic at unseen depths (R012). Candidate remedies: a held-out-depth calibration split (it would change the suite), abstention or uncertainty from the budget curve, or ensembles. This blocks any H5 claim.
- **Calibration method:** whether one global temperature is enough, or a slice-conditional calibrator is needed.
- **Candidate encoding:** a cheaper option for large candidate sets. Joint encoding costs one encoder pass per candidate.
- **Language and licensing:** model and license suitability, and the intended language coverage. Both current datasets are English.
- **Stage-2 settings:** latent dimensions, resource caps, and the training budget.
- **Primary endpoint:** the exact endpoint and the smallest useful effect for the first promoted scientific study.
- **Explanations:** whether they stop at faithful execution traces or require typed, checkable derivations. The latter needs additional grounding and rule work.

The immediate target is EXPAND v2 on the knowledge suite (§10.1, step 3b): the statement memory and its two gates, then branch reading with its controls. Seeds for R005 remain owed before promotion. **The instrument exists and has useful direct baselines; expansion must beat them, and a random-concept control, not merely beat chance.**

## 13. Lessons from Stage 1

Each lesson cites its evidence in [RESULT_LOG.md](../RESULT_LOG.md).

1. **Frozen encoders are not enough, even for the direct decision** (R000, R001, R005).
   - Hashed, frozen ModernBERT and frozen BERT direct scorers all stay at chance on `reasoning_v0`. They overfit the training worlds.
   - On realistic data, the same scorer over a fine-tuned encoder reaches 0.768 accuracy.
   - Consequence: fine-tuning is the default for useful predictors. "Frozen" survives only where retrieval geometry requires it.
2. **Shortcut baselines are strong** (R002). Word overlap solves 82% of SciQ. Report results per source and compare against the lexical fixture.
3. **Short runs can mislead** (R003, R005, R006). A run that did not move in 150 steps learned over 6,217. Check the pipeline on an easy source, and give long runs a stop criterion rather than a verdict drawn from a smoke run. A fine-tuned model sitting at ln 4 on its *training* loss means "not yet learned", not "cannot learn". For R006, R007 and R009 it turned out to mean collapse (lesson 8).
4. **Calibration must be checked per slice** (R004). A global temperature improved the overall metrics but harmed the easy slice. The well-trained model (R005) needed almost no temperature (T = 1.06).
5. **Record timing and source.**
   - R005's training time was lost, because the trainer logged no wall-clock time. Bundles and step logs now record `wall_time_seconds` and examples/s (first used in R006).
   - R005 was recorded against uncommitted code, so its manifests do not name a clean source revision.
   - Log timings in bundles, and commit before recorded runs.
6. **Resource envelope (RTX 2060, 6 GB).**
   - Fine-tuning ModernBERT-base with joint encoding at 384 tokens, with fp16 and gradient checkpointing, runs at about 5.5 training examples/s and uses about 3.5 GB of GPU memory.
   - A full `mcqa_v0` epoch takes about 5 h. fp16 inference runs at about 21 examples/s.
   - Micro-batch 4 at 512 tokens without checkpointing runs out of memory.
   - Concurrent GPU jobs cause out-of-memory failures.
   - A fine-tuned bundle is about 1.7 GB: 0.6 GB of weights and 1.2 GB of optimizer state.
7. **Keep the measuring instrument.** Every result above was scored from saved predictions by `metanoia_analysis`, with no model code involved. Paired comparison with both reversal directions worked unchanged on a new suite. The Stage-0 investment paid off, so keep that boundary intact.
8. **"At chance" can mean "collapsed", not "cannot"** (R007–R009). A cold-start model trained on shortcut-free data sat on the uniform solution: its training loss stayed at ln 4 and its logits were identical. More data (20k examples) and an easier curriculum (depth 1) did not help. A warm start from a reading-comprehension model solved it in 150 steps. Check a model's logit spread before concluding that an architecture cannot learn a task.
9. **A direct scorer solves in-distribution rule chains; extrapolation is the gap** (R008). Trained on depths 1–3 it scores ≈100% there, 0.786 at depth 4, and stays confident when wrong (ECE 0.15). The length control shows depth, not input length, drives most of the drop.
10. **Calibration does not transfer to unseen difficulty** (R008). A temperature fitted on in-distribution depths sharpened the output and worsened depth-4 NLL, from 0.80 to 1.00. H5 has to be tested on held-out depths.
11. **Simple English deductions already work** for R005 and R008 (qualitative, R008). Reported deduction failures need to be collected as concrete examples before being attributed to missing reasoning operations.
12. **Padding with distractor text slows learning; an unconverged baseline flatters any mechanism** (R011). With the same recipe as R008, 46-sentence inputs gave 0.61–0.74 on trained depths after 300 steps, against about 1.0 unpadded. Its depth curve (0.74 falling to 0.40) mixes a training deficit with an extrapolation deficit.
13. **Extra computation is only learned if it is needed** (R012). A residual workspace on top of a direct path that already fits the training data stays switched off: readout ‖w‖ ≈ 0.07 against logit spreads of 25–30, and identical answers at 0 and 16 rounds. Design the model so the budgeted operations are the only route to the answer, or train on problems a single pass cannot solve.
14. **Perfect in-distribution accuracy breaks temperature calibration** (R012). The fitted temperature of 0.07 turned an already overconfident model (NLL 4.4 at depth 6) into a catastrophic one (NLL 16.7). Never ship a calibration without checking it on shifted slices. The interface now applies only calibrations kept in `artifacts/calibrations/`.
15. **Learned relational matching over nonce symbols needs the pretrained reader** (R013). Small recurrent modules (an isolated workspace, a memory network) trained from scratch either collapsed or memorized world-specific symbols, even at trained depths. Build budgeted computation around the component that already reads rules, not beside it.
16. **Keep the owner's idea as the thing under test** (decision 0010). A scientifically convenient substitute (decision 0007's sentence workspace on logic puzzles) consumed a night of compute and tested something the owner does not consider the idea. Reordering stages or replacing a core operation needs explicit agreement, and status reports are written in the idea's own terms.
17. **A pooled vector of "item + input" does not carry the item** (R016, R017). The frozen reader's pooled encodings of different concepts on the same input were more alike (cosine 0.93) than encodings of different inputs (0.89). A head over them learned to ignore the concepts entirely. Put what must be compared in the same input, read by a reader that is trained for it.
18. **Test the knowledge before the mechanism** (R017). An evaluation-only probe, with the knowledge pasted into the input of an existing reader, took 20 minutes. It showed which knowledge helps (a relevant fact: +23 points) and which does not (definitions: +2). Run such a probe before training any mechanism that consumes a memory.
19. **Small knowledge-suite fine-tuning overfits in one epoch** (R016). On 8k questions, dev NLL was best after epoch 1 (0.99) and 2.45 by epoch 3. Keep dev selection, and let the interface serve the dev-selected checkpoint (a card's `checkpoint` field).
20. **Expansion helps when the reader reads the expansion itself** (R020). Branch reading with no extra weights gave the first budget curve: 0.537 → 0.611, above random statements and the matched direct model. Aggregate per-source numbers before claiming a general gain, though: all of it came from ARC-Easy.
21. **Check what the branches contain, not only the score** (R021, R022). A positive budget curve hid branches that were mostly question-topic statements, shared across candidates, and half of them never mentioned their candidate. Swapping, sharing and gold-fact oracle conditions tell use apart from selection quality in minutes, without training.
22. **An expansion must relate the candidate to the question** (R022). Statements about the candidate alone (anchored) are specific but irrelevant; statements about the question alone are relevant but shared. Proposals come from the intersection, and what counts as useful should eventually be learned from the reader (decision 0013).
23. **Try several methods and compose them; do not force one to work** (owner, decision 0014; R023). Feasibility checks showed each promising ingredient alone is weak: paths do not judge, and rerankers do not rank short facts well. Screen alternatives under one fixed protocol, combine generators and judges, and promote only what the protocol selects.
24. **Screening gains of 2–3 points need replication before they count** (R025, R026). The fused ordering led focused by 2.5 points on 798 dev questions and tied it on test after retraining. Keep promotion rules, but expect regression to the mean near the noise floor.
25. **Know the ceiling before optimizing toward it** (R021, R027). The gold fact lifts OBQA only to 0.674, and only half the gold facts are in the memory. Measure what the memory holds and what the reader can use before tuning the step in between.
26. **Encode all options in one sequence** (R029, R031, R035). Marker encoding trains 3× faster than joint encoding at 512 against 384 tokens, and on the matched quick run it is +2.6 points [1.3, 3.9] with equal calibrated NLL; with four-rotation averaging +5.2 points.
27. **A format that sees all options is order-sensitive, and the obvious fix fails** (R032, R033). The marker model changed its answer with option order on 67% of questions at 150 steps. Resetting all option positions to the same ids collapses (markers cannot find their own tokens, cosine 0.9999). Hashed positions derived from option text cut the dependence to 27% at the same accuracy. Measure with the permutation check, not with identical options (which hashing flatters).
28. **Check a diagnostic before building on it** (R032). The reset variant looked like a clean idea and an "equivariant" score (identical options gave identical logits) while learning nothing. An invariance that holds because the model is constant is not a result.
29. **Confidence is informative only where the model is competent** (R034, R035). Dev-fitted thresholds transfer to test on SciQ (89–91% coverage at about 96% precision) and select almost nothing on RACE, where the quick model has not learned. A tree triggered by confidence needs a System 1 that is competent on the source.
