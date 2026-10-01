# Result log

This log lists every experimental result, positive or negative, newest first. Entries
are never rewritten. If a result turns out to be wrong, a new entry corrects it and
names the entry it supersedes.

## Where the files live

- Each entry `Rnnn` has a folder `result_logs/Rnnn_<date>_<slug>/`. It holds the small
  files behind the entry: bundle manifests, training logs, run manifests, raw
  predictions, calibrations, analyses, the configs used, and a `SHA256SUMS` file.
- Large files (weights, optimizer state, datasets) stay in the gitignored
  `metanoia/artifacts/`, under the same names. They can be identified by the hashes in
  the archived manifests.
- Unless stated otherwise:
  - every number comes from the archived `analysis.json`;
  - intervals are 95% percentile bootstraps over groups;
  - "calibrated" means one temperature fitted on the calibration split only.

## Summary

| ID | Date | Suite | Predictor | Test acc | Test NLL (cal.) | Verdict |
|---|---|---|---|---|---|---|
| R035 | 2026-10-01 | `mcqa_v0` | seq_sciq_quick with 4-rotation averaging | **0.478** | **1.159** | Rotation averaging on the full test set: +2.6 [1.6, 3.6] over the same weights without it, calibrated NLL 1.159 (R031: 1.186; joint R004: 1.196), 4× the inference compute. Total +5.2 over the joint model |
| R034 | 2026-10-01 | `mcqa_v0` test | R031 predictions, selective metrics (spec r2) | 0.452 | – | New analysis revision: confidence AUROC 0.706; **SciQ answers 89% of questions at ≥95% precision** (threshold from dev), RACE essentially none. Confidence is informative where the model is competent |
| R033 | 2026-10-01 | `mcqa_v0` test (400-question sample) | seq_modernbert_base_hash quick (hashed option positions) | 0.440 (order as given) | – | **Hashed positions cut order dependence from 67% to 27%** of questions at the same accuracy (0.440 vs 0.433) and learn normally; not yet at the 2% bar. Quick run, one seed |
| R032 | 2026-10-01 | `mcqa_v0` test (400-question sample) | seq_modernbert_base quick; position-reset variant | 0.433 (order as given); 0.480 rotation-averaged | – | **Marker model is strongly order-sensitive** (answer changes with order on 67% of questions; rotation averaging +4.7 points). **Position-reset variant collapses** (4 epochs of evidence: loss ln 4): resets make an option's marker unable to find its own tokens |
| R031 | 2026-10-01 | `mcqa_v0` | seq_modernbert_base, 2.4k SciQ, 1 epoch (matched to R004) | **0.452** | **1.186** | Marker encoding **beats joint encoding on the matched quick run**: +2.6 [1.3, 3.9] accuracy, equal calibrated NLL (1.186 vs 1.196), 5× faster to train. One seed |
| R030 | 2026-10-01 | `mix_v1` (new) | lexical overlap fixture | 0.314 (macro over all 23,118) | 7.1 (raw) | Built the 10-source mix (355k train, law and medicine included); every source near chance for the fixture except SciQ (0.820); BoolQ passages leak across train/test, CaseHOLD has 5.6% duplicate option sets |
| R029 | 2026-10-01 | RACE validation (probe, 88 questions) | seq_modernbert_base (untrained) vs direct_modernbert_ft_joint | – | – | Marker encoding trains 3.0× faster than joint (17.9 vs 5.9 examples/s) at 3.3 GB; ModernBERT's sdpa path honours `position_ids`. Pipeline only |
| R028 | 2026-10-01 | candidate sources (metadata + validation statistics) | – | – | – | Step 0 of PLAN_V3: general reading (RACE 87.9k) and law (CaseHOLD 45k, CC-BY-4.0) are the largest native multiple-choice sources with a passage; QuALITY is the only long-document set (2.5k train); lexical shortcuts are weak on all three |
| R027 | 2026-09-30 | `knowledge_v0` OBQA dev (diagnostic) | Are OBQA's gold facts in `generics_v0`, and do branches contain them? | – | – | About half are in the memory (49% at similarity ≥ 0.9); focused branches hold about half of those (24%). OBQA is limited by memory coverage *and* retrieval |
| R026 | 2026-09-30 | `knowledge_v0` r3 | EXPAND v2 with fused proposals (round 2 of decision 0014), matched to R020/R024 | 0.652 (p16) / 0.647 (p32) | 0.874 / 0.867 (raw) | **Ties focused** (−0.4 [−1.6, +0.9] at p32); +6.6 to +7.1 over direct. The screening gain did not replicate; one-hop proposal tuning has plateaued |
| R025 | 2026-09-29 | `knowledge_v0` dev + calibration (screening) | Decision 0014 round 1: 14 generator × judge configurations, read by both trained v2 readers | best 0.651–0.657 vs focused 0.633–0.648 | – | **Promoted: similarity + reranker fusion** (+2.5 on OBQA + ARC-Challenge). Concept paths: most candidate-specific, but no accuracy or recall gain; follow-up search adds nothing |
| R024 | 2026-09-29 | `knowledge_v0` r3 | EXPAND v2 with focused proposals (decision 0013), matched to R020 | **0.652** (p32) | **0.862** (raw) | **Best model so far**: +7.0 over the matched direct model, +4.1 over similarity proposals; ARC-Challenge now gains (+4.5); OBQA still flat |
| R023 | 2026-09-29 | `knowledge_v0` dev + calibration (CPU checks) | Plan verification: concept graph over `generics_v0`, and off-the-shelf rerankers | – | – | Graph is cheap (19 s index) but path existence barely separates gold from distractors (1-hop 48% vs 43%); rerankers are no free win (gold fact top-2: 10–25% vs 33% by similarity). Methods must be composed |
| R022 | 2026-09-29 | `knowledge_v0` dev + calibration (probe) | Proposal rules read by the trained v2 model: anchored, chained, focused, distinct | 0.556–0.591 vs 0.610 (k = 2) | – | Anchoring on the candidate loses relevance; **focused** keeps it (gold-fact recall 0.28 vs 0.25, 85% on-candidate) within noise of accuracy; a retrain is needed to judge |
| R021 | 2026-09-29 | `knowledge_v0` dev + calibration (diagnostic) | What the trained EXPAND v2 model takes from its branches | own 0.614 vs empty 0.562, swapped 0.571 | – | Branch content is used; **selection is weak**: half the branches never mention their candidate, 55–60% of statements shared; gold fact as branch: OBQA 0.674 |
| R020 | 2026-09-29 | `knowledge_v0` r3 | Stage 2 matched: EXPAND v2 branch reading, retrieved / random / none | **0.611** / 0.556 / 0.581 (p32) | 0.953 / 1.049 / 0.998 (raw) | **First positive budget curve**: 0.537 → 0.611 with statements; +2.9 over the direct control, +5.5 over random; the gain is all ARC-Easy |
| R019 | 2026-09-29 | `knowledge_v0` (480 train) | EXPAND v2 smoke: branch reading at width 8, 30 steps | – | dev 1.210 (96 ex.) | Fits the GPU (4.6 GB peak), about 15 examples/s at the widest branch; loss falls. Pipeline only |
| R018 | 2026-09-29 | `knowledge_v0` dev + calibration (probe) | Memory `generics_v0` (1.02M statements) and the two gates of decision 0012 | – | – | Gate (a) a tie with WordNet; **gate (b) passed**: branch reading with retrieved statements beats random ones by 10–13 points and WordNet by 4–5 |
| R017 | 2026-09-29 | `knowledge_v0` test (diagnostic) | Knowledge read *jointly* as context by R005 and by the R016 direct model, no training | OBQA 0.388 → **0.618** with the gold fact (R005) | – | Joint reading uses a relevant fact (+23 points, no training); WordNet glosses help little (+1 to +4) and hurt a reader fine-tuned without context |
| R016 | 2026-09-29 | `knowledge_v0` r2 | Stage 2 matched: direct fine-tuned; EXPAND v1 retrieved / random / none | 0.581 / 0.519 / 0.519 / 0.516 | 1.010 / 1.129 / 1.131 / 1.140 (raw) | **Negative for EXPAND v1**: retrieved = random (evidence 99.7% correlated); the gain over R005 is the head, not the concepts; flat over budgets 8–128 |
| R015 | 2026-09-28 | `knowledge_v0` (train+dev) | Concept memory `wordnet_v0`: access recall of gold-fact words | – | – | Novel fact words at 16/32 per candidate: 19% / 26% (random 2% / 4%) |
| R014 | 2026-09-28 | `knowledge_v0` | Suite validation: lexical fixture | 0.251 | – | No lexical shortcut without a passage |
| R013 | 2026-09-28 | micro-worlds (diagnostic) | Isolated-encoding workspace (decision 0009) and a memory-network probe | chance | – | **Fails feasibility**: collapses or memorizes; no transfer to new worlds |
| R012 | 2026-09-28 | `reasoning_v1` r2 | Stage 3a matched: converged direct baseline, statement workspace, latent control | 0.753 / 0.778 / 0.759 | 1.97 / 1.49 / 1.25 (raw) | **Rounds change nothing (0–16)**: the workspace went unused; the baseline converged (depth 4 0.91, depth 8 0.53) but is badly overconfident |
| R011 | 2026-09-27 | `reasoning_v1` | Warm-started direct baseline, 640 tokens, 300 steps | 0.510 | 1.071 | Depth curve 0.74→0.40, but **undertrained** (depths 1–3 only 0.61–0.74) |
| R010 | 2026-09-27 | `reasoning_v1` | Suite validation: symbolic oracle and lexical fixture | 1.000 / 0.250 | – | Labels agree with text; no lexical shortcut at any depth |
| R009 | 2026-09-27 | `reasoning_v0` (dev) | Fine-tuned ModernBERT, cold start, depth-1-only curriculum (300 steps) | – | dev 1.386 | Collapsed like R007: an easy curriculum alone is not enough |
| R008 | 2026-09-27 | `reasoning_v0` | Fine-tuned ModernBERT, **warm start from R005**, 4.8k puzzles (300 steps) | **0.942** | 0.270 | Depths 1–3 ≈ 100%; **held-out depth 4: 0.786, overconfident** |
| R007 | 2026-09-27 | `reasoning_v0` | Fine-tuned ModernBERT, joint, 20k fresh train worlds (3,750 steps) | 0.269 | 1.386 | Chance; **collapsed to uniform**; training loss never moved |
| R006 | 2026-09-27 | `reasoning_v0` | Fine-tuned ModernBERT, joint, 3 epochs (339 steps) | 0.259 | 1.386 | Chance; does not fit train; inconclusive |
| R005 | 2026-09-27 | `mcqa_v0` | Fine-tuned ModernBERT, joint, full data | **0.768** | **0.608** | Works; current best |
| R004 | 2026-09-27 | `mcqa_v0` | Fine-tuned ModernBERT, joint, 2.4k SciQ | 0.426 | 1.196 | SciQ only; RACE ≈ lexical |
| R003 | 2026-09-27 | (dev only) | Fine-tuned ModernBERT, 2.4k mixed | – | dev 1.386 | No learning in 150 steps |
| R002 | 2026-09-27 | `mcqa_v0` | Lexical overlap fixture | 0.399 | 7.42 (raw) | Strong shortcut on SciQ |
| R001 | 2026-09-27 | `reasoning_v0` | Frozen ModernBERT / frozen BERT | 0.258 / 0.233 | 1.386 / 1.386 | Chance |
| R000 | 2026-09-26 | `reasoning_v0` | Hashed-token direct scorer | 0.219 | 1.388 | Chance (CPU era) |

Chance on both suites is 0.25 accuracy and NLL ln 4 = 1.386.

---

## R035: rotation averaging on the full test set (decision 0015)

- **Folder:** `result_logs/R035_2026-10-01_marker-rotation-averaging-sciq-quick/`
  (run manifest and predictions, calibration, selective analysis, paired comparison with
  R031, collection config, `SHA256SUMS`).
- **Question.** R032 saw +4.7 points from averaging over option rotations on 400 questions.
  Does it hold on the full `mcqa_v0` test set, and how does it affect calibration?
- **Predictor.** The same weights as R031 (`seq_modernbert_base@ee3ced731194`, bundle
  `seq_sciq_quick`), run with `rotations = 4` (mean log-probability over the four cyclic
  rotations). The identity is `…+rot4`, so its calibration is separate (temperature fitted on
  the 2,990 calibration questions: 3.006, against 3.546 without rotations).
- **Setup.** Same suite and splits as R031 (11,802 questions). RTX 2060, fp16. Collection
  took 10 m 33 s, 3.5× the unrotated 3 minutes. Uncommitted code on `5f52b54`.

| test (n = 5,922) | R004 joint | R031 marker | R035 marker, 4 rotations |
|---|---|---|---|
| accuracy, all | 0.426 | 0.452 | **0.478** [0.465, 0.492] |
| accuracy, RACE / SciQ | 0.328 / 0.910 | 0.360 / 0.908 | **0.389** / 0.922 |
| NLL raw / calibrated | 1.352 / 1.196 | 1.642 / 1.186 | 1.459 / **1.159** |
| ECE raw / calibrated | 0.155 / 0.041 | 0.261 / 0.047 | 0.192 / **0.028** |

- Paired, R035 against R031 (same weights): accuracy **+0.026 [0.016, 0.036]**, raw NLL
  −0.183 [−0.202, −0.165], wrong-to-right 8.8%, right-to-wrong 6.1%.
- Selective (dev-fitted thresholds, calibrated): SciQ answers 91% of questions at 96% precision
  (target 95%); RACE answers about 1%; AUROC all 0.726, RACE 0.63, SciQ 0.96.

**Interpretation.**
- **Order averaging is worth it for this model:** +2.6 points and better calibration on the
  whole test set (smaller than R032's 400-question +4.7, which was a noisy estimate), for
  about 4× the inference cost. Against the joint model of R004, the marker model with four
  rotations is +5.2 points, with calibrated NLL 1.159 against 1.196.
- **It does not cure order dependence:** it hides it by symmetrising, as Laya's rotations do.
  Hashed positions (R033) attack the cause. They can be combined with rotations.
- **Cost accounting.** Marker with 4 rotations costs 4 passes of 512 tokens, about the cost
  of joint encoding's N + 1 = 5 passes of 384 tokens; the gain is the accuracy, not speed.

**Limitations.** One quick model (150 steps on 2,400 SciQ examples); one seed; at full
training the order dependence, and so the benefit of averaging, will change.

---

## R034: selective metrics, first look (PLAN_V3 A4)

- **Folder:** `result_logs/R034_2026-10-01_selective-metrics-first-look/` (analysis, spec,
  the metric module, `SHA256SUMS`).
- **Question.** What do coverage at a precision target, confidence AUROC and accuracy per
  confidence tertile say about R031's quick marker model? (New analysis revision `mcqa_v0`
  r2; the run and its predictions are R031's, unchanged.)
- **Method.** `metanoia_analysis/selective.py`. Confidence is the top probability. For each
  target precision, the threshold is fitted **on the dev split** (per slice) and applied to
  test ("transfer", honest); "oracle" thresholds fitted on test itself are in the archived
  analysis and are optimistic. Tertile edges are also fixed on dev. Calibration (R031's
  temperature 3.55) is monotone, so it cannot change rankings; it moves thresholds.
- **Results** (test, raw probabilities, transfer thresholds):

| slice | n | AUROC | coverage at 0.95 precision (achieved precision) | accuracy, least / middle / most confident third |
|---|---|---|---|---|
| all | 5,922 | 0.706 | 0.159 (0.940) | 0.31 / 0.36 / 0.69 |
| RACE | 4,931 | 0.582 | 0.002 (1.00) | 0.30 / 0.35 / 0.44 |
| SciQ | 991 | 0.961 | **0.894 (0.967)** | 0.71 / 1.00 / 1.00 |

  Calibrated variant: AUROC 0.708 overall; SciQ coverage at 0.95 is 0.902 (0.963).

**Interpretation.**
- **Confidence tracks competence.** On SciQ, which the model was trained on, it answers
  89% of questions at 96.7% precision (target 95%), and its least-confident third is
  still 71% right. On RACE it has not learned the task (R031), and confidence barely
  separates right from wrong (AUROC 0.58). The overall AUROC 0.71 mixes these two.
- **The metrics work as an instrument:** thresholds fixed on dev transfer to test with
  achieved precision close to the target on SciQ (0.967 against 0.95). On the weak slice they
  select almost nothing, which is the right behaviour.
- **For the plan.** These are the numbers on which the tree's trigger and Gate C are built:
  coverage at 95% precision and accuracy in the least-confident third. A model that is
  mostly wrong has no useful third to improve; the tree should be tested on a System 1
  that is competent on the source.

**Limitations.** One quick model; achieved precision at a threshold is noisy when coverage is
tiny (RACE: 0.2% of 4,931 questions); AUROC has no interval yet; thresholds are fitted on
a dev split of 2,890 questions.

---

## R033: hashed option positions, quick run (decision 0015)

- **Folder:** `result_logs/R033_2026-10-01_hashed-positions-quick/`.
- **Question.** R032 found that the marker model is order-sensitive and that resetting every
  option to the same position cannot learn. Does giving each option a start position
  derived from its own text (stable hash into a range of 512, so positions do not depend on
  the presented order but options stay distinguishable) lower the order dependence without
  losing accuracy?
- **Predictor.** `seq_modernbert_base_hash@d8699e6866a9`: R031's recipe (2,400 SciQ examples,
  1 epoch, seed 0, 150 steps, 86 s) with `position_hash = true`, `position_hash_range = 512`.
  Uncommitted code on `85cd462`; RTX 2060.
- **Check.** The same 400 `mcqa_v0` test questions and checks as R032.

| 400 test questions | R032 sequential positions | R033 hashed positions |
|---|---|---|
| answer changes across 8 orders | 67.3% | **26.5%** |
| correct in all 8 orders | 0.230 | **0.345** |
| accuracy, order as given | 0.433 | 0.440 |
| accuracy, 4-rotation average | 0.480 | 0.428 |
| mean top probability | 0.721 | 0.594 |
| training / dev loss | 0.40 / 0.266 | 0.66 / 0.409 |
| identical options: max − min logit | 2.24 | 0.22 (see below) |

**Interpretation.**
- **Order dependence falls by 60%** (67% to 27% of questions) with no accuracy loss at the
  given order. Rotation averaging no longer helps (0.428), because there is less bias to
  average away; it is not harmful beyond noise (400 questions, about ±2.5 points).
- **It learns more slowly:** dev loss 0.41 against 0.27 after the same 150 steps, and it
  is less confident (mean top probability 0.59 against 0.72). At full training the
  difference may shrink or grow.
- **Not at the bar:** 27% is far from 2% (Gate A1). ModernBERT's local attention layers
  use token *index* distance, not position ids, so option order still matters through them;
  that is the likely remaining source (a hypothesis, not tested). Block attention masks or
  segment embeddings would address it.
- **The identical-option number is flattering:** identical texts hash to identical start
  positions, so the check partly reduces to the reset case. The permutation check is the
  meaningful one.

**Limitations.** 400 questions, one seed, 150 steps, quick recipe on one source; no
intervals; accuracy differences are within noise. No full-suite run, so no calibrated
numbers.

---

## R032: order robustness of the marker model, and the position-reset variant (decision 0015)

- **Folder:** `result_logs/R032_2026-10-01_marker-order-checks/` (script, outputs, configs, the
  three bundles' manifests and training logs, `SHA256SUMS`).
- **Question.**
  1. How order-sensitive is the quick marker model (R031)? (PLAN_V3 Gate A1: order-change
     rate at most 2%.)
  2. Does the position-reset variant (every option segment restarts at the same position id)
     remove the order dependence?
- **Setup.** `scripts/order_checks.py`, 400 random `mcqa_v0` test questions (RACE + SciQ).
  - *Identical-option check:* 200 questions whose four options are replaced by the same text;
    any logit spread is position bias.
  - *Permutation check:* every question scored in all 4 cyclic rotations plus 4 random orders
    (8 orders); "changes" means the predicted option is not the same in all 8.
  - Bundles: `seq_sciq_quick` (R031), `seq_reset_sciq_quick` (same recipe, position resets,
    1 epoch) and `seq_reset_sciq_e3` (3 epochs, 450 steps). Uncommitted code on `85cd462`;
    RTX 2060.

| 400 test questions | seq_sciq_quick (no reset) | seq_reset_sciq_quick (reset) |
|---|---|---|
| identical options: mean logit by slot (centred) | −1.01, +0.62, +0.93, −0.53 | 0.00, 0.00, 0.00, 0.00 |
| identical options: max − min logit | 2.24 | 0.004 |
| answer changes across the 8 orders | **67.3%** | 93.5% |
| accuracy, order as given | 0.433 | 0.258 |
| accuracy, rotation-averaged (4 rotations) | **0.480** | 0.290 |
| correct in all 8 orders | 0.230 | 0.018 |
| training loss at the end | 0.40 (dev 0.266) | 1.387 (dev 1.386 = ln 4) |

- The 3-epoch reset run (450 steps) also stays at ln 4: dev loss 1.3863, 1.3863, 1.3864 and
  logit spread 0.0007 to 0.0016, so it has collapsed (lesson 8), not only started slowly.
- *Why the reset variant cannot learn (diagnostic `dbg_reset.py`, pretrained encoder, no
  training).* The marker states of the four options of one question have mean cosine
  similarity 0.949 with default positions and **0.9999** with resets: with every option's
  tokens at identical positions, a `[MASK]` marker cannot tell its own option's tokens from
  the others' (rotary positions are relative), so all markers read the same mixture.

**Interpretation.**
- **The marker format is position-sensitive in practice.** At this early stage of training
  (150 steps) the model prefers some slots (middle slots by about +0.6 to +0.9 logits) and
  its answer depends on the order for about two thirds of questions. The Gate A1 bar of 2%
  is far away. Whether training longer, more data, or more augmentation closes it is
  the open question; the quick model has barely begun to learn.
- **Rotation averaging helps a lot here:** +4.7 points on 400 questions (0.433 to 0.480),
  at 4× the inference compute. It is therefore the default way to use a marker model.
- **Position resets as designed do not work.** They remove the bias (identical options give
  identical scores) only because the model learns nothing. Keeping option identity requires
  a different mechanism, for example a block attention mask that lets each option attend to
  its own tokens, the question and the context (and mixes options only in the head layers),
  or a learned segment embedding. Neither is implemented.

**Limitations.**
- 400 questions, one seed, quick training: the order-change rate will change with training;
  intervals were not computed.
- "Answer changes" counts any disagreement among 8 orders, a strict measure.
- The reset diagnostic uses 6 questions.
- `position_reset = true` remains a valid configuration value but is known not to learn
  (config docstring updated).

---

## R031: marker-sequence encoding against joint encoding, matched quick run (decision 0015)

- **Folder:** `result_logs/R031_2026-10-01_marker-encoding-sciq-quick_mcqa_v0/` (bundle
  manifest and training log, run manifest and predictions, calibration, analysis, paired
  comparison against R004, configs, `SHA256SUMS`).
- **Question.** At the same data, steps and learning rates as R004 (2,400 SciQ questions,
  one epoch), does the single-sequence marker model (all options in one sequence, scores
  read at `[MASK]` markers, two extra transformer layers) match or beat the joint model?
  This is a first, small look at PLAN_V3 Gate A1 (accuracy part).
- **Predictor.** `seq_modernbert_base@ee3ced731194`: ModernBERT-base fine-tuned, 159.1M
  parameters, 512 tokens, option order randomised in training, no rotation averaging, no
  position reset. Config `seq_sciq_quick` is R004's with micro-batch 8 instead of 4
  (accumulation to the same batch of 16). Seed 0, same 2,400 random SciQ examples as R004.
- **Data and suite.** Train `mcqa_sciq_v0`; evaluated on `mcqa_v0` (r1): test 5,922 (RACE
  4,931, SciQ 991); calibration split 2,990 for the temperature.
- **Hardware.** RTX 2060, fp16 mixed. Training 83 s (28.9 examples/s). Collection of all
  splits (11,802 questions) about 3 minutes. R004's training took about 8 minutes.
- **Source state.** Uncommitted code on top of `85cd462` (the trainer's mixed sampling and
  the new sources were added after it); the model code itself matches `85cd462`.

| test (n = 5,922) | R004 joint | R031 marker | difference [95% CI] |
|---|---|---|---|
| accuracy, all | 0.426 | **0.452** | +0.026 [+0.013, +0.039] |
| accuracy, RACE | 0.328 | 0.360 | – |
| accuracy, SciQ | 0.910 | 0.908 | – |
| NLL, raw | 1.352 | 1.642 | +0.289 [+0.252, +0.326] (worse) |
| NLL, calibrated | 1.196 | 1.186 | – |
| ECE, raw / calibrated | 0.155 / 0.041 | 0.261 / 0.047 | – |
| temperature fitted | – | 3.55 | – |

- Wrong-to-right 14.4%, right-to-wrong 11.8% (paired, all).
- Differences are b − a with b the marker model. Per-source differences were not
  computed. Per-source accuracies are from the two analyses.

**Interpretation.**
- **Accuracy.** The marker model is better by 2.6 points on the whole test set, with an
  interval clear of zero. Nearly all of it is RACE (0.360 against 0.328), which neither run
  trained on: the marker format transfers from SciQ to RACE better. SciQ, which both saw,
  is a tie (0.908 against 0.910).
- **Calibration.** The raw model is much more overconfident (raw NLL 1.64 against 1.35,
  ECE 0.26 against 0.16), and the fitted temperature is 3.5. After calibration NLL is equal
  (1.186 against 1.196) and ECE similar. The same temperature that fixes RACE hurts SciQ
  (SciQ ECE 0.148 calibrated against 0.018 raw), the per-slice problem of lesson 4. This
  supports PLAN_V3 A4 (a temperature per question type or source).
- **Cost.** Training took 83 s against about 8 minutes (R004); inference 5,922 questions in
  about 100 s.
- **Reading.** On this matched quick run the marker encoding is at least as good as joint
  encoding at about a fifth of the training time. It is the first step of Gate A1. Order
  robustness, which Gate A1 also requires, is not measured here.

**Limitations.**
- One seed, 2,400 training examples, one epoch: a quick run, not a converged comparison.
  The gap may change at full training.
- No permutation or identical-option check yet, so position bias is unmeasured.
- Option order at inference was the dataset's order; rotation averaging was not used.
- The temperature was fitted on a calibration split that includes RACE, which the model
  never saw in training.

---

## R030: `mix_v1` dataset and its lexical fixture (PLAN_V3 stage A2)

- **Folder:** `result_logs/R030_2026-10-01_mix_v1_dataset_and_lexical_fixture/` (dataset
  manifest, protocol, configs, audit, analysis, run manifest, `SHA256SUMS`). The dataset
  itself stays in `metanoia/artifacts/datasets/mix_v1`.
- **Question.** Can one pinned, leak-checked mix cover general reading, law with a passage
  and medicine without a passage, in the existing instrument? What do trivial rules score
  on it?
- **Setup.**
  - `metanoia-research prepare --config configs/data/mix_v1.toml`: 10 sources at pinned
    revisions (RACE, SciQ, ARC-Easy, ARC-Challenge, OBQA, QASC, CommonsenseQA, BoolQ,
    CaseHOLD, MedMCQA); 2 minutes on the CPU. Suite `suites/mix_v1`.
  - `scripts/audit_dataset.py`: per-source counts, split leakage, trivial rules.
  - Lexical fixture collected and analyzed (`lexical_overlap@1`).
  - Uncommitted code on top of `85cd462`.
- **Dataset.** `cases_sha256 4ca9f0dd…8f7a8`, `labels_sha256 0279e836…133565`.
  Train 355,114; dev 9,073; calibration 9,008; test 23,118. 1,021 items skipped for
  duplicate option texts. Regenerating `mcqa_v0` with the extended code reproduces its
  pinned hashes exactly.

| source | train | test | options | passage (median words) | lexical fixture, test acc [95% CI] |
|---|---|---|---|---|---|
| RACE | 87,816 | 4,931 | 4 | 279 | 0.314 [0.303, 0.328] |
| CaseHOLD (law) | 44,973 | 3,595 | 5 | 137 | 0.225 [0.211, 0.238] |
| MedMCQA (medicine) | 178,402 | 4,162 | 4 | none | 0.262 [0.255, 0.269] |
| SciQ | 11,643 | 991 | 4 | 49 | 0.820 [0.800, 0.841] |
| BoolQ | 8,217 | 3,270 | 2 | 84 | 0.497 [0.491, 0.502] |
| QASC | 7,189 | 926 | 8 | none | 0.131 [0.121, 0.140] |
| CommonsenseQA | 8,552 | 1,197 | 5 | none | 0.190 [0.183, 0.198] |
| OBQA | 4,954 | 500 | 4 | none | 0.242 [0.226, 0.259] |
| ARC-Easy | 2,251 | 2,375 | 3–5 | none | 0.266 [0.256, 0.276] |
| ARC-Challenge | 1,117 | 1,171 | 3–5 | none | 0.225 [0.211, 0.240] |
| all | 355,114 | 23,118 | | | 0.314 [0.309, 0.318] |

**Audit findings** (`audit.json`).
- **BoolQ:** 623 of its test passages also occur in train (different questions). A
  yes/no model may learn passage-specific priors. Always-"yes" scores about 0.62 (from the
  hub's label counts, 2,033 of 3,270), against the 0.5 chance of a seeded-order fixture.
- **CaseHOLD:** 5.6% of test items have a question and option set identical to some train
  item (same holdings, different citing text); RACE 1.4%, OBQA 2.8%, ARC-Easy 0.1%.
- **Shortcuts:** longest-option and passage-overlap rules are near chance everywhere except
  SciQ (overlap 0.817, the known SciQ shortcut, R002).
- **Medicine:** MedMCQA is 50% of the training examples (178k of 355k), so training needs
  source-balanced sampling (`mix_alpha`, added to the trainer; not yet used in a run).

**Interpretation.**
- The mix gives the owner's with-passage (law) and without-passage (medicine) comparison
  inside one suite, and a held-out structure for per-source reporting. The fixture does not
  exceed chance except on SciQ, so the other sources are not solvable by word overlap.
- Gold explanations for 162,366 MedMCQA questions (about 87% of its 186,276 kept items),
  9,060 QASC fact pairs and 5,954 OBQA facts are in `proofs.jsonl`, for with-passage
  ablations.

**Limitations.**
- ARC and OBQA slices use a different seed than `knowledge_v0` (dev/calibration split
  differs); use `knowledge_v0` for numbers about that suite.
- The macro average is accuracy over all test examples, which weights sources by test size.
- Contamination of pretrained encoders by these public sets is untested.
- The lexical fixture is not a trained options-only baseline; that is still to run.
- Run on a dirty tree (R028–R030 code uncommitted at the time).

---

## R029: marker-sequence encoding, position-id and throughput probe (decision 0015)

- **Folder:** `result_logs/R029_2026-10-01_sequence-encoding-probe/` (script, the two
  model configs, output, `SHA256SUMS`).
- **Question.**
  1. Does ModernBERT-base on this machine honour `position_ids` (needed for the
     position-reset variant)?
  2. What do training throughput and memory look like for the single-sequence marker
     model against the joint model, on real RACE questions?
- **Setup.** `scripts/probe_sequence.py`; uncommitted working tree on top of `a22c7b7`
  (the sequence model, its tests and configs); RTX 2060 6 GB, fp16 autocast, gradient
  checkpointing, AdamW, batch 8, 10 timed steps after one warm-up step; transformers
  attention implementation `sdpa`. Real ModernBERT-base weights, **untrained head**.
  Marker model: 512 tokens; joint model: 384 tokens (its R005 setting).

| | position reset effect on encoder states | examples/s | peak GPU memory |
|---|---|---|---|
| marker model (`seq_modernbert_base`) | max abs diff 38.3 vs default positions (identical ids) | **17.91** | 3.30 GB |
| joint model (`direct_modernbert_ft_joint`) | – | 5.89 | 3.47 GB |

- Passing default position ids explicitly gives exactly the same states as passing none
  (max abs diff 0.0).

**Interpretation.**
- **Positions.** The `sdpa` path honours `position_ids`: resetting option positions changes
  the encoder states, so the position-reset variant is a real change and is testable.
  This does not show that it removes order dependence. ModernBERT also alternates local
  (sliding-window) and global attention layers, and the local window follows token
  *index*, not position id, so options far apart in the sequence may still see each other
  differently. Only a trained model's permutation check can say.
- **Cost.** The marker model is 3.0× faster in training than joint encoding and slightly
  lighter in memory despite 512 against 384 tokens, so it reaches the predicted ≥2× saving
  (PLAN_V3 Gate A1, compute part). Accuracy and order-change are not measured here.
- Loss values (1.41, 1.40) are close to ln 4 = 1.386 as expected for untrained heads
  after a few steps; they say nothing about learning.

**Limitations.** One machine; 10 steps; throughput depends on passage length (RACE median
285 words) and padding; the loss values are not a result. Not an accuracy comparison.

---

## R028: Step 0 of PLAN_V3, survey of public multiple-choice sources with passages

- **Folder:** `result_logs/R028_2026-10-01_step0_source_survey/` (two scripts, their
  outputs, `SHA256SUMS`). Scripts also in `metanoia/scripts/`.
- **Question:** which domains have the most public labelled data that is already
  multiple choice and comes with its own passage? (PLAN_V3 §2: the owner's rule is
  "the domain with the most public labelled data".)
- **Setup.**
  - Hugging Face metadata only (licence, splits, file sizes) for 21 candidate sources,
    then the validation file of three sources (RACE, CaseHOLD, QuALITY), at pinned
    revisions.
  - Statistics on the validation splits and three trivial rules with no learning:
    longest option, most word overlap with the passage, most overlap with the question.
  - Commit `e755e02` plus the two new scripts (not yet committed); CPU; about 3 minutes.

**Native multiple choice with a passage (train sizes, licence).**

| source | domain | train / val / test | options | licence |
|---|---|---|---|---|
| RACE | general reading | 87,866 / 4,887 / 4,934 | 4 | "other" (research use) |
| CaseHOLD (LexGLUE) | law | 45,000 / 3,900 / 3,600 | 5 | CC-BY-4.0 |
| CosmosQA | commonsense reading | 25,262 / 2,985 / (test 6,963) | 4 | CC-BY-4.0 |
| DREAM | dialogue | 6,116 / 2,040 / 2,041 | 3 | unknown |
| ReClor | logical reasoning | about 4.6k (val 500) | 4 | non-commercial |
| QuALITY | long articles | 2,523 / 2,086 / test hidden | 4 | not stated on the card |

**Not native multiple choice, or no passage.**
- No passage: MedMCQA (182,822 train, Apache-2.0, medical knowledge), ARC, OBQA, QASC,
  CommonsenseQA.
- Passage, not exclusive-choice: BoolQ (9,427 yes/no, CC-BY-SA-3.0); MultiRC (27,243,
  several correct options); PubMedQA (1,000 labelled yes/no/maybe, 211,269 heuristically
  labelled; MIT); ContractNLI (about 10k, three-way, CC-BY-NC-SA).
- Span or free answer, so options would have to be constructed: HotpotQA (90,447,
  CC-BY-SA-4.0), MuSiQue, 2WikiMultihopQA (167k), SQuAD 2.0 (130k), DROP (77k),
  NarrativeQA (33k), CUAD (22k clause spans; licence CC-BY-4.0). LegalBench's repository
  holds task metadata only.

**Validation-split statistics (the three downloaded).**

| | RACE | CaseHOLD | QuALITY |
|---|---|---|---|
| questions | 4,887 | 3,900 | 2,086 |
| chance | 0.250 | 0.200 | 0.250 |
| passage words p10 / p50 / p90 / max | 152 / 285 / 379 / 894 | 136 / 146 / 157 / 188 | 2,034 / 4,848 / 5,616 / 6,162 |
| option words, mean | 5.8 | 24.7 | 7.9 |
| gold at position 0 / last | 0.218 / 0.251 | 0.197 / 0.203 | 0.235 / 0.253 |
| rows with duplicate options | 4 | 1 | 0 |
| longest option | 0.288 | 0.236 | 0.274 |
| overlap with passage | 0.297 | 0.361 | 0.248 |
| overlap with question | 0.226 | 0.197 | 0.219 |

**Interpretation.**
- **Volume.** Among native multiple choice with a passage, general reading (RACE) is the
  largest overall and law (CaseHOLD, 45k) the largest single specialised domain. No
  other domain reaches either; medicine is larger only without a passage (MedMCQA).
  This supports PLAN_V3's expected outcome: RACE-style data as the training backbone,
  and law as the primary domain stress test.
- **Passage length.** CaseHOLD passages are short (about 146 words), so CaseHOLD
  tests legal reading, not long documents. QuALITY is the only source with documents of
  thousands of words (median 4,848), and it is small (2.5k training questions); long-
  document behaviour will need constructed long inputs (PLAN_V3 §4) as well.
- **Shortcuts are weak.** No trivial rule is far above chance. The largest is overlap
  with the passage on CaseHOLD (0.361 against 0.200), so it must be reported as a
  fixture. Gold positions are balanced. This does not exclude artefacts an options-only
  trained model could find; that baseline is still to be run.
- **Multi-hop.** The multi-hop sets would need distractor options constructed by us,
  which risks type-matching shortcuts; deciding how is a separate design step.

**Limitations.**
- Not verified, written from memory: DREAM's option count (3), ReClor's training size
  (about 4.6k; the hub only gave file sizes), that CosmosQA's test labels are hidden,
  that MultiRC allows several correct options, and ContractNLI's size (about 10k).
- Other size numbers come from the hub's card metadata; I did not count rows in the
  parquet files for sources I did not download, and a card may be stale.
- Licences are as stated on the card, not legally reviewed. RACE, ReClor and ContractNLI
  carry research or non-commercial terms; this matters if the system is ever
  deployed commercially.
- Contamination of a pretrained encoder by any of these sets was not tested.
- Statistics are for the validation split only. Not archived: nothing else (all outputs are
  in the folder).

---

## R027: OBQA gold-fact coverage in `generics_v0` (diagnostic)

- **Folder:** `result_logs/R027_2026-09-30_obqa-fact-coverage/` (script and output).
- **Question:** OBQA has not moved with any proposal method (R020, R024, R026). Are
  its gold facts missing from the memory, or does retrieval miss them?
- **Setup.**
  - The 500 OBQA dev questions.
  - For each gold core fact, the memory statement closest to it: bge-small passage
    embeddings, exact search over 1,015,610 statements.
  - The closest statement among the focused branches (8 per candidate, R025's G1/J1).
  - Commit `6c02551` plus the script; RTX 2060.

| Similarity to the gold fact | In the memory | In the focused branches | Found when present |
|---|---|---|---|
| ≥ 0.95 | 0.094 | 0.028 | 0.30 |
| ≥ 0.90 | 0.486 | 0.238 | 0.49 |
| ≥ 0.85 | 0.906 | 0.550 | 0.61 |
| ≥ 0.80 | 0.990 | 0.790 | 0.80 |

- Medians: 0.898 in the memory, 0.857 in the branches.
- Examples: at 0.92, "Some animals live in deep water" for "deep sea animals live deep
  in the ocean" (close). At 0.89, "Many animals hatch from eggs" for "as the number of
  eggs laid … increases, the number that hatch will increase" (related, not
  equivalent).

**Interpretation.**
- **Coverage.** Roughly half of OBQA's gold facts have a close equivalent in
  GenericsKB. Below about 0.9 similarity, matches are usually *related* statements,
  not the needed relation.
- **Retrieval.** When a close equivalent exists, the focused branches contain it about
  half the time.
- **The ceiling.** Even the gold fact lifts the trained reader only to 0.674 on OBQA
  (R021), against about 0.61 now: at most about 6–7 points could come from better
  selection. OBQA questions often need the core fact *plus* a second piece of
  knowledge.
- **Consequence.** OBQA is limited jointly by memory coverage, by one-hop retrieval,
  and by the need to combine two facts. Re-ranking one-hop proposals from this memory
  cannot move it much; R024–R026 confirm it has not.

**Limitations.** Embedding similarity is a proxy for "the same fact"; thresholds are
judged from examples; dev only.

---

## R026: EXPAND v2 with fused proposals (round 2 of decision 0014)

- **Folder:** `result_logs/R026_2026-09-30_expand-v2-fused_knowledge_v0_r3/`
  (manifests, logs, runs, selection, analysis, comparisons,
  `fused_per_source.txt`, the proposal-cache manifest and build log, configs, script).
- **Question:** does the configuration promoted by R025 (similarity + reranker rank
  fusion), retrained on its own branches, beat focused proposals on test, especially
  on OBQA and ARC-Challenge?
- **Setup.**
  - `branch_v2_fused_knowledge_v0`: identical to R024 except `proposals = "fused"`
    (bge-reranker-v2-m3 pinned, a pool of 32, RRF k = 60).
  - The proposal cache (54,563 question–candidate orderings) took 4,764 s to build,
    at commit `6c02551`. The code is identical to `cc91ca5`, which changed only the
    interface.
  - Training: 867 s at commit `cc91ca5`, clean (the owner's interface commit). Dev NLL
    0.864 after epoch 1 and 0.910 after epoch 2, so **epoch 1 was selected**
    (`branch_v2_fused@78761cebebd7`).

| Test accuracy | b0 | p4 | p8 | p16 | p32 |
|---|---|---|---|---|---|
| **Fused** | 0.553 | 0.616 | 0.625 | **0.652** | 0.647 |
| Focused (R024) | 0.562 | 0.623 | 0.635 | 0.645 | **0.652** |
| Empty branches (R020) | 0.581 | = | = | = | = |

- **Fused NLL:** 1.058 at b0, then 0.936, 0.907, 0.874 and 0.867.
- **Paired, fused − focused** (all intervals include 0):
  - p8: −0.010 [−0.023, +0.003];
  - p16: +0.007 [−0.006, +0.020];
  - p32: −0.004 [−0.016, +0.009], NLL +0.005 [−0.011, +0.021].
- **Paired, fused − empty:** p16 +0.071 [+0.057, +0.087], NLL −0.124; p32 +0.066
  [+0.052, +0.081], NLL −0.131.
- **By source** (group bootstrap):

| | OBQA | ARC-Easy | ARC-Challenge |
|---|---|---|---|
| Fused − empty, p16 | +0.014 [−0.024, +0.054] | **+0.102** [+0.082, +0.123] | **+0.033** [+0.003, +0.062] |
| Fused − focused, p16 | +0.036 [−0.004, +0.076] | +0.005 | 0.000 |
| Fused − focused, p32 | +0.016 [−0.020, +0.054] | −0.001 | −0.020 [−0.045, +0.005] |

**Interpretation.**
- **Fused proposals tie focused ones.** The +2.5-point screening gain on OBQA +
  ARC-Challenge (R025) did not replicate after retraining. As R025 warned, it was at
  the edge of dev noise.
- **The mechanism is robust to how proposals are chosen, once they are
  candidate-relevant.** Two independently retrained variants (focused, fused) both
  land at +6.6 to +7.1 points over the matched direct model. Both are far above plain
  similarity (+2.9, R020) and random statements.
- **One-hop proposal tuning on this memory has plateaued.** OBQA stays within noise of
  the direct model for every method, and R027 shows why: its facts are only partly in
  the memory, and they often must be combined with a second fact. Continuing to
  re-rank one-hop proposals would be forcing a method that has given what it can.

**Limitations.** One seed per model; the reranker's CPU cost for new questions on the
interface (about 11 s per question on a cache miss).

---

## R025: proposal screening, round 1 (decision 0014)

- **Folder:** `result_logs/R025_2026-09-29_proposal-screening-round1/`
  - scripts: `screen_build.py`, `screen_eval.py`;
  - outputs: the branches of every configuration (`branches.json.gz`), the build
    statistics and example paths, both readers' results, and the promotion table
    (`promotion.txt`).
- **Question:** under the protocol fixed in decision 0014, do other generators (concept
  paths, follow-up search), judges (reranker) or their combinations choose better
  branches than focused similarity (G1/J1, R024)?
- **Setup.**
  - `knowledge_v0` dev + calibration (1,368; test untouched), with `generics_v0`.
  - Commit `e610aca` plus the scripts, on the RTX 2060.
  - The build took 19 min: 25 s for the concept index, 280 s for the paths, and 2.2 ms
    per reranked pair on the GPU (fp16).
- **Generators:**
  - **G1:** the top 32 for "question candidate";
  - **G3:** concept paths. One hop: statements mentioning a candidate concept and a
    question concept, weighted by inverse frequency, top 32. Two hops: through a bridge
    concept found in at most 5,000 statements, top 8 pairs. 86% of candidates have a
    one-hop path and 90% a two-hop path;
  - **G4:** a follow-up search from the focused top statement, top 16.
  - The mean pool is 66 statements per candidate.
- **Judges:**
  - J0: similarity;
  - J1: focused;
  - J2: bge-reranker-v2-m3, with a "Question: … Answer: …" query;
  - J1+J2: mention first, reranker within each group;
  - RRF(J0, J2): reciprocal-rank fusion;
  - J1+RRF.
- **Readers:** both trained v2 models, focused (`@950c83d9e163`) and similarity
  (`@277ebb247391`). Widths of 2 and 8 statements per candidate.

**Results** (dev + calibration). "Hard" is OBQA + ARC-Challenge accuracy (798
questions); Δ is against G1/J1 at the same width and reader. The full table is in
`promotion.txt`.

| Configuration | Focused reader: all (k=2 / 8) | Hard Δ | Similarity reader: all | Hard Δ | Gold recall (k=2 / 8) | On-candidate | Jaccard |
|---|---|---|---|---|---|---|---|
| G1/J1 (reference) | 0.633 / 0.648 | – | 0.596 / 0.591 | – | 0.260 / 0.467 | 0.79 | 0.15–0.22 |
| G1/J0 | 0.627 / 0.651 | −0.4 / +0.5 | 0.610 / 0.601 | +2.1 / +1.0 | 0.247 / 0.458 | 0.44–0.64 | 0.35 |
| G1/J2 | 0.632 / **0.661** | +1.5 / +1.1 | 0.591 / 0.604 | −1.0 / +1.4 | 0.179 / 0.411 | 0.49–0.67 | 0.31 |
| **G1/RRF(J0,J2)** | **0.651 / 0.657** | **+2.5 / +2.4** | 0.607 / 0.602 | +1.4 / +0.5 | 0.235 / 0.432 | 0.48–0.66 | 0.33–0.35 |
| G1/J1+RRF(J0,J2) | 0.640 / 0.658 | +1.1 / +1.9 | 0.593 / 0.593 | +0.1 / +0.6 | 0.252 / 0.455 | 0.79 | 0.14–0.22 |
| G3/J1 (paths) | 0.621 / 0.635 | −0.1 / −1.3 | 0.590 / 0.583 | −0.8 / −1.4 | 0.251 / 0.461 | 0.95 | 0.08 |
| G3/J2 (paths) | 0.621 / 0.637 | −2.0 / −2.1 | 0.585 / 0.581 | −1.8 / −2.0 | 0.229 / 0.450 | 0.91–0.95 | 0.07–0.08 |
| G1+G3/J1+J2 | 0.648 / 0.632 | +0.6 / −2.8 | 0.598 / 0.596 | −0.9 / −0.8 | 0.202 / 0.416 | 0.95 | 0.06–0.07 |
| G1+G3+G4/J1 | 0.626 / 0.632 | −0.3 / −2.3 | 0.583 / 0.573 | −1.6 / −2.0 | 0.250 / 0.463 | 0.95 | 0.07–0.08 |

**Promotion** (rule fixed in decision 0014, point estimates):
- **G1/RRF(J0,J2)** passes rule (b) at both widths under the focused reader (+2.5 and
  +2.4). **Promoted.**
- G1/J0 passes (b) at k = 2 only under the reader trained on it. That is R020's method,
  already tested (and beaten by focused, R024), so it is not retrained.
- Nothing passes rule (a): no configuration raises OBQA gold-fact recall.

**Interpretation.**
- **Concept paths, as built here, do not help.** They give the most candidate-specific
  branches of any method (95% on-candidate, overlap 0.07), but no more of the gold
  fact, and 1–2 points less accuracy. The example paths show why: lexical concept
  matching confuses senses ("boil" the infection for boiling, "aid" in "Kool-Aid"). And
  R023 showed that the existence of a path does not separate right from wrong answers.
  - This is a negative result for this *implementation* of relationship-based
    retrieval (lexical concepts, inverse-frequency ranking, untrained readers), not
    for the idea. Sense-aware concept linking and a judge trained on paths (J3/J4)
    remain untested.
- **Follow-up search adds nothing** in this form.
- **Fusing judges helps; replacing them does not.** The reranker alone loses gold
  recall. Fused with similarity (RRF), it gives the best accuracy on the hard sources
  under both readers, with 2.4–2.5 points on OBQA + ARC-Challenge under the focused
  reader. This is the composition the owner asked for.
- **Gold-fact word recall is a weak proxy.** RRF raises accuracy while *lowering*
  recall. Accuracy with a trained reader is the criterion that matters; recall stays a
  diagnostic.
- **Readers favour their own training distribution**: the similarity reader prefers
  G1/J0, as R022 found. So a retraining is needed before any test claim.

**Limitations.**
- Point estimates on 1,368 questions (half-width about ±0.026 overall and ±0.035 on
  the hard 798); differences of 2–3 points are at the edge of noise, which is why the
  promotion leads to a matched retraining and a test collection.
- Two readers, both trained on other proposal styles.
- Paths used lexical concept matching only.

---

## R024: EXPAND v2 with focused proposals (matched to R020)

- **Folder:** `result_logs/R024_2026-09-29_expand-v2-focused_knowledge_v0_r3/`
  (manifests, logs, dev and test runs, selection, analysis, comparisons,
  `focused_per_source.txt`, rejected calibrations, configs and script).
- **Question (decision 0013):** retrained on its own branches, does *focused*
  selection beat plain similarity? Focused takes the top 64 statements for "question
  candidate" and puts those mentioning the candidate first. Does OBQA or ARC-Challenge
  finally gain?
- **Setup.**
  - `branch_v2_focused_knowledge_v0`, identical to R020's retrieved model except
    `proposals = "focused"`: R005 initialization, 2 epochs, widths {0, 1, 2, 4, 8},
    seed 0.
  - Commit `ae34671` via `scripts/stage2_expand_v2_focused.sh`; dirty only by the
    owner's `interface/next.config.ts`.
  - Training took 926 s. Dev NLL was 0.885 after epoch 1 and 0.945 after epoch 2, so
    **epoch 1 was selected** (`branch_v2_focused@950c83d9e163`).
  - Controls: R020's random, empty-branch and similarity models (same suite revision
    and schedule).

| Test accuracy | b0 | p4 | p8 | p16 | p32 |
|---|---|---|---|---|---|
| **Focused** | 0.562 | 0.623 | 0.635 | 0.645 | **0.652** |
| Similarity (R020) | 0.537 | 0.592 | 0.608 | 0.609 | 0.611 |
| Random (R020) | 0.582 | 0.561 | 0.557 | 0.550 | 0.556 |
| Empty branches (R020) | 0.581 | = | = | = | = |

- **Focused NLL:** 1.045 at b0, 0.918, 0.895, 0.881, then **0.862** at p32. Test ECE
  at p32 is 0.018.
- **Paired, focused − empty:**
  - p8: +0.053 [+0.039, +0.069];
  - **p32: +0.070 [+0.056, +0.087], NLL −0.136 [−0.159, −0.113]**;
  - b0: −0.020 [−0.032, −0.007].
- **Paired, focused − similarity:** p8 +0.027 [+0.013, +0.041]; p32 +0.041 [+0.027,
  +0.053], NLL −0.092.
- **Paired, focused − random:** p32 +0.096 [+0.080, +0.112].

**By source** (p32, paired group bootstrap):

| Source | Focused | vs empty | vs similarity |
|---|---|---|---|
| OBQA | 0.612 | −0.010 [−0.056, +0.032] | −0.002 [−0.044, +0.042] |
| ARC-Easy | 0.711 | **+0.100** [+0.078, +0.120] | +0.036 [+0.019, +0.053] |
| ARC-Challenge | 0.549 | **+0.045** [+0.016, +0.073] | **+0.068** [+0.039, +0.096] |

**Calibration.**
- The fitted temperatures (T = 1.232 at p16, 1.208 at p32) *worsen* overall test ECE:
  0.022 → 0.029 and 0.018 → 0.033. They trade ARC-Easy for OBQA and ARC-Challenge.
- Both are rejected (`calibration_*_rejected.json`); the raw probabilities are already
  well calibrated.

**Interpretation.**
- **Candidate-specific expansion beats topic retrieval**, as R021 and R022 predicted.
  The same model, trained on branches about its own candidate, improves steadily with
  budget and is still rising at 8 per candidate. It is the best model in the project
  on this suite, and the gap to the direct model has grown from +2.9 to +7.0.
- **ARC-Challenge now benefits** (+4.5 over direct, +6.8 over similarity), so the gain
  is no longer only on easy questions.
- **OBQA still gains nothing.** Its questions need a linking fact that neither
  similarity nor focusing retrieves (R021: the gold fact alone gives the reader
  0.674). This is the target of the proposal portfolio (decision 0014: concept paths,
  follow-up search, learned judge).
- The model is less dependent on its branches than R020's: 2 points below the direct
  model at b0, against 4.5.
- **Accounting caveat:** usage reports the statements read, not the 64-statement pool
  considered per candidate (decision 0014; fixed by `knowledge_v0` r4).

**Limitations.** One seed. Selected at epoch 1 while R020's similarity model was
selected at epoch 2 (each by its own dev NLL).

---

## R023: feasibility checks for the proposal-method plan (CPU only)

- **Folder:** `result_logs/R023_2026-09-29_proposal-methods-feasibility/` (two
  scripts and their JSON outputs).
- **Question:** before committing to the plan of decision 0014, are its two main new
  ingredients viable?
  1. A concept graph over `generics_v0`, used to find paths from a candidate's
     concepts to the question's (the owner's relationship-based retrieval).
  2. An off-the-shelf reranker that reads (question + candidate, statement) together.
- **Setup:** CPU only; the GPU was running the owner's focused-proposal training.
  Commit `ae34671` plus the scripts. `knowledge_v0` dev + calibration (1,368) for the
  graph; 60 OBQA dev questions for the rerankers.

**1. Concept graph.**
- **Indexing:** 1,015,610 statements take 18.7 s on CPU to index by the 128,816
  concept terms they mention (1–3-word spans, singularized). That is 4.46 concepts per
  statement.
- **Degrees are extremely skewed:** "people" appears in 30,948 statements, "water"
  27,655, "plant" 25,766, "energy" 12,147. The median concept appears in 2 statements.
- **Path existence**, with concepts above 2,000 statements excluded from links:

| | Gold | Distractors |
|---|---|---|
| Candidate has any memory concept | 0.731 | 0.711 |
| At least one 1-hop statement linking candidate and question concepts | 0.482 | 0.435 |
| Mean number of such statements | 8.8 | 5.7 |
| At least one 2-hop bridge concept | 0.667 | 0.657 |
| OBQA only, 1-hop / 2-hop | 0.350 / 0.630 | 0.315 / 0.591 |

- **Within a question**, the gold candidate alone has the most 1-hop links in 297 of
  1,368 questions (22%) and ties for most in 72. A distractor has more in 637, and
  362 questions have no links at all.

**2. Rerankers.**
- **Setup:** for each question, a pool of the top 31 `generics_v0` statements for
  "question gold-candidate", plus the gold core fact. The table gives the rank of the
  gold fact among these 32.
- **Candidate sensitivity:** the share of questions where the fact scores higher
  under the gold candidate's query than under every distractor's (chance 0.25).

| Judge (query format) | Mean rank | Top 2 | Top 8 | Candidate-sensitive | CPU ms/pair |
|---|---|---|---|---|---|
| bge-small similarity (retrieval) | 15.9 | 0.33 | 0.48 | – | – |
| bge-reranker-base ("question candidate") | 16.5 | 0.10 | 0.30 | 0.47 | 20 |
| bge-reranker-base ("Question: … Answer: …") | 15.3 | 0.13 | 0.33 | 0.40 | 22 |
| bge-reranker-v2-m3 ("question candidate") | 12.9 | 0.22 | 0.43 | 0.53 | 66 |
| bge-reranker-v2-m3 ("Question: … Answer: …") | 12.3 | 0.25 | 0.48 | 0.58 | 88 |

Revisions: base `2cfc18c9…` (MIT), v2-m3 `953dc6f6…` (Apache-2.0). With 60
questions, proportions carry about ±0.12.

**Interpretation.**
- **The graph is cheap, but paths are a generator, not a judge.** Whether a
  candidate connects to the question barely differs between right and wrong answers:
  many statements link a distractor's concepts to the question's topic. Paths can
  propose linking statements that similarity search misses, but something that
  *reads* must decide which paths matter.
- **Common concepts need down-weighting, not a cap.** The cap removes "energy",
  "water" and "plant", which are central to science questions.
- **Off-the-shelf rerankers are not a free win.** Trained for web search, they rank a
  short generic fact worse than or no better than retrieval similarity at the top of
  the list. The larger v2-m3 is better on average and is candidate-sensitive (0.58
  against 0.25 by chance), so it may *add* to similarity rather than replace it. Its
  CPU cost (≈90 ms per pair) matters for the interface.
- The gold-fact rank is strict: other statements in the pool may be just as useful.
  The real test is the trained reader's accuracy with each method's branches
  (decision 0014, screening).
- **Consequence for the plan:** no single method is promising enough to back alone.
  Generators (similarity, graph paths, follow-up search) and judges (similarity,
  focused, reranker, the trained reader) are screened as a matrix, including
  combinations.

**Limitations.** CPU-only feasibility checks: lexical concept matching; 60 questions
for rerankers; the gold fact as the only positive.

---

## R022: proposal rules, read by the trained EXPAND v2 model (probe)

- **Folder:** `result_logs/R022_2026-09-29_proposal-rules-probe/` (both scripts, JSON
  results with example branches, stdout).
- **Question:** R021 shows the branches are dominated by question-topic statements.
  Do proposals that are about the candidate give better branches, *without*
  retraining the reader?
- **Setup.** The R020 retrieved model, `knowledge_v0` dev + calibration (1,368), and
  `generics_v0`. Proposal rules:
  - **retrieved** (as trained): top-k for "question candidate";
  - **anchored**: statements whose concept term is named in the candidate (1–3-word
    spans, WordNet-lemmatized; 94% of candidates have one), ranked by similarity to
    the question;
  - **chain**: the anchored first hop, interleaved with a second hop from the concept
    terms those statements mention, ranked by similarity to the question;
  - **focused**: the top 64 for "question candidate", reordered so statements
    mentioning the candidate's own (non-question) words come first;
  - **distinct**: focused, then statements retrieved by no other candidate first.

  Quality measures: mean pairwise Jaccard between a question's branches; the share of
  branches mentioning their candidate's own words; OBQA novel gold-fact word recall
  (union of branches).

| Rule, k | Acc all | OBQA | ARC-E | ARC-C | Jaccard | On-candidate | Gold recall |
|---|---|---|---|---|---|---|---|
| retrieved, 2 | **0.610** | 0.604 | 0.667 | 0.513 | 0.35 | 0.44 | 0.25 |
| retrieved, 8 | 0.603 | 0.584 | 0.668 | 0.510 | 0.36 | 0.64 | 0.48 |
| anchored, 2 | 0.556 | 0.550 | 0.607 | 0.470 | 0.10 | 0.92 | 0.19 |
| anchored, 8 | 0.542 | 0.538 | 0.588 | 0.463 | 0.10 | 0.93 | 0.38 |
| chain, 2 | 0.556 | 0.572 | 0.600 | 0.443 | 0.12 | 0.91 | 0.21 |
| chain, 8 | 0.534 | 0.524 | 0.574 | 0.473 | 0.12 | 0.93 | 0.37 |
| focused, 2 | 0.591 | 0.582 | 0.656 | 0.483 | 0.11 | 0.85 | 0.28 |
| focused, 8 | 0.586 | 0.570 | 0.642 | 0.507 | 0.17 | 0.85 | 0.49 |
| distinct, 2 | 0.580 | 0.580 | 0.633 | 0.480 | 0.01 | 0.80 | 0.28 |
| distinct, 8 | 0.575 | 0.558 | 0.618 | 0.520 | 0.04 | 0.84 | 0.52 |

Accuracy half-widths about ±0.026 overall and ±0.043 on OBQA. The two runs of the
"retrieved" rows agree to within 0.004 (fp16 batching).

**Interpretation.**
- **Anchoring on the candidate's own concept is the wrong kind of expansion on its
  own.** Branches become specific (92% on-candidate, overlap 0.10), but statements
  about "cotton" or "puppies" rarely relate the candidate to the question. Gold-fact
  recall falls, and so does accuracy (−5).
- **The naive second hop does not help.** Concepts mentioned by first-hop statements,
  ranked by the question, do not find the linking fact any more often. Recursion
  needs a better choice of the next concept (decision 0013).
- **Focused reordering keeps relevance and gains specificity.** 85% of branches are
  on-candidate, overlap falls to 0.11–0.17, and gold-fact recall is equal or higher.
  Accuracy is within noise of the trained setting (−1.9 at k = 2) for a reader
  trained on the other distribution.
- **Topic statements shared by all candidates carry real signal.** Removing them
  ("distinct") lowers accuracy further. A design may want a shared question context
  alongside the per-candidate branches.
- None of these can be judged fairly without retraining the reader on its own
  proposals. The next run trains v2 with focused proposals (decision 0013).

**Limitations.** Evaluation only, one reader trained on "retrieved" branches; crude
lexical matching; dev + calibration. The model option added with this entry
(`proposals = "focused"`) matches words with a simple plural rule rather than this
probe's WordNet lemmatizer, so its branches can differ slightly.

---

## R021: what the trained EXPAND v2 model takes from its branches (diagnostic)

- **Folder:** `result_logs/R021_2026-09-29_branch-use-diagnostic/` (script, JSON,
  stdout).
- **Question** (and the owner's observation, "it does not seem to use the branches
  much, and the selected statements often do not look useful"): does the model use
  candidate-specific branch content, and how good are the selected statements?
- **Setup.** The R020 retrieved model (`branch_v2_retrieved@277ebb247391`),
  evaluation only, on `knowledge_v0` dev + calibration (1,368; test untouched). Only
  the branches' content changes between conditions. Run at commit `a0b0c4f` plus the
  script.

| Condition (k per candidate) | All | OBQA | ARC-Easy | ARC-Challenge |
|---|---|---|---|---|
| empty | 0.562 | 0.574 | 0.593 | 0.483 |
| own branches, k = 2 / 8 | **0.614** / 0.600 | 0.608 / 0.576 | 0.667 / 0.674 | 0.523 / 0.500 |
| swapped (candidate i reads i+1's), k = 2 / 8 | 0.571 / 0.564 | 0.592 / 0.588 | 0.600 / 0.591 | 0.480 / 0.470 |
| shared (same round-robin union for all), k = 2 / 8 | 0.596 / 0.613 | 0.612 / 0.612 | 0.640 / 0.667 | 0.487 / 0.513 |
| question-only retrieval, same for all, k = 2 / 8 | 0.598 / 0.600 | 0.604 / 0.606 | 0.644 / 0.651 | 0.500 / 0.493 |
| gold core fact as every branch (oracle) | – | **0.674** | – | – |
| gold fact + own, k = 2 / 8 | – | 0.666 / 0.654 | – | – |

Approximate 95% half-widths: ±0.026 overall, ±0.043 on OBQA.

**Branch statistics:**
- **Overlap.** Mean pairwise Jaccard between a question's branches is 0.35. 55% (k = 2)
  and 60% (k = 8) of a candidate's statements also appear in another candidate's
  branch.
- **Mentions.** Only 49% (k = 2) and 69% (k = 8) of branches mention any of their own
  candidate's words that are not in the question.
- **Swaps.** With swapped branches, 29–32% of decisions change, but the choice follows
  its old branch only 12.5–15% of the time (chance 25%). The candidate's own text still
  dominates.
- **Leave-one-out.** A node moves its candidate's logit by 0.45 on average at k = 2,
  and 0.22 at k = 8. The centered logits move by 0.58–0.66, against a direct logit
  spread of 1.44.

**Interpretation.**
- **The model does use branch content, and uses it per candidate.** Swapping branches
  removes nearly all the gain (0.614 → 0.571). The gain is not a formatting effect.
- **The owner's impression is right: selection is the weak part.** Half of the
  branches at k = 2 are not about their candidate at all. They are statements about
  the question's topic, which every candidate retrieves. Giving every candidate the
  same pooled statements works as well as candidate-specific branches at k = 8. Much
  of what the model uses is topic knowledge, not evidence found by expanding the
  candidate.
- **Reading is not the bottleneck; retrieval is.** With the gold fact, the same
  trained model reaches 0.674 on OBQA, against 0.608 with its own retrieval and 0.574
  empty.
- Next (PLAN §10.1, step 3c): proposals anchored on the candidate's own concepts, and
  a second hop (R022 tests both without retraining).

**Limitations.** Evaluation only; one model; dev + calibration; approximate intervals.

---

## R020: Stage 2 matched experiment, EXPAND v2 (positive, with caveats)

- **Folder:** `result_logs/R020_2026-09-29_expand-v2-matched_knowledge_v0_r3/`
  (manifests, training logs, dev and test runs, selections, analyses, paired
  comparisons, `per_source_paired.txt`, calibrations, configs).
- **Question (decision 0012):** when each candidate reads its own branch of retrieved
  GenericsKB statements, is the decision better than with random statements and than
  the same training with empty branches? Does it improve as it reads more?

**Setup.**
- Three runs on `knowledge_v0` train (8,322), 2 epochs (1,042 steps), seed 0, all from
  R005, with widths per batch from {0, 1, 2, 4, 8}.
- Commit `a0b0c4f`, via `scripts/stage2_expand_v2.sh` on the RTX 2060. The manifests
  are marked dirty only by the owner's `interface/next.config.ts` edit.
- Selection by dev NLL at p32; collection on `knowledge_v0` revision 3.

| Model | Selected | Train time | Dev NLL, epoch 1 / 2 | Train loss, epoch 1 / 2 |
|---|---|---|---|---|
| Retrieved | `branch_v2_retrieved@277ebb247391` (final, epoch 2) | 894 s | 1.245 / 0.961 | 1.292 / 0.975 |
| Random | `branch_v2_random@494a3b380220` (epoch 1) | 894 s | 1.017 / 1.051 | 1.036 / 0.560 |
| None (direct control) | `branch_v2_none@9d4fae379848` (epoch 1) | 581 s | 0.962 / 1.086 | 1.000 / 0.492 |

- Retrieved and random ran for the same time, because they saw the same widths per
  batch; their losses differ.
- The retrieved model learned more slowly and did *not* overfit in 2 epochs, unlike
  both controls.

**Test results** (raw; about 1, 2, 4, 8 statements per candidate at p4 … p32):

| Model | b0 | p4 | p8 | p16 | p32 |
|---|---|---|---|---|---|
| Retrieved, accuracy | 0.537 [0.521, 0.554] | 0.592 | 0.608 | 0.609 | **0.611** [0.597, 0.626] |
| Random, accuracy | 0.582 | 0.561 | 0.557 | 0.550 | 0.556 [0.541, 0.571] |
| None, accuracy | 0.581 [0.566, 0.596] | = | = | = | = |
| Retrieved, NLL | 1.099 | 0.993 | 0.970 | 0.956 | **0.953** [0.928, 0.978] |
| Random, NLL | 1.013 | 1.038 | 1.038 | 1.043 | 1.049 |
| None, NLL | 0.998 | = | = | = | = |

- **Paired, retrieved − none:**
  - p8: accuracy +0.027 [+0.012, +0.043], NLL −0.027 [−0.050, −0.005];
  - p32: accuracy +0.029 [+0.014, +0.046], NLL −0.045 [−0.068, −0.023];
  - b0: accuracy −0.045 [−0.058, −0.031].
- **Paired, retrieved − random:** p32 accuracy +0.055 [+0.039, +0.071], NLL −0.096
  [−0.117, −0.076].
- **By source** (paired group bootstrap, retrieved − none, p32):
  - OBQA −0.008 [−0.052, +0.034];
  - **ARC-Easy +0.064 [+0.042, +0.085]**;
  - ARC-Challenge −0.023 [−0.054, +0.008].
- Against random at p32: OBQA +0.036 [−0.014, +0.080], ARC-Easy +0.084, ARC-Challenge
  +0.006.
- **Calibration.**
  - At p16 (the interface default): T = 1.186. Test ECE 0.033 → 0.019, NLL 0.956 →
    0.949. OBQA and ARC-Challenge improve; ARC-Easy ECE worsens slightly, 0.019 →
    0.046. **Applied.**
  - At p32: T = 1.171. Overall ECE worsens, 0.028 → 0.032. Rejected
    (`calibration_p32_rejected.json`).

**Interpretation.**
- **The first result in the project where expansion helps, and more budget helps
  more.**
  - The same trained model improves from 0.537 to 0.611 as its branches grow from 0 to
    about 8 statements per candidate, saturating at about 2.
  - It beats random statements at equal cost and the matched direct model, with
    intervals clear of zero.
  - This is the H1 pattern, on one seed and one suite.
- **The caveats matter.**
  1. The whole gain is on ARC-Easy. OBQA and ARC-Challenge, which need a less obvious
     fact or two hops, gain nothing.
  2. Without statements the model is 4.5 points *worse* than the direct model: it
     learned to depend on its branches.
  3. R021 shows much of what it reads is question-topic knowledge shared by all
     candidates, not evidence found by expanding each candidate.
- GenericsKB-Best contains about 199k statements derived from the ARC corpus, ARC's
  official knowledge source. This legitimately explains why ARC-Easy benefits most. It
  is not label leakage: the memory holds no questions or answers.

**Limitations.** One seed per model. The retrieved model's selected checkpoint is
epoch 2, while the controls' is epoch 1 (each by its own dev NLL). During collection
the memory's keys stayed on the CPU: a loading-order bug, fixed with R022. It made
searches slower but did not change their results. The budget grid is
per question, so the rare 5-option questions get fewer nodes per candidate.

---

## R019: EXPAND v2 training smoke run (pipeline check)

- **Folder:** `result_logs/R019_2026-09-29_expand-v2-smoke/` (configs, bundle manifest,
  training log). The weights were deleted after the run; not a model.
- **Question:** does branch reading (decision 0012) train on the RTX 2060 at the
  widest branch (8 statements per candidate), and how fast?
- **Setup:** `branch_v2_retrieved` with every batch at width 8, warm start from R005,
  480 train and 96 dev questions of `knowledge_v0`, 30 steps (batch 16, micro-batch
  8), fp16. Uncommitted code (committed with this entry).
- **Result:**
  - peak GPU memory 4.6 GB;
  - 14.8 training examples/s at width 8;
  - train loss 1.40 → 1.27 over 30 steps; dev loss 1.210;
  - one non-finite gradient step (fp16 scaler warm-up).
- **Consequence:** the matched runs (`scripts/stage2_expand_v2.sh`) fit in about
  1–1.5 h, since mixed widths are faster than width 8.
- **Limitations:** too short to say anything about learning.

---

## R018: memory `generics_v0` and the gates of decision 0012

- **Folder:** `result_logs/R018_2026-09-29_generics_v0-memory-gates/` (memory manifest
  and config, four access-recall reports, the probe script, its JSON results and
  stdout).
- **Question:** before training EXPAND v2, does a memory of statements about concepts
  hold knowledge that retrieval can find and the reader can use, better than WordNet
  definitions?

**Memory.**
- `generics_v0`: GenericsKB-Best (Hugging Face `community-datasets/generics_kb` at
  revision `aba8bf00…`, parquet sha256 `31a202ba…`, CC-BY-4.0).
- 1,020,868 rows give 1,015,610 unique statements, each keyed by its concept term.
- Keys come from the same frozen bge-small encoder as `wordnet_v0`. The build took
  5 min 45 s on the RTX 2060 (vectors: 780 MB, float16).
- Built at commit `b325cb8`. Its manifest is marked dirty: the owner's uncommitted
  `interface/next.config.ts` edit, plus the access-recall and search changes committed
  in `4afc3c2`.
- Exact search is now chunked over queries, since a million keys cannot be scored
  against all queries at once.

**Gate (a): access recall** (OpenBookQA train + dev, 5,288 questions, as in R015; novel
fact words, retrieved against random).

| k per candidate | generics, name match | generics, text match | WordNet, text match | WordNet, name match (R015) |
|---|---|---|---|---|
| 4 | 0.095 / 0.015 | 0.314 / 0.083 | 0.301 / 0.063 | 0.093 / 0.005 |
| 8 | 0.134 / 0.028 | 0.414 / 0.145 | 0.403 / 0.107 | 0.142 / 0.013 |
| 16 | 0.184 / 0.050 | 0.519 / 0.243 | 0.501 / 0.187 | 0.194 / 0.020 |
| 32 | 0.238 / 0.087 | 0.622 / 0.378 | 0.604 / 0.293 | 0.258 / 0.040 |

- "Text match" counts a fact word as found if it appears anywhere in a retrieved
  node's text (new option `--match text`).
- **Verdict: a tie.** Generics lead by only 1–2 points. Word coverage cannot
  distinguish a definition from a statement of the needed relation, so gate (b)
  decides.

**Gate (b): reading probe.**
- **Setup.** Frozen R005 reader, no training. `knowledge_v0` dev + calibration
  (1,368; test untouched).
- **Formats:**
  - branch: each candidate reads only its own top-k nodes in its own joint input,
    which is exactly EXPAND v2's input;
  - shared: the union of all branches as one context, as in R017.
- A sanity check passes: empty branches reproduce the direct decision exactly.

| Condition | generics_v0 acc / NLL | wordnet_v0 acc / NLL |
|---|---|---|
| no knowledge | 0.472 / 1.274 | 0.472 / 1.274 |
| branch, retrieved, k = 1 | **0.506** / 1.350 | 0.462 / 1.457 |
| branch, retrieved, k = 2 | 0.499 / 1.462 | 0.446 / 1.530 |
| branch, retrieved, k = 4 | 0.482 / 1.493 | – |
| branch, retrieved, k = 8 | **0.511** / 1.390 | – |
| branch, random, k = 1 / 2 / 4 / 8 | 0.409 / 0.398 / 0.393 / 0.381 | 0.397 / 0.352 (k = 1 / 2) |
| shared, retrieved, k = 1 / 2 | 0.507 / **0.541** | 0.452 / 0.487 |
| shared, random, k = 1 / 2 | 0.469 / 0.466 | 0.467 / 0.471 |

Approximate 95% half-width ±0.026 (binomial).

**Interpretation.**
- **Gate (b) is passed.** In the branch format, retrieved statements beat random ones
  by 10–13 points at every width, and beat WordNet by 4–5 points. The untrained reader
  gains up to 4 points from its own branch, and 7 points when it sees all branches
  (shared, k = 2).
- **The untrained reader is thrown by branch-specific text.** Random branches cost
  6–9 points, and NLL rises even when accuracy rises. It was never trained with
  candidate-specific context. Training with sampled widths (decision 0012) is meant to
  fix this.
- **Retrieval is dominated by the question's topic** (`examples` in the JSON): the four
  branches of a question often retrieve the same statements ("Most angler fish live in
  deep sea" under every candidate). Branches then differ little, which caps what
  branch reading can gain. The shared format's advantage fits this. Better,
  candidate-focused proposals are the first lever after v2's matched runs.
- WordNet is kept for reproducing R016. v2 uses `generics_v0`.

**Limitations.** No training; one probe format; dev + calibration only; approximate
intervals.

---

## R017: reading knowledge jointly (evaluation-only probe)

- **Folder:** `result_logs/R017_2026-09-29_joint-reading-probe_knowledge_v0/` (script,
  `context_probe.json`, stdout). No new model; per-example predictions were not saved.
- **Question:** R016 shows the EXPAND v1 head ignores concept content. Is the knowledge
  useless, or is the *way it is read* the problem? If the same frozen reader reads
  knowledge in its input text, next to the question and each candidate, does it use it?
- **Setup.** `knowledge_v0` test (4,046; OpenBookQA 500). Context conditions, rendered
  as the request's context:
  - none (as trained);
  - retrieved: the union of the top 4 `wordnet_v0` concepts per candidate, queried
    with "question candidate" exactly as EXPAND v1 does (about 16 glosses);
  - random: the same number of random concepts;
  - qret: the top 16 concepts for the question alone;
  - gold: OpenBookQA's core fact (OpenBookQA only; an oracle, never a model input).

  Readers: R005 (`direct_modernbert_ft_joint@7345459d01da`, the frozen path inside
  EXPAND v1) and the R016 direct model (`@365477be31f5`, fine-tuned on `knowledge_v0`
  with empty contexts). RTX 2060, fp16, about 20 min, commit `37b6751` plus the
  uncommitted probe script.

| Reader | Context | OBQA acc / NLL | ARC-Easy | ARC-Challenge | All acc / NLL |
|---|---|---|---|---|---|
| R005 | none | 0.388 / 1.494 | 0.587 | 0.395 | 0.507 / 1.175 |
| R005 | retrieved | 0.426 / 1.389 | 0.592 | 0.410 | 0.519 / 1.167 |
| R005 | random | 0.408 / 1.276 | 0.565 | 0.398 | 0.497 / 1.178 |
| R005 | qret | 0.446 / 1.248 | 0.595 | 0.422 | 0.526 / 1.115 |
| R005 | **gold fact** | **0.618 / 1.029** | – | – | – |
| R016 direct | none | 0.614 / 0.968 | 0.618 | 0.494 | 0.581 / 1.010 |
| R016 direct | retrieved | 0.566 / 1.065 | 0.599 | 0.454 | 0.553 / 1.084 |
| R016 direct | random | 0.562 / 1.132 | 0.563 | 0.465 | 0.535 / 1.160 |
| R016 direct | qret | 0.558 / 1.074 | 0.576 | 0.461 | 0.541 / 1.124 |
| R016 direct | **gold fact** | **0.736 / 0.711** | – | – | – |

Approximate 95% half-widths (binomial, not bootstrapped): ±0.043 on OBQA, ±0.015 on all.

**Concept-sensitivity probe** (R005 reader, as EXPAND v1 feeds it: `concept: … \n input`,
pooled, 64 test inputs × 16 retrieved concepts): mean cosine between the vectors of
*different concepts on the same input* is 0.931, higher than between *different
inputs* (0.893). The pooled vector is dominated by the question; the concept barely
moves it.

**Interpretation.**
- **Reading knowledge jointly works when the knowledge is right.** One relevant fact
  in the text lifts OpenBookQA by 23 points with no training (R005) and by 12 points
  on the knowledge-tuned reader. The reader can combine a fact with a candidate; the
  EXPAND v1 head, which only sees pooled vectors, cannot (R016).
- **WordNet glosses are weak knowledge for these questions.** Definitions ("predator:
  any animal that lives by preying…") rarely state the relation the question needs
  ("metal conducts heat"). The best gloss condition gains 2 points overall (qret,
  R005), and retrieved glosses beat random ones by about 2 points.
- **A reader must be trained with knowledge present.** The R016 direct model was
  fine-tuned on empty contexts; any added text, relevant or random, costs it 3–5
  points. Gold facts still help it.
- Consequences (decision 0012): expanded nodes must be *read in the reader's input*
  next to the candidate, with the reader trained that way; and the memory must hold
  *statements about concepts*, not only definitions.

**Limitations.** One pass, no training, a single context format (glosses joined as
sentences). Gold facts exist for OpenBookQA only. Intervals are approximate.

---

## R016: Stage 2 matched experiment, EXPAND v1 (negative for this design)

- **Folder:** `result_logs/R016_2026-09-29_expand-v1-matched_knowledge_v0_r2/`
  (manifests, logs, dev and test runs, selections, analyses, the paired comparisons,
  the calibrations not applied, the configs, `evidence_similarity.txt`).
- **Question (decision 0011):** when each candidate expands into concepts retrieved
  from `wordnet_v0` and the evidence aggregates back, is the decision better than with
  random concepts, with the same head and no concepts, and than a fine-tuned direct
  model? Does it improve as more concepts are read?

**Setup.** Four runs on `knowledge_v0` train (8,322), 3 epochs (1,563 steps), seed 0,
all initialized from R005, at commit `37b6751` (clean), via
`scripts/stage2_expand_v1.sh` on the RTX 2060. Selection by dev NLL (direct at b0,
expansion at p64); collection on `knowledge_v0` revision 2 (b0, p8 … p128).

| Model | Selected predictor | Train time | Examples/s |
|---|---|---|---|
| Direct, fully fine-tuned | `direct_modernbert_ft_joint@365477be31f5` (step 521 of 1,563) | 846 s | 29.5 |
| EXPAND v1, retrieved | `expand_v1_retrieved@65d4716203c1` (step 1,042) | 3,071 s | 8.1 |
| EXPAND v1, random | `expand_v1_random@4426ed952cbe` (step 1,042) | 3,430 s | 7.3 |
| EXPAND v1, none | `expand_v1_none@a15da5c621af` (step 1,042) | 292 s | 85.5 |

The direct model overfits quickly: dev NLL 0.988 after epoch 1, 1.058 after epoch 2,
2.449 after epoch 3 (train 0.16). The final weights (`@156d0e6982a4`) are not the
selected ones.

**Test results** (raw; b0 of the expansion models is the frozen R005):

| Model | b0 | p8 | p16 | p32 | p64 | p128 |
|---|---|---|---|---|---|---|
| Direct | 0.581 [0.566, 0.597] | = | = | = | = | = |
| Retrieved | 0.507 | 0.519 | 0.518 | 0.519 | 0.519 [0.503, 0.534] | 0.519 |
| Random | 0.507 | 0.519 | 0.520 | 0.519 | 0.519 | 0.518 |
| None | 0.507 | 0.516 | 0.516 | 0.516 | 0.516 | 0.516 |

- **NLL at p64:** direct 1.010 [0.988, 1.034]; retrieved 1.129 [1.107, 1.153]; random
  1.131; none 1.140; R005 (b0) 1.175. Test ECE: 0.024, 0.019, 0.023, 0.015.
- **Paired, p64:**
  - retrieved − random: accuracy 0.000 [−0.004, 0.003]; NLL −0.001 [−0.002, 0.000];
  - retrieved − none: accuracy +0.003 [−0.003, 0.009]; NLL −0.011 [−0.014, −0.008];
  - retrieved − direct: accuracy −0.063 [−0.077, −0.049]; NLL +0.119 [+0.104, +0.136].
- **By source (p64):** OBQA 0.532 vs R005 0.388; ARC-Easy 0.569 vs 0.588; ARC-Challenge
  0.410 vs 0.395. The direct model scores 0.614 / 0.617 / 0.494.
- **Budget:** flat from 8 to 128 concepts in total (predictions at p8 and p128
  correlate at 0.996).
- **The evidence ignores the concepts** (`evidence_similarity.txt`): the centered
  evidence added to each candidate at p64 correlates at 0.986 between retrieved and
  none, 0.990 between random and none, and 0.997 between retrieved and random. The
  learned same-branch bias is about 0 (−0.005, 0.003, 0.000).
- Calibration on the calibration split gives T = 0.978 (direct, b0) and 1.100
  (retrieved, p64). It does not improve test ECE (0.026 and 0.028 calibrated), so
  neither is applied; both are archived as `calibration_*_not_applied.json`.

**Interpretation.**
- **EXPAND v1 does not use concept content.** Retrieved and random concepts give the
  same predictions, and both match the head with no concepts to within 1–3% of the
  evidence's variance. The gain over R005 (OBQA +14 points) is the head re-scoring
  candidates for the new question style, which the no-concept control also learns.
- **Why** (R017): in EXPAND v1 a concept reaches the decision only as a pooled vector
  of "concept + input" from a frozen reader, and that vector is dominated by the input
  (cosine 0.93 between different concepts on one input). The head cannot tell
  concepts apart. The frozen reader never learned to judge a concept against a
  candidate.
- **This tests the design, not the idea.** R017 shows the same reader *does* use
  knowledge when it reads it in its input, and that WordNet definitions are weak
  knowledge for these questions. Decision 0012 changes both.
- **Lesson 13 again:** the frozen design kept the direct path from absorbing the task,
  but it also froze the only component able to relate a concept to a candidate.
- The practical model for passage-free science questions is now the fine-tuned direct
  model (0.581), still well below what a relevant fact would give it (R017).

**Limitations.** One seed per model. The direct model trained its encoder while the
expansion models kept it frozen (matched by design, as in decision 0011). The budget
grid counts total concepts at depth 1 only.

---

## R015: concept memory `wordnet_v0`, access recall (the Stage-2 gate)

- **Folder:** `result_logs/R015_2026-09-28_wordnet_v0-access-recall/`
- **Question** (PLAN §10.1, step 2): does retrieval from "question + candidate" surface
  the concepts a question needs, before any expansion model is trained? Nothing absent
  from the proposal pool can be rescued later (foundations §5.2).

**Setup.**
- **Memory:** `wordnet_v0`, all 117,659 WordNet 3.0 synsets (archive sha256
  `cbda5ea6…`). Each concept text is "name: gloss".
- **Keys:** a frozen `BAAI/bge-small-en-v1.5` at revision `5c38ec7c…` (CLS pooling,
  normalized, 384 dimensions, float16). Build time 95 s on the RTX 2060.
- **Search:** exact cosine.
- **Queries:** "question + candidate", one per candidate, with the model's
  retrieval-query prefix.
- **Reference:** OpenBookQA's gold core fact (train and dev questions only, never
  test; 5,288 questions).
  - Content words are lemmatized with WordNet morphy, after removing stopwords.
  - A fact word counts as found if it is a word of a retrieved concept's name.
  - "Novel" words are fact words absent from the question and candidates (5,011
    questions have at least one).
  - The chance reference is an equal number of random concepts.
- **Code:** commit `78e30d3` plus uncommitted memory code, committed with this entry.

**Results** (mean per-question recall; k is concepts per candidate, 4 to 5 candidates):

| k | All fact words | Random | Novel fact words | Random |
|---|---|---|---|---|
| 4 | 0.204 | 0.005 | 0.093 | 0.005 |
| 8 | 0.265 | 0.010 | 0.142 | 0.013 |
| 16 | 0.325 | 0.022 | 0.194 | 0.020 |
| 32 | 0.390 | 0.042 | 0.258 | 0.040 |

**Interpretation.**
- Retrieval brings in about a fifth to a quarter of the gold fact's *new* words at 16
  to 32 concepts per candidate, 6–18 times the random rate. There is real material for
  EXPAND to select from.
- No numeric gate threshold was declared beforehand, so this is reported as evidence,
  not as a passed bar.
- The qualitative example (`example_proposals.txt`) shows proposals dominated by the
  *question's* topic (gas, fluid, bottle) more than by each candidate. EXPAND's
  contextual scoring must separate them. A depth-2 expansion (concepts expanding into
  concepts) may reach words one hop further; to be measured when it exists.

**Limitations.**
- Word overlap with fact text is a proxy for relevance.
- WordNet glosses are definitions, not facts, so relations such as "liquids have
  definite volume" are only partly expressible as concepts.
- Recall is measured on OpenBookQA only; ARC has no gold facts.

---

## R014: `knowledge_v0` suite validation

- **Folder:** `result_logs/R014_2026-09-28_knowledge_v0-suite-validation/`
- **Suite:** `knowledge_v0` revision 1 (cases `3fd83e1b…`, labels `5bdce011…`).
  Deterministic (regenerated twice); `mcqa_v0` still reproduces its pinned hash.

**Contents.**
- OpenBookQA (with `fact1` as a proof, never an input), ARC-Easy and ARC-Challenge.
  The context is empty.
- Splits: train 8,322; dev 665; calibration 703 (the sources' validation splits,
  divided by question); test 4,046.
- 8 rows with duplicate options were skipped.

**Result** (lexical fixture, test): accuracy 0.251 [0.243, 0.258].
- By source: OpenBookQA 0.242, ARC-Easy 0.266, ARC-Challenge 0.225.
- There is no lexical shortcut without a passage, unlike SciQ (R002).

---

## R013: isolated-encoding workspace feasibility (negative diagnostics)

- **Folder:** `result_logs/R013_2026-09-28_isolated-workspace-feasibility/`: scripts,
  configs, outputs and the GPU smoke bundle manifest. The CPU probes are standalone
  scripts; their models were not saved.
- **Question:** before committing about 2 h of GPU time to decision 0009, can a
  workspace whose rounds are the *only* route between sentences learn rule chaining
  that transfers to unseen worlds?

**Probes** (all train on depths 1–2 or 1–3 and evaluate on held-out worlds):

| Probe | Encoder | Result |
|---|---|---|
| CPU 1 | Isolated workspace, hashing, bag of words | Memorizes: train loss 0.19; test 0.21–0.34 at every round count (0–8) |
| CPU 2 | Isolated workspace, hashing with 2 layers (order-aware) | Collapses: train loss stays at about ln 4 |
| CPU 3 | Memory network (recurrent query, 1–4 hops trained) | Memorizes: train loss 0.12; test 0.19–0.33 for 1–8 hops, even at trained depths |
| GPU | Isolated workspace, ModernBERT-640 warm-started from R005, 150 steps | Collapses: dev logit spread 0.010, dev loss 1.3868 |

**Interpretation.**
- Mechanisms that must learn relational matching over *nonce* symbols from scratch
  either collapse to uniform or memorize world-specific symbol combinations.
- None of them transfers to new worlds, even at trained depths.
- The only component that has learned rule chaining that transfers is the pretrained,
  full-context encoder (R008, R012).
- A pooled sentence vector probably cannot carry "which word, on which side of the
  rule" well enough for rounds of attention to chain on.
- Decision 0009 **fails its feasibility check** and was not run at scale.

**Limitations.**
- Short, untuned probes: one learning rate each, and 150 steps on the GPU.
- Lesson 3 applies: short runs can mislead. The collapse signature (logit spread about
  0) and the memorization signature (train ≪ test loss) are clear, though.

---

## R012: Stage 3a matched experiment (negative for this design: rounds unused)

- **Folder:** `result_logs/R012_2026-09-28_stage3a-matched_reasoning_v1_r2/`
- **Question (H1, decision 0008):** does one fixed trained model become more accurate at
  held-out depths as its reasoning rounds grow? And does a workspace over the input's
  sentences beat an equal-cost latent control and a converged direct baseline?

**Setup.** Three runs matched on everything except the workspace:
- **Initialization:** warm start from R005 (`@7345459d01da`).
- **Data:** the 20,000 padded examples of `microworld_v1_train20k` (depths 1–3).
- **Schedule:** 1,250 steps, the same optimizer and schedule, seed 0.
- **Hardware and code:** the RTX 2060, overnight via `scripts/stage3a_reasoning_v1.sh`,
  at commit `c66da2b`, clean.
- **Workspaces:** rounds sampled from 1–4 per batch, with deep supervision.
- **Selection:** by dev NLL, at b0 for the direct model and at r4 for the workspaces.
- **Collection:** on `reasoning_v1` revision 2 (budgets b0, r1, r2, r4, r8, r16).

| Model | Predictor (selected) | Train time | Examples/s | Selected |
|---|---|---|---|---|
| Direct baseline | `direct_modernbert_ft_joint_640@4810281132d1` | 6,220 s | 3.22 | final |
| Statement workspace | `ws_statements_640@5259bb08c5e9` | 6,296 s | 3.18 | step 1,000 |
| Latent control | `ws_latent_640@1a8013668375` | 6,353 s | 3.15 | final |

**Convergence gate: met for all three.** Dev NLL (depths 1–3) reached ≈0 and was flat
over the last checkpoints. The latent control started learning late: dev NLL was still
1.38 at step 375 and 0.11 at step 625.

**Results** (test, raw accuracy; every budget is shown where it varies):

| Depth | 1–3 | 4 | 5 | 6 | 7 | 8 | all |
|---|---|---|---|---|---|---|---|
| Direct, every budget | 1.00 / 1.00 / 0.996 | 0.913 | 0.614 | 0.459 | 0.481 | 0.529 | 0.753 [0.732, 0.772] |
| Statements, b0 … r16 | 1.00 | 0.909 | 0.674 | 0.541 | 0.522 | 0.558 (0.555 at r1–r4) | 0.778 [0.759, 0.799] |
| Latent, b0 … r16 | 1.00 | 0.918 | 0.658 | 0.525 | 0.483 | 0.471 | 0.759 [0.737, 0.782] |

- **Accuracy does not change with the budget.** For both workspaces, b0 and r16 give
  the same accuracy at every depth, except one depth-8 example.
- **NLL is flat, or slightly worse, with rounds.** Statement workspace overall NLL: b0
  1.486, r1 1.512, r16 1.496. Latent: 1.252, then about 1.253–1.256.
- **Paired differences against the direct baseline** (identical at every budget):
  statements +0.025 [0.005, 0.046]; latent +0.007 [−0.014, 0.027]. These come from the
  *budget-0* paths, which bypass the workspace entirely. They are therefore differences
  in how the shared encoder and scorer trained, not effects of reasoning rounds. With
  one seed per model, they indicate run-to-run spread of a few points.
- **Why the rounds do nothing** (`workspace_readout_norms.txt`). The zero-initialized
  workspace readout grew only to ‖w‖ ≈ 0.07, against direct-path logit spreads of
  21–32. The direct path (22 fine-tuned layers reading the whole context) solved the
  training depths by itself; with training loss ≈0, the workspace received almost no
  gradient. The rounds were never *needed* during training.
- **The converged direct baseline is a real improvement over R011** (depth 4: 0.48 →
  0.91; depth 8: 0.40 → 0.53), but **it is badly overconfident on held-out depths**:
  - raw ECE 0.34–0.44 at depths 5–8, and NLL up to 4.4;
  - the temperature fitted on the calibration split (depths 1–3, where it is always
    right) is 0.071, which sharpens further: calibrated NLL 16.7 at depth 6 and 7.7
    overall.

  All three b0 calibrations are archived as `calibration_b0_rejected.json` and moved to
  `artifacts/calibrations_rejected/`, so the interface never applies them.
- **Reading-comprehension retention** of the direct baseline (`mcqa_v0` dev;
  `mcqa_dev_score_*.json`):
  - accuracy 0.763 and NLL 0.817, against R005's 0.781 and 0.585;
  - RACE 0.729, SciQ 0.941;
  - most of the accuracy is kept, but it is more overconfident.
  - Three early checkpoints (steps 125–375) were also collected on `mcqa_v0` dev by
    mistake (an unintended `collect-checkpoints` call, stopped). They are kept but
    unused.

**Interpretation.**
- **This is not evidence against H1. It shows that this training design gives the
  rounds no role.** A residual workspace on top of a direct path that can already fit
  the training distribution learns nothing.
- For budgeted computation to be learned and tested, the extra computation must be
  *necessary* on the training data. That means:
  - the direct path cannot be allowed to solve the task alone, for example by
    encoding each sentence independently so that crossing sentences requires workspace
    rounds;
  - and/or the training data must contain problems a single pass cannot solve.
- The follow-up design is decision 0009.
- **Calibration under distribution shift (H5) fails badly** for a model that is perfect
  in-distribution. No held-out-depth calibration method exists yet.

**Limitations.**
- One seed per model.
- Selection of the workspaces used r4.
- The dense block and the readout sizes were not tuned.

---

## R011: warm-started direct baseline on `reasoning_v1` (undertrained)

- **Folder:** `result_logs/R011_2026-09-27_warm-baseline_reasoning_v1/`
- **Question:** what is the reference direct baseline's accuracy per depth on the new
  suite (PLAN §10, step 2)?
- **Predictor:** `direct_modernbert_ft_joint_640@aec6ecf34c53` (`mbft640_joint_reasoning_v1_warm`),
  initialized from R005 (`@7345459d01da`).

**Setup.**
- The R008 recipe with the 640-token config, on the 4,800 `microworld_v1` train
  examples: depths 1–3, padded to 46 sentences.
- 1 epoch = 300 steps, a checkpoint every 50 steps.
- The owner ran it: 1,530 s at 3.14 examples/s.
- Code: commit `97d9144`, clean.
- Selection by dev NLL (`selection.json`) chose the **final** weights.

**Learning curve** (dev, 800 examples, depths 1–3; `dev_accuracy_by_checkpoint.json`):

| Step | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| Dev accuracy | 0.272 | 0.276 | 0.365 | 0.561 | 0.677 | 0.700 |
| Dev NLL | 1.384 | 1.379 | 1.304 | 0.890 | 0.656 | 0.586 |

**Results** (test; calibrated T = 1.056 changes almost nothing):

| Depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| n | 191 | 181 | 236 | 208 | 184 | 183 | 209 | 208 |
| Accuracy | 0.744 | 0.729 | 0.610 | 0.476 | 0.402 | 0.317 | 0.402 | 0.399 |
| 95% interval | [.68, .80] | [.66, .80] | [.55, .67] | [.41, .54] | [.33, .48] | [.26, .39] | [.34, .47] | [.33, .47] |
| NLL (raw) | 0.535 | 0.507 | 0.781 | 1.231 | 1.460 | 1.530 | 1.353 | 1.297 |
| ECE (raw) | 0.047 | 0.062 | 0.071 | 0.173 | 0.220 | 0.257 | 0.174 | 0.147 |

- Overall: accuracy 0.510 [0.486, 0.535], NLL 1.085 raw and 1.071 calibrated.
- Conjunctive and plain chains score about the same (0.515 and 0.508).

**Interpretation.**
- **Undertrained.** This is not yet a usable reference.
  - On depths it was trained on, it reaches only 0.61–0.74. R008 reached about 100% on
    the unpadded `reasoning_v0`.
  - Dev accuracy was still rising steeply when the linear schedule brought the learning
    rate to zero.
  - Learning started later than in R008: around step 150, against 75. The padded
    46-sentence inputs are about 3 times longer and mostly distractor text.
- **Accuracy falls with depth** from 0.74 to about 0.40, and stays above chance at every
  held-out depth.
- **Overconfidence grows at held-out depths**: ECE rises from 0.05 to 0.26.
- The upturn at depths 7–8 (0.40 against 0.32 at depth 6) lies within overlapping
  intervals. It is not evidence of non-monotonicity.
- **Consequence:** a Stage-3a mechanism compared against this baseline would absorb the
  baseline's training deficit. The reference baseline must first be trained to near
  ceiling on depths 1–3 (PLAN §10).

**Limitations.** One seed, one short schedule.

---

## R010: `reasoning_v1` suite validation (oracle and lexical fixture)

- **Folder:** `result_logs/R010_2026-09-27_reasoning_v1-suite-validation/`
- **Question:** are the new suite's labels recoverable from the text alone, at every
  depth? And is it free of the lexical shortcut that the length padding could
  introduce?
- **Suite:** `reasoning_v1` revision 1, dataset `microworld_v1` (cases `e828fb9f…`,
  labels `ace50c60…`). Generator revision 2, deterministic (regenerated twice).
  Revision-1 datasets (`microworld_v0`, `microworld_train20k_v0`,
  `microworld_d1_train_v0`) still reproduce their hashes byte for byte.

**Results** (test):

| Predictor | Depths 1 to 8, accuracy each | All |
|---|---|---|
| `microworld_oracle` (symbolic) | 1.000 | 1.000 |
| `lexical_overlap@1` | 0.250 | 0.250 |

- n per depth: 191, 181, 236, 208, 184, 183, 209, 208.
- Input size: 46 sentences everywhere. Joint ModernBERT tokens: median 481, maximum
  609 (`token_counts.txt`, from a 100-world trial with the same settings).

**Interpretation.** The suite is valid for its question:
- labels agree with the text;
- word overlap carries no signal at any depth;
- length is constant by construction.

It needs a 640-token encoder.

**Limitations.** Unary predicates only, and template families are shared between train
and test.

---

## R009: cold start with a depth-1-only curriculum (negative)

- **Folder:** `result_logs/R009_2026-09-27_depth1-curriculum-cold_reasoning_v0/`
- **Question:** does an easier curriculum (depth 1 only) give a from-scratch model the
  foothold it lacked in R007?
- **Predictor:** `direct_modernbert_ft_joint@a30dafb645da` (`mbft_joint_reasoning_d1_diag`).

**Setup.**
- Same model and optimizer settings as R007.
- 4,800 random examples from `microworld_d1_train_v0`: 5,000 fresh depth-1 worlds, no
  context overlap with `reasoning_v0` evaluation (checked).
- 1 epoch = 300 steps, 431 s at 11.1 examples/s.
- Code: commit `51c5f8d`, clean.
- Evaluated on `reasoning_v0` dev only, scored by `metanoia-analysis select`
  (`dev_score_*.json`).

**Result.**
- Dev accuracy 0.247 and NLL 1.3863, both at chance.
- Training loss 1.388, never moving; the same collapse as R007.

**Interpretation.**
- At this budget, an easier task did not escape the collapse. Contrast R008, where a
  reading-comprehension warm start did.
- The foothold therefore seems to come from prior skill at matching candidates to
  context evidence, not from task difficulty.
- Depth-0 lookups were not tried (the generator does not support them), and 300 steps
  is short (§13, lesson 3).

---

## R008: warm start from R005, then rule puzzles (positive; exposes the depth gap)

- **Folder:** `result_logs/R008_2026-09-27_warm-start-from-R005_reasoning_v0/`
- **Question:** does starting from R005's weights, which already separate micro-world
  candidates (R007 diagnostic), avoid the collapse?
- **Predictor:** `direct_modernbert_ft_joint@b5eb1b748344` (`mbft_joint_reasoning_warm_diag`),
  with `training.initialized_from = direct_modernbert_ft_joint@7345459d01da` (R005).

**Setup.**
- Same model and optimizer settings as R005–R007.
- 4,800 random examples from `microworld_train20k_v0` (depths 1–3).
- 1 epoch = 300 steps, 528 s at 9.1 examples/s.
- Code: commit `51c5f8d`, clean.
- A single final checkpoint, collected on the full suite.

**Training.** The loss left chance by step 75 (1.27), reached 0.27 by step 150 and
0.004 by step 275.

**Results on `reasoning_v0`** (raw):

| Split / slice | n | Accuracy | NLL | ECE |
|---|---|---|---|---|
| dev, all (depths 1–3) | 300 | 1.000 | 0.0015 | 0.001 |
| test, depths 1 / 2 / 3 | 155 / 132 / 154 | 0.994 / 1.000 / 1.000 | 0.014 / 0.000 / 0.002 | ≤ 0.006 |
| **test, depth 4 (held out)** | 159 | **0.786 [0.721, 0.847]** | **0.804** | **0.153** |
| test, all | 600 | 0.942 [0.923, 0.960] | 0.217 | 0.042 |

- **Calibration.** T = 0.754 was fitted on the calibration split, which contains
  in-distribution depths only. It sharpens the output and makes held-out depth 4 worse
  (NLL 0.80 → 1.00; ECE 0.153 → 0.156). Overall calibrated NLL is 0.270.
- **Length confound** (`diagnostics/length_by_depth.txt`).
  - No input is truncated: the maximum is 357 tokens against the 384 limit.
  - Median length grows with depth: 102, 132, 167 and 198 tokens.
  - The shortest third of depth-4 inputs (103–182 tokens, as short as depth-2/3
    inputs, which score 100%) still scores only 0.830.
  - So derivation depth, not length, drives most of the drop. The longest third scores
    0.698, so length adds some difficulty.
- **Forgetting check** (`mcqa_v0` dev): accuracy 0.759, NLL 0.673, against R005's 0.781
  and 0.585. By source: RACE 0.724 (R005: 0.749), SciQ 0.938 (R005: 0.945). Most of the
  reading skill is kept.
- **Qualitative, not archived.** Both this model and R005 answered three simple English
  two-step deductions correctly: birds that fly; study → pass → diploma; heated metal →
  expands → longer. This model's logits are far larger (about +35, against +4 for R005).

**Interpretation.**
- **The direct baseline can learn rule chaining, and does so easily when started from
  a reading-comprehension model.** R006, R007 and R009 were optimization collapses,
  not capacity limits.
- **`reasoning_v0` at trained depths (1–3) is solved by a direct scorer.** At those
  depths it has no headroom for reasoning operations.
- **The headroom is depth extrapolation.** One step beyond the trained depths, accuracy
  falls to 0.79 and the model stays confident. That is exactly H1's territory: can
  extra, budgeted computation extend the chain the model can follow?
- **Training protocol is part of every comparison.** Every model compared on these
  suites must receive the same initialization and curriculum (foundations §13.4:
  "match curriculum and supervision").

**Limitations.**
- One seed and 300 steps.
- Depth 4 is the only held-out depth, and the depth-4 interval is wide.
- Only one training distribution was tried, with templates shared between train and
  test.

---

## R007: fine-tuned joint ModernBERT on 20k fresh train worlds (negative: collapse)

- **Folder:** `result_logs/R007_2026-09-27_finetuned-modernbert-train20k_reasoning_v0/`
- **Question:** the data-scale follow-up to R006. With 11 times more examples and
  steps, does the fine-tuned direct baseline learn `reasoning_v0`?
- **Predictors:**
  - final: `direct_modernbert_ft_joint@9a5fa4bebe73`;
  - selected (step 3,125, dev NLL): `@105ca5406ccf`.

**Setup.**
- Model and optimizer settings as in R005 and R006, config
  `mbft_joint_reasoning_train20k`.
- Training data: 20,000 train-only worlds from `microworld_train20k_v0` (depths 1–3,
  seed 20260927), with no context overlap with `reasoning_v0` dev or test. Dev came
  from `reasoning_v0`.
- 3 epochs = 3,750 steps. The owner ran it: 6,573 s at 9.13 examples/s on the RTX 2060.
- Code: commit `11c072a`, **clean**, the first recorded run without uncommitted
  changes.
- Selection: six checkpoints were collected on dev and selected by NLL
  (`selection.json`). All scored between 1.38625 and 1.38649.

**Results** (raw; temperature fit T = 148.4, at the search bound):

| Split / slice | n | Accuracy | NLL |
|---|---|---|---|
| dev, all | 300 | 0.249 [0.198, 0.299] | 1.3862 |
| test, all | 600 | 0.269 [0.236, 0.300] | 1.3863 |
| test, depth 1 / 2 / 3 / 4 | 132–159 each | 0.305 / 0.273 / 0.260 / 0.239 | 1.386 each |

- Training loss per epoch: 1.3877, 1.3871, 1.3874. It was at ln 4 from start to end.

**Diagnostic** (`diagnostics/dilution.py`, 40 dev examples per dataset). It measures how
different the four candidates' pooled joint vectors are, and how spread out their
logits are:

| Model | Data | Candidate spread / norm | Mean cosine | Logit std |
|---|---|---|---|---|
| R005 `mbft_joint_mcqa_v0` | micro-worlds | 0.074 | 0.992 | **1.39** |
| R005 `mbft_joint_mcqa_v0` | RACE + SciQ | 0.108 | 0.981 | 2.00 |
| R007 final | micro-worlds | 0.021 | 0.999 | **0.001** |
| R007 final | RACE + SciQ | 0.069 | 0.987 | 0.006 |

**Interpretation.**
- **More data did not help, so the problem is not generalization.** The model never
  fit its own training set.
- **The failure is a collapse.** The model learned to give every candidate the same
  score (logit std 0.001), which is the loss-minimizing answer when it finds no signal.
  It then stayed on that plateau.
- **This is not a hard representation limit.** R005's weights, never trained on
  micro-worlds, still separate these candidates (logit std 1.39). Joint mean pooling
  dilutes nonce-name candidates (spread 0.07 against 0.11 on RACE), but it does not
  erase them.
- **The likely cause is the suite design.** The shortcut controls of decision 0003
  remove every shallow cue, so at initialization there is no partial solution to
  climb. RuleTaker-style datasets include depth-0 lookups that provide one (recalled,
  not measured here).
- **Hypotheses to test next**, as short diagnostics:
  - (a) a warm start from R005, which already discriminates candidates;
  - (b) an easier curriculum: depth 1 only, then depth 0 if needed.
- **Consequence for the plan:** "the direct baseline is at chance" is **not yet
  evidence that `reasoning_v0` needs reasoning operations**. It may be an optimization
  failure that any architecture trained this way would share.

**Limitations.**
- One seed and one learning-rate setting.
- The diagnostic used only 40 examples.
- The collapse explanation fits the logit statistics but is not proven causal.

---

## R006: fine-tuned joint ModernBERT on `reasoning_v0`, 3 epochs (inconclusive)

- **Folder:** `result_logs/R006_2026-09-27_finetuned-modernbert-3ep_reasoning_v0/`
- **Question:** does the strongest Stage-1 baseline, the fine-tuned joint direct scorer
  of R005, already solve the controlled reasoning suite? If it does, `reasoning_v0` is
  too easy for Stages 3–4 (PLAN §10).
- **Predictors:**
  - final: `direct_modernbert_ft_joint@32baf83cf99d`;
  - selected checkpoint (step 113): `@e96d68733f99`.

**Setup.**
- Same model config and optimizer settings as R005 (`mbft_joint_reasoning_v0_e3`),
  1,800 train examples, 3 epochs (339 steps), one checkpoint per epoch.
- Measured: 601.8 s of training at 8.97 examples/s on the RTX 2060.
- Code: commit `d0de1de` plus an uncommitted README edit (`dirty: true`). The edit was
  documentation only.
- **This is the first use of dev-based checkpoint selection.**
  1. `collect-checkpoints` collected the three checkpoints on dev.
  2. `metanoia-analysis select` chose by dev NLL: step 113 at 1.3862, against 1.3863
     and 1.3864. The artifact is `selection.json`.
  3. Only the chosen checkpoint was collected on the full suite.

**Results** (raw; the fitted temperature is T = 0.075, which only sharpens noise, so the
calibrated numbers are not meaningful):

| Split / slice | n | Accuracy | NLL |
|---|---|---|---|
| dev, all | 300 | 0.251 [0.213, 0.286] | 1.3862 |
| test, all | 600 | 0.259 [0.226, 0.293] | 1.3863 |
| test, depth 1 / 2 / 3 / 4 | 155–159 each | 0.268 / 0.284 / 0.255 / 0.232 | 1.386 each |

- Training loss stayed at ln 4: 1.388 → 1.388 → 1.385.
- Unlike the frozen models in R001, the fine-tuned model **does not even fit the
  training set**.

**Interpretation.** The run is **inconclusive**, not negative.
- R003 and R005 showed that the same setup can sit at ln 4 for hundreds of steps and
  then learn.
- 339 steps on 1,800 examples is small next to R005's 6,217 steps on 99k examples.
- Published rule-reasoning transformers (RuleTaker-style) were trained on tens of
  thousands of examples. This is recalled, not measured here.
- Next step: the data-scale follow-up `mbft_joint_reasoning_train20k`:
  - 20,000 fresh train-only worlds (`microworld_train20k_v0`), with no context overlap
    with `reasoning_v0` dev or test (checked);
  - 3,750 steps, about 2 h, handed to the owner.

**Limitations.** One seed, a short schedule, and a linear LR decay that reaches zero
after only 339 steps.

---

## R005: fine-tuned ModernBERT on the full RACE + SciQ training split

- **Folder:** `result_logs/R005_2026-09-27_finetuned-modernbert-full_mcqa_v0/`
- **Predictor:** `direct_modernbert_ft_joint@7345459d01da`, bundle
  `artifacts/predictors/mbft_joint_mcqa_v0`.
- **Question:** does the direct scorer give accurate and calibrated probabilities on
  realistic multiple-choice questions once the encoder is fine-tuned?

**Setup.**
- Model config `direct_modernbert_ft_joint`: ModernBERT-base at revision `8949b909…`,
  fine-tuned, 384 tokens, mean pooling, gradient checkpointing, joint candidate
  encoding. Scorer: latent 256, hidden 512, dropout 0.1.
- Training config `mbft_joint_mcqa_v0`:
  - 1 epoch over all 99,459 train examples;
  - batch 16 (micro-batch 4), 6,217 steps;
  - AdamW with encoder LR 2e-5 and scorer LR 1e-4;
  - 6% warmup, then linear decay;
  - fp16 mixed precision, seed 0.
- Hardware: RTX 2060 6 GB. The owner ran it overnight. Training wall time was not
  recorded (estimated at about 5 h). Collection with fp16 inference took 9.5 min for
  11,802 examples (about 21 examples/s).
- Code: commit `d65d89c` with uncommitted changes. The diff hash is in `run.json`.

**Results** (calibrated, T = 1.056):

| Split / slice | n | Accuracy | NLL | ECE |
|---|---|---|---|---|
| dev, all | 2,890 | 0.781 [0.764, 0.797] | 0.585 | 0.021 |
| test, all | 5,922 | **0.768 [0.756, 0.779]** | **0.608 [0.584, 0.636]** | 0.016 |
| test, RACE | 4,931 | 0.730 [0.716, 0.742] | 0.708 | 0.019 |
| test, SciQ | 991 | 0.956 [0.941, 0.968] | 0.113 | 0.017 |

- Before calibration (raw): test accuracy 0.768, NLL 0.610, ECE 0.019. Calibration
  barely matters: the fine-tuned model is already close to calibrated on this
  distribution.
- Final training loss 0.800; dev loss 0.604; 6 steps skipped by the fp16 grad scaler.
- **Paired comparison against R002 (lexical), test, raw**
  (`comparison_vs_lexical.analysis.json`):
  - accuracy +0.369 [0.354, 0.383];
  - NLL −6.81 [−7.11, −6.52];
  - Brier −0.724 [−0.746, −0.700];
  - wrong→right 45.8%, right→wrong 4.4%.

**Interpretation.**
- The direct scorer with a fine-tuned encoder is a usable decision model on this
  distribution. It is the first non-trivial Stage-1 result.
- RACE accuracy (0.73) is in the range usually reported for fine-tuned base-size
  encoders. That comparison is from memory and was not measured here.
- A single qualitative check shows the right answer ranked first but with a large
  share of probability on a wrong option (53% against 33%). Inference-type questions
  are uncertain.

**Limitations.**
- One seed and no tuning.
- Final epoch only, with no dev-based checkpoint selection.
- Training time was not logged.
- The public benchmarks may overlap with ModernBERT's pretraining data.
- ECE is binned (15 equal-width bins).
- Says nothing about reasoning depth, because R001 shows `reasoning_v0` is still
  unsolved.

---

## R004: fine-tuned ModernBERT, quick run on 2,400 SciQ examples

- **Folder:** `result_logs/R004_2026-09-27_finetuned-modernbert-sciq-quick_mcqa_v0/`
- **Predictor:** `direct_modernbert_ft_joint@98d789393a90` (`mbft_joint_sciq_quick`).
- **Setup:**
  - same model config as R005;
  - trained on the `mcqa_sciq_v0` dataset, 2,400 random train examples, 1 epoch
    (150 steps), about 8 min;
  - evaluated on `mcqa_v0`.

**Results** (test, calibrated, T = 2.641):

| Slice | Accuracy | NLL | ECE |
|---|---|---|---|
| all | 0.426 [0.412, 0.439] | 1.196 | 0.041 |
| RACE | 0.328 | 1.368 | 0.038 |
| SciQ | 0.910 | 0.340 | 0.135 |

- On SciQ the raw NLL (0.217, ECE 0.020) is **better** than the calibrated one. A
  single temperature fitted on a mixture dominated by RACE over-softens the easy SciQ
  slice.

**Interpretation.**
- Training on one source does not transfer to the other: RACE stays near the lexical
  baseline (0.314).
- A global temperature can damage calibration on a slice when slices differ in
  difficulty.

---

## R003: two mixed RACE + SciQ smoke runs did not learn (negative)

- **Folder:** `result_logs/R003_2026-09-27_mixed-smoke-runs_no-learning/`
- **Predictors:**
  - `…@e2e17e2b2b77`: the first 2,400 train rows, all RACE;
  - `…@36bc7cc8c3db`: a random sample of 2,400.
- **Setup:** same model and optimizer settings as R005; 150 steps each; no collection
  run, so the evidence is the training logs only.
- **Result:** train loss 1.387 / 1.388 and dev loss 1.386 / 1.386. There was no
  movement off chance.
- The first run's training config file was later renamed and changed. Its exact config
  is the one recorded in its `bundle.json`.

**Related observation (not archived).** An in-process diagnostic script trained the same
model on 1,600 SciQ-only examples for 100 steps. It reached 0.863 dev accuracy on 300
SciQ dev examples, and the loss started falling around step 50. The script was deleted,
so this observation has no artifact.

**Interpretation.**
- R005 shows RACE is learnable. It needs far more than 150 steps; the transition point
  is unknown because R005 logged the loss only every 50 steps and the owner's console
  log was not kept.
- Short smoke runs on RACE cannot tell a broken setup from a slow start. Use a SciQ
  smoke run to check that the pipeline learns at all.

---

## R002: lexical overlap fixture on `mcqa_v0`

- **Folder:** `result_logs/R002_2026-09-27_lexical-baseline_mcqa_v0/`
- **Predictor:** `lexical_overlap@1`, which has no training.

**Results** (test, raw; this fixture is not calibrated):

| Slice | Accuracy | NLL | ECE |
|---|---|---|---|
| all | 0.399 [0.387, 0.412] | 7.42 | 0.431 |
| RACE | 0.314 | 8.78 | 0.516 |
| SciQ | 0.820 | 0.683 | 0.221 |

**Interpretation.**
- Word overlap between an option and the passage solves 82% of SciQ. SciQ accuracy
  alone is therefore weak evidence of understanding.
- Report RACE separately, and compare against this baseline, not only against chance.

---

## R001: frozen pretrained encoders on `reasoning_v0` (negative)

- **Folder:** `result_logs/R001_2026-09-27_frozen-encoders_reasoning_v0/`
- **Predictors:**
  - `direct_modernbert@471b2717df4f`: ModernBERT-base `8949b909…`, frozen, mean
    pooling;
  - `direct_bert@7056aff3d45a`: BERT-base-uncased `86b5e093…`, frozen, mean pooling.
- **Shared setup:**
  - scorer: latent 128, hidden 256;
  - training: 15 epochs, LR 3e-4, batch 32;
  - about 12 min (ModernBERT) and 8 min (BERT) on the RTX 2060.

**Results:**

| Predictor | Test acc | Raw NLL | Cal. NLL | T |
|---|---|---|---|---|
| Frozen ModernBERT | 0.258 [0.225, 0.295] | 1.620 | 1.386 | 148.4 (search bound) |
| Frozen BERT | 0.233 [0.200, 0.267] | 1.548 | 1.386 | 148.4 (search bound) |

- Both overfit: train loss fell to 1.13 / 1.19, while dev loss rose from epoch 1, to
  1.61 / 1.51.
- No depth or conjunction slice differs from chance.

**Interpretation.**
- A frozen, mean-pooled encoder plus the one-attention direct scorer cannot do
  multi-hop chaining over nonce words from 1,800 examples.
- Neither pretrained encoder helps over the hashed-token encoder (R000).
- This suite remains the open test for the reasoning operations (Stages 2–4).

---

## R000: hashed-token direct scorer on `reasoning_v0` (CPU era, before this log)

- **Source:** `handoff.md` §2. The artifacts were not copied to the GPU machine, so
  there is no folder.
- **Predictor:** `direct_hashing@21a3aeb6b1c0`.
- **Result:** test accuracy 0.219 [0.189, 0.253]; raw NLL 3.06; calibrated NLL 1.388
  with T = 148.4, at the search bound.
- For comparison, the oracle scored 1.000 and the lexical fixture 0.250.
- To reproduce, follow the workflow in `metanoia/README.md`.
