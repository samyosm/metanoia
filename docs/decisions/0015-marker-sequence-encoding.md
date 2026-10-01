# 0015 — Single-sequence marker encoding (after Laya), and what it needs

**Context.**
- PLAN_V3 (owner, 2026-10-01) adopts Laya's *strategy*, not its weights: one marked sequence
  for all options, soft or broad training, calibration, selective answering. `LAYA_STUDY.md`
  explains why it works.
- The current direct model encodes each candidate with the input in its own pass (decision
  0005, "joint", N + 1 encoder passes) and compares candidates in a set scorer over pooled
  vectors. It is order-equivariant by construction but slow (5.9 training examples/s on the
  RTX 2060) and compares candidates only after pooling.

**Choice.** `scorer.candidate_encoding = "sequence"` with a `[scorer.sequence]` section
(`SequenceConfig`). Implemented in `metanoia/encoding/sequence.py` (rendering and
tokenisation) and `metanoia/sequence_model.py` (the model).

1. **Format.** `[CLS] question: Q [SEP] [MASK] opt0 [MASK] opt1 … [SEP] context [SEP]`.
   Options come before the context under a head budget (`head_max_len`, `option_max_tokens`),
   so truncation cuts the context, never an option. The score of an option is read from the
   final state of its `[MASK]` marker.
2. **Head.** Two transformer layers over the whole sequence (`head_layers`), then a LayerNorm
   and an MLP on each marker. The candidate scorer of decision 0005 is not used. The head
   is stored as the model's `scorer` so bundles, learning-rate groups and `device` work
   unchanged.
3. **Order robustness.** The marker format depends on option order (R032), unlike joint
   encoding. Three tools, all selectable:
   - `shuffle_options` (default on): random option order in every training forward;
   - `rotations`: at inference, average the log-probabilities over K cyclic rotations, in one
     batch. K > 1 changes predictions, so `load_predictor(rotations=K)` appends `+rotK` to the
     predictor id and a calibration does not carry over;
   - `position_hash`: each option segment starts at a position derived from a stable hash of
     its own text, so positions do not depend on the presented order but options stay
     distinguishable (R033: order dependence 67% → 27% of questions on a quick run).
4. **Not valid:** `position_reset` (all option segments at the same positions) is kept as a
   configuration value but **does not learn**: markers cannot find their own option's tokens
   (R032, cosine 0.9999 between markers). Do not use it.
5. **Training.** `TrainingConfig.mix_alpha` and `epoch_examples` draw an epoch from several
   sources with shares proportional to `n_s ** alpha` (0 = equal, 1 = proportional).
   Unset, behaviour is the old shuffled pass.
6. **Checks added to the instrument.** `scripts/order_checks.py` (identical options;
   all-rotation and random-order permutations), and selective metrics in the analysis
   package (`selective_precisions`; R034).

**Alternatives considered.**
- *Cross-encoder head on `[CLS]` per candidate:* same cost as joint; rejected.
- *Block attention masks* (each option attends to the question, the context and itself;
  options mix only in the head): exactly order-equivariant, but needs a custom attention mask
  through ModernBERT's alternating local/global layers. Not implemented; the next step if
  hashed positions are not enough.
- *Segment embeddings:* needs new embedding weights; not tried.

**Compatibility.** New config fields have defaults. Existing configs and bundles are
unchanged. Sequence models support the direct decision only (no workspace, expansion or
branch yet): the tree is to be placed inside the sequence later (PLAN_V3 stage C).

**Evidence so far.** R029 (3.0× faster training than joint at 512 vs 384 tokens, positions
honoured by `sdpa`), R031 (+2.6 points over joint on the matched quick run), R032 and R033
(order dependence), R034 (selective metrics).

**Revisit when** full-scale runs show the order-change rate at or above 2% with hashed
positions plus rotation averaging, or when the tree needs node markers inside the sequence.
