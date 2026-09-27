# Suite reasoning_v1

**Question.** Trained on Horn-rule derivations of depth 1–3, how well and how reliably
does a predictor answer at held-out depths 4–8? This is the depth-extrapolation gap
found in R008 (decision 0007).

**Data.** `configs/data/microworld_v1.toml`: micro-world generator revision 2.
- 1,600 worlds with 5 examples each, split by world:
  - train 4,800 examples, depths 1–3;
  - dev 800 and calibration 800, depths 1–3;
  - test 1,600, depths 1–8, 181–236 examples per depth.
- A 400-word nonce vocabulary. No predicate is an English word or a whole-word
  ModernBERT token.
- **Length control.** Every example has exactly 46 sentences. Padding consists of
  facts and dead-end chains about non-candidate entities, so input length does not
  track depth and candidate mention counts stay equal.
- Joint inputs take up to 609 ModernBERT tokens (median 481), so use the 640-token
  model config.

**Validity** (RESULT_LOG R010): the symbolic oracle scores 1.000 at every depth; the
lexical-overlap fixture scores 0.250 at every depth.

**Main endpoint.** Accuracy and NLL per held-out depth (4–8), raw and calibrated, and,
once Stage 3a exists, their change with the reasoning budget. Calibration is fitted on
the calibration split, which contains depths 1–3 only; R008 showed that this can hurt
held-out depths, so report both.

**Reference baseline.** The warm-started direct scorer
(`configs/training/mbft640_joint_reasoning_v1_warm.toml`, the R008 recipe).

**Limitations.** Unary predicates only: no role binding (H2), no negation, three
template families shared between train and test. The budget grid for the reasoning
operations will come in a new revision of this suite.
