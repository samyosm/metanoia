# 0007 — Target depth extrapolation with `reasoning_v1`; build Stage 3a before Stage 2

**Status: the stage reordering is superseded by decision 0010.** The core path is Stage 2
(concept memory and candidate expansion), then Stage 3. The `reasoning_v1` suite and its
findings are kept as a parked track (PLAN §10.2).

**Evidence** (RESULT_LOG R006–R009).
- **Cold start collapses.** A direct scorer fine-tuned from plain ModernBERT on
  `reasoning_v0` collapses to uniform output. Neither 20k examples nor a depth-1-only
  curriculum changes this.
- **Warm start learns.** The same scorer warm-started from the reading-comprehension
  model R005 learns in about 150 steps. It reaches ≈100% on the trained depths 1–3 and
  0.786 at the held-out depth 4, with ECE 0.15.
- **No truncation.** No input exceeds the encoder limit.
- **Depth, not length, drives the drop.** Depth-4 inputs as short as depth-2/3 inputs
  still reach only 0.83.

**Choice.**
1. **`reasoning_v1`, a new suite revision; `reasoning_v0` stays frozen.**
   - Train, dev and calibration use depths 1–3.
   - Test holds out depths 4–8.
   - A larger nonce vocabulary, as a new generator revision.
   - Input length controlled independently of depth, by padding with distractor rules.
   - An `input_length` annotation.
   - The primary endpoint is accuracy and NLL per held-out depth, and the risk–budget
     curve at those depths (H1).
2. **The reference baseline is the R008 recipe**: warm start from R005, then puzzles,
   dev-selected. Every compared model gets the same initialization and curriculum.
3. **Build Stage 3a before Stage 2.** Stage 3a is the workspace with aggregation and
   connections over the context's own statements and candidates.
   - The controlled suites are closed-world, so an external concept memory cannot
     help them and would confound the first test of H1.
   - The Stage-2 memory work waits until a suite needs knowledge that is absent from
     the context.

**Alternatives.**
- **Keep `reasoning_v0` and add depth 5 only.** The current vocabulary allows at most
  depth 5. This is too short a curve, and it does not control length.
- **Move to published suites first** (ProofWriter, CLUTRR). Worth doing later, but
  their depth and length are not independently controlled.
- **Build Stage 2 first, as originally planned.** It does not address the measured
  gap.

**Revisit when** the warm-started direct baseline also extrapolates well on
`reasoning_v1` (then the suite needs harder structure: binary relations, negation,
branching), or when a realistic suite requires external knowledge.
