# 0006 — Local interface over a JSON-lines worker (`metanoia serve`)

**Choice.** A Next.js app in `../interface` (outside the Python source) talks to the
core package only through `metanoia serve`, a long-lived process that reads
`{"id", "request"}` JSON lines on stdin and writes `{"id", "result" | "error"}` lines.
It uses the public `DecisionRequest` / `DecisionResult` contract (PLAN §5.1). The
trace of the direct predictor now records its actual path: `render` (the rendered
text), `encode` (token count, `max_length`, truncation, candidate encoding),
`direct_score` (logits) and `calibrate` (temperature, probabilities).

**Reason.**
- Loading a fine-tuned bundle takes seconds, so a warm worker is needed for
  interactive use.
- stdin/stdout needs no port, authentication or HTTP framework, and it keeps the core
  free of web dependencies.
- The PLAN defers HTTP until it is needed (§3, Stage 6); this is the smallest boundary
  that works.

**Compatibility.**
- Adds `TextEncoder.token_count` to the encoder protocol. Both existing encoders
  implement it.
- Traces of the direct predictor change from two events to four. Traces are
  diagnostics, never scored, so no suite or analysis revision changes.
- Tracing still does not change predictions (tested).

**Revisit when** several clients or machines need the model (then add a thin HTTP
adapter over the same contract), or when Stage 3 traces need streaming or
graph-shaped views.

**Addendum (2026-09-29): live decisions.**
- **The model side.** `TorchPredictor.decide_stream` yields `DecisionUpdate`s
  (`metanoia_contracts.updates`) for stages that are actually computed, then the
  ordinary `DecisionResult`, which is identical to `decide()`.
  - For branch models (EXPAND v2) the stages are: the rendered input; the direct answer
    (empty branches); each candidate's proposed statements; the answer after reading 1,
    2, 4, … statements per candidate, up to the budget; then each candidate's per-node
    leave-one-out contributions, one candidate at a time.
  - Proposals are computed once and reused by every stage.
  - Other models compute in one pass, so they stream only the rendered input and the
    result. Nothing is replayed to look live.
- **The protocol.** `metanoia serve` accepts `"stream": true` on a request line and
  answers with `{"id", "update"}` lines before the `{"id", "result"}` line.
- **The interface.** `POST /api/decide` returns newline-delimited JSON: `start`,
  `update`… then `result` or `error`. The page shows:
  - a status line;
  - live probability bars;
  - the expansion tree, where statements appear as proposed, turn solid when read, and
    gain signed effects when measured;
  - a chart of how each candidate's probability changed as its branch grew;
  - the full computation path once done.
- **Calibration.** Live probabilities are an uncalibrated softmax. A calibration
  applies only to the final result, at the budget it was fitted for.
