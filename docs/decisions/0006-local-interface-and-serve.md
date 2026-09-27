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
