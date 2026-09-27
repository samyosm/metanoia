# Suite reasoning_v1, revision 2

Same dataset, splits, question and analysis specification as revision 1
(`../reasoning_v1/`). The only change is the **budget grid**, for the Stage-3a workspace
(decision 0008):

| Budget | max_rounds | max_nodes | max_pairs |
|---|---|---|---|
| b0 | 0 | 0 | 0 |
| r1, r2, r4, r8, r16 | R | 64 | R × 4096 |

Allowing 4,096 pairs per round covers the statement workspace (about 51² ≈ 2,600) and the
latent control (about 20² + 20 × 47 ≈ 1,340). A predictor that cannot afford a round
reports `budget_exhausted`.

Workspace models are trained with 1–4 rounds, so budgets r8 and r16 test whether
extra computation keeps helping beyond the trained range. A direct predictor answers
identically at every budget. Its run is the flat reference line.
