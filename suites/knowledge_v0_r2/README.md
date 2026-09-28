# Suite knowledge_v0, revision 2

Same dataset, splits, question and analysis specification as revision 1
(`../knowledge_v0/`), plus a budget grid for EXPAND v1 (decision 0011).

| Budget | max_depth | max_queries | max_proposals = max_nodes | Concepts per candidate (4 / 5 options) |
|---|---|---|---|---|
| b0 | 0 | 0 | 0 | none (direct scorer) |
| p8 | 1 | 8 | 8 | 2 / 1 |
| p16 | 1 | 8 | 16 | 4 / 3 |
| p32 | 1 | 8 | 32 | 8 / 6 |
| p64 | 1 | 8 | 64 | 16 / 12 |
| p128 | 1 | 8 | 128 | 32 / 25 |

The budget is the *total* number of concepts retrieved and read, so questions with more
options get fewer concepts per candidate at the same cost. Expansion models are trained
with 16 concepts per candidate, so p128 tests widths beyond training.
