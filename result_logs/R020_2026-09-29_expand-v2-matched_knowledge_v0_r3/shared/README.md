# Suite knowledge_v0, revision 3

Same dataset, splits, question and analysis specification as revisions 1 and 2
(`../knowledge_v0/`), with a budget grid for EXPAND v2, branch reading (decision 0012).

| Budget | max_depth | max_queries | max_proposals = max_nodes | Nodes per candidate (4 / 5 options) |
|---|---|---|---|---|
| b0 | 0 | 0 | 0 | none (empty branches: the direct input) |
| p4 | 1 | 8 | 4 | 1 / 0 |
| p8 | 1 | 8 | 8 | 2 / 1 |
| p16 | 1 | 8 | 16 | 4 / 3 |
| p32 | 1 | 8 | 32 | 8 / 6 |

The budget is the *total* number of nodes retrieved and read, so questions with more
options get fewer nodes per candidate at the same cost (at p4, five-option questions
get none). Branch models are trained with 0, 1, 2, 4 or 8 nodes per candidate, sampled
per batch.
