# 0011 — EXPAND v1: candidates expand into retrieved concepts; evidence aggregates back

**Goal.** The first slice of the core idea (decision 0010). Each original candidate
proposes related concepts from the concept memory. The model reads each one in context,
scores its relevance, and aggregates candidate-conditioned evidence from *all* retained
concepts (across branches) back into each candidate's score. The budget is how many
concepts each candidate may expand into.

**Mechanism** (`[expansion]` in a model config; foundations §5.2 and §5.5).
1. **Direct decision.** The Stage-1 scorer gives logits s⁰ (the R005 reader).
2. **Proposals.** For each candidate i, the query "question + candidate" goes through the
   memory's frozen retrieval encoder, and exact search returns the top k concepts. With
   `random_concepts = true` (the control), k concepts are drawn at random instead,
   deterministically from the question and candidate.
3. **Contextual reading.** The reader encodes `concept: {name: gloss}` together with the
   rendered input, giving a vector z_ij for each proposal. The relevance score is
   u_ij = g_exp(h, a_i, z_ij). It weights the proposal's evidence; it is not a
   probability that the concept is true.
4. **Aggregation.** Candidate i attends over every proposal of every candidate:
   β_i,v ∝ exp(q(a_i)·k(z_v)/√d + u_v + b·[v belongs to i's branch]), and
   r_i = Σ_v β_i,v · V(z_v). The final score is s_i = s⁰_i + g_out(a_i, r_i, a_i ⊙ r_i).
   Evidence found under one candidate can therefore support or weaken another.
5. **Budget.** A request expands only if `max_depth ≥ 1`, `max_queries ≥ n` and
   `max_proposals ≥ n·k'`, where k' = min(k, max_proposals // n) and `max_nodes`
   caps the retained concepts.
   - Budget 0 is the direct scorer exactly.
   - Usage reports queries, proposals, nodes and depth = 1.
6. **Trace.** An `expand` event per candidate (its concepts with relevance), then an
   `aggregate` event (each candidate's most-attended concepts and the evidence it
   added). This is the tree the interface shows.

**Training.**
- `freeze_direct = true` (v1): the encoder and direct scorer, loaded from R005, stay
  frozen; only the expansion parts train.
  - This avoids R012's failure, where a trainable direct path fits the data alone.
  - It is also fast, since no gradient flows through ModernBERT.
- During training and dev evaluation, expansion always runs with the configured k;
  the budget applies at inference.

**Matched comparison** (same data `knowledge_v0` train, schedule, seed and
initialization, R005):

| Model | What it isolates |
|---|---|
| Direct, fully fine-tuned | The practical baseline |
| Frozen R005 plus an equal-size head without concepts | Extra head capacity alone |
| Frozen R005 plus expansion with retrieved concepts | The idea |
| Frozen R005 plus expansion with random concepts | Whether the concept content matters |

`reasoning`-style budget sweeps come from a `knowledge_v0` revision with budgets of 0,
2, 4, 8, 16 and 32 concepts per candidate.

**Not yet.** Recursive expansion (concepts into concepts), CONNECT and COMPOSE. Those
are the next slices (PLAN §10.1, step 4).
