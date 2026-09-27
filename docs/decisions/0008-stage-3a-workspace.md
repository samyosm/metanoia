# 0008 — Stage 3a: a budgeted workspace over the context's statements

**Goal.** Test H1 on `reasoning_v1`: does one fixed trained model become more accurate
at held-out derivation depths as its inference budget (reasoning rounds) grows? And
does it beat an equal-cost recurrent control? (Foundations §1.3 and §13.4; PLAN §10.)

**Mechanism** (`workspace.mode = "statements"`).
1. **Encode as the direct model does.** The rendered input gives context token states;
   each candidate is jointly encoded into a vector a_i.
2. **Initialize the workspace.**
   - Nodes are the rendered input's sentences (question, facts, rules), each the mean
     of its tokens' states, plus one node per original candidate (a_i).
   - Nodes are typed (statement or candidate) and keep their source span, so traces can
     name the sentence behind a node.
   - Candidates are always retained.
3. **CONNECT and AGGREGATE, repeated for R rounds.** Each round is one pre-norm
   transformer block (multi-head attention over all nodes, then an MLP), with weights
   shared across rounds.
   - The attention weights are the round's question-conditioned edges, since the
     question sentence is a node.
   - The pairs scored are charged to the budget.
4. **Read out.** logits = direct logits + head(candidate node states). The head's last
   layer starts at zero, so a warm-started model begins identical to the direct
   baseline at every budget.
   - With R = 0 the workspace is skipped entirely: budget zero is exactly the direct
     scorer (PLAN §6.3).

**Equal-cost control** (`workspace.mode = "latent"`). The same block and readout, but
the non-candidate nodes are K learned latent slots with no statement structure. Each
round, the slots cross-attend to the raw context tokens. This is the "recurrent latent
processor with comparable parameters and cost" of foundations §13.4. It isolates what
grounding nodes in statements adds.

**Training.**
- Warm start from the direct baseline (`init_from`). Its encoder and scorer weights are
  loaded; the workspace weights start fresh.
- Each batch samples R uniformly from `train_rounds` (for example 1–4).
- Deep supervision averages the loss over every round's readout, so each intermediate
  budget is a trained operating point.
- The same data, schedule and initialization as the reference baseline (lesson 12).

**Budget and accounting.**
- `max_rounds` sets R. `max_nodes` caps the statement nodes (candidates are reserved
  separately). `max_pairs` caps the attention pairs scored over all rounds.
- A round that would exceed a cap is not run. Termination is then `budget_exhausted`
  instead of `completed`.
- Usage reports the rounds, nodes and pairs actually used. Composition is not built
  (Stage 3b), so composites and depth stay 0.

**Trace.** The direct events, then one `aggregate` event per round, giving:
- each candidate's logits after the round;
- each candidate's most-attended statements, as sentence indices and text excerpts.

This is the path the interface shows. It describes computation, not proof.

**Suite.** `reasoning_v1` revision 2 has the same dataset, with a budget grid of 0, 1,
2, 4, 8 and 16 rounds.

**Not claimed.**
- Attention weights are not explanations. Intervention tests (H6) come later.
- The dense block is O(n²) per round. With n ≈ 50 nodes this is small; sparsified
  edges are a later, separate control.

**Revisit when** the statement workspace shows no budget gain over the latent control
(then consider explicit typed edges or composition, Stage 3b), or when inputs outgrow
dense attention.
