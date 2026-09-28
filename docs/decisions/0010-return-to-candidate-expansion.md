# 0010 — Return to the core path: candidate expansion over a concept memory

**Context.**
- Decision 0007 reordered the stages. It built a sentence-attention workspace (Stage
  3a) on synthetic logic puzzles, before the concept memory and candidate expansion
  (Stage 2).
- That tested a substitute, not the project's idea.
- The owner confirmed the idea on 2026-09-28:
  1. each candidate expands into related concepts retrieved from a concept memory;
  2. those expand further, connect, and compose;
  3. evidence chains back up into each candidate's probability;
  4. more budget buys wider and deeper expansion.
- The substitute's results (R012, R013) say nothing about that idea.

**Choice.**
1. **Supersede 0007's reordering.** The order is again Stage 2 (concept memory,
   retrieval, EXPAND), then Stage 3 (CONNECT, COMPOSE and AGGREGATE over the expansion
   tree).
2. **A knowledge suite as the testbed.**
   - Science multiple-choice questions without a passage (OpenBookQA and ARC).
   - There, the direct model is knowledge-limited, so retrieved concepts can add
     evidence it does not have.
   - This also applies lesson 13: the extra computation must be *needed* on the
     training data.
3. **A concept memory v0.**
   - WordNet synsets: a name plus a gloss, about 117k concepts, general-purpose, from
     broad ("physics") to concrete ("transistor"), as in the idea document.
   - Retrieval uses a **separate, frozen sentence-embedding encoder**, exact search
     first. This is PLAN §7.1 option 3: the memory keys never go stale when the
     decision model is fine-tuned.
   - Proximity only *proposes* concepts. The decision model scores them in context
     (foundations §5.2).
4. **The first gate is access recall.** Before any expansion training, measure whether
   retrieval from question plus candidate brings up concepts that matter (OpenBookQA's
   gold "core fact" is the reference). Nothing can be rescued from a pool that lacks
   the relevant concepts.
5. **Controls from the start.**
   - The same model with random concepts (equal cost), which tests whether the
     *content* matters.
   - The direct model on the same data and schedule.
   - Inspect the readout magnitude and evidence attention, so a switched-off expansion
     is visible (R012).
6. **The logic-puzzle track is parked, not abandoned.** It keeps `reasoning_v0` and
   `reasoning_v1`, the converged baseline R012, the workspace code (decisions 0008 and
   0009), and the open fork (PLAN §10.2). It becomes relevant again for COMPOSE and
   derivations, and as a controlled depth test once expansion exists.

**Revisit when** access recall is too low with WordNet (then try ConceptNet, or a fact
memory such as the OpenBookQA "open book" corpus, as a second memory type), or when
the knowledge suite shows no headroom over the direct model.
