# 0012 — EXPAND v2: the reader reads each candidate's branch; a memory of statements

**Context.**
- EXPAND v1 (decision 0011) was tested in R016: retrieved, random and no concepts give
  the same decisions. Its evidence correlates at 0.997 with the random-concept control.
  Its gain over R005 comes from the trained head, and it is flat from 8 to 128 concepts.
- R017 explains why, and what works instead.
  1. **How a concept reaches the decision.** In v1, a concept reaches the decision only
     as the frozen reader's pooled vector of "concept + input". That vector is
     dominated by the input: the cosine between different concepts on one input is
     0.93. The small head cannot tell concepts apart.
  2. **Reading knowledge in the input works.** The same reader, given one relevant
     fact *in its input next to the candidate*, gains 23 points on OpenBookQA with no
     training (0.388 → 0.618); the knowledge-tuned reader gains 12 (0.614 → 0.736).
  3. **WordNet glosses are weak knowledge here.** Definitions rarely state the
     relation a question needs ("metal conducts heat"). The best gloss condition adds
     2 points.
  4. **A reader trained on empty contexts is hurt by any added text.** Expansion must
     therefore be present during training.
- The idea is unchanged (decision 0010): candidates expand, the expansions are read in
  relation to the candidate, and their evidence chains back into each candidate's
  probability. What changes is *how* nodes are read, and *what* a node holds.

**Choice.**
1. **Branch reading.** Each candidate's expansion (its branch) is rendered into the
   reader's input together with the candidate and the question:
   ```
   answer: {candidate}
   question: … context: …
   knowledge: [1] {node} [2] {node} …
   ```
   - The reader is **trainable** (warm start from R005), so it learns to judge nodes
     against the candidate.
   - The existing candidate-conditioned scorer compares branches across candidates,
     so evidence found for one candidate can weaken another.
   - Budget 0 means an empty branch: exactly the direct model's input.
2. **Nodes hold statements about concepts.** A new memory `generics_v0` holds short
   generic statements, each keyed by the concept it is about ("metals conduct heat",
   term "metal"). A node is therefore (concept, statement). Retrieval is the same
   frozen sentence encoder with exact search, from "question + candidate". WordNet
   stays available as `wordnet_v0`.
3. **Training with variable width.** The number of nodes per candidate is sampled per
   batch from {0, 1, 2, 4, 8}. The reader then learns both to answer without expansion
   and to use whatever expansion the budget buys. This is lesson 13 applied without
   freezing the reader: expansion is needed on some training batches, and absent on
   others.
4. **Controls, matched.** All are trained identically, from R005, on `knowledge_v0`:
   - the same model with random statements (content control);
   - the same model with empty branches (the direct control, trained on the same
     schedule).

   The budget grid is 0, 1, 2, 4, 8 nodes per candidate (`knowledge_v0` revision 3).
5. **Gates before training.**
   - **(a) Access recall.** Recall of OpenBookQA core-fact words for `generics_v0`,
     measured as in R015.
   - **(b) Reading probe.** The frozen R005 reader with retrieved statements as its
     input (R017's protocol) must beat random statements and the empty input. It then
     shows the memory holds knowledge the reader can use before any training.
6. **Trace.** For each candidate, one `expand` event lists its nodes (concept,
   statement, retrieval similarity). An `aggregate` event gives each node's
   contribution, measured by leaving the node out at inference (trace mode only). This
   is the tree in the interface, with real per-node evidence.

**Not yet (next slices).**
- **Recursive expansion:** a node's concept becomes a new query, giving depth-2 nodes
  in the same branch (QASC-style two-hop chains).
- **Stop when certain enough:** expand a branch deeper only while the decision is
  uncertain.
- CONNECT and COMPOSE.

**Revisit when:**
- gate (b) fails, which means the memory lacks usable knowledge. Try the ARC corpus
  or another fact source.
- or the trained model's budget curve is flat. Then check whether the reader ignores
  the branch, by comparing against the random-statement control and by per-node
  leave-one-out.
