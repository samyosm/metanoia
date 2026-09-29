# 0014 — A portfolio of proposal methods, composed and screened together

**Context.**
- EXPAND v2 works when each candidate reads statements next to itself (R020). What
  limits it is *which* statements each candidate expands into (R021, R022).
- The owner's guidance (2026-09-29):
  - **try several methods** rather than force one to work;
  - **compose methods** when possible;
  - **relationship-based retrieval**: link the candidate, the question and the context
    through the concepts statements mention, rather than taking nearest neighbours in
    vector space.
- R023 checked the two main new ingredients before any build.
  - **A concept graph over `generics_v0` is cheap** (19 s to index). But whether a
    path from candidate to question exists barely separates right answers from wrong
    ones: 48% against 43% for one hop. Paths can *propose*; they cannot *judge*.
  - **Off-the-shelf rerankers are no free win.** They are no better than similarity at
    ranking the gold fact first. The larger v2-m3 is better on average, and is
    candidate-sensitive (0.58 against 0.25 by chance).
- Decision 0013's learned scorer remains the principled judge. It needs a pool that
  contains the right statements, and that is what generators must supply.

**Choice: proposals as a pipeline of interchangeable, composable stages.**
Each stage is a selectable option in the branch config, recorded in the bundle, and
visible in the trace.

1. **Generators** (the union of their outputs forms the pool):
   - **G1 similarity:** top-n for "question candidate" (have);
   - **G2 anchored:** statements about concepts named in the candidate (R022 probe);
   - **G3 concept paths** (the owner's method):
     - statements mentioning a candidate concept *and* a question or context concept
       (one hop);
     - statement pairs through a bridge concept (two hops);
     - common concepts down-weighted by inverse frequency, not capped;
   - **G4 follow-up search:** a second search with "question candidate + first
     statement". This is iterative retrieval, the principled form of the naive hop
     that failed in R022;
   - **G5 context nodes:** the request's context sentences join the graph, so paths
     can run through the given passage. This is needed for passage suites (`mcqa_v0`)
     and the parked logic puzzles; it is not used by `knowledge_v0`, whose context is
     empty;
   - **G6 generated bridges** (a later comparison, not a default): a small local
     language model writes the linking statement, which is then grounded by retrieval.
     It is kept separate because it can invent support for any candidate.
2. **Judges** (score each pooled statement for a candidate):
   - **J0 similarity** (have);
   - **J1 focused** (mentions the candidate; have);
   - **J2 reranker**, bge-reranker-v2-m3 with a question/answer query;
   - **J3 the trained reader's leave-one-out effect.** Exact but expensive, so it is
     used as a teacher;
   - **J4 learned scorer**, distilled from J3 (decision 0013);
   - **fusion:** combinations of judges by rank fusion. Judges are composed, not only
     chosen.
3. **Selection:**
   - top-k per candidate;
   - optionally, statements shared by all candidates move into one shared question
     context instead of every branch (R022: topic knowledge helps, but it should not
     be attributed to one candidate);
   - for paths, a branch stops when it reaches the question: the first form of "expand
     until certain enough" (PLAN §10.1, step 4).

**Screening protocol** (fixed before any results).
- **Data:** `knowledge_v0` dev + calibration (1,368). The test split stays untouched
  until a method is promoted.
- **For each generator × judge (× fusion) configuration, at 2 and 8 statements per
  candidate:**
  - OBQA gold-fact novel-word recall;
  - on-candidate rate and branch overlap (as in R022);
  - accuracy and NLL per source, read by *both* trained v2 readers (retrieved and
    focused), since a reader favours the proposal style it was trained on (R022);
  - cost: pool size, and ms per question on GPU and CPU.
- **Promotion to a matched retraining** (as in R020, owner-run, about 30 min each), at
  most two configurations per round. A configuration is promoted if, against the best
  current method at equal statements per candidate, either:
  - (a) OBQA gold-fact recall rises by at least 0.03, and dev accuracy falls by no more
    than 2 points under either reader; or
  - (b) dev accuracy on OBQA + ARC-Challenge rises by at least 2 points under either
    reader.
- **The final claim** comes only from test collection on `knowledge_v0`, against the
  existing matched controls (random, empty, similarity, focused).

**Rounds.**
1. **Screen G1, G3 and G4 with J0, J1, J2 and their fusions** (no training; GPU pieces
   of ≤10 min).
2. **Promote at most two;** the owner runs the matched retraining.
3. **Build J3 labels on train questions** over the union pool G1 ∪ G3 ∪ G4, and
   distil J4. Screen, then promote.
4. **Recursion** (PLAN §10.1, step 4): the best judge chooses each next hop, and a
   branch stops when it connects to the question or the decision is confident.
5. **G6 generated bridges** once, as a comparison family.

**Engineering consequences.**
- **Concept graph as an artifact.** The mention index becomes a versioned artifact
  of the memory: hashed, built by a command, and recorded in the bundle's
  `expansion_identity`.
- **Pinned reranker.** The reranker is a frozen external model with a pinned revision,
  recorded like the retrieval encoder.
- **Budget accounting.** Budgets separate what is *considered* from what is *read*:
  pool size counts against `max_proposals`, statements read against `max_nodes`, and
  hops against `max_depth`.
  - Today `usage.proposals` reports only the statements read, which understates
    focused proposals (a pool of 64). This is fixed with a new suite revision
    (`knowledge_v0` r4) whose budgets set these separately.
- **Trace.** The trace shows each branch as the path it came from (candidate →
  statement → concept → statement → question concept) and which judge chose it.

**Revisit when:**
- no generator raises gold-fact recall over G1, which means the memory is the limit:
  then add another knowledge source, e.g. the ARC corpus or QASC facts;
- or every judge fails to beat J0 under both readers: then go directly to J3 and J4.

**Round 1 outcome** (R025, 2026-09-29).
- **Promoted:** G1 with RRF(J0, J2), i.e. similarity retrieval ordered by rank fusion
  of similarity and the reranker. It adds +2.5 points on OBQA + ARC-Challenge under the
  focused reader.
- **Not promoted:**
  - concept paths (G3): the most candidate-specific branches, but lower accuracy and no
    gain in gold recall. Lexical concept matching confuses senses;
  - follow-up search (G4): adds nothing;
  - the reranker alone: loses gold recall.
- **Implementation:** `proposals = "fused"`, with a pinned `[branch.reranker]` and an
  optional proposal-cache artifact (`build-proposals`), so training need not run the
  reranker.
  - A cache is refused unless its identity (memory, retrieval encoder, reranker, pool,
    fusion constant) matches the model's.
  - Pairs missing from the cache are ranked live by the same code.
  - Against the screening's branches, the implementation gives the same 8 statements
    for 89% of candidates and the same top 2 for 93% (half-precision near-ties).
- **Round 2:** the owner runs `scripts/stage2_expand_v2_fused.sh`.
