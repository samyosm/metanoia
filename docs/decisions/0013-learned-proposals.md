# 0013 — Learn what each candidate expands into

**Context.**
- EXPAND v2 (decision 0012) is the first design where expansion helps (R020).
  - Test accuracy rises from 0.537 to 0.611 as branches grow.
  - It beats random statements (+5.5) and the matched direct model (+2.9).
  - But the gain is all ARC-Easy, and OBQA and ARC-Challenge gain nothing.
- R021 and R022 locate the limit in *proposals*, not reading.
  - **Retrieval follows the question's topic.** At 2 per candidate, half the branches
    never mention their candidate, 55–60% of statements are shared, and the same
    pooled statements work as well as per-candidate branches. A generic statement
    ("Most conduction carries heat") can land under a wrong candidate and raise it.
  - **The reader can use the right fact.** With the gold fact as its branch, the
    trained model reaches 0.674 on OBQA, against 0.608 with its retrieval.
  - **Anchoring on the candidate's concept alone** is specific but irrelevant: accuracy
    −5, and gold recall falls.
  - **Reordering the "question + candidate" pool** so that statements mentioning the
    candidate come first ("focused") is specific *and* keeps relevance: 85% mention
    their candidate, overlap 0.11, and OBQA gold-fact recall rises (0.28 against
    0.25). Accuracy stays within noise, with a reader trained on the old branches.
- The owner's idea says each candidate *generates* what it expands into. A fixed
  similarity search is the crudest version of that generation.

**Choice.**
1. **Proposal pool per candidate** from several routes:
   - "question + candidate" retrieval, the top 64 as a pool;
   - statements about concepts named in the candidate (anchored);
   - a second hop from concepts those statements mention (the start of recursion).
2. **Focused ordering as the non-learned baseline.** Pool statements that mention the
   candidate come first. This is `proposals = "focused"` in the branch config: a new
   model option, with the old behavior unchanged.
3. **A learned proposal scorer.**
   - **Labels:** on train questions, label each pooled statement by its leave-one-out
     effect on the trained reader's margin for the gold answer: how much the statement
     helps the right candidate against the others.
   - **Scorer:** trained on these labels. It is small and reads (question, candidate,
     statement): a cross-encoder head on the frozen retrieval encoder, or a fine-tuned
     small encoder.
   - **Reader:** retrained on branches chosen by the scorer, with the matched controls
     (focused, random, empty).
   - This is how a candidate learns what to expand into, without changing the frozen
     memory keys.
4. **Evaluation.**
   - The same `knowledge_v0` revision-3 grid.
   - Per-source results, since OBQA and ARC-Challenge are the targets.
   - Branch statistics from R021 and R022: mention rate, overlap and gold-fact recall.

**Order.** Focused ordering first, because it is cheap and needs one reader retraining
(about 20 min). The learned scorer comes next, if focused ordering leaves OBQA and
ARC-Challenge flat. Recursive expansion (PLAN §10.1, step 4) uses the scorer to choose
each next hop.
