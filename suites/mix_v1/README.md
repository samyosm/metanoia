# Suite mix_v1

**Question.** How accurately, and how reliably in probability, does a direct predictor
choose the correct option across a broad mix of multiple-choice sources, with and without a
passage? (PLAN_V3 stage A.)

**Data.** `configs/data/mix_v1.toml` → `metanoia_research.data.mcqa`, pinned Hugging Face
revisions. Sources and their role:

| source | domain | passage | options |
|---|---|---|---|
| RACE (`all`) | general reading | yes | 4 |
| CaseHOLD (LexGLUE) | law | yes (citing text) | 5 |
| MedMCQA | medicine | no | 4 |
| SciQ | science | partly | 4 |
| ARC-Easy, ARC-Challenge, OpenBookQA, QASC | science | no | 3–8 |
| CommonsenseQA | commonsense | no | 5 |
| BoolQ | yes/no reading | yes | 2 (`yes`, `no`, seeded order) |

Splits: a source's train split → `train`, except for sources whose test labels are hidden
(MedMCQA, QASC, CommonsenseQA, BoolQ): their validation split is `test`, and `dev` and
`calibration` are carved from train by group (1% of groups each for MedMCQA, 6% for the
others). For the other sources, validation (split by group) gives `dev` and `calibration`,
and the source's test gives `test`. RACE and SciQ use the same seed and rules as `mcqa_v0`,
so their partitions match it.

**Labels.** Gold option IDs live only in `labels.parquet`; annotation `source` is the slice.
Gold explanations (MedMCQA `exp`) and gold facts (QASC `fact1`, `fact2`; OBQA `fact1`) are
kept in `proofs.jsonl` and never enter a request: they enable with-passage ablations.

**Main endpoint.** Per-source test accuracy and NLL, calibrated with a temperature fitted on
the calibration split only; and the macro average over sources.

**Limitations.**
- BoolQ passages overlap between train and test (623 of its test groups also appear in
  train); its questions differ. Always-"yes" scores about 0.62.
- CaseHOLD: 5.6% of test questions have an option set identical to a train question
  (`scripts/audit_dataset.py`); RACE 1.4%, OBQA 2.8%.
- MedMCQA has no passage; its questions are medical knowledge.
- Public benchmarks may have been seen by pretrained encoders.
- Not a long-document suite: QuALITY is held out of the mix (its articles need thousands of
  tokens) and belongs to the document stage.
