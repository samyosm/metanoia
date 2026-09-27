# Suite mcqa_v0

**Question.** How accurately, and how reliably in probability, does a direct predictor
choose the correct option of human-written multiple-choice questions with a passage?

**Data.** `configs/data/mcqa_v0.toml` → `metanoia_research.data.mcqa`, pinned Hugging
Face revisions of RACE (`all`) and SciQ. Splits: source train → `train` (99,459);
source validation, split by passage group → `dev` (2,890) and `calibration` (2,990);
source test → `test` (5,922). 105 rows with duplicate option texts are skipped.

**Labels.** Gold option IDs (`a`–`d`) live only in `labels.parquet`; annotations
`source` and `n_candidates` are used for slicing (`analysis/specifications/mcqa_v0.toml`).

**Main endpoint.** Test accuracy and NLL, calibrated with a temperature fitted on the
calibration split only. Chance is 0.25 accuracy / 1.386 NLL.

**Limitations.** Public benchmarks likely seen in the pretraining data of some
encoders; SciQ support passages are sometimes empty; RACE contains cloze-style
questions ("_"). Not a reasoning-depth test: use `reasoning_v0` for that.
