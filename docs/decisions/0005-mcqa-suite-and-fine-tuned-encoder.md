# 0005 — Realistic suite `mcqa_v0`, fine-tunable encoder, joint candidate encoding

**Observed first.** On `reasoning_v0`, frozen ModernBERT-base and frozen
BERT-base-uncased direct scorers stay at chance on dev/test, like the hashed-token
encoder (0003): train loss falls, dev loss rises from epoch 1, and the temperature fit
reaches its upper bound. A frozen mean-pooled encoder plus one attention readout does
not solve multi-hop nonce-word chaining from 1,800 examples.

**Choice.**
1. `suites/mcqa_v0`: human-written multiple-choice reading comprehension (RACE `all`
   + SciQ, pinned Hugging Face revisions, `metanoia_research.data.mcqa`). Passage-grouped
   dev/calibration from the sources' validation splits; the sources' test splits as test.
   Spec `analysis/specifications/mcqa_v0.toml` slices by `source`.
2. `HFEncoderConfig.frozen = false` fine-tunes the encoder; its weights are then saved
   in the bundle (`external_prefixes = ()`), and the encoder identity records
   `"frozen": false`. Frozen identities are unchanged, so earlier bundles still load.
3. `DirectScorerConfig.candidate_encoding = "joint"` encodes each candidate together
   with the rendered input (`answer: …` first so truncation never drops it). The scorer
   (eq. 7) is unchanged and still permutation-equivariant; cost is N extra encoder passes.
4. Training options, all defaulting to the previous behaviour: separate encoder LR,
   warmup + linear decay, fp16 mixed precision with a grad scaler, micro-batch gradient
   accumulation, step logging, random subsets. Inference `precision = "fp16"` is an
   explicit predictor-spec field recorded in `describe()`.

**Alternatives.** A standard cross-encoder MC head (CLS → linear) would be simpler but
bypasses the shared scorer that later stages extend. Frozen encoders with a larger
scorer were not tried; the frozen results above give little reason to expect a large gain.

**Compatibility.** New config fields have defaults; hashed/frozen bundles load unchanged.
Fine-tuned bundles are large: about 1.7 GB, which is the encoder weights in `weights.pt` plus AdamW state in `train_state.pt`.

**Revisit when** retrieval/workspace stages need a frozen encoder again (memory keys
must share the encoder's geometry; fine-tuning invalidates a memory index), or when a
cheaper candidate encoding is needed for many candidates.
