#!/usr/bin/env bash
# Stage 2, EXPAND v2 with focused proposals on knowledge_v0 (decision 0013): one run,
# matched to the R020 runs (their retrieved, random and empty-branch models are the
# comparisons), trained, selected on dev at p32 and collected on the revision-3 grid.
# About 25-35 min on an RTX 2060. Run from metanoia/. Safe to re-run.
set -euo pipefail
R=.venv/bin/metanoia-research
A=.venv/bin/metanoia-analysis
D=artifacts/datasets/knowledge_v0
name=branch_v2_focused_knowledge_v0
budget=p32
bundle=artifacts/predictors/$name
selection=artifacts/selections/${name}__dev-${budget}.json
mkdir -p artifacts/selections logs

if [ ! -d "$bundle" ]; then
  resume=()
  [ -d "$bundle.incomplete" ] && resume=(--resume)
  $R train --config "configs/training/$name.toml" "${resume[@]}" 2>&1 | tee -a "logs/$name.train.log"
fi
$R collect-checkpoints --bundle "$bundle" --suite suites/knowledge_v0_r3 --dataset $D \
  --splits dev --budgets "$budget" --device cuda --precision fp16
if [ ! -f "$selection" ]; then
  $A select --dataset $D --split dev --budget "$budget" --metric nll --out "$selection" \
    $(for r in artifacts/runs/${name}__dev-${budget}__*; do echo --run "$r"; done)
fi
if [ ! -d "artifacts/runs/${name}_knowledge_v0_r3" ]; then
  $R collect --config "configs/collection/${name}_knowledge_v0_r3.toml" --from-selection "$selection"
fi
echo "stage 2 EXPAND v2 focused run complete"
