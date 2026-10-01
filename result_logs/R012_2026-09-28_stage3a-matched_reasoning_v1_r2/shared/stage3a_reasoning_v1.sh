#!/usr/bin/env bash
# Stage 3a on reasoning_v1 (decision 0008): three matched runs, each trained, selected on
# dev and collected on the full budget grid. About 8 h on an RTX 2060; run from metanoia/.
#   1. mbft640_joint_reasoning_v1_warm20k   the reference direct baseline (select at b0)
#   2. ws_statements_reasoning_v1_warm20k   the statement workspace (select at r4)
#   3. ws_latent_reasoning_v1_warm20k       the equal-cost latent control (select at r4)
# Safe to re-run: finished steps are skipped; an interrupted training resumes.
set -euo pipefail
R=.venv/bin/metanoia-research
A=.venv/bin/metanoia-analysis
D=artifacts/datasets/microworld_v1
mkdir -p artifacts/selections logs

run() {
  local name=$1 budget=$2
  local bundle=artifacts/predictors/$name selection=artifacts/selections/${name}__dev-${budget}.json
  if [ ! -d "$bundle" ]; then
    local resume=()
    [ -d "$bundle.incomplete" ] && resume=(--resume)
    $R train --config "configs/training/$name.toml" "${resume[@]}" 2>&1 | tee -a "logs/$name.train.log"
  fi
  $R collect-checkpoints --bundle "$bundle" --suite suites/reasoning_v1_r2 --dataset $D \
    --splits dev --budgets "$budget" --device cuda --precision fp16
  if [ ! -f "$selection" ]; then
    $A select --dataset $D --split dev --budget "$budget" --metric nll --out "$selection" \
      $(for r in artifacts/runs/${name}__dev-${budget}__*; do echo --run "$r"; done)
  fi
  if [ ! -d "artifacts/runs/${name}_reasoning_v1_r2" ]; then
    $R collect --config "configs/collection/${name}_reasoning_v1_r2.toml" --from-selection "$selection"
  fi
}

run mbft640_joint_reasoning_v1_warm20k b0
run ws_statements_reasoning_v1_warm20k r4
run ws_latent_reasoning_v1_warm20k r4
echo "stage 3a runs complete"
