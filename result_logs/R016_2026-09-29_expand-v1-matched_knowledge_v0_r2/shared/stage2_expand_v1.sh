#!/usr/bin/env bash
# Stage 2, EXPAND v1 on knowledge_v0 (decision 0011): four matched runs, each trained,
# selected on dev and collected on the revision-2 budget grid. About 3.5 h on an RTX 2060;
# run from metanoia/. Safe to re-run: finished steps are skipped; training resumes.
#   1. mbft_joint_knowledge_v0            direct, fully fine-tuned (select at b0)
#   2. expand_v1_retrieved_knowledge_v0   frozen R005 + expansion over retrieved concepts
#   3. expand_v1_random_knowledge_v0      the same with random concepts (content control)
#   4. expand_v1_none_knowledge_v0        the same head with no concepts (capacity control)
# Expansion models are selected at p64 (about 16 concepts per candidate, as trained).
set -euo pipefail
R=.venv/bin/metanoia-research
A=.venv/bin/metanoia-analysis
D=artifacts/datasets/knowledge_v0
mkdir -p artifacts/selections logs

run() {
  local name=$1 budget=$2
  local bundle=artifacts/predictors/$name selection=artifacts/selections/${name}__dev-${budget}.json
  if [ ! -d "$bundle" ]; then
    local resume=()
    [ -d "$bundle.incomplete" ] && resume=(--resume)
    $R train --config "configs/training/$name.toml" "${resume[@]}" 2>&1 | tee -a "logs/$name.train.log"
  fi
  $R collect-checkpoints --bundle "$bundle" --suite suites/knowledge_v0_r2 --dataset $D \
    --splits dev --budgets "$budget" --device cuda --precision fp16
  if [ ! -f "$selection" ]; then
    $A select --dataset $D --split dev --budget "$budget" --metric nll --out "$selection" \
      $(for r in artifacts/runs/${name}__dev-${budget}__*; do echo --run "$r"; done)
  fi
  if [ ! -d "artifacts/runs/${name}_knowledge_v0_r2" ]; then
    $R collect --config "configs/collection/${name}_knowledge_v0_r2.toml" --from-selection "$selection"
  fi
}

run mbft_joint_knowledge_v0 b0
run expand_v1_retrieved_knowledge_v0 p64
run expand_v1_random_knowledge_v0 p64
run expand_v1_none_knowledge_v0 p64
echo "stage 2 EXPAND v1 runs complete"
