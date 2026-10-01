#!/usr/bin/env bash
# Stage 2, EXPAND v2 on knowledge_v0 (decision 0012): three matched runs, each trained,
# selected on dev and collected on the revision-3 budget grid. Run from metanoia/.
# Safe to re-run: finished steps are skipped; training resumes.
#   1. branch_v2_retrieved_knowledge_v0   branches of retrieved generics_v0 statements
#   2. branch_v2_random_knowledge_v0      random statements (content control)
#   3. branch_v2_none_knowledge_v0        empty branches (direct control, same schedule)
# Selection by dev NLL at p32 (8 nodes per candidate, the widest trained width).
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
  $R collect-checkpoints --bundle "$bundle" --suite suites/knowledge_v0_r3 --dataset $D \
    --splits dev --budgets "$budget" --device cuda --precision fp16
  if [ ! -f "$selection" ]; then
    $A select --dataset $D --split dev --budget "$budget" --metric nll --out "$selection" \
      $(for r in artifacts/runs/${name}__dev-${budget}__*; do echo --run "$r"; done)
  fi
  if [ ! -d "artifacts/runs/${name}_knowledge_v0_r3" ]; then
    $R collect --config "configs/collection/${name}_knowledge_v0_r3.toml" --from-selection "$selection"
  fi
}

run branch_v2_retrieved_knowledge_v0 p32
run branch_v2_random_knowledge_v0 p32
run branch_v2_none_knowledge_v0 p32
echo "stage 2 EXPAND v2 runs complete"
