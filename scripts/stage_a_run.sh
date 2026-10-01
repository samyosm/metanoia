#!/usr/bin/env bash
# PLAN_V3 stage A: train one model on mix_v1, select a checkpoint on dev, collect the full
# suite, calibrate and analyze (with selective metrics), and run the order checks.
# Usage:  scripts/stage_a_run.sh NAME      (NAME is a configs/training/NAME.toml, e.g.
#         seq_base_mix_v1, seq_hash_base_mix_v1, joint_base_mix_v1, seq_large_mix_v1)
# Run from metanoia/ on a machine set up with `uv sync --extra research --extra analysis
# --extra hf`. One GPU job per machine. Safe to re-run: finished steps are skipped and an
# interrupted training resumes from its latest checkpoint.
set -euo pipefail
name=${1:?usage: stage_a_run.sh NAME}
R=.venv/bin/metanoia-research
A=.venv/bin/metanoia-analysis
D=artifacts/datasets/mix_v1
S=analysis/specifications/mix_v1_selective.toml
bundle=artifacts/predictors/$name
selection=artifacts/selections/${name}__dev.json
run=${name}_mix_v1
mkdir -p artifacts/selections artifacts/calibrations logs

if [ ! -d "$D" ]; then
  $R prepare --config configs/data/mix_v1.toml
  grep -q 4ca9f0dd20ff457bab4bd9fc9dec04caadd9e20909eeff6ccc70b8f72fb8f7a8 "$D/dataset.json" \
    || { echo "mix_v1 does not reproduce the pinned hashes"; exit 1; }
fi
if [ ! -d "$bundle" ]; then
  resume=()
  [ -d "$bundle.incomplete" ] && resume=(--resume)
  $R train --config "configs/training/$name.toml" "${resume[@]}" 2>&1 | tee -a "logs/$name.train.log"
fi
$R collect-checkpoints --bundle "$bundle" --suite suites/mix_v1 --dataset $D \
  --splits dev --device cuda --precision fp16
if [ ! -f "$selection" ]; then
  $A select --dataset $D --split dev --budget b0 --metric nll --out "$selection" \
    $(for r in artifacts/runs/${name}__dev__*; do echo --run "$r"; done)
fi
if [ ! -d "artifacts/runs/$run" ]; then
  cat > "logs/$run.collect.toml" <<TOML
run_id = "$run"
suite = "suites/mix_v1"
dataset = "$D"
output_root = "artifacts/runs"

[predictor]
kind = "bundle"
path = "$bundle"
device = "cuda"
precision = "fp16"
TOML
  $R collect --config "logs/$run.collect.toml" --from-selection "$selection"
fi
cal=artifacts/calibrations/${name}_b0.json
[ -f "$cal" ] || $A calibrate --run "artifacts/runs/$run" --dataset $D --budget b0 --out "$cal"
$A analyze --run "artifacts/runs/$run" --dataset $D --spec $S --calibration "$cal" \
  | tee "logs/$name.analysis.txt"
# Order robustness, for marker models (400 test questions per check, ~3 min).
if grep -q '"candidate_encoding": "sequence"' "$bundle/bundle.json"; then
  .venv/bin/python scripts/order_checks.py "$bundle" $D test 400 | tee "logs/$name.order_checks.json"
fi
echo "stage A run $name complete"
