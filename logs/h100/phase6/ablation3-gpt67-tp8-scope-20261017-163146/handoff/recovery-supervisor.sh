#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/TempName
SESSION=ablation3-gpt67-tp8-scope-20261017-163146
OUT="$REPO/outputs/h100/phase6/$SESSION"
LOG="$REPO/logs/h100/phase6/$SESSION"
PY=/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python

export PYTHONNOUSERSITE=1
export DEBUG=0
export MAKEFLAGS=
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS=1

run_compare() {
  local name="$1"
  local compare="$LOG/$name"
  echo "${name}_started=$(date -Is)"
  "$PY" "$REPO/tools/h100/phase6/ablation3.py" prepare "$compare" \
    --build "$OUT/candidates" --megatron /workspace/Megatron-LM \
    --blocks 9 --rounds 1 \
    > "$LOG/handoff/${name}-prepare.log" 2>&1
  "$PY" "$REPO/tools/h100/phase6/ablation3.py" run "$compare" \
    > "$LOG/handoff/${name}-dry-run.log" 2>&1
  "$PY" "$REPO/tools/h100/phase6/ablation3.py" run "$compare" \
    --stage all --python "$PY" --cuda-home /usr/local/cuda-12.4 --execute \
    > "$LOG/handoff/${name}-controller.log" 2>&1
  "$PY" "$REPO/tools/h100/phase6/ablation3.py" report "$compare" \
    > "$LOG/handoff/${name}-report.log" 2>&1
  test -s "$compare/report.json"
  test -s "$compare/report.md"
  "$PY" - "$compare/all-state.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1])).get('status') == 'completed'
PY
  echo "${name}_completed=$(date -Is)"
}

echo "recovery_supervisor_started=$(date -Is)"
echo "preserving_incomplete_compare=$LOG/compare"
run_compare compare-recovery0
for repeat in 1 2 3 4 5 6 7; do
  run_compare "compare-repeat$repeat"
done
echo "recovery_supervisor_completed=$(date -Is)"
