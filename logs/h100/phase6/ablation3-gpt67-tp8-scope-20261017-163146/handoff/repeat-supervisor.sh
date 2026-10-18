#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/TempName
SESSION=ablation3-gpt67-tp8-scope-20261017-163146
OUT="$REPO/outputs/h100/phase6/$SESSION"
LOG="$REPO/logs/h100/phase6/$SESSION"
PY=/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python
CURRENT_CONTROLLER_PID=315448

export PYTHONNOUSERSITE=1
export DEBUG=0
export MAKEFLAGS=
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS=1

echo "supervisor_started=$(date -Is)"
echo "waiting_for_current_controller=$CURRENT_CONTROLLER_PID"
while kill -0 "$CURRENT_CONTROLLER_PID" 2>/dev/null; do
  sleep 30
done
echo "current_controller_exited=$(date -Is)"

"$PY" tools/h100/phase6/ablation3.py report "$LOG/compare" \
  > "$LOG/handoff/repeat0-report.log" 2>&1

python - "$LOG/compare/report.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p.get('timing_windows') == 81, p.get('timing_windows')
assert p.get('round_count') == 1, p.get('round_count')
PY

for repeat in 1 2 3 4 5 6 7; do
  compare="$LOG/compare-repeat$repeat"
  echo "repeat${repeat}_started=$(date -Is)"
  "$PY" tools/h100/phase6/ablation3.py prepare "$compare" \
    --build "$OUT/candidates" --megatron /workspace/Megatron-LM \
    --blocks 9 --rounds 1 \
    > "$LOG/handoff/repeat${repeat}-prepare.log" 2>&1
  "$PY" tools/h100/phase6/ablation3.py run "$compare" \
    > "$LOG/handoff/repeat${repeat}-dry-run.log" 2>&1
  "$PY" tools/h100/phase6/ablation3.py run "$compare" \
    --stage all --python "$PY" --cuda-home /usr/local/cuda-12.4 --execute \
    > "$LOG/handoff/repeat${repeat}-controller.log" 2>&1
  "$PY" tools/h100/phase6/ablation3.py report "$compare" \
    > "$LOG/handoff/repeat${repeat}-report.log" 2>&1
  python - "$compare/report.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p.get('timing_windows') == 81, p.get('timing_windows')
assert p.get('round_count') == 1, p.get('round_count')
PY
  echo "repeat${repeat}_completed=$(date -Is)"
done

echo "supervisor_completed=$(date -Is)"
