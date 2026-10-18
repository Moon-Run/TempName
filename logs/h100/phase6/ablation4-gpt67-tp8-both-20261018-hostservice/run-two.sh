#!/usr/bin/env bash
set -euo pipefail

REPO="$1"
PYTHON="$2"
LOG="$3"
cd "$REPO"
source "$LOG/runtime-env.sh"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0

for TRIAL in compare1 compare2; do
  "$PYTHON" tools/h100/phase6/ablation4.py run "$LOG/$TRIAL" \
    --stage all --python "$PYTHON" --cuda-home /usr/local/cuda-12.4 --execute \
    > "$LOG/${TRIAL}-controller.log" 2>&1
  "$PYTHON" tools/h100/phase6/ablation4.py report "$LOG/$TRIAL" \
    > "$LOG/${TRIAL}-report.log" 2>&1
  "$PYTHON" - "$LOG/$TRIAL" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
report = json.loads((root/'report.json').read_text())
assert report['completed'] and report['round_count'] == 2 and report['timing_windows'] == 50
state = json.loads((root/'all-state.json').read_text())
assert state['status'] == 'completed'
PY
done
