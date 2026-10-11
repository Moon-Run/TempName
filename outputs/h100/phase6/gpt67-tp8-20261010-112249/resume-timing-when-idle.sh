#!/usr/bin/env bash
set -Eeuo pipefail

SESSION_ROOT=/workspace/TempName/outputs/h100/phase6/gpt67-tp8-20261010-112249
source "$SESSION_ROOT/env.sh"
cd "$REPO"

PYTHON=/opt/conda/envs/flux-h100/bin/python
COMPARE=/workspace/TempName/logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517
STATE=/workspace/TempName/outputs/h100/phase6/gpt67-tp8-20261010-112249/watchdog-state.json
LOG=/workspace/TempName/outputs/h100/phase6/gpt67-tp8-20261010-112249/resume-timing.log
BASE_BUILD="$H100_BUILD/base"
CANDIDATES="$H100_BUILD/candidates-20261010-153517"
PLAN="$H100_BUILD/selection-plan-20261010-153517.json"

write_state() {
  local status="$1" stage="$2" detail="$3" compare="${4:-$COMPARE}"
  WATCHDOG_STATUS="$status" WATCHDOG_STAGE="$stage" WATCHDOG_DETAIL="$detail" \
  WATCHDOG_COMPARE="$compare" WATCHDOG_STATE="$STATE" "$PYTHON" - <<'PY'
import datetime
import json
import os
from pathlib import Path

zone = datetime.timezone(datetime.timedelta(hours=8))
path = Path(os.environ["WATCHDOG_STATE"])
data = {
    "status": os.environ["WATCHDOG_STATUS"],
    "stage": os.environ["WATCHDOG_STAGE"],
    "detail": os.environ["WATCHDOG_DETAIL"],
    "updated_beijing": datetime.datetime.now(zone).isoformat(),
    "pid": os.getppid(),
    "run_tag": "20261010-153517",
    "calibration": "/workspace/TempName/logs/h100/phase6/gpt67-tp8-20261010-112249/calibration-20261010-153517",
    "plan": "/workspace/TempName/outputs/h100/phase6/gpt67-tp8-20261010-112249/selection-plan-20261010-153517.json",
    "candidates": "/workspace/TempName/outputs/h100/phase6/gpt67-tp8-20261010-112249/candidates-20261010-153517",
    "compare": os.environ["WATCHDOG_COMPARE"],
}
temp = path.with_suffix(path.suffix + ".tmp")
temp.write_text(json.dumps(data, indent=2) + "\n")
temp.replace(path)
PY
}

on_error() {
  local rc=$?
  write_state failed resume "resume/repeat pipeline exited with status $rc"
  log "FAILED status=$rc"
  exit "$rc"
}

trap on_error ERR

log() {
  printf '[%s] %s\n' "$(TZ=Asia/Shanghai date '+%F %T %Z')" "$*" | tee -a "$LOG"
}

idle_once() {
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F',' 'NF == 2 {gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); if ($1 > 1024 || $2 > 5) bad=1; count++} END {exit !(count == 8 && !bad)}'
}

log "Waiting for eight idle GPUs before resuming incomplete timing windows"
write_state waiting waiting-for-gpu "waiting to resume 29 incomplete timing windows"
while true; do
  if idle_once; then
    log "GPU idle confirmation 1/3"
    sleep 10
    idle_once || { log "GPU became busy during confirmation 1"; sleep 600; continue; }
    log "GPU idle confirmation 2/3"
    sleep 10
    idle_once || { log "GPU became busy during confirmation 2"; sleep 600; continue; }
    log "GPU idle confirmation 3/3"
    break
  fi
  log "GPUs are still busy; next check in 600s"
  write_state waiting waiting-for-gpu "waiting to resume 29 incomplete timing windows; next check in 600s"
  sleep 600
done

log "Resuming timing windows in $COMPARE"
write_state running timing-resume "skipping 43 completed windows and running the remaining 29"
"$PYTHON" "$SESSION_ROOT/resume_timing.py" "$COMPARE" \
  --python "$PYTHON" --cuda-home "$CUDA124" >> "$LOG" 2>&1

log "Timing resume finished; regenerating report"
"$PYTHON" tools/h100/phase6/report.py "$COMPARE" >> "$LOG" 2>&1
log "Current experiment completed"
write_state completed first-run-completed "72 timing windows and report completed"

# Run one independent full repeat in a new compare directory after the first
# run. The frozen calibration and compiled libraries are reused; results are
# kept in a separate directory and are never written over the first run.
log "Waiting for eight idle GPUs before independent full repeat"
while true; do
  if idle_once; then
    log "Repeat GPU idle confirmation 1/3"
    sleep 10
    idle_once || { log "GPU became busy during repeat confirmation 1"; sleep 600; continue; }
    log "Repeat GPU idle confirmation 2/3"
    sleep 10
    idle_once || { log "GPU became busy during repeat confirmation 2"; sleep 600; continue; }
    log "Repeat GPU idle confirmation 3/3"
    break
  fi
  log "Repeat GPUs are still busy; next check in 600s"
  sleep 600
done

REPEAT_TAG="$(TZ=Asia/Shanghai date '+%Y%m%d-%H%M%S')-repeat"
REPEAT_COMPARE="$H100_LOG/compare-$REPEAT_TAG"
log "Preparing independent full repeat at $REPEAT_COMPARE"
write_state running repeat-prepare "preparing independent full repeat" "$REPEAT_COMPARE"
"$PYTHON" tools/h100/phase6/prepare.py "$REPEAT_COMPARE" \
  --base-build "$BASE_BUILD" --candidate-build "$CANDIDATES" \
  --plan "$PLAN" --megatron "$MEGATRON_DIR" >> "$LOG" 2>&1
log "Running repeat smoke/profile preflight"
write_state running repeat-preflight "running repeat smoke/profile preflight" "$REPEAT_COMPARE"
"$PYTHON" tools/h100/phase6/run.py "$REPEAT_COMPARE" --stage preflight --execute \
  --python "$PYTHON" --cuda-home "$CUDA124" >> "$LOG" 2>&1
log "Running repeat 72 timing windows"
write_state running repeat-timing "running repeat 72 timing windows" "$REPEAT_COMPARE"
"$PYTHON" tools/h100/phase6/run.py "$REPEAT_COMPARE" --stage timing --execute \
  --python "$PYTHON" --cuda-home "$CUDA124" >> "$LOG" 2>&1
log "Generating repeat report"
"$PYTHON" tools/h100/phase6/report.py "$REPEAT_COMPARE" >> "$LOG" 2>&1
log "Independent full repeat completed: $REPEAT_COMPARE"
write_state completed repeat-completed "first run and independent full repeat completed" "$REPEAT_COMPARE"
trap - ERR
