#!/usr/bin/env bash
set -Eeuo pipefail

SESSION_ROOT=/workspace/TempName/outputs/h100/phase6/gpt67-tp8-20261010-112249
source "$SESSION_ROOT/env.sh"
cd "$REPO"

PYTHON=/opt/conda/envs/flux-h100/bin/python
BASE_BUILD="$H100_BUILD/base"
NCCL_ROOT="$REPO/outputs/h100/phase6/tp8-20261009-deployment/nccl-release"
STATE="$H100_BUILD/watchdog-state.json"
LOCK="$H100_BUILD/watchdog.lock"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-600}"
IDLE_MEMORY_MIB="${IDLE_MEMORY_MIB:-1024}"
IDLE_UTIL_PERCENT="${IDLE_UTIL_PERCENT:-5}"
IDLE_CONFIRMATIONS="${IDLE_CONFIRMATIONS:-3}"
IDLE_CONFIRM_INTERVAL_SECONDS="${IDLE_CONFIRM_INTERVAL_SECONDS:-10}"

RUN_TAG=""
CALIBRATION=""
PLAN=""
CANDIDATES=""
COMPARE=""
CURRENT_STAGE=starting

log() {
  printf '[%s] %s\n' "$(TZ=Asia/Shanghai date '+%F %T %Z')" "$*"
}

write_state() {
  local status="$1" stage="$2" detail="$3"
  WATCHDOG_STATUS="$status" WATCHDOG_STAGE="$stage" WATCHDOG_DETAIL="$detail" \
  WATCHDOG_RUN_TAG="$RUN_TAG" WATCHDOG_CALIBRATION="$CALIBRATION" \
  WATCHDOG_PLAN="$PLAN" WATCHDOG_CANDIDATES="$CANDIDATES" \
  WATCHDOG_COMPARE="$COMPARE" WATCHDOG_STATE="$STATE" "$PYTHON" - <<'PY'
import datetime
import json
import os
from pathlib import Path

path = Path(os.environ["WATCHDOG_STATE"])
zone = datetime.timezone(datetime.timedelta(hours=8))
data = {
    "status": os.environ["WATCHDOG_STATUS"],
    "stage": os.environ["WATCHDOG_STAGE"],
    "detail": os.environ["WATCHDOG_DETAIL"],
    "updated_beijing": datetime.datetime.now(zone).isoformat(),
    "pid": os.getppid(),
    "run_tag": os.environ["WATCHDOG_RUN_TAG"],
    "calibration": os.environ["WATCHDOG_CALIBRATION"],
    "plan": os.environ["WATCHDOG_PLAN"],
    "candidates": os.environ["WATCHDOG_CANDIDATES"],
    "compare": os.environ["WATCHDOG_COMPARE"],
}
temp = path.with_name(path.name + ".tmp")
temp.write_text(json.dumps(data, indent=2) + "\n")
temp.replace(path)
PY
}

on_error() {
  local rc=$?
  write_state failed "$CURRENT_STAGE" "pipeline exited with status $rc"
  log "FAILED stage=$CURRENT_STAGE status=$rc"
  exit "$rc"
}

on_signal() {
  write_state stopped "$CURRENT_STAGE" "watchdog received a stop signal"
  log "STOPPED stage=$CURRENT_STAGE"
  exit 130
}

trap on_error ERR
trap on_signal INT TERM

exec 9>"$LOCK"
if ! flock -n 9; then
  log "Another watchdog already owns $LOCK"
  exit 0
fi

gpu_idle_once() {
  local rows row mem util
  mapfile -t rows < <(nvidia-smi \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits)
  [[ "${#rows[@]}" -eq 8 ]] || return 1
  for row in "${rows[@]}"; do
    IFS=',' read -r mem util <<<"$row"
    mem="${mem//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    [[ "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || return 1
    (( mem <= IDLE_MEMORY_MIB && util <= IDLE_UTIL_PERCENT )) || return 1
  done
}

gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr '\n' ';'
}

wait_for_idle() {
  local reason="$1" confirmation stable
  CURRENT_STAGE=waiting-for-gpu
  write_state waiting "$CURRENT_STAGE" "$reason"
  while true; do
    if gpu_idle_once; then
      stable=1
      log "GPU idle confirmation 1/$IDLE_CONFIRMATIONS for $reason"
      for ((confirmation=2; confirmation<=IDLE_CONFIRMATIONS; confirmation++)); do
        sleep "$IDLE_CONFIRM_INTERVAL_SECONDS"
        if gpu_idle_once; then
          log "GPU idle confirmation $confirmation/$IDLE_CONFIRMATIONS for $reason"
        else
          stable=0
          log "GPU became busy during confirmation: $(gpu_snapshot)"
          break
        fi
      done
      if (( stable )); then
        log "Eight GPUs are stably idle for $reason"
        return 0
      fi
    else
      log "Waiting for eight idle GPUs: $(gpu_snapshot)"
    fi
    write_state waiting "$CURRENT_STAGE" "$reason; next check in ${CHECK_INTERVAL_SECONDS}s"
    sleep "$CHECK_INTERVAL_SECONDS"
  done
}

CURRENT_STAGE=validating-inputs
write_state running "$CURRENT_STAGE" "checking frozen 6.7B baseline and environment"
"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("/workspace/TempName")
manifest = json.loads((root / "outputs/h100/phase6/gpt67-tp8-20261010-112249/base/manifest.json").read_text())
config = json.loads((root / "tools/h100/phase6/config.json").read_text())
assert manifest["status"] == "built"
assert manifest["config"] == config
assert config == {
    "tp": 8, "sm_count": 132, "layers": 32, "hidden": 4096,
    "ffn": 16384, "heads": 32, "sequence": 2048,
    "micro_batch": 1, "global_batch": 4, "seed": 1234,
    "token_seed": 4321, "vocab": 32768, "warmup_steps": 10,
    "timed_steps": 20, "window": 64, "fraction": 0.015625,
    "bases": ["remote_first", "interleaved"], "target_percent": 4.0,
}
PY
test -f "$NCCL_ROOT/include/nccl.h"
test -f "$NCCL_ROOT/lib/libnccl_static.a"
test "$(git -C "$MEGATRON_DIR" rev-parse HEAD)" = 3ea68ad6042cc1204386ae9364358f7c4de1bc37
test -z "$(git -C "$MEGATRON_DIR" status --porcelain --untracked-files=no)"

wait_for_idle "6.7B calibration"

RUN_TAG="$(TZ=Asia/Shanghai date '+%Y%m%d-%H%M%S')"
CALIBRATION="$H100_LOG/calibration-$RUN_TAG"
PLAN="$H100_BUILD/selection-plan-$RUN_TAG.json"
CANDIDATES="$H100_BUILD/candidates-$RUN_TAG"
COMPARE="$H100_LOG/compare-$RUN_TAG"
write_state running calibration "starting fresh H100 calibration"

CURRENT_STAGE=calibration
log "Preparing calibration at $CALIBRATION"
"$PYTHON" tools/h100/phase6/calibrate.py prepare "$CALIBRATION" \
  --build "$BASE_BUILD"
log "Running calibration"
"$PYTHON" tools/h100/phase6/calibrate.py run "$CALIBRATION" \
  --cuda-home "$CUDA124" --execute

CURRENT_STAGE=selection-plan
write_state running "$CURRENT_STAGE" "fitting arrival order and selective quantization mask"
log "Generating selection plan $PLAN"
"$PYTHON" tools/h100/phase6/plan.py "$CALIBRATION" "$PLAN"

CURRENT_STAGE=candidate-build
write_state running "$CURRENT_STAGE" "building two plan-specific candidate libraries"
log "Preparing candidate build $CANDIDATES"
"$PYTHON" tools/h100/phase6/build.py prepare "$CANDIDATES" \
  --config "$H100_CONFIG" --plan "$PLAN"
log "Compiling candidate libraries"
"$PYTHON" tools/h100/phase6/build.py compile "$CANDIDATES" \
  --cuda-home "$CUDA128" --jobs 2 --nccl-root "$NCCL_ROOT"

CURRENT_STAGE=prepare-comparison
write_state running "$CURRENT_STAGE" "freezing six-policy comparison inputs"
log "Preparing comparison $COMPARE"
"$PYTHON" tools/h100/phase6/prepare.py "$COMPARE" \
  --base-build "$BASE_BUILD" --candidate-build "$CANDIDATES" \
  --plan "$PLAN" --megatron "$MEGATRON_DIR"

wait_for_idle "6.7B smoke/profile preflight and timing"

CURRENT_STAGE=preflight
write_state running "$CURRENT_STAGE" "running six smoke and six profile policies"
log "Running smoke/profile preflight"
"$PYTHON" tools/h100/phase6/run.py "$COMPARE" --stage preflight \
  --python "$PYTHON" --cuda-home "$CUDA124" --execute

CURRENT_STAGE=timing
write_state running "$CURRENT_STAGE" "running 72 full optimizer-step timing windows"
log "Running 72 timing windows"
"$PYTHON" tools/h100/phase6/run.py "$COMPARE" --stage timing \
  --python "$PYTHON" --cuda-home "$CUDA124" --execute

CURRENT_STAGE=report
write_state running "$CURRENT_STAGE" "recomputing paired statistics"
log "Recomputing final report"
"$PYTHON" tools/h100/phase6/report.py "$COMPARE"

CURRENT_STAGE=completed
write_state completed "$CURRENT_STAGE" "calibration, candidates, preflight, timing, and report completed"
log "COMPLETE compare=$COMPARE report=$COMPARE/report.json"
trap - ERR
