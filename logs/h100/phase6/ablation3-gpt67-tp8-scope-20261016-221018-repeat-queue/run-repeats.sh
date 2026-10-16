#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/TempName
PY=/opt/conda/envs/flux-h100/bin/python
PARENT=ablation3-gpt67-tp8-scope-20261016-221018-release1
QUEUE="$REPO/logs/h100/phase6/ablation3-gpt67-tp8-scope-20261016-221018-repeat-queue"
EXPECTED_UUIDS=(
  bc561ab8-87a1-aaa1-6b5d-f379fc7571b7
  bd6c7a45-7290-f263-a277-e3afefe2484e
  571bb6d9-c967-403e-3be3-38113e2cc7b3
  e69c65ae-0527-2a67-1b3b-3b20f31465a2
  85309d21-1193-de40-5836-af3211e41112
  9df7bad7-654c-d1af-e88c-ded6cc7467fd
  ad6311ce-0286-7a7d-241a-a4913459728c
  98d8b4b3-eab2-35a9-2aca-138b9c44766e
)

cd "$REPO"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export DEBUG=0 MAKEFLAGS= CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11
export GIT_CONFIG_COUNT=3
export GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/workspace/TempName/3rdparty/cutlass
export GIT_CONFIG_KEY_1=safe.directory GIT_CONFIG_VALUE_1=/workspace/TempName/3rdparty/nccl
export GIT_CONFIG_KEY_2=safe.directory GIT_CONFIG_VALUE_2=/workspace/Megatron-LM

set_state() {
  "$PY" - "$QUEUE/queue-state.json" "$1" "$2" <<'PY'
import json, sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
path, status, detail = sys.argv[1:]
Path(path).write_text(json.dumps({
    'status': status,
    'detail': detail,
    'updated_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
}, ensure_ascii=False, indent=2) + '\n')
PY
}

set_state waiting-for-parent "$PARENT"
PARENT_STATE="$REPO/logs/h100/phase6/$PARENT/compare/all-state.json"
while [[ ! -f "$PARENT_STATE" ]]; do sleep 30; done
while true; do
  state=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "missing"))' "$PARENT_STATE")
  case "$state" in
    completed) break ;;
    failed)
      set_state failed "parent run failed; repeats not started"
      exit 1
      ;;
    *) sleep 30 ;;
  esac
done

"$PY" tools/h100/phase6/ablation3.py report "$REPO/logs/h100/phase6/$PARENT/compare"
"$PY" - "$REPO/logs/h100/phase6/$PARENT/compare/report.json" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
assert report.get('completed') is True and report.get('timing_windows') == 162
PY

for n in 1 2 3; do
  session="ablation3-gpt67-tp8-scope-20261016-221018-repeat$n"
  log="$REPO/logs/h100/phase6/$session"
  compare="$log/compare"
  checks="$QUEUE/repeat$n-gpu-checks.jsonl"
  consecutive_idle=0
  set_state waiting-for-gpu "$session"

  while (( consecutive_idle < 3 )); do
    gpu_csv=$(nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits)
    apps_csv=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
    is_match=$(EXPECTED_UUIDS="${EXPECTED_UUIDS[*]}" GPU_CSV="$gpu_csv" APPS_CSV="$apps_csv" "$PY" - <<'PY'
import csv, os
expected=os.environ['EXPECTED_UUIDS'].split()
rows=list(csv.reader(os.environ['GPU_CSV'].splitlines()))
apps=os.environ['APPS_CSV'].strip()
try:
    uuids=[r[1].strip().removeprefix('GPU-') for r in rows]
    idle=(len(rows)==8 and uuids==expected and not apps and
          all('H100' in r[2] and int(r[3])<=1024 and int(r[4])<=5 for r in rows))
except (IndexError, ValueError):
    idle=False
print('1' if idle else '0')
PY
)
    GPU_CSV="$gpu_csv" APPS_CSV="$apps_csv" IDLE="$is_match" SESSION="$session" "$PY" - "$checks" <<'PY'
import json, os, sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
Path(sys.argv[1]).open('a').write(json.dumps({
    'session':os.environ['SESSION'],
    'checked_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'passed':os.environ['IDLE']=='1',
    'gpu_csv':os.environ['GPU_CSV'],
    'apps_csv':os.environ['APPS_CSV'],
},ensure_ascii=False)+'\n')
PY
    if [[ "$is_match" == 1 ]]; then
      consecutive_idle=$((consecutive_idle+1))
      if (( consecutive_idle < 3 )); then sleep 10; fi
    else
      consecutive_idle=0
      sleep 600
    fi
  done

  set_state running "$session"
  if ! "$PY" tools/h100/phase6/ablation3.py run "$compare" \
      --stage all --python "$PY" --cuda-home /usr/local/cuda-12.4 --execute; then
    set_state failed "$session execution failed; remaining repeats stopped"
    exit 1
  fi
  if ! "$PY" tools/h100/phase6/ablation3.py report "$compare"; then
    set_state failed "$session strict report failed; remaining repeats stopped"
    exit 1
  fi
  set_state completed "$session"
done

set_state completed all-three-repeats
