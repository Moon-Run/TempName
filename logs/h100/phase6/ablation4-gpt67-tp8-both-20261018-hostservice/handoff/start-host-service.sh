#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${1:?Usage: start-host-service.sh <75-container-name> }"
UNIT=h100-ablation4-two-runs-20261018-hostservice
REPO=/workspace/TempName
LOG=/workspace/TempName/logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice
PYTHON=/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python

DOCKER="$(command -v docker)"
"$DOCKER" inspect "$CONTAINER" --format '{{.State.Status}} {{.Name}}'
test "$("$DOCKER" inspect "$CONTAINER" --format '{{.State.Status}}')" = running
test -x "$LOG/run-two.sh"

sudo systemd-run --unit="$UNIT" \
  --property=Type=exec --property=Restart=no \
  --property=RemainAfterExit=yes --property=StandardInput=null \
  --property=StandardOutput=journal --property=StandardError=journal \
  "$DOCKER" exec --workdir /workspace/TempName "$CONTAINER" \
  /bin/bash "$LOG/run-two.sh" "$REPO" "$PYTHON" "$LOG"

sudo systemctl show "$UNIT.service" \
  -p ActiveState -p SubState -p MainPID -p ControlGroup -p Result -p ExecMainStatus
sudo systemctl status "$UNIT.service" --no-pager -l
sudo journalctl -u "$UNIT.service" -n 50 --no-pager
"$DOCKER" top "$CONTAINER" -eo pid,ppid,args
