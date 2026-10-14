#!/usr/bin/env bash
set -eo pipefail
source /workspace/TempName/outputs/h100/phase6/ablation2-gpt67-tp8-interleaved-20261014-221952/env.sh
cd "$REPO"
exec 9>/tmp/tempname-h100-phase6-repeat-queue.lock
flock -n 9
exec python -u "$H100_OUT/run-session.py"
