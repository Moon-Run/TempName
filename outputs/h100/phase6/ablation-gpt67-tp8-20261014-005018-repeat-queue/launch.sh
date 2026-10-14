#!/usr/bin/env bash
set -eo pipefail
source /workspace/TempName/outputs/h100/phase6/ablation-gpt67-tp8-20261013-233551/env.sh
cd "$REPO"
exec python -u "$REPO/outputs/h100/phase6/ablation-gpt67-tp8-20261014-005018-repeat-queue/queue.py"
