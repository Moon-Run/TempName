#!/usr/bin/env bash
set -euo pipefail
source /workspace/TempName/outputs/h100/phase6/h100-ablation2-scope-20261015-010547-queue/env.sh
cd "$REPO"
exec python -u "$H100_BATCH_OUT/queue.py" --execute
