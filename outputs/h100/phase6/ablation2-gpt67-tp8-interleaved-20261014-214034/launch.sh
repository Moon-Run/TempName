#!/usr/bin/env bash
set -eo pipefail
source /workspace/TempName/outputs/h100/phase6/ablation2-gpt67-tp8-interleaved-20261014-214034/env.sh
cd "$REPO"
exec python -u "$H100_OUT/run-session.py"
