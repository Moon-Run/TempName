#!/usr/bin/env bash
set -eo pipefail
source /workspace/TempName/outputs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/env.sh
case "${1:-}" in
    gpt67-tp8-fusedv2-20261012-205424-repeat1|gpt67-tp8-fusedv2-20261012-205424-repeat2) ;;
    *) exit 2 ;;
esac
export H100_SESSION="$1"
export H100_LOG="$REPO/logs/h100/phase6/$H100_SESSION"
export H100_QUEUE_OUTPUT="$REPO/outputs/h100/phase6/gpt67-tp8-fusedv2-20261012-205424-queue"
mkdir -p "$H100_LOG/handoff"
exec 9>"$H100_LOG/launch.lock"
flock -n 9 || exit 1
test ! -e "$H100_LOG/compare/submission.json"
cd "$REPO"
python -u "$H100_BUILD/record-inputs.py"
python -u "$H100_QUEUE_OUTPUT/verify-inputs.py"
python -u "$H100_BUILD/execute.py"
python -u "$H100_BUILD/verify-results.py"
