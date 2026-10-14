#!/usr/bin/env bash
set -eo pipefail
source /workspace/TempName/outputs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-003756/env.sh
cd "$REPO"
exec python -u "$H100_SCOPE_OUT/resume-scope.py"
