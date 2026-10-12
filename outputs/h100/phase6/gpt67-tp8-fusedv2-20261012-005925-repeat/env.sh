source /workspace/TempName/outputs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/env.sh

# H100_BUILD deliberately keeps the verified baseline build from the first run.
export H100_SESSION=gpt67-tp8-fusedv2-20261012-005925-repeat
export H100_OUTPUT="$REPO/outputs/h100/phase6/$H100_SESSION"
export H100_LOG="$REPO/logs/h100/phase6/$H100_SESSION"
export H100_REUSED_SESSION=gpt67-tp8-fusedv2-20261011-233427
