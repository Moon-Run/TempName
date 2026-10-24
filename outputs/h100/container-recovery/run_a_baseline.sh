#!/usr/bin/env bash
set -euo pipefail
source /workspace/TempName/outputs/h100/container-recovery/env.sh
cd /workspace/TempName
exec /workspace/conda-envs/flux-h100-ablation3-20261017/bin/python /workspace/TempName/tools/h100/phase6/detach_a_baseline.py --build-cuda-home /usr/local/cuda-12.8 --cuda-home /usr/local/cuda-12.4 --megatron /workspace/Megatron-LM "$@"
