source /workspace/TempName/outputs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/env.sh
export H100_SESSION=ablation-gpt67-tp8-20261013-233551
export H100_OUT="$REPO/outputs/h100/phase6/$H100_SESSION"
export H100_LOG="$REPO/logs/h100/phase6/$H100_SESSION"
export H100_FREEZE="$REPO/outputs/h100/phase6/frozen-candidate-v1-20261013"
export H100_MEGATRON="$MEGATRON_DIR"
export H100_CUDA_BUILD="$CUDA128"
export H100_CUDA_RUN="$CUDA124"
