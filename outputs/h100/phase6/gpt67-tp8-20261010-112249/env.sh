source /opt/conda/etc/profile.d/conda.sh
conda activate flux-h100

export PYTHONNOUSERSITE=1
export REPO=/workspace/TempName
export MEGATRON_DIR=/workspace/Megatron-LM
export APEX_SRC=/workspace/apex-flux
export CUDA124=/usr/local/cuda-12.4
export CUDA128=/usr/local/cuda-12.8
export CUDA_HOME="$CUDA124"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export MAX_JOBS=2
export DEBUG=0
export CMAKE_BUILD_PARALLEL_LEVEL=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export H100_SESSION=gpt67-tp8-20261010-112249
export H100_BUILD="$REPO/outputs/h100/phase6/$H100_SESSION"
export H100_LOG="$REPO/logs/h100/phase6/$H100_SESSION"
export H100_CONFIG="$REPO/tools/h100/phase6/config.json"

export GIT_CONFIG_COUNT=4
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$REPO"
export GIT_CONFIG_KEY_1=safe.directory
export GIT_CONFIG_VALUE_1="$REPO/3rdparty/cutlass"
export GIT_CONFIG_KEY_2=safe.directory
export GIT_CONFIG_VALUE_2="$REPO/3rdparty/nccl"
export GIT_CONFIG_KEY_3=safe.directory
export GIT_CONFIG_VALUE_3="$MEGATRON_DIR"
