# 新服务器：单机 8 张 H100 部署与运行

更新：2026-10-09。以下命令在**新服务器**执行，使用当前受维护的 `tools/h100/phase6/` 入口。无需 Slurm；如果服务器受调度器管理，GPU 步骤须在已分配的计算节点内执行。

当前完成的是 CPU 检查与 CUDA 编译验证，尚未完成 H100 GPU 正确性和性能验证。2026-10-09 的四卡短测因 Slurm 返回 `AssocMaxSubmitJobLimit` 未能提交，用户随后要求保留现有作业并停止测试，没有产生 GPU 结果。本指南提供部署步骤，不表示新服务器已经验证成功。

## 1. 获取完整源码

先确保远端仓库包含 H100 适配代码，再在新机器 clone。不要只复制 `tools/h100/`：构建器还读取仓库的 `src/`、`include/`、`python/` 和受维护的 `tools/a800/` 辅助实现。

目录可以按下面的布局安排，实际路径可自行修改：

```text
~/data/gyd-h/
├── TempName/
├── Megatron-LM/
└── apex-flux/
```

```bash
mkdir -p "$HOME/data/gyd-h"
cd "$HOME/data/gyd-h"

# 将占位内容替换成你自己的仓库地址；已有 clone 时跳过这条。
git clone --recurse-submodules '<你的仓库地址>' TempName
cd TempName
export REPO="$PWD"
git submodule update --init --recursive
test -f tools/h100/phase6/build.py

export MEGATRON_DIR="$(dirname "$REPO")/Megatron-LM"
# 已有 Megatron-LM 目录时跳过 clone，在干净工作区中核对固定提交。
git clone https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON_DIR"
git -C "$MEGATRON_DIR" checkout --detach \
  3ea68ad6042cc1204386ae9364358f7c4de1bc37
git -C "$MEGATRON_DIR" rev-parse HEAD
git -C "$MEGATRON_DIR" status --short
```

Megatron 必须是上述精确提交，受版本控制的文件保持未修改。`outputs/`、`logs/` 和 Conda 环境不会随 clone 迁移；旧机器编译的 `.so`、诊断产物和 A800 到达计划都不作为新机器的运行输入。

## 2. 配置软件环境

以已经过本机编译检查的软件组合为起点：

| 组件 | 版本 / 要求 |
| --- | --- |
| Python | 3.10.20，独立 Conda 环境 |
| PyTorch | 2.6.0+cu124 |
| Apex | `e13873debc4699d39c6861074b9a3b2a02327f92`，编译 C++/CUDA 扩展 |
| CUDA Toolkit | 12.4 用于 Apex/运行环境；12.8 用于 Flux/TACO 编译 |
| C/C++ 工具 | GCC/G++ 11、Make、Git、Ninja |
| CMake | 3.30.0；本次 H100 编译使用此版本 |
| Megatron | `core_v0.12.3`，以上述 SHA 为准 |

先让系统具备 Git、GCC/G++、Make 和两套 Toolkit。Ubuntu 22.04 已配置 NVIDIA CUDA 软件源时，Toolkit 软件包名称为 `cuda-toolkit-12-4`、`cuda-toolkit-12-8`；软件源安装方法见 [NVIDIA Linux 安装指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html)。已有 Toolkit 时直接使用其实际路径。`nvidia-smi` 显示的 CUDA 版本不能替代 `nvcc --version` 检查。

```bash
conda create -n flux-h100 -c conda-forge python=3.10.20 pip -y
conda activate flux-h100
export PYTHONNOUSERSITE=1

python -m pip install setuptools==75.8.0 wheel==0.45.1 packaging==24.2
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
  numpy==1.26.4 einops==0.8.1 regex==2024.11.6 \
  psutil==6.1.1 PyYAML==6.0.2 tqdm==4.67.1 six==1.17.0 \
  pybind11==2.13.6 ninja==1.11.1.3 cmake==3.30.0

# 按新服务器实际安装位置修改。
export CUDA124=/usr/local/cuda-12.4
export CUDA128=/usr/local/cuda-12.8
export CUDA_HOME="$CUDA124"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export MAX_JOBS=2

"$CUDA124/bin/nvcc" --version
"$CUDA128/bin/nvcc" --version
"$CXX" --version
```

PyTorch 安装源与版本见 [官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。这里不需要安装 `torchvision`、`torchaudio`、Transformer Engine、FlashAttention 或额外的 `megatron-core` wheel；当前模型使用源码 Megatron、local transformer、合成数据及 NullTokenizer。

在同一环境编译 Apex，固定源码并保留实验实际需要的三个扩展：

```bash
export APEX_SRC="$(dirname "$REPO")/apex-flux"
git clone https://github.com/NVIDIA/apex.git "$APEX_SRC"
git -C "$APEX_SRC" checkout --detach \
  e13873debc4699d39c6861074b9a3b2a02327f92

python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['APEX_SRC']) / 'setup.py'
s = p.read_text()
old = '    ext_modules=ext_modules,'
new = ('    ext_modules=[ext for ext in ext_modules if ext.name in '
       '{"apex_C", "amp_C", "fused_layer_norm_cuda"}],')
assert s.count(old) == 1, 'Apex source differs from the pinned revision'
p.write_text(s.replace(old, new))
PY

CUDA_VISIBLE_DEVICES='' TORCH_CUDA_ARCH_LIST='9.0' \
  python -m pip install -v --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" "$APEX_SRC"

python -m pip check
python - <<'PY'
import torch
import apex_C, amp_C, fused_layer_norm_cuda
from apex.optimizers import FusedAdam
from apex.normalization import FusedLayerNorm
print(torch.__version__, torch.version.cuda)
print('Apex compiled extensions: OK')
PY
```

Apex 编译方式依据 [固定提交的 NVIDIA Apex 说明](https://github.com/NVIDIA/apex/tree/e13873debc4699d39c6861074b9a3b2a02327f92)。已有符合要求的环境可跳过安装步骤；详细历史依赖见 [公共软件环境说明](../env.md)。

如果使用 Docker，上述配置在容器内完成。容器需要能访问 8 张 GPU、支持 GPU IPC，并同时挂载 TempName 与 Megatron-LM。可以沿用 `--gpus all --ipc=host --network=host` 和父目录到 `/workspace` 的挂载方式，此时 `REPO=/workspace/TempName`、`MEGATRON_DIR=/workspace/Megatron-LM`；后续整个流程保持在同一容器/路径下执行。镜像自带的 PyTorch 2.5.1 或 2.7.0 不等于本实验的目标环境。

## 3. 核对八卡配置

从这里开始的设备检查、校准及测试命令会访问新服务器的 GPU。裸机且八卡均分配给本次实验时，可显式选择设备：

```bash
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nvidia-smi
nvidia-smi topo -m

python - <<'PY'
import torch
assert torch.cuda.device_count() == 8, 'Exactly eight visible GPUs are required'
counts = set()
for i in range(8):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, (p.major, p.minor), p.multi_processor_count, p.uuid)
    assert 'H100' in p.name and (p.major, p.minor) == (9, 0)
    counts.add(p.multi_processor_count)
assert len(counts) == 1 and counts <= {114, 132}
assert all(torch.cuda.can_device_access_peer(i, j)
           for i in range(8) for j in range(8) if i != j)
print('CUDA peer access: all pairs OK')
PY
```

调度器已经设置 `CUDA_VISIBLE_DEVICES` 时保留其分配；容器内使用实际可见的设备编号。当前路径要求完整 H100、SM90、同质八卡、NVLink 路径和两两 peer access。单纯“插了八张 H100”并不足以保证任意 PCIe 拓扑受支持。

默认 [config.json](../../../tools/h100/phase6/config.json) 已经是 **TP8、DP1、PP1、CP1、132 SM**，适用于完整 H100 SXM。若查询到 114 SM，先修改配置里的 `sm_count`，并确认拓扑受支持；从 baseline 构建开始的全部阶段使用同一配置。

下面用一个全新会话名组织所有产物。后续命令在同一个 shell 中顺序执行；重新连接后恢复这些变量，继续已有阶段时不要重新生成会话名。

```bash
export H100_SESSION="tp8-$(date +%Y%m%d-%H%M%S)"
export H100_BUILD="$REPO/outputs/h100/phase6/$H100_SESSION"
export H100_LOG="$REPO/logs/h100/phase6/$H100_SESSION"
export H100_CONFIG="$REPO/tools/h100/phase6/config.json"
printf 'build=%s\nlog=%s\n' "$H100_BUILD" "$H100_LOG"
```

## 4. 构建 baseline 与采样库

```bash
python tools/h100/phase6/build.py prepare "$H100_BUILD/base" \
  --config "$H100_CONFIG"
python tools/h100/phase6/build.py compile "$H100_BUILD/base" \
  --cuda-home "$CUDA128" --jobs 2
```

这一步只编译，不使用 GPU；builder 在子进程中隐藏 GPU，不会修改当前 shell 的八卡选择。NCCL 默认从仓库子模块源码构建到本次输出目录。已有 NCCL 开发安装时才使用 `--nccl-root`，不能把 PyTorch 的 NCCL wheel 目录直接当作含静态库的开发安装。

构建日志在 `$H100_BUILD/base/compile.log`。必须成功得到 `status=built`，再执行下一步。不要使用根目录的 A800 构建脚本替代本入口，也不要先 `pip install byte-flux` 或全局 `pip install -e .` 来替代独立库。

## 5. 在这台 H100 上重新校准

```bash
python tools/h100/phase6/calibrate.py prepare "$H100_LOG/calibration" \
  --build "$H100_BUILD/base"

# 这条命令实际启动八卡校准；移除 --execute 只会打印执行计划。
python tools/h100/phase6/calibrate.py run "$H100_LOG/calibration" \
  --cuda-home "$CUDA124" --execute

python tools/h100/phase6/plan.py "$H100_LOG/calibration" \
  "$H100_BUILD/selection-plan.json"
```

到达计划和量化 mask 使用本机实测数据生成。后续对比必须保持同一节点、同一组 GPU UUID 及顺序。更换机器、GPU 分配、形状或 TP 后重新准备；不要迁移旧计划或把诊断产物的 `diagnostic_only` 标记删除后冒充正式构建。

## 6. 构建组合候选并准备对比

```bash
python tools/h100/phase6/build.py prepare "$H100_BUILD/candidates" \
  --config "$H100_CONFIG" --plan "$H100_BUILD/selection-plan.json"
python tools/h100/phase6/build.py compile "$H100_BUILD/candidates" \
  --cuda-home "$CUDA128" --jobs 2

python tools/h100/phase6/prepare.py "$H100_LOG/compare" \
  --base-build "$H100_BUILD/base" \
  --candidate-build "$H100_BUILD/candidates" \
  --plan "$H100_BUILD/selection-plan.json" \
  --megatron "$MEGATRON_DIR"
```

默认六组是：`native`、`original`、`native_taco`、`taco_fused`、`remote_arrival_selective`、`interleaved_arrival_selective`。前两组为参考，后两组为组合候选，两个 TACO baseline 为必须对照。无需自己给每个策略手动切换 `.so` 或再套一层 `torchrun`；控制器会管理各自的 Python 路径和八个 rank。

## 7. 先验证，再计时

```bash
# 查看计划，不运行 GPU。
python tools/h100/phase6/run.py "$H100_LOG/compare"

# GPU swizzle、完整模型 smoke 与独立 profile。
python tools/h100/phase6/run.py "$H100_LOG/compare" \
  --stage preflight --cuda-home "$CUDA124" --execute

# 只有前一步通过后才运行两轮完整 optimizer-step 对比。
python tools/h100/phase6/run.py "$H100_LOG/compare" \
  --stage timing --cuda-home "$CUDA124" --execute

# 离线核验并生成 Markdown/JSON 报告；成功的 timing 也会自动生成报告。
python tools/h100/phase6/report.py "$H100_LOG/compare"
```

查看 `$H100_LOG/compare/report.md`、`report.json`；各次独立进程的错误和日志位于 `compare/results/*/process.log`。失败时先查看日志，保留失败目录，并以新会话重做需要变更的阶段，不覆盖原记录。

第一次部署包含依赖安装、多套库编译、校准、正确性检查及完整对比，**没有 20 分钟内完成的保证**。默认对比使用 12 层、H2048/FFN8192、S2048、micro batch1/global batch4，并执行两轮各 36 个独立计时窗口。校准完成后不要移动源码/产物路径、修改被冻结的输入或更新 Megatron；manifest 会核验路径和哈希。

量化 MLP 是 SM90 编译的 Phase6 GemmV2 移植版，attention 与原生 Flux 参考使用 Hopper GemmV3。通过 preflight 后取得的实际计时数据才用于判断该机器上的运行效果；本指南不预设提速结果。
