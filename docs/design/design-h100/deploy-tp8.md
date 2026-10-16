# 新服务器：单机 8 张 H100 部署与九组消融实验3

更新：2026-10-16。本文用于在**新服务器**配置环境，并执行一次 [消融实验3](消融实验3.md)：三种策略 × MLP、Attention、MLP＋Attention 三种作用范围，共九组。模型为 GPT 6.7B、单节点8×H100、TP8，每卡必须为132 SM。无需Slurm；受调度器管理时，在已分配的计算节点内执行GPU步骤。

**本指南已替换原六策略运行流程，全程使用 `ablation3.py`。** 一次实验指一套两轮、162个正式窗口，另有18个预检窗口，不表示连续多次复测。详细分组和统计口径见 [消融实验3](消融实验3.md)，CLI说明及开发验证见 [README-ablation3.md](../../../tools/h100/phase6/README-ablation3.md)。

新服务器必须重新校准 Attention 和 MLP，重新编译实际运行库。本文不复用旧机器的到达计划、mask或二进制。开发端已通过85项CPU测试和CUDA对象编译，但尚未完成本九组的完整运行库链接或GPU验证；这些阶段在新服务器按下文执行。模型定义见 [模型说明](../model-gpt-6.7b.md)，交付规范见 [执行端说明](instruction-h100.md)。

## 可直接交给新服务器执行者的指令

> 在实际 TempName 仓库根目录工作，按照 `docs/design/design-h100/deploy-tp8.md` 完成环境配置，再实际执行一次 `docs/design/design-h100/消融实验3.md` 的九组实验。本次是H100执行阶段，允许GPU设备检查、两角色BF16校准、mapping、smoke/profile和正式计时，不只做CPU检查或dry-run。先确认已同步九组代码及所需冻结源码，核对单节点8张132 SM H100和软件版本。新建独立会话，在本机重新校准Attention与MLP，显式传入两份本机计划，完整编译采样库及base/arrival两类候选库。使用 `ablation3.py` 完成九组预检后执行两轮、每轮9块、共162个正式窗口，不增加旧baseline或其他实验组。保留全样本、单列慢窗口敏感性，生成报告并按文档交付原始记录。GPU忙时停止并说明，不终止其他任务、不创建自动等待队列；运行中不周期查询GPU占用，不自动追加独立复测。失败保留现场及缺项，不覆盖旧结果。

以上是给**有H100的新服务器**的执行任务。本次开发端只更新指南，不启动GPU实验。

用户在IDE中的 `hjr/TempName/` 是本地工作区前缀；进入新服务器实际仓库根目录后，使用上述 `docs/design/...` 相对路径。`消融实验3.md` 和README中“本次开发未进行GPU测试”的说明描述的是开发交付状态；复制上述执行指令后，任务包含本指南第3、5、7节的GPU步骤。实验分组以消融实验3为准，不执行旧五组、四组或六策略入口。

## 1. 获取完整源码

先将九组新增代码和本文提交并同步到新服务器使用的仓库版本；开发端未提交的新文件不会随 `git pull` 自动迁移。不要只复制 `tools/h100/`：构建器还读取 `src/`、`include/`、`python/`、共享A800工具、固定候选源码及其manifest。

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
test -f tools/h100/phase6/ablation3.py
test -f tools/h100/phase6/README-ablation3.md
test -f outputs/h100/phase6/frozen-candidate-v1-20261013/manifest.json
test -f outputs/h100/phase6/gpt67-tp8-20261010-112249/candidates-20261010-153517/manifest.json

export MEGATRON_DIR="$(dirname "$REPO")/Megatron-LM"
# 已有 Megatron-LM 目录时跳过 clone，在干净工作区中核对固定提交。
git clone https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON_DIR"
git -C "$MEGATRON_DIR" checkout --detach \
  3ea68ad6042cc1204386ae9364358f7c4de1bc37
git -C "$MEGATRON_DIR" rev-parse HEAD
git -C "$MEGATRON_DIR" status --short
```

Megatron 必须是上述精确提交，受版本控制的文件保持未修改。Git已携带部分H100日志、trace、配置与冻结源码；编译库、构建缓存和Conda环境通常不会随clone迁移。拉到历史结果不等于新机器已有可运行环境，旧机器的 `.so`、诊断产物和A800到达计划都不作为新机器的运行输入。

九组构建还需要下列已跟踪文本来源，不能因为位于 `outputs/` 而省略：

- `outputs/h100/phase6/frozen-candidate-v1-20261013/` 的冻结登记及源码副本。
- `outputs/h100/phase6/gpt67-tp8-20261010-112249/candidates-20261010-153517/` 的manifest，以及其中 `remote_arrival_selective/`、`interleaved_arrival_selective/` 两套登记源码。构建器会核验两套来源，即使本轮只测试交替顺序。
- 固定版本的CUTLASS、NCCL子模块及维护代码。链接脚本、`.clang-format`、已登记的Triton输入等也须按manifest保留。

这些历史文件提供可核验的**计算源码模板**，不表示复用旧机器的运行库或校准计划。缺少文件时补齐对应提交的原文件，不改manifest绕过检查。

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
import sys
import numpy as np
import torch
import apex_C, amp_C, fused_layer_norm_cuda
from apex.optimizers import FusedAdam
from apex.normalization import FusedLayerNorm
assert sys.version_info[:3] == (3, 10, 20), sys.version
assert torch.__version__ == '2.6.0+cu124', torch.__version__
assert torch.version.cuda == '12.4', torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI is False, 'Expected the recorded PyTorch ABI=0 environment'
assert np.__version__ == '1.26.4', np.__version__
print('Python:', sys.executable)
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('NumPy:', np.__version__, 'C++ ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)
print('Apex compiled extensions: OK')
PY

# 固定后续worker解释器；重新连接时先激活同一Conda环境，再恢复该变量。
export H100_PYTHON="$(python -c 'import sys; print(sys.executable)')"
```

Apex 编译方式依据 [固定提交的 NVIDIA Apex 说明](https://github.com/NVIDIA/apex/tree/e13873debc4699d39c6861074b9a3b2a02327f92)。已有符合要求的环境可跳过安装步骤；详细历史依赖见 [公共软件环境说明](../env.md)。

已有环境跳过安装时仍执行上述导入/版本检查，并设置 `H100_PYTHON`。这些检查不创建CUDA张量；Torch/Apex导入成功只说明依赖可加载，GPU可用性在第3节检查，实际候选运行库仍需后续完整编译及预检。编译和执行始终使用这个Conda环境，不能在链接扩展后切换PyTorch或ABI。

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
assert counts == {132}, 'Ablation3 requires eight 132-SM H100s'
assert all(torch.cuda.can_device_access_peer(i, j)
           for i in range(8) for j in range(8) if i != j)
print('CUDA peer access: all pairs OK')
PY
```

调度器已经设置 `CUDA_VISIBLE_DEVICES` 时保留其分配；容器内使用实际可见的设备编号。当前路径要求完整 H100、SM90、同质八卡、NVLink 路径和两两 peer access。单纯“插了八张 H100”并不足以保证任意 PCIe 拓扑受支持。

九组入口读取冻结模型配置，固定TP8/132 SM。目前不支持把 `sm_count` 改为114后直接运行本实验；114 SM、MIG或不满足peer access的设备应停止并说明，不修改冻结配置伪装成支持。GPU身份、可见顺序、容器和产物绝对路径在校准及正式实验期间保持一致。

## 4. CPU来源核验与新会话目录

在上述Conda环境、仓库根目录执行。下面的来源检查不导入Torch、不查询GPU：

```bash
cd "$REPO"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python - <<'PY_CHECK'
import sys
sys.path.insert(0, 'tools/h100/phase6')
from candidate import sources
freeze, root, manifest = sources()
print('Frozen compute sources verified:', root)
print('Measured source commit:', manifest['source_commit'])
PY_CHECK

export H100_A3_SESSION="ablation3-gpt67-tp8-scope-$(date +%Y%m%d-%H%M%S)"
export H100_A3_OUT="$REPO/outputs/h100/phase6/$H100_A3_SESSION"
export H100_A3_LOG="$REPO/logs/h100/phase6/$H100_A3_SESSION"
export H100_A3_PLANS="$H100_A3_OUT/plans/measured"
printf 'session=%s\nbuild=%s\nlog=%s\n' "$H100_A3_SESSION" "$H100_A3_OUT" "$H100_A3_LOG"

# 先记录本次会话的路径和环境，便于断开连接后找到同一套产物。
mkdir -p "$H100_A3_LOG/handoff"
python - <<'PY_SESSION'
import json
import os
from pathlib import Path
names = ('REPO', 'MEGATRON_DIR', 'CUDA124', 'CUDA128', 'H100_PYTHON',
         'H100_A3_SESSION', 'H100_A3_OUT', 'H100_A3_LOG', 'H100_A3_PLANS',
         'CUDA_VISIBLE_DEVICES')
values = {name: os.environ.get(name) for name in names}
assert all(values[name] for name in names if name != 'CUDA_VISIBLE_DEVICES'), 'Missing session environment variable'
(Path(os.environ['H100_A3_LOG']) / 'handoff/session-paths.json').write_text(
    json.dumps(values, ensure_ascii=False, indent=2) + '\n')
PY_SESSION
git rev-parse HEAD > "$H100_A3_LOG/handoff/tempname-commit.txt"
git submodule status --recursive > "$H100_A3_LOG/handoff/submodules.txt"
git -C "$MEGATRON_DIR" rev-parse HEAD > "$H100_A3_LOG/handoff/megatron-commit.txt"
python -m pip freeze > "$H100_A3_LOG/handoff/pip-freeze.txt"
```

所有阶段输出目录必须是新目录；每一步成功后才执行下一步。`handoff/session-paths.json` 只记录路径与可见设备，不代表对应阶段已完成。重新连接后激活同一环境，按该文件恢复原会话变量，不重新生成会话名，也不对已启动的校准/compare目录再次执行整套任务；先查看阶段状态和原进程是否还在运行。可先用以下命令做CPU回归：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tools/h100/phase6 -p 'test_*.py' -v
```

## 5. 完整编译双shape采样库，并在新机器校准两角色

```bash
python tools/h100/phase6/ablation3.py calibration-build "$H100_A3_OUT/calibration" \
  --roles attn mlp
CUDA_VISIBLE_DEVICES='' python tools/h100/phase6/ablation3.py compile "$H100_A3_OUT/calibration" \
  --cuda-home "$CUDA128" --jobs 2
python tools/h100/phase6/ablation3.py calibration-prepare "$H100_A3_LOG/calibration" \
  --build "$H100_A3_OUT/calibration"

# 先打印计划，不启动GPU。
python tools/h100/phase6/ablation3.py calibration-run "$H100_A3_LOG/calibration"

# 以下命令实际启动本机八卡BF16校准。
python tools/h100/phase6/ablation3.py calibration-run "$H100_A3_LOG/calibration" \
  --python "$H100_PYTHON" --cuda-home "$CUDA124" --execute
python tools/h100/phase6/ablation3.py fit "$H100_A3_LOG/calibration" "$H100_A3_PLANS"
```

采样库要完成动态库、Python扩展和mapping程序的完整编译，manifest达到 `status=built`；仅 `compile-check` 的对象编译不能替代。编译阶段的空 `CUDA_VISIBLE_DEVICES` 只作用于单条命令，不改变当前shell的八卡选择。NCCL默认从固定子模块构建；若提供 `--nccl-root`，它必须是含头文件与静态库的开发安装，不能直接使用PyTorch的NCCL wheel目录。

Attention使用 `[2048,4096,4096]`、本地K512；MLP使用 `[2048,4096,16384]`、本地K2048。两角色分别拟合，pass0/1训练、pass2诊断，输出本机 `attn.json` 与 `mlp.json`。新服务器两份都必须重建，不采用README中的旧服务器计划复用示例。

启动前工具会核对GPU状态，忙时退出，不自动等待或重试。失败记录保留；修复需要变更的环境/源码后，在新目录重新准备对应阶段，不在旧目录改哈希或补写完成标记。

## 6. 从本机计划完整编译九组候选

```bash
python tools/h100/phase6/ablation3.py build-prepare "$H100_A3_OUT/candidates" \
  --attn-plan "$H100_A3_PLANS/attn.json" \
  --mlp-plan "$H100_A3_PLANS/mlp.json"
CUDA_VISIBLE_DEVICES='' python tools/h100/phase6/ablation3.py compile "$H100_A3_OUT/candidates" \
  --cuda-home "$CUDA128" --jobs 2
```

**新服务器必须显式传入上述两份新计划。** 不省略 `--mlp-plan`，因为省略时入口会尝试使用旧冻结MLP计划。正式会话不传 `--source-only`；该标记只用于CPU演练，生成的worker永久拒绝GPU执行，不能靠修改标记转为实测。

构建生成两套双shape库：`candidates/base/` 对应基础交替，`candidates/arrival/` 对应到达置换。A组共用base库，B/C共用arrival库，通过全远端/定向mask区分。两套库均需完成完整编译链接及Python扩展构建；编译日志分别位于两个目录的 `compile.log`，两份manifest都须为 `status=built`。不要使用旧六组的 `build.py prepare` 或 `prepare.py` 替代本节命令。

## 7. 准备并执行一次九组对比

```bash
python tools/h100/phase6/ablation3.py prepare "$H100_A3_LOG/compare" \
  --build "$H100_A3_OUT/candidates" --megatron "$MEGATRON_DIR" --blocks 9

# 默认dry-run：打印12项角色mapping和180个worker任务，不使用GPU。
python tools/h100/phase6/ablation3.py run "$H100_A3_LOG/compare"

# 实际执行：mapping、九组smoke/profile通过后，继续两轮正式计时。
python tools/h100/phase6/ablation3.py run "$H100_A3_LOG/compare" \
  --stage all --python "$H100_PYTHON" --cuda-home "$CUDA124" --execute

# all成功时已生成报告；这条命令在原环境重新做CPU核验并生成报告。
python tools/h100/phase6/ablation3.py report "$H100_A3_LOG/compare"
```

一次完整实验的固定规模为：

| 项目 | 数量或定义 |
| --- | --- |
| 策略 | A：交替＋完整量化；B：交替＋到达优先＋完整量化；C：交替＋到达优先＋选择性量化 |
| 作用范围 | 仅MLP、仅Attention、MLP＋Attention；三策略×三范围＝九组 |
| 正式窗口 | 九组同块，每轮9块，两轮共162窗口 |
| 预检 | 18个smoke/profile窗口、12个策略/角色mapping组合 |
| 正式计时 | 每窗口10步预热＋20步完整optimizer step；共3240个正式step |
| 完整step | 整个32层模型、4个微批的前向与反向、通信/同步、一次参数更新、学习率调度及清梯度 |

完整量化只针对所选模块全部远端贡献；GEMM、本地贡献和未选中模块仍为BF16。Attention仅指 `self_attention.linear_proj`，MLP仅指 `mlp.linear_fc2`；未选中模块统一保留原生Megatron BF16。每组均由独立进程执行，控制器管理库路径与八个rank，不需额外套 `torchrun`。

九组不增加原生Megatron、朴素量化、固定融合v2或其他对照；不套用旧六组的4%门槛。所有ms/step均是完整训练步，报告保留全样本及负收益；逐轮给出九项同范围策略比较和九项同策略范围比较。慢窗口筛选仅为附表，不能替代主报告或删改原始数据。

预检失败时不会进入正式计时；GPU忙时停止并说明，不建立等待队列。一套运行中不周期查询GPU占用，不额外启动独立复测。显存、数值及性能以这台新机器的实际预检和计时为准，不能用开发端CPU检查或旧服务器结果替代。

## 8. 核验结果与交付

检查 `$H100_A3_LOG/compare/report.json`、`report.md`、`preflight.json` 和 `all-state.json`；成功应具有完整数据及完成状态。失败时查看相应阶段状态与 `results/*/process.log`，保留失败目录及原因。新服务器首次部署包含安装、完整链接、双角色校准及180个worker窗口，不承诺固定分钟数内完成。

一次默认完整交付至少包含：

- 本机两角色校准原始记录、mapping、profile、完成状态与 `attn.json`/`mlp.json`。
- 采样库和base/arrival候选库的manifest、源码快照、完整编译日志、实际运行库哈希及依赖版本。
- 九组submission、冻结scripts、schedule、12项角色mapping与preflight。
- 1296份正式计时rank、144份预检rank、72份压缩profile trace，以及命令和进程日志。
- 全样本报告、敏感性附表（如生成）、阶段状态和 `handoff/` 中的执行说明，包括源码提交/未提交补丁、环境、会话路径、完成状态及缺项。

正式实验文本和trace按 [交付规范](instruction-h100.md#测试完成或中断后交什么) 回传；运行库和编译缓存由执行端保留，不提交`.so/.o/.a`、环境或checkpoint。缺项如实列出，不修改manifest绕过验证。将数据复制回开发端分析时保留记录中的原绝对路径；接收端可用 `ablation3.py report <compare目录> --offline` 核验文本证据，但这不替代原环境的运行库核验。
