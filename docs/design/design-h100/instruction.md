# H100 单节点 Phase6 接续说明

更新：2026-10-10。本目录记录 H100 代码移植，与 [A800 接续说明](../design-a800/instruction.md) 分开维护。

当前目标是单节点 TP4/TP8 下的 baseline 与“基础顺序＋到达优先＋选择性量化”完整 optimizer-step 对比。默认 TP8/DP1，不支持跨节点。**后续主模型已改为 GPT 6.7B：32层、H4096、FFN16384、32 heads、词表32768；默认 S2048/mb1/global4。** 参数量按当前独立嵌入/输出头及词表补齐约6.729B，详见[统一模型说明](../model-gpt-6.7b.md)。本次按用户要求仅改代码与文档，不编译、不测试、不提交作业。

已有 `gxn74` 的[六组两轮实测](base-phase6.md)属于旧12层、约0.64B模型，记录原样保留；此前本机四卡短测未能提交的记录也保留。旧模型的编译/正确性/性能结果不代表6.7B已通过验证，新模型需要新的构建、校准、计划和预检。

## 入口

| 内容 | 路径 |
| --- | --- |
| A800/H100 6.7B模型定义、形状与准备入口 | [model-gpt-6.7b.md](../model-gpt-6.7b.md) |
| 新服务器环境、源码迁移与八卡运行步骤 | [单机 TP8 部署指南](deploy-tp8.md) |
| 完整命令、baseline 定义与实现边界 | [H100 Phase6 README](../../../tools/h100/phase6/README.md) |
| 模型、TP、SM 数、基础顺序和预算 | [config.json](../../../tools/h100/phase6/config.json) |
| 6.7B TP4、S1024、micro1/global8 场景 | [config-tp4-s1024.json](../../../tools/h100/phase6/config-tp4-s1024.json) |
| 历史0.64B配置，仅用于旧模型复现 | [TP8](../../../tools/h100/phase6/config-0.64b-tp8.json) · [TP4 S1024](../../../tools/h100/phase6/config-0.64b-tp4-s1024.json) |
| 历史0.64B的H100实测 | [base-phase6.md](base-phase6.md) |
| 独立源码快照与无 GPU 编译 | [build.py](../../../tools/h100/phase6/build.py) |
| 新 H100 校准准备/执行 | [calibrate.py](../../../tools/h100/phase6/calibrate.py) |
| 到达表与量化 mask 拟合 | [plan.py](../../../tools/h100/phase6/plan.py) |
| 冻结模型实验与基线 | [prepare.py](../../../tools/h100/phase6/prepare.py) |
| 单节点 torchrun 控制器 | [run.py](../../../tools/h100/phase6/run.py) |
| 完整性核验与配对统计 | [report.py](../../../tools/h100/phase6/report.py) |
| CPU 协议检查 | [test_protocol.py](../../../tools/h100/phase6/test_protocol.py) |

## 生成数据与文件放置规范

所有路径都以 **TempName 仓库根目录**为基准。每个模型/TP/形状/节点校准使用独立会话名，6.7B建议 `gpt67-tp8-YYYYMMDD-HHMMSS`；不要复用旧0.64B的会话目录，也不要把产物放进 `src/`、`tools/` 或 `docs/`。

| 文件类别 | 放置位置 | 是否纳入 Git |
| --- | --- | --- |
| 维护的模型配置、脚本、CUDA/C++源码 | `tools/h100/phase6/`、`src/`、`include/`、`python/` | 是 |
| 实验设计、已核验结论、结果摘要 | `docs/design/design-h100/` | 是；标明模型、节点、配置和证据路径 |
| baseline/sampler源码快照、编译库、NCCL开发构建 | `outputs/h100/phase6/<会话>/base/` | 否 |
| 到达置换表与量化mask合并计划 | `outputs/h100/phase6/<会话>/selection-plan.json` | 否；由真实校准生成 |
| 候选源码快照、查表头文件、动态库 | `outputs/h100/phase6/<会话>/candidates/` | 否 |
| 接收端到达采样、GPU映射CSV、校准worker与日志 | `logs/h100/phase6/<会话>/calibration/` | 否 |
| 对比用脚本/配置快照、输入清单 | `logs/h100/phase6/<会话>/compare/scripts/`、`compare/submission.json` | 否 |
| smoke/profile/计时的每rank记录、trace、进程日志 | `logs/h100/phase6/<会话>/compare/results/` | 否 |
| 自动生成的统计结果 | `compare/report.json`、`compare/report.md`、`compare/preflight.json`及阶段状态文件 | 否；择要归纳到设计文档 |
| 如以后使用真实训练语料 | `data/h100/<数据集>/`或外部数据盘 | 否；当前入口不读取这里 |
| 如以后启用保存权重 | `checkpoints/h100/<会话>/` | 否；当前测量入口不保存checkpoint |

**当前合成数据不用提前生成磁盘语料文件。** `--mock-data`/NullTokenizer入口由worker按固定 `token_seed` 生成GPU驻留token bank，模型初始化使用 `seed`；同配置的对照组复用相同生成规则。默认有效token为0–32767，EOD/词表补齐由Megatron处理。种子、模型配置和token哈希保存在配置快照及每rank报告中，不生成待提交Git的 `.bin/.idx` 数据集。

典型目录结构如下，文件由对应工具生成，不手工补写“完成”标志或虚构测量数据：

```text
TempName/
├── tools/h100/phase6/config.json               # 维护的6.7B默认配置
├── docs/design/design-h100/                   # 设计说明和结果摘要
├── outputs/h100/phase6/gpt67-tp8-<时间>/
│   ├── base/                                 # build.py prepare/compile
│   │   ├── manifest.json
│   │   ├── config.json
│   │   ├── compile.log
│   │   ├── original/
│   │   ├── taco_fused/
│   │   ├── sampler_remote_first/
│   │   └── sampler_interleaved/
│   ├── selection-plan.json                   # plan.py
│   └── candidates/                           # 第二次build.py prepare/compile
└── logs/h100/phase6/gpt67-tp8-<时间>/
    ├── calibration/                          # calibrate.py
    │   ├── worker.py / config.json / plan.json
    │   ├── mapping-*.csv
    │   ├── b0-*-instrumented-rank*.json        # 原始到达观察
    │   ├── *.log / execution.json
    │   └── calibration-complete.json          # 全部校准成功后生成
    └── compare/                              # prepare.py / run.py / report.py
        ├── scripts/                          # 包含冻结的selection-plan.json副本
        ├── build/                            # 指向本会话动态库的链接
        ├── submission.json / preflight.json
        ├── results/
        │   ├── smoke-<策略>/rank*.json
        │   ├── profile-<策略>/rank*-profile.json.gz
        │   ├── round1/window-*/rank*.json
        │   └── round2/window-*/rank*.json
        └── report.json / report.md
```

在同一个shell中设置并保存会话路径，然后按[部署指南](deploy-tp8.md)传给每个阶段：

```bash
export REPO="$PWD"  # 必须在TempName根目录执行
export H100_SESSION="gpt67-tp8-$(date +%Y%m%d-%H%M%S)"
export H100_BUILD="$REPO/outputs/h100/phase6/$H100_SESSION"
export H100_LOG="$REPO/logs/h100/phase6/$H100_SESSION"
export H100_CONFIG="$REPO/tools/h100/phase6/config.json"
```

相邻 `Megatron-LM`、Apex源码和Conda环境按[环境说明](../env.md)管理，不放在实验结果目录冒充生成数据。`outputs/`、`logs/`、`data/`、`checkpoints/` 已被根目录 `.gitignore` 忽略；clone不会携带它们。原始证据需要另行备份，文档只保存摘要与来源路径。冻结后保持本机路径、代码/配置哈希、GPU UUID及顺序一致；改变配置、移动会话或更换服务器时建立新会话并重新准备/校准，不通过改manifest绕过核验。

## 实现边界

量化 MLP 使用重新编译为 SM90 的 Phase6 GemmV2 算法；attention 和原生 Flux 参考使用 Hopper GemmV3。该版本没有实现基于 WGMMA/TMA 的 Hopper 原生融合 TACO epilogue，不能将兼容移植称为完成 H100 性能优化。

必须 baseline 为原生 Megatron＋朴素 TACO、原始顺序 Flux＋全远端融合 TACO；参考为原生 Megatron、原生 Hopper Flux。默认候选为远端优先和交替两种基础顺序，可配置另外两种交替顺序。范围仅 MLP `linear_fc2` 前向，BF16 GEMM、本地 BF16、H128/E4M3 远端协议及 BF16 STE 反向沿用现有设计。

构建与结果分别放 `outputs/h100/phase6/`、`logs/h100/phase6/`。必须从 H100 重新采集到达观察并生成选择计划；工具拒绝 A800 计划和未完成构建。候选准备依赖真实校准，不用合成 CPU 测试输入替代 GPU 证据。

后续 GPU 工作需再获用户指令，届时按 README 执行校准、独立候选编译、swizzle/模型 smoke/profile 和完整六组/八组两轮测量。本次未启动这些 GPU 步骤。

## 本次无 GPU 验证（2026-10-09）

以下是2026-10-09、旧0.64B配置的历史记录，未在6.7B上重跑。编译和导入进程均设置 `CUDA_VISIBLE_DEVICES=''`：

| 检查 | 结果与范围 |
| --- | --- |
| Python CPU 协议测试 | 15 项通过；覆盖 TP4/TP8、四种基础顺序、到达计划/量化预算、留出数据隔离、输入冻结和 dry-run 不启动 worker |
| 注册代码生成器 | 132 SM 和 114 SM 两种配置的主机 C++ 编译通过；完整动态库编译采用默认 132 SM 配置 |
| CUDA 交叉编译 | SM90(a) 的 TACO runtime、融合 V2、采样 V2、合成到达表候选和实际 occupancy 查询函数通过；`cuobjdump` 确认量化 V2 对象包含 `sm_90a.cubin` |
| 完整库与 Python binding | `original`、`taco_fused`、`sampler_remote_first`、`sampler_interleaved` 四套独立库完成编译、链接和 Python 扩展构建 |
| CPU 导入 | 四套库使用 CUDA SDK 驱动桩通过 Torch/Flux 导入、绑定与导出符号检查，确认没有初始化 CUDA 上下文；没有调用 occupancy 查询或任何 kernel |
| GPU mapping 检查器 | 融合 baseline 和两套采样库的检查器完成编译，未执行 |
| 最终源码生成 | 重新执行 `build.py prepare` 成功；与已编译快照核对，只有等价的链接导出符号排列顺序不同 |

本机编译环境为 Python 3.10.20、PyTorch 2.6.0+cu124、CUDA Toolkit 12.8、G++ 11.4 和 CMake 3.30；编译检查使用现有 NCCL 开发安装。编译产物与记录位于 `outputs/h100/phase6/compile-check-20261009-r2/`，汇总为其中的 `validation.json`。该目录设为 `diagnostic_only=true`，工具禁止将其直接用于正式测量；这些被 Git 忽略的本地产物不会随 clone 迁移。

修复了无 GPU 编译发现的两处问题：H100 快照避免 `setup_requires` 再次联网下载 CMake；删除没有调用方的全局 CUDA 分配，使导入发生在绑定 rank 设备之前时也不会隐式分配设备内存。检查器改为查询实际编译 kernel 的 occupancy，不再沿用 A800 的固定数值。

上述编译记录当时不包含GPU验证；后来旧0.64B在gxn74的实测见[独立记录](base-phase6.md)。6.7B的真实校准、候选构建、GPU正确性、profile、性能和收敛均未验证。合成表只用于历史编译诊断，不作为真实校准计划或性能证据。
