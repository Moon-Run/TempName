# 开发与分析端：H100项目接续说明

更新：2026-10-11。**本页供 `/data/run01/scyb672` 这台服务器新开的Codex对话阅读。** 仓库根目录为 `/data/run01/scyb672/hjr/TempName`。本机负责源码维护、接收H100证据、CPU分析和文档/论文整理；本任务的H100测试在另一台服务器 `gxn74` 上执行。

H100端“怎么测试、测试完交什么”统一维护在 [instruction-h100.md](instruction-h100.md)，部署命令在 [deploy-tp8.md](deploy-tp8.md)。本页负责告诉开发端在哪里找代码、数据、结论，以及哪些事情尚未验证。最新用户任务决定本轮范围，不因历史实验记录或待验证列表自动启动新测试。

## 新对话先读什么

1. 先看工作区状态与最近提交，保留用户未提交改动；确认是否新拉取了数据，不能把下方历史快照当作远端实时进度。
2. 读本页[数据索引](#已接收数据与当前缺项)，然后读 [base-phase6.md](base-phase6.md) 的对应模型结果，6.7B统一展示第1–4轮：原实验为第1–2轮、复测为第3–4轮；旧0.64B另列。
3. 性能分析约定放在 `docs/design/design-h100/thinking.md`，当前磁盘缺失，先确认是否另存或尚未提交。不要假称已读到其内容。**旧小模型在H100上也没有相对融合v2的稳定优势，不能主要归因于6.7B变大或其OOM。**
4. 涉及方法或实现时，按下方源码地图看维护入口及对应实验快照；涉及新测试/补交数据时，将要求与执行端文档对齐。
5. 读用户当前指定的文件再工作；论文目录的实际内容以文件为准，不把此前讨论中的计划当作已完成论文。

可以先执行以下只读命令，不启动GPU：

```bash
cd /data/run01/scyb672/hjr/TempName
git status --short
git log -5 --oneline
rg --files docs/design/design-h100 tools/h100/phase6
```

## 文档和代码在哪里

| 目的 | 入口 |
| --- | --- |
| H100执行与数据交付约定 | [instruction-h100.md](instruction-h100.md) |
| 新服务器环境、八卡部署命令 | [deploy-tp8.md](deploy-tp8.md)、[公共环境](../env.md) |
| H100模型结果与证据链接 | [base-phase6.md](base-phase6.md) |
| H100收益减弱的分析 | `docs/design/design-h100/thinking.md`（当前缺失，待恢复/补交） |
| 模型层数、参数量、全局/局部shape | [model-gpt-6.7b.md](../model-gpt-6.7b.md) |
| 原始方法与A800历史 | [A800接续](../design-a800/instruction.md)、[idea.md](../design-a800/idea.md) |
| A800可比数据 | [TP8](../design-a800/base-phase6-tp8.md)、[TP4](../design-a800/base-phase6-tp4.md)、[TP4优化](../design-a800/optimize/tp4/optimize2.md)；不要混入跨节点/DP实验 |
| 论文草稿与思想调研工作区 | `docs/design/paper/`；检查实际文件及完成状态 |
| H100协议、baseline定义 | [tools/h100/phase6/README.md](../../../tools/h100/phase6/README.md) |
| H100配置 | [config.json](../../../tools/h100/phase6/config.json)、[TP4配置](../../../tools/h100/phase6/config-tp4-s1024.json)；旧模型使用同目录 `config-0.64b-*.json` |
| 独立构建与注册项 | [build.py](../../../tools/h100/phase6/build.py)、[generator.cc](../../../tools/h100/phase6/generator.cc) |
| 校准、置换与选择计划 | [calibrate.py](../../../tools/h100/phase6/calibrate.py)、[plan.py](../../../tools/h100/phase6/plan.py)；复用 `tools/a800/arrival/plan.py`、`tools/a800/phase5/plan.py` 的算法 |
| 模型冻结、运行、统计 | [prepare.py](../../../tools/h100/phase6/prepare.py)、[run.py](../../../tools/h100/phase6/run.py)、[report.py](../../../tools/h100/phase6/report.py)、[common.py](../../../tools/h100/phase6/common.py) |
| 共享模型worker与适配 | `tools/a800/phase5/e2e/`；H100 prepare会生成修改后的worker快照 |
| Python算子、CUDA codec与接收端 | [gemm_rs_taco.py](../../../python/flux/gemm_rs_taco.py)、[taco_codec.cuh](../../../src/gemm_rs/taco_codec.cuh)、[taco_runtime.cu](../../../src/gemm_rs/taco_runtime.cu) |
| GEMM封装与tile调度 | [gemm_reduce_scatter.cc](../../../src/gemm_rs/ths_op/gemm_reduce_scatter.cc)、[threadblock_swizzle.hpp](../../../src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp) |

`src/`、`include/`、`python/`、`tools/`是受维护代码；`outputs/h100/phase6/<会话>/`内源码是**实验冻结证据**。修改实现应回到维护入口，再生成新构建，不直接修改旧快照。新旧源码可能不同，解释历史结果以该会话manifest及快照为准。H100工具依赖受维护的A800辅助实现，不要仅凭目录名把 `tools/a800/` 当作无关代码。

## 本机数据分布与读取顺序

所有相对路径均从仓库根目录起算：

| 想查看的内容 | 去哪里找 |
| --- | --- |
| 最近交接、环境与缺项 | `logs/h100/phase6/<会话>/handoff/`；历史会话若没有，结合结果文档、submission及状态文件，不虚构交接记录 |
| 模型、策略、种子、run_id和运行库来源 | `<compare>/submission.json`，以及 `<compare>/scripts/config.json` |
| 校准原始观察与有效计划 | `logs/.../<calibration>/`、`outputs/.../selection-plan*.json`；先确认校准完成，避免选到失败目录 |
| 构建来源、配置、源码快照及库哈希 | `outputs/.../base/manifest.json`、`outputs/.../<candidates>/manifest.json`及各变体源码 |
| 正确性和实际执行内核 | `<compare>/preflight.json`、`results/smoke-*/rank*.json`、`results/profile-*/rank*.json`、`rank*-profile.json.gz`、mapping CSV |
| 正式step耗时、loss、梯度、显存 | `<compare>/results/round*/window-*/rank*.json`；必须看全部rank |
| 启动参数与错误现场 | 各窗口的 `command.json`、`process.log`，以及失败备份、阶段状态与续跑日志 |
| 完整统计与验收 | 完成后的 `<compare>/report.json`、`report.md`；若有独立核验，连同脚本一起查看 |
| 历史A800原始数据 | `logs/a800/`、`outputs/a800/`；本地可能存在，但不保证随clone携带，缺失时明确只能依据文档摘要 |

自 `279fc32` 起，Git已携带H100日志、trace及部分构建文本证据；`.so/.a/.o`、构建缓存、Conda/Docker环境通常不在Git中。具体提交规则与完整产物树见[执行端文件规范](instruction-h100.md#生成数据与文件放置规范)。`data/`和`checkpoints/`不是当前测量入口的输入：当前为固定seed的GPU合成token，不需要另外寻找语料 `.bin/.idx` 或模型权重。

建议顺序：**交接/状态 → submission确认身份 → report了解结果 → 原始rank记录核对统计 → 校准/计划/profile解释原因 → 冻结源码解释实现**。只有摘要不足以验证性能；仅有profile也不能替代正式计时。

## 已接收数据与当前缺项

**本次以提交 `e7a0c49` 的工作区为准，最新6.7B对比是 `compare-20261011-114343-repeat`。** 数据接收状态不代表远端当前进程状态；下一次pull后应重新核对报告与run_id。

| 模型 / 对比 | 结果目录（仓库根目录起算） | 已接收状态 |
| --- | --- | --- |
| 6.7B，第3–4轮（复测） | `logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/` | 72窗口、576份计时rank记录，独立预检和最终报告完整；两候选均`accepted=false`，timing为`completed`，未设置resumed |
| 6.7B，第1–2轮（原实验） | `logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/` | 72窗口、576份计时rank记录，最终报告完整；两候选均`accepted=false`，timing为`completed, resumed=true` |
| 旧0.64B | `logs/h100/phase6/tp8-20261009-deployment/compare/` | 完整72窗口；两候选均`accepted=false`，另有独立核验与窗口统计 |

优先读[最新report.json](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/report.json)、[最新report.md](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/report.md)、[最新timing状态](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/timing-state.json)和[最新submission](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/submission.json)。最新复测run_id为 `b113bd6141374034aa3e050cbf3aab95`，前次为 `e1cc608329cc4bb2ae8e2506f8352ab5`。在base-phase6中统一为四轮：前次报告round1/round2对应第1/2轮，复测报告round1/round2对应第3/4轮。各轮保留原配对统计，查数据时按此映射选择run_id和原目录，不修改原报告编号。

两次6.7B对比复用相同构建、计划和校准设备信息，对应路径如下；使用最新compare的submission核对关联，而不是按名称猜测：

| 内容 | 仓库相对路径 |
| --- | --- |
| baseline构建 | `outputs/h100/phase6/gpt67-tp8-20261010-112249/base/` |
| 成功校准 | `logs/h100/phase6/gpt67-tp8-20261010-112249/calibration-20261010-153517/` |
| 选择计划 | `outputs/h100/phase6/gpt67-tp8-20261010-112249/selection-plan-20261010-153517.json` |
| 候选构建 | `outputs/h100/phase6/gpt67-tp8-20261010-112249/candidates-20261010-153517/` |
| 前次OOM备份 | `logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/results/round2/window-007-native_taco.failed-20261011-112116/` |

早先无后缀`calibration/`是失败目录，不作为有效校准。不要重命名产物或修改冻结来源路径。构建manifest的源码提交是 `ff38f5b1ff1c80cb831f58c64b95f42b0379879a`，与上传数据的commit不同；模型为32层/H4096/F16384，全局MLP shape `[2048,4096,16384]`、TP8本地K=2048。

2026-10-11本次对两套6.7B对比共1152份计时rank记录进行了CPU核对，包含完成状态、身份、步数及preflight指纹；各自冻结scripts/preflight文件哈希、中位数、配对收益和95%区间均与报告一致。没有在本机重新执行GPU或原路径动态库核验，环境独占性也不能从这些检查推断。

**新对话注意：`base-phase6.md`、`thinking.md`的旧副本若出现“43/72、未完成”，那是过时快照，阅读时以对应会话最新回传report和阶段状态为准。** 本次已直接替换base-phase6的过时进度段落，合并原实验与复测为完整第1–4轮结果；不保留旧进度表。`thinking.md`当前未在磁盘找到，若以后恢复，需要更新其证据范围，不能声称已读到缺失文件。交接/环境记录如果缺失，按[执行端交付清单](instruction-h100.md#交接说明与环境记录放在哪里)补交。

## 本机收到新数据后怎么检查

1. 记录本次数据提交SHA，核对新增文件属于哪个会话；实验代码commit、数据上传commit分别记录。比较 `submission.json`、base/candidate manifest、计划中的模型/TP/种子和GPU身份。
2. 检查阶段状态和实际文件覆盖。默认六策略TP8应有两轮72窗口、576份计时rank记录，另有48份smoke、48份profile rank记录及48份trace；OOM目录、`*.failed-*`备份和仅有process.log的目录不算完成。
3. 按原清单检查文件哈希：submission与preflight的绑定、冻结scripts、preflight证据；然后检查全部rank的 `completed/passed`、零skip/零fallback、计时步数、初始化/RNG/token指纹、实际库哈希和模型字段。检查范围及缺项如实记录。
4. 重算时取每窗口最慢rank的elapsed除以正式step数，按策略、轮次、配对块组织数据；收益用 `100*(1-exp(mean(log(T候选/T基线))))`，按块bootstrap。不要用中位数相除替代配对收益，不混用第一轮和第二轮补空缺。
5. 更新 [base-phase6.md](base-phase6.md) 的事实与数据链接，以及本页的会话状态；机制解释写入 `thinking.md`，分别标明事实、线索和假设。若新数据改变结论，再更新相关论文表述。

`report.py`虽是CPU逻辑，却核验执行端**原绝对路径和实际库文件**。本机没有完整构建时直接运行它可能失败；拉回的 `compare/build/` 符号链接也可能指向 `/workspace/TempName`。不要通过改manifest路径/哈希绕过检查。原服务器严格核验与报告由H100端交付，本机可以按相对路径解析副本并独立复算，但要明确“原始数据统计已复核”和“原运行二进制已核验”的区别。

profile只用来解释机制：解压已有JSON即可，不需要GPU。内核duration可能含等待，多个stream可能重叠，不能把内核累计时间相加当作正式step耗时；跨rank时间戳也不能直接当作同一时钟。详细归类口径应随 `thinking.md` 的恢复核对。

## 当前结论与工作边界

- 默认新模型是6.7B，但历史约0.64B实验仍是重要独立证据。**旧小模型H100 TP8也未建立相对融合v2的稳定收益**；A800同模型TP8约2%，其TP4最佳约4%有不同batch/shape与专项优化，不能混为单纯GPU型号差异。
- 6.7B已通过续跑补齐72窗口，远端报告两候选均未通过性能门槛；最新复测也已完整回传且两候选未达门槛；本机已分别复算两套数据，运行条件仍需审查。OOM和窗口波动不能解释旧小模型同样无稳定收益。性能未达标、测量未完成、正确性失败是不同状态。
- 当前分析线索是H100全量化decode成本下降、少量化的可回收时间减少、到达计划留出命中率较低，以及局部节省未稳定进入整步关键路径。这些是此前分析线索，需对照新复测继续核查，不能宣称已证明单一硬件根因。
- 本机按当前任务做源码/文档维护、静态检查与CPU分析。不要执行拉回的 `env.sh`、watchdog、`resume_timing.py`或带 `--execute` 的GPU入口来验证数据，也不要因为文档列出待验证方向而启动远端测试。若用户安排H100实验，交给执行端按其说明运行并回传。
- 不覆盖历史会话、不修改已有观测/时间/完成状态、不移动实验运行目录来修复路径报错；用户新增数据和工作区修改先核对再处理。

## 实现边界

量化 MLP 使用重新编译为 SM90 的 Phase6 GemmV2 算法；attention 和原生 Flux 参考使用 Hopper GemmV3。该版本没有实现基于 WGMMA/TMA 的 Hopper 原生融合 TACO epilogue，不能将兼容移植称为完成 H100 性能优化。

必须 baseline 为原生 Megatron＋朴素 TACO、原始顺序 Flux＋全远端融合 TACO；参考为原生 Megatron、原生 Hopper Flux。默认候选为远端优先和交替两种基础顺序，可配置另外两种交替顺序。范围仅 MLP `linear_fc2` 前向，BF16 GEMM、本地 BF16、H128/E4M3 远端协议及 BF16 STE 反向沿用现有设计。

构建与结果分别放 `outputs/h100/phase6/`、`logs/h100/phase6/`。必须从 H100 重新采集到达观察并生成选择计划；工具拒绝 A800 计划和未完成构建。候选准备依赖真实校准，不用合成 CPU 测试输入替代 GPU 证据。

后续GPU工作由H100执行端按用户任务和预算安排，按README执行校准、独立候选编译、swizzle/模型smoke/profile及完整六组/八组两轮测量；本页是交接规范，不会自动启动或恢复这些步骤。

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

上述编译记录当时不包含GPU验证；后来旧0.64B以及6.7B的阶段实测见[独立记录](base-phase6.md)。截至本次已接收快照，6.7B已完成真实校准、候选构建、smoke与profile，完整两轮的远端报告已回传但两候选均未达门槛，也没有收敛验证。合成表只用于历史编译诊断，不作为真实校准计划或性能证据。
