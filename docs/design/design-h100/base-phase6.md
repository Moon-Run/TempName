# H100 单节点 Phase6 TP8 实测

模型范围：默认模型为[32层GPT 6.7B](../model-gpt-6.7b.md)。融合基线对齐后的新实验与历史 `phase6-warp-v1` 四轮结果分别记录，旧0.64B结果另列；不同模型和基线版本的收益不混用。

## GPT 6.7B TP8 融合 v2 基线对齐实验（执行中）

新会话：`gpt67-tp8-fusedv2-20261011-233427`（北京时间）。执行源码为 `d786da51e0aeafec1bb2b5d071c5c59e93e6de43`，初始工作区干净，CUTLASS/NCCL 子模块及工作区记录见本会话 `handoff/`。本轮 `taco_fused` 必须标识为 `a800-fused-v2-cta-v1`，使用 A800 冻结融合 v2 的非模板 CTA 解码及旧配置流程；历史结果中的融合列对应 `phase6-warp-v1`，不能作为本轮融合 v2 基线。

使用默认 32 层、H4096/FFN16384、32 heads、词表32768、S2048/mb1/global4、BF16、TP8 配置，不调参。新 base 通过当前 `build.py prepare/compile` 完整构建，编译 CUDA 12.8，GPU 运行 CUDA 12.4。核验原会话候选、计划和校准的配置、源码/库/输入哈希、节点、八卡 UUID/顺序、peer access 及运行软件一致后，复用 `gpt67-tp8-20261010-112249/candidates-20261010-153517/` 和 `selection-plan-20261010-153517.json`，保持其内容不变。复用证据见 [reuse-verification.json](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/handoff/reuse-verification.json)。

本轮计划重新执行六组 smoke/profile、两候选 GPU mapping 和全部 72 个正式窗口（两轮各六个配对块，每窗口10步预热、20步完整 optimizer step）；预检必须在新融合基线的全部八 rank trace 中检出非模板 `taco_decode_ring_kernel`。构建、预检和计时尚在执行，最终统计将在本轮完成及核验后补充；全部新产物分别位于 `outputs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/` 和 `logs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/`。

## GPT 6.7B TP8 历史完整四轮结果（phase6-warp-v1）

更新：2026-10-11，依据已回传的两份完整报告。原实验的两轮记为**第1–2轮**，复测的两轮顺延为**第3–4轮**，在同一组表中展示。共144个正式窗口、1152份计时rank记录、2880个正式optimizer step。各轮沿用原报告的独立配对统计，原始文件中的轮次编号不修改。

模型为32层GPT，H4096、FFN16384、32 heads、词表32768，约6.729B；单节点8×H100，S2048/mb1/global4、BF16、TP8/DP1/PP1/CP1。每step累积4个微批、8192 token；MLP全局shape为 `[2048,4096,16384]`，TP8本地K为2048。window64、量化预算上限1/64。

四轮的模型配置、计划SHA、构建文件清单/哈希、源码manifest、控制脚本哈希及校准设备信息一致。第3–4轮复用既有校准与构建，在新的compare目录重新执行预检和测量；不是算法改版。两份报告均为 `completed=true`，两候选均为 **`accepted=false`**。

### 完整step对照结果

每轮6个位置平衡块，每个策略测6个窗口；每窗口先预热10步，再连续计时20个完整step，包含前向、反向、量化/反量化、通信同步、optimizer更新和清梯度。窗口耗时取8rank最大elapsed除以20，表内ms/step为6个窗口的中位数。

下降百分比采用配对块耗时比的几何平均，95%区间由10000次块bootstrap得到；参考组“耗时增加”为对应配对下降的相反数。不能用表内中位数相除替代，也不把四轮窗口合并重算为一个收益数。

**远端组合 `remote_arrival_selective`**

| 轮次 | ms/step | 对Megatron＋量化下降% [95% CI] | 对融合v2下降% [95% CI] | 比原生Megatron耗时增加% | 比原生Flux耗时增加% |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 554.391 | +33.998 [+32.704, +35.245] | -0.882 [-2.611, +0.545] | -7.839 | -0.192 |
| 2 | 534.545 | +34.579 [+34.230, +34.898] | +3.003 [-0.876, +9.568] | -7.095 | -0.910 |
| 3 | 537.350 | +32.395 [+28.849, +35.225] | -0.521 [-5.065, +2.413] | -7.570 | +2.949 |
| 4 | 534.255 | +34.822 [+34.393, +35.232] | +0.724 [-0.904, +2.149] | -8.579 | -4.259 |

**交替组合 `interleaved_arrival_selective`**

| 轮次 | ms/step | 对Megatron＋量化下降% [95% CI] | 对融合v2下降% [95% CI] | 比原生Megatron耗时增加% | 比原生Flux耗时增加% |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 548.140 | +34.482 [+33.529, +35.535] | -0.143 [-0.916, +0.760] | -8.515 | -0.924 |
| 2 | 536.271 | +32.704 [+29.201, +34.499] | +0.223 [-8.957, +9.567] | -4.432 | +1.930 |
| 3 | 553.728 | +29.876 [+24.340, +34.296] | -4.265 [-10.074, -0.348] | -4.128 | +6.784 |
| 4 | 537.893 | +33.869 [+32.343, +35.126] | -0.727 [-2.223, +0.346] | -7.242 | -2.860 |

六组绝对耗时如下，单位ms/step：

| 策略 | 第1轮 | 第2轮 | 第3轮 | 第4轮 |
| --- | ---: | ---: | ---: | ---: |
| 原生Megatron `native` | 589.601 | 574.219 | 580.638 | 570.325 |
| 原生Hopper Flux `original` | 553.859 | 540.876 | 544.512 | 546.045 |
| Megatron＋量化 `native_taco` | 836.608 | 817.482 | 825.564 | 822.847 |
| 冻结Flux＋融合V2 `taco_fused` | 549.006 | 534.061 | 542.422 | 536.254 |
| 远端组合 `remote_arrival_selective` | 554.391 | 534.545 | 537.350 | 534.255 |
| 交替组合 `interleaved_arrival_selective` | 548.140 | 536.271 | 553.728 | 537.893 |

四轮中，两候选相对Megatron＋朴素量化均有明确下降，但相对融合v2尚无稳定优势。远端组合四轮的区间均跨零；交替组合第3轮为−4.265% [−10.074,−0.348]，该轮显著变慢，其余三轮区间跨零。原协议按每次两轮分别要求相对两个必须baseline下降至少4%、且95%区间下界大于0；两次报告都未通过。统一四轮展示不改变原验收结果。

### 轮次来源与核验

下表将本文轮次映射到原始报告；原文件仍各自使用round1/round2。

| 本文轮次 | 原始对比目录 | 原报告轮次 | 运行标识与状态 |
| --- | --- | --- | --- |
| 第1–2轮 | `compare-20261010-153517` | round1 / round2 | run_id=`e1cc608329cc4bb2ae8e2506f8352ab5`；completed，resumed=true |
| 第3–4轮 | `compare-20261011-114343-repeat` | round1 / round2 | run_id=`b113bd6141374034aa3e050cbf3aab95`；completed，未设置resumed=true |

第1–2轮曾在 `native_taco` 解码时OOM，随后在原会话恢复；失败备份不参与计时统计。第2轮融合v2的一个窗口为659.164ms/step，明显高于该轮其他窗口，按既定协议保留，可影响配对点估计和置信区间。第3–4轮状态未标续跑，但不等于已有全过程独占GPU的证据；运行条件仍需结合环境/占用记录判断。

2026-10-11已在开发端CPU核对两套对比共1152份计时rank记录：身份、完成/通过、零skip/零fallback、预热/计时步数和preflight指纹均通过；冻结scripts/preflight哈希、各轮中位数、全部配对收益及95%区间与原报告一致。没有重新执行GPU或在本机核验原路径动态库。本次仅合并文档表格和轮次编号，不修改任何原始数据或统计。

实现范围仍限MLP `linear_fc2` 前向：量化MLP为SM90编译的GemmV2，attention及原生Flux参考走Hopper V3。固定随机token的短程step与数值检查不构成真实语料收敛证据。旧小模型也未显示相对融合v2的稳定优势，不能仅归因于模型扩大或本次OOM。

### 证据入口

- 第1–2轮：[JSON](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/report.json) · [Markdown](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/report.md) · [状态](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/timing-state.json) · [冻结输入](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/submission.json) · [预检](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/preflight.json) · [原始结果](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/results/)
- 第3–4轮：[JSON](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/report.json) · [Markdown](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/report.md) · [状态](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/timing-state.json) · [冻结输入](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/submission.json) · [预检](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/preflight.json) · [原始结果](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/results/)
- [原OOM日志](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/results/round2/window-007-native_taco.failed-20261011-112116/process.log)

源码与数据索引见 [instruction.md](instruction.md)，测试及数据交付见 [instruction-h100.md](instruction-h100.md)。

## 2026-10-10 完整量化六组两轮首测

北京时间 2026-10-10 00:22:34–00:59:39（UTC 2026-10-09 16:22:34–16:59:39），在 `gxn74` 单节点 8×H100 80GB HBM3 上，按 [A800 TP8 完整量化首测](../design-a800/base-phase6-tp8.md#2026-10-06-完整量化组合适配与首测不优化) 的六组、两轮完整 optimizer-step 口径完成测量，耗时约 37 分钟。**全部 72 个窗口及独立复算通过，两候选均未通过 4% 性能门槛。** 会话目录沿用部署时的 UTC 日期 `tp8-20261009-deployment`。

本轮使用冻结的默认 window64、1/64 预算和本机实测到达计划，不扫描参数。A800 文档顶部后续 window16 优化结果不作为本轮配置或对照。H100 GPU 正确性和性能证据在本记录独立维护，部署指南及接续说明中的旧“尚未 GPU 验证”描述属于此前状态。

### 环境与模型

| 项目 | 本次配置 |
| --- | --- |
| 节点 / GPU | `gxn74`；8×NVIDIA H100 80GB HBM3，每卡 132 SM，SM90；两两 NV18 / peer access |
| 驱动 | 580.178.04 |
| Python / PyTorch | `/opt/conda/envs/flux-h100`；Python 3.10.20，PyTorch 2.6.0+cu124，NumPy 1.26.4 |
| CUDA / 编译器 | CUDA 12.4 用于运行与 Apex，CUDA 12.8.93 用于 Flux 编译；GCC/G++ 11.4.0，CMake 3.30.0 |
| NCCL | 运行时 2.21.5；独立 Release 静态库 2.19.3，SM90 |
| Megatron | `core_v0.12.3`，提交 `3ea68ad6042cc1204386ae9364358f7c4de1bc37` |
| Apex | 提交 `e13873debc4699d39c6861074b9a3b2a02327f92`；`apex_C`、`amp_C`、`fused_layer_norm_cuda` |
| 并行 | TP8 / DP1 / PP1 / CP1，sequence parallel |
| 模型 | 12 层 GPT，H2048、FFN8192、32 heads，BF16 |
| 输入 / batch | S2048 / micro batch 1 / global batch 4；每 step 累积 4 个微批、8192 token；实际 padded vocab 9216 |
| 初始化 | 模型 seed 1234，token seed 4321；六组使用同一参数、RNG 与输入指纹 |

### 六组定义与实现边界

| 策略 | 角色与实现 |
| --- | --- |
| `native` | 原生 Megatron，参考组；不加载 Flux |
| `original` | 原生 Hopper Flux，参考组；attention `linear_proj` 与 MLP `linear_fc2` 使用 Hopper GemmV3 |
| `native_taco` | 必须 baseline：原生 Megatron＋朴素 TACO，保留原生 linear forward，不加载 Flux |
| `taco_fused` | 必须 baseline：原始顺序 Flux＋全远端融合 TACO，使用本次冻结的 H100 库 |
| `remote_arrival_selective` | 远端优先基础排序＋到达优先＋选择性量化 |
| `interleaved_arrival_selective` | 交替基础排序＋到达优先＋选择性量化 |

量化只作用于 MLP `linear_fc2` 前向的远端贡献，使用 FP8 E4M3/H128；本地贡献、GEMM 与 STE 反向保持 BF16。Flux 量化 MLP 是重新编译为 SM90 的 Phase6 GemmV2 路径，attention 仍用 Hopper GemmV3；不是基于 WGMMA/TMA 的 Hopper 原生融合 TACO epilogue。基线及候选库均为本次 H100 独立构建，不复用 A800 二进制。本轮计时不启用采样、profiler 或 CUDA Graph。

### 本机校准与预检

同一节点、同一组 GPU UUID 和设备顺序重新采集两种 BF16 基础排序的到达数据，以 pass 0/1 拟合、pass 2 留出诊断。采用 window64 和 1/64 上限；模型对应 shape `[2048,2048,8192]`，每来源 256 个 tile 中有 224 个远端 tile。

| 计划 | 每来源选中 tile，按 rank 0–7 | 总选中 / 总远端 tile | 实际选择率 | 有效训练 tile | 留出近临界命中率 | 采样中位扰动 |
| --- | --- | --- | ---: | --- | ---: | ---: |
| 远端优先 | `[0,3,0,3,0,3,0,3]` | 12 / 1792 | 0.6696% | 254 / 256 | 33.33% | 21.52% |
| 交替 | `[3,3,3,3,3,3,3,3]` | 24 / 1792 | 1.3393% | 125 / 256 | 41.67% | 10.21% |

掩码是本机校准的实际产物，不强行补齐未选中的来源。交替计划有效训练 tile 较少，且两种计划留出命中率均有限；这些指标仅为校准诊断，不代表重排量化后的收益。没有进行重排或量化后的联合重新校准。

部署阶段依赖检查、15 项 CPU 协议测试和八 rank NCCL all-reduce / all-gather / reduce-scatter / all-to-all 已通过。四组基线/采样库及两组真实计划候选构建完成；六组完整模型 smoke 和六组独立 profile 共 96 份 rank 记录全部完成并通过，零 skip、零 fallback。48 份 GPU trace 核验了对应 Hopper V3、SM90 编译的 V2 及 TACO decode 内核；两候选实际 GPU swizzle 均匹配冻结计划。每层量化 mask、逻辑通信字节、初始参数/RNG/token 指纹、实际加载库路径和哈希均通过预检。

上述是 H100 整模型前向与执行预检，不将 A800 的独立 codec、连续偏斜调用或 STE 专项报告算成本次 H100 验证。共 864 条模块前向检查，其中 384 条为量化前向双参考预算检查。整模型量化前向最大相对 BF16 L2 误差为 2.5124%，相对 codec 参考最大为 0.0198%，均低于既定 5% / 1% 预算。

| 策略 | 最大相对 BF16 L2 误差 | 最大相对 codec L2 误差 |
| --- | ---: | ---: |
| `native_taco` | 2.5118% | 0.0000% |
| `taco_fused` | 2.5124% | 0.0198% |
| `remote_arrival_selective` | 0.3103% | 0.0013% |
| `interleaved_arrival_selective` | 0.7703% | 0.0107% |

### 统计口径

每轮 6 个位置平衡块，每块覆盖六组，第二轮反转块与组顺序。两轮共 72 个独立八 rank 窗口；每窗口 10 步预热＋20 步完整 optimizer step，总计 720 步预热、1440 步正式 step、576 份计时 rank 记录。窗口耗时取 8 rank 最大 elapsed 除以 20；完整 step 包含前向、反向、梯度处理和 optimizer 更新。

ms/step 为每轮同策略 6 个窗口的中位数。配对下降百分比为 `100 × (1 − exp(mean(log(T候选/T基线))))`，正值表示更快；95% 区间以配对块 bootstrap 10000 次得到，不能用表中中位数相除替代。参考组“耗时增加%”为上述下降值的相反数，负值表示候选更快。预检/profile 不混入计时统计。

验收沿用既定 4% 标准：每个候选在两轮中，分别对 `native_taco` 和 `taco_fused` 的点估计均至少下降 4%，且各 95% 区间下界大于零。原生 Megatron 与原生 Flux 只作参考。

### 单节点TP8完整量化六组对照表

下表按 A800 文档格式汇总“远端组合”`remote_arrival_selective`。本次直接在节点 `gxn74` 执行，“配置 / 作业”列以节点名标识实验。所有百分比从同块窗口计算，不能用中位数相除。

| 配置 / 作业 | 轮次 | 远端组合 ms/step | 对Megatron＋量化下降% [95% CI] | 对融合v2下降% [95% CI] | 比原生Megatron耗时增加% | 比原生Flux耗时增加% |
| :------ | :- | -----------: | :----------------------- | :---------------- | ---------------: | -----------: |
| S2048/mb1/gb4 / gxn74 | 1 | 205.601 | +33.636 [+33.255, +34.004] | +0.532 [-0.271, +1.583] | -6.781 | -1.449 |
| S2048/mb1/gb4 / gxn74 | 2 | 207.051 | +33.441 [+33.142, +33.914] | -0.368 [-0.934, +0.084] | -6.119 | -1.235 |

交替组合 `interleaved_arrival_selective` 使用相同口径，补充如下：

| 配置 / 作业 | 轮次 | 交替组合 ms/step | 对Megatron＋量化下降% [95% CI] | 对融合v2下降% [95% CI] | 比原生Megatron耗时增加% | 比原生Flux耗时增加% |
| :------ | :- | -----------: | :----------------------- | :---------------- | ---------------: | -----------: |
| S2048/mb1/gb4 / gxn74 | 1 | 205.333 | +33.605 [+33.494, +33.720] | +0.485 [-0.237, +1.495] | -6.737 | -1.402 |
| S2048/mb1/gb4 / gxn74 | 2 | 206.494 | +33.554 [+33.096, +34.097] | -0.197 [-0.761, +0.299] | -6.279 | -1.403 |

六组每轮的绝对耗时如下，单位 ms/step：

| 策略 | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| 原生 Megatron `native` | 220.847 | 219.944 |
| 原生 Hopper Flux `original` | 208.521 | 208.876 |
| Megatron＋量化 `native_taco` | 309.753 | 310.567 |
| 冻结 Flux＋融合 V2 `taco_fused` | 206.101 | 206.031 |
| 远端组合 `remote_arrival_selective` | 205.601 | 207.051 |
| 交替组合 `interleaved_arrival_selective` | 205.333 | 206.494 |

**H100 TP8 环境可以执行本轮实验，完整测量与数值检查已完成；两候选尚未建立相对冻结融合 V2 的稳定整步收益。** 对 Megatron＋量化的两轮下降约 33.4%–33.6%，区间均为正；对冻结融合 V2 则第一轮略快、第二轮略慢，四个区间均跨零，点估计也均不足 4%。因此两候选 `accepted=false`。本轮止于默认配置首测，没有继续优化，也不据此计算 H100 相对 A800 的硬件加速比。

计时阶段 576 份 rank 记录全部完成并通过，零 skip、零 fallback；模型、TP 拓扑、GPU UUID、运行库哈希和参数/RNG/token 指纹与预检一致。每策略的 12 个窗口中，同 rank 的初始 loss、20 步正式 loss 和梯度范数记录完全一致，最大差异均为零。跨全部窗口及 rank 的峰值 allocated / reserved 显存为 2.185 / 2.301 GiB；实验结束后八卡均为 0 MiB、0% 利用率。

本次输入是固定 seed 的随机 token，测量的是短程完整训练 step。前向误差、有限 loss/梯度和同策略轨迹重复通过不等于真实数据训练或收敛验证。量化 MLP 的 SM90 编译 V2 实现边界仍如上所述。

### 原始产物与复算

构建与计划：`outputs/h100/phase6/tp8-20261009-deployment/`；校准与本轮实验：`logs/h100/phase6/tp8-20261009-deployment/`。

- [冻结选择计划](../../../outputs/h100/phase6/tp8-20261009-deployment/selection-plan.json)
- [冻结提交与输入清单](../../../logs/h100/phase6/tp8-20261009-deployment/compare/submission.json)
- [完整预检](../../../logs/h100/phase6/tp8-20261009-deployment/compare/preflight.json)
- [完整配对报告](../../../logs/h100/phase6/tp8-20261009-deployment/compare/report.md) / [未舍入统计](../../../logs/h100/phase6/tp8-20261009-deployment/compare/report.json)
- [独立复算与数值核验](../../../logs/h100/phase6/tp8-20261009-deployment/compare/verification.json)
- [全部窗口及八 rank 原始耗时汇总](../../../logs/h100/phase6/tp8-20261009-deployment/compare/window-statistics.json)
- [计时原始文件哈希](../../../logs/h100/phase6/tp8-20261009-deployment/compare/timing-artifact-hashes.json)
- [计时完成状态](../../../logs/h100/phase6/tp8-20261009-deployment/compare/timing-state.json)
- [各窗口原始记录](../../../logs/h100/phase6/tp8-20261009-deployment/compare/results/)

首次执行命令如下；控制器拒绝覆盖已开始的阶段，不能在原目录重跑计时。

```bash
cd /workspace/TempName
source outputs/h100/phase6/tp8-20261009-deployment/env.sh
python tools/h100/phase6/run.py "$H100_LOG/compare" \
  --stage timing --python /opt/conda/envs/flux-h100/bin/python \
  --cuda-home "$CUDA124" --execute
```

完成后的 CPU 复算命令如下，不会重新启动 GPU 测量：

```bash
source outputs/h100/phase6/tp8-20261009-deployment/env.sh
python tools/h100/phase6/report.py "$H100_LOG/compare"
python "$H100_BUILD/verify-timing.py"
```
