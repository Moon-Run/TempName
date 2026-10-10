# H100 单节点 Phase6 TP8 实测

模型范围：下文是12层、约0.64B的历史实测，数值和原始会话均保留。2026-10-10后的默认模型已改为[32层GPT 6.7B](../model-gpt-6.7b.md)，新模型尚未测试，不能沿用本页性能、显存或到达表作为其结论。

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
