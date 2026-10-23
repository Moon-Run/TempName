# H100 新机制 baseline 汇总

更新：2026-10-23。按 [base-baseline.md](../base-baseline.md) 的指标整理，列顺序为 **CR2、R、C、B、A、Flux+改进量化、Flux＋CUDA量化、Megatron＋CUDA量化、Megatron+朴素量化、原生Megatron、原始Flux**。R 指 MA2-R；Flux+改进量化对应 [base-phase6.md](../base-phase6.md) 新监控体系 a–h 轮的“新融合 v2”。最后三列取自 [Attention/MLP 作用范围文档](../base-phase6-mlp-attention.md) 对应的原始数据，Megatron+朴素量化采用 Attention＋MLP 版本。

| **组别** | **策略概括** |
| --- | --- |
| **CR2** | **C＋打包来源预取＋原生提交缓存**：缓存固定 GEMM 执行信息，省去重复算子查找和中间视图，直接返回独立输出。 |
| **R** | **交替排序＋打包来源预取＋执行优化**：实例持有配置，返回独立输出，省去逐次配置与输出 clone。 |
| **C** | **交替排序＋到达优先＋选择性量化＋双工作区**：实例持有配置与独立输出。 |
| **B** | **交替排序＋到达优先** |
| **A** | **交替排序** |
| **Flux+改进量化** | **Flux 融合 v2：GEMM 内融合编码＋非模板 CTA 解码归约** |
| **Flux＋CUDA量化** | **原生 Flux GEMM＋独立 TACO CUDA 编解码＋NCCL All-to-All**。 |
| **Megatron＋CUDA量化** | **原生 Megatron GEMM＋独立 TACO CUDA 编解码＋NCCL All-to-All**。 |
| **Megatron+朴素量化** | **原生 Megatron GEMM＋PyTorch 张量版 TACO 编解码＋NCCL All-to-All**，量化范围为 Attention＋MLP。 |
| **原生Megatron** | **原生 Megatron BF16 训练路径**，Attention/MLP 不量化。 |
| **原始Flux** | **原生 Hopper Flux BF16 GEMM＋Reduce-Scatter**，Attention/MLP 不量化。 |

“到达优先”使用冻结的原到达表；C/CR2 使用原 1/64 预算上限的选择性量化 mask，A/B/R、两组 CUDA 量化基线及 Megatron+朴素量化使用全远端量化。Flux+改进量化在 MLP 投影中使用全远端量化，未将量化扩展到 Attention。原生Megatron、原始Flux 均不量化；所有组的本地贡献保持 BF16。

**本表是已有结果汇总，不是一次十一组同场实验。** CR2/R/C 使用最新 CR2 批次；B/A 和两组 CUDA 量化基线使用原 baseline 批次。Flux+改进量化、原生Megatron、原始Flux 使用新监控体系 a–h；Megatron+朴素量化使用作用范围实验 a–d。这四个历史参考列每窗口计时 **20 步**，其他列为 **50 步**；量化作用范围与是否量化见策略表。

| **指标** | **CR2** | **R** | **C** | **B** | **A** | **Flux+改进量化** | **Flux＋CUDA量化** | **Megatron＋CUDA量化** | **Megatron+朴素量化** | **原生Megatron** | **原始Flux** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 平均耗时（ms/step） | **511.297** | 515.716 | 516.784 | 536.041 | 536.692 | 647.284 | 610.632 | 632.905 | 1052.781 | 652.615 | 593.615 |
| 全部窗口中位耗时（ms/step） | **510.450** | 515.154 | 516.811 | 535.755 | 536.618 | 548.243 | 610.428 | 632.682 | 1052.199 | 575.367 | 542.205 |
| CR2 相对该列耗时下降（中位数比） | +0.000% | +0.913% | +1.231% | +4.723% | +4.877% | +6.894% | +16.378% | +19.320% | +51.487% | +11.283% | +5.857% |
| 平均训练 loss | 3.793364 | 3.821014 | 3.793364 | 3.821014 | 3.821014 | 7.332271† | 3.821911 | 3.821911 | 6.720901† | 7.339843† | 7.339843† |
| 最终训练 loss 的平均 | 3.057346e-04 | 3.261156e-05 | 3.057346e-04 | 3.261156e-05 | 3.261156e-05 | 4.100240e+00† | 1.916869e-02 | 1.916869e-02 | 1.926777e+00† | 3.302796e+00† | 3.302796e+00† |
| 吞吐（tokens/s） | **16022.00** | 15884.70 | 15851.88 | 15282.41 | 15263.87 | 12655.96 | 13415.61 | 12943.49 | 7781.29 | 12552.57 | 13800.19 |
| 数据来源 | CR2 批次 | CR2 批次 | CR2 批次 | 原 baseline 批次 | 原 baseline 批次 | 新监控体系 a–h | 原 baseline 批次 | 原 baseline 批次 | 作用范围实验 a–d | 新监控体系 a–h | 新监控体系 a–h |

† Flux+改进量化、Megatron+朴素量化、原生Megatron、原始Flux 的平均 loss 使用每窗口 20 个正式步，最终 loss 为第 20 步；其他列使用 50 步，不能直接用末步 loss 判断训练质量优劣。所有列保留对应范围内的全部窗口；Flux+改进量化受慢窗口影响，均值 **647.284 ms** 明显高于中位数 **548.243 ms**，最大窗口为 **1829.941 ms**。原生Megatron、原始Flux 的均值也包含历史慢窗口。

[作用范围文档](../base-phase6-mlp-attention.md#配对耗时下降) 中上述新增三组的“平均耗时”分别为 **1052.654 / 575.986 / 542.284 ms**，其口径是**八个轮次中位数的平均**。本表为保持各列一致，重新使用全部原始窗口计算算术均值，得到 **1052.781 / 652.615 / 593.615 ms**；二者不是同一统计量，中位数另列在第二行。

数据来源：

- **CR2/R/C：** `ma2-cr2-2x-50steps-20261023-153237-090898`，合并 compare1、compare2，各组保留 16 个正式窗口。见[原始报告](../../../../logs/h100/phase6/ma2-cr2-2x-50steps-20261023-153237-090898/fullstep/report.json)及[实测总结](ma-2-ablation.md#11-cr2-八卡实测结果与离线复核2026-10-23)。C 与原 baseline C 为相同实现，但这里使用新批次实测均值 516.784 ms，不混入旧批次的 517.734 ms。
- **Flux+改进量化：** 使用 [base-phase6.md 新监控体系 a–h](../base-phase6.md#gpt-67b-tp8-融合-v2-基线对齐实验) 的 `taco_fused`，实现标识 `a800-fused-v2-cta-v1`。来源为 [a/b](../../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat1/compare/report.json)、[c/d](../../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat2/compare/report.json)、[e/f](../../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat1/compare/report.json)、[g/h](../../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat2/compare/report.json) 四套会话，每轮 6 个窗口，合计 **48 个不重复窗口、384 份 rank 记录、960 个正式 step**。原文两种候选共用同一列基线，本表只计一次；不加入原第 1–4 轮，不剔除慢窗口，也不将八个轮次的中位数当作窗口均值。已由原始 rank 记录重新计算，并核对 a–h 各轮中位数与报告一致，见[汇总统计](../../../../outputs/h100/phase6/fused-v2-ah-summary-20261023/summary.json)及[计算脚本](../../../../outputs/h100/phase6/fused-v2-ah-summary-20261023/summarize.py)。
- **B/A/Flux＋CUDA量化/Megatron＋CUDA量化：** `baseline-taco-cuda-3x-50steps-20261021-162929-989672`，合并 compare1、compare2、compare3，各组保留 30 个正式窗口。见[原 baseline 汇总](../base-baseline.md#3-完成范围与全样本结果)及[原始审计统计](../../../../outputs/h100/phase6/baseline-taco-cuda-results-audit-20261021/audit.json)。
- **Megatron+朴素量化：** 取 [base-phase6-mlp-attention.md](../base-phase6-mlp-attention.md) 的 `native_taco_attn_mlp`，即 Attention＋MLP 同时使用朴素张量 codec。来源为 `TTention-MLP-gpt67-tp8-20261015-012238-repeat1` 至 `repeat4`：报告 [a](../../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat1/compare/report.json)、[b](../../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat2/compare/report.json)、[c](../../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat3/compare/report.json)、[d](../../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat4/compare/report.json)。每次两轮、每轮四个该策略窗口，合计 **32 窗口、256 份 rank 记录、640 正式步**；不混入 Attention-only 或 MLP-only 的朴素量化结果。
- **原生Megatron、原始Flux：** 分别为上述作用范围文档引用的新监控体系 a–h 中 `native`、`original`；会话与 Flux+改进量化相同，各 **48 窗口、384 份 rank 记录、960 正式步**。三组新增列均核对原始 rank 记录的完成状态、10/20 步协议、实际路由、有限 loss/gradient 和逐 rank loss 一致性；各轮中位数与报告吻合。见[重算统计](../../../../outputs/h100/phase6/historical-baselines-summary-20261023/summary.json)及[计算脚本](../../../../outputs/h100/phase6/historical-baselines-summary-20261023/summarize.py)。

各列都测完整 optimizer step，包含前后向、通信、参数更新与清梯度。每窗口预热 10 步；四个标 † 的历史参考列正式计时 20 步，其他列 50 步。步时取八 rank 最大墙钟除以各自正式步数；平均耗时为全部窗口的算术均值，中位数直接从全部窗口计算。每步 8192 token，吞吐为 `8192000 / 平均ms`。平均 loss 先取每窗口 rank0 的全部正式值平均，再跨窗口平均；最终 loss 取各窗口最后一个正式步，不含预热、不重复计八个 TP rank。

“CR2 相对该列耗时下降”为 `100 × (1 − CR2中位ms / 该列中位ms)`，使用第二行对应的未舍入窗口中位数计算。**除同场 C/R 外，相对其余列的百分比仅为跨批次描述性差值，没有十一组同场配对置信区间。历史参考列还存在量化范围、是否量化、20/50 步协议及慢窗口等差异，不能将这些百分比解释为单纯量化或排序的净收益。** 对同场 C/R，该中位数比也不替代逐轮完整块配对统计：CR2/C 两轮为 +1.027% / +1.097%，CR2/R 为 +0.630% / +1.083%，详见实测总结。

各组量化范围与训练轨迹并非全部等价，表中 loss 仅描述固定合成输入的短程结果。两组 CUDA 量化基线为作者 TACO CUDA codec＋NCCL 通信的移植实现，不包含完整 COCCL 调度与重叠；相对它们的差值不能表述为相对论文完整 TACO 系统的收益。
