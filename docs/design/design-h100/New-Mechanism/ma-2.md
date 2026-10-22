# MA2：连续系数打包预取，目标完整 step 提升 1%

更新：2026-10-22。MA2 是 [MA1](ma-1.md) 的后续优化，针对旧预取实现的细粒度读取和寄存器开销。目标是相对原始交替 A 达到至少 1% 的完整 step 点估计收益，并在两次独立比较中确认优于旧 MA1 P。

**最新八卡批次已完成两次比较、72 个训练窗口和 3600 个正式 optimizer step，正确性与执行路径检查均通过，但性能目标未通过。** 两次 V/A 配对收益分别为 **−0.510%［95% CI：−5.949%, +3.765%］**、**−1.844%［−6.584%, +0.834%］**；V/P 也未得到稳定正收益。这是正常完成后的性能验收未通过，没有复现此前 MA1 的 profile 解析失败。完整结果与限制见第 7 节。

完整训练结果如下，正值表示加速，方括号为 95% CI。A 为原交替实现，P 为旧 MA1 预取，V 为 MA2 打包预取。

| 比较 | A / P / V 平均步时（ms） | V/A 收益 % | V/P 收益 % | P/A 收益 % |
| --- | --- | --- | --- | --- |
| compare1 | 587.903 / 566.581 / 589.044 | −0.510 [−5.949, +3.765] | −3.926 [−10.087, +0.575] | +3.287 [−0.716, +9.547] |
| compare2 | 536.483 / 556.064 / 547.882 | −1.844 [−6.584, +0.834] | +1.178 [−6.365, +9.033] | −3.058 [−9.046, −0.040] |

## 1. 为什么继续优化数据读取

MA1 最新完整训练中，P/A 为 +0.333%、P/W 为 +0.400%，但独立 MLP 算子的 P/W 为 −1.282%，训练 profile 中 P 的 decoder 也略慢于 W。这说明应保留完整训练中的小幅正结果，同时减少预取本身的指令和寄存器负担，不能简单增加预取深度。

旧 MA1 P 中，每个 lane 持有 `lane+32*j` 的四个系数。每个来源需四条 8-bit payload load 和两条 scale load；当前与下一来源各保留四个原始字节寄存器及两个 scale。此次改为每 lane 持有 `4*lane+j` 四个连续系数，利用一次对齐的 32-bit load 读取四个 FP8 字节，保留原来的下一来源预取。

| 接收端数据操作 | 原 MA1 P | MA2 V |
| --- | --- | --- |
| 每 lane 的四个系数 | `lane, lane+32, lane+64, lane+96` | `4*lane, 4*lane+1, 4*lane+2, 4*lane+3` |
| 每来源 payload 读取 | 4 条 `ld.global.u8` | 1 条 `ld.global.u32` |
| 两个 scale 读取 | 2 条 FP32 load | 相同的 2 条 FP32 load |
| 四个 FP8 转为 half | 四次单值转换 | 两次 FP8x2 转换，再分别转 FP32 |
| 本地 BF16 读取 / 输出写回 | 四个分散元素 | 对齐的 64-bit load / store |
| 累加算术 | 逐来源、两处 BF16 舍入 | 保持相同，不改成 FP32 来源求和 |

“打包”只改变接收端寄存器表示和访存宽度，**不改变网络 packet 布局、量化范围或通信字节数**。各源仍发送原来的 H128/E4M3 数据及两个 FP32 scale，来源顺序仍为 `rank+1,…,rank`，本地 BF16 最后加入。

## 2. Hadamard 阶段和同步保持

连续系数布局使 H128 的距离 1、2 两级在单线程的四个寄存器中执行；距离 4、8、16、32、64 则使用 warp shuffle，其 lane 距离为物理距离除以 4。每个物理系数仍严格经历 `1→2→4→8→16→32→64` 的原加减顺序。

这里不是将旧布局的最后两级直接搬到开头，而是同时改变系数归属，使每个物理元素的计算图不变。缩放乘法、归一化、非有限过滤、除以 adaptive scale、解码后的 BF16 舍入及每次来源相加后的 BF16 舍入均保持。原始来源数据只在发布屏障后读取，复用屏障继续保护工作区。

连续布局还带来一个需要显式处理的编译器问题：原来乘法与第一级加减之间隔着 shuffle，现在第一级在寄存器内执行，nvcc 默认可能将 `q*x + q*y` 的一项乘法收缩进 FMA，改变原有的乘法舍入。V 对四个量化缩放乘法使用 `__fmul_rn`，保留先乘法舍入、再加减的顺序；没有全局关闭 FMA，也没有启用 fast math。回归用例包含确实产生一位 FP32 差异的反例，最终 SASS 中循环和最后远端来源均保留四条独立 FMUL。

向量读取的地址满足所需自然对齐：payload 的 group 偏移为 `128*group` 字节，lane 偏移为 `4*lane` 字节；本地及输出的四个 BF16 对应 8-byte 对齐地址。自然对齐要求见 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands)，FP8x2 转换语义见 [CUDA 12.8 Math API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html)。

实现见 [ma2_decode.cuh](../../../../tools/h100/phase6/ma2_decode.cuh) 与 [ma2_sources.py](../../../../tools/h100/phase6/ma2_sources.py)。使用精确形状 `TP8, M2048, N4096`，Attention/MLP 本地 K 为 512/2048。GEMM、交替映射、encoder、两次屏障、单工作区、逐次配置/复位及输出 clone 保持相同。

## 3. 同场 A/P/V 对照

| 组 | 本次实现 | 用途 |
| --- | --- | --- |
| A | 原融合 v2 scalar decoder，基础交替、全远端量化 | 原始交替参照 |
| P | 原 MA1 的预取 runtime/header，生成源码逐字保持 | 复测上一版，不使用历史计时替代 |
| V | 新连续四系数布局、打包读取/转换/写回、来源预取 | 本次优化候选 |

三组均不使用到达表或选择性量化。本轮不包含旧 MA1 W，也不包含 baseline C；V/P 检查新增打包优化的净效果，P/A 记录旧 MA1 是否在同场重现。若 V/A 提升但 V/P 没有正增量，不能认定新优化优于上一版。

[ma2_build.py](../../../../tools/h100/phase6/ma2_build.py) 为三组建立独立目录并冻结源码。P 直接复用原 `ma1_sources.sources(..., 'P')`，V 只替换接收 runtime 并新增自己的 decoder 头文件。旧 `ma1*.py`、`ma1_decode.cuh` 与已有结果保留，原实验仍可复现。

已将 MA1 成功批次 `ma1-prefetch-1x-50steps-20261021-233916-307862` 与本批的 `fullstep/plan.json` 构建身份逐文件对比：去除不同批次目录前缀后，**A 的 351 个、P 的 352 个登记源码文件哈希分别全部一致**。两批模型配置、预热/正式步数、共享环境字段和 Megatron commit 也一致。因此这里的 A/P 与 MA1 是相同实现；各批次重新构建、重新计时，不代表跨批次动态库文件或测得耗时必须相同。

## 4. 两次独立比较与 1% 判定

本次“两轮”指 **compare1、compare2 两次独立完整训练比较**；每次内部仍有顺序平衡的前后两个半程，不把这两个半程当作两次独立实验。

**compare1 与 compare2 之间没有切换机制或调参。** 两次共用本批同一套 A/P/V 构建，模型参数、模型 seed1234、token_seed4321、预热/计时步数及 12 块的 A/P/V 执行顺序均相同。变化的是 trial/run ID、结果目录和实际执行时段；compare1 完成后再执行 compare2，每个窗口均为新进程并从相同初始化开始，不承接上一轮训练权重。这是相同配置的运行重复，不是不同随机种子或不同训练数据的对照。

| 项目 | 每次比较 | 两次合计 |
| --- | ---: | ---: |
| 完整 A/P/V 块 | 12 | 24 |
| 正式训练窗口 | 36 | **72** |
| 正式 rank 记录 | 288 | 576 |
| 正式 optimizer step | 1800 | **3600** |
| 正式窗口预热步 | 360 | 720 |

每窗口独立初始化，10 步预热＋50 步正式；两次比较有不同 run ID。模型继续使用 GPT6.7B、32 层、H4096/FFN16384、32 heads、S2048、TP8、BF16、micro batch1/global batch4，每步 8192 token。`NCCL_NVLS_ENABLE=0`、`CUDA_DEVICE_MAX_CONNECTIONS=1`、初始化/RNG/token、全远端量化和反向路径均固定。

先进行算子预检、36 个算子计时窗口及 48 份算子 profile；完整训练共用一套模型 smoke 预检和独立采集的 24 份完整 step profile。所有正式窗口采用新进程，旧批次计时不混入。算子性能是否为正不作为跳过完整训练的理由，但正确性或执行证据失败会停止并保留现场。

性能指标仍为八 rank 最大窗口墙钟除以 50；按完整三组块配对 bootstrap 10000 次，seed20261021，报告全样本、负收益和区间，不选择最好一轮。

两次比较**各自**须同时满足：

1. **V/A 点估计 ≥1.0%，且其 95% CI 下界 >0。** 这是本次 1% 点估计目标。
2. **V/P 点估计及 95% CI 下界均 >0。** 确认新增优化比旧 MA1 P 更快。
3. 算子逐位正确性、模型逐 rank loss/gradient 一致性、实际调用和 profile 路径检查全部通过。

满足后，批次 `target_validated=true`。报告同时单列 **V/A 的 95% CI 下界是否也达到 1%**：`strict_one_percent_confidence_validated`。点估计达到 1% 不等于已用置信区间证明收益至少 1%；两种结论不能混用。P/A 是复测控制，只记录，不强求上一版这次必须通过。

两次的结果分别报告；只一轮达到目标也不能将整个批次写成通过。固定合成输入上的短程训练不代表真实语料长期收敛。

## 5. 一行启动

同步本次新增 `ma2*.py`、`ma2_decode.cuh`、`detach_ma2.py` 和已有工具依赖后，在原服务器 `/workspace/TempName` 执行：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_ma2.py --execute
```

默认重新准备和编译 A/P/V，CUDA12.8 编译、CUDA12.4 运行、jobs2、相邻 `../Megatron-LM`；双 fork 后台托管。容器和主机需持续运行。输出为 `logs/h100/phase6/ma2-packed-prefetch-2x-50steps-<北京时间戳>/`，其中：

```text
batch-state.json                    完成状态与两个 1% 判定字段
handoff/container-supervisor.log   后台总日志
build/                            A/P/V 隔离构建和 manifest
operator/                         算子逐位预检、计时、实际 mapping 和 profile
fullstep/                         两次比较的 plan、原始 rank 记录与 report.json/report.md
```

去掉 `--execute` 只预览，无目录创建、编译、fork 或 GPU 查询。`--foreground` 可前台诊断，`--python`、`--megatron`、`--cuda-home`、`--build-cuda-home`、`--nccl-root` 可指定环境。`--ma2-build` 只接受已完成且通过验证的本轮 A/P/V 构建，不能传入旧 MA1 的 A/W/P 构建。

**再次复测两轮时，直接重新执行上述 `detach_ma2.py --execute` 命令即可。** 默认按新的北京时间戳创建独立批次目录，重新构建 A/P/V，再完成两次比较，共新增 72 窗口、3600 正式步；模型、机制、种子和验收条件保持不变。无需修改脚本，也不要将 `--out` 指向已完成的旧批次。新结果单独报告，不覆盖或自动合并第 7 节已有结果；启动前让上一批及其他训练任务结束，避免复测任务相互竞争资源。

回传整个批次，保留冻结脚本、plan、rank JSON、mapping、trace、manifest 和状态。离线报告入口：

```bash
python3 tools/h100/phase6/ma2_bench.py report /path/to/batch/operator --offline
python3 tools/h100/phase6/ma2_bench.py profile-report /path/to/batch/operator --offline
python3 tools/h100/phase6/ma2_fullstep.py report /path/to/batch/fullstep --offline
```

## 6. 上机前本地验证与限制

2026-10-22 已完成以下本地检查：

| 检查 | 结果与证据 |
| --- | --- |
| 构建身份、数值模拟、报告、profile 解析、两轮编排与启动器 | **70 项 CPU 测试通过**；[测试登记](../../../../outputs/h100/phase6/ma2-validation-20261022/cpu-validation.json)、[完整日志](../../../../outputs/h100/phase6/ma2-validation-20261022/cpu-tests.log) |
| A/P/V 实际 runtime 编译与两种 K 的交替 mapping | 三组均通过 host mapping 和 CUDA 12.8 **SM90 目标文件编译**，保存 PTX/SASS；[编译登记](../../../../outputs/h100/phase6/ma2-validation-20261022/runtime-check/validation.json) |
| P/V 同一编译单元、相同选项的 decoder 比较 | 最终 P **40**、V **34** 个寄存器/线程；两者 stack/spill 均为 0；[指令与数值审计](../../../../outputs/h100/phase6/ma2-validation-20261022/decoder-check/validation.json) |
| FP8x2 与单值转换的主机实现 | 穷举 **65,536 对字节**（其中 64,516 对均为有限值），half 位模式全部一致；[转换检查](../../../../outputs/h100/phase6/ma2-validation-20261022/decoder-check/fp8x2-host.json) |
| 一行命令预览 | 正确生成两次比较、72 窗口、3600 正式步的计划，未创建目标目录或启动训练；[预览记录](../../../../outputs/h100/phase6/ma2-validation-20261022/launch-preview.txt) |

SASS 确认每来源读取指令从 6 条降到 3 条、FP8 转换从 4 条降到 2 条，本地读取和最终写回分别从 4 条降到 1 条。每来源仍有 20 条 shuffle；下一来源的读取确实发射在当前来源首条 shuffle 之前，并在后续迭代消费。这些是编译产物中的指令观察，不是运行时计数或耗时。

本节记录的是上机前本地检查，当时没有执行 GPU kernel，也没有完成整套 Flux 动态库链接、八卡逐位正确性或完整训练性能验证。后续服务器批次已完成运行，结果见第 7 节。静态指令或寄存器减少不等于已经改善实际 occupancy、访存重叠或完整 step，寄存器分配粒度也可能使 occupancy 不变；不能将 MA1 的 0.3%–0.4% 与一个推测收益相加。

## 7. 两轮八卡实测总结（2026-10-22）

### 7.1 批次完成情况与证据

本批目录为 `logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/`。依据[批次状态](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/batch-state.json)与[后台总日志](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/handoff/container-supervisor.log)，构建、算子计时、算子 profile、完整训练计时及完整训练 profile 均执行完成，最终 `status=completed`、`stage=done`。

| 检查项 | 本批结果 |
| --- | --- |
| 正式训练工作量 | 两次比较，各 12 个 A/P/V 块；合计 72 窗口、576 份 rank 计时记录、3600 个正式 optimizer step |
| 算子正确性 | `correctness_passed=true`、`bitwise_equal=true` |
| 模型正确性 | `correctness_passed=true`、`identical_trajectories=true`；配对 loss/gradient 轨迹一致 |
| 算子执行路径 | `operator_execution_path_validated=true` |
| 完整训练执行路径 | `fullstep_execution_path_validated=true` |
| 1% 点估计目标 | 两次均未通过；批次 `one_percent_point_target_validated=false` |
| 新增打包机制收益 | 两次均未通过；批次 `packed_benefit_validated=false` |
| 最终性能验收 | `target_validated=false`、`acceptance_status=not_validated` |

已从回传原始 rank 记录离线复算完整训练报告，结果与保存的 `fullstep/report.json` 完全一致。**`completed` 和性能 `target_validated` 是两个不同状态，本批不存在因启动退出而少跑窗口的问题。** 此前 MA1 的 CPU/GPU 镜像 annotation 解析故障没有在本批复现；没有证据将本批性能波动归因于进程绑定，因此未以“修复启动器”为由改动脚本或放宽门槛。

### 7.2 完整训练：未达到 1% 目标

完整训练汇总表已置于页首，数据来自[完整训练报告](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/fullstep/report.md)及[原始统计 JSON](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/fullstep/report.json)。正值表示候选更快，负值表示更慢；方括号为配对 bootstrap 95% CI。

平均步时为窗口步时的算术平均；收益使用配对块耗时比的对数均值再转换为百分比，不直接用表中算术均值相除。两次 V/A 点估计均为负，V/A 与 V/P 的四个区间均跨零。compare2 的 V/P 虽有 +1.178% 点估计，但区间很宽且跨零，也不是相对原交替 A 的 1% 收益，不能据此判定通过。

旧 MA1 P 在本批同场对照中也未复现稳定正收益：compare1 的 P/A 区间跨零，compare2 的 P/A 为负且区间低于零。这不改写上一批 MA1 的测量结果，但说明其 +0.3%–0.4% 尚不能作为跨批次稳定收益。两批计时不合并，不选择其中最好的一轮。

### 7.3 波动与半程结果

compare1 前半程 A/P/V 平均步时为 641.272/599.859/642.128 ms，后半程为 534.533/533.302/535.960 ms。全部窗口中，`compare1/block02-P` 达到 787.549 ms/step，`compare1/block01-A` 达到 776.831 ms/step；compare2 也有 `block05-P` 的 775.848 ms/step，而该轮 A/P/V 窗口中位数分别为 534.990/537.119/534.964 ms。不同臂均出现明显较慢的窗口，与宽置信区间相对应。

即使只作诊断查看较平稳的 compare1 后半程，V/A 仍为 −0.269% [−0.995, +0.476]，V/P 为 −0.499% [−1.112, −0.007]，也没有显示新增打包机制的正收益。compare2 前、后半程 V/A 分别为 +0.644% [−0.037, +1.855] 和 −4.395% [−13.638, +0.105]，方向不一致。

这些观察说明本批计时存在明显波动，**不能单凭它们确定是频率、资源竞争、主机调度或其他原因，也不能把所有负结果都归因于噪声**。保留全部窗口，不事后删除慢窗口或仅凭中位数重新判定通过。

### 7.4 算子与 decoder profile：局部改善未转化为整步收益

独立[算子报告](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/operator/report.json)采用完整 `op.forward().clone()` 路径计时：

| 算子 | A / P / V 平均耗时（µs） | V/A 收益 %［95% CI］ | V/P 收益 %［95% CI］ |
| --- | --- | --- | --- |
| Attention | 192.761 / 147.686 / 146.792 | +23.848 [+23.514, +24.158] | +0.606 [+0.263, +0.902] |
| MLP | 266.845 / 227.662 / 226.541 | +15.105 [+14.404, +15.754] | +0.495 [−0.468, +1.345] |

V/A 的较大算子收益包含原 MA1 warp/预取路径相对 scalar A 的差异，不能全部归给本次新增打包机制。V/P 才是本次增量：Attention 有小幅正收益，MLP 区间跨零。

独立[完整训练 profile](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/fullstep/profile-summary.json)中，先取各 rank 的 decoder 时长中位数，再对八 rank 求算术平均，得到：

| decoder 时长（µs） | A | P | V |
| --- | ---: | ---: | ---: |
| Attention | 50.722 | 15.752 | 15.004 |
| MLP | 52.810 | 15.962 | 15.220 |

该 profile 支持 V 的 decoder 在本次独立采集中比 P 更快，且执行路径检查通过；它不等同于正式训练窗口的墙钟收益。**本批最终结论是：MA2 实现及正确性验证通过，局部 decoder/Attention 算子有改善，但未证明完整训练达到 1%，也未证明稳定优于 MA1。** 后续优化应结合整步关键路径和计时波动继续定位，不能仅凭寄存器、指令数或单独 decoder 时长宣称目标已达成。
