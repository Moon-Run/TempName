# MA2：连续系数打包预取，目标完整 step 提升 1%

日期：2026-10-22。MA2 是 [MA1](ma-1.md) 的后续优化，针对旧预取实现的细粒度读取和寄存器开销。**这次目标是相对原始交替 A 达到至少 1% 的完整 step 点估计收益，并在两次独立比较中确认优于旧 MA1 P。当前没有 MA2 的八卡实测结果。**

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

## 4. 两次独立比较与 1% 判定

本次“两轮”指 **compare1、compare2 两次独立完整训练比较**；每次内部仍有顺序平衡的前后两个半程，不把这两个半程当作两次独立实验。

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

回传整个批次，保留冻结脚本、plan、rank JSON、mapping、trace、manifest 和状态。离线报告入口：

```bash
python3 tools/h100/phase6/ma2_bench.py report /path/to/batch/operator --offline
python3 tools/h100/phase6/ma2_bench.py profile-report /path/to/batch/operator --offline
python3 tools/h100/phase6/ma2_fullstep.py report /path/to/batch/fullstep --offline
```

## 6. 当前验证与限制

2026-10-22 已完成以下本地检查：

| 检查 | 结果与证据 |
| --- | --- |
| 构建身份、数值模拟、报告、profile 解析、两轮编排与启动器 | **70 项 CPU 测试通过**；[测试登记](../../../../outputs/h100/phase6/ma2-validation-20261022/cpu-validation.json)、[完整日志](../../../../outputs/h100/phase6/ma2-validation-20261022/cpu-tests.log) |
| A/P/V 实际 runtime 编译与两种 K 的交替 mapping | 三组均通过 host mapping 和 CUDA 12.8 **SM90 目标文件编译**，保存 PTX/SASS；[编译登记](../../../../outputs/h100/phase6/ma2-validation-20261022/runtime-check/validation.json) |
| P/V 同一编译单元、相同选项的 decoder 比较 | 最终 P **40**、V **34** 个寄存器/线程；两者 stack/spill 均为 0；[指令与数值审计](../../../../outputs/h100/phase6/ma2-validation-20261022/decoder-check/validation.json) |
| FP8x2 与单值转换的主机实现 | 穷举 **65,536 对字节**（其中 64,516 对均为有限值），half 位模式全部一致；[转换检查](../../../../outputs/h100/phase6/ma2-validation-20261022/decoder-check/fp8x2-host.json) |
| 一行命令预览 | 正确生成两次比较、72 窗口、3600 正式步的计划，未创建目标目录或启动训练；[预览记录](../../../../outputs/h100/phase6/ma2-validation-20261022/launch-preview.txt) |

SASS 确认每来源读取指令从 6 条降到 3 条、FP8 转换从 4 条降到 2 条，本地读取和最终写回分别从 4 条降到 1 条。每来源仍有 20 条 shuffle；下一来源的读取确实发射在当前来源首条 shuffle 之前，并在后续迭代消费。这些是编译产物中的指令观察，不是运行时计数或耗时。

上述检查没有执行 GPU kernel，也没有完成整套 Flux 动态库链接、八卡逐位正确性或完整训练性能验证。主机转换穷举不能替代设备指令验证；这些检查由启动脚本在目标服务器继续执行。静态指令或寄存器减少不等于已经改善实际 occupancy、访存重叠或完整 step，寄存器分配粒度也可能使 occupancy 不变。是否达到 1% 由上述两次实测决定，不将 MA1 的 0.3%–0.4% 与一个推测收益相加。
