# MA2-R × C：组合收益四组实验

日期：2026-10-23。按更正后的需求，本实验比较 **MA2-R＋C、C、MA2-R、A**，分别记为 **CR、C、R、A**。这里的 R 是 [MA2-R](ma-2.md#9-ma2-r优化固定配置提交与输出生命周期)，不是原 MA2 V。实现与测试脚本已建立，当前没有本实验的八卡计时结果。

| 组 | 实现 | 生产顺序与量化 | 工作区、输出与同步 |
| --- | --- | --- | --- |
| **CR：MA2-R＋C** | C 的完整执行框架，接收端接入 MA2-R 的打包来源预取 | C 的原到达表与原 1/64 冻结 mask | C 的双工作区、实例配置、独立输出；每调用原发布屏障，跨调用保证复用安全 |
| **C** | [base-baseline.md](../base-baseline.md) 中的原 C，按登记源码精确重建 | 交替＋原到达表；原 1/64 冻结 mask | 保留原 C 的双工作区、实例配置、独立输出和同步 |
| **R：MA2-R** | 原 MA2-R，逐文件保持其生成源码 | 基础交替、无到达表；全远端 H128/E4M3 量化 | 单工作区、实例配置、独立输出；发布与复用两道屏障 |
| **A** | 原融合 v2 scalar 接收端 | 基础交替、无到达表；全远端 H128/E4M3 量化 | 单工作区、逐次配置/复位、输出 clone；两道屏障 |

最终结果按下表填写，不能将历史 C 或 R 的耗时直接填入新实验。

| 比较 | CR / C / R / A 平均步时（ms） | CR/C 收益 %［95% CI］ | CR/R 收益 %［95% CI］ | CR/A 收益 %［95% CI］ |
| --- | --- | --- | --- | --- |
| compare1 | 待实测 | 待实测 | 待实测 | 待实测 |
| compare2 | 待实测 | 待实测 | 待实测 | 待实测 |

## 1. 组合的实际含义

MA2-R 已在上一批两次比较中得到约 3.7% 的 R/A 整步收益，但这主要是固定配置提交与输出路径的联合改动，不能将其全部归给打包 decoder。C 本来就有实例配置与独立输出管理，因此 **不能把“C 的收益＋R 的收益”相加，也不能通过再次去 clone 获得同一份收益**。

本次 CR 保留 C 的生产、选择性量化和生命周期，把 R 继承自 MA2 的连续四系数布局、打包载入/转换/写回、下一来源预取适配到 C 的混合数据路径。相对于 C，声明改动仅为接收 runtime 与两个新增 decoder 头文件；C 的 GEMM、encoder、mask、到达表、Python/C++ 包装、双工作区和屏障逻辑保持原样。构建器会核对这些文件的哈希。

C 的 mask 仍为 Attention **53/3584**、MLP **50/3584** 个远端贡献被量化，合计 **103/7168（约 1.437%）**；不是全远端量化，也不替换成 1/16 或其他实验的 mask。未选中贡献与本地贡献继续使用 BF16。

## 2. 接收端如何适配 C

对每个 H128 输出组，先按照 C 的物理 tile 与源 rank 查询冻结 mask，分两条路径：

1. **全部来源为 BF16：** 保留 C 已有的连续四元素读取和 packed BF16 归约算法，不强制经过 FP8 解码。
2. **至少一个远端来源为 FP8：** 每 lane 负责 `4*lane+j` 四个连续系数。选中的来源读取四个 FP8 字节及两个 scale，未选中的来源直接读取对应 BF16 槽；在处理当前来源时提前读取下一来源。

来源顺序仍为 `(rank+1)%8,…,rank`，本地来源最后加入。对 FP8 来源保留原 Hadamard 阶段顺序、显式舍入的缩放乘法、解码后 BF16 舍入，以及每次来源相加后的 BF16 舍入；不合并来源、不改为 FP32 跨来源求和。未选中的 FP8 槽不读取，避免将旧 packet 内容当作当前数据。

C 的双工作区继续交替使用。每次调用保留原发布屏障，下一次调用的发布屏障按原 C 协议保护此前槽位的复用；CR 不添加新的到达标记、辅助 stream 或轮询 kernel，也不改变这一同步协议。R/A 则继续保留它们原有的两道屏障，不能为了让四组屏障数相同而改写对照组。

接收端实现见 [ma2_c_decode.cuh](../../../../tools/h100/phase6/ma2_c_decode.cuh) 与 [ma2_ablation_sources.py](../../../../tools/h100/phase6/ma2_ablation_sources.py)。C 的纯 BF16 算法虽然保留，整个 kernel 的寄存器分配和调度仍可能变化；静态指令减少不能直接证明所有路径更快。由于 C 的量化覆盖很小，组合的增量也可能很小或为负。

## 3. 四组对照能回答什么

| 对比 | 可以回答的问题 | 归因限制 |
| --- | --- | --- |
| **CR/C** | 在相同 C 框架、mask、到达表与生命周期下，新增接收端优化是否产生净收益 | 本实验最主要的组合增量 |
| **CR/R** | 组合是否比单独 MA2-R 更快 | 同时包含选择性量化、排序、双工作区和实现差异 |
| **CR/A** | 组合相对原交替实现的整体收益 | 不能单独归给某一个机制 |
| R/A | 同场复测已测过的 MA2-R | 不使用上一批约 3.7% 替代新样本 |
| C/A | 同场复测 baseline C | 不能单独解释成到达排序或量化覆盖收益 |
| R/C | 单独两个方案谁更快 | 两者的量化范围、数值轨迹和生命周期不同 |

这四组是“组合与两种已有方案的对照”，不是所有因素均独立切换的完整因子设计。若 CR/C 为正但 CR/R 不为正，只能说接收端改动改善了 C，不能说组合优于两个单独方案。

## 4. 源码、计划和环境冻结

[ma2_ablation_build.py](../../../../tools/h100/phase6/ma2_ablation_build.py) 从仓库固定 candidate 源码恢复 C，并逐文件匹配 `base-baseline.md` 所用 `baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare1` 中登记的 C manifest。**C 的 351 个源码文件与该记录全部一致；A 的 351 个文件与原 A 一致，R 的 352 个文件等于原 MA2-R 变换。** CR 只相对 C 改动 `taco_runtime.cu`，并增加 `ma2_decode.cuh`、`ma2_c_decode.cuh`，共 353 个登记文件。

原角色计划、到达表和 mask 由该 baseline 的 `scripts/selection-plan.json` 冻结，构建器固定其文件摘要，不重做校准，也不以相同预算重新生成另一张 mask。默认重新编译四份隔离构建，不要求旧 C 动态库仍在；但仓库内固定 candidate 源码及上述 baseline 的来源记录必须完整，缺失或哈希不符会停止。

同场四组统一使用 `NCCL_NVLS_ENABLE=0`、`CUDA_DEVICE_MAX_CONNECTIONS=1`，其余通信环境捕获后核验。重新编译和统一环境意味着历史批次均值只能作背景，不与当前窗口混算，也不直接相减推导收益。

## 5. 正确性按两对检查

**只在等价量化范围内要求逐位一致：CR/C 一对、R/A 一对。** C 与 A 的历史训练轨迹原本不同，不能要求四组输出或 loss 全部相同，也不能因此放宽 CR/C 的一致性门槛。

- 四组输入、权重、模型初始化、RNG/token、软件与硬件身份一致；实际 mapping 分别符合各自冻结计划，CR/C 的 mapping 相同，R/A 的 mapping 相同。
- 算子覆盖零值、小值、离群值、随机输入、两 seed、多轮工作区复用以及 rank0/rank7 延迟；CR/C 使用原 selective mask 的独立 codec 参考，R/A 使用全远端参考。
- 保留原相对 BF16 5%、相对独立 codec 1% 的误差门槛，同时要求等价对内输出逐位一致；不根据新结果放宽阈值。
- 连续保留至少六次输出，检查独立存储与后续调用不覆盖；交错调用不同 K 的实例，检查工作区和配置隔离。
- 完整模型检查全部 64 个投影、有限 loss/gradient、零 skip/fallback、真实调用数；CR/C、R/A 对内逐 rank 的 loss/gradient 轨迹一致。各组正式 50 步平均 loss、末步 loss 与末步 grad norm 单列到报告。
- Profile 必须显示每臂实际 decoder。CR/C 每调用一道原发布屏障且零输出 clone；R 为两道屏障、零 clone；A 为两道屏障、一次 clone。CPU annotation 与 GPU 镜像 annotation 分开解析，避免重现旧启动报告问题。

报告字段使用 `bitwise_equal_within_pairs`、`identical_trajectories_within_pairs`，并保存 `equivalence_pairs=[["CR","C"],["R","A"]]`，不把局部等价表述为四组数值相同。

## 6. 两轮完整训练及判定

模型沿用 GPT6.7B：32 层、H4096/FFN16384、32 heads、S2048、TP8，BF16，micro batch1/global batch4，seed1234/token_seed4321，每 optimizer step 8192 token。每窗口独立初始化，10 步预热＋50 步正式；每次比较先做四个循环移位块，后四块反转块顺序与块内顺序，四组在每个位置出现次数相等。

| 项目 | 每次比较 | compare1＋compare2 |
| --- | ---: | ---: |
| 完整四组块 | 8 | 16 |
| 正式窗口 | 32 | **64** |
| 正式 rank 记录 | 256 | 512 |
| 正式 optimizer step | 1600 | **3200** |
| 正式窗口预热步 | 320 | 640 |

两次比较使用同一冻结配置、不同 run ID 与新进程；不是更换种子或数据的实验。算子阶段另有 32 个计时窗口、64 份 trace；完整模型共用预检，另采集 32 份完整 step trace，均不混入正式训练计时。

窗口指标为八 rank 最大墙钟除以 50。收益为 `100×[1−exp(mean(log(T候选/T参照)))]`；以完整四组块配对 bootstrap 10000 次，seed20261021。两次分别报告六项比较、前后半程、全量窗口及负收益，不把 rank 或窗口内 step 当独立样本，不删除慢窗口。

**组合通过条件：两次各自的 CR/C 与 CR/R 点估计、95% CI 下界均大于零，且正确性与执行路径检查通过。** CR/C 点估计是否达到 1%、其区间下界是否达到 1% 另列为更强诊断；本次“进一步优化”的主判定不把 1% 当作必需门槛。所有比较仍完整保留，即使算子性能为负也继续完整训练；正确性或执行证据失败则停止并保存现场。

## 7. 一行启动

同步本次新增工具和仓库中的固定来源记录后，在服务器 `/workspace/TempName` 执行：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_ma2_ablation.py --execute
```

默认重新准备并编译 CR/C/R/A，CUDA12.8 编译、CUDA12.4 运行、jobs2，使用相邻 `../Megatron-LM`。后台托管输出目录为 `logs/h100/phase6/ma2r-c-ablation-2x-50steps-<北京时间戳>/`：

```text
batch-state.json                   完成状态及组合收益判定
handoff/container-supervisor.log  后台总日志
build/                            四组源码、冻结 selection-plan 与构建身份
operator/                         算子正确性、计时、mapping 和 profile
fullstep/                         两次完整训练的计划、原始记录与报告
```

去掉 `--execute` 仅预览，不创建目录或查询 GPU。`--ablation-build` 可复用本实验已编译并核验的四组构建，不能传入旧三组 MA2 或 MA2-R 构建。沿用 `--python`、`--megatron`、`--build-cuda-home`、`--cuda-home`、`--nccl-root`、`--jobs`、`--worker-timeout`、`--foreground` 参数。容器和主机需持续运行；启动前结束其他训练任务，避免相互竞争资源。

离线复算入口：

```bash
python3 tools/h100/phase6/ma2_ablation_bench.py report /path/to/batch/operator --offline
python3 tools/h100/phase6/ma2_ablation_bench.py profile-report /path/to/batch/operator --offline
python3 tools/h100/phase6/ma2_ablation_fullstep.py report /path/to/batch/fullstep --offline
```

## 8. 当前验证与限制

| 本地检查 | 结果与证据 |
| --- | --- |
| 混合数据、实验编排与报告回归 | **31 项 CPU 测试通过**；[验证登记](../../../../outputs/h100/phase6/ma2-c-ablation-validation-20261023/cpu-validation.json)、[测试日志](../../../../outputs/h100/phase6/ma2-c-ablation-validation-20261023/cpu-tests.log) |
| 四组映射与实际 runtime | 两种 K 的 host mapping、四组 SM90 runtime 目标文件编译通过；[编译登记](../../../../outputs/h100/phase6/ma2-c-ablation-validation-20261023/runtime-check/validation.json) |
| R 原生包装 | 真实 PyTorch C++ wrapper 目标文件编译通过 |
| 一行命令预览 | 确认 64 窗口、3200 正式步，未创建目标目录或启动 GPU；[预览记录](../../../../outputs/h100/phase6/ma2-c-ablation-validation-20261023/launch-preview.txt) |

四组准确源码重建已通过。新的 CR kernel 在本地编译中使用 44 个寄存器、8-byte stack，0 spill；这些不是吞吐或占用率实测。本地尚未完成整套 Flux 动态库链接，目标服务器会执行完整编译与运行门禁。

本地回归覆盖冻结 mask、混合来源的 BF16 舍入、四组顺序平衡、两对数值一致性、错误 mask/工作区拒绝、完整训练统计和不同屏障/clone 路径。既有 C/A/R 的真实 rank0 完整训练 trace 通过新解析器的兼容检查；它们不是 CR 的 GPU 运行证据。

当前未在八卡上执行 CR，也没有本轮性能结果。只有新批次的 CR/C 与 CR/R 同场结果才能判断组合是否进一步优化；短程固定合成输入的一致性也不能替代真实语料长期收敛验证。
