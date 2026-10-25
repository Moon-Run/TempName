# MA5：在 CR2 上调节 tile 与 block

更新：2026-10-25。**本轮已实现可测试候选与独立测试入口，H100 性能结果待上机；500 ms 是目标，尚未达成。** 原始 CR2、历史 baseline 和已有实验入口保留。

## 1. 目标与边界

[baseline.md](baseline.md) 中 CR2 的全部窗口平均为 **511.297 ms/step**、中位数为 **510.450 ms/step**。从该均值降到 500 ms，需要节省约 **11.297 ms（2.21%）**。新实验使用本次重新构建、同场运行的 CR2 作为对照；历史 511.297 ms 只用于说明目标，不参与新配对收益计算。

本轮保持 GPT6.7B、TP8、Attention＋MLP、原冻结到达表、原 1/64 selective mask、双工作区、独立输出、原生提交缓存及每调用一次发布屏障。量化覆盖仍为 Attention **53/3584**、MLP **50/3584**。不改变 H128 编解码、来源累加顺序或 BF16 舍入规则。

CR2 上存在三个不同概念，调参必须落在实际执行路径上：

| 参数 | CR2 当前值 | MA5 的处理 |
| --- | --- | --- |
| GEMM CTA tile / warp tile | `128×128×32` / `64×64×32` | 默认保持；可选 K tile 同时改为 64 |
| decoder 每 block 线程数 / 一次处理的输出元素 | `128` / `512` | 支持 64、128、256、512 线程，即 256、512、1024、2048 元素 |
| decoder grid 的 block 数 | `2048` | 独立指定，通过 grid-stride 循环覆盖全部输出 |
| GEMM `avail_sms` | `-1`，由 CUTLASS 使用设备默认 | 可选 128 或 132，改变 Stream-K 调度输入；**不是直接指定 GEMM grid 数** |
| 旧 RS option 的 `num_blocks` | `6` | CR2 走直接 decoder 路径，单改该值不能调节实际 decoder grid，且会触发固定配置检查 |

CR2 历史训练 profile 中 decoder 合计约 2.8 ms/step，远小于 11.3 ms 的目标差距；profile 还存在观测开销与重叠。因此 decoder 调参是局部机会，不能据此保证达到 500 ms。GEMM K tile 和 SM 调度候选用于继续探索生产端收益。

## 2. 已实现的改动

### 2.1. Decoder：tile 大小与 grid 解耦

原 decoder 为 `<<<2048,128>>>`，每 warp 处理一个完整 H128 组，每 block 4 个 warp。每 rank 输出 `[256,4096]`，共 **8192 个 H128 组**。

MA5 将原逐组计算原样提取为设备函数，再使用 warp 粒度的 grid-stride 循环：

```cpp
group = blockIdx.x * warps_per_block + threadIdx.x / 32;
group < 8192;
group += gridDim.x * warps_per_block;
```

一个 warp 内的循环条件一致，原 full-mask shuffle 的参与线程不变。纯 BF16 路径的提前 `return` 留在逐组函数内部，不会漏掉同一 warp 后续负责的组。量化组继续使用原来源预取、原解码与逐次舍入；每个输出组只被一个 warp 写入。

默认候选为 **256 线程、528 blocks**：单次 CTA tile 从 512 增至 1024 个输出元素，grid 从 2048 减至 528。528 是目标 H100 SM132 的四倍，但不意味着硬件一定同时驻留四个 CTA/SM；驻留情况仍取决于资源使用。该配置大部分 warp 处理两个 H128 组，尾部按边界退出。

`--sweep` 顺序运行以下四组，每组都有自身同场 CR2 对照：

| 候选 | threads/block | 每次 CTA 输出 tile | decoder blocks | GEMM tile K / avail_sms |
| --- | ---: | ---: | ---: | --- |
| `t128-b528-k32-s-1` | 128 | 512 元素 | 528 | 32 / -1 |
| `t256-b1024-k32-s-1` | 256 | 1024 元素 | 1024 | 32 / -1 |
| `t256-b528-k32-s-1`（默认） | 256 | 1024 元素 | 528 | 32 / -1 |
| `t256-b264-k32-s-1` | 256 | 1024 元素 | 264 | 32 / -1 |

这组比较分别探查更大的 CTA tile、较少的 CTA 数量及二者组合。候选也可能因寄存器、驻留或循环开销而变慢。四组都保留，不依据算子微测的速度提前淘汰。

### 2.2. GEMM：可选 K64 与调度 SM 数

`--gemm-tile-k 64` 同时将 CTA tile 改为 **128×128×64**、warp tile 改为 **64×64×64**，保持 4 warp、128 线程和一个 K partition。不能只改 CTA K：原自动 warp K=32 会产生两个 K partition，破坏融合 encoder 的四 warp 布局假设。

K64 的 epilogue 使用精确匹配的新 thread map，并以编译期断言核对其基础输出布局和 `ThreadMapShape` 与 K32 相同。M/N tile 保持 128×128，因此原 **512 个物理 tile** 的 mask 和到达表可以继续使用。该候选增加 mainloop 的共享内存需求，可能降低 occupancy；也可能改变 Stream-K 分解及浮点计算轨迹，收益与数值一致性都必须实测。

`--gemm-sms 128` 将 CR2 缓存的 `GemmReduceScatterArguments.avail_sms` 从 -1 改为 128。512 个物理 tile 与 128 的整除关系提供了探索动机，但实际 grid 仍由 CUTLASS、K、occupancy 和 Stream-K 策略共同决定。报告记录真实 GEMM grid/block，不把 SM 预算冒充实际 block 数。

### 2.3. 来源与执行身份

代码入口：

- [ma5_sources.py](../../../../tools/h100/phase6/ma5_sources.py)：decoder 循环、真实 launch 参数与 GEMM 调度参数。
- [ma5_gemm.py](../../../../tools/h100/phase6/ma5_gemm.py)：可选 K64 generator/epilogue 变换和实际 GEMM 编译探针。
- [ma5_build.py](../../../../tools/h100/phase6/ma5_build.py)：从原 CR2 重建 `T/CR2` 两组，冻结 tuning、源码、mask 和构建输入。
- [ma5_bench.py](../../../../tools/h100/phase6/ma5_bench.py)、[ma5_fullstep.py](../../../../tools/h100/phase6/ma5_fullstep.py)：算子检查、完整训练及独立 profile。
- [detach_ma5.py](../../../../tools/h100/phase6/detach_ma5.py)、[run_ma5_tile_block.sh](../../../../tools/h100/phase6/run_ma5_tile_block.sh)：预览、后台启动和批次检查。

`T` 表示本次 tuning 候选，`CR2` 为逐文件核对的原实现。构建后的 C ABI 返回候选 tuning；算子与完整训练均读取它，并检查两组的 CR2 缓存初始化次数、调用次数、K、packet 地址及失败状态。Profile 进一步核对真实 decoder grid/block、GEMM K tile 与 128 线程，以及 GEMM→发布屏障→decoder 的执行顺序。

## 3. 测试命令

在八卡 H100 服务器的仓库根目录（如 `/workspace/TempName`），先运行默认 decoder 候选：

```bash
bash tools/h100/phase6/run_ma5_tile_block.sh --trials 1 --execute
```

启动方式与 `run_basic_quant.sh` 一致，优先读取 `outputs/h100/container-recovery/env.sh`，复用恢复环境中的 Python 和 CUDA 路径。默认后台托管；`--foreground` 可在当前终端顺序执行。**去掉 `--execute` 只打印计划，不创建批次目录、不编译、不查询 GPU。** 本地只预览时可用 `MA5_PYTHON=/usr/bin/python3 bash tools/h100/phase6/run_ma5_tile_block.sh --trials 1` 显式覆盖服务器 Python 路径。

一次完整 decoder 扫描：

```bash
bash tools/h100/phase6/run_ma5_tile_block.sh --trials 1 --sweep --execute
```

四个候选依次完成独立构建、算子检查和完整训练，不并发占用八张 GPU。每个候选都重新测量 CR2；首次扫描会编译四套候选/对照，耗时较长。`--sweep` 固定 decoder 参数，不能与单独的 decoder 覆盖参数混用。

单独检查 GEMM K64，decoder 保持原线程数和 grid：

```bash
bash tools/h100/phase6/run_ma5_tile_block.sh --trials 1 \
  --decoder-threads 128 --decoder-blocks 2048 --gemm-tile-k 64 --execute
```

单独检查 GEMM SM 调度，K tile 和 decoder 保持原值：

```bash
bash tools/h100/phase6/run_ma5_tile_block.sh --trials 1 \
  --decoder-threads 128 --decoder-blocks 2048 --gemm-sms 128 --execute
```

得到有希望的配置后，以明确参数在新批次复测两轮。例如，若默认 decoder 候选值得复测：

```bash
bash tools/h100/phase6/run_ma5_tile_block.sh --trials 2 \
  --decoder-threads 256 --decoder-blocks 528 --execute
```

如需合并 K64、SM128 和 decoder 调参，可同时提供三个选项，但要用同场训练结果验证组合收益。不同独立实验的加速不能直接相加。不同批次不要同时运行；`--execute` 返回只表示后台启动成功，完成状态以报告为准。

常用选项：`--out <新目录>`、`--jobs 2`、`--worker-timeout 1800`、`--cuda-home`、`--build-cuda-home`、`--nccl-root`。`--ma5-build <已有MA5构建>` 可复用已验证的两组构建，但必须显式给出与该构建一致的 tuning；不能用于 `--sweep`。线程数必须为 64/128/256/512，blocks 为正整数且不得超过 `8192/(threads/32)`；例如 512 线程最多 512 blocks，不能直接搭配默认 528。

## 4. 工作量、正确性与判定

每个候选、每个 trial 为 **8 个平衡 T/CR2 块、16 个正式窗口、128 份 rank 记录、800 个正式 optimizer step**。每窗口独立进程，先预热 10 步，再正式计时 50 步。`--trials 1` 只产生 compare1；`--trials 2` 为 32 个正式窗口、1600 正式步。四候选单轮 sweep 合计 64 窗口、3200 正式步。另有算子测试、训练 preflight 及独立 profile，不混入正式计时。

保持原完整 optimizer step 计时，包括前后向、通信、参数更新和清梯度；窗口步时为八 rank 最大墙钟 / 50。报告保留全部窗口，给出算术均值、中位数、前后半程，以及完整块配对 log 耗时比的 10000 次 bootstrap 95% CI。

必须满足：

1. 两种 K、多个 seed、零值/小值/离群值/随机值、输出保留、双槽复用、实例隔离和慢 rank 压力检查通过；T/CR2 输出逐位一致。
2. 完整训练的对应 rank loss/gradient 轨迹一致，无跳过、fallback 或未覆盖模块；两组都真实走 CR2 缓存。
3. 算子和完整训练 profile 的实际 launch 与冻结 tuning 一致，屏障、工作区和输出路径符合定义。

如果 K64 或 SM 调度改变了数值轨迹，该候选按当前协议拒绝，不静默放宽误差预算。性能退化、区间跨零和未达 500 ms 均为有效结果，仍完成报告。

| 报告字段 | 含义 |
| --- | --- |
| `increment_over_CR2_validated` | 每个请求的 trial 中，T/CR2 收益点估计与 95% CI 下界均 >0 |
| `absolute_500ms_observed` | 每个 trial 的 T 全部窗口算术均值 ≤500 ms；不是最低窗口或中位数阈值 |
| `target_validated` | 上述两项同时成立；批次完成还要求正确性与独立 profile 通过 |

`absolute_500ms_observed` 只描述本批均值，并不等于以置信区间证明长期均值 ≤500 ms。单轮结果与 sweep 属于探索；从多个候选中挑出最快者之后，应在新批次用两轮复测，不能把挑选后的点估计当作独立确认。

## 5. 查看结果与本地验证

默认批次目录：`logs/h100/phase6/ma5-tile-block-1x-50steps-<北京时间戳>/`。

```text
batch-state.json                       # 状态、候选列表、完成情况与训练结论
handoff/container-supervisor.log        # 后台总日志
t256-b528-k32-s-1/
  build/tuning.json                    # 实际构建参数
  operator/report.json                 # 正确性与独立算子时间
  operator/profile-summary.json        # 算子实际 launch
  fullstep/report.md                   # 易读的整步结果
  fullstep/report.json                 # 配对统计、全部窗口与数值摘要
  fullstep/profile-summary.json        # 训练期间的执行证据
```

失败时保留阶段日志和原始记录，不覆盖旧批次；`batch-state.json` 的 `status=completed, stage=done` 才表示所有请求的候选已完成。`handoff/controllers.json` 冻结控制器及依赖源码，每个阶段前后核对，运行期间不要修改工具源码。回传整批目录即可按原始 rank 记录重新汇总。

本地 CPU 检查命令：

```bash
python3 -m unittest discover -s tools/h100/phase6 -p 'test_ma5*.py' -v
bash -n tools/h100/phase6/run_ma5_tile_block.sh
```

本地 **37 项 CPU 检查通过**，覆盖完整输出组覆盖、候选参数边界、原 CR2 源码冻结、K64 thread map、launch 身份、缓存状态、单/双 trial 工作量、缺失 rank 与轨迹漂移拒绝、负收益正常完成、只预览不执行和控制器漂移拒绝。shell 语法及带空格参数转发也已检查。见 [CPU 日志](../../../../outputs/h100/phase6/ma5-validation-20261025/cpu-tests.log)、[默认命令预览](../../../../outputs/h100/phase6/ma5-validation-20261025/launch-preview.txt)、[扫描预览](../../../../outputs/h100/phase6/ma5-validation-20261025/sweep-preview.txt)。

最终冻结源码已完成两组 host mapping、SM90 decoder runtime、GPU mapping 程序编译、真实 K32 CUTLASS/encoder 探针，以及候选 PyTorch C++ wrapper 和 pybind 目标文件编译。见 [最终编译验证](../../../../outputs/h100/phase6/ma5-validation-20261025/final/compile-default/validation.json)及[编译日志](../../../../outputs/h100/phase6/ma5-validation-20261025/final/compile-default/compile.log)。默认 decoder 的编译器报告为 40 registers、零 spill，原 CR2 为 44 registers、零 spill；这是编译诊断，不是性能结论。

可选 K64 的真实 CUTLASS/encoder 探针也已通过 SM90 编译，包含实际 `StoreD::kTacoSupported` 与 128 线程断言；最终源码与该探针逐文件核对一致，并编译了当前 K64/SM128 的 mapping 程序。见 [K64 编译记录](../../../../outputs/h100/phase6/ma5-validation-20261025/k64-probe-supported/validation.json)与[最终源码和 mapping 复核](../../../../outputs/h100/phase6/ma5-validation-20261025/final/k64-validation.json)。

**本地没有运行 GPU、完整 Flux 动态库链接或训练性能实验。** mapping 程序只编译未执行；H100 上的完整构建、数值门槛、真实 launch 和整步时间由上述测试入口完成。当前不能把候选登记为快于 CR2，也不能写成已达到 500 ms。
