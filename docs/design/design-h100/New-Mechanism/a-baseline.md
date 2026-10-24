# Flux 基础量化、改进量化与独立 CUDA 量化的同场比较

更新：2026-10-24。新增三组实验，**统一覆盖 Attention＋MLP 输出投影**。本页的“Flux＋基础量化”仅保留生产端融合：GEMM 内编码并向目标 GPU 写入；接收端先独立解码，再独立归约。“Flux+改进量化”保留既有融合 v2 的生产端融合与接收端解码归约融合。“Flux＋CUDA量化”沿用原生 Hopper GemmOnly＋作者独立 TACO CUDA codec＋NCCL All-to-All。

**当前没有本实验的八卡计时结果。** 下表等待新批次回传，不填入 [历史汇总](baseline.md) 的跨批次数字。历史“改进量化”只覆盖 MLP，本次将其相同实现用于 Attention＋MLP，不能把历史步时当成本次对照。

| 指标 | **Flux+改进量化** | **Flux＋基础量化** | **Flux＋CUDA量化** |
| --- | --- | --- | --- |
| 策略 ID | `improved` | `basic` | `cuda` |
| 融合范围 | GEMM 内编码；解码与归约融合 | 仅 GEMM 内编码；解码与归约分开 | GEMM、编码、通信、解码、归约分开 |
| 作用范围 | Attention＋MLP | Attention＋MLP | Attention＋MLP |
| 量化覆盖 | 全远端 H128/E4M3 | 同左 | 同左 |
| 平均步时（ms/step） | 待实测 | 待实测 | 待实测 |
| 全部窗口中位步时（ms/step） | 待实测 | 待实测 | 待实测 |
| 平均 / 最终训练 loss | 待实测 | 待实测 | 待实测 |
| 吞吐（tokens/s） | 待实测 | 待实测 | 待实测 |

用户希望检验的耗时关系为：

```text
T(Flux+改进量化) < T(Flux＋基础量化) < T(Flux＋CUDA量化)
```

这是实验假设，不是必须实现的排序。基础版的来源解码可并行，CUDA 组使用不同 GEMM 后端；中间访存、GPU 并发和主机提交之间存在权衡，不能仅凭融合数量保证上述关系。报告保留全部窗口及负收益，性能排序不成立也继续完成所选次数的比较；正确性或执行证据不通过则停止。

## 1. 为什么这样定义“基础版”

原历史“改进量化”是冻结的 `a800-fused-v2-cta-v1`，它已经使用较朴素的 scalar CTA codec，并没有 MA1/MA2 来源预取、CR2 提交缓存或 CUDA Graph。若仅复制并改名，就无法得到一个具有明确区别的基础版。

本实验按融合边界定义三个实现层次：

```text
Flux＋CUDA量化：
  原生 Hopper GemmOnly → 整理远端 BF16
  → 作者独立编码 kernel → NCCL All-to-All
  → 作者独立解码 kernel → 逐来源 BF16 相加 → 输出

Flux＋基础量化：
  [原 GEMM＋编码＋远端写入] → 原发布屏障
  → [各来源独立解码到 BF16 中间张量]
  → [按固定来源顺序归约] → 原复用屏障 → 独立输出 clone

Flux+改进量化：
  [相同 GEMM＋编码＋远端写入] → 相同发布屏障
  → [解码＋固定顺序归约融合] → 相同复用屏障 → 独立输出 clone
```

“基础版”与“改进版”都沿用原始生产顺序、单工作区、逐次配置与原输出 clone。没有加入交替排序、到达优先、选择性 mask、打包来源预取、双工作区、持久配置优化、提交缓存或 Graph。这些历史优化不是本实验的组成部分。

## 2. 基础融合 kernel 的实现

实现由 [a_baseline_sources.py](../../../../tools/h100/phase6/a_baseline_sources.py) 注入隔离构建；新增接收端计算体见 [a_baseline_decode.cuh](../../../../tools/h100/phase6/a_baseline_decode.cuh)。

基础版与改进版共享相同 GEMM、epilogue 编码、packet 格式和 codec 数学代码。基础版仅在接收时增加必要的分阶段数据流：

1. 分配七个远端来源的 BF16 临时张量，形状为 `[7,256,4096]`。
2. 每个 CTA 负责某个来源的一个 H128 组，调用原 scalar CTA 解码过程，将结果舍入 BF16 后写入中间张量。
3. 独立归约 kernel 按 `(rank+1)%8, …, (rank+7)%8, rank` 的原顺序相加。最后一个是本地 BF16 贡献，不经过量化解码；每次相加后保留原 BF16 舍入。
4. 沿用原来的工作区复用屏障和输出所有权规则。

临时张量大小为 `7×256×4096×2 = 14,680,064` 字节，即 **14 MiB/调用**。它由 PyTorch 分配器管理，计入正式计时；可由缓存分配器复用，不表示每个模型必须永久持有 64 份。没有增加重复解码、延时循环或额外跨 rank 屏障。

改进版直接在同一个 kernel 中解码并累加，不生成这份完整的 BF16 中间张量。基础/改进的直接比较主要回答：**在同一生产端下，去掉接收端中间数据写回和独立归约提交，是否改善完整训练步。**

两版必须保持解码后舍入、来源顺序和逐次累加舍入，不能将基础版改为 FP32 跨来源求和。由于原始数学相同，要求基础/改进的实际投影输出及训练 loss/gradient 轨迹一致。

## 3. 三组固定条件与对比边界

| 项目 | 基础 / 改进 | 独立 CUDA |
| --- | --- | --- |
| GEMM | 相同 SM90 编译的 GemmV2 | 原生 Hopper GemmOnly |
| 编码 | 相同冻结 epilogue 内编码 | 作者原 CUDA 编码 kernel |
| 远端传输 | 原融合路径直接写目标工作区 | NCCL All-to-All |
| 接收 | 基础分开解码/归约，改进融合两者 | 作者 CUDA 解码后逐来源 BF16 相加 |
| 排序/选择 | 原始顺序，无到达表、无选择性 mask | 原生 GemmOnly，无上述策略 |
| 本地贡献 | BF16 | BF16 |
| 反向 | 原 BF16/STE 适配 | 原 BF16/STE 适配 |

基础/改进是一对有明确融合差异的直接对照；与 CUDA 组比较还包含 GEMM 后端和通信路径差异，不能把其全部差距归因于一个 kernel 的融合。

独立 CUDA 组复用 [baseline_codec.py](../../../../tools/h100/phase6/baseline_codec.py) 及固定提交 `ca717e05b6dd6d21fff4d2617d209c95be63f3ef` 中的作者 CUDA 编解码计算体。它是作者 codec 的移植基线，未使用完整 COCCL 通信调度与重叠，不代表原论文完整 TACO 系统。

训练固定 GPT6.7B：32 层、H4096、FFN16384、32 heads、S2048、TP8，micro batch1/global batch4，BF16，seed1234/token_seed4321。每个 optimizer step 为四个微批、8192 token，覆盖 32 个 Attention 输出投影及 32 个 MLP 输出投影；本地 K 分别为 512、2048。统一 `NCCL_NVLS_ENABLE=0`、`CUDA_DEVICE_MAX_CONNECTIONS=1`。

## 4. 完整训练次数与结果判定

`--trials 1` 只执行 `compare1`；默认 `--trials 2`，依次执行 `compare1`、`compare2`。每次完整比较内部仍包含两轮、每轮六个完整三组块，组内顺序平衡，第二轮反转；每个窗口独立初始化，10 步预热＋50 步正式计时。**这里的 trials 是独立比较次数，不是每次比较内部的 rounds；选择一次不会取消内部第二轮的顺序平衡。**

| 项目 | 单次（`--trials 1`） | 两次合计（默认） |
| --- | ---: | ---: |
| 完整三组块 | 12 | 24 |
| 正式窗口 | 36 | **72** |
| 正式 rank 记录 | 288 | **576** |
| 正式 optimizer step | 1800 | **3600** |
| 正式窗口的预热 step | 360 | 720 |
| 独立 smoke/profile 窗口 | 6 | 12 |
| 训练 profile trace | 24 | 48 |

1800 / 3600 步分别是一次 / 两次比较中独立窗口的累计工作量，不是同一个模型连续训练这些步数。profile 单独执行，不混入正式计时。每个窗口以八 rank 最大墙钟除以 50 得到步时。只运行一次时，结果支持本次比较，不声称已完成两次独立重复验证。

完整报告三项配对：

- **improved/basic：** 接收端解码归约融合的增量。
- **basic/cuda：** 基础融合执行路径相对独立 CUDA 量化的整体差距。
- **improved/cuda：** 双端融合执行路径相对独立 CUDA 量化的整体差距。

收益使用完整三组块配对的对数耗时比和 bootstrap 95% CI；同时报告算术均值、中位数、吞吐、平均/末步 loss、各轮结果和全部窗口。预期顺序的点估计与置信区间确认情况分别列出，不使用历史耗时，也不因为某一轮不符合期望而只保留另一轮。

smoke/profile 检查需确认：

- 三组初始化、输入、模型与设备身份一致，实际覆盖全部 64 个目标投影，无 skip/fallback。
- 基础/改进的投影输出及训练轨迹一致；CUDA 组通过自己的独立 codec 参考和原误差门槛，不要求跨 GEMM/codec 实现逐位等价。
- 返回输出不被后续调用覆写，正确性检查在正式计时外。
- 基础版实际出现独立解码及归约 kernel；改进版出现原融合解码归约 kernel；CUDA 组出现实际 GemmOnly、作者编解码和 NCCL 通信。
- 分开统计 CPU annotation 与 GPU 镜像 annotation，避免仅凭重复同名标记误判；编译或真实路径错误保留现场并停止。

## 5. 一行启动

将本次新增 `a_baseline*`、`detach_a_baseline.py` 与现有 `tools/h100/phase6` 依赖同步到原服务器，在 `/workspace/TempName` 仓库根目录执行。**只测一次完整比较，36 个窗口、1800 个正式 optimizer step：**

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_a_baseline.py --trials 1 --execute
```

不传 `--trials` 时仍默认测两次，共 72 个窗口、3600 个正式步；原命令保持有效：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_a_baseline.py --execute
```

默认重新构建基础/改进两份隔离 Flux 库、独立 Hopper GemmOnly 库和作者 CUDA codec 库，CUDA12.8 编译、CUDA12.4 运行、jobs2；使用相邻 `../Megatron-LM` 的固定提交。无需复用旧 C 校准或到达表。已有库可用 `--fused-build`、`--flux-build`、`--codec-build` 指定，仍须通过本协议的来源与库身份检查。

**去掉 `--execute` 只打印计划**，不创建目录、不编译、不 fork、不查询 GPU。路径可以通过 `--python`、`--megatron`、`--build-cuda-home`、`--cuda-home`、`--nccl-root`、`--out` 指定；输出目录必须不存在。单个 worker 默认超时 1800 秒，可通过 `--worker-timeout` 调整；超时保留日志并终止该 worker 进程组，停止后续比较。

沿用已有双 fork 容器后台托管流程，确认监督进程由容器 PID1 接管后才开始工作。`--foreground --execute` 可用于有意前台诊断。启动成功只表示后台进程建立，实验完成需要检查批次状态；容器和主机必须保持运行。

默认输出目录随所选次数使用 `1x` 或 `2x`：

```text
logs/h100/phase6/a-baseline-fusion-<1x或2x>-50steps-<北京时间戳>/
  batch-state.json
  handoff/container-supervisor.log
  handoff/supervisor.json
  fused-build/                    基础/改进的隔离源码与库
  flux-build/                     原生 Hopper GemmOnly
  codec-build/                    作者独立 CUDA codec
  compare1/
    submission.json
    scripts/
    results/                      每窗口的进程日志、八 rank JSON 和 profile
    preflight.json
    report.json / report.md
  compare2/                       仅 --trials 2 时生成，第二次独立比较
```

各阶段命令与日志同时保存在批次根目录。`status=completed` 表示所选次数的完整比较及正确性检查完成，不表示预期速度排序一定成立。离线复核入口为：

```bash
python3 tools/h100/phase6/a_baseline.py report /path/to/batch/compare1 --offline
# 仅测两次时执行：
python3 tools/h100/phase6/a_baseline.py report /path/to/batch/compare2 --offline
```

## 6. 本地验证状态

新增 `--trials 1` 后，启动器 **7 项 CPU 回归通过**，覆盖一次比较完成状态、36/1800/288 计数、无 compare2、默认两次流程和失败停止规则；一次比较的命令预览未创建批次目录。见 [单次比较验证](../../../../outputs/h100/phase6/a-baseline-validation-20261024/single-trial-validation.json)、[测试日志](../../../../outputs/h100/phase6/a-baseline-validation-20261024/cpu-single-trial-tests.log)及[命令预览](../../../../outputs/h100/phase6/a-baseline-validation-20261024/launch-preview-single-trial.txt)。本次只调整启动次数，以下保留原实现的完整验证记录。

本次已完成以下检查，汇总见 [validation.json](../../../../outputs/h100/phase6/a-baseline-validation-20261024/validation.json)：

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 新构建、BF16 舍入/来源顺序、训练适配、完整报告与启动门禁 | **22 项 CPU 测试全部通过，无跳过** | [测试日志](../../../../outputs/h100/phase6/a-baseline-validation-20261024/cpu-tests.log) |
| 两版真实接收 runtime 的 SM90 对象、PTX、SASS 编译 | CUDA12.8 编译通过；未启动 GPU | [编译登记](../../../../outputs/h100/phase6/a-baseline-validation-20261024/runtime-agent/validation.json)、[编译日志](../../../../outputs/h100/phase6/a-baseline-validation-20261024/runtime-agent/compile.log) |
| 启动命令预览 | 两次比较、72 窗口、3600 正式步；未创建批次目录 | [命令预览](../../../../outputs/h100/phase6/a-baseline-validation-20261024/launch-preview.txt) |
| 新 profile 解析器对已回传真实 trace 的兼容性 | 原融合 A 与独立 Flux CUDA 的 rank0 trace 均通过，每角色 128 调用 | 汇总登记中的原文件路径及 SHA256 |

CPU 模拟测试包括完整 36 窗口/288 rank 报告，并刻意构造与预期相反的速度顺序，确认仍会正常记录完成；也检查首轮报告不完整会阻止第二次训练、source-only 不能启动 GPU、返回输出别名或覆盖会失败，以及 CPU/GPU 镜像 annotation 不重复计数。

上述历史 trace 只用于验证解析器兼容性，融合 A 的交替生产顺序不作为本实验的原始顺序证据。**当前没有完整链接本次 Flux 动态库，也没有运行基础版的新 GPU kernel 或八卡训练。** 启动命令会在原环境完成完整构建、实际 mapping、数值及路径预检，再进行正式计时。本页没有填入推测性能。
