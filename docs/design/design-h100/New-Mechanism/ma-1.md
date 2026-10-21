# MA1：交替排序＋接收端来源预取

日期：2026-10-21。**本页提出并实现一个待八卡 H100 完整训练验证的新机制，不预先承诺加速。** 比较目标是：在相同交替生产顺序、全远端量化、发布屏障、输出所有权和完整训练配置下，提前读取下一来源的数据，能否缩短接收端解码归约及完整 optimizer step。启动命令见第 6 节。

**最新回传：算子预检和计时已完成，原批次在 profile 汇总时因解析错误停止，完整训练尚未开始。** 解析器已修复并通过全部回传 trace 的离线核验；已有算子数据尚未证明 P/W 的预取净收益，详见第 8 节。

## 1. 为什么这次不继续调到达表

[核心思路](../idea.md) 用贡献到达时间拟合分区内排序，但[消融7](../消融实验/base-phase6-ablation7.md)的改进 B 在 Attention、MLP 上都接受了零次交换，完整 step 的 B/A 区间均跨零。[最新 baseline](../base-baseline.md)里 B/A 三次仍没有一致的收益方向。交替顺序已经改善了生产节奏；继续提前某些贡献，未必缩短消费者的实际工作。

现有证据还排除了直接重复下列方案：

- [op3](../到达优先/base-op3.md) 已尝试 fragment/epoch 发布、辅助 stream 提前消费和取消原发布屏障。存在提前消费证据，完整 step 却约慢 0.84%–0.93%。因此 MA1 不再增加生产端发布或轮询工作。
- op3 的 warp decoder 控制 D 在算子上更快，但完整 step 的 D/A 区间跨零。不能把已有的 warp 化重新算作预取的增量收益。
- [op4](../到达优先/base-op4.md) 已尝试量化闭包与合并逆 Hadamard；模型梯度门禁失败。MA1 保留逐来源解码、逐来源 BF16 舍入，不合并来源算术。

MA1 转而利用一个明确的数据依赖事实：**来源 s+1 的载荷和 scale 地址不依赖来源 s 的逆 Hadamard 或累加结果。** 当前来源的数学计算必须有序，下一来源的数据读取可以提前。

## 2. 机制：来源间预取，来源内保持算术

原融合 v2 接收路径可概括为：

```text
全部 GEMM/远端写完成 → 发布屏障
  → 读取来源1 → 解码1 → BF16累加1
  → 读取来源2 → 解码2 → BF16累加2
  → … → 本地贡献 → 复用屏障 → 独立输出
```

MA1 在发布屏障后，每个 warp 负责一个 H128 组，每个 lane 持有 `lane+32*j, j=0..3` 四个系数。四条 32 元素 Hadamard 链互不依赖，按变换层级交错执行；最后两个层级在寄存器内组合。所有单个元素经历的加减、归一化、非有限过滤与尺度除法顺序仍与原 scalar 实现相同。

在同一 warp 内提前发射下一来源的 FP8 系数与两个 FP32 scale 的读取，在计算当前来源时让下一来源的读取继续进行：

```text
先载入来源1
发射来源2读取 → 解码来源1 → 按原顺序BF16累加1
发射来源3读取 → 解码来源2 → 按原顺序BF16累加2
…
提前读取本地BF16 → 解码最后远端来源 → 累加最后远端 → 累加本地
```

来源顺序始终为 `(rank+1)%8, …, rank`。每个远端来源仍先将解码值舍入 BF16，然后与 BF16 累加器相加并再次舍入。本地来源最后加入，不读取未使用的本地 FP8 packet。

实现位于 [ma1_decode.cuh](../../../../tools/h100/phase6/ma1_decode.cuh)，由 [ma1_sources.py](../../../../tools/h100/phase6/ma1_sources.py) 注入隔离构建。只改接收端 runtime 和新增 decoder 头文件；GEMM、交替 swizzle、encoder、packet 格式、Python 前端、C++ wrapper、两次原有屏障和输出 clone 均由同一份 A 源码复制并校验哈希。

这是一条**发布之后、同一 kernel 内的来源读取流水线**，没有 GEMM 与 decode 的跨 kernel 重叠，也没有提前消费尚未发布的数据。这里的“预取”是读入后续计算要用的寄存器，不能等同于更早完成网络传输。

实现使用 inline PTX 的显式加载和编译器约束；NVIDIA 文档说明 `volatile` 与 `memory` clobber 用于约束编译器对 inline assembly 的优化，但不能据此直接认定最终机器指令的时序。因此还保存 PTX/SASS、寄存器和 spill 信息，核对预取是否实际发射在当前来源计算之前。[NVIDIA Inline PTX 文档](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)

## 3. 三组对照与可归因范围

| 组 | 生产顺序 | 接收端 | 远端量化与生命周期 |
| --- | --- | --- | --- |
| A | 基础交替，无到达表 | 原融合 v2 scalar decoder | 全远端 H128/E4M3；单工作区、原配置/复位、输出 clone |
| W | 与 A 相同 | warp decoder，四链交错，逐来源即时读取 | 与 A 相同 |
| P | 与 A 相同 | 与 W 相同的算术、线程布局和加载指令；提前读取下一来源 | 与 A 相同 |

W/P 使用同一个头文件的模板实例，唯一机制开关为预取。W 并不等同于历史 op3 的 D，因为四条独立链的执行调度进一步交错；历史计时不混入本次样本。

- **P/A**：用户关心的“交替＋接收端优化”相对原交替实现的完整收益。
- **P/W**：预取的净增量，只有该比较的配对区间也为正，才能标记本次比较中的预取收益通过。
- **W/A**：warp 布局与独立计算链调度的控制项，单独报告。

本轮“单纯交替”以当前 baseline 的 **A（交替＋全远端量化）** 为参照，而不是改为另一套全 BF16 实现。三组固定相同量化范围，避免把减少 codec 工作量误算成预取收益。若后续要评价全 BF16 或选择性量化，应另建同实现配对实验。

**MA1 的 P 与 [base-baseline.md 中的 C](../base-baseline.md#2-五组与固定条件) 是不同实现；本次不是在 C 上叠加预取。**

| 项目 | baseline C | MA1 P |
| --- | --- | --- |
| 基础实现与排序 | 原 candidate，交替＋原到达表 | 基础交替，无到达表；基于原融合 v2 |
| 量化范围 | 1/64 预算上限及冻结 mask，仅少量远端贡献量化 | 全远端 H128/E4M3 量化 |
| 工作区及输出 | 双工作区，实例持有配置/输出 | 单工作区，原配置/复位，输出 clone |
| 本轮新增机制 | 沿用历史 C 实现 | warp decoder＋来源间预取，由 W 隔离预取增量 |

C 的冻结 mask 实际量化 103/7168 个远端贡献，约 1.437%；其历史 C/A 收益同时包含量化覆盖、codec 和生命周期等差异。MA1 固定 A 的这些条件，验证接收端优化能否在 A 上产生增量。**当前没有 C/P 同场比较结果，不能据此判断 MA1 是否快于 C，也不能拿历史 C/A 收益当作 MA1 的收益。**

当前只支持 `TP8, M=2048, N=4096`，Attention/MLP 的本地 K 为 512/2048。其他接收形状直接拒绝，不静默回退。

## 4. 预期收益与失败条件

预取希望缩短串行的 `load → H128 → fold → 下一来源load` 依赖链。归约本身仍有数据依赖；隐藏的是下一来源的读取延迟，不是消除 BF16 累加的顺序要求。

以下情况都可能使它没有收益甚至变慢：

1. 原 decoder 已通过足够多的驻留 warp 隐藏了读取延迟。
2. 增加的寄存器使 occupancy 降低，或产生 local-memory spill。
3. 数据已在缓存中，瓶颈主要是 Hadamard 指令而非读取。
4. 编译器已把 W 的读取安排得足够早，P 没有进一步改善。
5. 接收端虽然加速，但在完整 step 的占比太小，或 Python/launch/其他训练阶段掩盖了收益。

因此不以源码结构、算子最好单点或跨批次均值作为结论。算子、decoder profile 和完整 step 分别报告；性能为负也继续完成预注册的完整训练比较。

## 5. 实验与验收

所有组使用相同 GPT6.7B：32 层、H4096、FFN16384、32 heads、S2048、TP8，BF16、micro batch1/global batch4，seed1234/token_seed4321，每步 8192 token。覆盖 32 个 Attention 输出投影和 32 个 MLP 输出投影，反向沿用原 BF16/STE 路径。固定 `NCCL_NVLS_ENABLE=0`、`CUDA_DEVICE_MAX_CONNECTIONS=1`。

先运行算子正确性和 profile，再运行**一次完整 A/W/P 比较**：12 个完整三组块，共 **36 窗口、1800 个正式 optimizer step**。每窗口独立初始化，10 步预热、50 步正式。1800 步是独立窗口的累计正式工作量，不是同一个模型连续训练 1800 步；另有 360 步预热。

块内顺序平衡，后半反转。每个窗口用八 rank 最大墙钟除以 50 计算 ms/step；按完整三组块配对 bootstrap 置信区间，不把八个 rank 或窗口内每步当独立样本。报告所有窗口和负收益。只有本次比较的 **P/A、P/W 的 95% 区间下界均大于零**，才标记本次机制收益通过；正确性通过、运行完成与性能通过分别记录。单次比较只能说明本批次的结果，不能声称已经跨独立重复稳定复现。

正确性检查包括算子 A/W/P 输出逐位一致、零值/小值/离群值/随机输入、返回输出不被后续调用覆盖，以及实际交替 mapping。完整模型保留全部 64 个投影的前向误差检查、初始化/RNG/token 一致、有限 loss/grad norm、零 skip/fallback 和真实调用次数，并检查跨臂 loss/grad 轨迹一致。profile 应出现每臂实际 decoder，不能通过切回 A 路径冒充 P。

本实验仍反复训练固定合成微批，短程 loss 不代表真实语料长期收敛。

## 6. 一行启动

同步本次新增 `ma1*.py`、`ma1_decode.cuh`、`detach_ma1.py` 和已有 `tools/h100/phase6` 依赖到原服务器。在 `/workspace/TempName` 内执行：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_ma1.py --execute
```

默认从固定融合 v2 源码生成三份基础交替构建，**仅运行一次 A/W/P 完整训练比较（36 窗口、1800 正式步）**，不重新拟合到达表、不依赖旧实验动态库。CUDA12.8 编译、CUDA12.4 运行、jobs2；使用相邻 `../Megatron-LM`。采用既有双 fork 后台托管，终端关闭后可继续，容器和主机仍须保持运行。

去掉 `--execute` 仅打印计划，不创建输出、编译、fork 或查询 GPU。`--foreground` 用于前台诊断；路径可通过 `--python`、`--megatron`、`--build-cuda-home`、`--cuda-home`、`--nccl-root` 指定。`--ma1-build` 只接受本实验已完成且核验一致的 A/W/P 构建。任一编译、正确性或证据检查失败，保留现场并停止；不会自动跳过或放宽门槛。

输出目录中的 `batch-state.json` 登记整体状态，`handoff/container-supervisor.log` 保存后台日志，`build/` 保存隔离构建，`operator/` 保存算子与 profile 证据，`fullstep/` 保存本次完整训练比较及 `report.json/report.md`。**`completed` 表示实验完成；是否加速需要另看 `target_validated`、`total_benefit_validated`、`prefetch_benefit_validated` 与配对区间。**

完整回传批次后，可仅依靠冻结脚本和原始记录复算报告：

```bash
python3 tools/h100/phase6/ma1_bench.py report /path/to/batch/operator --offline
python3 tools/h100/phase6/ma1_bench.py profile-report /path/to/batch/operator --offline
python3 tools/h100/phase6/ma1_fullstep.py report /path/to/batch/fullstep --offline
```

不要只回传汇总值：保留 `plan.json`、`scripts/`、全部 rank JSON、mapping CSV、profile trace、构建 manifest 和状态文件。

## 7. 当前验证边界

本次改为一次比较后，已重新运行 fullstep/detach 相关 **20 项 CPU 测试，全部通过**；实际启动预览确认 `--trials 1`、36 窗口、1800 正式步及 `1x` 输出目录。此次仅调整实验次数，未启动服务器上的八卡训练。

当前编辑环境不是原 `/workspace` 八卡 H100 运行环境。以下保留机制实现阶段已经完成的本地验证记录，其中启动预览对应当时的三次比较配置；不代表本次改为一次比较后重新执行了这些验证：

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| CPU 数值、构建隔离、mapping、统计、证据拒绝与后台流程 | **56 项全部通过，无跳过** | [验证登记](../../../../outputs/h100/phase6/ma1-validation-20261021/cpu-validation.json)、[日志](../../../../outputs/h100/phase6/ma1-validation-20261021/cpu-tests.log) |
| A/W/P 实际接收 runtime 的 SM90 编译及两种 shape 的 host mapping | 全部通过；保存 `.o`、PTX、SASS | [编译登记](../../../../outputs/h100/phase6/ma1-validation-20261021/runtime-check/validation.json) |
| 独立 decoder 指令检查 | W 31/P 40 寄存器，均 0 stack、0 spill；确认 P 下一来源载入早于当前来源 butterfly，后续才读取对应寄存器 | [指令核验](../../../../outputs/h100/phase6/ma1-validation-20261021/decoder-check/validation.json) |
| 启动预览 | 成功，未创建目标目录、编译或启动 GPU | [命令预览](../../../../outputs/h100/phase6/ma1-validation-20261021/launch-preview.txt) |

W/P 共同使用 `__grid_constant__` 保留只读 kernel 参数，避免动态 peer 索引产生参数的线程本地副本；该项属于 W/A 的公共接收端优化，不算作 P/W 预取增量。指令核验还显示最终本地 BF16 读取可能穿插在最后远端来源的前几级 butterfly 之间，不能宣称全部本地读取都严格早于该来源的第一条计算指令。

本地只验证了编译对象和 CPU 行为，**没有完成三份完整 Flux 动态库的链接，也未运行 GPU kernel、八卡数值检查或训练计时**。SASS 发射顺序不能证明实测访存重叠或速度收益。完整链接、实际路径、逐位正确性及性能由上述服务器启动入口继续验证；尚不能写成“已经比交替排序更快”。

## 8. 首次回传：profile 解析失败，完整训练未开始

回传批次为 `ma1-prefetch-1x-50steps-20261021-210732-317969`，来自提交 `ea771841`。其 [batch-state.json](../../../../logs/h100/phase6/ma1-prefetch-1x-50steps-20261021-210732-317969/batch-state.json) 为 `failed`，停止阶段是 `operator-profile`。

**失败与后台进程托管无关。** [supervisor.json](../../../../logs/h100/phase6/ma1-prefetch-1x-50steps-20261021-210732-317969/handoff/supervisor.json) 记录 `ready=true`、`pid=3426600`、`ppid=1`；GPU worker 在初始化通信前也执行了 `torch.cuda.set_device(local_rank)`。本批已完成三组编译、八卡算子正确性检查、36 个算子计时窗口，以及三组各八 rank、两种 shape 的 48 份 profile trace。完整训练的 `fullstep/` 尚未创建，因此 **36 个完整训练窗口、1800 个正式 optimizer step 均未执行**。

具体错误见 [operator-profile.log](../../../../logs/h100/phase6/ma1-prefetch-1x-50steps-20261021-210732-317969/operator-profile.log)：`Missing or duplicated profiled invocation marker`。实际 trace 同时包含同名的 CPU `user_annotation` 与 GPU `gpu_user_annotation`。旧 `trace_path()` 只按名称和 `ph=X` 匹配，得到两份记录，错误地判定调用标记重复。

修复仅让调用边界匹配 CPU `user_annotation`；实际 kernel 仍通过 launch correlation 核验。CPU 标记确实缺失或重复时仍报错，GEMM、两屏障、decoder、clone 与 stream 顺序检查均保留。**26 项相关 CPU 测试通过，全部 48 份真实 trace、192 次算子调用均通过修复后的解析器。** [离线审计](../../../../outputs/h100/phase6/ma1-profile-fix-20261021/audit.json) 与 [重建的 profile 汇总](../../../../outputs/h100/phase6/ma1-profile-fix-20261021/profile-summary.json) 保存在独立输出目录；原日志、冻结脚本及原失败状态未修改，未重新运行 GPU。

[原算子报告](../../../../logs/h100/phase6/ma1-prefetch-1x-50steps-20261021-210732-317969/operator/report.json) 已离线复算一致，`correctness_passed`、`bitwise_equal`、`actual_mapping_validated` 均为 `true`。批次状态中的 `operator_bitwise_equal=false` 是因为 supervisor 要等 profile 汇总成功后才登记阶段结果，不代表实际出现逐位错误。

| 算子 | P/A 配对耗时下降 % [95% CI] | P/W 预取净增量 % [95% CI] |
| --- | ---: | ---: |
| Attention 输出投影 | +23.377 [+22.992, +23.697] | −0.258 [−0.664, +0.089] |
| MLP 输出投影 | +14.365 [+13.795, +14.894] | −0.517 [−1.369, +0.282] |

正值表示 P 更快。P/A 有正算子收益，但 P/W 两项区间均跨零且点估计为负，**尚无预取相对 W 的额外收益证据**。这些是完整 `GemmRS.forward().clone()` 的算子计时，不能换算成已经获得完整训练收益。

同步修复后的工具到原服务器后，可复用仍保留在服务器上的构建，启动一个新批次，跳过重新编译：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_ma1.py --ma1-build /workspace/TempName/logs/h100/phase6/ma1-prefetch-1x-50steps-20261021-210732-317969/build --execute
```

该命令仍执行一次完整训练比较，并先重新完成算子预检、计时和 profile；不覆盖旧批次，也不修改旧批次的冻结脚本或哈希。构建目录若已删除，则使用第 6 节的默认命令重新构建。
