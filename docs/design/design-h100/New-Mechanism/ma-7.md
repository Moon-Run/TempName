# MA7：CR3 的权重梯度异步旁路

**状态：2026-10-26，机制与对比入口已实现，等待 H100 TP8 实测。本文没有新的 GPU 性能结果。** 新候选记为 `F4-wgrad`，对照 `F4` 就是 [baseline.md](baseline.md) 中的 CR3。目标是让输入梯度继续向前一层传播时，权重梯度在辅助 CUDA stream 上计算；每个 microbatch 的反向结束后统一建立完成依赖，再允许梯度同步和参数更新。

启动入口为 [run_ma7.sh](../../../../tools/h100/phase6/run_ma7.sh)：

```bash
# 在 H100 容器的 TempName 仓库根目录执行
bash tools/h100/phase6/run_ma7.sh --execute
```

这是一项新的**反向调度实验**。前向仍完整使用原 CR3，不叠加 MA6 mask/layout、tile、异步 bias/AllGather 或 CUDA Graph。最终是否缩短约 500 ms 的完整训练步，必须看本轮同期 `F4-wgrad/F4` 的配对结果。

## 1. 为什么选择 wgrad

[MA6](ma-6.md) 已经给出几条有用的边界：

| 已有路线 | 已有证据 | 对 MA7 的意义 |
| --- | --- | --- |
| decoder 的四行 tile | v2 整步 −0.339%，区间为负 | 不继续扩大单个 CTA 的工作量 |
| bias reduction 与 AllGather 重叠 | v2 整步 −0.680%，区间为负 | 不能把较长 NCCL kernel 时长直接当作可利用的空闲时间 |
| packed mask | v3 局部 decoder 约省 1.9–2.0 μs，整步 CI 跨零 | 局部指令缩减不足以证明整步收益 |
| 同 tile 的 warp 布局 | v3 整步 +0.251%［+0.062%, +0.444%］，约省 1.284 ms | 有小幅正结果，但尚未独立复现 |
| decoder＋LayerNorm | 尚缺目标 Apex 统计归约实现的身份 | 暂不在逐位数值门槛下重写 LN |

MA6 第 5 节已指出另一条尚未实现的依赖：**前一层需要 dgrad，不需要当前投影的 wgrad；wgrad 的最终消费者是参数梯度累积和 optimizer。** 原 CR3 将二者放在同一个 backward 的主 stream 上顺序计算，因而 wgrad 会阻塞后继的反向传播。

MA4-F 的独立 profile 中，每 rank 每步目标投影 wgrad 的工作量之和中位数为 **8.569 ms**，dgrad 结束到 wgrad 结束的区间和中位数为 **9.622 ms**，见 [MA6 第 5 节](ma-6.md#5-有更大工作量的-wgrad-延迟为何暂不放第一位)。这些来自有 profiler 的历史运行，是选择研究方向的依据，**不是本候选可兑现的节省量，更不是已经确认的关键路径占比**。

以历史 CR3 的 501.433 ms 作量级参考，1% 净收益约需要 5.014 ms，2% 约需要 10.029 ms。即使粗略地把全部 8.569 ms 隐藏，直接 wgrad 工作量的尺度也只有约 1.71%；真实收益还要扣除依赖事件、提交、SM/带宽争用和末尾等待。因此不预先宣称降到某个绝对步时，也不将历史各项收益相加。

## 2. 机制：拆开传播依赖，在 microbatch 末尾收口

### 2.1. CR3 当前顺序

原 [ConsumerGemmRSFunction.backward](../../../../tools/h100/phase6/ma4_f_adapter.py) 为：

```text
主 stream：bias reduction → 原 AllGather 的消费依赖 → dgrad GEMM → wgrad GEMM
                                                                  ↓
                                     返回梯度，继续前一层 backward
```

数学计算为：

```text
G = 原 _gather_along_first_dim(grad_output.contiguous())
dX = G.matmul(W)
dW = G.reshape(-1, H).T.matmul(X.reshape(-1, K))
```

训练路径中的原 DDP 随后执行 `W.main_grad.add_(dW)`。本实验固定 BF16 的 dW GEMM 输出及原累积目标 dtype，不改用直接 FP32 GEMM 累积。

### 2.2. MA7 顺序

```text
主 stream：bias → 原 AllGather → dgrad → 返回 dX/dbias/dresidual → 上游 backward …
                                     │                                      │
                                     └─ 辅助 stream 等待已提交的主 stream    │
辅助 stream：                            dW GEMM → 原 main_grad.add_(dW) …  │
                                                                            │
microbatch backward 返回后：主 stream 等待辅助 stream 已提交的全部工作 ←───────┘
随后：下一 microbatch，或 finalize_model_grads → optimizer → 清梯度
```

具体规则：

1. 每个 rank 的 adapter 使用**一条共享辅助 stream**。64 个目标模块的任务按实际 backward 提交次序入队；不为每层创建 stream，也不跨 microbatch 延迟到下一步。
2. bias reduction、原 AllGather helper、dgrad 的主 stream 顺序不变。dgrad 提交后才在辅助 stream 建立输入就绪依赖；第一版不尝试让同一次投影的 dgrad/wgrad 同时启动。
3. 辅助 stream 执行与 CR3 相同的 `transpose → matmul` 表达式，得到原 dtype 的 dW，然后对绑定的原 `main_grad` 做相同 `add_`。返回 autograd 的 weight 梯度为 `None`，避免再执行一次原 DDP 累积；其余三类梯度照常返回。
4. 包装当前 Megatron 的 `schedules.backward_step`：先完整调用原函数，再为辅助 stream 建立等待。这样每个 microbatch 的传播链中没有逐投影的 wgrad 等待，整个 backward 返回的边界又包含所有权重梯度。
5. `finalize_model_grads` 之前检查队列已收口。梯度同步、grad norm、optimizer、参数指纹读取和清梯度都不能越过这个依赖。

这里的“收口”是 CUDA stream 依赖，不是在每层调用 `cudaDeviceSynchronize` 或强迫 CPU 等待 GPU。GPU 实际能否并行仍取决于硬件资源和提交顺序，应通过独立 profile 核实。PyTorch 对辅助 stream 的就绪依赖和分配生命周期要求见 [CUDA stream 文档](https://docs.pytorch.org/docs/2.6/notes/cuda.html#cuda-streams)。

### 2.3. 为什么不只把返回的 grad_weight 放到另一条 stream

若仍从原 autograd 节点返回辅助 stream 上的 grad_weight，下游 AccumulateGrad 的 stream 依赖可能立即把主 stream 再次挡住。原 DDP 在初始化时已创建权重叶子累积节点；改变一个 matmul 所在 stream 并不自动拆掉传播依赖。

MA7 显式管理限定范围内的 dW 和原梯度缓冲区，并在可核查的 microbatch 边界收口。这个选择增加了生命周期与集成责任，但可以明确验证“每个投影没有把 wgrad 等待重新插回主 stream”。PyTorch 的相关实现依据为 [InputBuffer 跨 stream 等待](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/autograd/input_buffer.cpp)和 [AccumulateGrad](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/autograd/functions/accumulate_grad.cpp)。

## 3. 数值和生命周期边界

### 3.1. 完整保留的计算

两组 native 源码都由注册的 MA4-F 变换生成，逐文件哈希相同；保留同一 producer、交替排序、原选择 mask、原 1/64 预算、FP8 packet、逐 source 归约顺序、两次 BF16 bias/residual 舍入、独立输出所有权和原 LayerNorm。Attention K=512、MLP K=2048，decoder 仍是 2048 blocks × 128 threads。

反向仍使用原 bias reduction、同一 AllGather、同一 dgrad 和 wgrad 表达式。每参数每 microbatch 的 BF16 dW 先形成，再向原梯度缓冲区累积，四个 microbatch 的累积次序不变。不能把四个 microbatch 拼成一个大 GEMM，也不能将 BF16 舍入移到累积之后。

### 3.2. 队列必须维护的约束

- 参数、`main_grad`、设备和原生投影实例绑定后保持身份；buffer 替换、重复提交、重复 backward 或失败后重试均应报错。
- 共享辅助 stream 按提交次序执行；输入 X、AllGather 输出 G 和梯度目标的存储在异步使用结束前受保护。不能仅保存裸指针后让 allocator 提前复用。
- 主 stream 的清零、原输入生成和 AllGather 必须先于辅助 stream 读写。末尾等待反向保证下一 microbatch、梯度同步和下一次清零都发生在 dW 写入之后。
- 原 F4 组不使用辅助 stream，也不手动累积 dW。候选必须有完整异步提交、累积、收口计数，不能回退到原串行 backward 后仍报告候选成功。
- 仅支持当前固定实验：TP8、PP=CP=DP=1、32 层、hidden=4096、sequence=2048、global batch=4、四个 microbatch、BF16、零 dropout，无 recompute/CUDA Graph，`gradient_accumulation_fusion=False`、`overlap_grad_reduce=False`、`overlap_param_gather=False`。超出边界需要重新设计集成，不能直接复用本实验结论。

辅助 stream 会延长 X/G 的使用期。单次 G 为 2048×4096 BF16，即 16 MiB；Attention/MLP 的 X 分别约 2/8 MiB。如果一个 microbatch 的 64 个投影任务全部未释放，仅这些逻辑张量就约 1.31 GiB，另有 dW/库工作区；其中 X 本来就是保存的激活，不能把这一数值直接称为新增显存。实现需要依赖存储生命周期保护，运行时仍应观察峰值显存和排队尾部。显存不足或尾部等待抵消收益均应作为负结果保留。

## 4. 实现与实验组

| 文件 | 作用 |
| --- | --- |
| [ma7_adapter.py](../../../../tools/h100/phase6/ma7_adapter.py) | 原 CR3 consumer、辅助 stream 的 wgrad 队列、梯度目标绑定及 microbatch 收口 |
| [ma7_build.py](../../../../tools/h100/phase6/ma7_build.py) | 从原 F4 生成两组字节相同的 native 源码，冻结 build 输入与动态库身份 |
| [ma7_bench.py](../../../../tools/h100/phase6/ma7_bench.py) | 前向逐位、四类梯度、三次更新及原前向路径检查 |
| [ma7_fullstep.py](../../../../tools/h100/phase6/ma7_fullstep.py) | 完整模型梯度/更新预检、同期完整 step 配对计时及反向路径分析 |
| [detach_ma7.py](../../../../tools/h100/phase6/detach_ma7.py) | 构建、预检、计时、profile 的后台监督和结果审计 |

协议为 `h100-ma7-f4-wgrad-v1`，算子与训练分别追加 `-operator` / `-fullstep`。只有 `F4` 和 `F4-wgrad` 两组；原 MA6 代码和历史日志继续保留。

两组使用相同的 Python 覆盖计数和 microbatch 边界检查，以免把审计开销只放在候选一侧。F4 的数学 backward 直接调用原实现，边界没有辅助 stream 等待；候选的额外提交、存储保护、累积和等待全部包含在正式计时中。因此仍需比较本批的同期 F4，不能用历史 501.433 ms 减去候选新测值。

算子阶段的计时范围仍是完整前向 `projection+bias+residual`。由于两组前向相同，它用于检查基线和执行身份，**不是 wgrad 加速指标**。算子预检另行运行实际候选 backward，在读取权重梯度之前显式收口，逐位检查四类梯度及三次更新。完整训练才使用 Megatron 的原 `main_grad` 和真实 microbatch 边界。

## 5. 怎样判断机制是否有效

### 5.1. 正确性先行

算子预检要求原选择 mask 下前向逐位一致，input/weight/bias/residual 四类梯度及三次参数更新一致。完整模型继续比较每 rank 全部 **389 个可训练参数张量**的梯度和更新指纹，不只检查被改写的 64 个权重；正式窗口比较同配对块的 loss、grad_norm 轨迹和最终参数指纹。

任何预检失败都停止后续正式比较，不放宽容差、不替换成只看有限 loss，也不把跨 stream 的数值差异视为可接受噪声。

### 5.2. 用 profile 核实依赖，而不把 kernel 时长相加

两组前向都必须实际运行原 F4 producer、发布屏障和 fused decoder，随后进入原 LayerNorm。反向按 forward/backward 的 autograd sequence 和 CPU launch correlation 关联每个投影：

- F4 的 bias、dgrad、wgrad 在原 compute stream，保留 bias→AllGather→dgrad→wgrad 顺序。
- 候选 bias/AllGather/dgrad 顺序相同，wgrad 必须位于另一条 compute stream；开始时间不能早于它的输入就绪。
- 检查实际 wgrad 和梯度累积覆盖，记录 `ma7.wgrad`、`ma7.drain` 边界，确认每个 microbatch 收口，且没有逐投影的主 stream 反向等待。
- 报告 wgrad 与主 stream 后续工作实际相交的次数/时间，以及尾部等待和资源争用的线索。存在另一条 stream 不等于有重叠；有重叠也不等于完整 step 更快。

profile 是独立运行，不能用有 profiler 的 GPU span 代替正式窗口步时。NCCL kernel 的长驻留时间包含等待，不能将“重叠的 NCCL 时长”直接算成节省。

### 5.3. 同期完整 step 是主指标

默认一轮 **8 个完整两组配对块、16 个正式训练窗口、128 份 rank 记录、800 个正式 optimizer steps**。每窗口 10 步预热＋50 步正式计时；正式计时覆盖前后向、所有队列收口、梯度同步、参数更新和清梯度。每窗口步时取八 rank 的最大总墙钟 / 50。

两组顺序平衡，所有完整块、慢窗口和负结果全部保留。配对收益按完整块的对数耗时比及 10000 次 bootstrap 得到 95% CI，正数表示 `F4-wgrad` 更快。`--trials 2` 为两轮、32 窗口、256 份 rank 记录、1600 正式步；不能把一轮的前后半程称作独立复现。

只有数值与执行路径门槛均通过，且每个请求的独立 trial 中配对点估计与 CI 下界都为正，才标记 `increment_over_F4_validated["F4-wgrad"]=true`。单轮仍是探索性结果。另行检查 profile 是否真的支持“传播继续与 wgrad 重叠”这一解释；整步正收益不自动证明某个具体争用或等待归因。

停止或回退到 CR3 的条件包括：数值不一致；队列/收口未覆盖；实际没有预期旁路；显存或辅助 stream 开销明显增加；末尾等待、通信或 dgrad 争用抵消收益；完整步时区间跨零或为负。不能因为候选名称是“异步”就默认启用它。

## 6. 启动与查看结果

```bash
# 只预览命令，不构建、不创建批次、不启动 GPU worker
bash tools/h100/phase6/run_ma7.sh

# 默认一轮，后台执行完整流程
bash tools/h100/phase6/run_ma7.sh --execute

# 两轮独立比较
bash tools/h100/phase6/run_ma7.sh --trials 2 --execute

# 需要前台等待时
bash tools/h100/phase6/run_ma7.sh --foreground --execute

# 只复用当前 MA7 协议、源码和动态库身份均通过验证的 build
bash tools/h100/phase6/run_ma7.sh \
  --ma7-build /absolute/path/to/verified-ma7-build --execute
```

沿用既有 H100 容器恢复环境；默认 CUDA 12.8 构建、CUDA 12.4 运行。启动时打印实际批次路径，依次完成构建、算子预检/计时/profile、训练预检/计时/profile。查看：

- `batch-state.json` 与 `handoff/container-supervisor.log`：进度、失败阶段和完整流程结果。
- `operator/report.json`、`operator/profile-summary.json`：前向、四类梯度和原 native 路径。
- `fullstep/preflight.json`、`fullstep/report.json`：完整梯度/更新与正式同期配对收益。
- `fullstep/profile-summary.json`：辅助 stream、wgrad/收口实际执行证据。

训练报告另含 `peak_cuda_memory_gib`，分别汇总每窗口八 rank 最大 allocated/reserved 峰值的均值与最大值；它衡量异步保留存储的实际代价。

`completed` 只表示实验流程完成，性能为负仍会产出完整报告。应以 **`batch-state.json` 完成全部阶段后的验收字段**和原始配对 CI 为准：训练 `report.json` 的统计先于独立 profile 生成，仅该文件中的正收益尚不代表后续路径门槛已经通过。不能把“脚本退出成功”当作“已比 CR3 更快”。

## 7. 本地验证边界

当前工作环境可使用 PyTorch 2.6.0＋CUDA 12.4 的 Python，但没有可见 GPU。CPU 数值/调度测试、冻结源码与 build 身份测试、离线报告隔离、启动预览可在这里完成；实际 H100 的完整链接、CUDA stream 并发、NCCL 路径和整步性能仍需运行上述入口。

已完成的检查：

- 通用套件 **44 项：35 通过、9 项因系统 Python 无 PyTorch 跳过**，见 [cpu-tests.log](../../../../outputs/h100/phase6/ma7-validation-20261026/cpu-tests.log)。
- 在实际 PyTorch 环境补跑 adapter，**12/12 通过、无跳过**，覆盖上述 9 项、四类梯度、三步更新、四 microbatch 的 BF16→FP32 累积和边界/失败检查，见 [torch-tests-nosite.log](../../../../outputs/h100/phase6/ma7-validation-20261026/torch-tests-nosite.log)。本机一次普通隔离 Python 启动在输出版本/测试信息前超时；成功记录使用 `-S` 加显式 site-packages 路径，完整命令保存在验证 JSON 中。
- 假 CUDA stream 测试确认提交时只有辅助 stream 等待主 stream，收口时才反向等待，并检查存储生命周期保护；这不是实际 GPU 并发验证。
- 两臂注册源码身份、在线冻结 worker 的源码/动态库变更拒绝、冻结训练控制器的只读导入、正负性能判定及 trace 的缺失等待/错误 stream 拒绝均通过。
- 实际 shell 两轮预览显示 **32 窗口／1600 正式步**，没有创建批次目录；shell 语法与生成的 worker/adapter 源码解析通过。

完整验证记录见 [validation.json](../../../../outputs/h100/phase6/ma7-validation-20261026/validation.json)，其中 `gpu_used`、`gpu_validated`、`performance_validated` 和 `linked_runtime` 均为 `false`。在八卡结果回传前，MA7 只是一项有实现和严格验收流程的候选机制；历史 501.433 ms 继续是 CR3 的历史结果，不写入未经测量的 MA7 步时。
