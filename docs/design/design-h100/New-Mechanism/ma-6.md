# MA6：在 F4 上缩短真实依赖链的下一步设计

**2026-10-25 第二轮代码更新（MA6-v2，尚未八卡实测）：** 本轮已确认继续以 **F4** 为基线。新候选将 decoder 改为可配置的多行 tile，共享物理 tile 的量化选择 mask；反向路径缓存固定 TP 元数据，并补上 `wait-first` 调度对照与逐 rank 主机提交间隙统计。当前入口仍为 `bash tools/h100/phase6/run_ma6.sh --execute`，具体配置、验证边界见[第 12 节](#12-ma6-v2重新减少工作量与反向提交开销)。**下文第 9–11 节的参数、命令和测量是 v1 历史记录；它们不代表 v2 已获加速，旧 build 不能复用于 v2。**

更新：2026-10-25，已纳入修复后完整八卡批次 `ma6-f4-tile-gather-1x-50steps-20261025-160359-496883`。**这次两项优化都没有验证相对 F4 的端到端增量：F4-tile 为 −0.146%［95% CI：−0.551%, +0.242%］；F4-gather 为 −0.876%［−1.257%, −0.482%］，本轮明确退化。** 三组平均步时分别为 **508.637 / 509.381 / 513.096 ms**。批次正常完成，正确性与实际执行路径通过，详见第 11 节。

设计阶段优先验证了反向 AllGather 与 bias 梯度归约的重叠。**本轮确实观察到重叠，但没有兑现为整步收益**：F4-gather 的 2048 次 rank-local backward 中有 2016 次重叠，完整 step 仍比同期 F4 平均多 4.459 ms。因此当前继续保留 F4，不默认叠加这版 gather 调度；当前 tile 配置也不列为已验证加速方案。

第 1–8 节保留当时基于原 F4 trace 的设计依据，原 B6 对应本轮 F4-gather；v1 未实现 S6、N6，v2 新增的 S6 串行对照入口见第 12 节。第 9 节记录三组实现，第 10 节保留首次启动失败与修复，第 11 节给出本轮结果。下文的设计假设不视为已实现的收益，尤其不能将 barrier/NCCL 时长或局部 overlap 直接折算成整步节省。

## 1. 起点是同期 F4，不再从历史 CR2 估算增量

F4 批次为 `ma4-f-1x-50steps-20261025-115124-977713`。完整训练 F4 / CR2 平均为 **501.433 / 509.852 ms/step**，配对收益 **+1.652%［95% CI：+1.297%, +2.034%］**，8 个块全部更快。逐位前向、梯度/参数更新指纹和实际融合路径通过；但仍只有一次独立 trial。见 [MA4 实测](ma-4.md#6-本轮-ma4-f-结果相对-cr2-有端到端增量)及[原始报告](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/report.json)。

下一次应冻结原 F4，使用**新候选/F4 同场比较**。历史 501.433 ms 只说明量级：再减少 1% 约需净省 **5.014 ms/step**；它不是新批次的固定基准。`baseline.md` 的跨批次中位数比也不用于验收新机制，更不能把多个批次的收益相加。

固定当前模型和数值边界：TP8、32 层、hidden=4096、sequence=2048、global batch=4、四个 microbatch、BF16、零 dropout、原 selective mask、原逐来源累加与舍入。每步每 rank 有 Attention/MLP 投影各 128 次，合计 256 次。

## 2. 重新读 F4 时间线：机会与不能相加的时间

此次只读分析覆盖 F4 **全部 8 份训练 trace、2048 次前向投影及其 2048 次反向**。它们来自独立 profile，每 rank 记录一个完整 optimizer step；不是从正式窗口中抽出的无 profiler 步时。

### 2.1. 反向存在可拆开的依赖

当前 [ConsumerGemmRSFunction.backward](../../../../tools/h100/phase6/ma4_f_adapter.py) 的顺序为：

```text
grad_bias = sum_to_size(grad_output)
    → AllGather(grad_output)
    → grad_input GEMM
    → grad_weight GEMM
    → 返回 input/weight/bias/residual 的梯度
```

冻结 [mappings.py](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/megatron/megatron/core/tensor_parallel/mappings.py) 中 `_gather_along_first_dim` 调用默认同步 collective API。这里“同步”主要描述 CUDA 输出消费依赖，不能简单解释为 CPU 等待整个通信结束。

以下先取每 rank 的角色中位数，再取八 rank 中位数，单位 μs；不同指标的中位数不能直接相加还原一次调用。

| 反向指标 | Attention | MLP |
| --- | ---: | ---: |
| bias 梯度归约 kernel | 8.872 | 8.872 |
| AllGather kernel，包含等待等影响 | 494.831 | 565.976 |
| dgrad GEMM | 17.128 | 46.528 |
| wgrad GEMM | 16.768 | 50.160 |
| bias 结束→AllGather 开始 | 4.112 | 4.072 |
| AllGather 结束→dgrad 开始 | 2.704 | 2.736 |

全部 2048 次调用中，AllGather 都在 bias kernel 结束后开始，最小间隔 **1.120 μs**；dgrad 都在 AllGather 结束后开始，wgrad 都在 dgrad 结束后。源码顺序与实际 GPU 链一致。

关键区别是：**dgrad/wgrad 需要完整 gathered gradient，而 bias 梯度只需要本 rank 的 `grad_output`。** bias 归约和 AllGather 都是对该输入的只读消费者，输出地址独立。当前将二者串行，属于值得移除的调度依赖；这还不是全局最长路径已被证明的结论。

### 2.2. 长 AllGather 不等于纯传输瓶颈

不同 rank 的同类通信时长差异很大，不能只展示八 rank 中位数：

| rank | profile GPU span，ms | 本步 bias 归约工作和，ms | backward AG 中位 μs：Attention / MLP |
| --- | ---: | ---: | ---: |
| 0 | 836.798 | 2.275 | 494.399 / 566.543 |
| 1 | 825.218 | 2.278 | 73.792 / 73.888 |
| 2 | 821.110 | 2.279 | 494.017 / 565.408 |
| 3 | 822.391 | 2.276 | 498.015 / 570.416 |
| 4 | 821.104 | 2.282 | 496.048 / 569.377 |
| 5 | 819.210 | 2.279 | 495.744 / 564.561 |
| 6 | 818.808 | 2.310 | 492.368 / 564.864 |
| 7 | 824.284 | 2.295 | 495.263 / 567.119 |

rank1 的 backward AllGather 在提交 API 结束后约 **1.44 / 1.43 μs** 开始；其他 rank 常已提前排队数毫秒甚至二十多毫秒。这支持存在到达不均衡和调度等待，**不支持把 494/566 μs 都当作网络传输或可节省预算**。也不能凭这一次 profile 固定优化 rank1，或未经时钟对齐就跨 rank 相减绝对时间戳。

这些 profile 的 GPU span 为 **818.808–836.798 ms**，中位 **821.750 ms**，明显高于正式 F4 的 501.433 ms。profile 足以观察调用和依赖形态，不能直接给无 profiler 整步做时间分账。

### 2.3. 前向还剩的相邻工作

全部 2048 次 F4 decoder 后面均直接接原 LayerNorm。每 rank 共 260 个前向 LN，其中 256 个是投影后继，另外 4 个是各 microbatch 首层输入 LN。

| 前向指标，μs | Attention | MLP |
| --- | ---: | ---: |
| F4 decoder | 11.296 | 11.776 |
| 原后继 LayerNorm | 18.000 | 17.856 |
| decoder 结束→LayerNorm 开始 | 1.296 | 1.296 |
| decoder 开始→LayerNorm 结束 | 30.568 | 30.896 |

256 个后继 LN 的 kernel 工作之和，每 rank 为 **4.549–4.583 ms**，跨 rank 中位 **4.571 ms**。归一化计算必须保留，因此这不是可删除的 4.571 ms。LN 后到同 stream 下一 GEMM 前还存在 AllGather，后续方案必须追踪到通信/GEMM的开始，不能只停在 LN 结束。

## 3. 第一候选 B6：让 bias 梯度归约与原 AllGather 重叠

### 3.1. 改变依赖，不改变公式

记本 rank 的输出梯度为 `g`，仍使用原参数、原 TP group、原 rank 拼接顺序和原 BF16/FP32 规则：

```text
db = sum_to_size(g, bias.shape)
G  = AllGather(g)
dx = G @ weight
dW = G.reshape(...).T @ saved_input.reshape(...)
dresidual = g
```

`db` 不读取 `G`，`G` 也不读取 `db`。拟议的执行关系为：

```mermaid
flowchart LR
    P[上游 g 就绪] --> A[原 AllGather：通信 stream]
    P --> B[原 bias 梯度归约：计算 stream]
    A --> J[计算 stream 建立完成依赖]
    B --> J
    J --> D[原 dgrad GEMM]
    D --> W[原 wgrad GEMM]
    W --> R[按原边界返回全部梯度]
```

第一版本只调整同一次 backward 内的提交与等待：

1. 准备与原实现相同的 contiguous 通信输入和独立 gather 输出。继续对原 `g` 执行原 `sum_to_size`，不改变它的归约布局、dtype 或算术顺序。
2. 从正确的计算 stream 异步启动原 AllGather，使通信的输入就绪事件位于 bias 归约之前。
3. 在原计算 stream 上执行 bias 归约；不在这里做全设备同步。
4. **在 bias 归约提交之后、首次读取 `G` 之前**调用对应 `Work.wait()`，建立当前计算 stream 的安全消费依赖。
5. 原样执行 dgrad、wgrad，按原接口返回全部梯度。第一版不提前返回 residual/dgrad，不延迟跨层 wgrad，不改变四个 microbatch 的梯度累积顺序。

仅将 `async_op=True` 写上去不够：若先 `wait()` 再做 bias，就恢复串行；若 bias 之后才建立通信输入事件，AllGather 仍可能等它。PyTorch 2.6 的 NCCL 实现分别在 collective 提交与 `Work.wait()` 中建立输入就绪、输出可消费的 stream 依赖，应以实际使用版本核对，不能靠 CPU API 返回时间代替 GPU 完成判断。见 [PyTorch 2.6 collective 语义](https://docs.pytorch.org/docs/2.6/distributed.html#synchronous-and-asynchronous-collective-operations)、[v2.6.0 ProcessGroupNCCL 的 syncStream / synchronizeStream](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp)。

### 3.2. 为什么它比再次缩短 CPU 提交更有针对性

旧 G 主要减少主机提交，但多数调用早已排队。B6 的目标是更改**GPU 必须满足的先后关系**，使通信无需等待一个无数据依赖的 reduction。理想局部模型可由 `B + A` 变为 `max(B,A)`，但只在到达时序、资源使用等其他条件不变时成立。

当前 bias 归约全步工作之和的 rank 中位为 **2.279 ms**，约为正式步时的 **0.455%**。这是直接可探索重叠的工作量级，**既不是保证收益，也不是包含调度变化后的严格全局上界**。它不足以预先承诺额外 1%；若实际改变了最晚 rank 的通信到达，效果可能与局部工作和不同，需要完整训练回答。

这也回应“优化的是否还是已经 overlap 的部分”：现有 trace 中 bias 与该次 AllGather 没有 overlap，拟议改动有明确可观测对象；但它们是否决定最终整步，仍须新旧同场结果检验。若只是早到 rank 更早等待其他 rank，整步可能不变。

### 3.3. 不能省掉的实现约束

- **输入真实就绪。** NCCL 只能读已完成生产的 `g`；若调用方使用其他 stream，需沿用正确 producer→current stream 的依赖，不能为了提前通信跳过它。
- **生命周期。** input、gather output 与 Work 必须活到相关 GPU 使用结束；复用缓冲区时必须有完成边界。首版保留独立输出及原 allocator 管理，不同时引入全局 buffer 池。
- **集体通信顺序。** 所有 rank 使用相同 TP group、相同 shape、同样的逐层/逐 microbatch collective 顺序；只移动该次 bias kernel，不跨其他 collective 重排。同一次 backward 只保留这一项未完成 Work；异常后不能回退并重复发起 AG。
- **保持输入副本事实。** 本次 2048 次 backward 中未观察到 contiguous/clone/copy 或 GPU memcpy，现有 `.contiguous()` 没有实际复制。新路径不能靠新增输入 clone 才工作，也不能把 MLP wgrad 前的 memset 误判为 gather 拷贝。
- **保持完整数值检查。** 归约算子、dgrad/wgrad 的形状、布局和调用保持一致；同读输入并不自动保证调度改动后逐位一致，仍须跑原全参数梯度/更新检查。
- **保持环境作为控制变量。** 首版继续 `CUDA_DEVICE_MAX_CONNECTIONS=1` 和原 NCCL 配置。不同 stream 只提供并发机会，不保证并发；资源争用、隐式同步或调度都可能使 B6 无收益。若要改连接数，另设双方共同改变环境的实验。见 [CUDA 12.4 并发执行与 stream 同步说明](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)。

## 4. 第二候选 N6：先做 Attention 的 decoder＋LayerNorm 融合

这条方向比 B6 更接近 F4 的前向融合，但复杂度也更高。建议先做 **Attention-only**：Attention BDA 后在同一个 `_forward_attention` 中进入 `pre_mlp_layernorm`，接口边界较清楚。MLP 输出则跨到下一层 `input_layernorm`，最后一层进入 `final_layernorm`，应作为后续独立扩展。

### 4.1. 必须保留两个输出

```text
z = BF16(residual + BF16(projection + bias))
y = 原 LayerNorm(z; gamma, beta, epsilon)
```

`z` 仍是下一子层的未归一化 residual，`y` 才是归一化后的计算输入。因此候选必须同时保留 `z` 和 `y`；不能只返回 `y`，也不能声称删除 `z` 的全部写出。LN 统计必须基于已经完成两次 BF16 舍入的 `z`，不能改成对未舍入的 FP32 三项和求均值/方差。

每次 `[256,4096]` BF16 的 `z` 为 **2 MiB**。可能省掉的是 LN 对 `z` 的独立读取及一个 launch/依赖边界；Attention-only 对应每步 128 次、约 **256 MiB 逻辑读取**，覆盖全部后继则约 512 MiB。它们可能命中缓存，不能直接按 HBM 带宽换算节省；原归一化、`z/y` 写出及 backward 保存仍有成本。

### 4.2. 现有 H128 decoder 不能直接附加整行归约

当前每 warp 处理一个 H128 组，每 CTA 四个组；一行 4096 元素有 32 个 H128 组，跨 **8 个 CTA**。`__syncthreads()` 不能跨这些 CTA 完成一行统计。

候选需让一个 CTA 拥有整行，例如 128/256 线程，每 warp 分别循环处理 8/4 个 H128 组，再进行行内归约。这只是待评估布局，不是确定参数。必须保留原 H128 shuffle、逐来源顺序和量化路径，核对寄存器、共享内存、spill 与实际 occupancy。F4 decoder 已为 51 registers，原 LN 为 48 registers；融合后的资源不能按简单相加或取较小值估算。

本批 LN 实际为非 persistent `cuApplyLayerNorm<BFloat16,float,BFloat16>`，epsilon 为 `1e-5`，`apply_layernorm_1p=false`。冻结快照有 Megatron 包装，但尚缺本次服务器 **Apex CUDA LN 源码及版本身份**；实现前需补齐其统计归约树、affine 与舍入规则，不能用本地另一版 Apex 或任意 Welford 算法替代后再放宽验证。

### 4.3. 反向与判定

双输出 `(z,y)` 会带来 residual 分支和 LN 分支两路梯度。需要按原 dtype、相加与舍入语义合并它们，再走 F4 的 bias、AllGather、dgrad、wgrad；同时保持 gamma/beta 的参数身份、梯度、sequence-parallel 属性和 mean/invvar 等保存量。优先研究复用原 LN backward，而不是同时重写前后向算法。

Attention-only 最多消除 128 个独立前向 LN 调用，不能声称消除全部 260 个。应同时测 **decoder 开始→LN 结束→后继 FC1 开始**与整步；如果只让 LN 更早结束，但后继 AllGather/FC1 和完整 step 未提前，就不能认定关键路径已缩短。

停止条件包括：融合后 decoder 的新增成本抵消原边界/读取收益；行级布局明显损伤 decoder 吞吐；原数值门槛无法保持；或完整训练区间跨零。它是值得做受控原型的第二候选，不是已经确认的下一项收益。

## 5. 有更大工作量的 wgrad 延迟，为何暂不放第一位

本次 trace 中，目标 wgrad 的每 rank 全步工作和中位为 **8.569 ms**；dgrad 结束到 wgrad 结束的区间和中位为 **9.622 ms**。从数学依赖看，前一层通常只需要 dgrad，wgrad 只需在参数更新前完成，存在延迟执行的研究动机。

但当前 Function 一次返回全部梯度；仅把 wgrad 放到另一 stream，随后在相同返回点等待，并不会缩短传播路径。真正重叠需要重新安排 autograd 节点、梯度累积 hook、`main_grad` 生命周期和输入保留，可能改变多个 microbatch 的归约顺序。若借用通信等待执行 wgrad，还会争用 SM/带宽。

不能直接用 FP32 GEMM 累加替代原 BF16 wgrad→原累积并宣称逐位等价。也不能让 grad norm、optimizer 或清梯度越过尚未完成的 wgrad。trace 在当前 wgrad 后出现的两个 `aten::add_` 尚未逐项确定数学来源，不据此认定某个一定是 residual 梯度合并。

这一方向可能比单个 bias reduction 有更大机会，但首先要证明**下一层 dgrad 能提前、最终 optimizer 前的尾部没有等量增长、显存峰值可接受**。本轮只列为后续研究，不与 B6/N6 同时实现。

## 6. 暂不重复的路线及理由

| 路线 | 现有证据 | MA6 判断 |
| --- | --- | --- |
| 再缩短 CPU 提交 / 直接叠加 G | 旧 G/CR2 整步 +0.107%，CI 跨零；多数调用有充足排队 | 暂不作为主线，除非新的无 profiler 证据显示关键 rank 持续供给不足 |
| 直接套 MA5 tile/block | CR2 上默认候选整步 +0.069%，CI 跨零；F4 decoder 资源已变化 | 可作以后 F4 专属消融，不能直接移植收益或替换成丢失 bias/residual 的旧 decoder |
| 只融合发布 barrier 与 decoder | F4 barrier 结束→decoder 开始的八 rank 中位约 1.360 μs；每 rank 全步间隙和中位约 0.415 ms | 仅边界机会很小，不能把 barrier 的几百 μs 都记作可删开销 |
| tile-ready＋辅助 stream 提前消费 | 旧 op3-v2 有有效提前消费，O/A 两轮仍慢 0.928% / 0.844%；O/S 也退化 | 暂缓直接复制，必须先找到 F4 上不同于旧实验的可利用尾部 |

op3 的负结果来自另一基线，不证明 F4 上永远无机会；但它已经实现 fragment/epoch 发布、辅助消费并取消原发布屏障，不能把同一思路重新当成未经尝试的方案。见 [op3-v2 实测](../到达优先/base-op3.md)、[MA5 实测](ma-5-tile&block.md#6-默认候选八卡实测结果与离线复核2026-10-25)。

如果以后重启发布路线，关键应是**最晚必需分片到达后的剩余消费尾部**，而不是“多少非关键分片提前完成”。若各 destination 的消费耗时近似相同、后继仍等待全量 TP AllGather，一个简化边界为 `max_destination(max_source ready[source,destination] + consumer_time[destination])`；提前完成早到 destination 可能只增加下一道集合通信的等待。

协议方面还需覆盖量化 payload/scale、未量化远端 BF16 与本地 BF16。Stream-K 的一个 epilogue callback 可能只完成 tile 的 fragment，不能直接宣布整 tile ready；每 destination 有 64 个物理 tile、每 tile 八个 fragment。system release/acquire 必须覆盖实际写入完成，双槽 epoch 和旧消费者结束后才能复用的保证也必须保留。不要让所有驻留消费者 CTA 忙等本地尚未获得调度的 producer，或假设不同 stream 一定同时运行。

## 7. 拟议验证：先区分重排收益与真正重叠

### 7.1. B6 用三组隔离变量

以下为设计阶段的机制归因方案。本次按用户指定，实际运行组调整为 **F4 / F4-tile / F4-gather**，不包含 S6，见第 9 节；因此本批能判断两项优化相对 F4 的净效果，不能独立分离 gather 的重排贡献与异步重叠贡献。

| 拟议组 | 同一次 backward 的顺序 | 作用 |
| --- | --- | --- |
| F4 | bias → 原同步 API 的 AG → dgrad → wgrad | 冻结基线 |
| S6 | 与 B6 相同的异步 AG helper → 立即 wait → bias → dgrad → wgrad | 先提交通信但保留串行，作为 B6 的等待位置对照 |
| B6 | 异步 AG 与 bias 并行 → wait → dgrad → wgrad | 在重排基础上允许重叠 |

三组均保留 F4 前向和相同数学算子，不同时改 GEMM、NCCL 算法、量化 mask、工作区或连接数。

**B6/F4** 预先指定为主要比较，回答用户关心的净增量；**B6/S6** 为机制归因比较，使用相同异步 API/helper，只改变 wait 相对 bias 的位置，判断允许重叠是否有额外贡献。**S6/F4** 包含通信提前提交以及原 helper→显式异步 helper 的差异，不能称为完全隔离的纯重排收益。若需进一步分解，可另加 `bias → 异步 AG → 立即 wait` 的串行桥接控制；不要为统一 helper 而悄悄改掉 F4 基线。

若 B6/F4 为正而 B6/S6 不明确，只能报告组合调度改善，不能单独宣称重叠贡献已被证实。即使 B6/S6 为正，也要结合实际 GPU overlap 证据，区分重叠与提交顺序引起的其他变化。

### 7.2. 三层检查

1. **数值与生命周期：** 独立检查 input/weight/bias/residual 完整梯度和多步更新；沿用 F4 smoke 的每 rank 389 个可训练参数张量、三步梯度/参数指纹比较；正式窗口 loss/grad_norm 轨迹、最终参数指纹一致。测试输入 storage 变化、多个存活输出、慢 rank 与不同模块/双槽复用，记录零 skip/fallback。
2. **依赖与路径：** 每 rank 每步仍是 256 次目标 backward、256 个原 bias reduction 和 256 次相同 AllGather，原 dgrad/wgrad 数量不变。按同一次调用关联 bias、AG、GEMM，检查通信输入事件在 bias 之前、wait 在首次读 `G` 之前；单列 bias/AG 的真实 GPU 区间交集、AG 结束和 dgrad 开始。优先比较**同一上游 `g` 的真实 GPU 生产结束→dgrad/wgrad 结束及后继 backward 开始**；若暂时无法关联这个上游节点，至少使用同次调用 `min(bias_start, AG_start)→wgrad_end` 作为完整局部执行 span。不能继续只从 bias 开始计时，否则 B6/S6 提前执行的 AG 部分被漏掉；也不能用不相关调用的重叠冒充本次收益。
3. **完整训练：** 正式计时关闭 profiler，以每窗口八 rank 最大墙钟/50 计算步时；每窗口独立初始化，10 步预热＋50 步正式。建议探索阶段 12 个完整三组块，每半程包含全部六种块内顺序一次，运行前固定随机种子并冻结完整顺序，共 36 窗口、288 份 rank 记录、1800 正式 optimizer step；全部窗口保留，不按微测速度筛掉负结果。以完整块做配对 bootstrap，不能将八 rank 或 50 步当成独立重复。

B6/F4 的 95% CI 下界大于零、数值与路径通过，才报告该批正增量；之后用新批次独立复测，再谈稳定收益。N6 另开候选/F4 实验，不在第一次 B6 验证中叠加。任何新配置都以其同期 F4 为控制，不能用历史均值凑出百分比。

### 7.3. 收益没有兑现时应如何判断

- **没有发生 AG/bias 重叠：** 先检查事件、wait 位置与环境限制；不能把 `async_op=True` 当作已实现机制。
- **发生重叠但 AG 延长、dgrad 未提前：** 通信与 reduction 的资源竞争或其他 rank 到达可能吞掉了机会。
- **局部 backward 变短但整步 CI 跨零：** 说明这条改动尚未形成可确认的全局增量，保留 F4，不继续用局部百分比作结论。
- **数值或生命周期失败：** 停止候选，不通过减少参数检查、允许隐式回退或放宽到“loss 接近”来过关。

## 8. 证据来源与仍待补齐的部分

分析使用[冻结计划](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/plan.json)、[已有训练 profile 汇总](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/profile-summary.json)以及以下完整 F4 trace：[rank0](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/results/profile-F4/rank0-profile.json.gz)、[rank1](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/results/profile-F4/rank1-profile.json.gz)、[rank2](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/results/profile-F4/rank2-profile.json.gz)、[rank3](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/results/profile-F4/rank3-profile.json.gz)、[rank4](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/results/profile-F4/rank4-profile.json.gz)、[rank5](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/results/profile-F4/rank5-profile.json.gz)、[rank6](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/results/profile-F4/rank6-profile.json.gz)、[rank7](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/results/profile-F4/rank7-profile.json.gz)。

复核方法：先用 `scope.attn.candidate` / `scope.mlp.candidate` 定位前向，再用 CPU `Sequence number` 关联对应 `ConsumerGemmRSFunctionBackward`；在该 CPU 区间内取 CUDA runtime correlation，并与 External ID 共同关联 GPU kernel。逐调用检查 bias/AG/dgrad/wgrad 的数量与时序，各角色每 rank 均为 128 次，再汇总。前向按真实 stream 的相邻 kernel 找 decoder、LN 和下一 GEMM，同时保留其他 stream 的通信。

每次 backward 虽然记录了四个 `cudaStreamWaitEvent`，该原 F4 批次的 Kineto 参数缺少足够的 event handle、源/目标 stream 信息，不能只用这些记录重建完整事件因果图。**本节设计分析证明的是源码依赖和实际执行顺序相符，尚未证明八卡全局最长路径。** 新机制的完整训练结果见第 11 节；若后续要做完整因果归因，仍需补充能关联具体事件/collective 的诊断。

代码边界见 [F4 decoder 生成](../../../../tools/h100/phase6/ma4_f_sources.py)、[F4 adapter/autograd](../../../../tools/h100/phase6/ma4_f_adapter.py)、[原 H128 decoder](../../../../tools/h100/phase6/ma2_c_decode.cuh)、[CR2 提交与屏障](../../../../tools/h100/phase6/ma2_cr2_sources.py)、[冻结 TransformerLayer](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/megatron/megatron/core/transformer/transformer_layer.py)、[冻结 LN 包装](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/megatron/megatron/core/fusions/fused_layer_norm.py)。

NVIDIA 目录中的 [Megatron-Bridge 通信重叠资料](https://github.com/NVIDIA/skills/blob/main/skills/nemo-mbridge-perf-tp-dp-comm-overlap/SKILL.md)面向 Bridge/TE 等集成配置；当前目标是仓库自己的 F4 autograd 路径，不能据此认为开启一个配置就会自动重叠。本次没有安装新技能或修改训练环境。

## 9. 本轮实现：F4 / F4-tile / F4-gather 同场比较

### 9.1. 三组各自改变什么

| 组别 | 前向 decoder | 反向调度 |
| --- | --- | --- |
| **F4** | 原融合 bias/residual decoder，两角色均 `<<<2048,128>>>` | 原 bias reduction → AllGather → dgrad → wgrad |
| **F4-tile** | 默认 Attention `<<<528,256>>>`，MLP 原 `<<<2048,128>>>`；保留全部 F4 消费融合 | 与 F4 相同 |
| **F4-gather** | 与 F4 相同，原生源码逐文件一致 | 先异步 AllGather，再原 bias reduction，wait 后执行原 dgrad/wgrad |

本轮没有 tile＋gather 合并组，便于分别归因。**调节的是 decoder 的 CTA 输出 tile 和 grid，不改变 GEMM 的 `128×128×32` tile 或 SM 设置。** 每 warp 仍负责一个 H128 组，128/256 threads 对应每 CTA 每次处理 512/1024 个元素；grid-stride 循环覆盖全部 8192 组，纯 BF16 路径的提前返回仅结束当前组。

默认采用 MA5 的角色拆分作为一个待测候选，不把它当作已验证最优配置。MA5-R 新回传结果出现同期两组从约 510 ms 跳到约 5.6 秒的窗口，用户已确认当时有其他进程占用 GPU；该批区间跨零，不能作为 F4-tile 已有收益的依据。本次按要求不增加每窗口 GPU 状态记录。Attention、MLP 的 threads/blocks 都可独立指定，所有负收益和慢窗口保留。

F4-tile 按实例固定 K 选择解码入口，K=512 对应 Attention、K=2048 对应 MLP；不在每次 forward 修改 TLS 配置。某角色使用 `128 threads / 2048 blocks` 时，直接保留原 F4 kernel，但实例固定函数选择接口的新增开销仍计入比较。构建记录 tile 参数，原生 ABI 和真实 profile 分别核对选择与实际 launch。

F4-gather 使用第 3 节的调度，bias 仍对原本地 `grad_output` 执行 `sum_to_size`，不对 gathered tensor 归约；输入梯度、权重梯度、残差梯度和更新顺序保持原规则。三组共用相同的实验计数接入；F4 保留原 native、数学与反向顺序，但不把新增公共计数包装声称为完整 Python 文件逐字不变。

代码入口：

- [ma6_sources.py](../../../../tools/h100/phase6/ma6_sources.py)、[ma6_build.py](../../../../tools/h100/phase6/ma6_build.py)：生成独立三组构建、冻结原 F4 和角色参数。
- [ma6_adapter.py](../../../../tools/h100/phase6/ma6_adapter.py)：每模块反向身份与计数、F4-gather 的异步 gather 调度。
- [ma6_bench.py](../../../../tools/h100/phase6/ma6_bench.py)、[ma6_fullstep.py](../../../../tools/h100/phase6/ma6_fullstep.py)：三组正确性、完整训练与实际路径检查。
- [detach_ma6.py](../../../../tools/h100/phase6/detach_ma6.py)、[run_ma6.sh](../../../../tools/h100/phase6/run_ma6.sh)：顺序构建与后台测试。

### 9.2. 启动三组对比

在目标八卡 H100 容器的仓库根目录运行：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_ma6.py --execute
```

环境恢复入口：

```bash
bash tools/h100/phase6/run_ma6.sh --execute
```

默认 **一轮、12 个完整三组块、36 个正式窗口、288 份 rank 记录、1800 个正式 optimizer step**，每窗口 10 步预热＋50 步正式。每半程六种块内顺序各出现一次，顺序在计时前冻结。另有算子正确性、三步完整梯度/更新、训练 smoke 和独立 profile。`--trials 2` 为两轮、72 窗口、3600 正式步；它会运行完整新比较，不从历史结果中拼对照。

去掉 `--execute` 只预览，不创建批次、不查询 GPU或编译；`--foreground` 前台运行。默认结果目录：

```text
logs/h100/phase6/ma6-f4-tile-gather-1x-50steps-<北京时间戳>/
  batch-state.json
  handoff/container-supervisor.log
  operator/report.json
  fullstep/report.json
  fullstep/report.md
  fullstep/profile-summary.json
```

例如要让 F4-tile 两个角色都使用原 MA5 的统一候选，可显式指定：

```bash
bash tools/h100/phase6/run_ma6.sh --execute \
  --attn-threads 256 --attn-blocks 528 \
  --mlp-threads 256 --mlp-blocks 528
```

threads 支持 64/128/256/512；blocks 必须为正且不超过 `8192 / (threads/32)`。每次只测一份明确配置，不自动扫描挑选最佳。`--ma6-build <目录>` 仅复用经过验证、参数与命令一致的 MA6 三组构建；不能混用旧 F4/CR2 或 MA5 构建。`MA6_PYTHON` 可覆盖 shell 入口的 Python 路径。

### 9.3. 看哪些结果

主要比较为 **F4-tile/F4、F4-gather/F4**，并报告 F4-tile/F4-gather。各候选单独记录 `increment_over_F4_validated`，要求每个请求 trial 中相对 F4 的配对点估计及 95% CI 下界都为正；一组失败不意味着另一组失败。数值/路径失败会停止并保留现场，性能负收益则完成并保留全部窗口。

算子前向微测三组都是完整 `projection+bias+residual`，**它无法评估 F4-gather 的反向优化**。gather 的结论必须来自完整训练、实际 backward 调度计数和独立 GPU trace；`async_op=True` 不等于已经发生 GPU overlap，无 overlap 也应如实记录。

反向 profile 需按同次调用识别 bias、AG、dgrad、wgrad，记录 AG/bias 实际重叠，以及 `min(bias_start, AG_start)→wgrad_end` 的完整局部 span。该区间仍不能替代整步验收。正性能结论以关闭 profiler 的全 optimizer step 为准，不能只报告快的 Attention 或剔除后半慢窗口。

本轮不测 C/R，也不验证原相对 C/R 的 2% 目标。与第 7 节设计不同，本次没有 S6，所以即使 F4-gather 有增量，也先表述为“提前发起 gather 并延后等待的整体调度收益”；实际 overlap 及其独立贡献需要分别说明。

### 9.4. 实现阶段的本地验证（历史记录）

本节保留首次实现时的验证记录；它没有覆盖本次发现的冻结 worker 在线 builder 导入路径，新增回归见第 10 节。

实现阶段的本地工作区没有 `/dev/nvidia0`，也没有目标服务器的 `/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python`，当时只做 CPU/源码/编译检查。服务器随后已完成三组八卡实验，结果见第 11 节；本地验证记录本身不构成 GPU 性能证据。

已完成 **74 项 MA6 与原 MA4-F 统一 CPU 检查，全部通过、零跳过**。最后对冻结只读重放的修正又定向复测 **21 项**，对启动器模板哈希冻结复测 **6 项**，均通过。这些复测与统一套件有重叠，不作为新的独立性能样本。覆盖真实 PyTorch CPU 的四类完整梯度和三步更新、异步 API 模拟下的提交/等待与异常处理、三组均衡顺序、独立结果判定、完整参数门槛及冻结脚本隔离导入。

三臂 host mapping、SM90 runtime、GPU mapping 程序及 PTX/SASS 编译通过，F4-tile 的真实 C++ wrapper 和 pybind 对象也编译通过。默认 F4-tile 的 MLP 与 F4-gather 保留的原 F4 decoder **PTX 完全一致**。默认 Attention 新 kernel 为 **44 registers**，原 F4 为 **51 registers**，均为 **0 spill、8-byte stack frame**；资源变化不等于性能收益。

上述本地 GPU mapping 程序仅编译未运行，整套 Flux 动态库链接及八卡数值、路径、性能验证由后续服务器实验补充。实现阶段证据见[验证汇总](../../../../outputs/h100/phase6/ma6-validation-20261025/validation.json)、[统一 CPU 日志](../../../../outputs/h100/phase6/ma6-validation-20261025/cpu-tests.log)、[最终 harness 复测](../../../../outputs/h100/phase6/ma6-validation-20261025/final-harness-tests.log)、[最终 launcher 复测](../../../../outputs/h100/phase6/ma6-validation-20261025/final-launcher-tests.log)、[编译验证](../../../../outputs/h100/phase6/ma6-validation-20261025/native-compile/validation.json)和[启动预览](../../../../outputs/h100/phase6/ma6-validation-20261025/launch-preview.txt)。

## 10. 首次上机失败与修复：冻结 worker 的模块名冲突

失败批次为 `ma6-f4-tile-gather-1x-50steps-20261025-153202-551639`。[批次状态](../../../../logs/h100/phase6/ma6-f4-tile-gather-1x-50steps-20261025-153202-551639/batch-state.json)记录 `failed / operator-timing`，实际停在该阶段的第一个 `preflight-F4`，还未进入正式算子计时或完整训练。三组编译及六个角色 mapping 命令已完成；不能把本次中断解释为 tile/gather 性能退化。

### 10.1. 是代码问题，发生在 worker 使用 GPU 之前

所有 rank 的[原始进程日志](../../../../logs/h100/phase6/ma6-f4-tile-gather-1x-50steps-20261025-153202-551639/operator/results/preflight-F4/process.log)都报：

```text
FileNotFoundError: .../operator/scripts/ma6_build.py
```

错误链为 `worker → verify(offline=False) → build_identity → ma6_build.verify`。冻结 worker 已经导入 `operator/scripts/common.py`，随后虽然从正确的仓库目录导入了 `ma6_build.py`，其 `from common import HERE` 却复用了 `sys.modules` 中的冻结版本，导致 builder 的 `HERE` 指向 `operator/scripts`。嵌套 builder 的 `REPO` 和相关来源模块也有同样的污染风险；只调整 `sys.path` 顺序或手动改一个 `HERE` 无法完整解决。

失败点位于该 worker 的 `import torch`、设备选择和分布式初始化之前。这次报错不是 GPU 占用、OOM 或 NCCL 失败；该失败批次没有产生三组训练性能结论。原失败批次及其冻结脚本保留原样，修复后的新批次另见第 11 节。

### 10.2. 修复方式与新增回归

[ma6_bench.py](../../../../tools/h100/phase6/ma6_bench.py) 将在线构建身份校验放入干净的 `python -I` 子进程，显式从真实 builder 目录加载仓库控制器。子进程核对仓库控制器与冻结 worker 的 SHA256 后，继续执行原完整构建校验并返回同一份身份信息；父进程的冻结模块不被替换。**没有改成 offline，也没有删掉 manifest、输入、源码或动态库的哈希检查。** 这些操作发生在测量窗口之外，不修改前向、反向或 CUDA kernel。

新增 [test_ma6_online_worker.py](../../../../tools/h100/phase6/test_ma6_online_worker.py) 从实际冻结 `worker(args)` 入口执行在线校验，在首次导入 Torch 时停止：三组均须到达这个边界；构建输入摘要、原生源码或动态库被改动时须在 GPU 初始化前拒绝。测试同时复制原失败控制器到临时目录，确认旧代码仍复现相同错误，且新路径不污染父进程的 `common`。

此前的冻结脚本导入和 `offline=True` 测试没有走这条路径，部分 prepare 测试还模拟了 `build_identity`，因此漏掉了错误。新增测试使用 `/tmp` 中的注册源码和明确标记的占位库，只校验路径与哈希，不加载 GPU 库；不能把通过 CPU 回归写成八卡正确性或性能已验证。

修复后 **5 项真实冻结 worker 在线启动回归＋27 项现有相关回归，共 32 项全部通过、零跳过**。原失败证据的 SHA256 在复核前后不变。见[修复验证汇总](../../../../outputs/h100/phase6/ma6-fix-20261025/validation.json)、[在线启动回归日志](../../../../outputs/h100/phase6/ma6-fix-20261025/online-worker-tests.log)、[相关回归日志](../../../../outputs/h100/phase6/ma6-fix-20261025/regression-tests.log)。

### 10.3. 同步修复后，复用服务器已完成的构建重新启动

本次未改动 22 个原生 builder/source 输入，它们与失败批次的冻结哈希一致。服务器保留原 build 时，可在仓库根目录用新批次重跑，省去再次编译：

```bash
bash tools/h100/phase6/run_ma6.sh --execute \
  --ma6-build /workspace/TempName/logs/h100/phase6/ma6-f4-tile-gather-1x-50steps-20261025-153202-551639/build
```

复用仍须通过当前完整构建验证。不要直接运行旧批次 `operator/scripts/bench.py`，它是为保留证据而冻结的旧代码；也不要覆盖旧批次目录。上述命令默认生成新目录并完成同一轮 F4/F4-tile/F4-gather 比较，仍为 36 个正式窗口、1800 个正式 optimizer step。

## 11. 修复后八卡实测：tile 未验证增量，gather 虽重叠但整步变慢

### 11.1. 完整完成，性能未通过不等于运行失败

批次 `ma6-f4-tile-gather-1x-50steps-20261025-160359-496883` 的[状态](../../../../logs/h100/phase6/ma6-f4-tile-gather-1x-50steps-20261025-160359-496883/batch-state.json)为 **`completed/done`**。[启动记录](../../../../logs/h100/phase6/ma6-f4-tile-gather-1x-50steps-20261025-160359-496883/handoff/launch.json)确认复用了首次失败批次已编译的三组构建，并以修复后的控制器创建了新批次；旧批次的数据未混入统计。

实际完成 **一轮、12 个完整三组块、36 个正式窗口、288 份 rank 记录、1800 个正式 optimizer step**。每组各 12 窗口、600 正式步；每窗口独立进程、10 步预热＋50 步正式。另有完整算子检查与计时、48 份算子 trace 和 24 份训练 trace，全部保留。

实测 F4-tile 仅将 Attention decoder 改为 **528 blocks ×256 threads**，MLP 保留原 F4 的 **2048×128**；F4-gather 的两个角色均保留原 F4 decoder。三组 GEMM 参数保持原设置，实际 launch 均为 **512 blocks ×128 threads**。本批没有运行 tile sweep、其他 MLP 配置、S6 或 tile＋gather 合并候选。

### 11.2. 完整 optimizer step：以同期 F4 为准

以下取自[训练报告](../../../../logs/h100/phase6/ma6-f4-tile-gather-1x-50steps-20261025-160359-496883/fullstep/report.json)。步时为每窗口八 rank 最大墙钟 / 50；收益为完整块配对对数比、10000 次 bootstrap，正值表示候选更快。

| 范围 | 块数 | F4 平均 ms | F4-tile 平均 ms | F4-gather 平均 ms | tile/F4 收益 %［95% CI］ | gather/F4 收益 %［95% CI］ |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 前半程 | 6 | 509.279 | 508.974 | 514.558 | +0.061 [−0.468, +0.517] | −1.035 [−1.581, −0.553] |
| 后半程 | 6 | 507.994 | 509.788 | 511.634 | −0.354 [−0.934, +0.228] | −0.718 [−1.249, −0.099] |
| **完整一轮** | **12** | **508.637** | **509.381** | **513.096** | **−0.146 [−0.551, +0.242]** | **−0.876 [−1.257, −0.482]** |

- **F4-tile：** 6 个块更快、6 个更慢，算术均值多 **0.744 ms/step**。前后半程和整体区间都跨零，未验证增量，也不能据此断言该 tile 必然退化。
- **F4-gather：** 1 个块更快、11 个更慢，算术均值多 **4.459 ms/step**。整体及前后半程区间均为负，本轮有明确退化证据。
- 次要比较 **F4-tile/F4-gather 为 +0.723%［+0.268%, +1.165%］**，只说明 tile 组比 gather 组快，不能将其写成 tile 相对 F4 的收益。

三组所有窗口的范围分别为 **504.515–513.174 / 504.349–513.885 / 507.905–519.644 ms**，没有出现 MA5-R 那种约 5.6 秒的窗口；所有窗口原样保留，没有筛选快窗口或重新配对。这里没有新增 GPU 状态记录，不能据此推断本次运行完全没有资源波动。

报告中两个候选的 `increment_over_F4_validated` 均为 false；数值和路径检查通过，所以状态仍为 completed。前后半程是同一次 trial 的拆分，不是两次独立复测。历史 F4 的 501.433 ms 不参与本次收益计算；本批也未同期测 C/R，原相对 C/R 的 2% 目标仍未评估。

### 11.3. 正确性和两项机制确实执行

三组算子输出逐位一致，input/weight/bias/residual 完整梯度及三步更新检查通过。训练 smoke 逐项核对每 rank 的 **389 个可训练参数张量**、三步梯度和更新后指纹；正式窗口的 loss/grad_norm 轨迹及最终参数指纹也一致。三组 mean loss 均为 `3.793363792030723`，final loss 为 `0.00030573460389859974`，final grad norm 为 `0.01800245891878419`。这些是固定合成输入上的等价性检查，不是长期收敛结论。

每组的 96 份正式 rank 记录共有 **6144 份模块记录、12288 份槽记录**。每槽初始化一次，`successful_calls=bias_residual_calls=120`，绑定 K=512/2048，`failed=false`；每模块 forward/backward 均为 240 次，包含预热。F4/F4-tile 每模块原反向调用 240 次、异步 gather/wait 为 0；F4-gather 的异步 gather 和 wait 均为 240 次、原反向调用为 0。64 个目标模块完整覆盖，零 skip/fallback。

真实 [训练 profile](../../../../logs/h100/phase6/ma6-f4-tile-gather-1x-50steps-20261025-160359-496883/fullstep/profile-summary.json) 还核对了各角色的 kernel 名、grid/block、原发布屏障及后继 LayerNorm。三组每 rank 每角色均有 128 次投影，独立 bias/residual add 仍为 0。因此此次无收益不是候选没有启用或退回旧实现。

### 11.4. Tile：更少寄存器和 CTA 没有兑现为收益

[算子报告](../../../../logs/h100/phase6/ma6-f4-tile-gather-1x-50steps-20261025-160359-496883/operator/report.json)计时范围是前向 `projection+bias+residual`，每个角色有 12 个完整三组块：

| 角色 | F4 平均 μs | F4-tile 平均 μs | F4-gather 平均 μs | tile/F4 收益 %［95% CI］ |
| --- | ---: | ---: | ---: | --- |
| Attention | 93.125 | 92.974 | 93.923 | +0.162 [−0.161, +0.493] |
| MLP | 139.555 | 139.709 | 140.387 | −0.110 [−0.379, +0.137] |

独立算子同样未验证 tile 的正增量。训练 profile 先取每 rank 的角色中位数，再取八 rank 中位数，得到：

| 指标，μs | F4：Attention / MLP | F4-tile：Attention / MLP |
| --- | ---: | ---: |
| decoder 本体 | 11.272 / 11.663 | 11.312 / 11.568 |
| decoder 开始→LayerNorm 结束 | 30.464 / 30.656 | 30.384 / 30.504 |

这些局部差异很小，不能将编译时 51→44 registers、2048→528 blocks 直接解释为加速。结论只针对本批默认角色配置；尚不能否定所有 tile/grid 参数，也没有验证它与 gather 叠加后的结果。

F4-gather 的前向微测相对 F4 为 Attention **−0.847%［−1.763%, −0.099%］**、MLP **−0.587%［−1.522%, +0.042%］**。这里没有执行反向 gather 优化，且二者使用相同原生前向源码，所以保留这些观测值，但不把差值归因于反向调度。

### 11.5. Gather：真实 overlap 已出现，但整步更慢

训练 trace 验证了 F4-gather 在每次 backward 中先提交 AllGather，再提交 bias reduction，随后保持 dgrad/wgrad 的正确依赖和顺序。实际 GPU 区间交集为：

| 组别 | Attention overlap / 1024 次 | MLP overlap / 1024 次 | 合计 |
| --- | ---: | ---: | ---: |
| F4 | 0 | 0 | 0 / 2048 |
| F4-tile | 0 | 0 | 0 / 2048 |
| **F4-gather** | **1007** | **1009** | **2016 / 2048，98.438%** |

F4-gather 不是所有调用都发生重叠：rank2 的 Attention 为 125/128，rank5 的 Attention/MLP 为 114/128、113/128，其余对应角色均为 128/128。不能只展示 rank0 的 100%。

以下同样是各 rank 角色中位数的八 rank 中位数，单位 μs：

| backward 指标 | F4：Attention / MLP | F4-gather：Attention / MLP |
| --- | ---: | ---: |
| bias reduction | 8.849 / 8.847 | 9.296 / 9.264 |
| AllGather kernel | 509.463 / 593.767 | 490.047 / 563.400 |
| bias 与 AllGather 实际重叠 | 0 / 0 | 9.216 / 9.152 |
| `min(bias_start, AG_start)→wgrad_end` | 560.807 / 710.535 | 528.407 / 667.734 |
| AllGather 结束→dgrad 开始 | 2.712 / 2.736 | 2.720 / 2.752 |

因此，“调度改了、实际发生 overlap、选定局部 span 变短”三点都有 profile 证据；**“完整训练变快”被本轮同期整步结果否定**。不能把局部 span 的减少乘以 256，抵消或解释正式计时中的 4.459 ms 退化。

三组独立 profile 的 GPU span，八 rank 中位分别为 **834.371 / 825.616 / 836.858 ms**，明显高于正式计时的 508.637 / 509.381 / 513.096 ms。观测开销和调度变化不能忽略，profile 中更短的 kernel 或整体 span 都不能替代无 profiler 的配对训练结果。

仍有两个关键边界。第一，报告明确 `upstream_gradient_ready_event_identified=false`：局部 span 已计入提前发起的 AG，但没有从同一个上游梯度真正就绪点起算，也没有重建八卡全局最长路径。第二，rank 间等待分布发生变化：F4 的 rank2 AG 中位约 **73.360/73.280 μs**，F4-gather 中短 AG 的 rank 变为 rank5，约 **73.616/73.792 μs**；该 rank 的 AG 结束→dgrad 开始反而约为 **85.136/79.632 μs**，八 rank 中位数会隐藏这种差异。

这些数据说明不能只看多数 rank 的局部区间；它们尚不能把退化精确归因于 CPU 提交、GPU 争用、到达不均衡或某一 rank。并发后 bias kernel 略变长，也不足以独立解释整步差值。本轮没有 S6 控制，不能进一步拆分“提前发起通信”和“延后等待允许重叠”的独立贡献。

### 11.6. 离线复核与当前选择

已从原始 rank 记录和全部 72 份 trace 离线重算**算子报告、训练报告、算子 profile 汇总、训练 profile 汇总**，四份结果均与回传 JSON 逐字段一致；冻结控制器及 **60 个本地依赖**哈希一致。原 batch-state、plan、preflight 和报告在复核前后保持原 SHA256。脚本与结果见 [audit.py](../../../../outputs/h100/phase6/ma6-results-audit-20261025/audit.py)、[audit.json](../../../../outputs/h100/phase6/ma6-results-audit-20261025/audit.json)、[audit.log](../../../../outputs/h100/phase6/ma6-results-audit-20261025/audit.log)。

复核只使用拉取的数据，没有重跑 GPU，也没有重新加载或校验服务器动态库本体。当前选择是：**继续使用 F4；默认 F4-tile 尚未证明有增量，当前 F4-gather 不作为性能优化默认启用。** 保留这次负结果和全部原始窗口；单轮结果不外推到其他模型、其他参数或长期训练。

2026-10-26 提交前复查修复了审计入口的版本依赖：`audit.py` 现在校验并加载批次内冻结的 v1 控制器与脚本，不要求当前 v2 工作区仍等于旧代码。默认只读运行，不覆盖上面的 2026-10-25 `audit.json`/`audit.log`；需要保存新复核结果时，可用 `--out` 指定尚不存在的文件。原审计所述 60 个本地依赖匹配是当时的记录，新的重放校验对象为批次冻结脚本。

```bash
python3 -I outputs/h100/phase6/ma6-results-audit-20261025/audit.py
```

## 12. MA6-v2：重新减少工作量与反向提交开销

### 12.1. 从上一轮负结果缩小假设

本轮继续使用 F4 基线，保留 v1 的原始批次和冻结脚本。`baseline.md` 的约 2% 是跨批次中位数比；F4/CR2 的同期配对收益为 1.652%。本轮验收仍只看新候选相对**同期 F4** 的完整训练增量。

旧 tile 只改变一个 warp 处理 H128 group 的调度和 grid，没有减少每个 group 重复执行的 mask 查询。旧 gather 已出现 98.438% overlap，继续提高重叠比例的空间很小；应该检查提前发起通信所增加的主机路径，以及最晚到达 rank 的后继提交。

对历史原始 trace 的进一步只读检查发现，F4 rank2 / F4-gather rank5 的 256 次 backward（合并 Attention/MLP）中位值分别为：

| 指标（μs） | F4 rank2 | v1 F4-gather rank5 |
| --- | ---: | ---: |
| CPU backward | 373.669 | 387.633 |
| AG launch API 结束→dgrad launch 开始 | 118.236 | 150.666 |
| AG GPU 结束→dgrad launch 开始 | 42.930 | 71.463 |
| dgrad launch API 结束→dgrad GPU 开始 | 1.357 | 1.315 |

这说明这些调用的后继间隙主要在主机尚未提交 dgrad 时出现。它只是 profiler 下的研究线索：两组来自独立 profile，不能把差值当作正式整步退化的因果分账，也不应针对固定 rank 编写优化。证据为历史批次的 [F4 rank2 trace](../../../../logs/h100/phase6/ma6-f4-tile-gather-1x-50steps-20261025-160359-496883/fullstep/results/profile-F4/rank2-profile.json.gz) 和 [F4-gather rank5 trace](../../../../logs/h100/phase6/ma6-f4-tile-gather-1x-50steps-20261025-160359-496883/fullstep/results/profile-F4-gather/rank5-profile.json.gz)。

### 12.2. Tile：增加多行工作 tile，复用选择性量化 mask

[ma6_sources.py](../../../../tools/h100/phase6/ma6_sources.py) 新增每角色 `decoder_rows`。原 F4 每个 warp 处理一行的 128 个元素；新候选让一个 warp 处理同一列组的连续 R 行，默认 **R=4、128 threads、512 blocks**，Attention/MLP 分别绑定且默认相同。

```text
原 F4：8192 个 H128 group → 每个 group 独立构造 8 项来源选择 mask
v2：   2048 个 4×128 行 tile → 每个 tile 构造一次 mask，再逐行执行原解码与消费
```

量化选择 mask 的边界是物理 128×128 tile。R 取 1/2/4/8/16，均整除 128；映射确保一个行 tile 不跨这个边界，因此可以共享 `selected[8]` 和 `any_quant`。八项中本地来源直接为 false，只查询七个远端来源的选择表。默认每 CTA 四个 warp、每 warp 四行；512 个 CTA 正好覆盖 2048 个行 tile。源码层面的 mask 求值组数从 8192 减至 2048；实际指令、cache 和耗时仍需测量。

每行仍使用原 H128 shuffle、原逐来源累加、原 BF16 量化路径，以及 decoder 内两次独立的 bias/residual BF16 舍入。行循环保持 rolled，控制代码大小；没有增加跨 warp/CTA 归约。原 F4 kernel 原文保留，显式使用 `rows=1, threads=128, blocks=2048` 时仍调用它。

这里调整的是 **decoder 的逻辑行 tile 和 block 数量**。producer GEMM 的 tile、launch、Stream-K 分片、发布屏障和通信格式不变。改变 GEMM tile 会影响部分和、epilogue 分片与数值边界，不能把它当作同等安全的 launch 调参。本轮也没有减少输出字节、通信量或逐来源算术。多行处理会减少可独立调度的 warp，可能增加寄存器压力或串行尾部；默认参数是一份待测候选，未被称为最优配置。

约束为：`threads ∈ {64,128,256,512}`，`rows ∈ {1,2,4,8,16}`，`1 ≤ blocks ≤ (8192 / rows) / (threads / 32)`。参数在构建中冻结，native 导出 threads/blocks/rows 三项，算子和训练 worker 均核对它们，profile 继续核对实际 kernel/grid/block。无需按一张卡的 SM 数硬编码 528 blocks。

### 12.3. Gather：缓存固定元数据，并补充 wait 位置对照

[ma6_adapter.py](../../../../tools/h100/phase6/ma6_adapter.py) 在每模块首次 backward（预热期间）绑定 TP group、world size、CUDA device 与完整 gather shape。后续调用检查输入 shape/dtype/device/stride 与绑定签名一致，复用固定元数据，减少反复查询 group/world/device 和构造输出 shape 的主机工作。TP group 生命周期固定为整个实验，不能在模块存活期间重建并替换 group。

每次仍保留独立 gather 输出和原 `.contiguous()`，对**原始本地** `grad_output` 执行 `sum_to_size`，再执行原 dgrad/wgrad；不复用跨调用输出缓冲。每模块缓存一份完整 `[2048,1,4096]` BF16 gather 输出会使 64 个模块约常驻额外 1 GiB/rank，跨模块共享又需引入新的 stream 生命周期协议，因此此次只缓存元数据。异常后的 Work/input/output 继续由失败状态保留，不重新发起 collective 或回退。

新 CLI `--gather-schedule` 冻结两种可选调度：

| 值 | F4-gather 顺序 | 用途 |
| --- | --- | --- |
| `overlap`（默认） | cached helper 发起 async AG → 原 bias → wait → 原 dgrad/wgrad | 检验减少主机准备开销后，原重叠是否能带来净收益 |
| `wait-first` | 同一 cached helper 发起 async AG → wait → 原 bias → 原 dgrad/wgrad | S6 串行控制；只改变 wait 相对 bias 的位置 |

`Work.wait()` 的作用仍是在当前 CUDA stream 建立后续消费依赖，不能按 CPU 返回时间判断通信完成，见 [PyTorch 2.6 collective 语义](https://docs.pytorch.org/docs/2.6/distributed.html#synchronous-and-asynchronous-collective-operations)。两种模式保留相同 NCCL、连接数及 collective 顺序。模式写入 plan、生成 adapter 的常量、模块执行状态和报告；不由正式计时期间的环境变量临时切换。`wait-first` 的 profile 额外要求 AG GPU 结束后 bias 才开始。

本轮继续三组 F4 / F4-tile / F4-gather，每次只选一种 gather 模式。**两条命令运行得到的是两个独立批次，不是同一完整块中的 overlap/wait-first 配对；不可直接相减它们的收益来声称已经隔离重叠贡献。** 如果需要严格的 B6/S6 统计归因，应另行扩展为同时包含两种调度的均衡实验。当前 `wait-first` 提供可执行的控制和 trace 检查，不宣称已经运行该控制。

### 12.4. 新增诊断与验收

[ma6_fullstep.py](../../../../tools/h100/phase6/ma6_fullstep.py) 按同一次 backward 的 correlation 关联 launch 与 GPU kernel，每 rank、每角色增加 CPU backward、AG launch 结束→bias/dgrad launch、AG GPU 结束→dgrad launch、dgrad launch→GPU 开始等中位指标。AG GPU 结束→dgrad launch 保留有符号数：负值表示主机提前排队，不能截断成零或误称等待时间。继续记录实际 overlap、AG 结束→dgrad GPU 开始和完整局部 span。

这里新增的 `dgrad_launch_to_gpu_start_us` 从 launch API **开始**计时，包含 API 自身时长；第 12.1 节历史表的最后一行则从 API **结束**计时，二者不可直接混用。

这些指标只帮助检查提交、执行与排队位置；仍未识别统一的上游梯度真实就绪事件，也未重建八卡全局最长路径。最终性能门槛仍是关闭 profiler 的完整 optimizer step，相对同期 F4 的配对点估计及 95% CI 下界在每个请求 trial 均为正。两项候选独立判定，保留全部慢窗口和负结果；不把前向微测用于判断 gather 的反向收益。

构建/算子/训练协议升级为 `h100-ma6-f4-v2` 及对应后缀，native tile 协议为 2。原 F4 native 版本仍为 1。旧 v1 build 缺少行 tile 身份且输入哈希不同，不能通过 `--ma6-build` 复用；旧数据仍用其冻结脚本离线重放，不由 v2 覆盖或重算替换。

### 12.5. 测试入口

在目标八卡 H100 容器的仓库根目录执行：

```bash
bash tools/h100/phase6/run_ma6.sh --execute
```

默认新建 `logs/h100/phase6/ma6-v2-overlap-1x-50steps-<北京时间戳>/`，完成构建、算子正确性/计时/profile、完整训练正确性/计时/profile。仍为一轮 12 个三组完整块、36 个正式窗口、1800 个正式 optimizer step，每窗口 10 步预热＋50 步正式；`--trials 2` 为 72 窗口、3600 正式步。默认后台执行，`--foreground` 可改为前台。无 `--execute` 只打印计划。

串行 wait 控制运行：

```bash
bash tools/h100/phase6/run_ma6.sh --execute --gather-schedule wait-first
```

较小的两行 tile 可作为另外一份预先指定的候选，仍独立测试而不自动择优：

```bash
bash tools/h100/phase6/run_ma6.sh --execute \
  --attn-rows 2 --attn-threads 128 --attn-blocks 1024 \
  --mlp-rows 2 --mlp-threads 128 --mlp-blocks 1024
```

若要对照 v1 的几何配置，须显式指定 `--attn-rows 1 --attn-threads 256 --attn-blocks 528 --mlp-rows 1 --mlp-threads 128 --mlp-blocks 2048`；这只恢复 tile 几何，gather 仍是 v2 的缓存实现，不是 v1 整套复刻。输出依然保存在新批次，不能覆盖旧结果。

优先查看 `batch-state.json`、`fullstep/report.json` 的 `gather_schedule` 和 `increment_over_F4_validated`，以及 `fullstep/profile-summary.json` 的逐 rank 记录。`MA6_PYTHON` 可覆盖解释器。只有完成八卡数值、路径和正式训练后，才能把候选升级为已验证优化；本轮本地环境无 GPU，尚无新的速度结论。

### 12.6. 本轮本地验证

**60 项 CPU 检查全部通过、零跳过**：adapter 14 项、tile/source/build 13 项、harness/launcher/frozen replay 28 项、真实冻结 worker 在线启动 5 项。数值部分使用 PyTorch `2.6.0+cu124` 的 CPU 执行，覆盖两种 gather 调度、BF16/FP32 的 input/weight/bias/residual 梯度及三步参数更新；collective 为 CPU 模拟，不能证明 NCCL 的实际并发。

新增覆盖包括：所有支持的 rows/threads 配合代表性 grid，对 8192 个 H128 group 的唯一完整覆盖、八 rank 的物理 mask 边界一致性、逐行算术原文保持；元数据只绑定一次、tensor signature 漂移拒绝、异常后 pending Work/缓冲保留；wait-first GPU 串行轨迹校验、签名一致但调度不同的 plan/generated adapter 拒绝，以及原在线 builder 导入冲突的失败复现。在线回归为临时旧控制器夹具调整 v1 外层标签/协议，历史文件和 v2 构建本体未改动。

CPU 日志与计数见[验证汇总](../../../../outputs/h100/phase6/ma6-v2-validation-20261025/cpu-validation.json)、[adapter](../../../../outputs/h100/phase6/ma6-v2-validation-20261025/adapter-tests.log)、[tile/build](../../../../outputs/h100/phase6/ma6-v2-validation-20261025/tile-tests.log)、[harness/launcher](../../../../outputs/h100/phase6/ma6-v2-validation-20261025/harness-tests.log)、[在线启动](../../../../outputs/h100/phase6/ma6-v2-validation-20261025/online-worker-tests.log)。在线日志明确标为已捕获工具输出的转录，不计作一次额外复测。

原生检查也已完成：三组 host mapping 通过，SM90 runtime、GPU mapping 程序、PTX/SASS 及真实 C++ wrapper/pybind 对象编译通过；GPU mapping 程序只编译、未执行。三组保留的原 F4 decoder PTX 一致。新 Attention/MLP 行 tile kernel 均为 **48 registers、8-byte stack、零 spill**，原 F4 为 51 registers。这个资源变化不是速度结论；本地未链接完整 Flux 动态库，也未执行八卡数值或性能测试。见[原生编译结果](../../../../outputs/h100/phase6/ma6-v2-validation-20261025/native-compile/validation.json)、[编译日志](../../../../outputs/h100/phase6/ma6-v2-validation-20261025/native-compile/compile.log)和[代码/证据摘要](../../../../outputs/h100/phase6/ma6-v2-validation-20261025/validation.json)。

shell 语法、Python AST 和两种启动计划检查通过；[默认预览](../../../../outputs/h100/phase6/ma6-v2-validation-20261025/launch-preview.txt)及[wait-first 预览](../../../../outputs/h100/phase6/ma6-v2-validation-20261025/wait-first-preview.txt)只打印命令，不创建正式批次。服务器测试结果尚未产生，v1 的负结果和 F4 作为已验证基线的选择保持有效。
