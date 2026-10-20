# op2：停止默认使用静态到达排序，先缩短串行解码关键路径

更新：2026-10-20。验收对照按本次确认固定为**现有消融7的 A 组：交替重排＋全远端量化**。本机没有 GPU，用户将在 H100 环境运行新实验。本文给出原因判断、已经实现的修改和测试入口；**尚未获得新 GPU 数值或性能结果，不能声称已达到 1.5%。**

## 1. 判断：不是表没生效，而是排序目标与执行结构不匹配

对 [idea.md](../idea.md)、[消融4](../base-phase6-ablation4.md)、[消融7](../base-phase6-ablation7.md)、[op-1.md](op-1.md) 和 [最新两次实测](base-op1.md) 的综合判断是：

1. **没有发现到达表没有接入、排列方向反了、物理 tile 丢失等功能性故障。** 消融4已经大幅改变顺序；最新 v3/1ε也在 GPU 上执行了非 identity 排列，两者均无稳定净收益。
2. **当前静态到达优先不适合作为默认加速机制。** 它试图让局部贡献更早到齐，但消费者必须等待整个生产阶段和全 rank 发布屏障。局部到达改善通常没有直接缩短消费者的开始时间。
3. **执行代码确实有可优化的开销。** A 组采用每 H128 组一个 CTA 的全量解码器，存在大量跨 warp 同步。op2 优先优化这条确定会执行的串行路径，而不继续调低排序门槛。

这不是对所有到达感知算法的否定。真正“到达即消费”需要改变生产者与消费者的依赖粒度；当前代码没有这种机制。op2 明确把**解码优化**作为主候选，把旧到达表保留为独立诊断，二者不混淆归因。

`op-1.md` 开头“v3 尚未运行”的表述属于旧时间点；本次以 `base-op1.md` 中两次已完成的新结果为准，旧文件及原始实验未改写。

## 2. 实验已经排除了哪些解释

| 实验 | Attention / MLP 改变槽位数，每角色4096 | B/A 完整 step 配对耗时下降 | 含义 |
| --- | ---: | --- | --- |
| 消融4，两次60步 | 2491 / 2422 | −0.185%、−0.135%，区间均跨零 | 大幅旧重排没有稳定收益 |
| 旧消融7及 op1/6ε | 0 / 0 | 微小正负差异，区间跨零 | identity 对照不能证明新排序有效 |
| op1/v3/1ε，两次50步 | 119 / 42 | −0.111% [−0.435%, +0.192%]；−0.098% [−0.379%, +0.160%] | 已实际重排，仍无稳定收益 |

本轮独立重算了最新批次的 **576 份 rank 记录、72 个计时窗口和8份 A/B mapping**。所有窗口的 `max(rank elapsed) / 50` 与原报告一致。复核脚本和输入哈希见 [audit.py](../../../../outputs/h100/phase6/arrival-op2-cpu-20261020/evidence/audit.py)、[audit.json](../../../../outputs/h100/phase6/arrival-op2-cpu-20261020/evidence/audit.json)。

两次 A 组平均为 **534.264115 ms/step**，B 为约534.821 ms/step。最新 v3 的两遍训练预测改善没有在留出诊断及整步计时中得到稳定支持；“重排数量较少”也不是充分解释，因为消融4已经重排了约六成槽位。

还存在两个建模问题：

- 排序标签来自 **BF16 发布采样**，执行对照是**全远端量化**。编码成本、访存与传输行为不同。
- 各 tile 的稀疏样本来自不同调用。PAVA 趋势和残差并不构成同一次执行的联合时间线；把某个 tile 移到新槽位后，其残差也未必保持。

因此，继续扩大位移或降低 ε 倍率可能得到更多置换，但没有理由据此预期达到 1.5%。

## 3. 为什么“更早到达”没有直接转换成耗时下降

A/B 的实际路径为：

```text
GEMM＋远端编码写入
        ↓
全 rank 发布屏障
        ↓
H128 解码＋按来源顺序归约
        ↓
全 rank 工作区复用屏障
        ↓
输出 clone／后续计算
```

对最新两次、A/B、rank0/rank4的8份 profile 检查，**2048/2048 次投影都在同一 stream 串行执行上述阶段，GEMM/decode 重叠为0**。核验范围仅限这些 trace。

设 `J(t)` 是某个输出 tile 的最后贡献到达时间。只减少某个 `J(t)`，但没有减少整个 GEMM 的完成时间、最终发布等待或后继解码时间，整次投影就未必提前结束。在固定目的地轮转、固定 tile 数、固定编码字节数的条件下，分区内部置换剩余的作用主要是局部性和尾部资源争用，不能视为必然的通信计算重叠。

op2 **保留两道跨 rank 屏障**。删除它们需要发布与获取的内存可见性、fragment 完整性、跨调用代数、工作区覆盖防护，以及消费者不会占满 SM 导致生产者饿死等协议。没有这些实现和 GPU 验证，直接删屏障不能作为优化。

## 4. 新思路：先优化必须经过的消费者，再独立检验排序

### 4.1 主候选 D：一个 warp 独立解码 H128

旧 A 的 `taco_decode_ring_kernel` 中，128个线程共同解码一个128系数组。Hadamard 的前5级在 warp 内完成，后2级通过共享内存交换，单个远端来源执行5次 `__syncthreads()`；TP8 有7个远端来源，来源循环另有8次同步：

```text
每个 CTA 的动态块同步数 = 7 × 5 + 8 = 43
H128 组数 = (2048 / 8) × (4096 / 128) = 8192
旧解码 launch = 8192 CTA × 128 threads
```

新 [decoder.cuh](../../../../tools/h100/phase6/arrival_op2/decoder.cuh) 让 lane `l` 持有 `l, l+32, l+64, l+96` 四个系数：

1. 前5级仍按原顺序执行 warp shuffle。
2. 第6、7级在同一线程的4个寄存器值之间执行蝶形运算。
3. 每个 warp 拥有完整 H128，CTA 内的4个 warp 独立处理4个组。
4. TP8 主 shape 的 CTA 数降到2048，解码内部不再需要块同步。
5. 固定 TP2/4/8 分派，TP8远端来源循环保持紧凑；本地贡献仍最后加入。

保持以下语义：

- GEMM、epilogue 编码代码、E4M3转换、两份 scale、packet 布局和全部远端量化范围；
- 来源顺序 `rank+1, …, rank-1, rank`；
- **每个来源解码后先舍入到 BF16，每次相加后再次舍入到 BF16**；
- N 尾部仍解码完整128个系数，只屏蔽越界输出；
- 同一个工作区、逐次配置/复位、输出 clone、发布及复用屏障。

这复用了仓库已有 warp Hadamard 的实现思路，移植到 A 的固定全量量化路径；不是新的 codec，也不是新的到达调度算法。新头文件只适用于全远端量化，不用于带选择性 mask 的 C。

### 4.2 三组对照及归因

| 组 | 生产顺序 | 解码器 | 与谁比较／解释 |
| --- | --- | --- | --- |
| **A** | 原交替表 | 原128-thread CTA解码 | 用户指定的消融7 A基线 |
| **D：预先指定的主候选** | 与A完全相同 | 新warp解码 | **D/A：解码执行优化收益，验收1.5%** |
| **R：诊断组** | 冻结v3/1ε到达表 | 与D完全相同 | **R/D：到达排序的额外收益** |

三组均为全远端量化，Attention/MLP 都启用。本轮没有借用 C 的稀疏 mask、双工作区或其他生命周期优化。R 的 BF16 校准局限仍然存在。

如果 D/A 达标而 R/D 无收益，结论是“串行解码优化有效，静态到达表仍无额外证据”。即使 R/A 更快，也不能把全部差距归因于到达优先。不会跑完后从 D/R 中择优改写主比较。

### 4.3 构建确实改变运行库

[arrival_op2_build.py](../../../../tools/h100/phase6/arrival_op2_build.py) 从原消融4构建中复用 A 的已登记运行库，独立构建 `decode_interleaved` 和 `decode_arrival`：

- D 相对 A 只新增解码头文件，并替换 `taco_decode_reduce()` 的调用；原 scalar kernel保留以便核对。
- R 相对 D 只改变主机/设备映射表。
- 源码哈希核验只允许这些差异；编码、wrapper及屏障发生变化会拒绝构建验证。
- 不修改 `fused_v2/`、历史运行库、旧实验入口或旧报告。直接修改仓库当前 `src/` 也不会被误当成此实验已生效。

## 5. 1.5%目标需要节省多少时间

本实验将“快1.5%”明确为**完整 optimizer step 耗时下降至少1.5%**：

```text
下降率 = 100 × [1 − exp(mean(log(T_D / T_A)))]
```

按历史 A 的534.264115 ms/step估算，需节省：

```text
534.264115 × 1.5% = 8.013962 ms/step
32层 × 2种投影 × 4微批 = 256次投影/step
平均每次投影需节省 31.304538 μs
```

526.250154 ms只是代入历史均值的描述性参考，**正式验收与新批次同场 A 比较**，不能用跨日期的旧均值判成功。

历史最新 A 的 profile 解码均值约48.770—61.278 μs/次，说明解码值得测量；但 profile 有扰动，也包含 rank 差异，不能据此预测新解码耗时或给出加速下界。C 的解码更快还包含大量 BF16 分支，不能拿 C 的时间当 D 的预测。

若令两种投影分别节省 `Δattn`、`Δmlp`，设计预算约为：

```text
128 × (Δattn + Δmlp) ≥ 8013.962 μs
即两角色每次节省量之和 ≥ 62.609076 μs
```

这只是预算；通信等待转移、launch成本及后续依赖都可能改变整步收益。

## 6. 本地验证与 GPU 检查范围

已经完成的 CPU/编译验证及输出见 [arrival-op2-cpu-20261020](../../../../outputs/h100/phase6/arrival-op2-cpu-20261020/)：

- **49项回归通过**：26项新协议与批次测试，加23项既有消融4、v3门槛及双shape接入回归。覆盖完整合成报告、基线不变、源码差异限制、真实执行入口保护、非有限统计拒绝、失败中止及1.5%验收边界。见 [tests.log](../../../../outputs/h100/phase6/arrival-op2-cpu-20261020/tests.log)。
- source-only父构建→D/R源码生成→compare准备→严格核验流程通过；两组的双角色主机表、映射及runtime的SM90对象编译通过。此处使用CPU fixture，不代表真实GPU构建。见 [integrated-compile/validation.json](../../../../outputs/h100/phase6/arrival-op2-cpu-20261020/integrated-compile/validation.json)。
- 对已校验原始JSON/CSV哈希的真实到达观测重放：A/D物理entry一致，R重现119/42个改变槽位；D/R各8192个主机坐标及真实非identity设备表编译通过。见 [measured-plan/validation.json](../../../../outputs/h100/phase6/arrival-op2-cpu-20261020/measured-plan/validation.json)。
- 最终独立decoder测试程序编译成功，**未执行GPU**。见 [decoder-final/validation.json](../../../../outputs/h100/phase6/arrival-op2-cpu-20261020/decoder-final/validation.json) 和 [编译资源日志](../../../../outputs/h100/phase6/arrival-op2-cpu-20261020/decoder-final/compile.log)。

**真实归档存在一个完整性限制：** 本地回传目录缺少父校准登记的 `calibration/__pycache__/runtime.cpython-310.pyc`，所以完整父档案严格校验未通过。上述真实重放只跳过父辅助文件存在性检查，仍核验原始到达JSON/CSV哈希；产物明确标记 `diagnostic_only=true`、`full_parent_seal_validated=false`，不能作为可执行构建。**生产入口未放宽校验**，在H100上须使用原服务器完整父构建及校准档案；缺文件时应补回原档案，不能把诊断产物用于正式计时。

CUDA12.8、SM90编译的资源结果：

| decoder | 寄存器/线程 | 共享内存/CTA | 编译器 barrier资源 | spill |
| --- | ---: | ---: | ---: | --- |
| 原A scalar | 36 | 536 B | 1 | 0 |
| op2 TP8 | 30 | 0 B | 0 | 0 |

这里 barrier资源数不是动态执行次数，旧算法动态次数为前述43。减少资源占用不等于已获得加速。

[decoder_bench.cu](../../../../tools/h100/phase6/arrival_op2/decoder_bench.cu) 直接链接**未修改的旧 scalar runtime**，在执行端检查：

- 所有 TP2/4/8来源 rank索引、实际 H100输出shape、尾部shape和不足4个warp的尾块；
- 独立 Hadamard 单系数解析结果、随机有限 E4M3 packet、多种尺度，以及原 block encoder产生的 BF16 packet；
- A/D 输出逐位一致、输出有限、输出尾部guard不被覆盖；
- 更换输入后重复执行；独立 decoder 的 CUDA event配对计时，仅作诊断。

独立 GPU 校验失败立即停止，随后才构建完整 D/R 运行库。完整模型另做8 rank数值预检、实际映射和profile kernel身份核验。CPU fixture及只编译结果不能通过GPU验收。

本机没有 PyTorch、NVIDIA设备节点或驱动，**没有执行 CUDA kernel，没有验证运行时逐位一致，也没有编译完整 Flux/PyTorch 动态库**。这些是执行端仍须完成的验证。

## 7. 在原 H100 容器运行

同步本次完整 `tools/h100/phase6/` 新文件及 `arrival_op2/` 子目录；保留原父构建及两角色校准原始数据。新默认是**两次独立比较**，不是旧op1的四次。每次6块×2轮×3组＝36计时窗口，每窗口10步预热＋50个完整optimizer step；合计72窗口、3600正式step、576份rank记录。

```bash
cd /workspace/TempName
export H100_OP2_PYTHON=/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX="$CXX"
export DEBUG=0
export MAKEFLAGS=

# 只预览：不创建目录、不启动子进程、不查询GPU。
"$H100_OP2_PYTHON" tools/h100/phase6/detach_arrival_op2.py

# 后台执行：独立解码逐位检查 → 构建D/R一次 → 两次各自预检和正式计时。
"$H100_OP2_PYTHON" tools/h100/phase6/detach_arrival_op2.py --execute
```

默认父构建仍为：

```text
outputs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/build
```

支持 `--build`、`--python`、`--megatron`、`--build-cuda-home`、`--cuda-home`、`--nccl-root`、`--jobs`、`--out`。默认编译CUDA12.8、训练运行CUDA12.4、jobs2、NVLS0。指定 `--trials 4` 可做四次；`--foreground --execute` 用于前台诊断。只有本协议已编译的构建可通过 `--op2-build` 复用，旧B库不能代替。

启动器默认输出：

```text
logs/h100/phase6/arrival-op2-2x-50steps-北京时间戳/
  decoder-check/validation.json         # GPU逐位检查及decoder微基准
  build/arrival-op2-manifest.json        # 独立D/R构建及源码身份
  compare1/report.json、report.md
  compare2/report.json、report.md
  batch-state.json
  handoff/container-supervisor.log
```

默认后台方式沿用已有容器PID1托管流程。返回“Started”仅代表启动；`batch-state.json` 的 `status=completed` 才表示全部实验完成。实验或证据检查失败立即停止，不自动重试、不覆盖结果。性能未达标会如实记录，并继续完成预定重复。

也可以先单独运行解码测试：

```bash
"$H100_OP2_PYTHON" tools/h100/phase6/arrival_op2.py decoder-check \
  --out /tmp/arrival-op2-decoder-gpu \
  --cuda-home /usr/local/cuda-12.8 --execute
```

省略 `--execute` 只编译，不会产生GPU通过状态。输出目录须不存在。

## 8. 读结果及最终采用规则

主结果使用全部完整三组块配对；每次独立比较12块，bootstrap10000次、seed20261009。八个rank和窗口内50步不当作独立样本。两轮反转顺序，慢窗口敏感性另列，不替代全样本主结果。

**本次预先固定的目标验收：每次独立比较的 combined D/A 点估计和95%区间下界都达到1.5%，且所有指定比较均通过数值及证据检查。** 两次默认都满足时，批次标记：

```json
{
  "status": "completed",
  "target_validated": true,
  "acceptance_status": "validated_at_least_1.5_percent"
}
```

若 `status=completed` 但 `target_validated=false`，表示实验完成、1.5%目标尚未得到验证。点估计超过1.5%而区间下界不足，不写成“已稳定达到1.5%”。逐轮结果仍完整显示；R/D及R/A为诊断，没有多重比较校正，不用于择优替换主候选。

若 D 逐位检查通过但整步未达标，先看同场无采样整步结果及decoder微基准，确认实际节省是否够31.3 μs/次，再判断是否值得继续优化消费者。若要重新研究“到达优先”本身，下一阶段需要同codec观测与真正可提前消费的依赖设计；本轮不再把单纯放宽静态排序门槛作为默认后续路线。
