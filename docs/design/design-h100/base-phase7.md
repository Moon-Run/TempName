# H100 Phase7：在原融合路径上改进到达优先

更新：2026-10-17。本轮在原有**交替顺序＋到达优先＋选择性量化**基础上改进静态 tile 排序。交替目标顺序、GemmV2、融合编码、解码、IPC、屏障、双缓冲、BF16/STE 反向及物理量化 mask 保持一致，只派生新的 CUDA/host 映射表。

当前已完成代码、历史真实采样的离线拟合、三套同源候选源码和九组完整模型实验入口。**尚未运行 H100 实验，不能宣称加速或训练质量改善。** 此前独立的分块原型、相关文档、开发会话与 CPU 验证结果已删除；本页只记录当前融合实现。设计依据见 [update.md](update.md)。

## 1. 本轮改进与原思想的关系

| 原思想 | 本轮处理 | 作用 |
| --- | --- | --- |
| 交替顺序 | 保留来源rank s的目标轮转 `s,s+1,...` 和每目标64个合法槽位 | 保持原来的目标分区服务节奏 |
| 到达优先 | 从真实接收端观测估计生产位置趋势，使用跨两遍稳定的剩余延迟生成协调置换 | 尝试减少仅按历史汇合时刻独立排序的局限 |
| 选择性量化 | 三种顺序使用逐元素相同的原物理mask | 先单独识别排序变化的收益；避免同时改变精度和流量 |
| 融合执行 | 保留原SM90编译的GemmV2及相关封装 | 不引入小GEMM循环或新的张量通信后端 |

新顺序仍然是离线静态计划，运行时继续通过原GEMM swizzle查表。它不增加每step的CPU调度，不实现实时就绪队列，也不删除现有完整生产后的屏障。

首版只改进到达优先。新的顺序经过目标机器验证之后，再按新顺序重新采样并调整量化对象，避免一次改动多个因素而无法解释结果。

## 2. 改进到达优先的实现

### 2.1 原始证据与时间口径

[fused_plan.py](../../../tools/h100/phase7/fused_plan.py) 从父计划 `inputs` 登记的原始 `b0-interleaved-instrumented-rank*.json` 读取数据，经 `resolve_input` 按精确SHA256定位，检查rank、硬件顺序、run ID及完整shape。

同一目标GPU、同一次观察中，一个来源对一个tile的完整到达时间由全部fragment的最晚记录确定，再减去该接收GPU本次调用起点。保留合法的负相对时间，不比较不同GPU的绝对时钟。

pass 0、1用于拟合；pass 2只在选择完成后作诊断，不决定映射、参数或回退。原始采样每次只观察部分tile，因此不同tile的观测来自不同调用，不能拼成一次完整调用的真实时间线。

### 2.2 区分生产位置与剩余延迟

对于每个目标分区d、来源s和训练遍p，令 `i_base(s,t)` 为tile t在原交替顺序中的逻辑槽位，`A(s,t,p)`为观测到达时间。先用Theil–Sen中位斜率拟合非负位置趋势：

```text
alpha(s,d,p) = max(0, median(pairwise arrival difference / slot difference))
residual(s,t,p) = A(s,t,p) - alpha(s,d,p) × i_base(s,t)
```

再使用保守的剩余延迟分数：

```text
priority(s,t) = min(residual(s,t,0), residual(s,t,1))
                - max(poll_uncertainty(t,0), poll_uncertainty(t,1))
```

这会压低只在一遍中明显变慢的贡献，并扣除采样轮询的不确定度。该分数不是统计置信下界，也不能解释为压缩或排序可节省的微秒数。

### 2.3 跨来源协调与局部微调

每个destination分别生成两个候选，保持完整tile集合和原合法槽位：

1. **共享剩余延迟顺序**：按同一tile在各来源中最大的稳定分数降序，形成各来源共同的物理tile顺序。
2. **协调剩余延迟顺序**：先按上述共享顺序分成固定的8-tile组，各来源只在每组内部按自己的分数微调。

共同的粗顺序用于减少各来源优先生产完全不同tile的情况，组内调整则保留来源差异。这里的8-tile组是静态排序约束，不会拆分GEMM或启动分块通信。

两种候选均写回原destination的64个槽位；跨目标轮转、物理tile归属、本地/远端判定和量化对象不变。

### 2.4 保守筛选与回退

使用下面的静态近似，只对候选排序作内部筛选：

```text
A_surrogate(s,t,p) = A(s,t,p)
                     + alpha(s,d,p) × (i_new(s,t) - i_base(s,t))
J_surrogate(t,p) = max_s A_surrogate(s,t,p)
```

每个destination独立与**原到达优先legacy**比较。候选须满足两训练遍的p95和最大tile汇合代理值均不恶化，并且至少一个指标的改善超过该分区的中位轮询不确定度；若多个候选通过，用预先固定的代理指标顺序选择。固定绝对数值精度用于稳定同分排序。

任一训练tile缺失时，该destination整个保留legacy；没有候选通过时也保留legacy。所有映射必须是完整双射，目标分区保持不变，不能删tile或用零到达时间补缺。

**这个线性位置模型是假设，不是真实执行预测。** CTA多波并发、Stream-K、缓存、编码分支和链路争用都可能破坏“位置变化与到达变化近似线性”的关系。输出字段明确标为 `surrogate_*`、`not_recalibrated=true`、`measured_performance=false`。最终收益只由采样关闭后的新H100配对实验决定。

## 3. 真实历史数据的离线拟合结果

以下是读取历史原始记录后生成的映射变化，不是新性能测量：

| 角色 | 接受新候选的目标分区（rank从0起） | 缺训练样本而回退的分区 | 各来源相对legacy改变的槽位数 |
| --- | --- | --- | --- |
| Attention | 2 | 4 | `[63,63,64,64,61,64,64,63]` |
| MLP | 0、2、4、5、6 | 1、3 | `[320,318,319,320,315,319,320,320]` |

其余分区因不满足保守条件保留legacy。Attention分区2和MLP分区0/2/5/6选择协调候选，MLP分区4选择完全共享候选。两角色分别拟合，不用MLP记录代替Attention记录。

三个顺序版本始终保留父计划的量化数目：Attention各来源为 `[7,7,7,2,7,7,7,1]`，MLP为 `[2,7,7,2,7,7,7,0]`。`verify --refit`可重新读取原始证据，重建映射和诊断并检查完全一致。

## 4. 同一融合实现上的九组对照

| 顺序族 | 基础顺序 | 到达优先 | 量化mask |
| --- | --- | --- | --- |
| `base` | 原interleaved | 关闭，恒等置换 | 原选择性mask |
| `legacy` | 原interleaved | 原来的到达置换 | 同一选择性mask |
| `improved` | 原interleaved | 本轮生成的协调置换，必要时回退legacy | 同一选择性mask |

每族分别作用于 `mlp`、`attn`、`attn_mlp`，策略ID为 `fused_<order>_<scope>`，共九组。**本轮base不是消融3的A组：A是全远端量化，本轮base仍使用原选择性mask。**

主比较为同范围的 `improved / legacy`；`legacy / base`和`improved / base`是辅助比较。另保留同顺序作用范围比较。新旧到达优先的差异仅来自静态映射，不能把作用范围扩展的收益归因于排序。

配置仍为GPT6.7B、32层、H4096/FFN16384、S2048、micro batch1/global batch4、BF16、TP8/DP1/PP1/CP1。每窗口10步预热＋20步完整optimizer step，每step含4个微批和8192 token。

默认两轮，每轮9个完整配对块、每块九组各一次，共162个正式窗口。循环移位平衡位置，第二轮反转执行顺序。另有9组smoke和9组profile预检，共180个worker窗口。数值和有限loss检查仍来自固定合成输入，不构成真实语料收敛验证。

## 5. 代码与构建不变量

| 文件 | 职责 |
| --- | --- |
| [fused.py](../../../tools/h100/phase7/fused.py) | 构建准备、校验、编译、模型准备、执行、报告入口 |
| [fused_plan.py](../../../tools/h100/phase7/fused_plan.py) | 原始采样解析、稳定剩余延迟排序、保守回退及拟合重建 |
| [fused_build.py](../../../tools/h100/phase7/fused_build.py) | 从原冻结融合实现生成三套候选，只允许两份映射源码不同 |
| [fused_protocol.py](../../../tools/h100/phase7/fused_protocol.py) | 九组路由、同mask定义、调度与比较集合 |
| [fused_prepare.py](../../../tools/h100/phase7/fused_prepare.py) | 冻结旧融合模型adapter/worker及新控制代码 |
| [fused_run.py](../../../tools/h100/phase7/fused_run.py) | 参数化复用原模型runner，默认dry-run |
| [fused_report.py](../../../tools/h100/phase7/fused_report.py) | 实际库、映射、数值、profile、完整rank与配对报告验证 |

每个候选保存351份注册源码。三个版本只允许以下两份文件的哈希不同：

```text
src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp
phase6_host.hpp
```

GEMM、epilogue、codec、IPC、双缓冲封装与反向代码保持同源一致。构建验证同时比较三版本计算源码，并检查它们相对原Phase6计算来源的差异；不能靠修改manifest将其他计算变化伪装成顺序变化。

未选中投影保持原生Megatron BF16。所选投影仍使用原 `GemmRSTacoDoubleBuffered`、融合H128/E4M3、固定来源归约及BF16/STE反向，不导入新的张量通信runtime。

## 6. 原版保留和本地开发产物

Phase6原版快照仍位于本地 `outputs/h100/phase7/phase6-baseline-20261017/`，维护提交为 `aa5d3d933319e5e9f38cb11fca5758d4781f9da0`，原计算候选来源为 `ff38f5b1ff1c80cb831f58c64b95f42b0379879a`。快照和原始记录未被本轮改写，复现步骤见 [Phase6保存说明](base-phase6-mlp-attention.md#phase7-开发前的原版保存与复现2026-10-17)。

本轮开发目录为 `outputs/h100/phase7/fused-arrival-20261017-dev/`，包含三套候选源码、计划和本地编译检查。它由 `.gitignore` 排除，不随Git同步。新机器应依据下面的命令重新生成，或显式携带对应本地文件与原始校准证据。

此前分块原型的10个实现/测试文件、原型文档、三个chunk8开发会话及 `logs/h100/phase7/cpu-20261017/` 已删除。当前测试目录仅保留融合方案的测试；`snapshot.py`继续用于校验Phase6原版。

## 7. 使用方式

### 7.1 生成与核对源码

在仓库根目录执行，输出目录必须不存在。默认使用已有真实Attention/MLP校准计划，也可通过 `--attn-plan`、`--mlp-plan`传入新计划。

```bash
python3 tools/h100/phase7/fused.py build-prepare \
  outputs/h100/phase7/fused-arrival-run1/candidates

python3 tools/h100/phase7/fused.py verify \
  outputs/h100/phase7/fused-arrival-run1/candidates --refit
```

`--source-only`用于无真实到达拟合的CPU流程演练，improved保留legacy且永久不能升级为可执行GPU会话。正式候选必须使用真实父计划及其完整哈希证据。

仅做host映射执行与SM90目标文件编译，不启动GPU：

```bash
python3 tools/h100/phase7/fused.py compile-check \
  outputs/h100/phase7/fused-arrival-run1/candidates \
  --out outputs/h100/phase7/fused-arrival-run1/compile-check \
  --cuda-home /data/apps/cuda/12.8
```

这个检查不链接完整Flux动态库，不证明实际GPU映射、占用或训练正确性；不能将其当成 `compile` 或GPU预检。

### 7.2 H100端完整编译和模型实验

下面的路径按H100主机配置设置。若更换host、GPU UUID/顺序，应先按原Phase6流程重新采集两角色BF16校准记录，而不是绕过硬件检查。

```bash
export PHASE7_PY=/opt/conda/envs/flux-h100/bin/python
export PHASE7_ROOT="$PWD/outputs/h100/phase7/fused-arrival-run1"
export PHASE7_CUDA=/usr/local/cuda-12.8
export PHASE7_NCCL="$PWD/outputs/h100/phase6/tp8-20261009-deployment/nccl-release"

DEBUG=0 "$PHASE7_PY" tools/h100/phase7/fused.py compile \
  "$PHASE7_ROOT/candidates" --cuda-home "$PHASE7_CUDA" \
  --nccl-root "$PHASE7_NCCL" --jobs 2

"$PHASE7_PY" tools/h100/phase7/fused.py prepare "$PHASE7_ROOT/compare" \
  --build "$PHASE7_ROOT/candidates" --megatron /workspace/Megatron-LM

# 默认只列出任务，不查询GPU、不启动worker。
"$PHASE7_PY" tools/h100/phase7/fused.py run "$PHASE7_ROOT/compare"

"$PHASE7_PY" tools/h100/phase7/fused.py run "$PHASE7_ROOT/compare" \
  --stage preflight --python "$PHASE7_PY" --cuda-home "$PHASE7_CUDA" --execute

"$PHASE7_PY" tools/h100/phase7/fused.py run "$PHASE7_ROOT/compare" \
  --stage timing --python "$PHASE7_PY" --cuda-home "$PHASE7_CUDA" --execute
```

Megatron固定 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`。准备时冻结Phase6依赖控制代码和本轮 `fused_*` 控制代码；执行前核对实际控制脚本、构建库和输入哈希，不能在已准备会话中继续改代码后直接续跑。

预检包含九组smoke、九组独立profile、八rank完整性和12项策略/角色GPU映射；profile须出现实际V2与解码kernel，未选中角色不能进入候选。正式计时不使用采样库。失败会话保留现场，新实验使用新目录。

### 7.3 CPU回归

```bash
python3 -m unittest discover -s tools/h100/phase7 -p 'test_fused_*.py' -v
```

新回归验证真实父源码派生、三版本同mask与同计算、合法映射、负相对时间、缺样本回退、heldout隔离、模型预检入口和完整配对统计。测试通过只能说明这些检查通过，不能替代H100数值与性能验证。

本次清理旧代码后，32项融合方案CPU回归通过；真实采样的拟合重建一致，三顺序×两角色的host映射检查通过，三个SM90映射目标文件及共享解码runtime目标文件编译通过。开发compare会话仅用于source-only dry-run，不能执行GPU；相关本地产物位于已忽略的开发目录，不新增测试结果文件到Git。

## 8. 收益边界与后续改进

本轮保留完整生产后的屏障，所以新排序仍主要通过生产/远端写的时序、拥塞和尾部变化影响完成时间。静态代理的tile汇合改善不保证最后一个生产者更早结束，也不保证step更快。

当前最重要的实验是同一融合实现上的 `improved / legacy`。报告使用八rank最大elapsed除以20，逐轮按完整九组配对块计算耗时下降和bootstrap区间；没有预设通过门槛，负收益和跨零区间都保留。

若新顺序没有稳定收益，保留legacy并分析真实profile；若有收益，再按新顺序重新测量剩余拖尾，更新选择性mask及量化预算。更进一步的分块ready消费需要在原融合epilogue和IPC生命周期内实现，单独检验新增协议与资源竞争成本。
