# 到达优先复核：优化生产尾部，联合求解，并以同实现实测决定是否采用

更新：2026-10-19。依据 [idea.md](../idea.md)、[消融4](../base-phase6-ablation4.md)、[消融7改进版](../base-phase6-ablation7.md)，重新核查冻结计划、原始校准、实际 mapping、profile 和配对计时，并实现下一版候选生成与实测准入工具。

**结论：主要问题是优化目标与实际消费路径不匹配，以及校准信号不足；没有找到“到达表未生效、映射方向写反”这类实现故障。改进版还存在可以修复的单交换搜索盲点，但它不是本批零收益的全部原因。** 当前证据不支持继续把“较晚到达的贡献提前生产”作为默认有效的加速机制，也不足以否定所有到达感知方法。

本次选择的修正是：**保留交替基线，把到达信息用于提出受约束的生产尾部候选；联合评估所有被提前和被推迟的贡献；没有可信信号就保留基线，非 identity 候选必须经过同实现、同量化方式的独立配对实测。** 不把代理模型改善写成真实加速，不通过降低误差门槛强行得到置换。

已完成联合求解、计划生成接入、回退诊断和历史实测准入审计。**本机没有可用的 H100 执行环境，本次没有启动新 GPU 性能实验。** 本文中的已有 GPU 结果来自原始回传证据；新算法的验证为 CPU 重放、回归及映射编译检查。

## 1. 两轮实验实际证明了什么

| 比较 | B/A 配对完整 step 耗时下降 | 实际调度 | 可以支持的结论 |
| --- | --- | --- | --- |
| 消融4，60步，compare1 / 2 | −0.185% / −0.135%，两区间跨零 | 旧到达表确实改变物理顺序 | 当前旧启发式没有稳定净收益 |
| 消融7，50步，compare1 / 2 / 3 | −0.151% / +0.119% / −0.096%，三区间跨零 | 新B两个角色均0交换，与A相同 | 测到了保守回退，未测到非 identity 新重排的效果 |

以上点估计从原始 `report.all_samples` 的同轮同块耗时重新计算，使用 `100×[1−exp(mean(log(T_B/T_A)))]`。完整区间及样本量见原结果页；跨零不等于通过严格等效检验。

C 相对 A/B 的约3%—4%差距同时涉及 codec、量化覆盖率、工作区和输出生命周期，不能拿来证明到达优先有效。消融7的C仍使用旧到达表，也不是新B的同实现对照。

## 2. 代码没有失联，但“到达”没有直接触发消费

### 2.1 旧映射确实进入了执行路径

消融4两次实际 mapping 均显示：Attention 改变 **2491/4096** 个槽位，MLP 改变 **2422/4096** 个槽位；目标分区改变数均为0。每个来源的512个物理 tile 仍一一覆盖。

[scope_build.py](../../../../tools/h100/phase6/scope_build.py) 将新槽位先映射到旧逻辑槽位，再通过交替规则得到物理 tile，即 `maps[source][new_slot] = old_slot`。Attention/MLP 分别按 `K_local=512/2048` 分派。核查没有发现方向颠倒或只改主机表、设备表未接入的问题。

### 2.2 当前同步结构限制了收益来源

消融4及7的A/B使用 [fused_v2 wrapper](../../../../tools/h100/phase6/fused_v2/src/gemm_rs/ths_op/gemm_reduce_scatter.cc)，顺序为：

```text
GEMM及输出发布 → 全rank发布屏障 → 解码归约 → 全rank复用屏障
```

[build.py](../../../../tools/h100/phase6/build.py) 使TACO实例选择V2路径，H100上这道发布屏障确实执行。独立重放两次消融4中 A/B、rank0/rank4 的8份profile，确认 **2048/2048次投影同流串行，GEMM/decode重叠0次**。这仅覆盖这些trace；profile耗时受采样扰动，不用于估算无采样加速上限。

所以，某个 tile 提早收齐，不能让当前消费者立即解码。应关注的是生产阶段最终完成、跨rank屏障等待和整个算子耗时。保持来源总tile数、字节数和目的地轮转频率不变的分区内重排，主要只能通过局部性、资源争用或最终尾部改变性能；它没有直接减少计算量或通信量。

**不能直接删除屏障。** 真正“到达即消费”需要逐fragment/tile发布、内存可见性、generation、消费者等待与工作区复用协议，以及GPU竞态、死锁和数值验证。这是另一项执行机制改造，本次没有将未经验证的同步改动混入历史消融。

另外，消融7的B从A的已登记源码复制并替换映射，见 [ablation7_build.py](../../../../tools/h100/phase6/ablation7_build.py)。只修改仓库当前 `src/gemm_rs` 不会自动改变它的冻结运行库。

## 3. 为什么改进版仍然零交换

输入为消融7 [compare1冻结计划](../../../../logs/h100/phase6/ablation6-then7-3x-50steps-20261019-162154-480841/ablation7/compare1/scripts/selection-plan.json)。两角色每个都有8来源×8目的地＝64个窗口，每窗口64个tile。pass0/1用于拟合，pass2仅诊断。

| 重新计算的诊断 | Attention | MLP |
| --- | ---: | ---: |
| 两训练pass最大到达槽位相同 | 3/64窗口 | 2/64窗口 |
| 两pass各自存在单交换改善、但不存在共同改善的单交换 | 49/64窗口 | 61/64窗口 |
| 每角色穷举的单交换数量 | 129024 | 129024 |
| 任意单交换的最大 `min(gain_pass0,gain_pass1)` | 0.192 μs | 1.184 μs |
| 窗口最大观测误差代理 ε | 2.816—3.008 μs | 3.072—3.584 μs |
| 既有6ε门槛 | 16.896—18.048 μs | 18.432—21.504 μs |
| `receiver_sparse` 相对同repeat `off` 的中位开销 | 15.442% | 5.679% |
| 旧1/64 mask选中贡献在pass2仍接近临界 | 18/53＝33.962% | 11/50＝22.000% |

表中的“任意单交换”甚至没有施加旧版的重复正残差准入条件，是更宽松的搜索。即使只要求超过 **1ε**，仍然没有合格交换。留出比例只描述旧mask选中的贡献，不是全部贡献的预测准确率。

此外，采样器测的是 **BF16 fragment发布**，A/B实际执行的是 **全远端量化**。编码、包格式、资源占用和传输行为均不同；不能直接把BF16标签当作量化路径的稳定时延模型。各tile来自不同稀疏采样调用，接收端相对起点也不能消除调用间波动。PAVA只能拟合趋势，不能恢复一次真实联合时间线，更不能自动证明某个残差会随tile移到新槽位。

### 3.1 单交换确实存在搜索盲点

两遍训练均取下面的合成输入，ε＝0.01 μs：

```text
观测到达：       [0, 0, 10, 10, 8, 8]
PAVA排位趋势：   [0, 0,  9,  9, 9, 9]
残差：          [0, 0,  1,  1,-1,-1]
```

单独提前任一10，另一个10仍决定最大值，旧规则拒绝交换；同时交换 `0↔2、1↔3` 后，预测到达变为 `[1,1,9,9,8,8]`，最大值由10降至9，超过同样的6ε＝0.06门槛。这说明“每一步都严格改善最大值”不是完整搜索。

但真实数据精确并列尾部的窗口仅为Attention 0个、MLP 1个。不能把这个合成反例直接当成实际零交换的主因。

### 3.2 更强结论：当前数据连理想联合排列也过不了原门槛

对某一训练pass，记非降PAVA趋势为 `b_p[i]`，残差为 `r_p[j]`，原最大到达为 `M_p`。在“槽位趋势＋tile残差”这个代理模型内，允许任意排列时，单pass的最优最大值为：

```text
L_p = max_i (b_p[i] + sort_desc(r_p)[i])
U_p = M_p - L_p
```

理由是：若较小残差被安排在较早槽位、较大残差被安排在较晚槽位，交换二者不会增大二者的最大预测到达；逐步交换即可得到残差降序。`U_p` 是单pass的最大可改善量。

这里还允许两个pass各选自己的排列，且没有位移约束，比实际要求同一排列更宽松。因此，任一pass的 `U_p≤6ε_p` 都足以排除该窗口所有共同排列。

实际计算得到：

| 乐观上界诊断 | Attention | MLP |
| --- | ---: | ---: |
| 两pass各自理想最优均超过6ε的窗口 | 0/64 | 0/64 |
| `max_window min_pass(U_p/ε_p)` | 3.8393 | 4.2331 |

**所以，在现有代理模型和6ε规则内，这批数据无论采用单交换还是联合排列，都只能回退identity。** 这是模型内的不可能性结论，不是实际GPU调度的性能上界，也不说明6ε是统计置信区间；它只使用原实现对轮询误差的保守处理，尚不覆盖模型失配。

## 4. 新思想与本次代码

### 4.1 从“晚到优先”改为“联合优化生产尾部的候选”

新 [arrival_assignment.py](../../../../tools/h100/phase6/arrival_assignment.py) 使用标识 `receiver-tail-assignment-v2`，选项为 `tail_assignment`。它继续在每个来源、目的地、合法窗口内建模，但直接求一个完整排列，不要求中间每次交换都改善最大值。

将原槽位j的tile放到新槽位i的代价定义为：

```text
cost[i,j] = max_{p∈{0,1}} (b_p[i] + r_p[j] - M_p + 6ε_p)
目标：min_π max_i cost[i,π(i)]
```

这样，瓶颈代价严格小于0，就表示**同一个排列在两遍训练中都降低整个窗口最大值，且分别超过原6ε门槛**。所有被后移tile都已计入，不会只计算提前者的收益。

实现顺序：

1. 检查完整训练观测，先用上一节的单pass乐观上界剪枝。
2. 默认仅允许tile在同目的地窗口内移动最多8个位置。TP8下最多相差64个全局逻辑槽位；这只是扰动约束，不等于CTA wave或实际执行时间。
3. 用二分代价阈值及二分图完全匹配求精确瓶颈最优值；同最优值下再优先最少改变槽位、最小总位移，结果确定。
4. 再核对两pass的最终完整窗口最大值、有限数值和严格门槛。缺失数据、无模型改善、噪声限制、位移限制或跨pass冲突分别记录原因并回退。
5. pass2只诊断冻结排列，绝不参与候选选择；来源、目的地、窗口、物理tile覆盖和量化mask保持约束。

单个来源窗口的代理改善仍不保证整组关键路径改善。模型也没有解决BF16与量化路径的分布差异，因此它是**候选生成器**，不是运行时间预测器或自动采用依据。

### 4.2 只有同实现实测才能决定采用

新增 [arrival_probe.py](../../../../tools/h100/phase6/arrival_probe.py) 作为已有实验的**只读、CPU离线准入审计器**，复用消融4/7严格报告流程。它不是新GPU测量器。

- 核验A/B的实现、codec、mask、shape、来源文件与冻结映射，只允许调度相关源码不同。
- 若两个角色的物理映射完全相同，直接返回 `not_tested_identity`，不把重复计时解释为新调度验证。
- 非identity时，`--evaluate` 在临时目录重新运行严格预检和完整step报告，原始结果不被覆盖。
- 至少需要两次独立启动、不同run_id的复测，且计划、运行库登记、模型、硬件、输入及测量身份一致；每次两轮及合并B/A的完整块配对95%区间下界均须大于0，才标记 `validated_for_same_configuration`。不同run_id本身不证明样本在统计意义上独立。
- 只认定Attention＋MLP联合策略在这套配置下的证据，不从完整step拆分逐角色加速，不自动部署。反复尝试候选时，这些逐项区间不提供多重选择校正；确认实验必须在候选冻结之后独立完成。

下一轮有价值的新GPU工作应先取得同codec、重复、可干预的观测，并测量无采样完整算子与step。仅重跑旧BF16校准或者再次计时identity B，不能解决当前问题。**本次没有把现有BF16采样器改写成量化路径采样器，也没有宣称已完成这项新校准。**

### 4.3 代码接入及历史兼容

| 文件 | 本次改动 |
| --- | --- |
| [arrival_assignment.py](../../../../tools/h100/phase6/arrival_assignment.py) | 联合瓶颈求解、位移约束、乐观上界及分类回退 |
| [plan.py](../../../../tools/h100/phase6/plan.py) | 接入 `arrival_order='tail_assignment'`，独立保留原mask生成 |
| [scope_calibrate.py](../../../../tools/h100/phase6/scope_calibrate.py) | 将新求解器及其PAVA依赖的源码哈希写入新角色计划 |
| [ablation4.py](../../../../tools/h100/phase6/ablation4.py) | `fit --arrival-order tail_assignment` 显式入口 |
| [arrival_replay.py](../../../../tools/h100/phase6/arrival_replay.py) | 对哈希校验过的raw重放新算法，输出候选、采样诊断与算法源码哈希 |
| [arrival_probe.py](../../../../tools/h100/phase6/arrival_probe.py) | identity识别及同实现独立实测准入 |

历史 `legacy_tail` 默认值和消融7 `tail_residual` 固定协议不变，新方法必须显式选择。旧结果、冻结计划和运行库保持原样。用新角色计划重新构建消融4会派生新的B/C映射，不能将它冒充历史消融7的B，也不能将新C与旧C混为同一策略。

## 5. 本次真实数据重放结果

重放先校验已登记raw JSON/CSV哈希，并完整复现旧 `maps/coords/base_coords/mask`，然后运行联合求解。

| 新求解器结果 | Attention | MLP |
| --- | ---: | ---: |
| 接受联合排列的窗口 | 0/64 | 0/64 |
| 改变槽位 | 0/4096 | 0/4096 |
| `uncertainty_limited_identity` | 52 | 64 |
| `no_model_gain_identity` | 12 | 0 |
| 旧物理量化mask是否保持 | 是 | 是 |

这与乐观上界结论一致。新代码的改进是搜索完整性、回退可解释性和实测准入，不是从本批数据获得了新的加速。合成反例已覆盖非identity分支，真实数据没有被人为放松规则以制造重排。

本次证据保存在 [arrival-op1-cpu-20261019](../../../../outputs/h100/phase6/arrival-op1-cpu-20261019/)：

- [assignment-replay-final.json](../../../../outputs/h100/phase6/arrival-op1-cpu-20261019/assignment-replay-final.json)：完整新算法重放及输入哈希，包含观测提取器、mask选择器和新求解器的源码身份。
- [residual-analysis.json](../../../../outputs/h100/phase6/arrival-op1-cpu-20261019/residual-analysis.json)：每窗口单交换穷举及独立最优上界。
- [profile-audit.json](../../../../outputs/h100/phase6/arrival-op1-cpu-20261019/profile-audit.json)、[paired-summary.json](../../../../outputs/h100/phase6/arrival-op1-cpu-20261019/paired-summary.json)：旧映射、实际串行路径及配对点估计复算。

## 6. 使用入口与后续决策

在仓库根目录执行CPU诊断；输出文件必须不存在：

```bash
python3 tools/h100/phase6/arrival_replay.py \
  logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare1/scripts/selection-plan.json \
  --arrival-order tail_assignment --out /tmp/arrival-assignment-replay.json

# 默认仅核验冻结证据和实际映射；--evaluate也只做已有证据的CPU重算。
python3 tools/h100/phase6/arrival_probe.py \
  logs/h100/phase6/ablation6-then7-3x-50steps-20261019-162154-480841/ablation7/compare1 \
  logs/h100/phase6/ablation6-then7-3x-50steps-20261019-162154-480841/ablation7/compare2 \
  --evaluate --out /tmp/arrival-evidence-gate.json
```

若在执行端已有一份完整、通过原校准核验的会话，可生成**新的**候选角色计划：

```bash
python3 tools/h100/phase6/ablation4.py fit \
  /path/to/completed-calibration /path/to/new-role-plans \
  --arrival-order tail_assignment
```

这一入口继续使用现有BF16校准格式，不会自动获得同codec标签；CPU replay也不会把缺少动态库的回传归档晋升为可执行构建。先检查两角色 `arrival_order.identity_fallback`。全identity时保留交替基线，无须为了“验证新调度”重新编译、长跑一套相同映射；非identity时才进入独立同实现数值预检和配对实测。

研究表述应据此调整为：**到达信息提供候选，联合尾部目标约束候选，独立实测决定是否保留。** 在当前数据和当前屏障结构下，应接受基础交替作为回退结果，撤回“到达优先已经带来稳定额外收益”的主张。

## 7. 验证记录与边界

- **115项相关CPU回归通过**，覆盖新联合求解及既有协议、scope、消融4/7、冻结输入和报告流程，见 [tests.log](../../../../outputs/h100/phase6/arrival-op1-cpu-20261019/tests.log)。联合求解的测试包括单交换停滞反例、两pass冲突、噪声/缺失回退、heldout隔离、窗口约束，以及小规模全排列穷举对照最优瓶颈、最少改变槽位和最小位移。
- 最终准入审计器的 **13项专项测试**通过，覆盖同实现/预算漂移、跨零拒绝、重复run_id拒绝、原证据只读及JSON序列化顺序变化，见 [probe-tests-verified.log](../../../../outputs/h100/phase6/arrival-op1-cpu-20261019/probe-tests-verified.log)。TP8双shape接入的6项测试另行复测通过，见 [integration-tests-final.log](../../../../outputs/h100/phase6/arrival-op1-cpu-20261019/integration-tests-final.log)；两个角色使用不同的合成非identity排列，物理mask与旧拟合结果一致。
- 另生成两角色不同的非identity映射，各来源改变32/512槽位；C++程序核对全部8192个主机表坐标和错误K拒绝逻辑，设备表用CUDA12.8编译为SM90目标对象。见 [compile-final/validation.json](../../../../outputs/h100/phase6/arrival-op1-cpu-20261019/compile-final/validation.json)。这验证的是映射表的生成及编译，**不是完整Flux动态库、CUDA运行数值或性能验收**。
- 消融7三次50步记录通过实际冻结计划核验，准入结论均为 `not_tested_identity`，见 [ablation7-admission.json](../../../../outputs/h100/phase6/arrival-op1-cpu-20261019/ablation7-admission.json)。该结论不需要再对相同映射做性能bootstrap。
- 消融4最新两次60步记录通过严格预检及完整报告的离线重算；B/A合并区间分别为 **−0.185% [−0.443%, +0.142%]**、**−0.135% [−0.493%, +0.206%]**。准入结论为 `not_validated_no_repeatable_gain`，见 [ablation4-admission.json](../../../../outputs/h100/phase6/arrival-op1-cpu-20261019/ablation4-admission.json)。这是证据核验成功、性能准入不通过，不是工具执行失败。

本机检查不到NVIDIA设备节点、驱动或PyTorch，仅有CUDA编译工具。没有新H100运行、没有新加速率，也没有宣称跨配置可推广的收益；现有模型、codec、同步和量化mask的限制仍适用。

## 8. 在H100服务器后台连续测四次

新增入口 [detach_ablation7_new.py](../../../../tools/h100/phase6/detach_ablation7_new.py)，只运行新的C/A/B三组比较，顺序为 `compare1 → compare2 → compare3 → compare4`。新B固定使用 `tail_assignment`；C保留旧到达表、原candidate实现及1/16量化预算，A保持交替＋全远端量化。

新协议为 `h100-gpt67-tp8-ablation7-new-v1`，由 [ablation7_new.py](../../../../tools/h100/phase6/ablation7_new.py) 及其独立计划、构建、worker和报告模块执行。旧 `ablation7.py` 与 `detach_ablation6_then7.py` 的行为不变。同步到服务器时需要包含配套的 `ablation7_new*.py`、`arrival_assignment.py` 等依赖，建议同步本次完整 `tools/h100/phase6/` 改动，而不是只复制启动脚本。

| 项目 | 单次比较 | 四次合计 |
| --- | ---: | ---: |
| 块数 / 轮数 | 6块×2轮，三组块内位置均衡 | 4次独立run_id |
| 正式窗口 | 36 | 144 |
| 每窗口预热 / 正式step | 10 / 50 | 相同 |
| 正式optimizer step | 1800 | 7200 |
| 正式rank记录 | 288 | 1152 |
| 预检rank记录 / profile trace / mapping | 48 / 24 / 6 | 192 / 96 / 24 |

### 启动命令

在原8×H100实验容器、仓库根目录执行，沿用原训练Python、CUDA12.8编译环境和CUDA12.4运行环境：

```bash
cd /workspace/TempName
export H100_A7_NEW_PYTHON=/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX="$CXX"
export DEBUG=0
export MAKEFLAGS=

# 预览：不创建目录、不编译、不查询GPU。
"$H100_A7_NEW_PYTHON" tools/h100/phase6/detach_ablation7_new.py

# 后台执行四次；确认由容器PID 1托管后返回。
"$H100_A7_NEW_PYTHON" tools/h100/phase6/detach_ablation7_new.py --execute
```

默认父构建为 `outputs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/build`，需要其完整C/A运行库和已登记的两角色校准证据。执行前沿用原主机/GPU顺序及空闲检查；如果换了主机或GPU顺序，需要先获得对应校准的父构建，不能把旧计划直接当作新设备的校准。

可用 `--build` 指定父构建，`--out` 指定全新批次路径，`--python`、`--megatron`、`--cuda-home`、`--build-cuda-home`、`--nccl-root`、`--jobs` 指定环境。默认jobs=2；未指定NCCL目录时，先校验父构建登记的NCCL头文件及静态库，再复用该目录。需要前台诊断时加 `--foreground --execute`。

### 构建、失败处理与结果

脚本复用原C/A运行库，**只构建一次新B**，四次共用同一冻结计划和新B库。每次依次完成prepare、dry-run、独立预检、正式计时和严格报告；上一份报告通过后才开始下一次。任一步执行或证据校验失败即停，不自动重试，不覆盖已有输出。性能负收益或区间跨零会如实报告，不会因此取消后面的重复实验。

默认输出为 `logs/h100/phase6/ablation7-new-4x-50steps-北京时间戳/`，主要文件：

```text
batch-state.json                       # 四次进度、实际identity状态、完成计数
handoff/container-supervisor.log        # 脱离终端后的总日志
handoff/supervisor.json                 # 托管进程身份
build/ablation7-new-manifest.json        # 新B构建身份
compare1/report.json、report.md
compare2/report.json、report.md
compare3/report.json、report.md
compare4/report.json、report.md
```

四次各自的 `report.json` 均须通过，且运行身份、角色计划一致，批次才标为 `completed`。启动后返回只表示已托管，不表示已测完；关闭终端后可继续运行，容器和主机须保持运行。

可以通过 `--ablation7-build /path/to/new-build` 复用已编译的新B构建；它必须包含本新协议的 `ablation7-new-manifest.json` 并来自相同父构建。旧版消融7的 `ablation7-manifest.json` 不可替代。

**本次按指定的四次重复执行，即使新B仍回退identity，也不自动跳过。** 启动日志、`batch-state.json` 和每份报告都会显示实际改变槽位、接受窗口及回退状态。对现有真实旧校准的CPU重放仍是两角色0窗口；因此这四次可能继续检验回退路径，不能预先称为非identity重排的收益实验。

四次测完后，可选择只读审计新实验的准入证据；该命令不会再次启动GPU，也不修改原报告：

```bash
python3 tools/h100/phase6/arrival_probe.py \
  /path/to/new-batch/compare1 /path/to/new-batch/compare2 \
  /path/to/new-batch/compare3 /path/to/new-batch/compare4 \
  --evaluate --out /tmp/ablation7-new-admission.json
```

该入口已通过9项新后端测试、14项四次调度测试和13项既有准入工具回归；覆盖真实格式的合成原始校准产生非identity新排列、独立协议、完整报告、第四次失败/输入漂移停止以及identity仍执行四次。默认命令预览确认不创建目录、不启动子进程。这里的通过均为CPU验证，四次H100实测要在执行端启动后才会产生。
