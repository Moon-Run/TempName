# H100 Phase7：分块消费与共享顺序首版实现

更新：2026-10-17。依据 [改进方案](update.md) 实现独立的 Phase7 代码路径，并保存 Phase6 原版。**当前完成源码、CPU 回归和可执行实验入口；开发节点没有可见 GPU，尚未完成 H100 数值、并发、性能或真实数据训练验证。本文不填写未测的耗时或加速比。**

## 1. 本次实现范围

首版采用可独立验证的 **PyTorch 分块 GEMM＋NCCL 变长 all-to-all＋H128/E4M3 编解码原型**，验证“完整生产后消费”与“分块交换完成后提前消费”的差别。

这是 [update.md](update.md) 中分块原型阶段的实现。它会改变 GEMM 切分、kernel launch 数量、通信后端调用粒度和编码实现，不能把它与历史 Flux 候选的绝对耗时差直接归因于移除屏障。没有直接删除原 Flux V2 的 IPC barrier，也没有修改已有源码中的双缓冲复用条件。

| 设计项 | 本次状态 |
| --- | --- |
| 保存 Phase6 源码、计划、来源和复现入口 | 已完成实际快照与逐文件校验 |
| 同一目标内的共享分块顺序、目标交替生产 | 已实现；默认从旧计划生成明确标注的逻辑槽位代理顺序 |
| 物理 tile 量化 mask、BF16/全远端/旧选择/匹配随机 | 已实现；四组结构实验保持同一物理 mask |
| 分块通信完成后消费、固定来源顺序归约 | 已实现独立 runtime，待 GPU 验证 |
| 缓冲区安全与输出生命周期 | 使用每次调用独立张量、Work/event 依赖与 `record_stream`；不复用旧 IPC 工作区 |
| 按实际路径重新校准后的净收益选择 | 已实现严格证据输入、完整候选选择与冻结评估 API；本次未采集新 GPU 校准数据 |
| 独立算子与完整 GPT6.7B 训练入口 | 已实现，GPU执行需在目标环境显式启动 |
| 四组配对统计及同步×顺序交互效应 | 已实现；缺 rank、缺块或身份不一致拒绝生成正式报告 |
| 融合 epilogue 的 ready/epoch/consumed、乱序 ready 队列 | 尚未实现；需先评估本原型的收益空间，再移植融合路径 |
| 运行中低频更新、真实数据长期质量评估 | 后续工作，不以当前合成输入代替 |

实现目录：[tools/h100/phase7](../../../tools/h100/phase7/README.md)。运行时不依赖新编译的 Flux 动态库，因此本次没有 CUDA 编译成功或新 `.so` 的声明。

## 2. Phase6 原版保存

实际快照位于 [phase6-baseline-20261017（本地）](../../../outputs/h100/phase7/phase6-baseline-20261017/README.md)，保存3999个文件、61,625,410字节（约58.77 MiB）。维护 HEAD 为 `aa5d3d933319e5e9f38cb11fca5758d4781f9da0`，实测计算来源及子模块版本另在 manifest 中登记。

**`outputs/h100/phase7/phase6-baseline-20261017/` 仅保留本地，不随 Git 同步；克隆仓库不会自动获得快照。** 下列校验命令以本地目录已存在为前提，异机复现须另行复制快照及外部校准记录。

快照 manifest SHA256：

```text
34fda4720799ad9cec6fded3aed9913c1d853e007db9d6ae30cbeee521812c9a
```

原版来源区分、归档范围、未复制的大型校准证据及完整复建说明已追加至 [Phase6 MLP/Attention 文档](base-phase6-mlp-attention.md#phase7-开发前的原版保存与复现2026-10-17)。旧源码、冻结候选、计划和历史日志未被 Phase7 改写。

```bash
python3 tools/h100/phase7/snapshot.py verify \
  outputs/h100/phase7/phase6-baseline-20261017 --original
```

快照不包含动态库、构建树、环境和全部原始日志。24项大型校准记录保留原路径及哈希引用，迁移复现时需要同时携带；不能只复制快照就宣称具备完整历史实测证据。

## 3. 分块 runtime 的执行方式

### 3.1 几何与顺序

保持两种真实输出投影形状：

| 角色 | 全局 `[M,N,K]` | TP8本地K |
| --- | --- | ---: |
| Attention | `[2048,4096,4096]` | 512 |
| MLP | `[2048,4096,16384]` | 2048 |

物理 tile 为128×128；每目标rank有64个tile。`chunk_tiles` 支持1、2、4、8、16、32、64，默认8，即每目标8个消费分块。小于等于32时为一个tile行内的连续列区域；64为该目标的完整输出分区。

每个通信stage为每个destination生产一个分块。来源rank s按 `s,s+1,...` 的循环目标顺序计算，保留本地目标优先及来源之间错开的目标轮转；各来源对同一destination使用相同的 `chunk_order[destination][stage]`。打包按目标rank编号排列，通信调用顺序在全部rank上完全相同。

共享顺序控制消费分块的生产顺序，**不是旧 Phase6 每来源独立的 CTA swizzle**。首次准备默认按旧计划 `coords` 中各分块最晚逻辑槽位生成共同优先序，保留旧物理 mask，并记录 `evidence_kind=legacy_proxy`、`recalibrated=false`、`performance_validated=false`。它不是排序后的新到达测量。

### 3.2 量化与变长传输

每个物理tile按 `[source, physical_tile]` 查询mask。本地贡献保留BF16；远端贡献按mask使用原H128/E4M3协议或原始BF16字节。完整H128组的FP8 payload为128字节，附两个FP32 scale，共136字节，对应原BF16的256字节。

每stage分别计算每个source/destination的字节split，使用 `all_to_all_single(..., async_op=True)` 传输。接收方按同一物理mask解析变长包。初始化阶段跨rank核对完整计划和消费模式，避免不同split或不同stage顺序导致通信错误。

### 3.3 两种消费模式

| 模式 | 放行条件 | 共同部分 |
| --- | --- | --- |
| `barrier` | 本次全部stage的通信Work完成后，才允许解码归约 | 相同GEMM切分、编码、字节布局、mask与固定归约数学实现 |
| `pipeline` | 每stage的通信Work完成后即可消费该stage | 与后续stage生产形成潜在重叠 |

这里的barrier模式是本原型的“整次交换完成后消费”对照，不是调用旧 Flux 的 `group_barrier`。stage内的一次交换仍同时覆盖各目标的相应分块；当前消费者按stage顺序处理，没有设备端任意就绪块领取队列。

runtime使用生产、通信launch、消费三条CUDA stream。生产者等待调用方输入就绪；通信launch等待对应生产event；消费者在自身stream上下文调用NCCL `Work.wait()` 后读取数据及scale。调用方等待最后消费event后才能使用完整输出。生产者不等待消费者，避免建立相反方向的循环依赖。

每次调用使用新张量，并为跨stream借用的输入、权重、发送包、接收包、本地贡献和输出登记生命周期。没有重复使用可被下一轮提前覆盖的IPC槽位，因此本原型不需要手写远端ready/ACK；将来复用持久IPC工作区时仍必须实现相应协议。此处的CPU模拟验证不能替代真实GPU并发和长时间运行检查。

分块内部归约保持 `rank+1,...,rank` 的来源顺序，每次加法按BF16舍入。改变消费模式不改变数学归约次序。CUDA graph和隐式CPU fallback均拒绝；开启NCCL blocking wait会破坏预期的异步放行，worker明确拒绝该配置。

## 4. 计划选择与数值边界

[planner.py](../../../tools/h100/phase7/planner.py) 提供：

- `import_phase6_plan`：迁移旧物理mask，派生自然/共享分块顺序，保存来源entry哈希。
- `with_mask_mode`：从已有计划派生BF16、全远端或每来源数量匹配的随机mask，不改变其分块顺序。
- `make_plan`：从明确区分train/heldout/final的证据生成共享顺序和量化计划。
- `evaluate_frozen_plan`：对固定计划使用独立配对证据评估，不修改或重新拟合计划。

定向选择以**已经测量过的完整候选mask**为单位，比较配对完整算子 `baseline_us−candidate_us` 的中位数净收益，同时满足每来源预算和最大绝对输出误差约束。包含并列拖尾来源的组合可以作为一个完整候选；不拼接未经测量的单项收益，也不在完整算子耗时之外再次扣除已经包含的编解码成本。

证据必须包含实际runtime/input身份、接收端相对时钟、分块到达记录、完整算子对照、编码/解码/调度诊断成本和数值误差；干预记录必须匹配冻结顺序哈希并声明在该顺序后重新测量。缺少train干预证据时，`targeted`报错，不能自动伪造优化mask。确有测量但没有正净收益候选时允许回到全零mask。

用于调参的数据不能再充当最终独立评估。final评估要求新的配对块、真实GPU来源、采样关闭和同runtime身份；合成测试证据始终标记rehearsal，不能升级为真实校准。

目前的通用benchmark尚不自动采集上述精细校准schema，不能直接将普通窗口耗时JSON当作到达校准输入。默认实验仍使用旧计划代理；新的定向计划可通过Python API传给 `ChunkedGemmRS`，在独立校准后再冻结到后续实验版本。

## 5. 四组实验与完整模型接入

| 组别 | 消费方式 | 分块顺序 |
| --- | --- | --- |
| E0 | barrier | identity |
| E1 | barrier | shared |
| E2 | pipeline | identity |
| E3 | pipeline | shared |

同一会话四组的物理mask相同。默认每轮4个完整配对块，循环移位平衡执行位置；第二轮反转块和块内顺序。共32个正式窗口，另有4个smoke窗口；每窗口重新启动8个rank，每次默认10步预热＋20步计时。profile单独运行，每组只采集1个正式step，不进入性能报告。

`kind=operator`测完整分块GEMM/通信/解码算子；`kind=model`测一次完整optimizer step，固定GPT6.7B、32层、S2048、micro batch1/global batch4、TP8/DP1/PP1/CP1、BF16，共8192 token/step。不能把operator耗时标成完整step耗时。

[adapter.py](../../../tools/h100/phase7/adapter.py)仅接入选中的 `self_attention.linear_proj` 和 `mlp.linear_fc2`。未选中投影使用原生Megatron；反向采用sequence AllGather及BF16 dgrad/wgrad，即与前期一致的STE近似。拒绝不支持的形状、expert路径及梯度累积融合配置，不做静默fallback。

独立算子smoke包含全GEMM＋独立codec参考、BF16参考，以及STE梯度对照；完整模型smoke检查所有启用投影，并记录有限loss、梯度范数及模块调用数。前向门槛沿用相对BF16 L2不超过5%、相对codec参考L2不超过1%；这不是长期训练质量保证。

模型计时继续使用固定合成token，每窗口重新初始化，保持与Phase6的性能工作负载对应。该loss不能代表真实验证集或多种子收敛结果。

## 6. 运行方式

### 6.1 CPU准备与检查

在仓库根目录执行。输出路径必须不存在，不覆盖失败会话或历史证据。

```bash
python3 tools/h100/phase7/phase7.py prepare \
  outputs/h100/phase7/attention-chunk8-run1 \
  --kind operator --scope attention --chunk-tiles 8 --quantization legacy

python3 outputs/h100/phase7/attention-chunk8-run1/scripts/phase7.py verify \
  outputs/h100/phase7/attention-chunk8-run1

# 默认只打印36个worker窗口，不查询GPU、不启动进程。
python3 outputs/h100/phase7/attention-chunk8-run1/scripts/phase7.py run \
  outputs/h100/phase7/attention-chunk8-run1
```

MLP算子使用新的输出目录和 `--scope mlp`。`--quantization bf16`为同分块runtime的全零mask对照，另支持`all_remote`和`random_matched`。改变粒度、预算或mask后新建会话，不能重写已冻结计划。

准备阶段复制运行时、控制/统计脚本、独立codec参考、两种顺序计划和父计划，逐文件记录SHA256。后续优先调用会话内的 `scripts/phase7.py`；live控制脚本若与冻结版本不一致，会拒绝运行或生成报告。

本次已实际准备并完成dry-run的三个会话如下。每个会话登记36个worker窗口，`results/`为空，`gpu_validated=false`；这些是可审查的冻结输入，不是GPU结果。**下表三个 `outputs/h100/phase7/*-20261017-dev/` 目录仅保留本地，不随 Git 同步；克隆后须另行复制，或按本节命令新建会话。**

| 会话 | 本地冻结manifest | 本地dry-run |
| --- | --- | --- |
| Attention算子 | [attention-chunk8-20261017-dev](../../../outputs/h100/phase7/attention-chunk8-20261017-dev/manifest.json) | [命令与调度](../../../outputs/h100/phase7/attention-chunk8-20261017-dev/dry-run.txt) |
| MLP算子 | [mlp-chunk8-20261017-dev](../../../outputs/h100/phase7/mlp-chunk8-20261017-dev/manifest.json) | [命令与调度](../../../outputs/h100/phase7/mlp-chunk8-20261017-dev/dry-run.txt) |
| 双模块完整模型 | [model-both-chunk8-20261017-dev](../../../outputs/h100/phase7/model-both-chunk8-20261017-dev/manifest.json) | [命令与调度](../../../outputs/h100/phase7/model-both-chunk8-20261017-dev/dry-run.txt) |

完整模型会话登记的目标Megatron路径为 `/workspace/Megatron-LM`。迁移时可以移动整个会话并用新位置调用冻结脚本；若目标依赖路径或实验配置不同，则重新prepare，不能编辑已有manifest适配。

### 6.2 在H100端执行

需要单节点8张132-SM H100和兼容的PyTorch/NCCL环境；本次CPU张量测试使用PyTorch `2.6.0+cu124`。以下路径按实际H100环境设置，命令本身尚未在该机器执行：

```bash
export PHASE7_PY=/opt/conda/envs/flux-h100/bin/python
export PHASE7_ROOT="$PWD/outputs/h100/phase7/attention-chunk8-run1"
unset TORCH_NCCL_BLOCKING_WAIT NCCL_BLOCKING_WAIT

"$PHASE7_PY" "$PHASE7_ROOT/scripts/phase7.py" run "$PHASE7_ROOT" \
  --stage preflight --python "$PHASE7_PY" --execute

"$PHASE7_PY" "$PHASE7_ROOT/scripts/phase7.py" run "$PHASE7_ROOT" \
  --stage timing --python "$PHASE7_PY" --execute

# 独立profile，须先通过预检，产生各rank的压缩Chrome trace。
"$PHASE7_PY" "$PHASE7_ROOT/scripts/phase7.py" run "$PHASE7_ROOT" \
  --stage profile --python "$PHASE7_PY" --execute
```

代码不强制 `CUDA_DEVICE_MAX_CONNECTIONS=1`；运行环境值会被记录并要求跨rank/窗口一致。先在目标主机确认资源可用，再执行显式GPU入口。本次未启动远端作业或排队程序。

### 6.3 完整模型会话

Megatron固定提交为 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`，执行前检查提交及tracked工作树。建议先完成两种shape的算子预检，再启动模型实验。

```bash
python3 tools/h100/phase7/phase7.py prepare \
  outputs/h100/phase7/model-both-chunk8-run1 \
  --kind model --scope both --chunk-tiles 8 --quantization legacy \
  --megatron /workspace/Megatron-LM
```

之后使用该会话冻结脚本执行 `preflight`、`timing`及必要的`profile`。`--scope attention`或`--scope mlp`分别限制启用投影。每个scope使用独立会话，不把不同scope的运行拼成同场配对块。

## 7. 报告和证据检查

窗口耗时为八rank最大elapsed除以正式step数；算子报告单位为 `ms/operator`，模型报告为 `ms/full_optimizer_step`。逐轮报告E1/E0、E3/E2、E2/E0、E3/E1及顺序×消费方式的交互效应。

交互使用每块的 `log(E3/E2)−log(E1/E0)`，正向改善按 `100×(1−exp(mean(log交互比)))` 表示。置信区间按完整四组配对块bootstrap10000次，不把rank、微批或step当独立重复，也不通过比较两项显著性代替交互检验。各区间不做多重比较校正。

报告要求八rank完整、GPU身份唯一、固定设备与软件环境、同rank跨窗口初始化/输入指纹一致、正确variant/round/block/directory、冻结计划与控制脚本哈希一致。模型还要求所有目标层调用齐全、20步loss/梯度有限及零skip。预检必须覆盖全部四组；正式报告不接受缺块、不完整窗口或把同一窗口复制到另一个block。

原始记录在 `results/`，预检为 `preflight.json`，逐轮结果为 `report.json`；profile在各组的 `rank*-profile.json.gz`。阶段开始与结束状态分文件记录；已启动阶段或已有输出目录拒绝覆盖，失败保留现场后新建会话。

新增机制是否有价值，应同时看最终算子/step耗时和trace里的生产/消费重叠。原型内部加速不等于优于历史原生Flux、Megatron或融合候选；移植融合路径后仍需重新验证。

## 8. 本次CPU验证

本地存在CUDA 12.8编译器，但没有可见GPU；使用已有环境 `/data/home/scyb672/run/conda_envs/flux-megatron-a800/bin/python`，PyTorch `2.6.0+cu124`，`torch.cuda.is_available()`为False。没有安装新环境或执行GPU测试。

```bash
/data/home/scyb672/run/conda_envs/flux-megatron-a800/bin/python \
  -m unittest discover -s tools/h100/phase7 -p 'test_*.py' -v

python3 -m unittest discover -s tools/h100/phase6 -p 'test_ablation3.py' -v
python3 tools/h100/phase7/snapshot.py verify \
  outputs/h100/phase7/phase6-baseline-20261017 --original
```

Phase7最终完整回归57项全部通过，包括24项计划测试、21项协议/报告/模型接入测试、12项真实CPU张量及传输调度模拟测试。覆盖：独立Phase4编码参考逐位一致、变长包偏移与split守恒、所有粒度的物理tile覆盖、固定ring归约舍入、屏障/流水的消费放行差异、跨stream生命周期登记、STE反向、计划证据隔离和冻结输入/窗口篡改拒绝。

Phase6原有消融3回归13项全部通过；原版快照3999个文件及91项校准输入引用校验通过，Python语法编译、文档本地链接和diff空白检查通过。详细测试日志和核验JSON仅保存在本地 `logs/h100/phase7/cpu-20261017/`；`logs/h100/phase7/cpu-*/` 已加入 `.gitignore`，不随仓库同步。测试源码和本节验证摘要继续保留。

CPU传输模拟不运行CUDA/NCCL；它验证控制依赖和数据布局，不能证明GPU前进性或重叠。合成报告测试仅使用临时目录，并标记非GPU证据。

本次实现的下一道门槛是目标H100端的算子smoke、受控延迟/重复调用和profile检查。通过后再进行完整step测量；只有确认净收益空间，才将分块发布与消费机制迁移到原融合GEMM的epilogue及持久工作区。
