# H100 GPT 6.7B TP8 消融实验4：两次完整实测与复测入口

更新：2026-10-18。依据回传提交 `62e07733`，对 `ablation4-gpt67-tp8-both-20261018-hostservice` 的 `compare1`、`compare2` 进行 CPU 离线严格核验与复算。**两次均已完成，每次 `--blocks 5 --rounds 2 --nvls-enable 0`，每次50个正式窗口，两次共100个。** 两套 `report.json.completed=true`、`all-state.json.status=completed`；本页统计只使用这两套消融4实测，不混入消融3。

**本页实测表格仍为每窗口20个正式step的历史结果。** 后续新建的消融4/5会话统一改为预热10步、正式60步；两次复测共6000个正式step。新串行脚本先消融5两次、再消融4两次，见[执行说明](消融实验5-选择性量化.md#9-先消融5两次再消融4两次统一60步)。尚无60步GPU结果，本页既有耗时/loss和原始会话不改写为60步。

## 1. 实测结论

- **C 在两次实验、四轮中均快于融合 v2。** 两次合并各自两轮后的配对耗时下降分别为 **3.539% [3.312%, 3.766%]**、**3.199% [2.882%, 3.486%]**；四个逐轮点估计为3.171%–3.578%，对应区间均为正。C 的整步平均为516.361、517.240 ms，融合 v2 为535.309、534.331 ms。
- **交替排序与到达优先的单独增益没有得到本次数据支持。** A/Flux 两次合并的区间都跨零；B/A 分别为−0.245%、−0.267%，区间也跨零。compare1第2轮、compare2第1轮的 B/A 区间完全为负；compare2合并的 B/Flux 为−0.354% [−0.634%, −0.045%]。不能用“差距小”宣称严格等效，也不能把个别负区间推广成长期稳定退化。
- **固定融合实现已经占据相对朴素量化的大部分收益。** Flux v2 相对 Megatron＋量化的配对耗时下降为46.942%、47.031%；C 相对 Megatron 为48.820%、48.725%。A/Flux 与 B/A 是本实验可以分离的调度比较，C/B 的3.768%、3.540%同时包含选择性量化和候选实现变化，不能解释为纯 mask 收益。
- 四轮均未触发“本轮同组中位数×1.10”的慢窗口阈值，主结果没有删除窗口，敏感性结果与主表一致。

## 2. 数据完整性与固定条件

| 项目 | compare1 | compare2 | 合计 |
| --- | ---: | ---: | ---: |
| 轮数 / 五组完整配对块 | 2 / 10 | 2 / 10 | 4 / 20 |
| 正式窗口 / rank记录 | 50 / 400 | 50 / 400 | 100 / 800 |
| 正式 optimizer step（不重复计rank） | 1000 | 1000 | 2000 |
| smoke/profile 窗口 / rank记录 | 10 / 80 | 10 / 80 | 20 / 160 |
| profile trace / 策略角色mapping | 40 / 8 | 40 / 8 | 80 / 16 |

两次 run_id 分别为 `36f5fe380921468ca16a217009f607f3`、`4587cb5d32a446638726ee3c4ef6aa1f`。两次独立新建模型、窗口和预检；运行库、冻结 worker、校准计划、模型/输入/环境指纹及通信配置的登记一致，未复用计时窗口。

单节点 `gxn75`，8×H100 80GB HBM3、每卡132 SM。GPT6.7B、32层、H4096/FFN16384、32 heads、S2048、micro batch1/global batch4、TP8/DP1/PP1/CP1、BF16，seed1234、token_seed4321，padded vocab33792。Python3.10.20、PyTorch2.6.0+cu124、运行CUDA12.4、NCCL2.21.5，编译CUDA12.8/GCC11。Megatron提交固定为 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`；本次构建/准备登记的维护提交为 `92cd94bc58e41aae060959782f65181e89ca404b`，C计算来源为 `ff38f5b1ff1c80cb831f58c64b95f42b0379879a`。

五组均作用于32个 `mlp.linear_fc2` 和32个 `self_attention.linear_proj`，共64个输出投影：

| 简称 | 调度与执行实现 | 远端量化 |
| --- | --- | --- |
| Megatron | 原生GEMM＋Tensor TACO通信；不导入Flux | 全远端 |
| Flux v2 | 原始Flux顺序；固定融合v2、CTA解码、单工作区、逐次配置/复位、输出clone | 全远端 |
| A | 与Flux v2同实现，改为基础交替顺序 | 全远端 |
| B | 与Flux/A同实现，在交替基础上增加冻结到达表 | 全远端 |
| C | 与B同到达表；保留候选优化codec、warp解码、双工作区及实例持有配置/输出 | 定向选择性mask |

Attention shape为 `[2048,4096,4096]`、本地K512；MLP为 `[2048,4096,16384]`、本地K2048。全量组每来源量化448个远端tile；本次C的Attention每来源选择数为 `[7,7,7,7,4,7,7,7]`，MLP为 `[7,7,7,7,1,7,7,7]`。本地贡献、GEMM及既定反向路径保持BF16/STE；“完整量化”不表示整网FP8或权重量化。

**C的选择性量化预算上限是远端贡献的 `1/64 = 1.5625%`，不是 `1/16 = 6.25%`。** [冻结配置](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/submission.json)中的 `fraction=0.015625`；每来源448个远端tile最多选择 `448/64=7` 个，并不要求用满预算。选择单位是 `[source rank, physical tile]` 的来源贡献：本次 [实际mask](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/scripts/selection-plan.json) 中，Attention合计53/3584（约1.479%），MLP合计50/3584（约1.395%）；两角色按相同调用次数合计103/7168（约1.437%）。这是远端通信贡献的量化比例，不是模型权重的量化比例；其余远端贡献和本地贡献仍为BF16。

两次均冻结 `NCCL_NVLS_ENABLE=0`、`NCCL_DEBUG=WARN`、`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`；通信快照中的 `/etc/nccl.conf` 和 `/root/.nccl.conf` 均不存在。完整五组定义、构建隔离及通信快照规则见 [消融实验4](消融实验4.md)。

## 3. 全样本完整step耗时

每窗口先预热10步、计时20步，每步含4个微批、8192 token，覆盖全模型前向/反向、通信、optimizer更新、调度及清梯度。窗口耗时为 `max(8 rank elapsed_seconds) × 1000 / 20`，不是某层、某个GEMM或八rank均值。每组每轮5个窗口，每次合并10个窗口，两次合计每组20个窗口。

| **指标** | **C：完整版（选择性量化）** | **A：交替＋完整量化** | **B：交替＋到达＋完整量化** | **Megatron＋量化** |
| --- | ---: | ---: | ---: | ---: |
| 平均耗时（ms/step） | 516.800 | 535.028 | 536.399 | 1008.837 |
| C完整版相对耗时下降 | 0.000% | +3.407% | +3.654% | +48.773% |
| 平均训练loss | 8.432632 | 8.344758 | 8.344758 | 6.720901 |
| 最终训练loss的平均 | 5.509021 | 6.573645 | 6.573645 | 1.926777 |

上表合并compare1、compare2的全部正式窗口，每组20个窗口、400个正式optimizer step。平均耗时直接对20个未舍入窗口取算术平均；由于两次各10个窗口，也等于两次未舍入均值的等权平均。

“C完整版相对耗时下降”按 `100 × (1 − C平均耗时 / 该列平均耗时)` 计算，正值表示C更快，C自身为0。此行是合并均值的描述性对照，不是第4节同轮同块的配对几何平均下降，也不附带配对置信区间；计算使用未舍入值，最后显示三位小数。

**两项loss都从本次原始正式计时记录计算，不包含预热、smoke或profile。** 已核验100个窗口的800份rank文件哈希、完成/运行身份、20步loss长度与有限值；每窗口八rank的 `losses` 序列完全一致，只计rank0一次，不能把同一TP组的八份记录当成八个独立样本。每组取窗口数 `W=20`：

```text
平均训练loss = sum_w(sum_s(losses_rank0[w][s])) / (W × 20)
最终训练loss的平均 = sum_w(losses_rank0[w][-1]) / W
```

“最终训练loss”是各窗口第20个正式step的 `losses[-1]`，先取每窗口末步值，再跨20个窗口求算术平均，不是只取整套实验最后一个窗口。loss均使用未舍入值计算，显示六位小数。两次实验各组的这两项loss均值分别一致；Flux/A/B的loss均值相同来自本次原始记录。每窗口重新初始化模型，这些短程合成token记录不代表长期收敛。

分实验耗时与合并吞吐保留如下。吞吐按总token数除以总时间计算，即 `8192000 / 平均ms`；中位耗时分别基于各次10个窗口，不将两次中位数取平均。

| **指标** | **C：完整版（选择性量化）** | **A：交替＋完整量化** | **B：交替＋到达＋完整量化** | **Megatron＋量化** |
| --- | ---: | ---: | ---: | ---: |
| compare1 平均耗时（ms/step） | 516.361 | 535.266 | 536.579 | 1008.917 |
| compare1 中位耗时（ms/step） | 516.218 | 535.903 | 536.741 | 1007.344 |
| compare2 平均耗时（ms/step） | 517.240 | 534.790 | 536.220 | 1008.758 |
| compare2 中位耗时（ms/step） | 517.046 | 534.563 | 536.204 | 1007.207 |
| 两次合并吞吐（tokens/s） | 15851.38 | 15311.35 | 15272.21 | 8120.24 |

逐轮平均耗时（完整step；每格5个窗口）：

| **指标** | **C：完整版（选择性量化）** | **A：交替＋完整量化** | **B：交替＋到达＋完整量化** | **Megatron＋量化** |
| --- | ---: | ---: | ---: | ---: |
| compare1 第1轮平均耗时（ms/step） | 516.670 | 536.480 | 535.988 | 1007.047 |
| compare1 第2轮平均耗时（ms/step） | 516.051 | 534.051 | 537.169 | 1010.786 |
| compare2 第1轮平均耗时（ms/step） | 516.585 | 534.517 | 536.435 | 1008.357 |
| compare2 第2轮平均耗时（ms/step） | 517.896 | 535.063 | 536.005 | 1009.160 |

## 4. 同场配对比较

以下每列使用对应实验的10个完整五组块，将两轮合并；保持同一轮、同一块内的X/Y配对。配对下降为 `100 × [1 − exp(mean(log(T_X/T_Y)))]`，正值表示X更快。95%区间按完整五组块bootstrap10000次、seed20261009计算；20个step和8个rank不视作独立样本。未给两次实验额外拼接一个跨批次置信区间。

| X / 参考 Y | compare1 配对耗时下降% [95% CI] | compare2 配对耗时下降% [95% CI] |
| --- | ---: | ---: |
| Flux v2 / Megatron | +46.942 [+46.769, +47.116] | +47.031 [+46.887, +47.181] |
| A / Megatron | +46.946 [+46.789, +47.125] | +46.985 [+46.847, +47.132] |
| A / Flux v2 | +0.008 [-0.274, +0.337] | -0.086 [-0.295, +0.104] |
| B / Megatron | +46.816 [+46.657, +46.977] | +46.843 [+46.676, +47.029] |
| B / Flux v2 | -0.237 [-0.566, +0.055] | -0.354 [-0.634, -0.045] |
| B / A | -0.245 [-0.583, +0.086] | -0.267 [-0.521, +0.013] |
| C / Megatron | +48.820 [+48.684, +48.972] | +48.725 [+48.554, +48.899] |
| C / Flux v2 | +3.539 [+3.312, +3.766] | +3.199 [+2.882, +3.486] |
| C / A | +3.532 [+3.336, +3.726] | +3.282 [+2.913, +3.594] |
| C / B | +3.768 [+3.494, +4.030] | +3.540 [+3.226, +3.893] |

逐轮负例保留：compare1第2轮 B/A 为 **−0.584% [−0.997%, −0.052%]**；compare2第1轮 B/A 为 **−0.359% [−0.634%, −0.041%]**。其余轮次与最好单块、配对中位加速比可直接查两套原始报告。C/B改变codec、工作区、配置/输出生命周期与量化覆盖率，不能用该比较单独证明选择性量化有效。所有区间未做多重比较校正，协议无4%验收门槛。

## 5. 慢窗口、数值检查与结论边界

| 实验 / 轮次 | 最大窗口耗时 / 本轮同组中位数 | 保留完整块 | 排除块 |
| --- | ---: | ---: | --- |
| compare1 / 第1轮 | 1.005442 | 5/5 | 无 |
| compare1 / 第2轮 | 1.006074 | 5/5 | 无 |
| compare2 / 第1轮 | 1.008560 | 5/5 | 无 |
| compare2 / 第2轮 | 1.010074 | 5/5 | 无 |

固定阈值为本轮同组原始窗口中位数的1.10倍，任一组触发则同步排除对应整块五组；不迭代阈值。本次全部20个完整块保留，不能把相对朴素量化的大幅差距解释为剔除慢窗口造成。

严格报告逐窗口核验八rank齐全、实际库登记、64个投影的路由/调用、通信快照、初始化/RNG/token/运行环境、有限loss/梯度及零skip/意外fallback；数值预检采用相对BF16 L2≤5%、相对独立codec参考L2≤1%的既定门槛。每个窗口重新初始化训练，因此这些合成token上的短程记录不构成真实语料准确率、验证困惑度或长期收敛结论。

## 6. 核验与原始证据

本次在临时目录建立只读输入链接，调用维护代码 `ablation4_report.report(..., offline=True)`，重新核验冻结文件、预检/计时证据哈希及全部800份正式rank，并复算两套完整报告。JSON序列化后仅 `offline` 标志与回传报告不同，其他字段（含逐窗口耗时、配对区间、敏感性与证据哈希）全部一致；两次 `measurement_identity` 和角色计划哈希也相同。未写入或覆盖原始会话的报告、日志、状态及rank记录，未在本机重新读取H100动态库或启动GPU。

60步改动后的兼容性复核也已完成：再次在临时目录复算这两套20步历史报告，除新增的 `execution_timing={warmup_steps:10,timed_steps:20}`、每次 `formal_optimizer_steps=1000` 以及离线标志外，全部原有JSON字段仍与回传报告一致。

- compare1：[报告JSON](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/report.json)、[报告Markdown](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/report.md)、[submission](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/submission.json)、[预检](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/preflight.json)、[完成状态](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/all-state.json)。
- compare2：[报告JSON](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare2/report.json)、[报告Markdown](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare2/report.md)、[submission](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare2/submission.json)、[预检](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare2/preflight.json)、[完成状态](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare2/all-state.json)。
- [原始compare1窗口](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/results/)、[原始compare2窗口](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare2/results/)、[执行端路径/环境登记](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/handoff/session-paths.json)、[构建登记快照](../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/provenance/build-manifest.json)。

## 7. 容器内再跑两次

已新增 [`ablation4_repeat.py`](../../../tools/h100/phase6/ablation4_repeat.py)。在原75容器、仓库根目录执行：

```bash
cd /workspace/TempName
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  tools/h100/phase6/ablation4_repeat.py --execute
```

默认复用本次已完成的四套构建 `outputs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/build`，在 `logs/h100/phase6/ablation4-repeat-时间戳/` 下创建全新的 `compare1`、`compare2`，不覆盖上面这批结果。默认Python为启动脚本的解释器，运行CUDA为 `/usr/local/cuda-12.4`。换路径时显式传参：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  tools/h100/phase6/ablation4_repeat.py \
  --build /workspace/TempName/outputs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/build \
  --out /workspace/TempName/logs/h100/phase6/ablation4-new-batch \
  --megatron /workspace/Megatron-LM --cuda-home /usr/local/cuda-12.4 --execute
```

脚本固定每次 `--blocks 5 --rounds 2 --nvls-enable 0`，新窗口预热10步、正式60步，每次3000个正式step，两次共6000个。两套先分别prepare和dry-run，核对输入/步数一致、run_id独立；然后顺序执行各自的 `run --stage all --execute`，各自重做8项mapping和10个smoke/profile窗口，再执行50个正式窗口并调用严格 `report`。第一套报告与阶段状态通过后才开始第二套；任一步失败即停止，保留现场，不自动重试。每次启动沿用原控制器的GPU空闲检查，运行中不周期查询占用。

输出包括 `compare1/report.{json,md}`、`compare2/report.{json,md}`、各阶段日志和 `batch-state.json`。只有两套严格报告均通过且运行身份一致，批次状态才为 `completed`，登记100个正式窗口/800份正式rank。指定的 `--out` 必须不存在；省略 `--execute` 只打印完整命令计划，不创建目录、不启动子进程或查询GPU。脚本前台执行，不自动建立后台服务；需要独立托管时可由既有容器监督进程调用。

脚本的5项CPU测试已通过，覆盖固定参数与两次顺序、首轮失败/报告不完整阻止次轮、输入漂移拒绝、旧目录保护及只读计划。测试用临时模拟记录，不代表已经启动新一批实验。
