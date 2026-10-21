# H100新baseline：保持消融6的A/B/C，换用作者原版TACO CUDA编解码

更新：2026-10-21。**本轮重新测五组：消融6的A/B/C保持不变，另两组改为“Flux＋独立TACO CUDA量化”和“Megatron＋独立TACO CUDA量化”。** 两个baseline共用作者源码中的独立融合压缩、解压kernel；通信使用PyTorch/NCCL All-to-All。实现与启动器已完成，本页尚无新的八卡实测结果。

旧 [消融6](消融实验/消融实验6-对比baseline.md) 的两组baseline使用PyTorch张量操作实现编解码，未执行作者的融合CUDA kernel。旧 [约48%的耗时下降](消融实验/base-phase6-ablation6.md) 只针对那两个朴素baseline，不能当作本轮结果，也不能解释为相对论文完整TACO系统的收益。

## 1. 五组的固定范围

| 组别 | 实现 | 量化范围及生命周期 |
| --- | --- | --- |
| A | 消融6原A：融合v2＋基础交替 | 全远端量化；单工作区、原配置/复位与输出clone |
| B | 消融6原B：融合v2＋原到达表 | 全远端量化；与A相同生命周期 |
| C | 消融6原C：原candidate、交替＋原到达表 | **原1/64预算上限与冻结mask**；双工作区、实例持有配置/输出 |
| Flux＋量化 | 原生Hopper SM90 GemmOnly产生BF16 partial，再独立编码、通信、解码、归约 | 全远端量化；不使用GemmRS或融合v2 |
| Megatron＋量化 | 保留原生Megatron GEMM，将前向RS替换为同一独立编码、通信、解码、归约 | 全远端量化；不加载Flux |

A/B/C直接复用原消融6使用的ablation4父构建，包含原动态库、角色计划、到达表和C的mask；不重做校准，不改成消融7的1/16或op4的闭包mask。本轮新代码没有修改旧A/B/C源文件。默认父构建为：

```text
outputs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/build
```

五组作用范围均为32层的Attention输出投影和MLP输出投影，共64个RowParallelLinear前向RS；本地贡献保持BF16。反向继续沿用原实验的BF16 AllGather及dgrad/wgrad流程，不另外扩大到其他通信位置。完整训练每步仍包含前向、反向、通信、optimizer更新和清梯度。

## 2. “接近原本TACO”的具体含义

参考作者 [COCCL_TACO仓库](https://github.com/liuman1125-ops/COCCL_TACO)，固定提交 **`ca717e05b6dd6d21fff4d2617d209c95be63f3ef`**。本地已随工具保存 [原始taco.cu](../../../tools/h100/phase6/baseline_upstream/taco.cu)、[RS配置](../../../tools/h100/phase6/baseline_upstream/taco_RS.config)、[许可证](../../../tools/h100/phase6/baseline_upstream/LICENSE.txt) 和 [来源及哈希](../../../tools/h100/phase6/baseline_upstream/provenance.json)，远端无需下载另一份TACO仓库。

构建器从固定源码中原文提取设备辅助函数及以下两个kernel，**不重写其设备端计算体**：

- `fused_ash_compress_kernel<__nv_bfloat16,128>`：统计、自适应缩放、Hadamard、FP8量化在独立kernel中完成。
- `fused_ash_decompress_kernel<__nv_bfloat16,128>`：独立解压、逆Hadamard及尺度恢复，输出BF16。

固定E4M3、SATFINITE、H128、`target_range=448`、`lambda=1e-6`，每组两个FP32 scale。保持作者的128线程CTA及shared-memory变换，不采用融合v2的单warp四元素重写。必要的移植仅包括头文件、C ABI启动包装及当前PyTorch CUDA stream接入；不调用COCCL的配置解析、内存管理或集合通信API。

每个目标rank的packet保持作者的布局：`[FP8 payload][FP32 quant scales][FP32 adaptive scales]`。每128个输入元素对应136字节，七个远端目标的每rank每投影逻辑发送量为 **7,798,784字节**，BF16对应14,680,064字节，减少46.875%。这是逻辑字节数，不是实测链路流量。

只支持本实验完整H128分组：M2048、N4096、TP8；Attention/MLP的K_local分别为512/2048。拒绝不整除的尾组，避免将作者尾组处理与本项目补零布局混为同一协议。

两组执行路径为：

```text
本地BF16 GEMM（Flux GemmOnly或原生Megatron）
  → 拼接七个目标的远端partial
  → 作者独立融合压缩kernel，一次launch覆盖七个packet
  → NCCL All-to-All（本地split为0）
  → 作者独立融合解压kernel，一次launch覆盖七个packet
  → 原来源顺序rank+1,…,rank逐次BF16相加，本地partial直接加入
  → 返回独立输出
```

这里保留独立codec之外的张量拼接、临时分配和逐来源归约开销；没有增加GEMM-量化融合、解码-归约融合、双缓冲或自定义通信流水。作者kernel内部已有的融合是要保留的原实现部分。

**因此本轮是“原版TACO codec＋NCCL通信”的移植基线，仍不等同于论文完整TACO系统。** COCCL的集合通信调度、分块重叠及完整系统集成不在本轮范围内。比较结果应按此名称报告。

## 3. 实验规模及统计口径

沿用消融6的GPT6.7B、32层、H4096/FFN16384、32 heads、S2048、TP8/DP1/PP1/CP1、BF16、micro batch1/global batch4；seed1234、token_seed4321，每optimizer step为8192 token。固定`NCCL_NVLS_ENABLE=0`，其余通信环境沿用消融6的捕获与核验机制。

默认串行运行 **compare1、compare2、compare3三次独立比较**。每次5块×2轮×5组＝50个正式窗口；第二轮反转块顺序和块内顺序，每窗口独立初始化、10步预热＋50步正式计时。总计：

| 项目 | 每次比较 | 三次合计 |
| --- | ---: | ---: |
| 正式计时窗口 | 50 | 150 |
| 正式rank记录 | 400 | 1200 |
| 正式optimizer step | 2500 | 7500 |
| smoke/profile窗口 | 10 | 30 |
| profile trace | 40 | 120 |

7500是独立窗口累计工作量，不是同一个模型连续训练7500步。每次都有独立run ID、预检和严格报告，上一比较完成后才准备下一比较。

窗口时间为八rank最大墙钟除以50。配对下降率使用`100×[1−exp(mean(log(T_X/T_Y)))]`，按完整五组块bootstrap10000次、seed20261009。主表保留全部样本及负收益；慢窗口只进入独立敏感性附表，无4%或其他预设性能通过门槛。报告同时保存平均/中位step耗时、吞吐、rank0正式50步平均loss及末步loss。

Flux/Megatron对照共用同一codec和通信，但仍包含GEMM及实现管理开销。C/A、C/B及A/新Flux基线也各自包含多项实现差异，不能将全部收益归因于单一调度或量化策略。训练数据仍为固定四个合成微批的重复训练，50步loss不代表真实语料长期收敛。

## 4. 计时前必须通过的检查

1. **A/B/C来源不变。** 核验父构建、动态库、角色计划、实际mapping与C的mask；三次比较的输入、构建和软件身份必须一致。
2. **两基线实际加载同一CUDA库。** 固定源码、设备端提取片段、编译命令和动态库SHA256；禁止用旧PyTorch codec或融合v2库替代。A/B/C不得加载新codec库。
3. **独立codec检查。** 每个新基线的smoke进程在模型初始化前验证零值、小值、离群值及实际七packet完整shape，比较独立tensor参考及实际packet解释结果，并检查后续调用未覆盖已返回输出。BF16相对L2上限5%；与codec参考之差按BF16输入范数归一化后上限1%；实际packet检查上限0.5%。诊断使用独立随机数生成器，不进入模型计时或profile。
4. **模型检查。** 保留消融6全部64个目标投影的forward精度检查、初始化/RNG/token身份、有限loss/grad norm、零skip/fallback和调用次数核验。正式每投影200次，另有40次预热；不根据新结果放宽阈值。
5. **实际GPU路径检查。** 新基线每份完整step profile必须有256次作者压缩kernel、256次作者解压kernel及对应通信标记；Flux组还须有256次GemmOnly标记。若执行融合v2的解码归约kernel或缺少实际CUDA codec调用则拒绝验收。

任何编译、精度、来源、执行路径或记录完整性检查失败都保留现场并停止，不自动重试或跳过。通过这些短程检查不等于证明与原系统逐位等价或长期收敛。

## 5. 另一台服务器的一行启动

同步本次新增`baseline*.py`、`baseline_codec.cu`、整个`baseline_upstream/`目录及 [detach_baseline.py](../../../tools/h100/phase6/detach_baseline.py)，连同已有实验工具。在原服务器/容器的`/workspace/TempName`目录执行：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_baseline.py --execute
```

默认复用上述A/B/C构建，fresh编译一个原生Flux GemmOnly构建和一个独立TACO CUDA库。沿用CUDA12.8编译、CUDA12.4运行、jobs2、相邻`../Megatron-LM`及原NCCL开发库记录。双fork后台托管，终端关闭后继续；需保持容器和主机运行。

去掉`--execute`只打印计划，不创建目录、不编译、不fork或查询GPU。加`--foreground`可前台诊断。路径变更时使用`--build`、`--megatron`、`--build-cuda-home`、`--cuda-home`、`--python`；若原NCCL开发库已移动，传入`--nccl-root`，需包含`include/nccl.h`与`lib/libnccl_static.a`。

已有消融6的GemmOnly构建可用`--flux-build /path/to/flux-build`复用，仍会严格核验。`--codec-build`仅接受本次固定协议构建的独立库。原A/B/C构建缺失时启动失败，不用新建mask或其他实验的库替代。

默认输出：

```text
logs/h100/phase6/baseline-taco-cuda-3x-50steps-<北京时间戳>/
  batch-state.json                    批次状态与三次比较完成数
  handoff/container-supervisor.log   后台总日志
  flux-build/                       原生GemmOnly构建
  codec-build/manifest.json          上游来源、编译命令、动态库哈希
  compare1/                         独立预检、profile、正式窗口与报告
  compare2/
  compare3/
```

每个compare含`submission.json`、`scripts/`、`provenance/`、`results/`、`preflight.json`及`report.json/report.md`。只有三份完整报告均通过后，批次才标记`status=completed`；它表示实验完成，不保证性能为正。

回传整个批次。若省略大型二进制，保留所有manifest、冻结脚本、rank JSON、trace和报告；离线报告入口为：

```bash
python3 tools/h100/phase6/baseline.py report /path/to/batch/compare1 --offline
```

## 6. 本地验证及待测项

已完成独立codec的真实SM90编译与动态库链接，见 [构建登记](../../../outputs/h100/phase6/baseline-validation-20261021/codec-build/manifest.json)。另在屏蔽GPU的进程中验证了动态库加载、C ABI、非法参数及替换库拒绝，见 [ABI检查](../../../outputs/h100/phase6/baseline-validation-20261021/abi-validation.json)。编译和这些检查均未运行GPU kernel。

**10项CPU测试全部通过，无跳过**，见 [CPU验证登记](../../../outputs/h100/phase6/baseline-validation-20261021/cpu-validation.json) 与 [测试代码](../../../tools/h100/phase6/test_baseline.py)。覆盖A/B/C定义和调度保持、真实生成adapter的五组路由与调用计数、上游kernel原文保持、源码变化拒绝、缺失CUDA profile拒绝、数值门槛/动态库替换拒绝、source-only不可执行、完整50窗口模拟报告及截断记录拒绝、三次启动顺序及失败停止。模拟rank和trace仅用于测试报告器，不作为实验结果。另已通过 [启动预览](../../../outputs/h100/phase6/baseline-validation-20261021/launch-preview.txt)，确认无副作用。

**GPU数值检查、NCCL通信及完整训练性能仍需在另一台八卡H100服务器执行。当前不预填新baseline耗时或收益。**
