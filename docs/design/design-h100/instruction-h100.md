# H100执行端：测试与数据交付说明

更新：2026-10-11。**本页供有H100的服务器上的执行者或新Codex对话阅读。** 当前执行端为 `gxn74`，容器内仓库通常为 `/workspace/TempName`；实际路径以本机部署为准。测试按用户已安排的任务范围和时间预算进行，阅读本页不等于自动启动实验。

另一台开发服务器没有本任务需要的H100，负责接收数据、维护代码和分析；它的新对话入口是 [instruction.md](instruction.md)。本页维护测试流程、产物布局和交付清单；已接收会话的最新索引在[开发端数据索引](instruction.md#已接收数据与当前缺项)，实测结论在 [base-phase6.md](base-phase6.md)。

## 先确定本次任务

主模型为GPT 6.7B：32层、H4096、FFN16384、32 heads、词表32768；默认单节点TP8/DP1/PP1/CP1，S2048/mb1/global4，约6.729B参数。历史约0.64B是12层H2048模型，两者配置、计划和结果分开。模型详情见[统一定义](../model-gpt-6.7b.md)。

默认六组全部保留：参考组 `native`、`original`，必须baseline `native_taco`、`taco_fused`，候选 `remote_arrival_selective`、`interleaved_arrival_selective`。不能因为某个baseline慢或发生OOM而悄悄删除它。当前仅比较完整optimizer step，不把单算子或profile耗时当作主结果。

开始前明确是“新实验”“同条件复测”“原条件续跑”还是“只补交已有数据”。截至提交 `e7a0c49`，旧0.64B完整实验、6.7B前次续跑和最新复测均有72窗口及最终报告，两候选均未达性能门槛；不要重复补跑已完成窗口。最新6.7B对比为 `logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/`，前次为同父目录下 `compare-20261010-153517/`，前次状态注明 `resumed=true`。先核对实际状态与run_id，完整索引见开发端说明。

## 入口

| 内容 | 路径 |
| --- | --- |
| A800/H100 6.7B模型定义、形状与准备入口 | [model-gpt-6.7b.md](../model-gpt-6.7b.md) |
| 新服务器环境、源码迁移与八卡运行步骤 | [单机 TP8 部署指南](deploy-tp8.md) |
| 完整命令、baseline 定义与实现边界 | [H100 Phase6 README](../../../tools/h100/phase6/README.md) |
| 模型、TP、SM 数、基础顺序和预算 | [config.json](../../../tools/h100/phase6/config.json) |
| 6.7B TP4、S1024、micro1/global8 场景 | [config-tp4-s1024.json](../../../tools/h100/phase6/config-tp4-s1024.json) |
| 历史0.64B配置，仅用于旧模型复现 | [TP8](../../../tools/h100/phase6/config-0.64b-tp8.json) · [TP4 S1024](../../../tools/h100/phase6/config-0.64b-tp4-s1024.json) |
| 6.7B阶段记录及历史0.64B完整实测 | [base-phase6.md](base-phase6.md) |
| 独立源码快照与无 GPU 编译 | [build.py](../../../tools/h100/phase6/build.py) |
| 新 H100 校准准备/执行 | [calibrate.py](../../../tools/h100/phase6/calibrate.py) |
| 到达表与量化 mask 拟合 | [plan.py](../../../tools/h100/phase6/plan.py) |
| 冻结模型实验与基线 | [prepare.py](../../../tools/h100/phase6/prepare.py) |
| 单节点 torchrun 控制器 | [run.py](../../../tools/h100/phase6/run.py) |
| 完整性核验与配对统计 | [report.py](../../../tools/h100/phase6/report.py) |
| CPU 协议检查 | [test_protocol.py](../../../tools/h100/phase6/test_protocol.py) |

## 新实验怎么测试

首次部署先完成[部署指南第1–3节](deploy-tp8.md)：固定源码/子模块、安装环境、核对同质H100与GPU分配。Megatron固定提交 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`，软件版本详见[环境说明](../env.md)。确认卡数、SM数、GPU UUID顺序和peer access；默认配置是132 SM，不能对114 SM设备直接沿用。

按以下顺序执行，**命令全文以部署指南和工具README为准**。只看命令计划时不要加 `--execute`；下面标为GPU的阶段会实际使用设备。

| 顺序 | 入口及动作 | 必须得到的证据 / 下一步条件 |
| --- | --- | --- |
| 1 | 新建会话；保存模型配置、代码commit/submodule及补丁、环境和设备记录 | 后续所有阶段使用相同配置与来源；不覆盖历史会话 |
| 2 | `build.py prepare`、`build.py compile`生成 `base/`，仅编译 | baseline/sampler的manifest为 `status=built`，保留compile.log |
| 3 | `calibrate.py prepare`，再 `calibrate.py run --execute`，使用GPU | 全部rank和基础顺序校准成功，生成真实 `calibration-complete.json` |
| 4 | `plan.py`，仅CPU拟合 | 得到本节点/本shape的 `selection-plan.json`；pass 0/1训练，pass 2只诊断 |
| 5 | `build.py prepare --plan`、`build.py compile`生成 `candidates/` | 候选完成编译，配置与base一致，绑定实际计划 |
| 6 | `prepare.py`冻结 `compare/` | submission、scripts、计划副本与库哈希齐全 |
| 7 | `run.py --stage preflight --execute`，使用GPU | swizzle映射、全部策略smoke/profile通过，生成preflight |
| 8 | `run.py --stage timing --execute`，使用GPU | 两轮完整计时；默认六组72窗口、每窗口10步预热＋20步计时 |
| 9 | 原环境执行 `report.py`，仅CPU核验 | 最终report.json/report.md和完成状态；失败时保留已有证据 |
| 10 | 按下方交付清单整理、提交并回传 | 告知数据提交SHA、会话路径、是否完整及尚缺文件 |

具体参数和依赖路径见[部署指南第4–7节](deploy-tp8.md#4-构建-baseline-与采样库)。该流程适用于新会话；同条件复测可以在校准/配置/构建/设备身份核验一致的前提下复用原产物，但必须新建compare、冻结新run_id并重做完整预检和两轮计时，逐项记录复用关系。配置或设备等变化时重新构建/校准。旧失败会话的续跑须遵守本页末尾恢复约定，不能直接重新执行全部命令。

完整结果取每窗口所有rank的最大elapsed除以正式step数，配对收益为 `100*(1-exp(mean(log(T候选/T基线))))`，置信区间按配对块bootstrap计算。默认要求每候选在两轮分别对两个必须baseline达到4%且95%区间下界大于0；门槛如需变化在新实验前固定。`accepted=false`仍是有效结果，执行失败和性能未达标分开记录。

## 生成数据与文件放置规范

所有路径都以 **TempName 仓库根目录**为基准。每个模型/TP/形状/节点校准使用独立会话名，6.7B建议 `gpt67-tp8-YYYYMMDD-HHMMSS`；不要复用旧0.64B的会话目录，也不要把产物放进 `src/`、`tools/` 或 `docs/`。

| 文件类别 | 放置位置 | 是否纳入 Git |
| --- | --- | --- |
| 维护的模型配置、脚本、CUDA/C++源码 | `tools/h100/phase6/`、`src/`、`include/`、`python/` | 是 |
| 实验设计、已核验结论、结果摘要 | `docs/design/design-h100/` | 是；标明模型、节点、配置和证据路径 |
| baseline/sampler源码快照、构建配置、manifest和编译日志 | `outputs/h100/phase6/<会话>/base/` | 是；提交文本证据，编译库/构建缓存除外 |
| 到达置换表与量化mask合并计划 | `outputs/h100/phase6/<会话>/selection-plan.json` | 是；由真实校准生成 |
| 候选源码快照、查表头文件、manifest | `outputs/h100/phase6/<会话>/candidates/` | 是；动态库及构建缓存除外 |
| 接收端到达采样、GPU映射CSV、校准worker与日志 | `logs/h100/phase6/<会话>/calibration/` | 是；全部rank与pass，含失败记录 |
| 对比用脚本/配置快照、输入清单 | `logs/h100/phase6/<会话>/compare/scripts/`、`compare/submission.json` | 是 |
| smoke/profile/计时的每rank记录、trace、进程日志 | `logs/h100/phase6/<会话>/compare/results/` | 是；包括原始 `rank*-profile.json.gz` |
| 自动生成的统计结果 | `compare/report.json`、`compare/report.md`、`compare/preflight.json`及阶段状态文件 | 是；同时在设计文档归纳结论 |
| 每次交接说明、环境记录与补传清单 | `logs/h100/phase6/<会话>/handoff/` | 是；由执行方补充，不冒充工具自动生成文件 |
| `.so`、`.a`、`.o`、wheel、CMake/Ninja构建树、NCCL安装、Conda环境 | 原H100会话目录或环境目录 | 否；执行方保留，必要时另行传输 |
| 如以后使用真实训练语料 | `data/h100/<数据集>/`或外部数据盘 | 否；当前入口不读取这里 |
| 如以后启用保存权重 | `checkpoints/h100/<会话>/` | 否；当前测量入口不保存checkpoint |

**当前合成数据不用提前生成磁盘语料文件。** `--mock-data`/NullTokenizer入口由worker按固定 `token_seed` 生成GPU驻留token bank，模型初始化使用 `seed`；同配置的对照组复用相同生成规则。默认有效token为0–32767，EOD/词表补齐由Megatron处理。种子、模型配置和token哈希保存在配置快照及每rank报告中，不生成待提交Git的 `.bin/.idx` 数据集。

典型目录结构如下，文件由对应工具生成，不手工补写“完成”标志或虚构测量数据：

```text
TempName/
├── tools/h100/phase6/config.json               # 维护的6.7B默认配置
├── docs/design/design-h100/                   # 设计说明和结果摘要
├── outputs/h100/phase6/gpt67-tp8-<时间>/
│   ├── base/                                 # build.py prepare/compile
│   │   ├── manifest.json
│   │   ├── config.json
│   │   ├── compile.log
│   │   ├── original/
│   │   ├── taco_fused/
│   │   ├── sampler_remote_first/
│   │   └── sampler_interleaved/
│   ├── selection-plan.json                   # plan.py
│   └── candidates/                           # 第二次build.py prepare/compile
└── logs/h100/phase6/gpt67-tp8-<时间>/
    ├── calibration/                          # calibrate.py
    │   ├── worker.py / config.json / plan.json
    │   ├── mapping-*.csv
    │   ├── b0-*-instrumented-rank*.json        # 原始到达观察
    │   ├── *.log / execution.json
    │   └── calibration-complete.json          # 全部校准成功后生成
    └── compare/                              # prepare.py / run.py / report.py
        ├── scripts/                          # 包含冻结的selection-plan.json副本
        ├── build/                            # 指向本会话动态库的链接
        ├── submission.json / preflight.json
        ├── results/
        │   ├── smoke-<策略>/rank*.json
        │   ├── profile-<策略>/rank*-profile.json.gz
        │   ├── round1/window-*/rank*.json
        │   └── round2/window-*/rank*.json
        └── report.json / report.md
```

在同一个shell中设置并保存会话路径，然后按[部署指南](deploy-tp8.md)传给每个阶段：

```bash
export REPO="$PWD"  # 必须在TempName根目录执行
export H100_SESSION="gpt67-tp8-$(date +%Y%m%d-%H%M%S)"
export H100_BUILD="$REPO/outputs/h100/phase6/$H100_SESSION"
export H100_LOG="$REPO/logs/h100/phase6/$H100_SESSION"
export H100_CONFIG="$REPO/tools/h100/phase6/config.json"
```

相邻 `Megatron-LM`、Apex源码和Conda环境按[环境说明](../env.md)管理，不放在实验结果目录冒充生成数据。**自 `279fc32` 起，根目录 `.gitignore` 已对 H100 日志和构建文本证据设置例外，不能再认为所有 `logs/`、`outputs/` 都不会随 clone/pull 迁移。** 当前规则保留 `logs/h100/` 下的原始记录及 `outputs/h100/` 下的配置/源码等白名单文件，仍排除编译二进制、主要构建缓存和环境。已跟踪文件也不会因后来加入 ignore 规则而自动停止跟踪。

冻结后保持执行端绝对路径、代码/配置哈希、GPU UUID及顺序一致；改变配置、移动运行目录或更换服务器时建立新会话并重新准备/校准，不通过改manifest绕过核验。接收端可以在自己的仓库根目录下保存相同的相对路径用于分析，原文件内的 `/workspace/TempName/...` 等来源路径保持原样。

## 测试完成或中断后交什么

### 每次必须回传的文件

以下路径中的 `<calibration>`、`<compare>`、`<candidates>` 指本会话的**实际目录名**，允许带时间后缀。现有会话按[开发端数据索引](instruction.md#已接收数据与当前缺项)的实际路径提交，新会话按前面的放置规范生成。

| 类别 | 必须回传的内容 | 分析用途 |
| --- | --- | --- |
| 构建来源 | `base/`及`<candidates>/`内的 `manifest.json`、`config.json`、`compile.log`、源码快照、生成的查表头文件；会话下的环境/辅助脚本 | 确定实际编译内容、模型配置、代码与库哈希；仅上传当前维护源码不能替代冻结快照 |
| 校准原始数据 | `<calibration>/`全部文本产物：`worker.py`、`runtime.py`、`config.json`、`plan.json`、`execution.json`、成功时的 `calibration-complete.json`、`mapping-*.csv`、`b0-*-instrumented-rank*.json`、采样profile JSON及进程日志 | 复核接收端到达观察、采样扰动和GPU映射；保留全部rank及pass 0/1/2，不能只传拟合后的表 |
| 拟合结果 | 构建目录下实际 `selection-plan*.json` 和 `<compare>/scripts/selection-plan.json` 冻结副本 | 复核顺序、物理mask、实际选择率、训练/留出分离；预算是上限，不能人为补满 |
| 实验冻结输入 | `<compare>/submission.json`及 `scripts/`全部文件 | 确定run_id、策略、种子、模型、库路径和哈希 |
| 正确性与执行路径 | `<compare>/preflight.json`、预检状态、`results/mapping-*.csv`、所有 `smoke-*`/`profile-*`目录的rank JSON、`command.json`、`process.log`及 `rank*-profile.json.gz` | 复核数值预算、mask/逻辑字节、实际内核、初始参数/RNG/token一致性；trace不能用截图替代 |
| 完整step原始结果 | `<compare>/results/round*/window-*/`内**全部rank**的JSON，以及每窗口的命令与进程日志 | 取最慢rank耗时，重建配对块与loss/梯度/显存轨迹；只传rank0、均值或挑出的最快窗口不足以复算 |
| 完成报告 | 完成时的 `<compare>/report.json`、`report.md`、`timing-state.json`或`all-state.json`；另做独立核验时一并上传核验脚本和结果 | 区分数据完整、正确性通过、性能门槛通过；`accepted=false`同样是有效结论 |
| 失败与续跑 | 不完整窗口日志、`*.failed-*`备份、失败状态；实际使用的续跑/watchdog脚本与日志、启动/停止时刻及原因 | 解释缺失窗口、显存不足、等待资源和运行条件是否改变；失败记录也要回传 |
| 人工交接 | `logs/h100/phase6/<会话>/handoff/`，内容见下节 | 告知这次新增了哪些数据、哪些尚缺、下一步做什么 |

默认六策略TP8的完整覆盖应为：两种基础顺序共16份校准rank JSON；48份smoke与48份profile rank JSON、48份压缩trace、两份候选GPU映射CSV；两轮各36个计时窗口，共576份计时rank JSON。计时每窗口10步warmup＋20步正式step。改为TP4或增加基础顺序时，按冻结配置和调度表重新计算数量，不硬套这些数字。失败备份目录不参与完成数。

不要求上传合成token的独立数据集、模型checkpoint、Conda目录或Docker镜像；当前入口本就不生成训练语料文件或checkpoint。种子、token哈希、完整模型配置和版本才是重建合成输入所需信息。大型二进制构建仍在H100端保留，至少保留到该会话核验和问题定位结束。

### 交接说明与环境记录放在哪里

在 `logs/h100/phase6/<会话>/handoff/` 中新增 `handoff-YYYYMMDD-HHMMSS.md`，每次回传一份，旧记录不覆盖。这是执行方填写的记录，当前工具不会自动生成。至少包含：

```text
会话 / 本次回传时间（含时区）：
状态：partial / failed / complete；最后完成阶段：
测量源码commit / submodule SHA / 未提交补丁路径：
节点、容器内仓库路径、GPU UUID及rank顺序：
实际base / calibration / plan / candidates / compare相对路径：
模型 / TP / sequence / micro batch / global batch / 两个seed：
完成窗口数 / 计划窗口数；本次新增或补传文件范围：
严格report核验：未执行（原因）/ 成功 / 失败（日志路径）：
性能验收accepted值：未形成最终报告 / 实际值：
失败窗口、OOM等原因、是否续跑及失败备份位置：
是否存在其他GPU进程、资源变化、分配器/NCCL设置变动：
尚缺文件与下一步；是否有仍存活的watchdog/等待任务：
环境记录和额外传输文件的位置、校验和：
```

同目录放 `environment-<时间>.md`、`pip-freeze-<时间>.txt`、`nvidia-smi-<时间>.txt`、`topology-<时间>.txt` 等。环境记录写明Python/PyTorch/CUDA运行时、两套nvcc、驱动、NCCL运行库及编译库、编译器/CMake、Megatron精确SHA、Apex SHA及本地补丁、Python解释器与CUDA路径。GPU占用记录应含测试前和失败时的进程/显存情况；只记录事后空闲状态不能解释计时期间的干扰。

运行相关环境变量至少记录 `CUDA_VISIBLE_DEVICES`、`CUDA_DEVICE_MAX_CONNECTIONS`、设置过的 `NCCL_*`、`PYTORCH_CUDA_ALLOC_CONF`/`PYTORCH_ALLOC_CONF`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS`。只收集与复现相关的项目，不直接上传完整环境变量、代理凭据或登录令牌。构建器manifest的 `gpu_validated=false`、准备阶段submission的同名字段不会随GPU测试自动更新；不得为了“表示完成”修改它们，应看preflight、实际rank记录和最终report。

### 上传方式和接收位置

**沿用当前Git回传方式。** 原始数据提交在既有 `logs/h100/phase6/<会话>/`，配置/计划/源码快照提交在 `outputs/h100/phase6/<会话>/`；结果说明更新 [base-phase6.md](base-phase6.md)。开发端pull后落到 `/data/run01/scyb672/hjr/TempName/` 下相同相对目录，无需另建一份“结果副本”放入 `docs/`。远端容器使用 `/workspace/TempName/` 也不影响这一约定。

每个阶段完成或失败后可以提交阶段快照；最终测量完成再补传报告。不要边写一个JSON/trace边提交它，选择文件已关闭的时点；尚在写入的窗口不算本次已交付，交接中说明。对于已关闭的成功窗口，不改内容、不重命名、不择优替换。

H100端可按如下方式检查待上传范围（提交/推送在执行方按实际分支进行，本页更新不会替它执行）：

```bash
cd /workspace/TempName                         # 按执行端实际仓库路径修改
export H100_SESSION=gpt67-tp8-20261010-112249   # 本次已有会话；新实验换新名称
git status --short -- \
  "logs/h100/phase6/$H100_SESSION" \
  "outputs/h100/phase6/$H100_SESSION" \
  docs/design/design-h100/base-phase6.md

# 仅添加本次会话与对应结论；不要git add整个仓库或强制添加二进制。
git add -- \
  "logs/h100/phase6/$H100_SESSION" \
  "outputs/h100/phase6/$H100_SESSION" \
  docs/design/design-h100/base-phase6.md
git diff --cached --shortstat
git diff --cached --name-only
# 确认无无关改动、环境/凭据、构建缓存后，按实际状态commit并push。
```

`.gitignore`只控制文件能否被发现，并不能代替交付清单：提交前核对缺项，不能用 `git add -f` 整体绕过忽略规则。本次已提交的源码快照是历史证据，不在其中继续开发；代码修复回到 `src/`、`tools/`等受维护入口，并产生新的构建快照。

若后续完整trace或专项profile超出仓库传输限制，可以单独打包传输，但必须在handoff写明归档文件名、下载/存储位置、SHA256和包内相对路径；在接收端恢复到同一会话目录后再核对。不要无说明地删掉trace只保留均值。既有 `279fc32` 的文件保持原样，不因本约定改写Git历史。

### 在原路径完成严格核验

`report.py`虽是CPU逻辑，但会读取manifest、库文件的原绝对路径并核对哈希。因此应由执行端在原环境、原目录完成核验，再回传报告；开发端拉取了文本和trace，不等于拥有可执行构建。

正式窗口全部齐全后执行：

```bash
# H100_LOG指向本次会话日志目录；已有复测须改为实际compare-20261011-114343-repeat等目录。
python tools/h100/phase6/report.py "$H100_LOG/compare"
```

部分完成时交付失败状态、原始窗口及日志，不手工填写完整report。保留 `.so` 等原构建供问题定位，不修改来源路径、哈希或完成标志来绕过核验。开发端的接收核验办法见 [instruction.md](instruction.md#本机收到新数据后怎么检查)。

### 资源、失败和续跑注意事项

1. 已回传的6.7B会话遇到 `native_taco` 解码时的OOM，下一次执行前应确认分配到的8张卡可供本实验使用，并记录显存及其他进程。不要擅自终止他人的任务，也不要把清理其他任务当作默认前提。watchdog的历史`waiting`状态不是当前空闲证明。
2. 保持同节点、同GPU UUID与顺序、同模型/种子/软件/编译库；在校准、预检、计时阶段保持约定的运行条件。调整batch、分配器设置、通信环境或codec来绕过OOM属于实验条件变化，需另建会话，不能拼接成原两轮结果。
3. 维护入口 `tools/h100/phase6/run.py` 会拒绝覆盖已开始的阶段。当前6.7B上传了专用 `outputs/h100/phase6/gpt67-tp8-20261010-112249/resume_timing.py` 及等待脚本；它是该会话的执行证据，含固定 `/workspace/TempName` 路径，不是通用恢复入口。不要删除状态文件后盲目重跑，也不要在本开发服务器启动它。
4. 若原条件下续跑，记录中断与重启时刻、资源状况；保留原失败目录的 `*.failed-<时间>`备份、续跑脚本/日志及重新核验报告。完成文件须先核验再跳过，不能只看文件存在。跨长时间或竞争资源状态变化后的合并结果要明确标为续跑数据；必要时另做完整独占对照，不能把补齐窗口等同于证明计时环境稳定。
5. 完成、用户停止或超出约定时间预算时，执行方停止本任务的控制器和等待/自动重试脚本，确认本任务子进程退出，记录实际状态后回传已有结果。不要因希望得到正收益就无预算地追加实验；完整构建和72窗口测量也不能承诺20分钟内结束。

## 实现边界

量化 MLP 使用重新编译为 SM90 的 Phase6 GemmV2 算法；attention 和原生 Flux 参考使用 Hopper GemmV3。该版本没有实现基于 WGMMA/TMA 的 Hopper 原生融合 TACO epilogue，不能将兼容移植称为完成 H100 性能优化。

必须 baseline 为原生 Megatron＋朴素 TACO、原始顺序 Flux＋全远端融合 TACO；参考为原生 Megatron、原生 Hopper Flux。默认候选为远端优先和交替两种基础顺序，可配置另外两种交替顺序。范围仅 MLP `linear_fc2` 前向，BF16 GEMM、本地 BF16、H128/E4M3 远端协议及 BF16 STE 反向沿用现有设计。

构建与结果分别放 `outputs/h100/phase6/`、`logs/h100/phase6/`。必须从 H100 重新采集到达观察并生成选择计划；工具拒绝 A800 计划和未完成构建。候选准备依赖真实校准，不用合成 CPU 测试输入替代 GPU 证据。

后续GPU工作由H100执行端按用户任务和预算安排，按README执行校准、独立候选编译、swizzle/模型smoke/profile及完整六组/八组两轮测量；本页是交接规范，不会自动启动或恢复这些步骤。


历史无GPU编译记录由[开发端说明](instruction.md#本次无-gpu-验证2026-10-09)保存，不作为当前模型GPU性能证据。
