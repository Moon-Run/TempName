# H100 单节点 Phase6 接续说明

更新：2026-10-11。本目录记录 H100 代码移植及两台服务器之间的实验交接，与 [A800 接续说明](../design-a800/instruction.md) 分开维护。

当前目标是单节点 TP4/TP8 下的 baseline 与“基础顺序＋到达优先＋选择性量化”完整 optimizer-step 对比。默认 TP8/DP1，不支持跨节点。**主模型为 GPT 6.7B：32层、H4096、FFN16384、32 heads、词表32768；默认 S2048/mb1/global4。** 参数量按当前独立嵌入/输出头及词表补齐约6.729B，详见[统一模型说明](../model-gpt-6.7b.md)。本次仅更新交接文档并在 CPU 上核对已拉取文件，没有启动编译、GPU 测试或远端任务。

目前只有另一台服务器 `gxn74` 具备本任务需要的 H100。它负责构建、GPU 校准/测试和原路径核验；本开发服务器负责代码维护、接收证据、CPU 分析和论文整理。以已拉取的提交 `279fc32`（`Add h100 test data`）为界：旧0.64B已完成两轮，6.7B已完成校准和预检，但正式计时只完成43/72窗口。详见下方[已接收数据](#已接收数据与当前缺项)。这是提交中的快照，不能据此判断远端进程目前是否还在运行。

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

## 已接收数据与当前缺项

以下核对基于本开发服务器已拉取的 `279fc32`，不代表此后远端的实时进度。

| 会话 | 已接收证据 | 本次交接状态 |
| --- | --- | --- |
| `tp8-20261009-deployment`，旧0.64B | `logs/h100/phase6/`和`outputs/h100/phase6/`下同名目录；72个计时窗口、576份计时rank记录，smoke/profile记录与trace、最终报告、独立核验及窗口统计 | 完整历史数据；`report.json`两候选均为`accepted=false`，不能当作6.7B结果 |
| `gpt67-tp8-20261010-112249`，6.7B | baseline/候选构建文本、真实校准与选择计划、48份smoke rank记录、48份profile rank记录、48份trace；43个完整计时窗口、344份计时rank记录，以及第44个窗口的失败现场 | 部分完成；`timing-state.json`为`failed`，尚无最终`report.json`/`report.md` |

6.7B的**有效路径**如下。带时间后缀的目录是真实产物，不要为统一命名而重命名，也不要选中早先失败的无后缀 `calibration/`：

| 内容 | 仓库相对路径 |
| --- | --- |
| baseline构建 | `outputs/h100/phase6/gpt67-tp8-20261010-112249/base/` |
| 成功校准 | `logs/h100/phase6/gpt67-tp8-20261010-112249/calibration-20261010-153517/` |
| 选择计划 | `outputs/h100/phase6/gpt67-tp8-20261010-112249/selection-plan-20261010-153517.json` |
| 候选构建 | `outputs/h100/phase6/gpt67-tp8-20261010-112249/candidates-20261010-153517/` |
| 预检、计时与后续报告 | `logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/` |
| 首次失败窗口 | 上述compare下的 `results/round2/window-007-native_taco/` |

该会话冻结配置和构建manifest中的源码提交为 `ff38f5b1ff1c80cb831f58c64b95f42b0379879a`；`279fc32`是**数据回传提交**，两者用途不同。模型以 `compare-20261010-153517/submission.json` 为准：`layers=32, hidden=4096, ffn=16384`，MLP全局形状为 `[2048,4096,16384]`，TP8的本地K为2048。

本次CPU接收核对确认：submission哈希与preflight一致，9个冻结脚本文件、170个preflight证据文件均匹配已有SHA256；43个窗口均具备8份 `completed/passed` 且零skip/零fallback的rank记录。**这仅是已回传文件的完整性检查，没有在本机重新验证GPU执行或动态库。** OOM窗口缺少完整rank输出，不能因存在目录或 `process.log` 就计为完成。

下一次从H100端补传的重点是：剩余29个窗口的结果（含失败窗口的有效重试）、失败备份与续跑日志、最终阶段状态、在原路径核验后生成的 `report.json`/`report.md`，以及下面规定的交接与环境说明。如果仍中断，提交新的部分进度即可，不填造最终报告。现有文档的第一轮统计只能标为阶段性结果；不能把第二轮缺失组用第一轮替代。

## 两台服务器的工作约定

### 执行端与分析端

- **H100执行端**：按用户安排的GPU任务和时间预算运行；记录实际源码版本与环境，完成校准、候选编译、预检及正式计时；在原路径执行严格核验，整理并回传证据。回传动作本身不要求重新跑已完成实验。
- **本开发服务器**：拉取同一会话的证据，检查文件、哈希、窗口/rank覆盖和统计口径，维护结论及论文。不要在这里执行拉回来的 `env.sh`、watchdog或续跑脚本来“验证能否运行”；其中包含H100端路径和GPU任务。
- **源码与数据分别记录提交**：H100端开始新实验前记录代码commit、submodule SHA和未提交补丁；测量过程中保持冻结输入不变，避免更新代码/环境。接收数据后可更新文档，但不能回写修改已冻结的脚本、mask、manifest和rank记录。
- **会话身份**：新实验建议 `gpt67-tp8-<节点>-YYYYMMDD-HHMMSS`，交接记录写明时区。TP、shape、模型、基础顺序、量化预算或运行环境改变时使用新会话；旧模型/旧节点的计划不能复用为新实验输入。

### 每次必须回传的文件

以下路径中的 `<calibration>`、`<compare>`、`<candidates>` 指本会话的**实际目录名**，允许带时间后缀。现有会话按上一节路径提交，新会话按前面的放置规范生成。

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

**沿用当前Git回传方式。** 原始数据提交在既有 `logs/h100/phase6/<会话>/`，配置/计划/源码快照提交在 `outputs/h100/phase6/<会话>/`；结果说明更新 [base-phase6.md](base-phase6.md)。本机pull后落到 `/data/run01/scyb672/hjr/TempName/` 下相同相对目录，无需另建一份“结果副本”放入 `docs/`。远端容器使用 `/workspace/TempName/` 也不影响这一约定。

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

若后续完整trace或专项profile超出仓库传输限制，可以单独打包传输，但必须在handoff写明归档文件名、下载/存储位置、SHA256和包内相对路径；在本机恢复到同一会话目录后再核对。不要无说明地删掉trace只保留均值。既有 `279fc32` 的文件保持原样，不因本约定改写Git历史。

### 原服务器严格核验与本机接收核验

`report.py`虽然是CPU逻辑，却会读取manifest和动态库的**原绝对路径**并核对哈希。Git没有携带全部二进制，`compare/build/`中的符号链接在本机也可能失效；因此不能把“文件已拉取”称为“原程序已在本机完整复算”。不要改JSON中的路径、哈希或完成标志来消除报错。

对当前6.7B会话，**只有正式窗口补齐后**，H100端才在原环境执行下列CPU核验并上传输出；此命令不启动GPU测量，部分数据时应保留失败状态而非制造完整报告：

```bash
cd /workspace/TempName
python tools/h100/phase6/report.py \
  logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517
```

本机接收时依次检查：会话/源码/模型身份，冻结输入与已有证据哈希，各窗口全部rank是否齐全且通过，两个轮次的完整配对块，实际选择mask和profile证据，报告与原始elapsed是否一致。独立CPU统计脚本及结果也需保存并注明核验范围；解析数据时可用仓库相对路径定位副本，但不修改来源文件。若另行传输了二进制且恢复原路径，可再做更完整的严格核验，其结果单独记录。

完整结果取每窗口所有rank的最大elapsed，除以正式step数；配对收益按 `100*(1-exp(mean(log(T候选/T基线))))`，使用配对块bootstrap置信区间。不能用主表中位数直接相除替代配对统计。默认4%门槛针对两轮、两个必须baseline分别检查，参考组也须回传；性能未达标与测试执行失败是两回事。短程随机token上的有限loss/梯度及前向误差检查不代表训练收敛。

### 资源、失败和续跑注意事项

1. 本次6.7B已遇到 `native_taco` 解码时的OOM，下一次执行前应确认分配到的8张卡可供本实验使用，并记录显存及其他进程。不要擅自终止他人的任务，也不要把清理其他任务当作默认前提。watchdog的历史`waiting`状态不是当前空闲证明。
2. 保持同节点、同GPU UUID与顺序、同模型/种子/软件/编译库；在校准、预检、计时阶段保持约定的运行条件。调整batch、分配器设置、通信环境或codec来绕过OOM属于实验条件变化，需另建会话，不能拼接成原两轮结果。
3. 维护入口 `tools/h100/phase6/run.py` 会拒绝覆盖已开始的阶段。当前6.7B上传了专用 `outputs/h100/phase6/gpt67-tp8-20261010-112249/resume_timing.py` 及等待脚本；它是该会话的执行证据，含固定 `/workspace/TempName` 路径，不是通用恢复入口。不要删除状态文件后盲目重跑，也不要在本开发服务器启动它。
4. 若原条件下续跑，记录中断与重启时刻、资源状况；保留原失败目录的 `*.failed-<时间>`备份、续跑脚本/日志及重新核验报告。完成文件须先核验再跳过，不能只看文件存在。跨长时间或竞争资源状态变化后的合并结果要明确标为续跑数据；必要时另做完整独占对照，不能把补齐窗口等同于证明计时环境稳定。
5. 完成、用户停止或超出约定时间预算时，执行方停止本任务的控制器和等待/自动重试脚本，确认本任务子进程退出，记录实际状态后回传已有结果。不要因希望得到正收益就无预算地追加实验；完整构建和72窗口测量也不能承诺20分钟内结束。

## 实现边界

量化 MLP 使用重新编译为 SM90 的 Phase6 GemmV2 算法；attention 和原生 Flux 参考使用 Hopper GemmV3。该版本没有实现基于 WGMMA/TMA 的 Hopper 原生融合 TACO epilogue，不能将兼容移植称为完成 H100 性能优化。

必须 baseline 为原生 Megatron＋朴素 TACO、原始顺序 Flux＋全远端融合 TACO；参考为原生 Megatron、原生 Hopper Flux。默认候选为远端优先和交替两种基础顺序，可配置另外两种交替顺序。范围仅 MLP `linear_fc2` 前向，BF16 GEMM、本地 BF16、H128/E4M3 远端协议及 BF16 STE 反向沿用现有设计。

构建与结果分别放 `outputs/h100/phase6/`、`logs/h100/phase6/`。必须从 H100 重新采集到达观察并生成选择计划；工具拒绝 A800 计划和未完成构建。候选准备依赖真实校准，不用合成 CPU 测试输入替代 GPU 证据。

后续GPU工作由H100执行端按用户任务和预算安排，按README执行校准、独立候选编译、swizzle/模型smoke/profile及完整六组/八组两轮测量；本页是交接规范，不会自动启动或恢复这些步骤。

## 本次无 GPU 验证（2026-10-09）

以下是2026-10-09、旧0.64B配置的历史记录，未在6.7B上重跑。编译和导入进程均设置 `CUDA_VISIBLE_DEVICES=''`：

| 检查 | 结果与范围 |
| --- | --- |
| Python CPU 协议测试 | 15 项通过；覆盖 TP4/TP8、四种基础顺序、到达计划/量化预算、留出数据隔离、输入冻结和 dry-run 不启动 worker |
| 注册代码生成器 | 132 SM 和 114 SM 两种配置的主机 C++ 编译通过；完整动态库编译采用默认 132 SM 配置 |
| CUDA 交叉编译 | SM90(a) 的 TACO runtime、融合 V2、采样 V2、合成到达表候选和实际 occupancy 查询函数通过；`cuobjdump` 确认量化 V2 对象包含 `sm_90a.cubin` |
| 完整库与 Python binding | `original`、`taco_fused`、`sampler_remote_first`、`sampler_interleaved` 四套独立库完成编译、链接和 Python 扩展构建 |
| CPU 导入 | 四套库使用 CUDA SDK 驱动桩通过 Torch/Flux 导入、绑定与导出符号检查，确认没有初始化 CUDA 上下文；没有调用 occupancy 查询或任何 kernel |
| GPU mapping 检查器 | 融合 baseline 和两套采样库的检查器完成编译，未执行 |
| 最终源码生成 | 重新执行 `build.py prepare` 成功；与已编译快照核对，只有等价的链接导出符号排列顺序不同 |

本机编译环境为 Python 3.10.20、PyTorch 2.6.0+cu124、CUDA Toolkit 12.8、G++ 11.4 和 CMake 3.30；编译检查使用现有 NCCL 开发安装。编译产物与记录位于 `outputs/h100/phase6/compile-check-20261009-r2/`，汇总为其中的 `validation.json`。该目录设为 `diagnostic_only=true`，工具禁止将其直接用于正式测量；这些被 Git 忽略的本地产物不会随 clone 迁移。

修复了无 GPU 编译发现的两处问题：H100 快照避免 `setup_requires` 再次联网下载 CMake；删除没有调用方的全局 CUDA 分配，使导入发生在绑定 rank 设备之前时也不会隐式分配设备内存。检查器改为查询实际编译 kernel 的 occupancy，不再沿用 A800 的固定数值。

上述编译记录当时不包含GPU验证；后来旧0.64B以及6.7B的阶段实测见[独立记录](base-phase6.md)。截至本次已接收快照，6.7B已完成真实校准、候选构建、smoke与profile，完整两轮性能验收尚未完成，也没有收敛验证。合成表只用于历史编译诊断，不作为真实校准计划或性能证据。
