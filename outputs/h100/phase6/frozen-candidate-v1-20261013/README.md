# H100 Phase6 优化前候选冻结版

冻结标识：`h100-phase6-candidate-v1-20261013`。这是后续候选优化的源码起点与旧实现对照，**不是必须 baseline `taco_fused` 的新名称**。本次仅归档和 CPU 哈希核对，未重新编译或运行 GPU。

## 冻结内容

| 内容 | 固定值 / 位置 |
| --- | --- |
| 维护源码提交 | `944b9df06a564cabd738dc608c3d4945e98bb103` |
| 独立源码副本 | `source/`，652 个已跟踪文件，约 5.22 MiB；覆盖 `src/include/cmake/python`、H100 工具、共享 A800 工具及根构建文件 |
| 冻结登记 | `manifest.json`：源码逐文件 SHA256、Git blob/权限、依赖版本、配置、实测来源、缺项与本次核验范围 |
| 原实测输入副本 | `evidence/`：候选/base manifest、选择计划、最新 g/h 轮 submission；原绝对路径及哈希保持原样 |
| 本目录完整性清单 | `SHA256SUMS.txt`，从本目录校验；不包含清单自身 |
| 源码比对清单 | `SOURCE.sha256.txt`，从 `source/` 或恢复出的仓库根目录校验 |
| 既有实测证据清单 | `REFERENCES.sha256.txt`，从 TempName 仓库根目录校验；不要求维护入口继续保持旧代码 |
| CUTLASS | `df8a550d3917b0e97f416b2ed8c2d786f7f686a3` |
| NCCL | `8c6c5951854a57ba90c4424fa040497f6defac46` |
| Megatron-LM | `3ea68ad6042cc1204386ae9364358f7c4de1bc37` |

源码副本与该提交逐文件一致；冻结前已有的 `base-phase6.md` 未提交修改属于结果文档，不是源码补丁。子模块、Megatron、软件环境、动态库和构建缓存不复制进本目录；`source/` 保留仓库原已跟踪的 Triton `cuda_extra.bc/.ll` 输入，这些不是本次编译产物。

## 已测版本的身份

以下路径从 TempName 仓库根目录起算：

| 内容 | 来源 |
| --- | --- |
| 两套候选 | `outputs/h100/phase6/gpt67-tp8-20261010-112249/candidates-20261010-153517/`，其中 `remote_arrival_selective/` 与 `interleaved_arrival_selective/` |
| 候选构建源码提交 | `ff38f5b1ff1c80cb831f58c64b95f42b0379879a` |
| 候选 manifest SHA256 | `af23e1a00445ca214e8d864daa0a40604e6cf5f9b29d28421314a71030921303` |
| 成功校准 | `logs/h100/phase6/gpt67-tp8-20261010-112249/calibration-20261010-153517/` |
| 选择计划 | `outputs/h100/phase6/gpt67-tp8-20261010-112249/selection-plan-20261010-153517.json` |
| 计划 SHA256 | `2d8eee0ffaf2c86fdd1719950c7a5ed5eb0c93d5803b7deaf1e65caad752d167` |
| 融合 v2 / 原生 Flux 构建 | `outputs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/base/` |
| baseline 构建源码提交 | `d786da51e0aeafec1bb2b5d071c5c59e93e6de43` |
| baseline manifest SHA256 | `2dbac964c7e71f81063edcbba861d09a9fb33f814424edb35688b5708066e939` |
| 必须融合 baseline 标识 | `a800-fused-v2-cta-v1`，CTA 解码；不是历史 `phase6-warp-v1` |

两套候选全部 702 个源码文件与原 manifest 一致。旧 baseline 回传快照有 28 个缺失条目（四个变体各缺 `cuda_extra.bc/.ll`、`version.ld` 及四个 `.clang-format`）；逐项名称及原哈希见 manifest 的 `validation.unavailable_entries_in_source_manifests`。本次不补写旧证据，也不声称在本机核验了原运行动态库。H100 端应保留原构建，补传缺少的文本来源文件；编译二进制沿用执行端保留、必要时单独传输的约定。

配置固定为 GPT 6.7B、32 层、H4096/F16384、TP8、132 SM、S2048、mb1/global4、seed1234/token_seed4321；窗口 64、量化预算 1/64、10 步预热＋20 步正式计时。候选仍是基础顺序＋到达优先＋选择性量化：BF16 GEMM/本地贡献，远端 H128/E4M3，BF16 STE 反向；MLP 用 SM90 编译的 GemmV2，attention/参考用 Hopper GemmV3。

`base-phase6.md` 的融合 v2 对齐第 1–4 轮及新监控体系 a–h 均复用上面同一候选构建。全部六套会话、run_id、文档轮次映射见 `manifest.json`。a–h 四套报告的两候选均未通过完整两轮验收；冻结保存的是当前实现，不代表性能门槛已通过。源码提交、构建提交、数据上传提交须分别记录，不能把旧构建的实测结果标成对本次源码快照的新验证。

## 只读核对与恢复

在任意一台已取得本目录和回传证据的服务器，从 TempName 根目录执行；这些命令不使用 GPU：

```bash
H100_FREEZE="$PWD/outputs/h100/phase6/frozen-candidate-v1-20261013"
(cd "$H100_FREEZE" && sha256sum --quiet -c SHA256SUMS.txt)
sha256sum --quiet -c "$H100_FREEZE/REFERENCES.sha256.txt"
```

缺失或哈希不匹配时定位实际文件，不重写清单让检查通过。冻结目录与被引用的旧实验目录按只读证据管理。新优化继续改受维护入口，不改这里的副本；不要直接在 `source/` 内运行构建工具，其 Git 根仍可能指向外层新仓库。

需要恢复旧源码进行构建时，另建工作树，保持当前优化工作区和旧产物原样：

```bash
# TempName 根目录；目标目录须不存在。
H100_FREEZE="$PWD/outputs/h100/phase6/frozen-candidate-v1-20261013"
git worktree add --detach ../TempName-h100-candidate-v1 \
  944b9df06a564cabd738dc608c3d4945e98bb103
git -C ../TempName-h100-candidate-v1 submodule update --init --recursive
(cd ../TempName-h100-candidate-v1 && sha256sum --quiet -c "$H100_FREEZE/SOURCE.sha256.txt")
```

随后按部署文档在新会话准备/编译；新路径或新设备下重新校准，不改原 manifest 内的 `/workspace/TempName/...`。重建库需要重新 smoke/profile 和测量，不能假定二进制哈希、性能等同于旧库。同节点原路径下复用原库时先按原 manifest 核对实际库及完整运行条件。优化结果必须记录这个冻结 ID、新 commit/补丁、新计划及构建哈希，并保留四个既有参考/必须 baseline；现有比较工具每次接收一套 candidate build，尚无“同时插入旧、新两套候选”的快捷开关。
