# H100 单节点 Phase6 接续说明

更新：2026-10-09。本目录记录 H100 代码移植，与 [A800 接续说明](../design-a800/instruction.md) 分开维护。

当前目标是单节点 TP4/TP8 下的 baseline 与“基础顺序＋到达优先＋选择性量化”完整 optimizer-step 对比。默认 TP8/DP1，不支持跨节点。首轮仅完成 CPU/编译验证；之后获准的四卡限时测试因提交名额限制未能启动，用户要求保留现有作业并停止。本地尚无 H100 正确性、到达校准、性能或收敛结果。

## 入口

| 内容 | 路径 |
| --- | --- |
| 新服务器环境、源码迁移与八卡运行步骤 | [单机 TP8 部署指南](deploy-tp8.md) |
| 完整命令、baseline 定义与实现边界 | [H100 Phase6 README](../../../tools/h100/phase6/README.md) |
| 模型、TP、SM 数、基础顺序和预算 | [config.json](../../../tools/h100/phase6/config.json) |
| TP4、S1024、micro/global batch8 场景 | [config-tp4-s1024.json](../../../tools/h100/phase6/config-tp4-s1024.json) |
| 独立源码快照与无 GPU 编译 | [build.py](../../../tools/h100/phase6/build.py) |
| 新 H100 校准准备/执行 | [calibrate.py](../../../tools/h100/phase6/calibrate.py) |
| 到达表与量化 mask 拟合 | [plan.py](../../../tools/h100/phase6/plan.py) |
| 冻结模型实验与基线 | [prepare.py](../../../tools/h100/phase6/prepare.py) |
| 单节点 torchrun 控制器 | [run.py](../../../tools/h100/phase6/run.py) |
| 完整性核验与配对统计 | [report.py](../../../tools/h100/phase6/report.py) |
| CPU 协议检查 | [test_protocol.py](../../../tools/h100/phase6/test_protocol.py) |

## 实现边界

量化 MLP 使用重新编译为 SM90 的 Phase6 GemmV2 算法；attention 和原生 Flux 参考使用 Hopper GemmV3。该版本没有实现基于 WGMMA/TMA 的 Hopper 原生融合 TACO epilogue，不能将兼容移植称为完成 H100 性能优化。

必须 baseline 为原生 Megatron＋朴素 TACO、原始顺序 Flux＋全远端融合 TACO；参考为原生 Megatron、原生 Hopper Flux。默认候选为远端优先和交替两种基础顺序，可配置另外两种交替顺序。范围仅 MLP `linear_fc2` 前向，BF16 GEMM、本地 BF16、H128/E4M3 远端协议及 BF16 STE 反向沿用现有设计。

构建与结果分别放 `outputs/h100/phase6/`、`logs/h100/phase6/`。必须从 H100 重新采集到达观察并生成选择计划；工具拒绝 A800 计划和未完成构建。候选准备依赖真实校准，不用合成 CPU 测试输入替代 GPU 证据。

后续 GPU 工作需再获用户指令，届时按 README 执行校准、独立候选编译、swizzle/模型 smoke/profile 和完整六组/八组两轮测量。本次未启动这些 GPU 步骤。

## 本次无 GPU 验证（2026-10-09）

在当前编译节点完成以下检查，编译和导入进程均设置 `CUDA_VISIBLE_DEVICES=''`：

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

真实 H100 校准、带真实计划的候选构建、GPU swizzle/模型正确性、profile、性能和收敛均未验证。合成表只用于验证候选 CUDA 代码可以编译，不作为校准计划或性能证据。
