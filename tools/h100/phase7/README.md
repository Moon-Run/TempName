# Phase7：独立分块 GEMM / 通信 / 解码原型

设计、实现边界、运行命令和验证记录见 [base-phase7.md](../../../docs/design/design-h100/base-phase7.md)。本目录不改写 Phase6、A800 或冻结候选源码。

- `phase7.py`：准备冻结会话、默认 dry-run、显式执行、四组配对报告。
- `planner.py`：共享分块顺序、物理 mask、旧计划导入和按完整候选净收益选计划。
- `runtime.py` / `codec.py`：分块 BF16 GEMM、H128/E4M3 mixed-wire NCCL all-to-all、屏障/流水消费。
- `adapter.py` / `worker.py`：独立算子与固定 GPT6.7B 的 BF16/STE 训练接入。
- `protocol.py`：冻结身份、完整rank/窗口检查、完整配对块统计与交互效应。
- `snapshot.py`：Phase6 原版来源归档及校验。

所有 GPU 入口必须显式传 `run --execute`。CPU 准备或测试不会升级为 GPU 验证。现有运行时使用新分块张量和 CUDA/NCCL 事件依赖，不是对旧 Flux epilogue 的直接无屏障改写，也没有实现设备端动态 ready 队列。

在带 PyTorch 2.6 的 Python 环境运行全部 CPU 回归：

```bash
python -m unittest discover -s tools/h100/phase7 -p 'test_*.py' -v
```

系统 Python 没有 PyTorch 时，纯标准库计划/协议测试仍可执行，依赖真实张量的 runtime 测试会明确 skip。
