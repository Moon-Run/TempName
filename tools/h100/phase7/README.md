# Phase7：原融合路径上的到达优先改进

主入口是 `fused.py`，完整设计、实现边界及运行命令见 [base-phase7.md](../../../docs/design/design-h100/base-phase7.md)，后续路线见 [update.md](../../../docs/design/design-h100/update.md)。

- `fused_plan.py`：真实receiver采样、稳定剩余延迟、协调排序、保守回退。
- `fused_build.py`：生成base/legacy/improved三套同源候选，仅CUDA/host映射表不同。
- `fused_protocol.py`、`fused_prepare.py`、`fused_run.py`、`fused_report.py`：原融合模型接入、九组配对及严格证据检查。
- `snapshot.py`：校验本地保留的Phase6原版快照。

三种顺序使用完全相同的原选择性量化mask；GemmV2、codec、IPC、屏障和双缓冲保持一致。默认dry-run，GPU执行须显式指定 `--execute`。静态模型筛选不等于性能提升。

```bash
python3 tools/h100/phase7/fused.py --help
python3 -m unittest discover -s tools/h100/phase7 -p 'test_fused_*.py' -v
```

此前独立分块原型及其CPU验证产物已删除。新生成的 `fused-*-dev/` 目录仅保留本地，不随Git提交。
