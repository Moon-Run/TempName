# MA6：F4 / F4-tile / F4-gather 完整训练结果

| trial | F4 / F4-tile / F4-gather 平均 ms | 比较 | 收益 %［95% CI］ |
| --- | --- | --- | --- |
| compare1 | 508.456 / 510.175 / 511.918 | F4-tile/F4 | -0.339 [-0.609, -0.093] |
| compare1 | 508.456 / 510.175 / 511.918 | F4-gather/F4 | -0.680 [-0.971, -0.382] |
| compare1 | 508.456 / 510.175 / 511.918 | F4-tile/F4-gather | +0.339 [+0.039, +0.660] |

F4-gather 调度：`overlap`。
全部保留 36 个窗口、1800 个正式 optimizer step。
每个候选分别判断相对同期 F4 的正增量；单轮为探索性结果。未测试 tile+gather 合并候选或 C/R。
F4-gather 的算子前向计时不测反向收益；需结合完整训练及独立 backward profile。慢窗口不删除。
