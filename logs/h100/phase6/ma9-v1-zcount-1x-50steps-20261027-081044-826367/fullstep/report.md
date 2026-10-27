# MA9：CR3/F4 与优化器零梯度统计融合的完整训练结果

| trial | F4 / F4-zcount 平均 ms | 收益 %［95% CI］ | 点估计 ≥2% | 区间下界 ≥2% |
| --- | --- | --- | --- | --- |
| compare1 | 503.310 / 489.878 | +2.669 [+2.306, +3.062] | True | True |

保留全部 16 个窗口、800 个正式 optimizer step。
每个 trial 分别验收；单轮仍为探索性结果。profile 必须另行通过，完整实验状态以 batch-state.json 为准。
F4 保留原 CR3 模型、autograd 与训练步；候选只融合 optimizer 零梯度统计，保留统计值与原 AllReduce。
完整梯度/更新指纹保持一致；显存峰值与全部慢窗口一并报告。
