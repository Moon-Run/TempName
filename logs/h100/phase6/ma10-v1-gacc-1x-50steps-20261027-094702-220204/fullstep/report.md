# MA10：CR3/F4 与 DDP 梯度入库批量融合的完整训练结果

| trial | F4 / F4-gacc 平均 ms | 收益 %［95% CI］ | 点估计 ≥2% | 区间下界 ≥2% |
| --- | --- | --- | --- | --- |
| compare1 | 504.544 / 536.320 | -6.299 [-6.809, -5.768] | False | False |

保留全部 16 个窗口、800 个正式 optimizer step。
每个 trial 分别验收；单轮仍为探索性结果。profile 必须另行通过，完整实验状态以 batch-state.json 为准。
F4 保留原 CR3 模型与训练步；候选在每次 microbatch backward 内批量执行原 FP32 main_grad += BF16 grad，保持原通信和优化器。
完整梯度/更新指纹保持一致；显存峰值与全部慢窗口一并报告。
