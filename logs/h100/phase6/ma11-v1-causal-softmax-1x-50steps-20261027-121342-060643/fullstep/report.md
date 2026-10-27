# MA11：相对 MA89 的注意力 mask/softmax 中间态消除

| trial | F4-cache-zcount / F4-cache-zcount-csoftmax 平均 ms | 收益 %［95% CI］ | 点估计 ≥2% | 区间下界 ≥2% |
| --- | --- | --- | --- | --- |
| compare1 | 459.638 / 450.662 | +1.952 [+1.711, +2.271] | False | False |

保留全部 16 个窗口、800 个正式 optimizer step。
每个 trial 分别验收；单轮仍为探索性结果。profile 必须另行通过，完整实验状态以 batch-state.json 为准。
两组均使用原 MA8-v2 激活缓存与 MA9 零统计；候选仅叠加 MA11 注意力机制。
2% 为相对同期 F4-cache-zcount 的增量目标，不是相对原 CR3 的总收益。
两组 smoke 核对零统计；候选另外逐次核对 softmax 前后向字节。审计不进入正式计时。
完整梯度/更新指纹保持一致；显存峰值与全部慢窗口一并报告。
