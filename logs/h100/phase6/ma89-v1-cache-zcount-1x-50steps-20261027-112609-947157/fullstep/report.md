# MA8＋MA9：相对 F4-cache 的优化器零梯度统计增量

| trial | F4-cache / F4-cache-zcount 平均 ms | 收益 %［95% CI］ | 点估计 ≥2% | 区间下界 ≥2% |
| --- | --- | --- | --- | --- |
| compare1 | 473.635 / 471.812 | +0.591 [-4.076, +3.042] | False | False |

保留全部 16 个窗口、800 个正式 optimizer step。
每个 trial 分别验收；单轮仍为探索性结果。profile 必须另行通过，完整实验状态以 batch-state.json 为准。
两组均使用原 MA8-v2 激活缓存；F4-cache 保留原 optimizer，F4-cache-zcount 只叠加已验证 MA9 零统计融合。
2% 为相对同期 F4-cache 的增量目标，不是相对原 CR3 的总收益。
零统计精确性来自候选 smoke 对原统计的逐步核对；正式窗口未新增零统计轨迹记录。
完整梯度/更新指纹保持一致；显存峰值与全部慢窗口一并报告。
