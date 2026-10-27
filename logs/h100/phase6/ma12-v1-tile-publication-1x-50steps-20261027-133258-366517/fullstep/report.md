# MA12：相对 MA89 的逐 tile 发布与接收端 acquire

| trial | F4-cache-zcount / F4-cache-zcount-tilepub 平均 ms | 收益 %［95% CI］ | 点估计 ≥2% | 区间下界 ≥2% |
| --- | --- | --- | --- | --- |
| compare1 | 459.554 / 460.477 | -0.205 [-0.889, +0.584] | False | False |

保留全部 16 个窗口、800 个正式 optimizer step。
每个 trial 分别验收；单轮仍为探索性结果。profile 必须另行通过，完整实验状态以 batch-state.json 为准。
两组均使用原 MA8-v2 激活缓存与 MA9 零统计；候选仅改变 MA12 原生发布与等待协议。
2% 为相对同期 F4-cache-zcount 的增量目标，不是相对原 CR3 的总收益。
两组 smoke 核对零统计与投影逐位结果；完整梯度和参数指纹另行核对。审计不进入正式计时。
完整梯度/更新指纹保持一致；显存峰值与全部慢窗口一并报告。
