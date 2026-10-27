# MA13：相对 MA11 的指数驻留与空归约参与者消除

| trial | F4-cache-zcount-csoftmax / F4-cache-zcount-csoftmax-resident 平均 ms | 收益 %［95% CI］ | 点估计 ≥2% | 区间下界 ≥2% |
| --- | --- | --- | --- | --- |
| compare1 | 451.935 / 450.462 | +0.326 [-0.114, +0.767] | False | False |

保留全部 16 个窗口、800 个正式 optimizer step。
每个 trial 分别验收；单轮仍为探索性结果。profile 必须另行通过，完整实验状态以 batch-state.json 为准。
两组均使用原 MA8-v2、MA9 与 MA11 反向；候选仅替换 softmax 前向。
2% 为相对同期 F4-cache-zcount-csoftmax 的增量目标，不是相对原 CR3 的总收益。
两组 smoke 核对零统计及 softmax 前后向字节；候选另外与原 MA11 前向核对。审计不进入正式计时。
完整梯度/更新指纹保持一致；显存峰值与全部慢窗口一并报告。
