# MA7：F4 / F4-wgrad 完整训练结果

| trial | F4 / F4-wgrad 平均 ms | 比较 | 收益 %［95% CI］ |
| --- | --- | --- | --- |
| compare1 | 516.226 / 564.637 | F4-wgrad/F4 | -9.378 [-9.722, -9.029] |

反向实验：`independent_weight_stream`；两臂 native 前向相同。
全部保留 16 个窗口、800 个正式 optimizer step。
按每个独立 trial 的完整块配对判断同期 F4 正增量；单轮为探索性结果。
profile 核对wgrad 提交、microbatch drain 及 CUDA stream；观察到重叠本身不代表整步收益。慢窗口不删除。
