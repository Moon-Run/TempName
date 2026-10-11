# H100 单节点 Phase6 完整 step 对比

原生 Flux 参考使用 Hopper V3；融合量化 baseline 与组合候选使用编译为 SM90 的 V2 MLP 路径。
融合基线实现：`a800-fused-v2-cta-v1`；不同基线版本的收益分开报告。
正值为配对耗时下降；95% 区间按配对块 bootstrap。只报告本次同节点、同模型数据。

| 轮次 | 候选 | 对照 | 下降% | 95%区间 |
| --- | --- | --- | ---: | --- |
| 1 | remote_arrival_selective | native | 7.066 | [6.164, 8.046] |
| 1 | remote_arrival_selective | original | 1.480 | [0.323, 2.872] |
| 1 | remote_arrival_selective | native_taco | 34.543 | [34.292, 34.787] |
| 1 | remote_arrival_selective | taco_fused | 2.938 | [0.832, 5.925] |
| 1 | interleaved_arrival_selective | native | -4.106 | [-28.860, 6.963] |
| 1 | interleaved_arrival_selective | original | -10.364 | [-36.041, 1.104] |
| 1 | interleaved_arrival_selective | native_taco | 26.674 | [8.572, 34.546] |
| 1 | interleaved_arrival_selective | taco_fused | -8.731 | [-36.910, 5.191] |
| 2 | remote_arrival_selective | native | 4.439 | [-0.497, 7.324] |
| 2 | remote_arrival_selective | original | -1.439 | [-6.502, 1.339] |
| 2 | remote_arrival_selective | native_taco | 33.196 | [29.649, 35.139] |
| 2 | remote_arrival_selective | taco_fused | 0.675 | [-1.689, 2.207] |
| 2 | interleaved_arrival_selective | native | 6.154 | [5.535, 6.657] |
| 2 | interleaved_arrival_selective | original | 0.381 | [-0.278, 1.015] |
| 2 | interleaved_arrival_selective | native_taco | 34.394 | [33.661, 35.058] |
| 2 | interleaved_arrival_selective | taco_fused | 2.458 | [0.370, 5.421] |

两轮均通过两个必须 baseline 的 4.0% 门槛：`{'remote_arrival_selective': False, 'interleaved_arrival_selective': False}`。
完整数值、每轮绝对耗时和区间见 report.json。
