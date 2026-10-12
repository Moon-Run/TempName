# H100 单节点 Phase6 完整 step 对比

原生 Flux 参考使用 Hopper V3；融合量化 baseline 与组合候选使用编译为 SM90 的 V2 MLP 路径。
融合基线实现：`a800-fused-v2-cta-v1`；不同基线版本的收益分开报告。
正值为配对耗时下降；95% 区间按配对块 bootstrap。只报告本次同节点、同模型数据。

| 轮次 | 候选 | 对照 | 下降% | 95%区间 |
| --- | --- | --- | ---: | --- |
| 1 | remote_arrival_selective | native | 14.876 | [-0.328, 33.958] |
| 1 | remote_arrival_selective | original | -1.898 | [-8.727, 1.832] |
| 1 | remote_arrival_selective | native_taco | 32.886 | [28.800, 35.004] |
| 1 | remote_arrival_selective | taco_fused | 10.293 | [-7.234, 31.316] |
| 1 | interleaved_arrival_selective | native | 4.074 | [-0.621, 6.880] |
| 1 | interleaved_arrival_selective | original | -14.829 | [-51.546, 1.301] |
| 1 | interleaved_arrival_selective | native_taco | 24.370 | [0.076, 34.745] |
| 1 | interleaved_arrival_selective | taco_fused | -1.091 | [-49.759, 30.847] |
| 2 | remote_arrival_selective | native | 5.946 | [4.976, 6.845] |
| 2 | remote_arrival_selective | original | 0.315 | [-1.141, 1.680] |
| 2 | remote_arrival_selective | native_taco | 34.195 | [33.631, 34.697] |
| 2 | remote_arrival_selective | taco_fused | 6.148 | [0.525, 15.177] |
| 2 | interleaved_arrival_selective | native | 6.530 | [6.130, 7.005] |
| 2 | interleaved_arrival_selective | original | 0.934 | [0.578, 1.306] |
| 2 | interleaved_arrival_selective | native_taco | 34.604 | [34.225, 35.098] |
| 2 | interleaved_arrival_selective | taco_fused | 6.731 | [1.249, 16.203] |

两轮均通过两个必须 baseline 的 4.0% 门槛：`{'remote_arrival_selective': False, 'interleaved_arrival_selective': False}`。
完整数值、每轮绝对耗时和区间见 report.json。
