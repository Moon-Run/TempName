# H100 单节点 Phase6 完整 step 对比

原生 Flux 参考使用 Hopper V3；融合量化 baseline 与组合候选使用编译为 SM90 的 V2 MLP 路径。
融合基线实现：`a800-fused-v2-cta-v1`；不同基线版本的收益分开报告。
正值为配对耗时下降；95% 区间按配对块 bootstrap。只报告本次同节点、同模型数据。

| 轮次 | 候选 | 对照 | 下降% | 95%区间 |
| --- | --- | --- | ---: | --- |
| 1 | remote_arrival_selective | native | -3.253 | [-27.171, 7.260] |
| 1 | remote_arrival_selective | original | -9.683 | [-34.541, 1.148] |
| 1 | remote_arrival_selective | native_taco | 27.432 | [10.835, 34.684] |
| 1 | remote_arrival_selective | taco_fused | 5.315 | [1.383, 12.051] |
| 1 | interleaved_arrival_selective | native | 6.900 | [6.594, 7.172] |
| 1 | interleaved_arrival_selective | original | 1.103 | [0.640, 1.682] |
| 1 | interleaved_arrival_selective | native_taco | 34.568 | [34.334, 34.836] |
| 1 | interleaved_arrival_selective | taco_fused | 14.625 | [1.536, 35.575] |
| 2 | remote_arrival_selective | native | 6.989 | [6.080, 8.292] |
| 2 | remote_arrival_selective | original | 1.246 | [0.783, 1.687] |
| 2 | remote_arrival_selective | native_taco | 34.736 | [34.496, 35.003] |
| 2 | remote_arrival_selective | taco_fused | 2.387 | [2.072, 2.652] |
| 2 | interleaved_arrival_selective | native | 6.841 | [5.827, 8.442] |
| 2 | interleaved_arrival_selective | original | 1.090 | [0.459, 1.820] |
| 2 | interleaved_arrival_selective | native_taco | 34.632 | [34.342, 34.973] |
| 2 | interleaved_arrival_selective | taco_fused | 2.232 | [1.843, 2.601] |

两轮均通过两个必须 baseline 的 4.0% 门槛：`{'remote_arrival_selective': False, 'interleaved_arrival_selective': False}`。
完整数值、每轮绝对耗时和区间见 report.json。
