# H100 单节点 Phase6 完整 step 对比

原生 Flux 参考使用 Hopper V3；融合量化 baseline 与组合候选使用编译为 SM90 的 V2 MLP 路径。
融合基线实现：`a800-fused-v2-cta-v1`；不同基线版本的收益分开报告。
正值为配对耗时下降；95% 区间按配对块 bootstrap。只报告本次同节点、同模型数据。

| 轮次 | 候选 | 对照 | 下降% | 95%区间 |
| --- | --- | --- | ---: | --- |
| 1 | remote_arrival_selective | native | 11.289 | [5.938, 20.479] |
| 1 | remote_arrival_selective | original | -6.245 | [-21.250, 0.868] |
| 1 | remote_arrival_selective | native_taco | 36.313 | [-15.055, 65.728] |
| 1 | remote_arrival_selective | taco_fused | 10.445 | [-54.895, 46.653] |
| 1 | interleaved_arrival_selective | native | 8.167 | [6.177, 11.044] |
| 1 | interleaved_arrival_selective | original | -9.985 | [-35.903, 1.582] |
| 1 | interleaved_arrival_selective | native_taco | 34.071 | [-29.093, 65.947] |
| 1 | interleaved_arrival_selective | taco_fused | 7.293 | [-73.436, 47.071] |
| 2 | remote_arrival_selective | native | -4.032 | [-32.350, 9.010] |
| 2 | remote_arrival_selective | original | -5.522 | [-41.656, 17.037] |
| 2 | remote_arrival_selective | native_taco | 12.147 | [-25.110, 34.533] |
| 2 | remote_arrival_selective | taco_fused | -1.832 | [-17.450, 8.716] |
| 2 | interleaved_arrival_selective | native | 22.515 | [6.305, 46.627] |
| 2 | interleaved_arrival_selective | original | 21.405 | [0.516, 44.374] |
| 2 | interleaved_arrival_selective | native_taco | 34.566 | [34.265, 34.884] |
| 2 | interleaved_arrival_selective | taco_fused | 24.154 | [1.681, 49.362] |

两轮均通过两个必须 baseline 的 4.0% 门槛：`{'remote_arrival_selective': False, 'interleaved_arrival_selective': False}`。
完整数值、每轮绝对耗时和区间见 report.json。
