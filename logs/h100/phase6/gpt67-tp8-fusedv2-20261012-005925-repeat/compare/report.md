# H100 单节点 Phase6 完整 step 对比

原生 Flux 参考使用 Hopper V3；融合量化 baseline 与组合候选使用编译为 SM90 的 V2 MLP 路径。
融合基线实现：`a800-fused-v2-cta-v1`；不同基线版本的收益分开报告。
正值为配对耗时下降；95% 区间按配对块 bootstrap。只报告本次同节点、同模型数据。

| 轮次 | 候选 | 对照 | 下降% | 95%区间 |
| --- | --- | --- | ---: | --- |
| 1 | remote_arrival_selective | native | 7.259 | [6.684, 7.787] |
| 1 | remote_arrival_selective | original | 12.138 | [0.636, 30.817] |
| 1 | remote_arrival_selective | native_taco | 34.928 | [34.246, 35.596] |
| 1 | remote_arrival_selective | taco_fused | 5.750 | [1.939, 9.856] |
| 1 | interleaved_arrival_selective | native | 6.888 | [6.585, 7.189] |
| 1 | interleaved_arrival_selective | original | 11.787 | [0.347, 30.381] |
| 1 | interleaved_arrival_selective | native_taco | 34.667 | [34.313, 35.111] |
| 1 | interleaved_arrival_selective | taco_fused | 5.373 | [1.555, 9.133] |
| 2 | remote_arrival_selective | native | -11.478 | [-37.881, 8.276] |
| 2 | remote_arrival_selective | original | -27.101 | [-76.730, 2.182] |
| 2 | remote_arrival_selective | native_taco | 14.403 | [-20.268, 35.006] |
| 2 | remote_arrival_selective | taco_fused | -14.421 | [-73.357, 21.409] |
| 2 | interleaved_arrival_selective | native | 14.556 | [5.425, 28.564] |
| 2 | interleaved_arrival_selective | original | 2.582 | [-0.260, 5.268] |
| 2 | interleaved_arrival_selective | native_taco | 34.394 | [33.365, 35.281] |
| 2 | interleaved_arrival_selective | taco_fused | 12.301 | [0.439, 23.280] |

两轮均通过两个必须 baseline 的 4.0% 门槛：`{'remote_arrival_selective': False, 'interleaved_arrival_selective': True}`。
完整数值、每轮绝对耗时和区间见 report.json。
