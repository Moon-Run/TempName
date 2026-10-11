# H100 单节点 Phase6 完整 step 对比

原生 Flux 参考使用 Hopper V3；融合量化 baseline 与组合候选使用编译为 SM90 的 V2 MLP 路径。
正值为配对耗时下降；95% 区间按配对块 bootstrap。只报告本次同节点、同模型数据。

| 轮次 | 候选 | 对照 | 下降% | 95%区间 |
| --- | --- | --- | ---: | --- |
| 1 | remote_arrival_selective | native | 7.839 | [4.597, 11.972] |
| 1 | remote_arrival_selective | original | 0.192 | [-1.598, 1.887] |
| 1 | remote_arrival_selective | native_taco | 33.998 | [32.704, 35.245] |
| 1 | remote_arrival_selective | taco_fused | -0.882 | [-2.611, 0.545] |
| 1 | interleaved_arrival_selective | native | 8.515 | [6.454, 11.999] |
| 1 | interleaved_arrival_selective | original | 0.924 | [0.634, 1.156] |
| 1 | interleaved_arrival_selective | native_taco | 34.482 | [33.529, 35.535] |
| 1 | interleaved_arrival_selective | taco_fused | -0.143 | [-0.916, 0.760] |
| 2 | remote_arrival_selective | native | 7.095 | [6.704, 7.592] |
| 2 | remote_arrival_selective | original | 0.910 | [0.198, 1.555] |
| 2 | remote_arrival_selective | native_taco | 34.579 | [34.230, 34.898] |
| 2 | remote_arrival_selective | taco_fused | 3.003 | [-0.876, 9.568] |
| 2 | interleaved_arrival_selective | native | 4.432 | [0.065, 6.921] |
| 2 | interleaved_arrival_selective | original | -1.930 | [-7.741, 1.218] |
| 2 | interleaved_arrival_selective | native_taco | 32.704 | [29.201, 34.499] |
| 2 | interleaved_arrival_selective | taco_fused | 0.223 | [-8.957, 9.567] |

两轮均通过两个必须 baseline 的 4.0% 门槛：`{'remote_arrival_selective': False, 'interleaved_arrival_selective': False}`。
完整数值、每轮绝对耗时和区间见 report.json。
