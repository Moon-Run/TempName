# H100 单节点 Phase6 完整 step 对比

原生 Flux 参考使用 Hopper V3；融合量化 baseline 与组合候选使用编译为 SM90 的 V2 MLP 路径。
正值为配对耗时下降；95% 区间按配对块 bootstrap。只报告本次同节点、同模型数据。

| 轮次 | 候选 | 对照 | 下降% | 95%区间 |
| --- | --- | --- | ---: | --- |
| 1 | remote_arrival_selective | native | 6.781 | [6.377, 7.159] |
| 1 | remote_arrival_selective | original | 1.449 | [0.819, 2.040] |
| 1 | remote_arrival_selective | native_taco | 33.636 | [33.255, 34.004] |
| 1 | remote_arrival_selective | taco_fused | 0.532 | [-0.271, 1.583] |
| 1 | interleaved_arrival_selective | native | 6.737 | [6.207, 7.302] |
| 1 | interleaved_arrival_selective | original | 1.402 | [1.108, 1.746] |
| 1 | interleaved_arrival_selective | native_taco | 33.605 | [33.494, 33.720] |
| 1 | interleaved_arrival_selective | taco_fused | 0.485 | [-0.237, 1.495] |
| 2 | remote_arrival_selective | native | 6.119 | [5.602, 6.573] |
| 2 | remote_arrival_selective | original | 1.235 | [0.796, 1.667] |
| 2 | remote_arrival_selective | native_taco | 33.441 | [33.142, 33.914] |
| 2 | remote_arrival_selective | taco_fused | -0.368 | [-0.934, 0.084] |
| 2 | interleaved_arrival_selective | native | 6.279 | [5.659, 6.974] |
| 2 | interleaved_arrival_selective | original | 1.403 | [1.036, 1.714] |
| 2 | interleaved_arrival_selective | native_taco | 33.554 | [33.096, 34.097] |
| 2 | interleaved_arrival_selective | taco_fused | -0.197 | [-0.761, 0.299] |

两轮均通过两个必须 baseline 的 4.0% 门槛：`{'remote_arrival_selective': False, 'interleaved_arrival_selective': False}`。
完整数值、每轮绝对耗时和区间见 report.json。
