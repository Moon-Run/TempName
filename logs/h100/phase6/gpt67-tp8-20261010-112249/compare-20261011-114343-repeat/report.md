# H100 单节点 Phase6 完整 step 对比

原生 Flux 参考使用 Hopper V3；融合量化 baseline 与组合候选使用编译为 SM90 的 V2 MLP 路径。
正值为配对耗时下降；95% 区间按配对块 bootstrap。只报告本次同节点、同模型数据。

| 轮次 | 候选 | 对照 | 下降% | 95%区间 |
| --- | --- | --- | ---: | --- |
| 1 | remote_arrival_selective | native | 7.570 | [0.941, 12.842] |
| 1 | remote_arrival_selective | original | -2.949 | [-8.795, 1.978] |
| 1 | remote_arrival_selective | native_taco | 32.395 | [28.849, 35.225] |
| 1 | remote_arrival_selective | taco_fused | -0.521 | [-5.065, 2.413] |
| 1 | interleaved_arrival_selective | native | 4.128 | [-5.041, 10.128] |
| 1 | interleaved_arrival_selective | original | -6.784 | [-15.613, 0.518] |
| 1 | interleaved_arrival_selective | native_taco | 29.876 | [24.340, 34.296] |
| 1 | interleaved_arrival_selective | taco_fused | -4.265 | [-10.074, -0.348] |
| 2 | remote_arrival_selective | native | 8.579 | [6.166, 12.914] |
| 2 | remote_arrival_selective | original | 4.259 | [0.275, 10.079] |
| 2 | remote_arrival_selective | native_taco | 34.822 | [34.393, 35.232] |
| 2 | remote_arrival_selective | taco_fused | 0.724 | [-0.904, 2.149] |
| 2 | interleaved_arrival_selective | native | 7.242 | [4.899, 9.892] |
| 2 | interleaved_arrival_selective | original | 2.860 | [0.718, 6.467] |
| 2 | interleaved_arrival_selective | native_taco | 33.869 | [32.343, 35.126] |
| 2 | interleaved_arrival_selective | taco_fused | -0.727 | [-2.223, 0.346] |

两轮均通过两个必须 baseline 的 4.0% 门槛：`{'remote_arrival_selective': False, 'interleaved_arrival_selective': False}`。
完整数值、每轮绝对耗时和区间见 report.json。
