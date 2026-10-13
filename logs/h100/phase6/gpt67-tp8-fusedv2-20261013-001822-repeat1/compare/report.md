# H100 单节点 Phase6 完整 step 对比

原生 Flux 参考使用 Hopper V3；融合量化 baseline 与组合候选使用编译为 SM90 的 V2 MLP 路径。
融合基线实现：`a800-fused-v2-cta-v1`；不同基线版本的收益分开报告。
正值为配对耗时下降；95% 区间按配对块 bootstrap。只报告本次同节点、同模型数据。

| 轮次 | 候选 | 对照 | 下降% | 95%区间 |
| --- | --- | --- | ---: | --- |
| 1 | remote_arrival_selective | native | 6.889 | [6.385, 7.312] |
| 1 | remote_arrival_selective | original | 13.023 | [0.502, 33.134] |
| 1 | remote_arrival_selective | native_taco | 34.853 | [34.592, 35.099] |
| 1 | remote_arrival_selective | taco_fused | 8.874 | [1.641, 21.398] |
| 1 | interleaved_arrival_selective | native | 6.934 | [6.657, 7.216] |
| 1 | interleaved_arrival_selective | original | 13.065 | [0.235, 33.133] |
| 1 | interleaved_arrival_selective | native_taco | 34.884 | [34.539, 35.274] |
| 1 | interleaved_arrival_selective | taco_fused | 8.918 | [1.757, 21.224] |
| 2 | remote_arrival_selective | native | 6.727 | [6.094, 7.468] |
| 2 | remote_arrival_selective | original | 0.995 | [0.051, 2.441] |
| 2 | remote_arrival_selective | native_taco | 34.955 | [34.458, 35.615] |
| 2 | remote_arrival_selective | taco_fused | 1.676 | [1.222, 2.205] |
| 2 | interleaved_arrival_selective | native | 6.614 | [6.292, 6.933] |
| 2 | interleaved_arrival_selective | original | 0.875 | [0.080, 1.848] |
| 2 | interleaved_arrival_selective | native_taco | 34.876 | [34.441, 35.284] |
| 2 | interleaved_arrival_selective | taco_fused | 1.556 | [1.007, 1.989] |

两轮均通过两个必须 baseline 的 4.0% 门槛：`{'remote_arrival_selective': False, 'interleaved_arrival_selective': False}`。
完整数值、每轮绝对耗时和区间见 report.json。
