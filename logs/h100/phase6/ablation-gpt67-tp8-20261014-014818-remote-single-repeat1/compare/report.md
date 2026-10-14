# H100 GPT 6.7B TP8 完整 optimizer step 消融

保留全样本；每窗口 10+20 步，取全部 rank 的最大 elapsed。95% 区间按完整配对块 bootstrap。
G5 各 seed 单独报告，seed-mean 在每块内取几何平均后配对；没有单项 4% 门槛。

| 基础顺序 | 轮次 | 比较 | 耗时下降 % | 95% 区间 | 结论 |
| --- | --- | --- | ---: | --- | --- |
| remote_first | 1 | G0/native | 6.219 | [5.718, 6.698] | 确认正增益 |
| remote_first | 1 | G1/native | 6.310 | [5.584, 6.919] | 确认正增益 |
| remote_first | 1 | G2/native | 6.152 | [5.873, 6.463] | 确认正增益 |
| remote_first | 1 | G3/native | 6.064 | [5.592, 6.556] | 确认正增益 |
| remote_first | 1 | G4/native | 6.302 | [5.663, 6.922] | 确认正增益 |
| remote_first | 1 | G5-s17/native | 6.480 | [5.894, 7.008] | 确认正增益 |
| remote_first | 1 | G5-s29/native | 6.505 | [5.905, 7.029] | 确认正增益 |
| remote_first | 1 | G5-s43/native | 6.596 | [6.126, 7.093] | 确认正增益 |
| remote_first | 1 | G6/native | 6.401 | [5.928, 6.798] | 确认正增益 |
| remote_first | 1 | G5-seed-mean/native | 6.527 | [6.045, 6.998] | 确认正增益 |
| remote_first | 1 | G1/G0 | 0.097 | [-0.725, 0.845] | 尚未确认正增益 |
| remote_first | 1 | G2/G1 | -0.169 | [-0.822, 0.576] | 尚未确认正增益 |
| remote_first | 1 | G4/G3 | 0.253 | [-0.375, 0.826] | 尚未确认正增益 |
| remote_first | 1 | G3/G1 | -0.263 | [-0.778, 0.280] | 尚未确认正增益 |
| remote_first | 1 | G4/G2 | 0.159 | [-0.602, 0.913] | 尚未确认正增益 |
| remote_first | 1 | G4/G6 | -0.106 | [-0.589, 0.370] | 尚未确认正增益 |
| remote_first | 1 | G4/G5-s17 | -0.190 | [-0.562, 0.179] | 尚未确认正增益 |
| remote_first | 1 | G4/G5-s29 | -0.218 | [-0.741, 0.272] | 尚未确认正增益 |
| remote_first | 1 | G4/G5-s43 | -0.316 | [-0.757, 0.124] | 尚未确认正增益 |
| remote_first | 1 | G4/G5-seed-mean | -0.241 | [-0.587, 0.138] | 尚未确认正增益 |
| remote_first | 1 | arrival_quantization_interaction | 0.421 | [-0.586, 1.429] | 尚未确认正增益 |
| remote_first | combined | G0/native | 6.219 | [5.718, 6.698] | 确认正增益 |
| remote_first | combined | G1/native | 6.310 | [5.584, 6.919] | 确认正增益 |
| remote_first | combined | G2/native | 6.152 | [5.873, 6.463] | 确认正增益 |
| remote_first | combined | G3/native | 6.064 | [5.592, 6.556] | 确认正增益 |
| remote_first | combined | G4/native | 6.302 | [5.663, 6.922] | 确认正增益 |
| remote_first | combined | G5-s17/native | 6.480 | [5.894, 7.008] | 确认正增益 |
| remote_first | combined | G5-s29/native | 6.505 | [5.905, 7.029] | 确认正增益 |
| remote_first | combined | G5-s43/native | 6.596 | [6.126, 7.093] | 确认正增益 |
| remote_first | combined | G6/native | 6.401 | [5.928, 6.798] | 确认正增益 |
| remote_first | combined | G5-seed-mean/native | 6.527 | [6.045, 6.998] | 确认正增益 |
| remote_first | combined | G1/G0 | 0.097 | [-0.725, 0.845] | 尚未确认正增益 |
| remote_first | combined | G2/G1 | -0.169 | [-0.822, 0.576] | 尚未确认正增益 |
| remote_first | combined | G4/G3 | 0.253 | [-0.375, 0.826] | 尚未确认正增益 |
| remote_first | combined | G3/G1 | -0.263 | [-0.778, 0.280] | 尚未确认正增益 |
| remote_first | combined | G4/G2 | 0.159 | [-0.602, 0.913] | 尚未确认正增益 |
| remote_first | combined | G4/G6 | -0.106 | [-0.589, 0.370] | 尚未确认正增益 |
| remote_first | combined | G4/G5-s17 | -0.190 | [-0.562, 0.179] | 尚未确认正增益 |
| remote_first | combined | G4/G5-s29 | -0.218 | [-0.741, 0.272] | 尚未确认正增益 |
| remote_first | combined | G4/G5-s43 | -0.316 | [-0.757, 0.124] | 尚未确认正增益 |
| remote_first | combined | G4/G5-seed-mean | -0.241 | [-0.587, 0.138] | 尚未确认正增益 |
| remote_first | combined | arrival_quantization_interaction | 0.421 | [-0.586, 1.429] | 尚未确认正增益 |

| 基础顺序 | 轮次 | 组别/seed | 中位 ms/step | 中位 tokens/s |
| --- | --- | --- | ---: | ---: |
| remote_first | 1 | native | 573.242 | 14290.70 |
| remote_first | 1 | G0 | 539.222 | 15192.25 |
| remote_first | 1 | G1 | 537.100 | 15252.28 |
| remote_first | 1 | G2 | 537.398 | 15243.84 |
| remote_first | 1 | G3 | 538.881 | 15201.88 |
| remote_first | 1 | G4 | 538.417 | 15214.98 |
| remote_first | 1 | G5-s17 | 534.587 | 15323.99 |
| remote_first | 1 | G5-s29 | 535.064 | 15310.32 |
| remote_first | 1 | G5-s43 | 536.078 | 15281.54 |
| remote_first | 1 | G6 | 537.084 | 15252.75 |
| remote_first | 1 | G5-seed-mean | 535.470 | 15298.73 |
| remote_first | combined | native | 573.242 | 14290.70 |
| remote_first | combined | G0 | 539.222 | 15192.25 |
| remote_first | combined | G1 | 537.100 | 15252.28 |
| remote_first | combined | G2 | 537.398 | 15243.84 |
| remote_first | combined | G3 | 538.881 | 15201.88 |
| remote_first | combined | G4 | 538.417 | 15214.98 |
| remote_first | combined | G5-s17 | 534.587 | 15323.99 |
| remote_first | combined | G5-s29 | 535.064 | 15310.32 |
| remote_first | combined | G5-s43 | 536.078 | 15281.54 |
| remote_first | combined | G6 | 537.084 | 15252.75 |
| remote_first | combined | G5-seed-mean | 535.470 | 15298.73 |
