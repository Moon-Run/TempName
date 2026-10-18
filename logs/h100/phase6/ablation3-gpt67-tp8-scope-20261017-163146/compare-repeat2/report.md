# H100 GPT6.7B TP8 九组消融

本次1轮，每轮9个完整配对块，共81个正式窗口。
所有ms/step均为完整32层模型、4微批的optimizer step。九组同块，全样本为主。
A=交替+全远端量化；B=交替+到达优先+全远端量化；C=交替+到达优先+选择性量化。
本地贡献及未选中模块保持BF16；没有同期native_taco或固定融合v2基线。
95% CI按完整九组配对块bootstrap 10000次、seed20261009，逐项区间未校正多重比较；无4%门槛。

单轮保留原第一轮的循环移位调度，没有第二轮反转验证；combined与第1轮是同一批数据，不增加样本数。

## 全样本

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | interleaved_all_mlp | 535.170 | 15307.28 |
| 1 | interleaved_arrival_all_mlp | 535.188 | 15306.78 |
| 1 | interleaved_arrival_selective_mlp | 535.679 | 15292.73 |
| 1 | interleaved_all_attn | 533.949 | 15342.30 |
| 1 | interleaved_arrival_all_attn | 534.413 | 15328.98 |
| 1 | interleaved_arrival_selective_attn | 534.675 | 15321.46 |
| 1 | interleaved_all_attn_mlp | 516.033 | 15874.96 |
| 1 | interleaved_arrival_all_attn_mlp | 517.115 | 15841.75 |
| 1 | interleaved_arrival_selective_attn_mlp | 518.974 | 15784.98 |
| combined | interleaved_all_mlp | 535.170 | 15307.28 |
| combined | interleaved_arrival_all_mlp | 535.188 | 15306.78 |
| combined | interleaved_arrival_selective_mlp | 535.679 | 15292.73 |
| combined | interleaved_all_attn | 533.949 | 15342.30 |
| combined | interleaved_arrival_all_attn | 534.413 | 15328.98 |
| combined | interleaved_arrival_selective_attn | 534.675 | 15321.46 |
| combined | interleaved_all_attn_mlp | 516.033 | 15874.96 |
| combined | interleaved_arrival_all_attn_mlp | 517.115 | 15841.75 |
| combined | interleaved_arrival_selective_attn_mlp | 518.974 | 15784.98 |

### 同范围策略比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_arrival_all_mlp/interleaved_all_mlp | 0.003 | [-0.349,0.440] | 0.999326 | 1.014327 | 1/3 | 538.131 / 530.530 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.234 | [-0.539,0.095] | 0.998549 | 1.008116 | 1/6 | 539.193 / 534.852 |
| 1 | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.231 | [-0.466,0.040] | 0.997010 | 1.006102 | 1/3 | 538.131 / 534.868 |
| 1 | interleaved_arrival_all_attn/interleaved_all_attn | -0.267 | [-0.752,0.139] | 0.997578 | 1.007671 | 1/2 | 536.554 / 532.469 |
| 1 | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | 0.091 | [-0.229,0.435] | 0.999138 | 1.009428 | 1/0 | 537.081 / 532.065 |
| 1 | interleaved_arrival_selective_attn/interleaved_all_attn | -0.176 | [-0.573,0.194] | 0.999043 | 1.005959 | 1/0 | 535.236 / 532.065 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | -0.226 | [-0.667,0.257] | 0.996898 | 1.012521 | 1/0 | 523.126 / 516.657 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | -0.271 | [-0.705,0.189] | 0.995535 | 1.010149 | 1/7 | 520.684 / 515.452 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | -0.497 | [-0.849,-0.127] | 0.994038 | 1.004479 | 1/7 | 517.761 / 515.452 |
| combined | interleaved_arrival_all_mlp/interleaved_all_mlp | 0.003 | [-0.349,0.440] | 0.999326 | 1.014327 | 1/3 | 538.131 / 530.530 |
| combined | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.234 | [-0.539,0.095] | 0.998549 | 1.008116 | 1/6 | 539.193 / 534.852 |
| combined | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.231 | [-0.466,0.040] | 0.997010 | 1.006102 | 1/3 | 538.131 / 534.868 |
| combined | interleaved_arrival_all_attn/interleaved_all_attn | -0.267 | [-0.752,0.139] | 0.997578 | 1.007671 | 1/2 | 536.554 / 532.469 |
| combined | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | 0.091 | [-0.229,0.435] | 0.999138 | 1.009428 | 1/0 | 537.081 / 532.065 |
| combined | interleaved_arrival_selective_attn/interleaved_all_attn | -0.176 | [-0.573,0.194] | 0.999043 | 1.005959 | 1/0 | 535.236 / 532.065 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | -0.226 | [-0.667,0.257] | 0.996898 | 1.012521 | 1/0 | 523.126 / 516.657 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | -0.271 | [-0.705,0.189] | 0.995535 | 1.010149 | 1/7 | 520.684 / 515.452 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | -0.497 | [-0.849,-0.127] | 0.994038 | 1.004479 | 1/7 | 517.761 / 515.452 |

### 同策略范围比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_all_attn_mlp/interleaved_all_mlp | 3.390 | [2.930,3.845] | 1.036316 | 1.046827 | 1/3 | 538.131 / 514.059 |
| 1 | interleaved_all_attn_mlp/interleaved_all_attn | 3.192 | [2.826,3.533] | 1.034096 | 1.040696 | 1/3 | 534.980 / 514.059 |
| 1 | interleaved_all_mlp/interleaved_all_attn | -0.205 | [-0.465,0.058] | 0.997405 | 1.004717 | 1/5 | 534.875 / 532.364 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.169 | [2.712,3.556] | 1.035221 | 1.043618 | 1/6 | 539.193 / 516.657 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.231 | [2.935,3.503] | 1.035067 | 1.039531 | 1/0 | 537.081 / 516.657 |
| 1 | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | 0.064 | [-0.354,0.505] | 0.997475 | 1.011086 | 1/7 | 541.121 / 535.188 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.133 | [2.915,3.342] | 1.032266 | 1.036505 | 1/7 | 534.269 / 515.452 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 2.881 | [2.479,3.300] | 1.030153 | 1.042878 | 1/7 | 537.554 / 515.452 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.260 | [-0.519,0.033] | 0.996471 | 1.006149 | 1/7 | 537.554 / 534.269 |
| combined | interleaved_all_attn_mlp/interleaved_all_mlp | 3.390 | [2.930,3.845] | 1.036316 | 1.046827 | 1/3 | 538.131 / 514.059 |
| combined | interleaved_all_attn_mlp/interleaved_all_attn | 3.192 | [2.826,3.533] | 1.034096 | 1.040696 | 1/3 | 534.980 / 514.059 |
| combined | interleaved_all_mlp/interleaved_all_attn | -0.205 | [-0.465,0.058] | 0.997405 | 1.004717 | 1/5 | 534.875 / 532.364 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.169 | [2.712,3.556] | 1.035221 | 1.043618 | 1/6 | 539.193 / 516.657 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.231 | [2.935,3.503] | 1.035067 | 1.039531 | 1/0 | 537.081 / 516.657 |
| combined | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | 0.064 | [-0.354,0.505] | 0.997475 | 1.011086 | 1/7 | 541.121 / 535.188 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.133 | [2.915,3.342] | 1.032266 | 1.036505 | 1/7 | 534.269 / 515.452 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 2.881 | [2.479,3.300] | 1.030153 | 1.042878 | 1/7 | 537.554 / 515.452 |
| combined | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.260 | [-0.519,0.033] | 0.996471 | 1.006149 | 1/7 | 537.554 / 534.269 |
## 慢窗口敏感性附表

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | interleaved_all_mlp | 535.170 | 15307.28 |
| 1 | interleaved_arrival_all_mlp | 535.188 | 15306.78 |
| 1 | interleaved_arrival_selective_mlp | 535.679 | 15292.73 |
| 1 | interleaved_all_attn | 533.949 | 15342.30 |
| 1 | interleaved_arrival_all_attn | 534.413 | 15328.98 |
| 1 | interleaved_arrival_selective_attn | 534.675 | 15321.46 |
| 1 | interleaved_all_attn_mlp | 516.033 | 15874.96 |
| 1 | interleaved_arrival_all_attn_mlp | 517.115 | 15841.75 |
| 1 | interleaved_arrival_selective_attn_mlp | 518.974 | 15784.98 |

### 同范围策略比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_arrival_all_mlp/interleaved_all_mlp | 0.003 | [-0.349,0.440] | 0.999326 | 1.014327 | 1/3 | 538.131 / 530.530 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.234 | [-0.539,0.095] | 0.998549 | 1.008116 | 1/6 | 539.193 / 534.852 |
| 1 | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.231 | [-0.466,0.040] | 0.997010 | 1.006102 | 1/3 | 538.131 / 534.868 |
| 1 | interleaved_arrival_all_attn/interleaved_all_attn | -0.267 | [-0.752,0.139] | 0.997578 | 1.007671 | 1/2 | 536.554 / 532.469 |
| 1 | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | 0.091 | [-0.229,0.435] | 0.999138 | 1.009428 | 1/0 | 537.081 / 532.065 |
| 1 | interleaved_arrival_selective_attn/interleaved_all_attn | -0.176 | [-0.573,0.194] | 0.999043 | 1.005959 | 1/0 | 535.236 / 532.065 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | -0.226 | [-0.667,0.257] | 0.996898 | 1.012521 | 1/0 | 523.126 / 516.657 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | -0.271 | [-0.705,0.189] | 0.995535 | 1.010149 | 1/7 | 520.684 / 515.452 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | -0.497 | [-0.849,-0.127] | 0.994038 | 1.004479 | 1/7 | 517.761 / 515.452 |

### 同策略范围比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_all_attn_mlp/interleaved_all_mlp | 3.390 | [2.930,3.845] | 1.036316 | 1.046827 | 1/3 | 538.131 / 514.059 |
| 1 | interleaved_all_attn_mlp/interleaved_all_attn | 3.192 | [2.826,3.533] | 1.034096 | 1.040696 | 1/3 | 534.980 / 514.059 |
| 1 | interleaved_all_mlp/interleaved_all_attn | -0.205 | [-0.465,0.058] | 0.997405 | 1.004717 | 1/5 | 534.875 / 532.364 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.169 | [2.712,3.556] | 1.035221 | 1.043618 | 1/6 | 539.193 / 516.657 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.231 | [2.935,3.503] | 1.035067 | 1.039531 | 1/0 | 537.081 / 516.657 |
| 1 | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | 0.064 | [-0.354,0.505] | 0.997475 | 1.011086 | 1/7 | 541.121 / 535.188 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.133 | [2.915,3.342] | 1.032266 | 1.036505 | 1/7 | 534.269 / 515.452 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 2.881 | [2.479,3.300] | 1.030153 | 1.042878 | 1/7 | 537.554 / 515.452 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.260 | [-0.519,0.033] | 0.996471 | 1.006149 | 1/7 | 537.554 / 534.269 |

最好值仅为单块观测；负值和跨零区间保留。范围差异不等于局部算子耗时，也不能直接相加。
筛选轮次1：剩余9块；触发与排除：`[]`。
