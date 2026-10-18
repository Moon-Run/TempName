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
| 1 | interleaved_all_mlp | 534.541 | 15325.30 |
| 1 | interleaved_arrival_all_mlp | 534.766 | 15318.85 |
| 1 | interleaved_arrival_selective_mlp | 535.409 | 15300.45 |
| 1 | interleaved_all_attn | 534.595 | 15323.75 |
| 1 | interleaved_arrival_all_attn | 534.790 | 15318.17 |
| 1 | interleaved_arrival_selective_attn | 535.848 | 15287.91 |
| 1 | interleaved_all_attn_mlp | 515.156 | 15901.99 |
| 1 | interleaved_arrival_all_attn_mlp | 514.911 | 15909.56 |
| 1 | interleaved_arrival_selective_attn_mlp | 514.968 | 15907.79 |
| combined | interleaved_all_mlp | 534.541 | 15325.30 |
| combined | interleaved_arrival_all_mlp | 534.766 | 15318.85 |
| combined | interleaved_arrival_selective_mlp | 535.409 | 15300.45 |
| combined | interleaved_all_attn | 534.595 | 15323.75 |
| combined | interleaved_arrival_all_attn | 534.790 | 15318.17 |
| combined | interleaved_arrival_selective_attn | 535.848 | 15287.91 |
| combined | interleaved_all_attn_mlp | 515.156 | 15901.99 |
| combined | interleaved_arrival_all_attn_mlp | 514.911 | 15909.56 |
| combined | interleaved_arrival_selective_attn_mlp | 514.968 | 15907.79 |

### 同范围策略比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_arrival_all_mlp/interleaved_all_mlp | 0.065 | [-0.212,0.353] | 1.000265 | 1.007262 | 1/8 | 535.429 / 531.569 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.220 | [-0.562,0.111] | 0.998171 | 1.007624 | 1/2 | 535.222 / 531.172 |
| 1 | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.155 | [-0.425,0.147] | 0.996623 | 1.007891 | 1/2 | 535.363 / 531.172 |
| 1 | interleaved_arrival_all_attn/interleaved_all_attn | 0.154 | [-0.220,0.543] | 1.000995 | 1.011986 | 1/5 | 537.581 / 531.214 |
| 1 | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | -0.056 | [-0.384,0.224] | 1.001845 | 1.005275 | 1/6 | 534.790 / 531.984 |
| 1 | interleaved_arrival_selective_attn/interleaved_all_attn | 0.098 | [-0.193,0.397] | 1.002437 | 1.010120 | 1/6 | 537.367 / 531.984 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | -0.263 | [-0.643,0.131] | 0.996256 | 1.007527 | 1/3 | 515.904 / 512.050 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | 0.163 | [-0.129,0.436] | 1.003006 | 1.009358 | 1/6 | 519.787 / 514.968 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | -0.100 | [-0.472,0.196] | 0.999083 | 1.005962 | 1/4 | 516.228 / 513.169 |
| combined | interleaved_arrival_all_mlp/interleaved_all_mlp | 0.065 | [-0.212,0.353] | 1.000265 | 1.007262 | 1/8 | 535.429 / 531.569 |
| combined | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.220 | [-0.562,0.111] | 0.998171 | 1.007624 | 1/2 | 535.222 / 531.172 |
| combined | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.155 | [-0.425,0.147] | 0.996623 | 1.007891 | 1/2 | 535.363 / 531.172 |
| combined | interleaved_arrival_all_attn/interleaved_all_attn | 0.154 | [-0.220,0.543] | 1.000995 | 1.011986 | 1/5 | 537.581 / 531.214 |
| combined | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | -0.056 | [-0.384,0.224] | 1.001845 | 1.005275 | 1/6 | 534.790 / 531.984 |
| combined | interleaved_arrival_selective_attn/interleaved_all_attn | 0.098 | [-0.193,0.397] | 1.002437 | 1.010120 | 1/6 | 537.367 / 531.984 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | -0.263 | [-0.643,0.131] | 0.996256 | 1.007527 | 1/3 | 515.904 / 512.050 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | 0.163 | [-0.129,0.436] | 1.003006 | 1.009358 | 1/6 | 519.787 / 514.968 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | -0.100 | [-0.472,0.196] | 0.999083 | 1.005962 | 1/4 | 516.228 / 513.169 |

### 同策略范围比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_all_attn_mlp/interleaved_all_mlp | 3.767 | [3.417,4.189] | 1.037277 | 1.053700 | 1/2 | 535.363 / 508.079 |
| 1 | interleaved_all_attn_mlp/interleaved_all_attn | 3.933 | [3.657,4.249] | 1.041151 | 1.052188 | 1/2 | 534.595 / 508.079 |
| 1 | interleaved_all_mlp/interleaved_all_attn | 0.172 | [-0.168,0.467] | 1.003866 | 1.006699 | 1/0 | 534.347 / 530.792 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.451 | [3.161,3.719] | 1.038843 | 1.040354 | 1/3 | 532.713 / 512.050 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.531 | [3.107,3.913] | 1.037004 | 1.045314 | 1/3 | 535.253 / 512.050 |
| 1 | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | 0.083 | [-0.209,0.384] | 1.000352 | 1.008761 | 1/8 | 536.226 / 531.569 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.820 | [3.532,4.092] | 1.040618 | 1.047510 | 1/1 | 537.757 / 513.367 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 3.742 | [3.531,3.947] | 1.038214 | 1.044062 | 1/1 | 535.987 / 513.367 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.081 | [-0.335,0.157] | 0.999995 | 1.004859 | 1/5 | 536.093 / 533.500 |
| combined | interleaved_all_attn_mlp/interleaved_all_mlp | 3.767 | [3.417,4.189] | 1.037277 | 1.053700 | 1/2 | 535.363 / 508.079 |
| combined | interleaved_all_attn_mlp/interleaved_all_attn | 3.933 | [3.657,4.249] | 1.041151 | 1.052188 | 1/2 | 534.595 / 508.079 |
| combined | interleaved_all_mlp/interleaved_all_attn | 0.172 | [-0.168,0.467] | 1.003866 | 1.006699 | 1/0 | 534.347 / 530.792 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.451 | [3.161,3.719] | 1.038843 | 1.040354 | 1/3 | 532.713 / 512.050 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.531 | [3.107,3.913] | 1.037004 | 1.045314 | 1/3 | 535.253 / 512.050 |
| combined | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | 0.083 | [-0.209,0.384] | 1.000352 | 1.008761 | 1/8 | 536.226 / 531.569 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.820 | [3.532,4.092] | 1.040618 | 1.047510 | 1/1 | 537.757 / 513.367 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 3.742 | [3.531,3.947] | 1.038214 | 1.044062 | 1/1 | 535.987 / 513.367 |
| combined | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.081 | [-0.335,0.157] | 0.999995 | 1.004859 | 1/5 | 536.093 / 533.500 |
## 慢窗口敏感性附表

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | interleaved_all_mlp | 534.541 | 15325.30 |
| 1 | interleaved_arrival_all_mlp | 534.766 | 15318.85 |
| 1 | interleaved_arrival_selective_mlp | 535.409 | 15300.45 |
| 1 | interleaved_all_attn | 534.595 | 15323.75 |
| 1 | interleaved_arrival_all_attn | 534.790 | 15318.17 |
| 1 | interleaved_arrival_selective_attn | 535.848 | 15287.91 |
| 1 | interleaved_all_attn_mlp | 515.156 | 15901.99 |
| 1 | interleaved_arrival_all_attn_mlp | 514.911 | 15909.56 |
| 1 | interleaved_arrival_selective_attn_mlp | 514.968 | 15907.79 |

### 同范围策略比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_arrival_all_mlp/interleaved_all_mlp | 0.065 | [-0.212,0.353] | 1.000265 | 1.007262 | 1/8 | 535.429 / 531.569 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.220 | [-0.562,0.111] | 0.998171 | 1.007624 | 1/2 | 535.222 / 531.172 |
| 1 | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.155 | [-0.425,0.147] | 0.996623 | 1.007891 | 1/2 | 535.363 / 531.172 |
| 1 | interleaved_arrival_all_attn/interleaved_all_attn | 0.154 | [-0.220,0.543] | 1.000995 | 1.011986 | 1/5 | 537.581 / 531.214 |
| 1 | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | -0.056 | [-0.384,0.224] | 1.001845 | 1.005275 | 1/6 | 534.790 / 531.984 |
| 1 | interleaved_arrival_selective_attn/interleaved_all_attn | 0.098 | [-0.193,0.397] | 1.002437 | 1.010120 | 1/6 | 537.367 / 531.984 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | -0.263 | [-0.643,0.131] | 0.996256 | 1.007527 | 1/3 | 515.904 / 512.050 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | 0.163 | [-0.129,0.436] | 1.003006 | 1.009358 | 1/6 | 519.787 / 514.968 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | -0.100 | [-0.472,0.196] | 0.999083 | 1.005962 | 1/4 | 516.228 / 513.169 |

### 同策略范围比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_all_attn_mlp/interleaved_all_mlp | 3.767 | [3.417,4.189] | 1.037277 | 1.053700 | 1/2 | 535.363 / 508.079 |
| 1 | interleaved_all_attn_mlp/interleaved_all_attn | 3.933 | [3.657,4.249] | 1.041151 | 1.052188 | 1/2 | 534.595 / 508.079 |
| 1 | interleaved_all_mlp/interleaved_all_attn | 0.172 | [-0.168,0.467] | 1.003866 | 1.006699 | 1/0 | 534.347 / 530.792 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.451 | [3.161,3.719] | 1.038843 | 1.040354 | 1/3 | 532.713 / 512.050 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.531 | [3.107,3.913] | 1.037004 | 1.045314 | 1/3 | 535.253 / 512.050 |
| 1 | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | 0.083 | [-0.209,0.384] | 1.000352 | 1.008761 | 1/8 | 536.226 / 531.569 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.820 | [3.532,4.092] | 1.040618 | 1.047510 | 1/1 | 537.757 / 513.367 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 3.742 | [3.531,3.947] | 1.038214 | 1.044062 | 1/1 | 535.987 / 513.367 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.081 | [-0.335,0.157] | 0.999995 | 1.004859 | 1/5 | 536.093 / 533.500 |

最好值仅为单块观测；负值和跨零区间保留。范围差异不等于局部算子耗时，也不能直接相加。
筛选轮次1：剩余9块；触发与排除：`[]`。
