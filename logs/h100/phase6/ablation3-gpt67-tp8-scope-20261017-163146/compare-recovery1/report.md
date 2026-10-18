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
| 1 | interleaved_all_mlp | 535.231 | 15305.54 |
| 1 | interleaved_arrival_all_mlp | 535.474 | 15298.59 |
| 1 | interleaved_arrival_selective_mlp | 534.639 | 15322.50 |
| 1 | interleaved_all_attn | 535.060 | 15310.43 |
| 1 | interleaved_arrival_all_attn | 535.264 | 15304.60 |
| 1 | interleaved_arrival_selective_attn | 532.210 | 15392.41 |
| 1 | interleaved_all_attn_mlp | 514.569 | 15920.13 |
| 1 | interleaved_arrival_all_attn_mlp | 514.065 | 15935.74 |
| 1 | interleaved_arrival_selective_attn_mlp | 514.865 | 15910.95 |
| combined | interleaved_all_mlp | 535.231 | 15305.54 |
| combined | interleaved_arrival_all_mlp | 535.474 | 15298.59 |
| combined | interleaved_arrival_selective_mlp | 534.639 | 15322.50 |
| combined | interleaved_all_attn | 535.060 | 15310.43 |
| combined | interleaved_arrival_all_attn | 535.264 | 15304.60 |
| combined | interleaved_arrival_selective_attn | 532.210 | 15392.41 |
| combined | interleaved_all_attn_mlp | 514.569 | 15920.13 |
| combined | interleaved_arrival_all_attn_mlp | 514.065 | 15935.74 |
| combined | interleaved_arrival_selective_attn_mlp | 514.865 | 15910.95 |

### 同范围策略比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_arrival_all_mlp/interleaved_all_mlp | -0.147 | [-0.654,0.323] | 0.999546 | 1.008193 | 1/8 | 537.788 / 533.418 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.089 | [-0.557,0.435] | 0.998059 | 1.015296 | 1/7 | 540.317 / 532.176 |
| 1 | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.236 | [-0.857,0.273] | 0.999637 | 1.007109 | 1/5 | 535.597 / 531.816 |
| 1 | interleaved_arrival_all_attn/interleaved_all_attn | 0.006 | [-0.583,0.513] | 1.005025 | 1.008833 | 1/0 | 535.744 / 531.054 |
| 1 | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | 0.381 | [-0.130,0.871] | 1.003189 | 1.013530 | 1/3 | 538.980 / 531.785 |
| 1 | interleaved_arrival_selective_attn/interleaved_all_attn | 0.388 | [0.084,0.700] | 1.001491 | 1.011968 | 1/7 | 537.029 / 530.678 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | 0.072 | [-0.253,0.421] | 1.000227 | 1.010660 | 1/1 | 519.121 / 513.645 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | 0.055 | [-0.181,0.291] | 1.001211 | 1.005458 | 1/3 | 516.203 / 513.401 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | 0.127 | [-0.171,0.397] | 1.002215 | 1.008167 | 1/1 | 519.121 / 514.915 |
| combined | interleaved_arrival_all_mlp/interleaved_all_mlp | -0.147 | [-0.654,0.323] | 0.999546 | 1.008193 | 1/8 | 537.788 / 533.418 |
| combined | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.089 | [-0.557,0.435] | 0.998059 | 1.015296 | 1/7 | 540.317 / 532.176 |
| combined | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.236 | [-0.857,0.273] | 0.999637 | 1.007109 | 1/5 | 535.597 / 531.816 |
| combined | interleaved_arrival_all_attn/interleaved_all_attn | 0.006 | [-0.583,0.513] | 1.005025 | 1.008833 | 1/0 | 535.744 / 531.054 |
| combined | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | 0.381 | [-0.130,0.871] | 1.003189 | 1.013530 | 1/3 | 538.980 / 531.785 |
| combined | interleaved_arrival_selective_attn/interleaved_all_attn | 0.388 | [0.084,0.700] | 1.001491 | 1.011968 | 1/7 | 537.029 / 530.678 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | 0.072 | [-0.253,0.421] | 1.000227 | 1.010660 | 1/1 | 519.121 / 513.645 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | 0.055 | [-0.181,0.291] | 1.001211 | 1.005458 | 1/3 | 516.203 / 513.401 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | 0.127 | [-0.171,0.397] | 1.002215 | 1.008167 | 1/1 | 519.121 / 514.915 |

### 同策略范围比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_all_attn_mlp/interleaved_all_mlp | 3.625 | [3.291,4.030] | 1.035175 | 1.051701 | 1/8 | 537.788 / 511.351 |
| 1 | interleaved_all_attn_mlp/interleaved_all_attn | 3.720 | [3.258,4.153] | 1.037059 | 1.048362 | 1/4 | 538.589 / 513.743 |
| 1 | interleaved_all_mlp/interleaved_all_attn | 0.098 | [-0.390,0.569] | 1.001105 | 1.012739 | 1/4 | 538.589 / 531.814 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.835 | [3.573,4.186] | 1.038206 | 1.053211 | 1/7 | 540.317 / 513.018 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.783 | [3.405,4.150] | 1.041388 | 1.050042 | 1/1 | 539.349 / 513.645 |
| 1 | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | -0.054 | [-0.503,0.393] | 0.999408 | 1.009888 | 1/3 | 538.980 / 533.702 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.974 | [3.613,4.402] | 1.038304 | 1.056254 | 1/4 | 543.829 / 514.865 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 3.468 | [3.184,3.777] | 1.034919 | 1.044519 | 1/4 | 537.787 / 514.865 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.527 | [-0.956,-0.111] | 0.995458 | 1.005670 | 1/5 | 534.831 / 531.816 |
| combined | interleaved_all_attn_mlp/interleaved_all_mlp | 3.625 | [3.291,4.030] | 1.035175 | 1.051701 | 1/8 | 537.788 / 511.351 |
| combined | interleaved_all_attn_mlp/interleaved_all_attn | 3.720 | [3.258,4.153] | 1.037059 | 1.048362 | 1/4 | 538.589 / 513.743 |
| combined | interleaved_all_mlp/interleaved_all_attn | 0.098 | [-0.390,0.569] | 1.001105 | 1.012739 | 1/4 | 538.589 / 531.814 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.835 | [3.573,4.186] | 1.038206 | 1.053211 | 1/7 | 540.317 / 513.018 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.783 | [3.405,4.150] | 1.041388 | 1.050042 | 1/1 | 539.349 / 513.645 |
| combined | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | -0.054 | [-0.503,0.393] | 0.999408 | 1.009888 | 1/3 | 538.980 / 533.702 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.974 | [3.613,4.402] | 1.038304 | 1.056254 | 1/4 | 543.829 / 514.865 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 3.468 | [3.184,3.777] | 1.034919 | 1.044519 | 1/4 | 537.787 / 514.865 |
| combined | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.527 | [-0.956,-0.111] | 0.995458 | 1.005670 | 1/5 | 534.831 / 531.816 |
## 慢窗口敏感性附表

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | interleaved_all_mlp | 535.231 | 15305.54 |
| 1 | interleaved_arrival_all_mlp | 535.474 | 15298.59 |
| 1 | interleaved_arrival_selective_mlp | 534.639 | 15322.50 |
| 1 | interleaved_all_attn | 535.060 | 15310.43 |
| 1 | interleaved_arrival_all_attn | 535.264 | 15304.60 |
| 1 | interleaved_arrival_selective_attn | 532.210 | 15392.41 |
| 1 | interleaved_all_attn_mlp | 514.569 | 15920.13 |
| 1 | interleaved_arrival_all_attn_mlp | 514.065 | 15935.74 |
| 1 | interleaved_arrival_selective_attn_mlp | 514.865 | 15910.95 |

### 同范围策略比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_arrival_all_mlp/interleaved_all_mlp | -0.147 | [-0.654,0.323] | 0.999546 | 1.008193 | 1/8 | 537.788 / 533.418 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.089 | [-0.557,0.435] | 0.998059 | 1.015296 | 1/7 | 540.317 / 532.176 |
| 1 | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.236 | [-0.857,0.273] | 0.999637 | 1.007109 | 1/5 | 535.597 / 531.816 |
| 1 | interleaved_arrival_all_attn/interleaved_all_attn | 0.006 | [-0.583,0.513] | 1.005025 | 1.008833 | 1/0 | 535.744 / 531.054 |
| 1 | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | 0.381 | [-0.130,0.871] | 1.003189 | 1.013530 | 1/3 | 538.980 / 531.785 |
| 1 | interleaved_arrival_selective_attn/interleaved_all_attn | 0.388 | [0.084,0.700] | 1.001491 | 1.011968 | 1/7 | 537.029 / 530.678 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | 0.072 | [-0.253,0.421] | 1.000227 | 1.010660 | 1/1 | 519.121 / 513.645 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | 0.055 | [-0.181,0.291] | 1.001211 | 1.005458 | 1/3 | 516.203 / 513.401 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | 0.127 | [-0.171,0.397] | 1.002215 | 1.008167 | 1/1 | 519.121 / 514.915 |

### 同策略范围比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_all_attn_mlp/interleaved_all_mlp | 3.625 | [3.291,4.030] | 1.035175 | 1.051701 | 1/8 | 537.788 / 511.351 |
| 1 | interleaved_all_attn_mlp/interleaved_all_attn | 3.720 | [3.258,4.153] | 1.037059 | 1.048362 | 1/4 | 538.589 / 513.743 |
| 1 | interleaved_all_mlp/interleaved_all_attn | 0.098 | [-0.390,0.569] | 1.001105 | 1.012739 | 1/4 | 538.589 / 531.814 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.835 | [3.573,4.186] | 1.038206 | 1.053211 | 1/7 | 540.317 / 513.018 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.783 | [3.405,4.150] | 1.041388 | 1.050042 | 1/1 | 539.349 / 513.645 |
| 1 | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | -0.054 | [-0.503,0.393] | 0.999408 | 1.009888 | 1/3 | 538.980 / 533.702 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.974 | [3.613,4.402] | 1.038304 | 1.056254 | 1/4 | 543.829 / 514.865 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 3.468 | [3.184,3.777] | 1.034919 | 1.044519 | 1/4 | 537.787 / 514.865 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.527 | [-0.956,-0.111] | 0.995458 | 1.005670 | 1/5 | 534.831 / 531.816 |

最好值仅为单块观测；负值和跨零区间保留。范围差异不等于局部算子耗时，也不能直接相加。
筛选轮次1：剩余9块；触发与排除：`[]`。
