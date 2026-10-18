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
| 1 | interleaved_all_mlp | 535.110 | 15308.99 |
| 1 | interleaved_arrival_all_mlp | 534.694 | 15320.92 |
| 1 | interleaved_arrival_selective_mlp | 535.816 | 15288.83 |
| 1 | interleaved_all_attn | 535.223 | 15305.76 |
| 1 | interleaved_arrival_all_attn | 533.713 | 15349.07 |
| 1 | interleaved_arrival_selective_attn | 532.509 | 15383.78 |
| 1 | interleaved_all_attn_mlp | 515.758 | 15883.42 |
| 1 | interleaved_arrival_all_attn_mlp | 515.795 | 15882.28 |
| 1 | interleaved_arrival_selective_attn_mlp | 515.344 | 15896.16 |
| combined | interleaved_all_mlp | 535.110 | 15308.99 |
| combined | interleaved_arrival_all_mlp | 534.694 | 15320.92 |
| combined | interleaved_arrival_selective_mlp | 535.816 | 15288.83 |
| combined | interleaved_all_attn | 535.223 | 15305.76 |
| combined | interleaved_arrival_all_attn | 533.713 | 15349.07 |
| combined | interleaved_arrival_selective_attn | 532.509 | 15383.78 |
| combined | interleaved_all_attn_mlp | 515.758 | 15883.42 |
| combined | interleaved_arrival_all_attn_mlp | 515.795 | 15882.28 |
| combined | interleaved_arrival_selective_attn_mlp | 515.344 | 15896.16 |

### 同范围策略比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_arrival_all_mlp/interleaved_all_mlp | 0.099 | [-0.142,0.355] | 1.000272 | 1.006520 | 1/0 | 537.864 / 534.379 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.225 | [-0.499,0.048] | 0.997699 | 1.005431 | 1/3 | 537.134 / 534.233 |
| 1 | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.126 | [-0.304,0.070] | 0.997971 | 1.004190 | 1/0 | 537.864 / 535.619 |
| 1 | interleaved_arrival_all_attn/interleaved_all_attn | 0.122 | [-0.240,0.500] | 0.999770 | 1.011991 | 1/7 | 535.974 / 529.623 |
| 1 | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | 0.208 | [-0.220,0.629] | 1.003203 | 1.010327 | 1/3 | 533.713 / 528.258 |
| 1 | interleaved_arrival_selective_attn/interleaved_all_attn | 0.330 | [-0.056,0.765] | 1.004109 | 1.016688 | 1/3 | 537.074 / 528.258 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | 0.030 | [-0.319,0.378] | 1.000641 | 1.009926 | 1/1 | 517.164 / 512.081 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | 0.054 | [-0.280,0.353] | 1.000874 | 1.006243 | 1/6 | 518.915 / 515.696 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | 0.084 | [-0.226,0.372] | 1.000895 | 1.007359 | 1/5 | 517.337 / 513.558 |
| combined | interleaved_arrival_all_mlp/interleaved_all_mlp | 0.099 | [-0.142,0.355] | 1.000272 | 1.006520 | 1/0 | 537.864 / 534.379 |
| combined | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.225 | [-0.499,0.048] | 0.997699 | 1.005431 | 1/3 | 537.134 / 534.233 |
| combined | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.126 | [-0.304,0.070] | 0.997971 | 1.004190 | 1/0 | 537.864 / 535.619 |
| combined | interleaved_arrival_all_attn/interleaved_all_attn | 0.122 | [-0.240,0.500] | 0.999770 | 1.011991 | 1/7 | 535.974 / 529.623 |
| combined | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | 0.208 | [-0.220,0.629] | 1.003203 | 1.010327 | 1/3 | 533.713 / 528.258 |
| combined | interleaved_arrival_selective_attn/interleaved_all_attn | 0.330 | [-0.056,0.765] | 1.004109 | 1.016688 | 1/3 | 537.074 / 528.258 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | 0.030 | [-0.319,0.378] | 1.000641 | 1.009926 | 1/1 | 517.164 / 512.081 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | 0.054 | [-0.280,0.353] | 1.000874 | 1.006243 | 1/6 | 518.915 / 515.696 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | 0.084 | [-0.226,0.372] | 1.000895 | 1.007359 | 1/5 | 517.337 / 513.558 |

### 同策略范围比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_all_attn_mlp/interleaved_all_mlp | 3.668 | [3.290,4.058] | 1.038257 | 1.048397 | 1/2 | 536.499 / 511.733 |
| 1 | interleaved_all_attn_mlp/interleaved_all_attn | 3.603 | [3.341,3.834] | 1.039197 | 1.041613 | 1/6 | 536.527 / 515.093 |
| 1 | interleaved_all_mlp/interleaved_all_attn | -0.067 | [-0.354,0.195] | 0.998712 | 1.005326 | 1/7 | 535.974 / 533.135 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.601 | [3.346,3.865] | 1.036086 | 1.044159 | 1/1 | 534.694 / 512.081 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.514 | [3.066,3.969] | 1.035226 | 1.050584 | 1/1 | 537.984 / 512.081 |
| 1 | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | -0.089 | [-0.446,0.256] | 0.999677 | 1.006528 | 1/8 | 537.574 / 534.087 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.869 | [3.576,4.212] | 1.038563 | 1.050305 | 1/5 | 539.393 / 513.558 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 3.366 | [3.012,3.663] | 1.036285 | 1.042495 | 1/6 | 537.610 / 515.696 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.524 | [-0.927,-0.142] | 0.996014 | 1.003348 | 1/6 | 537.610 / 535.816 |
| combined | interleaved_all_attn_mlp/interleaved_all_mlp | 3.668 | [3.290,4.058] | 1.038257 | 1.048397 | 1/2 | 536.499 / 511.733 |
| combined | interleaved_all_attn_mlp/interleaved_all_attn | 3.603 | [3.341,3.834] | 1.039197 | 1.041613 | 1/6 | 536.527 / 515.093 |
| combined | interleaved_all_mlp/interleaved_all_attn | -0.067 | [-0.354,0.195] | 0.998712 | 1.005326 | 1/7 | 535.974 / 533.135 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.601 | [3.346,3.865] | 1.036086 | 1.044159 | 1/1 | 534.694 / 512.081 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.514 | [3.066,3.969] | 1.035226 | 1.050584 | 1/1 | 537.984 / 512.081 |
| combined | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | -0.089 | [-0.446,0.256] | 0.999677 | 1.006528 | 1/8 | 537.574 / 534.087 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.869 | [3.576,4.212] | 1.038563 | 1.050305 | 1/5 | 539.393 / 513.558 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 3.366 | [3.012,3.663] | 1.036285 | 1.042495 | 1/6 | 537.610 / 515.696 |
| combined | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.524 | [-0.927,-0.142] | 0.996014 | 1.003348 | 1/6 | 537.610 / 535.816 |
## 慢窗口敏感性附表

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | interleaved_all_mlp | 535.110 | 15308.99 |
| 1 | interleaved_arrival_all_mlp | 534.694 | 15320.92 |
| 1 | interleaved_arrival_selective_mlp | 535.816 | 15288.83 |
| 1 | interleaved_all_attn | 535.223 | 15305.76 |
| 1 | interleaved_arrival_all_attn | 533.713 | 15349.07 |
| 1 | interleaved_arrival_selective_attn | 532.509 | 15383.78 |
| 1 | interleaved_all_attn_mlp | 515.758 | 15883.42 |
| 1 | interleaved_arrival_all_attn_mlp | 515.795 | 15882.28 |
| 1 | interleaved_arrival_selective_attn_mlp | 515.344 | 15896.16 |

### 同范围策略比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_arrival_all_mlp/interleaved_all_mlp | 0.099 | [-0.142,0.355] | 1.000272 | 1.006520 | 1/0 | 537.864 / 534.379 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.225 | [-0.499,0.048] | 0.997699 | 1.005431 | 1/3 | 537.134 / 534.233 |
| 1 | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.126 | [-0.304,0.070] | 0.997971 | 1.004190 | 1/0 | 537.864 / 535.619 |
| 1 | interleaved_arrival_all_attn/interleaved_all_attn | 0.122 | [-0.240,0.500] | 0.999770 | 1.011991 | 1/7 | 535.974 / 529.623 |
| 1 | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | 0.208 | [-0.220,0.629] | 1.003203 | 1.010327 | 1/3 | 533.713 / 528.258 |
| 1 | interleaved_arrival_selective_attn/interleaved_all_attn | 0.330 | [-0.056,0.765] | 1.004109 | 1.016688 | 1/3 | 537.074 / 528.258 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | 0.030 | [-0.319,0.378] | 1.000641 | 1.009926 | 1/1 | 517.164 / 512.081 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | 0.054 | [-0.280,0.353] | 1.000874 | 1.006243 | 1/6 | 518.915 / 515.696 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | 0.084 | [-0.226,0.372] | 1.000895 | 1.007359 | 1/5 | 517.337 / 513.558 |

### 同策略范围比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_all_attn_mlp/interleaved_all_mlp | 3.668 | [3.290,4.058] | 1.038257 | 1.048397 | 1/2 | 536.499 / 511.733 |
| 1 | interleaved_all_attn_mlp/interleaved_all_attn | 3.603 | [3.341,3.834] | 1.039197 | 1.041613 | 1/6 | 536.527 / 515.093 |
| 1 | interleaved_all_mlp/interleaved_all_attn | -0.067 | [-0.354,0.195] | 0.998712 | 1.005326 | 1/7 | 535.974 / 533.135 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.601 | [3.346,3.865] | 1.036086 | 1.044159 | 1/1 | 534.694 / 512.081 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.514 | [3.066,3.969] | 1.035226 | 1.050584 | 1/1 | 537.984 / 512.081 |
| 1 | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | -0.089 | [-0.446,0.256] | 0.999677 | 1.006528 | 1/8 | 537.574 / 534.087 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.869 | [3.576,4.212] | 1.038563 | 1.050305 | 1/5 | 539.393 / 513.558 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 3.366 | [3.012,3.663] | 1.036285 | 1.042495 | 1/6 | 537.610 / 515.696 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.524 | [-0.927,-0.142] | 0.996014 | 1.003348 | 1/6 | 537.610 / 535.816 |

最好值仅为单块观测；负值和跨零区间保留。范围差异不等于局部算子耗时，也不能直接相加。
筛选轮次1：剩余9块；触发与排除：`[]`。
