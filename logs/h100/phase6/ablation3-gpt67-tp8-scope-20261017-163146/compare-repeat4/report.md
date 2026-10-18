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
| 1 | interleaved_all_mlp | 534.846 | 15316.55 |
| 1 | interleaved_arrival_all_mlp | 535.289 | 15303.89 |
| 1 | interleaved_arrival_selective_mlp | 536.499 | 15269.38 |
| 1 | interleaved_all_attn | 534.327 | 15331.43 |
| 1 | interleaved_arrival_all_attn | 534.639 | 15322.48 |
| 1 | interleaved_arrival_selective_attn | 533.464 | 15356.22 |
| 1 | interleaved_all_attn_mlp | 515.232 | 15899.64 |
| 1 | interleaved_arrival_all_attn_mlp | 515.256 | 15898.89 |
| 1 | interleaved_arrival_selective_attn_mlp | 517.044 | 15843.93 |
| combined | interleaved_all_mlp | 534.846 | 15316.55 |
| combined | interleaved_arrival_all_mlp | 535.289 | 15303.89 |
| combined | interleaved_arrival_selective_mlp | 536.499 | 15269.38 |
| combined | interleaved_all_attn | 534.327 | 15331.43 |
| combined | interleaved_arrival_all_attn | 534.639 | 15322.48 |
| combined | interleaved_arrival_selective_attn | 533.464 | 15356.22 |
| combined | interleaved_all_attn_mlp | 515.232 | 15899.64 |
| combined | interleaved_arrival_all_attn_mlp | 515.256 | 15898.89 |
| combined | interleaved_arrival_selective_attn_mlp | 517.044 | 15843.93 |

### 同范围策略比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_arrival_all_mlp/interleaved_all_mlp | -0.312 | [-0.755,0.076] | 0.997942 | 1.005161 | 1/3 | 533.872 / 531.131 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.131 | [-0.547,0.354] | 0.996301 | 1.013165 | 1/6 | 535.289 / 528.333 |
| 1 | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.443 | [-0.839,0.028] | 0.995104 | 1.012409 | 1/6 | 534.890 / 528.333 |
| 1 | interleaved_arrival_all_attn/interleaved_all_attn | 0.078 | [-0.242,0.476] | 0.999999 | 1.013828 | 1/0 | 538.595 / 531.249 |
| 1 | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | 0.089 | [-0.230,0.423] | 0.999689 | 1.009334 | 1/8 | 536.144 / 531.186 |
| 1 | interleaved_arrival_selective_attn/interleaved_all_attn | 0.167 | [-0.052,0.379] | 1.002699 | 1.007109 | 1/0 | 538.595 / 534.794 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | -0.059 | [-0.379,0.241] | 1.002083 | 1.006919 | 1/8 | 515.908 / 512.363 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | -0.258 | [-0.516,0.052] | 0.996034 | 1.006218 | 1/0 | 515.685 / 512.498 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | -0.318 | [-0.567,-0.086] | 0.997660 | 1.002213 | 1/8 | 515.908 / 514.769 |
| combined | interleaved_arrival_all_mlp/interleaved_all_mlp | -0.312 | [-0.755,0.076] | 0.997942 | 1.005161 | 1/3 | 533.872 / 531.131 |
| combined | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.131 | [-0.547,0.354] | 0.996301 | 1.013165 | 1/6 | 535.289 / 528.333 |
| combined | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.443 | [-0.839,0.028] | 0.995104 | 1.012409 | 1/6 | 534.890 / 528.333 |
| combined | interleaved_arrival_all_attn/interleaved_all_attn | 0.078 | [-0.242,0.476] | 0.999999 | 1.013828 | 1/0 | 538.595 / 531.249 |
| combined | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | 0.089 | [-0.230,0.423] | 0.999689 | 1.009334 | 1/8 | 536.144 / 531.186 |
| combined | interleaved_arrival_selective_attn/interleaved_all_attn | 0.167 | [-0.052,0.379] | 1.002699 | 1.007109 | 1/0 | 538.595 / 534.794 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | -0.059 | [-0.379,0.241] | 1.002083 | 1.006919 | 1/8 | 515.908 / 512.363 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | -0.258 | [-0.516,0.052] | 0.996034 | 1.006218 | 1/0 | 515.685 / 512.498 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | -0.318 | [-0.567,-0.086] | 0.997660 | 1.002213 | 1/8 | 515.908 / 514.769 |

### 同策略范围比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_all_attn_mlp/interleaved_all_mlp | 3.550 | [3.259,3.838] | 1.037498 | 1.043268 | 1/0 | 535.116 / 512.922 |
| 1 | interleaved_all_attn_mlp/interleaved_all_attn | 3.620 | [3.369,3.961] | 1.037173 | 1.050053 | 1/0 | 538.595 / 512.922 |
| 1 | interleaved_all_mlp/interleaved_all_attn | 0.072 | [-0.140,0.311] | 0.999612 | 1.006503 | 1/0 | 538.595 / 535.116 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.793 | [3.283,4.320] | 1.040348 | 1.055322 | 1/8 | 540.709 / 512.363 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.487 | [3.186,3.817] | 1.035629 | 1.046413 | 1/8 | 536.144 / 512.363 |
| 1 | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | -0.318 | [-0.730,0.042] | 0.995916 | 1.003595 | 1/1 | 535.005 / 533.088 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.671 | [3.083,4.280] | 1.038840 | 1.057817 | 1/0 | 542.129 / 512.498 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 3.151 | [2.882,3.468] | 1.031759 | 1.043503 | 1/0 | 534.794 / 512.498 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.539 | [-0.954,-0.069] | 0.991875 | 1.010129 | 1/6 | 533.685 / 528.333 |
| combined | interleaved_all_attn_mlp/interleaved_all_mlp | 3.550 | [3.259,3.838] | 1.037498 | 1.043268 | 1/0 | 535.116 / 512.922 |
| combined | interleaved_all_attn_mlp/interleaved_all_attn | 3.620 | [3.369,3.961] | 1.037173 | 1.050053 | 1/0 | 538.595 / 512.922 |
| combined | interleaved_all_mlp/interleaved_all_attn | 0.072 | [-0.140,0.311] | 0.999612 | 1.006503 | 1/0 | 538.595 / 535.116 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.793 | [3.283,4.320] | 1.040348 | 1.055322 | 1/8 | 540.709 / 512.363 |
| combined | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.487 | [3.186,3.817] | 1.035629 | 1.046413 | 1/8 | 536.144 / 512.363 |
| combined | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | -0.318 | [-0.730,0.042] | 0.995916 | 1.003595 | 1/1 | 535.005 / 533.088 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.671 | [3.083,4.280] | 1.038840 | 1.057817 | 1/0 | 542.129 / 512.498 |
| combined | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 3.151 | [2.882,3.468] | 1.031759 | 1.043503 | 1/0 | 534.794 / 512.498 |
| combined | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.539 | [-0.954,-0.069] | 0.991875 | 1.010129 | 1/6 | 533.685 / 528.333 |
## 慢窗口敏感性附表

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | interleaved_all_mlp | 534.846 | 15316.55 |
| 1 | interleaved_arrival_all_mlp | 535.289 | 15303.89 |
| 1 | interleaved_arrival_selective_mlp | 536.499 | 15269.38 |
| 1 | interleaved_all_attn | 534.327 | 15331.43 |
| 1 | interleaved_arrival_all_attn | 534.639 | 15322.48 |
| 1 | interleaved_arrival_selective_attn | 533.464 | 15356.22 |
| 1 | interleaved_all_attn_mlp | 515.232 | 15899.64 |
| 1 | interleaved_arrival_all_attn_mlp | 515.256 | 15898.89 |
| 1 | interleaved_arrival_selective_attn_mlp | 517.044 | 15843.93 |

### 同范围策略比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_arrival_all_mlp/interleaved_all_mlp | -0.312 | [-0.755,0.076] | 0.997942 | 1.005161 | 1/3 | 533.872 / 531.131 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_all_mlp | -0.131 | [-0.547,0.354] | 0.996301 | 1.013165 | 1/6 | 535.289 / 528.333 |
| 1 | interleaved_arrival_selective_mlp/interleaved_all_mlp | -0.443 | [-0.839,0.028] | 0.995104 | 1.012409 | 1/6 | 534.890 / 528.333 |
| 1 | interleaved_arrival_all_attn/interleaved_all_attn | 0.078 | [-0.242,0.476] | 0.999999 | 1.013828 | 1/0 | 538.595 / 531.249 |
| 1 | interleaved_arrival_selective_attn/interleaved_arrival_all_attn | 0.089 | [-0.230,0.423] | 0.999689 | 1.009334 | 1/8 | 536.144 / 531.186 |
| 1 | interleaved_arrival_selective_attn/interleaved_all_attn | 0.167 | [-0.052,0.379] | 1.002699 | 1.007109 | 1/0 | 538.595 / 534.794 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_all_attn_mlp | -0.059 | [-0.379,0.241] | 1.002083 | 1.006919 | 1/8 | 515.908 / 512.363 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_all_attn_mlp | -0.258 | [-0.516,0.052] | 0.996034 | 1.006218 | 1/0 | 515.685 / 512.498 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_all_attn_mlp | -0.318 | [-0.567,-0.086] | 0.997660 | 1.002213 | 1/8 | 515.908 / 514.769 |

### 同策略范围比较

| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | interleaved_all_attn_mlp/interleaved_all_mlp | 3.550 | [3.259,3.838] | 1.037498 | 1.043268 | 1/0 | 535.116 / 512.922 |
| 1 | interleaved_all_attn_mlp/interleaved_all_attn | 3.620 | [3.369,3.961] | 1.037173 | 1.050053 | 1/0 | 538.595 / 512.922 |
| 1 | interleaved_all_mlp/interleaved_all_attn | 0.072 | [-0.140,0.311] | 0.999612 | 1.006503 | 1/0 | 538.595 / 535.116 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_mlp | 3.793 | [3.283,4.320] | 1.040348 | 1.055322 | 1/8 | 540.709 / 512.363 |
| 1 | interleaved_arrival_all_attn_mlp/interleaved_arrival_all_attn | 3.487 | [3.186,3.817] | 1.035629 | 1.046413 | 1/8 | 536.144 / 512.363 |
| 1 | interleaved_arrival_all_mlp/interleaved_arrival_all_attn | -0.318 | [-0.730,0.042] | 0.995916 | 1.003595 | 1/1 | 535.005 / 533.088 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_mlp | 3.671 | [3.083,4.280] | 1.038840 | 1.057817 | 1/0 | 542.129 / 512.498 |
| 1 | interleaved_arrival_selective_attn_mlp/interleaved_arrival_selective_attn | 3.151 | [2.882,3.468] | 1.031759 | 1.043503 | 1/0 | 534.794 / 512.498 |
| 1 | interleaved_arrival_selective_mlp/interleaved_arrival_selective_attn | -0.539 | [-0.954,-0.069] | 0.991875 | 1.010129 | 1/6 | 533.685 / 528.333 |

最好值仅为单块观测；负值和跨零区间保留。范围差异不等于局部算子耗时，也不能直接相加。
筛选轮次1：剩余9块；触发与排除：`[]`。
