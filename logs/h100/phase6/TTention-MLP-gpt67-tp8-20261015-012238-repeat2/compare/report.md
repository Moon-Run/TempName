# H100 GPT6.7B TP8 Attention/MLP四组作用范围对比

全样本为主；同作用范围配对。每窗口取8rank最大elapsed/20，8192 token/step。
双模块相对Attention-only是扩展整套实现的净变化，不是单独增加量化的纯算法收益。
最好加速比是最佳单块观测；95%区间按完整配对块bootstrap 10000次，seed20261009。无4%验收门槛。

## 全样本

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | native_taco_attn_mlp | 1042.903 | 7855.20 |
| 1 | native_taco_attn | 816.824 | 10029.09 |
| 1 | interleaved_full_attn | 559.332 | 14646.10 |
| 1 | interleaved_full_attn_mlp | 541.307 | 15133.76 |
| 2 | native_taco_attn_mlp | 1053.032 | 7779.49 |
| 2 | native_taco_attn | 814.340 | 10059.70 |
| 2 | interleaved_full_attn | 557.113 | 14704.37 |
| 2 | interleaved_full_attn_mlp | 539.588 | 15182.43 |
| combined | native_taco_attn_mlp | 1049.316 | 7807.00 |
| combined | native_taco_attn | 815.882 | 10040.67 |
| combined | interleaved_full_attn | 558.195 | 14675.87 |
| combined | interleaved_full_attn_mlp | 541.307 | 15133.76 |

| 轮次 | 候选 | 同范围基线 | 配对中位加速比 |
| --- | --- | --- | ---: |
| 1 | interleaved_full_attn | native_taco_attn | 1.459625 |
| 1 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.923974 |
| 2 | interleaved_full_attn | native_taco_attn | 1.461940 |
| 2 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.944703 |
| combined | interleaved_full_attn | native_taco_attn | 1.460500 |
| combined | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.939061 |

| 轮次 | 候选 | 最好加速比 | 所在轮/block(0起) | 基线ms | 候选ms |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | interleaved_full_attn | 1.465503 | 1/2 | 818.054 | 558.207 |
| 1 | interleaved_full_attn_mlp | 1.948885 | 1/0 | 1056.136 | 541.918 |
| 2 | interleaved_full_attn | 1.475811 | 2/2 | 823.779 | 558.187 |
| 2 | interleaved_full_attn_mlp | 1.967611 | 2/1 | 1055.719 | 536.548 |
| combined | interleaved_full_attn | 1.475811 | 2/2 | 823.779 | 558.187 |
| combined | interleaved_full_attn_mlp | 1.967611 | 2/1 | 1055.719 | 536.548 |

| 轮次 | 比较 | 配对耗时下降% | 95%区间 |
| --- | --- | ---: | --- |
| 1 | interleaved_full_attn/native_taco_attn | 31.476 | [31.257, 31.681] |
| 1 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.134 | [47.867, 48.509] |
| 1 | interleaved_full_attn_mlp/interleaved_full_attn | 3.220 | [2.701, 3.657] |
| 1 | native_taco_attn_mlp/native_taco_attn | -27.865 | [-28.715, -27.056] |
| 2 | interleaved_full_attn/native_taco_attn | 31.720 | [31.477, 32.059] |
| 2 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.669 | [48.413, 49.013] |
| 2 | interleaved_full_attn_mlp/interleaved_full_attn | 2.923 | [2.003, 3.723] |
| 2 | native_taco_attn_mlp/native_taco_attn | -29.131 | [-30.448, -27.827] |
| combined | interleaved_full_attn/native_taco_attn | 31.598 | [31.405, 31.815] |
| combined | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.402 | [48.114, 48.698] |
| combined | interleaved_full_attn_mlp/interleaved_full_attn | 3.071 | [2.510, 3.538] |
| combined | native_taco_attn_mlp/native_taco_attn | -28.496 | [-29.463, -27.634] |
## 慢窗口敏感性附表

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | native_taco_attn_mlp | 1042.903 | 7855.20 |
| 1 | native_taco_attn | 816.824 | 10029.09 |
| 1 | interleaved_full_attn | 559.332 | 14646.10 |
| 1 | interleaved_full_attn_mlp | 541.307 | 15133.76 |
| 2 | native_taco_attn_mlp | 1053.032 | 7779.49 |
| 2 | native_taco_attn | 814.340 | 10059.70 |
| 2 | interleaved_full_attn | 557.113 | 14704.37 |
| 2 | interleaved_full_attn_mlp | 539.588 | 15182.43 |

| 轮次 | 候选 | 同范围基线 | 配对中位加速比 |
| --- | --- | --- | ---: |
| 1 | interleaved_full_attn | native_taco_attn | 1.459625 |
| 1 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.923974 |
| 2 | interleaved_full_attn | native_taco_attn | 1.461940 |
| 2 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.944703 |

| 轮次 | 候选 | 最好加速比 | 所在轮/block(0起) | 基线ms | 候选ms |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | interleaved_full_attn | 1.465503 | 1/2 | 818.054 | 558.207 |
| 1 | interleaved_full_attn_mlp | 1.948885 | 1/0 | 1056.136 | 541.918 |
| 2 | interleaved_full_attn | 1.475811 | 2/2 | 823.779 | 558.187 |
| 2 | interleaved_full_attn_mlp | 1.967611 | 2/1 | 1055.719 | 536.548 |

| 轮次 | 比较 | 配对耗时下降% | 95%区间 |
| --- | --- | ---: | --- |
| 1 | interleaved_full_attn/native_taco_attn | 31.476 | [31.257, 31.681] |
| 1 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.134 | [47.867, 48.509] |
| 1 | interleaved_full_attn_mlp/interleaved_full_attn | 3.220 | [2.701, 3.657] |
| 1 | native_taco_attn_mlp/native_taco_attn | -27.865 | [-28.715, -27.056] |
| 2 | interleaved_full_attn/native_taco_attn | 31.720 | [31.477, 32.059] |
| 2 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.669 | [48.413, 49.013] |
| 2 | interleaved_full_attn_mlp/interleaved_full_attn | 2.923 | [2.003, 3.723] |
| 2 | native_taco_attn_mlp/native_taco_attn | -29.131 | [-30.448, -27.827] |

正确性及完整性通过；性能收益按区间单独判断，负值和跨零区间均保留。
筛选轮次 1：剩余4块；剔除及触发策略：`[]`。
筛选轮次 2：剩余4块；剔除及触发策略：`[]`。
