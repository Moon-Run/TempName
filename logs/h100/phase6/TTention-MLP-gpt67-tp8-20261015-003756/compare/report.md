# H100 GPT6.7B TP8 Attention/MLP四组作用范围对比

全样本为主；同作用范围配对。每窗口取8rank最大elapsed/20，8192 token/step。
双模块相对Attention-only是扩展整套实现的净变化，不是单独增加量化的纯算法收益。
最好加速比是最佳单块观测；95%区间按完整配对块bootstrap 10000次，seed20261009。无4%验收门槛。

## 全样本

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | native_taco_attn_mlp | 1061.756 | 7715.52 |
| 1 | native_taco_attn | 826.196 | 9915.35 |
| 1 | interleaved_full_attn | 559.851 | 14632.47 |
| 1 | interleaved_full_attn_mlp | 543.598 | 15070.20 |
| 2 | native_taco_attn_mlp | 1065.992 | 7684.86 |
| 2 | native_taco_attn | 831.554 | 9851.53 |
| 2 | interleaved_full_attn | 567.274 | 14441.01 |
| 2 | interleaved_full_attn_mlp | 553.716 | 14794.64 |
| combined | native_taco_attn_mlp | 1063.720 | 7701.29 |
| combined | native_taco_attn | 828.275 | 9890.44 |
| combined | interleaved_full_attn | 561.435 | 14591.19 |
| combined | interleaved_full_attn_mlp | 546.609 | 14986.98 |

| 轮次 | 候选 | 同范围基线 | 配对中位加速比 |
| --- | --- | --- | ---: |
| 1 | interleaved_full_attn | native_taco_attn | 1.478627 |
| 1 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.953227 |
| 2 | interleaved_full_attn | native_taco_attn | 1.462149 |
| 2 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.926679 |
| combined | interleaved_full_attn | native_taco_attn | 1.471680 |
| combined | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.946235 |

| 轮次 | 候选 | 最好加速比 | 所在轮/block(0起) | 基线ms | 候选ms |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | interleaved_full_attn | 1.495283 | 1/2 | 827.553 | 553.442 |
| 1 | interleaved_full_attn_mlp | 2.165057 | 1/2 | 1181.781 | 545.843 |
| 2 | interleaved_full_attn | 1.484330 | 2/2 | 842.248 | 567.427 |
| 2 | interleaved_full_attn_mlp | 1.955709 | 2/2 | 1085.027 | 554.800 |
| combined | interleaved_full_attn | 1.495283 | 1/2 | 827.553 | 553.442 |
| combined | interleaved_full_attn_mlp | 2.165057 | 1/2 | 1181.781 | 545.843 |

| 轮次 | 比较 | 配对耗时下降% | 95%区间 |
| --- | --- | ---: | --- |
| 1 | interleaved_full_attn/native_taco_attn | 32.450 | [32.050, 32.885] |
| 1 | interleaved_full_attn_mlp/native_taco_attn_mlp | 50.030 | [48.563, 52.567] |
| 1 | interleaved_full_attn_mlp/interleaved_full_attn | 2.684 | [1.841, 3.383] |
| 1 | native_taco_attn_mlp/native_taco_attn | -31.552 | [-38.913, -27.661] |
| 2 | interleaved_full_attn/native_taco_attn | 31.806 | [31.435, 32.371] |
| 2 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.143 | [47.539, 48.739] |
| 2 | interleaved_full_attn_mlp/interleaved_full_attn | 2.368 | [1.420, 3.689] |
| 2 | native_taco_attn_mlp/native_taco_attn | -28.389 | [-28.757, -27.946] |
| combined | interleaved_full_attn/native_taco_attn | 32.129 | [31.742, 32.535] |
| combined | interleaved_full_attn_mlp/native_taco_attn_mlp | 49.095 | [48.089, 50.596] |
| combined | interleaved_full_attn_mlp/interleaved_full_attn | 2.526 | [1.820, 3.264] |
| combined | native_taco_attn_mlp/native_taco_attn | -29.961 | [-33.627, -27.957] |
## 慢窗口敏感性附表

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | native_taco_attn_mlp | 1061.229 | 7719.35 |
| 1 | native_taco_attn | 824.840 | 9931.63 |
| 1 | interleaved_full_attn | 560.185 | 14623.74 |
| 1 | interleaved_full_attn_mlp | 541.465 | 15129.32 |
| 2 | native_taco_attn_mlp | 1065.992 | 7684.86 |
| 2 | native_taco_attn | 831.554 | 9851.53 |
| 2 | interleaved_full_attn | 567.274 | 14441.01 |
| 2 | interleaved_full_attn_mlp | 553.716 | 14794.64 |

| 轮次 | 候选 | 同范围基线 | 配对中位加速比 |
| --- | --- | --- | ---: |
| 1 | interleaved_full_attn | native_taco_attn | 1.474199 |
| 1 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.946534 |
| 2 | interleaved_full_attn | native_taco_attn | 1.462149 |
| 2 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.926679 |

| 轮次 | 候选 | 最好加速比 | 所在轮/block(0起) | 基线ms | 候选ms |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | interleaved_full_attn | 1.483055 | 1/3 | 830.785 | 560.185 |
| 1 | interleaved_full_attn_mlp | 1.959921 | 1/0 | 1061.229 | 541.465 |
| 2 | interleaved_full_attn | 1.484330 | 2/2 | 842.248 | 567.427 |
| 2 | interleaved_full_attn_mlp | 1.955709 | 2/2 | 1085.027 | 554.800 |

| 轮次 | 比较 | 配对耗时下降% | 95%区间 |
| --- | --- | ---: | --- |
| 1 | interleaved_full_attn/native_taco_attn | 32.225 | [31.934, 32.572] |
| 1 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.701 | [48.499, 48.978] |
| 1 | interleaved_full_attn_mlp/interleaved_full_attn | 3.116 | [2.580, 3.535] |
| 1 | native_taco_attn_mlp/native_taco_attn | -28.002 | [-28.688, -27.456] |
| 2 | interleaved_full_attn/native_taco_attn | 31.806 | [31.435, 32.371] |
| 2 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.143 | [47.539, 48.739] |
| 2 | interleaved_full_attn_mlp/interleaved_full_attn | 2.368 | [1.420, 3.689] |
| 2 | native_taco_attn_mlp/native_taco_attn | -28.389 | [-28.757, -27.946] |

正确性及完整性通过；性能收益按区间单独判断，负值和跨零区间均保留。
筛选轮次 1：剩余3块；剔除及触发策略：`[{'round': 1, 'block': 2, 'triggers': {'native_taco_attn_mlp': {'ms': 1181.7810831125826, 'threshold_ms': 1167.9314102500211}}}]`。
筛选轮次 2：剩余4块；剔除及触发策略：`[]`。
