# H100 GPT6.7B TP8 Attention/MLP四组作用范围对比

全样本为主；同作用范围配对。每窗口取8rank最大elapsed/20，8192 token/step。
双模块相对Attention-only是扩展整套实现的净变化，不是单独增加量化的纯算法收益。
最好加速比是最佳单块观测；95%区间按完整配对块bootstrap 10000次，seed20261009。无4%验收门槛。

## 全样本

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | native_taco_attn_mlp | 1059.926 | 7728.89 |
| 1 | native_taco_attn | 825.740 | 9920.80 |
| 1 | interleaved_full_attn | 572.083 | 14319.64 |
| 1 | interleaved_full_attn_mlp | 553.379 | 14803.63 |
| 2 | native_taco_attn_mlp | 1056.232 | 7755.90 |
| 2 | native_taco_attn | 828.329 | 9889.94 |
| 2 | interleaved_full_attn | 569.658 | 14380.55 |
| 2 | interleaved_full_attn_mlp | 549.061 | 14920.01 |
| combined | native_taco_attn_mlp | 1057.789 | 7744.45 |
| combined | native_taco_attn | 825.740 | 9920.80 |
| combined | interleaved_full_attn | 570.388 | 14362.17 |
| combined | interleaved_full_attn_mlp | 549.559 | 14906.49 |

| 轮次 | 候选 | 同范围基线 | 配对中位加速比 |
| --- | --- | --- | ---: |
| 1 | interleaved_full_attn | native_taco_attn | 1.448277 |
| 1 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.915663 |
| 2 | interleaved_full_attn | native_taco_attn | 1.462789 |
| 2 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.923911 |
| combined | interleaved_full_attn | native_taco_attn | 1.451606 |
| combined | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.921319 |

| 轮次 | 候选 | 最好加速比 | 所在轮/block(0起) | 基线ms | 候选ms |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | interleaved_full_attn | 1.452270 | 1/0 | 823.767 | 567.227 |
| 1 | interleaved_full_attn_mlp | 1.931582 | 1/2 | 1067.763 | 552.792 |
| 2 | interleaved_full_attn | 1.468100 | 2/1 | 836.332 | 569.670 |
| 2 | interleaved_full_attn_mlp | 1.926745 | 2/2 | 1058.109 | 549.169 |
| combined | interleaved_full_attn | 1.468100 | 2/1 | 836.332 | 569.670 |
| combined | interleaved_full_attn_mlp | 1.931582 | 1/2 | 1067.763 | 552.792 |

| 轮次 | 比较 | 配对耗时下降% | 95%区间 |
| --- | --- | ---: | --- |
| 1 | interleaved_full_attn/native_taco_attn | 30.856 | [30.553, 31.111] |
| 1 | interleaved_full_attn_mlp/native_taco_attn_mlp | 47.861 | [47.671, 48.112] |
| 1 | interleaved_full_attn_mlp/interleaved_full_attn | 3.333 | [3.238, 3.438] |
| 1 | native_taco_attn_mlp/native_taco_attn | -28.193 | [-28.984, -27.406] |
| 2 | interleaved_full_attn/native_taco_attn | 31.444 | [30.903, 31.832] |
| 2 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.020 | [47.952, 48.087] |
| 2 | interleaved_full_attn_mlp/interleaved_full_attn | 3.457 | [2.913, 3.909] |
| 2 | native_taco_attn_mlp/native_taco_attn | -27.330 | [-28.047, -26.454] |
| combined | interleaved_full_attn/native_taco_attn | 31.151 | [30.795, 31.501] |
| combined | interleaved_full_attn_mlp/native_taco_attn_mlp | 47.940 | [47.807, 48.067] |
| combined | interleaved_full_attn_mlp/interleaved_full_attn | 3.395 | [3.135, 3.640] |
| combined | native_taco_attn_mlp/native_taco_attn | -27.760 | [-28.404, -27.114] |
## 慢窗口敏感性附表

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | native_taco_attn_mlp | 1059.926 | 7728.89 |
| 1 | native_taco_attn | 825.740 | 9920.80 |
| 1 | interleaved_full_attn | 572.083 | 14319.64 |
| 1 | interleaved_full_attn_mlp | 553.379 | 14803.63 |
| 2 | native_taco_attn_mlp | 1056.232 | 7755.90 |
| 2 | native_taco_attn | 828.329 | 9889.94 |
| 2 | interleaved_full_attn | 569.658 | 14380.55 |
| 2 | interleaved_full_attn_mlp | 549.061 | 14920.01 |

| 轮次 | 候选 | 同范围基线 | 配对中位加速比 |
| --- | --- | --- | ---: |
| 1 | interleaved_full_attn | native_taco_attn | 1.448277 |
| 1 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.915663 |
| 2 | interleaved_full_attn | native_taco_attn | 1.462789 |
| 2 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.923911 |

| 轮次 | 候选 | 最好加速比 | 所在轮/block(0起) | 基线ms | 候选ms |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | interleaved_full_attn | 1.452270 | 1/0 | 823.767 | 567.227 |
| 1 | interleaved_full_attn_mlp | 1.931582 | 1/2 | 1067.763 | 552.792 |
| 2 | interleaved_full_attn | 1.468100 | 2/1 | 836.332 | 569.670 |
| 2 | interleaved_full_attn_mlp | 1.926745 | 2/2 | 1058.109 | 549.169 |

| 轮次 | 比较 | 配对耗时下降% | 95%区间 |
| --- | --- | ---: | --- |
| 1 | interleaved_full_attn/native_taco_attn | 30.856 | [30.553, 31.111] |
| 1 | interleaved_full_attn_mlp/native_taco_attn_mlp | 47.861 | [47.671, 48.112] |
| 1 | interleaved_full_attn_mlp/interleaved_full_attn | 3.333 | [3.238, 3.438] |
| 1 | native_taco_attn_mlp/native_taco_attn | -28.193 | [-28.984, -27.406] |
| 2 | interleaved_full_attn/native_taco_attn | 31.444 | [30.903, 31.832] |
| 2 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.020 | [47.952, 48.087] |
| 2 | interleaved_full_attn_mlp/interleaved_full_attn | 3.457 | [2.913, 3.909] |
| 2 | native_taco_attn_mlp/native_taco_attn | -27.330 | [-28.047, -26.454] |

正确性及完整性通过；性能收益按区间单独判断，负值和跨零区间均保留。
筛选轮次 1：剩余4块；剔除及触发策略：`[]`。
筛选轮次 2：剩余4块；剔除及触发策略：`[]`。
