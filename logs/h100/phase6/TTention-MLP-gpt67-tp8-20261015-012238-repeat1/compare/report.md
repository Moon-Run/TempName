# H100 GPT6.7B TP8 Attention/MLP四组作用范围对比

全样本为主；同作用范围配对。每窗口取8rank最大elapsed/20，8192 token/step。
双模块相对Attention-only是扩展整套实现的净变化，不是单独增加量化的纯算法收益。
最好加速比是最佳单块观测；95%区间按完整配对块bootstrap 10000次，seed20261009。无4%验收门槛。

## 全样本

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | native_taco_attn_mlp | 1049.767 | 7803.64 |
| 1 | native_taco_attn | 819.842 | 9992.29 |
| 1 | interleaved_full_attn | 559.068 | 14653.00 |
| 1 | interleaved_full_attn_mlp | 543.400 | 15075.50 |
| 2 | native_taco_attn_mlp | 1047.405 | 7821.27 |
| 2 | native_taco_attn | 818.547 | 10007.97 |
| 2 | interleaved_full_attn | 560.551 | 14614.35 |
| 2 | interleaved_full_attn_mlp | 541.356 | 15132.53 |
| combined | native_taco_attn_mlp | 1049.287 | 7807.20 |
| combined | native_taco_attn | 818.547 | 10007.97 |
| combined | interleaved_full_attn | 559.305 | 14646.76 |
| combined | interleaved_full_attn_mlp | 542.709 | 15094.65 |

| 轮次 | 候选 | 同范围基线 | 配对中位加速比 |
| --- | --- | --- | ---: |
| 1 | interleaved_full_attn | native_taco_attn | 1.466261 |
| 1 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.930455 |
| 2 | interleaved_full_attn | native_taco_attn | 1.459925 |
| 2 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.943454 |
| combined | interleaved_full_attn | native_taco_attn | 1.464163 |
| combined | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.936930 |

| 轮次 | 候选 | 最好加速比 | 所在轮/block(0起) | 基线ms | 候选ms |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | interleaved_full_attn | 1.469483 | 1/1 | 822.729 | 559.877 |
| 1 | interleaved_full_attn_mlp | 1.943701 | 1/3 | 1049.139 | 539.764 |
| 2 | interleaved_full_attn | 1.464773 | 2/0 | 818.417 | 558.733 |
| 2 | interleaved_full_attn_mlp | 1.945611 | 2/2 | 1056.719 | 543.130 |
| combined | interleaved_full_attn | 1.469483 | 1/1 | 822.729 | 559.877 |
| combined | interleaved_full_attn_mlp | 1.945611 | 2/2 | 1056.719 | 543.130 |

| 轮次 | 比较 | 配对耗时下降% | 95%区间 |
| --- | --- | ---: | --- |
| 1 | interleaved_full_attn/native_taco_attn | 31.803 | [31.686, 31.921] |
| 1 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.272 | [48.150, 48.454] |
| 1 | interleaved_full_attn_mlp/interleaved_full_attn | 2.858 | [2.211, 3.630] |
| 1 | native_taco_attn_mlp/native_taco_attn | -28.069 | [-28.570, -27.616] |
| 2 | interleaved_full_attn/native_taco_attn | 31.508 | [31.301, 31.714] |
| 2 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.413 | [48.113, 48.593] |
| 2 | interleaved_full_attn_mlp/interleaved_full_attn | 3.430 | [3.324, 3.525] |
| 2 | native_taco_attn_mlp/native_taco_attn | -28.215 | [-28.567, -27.817] |
| combined | interleaved_full_attn/native_taco_attn | 31.656 | [31.497, 31.800] |
| combined | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.342 | [48.178, 48.497] |
| combined | interleaved_full_attn_mlp/interleaved_full_attn | 3.145 | [2.718, 3.516] |
| combined | native_taco_attn_mlp/native_taco_attn | -28.142 | [-28.432, -27.850] |
## 慢窗口敏感性附表

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | native_taco_attn_mlp | 1049.767 | 7803.64 |
| 1 | native_taco_attn | 819.842 | 9992.29 |
| 1 | interleaved_full_attn | 559.068 | 14653.00 |
| 1 | interleaved_full_attn_mlp | 543.400 | 15075.50 |
| 2 | native_taco_attn_mlp | 1047.405 | 7821.27 |
| 2 | native_taco_attn | 818.547 | 10007.97 |
| 2 | interleaved_full_attn | 560.551 | 14614.35 |
| 2 | interleaved_full_attn_mlp | 541.356 | 15132.53 |

| 轮次 | 候选 | 同范围基线 | 配对中位加速比 |
| --- | --- | --- | ---: |
| 1 | interleaved_full_attn | native_taco_attn | 1.466261 |
| 1 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.930455 |
| 2 | interleaved_full_attn | native_taco_attn | 1.459925 |
| 2 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.943454 |

| 轮次 | 候选 | 最好加速比 | 所在轮/block(0起) | 基线ms | 候选ms |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | interleaved_full_attn | 1.469483 | 1/1 | 822.729 | 559.877 |
| 1 | interleaved_full_attn_mlp | 1.943701 | 1/3 | 1049.139 | 539.764 |
| 2 | interleaved_full_attn | 1.464773 | 2/0 | 818.417 | 558.733 |
| 2 | interleaved_full_attn_mlp | 1.945611 | 2/2 | 1056.719 | 543.130 |

| 轮次 | 比较 | 配对耗时下降% | 95%区间 |
| --- | --- | ---: | --- |
| 1 | interleaved_full_attn/native_taco_attn | 31.803 | [31.686, 31.921] |
| 1 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.272 | [48.150, 48.454] |
| 1 | interleaved_full_attn_mlp/interleaved_full_attn | 2.858 | [2.211, 3.630] |
| 1 | native_taco_attn_mlp/native_taco_attn | -28.069 | [-28.570, -27.616] |
| 2 | interleaved_full_attn/native_taco_attn | 31.508 | [31.301, 31.714] |
| 2 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.413 | [48.113, 48.593] |
| 2 | interleaved_full_attn_mlp/interleaved_full_attn | 3.430 | [3.324, 3.525] |
| 2 | native_taco_attn_mlp/native_taco_attn | -28.215 | [-28.567, -27.817] |

正确性及完整性通过；性能收益按区间单独判断，负值和跨零区间均保留。
筛选轮次 1：剩余4块；剔除及触发策略：`[]`。
筛选轮次 2：剩余4块；剔除及触发策略：`[]`。
