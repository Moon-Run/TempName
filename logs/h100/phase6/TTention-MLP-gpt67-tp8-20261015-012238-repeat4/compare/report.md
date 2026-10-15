# H100 GPT6.7B TP8 Attention/MLP四组作用范围对比

全样本为主；同作用范围配对。每窗口取8rank最大elapsed/20，8192 token/step。
双模块相对Attention-only是扩展整套实现的净变化，不是单独增加量化的纯算法收益。
最好加速比是最佳单块观测；95%区间按完整配对块bootstrap 10000次，seed20261009。无4%验收门槛。

## 全样本

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | native_taco_attn_mlp | 1050.362 | 7799.33 |
| 1 | native_taco_attn | 822.816 | 9956.05 |
| 1 | interleaved_full_attn | 563.982 | 14525.31 |
| 1 | interleaved_full_attn_mlp | 546.801 | 14981.69 |
| 2 | native_taco_attn_mlp | 1061.608 | 7717.78 |
| 2 | native_taco_attn | 826.003 | 9917.83 |
| 2 | interleaved_full_attn | 567.232 | 14442.54 |
| 2 | interleaved_full_attn_mlp | 549.437 | 14909.89 |
| combined | native_taco_attn_mlp | 1051.377 | 7791.75 |
| combined | native_taco_attn | 822.816 | 9956.05 |
| combined | interleaved_full_attn | 564.240 | 14518.66 |
| combined | interleaved_full_attn_mlp | 547.520 | 14962.03 |

| 轮次 | 候选 | 同范围基线 | 配对中位加速比 |
| --- | --- | --- | ---: |
| 1 | interleaved_full_attn | native_taco_attn | 1.459082 |
| 1 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.915402 |
| 2 | interleaved_full_attn | native_taco_attn | 1.456211 |
| 2 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.929710 |
| combined | interleaved_full_attn | native_taco_attn | 1.457737 |
| combined | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.924561 |

| 轮次 | 候选 | 最好加速比 | 所在轮/block(0起) | 基线ms | 候选ms |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | interleaved_full_attn | 1.466574 | 1/3 | 821.910 | 560.429 |
| 1 | interleaved_full_attn_mlp | 1.927383 | 1/1 | 1054.305 | 547.014 |
| 2 | interleaved_full_attn | 1.462229 | 2/1 | 821.686 | 561.940 |
| 2 | interleaved_full_attn_mlp | 1.961068 | 2/2 | 1080.252 | 550.849 |
| combined | interleaved_full_attn | 1.466574 | 1/3 | 821.910 | 560.429 |
| combined | interleaved_full_attn_mlp | 1.961068 | 2/2 | 1080.252 | 550.849 |

| 轮次 | 比较 | 配对耗时下降% | 95%区间 |
| --- | --- | ---: | --- |
| 1 | interleaved_full_attn/native_taco_attn | 31.458 | [31.205, 31.704] |
| 1 | interleaved_full_attn_mlp/native_taco_attn_mlp | 47.723 | [47.327, 48.116] |
| 1 | interleaved_full_attn_mlp/interleaved_full_attn | 2.867 | [2.360, 3.371] |
| 1 | native_taco_attn_mlp/native_taco_attn | -27.354 | [-28.191, -26.482] |
| 2 | interleaved_full_attn/native_taco_attn | 27.658 | [19.497, 31.519] |
| 2 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.189 | [47.634, 48.749] |
| 2 | interleaved_full_attn_mlp/interleaved_full_attn | 12.007 | [2.720, 27.535] |
| 2 | native_taco_attn_mlp/native_taco_attn | -22.861 | [-29.338, -12.803] |
| combined | interleaved_full_attn/native_taco_attn | 29.584 | [25.666, 31.562] |
| combined | interleaved_full_attn_mlp/native_taco_attn_mlp | 47.956 | [47.573, 48.342] |
| combined | interleaved_full_attn_mlp/interleaved_full_attn | 7.550 | [2.683, 16.182] |
| combined | native_taco_attn_mlp/native_taco_attn | -25.088 | [-28.361, -19.827] |
## 慢窗口敏感性附表

| 轮次 | 策略 | 中位ms/step | tokens/s |
| --- | --- | ---: | ---: |
| 1 | native_taco_attn_mlp | 1050.362 | 7799.33 |
| 1 | native_taco_attn | 822.816 | 9956.05 |
| 1 | interleaved_full_attn | 563.982 | 14525.31 |
| 1 | interleaved_full_attn_mlp | 546.801 | 14981.69 |
| 2 | native_taco_attn_mlp | 1048.450 | 7813.44 |
| 2 | native_taco_attn | 822.402 | 9961.06 |
| 2 | interleaved_full_attn | 563.949 | 14526.14 |
| 2 | interleaved_full_attn_mlp | 548.026 | 14948.20 |

| 轮次 | 候选 | 同范围基线 | 配对中位加速比 |
| --- | --- | --- | ---: |
| 1 | interleaved_full_attn | native_taco_attn | 1.459082 |
| 1 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.915402 |
| 2 | interleaved_full_attn | native_taco_attn | 1.458292 |
| 2 | interleaved_full_attn_mlp | native_taco_attn_mlp | 1.921784 |

| 轮次 | 候选 | 最好加速比 | 所在轮/block(0起) | 基线ms | 候选ms |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | interleaved_full_attn | 1.466574 | 1/3 | 821.910 | 560.429 |
| 1 | interleaved_full_attn_mlp | 1.927383 | 1/1 | 1054.305 | 547.014 |
| 2 | interleaved_full_attn | 1.462229 | 2/1 | 821.686 | 561.940 |
| 2 | interleaved_full_attn_mlp | 1.961068 | 2/2 | 1080.252 | 550.849 |

| 轮次 | 比较 | 配对耗时下降% | 95%区间 |
| --- | --- | ---: | --- |
| 1 | interleaved_full_attn/native_taco_attn | 31.458 | [31.205, 31.704] |
| 1 | interleaved_full_attn_mlp/native_taco_attn_mlp | 47.723 | [47.327, 48.116] |
| 1 | interleaved_full_attn_mlp/interleaved_full_attn | 2.867 | [2.360, 3.371] |
| 1 | native_taco_attn_mlp/native_taco_attn | -27.354 | [-28.191, -26.482] |
| 2 | interleaved_full_attn/native_taco_attn | 31.423 | [31.230, 31.611] |
| 2 | interleaved_full_attn_mlp/native_taco_attn_mlp | 48.121 | [47.379, 49.007] |
| 2 | interleaved_full_attn_mlp/interleaved_full_attn | 3.062 | [2.476, 3.447] |
| 2 | native_taco_attn_mlp/native_taco_attn | -28.140 | [-30.213, -26.746] |

正确性及完整性通过；性能收益按区间单独判断，负值和跨零区间均保留。
筛选轮次 1：剩余4块；剔除及触发策略：`[]`。
筛选轮次 2：剩余3块；剔除及触发策略：`[{'round': 2, 'block': 3, 'triggers': {'native_taco_attn': {'ms': 992.4448177451268, 'threshold_ms': 908.6034318991007}, 'interleaved_full_attn': {'ms': 842.8238774649799, 'threshold_ms': 623.9555533102248}}}]`。
