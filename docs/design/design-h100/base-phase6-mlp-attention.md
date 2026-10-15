# H100 GPT 6.7B TP8 Attention/MLP作用范围实测

更新：2026-10-15。测试a、b、c、d分别对应本批次repeat1、repeat2、repeat3、repeat4，每次两轮、32个正式窗口，合计128个正式窗口。

单节点8×H100，GPT 6.7B、32层、H4096/FFN16384、32 heads、S2048、micro batch1/global batch4、BF16、TP8。每窗口10步预热＋20步完整optimizer step，取八rank最大elapsed/20；表内ms/step为每轮4个配对块的中位数，使用全部样本。

Attention候选为 `interleaved_full_attn`，Attention＋MLP候选为 `interleaved_full_attn_mlp`；对应Megatron＋量化基线分别为 `native_taco_attn` 和 `native_taco_attn_mlp`。两个候选共享双shape库和Attention计划；Attention-only组的MLP保留原生Megatron BF16。

## 完整step耗时对照

| 测试 | 轮次 | 配对块数 | Attention＋MLP ms/step | Attention ms/step | Megatron＋量化 Attention ms/step | Megatron＋量化 Attention＋MLP ms/step |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| a | 1 | 4 | 543.400 | 559.068 | 819.842 | 1049.767 |
| a | 2 | 4 | 541.356 | 560.551 | 818.547 | 1047.405 |
| b | 1 | 4 | 541.307 | 559.332 | 816.824 | 1042.903 |
| b | 2 | 4 | 539.588 | 557.113 | 814.340 | 1053.032 |
| c | 1 | 4 | 553.379 | 572.083 | 825.740 | 1059.926 |
| c | 2 | 4 | 549.061 | 569.658 | 828.329 | 1056.232 |
| d | 1 | 4 | 546.801 | 563.982 | 822.816 | 1050.362 |
| d | 2 | 4 | 549.437 | 567.232 | 826.003 | 1061.608 |

## 配对耗时下降

平均耗时取表内各轮ms/step的算术平均。Attention四列取a–d两轮共8行；交替组合候选及其余三个参考列取 [base-phase6.md](base-phase6.md) 的a–h轮。

| Attention＋MLP ms/step | Attention ms/step | MLP ms/step | Megatron＋量化 Attention ms/step | Megatron＋量化 MLP ms/step | Megatron＋量化 Attention＋MLP ms/step | 原生 Megatron ms/step | 原生 Flux ms/step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 545.541 | 563.627 | 537.642 | 821.555 | 823.203 | 1052.654 | 575.986 | 542.284 |
| 0.000% | +3.209% | -1.469% | +33.597% | +33.729% | +48.175% | +5.286% | -0.601% |

上表第二行为Attention＋MLP相对各列的耗时下降百分比，按 `100×(1−545.541/该列平均耗时)` 计算；正值表示更快，负值表示更慢。

下降百分比采用 `100×(1−exp(mean(log(T候选/T参考))))`，95% CI按每轮4个完整配对块bootstrap 10000次、seed20261009计算。正值表示更快。两种作用范围分别对照各自Megatron＋量化基线；扩展MLP净下降以同轮Attention候选为参考。

| 测试 | 轮次 | Attention＋MLP ms/step | Attention ms/step | Attention 对同范围 Megatron＋量化下降% [95% CI] | Attention＋MLP 对同范围 Megatron＋量化下降% [95% CI] | 扩展MLP候选净下降% [95% CI] |
| --- | --- | ---: | ---: | --- | --- | --- |
| a | 1 | 543.400 | 559.068 | +31.803 [+31.686, +31.921] | +48.272 [+48.150, +48.454] | +2.858 [+2.211, +3.630] |
| a | 2 | 541.356 | 560.551 | +31.508 [+31.301, +31.714] | +48.413 [+48.113, +48.593] | +3.430 [+3.324, +3.525] |
| b | 1 | 541.307 | 559.332 | +31.476 [+31.257, +31.681] | +48.134 [+47.867, +48.509] | +3.220 [+2.701, +3.657] |
| b | 2 | 539.588 | 557.113 | +31.720 [+31.477, +32.059] | +48.669 [+48.413, +49.013] | +2.923 [+2.003, +3.723] |
| c | 1 | 553.379 | 572.083 | +30.856 [+30.553, +31.111] | +47.861 [+47.671, +48.112] | +3.333 [+3.238, +3.438] |
| c | 2 | 549.061 | 569.658 | +31.444 [+30.903, +31.832] | +48.020 [+47.952, +48.087] | +3.457 [+2.913, +3.909] |
| d | 1 | 546.801 | 563.982 | +31.458 [+31.205, +31.704] | +47.723 [+47.327, +48.116] | +2.867 [+2.360, +3.371] |
| d | 2 | 549.437 | 567.232 | +27.658 [+19.497, +31.519] | +48.189 [+47.634, +48.749] | +12.007 [+2.720, +27.535] |

测试d第2轮的block3包含慢窗口：Megatron＋量化Attention为992.445 ms/step、Attention候选为842.824 ms/step。上表保留该块；原报告的慢窗口敏感性附表另列筛选结果。

## 数据来源

- 测试a：[TTention-MLP-gpt67-tp8-20261015-012238-repeat1](../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat1/compare/report.json)，轮次1/2对应原报告round1/round2。
- 测试b：[TTention-MLP-gpt67-tp8-20261015-012238-repeat2](../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat2/compare/report.json)，轮次1/2对应原报告round1/round2。
- 测试c：[TTention-MLP-gpt67-tp8-20261015-012238-repeat3](../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat3/compare/report.json)，轮次1/2对应原报告round1/round2。
- 测试d：[TTention-MLP-gpt67-tp8-20261015-012238-repeat4](../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat4/compare/report.json)，轮次1/2对应原报告round1/round2。

第5次因启动时GPU繁忙停止，正式窗口为0；第6、7次尚未启动。
