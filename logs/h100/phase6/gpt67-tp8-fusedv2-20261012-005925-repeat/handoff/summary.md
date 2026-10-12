# GPT 6.7B TP8 aligned fused v2 result

Session: `gpt67-tp8-fusedv2-20261012-005925-repeat`; implementation: `a800-fused-v2-cta-v1`.

## remote_arrival_selective

| Round | ms/step | vs Megatron+quant % [95% CI] | vs fused v2 % [95% CI] | native overhead % | Flux overhead % |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 538.959 | +34.928 [+34.246, +35.596] | +5.750 [+1.939, +9.856] | -7.259 | -12.138 |
| 2 | 546.383 | +14.403 [-20.268, +35.006] | -14.421 [-73.357, +21.409] | +11.478 | +27.101 |

## interleaved_arrival_selective

| Round | ms/step | vs Megatron+quant % [95% CI] | vs fused v2 % [95% CI] | native overhead % | Flux overhead % |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 542.186 | +34.667 [+34.313, +35.111] | +5.373 [+1.555, +9.133] | -6.888 | -11.787 |
| 2 | 541.868 | +34.394 [+33.365, +35.281] | +12.301 [+0.439, +23.280] | -14.556 | -2.582 |

## Absolute timing

| Policy | Round 1 | Round 2 |
| --- | ---: | ---: |
| native | 580.558 | 579.234 |
| original | 545.284 | 554.080 |
| native_taco | 827.683 | 828.409 |
| taco_fused | 553.811 | 553.222 |
| remote_arrival_selective | 538.959 | 546.383 |
| interleaved_arrival_selective | 542.186 | 541.868 |

Accepted: `{'remote_arrival_selective': False, 'interleaved_arrival_selective': True}`; 72 windows / 576 rank records / 1440 timed steps.
