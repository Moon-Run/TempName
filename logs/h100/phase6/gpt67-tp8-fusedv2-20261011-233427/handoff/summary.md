# GPT 6.7B TP8 aligned fused v2 result

Session: `gpt67-tp8-fusedv2-20261011-233427`; implementation: `a800-fused-v2-cta-v1`.

## remote_arrival_selective

| Round | ms/step | vs Megatron+quant % [95% CI] | vs fused v2 % [95% CI] | native overhead % | Flux overhead % |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 541.272 | +34.543 [+34.292, +34.787] | +2.938 [+0.832, +5.925] | -7.066 | -1.480 |
| 2 | 541.223 | +33.196 [+29.649, +35.139] | +0.675 [-1.689, +2.207] | -4.439 | +1.439 |

## interleaved_arrival_selective

| Round | ms/step | vs Megatron+quant % [95% CI] | vs fused v2 % [95% CI] | native overhead % | Flux overhead % |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 545.156 | +26.674 [+8.572, +34.546] | -8.731 [-36.910, +5.191] | +4.106 | +10.364 |
| 2 | 540.493 | +34.394 [+33.661, +35.058] | +2.458 [+0.370, +5.421] | -6.154 | -0.381 |

## Absolute timing

| Policy | Round 1 | Round 2 |
| --- | ---: | ---: |
| native | 583.145 | 577.261 |
| original | 548.262 | 544.148 |
| native_taco | 829.511 | 825.354 |
| taco_fused | 553.170 | 549.213 |
| remote_arrival_selective | 541.272 | 541.223 |
| interleaved_arrival_selective | 545.156 | 540.493 |

Accepted: `{'remote_arrival_selective': False, 'interleaved_arrival_selective': False}`; 72 windows / 576 rank records / 1440 timed steps.
