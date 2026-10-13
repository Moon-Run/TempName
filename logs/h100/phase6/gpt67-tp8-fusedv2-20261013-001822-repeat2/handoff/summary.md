# GPT 6.7B TP8 aligned fused v2 result

Session: `gpt67-tp8-fusedv2-20261013-001822-repeat2`; implementation: `a800-fused-v2-cta-v1`.

## remote_arrival_selective

| Round | ms/step | vs Megatron+quant % [95% CI] | vs fused v2 % [95% CI] | native overhead % | Flux overhead % |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 539.750 | +36.313 [-15.055, +65.728] | +10.445 [-54.895, +46.653] | -11.289 | +6.245 |
| 2 | 538.590 | +12.147 [-25.110, +34.533] | -1.832 [-17.450, +8.716] | +4.032 | +5.522 |

## interleaved_arrival_selective

| Round | ms/step | vs Megatron+quant % [95% CI] | vs fused v2 % [95% CI] | native overhead % | Flux overhead % |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 536.086 | +34.071 [-29.093, +65.947] | +7.293 [-73.436, +47.071] | -8.167 | +9.985 |
| 2 | 536.662 | +34.566 [+34.265, +34.884] | +24.154 [+1.681, +49.362] | -22.515 | -21.405 |

## Absolute timing

| Policy | Round 1 | Round 2 |
| --- | ---: | ---: |
| native | 576.218 | 575.587 |
| original | 542.504 | 540.055 |
| native_taco | 824.155 | 818.670 |
| taco_fused | 551.103 | 547.133 |
| remote_arrival_selective | 539.750 | 538.590 |
| interleaved_arrival_selective | 536.086 | 536.662 |

Accepted: `{'remote_arrival_selective': False, 'interleaved_arrival_selective': False}`; 72 windows / 576 rank records / 1440 timed steps.
