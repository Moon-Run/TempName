# GPT 6.7B TP8 aligned fused v2 result

Session: `gpt67-tp8-fusedv2-20261012-211935-repeat1`; implementation: `a800-fused-v2-cta-v1`.

## remote_arrival_selective

| Round | ms/step | vs Megatron+quant % [95% CI] | vs fused v2 % [95% CI] | native overhead % | Flux overhead % |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 536.570 | +32.886 [+28.800, +35.004] | +10.293 [-7.234, +31.316] | -14.876 | +1.898 |
| 2 | 538.877 | +34.195 [+33.631, +34.697] | +6.148 [+0.525, +15.177] | -5.946 | -0.315 |

## interleaved_arrival_selective

| Round | ms/step | vs Megatron+quant % [95% CI] | vs fused v2 % [95% CI] | native overhead % | Flux overhead % |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 539.979 | +24.370 [+0.076, +34.745] | -1.091 [-49.759, +30.847] | -4.074 | +14.829 |
| 2 | 538.331 | +34.604 [+34.225, +35.098] | +6.731 [+1.249, +16.203] | -6.530 | -0.934 |

## Absolute timing

| Policy | Round 1 | Round 2 |
| --- | ---: | ---: |
| native | 578.916 | 575.296 |
| original | 542.286 | 542.045 |
| native_taco | 823.840 | 823.649 |
| taco_fused | 549.161 | 547.494 |
| remote_arrival_selective | 536.570 | 538.877 |
| interleaved_arrival_selective | 539.979 | 538.331 |

Accepted: `{'remote_arrival_selective': False, 'interleaved_arrival_selective': False}`; 72 windows / 576 rank records / 1440 timed steps.
