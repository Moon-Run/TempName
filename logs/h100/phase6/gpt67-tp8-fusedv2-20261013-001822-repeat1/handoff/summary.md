# GPT 6.7B TP8 aligned fused v2 result

Session: `gpt67-tp8-fusedv2-20261013-001822-repeat1`; implementation: `a800-fused-v2-cta-v1`.

## remote_arrival_selective

| Round | ms/step | vs Megatron+quant % [95% CI] | vs fused v2 % [95% CI] | native overhead % | Flux overhead % |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 536.234 | +34.853 [+34.592, +35.099] | +8.874 [+1.641, +21.398] | -6.889 | -13.023 |
| 2 | 537.758 | +34.955 [+34.458, +35.615] | +1.676 [+1.222, +2.205] | -6.727 | -0.995 |

## interleaved_arrival_selective

| Round | ms/step | vs Megatron+quant % [95% CI] | vs fused v2 % [95% CI] | native overhead % | Flux overhead % |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 537.680 | +34.884 [+34.539, +35.274] | +8.918 [+1.757, +21.224] | -6.934 | -13.065 |
| 2 | 536.422 | +34.876 [+34.441, +35.284] | +1.556 [+1.007, +1.989] | -6.614 | -0.875 |

## Absolute timing

| Policy | Round 1 | Round 2 |
| --- | ---: | ---: |
| native | 575.820 | 574.514 |
| original | 543.516 | 539.808 |
| native_taco | 825.594 | 823.403 |
| taco_fused | 548.523 | 545.486 |
| remote_arrival_selective | 536.234 | 537.758 |
| interleaved_arrival_selective | 537.680 | 536.422 |

Accepted: `{'remote_arrival_selective': False, 'interleaved_arrival_selective': False}`; 72 windows / 576 rank records / 1440 timed steps.
