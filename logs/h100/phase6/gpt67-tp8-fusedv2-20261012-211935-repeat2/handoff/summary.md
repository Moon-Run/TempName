# GPT 6.7B TP8 aligned fused v2 result

Session: `gpt67-tp8-fusedv2-20261012-211935-repeat2`; implementation: `a800-fused-v2-cta-v1`.

## remote_arrival_selective

| Round | ms/step | vs Megatron+quant % [95% CI] | vs fused v2 % [95% CI] | native overhead % | Flux overhead % |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 538.186 | +27.432 [+10.835, +34.684] | +5.315 [+1.383, +12.051] | +3.253 | +9.683 |
| 2 | 537.781 | +34.736 [+34.496, +35.003] | +2.387 [+2.072, +2.652] | -6.989 | -1.246 |

## interleaved_arrival_selective

| Round | ms/step | vs Megatron+quant % [95% CI] | vs fused v2 % [95% CI] | native overhead % | Flux overhead % |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 537.689 | +34.568 [+34.334, +34.836] | +14.625 [+1.536, +35.575] | -6.900 | -1.103 |
| 2 | 538.288 | +34.632 [+34.342, +34.973] | +2.232 [+1.843, +2.601] | -6.841 | -1.090 |

## Absolute timing

| Policy | Round 1 | Round 2 |
| --- | ---: | ---: |
| native | 576.683 | 574.857 |
| original | 544.092 | 543.966 |
| native_taco | 821.867 | 824.445 |
| taco_fused | 547.839 | 550.921 |
| remote_arrival_selective | 538.186 | 537.781 |
| interleaved_arrival_selective | 537.689 | 538.288 |

Accepted: `{'remote_arrival_selective': False, 'interleaved_arrival_selective': False}`; 72 windows / 576 rank records / 1440 timed steps.
