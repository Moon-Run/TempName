# H100 GPT 6.7B TP8 完整 optimizer step 消融

保留全样本；每窗口 10+20 步，取全部 rank 的最大 elapsed。95% 区间按完整配对块 bootstrap。
G5 各 seed 单独报告，seed-mean 在每块内取几何平均后配对；没有单项 4% 门槛。

| 基础顺序 | 轮次 | 比较 | 耗时下降 % | 95% 区间 | 结论 |
| --- | --- | --- | ---: | --- | --- |
| remote_first | 1 | G0/native | 4.379 | [1.539, 6.641] | 确认正增益 |
| remote_first | 1 | G1/native | 4.294 | [-1.796, 9.167] | 尚未确认正增益 |
| remote_first | 1 | G2/native | 4.498 | [-1.473, 9.256] | 尚未确认正增益 |
| remote_first | 1 | G3/native | 5.416 | [2.856, 7.252] | 确认正增益 |
| remote_first | 1 | G4/native | 7.110 | [3.798, 10.599] | 确认正增益 |
| remote_first | 1 | G5-s17/native | 9.439 | [6.675, 12.867] | 确认正增益 |
| remote_first | 1 | G5-s29/native | 4.437 | [-1.772, 9.431] | 尚未确认正增益 |
| remote_first | 1 | G5-s43/native | 6.166 | [3.118, 8.948] | 确认正增益 |
| remote_first | 1 | G6/native | 6.463 | [-0.175, 11.416] | 尚未确认正增益 |
| remote_first | 1 | G5-seed-mean/native | 6.704 | [4.664, 8.509] | 确认正增益 |
| remote_first | 1 | G1/G0 | -0.088 | [-7.790, 6.730] | 尚未确认正增益 |
| remote_first | 1 | G2/G1 | 0.213 | [-0.622, 0.984] | 尚未确认正增益 |
| remote_first | 1 | G4/G3 | 1.791 | [-3.481, 7.868] | 尚未确认正增益 |
| remote_first | 1 | G3/G1 | 1.172 | [-6.317, 8.421] | 尚未确认正增益 |
| remote_first | 1 | G4/G2 | 2.735 | [-0.067, 7.614] | 尚未确认正增益 |
| remote_first | 1 | G4/G6 | 0.691 | [-6.009, 7.839] | 尚未确认正增益 |
| remote_first | 1 | G4/G5-s17 | -2.572 | [-7.751, 0.198] | 尚未确认正增益 |
| remote_first | 1 | G4/G5-s29 | 2.797 | [-5.238, 10.186] | 尚未确认正增益 |
| remote_first | 1 | G4/G5-s43 | 1.006 | [0.324, 1.903] | 确认正增益 |
| remote_first | 1 | G4/G5-seed-mean | 0.435 | [-3.968, 3.949] | 尚未确认正增益 |
| remote_first | 1 | arrival_quantization_interaction | 1.582 | [-3.682, 7.641] | 尚未确认正增益 |
| remote_first | 2 | G0/native | 6.635 | [6.040, 7.198] | 确认正增益 |
| remote_first | 2 | G1/native | 5.989 | [5.015, 6.777] | 确认正增益 |
| remote_first | 2 | G2/native | 6.699 | [5.889, 7.355] | 确认正增益 |
| remote_first | 2 | G3/native | 6.418 | [6.007, 6.789] | 确认正增益 |
| remote_first | 2 | G4/native | 6.427 | [5.550, 7.174] | 确认正增益 |
| remote_first | 2 | G5-s17/native | 6.385 | [5.833, 6.926] | 确认正增益 |
| remote_first | 2 | G5-s29/native | 6.133 | [5.742, 6.503] | 确认正增益 |
| remote_first | 2 | G5-s43/native | 6.214 | [5.568, 6.798] | 确认正增益 |
| remote_first | 2 | G6/native | 6.659 | [6.274, 7.048] | 确认正增益 |
| remote_first | 2 | G5-seed-mean/native | 6.244 | [5.836, 6.637] | 确认正增益 |
| remote_first | 2 | G1/G0 | -0.692 | [-1.786, 0.285] | 尚未确认正增益 |
| remote_first | 2 | G2/G1 | 0.756 | [-0.481, 1.983] | 尚未确认正增益 |
| remote_first | 2 | G4/G3 | 0.009 | [-0.942, 0.724] | 尚未确认正增益 |
| remote_first | 2 | G3/G1 | 0.457 | [-0.404, 1.499] | 尚未确认正增益 |
| remote_first | 2 | G4/G2 | -0.292 | [-1.202, 0.612] | 尚未确认正增益 |
| remote_first | 2 | G4/G6 | -0.249 | [-1.213, 0.606] | 尚未确认正增益 |
| remote_first | 2 | G4/G5-s17 | 0.045 | [-0.913, 0.973] | 尚未确认正增益 |
| remote_first | 2 | G4/G5-s29 | 0.312 | [-0.653, 1.195] | 尚未确认正增益 |
| remote_first | 2 | G4/G5-s43 | 0.226 | [-0.801, 1.279] | 尚未确认正增益 |
| remote_first | 2 | G4/G5-seed-mean | 0.194 | [-0.742, 1.041] | 尚未确认正增益 |
| remote_first | 2 | arrival_quantization_interaction | -0.753 | [-1.824, 0.305] | 尚未确认正增益 |
| remote_first | combined | G0/native | 5.513 | [3.994, 6.699] | 确认正增益 |
| remote_first | combined | G1/native | 5.145 | [2.096, 7.655] | 确认正增益 |
| remote_first | combined | G2/native | 5.605 | [2.614, 8.061] | 确认正增益 |
| remote_first | combined | G3/native | 5.918 | [4.585, 6.856] | 确认正增益 |
| remote_first | combined | G4/native | 6.769 | [5.109, 8.535] | 确认正增益 |
| remote_first | combined | G5-s17/native | 7.924 | [6.449, 9.882] | 确认正增益 |
| remote_first | combined | G5-s29/native | 5.289 | [2.156, 7.809] | 确认正增益 |
| remote_first | combined | G5-s43/native | 6.190 | [4.654, 7.592] | 确认正增益 |
| remote_first | combined | G6/native | 6.561 | [3.228, 9.096] | 确认正增益 |
| remote_first | combined | G5-seed-mean/native | 6.474 | [5.395, 7.427] | 确认正增益 |
| remote_first | combined | G1/G0 | -0.390 | [-4.141, 3.163] | 尚未确认正增益 |
| remote_first | combined | G2/G1 | 0.485 | [-0.270, 1.239] | 尚未确认正增益 |
| remote_first | combined | G4/G3 | 0.904 | [-1.678, 4.099] | 尚未确认正增益 |
| remote_first | combined | G3/G1 | 0.815 | [-2.934, 4.495] | 尚未确认正增益 |
| remote_first | combined | G4/G2 | 1.233 | [-0.393, 3.963] | 尚未确认正增益 |
| remote_first | combined | G4/G6 | 0.222 | [-3.036, 3.923] | 尚未确认正增益 |
| remote_first | combined | G4/G5-s17 | -1.255 | [-3.947, 0.349] | 尚未确认正增益 |
| remote_first | combined | G4/G5-s29 | 1.562 | [-2.550, 5.682] | 尚未确认正增益 |
| remote_first | combined | G4/G5-s43 | 0.617 | [-0.022, 1.307] | 尚未确认正增益 |
| remote_first | combined | G4/G5-seed-mean | 0.315 | [-1.941, 2.215] | 尚未确认正增益 |
| remote_first | combined | arrival_quantization_interaction | 0.421 | [-2.218, 3.592] | 尚未确认正增益 |
| interleaved | 1 | G0/native | 6.852 | [3.446, 10.648] | 确认正增益 |
| interleaved | 1 | G1/native | 9.299 | [6.553, 12.859] | 确认正增益 |
| interleaved | 1 | G2/native | 7.002 | [3.674, 10.701] | 确认正增益 |
| interleaved | 1 | G3/native | 6.974 | [3.558, 10.494] | 确认正增益 |
| interleaved | 1 | G4/native | 6.966 | [3.537, 10.681] | 确认正增益 |
| interleaved | 1 | G5-s17/native | 6.679 | [3.370, 10.168] | 确认正增益 |
| interleaved | 1 | G5-s29/native | 9.447 | [6.742, 13.114] | 确认正增益 |
| interleaved | 1 | G5-s43/native | 7.052 | [3.505, 10.693] | 确认正增益 |
| interleaved | 1 | G6/native | 6.872 | [3.069, 10.704] | 确认正增益 |
| interleaved | 1 | G5-seed-mean/native | 7.734 | [6.610, 9.149] | 确认正增益 |
| interleaved | 1 | G1/G0 | 2.627 | [-0.320, 7.613] | 尚未确认正增益 |
| interleaved | 1 | G2/G1 | -2.533 | [-7.669, 0.189] | 尚未确认正增益 |
| interleaved | 1 | G4/G3 | -0.008 | [-0.348, 0.349] | 尚未确认正增益 |
| interleaved | 1 | G3/G1 | -2.564 | [-8.158, 0.258] | 尚未确认正增益 |
| interleaved | 1 | G4/G2 | -0.038 | [-8.129, 7.340] | 尚未确认正增益 |
| interleaved | 1 | G4/G6 | 0.101 | [-8.106, 7.914] | 尚未确认正增益 |
| interleaved | 1 | G4/G5-s17 | 0.308 | [-7.464, 7.666] | 尚未确认正增益 |
| interleaved | 1 | G4/G5-s29 | -2.740 | [-8.281, 0.169] | 尚未确认正增益 |
| interleaved | 1 | G4/G5-s43 | -0.092 | [-0.735, 0.388] | 尚未确认正增益 |
| interleaved | 1 | G4/G5-seed-mean | -0.832 | [-5.153, 2.574] | 尚未确认正增益 |
| interleaved | 1 | arrival_quantization_interaction | 2.463 | [-0.426, 7.406] | 尚未确认正增益 |
| interleaved | 2 | G0/native | 6.384 | [5.989, 6.824] | 确认正增益 |
| interleaved | 2 | G1/native | 6.636 | [6.337, 6.922] | 确认正增益 |
| interleaved | 2 | G2/native | 6.668 | [6.314, 7.008] | 确认正增益 |
| interleaved | 2 | G3/native | 6.545 | [6.216, 6.822] | 确认正增益 |
| interleaved | 2 | G4/native | 6.227 | [5.685, 6.737] | 确认正增益 |
| interleaved | 2 | G5-s17/native | 6.099 | [5.821, 6.396] | 确认正增益 |
| interleaved | 2 | G5-s29/native | 6.477 | [6.054, 6.882] | 确认正增益 |
| interleaved | 2 | G5-s43/native | 6.743 | [6.262, 7.218] | 确认正增益 |
| interleaved | 2 | G6/native | 6.661 | [6.445, 6.903] | 确认正增益 |
| interleaved | 2 | G5-seed-mean/native | 6.440 | [6.113, 6.735] | 确认正增益 |
| interleaved | 2 | G1/G0 | 0.269 | [-0.363, 0.727] | 尚未确认正增益 |
| interleaved | 2 | G2/G1 | 0.034 | [-0.415, 0.525] | 尚未确认正增益 |
| interleaved | 2 | G4/G3 | -0.340 | [-0.652, -0.028] | 尚未确认正增益 |
| interleaved | 2 | G3/G1 | -0.098 | [-0.489, 0.318] | 尚未确认正增益 |
| interleaved | 2 | G4/G2 | -0.472 | [-0.930, -0.041] | 尚未确认正增益 |
| interleaved | 2 | G4/G6 | -0.464 | [-0.979, -0.001] | 尚未确认正增益 |
| interleaved | 2 | G4/G5-s17 | 0.136 | [-0.377, 0.709] | 尚未确认正增益 |
| interleaved | 2 | G4/G5-s29 | -0.267 | [-0.677, 0.148] | 尚未确认正增益 |
| interleaved | 2 | G4/G5-s43 | -0.553 | [-1.058, -0.131] | 尚未确认正增益 |
| interleaved | 2 | G4/G5-seed-mean | -0.227 | [-0.637, 0.148] | 尚未确认正增益 |
| interleaved | 2 | arrival_quantization_interaction | -0.374 | [-0.890, 0.137] | 尚未确认正增益 |
| interleaved | combined | G0/native | 6.619 | [4.969, 8.580] | 确认正增益 |
| interleaved | combined | G1/native | 7.977 | [6.548, 9.946] | 确认正增益 |
| interleaved | combined | G2/native | 6.835 | [5.187, 8.720] | 确认正增益 |
| interleaved | combined | G3/native | 6.760 | [5.090, 8.584] | 确认正增益 |
| interleaved | combined | G4/native | 6.598 | [4.943, 8.541] | 确认正增益 |
| interleaved | combined | G5-s17/native | 6.389 | [4.760, 8.152] | 确认正增益 |
| interleaved | combined | G5-s29/native | 7.974 | [6.535, 9.963] | 确认正增益 |
| interleaved | combined | G5-s43/native | 6.897 | [5.167, 8.753] | 确认正增益 |
| interleaved | combined | G6/native | 6.766 | [4.862, 8.712] | 确认正增益 |
| interleaved | combined | G5-seed-mean/native | 7.089 | [6.473, 7.909] | 确认正增益 |
| interleaved | combined | G1/G0 | 1.455 | [-0.117, 4.115] | 尚未确认正增益 |
| interleaved | combined | G2/G1 | -1.241 | [-3.895, 0.199] | 尚未确认正增益 |
| interleaved | combined | G4/G3 | -0.174 | [-0.409, 0.074] | 尚未确认正增益 |
| interleaved | combined | G3/G1 | -1.324 | [-4.126, 0.169] | 尚未确认正增益 |
| interleaved | combined | G4/G2 | -0.255 | [-4.182, 3.547] | 尚未确认正增益 |
| interleaved | combined | G4/G6 | -0.181 | [-4.159, 3.873] | 尚未确认正增益 |
| interleaved | combined | G4/G5-s17 | 0.222 | [-3.550, 3.966] | 尚未确认正增益 |
| interleaved | combined | G4/G5-s29 | -1.496 | [-4.272, 0.014] | 尚未确认正增益 |
| interleaved | combined | G4/G5-s43 | -0.322 | [-0.723, 0.030] | 尚未确认正增益 |
| interleaved | combined | G4/G5-seed-mean | -0.529 | [-2.645, 1.164] | 尚未确认正增益 |
| interleaved | combined | arrival_quantization_interaction | 1.055 | [-0.481, 3.755] | 尚未确认正增益 |

| 基础顺序 | 轮次 | 组别/seed | 中位 ms/step | 中位 tokens/s |
| --- | --- | --- | ---: | ---: |
| remote_first | 1 | native | 579.243 | 14142.63 |
| remote_first | 1 | G0 | 543.378 | 15076.28 |
| remote_first | 1 | G1 | 544.455 | 15046.24 |
| remote_first | 1 | G2 | 541.881 | 15117.72 |
| remote_first | 1 | G3 | 541.097 | 15139.61 |
| remote_first | 1 | G4 | 538.528 | 15211.85 |
| remote_first | 1 | G5-s17 | 537.511 | 15240.63 |
| remote_first | 1 | G5-s29 | 541.976 | 15115.06 |
| remote_first | 1 | G5-s43 | 541.222 | 15136.13 |
| remote_first | 1 | G6 | 540.735 | 15149.81 |
| remote_first | 1 | G5-seed-mean | 540.934 | 15144.19 |
| remote_first | 2 | native | 576.140 | 14218.76 |
| remote_first | 2 | G0 | 538.888 | 15201.66 |
| remote_first | 2 | G1 | 538.430 | 15214.61 |
| remote_first | 2 | G2 | 536.137 | 15279.72 |
| remote_first | 2 | G3 | 537.344 | 15245.35 |
| remote_first | 2 | G4 | 539.158 | 15194.18 |
| remote_first | 2 | G5-s17 | 539.825 | 15175.32 |
| remote_first | 2 | G5-s29 | 540.349 | 15160.59 |
| remote_first | 2 | G5-s43 | 540.246 | 15163.47 |
| remote_first | 2 | G6 | 538.198 | 15221.16 |
| remote_first | 2 | G5-seed-mean | 540.629 | 15152.73 |
| remote_first | combined | native | 576.736 | 14204.09 |
| remote_first | combined | G0 | 540.895 | 15145.26 |
| remote_first | combined | G1 | 540.535 | 15155.36 |
| remote_first | combined | G2 | 539.710 | 15178.54 |
| remote_first | combined | G3 | 540.559 | 15154.69 |
| remote_first | combined | G4 | 538.528 | 15211.85 |
| remote_first | combined | G5-s17 | 538.347 | 15216.95 |
| remote_first | combined | G5-s29 | 541.171 | 15137.54 |
| remote_first | combined | G5-s43 | 540.897 | 15145.22 |
| remote_first | combined | G6 | 538.640 | 15208.66 |
| remote_first | combined | G5-seed-mean | 540.892 | 15145.35 |
| interleaved | 1 | native | 574.961 | 14248.07 |
| interleaved | 1 | G0 | 536.130 | 15279.90 |
| interleaved | 1 | G1 | 534.349 | 15330.83 |
| interleaved | 1 | G2 | 535.260 | 15304.74 |
| interleaved | 1 | G3 | 534.449 | 15327.95 |
| interleaved | 1 | G4 | 535.482 | 15298.39 |
| interleaved | 1 | G5-s17 | 536.446 | 15270.90 |
| interleaved | 1 | G5-s29 | 535.009 | 15311.89 |
| interleaved | 1 | G5-s43 | 534.873 | 15315.80 |
| interleaved | 1 | G6 | 534.514 | 15326.07 |
| interleaved | 1 | G5-seed-mean | 535.364 | 15301.77 |
| interleaved | 2 | native | 573.389 | 14286.98 |
| interleaved | 2 | G0 | 538.528 | 15211.84 |
| interleaved | 2 | G1 | 535.735 | 15291.14 |
| interleaved | 2 | G2 | 536.010 | 15283.29 |
| interleaved | 2 | G3 | 536.184 | 15278.35 |
| interleaved | 2 | G4 | 536.750 | 15262.26 |
| interleaved | 2 | G5-s17 | 538.127 | 15223.17 |
| interleaved | 2 | G5-s29 | 537.201 | 15249.53 |
| interleaved | 2 | G5-s43 | 534.893 | 15315.22 |
| interleaved | 2 | G6 | 535.829 | 15288.46 |
| interleaved | 2 | G5-seed-mean | 537.350 | 15245.20 |
| interleaved | combined | native | 573.389 | 14286.98 |
| interleaved | combined | G0 | 537.207 | 15249.25 |
| interleaved | combined | G1 | 535.352 | 15302.10 |
| interleaved | combined | G2 | 535.842 | 15288.10 |
| interleaved | combined | G3 | 535.940 | 15285.31 |
| interleaved | combined | G4 | 535.973 | 15284.35 |
| interleaved | combined | G5-s17 | 538.072 | 15224.72 |
| interleaved | combined | G5-s29 | 535.435 | 15299.72 |
| interleaved | combined | G5-s43 | 534.893 | 15315.22 |
| interleaved | combined | G6 | 535.565 | 15295.98 |
| interleaved | combined | G5-seed-mean | 536.256 | 15276.29 |
