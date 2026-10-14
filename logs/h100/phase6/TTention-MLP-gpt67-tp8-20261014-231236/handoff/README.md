# H100 Attention/MLP scope experiment

Failure details and missing items are documented in [failure-analysis.md](failure-analysis.md). The aggregate profile check rejected a total of 256 same-named role events (128 CPU annotations plus 128 GPU annotations) against an expected count of 128. Actual module calls in the profile rank reports are four per module. All formal timing is absent: 0/32 windows, no official performance report or speedup/95% interval.

Session: `TTention-MLP-gpt67-tp8-20261014-231236`. Updated Beijing: 2026-10-14T23:30:35.424572+08:00.

Status: **failed**. run-all.log exited 1; see /workspace/TempName/logs/h100/phase6/TTention-MLP-gpt67-tp8-20261014-231236/run-all.log.

Maintenance source commit: `5b3c41be6d2536ecf8d4c32d824071a819dba452`. Initial environment, submodules, source hashes, worktree and patch are in this handoff directory; all exact commands are saved as command JSON. Measured compute source is frozen candidate v1 from ff38f5b1ff1c80cb831f58c64b95f42b0379879a.

GPT 6.7B: 32 layers, H4096/FFN16384, 32 heads, S2048, micro1/global4, BF16, TP8. Attention shape [2048,4096,4096], local K512; MLP shape [2048,4096,16384], local K2048. Both roles receive NEW real H100 BF16 calibration in this session, training passes 0/1, heldout pass2. No old MLP-only library, source-only build or synthetic plan is used.

Four policies: `['native_taco_attn_mlp', 'native_taco_attn', 'interleaved_full_attn', 'interleaved_full_attn_mlp']`. Attention-only groups keep native Megatron BF16 MLP. Both candidates share one newly built dual-shape library and exactly the same Attention plan. Four paired blocks per round, two rounds, 32 new windows, each 10 warmup and 20 full optimizer steps.

Actual evidence: `{'timing_rank': 0, 'preflight_rank': 64, 'profile_traces': 32, 'candidate_mappings': 3, 'calibration_rank': 8, 'calibration_mappings': 2}`; expected: `{'timing_rank': 256, 'preflight_rank': 64, 'profile_traces': 32, 'candidate_mappings': 3, 'calibration_rank': 8, 'calibration_mappings': 2}`. Calibration ranks contain both full shapes. The whole calibration directory, both measured plans, source snapshots, build manifests, scripts, schedule, profile traces, command/process logs and report are preserved. Strict report is generated on H100 without offline mode. All samples are primary; slow-window sensitivity uses a separate whole-block table and keeps negative results.

GPU checks occur before calibration and before the comparison only. No Slurm, waiting queue, periodic runtime occupancy query, automatic retry or independent repeat. Any failed stage stops. Historical builds, baselines and data are unchanged. Libraries, object files and caches remain local and ignored by Git. Data-return commit SHA: not created.
