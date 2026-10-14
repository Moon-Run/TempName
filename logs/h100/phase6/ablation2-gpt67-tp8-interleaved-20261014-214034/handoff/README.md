# H100 interleaved five-group ablation

Session: `ablation2-gpt67-tp8-interleaved-20261014-214034`.

Status: **failed**. run-all.log exited 1; see /workspace/TempName/logs/h100/phase6/ablation2-gpt67-tp8-interleaved-20261014-214034/run-all.log. Updated Beijing: 2026-10-14T21:44:59.152280+08:00.

Failure reason: the entry rejected busy GPUs before mapping or model execution. At 21:44:58 Beijing on 2026-10-14, all eight H100s used 71887 MiB each at 100% utilization. Host and GPU UUID/order matched the parent calibration. See `compare/start-gpu-check.json` and `run-all.log`; the controller has exited and no wait queue was created.

Completed preparation: 57 CPU regressions, source freeze and references checks, local runtime/model/Megatron/submodule identity checks, verified reuse of four complete interleaved builds, new executable compare and dry-run of exactly five groups and 50 formal windows. Python/PyTorch/CUDA/NCCL, hardware, selected library hashes and relevant environment variables are in `environment.json`. No baseline or maintained implementation was changed.

GPU correctness, performance and speedup/95% intervals are unavailable. All 400 formal rank records, 80 preflight rank records, 40 traces and 4 mapping files remain missing. This is startup resource rejection, not numerical or candidate mapping failure. Once GPUs are available, prepare a new independent compare/session using the verified reuse registry; preserve this failed state and do not edit its submission or stage files to retry.

Code: `4eccd5da5bd6950d307e3fcfed0923fb77c76f82`; original worktree and patch are recorded in handoff. GPT 6.7B, 32 layers, H4096/FFN16384, 32 heads, S2048, mb1/global4, BF16, TP8. Five groups only: native_taco, G1, G2, G3, G4; interleaved; two rounds of five paired blocks. 50 fresh windows, each 10 warmup + 20 full optimizer steps. No Attention extension.

Reused source build: `/workspace/TempName/outputs/h100/phase6/ablation-gpt67-tp8-20261013-233551/candidates`; the four complete libraries stay on H100. The session candidates registry includes source references, hashes and all four plans. Compare preparation freezes baseline sources, controllers, worker, configuration and schedule.

Actual evidence counts: `{'timing_rank': 0, 'preflight_rank': 0, 'traces': 0, 'mappings': 0}`. Expected: 400 formal ranks, 80 preflight ranks, 40 compressed traces and 4 mapping CSVs. Missing evidence is not filled with historical timing. Strict report and paired speedups/95% intervals exist only after successful GPU execution.

Startup GPU availability is checked by ablation2.py run --stage all --execute. No automatic waiting, no repeated run, no other-task termination, no periodic runtime occupancy query. Failures are preserved. All-sample results are primary; whole-block slow-window sensitivity is separate. Libraries, object files, CMake/Ninja caches and Python caches remain ignored. Data return commit SHA: not created.
