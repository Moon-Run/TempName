# H100 interleaved five-group ablation

Session: `ablation2-gpt67-tp8-interleaved-20261014-215234`.

Status: **failed**. Source changed since validated CPU run. Updated Beijing: 2026-10-14T22:18:34.212672+08:00.

Code: `687e1888dea559d7b75c5015ac42dbd97145c08f`; original worktree and patch are recorded in handoff. GPT 6.7B, 32 layers, H4096/FFN16384, 32 heads, S2048, mb1/global4, BF16, TP8. Five groups only: native_taco, G1, G2, G3, G4; interleaved; two rounds of five paired blocks. 50 fresh windows, each 10 warmup + 20 full optimizer steps. No Attention extension.

Reused source build: `/workspace/TempName/outputs/h100/phase6/ablation-gpt67-tp8-20261013-233551/candidates`; the four complete libraries stay on H100. The session candidates registry includes source references, hashes and all four plans. Compare preparation freezes baseline sources, controllers, worker, configuration and schedule.

Actual evidence counts: `{'timing_rank': 0, 'preflight_rank': 0, 'traces': 0, 'mappings': 0}`. Expected: 400 formal ranks, 80 preflight ranks, 40 compressed traces and 4 mapping CSVs. Missing evidence is not filled with historical timing. Strict report and paired speedups/95% intervals exist only after successful GPU execution.

Startup GPU availability is checked by ablation2.py run --stage all --execute. An external watcher waits for GPUs before launching this controller; this controller performs one final availability gate, with no waiting or retries after launch. No repeated suite, other-task termination or periodic runtime occupancy query. Failures are preserved. All-sample results are primary; whole-block slow-window sensitivity is separate. Libraries, object files, CMake/Ninja caches and Python caches remain ignored. Data return commit SHA: not created.
