# H100 GPT 6.7B TP8 ablation

Session: `ablation-gpt67-tp8-20261013-233551`.

Status: **partial**; GPU ablation preflight started; formal timing follows only after all gates pass. Updated Beijing: 2026-10-14T00:35:06.503505+08:00.

One campaign only: remote_first/interleaved; native + G0-G6; G5 seeds 17/29/43; 10 blocks, schedule seed 20261013; 10 warmup + 20 full optimizer steps; 400 formal windows.

Source commit: `23e9cc214ef27ff22cecdd8425051a66c56fb5ad`. Initial source/environment/patch evidence is in this handoff directory; actual source snapshots and derived plans are in the session candidates directory. Parent freeze: h100-phase6-candidate-v1-20261013.

Original builds, fixed baselines and historical evidence are unchanged. Parent calibration reuse is allowed only after full local source/plan/library/environment/host/GPU-order verification. Each new candidate is built independently with CUDA 12.8 and executed using CUDA 12.4.

GPU occupancy is checked before GPU execution only; no periodic resource queries during preflight/timing or on failure. Occupancy sampling during execution is disabled; exclusive GPU use throughout measurement is not asserted.

No retries, slow-window deletion, independent repeat, or offline report substitution. Compiled libraries and caches remain on the H100 host. Data return commit SHA: pending; this background campaign does not publish Git commits. Final strict results: compare/report.json and compare/report.md when complete; otherwise retain all partial records and stage logs.
