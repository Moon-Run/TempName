# Independent H100 ablation repeat

Status: complete; Strict report passed; 100 fresh windows. Beijing: 2026-10-14T07:31:31.578241+08:00.

Reused build: `/workspace/TempName/outputs/h100/phase6/ablation-gpt67-tp8-20261013-233551/candidates`; parent report: `/workspace/TempName/logs/h100/phase6/ablation-gpt67-tp8-20261013-233551/compare/report.json`.

GPT 6.7B TP8; native + G0-G6; remote_first only, round1 only; G5 seeds 17/29/43; 10 blocks; schedule seed 20261013; 10 warmup + 20 full optimizer steps; 100 fresh windows. Each repeat creates a new compare and reruns all preflight and timing. No slow-window deletion or automatic retry. Runtime occupancy sampling disabled; exclusive use throughout execution is not asserted. Build sources and libraries remain in the referenced parent build; libraries/caches stay local. Data return commit SHA: pending.
