# Independent H100 ablation2 repeat 3

Session: `ablation2-gpt67-tp8-interleaved-20261015-012238-repeat3`. Status: **complete**. All fresh preflight/timing and strict report passed. Beijing: 2026-10-15T03:45:03.526372+08:00.

GPT 6.7B TP8; policies `['native_taco', 'G1', 'G2', 'G3', 'G4']`; two rounds; 5 paired blocks per round; 50 fresh windows, 10 warmup + 20 complete optimizer steps. Formal timing and all preflight are fresh.

Reused builds: `/workspace/TempName/outputs/h100/phase6/ablation2-gpt67-tp8-interleaved-20261014-221952/candidates`. Reference compare: `/workspace/TempName/logs/h100/phase6/ablation2-gpt67-tp8-interleaved-20261014-221952/compare`. Scope real two-shape calibration: `/workspace/TempName/logs/h100/phase6/TTention-MLP-gpt67-tp8-20261014-231236/calibration`. Reuse records and frozen plan/library/controller hashes accompany this compare. Original source and calibration evidence remain at their referenced locations.

Actual: `{'timing_rank': 400, 'preflight_rank': 80, 'traces': 40, 'mappings': 4}`; expected: `{'timing_rank': 400, 'preflight_rank': 80, 'traces': 40, 'mappings': 4}`. Reports use all samples as primary and keep slow-window filtering separate. No missing records are filled with historical results. Busy GPUs, input changes or failure stop the entire remaining batch; no retry. GPU occupancy is checked before each suite only. No claim of measured runtime exclusivity. Libraries, objects, build and Python caches stay ignored. Data-return commit: not created.
