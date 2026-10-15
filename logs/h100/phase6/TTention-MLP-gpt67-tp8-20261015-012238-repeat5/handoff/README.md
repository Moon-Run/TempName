# Independent H100 scope repeat 5

Session: `TTention-MLP-gpt67-tp8-20261015-012238-repeat5`. Status: **failed**. run-all.log exited 1; see /workspace/TempName/logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat5/run-all.log. Beijing: 2026-10-15T09:07:31.952829+08:00.

GPT 6.7B TP8; policies `['native_taco_attn_mlp', 'native_taco_attn', 'interleaved_full_attn', 'interleaved_full_attn_mlp']`; two rounds; 4 paired blocks per round; 32 fresh windows, 10 warmup + 20 complete optimizer steps. Formal timing and all preflight are fresh.

Reused builds: `/workspace/TempName/outputs/h100/phase6/TTention-MLP-gpt67-tp8-20261014-231236/candidates`. Reference compare: `/workspace/TempName/logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-003756/compare`. Scope real two-shape calibration: `/workspace/TempName/logs/h100/phase6/TTention-MLP-gpt67-tp8-20261014-231236/calibration`. Reuse records and frozen plan/library/controller hashes accompany this compare. Original source and calibration evidence remain at their referenced locations.

Actual: `{'timing_rank': 0, 'preflight_rank': 0, 'traces': 0, 'mappings': 0}`; expected: `{'timing_rank': 256, 'preflight_rank': 64, 'traces': 32, 'mappings': 3}`. Reports use all samples as primary and keep slow-window filtering separate. No missing records are filled with historical results. Busy GPUs, input changes or failure stop the entire remaining batch; no retry. GPU occupancy is checked before each suite only. No claim of measured runtime exclusivity. Libraries, objects, build and Python caches stay ignored. Data-return commit: not created.
