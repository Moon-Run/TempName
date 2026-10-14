# Serial H100 experiment batch

Seven independent five-group ablation2 suites, then seven independent four-group Attention/MLP suites. Wait for the current scope suite and strict report before taking the shared experiment lock. Each ablation2 suite has 50 windows; each scope suite has 32; all have two rounds, 10 warmup and 20 full optimizer steps. Total: 574 fresh windows.

Verified existing builds and real calibration are reused read-only. Each suite creates its own compare, run_id, controller/source snapshots, schedule, full mapping/smoke/profile, timing and strict reports. Slow-window sensitivity stays separate; no old timing is substituted. Input or controller changes, busy GPUs or any failed gate stop the batch. No automatic retry or additional runs.

Only parent/process/status heartbeats occur during execution. GPU occupancy is queried by the official entry before each suite only. No other tasks are stopped. Libraries, objects and caches remain local and ignored. Source/calibration references are retained in every handoff. Data-return commit: not created.

Plan: `/workspace/TempName/outputs/h100/phase6/h100-ablation2-scope-20261015-012238-queue/batch-plan.json`. Status: `/workspace/TempName/logs/h100/phase6/h100-ablation2-scope-20261015-012238-queue/watchdog-state.json`.
