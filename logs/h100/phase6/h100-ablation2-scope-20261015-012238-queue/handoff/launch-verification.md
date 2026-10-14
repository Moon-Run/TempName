# Serial repeat batch launch

Batch: `h100-ablation2-scope-20261015-012238-queue`.
Controller PID: 2061714. Started Beijing: 2026-10-15 01:24.

Order is seven five-group interleaved ablation2 suites (50 fresh windows
each), followed by seven four-group Attention/MLP suites (32 fresh windows
each). Every suite retains two rounds, 10 warmup and 20 full optimizer steps.
Total: 14 independent sessions and 574 new formal windows.

CPU batch schedule checks passed, including rejection of incomplete schedules
and source-only inputs. The repaired `batch.py` also passed direct standard
library Queue and PyTorch import checks without initializing CUDA. Complete
local candidates, plans, source/manifests and real dual-shape calibration
were verified against their frozen hashes. Software and communication settings
match the original sessions. The input hash list is in `input-verification.json`.

The parent Attention/MLP session completed all 32 windows and its strict
report at 2026-10-15 01:15:28 +08:00. Its records remain unchanged.

The first replacement suite passed reuse verification, preparation, dry-run,
and the official GPU startup gate at 2026-10-15 01:25:00 +08:00. All eight
assigned GPUs showed zero used memory and zero utilization, with no visible
compute processes. See that suite's `compare/start-gpu-check.json`.
The official mapping/smoke/profile and timing entry is now running.

Previous failed batches remain at `h100-ablation2-scope-20261015-010547-queue`
and `h100-ablation2-scope-20261015-011744-queue`: the former documents a fixed
Python module naming collision; the latter documents the busy GPU startup
gate. Neither produced formal windows. They are not included in these 14 runs.

Each suite creates a fresh comparison/run_id and produces its own strict
report and delivery inventory. GPU occupancy is checked only before each
suite, not during measurement. Any busy gate, changed input or failed test
stops the remaining batch without retry. Libraries, objects and caches stay
local and ignored. No data-return commit was created.

From the repository root:

```bash
cat logs/h100/phase6/h100-ablation2-scope-20261015-012238-queue/watchdog-state.json
tail -f logs/h100/phase6/h100-ablation2-scope-20261015-012238-queue/watchdog.log
```

The state exposes `current_session`, `run_index`, `completed_runs`, and the
current suite's window progress. Results for each session are under its own
`logs/h100/phase6/<session>/compare/`. All planned paths are frozen in the
batch plan and batch state.
