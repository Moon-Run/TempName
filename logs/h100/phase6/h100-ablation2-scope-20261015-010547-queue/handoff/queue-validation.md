# Batch launch verification

Beijing date: 2026-10-15. Batch controller PID: 2050443.

- CPU schedule regression passed: seven ablation2 suites followed by seven
  Attention/MLP suites, two rounds each, 574 fresh formal windows total.
  Incomplete schedules and source-only submissions are rejected.
- Existing reference submissions satisfy the same five-group/50-window and
  four-group/32-window schedule checks used for each new comparison.
- Local complete candidate libraries, source/manifests, physical plans, real
  Attention and MLP BF16 calibration records and frozen evidence hashes passed
  the input verification in `input-verification.json`.
- Python, PyTorch, CUDA runtime, NCCL runtime and communication/environment
  settings exactly match both references. This extra metadata check did not
  initialize CUDA contexts.
- Shell syntax and Git ignore checks passed. Scripts/results remain eligible
  for Git; dynamic libraries and compiled objects remain ignored.
- The parent Attention/MLP session completed 32 windows and its strict report
  at 2026-10-15 01:15:28 +08:00. Its frozen historical inputs were preserved.

The controller waits for parent completion and the shared experiment lock,
then uses each experiment's independent official entry. Every new suite reruns
mapping, smoke/profile and timing. Busy GPUs, failed gates or changed frozen
inputs stop remaining suites. Runtime GPU occupancy polling is disabled.

From the repository root, inspect the batch with:

```bash
cat logs/h100/phase6/h100-ablation2-scope-20261015-010547-queue/watchdog-state.json
tail -f logs/h100/phase6/h100-ablation2-scope-20261015-010547-queue/watchdog.log
```

`current_session` points to the suite being executed. Its `execution.json`
contains window progress; its `compare/report.json` and `compare/report.md`
are valid final results only after strict completion. Session paths for all
14 suites are listed in `outputs/.../batch-plan.json` and the batch state.
