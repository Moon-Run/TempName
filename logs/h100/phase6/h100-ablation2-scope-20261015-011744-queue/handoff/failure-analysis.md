# GPU startup gate stopped the batch

At 2026-10-15 01:20:14 +08:00, the first ablation2 suite passed local
environment, GPU identity/order/132-SM, library/plan/source hash validation,
reuse registration, fresh comparison preparation and the 50-window dry-run.

The official startup gate then recorded GPU 4 using 4399 MiB, exceeding the
unchanged 1024 MiB limit. All reported GPU utilization values were zero, and
the compute-app query returned no visible entries. Neither fact satisfies
the memory gate or proves that this allocation is available to the experiment.

The official entry and the entire remaining batch stopped as required.
No mapping, smoke/profile or formal training window started. No other tasks
were stopped. Nothing was retried and the occupancy threshold was not relaxed.

Evidence is preserved in the first suite:

- `compare/start-gpu-check.json`
- `compare/all-state.json`
- `run-all.log`
- `handoff/environment.json` and `handoff/repeat-inputs.json`

The previously completed Attention/MLP experiment remains complete with
32 fresh windows and its strict report. New repeats completed: zero of 14.

A fresh prepared batch is needed after availability is restored; this failed
batch and comparison must not be restarted by removing their state files.
Any GPU waiting monitor requires the user to revise the prior busy-stop rule.
