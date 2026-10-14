# Batch entry naming collision

The first suite stopped during environment verification before preparation,
mapping, smoke/profile or timing. No measured windows were produced and the
previous completed Attention/MLP experiment was unaffected.

The new helper was named `queue.py`. When executed directly, its directory
was on `sys.path`, so PyTorch's `from queue import Queue` imported this helper
instead of Python's standard-library module. The traceback is preserved in
the first suite's `reuse-verification.log` and this batch's `watchdog.log`.

The repair uses `batch.py` in a new directory containing no `queue.py`.
The new regression explicitly imports the standard-library Queue and PyTorch
and confirms that CUDA contexts are not initialized. Scheduling, models,
groups, blocks, seeds, plans, libraries and numerical gates are unchanged.

Replacement batch: `h100-ablation2-scope-20261015-011744-queue`. All fourteen
replacement suites have new paths. This failed batch, its initial first-suite
directory, state, frozen script and logs remain unchanged for provenance.
