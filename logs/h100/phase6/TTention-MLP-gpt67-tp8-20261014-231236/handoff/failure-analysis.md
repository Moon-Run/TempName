# Attention/MLP H100 preflight failure

Session: `TTention-MLP-gpt67-tp8-20261014-231236`. Source commit: `5b3c41be6d2536ecf8d4c32d824071a819dba452`. The controller stopped at 2026-10-14 23:30:35 Beijing. No formal timing was started; 0/32 windows and 0/256 formal rank records.

Both full shapes were newly calibrated on the same gxn74 eight 132-SM H100s in this session. All eight calibration ranks contain Attention `[2048,4096,4096]` / local K512 and MLP `[2048,4096,16384]` / local K2048. The calibration completion gate and fitting passed with training passes 0/1 and heldout pass 2. Actual selected counts by source rank:

- Attention: `[7,7,7,2,7,7,7,1]`.
- MLP: `[2,7,7,2,7,7,7,0]`.

New sampler and shared dual-shape candidate libraries were fully compiled with CUDA 12.8. The two candidate policies use the same library and identical Attention plan. CUDA 12.4 was used for GPU execution. All 71 CPU regressions and source freeze/reference checks passed. No source-only build, synthetic plan, historical timing or old MLP-only runtime library was used.

All four smoke windows completed with eight ranks each, zero skips and zero unexpected fallbacks. Maximum relative L2 errors against BF16 / independent codec were:

| Policy | vs BF16 | vs codec |
| --- | ---: | ---: |
| native_taco_attn_mlp | 0.025464761 | 0 |
| native_taco_attn | 0.025444034 | 0 |
| interleaved_full_attn | 0.003829081 | 0.000062348 |
| interleaved_full_attn_mlp | 0.003825618 | 0.000086355 |

These smoke values satisfy the fixed 0.05 / 0.01 budgets. They do not establish profile execution-path validation or performance.

All eight smoke/profile windows produced 64 raw rank records and 32 compressed traces. Three candidate/shape mapping CSVs are present. The preflight aggregate failed in `tools/h100/phase6/scope_report.py:142`:

```text
ValueError: Missing profile scopes: scope.attn.tensor
```

The counter includes every same-named `ph=X` event without restricting its category. Actual traces contain 128 CPU `user_annotation` events and 128 `gpu_user_annotation` events per Attention or MLP role, totaling 256. The checker expects `32*4=128`, so the aggregate rejects the trace. The actual rank reports record four calls per module for a single profile optimizer step; they are not two measured steps. This is a profile event-count mismatch, not missing smoke rank records.

The profile gate and role/kernel association have not been certified. No `preflight.json`, official `report.json`, `report.md`, speedup or 95% interval was generated. Missing delivery items are all 256 formal rank records and their timing commands/process logs, the successful preflight gate and performance reports. Existing calibration, plans, source snapshots, manifests, commands, environment, raw rank records, traces and failure logs are retained.

The experiment was stopped as requested. No check was relaxed, no failed group was skipped, no slow window was removed and no automatic repeat or waiting queue was created. The original frozen sources and historical data were not edited. Libraries, object files and build caches remain local and Git-ignored. The additional inspection is diagnostic only; it does not promote this failed session to a successful preflight. A corrected profile-count implementation would require a new compare/session and fresh preflight before formal timing.

Evidence: `compare/results/profile-*/rank*-profile.json.gz`, `compare/results/profile-*/rank*.json`, `compare/all-state.json`, `run-all.log`, `execution.json`, `handoff/profile-scope-diagnostics.json`, `handoff/partial-evidence-verification.json`, `handoff/partial-artifact-hashes.json`.
