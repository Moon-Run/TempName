# Profile-count correction and validation

The original failed session is `TTention-MLP-gpt67-tp8-20261014-231236`. Its records, submission, controller snapshots and failed gate remain unchanged.

The maintained `scope_report.profile_roles` now requires `cat=user_annotation` in addition to the exact name and `ph=X` when counting CPU spans and locating CUDA launches. GPU annotation mirrors are excluded from this CPU-span count. The expected count remains 128 per role. Kernel launch correlation, V2/mixed-wire requirements, native no-Flux checks, numerical budgets, actual module calls and all other evidence gates are unchanged.

Validation before this new GPU run:

- All 72 CPU regressions passed, including the new four-policy test with mirrored GPU annotations.
- A missing CPU span is rejected even with all GPU mirrors present.
- An extra CPU span is rejected even with all GPU mirrors present.
- Read-only application of the corrected role/kernel checks to all 32 original real traces passed, each role showing exactly 128 CPU spans. No old `preflight.json` was written.

The original real Attention and MLP calibration, both complete libraries, original source inventory, plans, device order and software environment passed reuse validation. New GPU mappings, smoke/profile, formal timing and strict report are generated only in this new compare. Calibration controller identities are verified against their immutable archived snapshots; the new comparison freezes the corrected maintained controller hashes and the uncommitted patch.
