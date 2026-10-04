# Phase6: execution cost and full-step acceptance

Only MLP `linear_fc2` forward changes. The two candidates retain their remote /
interleaved base order, a constrained arrival permutation, and a frozen physical
tile selection mask. BF16 GEMM, local BF16 contributions, E4M3/H128 codec and the
BF16 straight-through backward remain. Attention keeps its original backend.

The optimized implementation provides warp-local decoding, packed BF16 reduction
for groups with no quantized source, exact bounded FP32-to-E4M3 conversion on
SM80, instance-owned C++ configuration, and independently owned outputs. Optional
`--graph` captures only this MLP forward; input copy, replay and output clone are
included in full-step timing. Graph capture and mask upload happen during warmup.
The graph path requires fixed weight storage with in-place optimizer updates.

`--double-buffered` alternates two complete BF16/FP8 IPC workspaces and preserves
each invocation's post-GEMM publication barrier. Invocation k+1 cannot finish
that barrier until every rank has decoded k, protecting slot k's reuse at k+2.
This removes the separate post-decode barrier. It is mutually exclusive with
`--graph`. GPU checks add staggered-rank bursts and exact output-lifetime checks;
model profiles verify the reduced barrier count. The extra memory is measured.

The 2026-10-04 TP4 runtime additionally shares the immutable selection decision
over 16 rows of a physical tile, with columns varying first across CUDA blocks.
Its default dispatch is restricted to the measured M8192/N2048 and M8192/N4096
selective shapes; small H, M2048, N tails and other TP sizes keep the previous
decoder. The wire format, ring addition order, BF16 rounding and publication
barrier are unchanged. `bench_decode_variants.py` compares complete operators
and requires bitwise equality; `run_decode_diagnostics.py` also checks mixed
masks, tails, output lifetime and delayed-rank workspace reuse. The private
`taco_set_decode_variant` diagnostic switch accepts -1 (shape-based default),
0 (previous decoder), 1 (16 rows/tile-first), 2 (32 rows/tile-first), and
3 (16 rows/row-first). Formal runs use the frozen default, with no switch calls.

Builds and experiments are immutable and use fresh directories. Do not modify
old phase4/5 snapshots. Preserve allocation 179147 and its resident step 179147.1,
and preserve the outer hold loop of allocation 182708. Check Slurm before starting
and run GPU experiments serially within each allocation. With the user's explicit
authorization, independent TP4 experiments can run on both nodes concurrently;
every paired comparison must stay within one node and one model configuration.

```bash
python3 -m unittest discover -s tools/a800/phase6 -p 'test_*.py'
python3 tools/a800/phase5/build.py --out outputs/a800/phase6/NEW \
  --arrival-plan outputs/a800/arrival-v2-tp4-20261003/artifacts/plan.json
python3 tools/a800/phase6/prepare.py logs/a800/phase6/NEW \
  --build outputs/a800/phase6/NEW
squeue --steps -j 179147
srun --jobid=179147 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gpus=4 --kill-on-bad-exit=1 \
  /data/home/scyb672/run/conda_envs/flux-megatron-a800/bin/python \
  "$PWD/logs/a800/phase6/NEW/campaign.py" > logs/a800/phase6/NEW/driver.log 2>&1
python3 tools/a800/phase6/assess.py logs/a800/phase6/NEW
```

`prepare.py` freezes six groups: native Megatron, native Flux, native Megatron +
tensor TACO, **original-order Flux + frozen phase4 fused warp-v2 TACO**, and both
arrival/selective candidates. Formal measurement uses six position-balanced
blocks in each of two reversed rounds, 10 warmup and 20 complete optimizer steps
per window (72 windows / 288 rank records / 1440 timed global optimizer steps).
The frozen Flux required baseline is explicitly
`outputs/a800/phase4/taco-fused-warp-20261003`, never a candidate rebuild.

`--pilot` runs only the two candidates, original Flux and fused-v2 Flux, with two
blocks per round. It is exploratory and cannot pass `assess.py`. `--plan` and
`--arrival-plan` accept separately frozen selection/ordering plans. Smaller
budgets and a larger legal destination window are tuning choices; selection
based on old BF16 calibration is not post-quantization recalibration.

`assess.py` checks all ranks, reverse order, input/initialization/GPU identity,
mapping, numerical budgets, logical bytes, and the exact required baseline hash.
It reports `1 - exp(mean(log(T_candidate/T_baseline)))` and a 95% paired-block
bootstrap interval separately against both required baselines. Both rounds must
have point estimates at least the frozen target and positive intervals for a candidate to pass. New
`prepare.py` and `prepare_scenario.py` runs explicitly freeze 4% and an acceptance protocol version;
historical configurations without that field retain their original 5% target. It also
reports paired latency gaps to both native references, without inventing a
closeness threshold.

Diagnostics:

- `bench_operator.py` compares bounded selection budgets with/without graph
  replay. Operator times are diagnostic and cannot establish the training goal.
- `check_fp8.cu` compares the bounded conversion bit-for-bit with CUDA's codec
  on 4,194,304 deterministic random FP32 bit patterns plus rounding boundaries.
- `rebuild_runtime.py BASE OUT` clones a candidate and relinks only its runtime.
  ABI/epilogue/codec headers must match the source exactly; otherwise use the full
  builder. It never edits BASE.

GPU preflight includes all/none/checkerboard/calibrated masks, N tails, zero,
random and spiky inputs, repeated calls and output lifetimes. With `--graph`,
weights are updated in-place between codec cases. Model smoke/profile follows,
then both complete-step rounds. Finite loss and codec error budgets do not prove
real-data convergence; repetitions share one allocation and synthetic tokens.

## Communication-intensity and ordering extension

The user requested both remote-leading alternatives. Four independently
calibrated bases are supported: the existing rank+1 partition offset,
`L,R1,R2,R3`, `R1,L,R2,R3`, and `R1,R2,R3,L`. None of these names implies a strict
global ordering of all remote tiles before all local tiles. Per-destination
64-slot windows preserve the selected base's destination quota.

`build_remote_lead_sampler.py` creates isolated BF16 sampler libraries for the
two new orders (`--group-first` selects `R1,R2,R3,L`). `prepare_calibration.py`
freezes a calibration driver and exact-shape mapping checks. `plan.py` fits only
passes 0/1 and diagnoses pass 2. Observations rejected by the original validity
rules receive zero training priority; held-out availability never gates fitting.
Coverage and sampling overhead are recorded in the plan.

`build.py PLAN OUT` builds all calibrated candidates with MLP shape predicates;
attention bypasses their tables. `prepare_scenario.py OUT --build BUILD --plan
PLAN --hidden {512,1024,2048} [--pilot]` freezes eight same-model policies. The
default model is 12 layers, FFN=4H, S2048, micro/global batch4 and TP4. `--bases`
can select a subset of the four calibrated bases; formal runs use one balanced
block per policy. `--sequence` and `--tokens-per-microbatch` define a different
same-model scenario, not a speedup relative to another sequence length.
`--job-id` freezes the allocation expected by the campaign driver. Pilot rounds
have two blocks; the independent confirmation has eight balanced blocks and a
reversed second round (128 windows, 512 rank records, 2560 timed optimizer steps).
`scenario_suite.py` runs the diagnostic operator sweep, the three pilots and an
independent confirmation selected by the minimum repeated gain against both
required baselines. Candidate-to-candidate paired intervals are also reported.

`verify_run.py` loads each run's own frozen verifier and routing module, avoiding
mixing newer verification code with older snapshots. `assess.py` accepts only a
complete, balanced formal run, and checks the frozen required Flux baseline.

On 2026-10-04 the user explicitly stopped the H2048 confirmation after its first
round. All 64 first-round windows finished; the second round never started.
`verify_single_round.py RUN` audits this case without modifying frozen inputs or
relaxing `assess.py`'s two-round contract. See `verification-single-round.json`,
`report-single-round.md` and `user-stop.json` in the confirmation directory.
Campaign and suite states are `stopped_by_user`; do not automatically resume them.

The optional short-sequence/large-batch preparation arguments and the
`large-communication-calibration-20261003` directory were prepared but **not run**:
the user requested finishing the already started suite, documenting it, and
stopping. They supply no performance or numerical evidence. Do not automatically
launch them on a later continuation. Latest outcomes and the stopping boundary
are recorded in `docs/design/instruction.md` and `docs/design/base-phase6.md`.
