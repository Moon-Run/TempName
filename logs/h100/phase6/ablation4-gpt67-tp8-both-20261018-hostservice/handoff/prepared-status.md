# Ablation4 Detached-Run Status

Recorded: 2026-10-18 (Asia/Shanghai)  
Session: `ablation4-gpt67-tp8-both-20261018-hostservice`  
Host reported by the container: `gxn75`

## Preparation complete

- The four runtime variants were built and verified: pinned fused v2 `taco_fused`, `interleaved_all`, `interleaved_arrival_all`, and frozen `scope-candidate` C. Both the fused and candidate manifests report `status=built`.
- Both explicit measured role plans came from `/workspace/TempName/outputs/h100/phase6/ablation3-gpt67-tp8-scope-20261017-163146/plans/measured/`: Attention and MLP. The shared NCCL Release library came from that same completed NVLS=0 session.
- `compare1` and `compare2` are separate, newly frozen compare directories with unique run IDs, both using `--blocks 5 --rounds 2 --nvls-enable 0`.
- Each dry-run reports 50 formal windows and 60 total worker windows. Each includes 10 preflight windows (5 smoke and 5 profile) and 8 policy/role mapping commands.
- Both compare submissions freeze the same measured plans, build identity, schedule, and communication snapshot. `NCCL_NVLS_ENABLE=0`, `NCCL_DEBUG=WARN`, and `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` are recorded. The compare run IDs are distinct.
- `run-two.sh`, `runtime-env.sh`, and `handoff/start-host-service.sh` are prepared. The run script executes compare1, verifies its strict report, and only then executes and verifies compare2.
- `bash -n` passed for both scripts. Build `.so` files and build caches match `.gitignore` and remain excluded from Git.

## Environment and provenance

- Repository: `/workspace/TempName`, commit `92cd94bc58e41aae060959782f65181e89ca404b`.
- Megatron-LM: `/workspace/Megatron-LM`, fixed commit `3ea68ad6042cc1204386ae9364358f7c4de1bc37`.
- Python 3.10.20: `/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python`.
- PyTorch 2.6.0+cu124; CUDA runtime 12.4; Apex compiled extensions from the pinned source checkout.
- Build CUDA Toolkit `/usr/local/cuda-12.8`; run CUDA Toolkit `/usr/local/cuda-12.4`; GCC/G++ 11; CMake 3.30.0; Ninja 1.11.1.3.
- GPUs: eight H100 80GB, 132 SM, fixed visible order 0-7; Megatron and measured-plan hardware identities match the 75 host session.
- Fused source snapshot: `tools/h100/phase6/fused_v2` with provenance pinned by the build manifest. Selective candidate parent source commit: `ff38f5b1ff1c80cb831f58c64b95f42b0379879a`.
- NCCL/CUTLASS commits and package versions are in `submodules.txt` and `pip-freeze.txt`.
- `prepared-input-sha256.txt` records hashes for the build-independent scripts and both submissions. Each submission records the frozen build/plan/library hashes.

## Detached execution

The user authorized independent execution inside the container provided it is not a Codex app-server/VSCode child. A double-fork launcher started the sequential supervisor directly under container PID 1:

- Supervisor PID: `1062169`; PPID: `1`; SID/PGID: `1062168`.
- Cgroup: `0::/`.
- Command: `/bin/bash .../run-two.sh /workspace/TempName /workspace/conda-envs/flux-h100-ablation3-20261017/bin/python .../ablation4-gpt67-tp8-both-20261018-hostservice`.
- The compare1 controller PID `1062171` is a child of the supervisor. Process ancestry was checked and contains no Codex app-server or VSCode process.
- PID record: `handoff/container-supervisor.pid`; supervisor output: `handoff/container-supervisor.log`.
- At the latest check, compare1 had 8/8 mapping outputs and 5/5 smoke windows (40 rank records). Profile preflight was in progress: two profile directories had been started, one with eight compressed trace files. Formal timing had not started. No periodic GPU utilization polling is used.

This is detached from the editor process tree, but remains inside the running container. The container/runtime must remain alive for work to continue. The host systemd launcher remains only as a fallback artifact; no host service was created.

## Execution counts

- GPU smoke/profile/mapping: compare1 started; 8/8 mapping outputs and 5/5 smoke windows (40/40 rank records) complete; profile preflight in progress (2/5 directories started, 8/40 trace files present at last check).
- compare1: in progress; 0/50 formal windows at last check.
- compare2: prepared, waiting for compare1's strict report; 0/50 formal windows.
- Reports: none yet.

After host service completion, expected combined delivery is 100 formal windows / 800 formal rank files, 20 preflight windows / 160 preflight rank files, 80 compressed profile traces, and 16 policy/role mappings, plus both strict reports and all command/process logs.
