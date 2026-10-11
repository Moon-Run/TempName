# H100 TP8 Deployment Validation

Date: 2026-10-09

Result: the documented TP8 experiment can run on this node. Calibration,
candidate GPU mapping checks, and all six full-model smoke/profile policies
completed successfully. Full timing has not been run; no speedup claim is made.

## Environment

- Conda environment: `/opt/conda/envs/flux-h100`
- Python 3.10.20, PyTorch 2.6.0+cu124, NumPy 1.26.4
- CUDA 12.4 for Apex/runtime; CUDA 12.8.93 for Flux compilation
- GCC/G++ 11.4.0, CMake 3.30.0
- Apex commit: `e13873debc4699d39c6861074b9a3b2a02327f92`
- Apex extensions: `apex_C`, `amp_C`, `fused_layer_norm_cuda`
- Megatron commit: `3ea68ad6042cc1204386ae9364358f7c4de1bc37`
- NCCL runtime: 2.21.5; pinned NCCL static build: 2.19.3, Release, sm90
- Eight H100 80GB HBM3 GPUs, 132 SM each, pairwise NV18 links and CUDA peer access
- TP8, 12 layers, hidden 2048, FFN 8192, sequence 2048, micro batch 1, global batch 4

CUDA 12.8 compiler and development packages were installed alongside 12.4:
cudart, driver headers, NVML, cuobjdump, cuBLAS, cuRAND, NVRTC, NVTX,
cuSPARSE and cuSOLVER. The environment script selects 12.4 for execution.
Apex setup.py was patched to retain only the three documented extensions.
No tracked files in TempName or Megatron were changed.

## Validation

- Python dependency audit: `pip check` passed.
- CPU protocol tests: 15 passed.
- Eight-rank BF16 NCCL all-reduce, all-gather, reduce-scatter and all-to-all passed.
- Four baseline variants and two candidate variants compiled successfully.
- Both BF16 calibration orders completed on this exact GPU allocation.
- Selection plan has `gpu_calibrated=true` and binds the node/device ordering.
- Six policies passed full-model smoke and independent profile on all eight ranks:
  `native`, `original`, `native_taco`, `taco_fused`,
  `remote_arrival_selective`, `interleaved_arrival_selective`.
- All 96 rank records completed/passed with zero skips and zero fallback calls.
- Preflight verified finite loss/gradient records, forward checks, selection masks,
  logical wire bytes, identical initialization/input fingerprints and loaded library hashes.
- All 48 rank profiles contained the required policy kernel evidence.
- Both executed GPU swizzles matched the frozen selection plan.
- `compare/preflight-state.json`: `status=completed`.
- `compare/preflight.json`: `completed=true`.
- GPUs were idle with zero memory usage after verification.

## Artifacts

Builds and selection plan:
`/workspace/TempName/outputs/h100/phase6/tp8-20261009-deployment`

Calibration, NCCL checks, Apex build log and model preflight:
`/workspace/TempName/logs/h100/phase6/tp8-20261009-deployment`

## Formal Timing

The next command runs the documented full timing campaign on the same GPUs.
It has not been executed as part of this deployment feasibility check.

```bash
cd /workspace/TempName
source outputs/h100/phase6/tp8-20261009-deployment/env.sh
python tools/h100/phase6/run.py "$H100_LOG/compare" \
  --stage timing --cuda-home "$CUDA124" --execute
```

Keep this allocation, GPU ordering, model configuration and frozen artifacts.
Changing them requires a new calibration and prepared comparison.
