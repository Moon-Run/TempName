# Environment and frozen input record

Commit: `d786da51e0aeafec1bb2b5d071c5c59e93e6de43`
Host: `gxn74`
Runtime: flux-h100 / Python 3.10.20 / PyTorch 2.6.0+cu124 / CUDA 12.4 / NCCL 2.21.5.
Build: CUDA 12.8 / GCC 11 / CMake 3.30 / existing NCCL 2.19.3 SM90 static release.

Implementation: `a800-fused-v2-cta-v1`.
Candidates and measured plan are unchanged and reused after exact config/hash/hardware/runtime checks.
No historical timing is used. All six policies will be measured anew.

## Runtime environment

```json
{
  "PYTHONNOUSERSITE": "1",
  "NCCL_VERSION": "2.21.5-1",
  "CXX": "/usr/bin/g++-11",
  "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
  "CUDA_DEVICE_MAX_CONNECTIONS": "1",
  "OMP_NUM_THREADS": "1",
  "CUDA128": "/usr/local/cuda-12.8",
  "CUDA124": "/usr/local/cuda-12.4",
  "CC": "/usr/bin/gcc-11",
  "MKL_NUM_THREADS": "1"
}
```

## GPU order

0: `bc561ab8-87a1-aaa1-6b5d-f379fc7571b7` / NVIDIA H100 80GB HBM3 / 132 SM
1: `bd6c7a45-7290-f263-a277-e3afefe2484e` / NVIDIA H100 80GB HBM3 / 132 SM
2: `571bb6d9-c967-403e-3be3-38113e2cc7b3` / NVIDIA H100 80GB HBM3 / 132 SM
3: `e69c65ae-0527-2a67-1b3b-3b20f31465a2` / NVIDIA H100 80GB HBM3 / 132 SM
4: `85309d21-1193-de40-5836-af3211e41112` / NVIDIA H100 80GB HBM3 / 132 SM
5: `9df7bad7-654c-d1af-e88c-ded6cc7467fd` / NVIDIA H100 80GB HBM3 / 132 SM
6: `ad6311ce-0286-7a7d-241a-a4913459728c` / NVIDIA H100 80GB HBM3 / 132 SM
7: `98d8b4b3-eab2-35a9-2aca-138b9c44766e` / NVIDIA H100 80GB HBM3 / 132 SM

See reuse-verification.json, pip-freeze.txt, topology.txt, CUDA/compiler logs and gpu-occupancy.jsonl for evidence.
