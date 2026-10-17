# NVIDIA Driver Diagnosis

Read-only diagnosis recorded 2026-10-17 after the second calibration failure.

## Healthy observations

- NVIDIA driver/NVML: `580.65.06`; loaded `nvidia`, `nvidia_drm`, and `nvidia_modeset` modules all report `580.65.06`.
- GSP firmware on all eight GPUs: `580.65.06`; VBIOS: `96.00.A5.00.01`.
- Eight H100 80GB GPUs are visible, idle, and have no compute processes.
- `nvidia-smi -q -d FABRIC` currently reports each GPU `State: Completed`, `Status: Success`.
- All 18 NVLink links per GPU report `26.562 GB/s`.
- `nvidia-smi nvlink -e` reports zero Replay, Recovery, and CRC errors for all links.
- ECC volatile and aggregate counters are zero; no Xid error was found.
- PyTorch 2.6.0+cu124 sees eight devices, CUDA runtime 12.4, NCCL 2.21.5, and all CUDA peer-access pairs pass.
- Driver parameters show `NvLinkDisable: 0`; no GPU is excluded.

## Abnormal observations

- During both calibration attempts, all ranks failed at custom peer tensor creation with CUDA 401 (`the operation cannot be performed in the present state`).
- Kernel logs at both attempts contain `NV_ERR_FABRIC_STATE_OUT_OF_SYNC` from `mem_multicast_fabric.c:3034`.
- The host has been up for 81 days. The container is not booted with systemd, so `systemctl` cannot verify host services.
- No `nvidia-fabricmanager`, DCGM, or related package/process is visible inside this container. The presence and health of the host-side Fabric Manager service therefore remain unverified.

## Interpretation

The basic driver, GPU enumeration, PCIe/NVLink link state, ECC, and user-space CUDA/NCCL versions are internally consistent. The reproducible failure is specifically associated with the driver/NVLink fabric state used by peer/multicast allocation, despite `nvidia-smi` reporting the current fabric state as completed/success. This requires host-level Fabric Manager/NVSwitch and driver-state inspection or recovery before another calibration attempt.

All raw outputs are in this directory: `driver-nvidia-smi-q.txt`, `driver-nvlink-status.txt`, `driver-nvlink-errors.txt`, `driver-topology.txt`, `driver-services.txt`, `driver-modules.txt`, `driver-userspace-libraries.txt`, `driver-kernel-errors.txt`, and `driver-torch-nccl.txt`.

## Configured experiment environment

The retry used this independently configured environment:

- Repository: `/workspace/TempName`, commit `ccdb16a64594b1ed2be9262a14d325bd1b0afec7`
- Megatron-LM: `/workspace/Megatron-LM`, commit `3ea68ad6042cc1204386ae9364358f7c4de1bc37`
- CUTLASS submodule: `df8a550d3917b0e97f416b2ed8c2d786f7f686a3`
- NCCL source: `8c6c5951854a57ba90c4424fa040497f6defac46` (2.19.3); fresh `DEBUG=0` Release build at `$H100_A3_NCCL_ROOT`
- Python: 3.10.20 at `/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python`
- PyTorch: 2.6.0+cu124; CUDA runtime 12.4; PyTorch NCCL 2.21.5; Triton 3.2.0; NumPy 1.26.4; C++ ABI 0
- CUDA Toolkit 12.4.131: `/usr/local/cuda-12.4`; CUDA Toolkit 12.8.93: `/usr/local/cuda-12.8`
- GCC/G++: 11.4.0; CMake: 3.30.0; Ninja: 1.11.1.3
- Apex: source commit `e13873debc4699d39c6861074b9a3b2a02327f92` at `/workspace/apex-flux`, compiled with `apex_C`, `amp_C`, and `fused_layer_norm_cuda`
- Other pinned Python packages: `setuptools==75.8.0`, `wheel==0.45.1`, `packaging==24.2`, `einops==0.8.1`, `regex==2024.11.6`, `psutil==6.1.1`, `PyYAML==6.0.2`, `tqdm==4.67.1`, `six==1.17.0`, `pybind11==2.13.6`

The retry session was `ablation3-gpt67-tp8-scope-20261017-150923`. Its paths and runtime variables are recorded in `session-paths.json`: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, `DEBUG=0`, empty `MAKEFLAGS`, `CC=/usr/bin/gcc-11`, `CXX=/usr/bin/g++-11`, `CUDAHOSTCXX=/usr/bin/g++-11`, `PYTHONNOUSERSITE=1`, and `MAX_JOBS=2`. The output and log roots were `/workspace/TempName/outputs/h100/phase6/<session>` and `/workspace/TempName/logs/h100/phase6/<session>`.

The complete dependency list is `pip-freeze.txt`; source/environment evidence is in `session-paths.json`, `megatron-commit.txt`, `nccl-commit.txt`, and the repository handoff files.
