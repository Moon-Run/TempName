# CR2 remaining submission diagnosis (2026-10-23)

Evidence: all 32 separately collected full-training traces (CR2 / CR / C / R × 8 ranks), each one optimizer step, 128 attention + 128 MLP projection calls. Source SHA-256 values are recorded in both JSON outputs. This is a read-only analysis; no GPU run was performed.

**Limitation:** profiles are instrumented and separately scheduled; CPU costs, GPU kernel sums, communication waits, and stream gaps are not additive predictions of normal training speedup. Profile spans are 842–901 ms; timed CR2 steps are approximately 511 ms. The delayed rank also changes between runs.

## Remaining size of target

| Trial | CR2 mean (ms) | Further reduction to reach 2% relative C (ms) | Further reduction to reach 2% relative R (ms) |
|---|---:|---:|---:|
| compare1 | 511.715097 | 5.031202 | 7.053776 |
| compare2 | 510.879037 | 4.665727 | 4.736461 |

These target budgets use ratios of the reported arithmetic means for scale. Acceptance must keep the paired-block bootstrap calculation, contemporaneous controls, and two-trial requirement.

## Projection stage measurements

The next table is the median across ranks of each rank's per-call median, in microseconds.

| Arm | attn CPU | MLP CPU | attn CUDA API | MLP CUDA API | attn decoder | MLP decoder |
|---|---:|---:|---:|---:|---:|---:|
| CR2 | 99.656 | 99.408 | 27.230 | 26.522 | 10.584 | 11.040 |
| CR | 127.598 | 129.242 | 28.174 | 27.147 | 10.720 | 11.008 |
| C | 127.404 | 129.877 | 28.206 | 27.588 | 22.320 | 22.136 |
| R | 132.733 | 134.211 | 32.496 | 31.708 | 14.511 | 14.696 |

CR2 runtime calls per projection: one `cudaStreamIsCapturing`, one `cudaMemsetAsync`, and three `cudaLaunchKernel` calls (producer GEMM, publication barrier, decoder), plus occasional allocator `cudaEventQuery`. Per-step medians across ranks: launch APIs 4.658 ms, memset APIs 1.924 ms, capture-query APIs 0.308 ms. These are host API costs, not standalone GPU work.

CR2 decoder duration sums to median 2.786 ms/rank per step. Even a hypothetical zero-duration decoder would be smaller than the additional 4.7–7.1 ms needed by the target, before considering overlap. Decoder tuning alone is poorly supported as the next 1 percentage point.

## Rank differences in CR2

| Rank | attn GEMM queue median (µs) | MLP GEMM queue median (µs) | attn next-launch lateness median (µs) | Postdecode gap sum (ms) | Positive host lateness sum (ms) | Publication barrier sum (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 6892.922 | 7152.183 | -7270.690 | 0.462 | 0.000 | 79.132 |
| 1 | 16436.635 | 16695.275 | -16859.310 | 0.445 | 0.000 | 92.583 |
| 2 | 3335.886 | 3569.332 | -3772.347 | 0.715 | 0.313 | 83.242 |
| 3 | 2208.541 | 2504.674 | -2669.905 | 0.524 | 0.073 | 93.165 |
| 4 | 7431.588 | 7759.918 | -7876.332 | 0.450 | 0.000 | 96.649 |
| 5 | 0.960 | 1.070 | 38.675 | 5.421 | 5.088 | 9.281 |
| 6 | 574.397 | 820.003 | -980.519 | 0.589 | 0.220 | 84.609 |
| 7 | 3610.345 | 3914.583 | -4032.060 | 0.448 | 0.000 | 89.905 |

Queue is kernel start minus the end of its launch API. Positive next-launch lateness means the following same-stream kernel was submitted after decoding had already completed. Other-stream NCCL overlaps remain present, so none of these columns directly measures removable critical-path time.

In this CR2 profile, rank 5 has an attention postdecode gap median of 40.160 µs, next-launch lateness median of 38.675 µs, and only 10.94% of attention next launches already queued by decoder completion. Its MLP next launch is already queued 96.875% of the time. In CR profiles rank 5 is already queued by milliseconds, illustrating why a fixed rank-specific remedy cannot be justified.

All 256 immediate decoder consumers on rank 5 are BF16 elementwise `aten::add` kernels. This locates a launch boundary worth investigating, but trace names alone do not prove whether it is bias or residual addition. Fusion would require inspecting the actual consumer semantics and preserving BF16 rounding, output ownership, autograd, and dropout behavior.

## Choice supported by current evidence

There is remaining exposed submission latency, especially on attention projections, but there is no instrumentation range around `can_implement` / CUTLASS `initialize`. The current trace cannot assign them a measured savings budget. Replacing repeated initialization with a validated parameter update is a plausible exploratory change, not an evidence-backed guarantee of another 1%.

A new mechanism that batches the producer/reset/barrier/decoder submission has a larger identifiable host-call budget (four device work submissions per projection). Dynamic input/weight/output addresses, alternating workspaces, StreamK reset semantics, and identical rank ordering must all remain correct. Fixed-address graph replay without explicit pointer updates is not an acceptable implementation. Measure a host-only preparation candidate separately if implemented, so benefits are not attributed to graph replay prematurely.

Reproduce:

```bash
python3 outputs/h100/phase6/cr2-next-diagnosis-20261023/profile-analyze.py
python3 outputs/h100/phase6/cr2-next-diagnosis-20261023/profile-runtime.py
```
