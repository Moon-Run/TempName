// TACO adaptation; see tools/a800/phase4/TACO_LICENSE.txt (BSD-3-Clause).
// Isolated op-2 all-remote decoder. The builder copies this header into its
// generated source tree; the pinned fused_v2 sources are never edited.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "gemm_rs/taco_runtime.h"

namespace flux_taco_op2 {

// Lane l owns coefficients l + 32*j. Stages 1..16 have exactly the scalar
// decoder's shuffle order; stages 32 and 64 become register butterflies.
// Do not mask padded N coefficients: they are part of the inverse H128.
__device__ __forceinline__ void decode_group(
    const unsigned char* payload, float quant, float adaptive, float (&x)[4]) {
  const int lane = threadIdx.x & 31;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __half_raw raw =
        __nv_cvt_fp8_to_halfraw(payload[lane + 32 * j], __NV_E4M3);
    x[j] = __half2float(__half(raw)) * quant;
    #pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
      const float other = __shfl_xor_sync(0xffffffff, x[j], delta);
      x[j] = (lane & delta) ? other - x[j] : x[j] + other;
    }
  }
  float a = x[0], b = x[1], c = x[2], d = x[3];
  x[0] = a + b; x[1] = a - b; x[2] = c + d; x[3] = c - d;
  a = x[0]; b = x[1]; c = x[2]; d = x[3];
  x[0] = a + c; x[1] = b + d; x[2] = a - c; x[3] = b - d;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    x[j] *= rsqrtf(128.f);
    x[j] = (isfinite(x[j]) ? x[j] : 0.f) / fmaxf(adaptive, 1e-12f);
  }
}

template <int World>
__global__ void taco_decode_ring_kernel_op2(
    FluxTacoConfig c, const __nv_bfloat16* local, __nv_bfloat16* output) {
  constexpr int groups_per_cta = 4;
  const int lane = threadIdx.x & 31;
  const int64_t group = int64_t(blockIdx.x) * groups_per_cta + threadIdx.x / 32;
  const int nt = (c.n + 127) / 128;
  const int64_t groups = int64_t(c.m / World) * nt;
  if (group >= groups) return;  // Whole-warp exit: no cross-warp barriers.
  const int64_t row = group / nt;
  const int col = int(group % nt) * 128 + lane;
  const int64_t chunk = int64_t(c.m / World) * c.n;
  const int64_t source_bytes = groups * 136;
  __nv_bfloat16 accum[4] = {};

  // The unchanged ring order is rank+1,...,rank-1,rank. All remote
  // contributions are quantized in ablation7 A. The local BF16 value is last.
  // The loop body remains compact to avoid fully duplicating World-1 H128
  // transforms in the instruction cache.
  #pragma unroll 1
  for (int i = 1; i < World; ++i) {
    int src = c.rank + i;
    if (src >= World) src -= World;
    const unsigned char* slot = c.peers[c.rank] + int64_t(src) * source_bytes;
    const float* quant = reinterpret_cast<const float*>(slot + groups * 128);
    const float* adaptive = quant + groups;
    float value[4];
    decode_group(slot + group * 128, quant[group], adaptive[group], value);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      // Preserve BOTH source BF16 conversion and per-add BF16 conversion.
      accum[j] = __float2bfloat16(
          __bfloat162float(accum[j]) +
          __bfloat162float(__float2bfloat16(value[j])));
    }
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    if (col + j * 32 < c.n) {
      const int64_t offset = row * c.n + col + j * 32;
      const float value = __bfloat162float(local[int64_t(c.rank) * chunk + offset]);
      accum[j] = __float2bfloat16(
          __bfloat162float(accum[j]) +
          __bfloat162float(__float2bfloat16(value)));
      output[offset] = accum[j];
    }
  }
}

}  // namespace flux_taco_op2

// This entry point is ONLY for all-remote quantization, as in ablation7 A.
// It does not consume a selective mask, change the wire format, or alter the
// caller's producer/publication/workspace-reuse barriers.
inline int taco_decode_reduce_op2(
    FluxTacoConfig c, const void* local, void* output, cudaStream_t stream) {
  if (!c.enabled || !local || !output ||
      (c.world != 2 && c.world != 4 && c.world != 8) ||
      c.rank < 0 || c.rank >= c.world || !c.peers[c.rank] ||
      c.m <= 0 || c.m % c.world || c.n <= 0 || c.n > 2147483520)
    return int(cudaErrorInvalidValue);
  const int64_t blocks = (taco_groups(c) + 3) / 4;
  if (blocks > 2147483647) return int(cudaErrorInvalidConfiguration);
  const auto* source = static_cast<const __nv_bfloat16*>(local);
  auto* destination = static_cast<__nv_bfloat16*>(output);
  switch (c.world) {
    case 2:
      flux_taco_op2::taco_decode_ring_kernel_op2<2>
          <<<unsigned(blocks), 128, 0, stream>>>(c, source, destination);
      break;
    case 4:
      flux_taco_op2::taco_decode_ring_kernel_op2<4>
          <<<unsigned(blocks), 128, 0, stream>>>(c, source, destination);
      break;
    case 8:
      flux_taco_op2::taco_decode_ring_kernel_op2<8>
          <<<unsigned(blocks), 128, 0, stream>>>(c, source, destination);
      break;
  }
  return int(cudaGetLastError());
}
