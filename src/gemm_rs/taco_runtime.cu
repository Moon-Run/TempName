// TACO adaptation; see tools/a800/phase4/TACO_LICENSE.txt.
#include "gemm_rs/taco_codec.cuh"

static thread_local FluxTacoConfig active;
extern "C" FluxTacoConfig taco_config() { return active; }
extern "C" void taco_reset() { active = FluxTacoConfig{}; }
extern "C" int taco_instance_config_supported() { return 1; }
extern "C" void taco_set_config(FluxTacoConfig c) { active = c; }
extern "C" void taco_allow_capture() { active.allow_capture = 1; }
extern "C" void taco_defer_reuse_barrier() { active.deferred_reuse_barrier = 1; }
extern "C" int taco_placement() {
#ifdef FLUX_TACO_SEPARATE
  return 2;
#else
  return 1;
#endif
}
extern "C" int taco_configure(void** peers, int rank, int world, int m, int n) {
  if (active.enabled || !peers || (world != 2 && world != 4 && world != 8) ||
      rank < 0 || rank >= world || m <= 0 || m % (128 * world) || n <= 0 || n % 8)
    return int(cudaErrorInvalidValue);
  FluxTacoConfig c;
  c.rank = rank; c.world = world; c.m = m; c.n = n;
  for (int r = 0; r < world; ++r) {
    if (!peers[r]) return int(cudaErrorInvalidValue);
    c.peers[r] = static_cast<unsigned char*>(peers[r]);
  }
  c.enabled = 1; active = c;
  return int(cudaSuccess);
}

extern "C" int taco_configure_selective(void** peers, int rank, int world, int m, int n,
                                        const unsigned char* selected) {
  if (!selected || taco_placement() != 1) return int(cudaErrorInvalidValue);
  int result = taco_configure(peers, rank, world, m, n);
  if (!result) active.selected = selected;
  return result;
}

__global__ void taco_encode_scatter_kernel(FluxTacoConfig c, const __nv_bfloat16* staged) {
  __shared__ flux_taco::Scratch scratch;
  const int64_t groups = taco_groups(c);
  const int dst = blockIdx.x / groups;
  if (dst == c.rank) return;  // Local contribution stays BF16 in its original slot.
  const int64_t group = blockIdx.x % groups;
  const int nt = (c.n + 127) / 128;
  const int row = group / nt;
  const int col = (group % nt) * 128 + threadIdx.x;
  const int global_row = dst * (c.m / c.world) + row;
  float value = col < c.n ? __bfloat162float(staged[int64_t(global_row)*c.n+col]) : 0.f;
  unsigned char* slot = c.peers[dst] + c.rank*taco_source_bytes(c);
  float* qs = reinterpret_cast<float*>(slot + groups*128);
  float* as = qs + groups;
  const int valid = min(128, c.n - int(group % nt)*128);
  flux_taco::encode(value, valid, slot+group*128, qs+group, as+group, scratch);
}

extern "C" int taco_encode_scatter(FluxTacoConfig c, const void* staged, cudaStream_t stream) {
  if (!c.enabled || !staged) return int(cudaErrorInvalidValue);
  taco_encode_scatter_kernel<<<taco_groups(c)*c.world, 128, 0, stream>>>(
      c, static_cast<const __nv_bfloat16*>(staged));
  return int(cudaGetLastError());
}

template <bool ModelShape>
__global__ void taco_decode_ring_kernel(FluxTacoConfig c, const __nv_bfloat16* local,
                                        __nv_bfloat16* output) {
  // Each warp owns an H128 group. Four groups per CTA avoid cross-warp
  // scratch/barriers, including on the common unquantized selective path.
  const int lane = threadIdx.x & 31;
  const int world = ModelShape ? 4 : c.world;
  const int n = ModelShape ? 2048 : c.n;
  const int m = ModelShape ? 2048 : c.m;
  const int group = blockIdx.x * 4 + threadIdx.x / 32;
  const int nt = (n + 127) / 128;
  const int64_t groups = int64_t(m / world) * nt, source_bytes = groups * 136;
  if (group >= groups) return;  // Uniform within a warp.
  const int row = group / nt, col = (group % nt) * 128 + lane;
  const int64_t chunk = int64_t(m / world) * n;
  const int physical_tile = ((c.rank * (m / world) + row) / 128) * nt + group % nt;
  bool any_quant = !c.selected;
  if (c.selected) {
    #pragma unroll
    for (int src = 0; src < (ModelShape ? 4 : c.world); ++src)
      any_quant |= src != c.rank && c.selected[src * (m / 128) * nt + physical_tile];
  }
  if (!any_quant && (group % nt + 1) * 128 <= n) {
    // Most sparse-mask groups are entirely BF16. Four adjacent elements per
    // lane give one aligned 64-bit transaction and two packed BF16 additions.
    const int64_t offset = int64_t(row) * n + (group % nt) * 128 + lane * 4;
    uint2 packed = make_uint2(0, 0);
    auto* sum = reinterpret_cast<__nv_bfloat162*>(&packed);
    #pragma unroll
    for (int i = 1; i <= (ModelShape ? 4 : c.world); ++i) {
      int src = c.rank + i;
      if (src >= world) src -= world;
      uint2 value = *reinterpret_cast<const uint2*>(local + src * chunk + offset);
      auto* x = reinterpret_cast<__nv_bfloat162*>(&value);
      sum[0] = __hadd2(sum[0], x[0]);
      sum[1] = __hadd2(sum[1], x[1]);
    }
    *reinterpret_cast<uint2*>(output + offset) = packed;
    return;
  }
  __nv_bfloat16 accum[4] = {};
  // Match Flux ring_reduction=True: rank+1,...,rank with BF16 rounding
  // after decoding each source and after every addition.
  #pragma unroll
  for (int i = 1; i <= (ModelShape ? 4 : c.world); ++i) {
    int src = c.rank + i;
    if (src >= world) src -= world;
    float value[4];
    if (!taco_selected(c, src, c.rank, physical_tile)) {
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        value[j] = col + j * 32 < n ?
            __bfloat162float(local[src * chunk + int64_t(row) * n + col + j * 32]) : 0.f;
    } else {
      const unsigned char* slot = c.peers[c.rank] + src * source_bytes;
      const float* qs = reinterpret_cast<const float*>(slot + groups * 128);
      const float* as = qs + groups;
      flux_taco::decode_warp(slot + group * 128, qs[group], as[group], value);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      accum[j] = __float2bfloat16(__bfloat162float(accum[j]) +
                                __bfloat162float(__float2bfloat16(value[j])));
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    if (col + j * 32 < n) output[int64_t(row) * n + col + j * 32] = accum[j];
}

// A block stays inside one physical 128x128 tile, sharing its selection
// decision across Rows H128 groups. This avoids thousands of tiny blocks and
// dynamic source loops on the common TP4 selective path. Ring order and BF16
// rounding are identical to the generic decoder; N tails keep the generic path.
template <int Rows, bool RowMajor = false>
__global__ void taco_decode_tile4_kernel(FluxTacoConfig c, const __nv_bfloat16* local,
                                        __nv_bfloat16* output) {
  constexpr int world = 4;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const int nt = c.n / 128;
  const int tile = blockIdx.x / (128 / Rows);
  const int col_tile = RowMajor ? blockIdx.x : tile % nt;
  const int first_row = RowMajor ? blockIdx.y * Rows :
      (tile / nt) * 128 + (blockIdx.x % (128 / Rows)) * Rows;
  const int physical_tile = (c.rank * (c.m / (128 * world)) + first_row / 128) * nt + col_tile;
  const int tiles = (c.m / 128) * nt;
  const int64_t chunk = int64_t(c.m / world) * c.n;
  const int64_t groups = int64_t(c.m / world) * nt;
  bool selected[world];
  bool any_quant = false;
  #pragma unroll
  for (int src = 0; src < world; ++src) {
    selected[src] = src != c.rank && c.selected[src * tiles + physical_tile];
    any_quant |= selected[src];
  }
  if (!any_quant) {
    #pragma unroll 1
    for (int row = first_row + warp; row < first_row + Rows; row += 4) {
      const int64_t offset = int64_t(row) * c.n + col_tile * 128 + lane * 4;
      uint2 packed = make_uint2(0, 0);
      auto* sum = reinterpret_cast<__nv_bfloat162*>(&packed);
      #pragma unroll
      for (int i = 1; i <= world; ++i) {
        const int src = (c.rank + i) & 3;
        uint2 value = *reinterpret_cast<const uint2*>(local + src * chunk + offset);
        auto* x = reinterpret_cast<__nv_bfloat162*>(&value);
        sum[0] = __hadd2(sum[0], x[0]);
        sum[1] = __hadd2(sum[1], x[1]);
      }
      *reinterpret_cast<uint2*>(output + offset) = packed;
    }
    return;
  }
  #pragma unroll 1
  for (int row = first_row + warp; row < first_row + Rows; row += 4) {
    const int64_t group = int64_t(row) * nt + col_tile;
    __nv_bfloat16 accum[4] = {};
    #pragma unroll
    for (int i = 1; i <= world; ++i) {
      const int src = (c.rank + i) & 3;
      float value[4];
      if (!selected[src]) {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
          value[j] = __bfloat162float(local[src * chunk + int64_t(row) * c.n +
                                           col_tile * 128 + lane + j * 32]);
      } else {
        const unsigned char* slot = c.peers[c.rank] + src * groups * 136;
        const float* qs = reinterpret_cast<const float*>(slot + groups * 128);
        const float* as = qs + groups;
        flux_taco::decode_warp(slot + group * 128, qs[group], as[group], value);
      }
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        accum[j] = __float2bfloat16(__bfloat162float(accum[j]) +
                                  __bfloat162float(__float2bfloat16(value[j])));
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      output[int64_t(row) * c.n + col_tile * 128 + lane + j * 32] = accum[j];
  }
}

// Diagnostic A/B switch, scoped to the calling host thread. It changes launch
// geometry only, never the wire protocol or the immutable instance config.
// Enable only the two measured large-output shapes by default. Small H and
// the original M2048 shape retain the generic/specialized decoder.
static thread_local int decode_variant = -1;
extern "C" int taco_set_decode_variant(int variant) {
  if (variant < -1 || variant > 3) return int(cudaErrorInvalidValue);
  decode_variant = variant;
  return int(cudaSuccess);
}

extern "C" int taco_decode_reduce(FluxTacoConfig c, const void* local, void* output,
                                   cudaStream_t stream) {
  if (!c.enabled || !local || !output) return int(cudaErrorInvalidValue);
  const int variant = decode_variant < 0 ?
      ((c.m == 8192 && (c.n == 2048 || c.n == 4096)) ? 3 : 0) : decode_variant;
  if (c.world == 4 && c.selected && c.n % 128 == 0 && variant != 0) {
    const int blocks = (c.m / (4 * 128)) * (c.n / 128);
    if (variant == 1)
      taco_decode_tile4_kernel<16><<<blocks * 8, 128, 0, stream>>>(
          c, static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
    else if (variant == 2)
      taco_decode_tile4_kernel<32><<<blocks * 4, 128, 0, stream>>>(
          c, static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
    else
      taco_decode_tile4_kernel<16, true><<<dim3(c.n / 128, c.m / (4 * 16)), 128, 0, stream>>>(
          c, static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
  } else if (c.world == 4 && c.m == 2048 && c.n == 2048)
    taco_decode_ring_kernel<true><<<(taco_groups(c) + 3) / 4, 128, 0, stream>>>(
        c, static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
  else
    taco_decode_ring_kernel<false><<<(taco_groups(c) + 3) / 4, 128, 0, stream>>>(
        c, static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
  return int(cudaGetLastError());
}
