// TACO adaptation; see tools/a800/phase4/TACO_LICENSE.txt.
#include "gemm_rs/taco_codec.cuh"
#include "gemm_rs/arrival_op3_v2_decode.cuh"

static thread_local FluxTacoConfig active;
extern "C" FluxTacoConfig taco_config() { return active; }
extern "C" void taco_reset() { active = FluxTacoConfig{}; }
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

__global__ void taco_decode_ring_kernel(FluxTacoConfig c, const __nv_bfloat16* local,
                                        __nv_bfloat16* output) {
  __shared__ flux_taco::Scratch scratch;
  const int64_t group = blockIdx.x;
  const int nt = (c.n + 127) / 128;
  const int row = group / nt, col = (group % nt) * 128 + threadIdx.x;
  const int64_t groups = taco_groups(c), source_bytes = taco_source_bytes(c);
  const int64_t chunk = int64_t(c.m / c.world) * c.n;
  __nv_bfloat16 accum = __float2bfloat16(0.f);
  // Match Flux ring_reduction=True: rank+1,...,rank with BF16 rounding
  // after decoding each source and after every addition.
  for (int i = 1; i <= c.world; ++i) {
    int src = (c.rank + i) % c.world;
    float value;
    if (src == c.rank) {
      value = col < c.n ? __bfloat162float(local[src * chunk + int64_t(row) * c.n + col]) : 0.f;
    } else {
      const unsigned char* slot = c.peers[c.rank] + src * source_bytes;
      const float* qs = reinterpret_cast<const float*>(slot + groups * 128);
      const float* as = qs + groups;
      value = flux_taco::decode(slot + group * 128, qs[group], as[group], scratch);
    }
    accum = __float2bfloat16(__bfloat162float(accum) + __bfloat162float(__float2bfloat16(value)));
    __syncthreads();
  }
  if (col < c.n) output[int64_t(row) * c.n + col] = accum;
}

extern "C" int taco_decode_reduce(FluxTacoConfig c, const void* local, void* output,
                                   cudaStream_t stream) {
  if (!c.enabled || !local || !output) return int(cudaErrorInvalidValue);
  taco_decode_ring_kernel<<<taco_groups(c), 128, 0, stream>>>(
      c, static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
  return int(cudaGetLastError());
}

extern "C" int taco_op3_protocol_version() { return 2; }
extern "C" int taco_op3_configure(uint32_t epoch, int overlap, int rounds, int diagnostics) {
  if (!active.enabled || active.world != 8 || active.m != 2048 || active.n != 4096 ||
      taco_placement() != 1 || epoch == 0 || epoch > 0x7fffffffU ||
      (overlap != 0 && overlap != 1) || rounds < 1 || rounds > 4 ||
      (diagnostics != 0 && diagnostics != 1) ||
      taco_source_bytes(active) < kOp3ControlBytes)
    return int(cudaErrorInvalidValue);
  active.op3_epoch = epoch;
  active.op3_overlap = overlap;
  active.op3_rounds = rounds;
  active.op3_diagnostics = diagnostics;
  return int(cudaSuccess);
}
extern "C" int taco_op3_prepare(FluxTacoConfig c, cudaStream_t stream) {
  if (!c.enabled || !c.op3_epoch) return int(cudaErrorInvalidValue);
  // ready, group-done and stop all use epochs: no per-call control reset race.
  return c.op3_diagnostics ? int(cudaMemsetAsync(taco_op3_stats(c), 0,
      kOp3StatsCount * sizeof(unsigned long long), stream)) : int(cudaSuccess);
}
__global__ void taco_op3_v2_stop(FluxTacoConfig c) {
  auto* p = taco_op3_stop_ptr(c);
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(c.op3_epoch) : "memory");
  if (c.op3_diagnostics) taco_op3_stats(c)[5] = taco_op3_clock();
}
extern "C" int taco_op3_stop(FluxTacoConfig c, cudaStream_t stream) {
  if (!c.enabled || !c.op3_epoch) return int(cudaErrorInvalidValue);
  taco_op3_v2_stop<<<1, 1, 0, stream>>>(c);
  return int(cudaGetLastError());
}

// At most the first 64 physical fragments are eligible (1024 H128 groups).
// Every round visits the SAME set; group-done prevents expanding the budget.
// Four warps decode four rows concurrently, with at most four rows per warp.
// There is no spin on producers. Late CTAs and later rows stop taking work as
// soon as the main stream's post-GEMM stop epoch is observed.
__global__ void taco_op3_v2_try_decode(FluxTacoConfig c, const __nv_bfloat16* local,
                                      __nv_bfloat16* output) {
  __shared__ int ready;
  const int fragment = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  if (threadIdx.x == 0)
    ready = !taco_op3_stopped(c) && taco_op3_is_ready(c, fragment);
  __syncthreads();  // CTA acquire visibility for all participating payload readers.
  if (!ready) return;
  for (int compact = warp; compact < 16; compact += 4) {
    const int group = taco_op3_group(fragment, compact);
    int stopped = 0, done = 0;
    if (lane == 0) {
      stopped = taco_op3_stopped(c);
      done = taco_op3_done(c)[group] == c.op3_epoch;
    }
    stopped = __shfl_sync(0xffffffff, stopped, 0);
    if (stopped) return;  // Warp-uniform exit; no CTA barrier remains below.
    done = __shfl_sync(0xffffffff, done, 0);
    if (done) continue;
    if (c.op3_diagnostics && lane == 0)
      taco_op3_first(taco_op3_stats(c), taco_op3_clock());
    op3_v2_decode_warp(c, local, output, group);
    __syncwarp();  // Finish every output store before publishing this group's done.
    if (lane == 0) {
      taco_op3_done(c)[group] = c.op3_epoch;
      if (c.op3_diagnostics) {
        const auto now = taco_op3_clock();
        taco_op3_first(taco_op3_stats(c) + 3, now);
        atomicMax(taco_op3_stats(c) + 4, now);
        atomicAdd(taco_op3_stats(c) + 2, 1ULL);
        if (taco_op3_stopped(c)) atomicAdd(taco_op3_stats(c) + 6, 1ULL);
      }
    }
  }
}

// Same grouping as try: one fragment per CTA, four H128 warps. Tail can poll
// remote producers only after its OWN GEMM and finite auxiliary stream end.
__global__ void taco_op3_v2_tail_decode(FluxTacoConfig c, const __nv_bfloat16* local,
                                       __nv_bfloat16* output) {
  __shared__ int pending;
  const int fragment = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  if (threadIdx.x == 0) {
    pending = 0;
    for (int compact = 0; compact < 16; ++compact)
      pending |= taco_op3_done(c)[taco_op3_group(fragment, compact)] != c.op3_epoch;
    if (pending)
      while (!taco_op3_is_ready(c, fragment)) __nanosleep(64);
  }
  __syncthreads();
  if (!pending) return;
  for (int compact = warp; compact < 16; compact += 4) {
    const int group = taco_op3_group(fragment, compact);
    int done = lane == 0 ? taco_op3_done(c)[group] == c.op3_epoch : 0;
    done = __shfl_sync(0xffffffff, done, 0);
    if (!done) op3_v2_decode_warp(c, local, output, group);
  }
}
extern "C" int taco_op3_try(FluxTacoConfig c, const void* local, void* output,
                           cudaStream_t stream) {
  if (!c.enabled || !c.op3_epoch || !local || !output) return int(cudaErrorInvalidValue);
  for (int round = 0; round < c.op3_rounds; ++round) {
    taco_op3_v2_try_decode<<<kOp3EarlyCtas, 128, 0, stream>>>(c,
        static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return int(error);
  }
  return int(cudaSuccess);
}
extern "C" int taco_op3_tail(FluxTacoConfig c, const void* local, void* output,
                            cudaStream_t stream) {
  if (!c.enabled || !c.op3_epoch || !local || !output) return int(cudaErrorInvalidValue);
  taco_op3_v2_tail_decode<<<kOp3Fragments, 128, 0, stream>>>(c,
      static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
  return int(cudaGetLastError());
}
