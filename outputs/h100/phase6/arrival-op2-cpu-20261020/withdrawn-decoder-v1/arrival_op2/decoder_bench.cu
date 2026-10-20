// TACO adaptation; see tools/a800/phase4/TACO_LICENSE.txt (BSD-3-Clause).
// Standalone test: link the UNMODIFIED pinned fused_v2 taco_runtime.cu.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "gemm_rs/taco_codec.cuh"
#include "decoder.cuh"

namespace {

void check(cudaError_t error, const char* where) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(error));
}

std::string quote(const std::string& text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                         << std::setfill('0') << unsigned(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  size_t size;
  explicit DeviceBuffer(size_t count) : size(count) {
    check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  void upload(const std::vector<T>& data) {
    if (data.size() != size) throw std::runtime_error("upload size mismatch");
    check(cudaMemcpy(ptr, data.data(), size * sizeof(T), cudaMemcpyHostToDevice),
          "cudaMemcpy upload");
  }
  std::vector<T> download() const {
    std::vector<T> result(size);
    check(cudaMemcpy(result.data(), ptr, size * sizeof(T), cudaMemcpyDeviceToHost),
          "cudaMemcpy download");
    return result;
  }
};

struct Events {
  cudaEvent_t begin{}, end{};
  Events() {
    check(cudaEventCreate(&begin), "cudaEventCreate begin");
    check(cudaEventCreate(&end), "cudaEventCreate end");
  }
  ~Events() { cudaEventDestroy(begin); cudaEventDestroy(end); }
};

uint16_t bf16_bits(float value) {
  // Independent CPU round-to-nearest-even BF16 for finite test data.
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fff + ((bits >> 16) & 1);
  return uint16_t(bits >> 16);
}

float bf16_float(uint16_t bits) {
  uint32_t word = uint32_t(bits) << 16;
  float result;
  std::memcpy(&result, &word, sizeof(result));
  return result;
}

uint32_t mix(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15;
  x *= 0x846ca68b; x ^= x >> 16;
  return x;
}

// Independent forward encoder uses the original 128-thread block codec,
// never the new warp transform. It creates real BF16-origin packets, including
// padded tail groups, for comparison with the arbitrary finite-packet tests.
__global__ void encode_sources(FluxTacoConfig c, const __nv_bfloat16* source) {
  __shared__ flux_taco::Scratch scratch;
  const int64_t groups = taco_groups(c);
  const int src = blockIdx.x / groups;
  const int64_t group = blockIdx.x % groups;
  const int nt = (c.n + 127) / 128;
  const int64_t row = group / nt;
  const int col = int(group % nt) * 128 + threadIdx.x;
  const int64_t chunk = int64_t(c.m / c.world) * c.n;
  const float value = col < c.n ?
      __bfloat162float(source[int64_t(src) * chunk + row * c.n + col]) : 0.f;
  unsigned char* slot = c.peers[c.rank] + int64_t(src) * taco_source_bytes(c);
  float* quant = reinterpret_cast<float*>(slot + groups * 128);
  float* adaptive = quant + groups;
  const int valid = min(128, c.n - int(group % nt) * 128);
  flux_taco::encode(value, valid, slot + group * 128,
                    quant + group, adaptive + group, scratch);
}

constexpr size_t guard_size = 32;
constexpr uint16_t guard_value = 0x7fc1;

struct Fixture {
  FluxTacoConfig config;
  int rows, n, world, rank;
  int64_t groups, chunk, source_bytes;
  std::vector<uint16_t> local;
  std::vector<unsigned char> packets;
  DeviceBuffer<unsigned char> gpu_packets;
  DeviceBuffer<uint16_t> gpu_local, old_output, new_output;

  Fixture(int world_, int rank_, int rows_, int n_, uint32_t seed)
      : rows(rows_), n(n_), world(world_), rank(rank_),
        groups(int64_t(rows) * ((n + 127) / 128)),
        chunk(int64_t(rows) * n), source_bytes(groups * 136),
        local(size_t(world * chunk)), packets(size_t(world * source_bytes)),
        gpu_packets(packets.size()), gpu_local(local.size()),
        old_output(size_t(chunk) + guard_size),
        new_output(size_t(chunk) + guard_size) {
    config.enabled = 1; config.world = world; config.rank = rank;
    config.m = rows * world; config.n = n;
    config.peers[rank] = gpu_packets.ptr;
    for (size_t i = 0; i < local.size(); ++i) {
      const uint32_t v = mix(uint32_t(i) ^ seed);
      const float scale[] = {0.0003f, 0.125f, 1.f, 8.f, 1024.f};
      local[i] = bf16_bits((float(int(v % 8193) - 4096) / 1024.f) *
                          scale[(i / 19) % 5]);
    }
    gpu_local.upload(local);
    const std::vector<uint16_t> sentinel(old_output.size, guard_value);
    old_output.upload(sentinel); new_output.upload(sentinel);
  }

  void random_packets(uint32_t seed) {
    const float quant_options[] = {1e-12f, 1e-7f, 0.0037f, 0.5f, 1.f, 31.f, 1e6f};
    const float adaptive_options[] = {1e-3f, 0.03125f, 1.f, 3.7f, 448.f, 1e3f};
    for (int src = 0; src < world; ++src) {
      const size_t start = size_t(src * source_bytes);
      for (int64_t g = 0; g < groups; ++g) {
        for (int k = 0; k < 128; ++k) {
          unsigned char value = mix(seed ^ uint32_t((src * groups + g) * 128 + k));
          if ((value & 0x7f) == 0x7f) value ^= 1;  // Exclude E4M3 NaN only.
          packets[start + g * 128 + k] = value;
        }
        const float quant = quant_options[mix(uint32_t(g) ^ seed) % 7];
        const float adaptive = adaptive_options[mix(uint32_t(g + src * 17)) % 6];
        std::memcpy(packets.data() + start + groups * 128 + g * 4, &quant, 4);
        std::memcpy(packets.data() + start + groups * 132 + g * 4, &adaptive, 4);
      }
    }
    gpu_packets.upload(packets);
  }

  void impulse_packets() {
    std::fill(packets.begin(), packets.end(), 0);
    for (int src = 0; src < world; ++src) {
      const size_t start = size_t(src * source_bytes);
      const float quant = std::ldexp(1.f, src % 4 - 2);
      const float adaptive = std::ldexp(1.f, src % 3 - 1);
      for (int64_t g = 0; g < groups; ++g) {
        const int coefficient = (src * 17 + rank * 7 + int(g)) & 127;
        packets[start + g * 128 + coefficient] = (src & 1) ? 0xb8 : 0x38;
        std::memcpy(packets.data() + start + groups * 128 + g * 4, &quant, 4);
        std::memcpy(packets.data() + start + groups * 132 + g * 4, &adaptive, 4);
      }
    }
    gpu_packets.upload(packets);
  }

  void encode_packets() {
    encode_sources<<<unsigned(world * groups), 128>>>(
        config, reinterpret_cast<const __nv_bfloat16*>(gpu_local.ptr));
    check(cudaGetLastError(), "encode_sources");
    check(cudaDeviceSynchronize(), "encode_sources synchronize");
  }

  void launch(bool optimized) {
    const int error = optimized ?
        taco_decode_reduce_op2(config, gpu_local.ptr, new_output.ptr, nullptr) :
        taco_decode_reduce(config, gpu_local.ptr, old_output.ptr, nullptr);
    check(cudaError_t(error), optimized ? "op2 decode" : "pinned scalar decode");
  }

  uint16_t impulse_expected(int64_t index) const {
    const int row = int(index / n), col = int(index % n);
    const int64_t group = int64_t(row) * ((n + 127) / 128) + col / 128;
    uint16_t accum = 0;
    for (int i = 1; i <= world; ++i) {
      const int src = (rank + i) % world;
      uint16_t value;
      if (src == rank) value = local[size_t(src * chunk + index)];
      else {
        const int coefficient = (src * 17 + rank * 7 + int(group)) & 127;
        const int parity = __builtin_parity(unsigned(coefficient & (col & 127)));
        const float sign = ((src & 1) ^ parity) ? -1.f : 1.f;
        // A Hadamard matrix column is +/-1, so this independent oracle
        // does not implement the butterfly used by either GPU decoder.
        const float magnitude =
            std::ldexp(1.f, src % 4 - 2) / std::sqrt(128.f) /
            std::ldexp(1.f, src % 3 - 1);
        value = bf16_bits(sign * magnitude);
      }
      accum = bf16_bits(bf16_float(accum) + bf16_float(value));
    }
    return accum;
  }

  size_t verify(bool analytic) const {
    const auto before = old_output.download();
    const auto after = new_output.download();
    for (int64_t i = 0; i < chunk; ++i) {
      if (before[size_t(i)] != after[size_t(i)] ||
          !std::isfinite(bf16_float(after[size_t(i)])) ||
          (analytic && after[size_t(i)] != impulse_expected(i))) {
        std::ostringstream error;
        error << "bitwise/finite/analytic mismatch world=" << world << " rank=" << rank
              << " rows=" << rows << " n=" << n << " index=" << i
              << " scalar=0x" << std::hex << before[size_t(i)]
              << " op2=0x" << after[size_t(i)];
        if (analytic) error << " oracle=0x" << impulse_expected(i);
        throw std::runtime_error(error.str());
      }
    }
    for (size_t i = size_t(chunk); i < before.size(); ++i)
      if (before[i] != guard_value || after[i] != guard_value)
        throw std::runtime_error("output tail guard overwritten");
    return size_t(chunk);
  }
};

struct Correctness {
  int cases = 0, paired_launches = 0;
  size_t elements = 0;
};

Correctness run_correctness() {
  Correctness result;
  // Non-128 row counts deliberately exercise the final partial four-warp CTA;
  // decoder-only input contract permits them even though production's GEMM
  // configure requires M % (128*TP) == 0.
  const int shapes[][2] = {
      {1, 8}, {3, 120}, {5, 128}, {3, 136}, {7, 264}, {128, 2048},
      {256, 4096}};
  for (int world : {2, 4, 8}) {
    for (int rank = 0; rank < world; ++rank) {
      for (const auto& shape : shapes) {
        // Exact output shape of the H100 TP8 Attention and MLP experiment.
        if (shape[1] == 4096 && world != 8) continue;
        Fixture fixture(world, rank, shape[0], shape[1],
                        uint32_t(0x5eed + world * 97 + rank * 3 + shape[1]));
        for (int mode = 0; mode < 3; ++mode) {
          if (mode == 0) fixture.impulse_packets();
          else if (mode == 1) fixture.random_packets(uint32_t(913 + rank * 77 + shape[1]));
          else fixture.encode_packets();
          // Repeated launch order alternates, and new input data also replaces
          // the previous mode's packets. Stale output or unwritten tails fail.
          for (int repeat = 0; repeat < 3; ++repeat) {
            fixture.launch(bool(repeat & 1));
            fixture.launch(!bool(repeat & 1));
            check(cudaDeviceSynchronize(), "correctness synchronize");
            result.elements += fixture.verify(mode == 0);
            ++result.paired_launches;
          }
          ++result.cases;
        }
      }
    }
  }
  return result;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t half = values.size() / 2;
  return values.size() % 2 ? values[half] : (values[half - 1] + values[half]) * 0.5;
}

double measure(Fixture& fixture, bool optimized, int iterations, Events& events) {
  check(cudaEventRecord(events.begin), "record begin");
  for (int i = 0; i < iterations; ++i) fixture.launch(optimized);
  check(cudaEventRecord(events.end), "record end");
  check(cudaEventSynchronize(events.end), "synchronize end");
  float elapsed_ms = 0.f;
  check(cudaEventElapsedTime(&elapsed_ms, events.begin, events.end), "elapsed");
  if (!(elapsed_ms > 0.f)) throw std::runtime_error("nonpositive CUDA event time");
  return double(elapsed_ms) * 1000.0 / iterations;
}

void array_json(std::ostream& out, const std::vector<double>& values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}

std::string benchmark(int world, int m, int n, int samples, int iterations) {
  Fixture fixture(world, 0, m / world, n, uint32_t(m * 3 + world));
  fixture.encode_packets();
  for (int i = 0; i < 20; ++i) {
    fixture.launch(false); fixture.launch(true);
  }
  check(cudaDeviceSynchronize(), "benchmark warmup");
  fixture.verify(false);
  Events events;
  std::vector<double> before, after, ratios;
  for (int sample = 0; sample < samples; ++sample) {
    double a, b;
    if (sample & 1) {
      b = measure(fixture, true, iterations, events);
      a = measure(fixture, false, iterations, events);
    } else {
      a = measure(fixture, false, iterations, events);
      b = measure(fixture, true, iterations, events);
    }
    before.push_back(a); after.push_back(b); ratios.push_back(a / b);
  }
  fixture.verify(false);
  std::ostringstream out;
  out << std::setprecision(9)
      << "{\"world\":" << world << ",\"rank\":0,\"m\":" << m << ",\"n\":" << n
      << ",\"samples\":" << samples << ",\"iterations_per_sample\":" << iterations
      << ",\"measurement\":\"cuda_event_warm_cache_decoder_only\""
      << ",\"scalar_median_us\":" << median(before)
      << ",\"op2_median_us\":" << median(after)
      << ",\"paired_median_speedup\":" << median(ratios)
      << ",\"paired_median_latency_reduction_pct\":" << (1. - 1. / median(ratios)) * 100.
      << ",\"scalar_us\":";
  array_json(out, before);
  out << ",\"op2_us\":"; array_json(out, after);
  out << '}';
  return out.str();
}

int argument_int(const char* text) {
  size_t consumed = 0;
  const int value = std::stoi(text, &consumed);
  if (consumed != std::strlen(text) || value <= 0)
    throw std::runtime_error("expected positive integer");
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool check_only = false;
    int samples = 15, iterations = 100;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--check-only") check_only = true;
      else if (arg == "--samples" && i + 1 < argc) samples = argument_int(argv[++i]);
      else if (arg == "--iterations" && i + 1 < argc) iterations = argument_int(argv[++i]);
      else throw std::runtime_error("usage: decoder_bench [--check-only] "
                                    "[--samples N] [--iterations N]");
    }
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability != cudaSuccess || device_count == 0) {
      std::cout << "{\"schema\":\"arrival_op2.decoder.v1\","
                   "\"status\":\"gpu_unavailable\",\"completed\":false,"
                   "\"correctness_passed\":false,\"bitwise_equal\":false,"
                   "\"correctness\":{\"status\":\"not_run\"},"
                   "\"benchmarks\":[],\"reason\":"
                << quote(availability == cudaSuccess ? "no visible CUDA device" :
                         cudaGetErrorString(availability)) << "}\n";
      return 77;
    }
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    check(cudaSetDevice(0), "cudaSetDevice");
    int runtime = 0, driver = 0;
    check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    const Correctness correctness = run_correctness();
    std::vector<std::string> timings;
    if (!check_only) {
      for (int world : {2, 4, 8})
        for (int m : {2048, 8192})
          timings.push_back(benchmark(world, m, 2048, samples, iterations));
      timings.push_back(benchmark(8, 2048, 4096, samples, iterations));
    }
    std::cout << "{\"schema\":\"arrival_op2.decoder.v1\",\"status\":\"passed\","
                 "\"completed\":true,\"correctness_passed\":true,\"bitwise_equal\":true,"
                 "\"device\":{\"name\":" << quote(properties.name)
              << ",\"major\":" << properties.major << ",\"minor\":" << properties.minor
              << ",\"sm_count\":" << properties.multiProcessorCount
              << ",\"cuda_runtime\":" << runtime << ",\"cuda_driver\":" << driver
              << "},\"correctness\":{\"status\":\"passed\",\"comparison\":\"bitwise_bf16\","
                 "\"worlds\":[2,4,8],\"all_ranks\":true,\"packet_modes\":"
                 "[\"analytic_hadamard_impulse\",\"finite_e4m3_random\","
                 "\"pinned_block_encode_bf16\"],\"exact_h100_shape\":"
                 "{\"world\":8,\"m\":2048,\"n\":4096,\"all_ranks\":true},"
                 "\"cases\":" << correctness.cases
              << ",\"paired_launches\":" << correctness.paired_launches
              << ",\"compared_elements\":" << correctness.elements
              << ",\"output_tail_guard_elements\":" << guard_size
              << "},\"benchmarks\":[";
    for (size_t i = 0; i < timings.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << timings[i];
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cout << "{\"schema\":\"arrival_op2.decoder.v1\",\"status\":\"failed\","
                 "\"completed\":false,\"correctness_passed\":false,\"bitwise_equal\":false,"
                 "\"error\":" << quote(error.what()) << "}\n";
    return 1;
  }
}
