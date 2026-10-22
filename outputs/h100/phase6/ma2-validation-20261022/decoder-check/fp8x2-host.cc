#include <cuda_fp8.h>
#include <cstdio>
int main() {
  unsigned finite = 0, pairs = 0;
  for (unsigned packed=0;packed<65536;++packed) {
    __half2_raw pair = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed), __NV_E4M3);
    __half_raw low = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(packed), __NV_E4M3);
    __half_raw high = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(packed >> 8), __NV_E4M3);
    if (pair.x != low.x || pair.y != high.x) {
      std::printf("mismatch packed=%u vector=%u,%u scalar=%u,%u\n",packed,pair.x,pair.y,low.x,high.x);
      return 1;
    }
    ++pairs;
    if ((packed&127)!=127 && ((packed>>8)&127)!=127) ++finite;
  }
  std::printf("{\"pairs_checked\":%u,\"finite_pairs_checked\":%u,\"bitwise_equal\":true,\"gpu_used\":false}\n",pairs,finite);
}
