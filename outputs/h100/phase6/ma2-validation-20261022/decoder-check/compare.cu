#include "gemm_rs/taco_codec.cuh"
#include "ma1_decode.cuh"
#include "ma2_decode.cuh"
extern "C" cudaError_t compare_standalone(FluxTacoConfig c, const __nv_bfloat16* local, __nv_bfloat16* output, cudaStream_t stream, int mode) {
  if (!taco_ma2_exact_shape(c) || !local || !output || (mode != 1 && mode != 2)) return cudaErrorInvalidValue;
  if (mode == 1) taco_ma1_decode_kernel<true><<<2048,128,0,stream>>>(c,local,output);
  else taco_ma2_decode_kernel<<<2048,128,0,stream>>>(c,local,output);
  return cudaGetLastError();
}
