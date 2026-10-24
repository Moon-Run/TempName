#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
__global__ void probe(float* x) { x[threadIdx.x] = 1.0f; }
