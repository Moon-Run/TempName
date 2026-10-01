#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA(call) do { auto s=(call); if(s!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(s));return 1;} } while(0)
#define NCCL(call) do { auto s=(call); if(s!=ncclSuccess){fprintf(stderr,"NCCL: %s\n",ncclGetErrorString(s));return 1;} } while(0)
int main(){
  constexpr int nranks=2;
  constexpr size_t count=1<<20;
  int devices[2]={0,1}; ncclComm_t comms[2]; cudaStream_t streams[2];
  __nv_bfloat16 *send[2],*recv[2];
  int version=0; NCCL(ncclGetVersion(&version));
  if(version!=NCCL_VERSION_CODE) return 2;
  NCCL(ncclCommInitAll(comms,nranks,devices));
  std::vector<__nv_bfloat16> host(count*nranks);
  for(int r=0;r<nranks;++r){
    CUDA(cudaSetDevice(r)); CUDA(cudaStreamCreate(&streams[r]));
    CUDA(cudaMalloc(&send[r],host.size()*sizeof(host[0])));
    CUDA(cudaMalloc(&recv[r],count*sizeof(host[0])));
    for(size_t i=0;i<host.size();++i)host[i]=__float2bfloat16(float(r+1)+float(i%7)*0.125f);
    CUDA(cudaMemcpy(send[r],host.data(),host.size()*sizeof(host[0]),cudaMemcpyHostToDevice));
  }
  NCCL(ncclGroupStart());
  for(int r=0;r<nranks;++r){CUDA(cudaSetDevice(r));NCCL(ncclReduceScatter(send[r],recv[r],count,ncclBfloat16,ncclSum,comms[r],streams[r]));}
  NCCL(ncclGroupEnd());
  for(int r=0;r<nranks;++r){
    CUDA(cudaSetDevice(r));CUDA(cudaStreamSynchronize(streams[r]));
    CUDA(cudaMemcpy(host.data(),recv[r],count*sizeof(host[0]),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<count;++i){float expected=3.f+float((r*count+i)%7)*0.25f;if(__bfloat162float(host[i])!=expected){fprintf(stderr,"rank=%d index=%zu incorrect\n",r,i);return 3;}}
    NCCL(ncclCommDestroy(comms[r]));CUDA(cudaFree(send[r]));CUDA(cudaFree(recv[r]));CUDA(cudaStreamDestroy(streams[r]));
  }
  printf("PASS: project NCCL %d, 2-GPU BF16 ReduceScatter, %zu elements/rank\n",version,count);
}
