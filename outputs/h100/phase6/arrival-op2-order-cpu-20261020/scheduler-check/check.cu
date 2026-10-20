
#include "src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp"
#include "phase6_host.hpp"
using Swizzle=cutlass::gemm::threadblock::ThreadblockSwizzleStreamKRankOffset;
__global__ void check_mapping_device(Swizzle sw, int *out) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<512) { auto coord=sw.get_tile_offset(i);out[i]=coord.m()*32+coord.n(); }
}
Swizzle make(int k,int m=2048) {
 return Swizzle(cutlass::gemm::GemmUniversalMode::kGemm,{m,4096,k},{128,128,32},1,2,132,132,2,2,2,8);
}
int main(){
 setenv("LOCAL_RANK","0",1);setenv("LOCAL_WORLD_SIZE","8",1);
 unsetenv("H100_OP2_ATTN_SPAN");unsetenv("H100_OP2_MLP_SPAN");
 bool bad=false;try{make(512);}catch(const std::invalid_argument&){bad=true;}
 if(!bad)return 1;
 auto other=make(1024);if(other.scope_shape)return 2;
 other=make(512,1024);if(other.scope_shape)return 3;
 const int spans[]={1,4,8,16,32};const char*strings[]={"1","4","8","16","32"};
 for(int a=0;a<5;++a)for(int b=0;b<5;++b){
  setenv("H100_OP2_ATTN_SPAN",strings[a],1);setenv("H100_OP2_MLP_SPAN",strings[b],1);
  for(int role=0;role<2;++role){
   auto sw=make(role?2048:512);
   if(!sw.scope_shape||sw.scope_pair_span!=spans[role?b:a]||sw.sk_tiles!=0||sw.dp_blocks!=512||
      sw.dp_first_wave_tiles!=1||sw.cohort_raster||sw.remap_block_indices)return 4;
   for(int rank=0;rank<8;++rank){int i=0,span=sw.scope_pair_span;
    for(int col=0;col<32;col+=span)for(int row=0;row<2;++row)for(int n=col;n<col+span;++n)for(int d=0;d<8;++d,++i)
     if(phase6_host_lookup(16,32,role?2048:512,rank,i)!=((d+rank)%8)*64+row*32+n)return 5;
   }
  }
 }
 setenv("H100_OP2_ATTN_SPAN","04",1);bad=false;
 try{make(512);}catch(const std::invalid_argument&){bad=true;}
 if(!bad)return 6;
}
