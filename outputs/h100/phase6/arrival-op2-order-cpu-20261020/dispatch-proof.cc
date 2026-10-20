#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"
int main(){
using S=cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
for(int k: {512,2048}) S(cutlass::gemm::GemmUniversalMode::kGemm,{2048,4096,k},{128,128,32},1,2,132,132,2,2,2,8).Print();
}
