#include "gemm_rs/gemm_v2_reduce_scatter.hpp"
namespace bytedance::flux {
using namespace cute;
using Meta = decltype(make_gemm_meta(
    make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{}),
    _Sm80{}, _A100{}, _ReduceScatter{}, _RCR{}, _GemmV2{}, None{},
    make_reduce_scatter_meta(_False{}, _IntraNode{})));
using HP = decltype(materialize_hparams(Meta{}, make_gemm_hparams(
    Auto{}, Auto{}, Shape<_128, _128, _32>{}, Auto{}, _3{})));
using Kernel = decltype(GemmV2ReduceScatter_Kernel<Meta, HP>().gemm_kernel());
using Device = GemmV2ReduceScatter_Device<Meta, HP, Kernel>;
static_assert(Kernel::kThreadCount == 128, "MA12 producer must retain four warps");
extern "C" void ma12_gemm_compile_probe(std::any const &args, void *workspace,
                                      cudaStream_t stream) {
  Device op;
  op.run(args, workspace, stream);
}
}
