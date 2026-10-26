"""F4-tile: reuse physical-tile masks across rows of the exact fused decoder.

Only F4-tile receives these native transforms. F4 and F4-gather retain the
original F4 sources; gather overlap is an independent Python backward change.
"""
from pathlib import Path

from common import replace_once, require
import ma4_f_sources as f4

RUNTIME, HEADER, WRAPPER, DECODER = f4.RUNTIME, f4.HEADER, f4.WRAPPER, f4.DECODER
BASELINE_TUNING = dict(decoder_threads=128, decoder_blocks=2048, decoder_rows=1)
DEFAULT_TUNING = {role:dict(decoder_threads=128, decoder_blocks=512, decoder_rows=4)
                  for role in ('attn', 'mlp')}
ROLES = ('attn', 'mlp')
ROLE_K = dict(attn=512, mlp=2048)


def tuning_config(tuning=None):
    tuning = DEFAULT_TUNING if tuning is None else tuning
    require(isinstance(tuning, dict) and set(tuning) == set(ROLES),
            'MA6 tile tuning requires attn and mlp')
    result = {}
    for role in ROLES:
        item = tuning[role]
        require(isinstance(item, dict) and set(item) == set(BASELINE_TUNING),
                'MA6 only tunes decoder threads, blocks, and row tiles: ' + role)
        require(all(type(v) is int for v in item.values()), 'MA6 tile values must be integers')
        threads, blocks, rows = (item[key] for key in BASELINE_TUNING)
        require(threads in (64, 128, 256, 512), 'Unsupported MA6 decoder threads')
        require(rows in (1, 2, 4, 8, 16), 'Unsupported MA6 decoder rows')
        require(1 <= blocks <= (8192 // rows) // (threads // 32),
                'MA6 decoder blocks must be positive and no greater than the row-tile grid')
        result[role] = dict(decoder_threads=threads, decoder_blocks=blocks, decoder_rows=rows)
    return result


def kernel_name(role, tuning=None):
    tuning = tuning_config(tuning)
    require(role in ROLES, 'Unknown MA6 decoder role')
    if tuning[role] == BASELINE_TUNING:
        return 'taco_ma4f_decode_consumer_kernel'
    return 'taco_ma6_' + ('attention' if role == 'attn' else 'mlp') + '_decode_consumer_kernel'


def decode_entry(role, tuning):
    return ('taco_ma4f_decode_reduce' if tuning[role] == BASELINE_TUNING else
            'taco_ma6_decode_' + ('attention' if role == 'attn' else 'mlp'))


def decoder(source, tuning):
    tuning = tuning_config(tuning)
    signature = '''__global__ void taco_ma4f_decode_consumer_kernel(const __grid_constant__ FluxTacoConfig c,
    const __nv_bfloat16* local, __nv_bfloat16* output,
    const __nv_bfloat16* bias, const __nv_bfloat16* residual) {'''
    require(source.count(signature) == 1, 'MA6 requires the exact fused F4 kernel')
    original = source[source.index(signature):]
    group = replace_once(original, signature,
        '''__device__ __forceinline__ void taco_ma6_decode_consumer_group(const FluxTacoConfig& c,
    const __nv_bfloat16* local, __nv_bfloat16* output,
    const __nv_bfloat16* bias, const __nv_bfloat16* residual, int group) {''')
    group = replace_once(group, '  const int group = blockIdx.x * 4 + threadIdx.x / 32;\n', '')
    # The original kernel stays verbatim for any role with baseline geometry.
    # In the looped kernel the BF16-path return exits only this group helper.
    result = source + '\n' + group
    # The selection mask covers one physical 128x128 producer tile. Rows of a
    # decoder row tile share this mask, but each H128 group keeps its original
    # data, coefficient transforms, source folds and BF16 consumer rounding.
    mask_start = group.index('  const int row = group / 32;')
    arithmetic_start = group.index('  if (!any_quant) {', mask_start)
    mask = group[mask_start:arithmetic_start]
    masked_group = group[:mask_start] + group[arithmetic_start:]
    masked_group = replace_once(masked_group, 'taco_ma6_decode_consumer_group(',
                                 'taco_ma6_decode_consumer_masked_group(')
    masked_group = replace_once(masked_group,
        'const __nv_bfloat16* bias, const __nv_bfloat16* residual, int group) {',
        'const __nv_bfloat16* bias, const __nv_bfloat16* residual, int group,\n'
        '    const bool (&selected)[8], bool any_quant) {')
    if any(tuning[role]['decoder_rows'] > 1 for role in ROLES):
        result += '\n' + masked_group
    for role in ROLES:
        if tuning[role] == BASELINE_TUNING:
            continue
        rows = tuning[role]['decoder_rows']
        if rows == 1:
            body = '''  for (int group = blockIdx.x * kWarps + threadIdx.x / 32;
       group < 8192; group += gridDim.x * kWarps) {
    taco_ma6_decode_consumer_group(c, local, output, bias, residual, group);
  }
'''
        else:
            # Row tile boundaries divide 128; no tile can cross the source
            # selection boundary. The inner loop is kept rolled to bound
            # instruction size/register pressure for the quantized path.
            body = f'''  constexpr int kRows = {rows};
  for (int tile = blockIdx.x * kWarps + threadIdx.x / 32;
       tile < 8192 / kRows; tile += gridDim.x * kWarps) {{
    const int group = (tile / 32) * kRows * 32 + tile % 32;
{mask}    #pragma unroll 1
    for (int row_offset = 0; row_offset < kRows; ++row_offset) {{
      taco_ma6_decode_consumer_masked_group(c, local, output, bias, residual,
          group + row_offset * 32, selected, any_quant);
    }}
  }}
'''
        result += f'''
// Every warp owns complete H128 groups. Tile/row iterations are warp-uniform;
// unchanged full-mask shuffles never mix rows or change source reduction order.
__global__ void {kernel_name(role, tuning)}(const __grid_constant__ FluxTacoConfig c,
    const __nv_bfloat16* local, __nv_bfloat16* output,
    const __nv_bfloat16* bias, const __nv_bfloat16* residual) {{
  constexpr int kWarps = {tuning[role]['decoder_threads'] // 32};
{body}}}
'''
    return result


def runtime(source, tuning):
    tuning = tuning_config(tuning)
    require('taco_ma4f_protocol_version() { return 1; }' in source,
            'MA6 requires frozen F4 native consumption')
    result = source
    for role in ROLES:
        if tuning[role] == BASELINE_TUNING:
            continue
        params = tuning[role]
        result += f'''
extern "C" int {decode_entry(role, tuning)}(FluxTacoConfig c, const void* local,
    void* output, const void* bias, const void* residual, cudaStream_t stream) {{
  if (!c.enabled || !local || !output || !bias || !residual || !c.selected ||
      !taco_ma2_exact_shape(c) || !c.deferred_reuse_barrier || c.allow_capture)
    return int(cudaErrorInvalidValue);
  {kernel_name(role, tuning)}<<<{params['decoder_blocks']}, {params['decoder_threads']}, 0, stream>>>(c,
      static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output),
      static_cast<const __nv_bfloat16*>(bias), static_cast<const __nv_bfloat16*>(residual));
  return int(cudaGetLastError());
}}
'''
    result += '\nextern "C" int taco_ma6_protocol_version() { return 2; }\n'
    for field in BASELINE_TUNING:
        result += f'''extern "C" int taco_ma6_{field}(int k) {{
  return k == 512 ? {tuning['attn'][field]} : k == 2048 ? {tuning['mlp'][field]} : -2;
}}
'''
    return result


def header(source, tuning):
    tuning = tuning_config(tuning)
    for role in ROLES:
        if tuning[role] != BASELINE_TUNING:
            source += f'''\nextern "C" int {decode_entry(role, tuning)}(FluxTacoConfig,
    const void*, void*, const void*, const void*, cudaStream_t);
'''
    return source


def wrapper(source, tuning):
    tuning = tuning_config(tuning)
    source = replace_once(source, '  const int cr2_k = taco_cr2_bound_k();', f'''  const int cr2_k = taco_cr2_bound_k();
  // Immutable instance K, not mutable thread-local configuration on forward.
  const decltype(&taco_ma4f_decode_reduce) ma6_decode =
      cr2_k == 512 ? &{decode_entry('attn', tuning)} : &{decode_entry('mlp', tuning)};''')
    return replace_once(source, 'CUDA_CHECK(static_cast<cudaError_t>(taco_ma4f_decode_reduce(',
                        'CUDA_CHECK(static_cast<cudaError_t>(ma6_decode(')


def sources(reference, tuning=None):
    tuning = tuning_config(tuning)
    reference = Path(reference)
    return {name: transform((reference / name).read_text(), tuning)
            for name, transform in ((RUNTIME, runtime), (HEADER, header),
                                    (WRAPPER, wrapper), (DECODER, decoder))}
