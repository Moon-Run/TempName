"""F4 numerical-source preservation and complete warp-group tile coverage."""
import copy
from collections import Counter
import unittest

from common import HERE
import ma4_f_sources as f4
import ma6_sources as source


class TileSourceTests(unittest.TestCase):
    def setUp(self):
        self.original = f4.decoder((HERE / 'ma2_c_decode.cuh').read_text())

    def test_original_f4_kernel_and_fused_per_group_arithmetic_are_preserved(self):
        changed = source.decoder(self.original, source.DEFAULT_TUNING)
        self.assertTrue(changed.startswith(self.original + '\n'))
        old_body = self.original[self.original.index('  const int lane'):]
        old_body = old_body.replace('  const int group = blockIdx.x * 4 + threadIdx.x / 32;\n', '')
        start = changed.index('__device__ __forceinline__ void taco_ma6_decode_consumer_group(')
        new_body = changed[changed.index('  const int lane', start):]
        self.assertTrue(new_body.startswith(old_body))
        self.assertIn('taco_ma4f_store(output, bias, residual, offset, packed);\n    return;', new_body)
        # The early return remains in the per-group helper, not the outer loop.
        loop = changed[changed.index('__global__ void taco_ma6_attention_decode_consumer_kernel'):]
        self.assertNotIn('return;', loop)
        self.assertIn('taco_ma6_decode_consumer_masked_group(c, local, output, bias, residual,', loop)
        self.assertEqual(changed.count(f4.CONSUMER_STORE), 1)

    def test_each_legal_row_tile_grid_covers_all_h128_groups_exactly_once(self):
        for rows in (1, 2, 4, 8, 16):
            for threads in (64, 128, 256, 512):
                warps = threads // 32
                maximum = 8192 // rows // warps
                for blocks in sorted({1, min(132, maximum), min(528, maximum), maximum}):
                    groups = []
                    for block in range(blocks):
                        for warp in range(warps):
                            for tile in range(block * warps + warp, 8192 // rows, blocks * warps):
                                base = (tile // 32) * rows * 32 + tile % 32
                                groups.extend(base + row * 32 for row in range(rows))
                    self.assertEqual(Counter(groups), Counter(range(8192)),
                                     (rows, threads, blocks))

    def test_row_tile_mask_is_valid_for_every_rank_and_group(self):
        for rows in (1, 2, 4, 8, 16):
            for rank in range(8):
                for tile in range(8192 // rows):
                    base = (tile // 32) * rows * 32 + tile % 32
                    physical_tile = (rank * 2 + (base // 32) // 128) * 32 + base % 32
                    for row in range(rows):
                        group = base + row * 32
                        original_tile = (rank * 2 + (group // 32) // 128) * 32 + group % 32
                        self.assertEqual(physical_tile, original_tile)

    def test_masked_helper_keeps_every_original_arithmetic_statement(self):
        changed = source.decoder(self.original, source.DEFAULT_TUNING)
        start = changed.index('__device__ __forceinline__ void taco_ma6_decode_consumer_masked_group(')
        body = changed[changed.index('  if (!any_quant) {', start):]
        original_body = self.original[self.original.index('  if (!any_quant) {'):]
        self.assertTrue(body.startswith(original_body))
        kernel = changed[changed.index('__global__ void taco_ma6_attention_decode_consumer_kernel'):]
        self.assertLess(kernel.index('selected[src] ='), kernel.index('for (int row_offset'))
        self.assertIn('constexpr int kRows = 4;', kernel)
        self.assertNotIn('__syncthreads', kernel)

    def test_baseline_roles_use_the_original_f4_kernel(self):
        tuning = {role:dict(source.BASELINE_TUNING) for role in source.ROLES}
        self.assertEqual(source.kernel_name('mlp', tuning), 'taco_ma4f_decode_consumer_kernel')
        changed = source.decoder(self.original, tuning)
        self.assertNotIn('__global__ void taco_ma6_mlp_decode_consumer_kernel', changed)
        tuning['attn'] = dict(source.BASELINE_TUNING)
        tuning['mlp'] = dict(decoder_threads=512, decoder_blocks=264, decoder_rows=1)
        changed = source.decoder(self.original, tuning)
        self.assertNotIn('__global__ void taco_ma6_attention_decode_consumer_kernel', changed)
        self.assertIn('__global__ void taco_ma6_mlp_decode_consumer_kernel', changed)
        self.assertIn('constexpr int kWarps = 16;', changed)
        self.assertIn('taco_ma6_decode_consumer_group(c, local, output, bias, residual, group);', changed)
        self.assertNotIn('taco_ma6_decode_consumer_masked_group', changed)

    def test_role_function_is_bound_to_immutable_k_once(self):
        original = ('  const int cr2_k = taco_cr2_bound_k();\n'
                    'CUDA_CHECK(static_cast<cudaError_t>(taco_ma4f_decode_reduce(\n'
                    '    owned_taco, local, output, bias, residual, stream)));\n')
        changed = source.wrapper(original, source.DEFAULT_TUNING)
        self.assertEqual(changed.count('taco_cr2_bound_k()'), 1)
        self.assertIn('const decltype(&taco_ma4f_decode_reduce) ma6_decode', changed)
        self.assertIn('cr2_k == 512 ? &taco_ma6_decode_attention : &taco_ma6_decode_mlp;', changed)
        self.assertIn('ma6_decode(\n    owned_taco, local, output, bias, residual, stream)', changed)

    def test_native_tuning_identity_and_launch_agree(self):
        original = 'extern "C" int taco_ma4f_protocol_version() { return 1; }\n'
        changed = source.runtime(original, source.DEFAULT_TUNING)
        self.assertTrue(changed.startswith(original))
        self.assertIn('taco_ma6_attention_decode_consumer_kernel<<<512, 128, 0, stream>>>', changed)
        self.assertIn('return k == 512 ? 128 : k == 2048 ? 128 : -2;', changed)
        self.assertIn('return k == 512 ? 512 : k == 2048 ? 512 : -2;', changed)
        self.assertIn('!c.deferred_reuse_barrier || c.allow_capture', changed)
        self.assertIn('return k == 512 ? 4 : k == 2048 ? 4 : -2;', changed)
        self.assertIn('taco_ma6_protocol_version() { return 2; }', changed)

    def test_invalid_geometry_or_non_decoder_fields_are_rejected(self):
        for key, value in [('decoder_threads', True), ('decoder_threads', 96),
                           ('decoder_blocks', 0), ('decoder_blocks', 513),
                           ('decoder_rows', 3), ('decoder_rows', True), ('decoder_rows', 32)]:
            tuning = copy.deepcopy(source.DEFAULT_TUNING)
            tuning['attn'][key] = value
            with self.assertRaises(ValueError):
                source.tuning_config(tuning)
        tuning = copy.deepcopy(source.DEFAULT_TUNING)
        tuning['attn']['gemm_tile_k'] = 64
        with self.assertRaises(ValueError):
            source.tuning_config(tuning)
        with self.assertRaises(ValueError):
            source.tuning_config({'attn':dict(source.BASELINE_TUNING)})
        copied = source.tuning_config()
        copied['attn']['decoder_blocks'] = 1
        self.assertEqual(source.DEFAULT_TUNING['attn']['decoder_blocks'], 512)


if __name__ == '__main__':
    unittest.main()
