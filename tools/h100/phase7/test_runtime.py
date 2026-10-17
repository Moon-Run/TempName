"""CPU tests of transport layout, mixed packets, and independent codec math.

No test in this file validates CUDA concurrency, NCCL progress, or GPU speed.
Run with the documented torch environment; system-Python discovery skips it
when torch is absent rather than treating a GPU implementation as tested.
"""
import importlib.util
import pathlib
import random
import unittest
from contextlib import contextmanager, nullcontext
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    try:
        from . import codec, runtime
    except ImportError:
        import codec
        import runtime


def make_plan(chunk_tiles=8, world=8):
    m, n = world * 256, 4096
    tiles, per_destination = m // 128 * (n // 128), 64
    mask = [[int(src != tile // per_destination and (src * 7 + tile) % 5 == 0)
             for tile in range(tiles)] for src in range(world)]
    count = per_destination // chunk_tiles
    order = []
    for dst in range(world):
        row = list(range(count))
        random.Random(100 + dst).shuffle(row)
        order.append(row)
    return dict(shape=dict(role='attention', M=m, N=n, K_local=512, world_size=world,
                           tile_m=128, tile_n=128), chunk_tiles=chunk_tiles,
                chunk_order=order, quantize_mask=mask)


@unittest.skipIf(torch is None, 'PyTorch not installed in this interpreter')
class LayoutTests(unittest.TestCase):
    def test_each_granularity_covers_each_physical_tile_once(self):
        for count in (1, 2, 4, 8, 16, 32, 64):
            plan = make_plan(count)
            chunks = runtime.validate_plan(plan)
            self.assertEqual(sorted(t for chunk in chunks for t in chunk.local_tiles), list(range(64)))
            self.assertEqual(sum(chunk.rows * chunk.columns for chunk in chunks), 256 * 4096)
            for chunk in chunks:
                self.assertEqual(chunk.rows * chunk.columns, count * 128 * 128)

    def test_variable_splits_agree_at_both_endpoints(self):
        for count in (1, 4, 32, 64):
            plan = make_plan(count)
            total_by_rank = [0] * 8
            for stage in range(64 // count):
                splits = [runtime.stage_splits(plan, rank, stage) for rank in range(8)]
                for src in range(8):
                    for dst in range(8):
                        self.assertEqual(splits[src][1][dst], splits[dst][0][src])
                        if src == dst:
                            self.assertEqual(splits[src][1][dst], 0)
                    total_by_rank[src] += sum(splits[src][1])
            for rank in range(8):
                selected = sum(plan['quantize_mask'][rank])
                expected = 7 * 64 * codec.BF16_BYTES - selected * (codec.BF16_BYTES - codec.QUANT_BYTES)
                self.assertEqual(total_by_rank[rank], expected)

    def test_invalid_or_local_masks_fail_before_collectives(self):
        plan = make_plan()
        plan['quantize_mask'][3][3 * 64] = 1
        with self.assertRaisesRegex(ValueError, 'Local contributions'):
            runtime.validate_plan(plan)
        plan = make_plan()
        plan['quantize_mask'][2][0] = 2
        with self.assertRaisesRegex(ValueError, 'binary'):
            runtime.validate_plan(plan)

    def test_non_permutation_and_unsupported_chunks_fail(self):
        plan = make_plan()
        plan['chunk_order'][0][0] = plan['chunk_order'][0][1]
        with self.assertRaisesRegex(ValueError, 'permutation'):
            runtime.validate_plan(plan)
        plan = make_plan()
        plan['chunk_tiles'] = 3
        with self.assertRaisesRegex(ValueError, 'power of two'):
            runtime.validate_plan(plan)

    def test_modes_differ_only_in_consumer_admission(self):
        # Structural guard on the production loop: no mode-dependent GEMM,
        # mask, physical indexing, dtype or collective split changes.
        import ast
        source = pathlib.Path(runtime.__file__).read_text()
        tree = ast.parse(source)
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == '_forward')
        branches = [node for node in ast.walk(method) if isinstance(node, ast.If) and
                    any(isinstance(child, ast.Attribute) and child.attr == 'mode'
                        for child in ast.walk(node.test))]
        self.assertEqual(len(branches), 2)
        for branch in branches:
            calls = [node.func for node in ast.walk(branch) if isinstance(node, ast.Call)]
            self.assertFalse(any(isinstance(func, ast.Attribute) and
                                 func.attr in ('matmul', '_pack', 'all_to_all_single') for func in calls))


@unittest.skipIf(torch is None, 'PyTorch not installed in this interpreter')
class CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference_path = pathlib.Path(__file__).resolve().parents[2] / 'a800/phase4/reference.py'
        spec = importlib.util.spec_from_file_location('phase4_wire_reference', reference_path)
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)
        # This avoids launching many CPU threads for tiny physical tiles.
        torch.set_num_threads(1)

    def test_bf16_wire_is_bit_exact(self):
        tile = torch.randn((128, 128), generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
        packet = codec.encode_tile(tile, False)
        self.assertEqual(packet.numel(), 32768)
        self.assertTrue(torch.equal(codec.decode_tile(packet, False), tile))

    def test_quantized_codec_matches_independent_phase4_reference(self):
        generator = torch.Generator().manual_seed(6)
        for scale in (0.0, 1e-4, 1.0, 1000.0):
            tile = (torch.randn((128, 128), generator=generator) * scale).to(torch.bfloat16)
            expected, _ = self.reference.roundtrip(tile)
            packet = codec.encode_tile(tile, True)
            self.assertEqual(packet.numel(), 17408)
            self.assertTrue(torch.equal(codec.decode_tile(packet, True), expected))

    def test_mixed_multitile_packet_roundtrip_and_offsets(self):
        for chunk_tiles in (1, 8, 64):
            plan = make_plan(chunk_tiles)
            chunks = runtime.validate_plan(plan)
            source, destination = 3, 6
            producer = runtime.ChunkedGemmRS.__new__(runtime.ChunkedGemmRS)
            consumer = runtime.ChunkedGemmRS.__new__(runtime.ChunkedGemmRS)
            for instance, rank in ((producer, source), (consumer, destination)):
                instance.rank, instance._mask, instance.tiles_per_destination = rank, plan['quantize_mask'], 64
            chunk = chunks[plan['chunk_order'][destination][0]]
            piece = torch.randn((chunk.rows, chunk.columns),
                                generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
            packet = producer._pack(piece, chunk, destination)
            self.assertEqual(packet.numel(), runtime.packet_bytes(chunk, source, destination,
                                                                  plan['quantize_mask'], 64))
            decoded = consumer._unpack(packet, chunk, source)
            expected = piece.clone()
            for physical, tile in consumer._tiles(expected, chunk):
                if plan['quantize_mask'][source][destination * 64 + physical]:
                    value, _ = self.reference.roundtrip(tile)
                    tile.copy_(value)
            self.assertTrue(torch.equal(decoded, expected))

    def test_packet_dtype_length_and_tile_shape_are_checked(self):
        with self.assertRaises(ValueError):
            codec.encode_tile(torch.zeros(128, 127, dtype=torch.bfloat16), True)
        with self.assertRaises(ValueError):
            codec.decode_tile(torch.zeros(codec.QUANT_BYTES - 1, dtype=torch.uint8), True)
        with self.assertRaises(ValueError):
            codec.decode_tile(torch.zeros(codec.BF16_BYTES), False)

    def test_consumer_preserves_ring_order_and_bf16_rounding(self):
        # Exercise real CPU tensor decode/reduction. CUDA stream calls alone
        # are mocked; this test makes no GPU visibility or overlap assertion.
        plan = make_plan(1, world=4)
        plan['quantize_mask'] = [[0] * 256 for _ in range(4)]
        destination = 1
        chunk = runtime.validate_plan(plan)[plan['chunk_order'][destination][0]]
        receiver = runtime.ChunkedGemmRS.__new__(runtime.ChunkedGemmRS)
        receiver.rank, receiver.world, receiver.tiles_per_destination = destination, 4, 64
        receiver._mask, receiver.profile, receiver.consumer = plan['quantize_mask'], False, object()
        constants = [9984.0, 1.0, -9984.0, 4.0]
        pieces = [torch.full((128, 128), value, dtype=torch.bfloat16) for value in constants]
        packets = []
        incoming = []
        for src, piece in enumerate(pieces):
            if src == destination:
                incoming.append(0)
                continue
            packet = codec.encode_tile(piece, False)
            packets.append(packet)
            incoming.append(packet.numel())
        received = torch.cat(packets)
        output = torch.zeros((256, 4096), dtype=torch.bfloat16)
        pending = dict(work=mock.Mock(), chunk=chunk, received=received, local=pieces[destination],
                       send=torch.empty(0, dtype=torch.uint8), stage=0, incoming=incoming)
        with mock.patch.object(torch.cuda, 'stream', return_value=nullcontext()), \
                mock.patch.object(torch.Tensor, 'record_stream', lambda *_: None):
            receiver._consume(pending, output)
        pending['work'].wait.assert_called_once_with()
        actual = output[chunk.row:chunk.row + 128, chunk.column:chunk.column + 128]
        # Source order 2,3,0,1 gives 1; order 0,1,2,3 instead gives 4.
        self.assertTrue(torch.equal(actual, torch.ones_like(actual)))


@unittest.skipIf(torch is None, 'PyTorch not installed in this interpreter')
class AdmissionTests(unittest.TestCase):
    def test_configuration_cannot_change_consumer_mode_or_routes(self):
        op = runtime.ChunkedGemmRS.__new__(runtime.ChunkedGemmRS)
        op._mode, op._order, op._mask = 'barrier', ((0,), (0,)), ((0, 1), (1, 0))
        for name, value in (('mode', 'pipeline'), ('mask', []), ('order', [])):
            with self.assertRaises(AttributeError):
                setattr(op, name, value)
        with self.assertRaises(TypeError):
            op.mask[0][1] = 0
        with self.assertRaisesRegex(ValueError, 'mode'):
            runtime.ChunkedGemmRS(None, {}, mode='invalid')

    def test_admission_and_borrowed_tensor_lifetimes_on_a_cpu_transport_simulator(self):
        # Real CPU GEMM/pack/decode; fake NCCL supplies zero remote contributions.
        # This verifies host scheduling and record_stream registrations, not
        # CUDA event visibility, allocator implementation, or GPU progress.
        torch.set_num_threads(1)
        for mode in ('barrier', 'pipeline'):
            trace, registrations, transfers, gemm_rows = [], {}, [], []

            class Stream:
                def __init__(self, name):
                    self.name = name

                def wait_event(self, event):
                    self.assert_recorded(event)
                    trace.append(('event_wait', self.name, event.stream.name))

                @staticmethod
                def assert_recorded(event):
                    if not hasattr(event, 'stream'):
                        raise AssertionError('A stream waited for an unrecorded event')

            caller, producer, launcher, consumer = [Stream(name) for name in
                                                   ('caller', 'producer', 'launcher', 'consumer')]
            current = [caller]

            @contextmanager
            def stream_context(stream):
                old = current[0]
                current[0] = stream
                try:
                    yield
                finally:
                    current[0] = old

            class Event:
                def record(self, stream):
                    self.stream = stream
                    trace.append(('event_record', stream.name))

            class Work:
                def __init__(self, stage):
                    self.stage = stage

                def wait(self):
                    trace.append(('work_wait', self.stage, current[0].name))

            def exchange(received, send, **kwargs):
                stage = len(transfers)
                self.assertEqual(current[0].name, 'launcher')
                self.assertEqual(received.numel(), sum(kwargs['output_split_sizes']))
                self.assertEqual(send.numel(), sum(kwargs['input_split_sizes']))
                received.zero_()
                transfers.append((received, send))
                trace.append(('exchange', stage))
                return Work(stage)

            def register(tensor, stream):
                registrations.setdefault(id(tensor), set()).add(stream.name)

            plan = dict(shape=dict(M=512, N=128, K_local=32, world_size=2), chunk_tiles=1,
                        chunk_order=[[0, 1], [0, 1]], quantize_mask=[[0] * 4 for _ in range(2)])
            op = runtime.ChunkedGemmRS.__new__(runtime.ChunkedGemmRS)
            op.plan, op.chunks = plan, runtime.validate_plan(plan)
            op._mode, op.profile, op._epoch = mode, False, 0
            op._order, op._mask = plan['chunk_order'], plan['quantize_mask']
            op.rank, op.world, op.rows, op.n, op.group, op.tiles_per_destination = 0, 2, 256, 128, None, 2
            op.producer, op.launcher, op.consumer = producer, launcher, consumer
            op._splits = [runtime.stage_splits(plan, 0, stage) for stage in range(2)]
            op.plan_sha256, op.selection_sha256, op.wire_bytes_per_rank = 'plan', 'mask', 65536
            x = torch.randn(512, 32).to(torch.bfloat16)
            weight = torch.randn(128, 32).to(torch.bfloat16)
            matmul = torch.Tensor.matmul

            def compute(left, right):
                gemm_rows.append((left.data_ptr() - x.data_ptr()) // (32 * 2))
                return matmul(left, right)

            with mock.patch.object(torch.cuda, 'stream', stream_context), \
                    mock.patch.object(torch.cuda, 'current_stream', lambda: current[0]), \
                    mock.patch.object(torch.cuda, 'Event', Event), \
                    mock.patch.object(torch.Tensor, 'record_stream', register), \
                    mock.patch.object(torch.Tensor, 'matmul', compute), \
                    mock.patch.object(runtime.dist, 'all_to_all_single', exchange):
                result = op._forward(x, weight)
            self.assertTrue(torch.equal(result, x[:256].matmul(weight.t())))
            self.assertEqual(gemm_rows, [0, 256, 128, 384])
            exchanges = [i for i, event in enumerate(trace) if event[0] == 'exchange']
            waits = [i for i, event in enumerate(trace) if event[0] == 'work_wait']
            if mode == 'barrier':
                self.assertLess(max(exchanges), min(waits))
            else:
                self.assertLess(exchanges[0], waits[0])
                self.assertLess(waits[0], exchanges[1])
            self.assertTrue(all(event[2] == 'consumer' for event in trace if event[0] == 'work_wait'))
            self.assertEqual(registrations[id(x)], {'producer'})
            self.assertEqual(registrations[id(weight)], {'producer'})
            self.assertIn('consumer', registrations[id(result)])
            for received, send in transfers:
                self.assertIn('consumer', registrations[id(received)])
                self.assertTrue({'launcher', 'consumer'} <= registrations[id(send)])
            self.assertEqual(trace[-1], ('event_wait', 'caller', 'consumer'))


if __name__ == '__main__':
    unittest.main()
