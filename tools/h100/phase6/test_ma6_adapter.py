"""CPU numerical and scheduling contracts for the three MA6 consumer arms.

The CPU collective below models TP8 concatenation. It does not claim CUDA
overlap; real stream dependencies remain an eight-GPU profiling gate.
"""
import ast
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ma4_f_adapter
import ma5_fullstep
import ma6_adapter

try:
    import torch
except ImportError:
    torch = None


class SourceContractTests(unittest.TestCase):
    def test_reference_math_is_preserved_and_frozen_source_compiles(self):
        source = ma6_adapter.consumer_autograd_source()
        reference = ma4_f_adapter.consumer_autograd_source().replace(
            'class ConsumerGemmRSFunction(', 'class _MA6ReferenceConsumerGemmRSFunction(', 1)
        self.assertTrue(source.startswith(reference))
        ast.parse(source)
        original = ma4_f_adapter.adapter_source(ma5_fullstep.adapter_source())
        transformed = ma6_adapter.adapter_source(original)
        ast.parse(transformed)
        self.assertIn('dist_all_gather_func', transformed)
        self.assertNotIn("os.environ.get('H100_MA4_F_ARM') == 'F4'", transformed)
        self.assertIn("backward=new_backward_state(self._ma6_arm)", transformed)
        self.assertIn("consumer['backward'])", transformed)
        with self.assertRaisesRegex(ValueError, 'source contract changed'):
            ma6_adapter.adapter_source(transformed)

    def test_backward_state_is_independent_and_serializes_only_counters(self):
        left = ma6_adapter.new_backward_state('F4')
        right = ma6_adapter.new_backward_state('F4')
        left['forward_calls'] += 1
        left._ma6_op_id = 123
        self.assertEqual(right['forward_calls'], 0)
        self.assertIsNone(right._ma6_op_id)
        self.assertEqual(json.loads(json.dumps(left)), dict(left))
        self.assertEqual(set(left), {'protocol', 'arm', 'gather_schedule', 'forward_calls', 'backward_calls',
            'baseline_backward_calls', 'async_gather_calls', 'wait_calls', 'failed', 'active'})
        self.assertEqual(left['gather_schedule'], 'baseline')
        self.assertIsNone(left._ma6_gather_metadata)
        with self.assertRaises(ValueError):
            ma6_adapter.new_backward_state('CR2')

    def test_schedule_is_validated_and_baked_into_generated_source(self):
        with self.assertRaisesRegex(ValueError, 'schedule'):
            ma6_adapter.new_backward_state('F4-gather', 'unknown')
        with self.assertRaisesRegex(ValueError, 'schedule'):
            ma6_adapter.consumer_autograd_source('unknown')
        source = ma6_adapter.consumer_autograd_source('wait-first')
        self.assertIn("DEFAULT_GATHER_SCHEDULE = 'wait-first'", source)
        state = ma6_adapter.new_backward_state('F4-gather', 'wait-first')
        self.assertEqual(state['gather_schedule'], 'wait-first')


@unittest.skipIf(torch is None, 'Run with the actual PyTorch CPU environment')
class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.group = object()
        self.starts = []
        self.waits = []
        group, starts, waits = self.group, self.starts, self.waits

        def gather(input_):
            return torch.cat([input_] * 8, dim=0)

        def asynchronous(output, input_, *, group, async_op):
            self.assertIs(group, self.group)
            self.assertTrue(async_op)
            self.assertTrue(input_.is_contiguous())
            self.assertNotEqual(output.data_ptr(), input_.data_ptr())
            starts.append((input_.data_ptr(), output.data_ptr()))

            class Work:
                def wait(self):
                    # Filling only on wait also detects any premature dgrad
                    # consumer of an uninitialized gathered output.
                    output.copy_(gather(input_))
                    waits.append(True)
                    return True

            return Work()

        self.scope = dict(torch=torch,
            parallel_state=SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 8,
                                           get_tensor_model_parallel_group=lambda: group),
            _gather_along_first_dim=gather, dist_all_gather_func=asynchronous)
        exec(ma6_adapter.consumer_autograd_source(), self.scope)
        self.Function = self.scope['ConsumerGemmRSFunction']
        self.group_patch = patch.object(torch.distributed, 'get_world_size',
                                       side_effect=lambda group=None: 8 if group is self.group else 0)
        self.group_patch.start()
        self.addCleanup(self.group_patch.stop)

    @staticmethod
    def projection(x, weight):
        # All eight simulated sequence shards contribute to the local output.
        # Its exact adjoint is the TP8 gathered gradient used by the adapter.
        full = x.reshape(-1, x.shape[-1]).matmul(weight.t())
        return full.reshape(8, full.shape[0] // 8, full.shape[1]).sum(dim=0)

    def test_all_four_gradients_and_three_parameter_updates_match_reference(self):
        for dtype in (torch.bfloat16, torch.float32):
            combinations = [(arm, 'overlap') for arm in ma6_adapter.ARMS]
            combinations.append(('F4-gather', 'wait-first'))
            for arm, schedule in combinations:
                with self.subTest(dtype=dtype, arm=arm, schedule=schedule):
                    torch.manual_seed(487)
                    initial = [torch.randn(shape, dtype=dtype) for shape in
                               ((16, 1, 8), (12, 8), (12,), (2, 1, 12))]
                    candidate = [value.clone().requires_grad_() for value in initial]
                    reference = [value.clone().requires_grad_() for value in initial]
                    state = ma6_adapter.new_backward_state(arm, schedule)
                    projection = self.projection

                    class Op:
                        def forward_consumer(self, x, weight, bias, residual, reduce_scatter_option=None):
                            return residual + (projection(x, weight) + bias)

                    op = Op()
                    kept = []
                    for step in range(3):
                        output = self.Function.apply(*candidate, op, None, state)
                        expected = reference[3] + (projection(reference[0], reference[1]).reshape_as(reference[3]) + reference[2])
                        self.assertTrue(torch.equal(output, expected))
                        self.assertIsNone(output._base)
                        self.assertTrue(all(output.data_ptr() != previous.data_ptr() for previous, _ in kept))
                        grad = torch.randn_like(output)
                        output.backward(grad)
                        expected.backward(grad)
                        for index, (left, right) in enumerate(zip(candidate, reference)):
                            self.assertTrue(torch.equal(left.grad, right.grad), (dtype, arm, step, index))
                        kept.append((output, output.detach().clone()))
                        with torch.no_grad():
                            for left, right in zip(candidate, reference):
                                left.add_(left.grad, alpha=-0.03125)
                                right.add_(right.grad, alpha=-0.03125)
                                self.assertTrue(torch.equal(left, right))
                                left.grad = right.grad = None
                    self.assertTrue(all(torch.equal(value, snapshot) for value, snapshot in kept))
                    self.assertEqual(state['forward_calls'], 3)
                    self.assertEqual(state['backward_calls'], 3)
                    self.assertEqual(state['async_gather_calls'], 3 if arm == 'F4-gather' else 0)
                    self.assertEqual(state['wait_calls'], state['async_gather_calls'])
                    self.assertEqual(state['baseline_backward_calls'], 0 if arm == 'F4-gather' else 3)
                    self.assertFalse(state['failed'])
                    self.assertFalse(state['active'])
                    self.assertIsNone(state._ma6_pending)

    def test_metadata_binds_tp_group_once_without_retaining_tensor_storage(self):
        helper = self.scope['_ma6_gather_metadata']
        state = ma6_adapter.new_backward_state('F4-gather')
        tensor = torch.ones((2, 1, 12), dtype=torch.bfloat16)
        getter = self.scope['parallel_state'].get_tensor_model_parallel_group
        with patch.object(self.scope['parallel_state'], 'get_tensor_model_parallel_group',
                          wraps=getter) as group_getter:
            with patch.object(torch.distributed, 'get_world_size', return_value=8) as size_getter:
                metadata = helper(tensor, state)
                for _ in range(3):
                    self.assertIs(helper(tensor.clone(), state), metadata)
                self.assertEqual(group_getter.call_count, 1)
                self.assertEqual(size_getter.call_count, 1)
        self.assertEqual(metadata[2], (16, 1, 12))
        self.assertIs(metadata[1], self.group)
        self.assertFalse(any(isinstance(item, torch.Tensor) for item in metadata))

    def test_noncontiguous_gradient_keeps_original_bias_semantics(self):
        torch.manual_seed(16)
        values = [torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
                  for shape in ((16, 1, 8), (12, 8), (12,), (2, 1, 12))]
        projection = self.projection
        op = SimpleNamespace(forward_consumer=lambda x, w, b, r, **kwargs: r + (projection(x, w) + b))
        state = ma6_adapter.new_backward_state('F4-gather')
        output = self.Function.apply(*values, op, None, state)
        grad = torch.randn((2, 1, 24), dtype=torch.bfloat16)[..., ::2]
        self.assertFalse(grad.is_contiguous())
        output.backward(grad)
        self.assertTrue(torch.equal(values[2].grad, grad.sum_to_size(values[2].shape)))
        self.assertTrue(torch.equal(values[3].grad, grad))
        full = torch.cat([grad.contiguous()] * 8, dim=0)
        self.assertTrue(torch.equal(values[0].grad, full.matmul(values[1])))
        expected_weight = full.reshape(-1, 12).t().matmul(values[0].reshape(-1, 8))
        self.assertTrue(torch.equal(values[1].grad, expected_weight))

    def test_same_state_cannot_cross_native_instances_or_change_arm(self):
        validate = self.scope['_ma6_validate_state']
        state = ma6_adapter.new_backward_state('F4-gather')
        first, second = object(), object()
        validate(state, first)
        with self.assertRaisesRegex(RuntimeError, 'another projection'):
            validate(state, second)
        self.assertTrue(state['failed'])
        with self.assertRaisesRegex(RuntimeError, 'failed backward'):
            validate(state, first)
        changed = ma6_adapter.new_backward_state('F4')
        changed['arm'] = 'F4-gather'
        with self.assertRaisesRegex(RuntimeError, 'Invalid MA6'):
            validate(changed, first)


class SchedulingTests(unittest.TestCase):
    def make(self, arm, fail=None, schedule='overlap'):
        log = []

        class Tensor:
            is_cuda = False
            device = 'cpu'
            dtype = 'BF16'
            shape = (2, 1, 12)

            def __init__(self, name):
                self.name = name

            def size(self):
                return self.shape

            def stride(self):
                return (12, 12, 1)

            def contiguous(self):
                log.append('contiguous')
                return self

            def sum_to_size(self, shape):
                log.append('bias')
                if fail == 'bias':
                    raise RuntimeError('synthetic bias failure')
                return Tensor('bias')

            def reshape(self, *shape):
                return self

            def t(self):
                return self

            def matmul(self, other):
                stage = 'dgrad' if other.name == 'weight' else 'wgrad'
                log.append(stage)
                if fail == stage:
                    raise RuntimeError('synthetic ' + stage + ' failure')
                return Tensor('result')

        class Work:
            def wait(self):
                log.append('wait')
                if fail == 'wait':
                    raise RuntimeError('synthetic wait failure')

        def asynchronous(output, input_, *, group, async_op):
            self.assertTrue(async_op)
            log.append('async_gather')
            if fail == 'launch':
                raise RuntimeError('synthetic launch failure')
            return Work()

        def synchronous(input_):
            log.append('sync_gather')
            return Tensor('full')

        fake_torch = SimpleNamespace(autograd=SimpleNamespace(Function=object),
            distributed=SimpleNamespace(get_world_size=lambda group: 8),
            empty=lambda *args, **kwargs: Tensor('full'))
        scope = dict(torch=fake_torch,
            parallel_state=SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 8,
                                           get_tensor_model_parallel_group=lambda: object()),
            _gather_along_first_dim=synchronous, dist_all_gather_func=asynchronous)
        exec(ma6_adapter.consumer_autograd_source(schedule), scope)
        state = scope['new_backward_state'](arm)
        ctx = SimpleNamespace(ma6_state=state, ma6_backward_done=False,
                              saved_tensors=(Tensor('input'), Tensor('weight')), bias_shape=(12,))
        return scope['ConsumerGemmRSFunction'], ctx, Tensor('g'), state, log

    def test_only_gather_arm_moves_collective_before_bias_and_wait_after_bias(self):
        for arm in ma6_adapter.ARMS:
            function, ctx, grad, state, log = self.make(arm)
            result = function.backward(ctx, grad)
            self.assertIs(result[3], grad)
            self.assertEqual(len(result), 7)
            if arm == 'F4-gather':
                self.assertEqual(log, ['contiguous', 'async_gather', 'bias', 'wait', 'dgrad', 'wgrad'])
            else:
                self.assertEqual(log, ['bias', 'contiguous', 'sync_gather', 'dgrad', 'wgrad'])
            self.assertEqual(state['backward_calls'], 1)

    def test_wait_first_uses_same_collective_and_moves_only_wait(self):
        function, ctx, grad, state, log = self.make('F4-gather', schedule='wait-first')
        function.backward(ctx, grad)
        self.assertEqual(log, ['contiguous', 'async_gather', 'wait', 'bias', 'dgrad', 'wgrad'])
        self.assertEqual(state['async_gather_calls'], 1)
        self.assertEqual(state['wait_calls'], 1)

    def test_bound_tensor_signature_drift_is_rejected_before_second_collective(self):
        for field, replacement in (('shape', (3, 1, 12)), ('dtype', 'FP32'),
                                   ('device', 'cuda:1'), ('stride', lambda: (24, 24, 2))):
            with self.subTest(field=field):
                function, ctx, grad, state, log = self.make('F4-gather')
                function.backward(ctx, grad)
                ctx.ma6_backward_done = False
                setattr(grad, field, replacement)
                with self.assertRaisesRegex(RuntimeError, 'changed after binding'):
                    function.backward(ctx, grad)
                self.assertTrue(state['failed'])
                self.assertEqual(log.count('async_gather'), 1)

    def test_schedule_identity_mutation_is_rejected(self):
        function, ctx, grad, state, log = self.make('F4-gather')
        state['gather_schedule'] = 'wait-first'
        with self.assertRaisesRegex(RuntimeError, 'Invalid MA6'):
            function.backward(ctx, grad)
        self.assertEqual(log, [])

    def test_async_failures_never_fallback_or_allow_a_second_collective(self):
        for schedule in ma6_adapter.GATHER_SCHEDULES:
            for stage in ('launch', 'bias', 'wait'):
                with self.subTest(stage=stage, schedule=schedule):
                    function, ctx, grad, state, log = self.make('F4-gather', fail=stage,
                                                              schedule=schedule)
                    with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                        function.backward(ctx, grad)
                    self.assertTrue(state['failed'])
                    self.assertFalse(state['active'])
                    self.assertEqual(state['backward_calls'], 0)
                    self.assertEqual(log.count('async_gather'), 1)
                    self.assertNotIn('sync_gather', log)
                    self.assertNotIn('dgrad', log)
                    if stage != 'launch':
                        self.assertIsNotNone(state._ma6_pending)
                    with self.assertRaisesRegex(RuntimeError, 'failed backward'):
                        function.backward(ctx, grad)
                    self.assertEqual(log.count('async_gather'), 1)

    def test_repeated_backward_is_rejected_before_relaunch(self):
        function, ctx, grad, state, log = self.make('F4-gather')
        function.backward(ctx, grad)
        with self.assertRaisesRegex(RuntimeError, 'repeated backward'):
            function.backward(ctx, grad)
        self.assertEqual(log.count('async_gather'), 1)
        self.assertTrue(state['failed'])

    def test_consumer_failure_retains_work_and_input_output_until_abort(self):
        for schedule in ma6_adapter.GATHER_SCHEDULES:
            for stage in ('dgrad', 'wgrad'):
                with self.subTest(schedule=schedule, stage=stage):
                    function, ctx, grad, state, log = self.make('F4-gather', fail=stage,
                                                              schedule=schedule)
                    with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                        function.backward(ctx, grad)
                    self.assertTrue(state['failed'])
                    self.assertIs(state._ma6_pending[1], grad)
                    self.assertIsNot(state._ma6_pending[2], grad)
                    self.assertTrue(callable(state._ma6_pending[0].wait))
                    self.assertEqual(state['wait_calls'], 1)
                    with self.assertRaisesRegex(RuntimeError, 'failed backward'):
                        function.backward(ctx, grad)
                    self.assertEqual(log.count('async_gather'), 1)


if __name__ == '__main__':
    unittest.main()
