"""CPU semantic tests of the exact pinned Megatron arithmetic with fake TP I/O."""
import ast
from functools import partial
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
from typing import Optional
import unittest
from unittest.mock import patch

import torch
import ma8_adapter as ma8


REPO = Path(__file__).resolve().parents[3]
LAYERS = REPO.parent / 'Megatron-LM/megatron/core/tensor_parallel/layers.py'
FAILED_BATCH = REPO / 'logs/h100/phase6/ma8-v1-cache-1x-50steps-20261026-201336-451664'


def pinned_source(path=LAYERS):
    text = path.read_text()
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and
                n.name == 'LinearWithGradAccumulationAndAsyncCommunication')
    return ma8.normalized(ast.get_source_segment(text, node))


class Collectives:
    def __init__(self):
        self.buffer = None
        self.calls = []
        self.waits = []

    def get_tensor(self, shape, dtype, name):
        if self.buffer is None or list(self.buffer.shape) != list(shape) or self.buffer.dtype != dtype:
            self.buffer = torch.empty(shape, dtype=dtype)
        return self.buffer

    def gather(self, output, input, group=None, async_op=False):
        self.calls.append(('gather', async_op, output.data_ptr()))
        # Real collectives only move values. Distinct simulated rank slices
        # catch accidentally using only the local activation in wgrad.
        output.copy_(torch.cat([input + rank for rank in range(8)], dim=0))
        return types.SimpleNamespace(wait=lambda: self.waits.append('gather'))

    def scatter(self, output, input, group=None, async_op=False):
        self.calls.append(('scatter', async_op, output.data_ptr()))
        output.copy_(input[:output.shape[0]] * 8)
        return types.SimpleNamespace(wait=lambda: self.waits.append('scatter'))


class TorchProxy:
    # The raw pinned CPU test must preserve its CUDA destination expression;
    # only device resolution and TP collectives are substituted.
    cuda = types.SimpleNamespace(current_device=lambda: 'cpu')

    def __getattr__(self, name):
        return getattr(torch, name)


def fixture(include_scope=False, real_amp=False):
    io = Collectives()
    scope = dict(torch=TorchProxy(), custom_fwd=lambda f: f, custom_bwd=lambda f: f,
        get_tensor_model_parallel_world_size=lambda: 8,
        get_tensor_model_parallel_group=lambda: None,
        get_global_memory_buffer=lambda: io,
        dist_all_gather_func=io.gather, dist_reduce_scatter_func=io.scatter,
        prepare_input_tensors_for_wgrad_compute=lambda grad, inp:
            (grad.contiguous().view(-1, grad.shape[-1]), inp.view(-1, inp.shape[-1])))
    if real_amp:
        scope.update(custom_fwd=partial(torch.amp.custom_fwd, device_type='cuda'),
                     custom_bwd=partial(torch.amp.custom_bwd, device_type='cuda'))
    source = pinned_source()
    exec(compile(source, '<ma8-reference-test>', 'exec'), scope)
    exec(compile(ma8.cached_class_source(source), '<ma8-candidate-test>', 'exec'), scope)
    functions = ma8.make_functions(torch,
        scope['LinearWithGradAccumulationAndAsyncCommunication'], scope['_MA8CachedRawLinear'])
    return (io, functions, scope) if include_scope else (io, functions)


def inspectable_layers(testcase, path=LAYERS, changed_column=False):
    """Load original methods from a real file; never mock inspect.getsource.

    AST identifies the method's lines only. Its first-line indentation must
    remain intact, just as it does when inspect reads the production class.
    The constructor and CUDA collectives are not needed for installation.
    """
    text = path.read_text()
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)
               and n.name == 'ColumnParallelLinear')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    column = ''.join(text.splitlines(keepends=True)[method.lineno - 1:method.end_lineno])
    if changed_column:
        column = ma8.replace_once(column, 'return output, output_bias', 'return output, None')
    source = pinned_source(path) + '\n\nclass ColumnParallelLinear:\n' + column
    temp = tempfile.TemporaryDirectory(prefix='ma8-inspect-source-')
    testcase.addCleanup(temp.cleanup)
    file = Path(temp.name) / 'layers.py'
    file.write_text(source)
    _, _, namespace = fixture(include_scope=True)
    module = types.ModuleType('ma8_source_fixture_' + file.parent.name.replace('-', '_'))
    module.__file__ = str(file)
    module.__dict__.update(namespace, Optional=Optional)
    registered = patch.dict(sys.modules, {module.__name__: module})
    registered.start()
    testcase.addCleanup(registered.stop)
    exec(compile(source, str(file), 'exec'), module.__dict__)
    return module


def apply(function, values, state):
    return function.apply(*values, False, False, True, None, None, state)


def linear_with_grad_accumulation_and_async_allreduce(**kwargs):
    return 'untouched global reference'


def resetting_forward(self, input_):
    self._forward_impl = linear_with_grad_accumulation_and_async_allreduce
    return self._forward_impl(input=input_, weight=self.weight, bias=None,
        gradient_accumulation_fusion=False, allreduce_dgrad=False,
        sequence_parallel=True, grad_output_buffer=None, wgrad_deferral_limit=None)


class SourceTests(unittest.TestCase):
    def test_source_identity_and_bounded_change(self):
        source = pinned_source()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), ma8.CLASS_SHA256)
        candidate = ma8.cached_class_source(source)
        ast.parse(candidate)
        self.assertEqual(candidate.count('dist_all_gather_func('), 1)
        self.assertEqual(candidate.count('dist_reduce_scatter_func('), 1)
        self.assertNotIn('get_global_memory_buffer()', candidate)
        for math in ('grad_input = grad_output.matmul(weight)',
                     'grad_weight = grad_output.t().matmul(total_input)',
                     'grad_bias = grad_output.sum(dim=0) if use_bias else None',
                     'prepare_input_tensors_for_wgrad_compute(\n                grad_output, total_input'):
            self.assertIn(math, candidate)
        with self.assertRaises(RuntimeError):
            ma8.cached_class_source(source.replace('grad_input =', 'grad_input  ='))

    def test_column_forward_contract(self):
        layers = inspectable_layers(self)
        source = inspect.getsource(layers.ColumnParallelLinear.forward)
        self.assertTrue(source.startswith('    def forward('))
        ma8.require_source(source, ma8.COLUMN_FORWARD_SHA256)

    def test_returned_failure_is_reproduced_without_gpu_or_source_mock(self):
        path = FAILED_BATCH / 'fullstep/megatron/megatron/core/tensor_parallel/layers.py'
        adapter = FAILED_BATCH / 'fullstep/scripts/ma8_adapter.py'
        old_digest = next(ast.literal_eval(n.value) for n in ast.parse(adapter.read_text()).body
                          if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                          and t.id == 'COLUMN_FORWARD_SHA256' for t in n.targets))
        layers = inspectable_layers(self, path)
        runtime_source = inspect.getsource(layers.ColumnParallelLinear.forward)
        with self.assertRaisesRegex(RuntimeError, 'pinned, unmodified Megatron source'):
            ma8.require_source(runtime_source, old_digest)
        ma8.require_source(runtime_source, ma8.COLUMN_FORWARD_SHA256)

        text = path.read_text()
        cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)
                   and n.name == 'ColumnParallelLinear')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        # The old test accepted precisely this differently indented extraction.
        ma8.require_source(ast.get_source_segment(text, method), old_digest)
        self.assertNotEqual(old_digest, ma8.COLUMN_FORWARD_SHA256)

    def test_instance_binding_survives_forward_impl_reset_without_global_patch(self):
        class Tensor:
            device = 'cuda:0'
            dtype = torch.bfloat16
            is_cuda = True
            requires_grad = True
            shape = (256, 1, 4096)

            def data_ptr(self): return id(self)
            def is_contiguous(self): return True

        class Column:
            training = True
            forward = resetting_forward

            def __init__(self): self.weight = Tensor()

        selected, untouched = Column(), Column()
        state = ma8.CacheState('layer.mlp.linear_fc1', 'fc1', 'F4-cache')
        calls = []
        function = types.SimpleNamespace(apply=lambda *args: calls.append(args) or 'cached')
        ma8._bind_column(selected, state, function, resetting_forward, torch)
        self.assertEqual(selected.forward(Tensor()), 'cached')
        self.assertEqual(selected.forward(Tensor()), 'cached')
        self.assertEqual(untouched.forward(Tensor()), 'untouched global reference')
        self.assertIs(Column.forward, resetting_forward)
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][-1], state)
        selected.weight = Tensor()
        with self.assertRaisesRegex(RuntimeError, 'ownership'):
            selected.forward(Tensor())
        self.assertTrue(state['failed'])


class ArithmeticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8123)
        torch.set_num_threads(1)

    def test_bf16_fp32_gradients_and_three_updates(self):
        for dtype in (torch.bfloat16, torch.float32):
            for use_bias in (False, True):
                with self.subTest(dtype=dtype, bias=use_bias):
                    fixtures = [fixture(), fixture()]
                    states = [ma8.CacheState('layer.mlp.linear_fc1', 'fc1', arm) for arm in ma8.ARMS]
                    w0 = torch.randn(6, 8, dtype=dtype)
                    weights = [w0.clone().requires_grad_() for _ in ma8.ARMS]
                    b0 = torch.randn(6, dtype=dtype)
                    biases = [b0.clone().requires_grad_() if use_bias else None for _ in ma8.ARMS]
                    for step in range(3):
                        for microbatch in range(4):
                            x0 = torch.randn(2, 1, 8, dtype=dtype)
                            inputs = [x0.clone().requires_grad_() for _ in ma8.ARMS]
                            outputs = [apply(functions[arm], [x, w, b], state)
                                for (io, functions), arm, x, w, b, state in
                                zip(fixtures, ma8.ARMS, inputs, weights, biases, states)]
                            self.assertTrue(torch.equal(*outputs))
                            upstream = torch.randn(16, 1, 12, dtype=dtype)[..., ::2]
                            self.assertFalse(upstream.is_contiguous())
                            for output in outputs:
                                output.backward(upstream)
                            self.assertTrue(torch.equal(inputs[0].grad, inputs[1].grad))
                            self.assertTrue(torch.equal(weights[0].grad, weights[1].grad))
                            if use_bias:
                                self.assertTrue(torch.equal(biases[0].grad, biases[1].grad))
                        with torch.no_grad():
                            for w, b in zip(weights, biases):
                                w.add_(w.grad, alpha=-0.01)
                                w.grad = None
                                if b is not None:
                                    b.add_(b.grad, alpha=-0.01)
                                    b.grad = None
                        self.assertTrue(torch.equal(*weights))
                    for state, (io, functions) in zip(states, fixtures):
                        actual = ma8.ActivationCacheTracker(state.arm, {'column': state}).state()['column']
                        self.assertEqual(actual['forward_calls'], 12)
                        self.assertEqual(actual['backward_calls'], 12)
                        self.assertEqual(actual['pending'], 0)
                        self.assertEqual(actual['retained_bytes'], 0)
                        self.assertEqual(sum(name == 'gather' for name, _, _ in io.calls),
                            24 if state.arm == 'F4' else 12)
                        self.assertEqual(sum(name == 'scatter' for name, _, _ in io.calls), 12)
                        self.assertEqual(io.waits.count('gather'), 12 if state.arm == 'F4' else 0)
                        self.assertEqual(io.waits.count('scatter'), 12)

    def test_shared_global_buffer_overwrite_does_not_corrupt_cache(self):
        io, functions = fixture()
        inputs = [torch.randn(2, 1, 8, dtype=torch.bfloat16).requires_grad_() for _ in range(2)]
        weights = [torch.randn(6, 8, dtype=torch.bfloat16).requires_grad_() for _ in range(2)]
        states = [ma8.CacheState(f'layer{i}.mlp.linear_fc1', 'fc1', 'F4-cache') for i in range(2)]
        outputs, saved = [], []
        for inp, weight, state in zip(inputs, weights, states):
            outputs.append(apply(functions['F4-cache'], [inp, weight, None], state))
            saved.append(outputs[-1].grad_fn.saved_tensors[2])
        # Simulate unrelated column/row operations overwriting the shared buffer.
        global_buffer = io.get_tensor(saved[0].shape, saved[0].dtype, 'mpu')
        global_buffer.fill_(float('nan'))
        self.assertEqual(len({x.data_ptr() for x in saved} | {global_buffer.data_ptr()}), 3)
        expected = [x.clone() for x in saved]
        for index in (1, 0):
            upstream = torch.randn_like(outputs[index])
            expected_wgrad = upstream.view(-1, 6).t().matmul(expected[index].view(-1, 8))
            outputs[index].backward(upstream)
            self.assertTrue(torch.equal(weights[index].grad, expected_wgrad))
            self.assertTrue(torch.equal(saved[index], expected[index]))
        self.assertEqual(len([c for c in io.calls if c[0] == 'gather']), 2)

    def test_original_input_version_is_still_checked(self):
        _, functions = fixture()
        state = ma8.CacheState('layer.mlp.linear_fc1', 'fc1', 'F4-cache')
        inp = torch.randn(2, 1, 8, requires_grad=True)
        weight = torch.randn(6, 8, requires_grad=True)
        output = apply(functions['F4-cache'], [inp, weight, None], state)
        with torch.no_grad():
            inp.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'modified by an inplace'):
            output.sum().backward()
        self.assertTrue(state['failed'])
        with self.assertRaises(RuntimeError):
            ma8.ActivationCacheTracker('F4-cache', {'x': state}).state()

    def test_repeated_backward_and_inflight_forward_are_rejected(self):
        for repeat_forward in (False, True):
            _, functions = fixture()
            state = ma8.CacheState('layer.mlp.linear_fc1', 'fc1', 'F4-cache')
            values = [torch.randn(2, 1, 8, requires_grad=True), torch.randn(6, 8, requires_grad=True), None]
            output = apply(functions['F4-cache'], values, state)
            if repeat_forward:
                with self.assertRaises(RuntimeError):
                    apply(functions['F4-cache'], values, state)
            else:
                output.sum().backward(retain_graph=True)
                with self.assertRaises(RuntimeError):
                    output.sum().backward()
            self.assertTrue(state['failed'])

    def test_exception_does_not_report_completed_backward(self):
        io, functions = fixture()
        state = ma8.CacheState('layer.mlp.linear_fc1', 'fc1', 'F4-cache')
        values = [torch.randn(2, 1, 8, requires_grad=True), torch.randn(6, 8, requires_grad=True), None]
        output = apply(functions['F4-cache'], values, state)
        # A version failure is deterministic and occurs before collective launch.
        with torch.no_grad():
            values[1].add_(1)
        with self.assertRaises(RuntimeError):
            output.sum().backward()
        self.assertEqual(state['backward_calls'], 0)
        self.assertEqual(state['pending'], 1)
        self.assertTrue(state['failed'])

    def test_real_cpu_profiler_names_sequence_and_scope_nesting(self):
        for arm, class_name in zip(ma8.ARMS, ('MA8ReferenceLinear', 'MA8RetainedActivationLinear')):
            with self.subTest(arm=arm):
                _, functions = fixture()
                name = 'decoder.layers.0.self_attention.linear_qkv'
                state = ma8.CacheState(name, 'qkv', arm)
                values = [torch.randn(2, 1, 8, requires_grad=True), torch.randn(6, 8, requires_grad=True), None]
                with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
                    output = apply(functions[arm], values, state)
                    output.sum().backward()
                with tempfile.TemporaryDirectory(prefix='ma8-cpu-profile-') as tmp:
                    path = str(Path(tmp) / 'trace.json')
                    prof.export_chrome_trace(path)
                    events = json.loads(Path(path).read_text())['traceEvents']
                forward = [e for e in events if e.get('name') == class_name]
                backward = [e for e in events if e.get('name') == class_name + 'Backward']
                self.assertEqual(len(forward), 1)
                self.assertEqual(len(backward), 1)
                self.assertEqual(forward[0]['args']['Sequence number'], backward[0]['args']['Sequence number'])
                for phase, parent in (('forward', forward[0]), ('backward', backward[0])):
                    scopes = [e for e in events if e.get('name') == f'ma8.column.{phase}.' + name]
                    self.assertEqual(len(scopes), 1)
                    self.assertEqual(scopes[0]['cat'], 'user_annotation')
                    self.assertEqual(scopes[0]['tid'], parent['tid'])
                    self.assertLessEqual(parent['ts'], scopes[0]['ts'])
                    self.assertLessEqual(scopes[0]['ts'] + scopes[0]['dur'], parent['ts'] + parent['dur'] + 0.001)

    def test_real_amp_decorators_preserve_context_and_gradients(self):
        # CUDA AMP decorators execute on the CPU fixture, without enabling CUDA
        # autocast or claiming GPU coverage. This checks nested ctx bookkeeping.
        _, functions = fixture(real_amp=True)
        originals = [torch.randn(shape, dtype=torch.bfloat16)
                     for shape in ((2, 1, 8), (6, 8), (6,))]
        arms = [[value.clone().requires_grad_() for value in originals] for _ in ma8.ARMS]
        outputs = []
        for arm, values in zip(ma8.ARMS, arms):
            state = ma8.CacheState('layer.self_attention.linear_qkv', 'qkv', arm)
            output = apply(functions[arm], values, state)
            self.assertTrue(hasattr(output.grad_fn, '_fwd_used_autocast'))
            self.assertTrue(hasattr(output.grad_fn, '_dtype'))
            outputs.append(output)
        self.assertTrue(torch.equal(*outputs))
        upstream = torch.randn_like(outputs[0])
        for output in outputs:
            output.backward(upstream)
        for baseline, candidate in zip(*arms):
            self.assertTrue(torch.equal(baseline.grad, candidate.grad))


class InstallationTests(unittest.TestCase):
    def fixture(self, changed_column=False):
        layers = inspectable_layers(self, changed_column=changed_column)
        Column = layers.ColumnParallelLinear
        original_forward = Column.forward

        class FakeWeight:
            requires_grad = True
            dtype = torch.bfloat16
            is_cuda = True

            def __init__(self, role): self.shape = (1536 if role == 'qkv' else 2048, 4096)
            def is_contiguous(self): return True
            def data_ptr(self): return id(self)

        Column.training = Column.sequence_parallel = True
        Column.allreduce_dgrad = Column.gradient_accumulation_fusion = Column.explicit_expert_comm = False
        Column.disable_grad_reduce = Column.gather_output = False
        config = types.SimpleNamespace(num_layers=32, hidden_size=4096,
            tensor_model_parallel_size=8, pipeline_model_parallel_size=1,
            sequence_parallel=True, gradient_accumulation_fusion=False,
            recompute_granularity=None, enable_cuda_graph=False, external_cuda_graph=False,
            cpu_offloading=False, defer_embedding_wgrad_compute=False)
        modules = [(f'decoder.layers.{i}{suffix}', Column())
            for i in range(32) for suffix, role in ma8.SUFFIX_ROLES.items()]
        for name, module in modules:
            module.weight = FakeWeight('qkv' if name.endswith('linear_qkv') else 'fc1')
        model = types.SimpleNamespace(training=True, config=config, named_modules=lambda: iter(modules))
        packages = {name: types.ModuleType(name) for name in
            ('megatron', 'megatron.core', 'megatron.core.tensor_parallel')}
        packages['megatron.core.tensor_parallel'].layers = layers

        return model, modules, packages, original_forward

    def test_all_64_instances_and_duplicate_install_rejection(self):
        for arm in ma8.ARMS:
            with self.subTest(arm=arm):
                model, modules, packages, original = self.fixture()
                with patch.dict(sys.modules, packages):
                    tracker = ma8.install_activation_cache(model, arm)
                    self.assertEqual(len(tracker.states), 64)
                    self.assertTrue(all(module.forward.__func__ is not original for _, module in modules))
                    self.assertIs(type(modules[0][1]).forward, original)
                    with self.assertRaisesRegex(RuntimeError, 'installed twice'):
                        ma8.install_activation_cache(model, arm)

    def test_late_invalid_module_or_missing_scope_never_half_patches(self):
        for defect in ('last_shape', 'missing_module', 'source_hash', 'recompute'):
            with self.subTest(defect=defect):
                model, modules, packages, original = self.fixture(changed_column=defect == 'source_hash')
                if defect == 'last_shape': modules[-1][1].weight.shape = (999, 4096)
                if defect == 'missing_module': modules.pop()
                if defect == 'recompute': model.config.recompute_granularity = 'full'
                with patch.dict(sys.modules, packages):
                    with self.assertRaises(RuntimeError):
                        ma8.install_activation_cache(model, 'F4-cache')
                self.assertTrue(all(module.forward.__func__ is original for _, module in modules))
                self.assertFalse(hasattr(model, '_ma8_activation_cache_tracker'))


if __name__ == '__main__':
    unittest.main()
