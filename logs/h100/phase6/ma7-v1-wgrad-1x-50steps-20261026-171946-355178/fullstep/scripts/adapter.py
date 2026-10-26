"""Restricted BF16 RowParallelLinear GEMM-RS integration for the frozen GPT trial.

Only attention.linear_proj and mlp.linear_fc2 forward paths are replaced.
Backward keeps the native sequence AllGather and local dgrad/wgrad GEMMs.
Unsupported shapes/configurations fail before timing rather than silently falling back.
"""
import types
import os
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import RowParallelLinear
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
from taco_support import TacoForward, check_taco_output, check_consumer_output
from routing import codec_placement, enabled, is_native, is_fused_v2, module_role, route
from megatron_taco import TensorTacoRS, QuantizedRS


class GemmRSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, op, option):
        ctx.save_for_backward(input_, weight)
        shape = input_.shape
        flat = input_.reshape(-1, shape[-1]).contiguous()
        # Give the autograd graph its own output lifetime, independent of Flux workspaces.
        output = op.forward(flat, weight, reduce_scatter_option=option)
        if not getattr(op, 'owns_output', False):
            output = output.clone()
        return output.reshape(shape[0] // parallel_state.get_tensor_model_parallel_world_size(),
                              shape[1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        full_grad = _gather_along_first_dim(grad_output.contiguous())
        grad_input = full_grad.matmul(weight)
        grad_weight = full_grad.reshape(-1, full_grad.shape[-1]).t().matmul(
            input_.reshape(-1, input_.shape[-1]))
        return grad_input, grad_weight, None, None


class NativeRSControl:
    def forward(self, input_, weight, reduce_scatter_option=None):
        partial = input_.matmul(weight.t())
        output = torch.empty((input_.shape[0] // parallel_state.get_tensor_model_parallel_world_size(), weight.shape[0]), device=input_.device, dtype=input_.dtype)
        torch.distributed.reduce_scatter_tensor(output, partial,
            group=parallel_state.get_tensor_model_parallel_group())
        return output


class _MA7ReferenceConsumerGemmRSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, residual, op, option):
        if input_.ndim != 3:
            raise RuntimeError('MA4-F requires a three-dimensional projection input')
        expected = (input_.shape[0] // parallel_state.get_tensor_model_parallel_world_size(),
                    input_.shape[1], weight.shape[0])
        if (tuple(residual.shape) != expected or not residual.is_contiguous() or
                residual.dtype != input_.dtype or tuple(bias.shape) != (weight.shape[0],)):
            raise RuntimeError('MA4-F residual shape/dtype/layout differs from the frozen consumer')
        ctx.save_for_backward(input_, weight)
        ctx.bias_shape = bias.shape
        shape = input_.shape
        flat = input_.reshape(-1, shape[-1]).contiguous()
        output = op.forward_consumer(flat, weight, bias, residual.reshape(-1, residual.shape[-1]),
                                     reduce_scatter_option=option)
        if output._base is not None or output.numel() != residual.numel():
            raise RuntimeError('MA4-F native consumer must return an independent complete output')
        # Native output is a fresh allocation. Equal-numel resize changes only
        # its metadata and avoids a new MakeViewlessTensor path after MLP BDA.
        return output.resize_(residual.shape)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        # The original bias AddBackward unbroadcast precedes linear backward.
        grad_bias = grad_output.sum_to_size(ctx.bias_shape)
        grad_residual = grad_output
        # Preserve the baseline linear backward, including its sequence gather.
        full_grad = _gather_along_first_dim(grad_output.contiguous())
        grad_input = full_grad.matmul(weight)
        grad_weight = full_grad.reshape(-1, full_grad.shape[-1]).t().matmul(
            input_.reshape(-1, input_.shape[-1]))
        return grad_input, grad_weight, grad_bias, grad_residual, None, None


PROTOCOL = 'h100-ma7-f4-wgrad-v1'
ARMS = ('F4', 'F4-wgrad')
BACKWARD_SCHEDULE = 'independent_weight_stream'

class BackwardState(dict):
    def __init__(self, arm, allow_cpu=False):
        if arm not in ARMS:
            raise ValueError('Unknown MA7 arm: ' + str(arm))
        super().__init__(protocol=PROTOCOL, arm=arm, backward_schedule=BACKWARD_SCHEDULE,
            forward_calls=0, backward_calls=0, baseline_backward_calls=0,
            deferred_wgrad_calls=0, cuda_wgrad_calls=0, cpu_wgrad_calls=0,
            main_grad_accumulations=0, standalone_grad_assignments=0,
            drained_wgrad_calls=0, drain_calls=0, pending_wgrad=0,
            peak_pending_wgrad=0, async_gather_calls=0, wait_calls=0,
            failed=False, active=False)
        self._arm = arm
        self._allow_cpu = bool(allow_cpu)
        self._op = None
        self._weight = None
        self._weight_ptr = None
        self._main_grad = None
        self._main_grad_ptr = None
        self._queue = None
        self._training = False
        self._in_microbatch = False


def new_backward_state(arm, allow_cpu=False):
    return BackwardState(arm, allow_cpu)


def _ma7_poison(state, message):
    state['failed'] = True
    if state._queue is not None:
        state._queue.poison()
    raise RuntimeError(message)


def _ma7_validate_state(state, op=None):
    expected = {'protocol', 'arm', 'backward_schedule', 'forward_calls', 'backward_calls',
        'baseline_backward_calls', 'deferred_wgrad_calls', 'cuda_wgrad_calls',
        'cpu_wgrad_calls', 'main_grad_accumulations', 'standalone_grad_assignments',
        'drained_wgrad_calls', 'drain_calls', 'pending_wgrad', 'peak_pending_wgrad',
        'async_gather_calls', 'wait_calls', 'failed', 'active'}
    if (not isinstance(state, dict) or set(state) != expected or
            state['protocol'] != PROTOCOL or state['arm'] not in ARMS or
            getattr(state, '_arm', None) != state['arm'] or
            state['backward_schedule'] != BACKWARD_SCHEDULE or
            state['async_gather_calls'] != 0 or state['wait_calls'] != 0):
        if isinstance(state, dict): state['failed'] = True
        raise RuntimeError('Invalid MA7 backward state')
    if state['failed'] or state['active']:
        _ma7_poison(state, 'MA7 cannot retry or reenter a failed backward')
    if op is not None:
        if state._op is None:
            state._op = op
        elif state._op is not op:
            _ma7_poison(state, 'MA7 backward state belongs to another projection instance')


def _ma7_bind_weight(state, weight):
    if (not weight.is_leaf or not weight.requires_grad or not weight.is_contiguous() or
            weight.dtype not in (torch.bfloat16, torch.float32) or
            (not weight.is_cuda and not state._allow_cpu)):
        _ma7_poison(state, 'MA7 requires a CUDA contiguous leaf weight (CPU is test-only)')
    main_grad = getattr(weight, 'main_grad', None)
    if main_grad is not None and (main_grad.shape != weight.shape or
            main_grad.dtype != torch.float32 or main_grad.device != weight.device or
            not main_grad.is_contiguous() or main_grad.requires_grad):
        _ma7_poison(state, 'MA7 requires the original contiguous FP32 main_grad buffer')
    if state._weight is None:
        state._weight = weight
        state._weight_ptr = weight.data_ptr()
        state._main_grad = main_grad
        state._main_grad_ptr = None if main_grad is None else main_grad.data_ptr()
    elif (state._weight is not weight or state._weight_ptr != weight.data_ptr() or
            state._main_grad is not main_grad or
            state._main_grad_ptr != (None if main_grad is None else main_grad.data_ptr())):
        _ma7_poison(state, 'MA7 weight or main_grad ownership/storage changed')
    if state._training and (main_grad is None or weight.dtype != torch.bfloat16 or
            getattr(weight, 'zero_out_wgrad', False)):
        _ma7_poison(state, 'MA7 training requires the frozen BF16 weight and native main_grad')


class DeferredWgradQueue:
    """One stream, at most one contribution per module and 64 per microbatch.

    Strong input/full_grad/result references remain alive until the microbatch
    stream join is enqueued. record_stream additionally protects their allocator
    lifetime when those references are released before GPU completion.
    """
    def __init__(self, max_pending=64):
        self.max_pending = max_pending
        self.states = []
        self.pending = []
        self.failed = False
        self.device = None
        self.stream = None
        self.main_stream = None

    def register(self, state, weight):
        if self.failed:
            _ma7_poison(state, 'MA7 queue is poisoned')
        if state._queue not in (None, self):
            _ma7_poison(state, 'MA7 state already belongs to another queue')
        if self.device is None:
            self.device = weight.device
            if weight.is_cuda and state['arm'] == 'F4-wgrad':
                self.main_stream = torch.cuda.current_stream(weight.device)
                self.stream = torch.cuda.Stream(device=weight.device)
        if self.device != weight.device:
            _ma7_poison(state, 'MA7 queue cannot cross devices')
        if all(existing is not state for existing in self.states):
            self.states.append(state)
        state._queue = self

    def poison(self):
        self.failed = True
        for state in self.states:
            state['failed'] = True

    def submit(self, state, input_, full_grad, weight):
        _ma7_bind_weight(state, weight)
        if (self.failed or state['pending_wgrad'] != 0 or
                len(self.pending) >= self.max_pending or
                (state._training and not state._in_microbatch)):
            _ma7_poison(state, 'MA7 queue boundary or bounded capacity violated')
        if input_.device != self.device or full_grad.device != self.device:
            _ma7_poison(state, 'MA7 queue inputs changed device')
        if state._main_grad is not None and weight.grad is not None:
            _ma7_poison(state, 'MA7 deferred weight unexpectedly has a native .grad')
        try:
            with torch.autograd.profiler.record_function('ma7.wgrad'):
                if weight.is_cuda:
                    current = torch.cuda.current_stream(weight.device)
                    if current != self.main_stream:
                        _ma7_poison(state, 'MA7 backward moved to another main stream')
                    # Called AFTER dgrad submission. Only aux waits here; the
                    # main stream never waits for wgrad inside Consumer.backward.
                    with torch.cuda.stream(self.stream):
                        self.stream.wait_stream(current)
                        input_.record_stream(self.stream)
                        full_grad.record_stream(self.stream)
                        weight.record_stream(self.stream)
                        grad_weight = full_grad.reshape(-1, full_grad.shape[-1]).t().matmul(
                            input_.reshape(-1, input_.shape[-1]))
                        if state._main_grad is not None:
                            state._main_grad.record_stream(self.stream)
                            state._main_grad.add_(grad_weight)
                        grad_weight.record_stream(self.stream)
                    state['cuda_wgrad_calls'] += 1
                else:
                    if not state._allow_cpu:
                        _ma7_poison(state, 'MA7 CPU execution is test-only')
                    grad_weight = full_grad.reshape(-1, full_grad.shape[-1]).t().matmul(
                        input_.reshape(-1, input_.shape[-1]))
                    if state._main_grad is not None:
                        state._main_grad.add_(grad_weight)
                    state['cpu_wgrad_calls'] += 1
            self.pending.append((state, input_, full_grad, weight, grad_weight))
            state['deferred_wgrad_calls'] += 1
            state['pending_wgrad'] += 1
            state['peak_pending_wgrad'] = max(state['peak_pending_wgrad'], state['pending_wgrad'])
            if state._main_grad is not None:
                state['main_grad_accumulations'] += 1
        except BaseException:
            self.poison()
            raise

    def drain(self):
        if self.failed:
            raise RuntimeError('MA7 cannot drain a poisoned queue')
        try:
            if self.pending and self.stream is not None:
                current = torch.cuda.current_stream(self.device)
                if current != self.main_stream:
                    raise RuntimeError('MA7 drain moved to another main stream')
                current.wait_stream(self.stream)
            for state, input_, full_grad, weight, grad_weight in self.pending:
                _ma7_bind_weight(state, weight)
                if state._main_grad is None:
                    # Operator preflight uses ordinary leaf gradients. Its
                    # explicit drain is part of the tested backward boundary.
                    if grad_weight.is_cuda:
                        grad_weight.record_stream(self.main_stream)
                    if weight.grad is None:
                        weight.grad = grad_weight
                    else:
                        weight.grad.add_(grad_weight)
                    state['standalone_grad_assignments'] += 1
                state['drained_wgrad_calls'] += 1
                state['pending_wgrad'] -= 1
            self.pending.clear()
        except BaseException:
            self.poison()
            raise


def drain_backward_state(state):
    """Standalone operator boundary; training uses one shared queue join."""
    _ma7_validate_state(state)
    if state._training or state['backward_calls'] != state['drain_calls'] + 1:
        _ma7_poison(state, 'MA7 standalone drain has no exactly-once backward')
    with torch.autograd.profiler.record_function('ma7.drain'):
        if state._queue is not None:
            state._queue.drain()
        state['drain_calls'] += 1


class ConsumerGemmRSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, residual, op, option, backward_state):
        state = backward_state
        _ma7_validate_state(state, op)
        _ma7_bind_weight(state, weight)
        if state['pending_wgrad']:
            _ma7_poison(state, 'MA7 requires the previous backward drain before another forward')
        if state['arm'] == 'F4-wgrad' and state._queue is None:
            queue = DeferredWgradQueue(max_pending=1)
            queue.register(state, weight)
        ctx.ma7_state = state
        ctx.ma7_backward_done = False
        try:
            output = _MA7ReferenceConsumerGemmRSFunction.forward(
                ctx, input_, weight, bias, residual, op, option)
            state['forward_calls'] += 1
            return output
        except BaseException:
            state['failed'] = True
            if state._queue is not None: state._queue.poison()
            raise

    @staticmethod
    def backward(ctx, grad_output):
        state = ctx.ma7_state
        _ma7_validate_state(state)
        if ctx.ma7_backward_done:
            _ma7_poison(state, 'MA7 does not support repeated backward')
        if torch.is_grad_enabled():
            _ma7_poison(state, 'MA7 does not support higher-order backward')
        state['active'] = True
        try:
            if state['arm'] == 'F4':
                result = _MA7ReferenceConsumerGemmRSFunction.backward(ctx, grad_output)
                state['baseline_backward_calls'] += 1
            else:
                input_, weight = ctx.saved_tensors
                grad_bias = grad_output.sum_to_size(ctx.bias_shape)
                grad_residual = grad_output
                full_grad = _gather_along_first_dim(grad_output.contiguous())
                grad_input = full_grad.matmul(weight)
                state._queue.submit(state, input_, full_grad, weight)
                result = (grad_input, None, grad_bias, grad_residual, None, None)
            state['backward_calls'] += 1
            ctx.ma7_backward_done = True
            return (*result, None)
        except BaseException:
            state['failed'] = True
            if state._queue is not None: state._queue.poison()
            raise
        finally:
            state['active'] = False


def consumer_apply(input_, weight, bias, residual, op, option, state):
    return ConsumerGemmRSFunction.apply(input_, weight, bias, residual, op, option, state)


class Adapter:
    def __init__(self, model, policy, shapes, ring_reduction=False, control_backward=False):
        self.policy = policy
        self.ring_reduction = ring_reduction
        self.control_backward = control_backward
        self.option = None
        self.allowed = {tuple(s) for s in shapes}
        self.modules = {}
        self.calls = {}
        self.audit = True
        self.observed = {}
        self.forward_checks = {}
        self.placements = {}
        self.selection_checks = {}
        self.native_quant_forwards = {}
        self.scope_ops = {}
        self.actual_routes = {}
        self.profile = os.environ.get('E2E_MODE') == 'profile'
        self.flux = None
        if not is_native(policy):
            import flux
            self.flux = flux
            flux.init_flux_shm(parallel_state.get_tensor_model_parallel_group())
            self.option = flux.ReduceScatterOption()
            for key, value in dict(use_1d_ring=True, use_p2p_read=True, use_cudaMemcpyAsync=False,
                                   use_gemmk=False, per_tile_flags=False, use_barrier_queue=False,
                                   num_blocks=6, n_split=1).items():
                setattr(self.option, key, value)
            self.option.ring_mode = flux.RingMode.Ring1D
        for name, module in model.named_modules():
            if not isinstance(module, RowParallelLinear):
                continue
            assert name.endswith(('.self_attention.linear_proj', '.mlp.linear_fc2')), name
            assert module.input_is_parallel and module.sequence_parallel
            assert not module.explicit_expert_comm and not module.gradient_accumulation_fusion
            assert module.weight.dtype == torch.bfloat16 or (policy == 'native' and module.weight.dtype == torch.float32)
            assert module.config._cpu_offloading_context is None
            self.modules[name] = module
            self.calls[name] = 0
            native = module.forward
            module.forward = types.MethodType(self.make_forward(name, native), module)
        assert len(self.modules) == 2 * model.config.num_layers, list(self.modules)
        self._install_consumers(model)

    def make_forward(self, name, native):
        adapter = self
        op = None
        expected_shape = None
        native_quant = None

        def forward(module, input_):
            nonlocal op, expected_shape, native_quant
            if adapter.audit:
                shape = (input_.shape[0] * input_.shape[1], module.output_size, module.input_size)
                assert shape in adapter.allowed, (name, shape)
                assert input_.dtype == module.weight.dtype
                assert input_.shape[-1] == module.input_size_per_partition
                adapter.calls[name] += 1
                adapter.observed[name] = dict(M=shape[0], N=shape[1], K_global=shape[2],
                    input_shape=list(input_.shape), input_stride=list(input_.stride()),
                    weight_shape=list(module.weight.shape), K_local=input_.shape[-1])
            if not enabled(adapter.policy, name):
                adapter.actual_routes[name] = 'native_bf16'
                adapter.placements[name] = 'native'
                return native(input_)
            if is_native(adapter.policy):
                if native_quant is None:
                    assert native.__func__ is RowParallelLinear.forward
                    op=TensorTacoRS(parallel_state.get_tensor_model_parallel_group(),
                                   input_.shape[0]*input_.shape[1],module.output_size)
                    adapter.scope_ops[name] = op
                    adapter.actual_routes[name] = 'tensor'
                    def quantized_rs(partial):
                        result=QuantizedRS.apply(partial,op)
                        if os.environ.get('E2E_MODE')=='smoke' and name not in adapter.forward_checks:
                            adapter.forward_checks[name]=check_taco_output(
                                partial.reshape(-1,partial.shape[-1]),result.reshape(-1,result.shape[-1]),op.group)
                        return result
                    # Copy only the function globals for this module: preserve native
                    # GEMM/bias/linear backward and avoid a process-global monkeypatch.
                    globals_=dict(native.__func__.__globals__,reduce_scatter_to_sequence_parallel_region=quantized_rs)
                    function=types.FunctionType(native.__func__.__code__,globals_,native.__func__.__name__,
                                                native.__func__.__defaults__,native.__func__.__closure__)
                    native_quant=types.MethodType(function,module)
                    adapter.native_quant_forwards[name]=native_quant
                    adapter.placements[name]='tensor'
                    adapter.selection_checks[name]=dict(mask_sha256=None,wire_bytes=op.wire_bytes_per_rank,
                                                        bf16_wire_bytes=op.bf16_wire_bytes_per_rank)
                return native_quant(input_)
            if op is None:
                expected_shape = tuple(input_.shape)
                if enabled(adapter.policy, name) and not is_native(adapter.policy):
                    placement = codec_placement(adapter.policy)
                    op = TacoForward(parallel_state.get_tensor_model_parallel_group(),
                        input_.shape[0] * input_.shape[1], module.output_size,
                        input_.shape[-1], adapter.policy, module_role(name))
                    adapter.scope_ops[name] = op
                    adapter.actual_routes[name] = 'candidate'
                    adapter.placements[name] = op.codec.placement
                    adapter.selection_checks[name] = dict(mask_sha256=getattr(op.codec,'selection_sha256',None),
                        wire_bytes=op.codec.wire_bytes_per_rank,bf16_wire_bytes=op.codec.bf16_wire_bytes_per_rank)
                else:
                    raise RuntimeError('Unexpected scope fallback')
                    op = adapter.flux.GemmRS(parallel_state.get_tensor_model_parallel_group(), 1,
                        input_.shape[0] * input_.shape[1], module.output_size,
                        torch.bfloat16, torch.bfloat16, transpose_weight=False,
                        fuse_reduction=False, ring_reduction=adapter.ring_reduction)
                    adapter.placements[name] = 'bf16'
            assert tuple(input_.shape) == expected_shape
            consumer = adapter._consumer_contexts.get(name)
            if consumer is None:
                output = GemmRSFunction.apply(input_, module.weight, op, adapter.option)
            else:
                residual = consumer['residual']
                if (residual is None or consumer['output'] is not None or
                        id(module.bias) != consumer['bias_parameter_id']):
                    raise RuntimeError('MA4-F projection has no matching residual phase or bias parameter')
                output = consumer_apply(input_, module.weight, module.bias, residual, op, adapter.option, consumer['backward'])
                consumer['projection_calls'] += 1
                consumer['output'] = output
            if os.environ.get('E2E_MODE') == 'smoke' and name not in adapter.forward_checks:
                with torch.no_grad():
                    flat = input_.reshape(-1, input_.shape[-1])
                    partial = flat.matmul(module.weight.t())
                    ref = torch.empty_like(output).reshape(-1, output.shape[-1])
                    torch.distributed.reduce_scatter_tensor(ref, partial, group=parallel_state.get_tensor_model_parallel_group())
                    actual = output.reshape_as(ref)
                    if adapter.placements[name] in ('fused','separate'):
                        if consumer is None:
                            adapter.forward_checks[name] = check_taco_output(partial, actual,
                                parallel_state.get_tensor_model_parallel_group(), getattr(op.codec,'selection_cpu',None))
                        else:
                            adapter.forward_checks[name] = check_consumer_output(partial, actual,
                                parallel_state.get_tensor_model_parallel_group(), getattr(op.codec,'selection_cpu',None),
                                module.bias, residual.reshape_as(actual))
                    else:
                        torch.testing.assert_close(actual, ref, atol=.02, rtol=.02)
                        delta = actual.float() - ref.float()
                        adapter.forward_checks[name] = dict(max_abs=delta.abs().max().item(), relative_l2=(delta.norm()/ref.float().norm()).item())

            if consumer is not None:
                return output, None
            if module.skip_bias_add:
                return output, module.bias
            return output + module.bias if module.bias is not None else output, None
        return forward

    def stop_audit(self):
        self.audit = False
        expected = {name: route(self.policy,name) for name in self.modules}
        assert self.actual_routes == expected
        assert set(self.scope_ops) == {name for name in self.modules if enabled(self.policy,name)}
        assert len({id(op) for op in self.scope_ops.values()}) == len(self.scope_ops)
        self.post_audit_routes = {}
        self.forward_preserved = {}
        self.timed_calls = {name:0 for name in self.modules}
        self.workspace_checks = {} if is_native(self.policy) else {name:op.workspace_info for name,op in self.scope_ops.items()}
        for field in ('packet_ptr','mask_ptr'):
            pointers = [slot[field] for slots in self.workspace_checks.values() for slot in slots if field in slot]
            assert all(p > 0 for p in pointers) and len(pointers) == len(set(pointers))
        for name,module in self.modules.items():
            if not enabled(self.policy,name):
                target = types.MethodType(RowParallelLinear.forward,module)
                actual = 'native_bf16'
            elif is_native(self.policy):
                target = self.native_quant_forwards[name]
                actual = 'tensor'
            else:
                target = module.forward
                assert self.scope_ops[name].owns_output  is True
                assert self.scope_ops[name].owns_configuration  is True
                actual = 'candidate'
            self.forward_preserved[name] = target.__func__.__code__ is RowParallelLinear.forward.__code__
            assert self.forward_preserved[name] == (actual != 'candidate')
            self.post_audit_routes[name] = actual
            if enabled(self.policy,name):
                op = self.scope_ops[name]
                self.selection_checks[name].update(role=module_role(name),
                    shape=[self.observed[name][k] for k in ('M','N','K_global')],
                    plan_sha256=None if is_native(self.policy) else op.plan_sha256)
            def timed(module,input_,_target=target,_name=name,_role=module_role(name),_route=actual):
                self.timed_calls[_name] += 1
                if self.profile:
                    with torch.profiler.record_function('scope.'+_role+'.'+_route):
                        return _target(input_)
                return _target(input_)
            module.forward = types.MethodType(timed,module)
        assert self.post_audit_routes == expected


    def _install_consumers(self, model):
        self._ma7_arm = os.environ.get('H100_MA7_ARM')
        if self._ma7_arm not in ARMS:
            raise RuntimeError('MA7 requires an explicit valid experiment arm')
        self._consumer_enabled = True
        self._consumer_failed = False
        self._consumer_contexts = {}
        if not self._consumer_enabled:
            return
        config = model.config
        if (config.hidden_dropout != 0 or config.attention_dropout != 0 or
                config.bias_dropout_fusion or config.recompute_granularity is not None or
                config.enable_cuda_graph or config.external_cuda_graph or
                getattr(config, 'fp32_residual_connection', False) or not model.training):
            raise RuntimeError('MA4-F requires frozen BF16 training, zero dropout, no recompute or CUDA graphs')
        layers = dict(model.named_modules())
        for name, module in self.modules.items():
            if (not module.skip_bias_add or module.bias is None or
                    module.bias.dtype != torch.bfloat16 or not module.bias.is_contiguous() or
                    tuple(module.bias.shape) != (4096,) or
                    not getattr(module.bias, 'sequence_parallel', False)):
                raise RuntimeError('MA4-F requires the original sequence-parallel BF16 projection bias')
            attention = name.endswith('.self_attention.linear_proj')
            suffix = '.self_attention.linear_proj' if attention else '.mlp.linear_fc2'
            layer = layers[name[:-len(suffix)]]
            if (layer.recompute_input_layernorm or layer.recompute_pre_mlp_layernorm or layer.recompute_mlp or
                    layer.hidden_dropout != 0):
                raise RuntimeError('MA4-F does not support layer recomputation or dropout')
            state = dict(residual=None, output=None, residual_version=None, projection_calls=0, bda_calls=0,
                         bias_parameter_id=id(module.bias),
                         backward=new_backward_state(self._ma7_arm))
            self._consumer_contexts[name] = state
            phase = '_forward_attention' if attention else '_forward_mlp'
            original = getattr(layer, phase)
            wrapper = self._consumer_phase(original, state, 0 if attention else 1,
                                           'hidden_states' if attention else 'residual')
            setattr(layer, phase, types.MethodType(wrapper, layer))
            setattr(layer, 'self_attn_bda' if attention else 'mlp_bda', self._consumer_bda(state))

    def _consumer_phase(self, original, state, index, keyword):
        adapter = self
        def forward(layer, *args, **kwargs):
            if adapter._consumer_failed or state['residual'] is not None:
                raise RuntimeError('MA4-F consumer phase is failed or reentrant')
            before = state['projection_calls']
            try:
                if not layer.training or not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
                    raise RuntimeError('MA4-F requires the frozen grad-enabled training phase')
                residual = args[index] if len(args) > index else kwargs[keyword]
                state['residual_version'] = residual._version
                state['residual'] = residual
                result = original(*args, **kwargs)
                if (state['projection_calls'] != before + 1 or state['bda_calls'] != state['projection_calls'] or
                        state['output'] is not None):
                    raise RuntimeError('MA4-F phase did not consume exactly one fused projection')
                return result
            except BaseException:
                adapter._consumer_failed = True
                raise
            finally:
                state['residual'] = None
                state['residual_version'] = None
                state['output'] = None
        return forward

    def _consumer_bda(self, state):
        def consume(output_with_bias, residual, prob):
            output, bias = output_with_bias
            if (prob != 0 or bias is not None or residual is not state['residual'] or
                    residual._version != state['residual_version'] or
                    output is not state['output'] or output is None):
                raise RuntimeError('MA4-F BDA ownership, residual identity or dropout changed')
            state['output'] = None
            state['bda_calls'] += 1
            return output
        def factory(training, fused):
            if not training or fused:
                raise RuntimeError('MA4-F requires the frozen unfused training BDA interface')
            return consume
        return factory

    def consumer_state(self):
        if not self._consumer_enabled:
            return {}
        if self._consumer_failed:
            raise RuntimeError('MA4-F consumer evidence requested after a failed forward')
        result = {}
        for name, state in self._consumer_contexts.items():
            if state['residual'] is not None or state['output'] is not None:
                raise RuntimeError('MA4-F retains an unfinished consumer phase')
            result[name] = dict(projection_calls=state['projection_calls'], bda_calls=state['bda_calls'],
                bias_parameter_id=state['bias_parameter_id'], native=self.scope_ops[name].codec.ma4_consumer_state())
        return result


    def install_wgrad_schedule(self, schedules, config, model):
        if getattr(self, '_ma7_schedule_installed', False):
            raise RuntimeError('MA7 schedule cannot be installed twice')
        if (len(model) != 1 or len(self._consumer_contexts) != 64 or
                getattr(config, 'gradient_accumulation_fusion', True) or
                not getattr(config, 'sequence_parallel', False) or
                getattr(config, 'pipeline_model_parallel_size', None) != 1 or
                getattr(config, 'context_parallel_size', None) != 1 or
                parallel_state.get_tensor_model_parallel_world_size() != 8 or
                parallel_state.get_data_parallel_world_size() != 1):
            raise RuntimeError('MA7 requires frozen TP8, DP1/PP1/CP1, non-fused gradient training')
        ddp = getattr(model[0], 'ddp_config', None)
        if (ddp is None or getattr(ddp, 'overlap_grad_reduce', True) or
                getattr(ddp, 'overlap_param_gather', True)):
            raise RuntimeError('MA7 requires the original non-overlap Megatron DDP')
        if getattr(schedules.backward_step, '_ma7_wrapped', False):
            raise RuntimeError('MA7 schedule is already owned by another adapter')
        finalize = config.finalize_model_grads_func
        if not callable(finalize):
            raise RuntimeError('MA7 requires the original finalize_model_grads boundary')
        queue = DeferredWgradQueue(max_pending=64)
        states = []
        for name, consumer in self._consumer_contexts.items():
            state = consumer['backward']
            state._training = True
            weight = self.modules[name].weight
            _ma7_bind_weight(state, weight)
            queue.register(state, weight)
            states.append(state)
        if len({id(state._weight) for state in states}) != 64:
            queue.poison()
            raise RuntimeError('MA7 does not support shared projection weights')
        original = schedules.backward_step
        adapter = self
        def backward_step(*args, **kwargs):
            try:
                if queue.failed or queue.pending or any(s._in_microbatch for s in states):
                    raise RuntimeError('MA7 backward microbatch boundary is unfinished or reentrant')
                before = {id(s): s['backward_calls'] for s in states}
                for state in states:
                    _ma7_validate_state(state)
                    _ma7_bind_weight(state, state._weight)
                    if state['drain_calls'] != state['backward_calls']:
                        raise RuntimeError('MA7 previous microbatch was not drained')
                    state._in_microbatch = True
                result = original(*args, **kwargs)
                with torch.autograd.profiler.record_function('ma7.drain'):
                    if any(s['backward_calls'] != before[id(s)] + 1 for s in states):
                        raise RuntimeError('MA7 requires exactly one backward per target per microbatch')
                    queue.drain()
                    for state in states:
                        state['drain_calls'] += 1
                        state._in_microbatch = False
                return result
            except BaseException:
                queue.poison()
                adapter._consumer_failed = True
                raise
        backward_step._ma7_wrapped = True
        def finalize_checked(*args, **kwargs):
            if (queue.failed or queue.pending or any(s._in_microbatch or s['pending_wgrad'] or
                    s['drain_calls'] != s['backward_calls'] for s in states)):
                queue.poison()
                raise RuntimeError('MA7 finalization crossed an unfinished backward')
            return finalize(*args, **kwargs)
        schedules.backward_step = backward_step
        config.finalize_model_grads_func = finalize_checked
        self._ma7_queue = queue
        self._ma7_schedule_installed = True

    def ma7_backward_state(self):
        if self._consumer_failed or not getattr(self, '_ma7_schedule_installed', False):
            raise RuntimeError('MA7 requires a healthy installed schedule')
        result = {}
        for name, consumer in self._consumer_contexts.items():
            state = consumer['backward']
            _ma7_validate_state(state)
            _ma7_bind_weight(state, self.modules[name].weight)
            if state['pending_wgrad'] or state._in_microbatch:
                raise RuntimeError('MA7 evidence requested before microbatch drain')
            result[name] = dict(state)
        return result
