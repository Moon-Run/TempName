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
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim, dist_all_gather_func
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


class _MA6ReferenceConsumerGemmRSFunction(torch.autograd.Function):
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


PROTOCOL = 'h100-ma6-f4-v2'
ARMS = ('F4', 'F4-tile', 'F4-gather')
GATHER_SCHEDULES = ('overlap', 'wait-first')
DEFAULT_GATHER_SCHEDULE = 'overlap'

class BackwardState(dict):
    """One module's counters; private fields retain ownership and failed work."""

    def __init__(self, arm, schedule=DEFAULT_GATHER_SCHEDULE):
        if arm not in ARMS:
            raise ValueError('Unknown MA6 arm: ' + str(arm))
        if schedule not in GATHER_SCHEDULES:
            raise ValueError('Unknown MA6 gather schedule: ' + str(schedule))
        gather_schedule = schedule if arm == 'F4-gather' else 'baseline'
        super().__init__(protocol=PROTOCOL, arm=arm, gather_schedule=gather_schedule, forward_calls=0,
                         backward_calls=0, baseline_backward_calls=0,
                         async_gather_calls=0, wait_calls=0,
                         failed=False, active=False)
        self._ma6_arm = arm
        self._ma6_schedule = gather_schedule
        self._ma6_gather_metadata = None
        self._ma6_op_id = None
        self._ma6_pending = None


def new_backward_state(arm, schedule=DEFAULT_GATHER_SCHEDULE):
    return BackwardState(arm, schedule)


def _ma6_validate_state(state, op=None):
    expected = {'protocol', 'arm', 'gather_schedule', 'forward_calls', 'backward_calls',
                'baseline_backward_calls', 'async_gather_calls', 'wait_calls',
                'failed', 'active'}
    if (not isinstance(state, dict) or set(state) != expected or
            state['protocol'] != PROTOCOL or state['arm'] not in ARMS or
            not hasattr(state, '_ma6_op_id') or not hasattr(state, '_ma6_pending') or
            getattr(state, '_ma6_arm', None) != state['arm'] or
            getattr(state, '_ma6_schedule', None) != state['gather_schedule'] or
            state['gather_schedule'] not in (GATHER_SCHEDULES if state['arm'] == 'F4-gather' else ('baseline',))):
        raise RuntimeError('Invalid MA6 backward state')
    if state['failed'] or state['active'] or state._ma6_pending is not None:
        state['failed'] = True
        raise RuntimeError('MA6 cannot retry or reenter a failed backward')
    if op is not None:
        if state._ma6_op_id is None:
            state._ma6_op_id = id(op)
        elif state._ma6_op_id != id(op):
            state['failed'] = True
            raise RuntimeError('MA6 backward state belongs to another projection instance')


def _ma6_gather_metadata(grad_output, state):
    # The experiment fixes TP topology/device/layout. Bind expensive process-
    # group and CUDA-device discovery once (during warmup), then reject tensor
    # signature drift before any collective. Do not cache tensors or Work.
    signature = (tuple(grad_output.shape), grad_output.dtype, grad_output.device,
                 tuple(grad_output.stride()))
    metadata = state._ma6_gather_metadata
    if metadata is None:
        group = parallel_state.get_tensor_model_parallel_group()
        world = torch.distributed.get_world_size(group=group)
        if world != 8 or parallel_state.get_tensor_model_parallel_world_size() != world:
            raise RuntimeError('MA6 gather requires the frozen TP8 group')
        if grad_output.is_cuda and grad_output.device.index != torch.cuda.current_device():
            raise RuntimeError('MA6 gather input is not on the current CUDA device')
        full_shape = (signature[0][0] * world, *signature[0][1:])
        metadata = (signature, group, full_shape)
        state._ma6_gather_metadata = metadata
    elif signature != metadata[0]:
        raise RuntimeError('MA6 gather shape/dtype/device/stride changed after binding')
    return metadata


def _ma6_gather_and_bias(grad_output, bias_shape, state):
    _, group, shape = _ma6_gather_metadata(grad_output, state)
    gather_input = grad_output.contiguous()
    # Preserve independent output allocation and NCCL allocator ownership.
    full_grad = torch.empty(shape, dtype=gather_input.dtype, device=gather_input.device)
    work = dist_all_gather_func(full_grad, gather_input, group=group, async_op=True)
    state['async_gather_calls'] += 1
    state._ma6_pending = (work, gather_input, full_grad)
    if work is None or not callable(getattr(work, 'wait', None)):
        raise RuntimeError('MA6 asynchronous gather did not return a Work handle')
    if state['gather_schedule'] == 'wait-first':
        # S6 control: same helper/cache/collective, communication-first but
        # serialize bias on the current stream to isolate wait placement.
        work.wait()
        state['wait_calls'] += 1
        grad_bias = grad_output.sum_to_size(bias_shape)
    else:
        # Reduce the original local tensor, preserving reduction layout/tree.
        grad_bias = grad_output.sum_to_size(bias_shape)
        # Work.wait follows bias submission and precedes every full_grad read.
        work.wait()
        state['wait_calls'] += 1
    return full_grad, grad_bias


class ConsumerGemmRSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, residual, op, option, backward_state):
        _ma6_validate_state(backward_state, op)
        ctx.ma6_state = backward_state
        ctx.ma6_backward_done = False
        try:
            output = _MA6ReferenceConsumerGemmRSFunction.forward(
                ctx, input_, weight, bias, residual, op, option)
            backward_state['forward_calls'] += 1
            return output
        except BaseException:
            backward_state['failed'] = True
            raise

    @staticmethod
    def backward(ctx, grad_output):
        state = ctx.ma6_state
        _ma6_validate_state(state)
        if ctx.ma6_backward_done:
            state['failed'] = True
            raise RuntimeError('MA6 does not support repeated backward on a consumer output')
        state['active'] = True
        try:
            if state['arm'] == 'F4-gather':
                input_, weight = ctx.saved_tensors
                full_grad, grad_bias = _ma6_gather_and_bias(grad_output, ctx.bias_shape, state)
                grad_residual = grad_output
                grad_input = full_grad.matmul(weight)
                grad_weight = full_grad.reshape(-1, full_grad.shape[-1]).t().matmul(
                    input_.reshape(-1, input_.shape[-1]))
                result = (grad_input, grad_weight, grad_bias, grad_residual, None, None)
            else:
                # Same original math and bias -> gather -> dgrad -> wgrad order.
                result = _MA6ReferenceConsumerGemmRSFunction.backward(ctx, grad_output)
                state['baseline_backward_calls'] += 1
            # Keep input/output/Work references until all consumers have been
            # queued on the stream protected by Work.wait and NCCL's allocator
            # recording. On error they remain retained by the poisoned state.
            state._ma6_pending = None
            state['backward_calls'] += 1
            ctx.ma6_backward_done = True
            return (*result, None)
        except BaseException:
            state['failed'] = True
            raise
        finally:
            state['active'] = False


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
                output = ConsumerGemmRSFunction.apply(input_, module.weight, module.bias, residual, op, adapter.option, consumer['backward'])
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
        self._ma6_arm = os.environ.get('H100_MA6_ARM')
        if self._ma6_arm not in ARMS:
            raise RuntimeError('MA6 requires an explicit valid experiment arm')
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
                         backward=new_backward_state(self._ma6_arm))
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


    def ma6_backward_state(self):
        if self._consumer_failed:
            raise RuntimeError('MA6 cannot report a failed consumer adapter')
        result = {}
        for name, consumer in self._consumer_contexts.items():
            state = consumer['backward']
            _ma6_validate_state(state)
            if state['arm'] != self._ma6_arm:
                raise RuntimeError('MA6 backward arm identity changed')
            result[name] = dict(state)
        return result
