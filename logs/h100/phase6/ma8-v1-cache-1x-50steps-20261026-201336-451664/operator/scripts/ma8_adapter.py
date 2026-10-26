"""MA8: retain the forward sequence gather for QKV/FC1 weight gradients.

The pinned Megatron methods supply all arithmetic. The candidate changes only
the ownership of the forward gather output and its reuse in backward. No global
Megatron symbol is replaced; output-layer and row-parallel paths stay original.
"""
import hashlib
import inspect
import textwrap
import types

PROTOCOL = 'h100-ma8-f4-activation-cache-v1'
ARMS = ('F4', 'F4-cache')
CLASS_SHA256 = '0f1975a4582b1e31ce8d5a5a9189f820cfb8a35abb91a025e8c330a198258bea'
COLUMN_FORWARD_SHA256 = 'fd242d309b08e9711486439da4b3881513056deb16643f2d98d241da3191a8b9'
SUFFIX_ROLES = {'.self_attention.linear_qkv': 'qkv', '.mlp.linear_fc1': 'fc1'}


def normalized(source):
    return textwrap.dedent(source).strip() + '\n'


def require_source(source, expected):
    source = normalized(source)
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise RuntimeError('MA8 requires the pinned, unmodified Megatron source')
    return source


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('MA8 source anchor changed: ' + old[:100])
    return source.replace(old, new)


def cached_class_source(source):
    """Produce a bounded source change; original backward math remains literal."""
    source = require_source(source, CLASS_SHA256)
    source = replace_once(source,
        'class LinearWithGradAccumulationAndAsyncCommunication(', 'class _MA8CachedRawLinear(')
    source = replace_once(source, '        ctx.save_for_backward(input, weight)\n', '')
    source = replace_once(source,
        '            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")',
        '            all_gather_buffer = torch.empty(dim_size, dtype=input.dtype, device=input.device)')
    source = replace_once(source, '        output = torch.matmul(total_input, weight.t())',
        '        ctx.save_for_backward(input, weight, total_input)\n'
        '        output = torch.matmul(total_input, weight.t())')
    source = replace_once(source, '        input, weight = ctx.saved_tensors',
        '        input, weight, retained_input = ctx.saved_tensors')
    start = source.index('        if wgrad_compute:\n            if ctx.sequence_parallel:')
    end = source.index('        grad_input = grad_output.matmul(weight)', start)
    source = source[:start] + '        if wgrad_compute:\n            total_input = retained_input\n' + source[end:]
    source = replace_once(source,
        '        if ctx.sequence_parallel and wgrad_compute:\n            handle.wait()\n\n', '')
    return source


class CacheState(dict):
    def __init__(self, name, role, arm):
        if arm not in ARMS or role not in ('qkv', 'fc1'):
            raise ValueError('Unknown MA8 arm/column role')
        super().__init__(protocol=PROTOCOL, arm=arm, role=role,
            forward_calls=0, backward_calls=0, forward_all_gathers=0,
            backward_all_gathers=0, retained_forward_inputs=0,
            reused_backward_inputs=0, reduce_scatters=0, pending=0,
            peak_pending=0, retained_bytes=0, peak_retained_bytes=0, failed=False)
        self.name = name
        self.arm = arm


def make_functions(torch, reference, cached):
    """Add symmetric path evidence around the exact raw Function methods."""
    def forward(ctx, input, weight, bias, fusion, allreduce, sequence, buffer, limit, state):
        if (state['failed'] or state['pending'] or state['arm'] != state.arm or
                fusion or allreduce or not sequence or buffer is not None or limit is not None):
            state['failed'] = True
            raise RuntimeError('MA8 requires one live, nondeferred sequence-parallel call per module')
        ctx.ma8_state, ctx.ma8_consumed = state, False
        raw = cached if state.arm == 'F4-cache' else reference
        state['pending'] = 1
        state['peak_pending'] = 1
        try:
            with torch.autograd.profiler.record_function('ma8.column.forward.' + state.name):
                output = raw.forward(ctx, input, weight, bias, fusion, allreduce, sequence, buffer, limit)
            state['forward_calls'] += 1
            state['forward_all_gathers'] += 1
            if state.arm == 'F4-cache':
                state['retained_forward_inputs'] += 1
                # The output's sequence dimension is the gathered dimension;
                # its hidden dimension belongs to weight, not the saved input.
                state['retained_bytes'] = output.shape[0] * input.shape[1] * input.shape[2] * input.element_size()
                state['peak_retained_bytes'] = max(state['peak_retained_bytes'], state['retained_bytes'])
            return output
        except BaseException:
            state['failed'] = True
            raise

    def backward(ctx, grad_output):
        state = ctx.ma8_state
        if state['failed'] or ctx.ma8_consumed or state['pending'] != 1:
            state['failed'] = True
            raise RuntimeError('MA8 rejects repeated, missing or failed backward')
        try:
            raw = cached if state.arm == 'F4-cache' else reference
            with torch.autograd.profiler.record_function('ma8.column.backward.' + state.name):
                gradients = raw.backward(ctx, grad_output)
            state['backward_calls'] += 1
            state['reduce_scatters'] += 1
            state['backward_all_gathers'] += int(state.arm == 'F4')
            state['reused_backward_inputs'] += int(state.arm == 'F4-cache')
            state['pending'] = 0
            state['retained_bytes'] = 0
            ctx.ma8_consumed = True
            return (*gradients, None)
        except BaseException:
            state['failed'] = True
            raise

    class MA8ReferenceLinear(torch.autograd.Function):
        pass

    class MA8RetainedActivationLinear(torch.autograd.Function):
        pass

    for function in (MA8ReferenceLinear, MA8RetainedActivationLinear):
        function.forward = staticmethod(forward)
        function.backward = staticmethod(backward)
    return {'F4': MA8ReferenceLinear, 'F4-cache': MA8RetainedActivationLinear}


class ActivationCacheTracker:
    def __init__(self, arm, states):
        self.arm, self.states = arm, states

    def state(self):
        if any(s['failed'] or s['pending'] or s['retained_bytes'] for s in self.states.values()):
            raise RuntimeError('MA8 evidence requested with failed or unfinished column work')
        return {name: dict(state) for name, state in self.states.items()}


def _bind_column(module, state, function, original_forward, torch):
    weight_identity, weight_pointer = module.weight, module.weight.data_ptr()

    def dispatch(input, weight, bias, gradient_accumulation_fusion, allreduce_dgrad,
                 sequence_parallel, grad_output_buffer=None, wgrad_deferral_limit=None,
                 async_grad_allreduce=None):
        if (not module.training or not torch.is_grad_enabled() or torch.is_inference_mode_enabled() or
                weight is not weight_identity or weight.data_ptr() != weight_pointer or
                not weight.requires_grad or tuple(input.shape) != (256, 1, 4096) or
                not input.is_cuda or input.device != weight.device or input.dtype != torch.bfloat16 or
                not input.is_contiguous() or async_grad_allreduce is not None):
            state['failed'] = True
            raise RuntimeError('MA8 column training shape, ownership or dtype changed')
        return function.apply(input, weight, bias, gradient_accumulation_fusion,
            allreduce_dgrad, sequence_parallel, grad_output_buffer, wgrad_deferral_limit, state)

    # ColumnParallelLinear.forward resets _forward_impl at every invocation.
    # Clone only this instance's forward globals so that assignment selects our
    # dispatcher, while its complete original method body remains unchanged.
    namespace = dict(original_forward.__globals__)
    namespace['linear_with_grad_accumulation_and_async_allreduce'] = dispatch
    method = types.FunctionType(original_forward.__code__, namespace,
        original_forward.__name__, original_forward.__defaults__, original_forward.__closure__)
    method.__kwdefaults__ = original_forward.__kwdefaults__
    module.forward = types.MethodType(method, module)


def install_activation_cache(model, arm):
    """Install once, after the F4 row projection adapter and before training."""
    import torch
    from megatron.core.tensor_parallel import layers

    if arm not in ARMS or hasattr(model, '_ma8_activation_cache_tracker'):
        raise RuntimeError('MA8 arm is invalid or adapter was installed twice')
    config = model.config
    if (not model.training or config.num_layers != 32 or config.hidden_size != 4096 or
            config.tensor_model_parallel_size != 8 or config.pipeline_model_parallel_size != 1 or
            not config.sequence_parallel or config.gradient_accumulation_fusion or
            config.recompute_granularity is not None or config.enable_cuda_graph or
            config.external_cuda_graph or getattr(config, 'cpu_offloading', False) or
            config.defer_embedding_wgrad_compute):
        raise RuntimeError('MA8 requires frozen TP8/PP1 BF16 training without recompute or offload')
    reference = layers.LinearWithGradAccumulationAndAsyncCommunication
    source = require_source(inspect.getsource(reference), CLASS_SHA256)
    original_forward = layers.ColumnParallelLinear.forward
    require_source(inspect.getsource(original_forward), COLUMN_FORWARD_SHA256)
    namespace = dict(vars(layers))
    exec(compile(cached_class_source(source), '<ma8-cached-pinned-megatron>', 'exec'), namespace)
    functions = make_functions(torch, reference, namespace['_MA8CachedRawLinear'])
    states, selected = {}, []
    for name, module in model.named_modules():
        roles = [role for suffix, role in SUFFIX_ROLES.items() if name.endswith(suffix)]
        if not roles:
            continue
        role = roles[0]
        if (type(module) is not layers.ColumnParallelLinear or not module.training or
                not module.sequence_parallel or module.allreduce_dgrad or
                module.gradient_accumulation_fusion or module.explicit_expert_comm or
                module.disable_grad_reduce or module.gather_output or
                module.weight is None or not module.weight.requires_grad or
                module.weight.dtype != torch.bfloat16 or not module.weight.is_cuda or
                not module.weight.is_contiguous() or
                tuple(module.weight.shape) != (1536 if role == 'qkv' else 2048, 4096) or
                getattr(module.forward, '__func__', None) is not original_forward):
            raise RuntimeError('MA8 column module is incompatible: ' + name)
        state = CacheState(name, role, arm)
        states[name] = state
        selected.append((module, state))
    if (len(states) != 64 or sum(s['role'] == 'qkv' for s in states.values()) != 32 or
            sum(s['role'] == 'fc1' for s in states.values()) != 32):
        raise RuntimeError('MA8 must cover all 32 QKV and 32 FC1 modules')
    # All validation precedes mutation: failed installation cannot half-patch a model.
    for module, state in selected:
        _bind_column(module, state, functions[arm], original_forward, torch)
    tracker = ActivationCacheTracker(arm, states)
    model._ma8_activation_cache_tracker = tracker
    return tracker
