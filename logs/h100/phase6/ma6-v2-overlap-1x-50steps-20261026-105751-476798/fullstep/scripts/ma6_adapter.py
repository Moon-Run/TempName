"""MA6's matched consumer wrapper and bias-gradient / AllGather scheduling.

The original MA4-F autograd remains the reference for F4 and F4-tile. Only
F4-gather caches immutable TP metadata and moves the same gather before the
original bias reduction. The overlap / wait-first schedules differ only in
wait placement; gathered outputs remain independent per-call allocations.
CUDA overlap is measured separately, never inferred from async submission.
"""
import inspect

import ma4_f_adapter as baseline

PROTOCOL = 'h100-ma6-f4-v2'
GATHER_SCHEDULES = ('overlap', 'wait-first')
DEFAULT_GATHER_SCHEDULE = 'overlap'
ARMS = ('F4', 'F4-tile', 'F4-gather')


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


SUPPORT_SOURCE = '''
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


'''


ADAPTER_METHODS = '''
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
'''


def consumer_autograd_source(gather_schedule=DEFAULT_GATHER_SCHEDULE):
    """Standalone source; callers supply torch, TP helpers and gather alias."""
    if gather_schedule not in GATHER_SCHEDULES:
        raise ValueError('Unknown MA6 gather schedule: ' + str(gather_schedule))
    original = baseline.consumer_autograd_source()
    original = baseline.replace_once(original, 'class ConsumerGemmRSFunction(',
                                     'class _MA6ReferenceConsumerGemmRSFunction(')
    identity = ('PROTOCOL = ' + repr(PROTOCOL) + '\nARMS = ' + repr(ARMS) +
                '\nGATHER_SCHEDULES = ' + repr(GATHER_SCHEDULES) +
                '\nDEFAULT_GATHER_SCHEDULE = ' + repr(gather_schedule) + '\n\n')
    return (original + identity + inspect.getsource(BackwardState) + '\n\n' +
            inspect.getsource(new_backward_state) + '\n' + SUPPORT_SOURCE)


def adapter_source(source, gather_schedule=DEFAULT_GATHER_SCHEDULE):
    """Apply to the complete already-generated MA4-F adapter, exactly once."""
    source = baseline.replace_once(source, baseline.consumer_autograd_source(),
                                   consumer_autograd_source(gather_schedule))
    source = baseline.replace_once(source,
        'from megatron.core.tensor_parallel.mappings import _gather_along_first_dim',
        'from megatron.core.tensor_parallel.mappings import _gather_along_first_dim, dist_all_gather_func')
    source = baseline.replace_once(source,
        "        self._consumer_enabled = os.environ.get('H100_MA4_F_ARM') == 'F4'",
        """        self._ma6_arm = os.environ.get('H100_MA6_ARM')
        if self._ma6_arm not in ARMS:
            raise RuntimeError('MA6 requires an explicit valid experiment arm')
        self._consumer_enabled = True""")
    source = baseline.replace_once(source,
        "                         bias_parameter_id=id(module.bias))",
        "                         bias_parameter_id=id(module.bias),\n"
        "                         backward=new_backward_state(self._ma6_arm))")
    source = baseline.replace_once(source,
        'output = ConsumerGemmRSFunction.apply(input_, module.weight, module.bias, residual, op, adapter.option)',
        "output = ConsumerGemmRSFunction.apply(input_, module.weight, module.bias, residual, op, adapter.option, consumer['backward'])")
    return source + '\n' + ADAPTER_METHODS


def taco_source(source):
    """The native consumer interface remains exactly MA4-F's interface."""
    return baseline.taco_source(source)
