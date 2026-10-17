"""Restricted Megatron integration for the Phase7 chunk prototype, with BF16 STE."""
import types
import torch
import torch.distributed as dist

from runtime import ChunkedGemmRS, reference_forward


class ChunkedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, op):
        ctx.save_for_backward(input_, weight)
        ctx.group = op.group
        ctx.world = dist.get_world_size(op.group)
        flat = input_.reshape(-1, input_.shape[-1]).contiguous()
        out = op.forward(flat, weight)
        return out.reshape(input_.shape[0] // ctx.world, *input_.shape[1:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        local = grad_output.contiguous()
        full = torch.empty((local.shape[0] * ctx.world, *local.shape[1:]),
                           dtype=local.dtype, device=local.device)
        dist.all_gather_into_tensor(full, local, group=ctx.group)
        grad_input = full.matmul(weight)
        grad_weight = full.reshape(-1, full.shape[-1]).t().matmul(input_.reshape(-1, input_.shape[-1]))
        return grad_input, grad_weight, None


def check_output(input_, weight, output, group, mask):
    """Independent full-GEMM/codec reference; diagnostic only, outside timing."""
    with torch.no_grad():
        coded = reference_forward(input_, weight, group, mask)
        zeros = [[0] * len(row) for row in mask]
        bf16 = reference_forward(input_, weight, group, zeros)
        norm = max(float(bf16.float().norm()), 1e-20)
        errors = dict(relative_l2=float((output.float() - bf16.float()).norm()) / norm,
                      protocol_relative_l2=float((output.float() - coded.float()).norm()) / norm,
                      max_abs=float((output.float() - bf16.float()).abs().max()))
        errors['passed'] = bool(torch.isfinite(output).all()) and errors['relative_l2'] <= .05 and errors['protocol_relative_l2'] <= .01
        if not errors['passed']:
            raise RuntimeError('Phase7 forward check failed: ' + str(errors))
        return errors


class Adapter:
    """Only linear_proj/linear_fc2, sequence-parallel BF16, no fused wgrad/expert path."""
    def __init__(self, model, plans, mode, smoke=False, profile=False):
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.layers import RowParallelLinear
        self.group = parallel_state.get_tensor_model_parallel_group()
        self.plans, self.mode, self.smoke, self.profile = plans, mode, smoke, profile
        self.modules, self.originals, self.ops, self.checks, self.calls = {}, {}, {}, {}, {}
        found = {'attention': 0, 'mlp': 0}
        # Validate every route before making any mutation.
        for name, module in model.named_modules():
            if not isinstance(module, RowParallelLinear):
                continue
            role = 'attention' if name.endswith('.self_attention.linear_proj') else 'mlp' if name.endswith('.mlp.linear_fc2') else None
            if role is None:
                raise ValueError('Unsupported row-parallel projection: ' + name)
            found[role] += 1
            if role not in plans:
                continue
            if not (module.input_is_parallel and module.sequence_parallel and not module.explicit_expert_comm
                    and not module.gradient_accumulation_fusion and module.weight.dtype == torch.bfloat16
                    and module.config._cpu_offloading_context is None):
                raise ValueError('Unsupported Phase7 module configuration: ' + name)
            shape = plans[role]['shape']
            if (module.output_size, module.input_size_per_partition) != (shape['N'], shape['K_local']):
                raise ValueError('Role/weight shape mismatch: ' + name)
            self.modules[name] = (module, role)
        if found != {'attention': 32, 'mlp': 32}:
            raise ValueError('Only the frozen 32-layer GPT configuration is supported')
        for name, (module, role) in self.modules.items():
            self.originals[name] = module.forward
            self.calls[name] = 0
            module.forward = types.MethodType(self._forward(name, role), module)

    def _forward(self, name, role):
        def forward(module, input_):
            plan = self.plans[role]
            flat = input_.reshape(-1, input_.shape[-1]).contiguous()
            if list(flat.shape) != [plan['shape']['M'], plan['shape']['K_local']] or input_.dtype != torch.bfloat16:
                raise ValueError('Unexpected activation shape/dtype: ' + name)
            if name not in self.ops:
                self.ops[name] = ChunkedGemmRS(self.group, plan, mode=self.mode, profile=self.profile)
            op = self.ops[name]
            output = ChunkedLinear.apply(input_, module.weight, op)
            if self.smoke and name not in self.checks:
                self.checks[name] = check_output(flat, module.weight,
                                                output.reshape(-1, output.shape[-1]), self.group,
                                                plan['quantize_mask'])
            self.calls[name] += 1
            output_bias = module.bias if module.skip_bias_add else None
            if not module.skip_bias_add and module.bias is not None:
                output = output + module.bias
            return output, output_bias
        return forward

    def restore(self):
        for name, (module, _) in self.modules.items():
            module.forward = self.originals[name]
