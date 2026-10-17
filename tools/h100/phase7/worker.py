"""One fresh H100 TP8 worker window. Never import Torch without explicit execution."""
import argparse
from contextlib import nullcontext
import functools
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

if os.environ.get('PHASE7_EXECUTE') != '1':
    raise RuntimeError('Phase7 GPU worker requires the controller run --execute')

from protocol import MODEL, PROTOCOL, VARIANTS, digest, read, require, sha, verify, write_new


def profiler(job):
    return (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA])
            if job['stage'] == 'profile' else nullcontext())


def save_trace(trace, record):
    if trace is not None:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='phase7-trace-') as temp:
            raw = Path(temp) / 'trace.json'
            trace.export_chrome_trace(str(raw))
            target = OUTPUT / f'rank{record["rank"]}-profile.json.gz'
            with raw.open('rb') as source, gzip.open(target, 'wb') as dest:
                import shutil
                shutil.copyfileobj(source, dest)


def tensor_hash(items):
    h = hashlib.sha256()
    for name, tensor in items:
        h.update(name.encode())
        h.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def operator_work(m, plans, job, record):
    from runtime import ChunkedGemmRS
    from adapter import ChunkedLinear, check_output
    plan = plans[m['scope']]
    shape = plan['shape']
    gen = torch.Generator(device='cuda').manual_seed(MODEL['seed'] + dist.get_rank())
    x = torch.randn((shape['M'], shape['K_local']), generator=gen, device='cuda', dtype=torch.bfloat16)
    w = torch.randn((shape['N'], shape['K_local']), generator=gen, device='cuda', dtype=torch.bfloat16) * .02
    record['fingerprints'] = dict(input=tensor_hash([('input', x)]), weight=tensor_hash([('weight', w)]))
    op = ChunkedGemmRS(dist.group.WORLD, plan, mode=VARIANTS[job['variant']]['mode'], profile=job['stage'] == 'profile')
    if job['stage'] == 'smoke':
        with torch.no_grad():
            output = op.forward(x, w)
            record['forward_checks'] = {m['scope']: check_output(x, w, output, dist.group.WORLD, plan['quantize_mask'])}
        # Exercise the actual STE path, compare against the fixed full BF16 derivative.
        xx, ww = x.detach().requires_grad_(True), w.detach().requires_grad_(True)
        grad = torch.randn((shape['M'] // 8, shape['N']), generator=gen, device='cuda', dtype=torch.bfloat16) * .01
        ChunkedLinear.apply(xx, ww, op).backward(grad)
        full = torch.empty((shape['M'], shape['N']), dtype=torch.bfloat16, device='cuda')
        dist.all_gather_into_tensor(full, grad, group=dist.group.WORLD)
        expected_x, expected_w = full.matmul(w), full.t().matmul(x)
        for actual, expected in ((xx.grad, expected_x), (ww.grad, expected_w)):
            require(bool(torch.isfinite(actual).all()), 'Nonfinite STE gradient')
            require(torch.allclose(actual, expected, rtol=.01, atol=.01), 'STE gradient mismatch')
        record['backward_checked'] = True
    with torch.no_grad():
        for _ in range(m['warmup']):
            op.forward(x, w)
        dist.barrier(); torch.cuda.synchronize()
        with profiler(job) as trace:
            start = time.perf_counter()
            for _ in range(record['steps']):
                output = op.forward(x, w)
            torch.cuda.synchronize()
            record['elapsed_seconds'] = time.perf_counter() - start
        save_trace(trace, record)
        require(bool(torch.isfinite(output).all()), 'Nonfinite output')
    record.update(losses=[], grad_norms=[], skips=0, runtime=op.last_diagnostics)


def model_args():
    flags = []
    for flag, key in [('num-layers', 'layers'), ('hidden-size', 'hidden'), ('ffn-hidden-size', 'ffn'),
                      ('num-attention-heads', 'heads'), ('seq-length', 'sequence'), ('max-position-embeddings', 'sequence'),
                      ('micro-batch-size', 'micro_batch'), ('global-batch-size', 'global_batch'), ('seed', 'seed'),
                      ('vocab-size', 'vocab'), ('tensor-model-parallel-size', 'tp')]:
        flags += ['--' + flag, str(MODEL[key])]
    return flags + ['--transformer-impl', 'local', '--position-embedding-type', 'learned_absolute',
                    '--untie-embeddings-and-output-weights', '--no-persist-layer-norm', '--no-masked-softmax-fusion',
                    '--no-bias-gelu-fusion', '--no-bias-dropout-fusion', '--no-rope-fusion', '--train-iters', '1000', '--bf16',
                    '--optimizer', 'adam', '--lr', '1e-4', '--min-lr', '1e-4', '--lr-decay-style', 'constant',
                    '--weight-decay', '0.01', '--adam-beta1', '0.9', '--adam-beta2', '0.95', '--clip-grad', '1.0',
                    '--init-method-std', '0.02', '--hidden-dropout', '0', '--attention-dropout', '0',
                    '--no-gradient-accumulation-fusion', '--pipeline-model-parallel-size', '1', '--context-parallel-size', '1',
                    '--sequence-parallel', '--mock-data', '--tokenizer-type', 'NullTokenizer', '--split', '100,0,0',
                    '--num-workers', '0', '--eval-iters', '0', '--timing-log-level', '0', '--no-check-for-nan-in-loss-and-grad']


def model_work(m, plans, job, record):
    from megatron.training.initialize import initialize_megatron
    from megatron.training import get_args, get_timers
    from megatron.training.training import setup_model_and_optimizer
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.utils import get_model_config
    from megatron.training.utils import unwrap_model
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from pretrain_gpt import model_provider, loss_func
    from adapter import Adapter
    # Megatron initializes its own distributed groups, with the same fixed arguments as Phase6.
    sys.argv = [sys.argv[0], *model_args()]
    initialize_megatron()
    args = get_args()
    require(args.tensor_model_parallel_size == 8 and args.data_parallel_size == args.pipeline_model_parallel_size == args.context_parallel_size == 1,
            'Wrong model parallel configuration')
    require(args.bf16 and args.sequence_parallel and not args.gradient_accumulation_fusion,
            'Unsupported precision/parallel mode')
    model, optimizer, scheduler = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.finalize_model_grads_func = finalize_model_grads
    config.timers = get_timers()
    raw = unwrap_model(model)[0]
    model[0].train()
    record['fingerprints'] = dict(parameters=tensor_hash(raw.named_parameters()),
        rng=tensor_hash([('cpu', torch.get_rng_state()), ('cuda', torch.cuda.get_rng_state()),
                        *sorted(get_cuda_rng_tracker().get_states().items())]))
    adapter = Adapter(raw, plans, VARIANTS[job['variant']]['mode'], smoke=job['stage'] == 'smoke', profile=job['stage'] == 'profile')
    gen = torch.Generator(device='cuda').manual_seed(MODEL['token_seed'])
    tokens = torch.randint(0, MODEL['vocab'], (4, 1, 2049), generator=gen, device='cuda')
    record['fingerprints']['tokens'] = tensor_hash([('tokens', tokens)])
    positions = torch.arange(2048, device='cuda').unsqueeze(0)
    mask = torch.triu(torch.ones((1, 1, 2048, 2048), dtype=torch.bool, device='cuda'), diagonal=1)
    loss_mask = torch.ones((1, 2048), device='cuda')
    batches = [dict(tokens=t[:, :-1].contiguous(), labels=t[:, 1:].contiguous(),
                    position_ids=positions, attention_mask=mask, loss_mask=loss_mask) for t in tokens]
    forward_backward = get_forward_backward_func()

    def forward_step(iterator, current_model):
        batch = next(iterator)
        value = current_model(batch['tokens'], batch['position_ids'], batch['attention_mask'],
                              labels=batch['labels'], loss_mask=batch['loss_mask'])
        return value, functools.partial(loss_func, batch['loss_mask'])

    def clear():
        for chunk in model:
            chunk.zero_grad_buffer()
        optimizer.zero_grad()

    def step():
        losses = forward_backward(forward_step_func=forward_step, data_iterator=iter(batches), model=model,
                                  num_microbatches=4, seq_length=2048, micro_batch_size=1,
                                  decoder_seq_length=None, forward_only=False)
        success, grad_norm, _ = optimizer.step()
        scheduler.step(increment=4)
        clear()
        return losses, success, grad_norm

    def loss_number(losses):
        return float(sum(d['lm loss'][0] for d in losses) / sum(d['lm loss'][1] for d in losses))

    clear()
    for _ in range(m['warmup']):
        losses, ok, norm = step()
        require(bool(ok) and math.isfinite(float(norm)), 'Warmup failed')
    before = adapter.calls.copy()
    dist.barrier(); torch.cuda.synchronize()
    trajectory = []
    with profiler(job) as trace:
        start = time.perf_counter()
        for _ in range(record['steps']):
            trajectory.append(step())
        torch.cuda.synchronize()
        record['elapsed_seconds'] = time.perf_counter() - start
    save_trace(trace, record)
    record.update(losses=[loss_number(x[0]) for x in trajectory], grad_norms=[float(x[2]) for x in trajectory],
                  skips=sum(not bool(x[1]) for x in trajectory), forward_checks=adapter.checks,
                  module_calls={name: count - before[name] for name, count in adapter.calls.items()},
                  padded_vocab_size=args.padded_vocab_size, tokens_per_step=8192,
                  data_scope='Fixed synthetic batch, fresh model per window; no convergence claim')
    require(all(c == record['steps'] * 4 for c in record['module_calls'].values()), 'Missing projection calls')
    require(record['skips'] == 0 and all(math.isfinite(x) for x in record['losses'] + record['grad_norms']), 'Training failed')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    a = parser.parse_args()
    root = a.root.resolve()
    m = verify(root)
    job = json.loads(os.environ['PHASE7_JOB'])
    require(job['variant'] in VARIANTS and job['stage'] in ('smoke', 'timing', 'profile'), 'Unknown worker job')
    if job['stage'] == 'timing':
        require({k: job[k] for k in ('round', 'block', 'variant', 'directory')} in m['schedule'], 'Unscheduled window')
    folder = root / 'results' / job['directory']
    require(folder.is_dir() and folder.resolve().is_relative_to(root / 'results'), 'Invalid output directory')
    global torch, dist, OUTPUT
    OUTPUT = folder
    import torch
    import torch.distributed as dist
    local_rank = int(os.environ['LOCAL_RANK'])
    require(int(os.environ['WORLD_SIZE']) == 8 and int(os.environ.get('LOCAL_WORLD_SIZE', 8)) == 8, 'Single-node TP8 required')
    torch.cuda.set_device(local_rank)
    require(all(os.environ.get(name, '0') == '0' for name in ('TORCH_NCCL_BLOCKING_WAIT', 'NCCL_BLOCKING_WAIT')),
            'Blocking NCCL Work.wait would serialize the prototype; unset blocking-wait options')
    device = torch.cuda.get_device_properties(local_rank)
    require(device.major == 9 and device.minor == 0 and 'H100' in device.name and device.multi_processor_count == 132,
            'Only 132-SM H100 execution is admitted')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    roles = ('attention', 'mlp') if m['scope'] == 'both' else (m['scope'],)
    order = VARIANTS[job['variant']]['ordering']
    plans = {r: read(root / 'plans' / f'{r}-{order}.json') for r in roles}
    record = dict(protocol=PROTOCOL, run_id=m['run_id'], manifest_sha256=sha(root / 'manifest.json'),
                  rank=int(os.environ['RANK']), world_size=8, **job, kind=m['kind'], scope=m['scope'],
                  steps=1 if job['stage'] == 'profile' else m['steps'], forward_checks={}, completed=False, passed=False,
                  variant_definition=VARIANTS[job['variant']], plans_sha256={r: digest(p) for r, p in plans.items()},
                  device=dict(name=device.name, capability=[device.major, device.minor], sm_count=device.multi_processor_count,
                              uuid=str(getattr(device, 'uuid', local_rank))),
                  execution=dict(torch=torch.__version__, cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
                                 python=sys.version, connection_limit=os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'default'),
                                 blocking_wait=False),
                  model_config=m['model'], implementation=m['implementation'])
    try:
        if m['kind'] == 'operator':
            dist.init_process_group('nccl')
            operator_work(m, plans, job, record)
        else:
            model_work(m, plans, job, record)
        record.update(completed=True, passed=True)
        write_new(folder / f'rank{record["rank"]}.json', record)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
