"""Fresh process, complete-operator same-codec calibration worker."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/'config.json').read_text())
if (os.environ.get('H100_OP2_PROBE_EXECUTE') != '1' or PLAN['source_only'] or
        not PLAN['executable']):
    raise RuntimeError('Executable probe with explicit --execute required')
job = PLAN['schedule'][int(os.environ['H100_OP2_PROBE_JOB'])]
label = job['label']
span = dict(PLAN['catalog'])[label]
if any(os.environ.get(key) != str(span) for key in
       ('H100_OP2_ATTN_SPAN', 'H100_OP2_MLP_SPAN')):
    raise RuntimeError('Probe scheduler environment differs from the frozen candidate')
from communication import verify as verify_communication
comm = verify_communication(PLAN['communication'])
import torch
import torch.distributed as dist
rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
if world != 8 or rank != local:
    raise RuntimeError('Probe requires a complete one-node TP8 group')
torch.cuda.set_device(local)
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
from runtime import check_topology
hardware = check_topology(world)
import flux
from flux.gemm_rs_taco import GemmRSTaco
from codec_reference import check_taco_output
dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10))
group = dist.new_group(list(range(world)))
policy = PLAN['A'] if label == 'A' else PLAN['S']
expected_runtime = PLAN['runtime_identities'][policy]
library = (Path(flux.__file__).parent/'lib/libflux_cuda.so').resolve()
library_hashes = {name: hashlib.sha256(library.with_name(name).read_bytes()).hexdigest()
                  for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
if (library.parent != (Path(expected_runtime['root'])/'python/flux/lib').resolve() and
        library != (Path(expected_runtime['root'])/'libflux_cuda.so').resolve()):
    raise RuntimeError('Unexpected isolated Flux runtime')
if library_hashes != expected_runtime['libraries']:
    raise RuntimeError('Probe loaded different library bytes')
def tensor_hash(value):
    data = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()
record = dict(protocol=PLAN['protocol'], probe_protocol=PLAN['probe_protocol'],
              run_id=PLAN['run_id'], plan_sha256=hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),
              rank=rank, world_size=world, job=job, policy=policy, span=span,
              hardware=hardware, communication=comm, library_hashes=library_hashes,
              profiling=False, sampling=False, timing=PLAN['timing'], completed=False,
              software=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                            nccl=list(torch.cuda.nccl.version())),
              results={})
for role_index, (role, shape) in enumerate(PLAN['roles'].items()):
    m, n, k_local = shape
    torch.manual_seed(PLAN['input_seed']+role_index*10000+rank)
    inputs = (torch.randn(m, k_local, device='cuda')*.02).to(torch.bfloat16)
    weights = (torch.randn(n, k_local, device='cuda')*.02).to(torch.bfloat16)
    inputs_sha256, weights_sha256 = tensor_hash(inputs), tensor_hash(weights)
    partial = inputs@weights.t()
    op = GemmRSTaco(group, m, n, k_local, placement='fused')
    first = op.forward(inputs, weights).clone()
    correctness = check_taco_output(partial, first, group, mask=None)
    first_sha256 = tensor_hash(first)
    repeat = op.forward(inputs, weights).clone()
    if not torch.equal(first, repeat):
        raise RuntimeError('Repeated same-input operator output differs')
    del partial, repeat
    for _ in range(PLAN['timing']['warmup']):
        output = op.forward(inputs, weights).clone()
    dist.barrier(group=group)
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    started = time.perf_counter()
    for _ in range(PLAN['timing']['iterations']):
        output = op.forward(inputs, weights).clone()
    end.record()
    end.synchronize()
    elapsed = time.perf_counter()-started
    final_sha256 = tensor_hash(output)
    if not torch.isfinite(output).all() or first_sha256 != final_sha256:
        raise RuntimeError('Probe output changed across the timing window')
    record['results'][role] = dict(shape=shape, inputs_sha256=inputs_sha256,
        weights_sha256=weights_sha256, output_sha256=first_sha256,
        final_output_sha256=final_sha256, repeated_bitwise_equal=True,
        correctness=correctness, wall_us=elapsed*1e6/PLAN['timing']['iterations'],
        gpu_us=begin.elapsed_time(end)*1000/PLAN['timing']['iterations'])
    dist.barrier(group=group)
    torch.cuda.synchronize()
    del op, inputs, weights, first, output
record['completed'] = True
output_dir = Path(os.environ['H100_OP2_PROBE_RESULT'])
with (output_dir/f'rank{rank}.json').open('x') as handle:
    json.dump(record, handle, indent=2)
    handle.write('\n')
dist.destroy_process_group()
