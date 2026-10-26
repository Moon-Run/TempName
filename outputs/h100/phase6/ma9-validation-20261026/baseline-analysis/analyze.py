#!/usr/bin/env python3
"""Read-only, correlation-linked accounting of CR3's gradient zero statistics.

This is profiler evidence for actual work and scope, not an unprofiled speedup
estimate. MA8's instrumented F4 is intentionally excluded from the budget.
"""
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


REPO = Path(__file__).resolve().parents[5]
ORIGINAL = REPO / 'logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(path):
    events = json.load(gzip.open(path, 'rt'))['traceEvents']
    cpu = [e for e in events if e.get('cat') == 'cpu_op']
    nested = sorted((e for e in cpu if e['name'] == 'aten::count_nonzero'),
                    key=lambda e: (e['ts'], -e['dur']))
    assert nested, path
    # ATen records both overloads; only outer calls represent a parameter.
    outer = []
    for event in nested:
        if not outer or event['ts'] >= outer[-1]['ts'] + outer[-1]['dur']:
            outer.append(event)
    assert len(nested) == 2 * len(outer), (path, len(nested), len(outer))
    tid = outer[0]['tid']
    assert all(e['tid'] == tid for e in outer)
    start = outer[0]['ts']
    last_count_end = max(e['ts'] + e['dur'] for e in outer)
    collective = min((e for e in cpu if e['tid'] == tid and
                      e['name'] == 'c10d::allreduce_' and e['ts'] > last_count_end),
                     key=lambda e: e['ts'])
    scalar = min((e for e in cpu if e['tid'] == tid and
                  e['name'] == 'aten::item' and e['ts'] > collective['ts'] + collective['dur']),
                 key=lambda e: e['ts'])
    end = scalar['ts'] + scalar['dur']
    assert collective['ts'] - last_count_end < 1000
    region = [e for e in cpu if e['tid'] == tid and start <= e['ts'] < end]
    launches = [e for e in events if e.get('cat') == 'cuda_runtime' and
                e['tid'] == tid and start <= e['ts'] < end]
    correlations = {e.get('args', {}).get('correlation') for e in launches}
    correlations.discard(None)
    gpu = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') and
           e.get('args', {}).get('correlation') in correlations]
    kernels = [e for e in gpu if e.get('cat') == 'kernel']
    collectives = [e for e in kernels if 'nccl' in e['name'].lower()]
    compute = [e for e in kernels if e not in collectives]
    assert len(collectives) == 1 and 'AllReduce' in collectives[0]['name'], path
    assert kernels and compute
    families = defaultdict(lambda: {'calls': 0, 'gpu_work_ms': 0.0})
    for event in kernels:
        name = event['name']
        if 'nccl' in name.lower():
            kind = 'allreduce'
        elif 'CompareEqFunctor' in name or 'CompareNeFunctor' in name:
            kind = 'nonzero_predicate'
        elif 'direct_copy_kernel_cuda' in name:
            kind = 'predicate_to_int64'
        elif 'ReduceOp<long' in name:
            kind = 'int64_sum'
        elif 'CUDAFunctorOnOther_add<long>' in name:
            kind = 'numel_minus_nonzero'
        elif 'CUDAFunctor_add<float>' in name:
            kind = 'ordered_fp32_accumulation'
        else:
            kind = name
        families[kind]['calls'] += 1
        families[kind]['gpu_work_ms'] += event['dur'] / 1000
    all_kernels = [e for e in events if e.get('cat') == 'kernel']
    return {
        'trace': str(path.relative_to(REPO)), 'sha256': digest(path),
        'parameter_count': len(outer),
        'cpu_count_overload_events': len(nested),
        'cpu_region_ms': (end - start) / 1000,
        'cpu_count_outer_duration_sum_ms': sum(e['dur'] for e in outer) / 1000,
        'cpu_scalar_wait_ms': scalar['dur'] / 1000,
        'cpu_operations': dict(Counter(e['name'] for e in region)),
        'cuda_runtime_calls': len(launches),
        'gpu_kernel_calls': len(kernels),
        'gpu_compute_calls': len(compute),
        'gpu_compute_work_ms': sum(e['dur'] for e in compute) / 1000,
        'gpu_allreduce_ms': sum(e['dur'] for e in collectives) / 1000,
        'gpu_region_span_ms': (max(e['ts'] + e['dur'] for e in gpu) - min(e['ts'] for e in gpu)) / 1000,
        'gpu_families': dict(families),
        'profile_step_gpu_span_ms': (max(e['ts'] + e['dur'] for e in all_kernels) - min(e['ts'] for e in all_kernels)) / 1000,
        'selection': 'CPU count_nonzero block through following allreduce/item; GPU linked by CUDA runtime correlation on the same CPU thread.',
    }


def main():
    rows = [analyze(ORIGINAL / f'results/profile-F4/rank{rank}-profile.json.gz') for rank in range(8)]
    optimizer = ORIGINAL / 'megatron/megatron/core/optimizer/optimizer.py'
    clipping = ORIGINAL / 'megatron/megatron/core/optimizer/clip_grads.py'
    log = ORIGINAL / 'results/profile-F4/process.log'
    result = {
        'protocol': 'ma9-original-cr3-zero-statistics-readonly-v1',
        'baseline': 'Original MA4-F CR3, before MA8 column wrappers',
        'source_files': {str(p.relative_to(REPO)): digest(p) for p in (optimizer, clipping, log)},
        'log_num_zeros_in_grad': False,
        'zero_statistics_still_execute': True,
        'ranks': rows,
        'median': {key: median(r[key] for r in rows) for key in (
            'cpu_region_ms', 'gpu_compute_work_ms', 'gpu_allreduce_ms',
            'gpu_region_span_ms', 'profile_step_gpu_span_ms')},
        'limitations': [
            'Profiles materially perturb execution; these durations do not predict unprofiled step savings.',
            'CPU region includes submission and synchronization; kernel work, GPU span and CPU span overlap and must not be added.',
            'NCCL duration includes rank-arrival waiting and is not a standalone removable communication budget.',
            'The new candidate must retain exact per-parameter counts, original ordered FP32 accumulation, original allreduce and scalar result.',
            'MA8 instrumented F4 timing is excluded: 64 column modules use extra Function dispatch, record_function and state checks even while timing.',
        ],
    }
    assert 'log_num_zeros_in_grad ........................... False' in log.read_text()
    out = Path(__file__).with_name('analysis.json')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'ranks': [{k: r[k] for k in (
        'parameter_count', 'cpu_region_ms', 'gpu_compute_calls',
        'gpu_compute_work_ms', 'gpu_allreduce_ms', 'gpu_region_span_ms')}
        for r in rows], 'median': result['median']}, indent=2))


if __name__ == '__main__':
    main()
