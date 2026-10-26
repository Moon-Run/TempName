#!/usr/bin/env python3
"""Attribute original CR3 DDP gradient writes through CPU/CUDA correlations.

Read-only with respect to the historical experiment. Only analysis.json beside
this script is written. No count_zeros scalar add is included in this scope.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median


REPO = Path(__file__).resolve().parents[5]
BASE = REPO / 'logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep'
ENGINE = 'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad'
NODE = 'torch::autograd::AccumulateGrad'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def end(event):
    return event['ts'] + event.get('dur', 0)


def contains(parent, child):
    return (parent['tid'] == child['tid'] and
            parent['ts'] <= child['ts'] + .01 and end(child) <= end(parent) + .01)


class ThreadIndex:
    def __init__(self, events):
        self.rows = defaultdict(list)
        for event in events:
            self.rows[event['tid']].append(event)
        for rows in self.rows.values():
            rows.sort(key=lambda e: e['ts'])
        self.starts = {tid: [e['ts'] for e in rows] for tid, rows in self.rows.items()}

    def inside(self, scope):
        rows = self.rows[scope['tid']]
        starts = self.starts[scope['tid']]
        lo = bisect_left(starts, scope['ts'] - .01)
        hi = bisect_right(starts, end(scope) + .01)
        return [e for e in rows[lo:hi] if contains(scope, e)]


def analyze(path, rank):
    events = json.load(gzip.open(path, 'rt'))['traceEvents']
    cpu = [e for e in events if e.get('cat') == 'cpu_op']
    runtime = [e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')]
    cpu_index, runtime_index = ThreadIndex(cpu), ThreadIndex(runtime)
    device = defaultdict(list)
    for event in events:
        if event.get('cat') in ('kernel', 'gpu_memset', 'gpu_memcpy'):
            device[event.get('args', {}).get('correlation')].append(event)
    scopes = sorted((e for e in cpu if e['name'] == ENGINE), key=lambda e: e['ts'])
    assert len(scopes) == 389 * 4, (rank, len(scopes))
    nodes = [e for e in cpu if e['name'] == NODE]
    assert len(nodes) == len(scopes)
    adds, launches, kernels = [], [], []
    seen = set()
    for scope in scopes:
        children = cpu_index.inside(scope)
        own_nodes = [e for e in children if e['name'] == NODE]
        own_adds = [e for e in children if e['name'] == 'aten::add_']
        assert len(own_nodes) == len(own_adds) == 1, (rank, scope, own_nodes, own_adds)
        node, add = own_nodes[0], own_adds[0]
        # The original engine scope includes the DDP post-hook; the leaf-node
        # scope itself ends before the DDP add. Confusing those two loses work.
        assert end(node) <= add['ts'] + .01
        assert not any(e['name'] == 'aten::count_nonzero' for e in children)
        own_launches = [e for e in runtime_index.inside(add)
                        if e.get('args', {}).get('correlation') is not None]
        own_work = [e for launch in own_launches
                    for e in device[launch['args']['correlation']]]
        assert len(own_work) == 1 and own_work[0]['cat'] == 'kernel', (rank, own_work)
        kernel = own_work[0]
        assert 'CUDAFunctor_add<float>' in kernel['name'], kernel['name']
        correlation = kernel['args']['correlation']
        assert correlation not in seen
        seen.add(correlation)
        adds.append(add)
        launches.extend(own_launches)
        kernels.append(kernel)
    assert len(kernels) == 1556
    assert len({e['args']['stream'] for e in kernels}) == 1
    ordered = sorted(kernels, key=lambda e: e['ts'])
    assert all(end(a) <= b['ts'] + .01 for a, b in zip(ordered, ordered[1:]))
    all_kernels = [e for e in events if e.get('cat') == 'kernel']
    count_nodes = [e for e in cpu if e['name'] == 'aten::count_nonzero']
    durations = sorted(e['dur'] for e in kernels)
    rank_path = path.with_name(f'rank{rank}.json')
    parameters = json.loads(rank_path.read_text())['ma4_f_trainable_parameters']
    assert len(parameters) == 389 and all(p['dtype'] == 'torch.bfloat16' for p in parameters.values())
    elements = sum(math.prod(p['shape']) for p in parameters.values())
    return dict(
        rank=rank, trace=str(path.relative_to(REPO)), sha256=digest(path),
        rank_record=str(rank_path.relative_to(REPO)), rank_record_sha256=digest(rank_path),
        microbatches=4, trainable_parameters=389,
        gradient_elements_per_microbatch=elements,
        largest_source_tensor_mib=max(math.prod(p['shape']) for p in parameters.values()) * 2 / 2**20,
        logical_gradient_add_traffic_gib_per_step=elements * (2 + 4 + 4) * 4 / 2**30,
        traffic_definition='4 microbatches * elements * (BF16 source read 2B + FP32 main_grad read 4B + write 4B); logical traffic, not measured HBM traffic.',
        accumulate_grad_engine_calls=len(scopes), leaf_accumulate_grad_calls=len(nodes),
        ddp_add_calls=len(adds), correlated_gradient_add_kernels=len(kernels),
        cuda_runtime_calls_inside_adds=len(launches),
        kernel_names=dict(Counter(e['name'] for e in kernels)),
        compute_stream=kernels[0]['args']['stream'],
        cpu_engine_scope_work_ms=sum(e['dur'] for e in scopes) / 1000,
        cpu_leaf_node_work_ms=sum(e['dur'] for e in nodes) / 1000,
        cpu_add_work_ms=sum(e['dur'] for e in adds) / 1000,
        cpu_runtime_work_ms=sum(e['dur'] for e in launches) / 1000,
        gpu_gradient_add_work_ms=sum(e['dur'] for e in kernels) / 1000,
        gpu_gradient_add_duration_us=dict(min=durations[0], median=median(durations),
            p90=durations[int(.9 * (len(durations) - 1))], max=durations[-1]),
        profile_step_gpu_span_ms=(max(map(end, all_kernels)) - min(e['ts'] for e in all_kernels)) / 1000,
        excluded_count_nonzero_overload_events=len(count_nodes),
        selection=('Exactly one aten::add_ follows each leaf AccumulateGrad inside its '
            'engine-evaluation scope. CUDA runtime launch correlation identifies exactly '
            'one FP32 add kernel per hook. The count_zeros scalar adds are outside these scopes.'),
    )


def main():
    rows = [analyze(BASE / f'results/profile-F4/rank{rank}-profile.json.gz', rank) for rank in range(8)]
    ddp = BASE / 'megatron/megatron/core/distributed/distributed_data_parallel.py'
    schedules = BASE / 'megatron/megatron/core/pipeline_parallel/schedules.py'
    worker = BASE / 'scripts/worker.py'
    fields = ('cpu_engine_scope_work_ms', 'cpu_leaf_node_work_ms', 'cpu_add_work_ms',
              'cpu_runtime_work_ms', 'gpu_gradient_add_work_ms', 'profile_step_gpu_span_ms')
    result = dict(
        protocol='ma10-original-cr3-gradient-commit-readonly-v1',
        baseline='Original MA4-F CR3, before MA6/MA7/MA8/MA9 changes',
        source_files={str(p.relative_to(REPO)): digest(p) for p in (ddp, schedules, worker)},
        historical_unprofiled_reference_ms=501.433,
        two_percent_budget_ms=501.433 * .02,
        required_net_saving_us_per_original_add=501.433 * .02 * 1000 / 1556,
        ranks=rows, median={key: median(row[key] for row in rows) for key in fields},
        limitations=[
            'Profiled CPU and GPU durations are diagnostic evidence, not an unprofiled speedup estimate.',
            'CPU engine, leaf, add and CUDA runtime scopes are nested; their times must not be added.',
            'GPU add work remains necessary under batching: this mechanism removes launches/dispatch and may improve small-tensor execution, not all gradient memory traffic.',
            'Logical byte counts from frozen parameter shapes do not measure HBM traffic or cache hits and cannot be divided by peak bandwidth to predict a speedup.',
            'Hooks are interleaved with backward GEMMs and collectives; the sum of hook/kernel times is not a removable critical-path interval.',
            'All timing differences are within one rank; no absolute timestamps are subtracted across ranks.',
            'The approximate 10 ms target must be established by fresh-process paired full optimizer-step measurements against unchanged F4.',
            'Queue creation, pointer packing, native submission, flush, source lifetime and memory-pressure costs are included in candidate timing and may erase any gain.',
        ],
    )
    out = Path(__file__).with_name('analysis.json')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(ranks=[{key: row[key] for key in ('rank', 'ddp_add_calls',
        'cpu_engine_scope_work_ms', 'cpu_add_work_ms', 'gpu_gradient_add_work_ms')} for row in rows],
        median=result['median'], target_ms=result['two_percent_budget_ms']), indent=2))


if __name__ == '__main__':
    main()
