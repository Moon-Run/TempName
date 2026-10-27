#!/usr/bin/env python3
"""Audit original MA89 projection handoffs without changing any input evidence."""
from bisect import bisect_right
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[4]
TOOLS = REPO / 'tools/h100/phase6'
sys.path.insert(0, str(TOOLS))
from ma9_profile import Index, contains, end, require

BATCH = 'ma89-v1-cache-zcount-1x-50steps-20261027-104719-859382'
ROOT = REPO / 'logs/h100/phase6' / BATCH / 'fullstep'
BASELINE_MS = 459.8060027899919


def analyze(path, rank):
    with gzip.open(path) as source:
        events = json.load(source)['traceEvents']
    index = Index(events)
    scopes = defaultdict(list)
    for event in events:
        if event.get('cat') == 'user_annotation' and event['name'] in (
                'scope.attn.candidate', 'scope.mlp.candidate'):
            scopes[event['tid']].append(event)
    for values in scopes.values():
        values.sort(key=lambda e: e['ts'])
    starts = {tid: [event['ts'] for event in values] for tid, values in scopes.items()}
    calls = [event for event in index.cpu if event['name'] == 'ConsumerGemmRSFunction']
    require(len(calls) == 256, 'Expected all 256 row projections')
    used, used_scopes, records, names = set(), set(), [], defaultdict(set)
    for call in calls:
        tid = call['tid']
        pos = bisect_right(starts.get(tid, []), call['ts']) - 1
        require(pos >= 0, 'Missing parent projection annotation')
        scope = scopes[tid][pos]
        require(contains(scope, call), 'Projection outside parent annotation')
        require((tid, pos) not in used_scopes, 'Projection annotation reused')
        used_scopes.add((tid, pos))
        work, launches = index.work(call)
        path_events = sorted((event for event in work if event['cat'] == 'kernel'),
                             key=lambda event: event['ts'])
        require(len(path_events) == 3, 'Expected producer/publication/decoder kernel triple')
        producer, barrier, decoder = path_events
        require('flux_bf16' in producer['name'] and 'reducescatter' in producer['name'],
                'Wrong producer kernel')
        require('CudaIpcBarrierAllKernel' in barrier['name'], 'Wrong publication kernel')
        require('taco_ma4f_decode_consumer_kernel' in decoder['name'], 'Wrong decoder kernel')
        require(len({event['args']['stream'] for event in path_events}) == 1,
                'Projection stream changed')
        require(all(end(a) <= b['ts'] + .01 for a, b in zip(path_events, path_events[1:])),
                'Projection order changed')
        require(not used.intersection(map(id, path_events)), 'Kernel has duplicate ownership')
        used.update(map(id, path_events))
        role = scope['name'].split('.')[1]
        for label, event in zip(('producer', 'barrier', 'decoder'), path_events):
            names[label].add(event['name'])
            require(event['args']['correlation'] in launches, 'No host launch correlation')
        records.append(dict(role=role, cpu_us=call['dur'],
            producer_us=producer['dur'], barrier_us=barrier['dur'], decoder_us=decoder['dur'],
            producer_to_barrier_gap_us=max(0, barrier['ts'] - end(producer)),
            barrier_to_decoder_gap_us=max(0, decoder['ts'] - end(barrier)),
            producer_end_to_decoder_end_us=end(decoder) - end(producer),
            chain_us=end(decoder) - producer['ts']))
    require(len(used_scopes) == sum(map(len, scopes.values())), 'Unowned annotation')
    for token in ('CudaIpcBarrierAllKernel', 'taco_ma4f_decode_consumer_kernel'):
        require(all(id(event) in used for event in events if event.get('cat') == 'kernel'
                    and token in event['name']), 'Unowned handoff/decoder kernel')
    fields = [key for key in records[0] if key != 'role']
    groups = {}
    for role in ('attn', 'mlp', 'all'):
        rows = [row for row in records if role == 'all' or row['role'] == role]
        require(len(rows) == (256 if role == 'all' else 128), 'Wrong role coverage')
        groups[role] = dict(calls=len(rows),
            sum_ms={key.removesuffix('_us'): sum(row[key] for row in rows) / 1000 for key in fields},
            median_us={key: statistics.median(row[key] for row in rows) for key in fields})
    kernels = [event for event in events if event.get('cat') == 'kernel']
    return dict(rank=rank, path=str(path.relative_to(REPO)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        launch_correlated_projection_kernels=len(used),
        total_profile_gpu_span_ms=(max(map(end, kernels)) - min(event['ts'] for event in kernels)) / 1000,
        kernel_names={key: sorted(values) for key, values in names.items()}, groups=groups)


def main():
    records = []
    for rank in range(8):
        path = ROOT / 'results/profile-F4-cache-zcount' / f'rank{rank}-profile.json.gz'
        records.append(analyze(path, rank))
        print(f'rank {rank}: 256 projection paths / 768 uniquely owned launches', flush=True)
    fields = records[0]['groups']['all']['sum_ms']
    report = dict(protocol='h100-ma12-baseline-publication-audit-v1',
        baseline_batch=BATCH, baseline_arm='F4-cache-zcount', source_traces=8,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        launch_index_sha256=hashlib.sha256((TOOLS / 'ma9_profile.py').read_bytes()).hexdigest(),
        historical_step_ms=BASELINE_MS, target_percent=2.0,
        required_historical_scale_saving_ms=BASELINE_MS * .02,
        required_saving_us_per_projection=BASELINE_MS * .02 * 1000 / 256,
        records=records,
        median_rank_sum_ms={key: statistics.median(
            record['groups']['all']['sum_ms'][key] for record in records) for key in fields},
        median_rank_profile_gpu_span_ms=statistics.median(
            record['total_profile_gpu_span_ms'] for record in records),
        candidate_gpu_tested=False, performance_target_validated=False,
        limitation='Each kernel is attributed through a runtime launch within its original '
            'ConsumerGemmRSFunction, with an enclosing attention/MLP scope. Profile durations '
            'include rank skew and instrumentation. Barrier work is not all removable critical-path '
            'time. CPU and GPU quantities are not additive. No cross-rank timestamps are subtracted. '
            'A new unprofiled paired full-step experiment is required to judge MA12.')
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['median_rank_sum_ms'], indent=2), flush=True)


if __name__ == '__main__':
    main()
