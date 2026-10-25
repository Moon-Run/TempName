"""Diagnose queued projection work and its serial consumers in returned traces.

This is a same-stream dependency diagnosis, not reconstruction of the global
eight-GPU longest path. CUPTI/PyTorch profiling alters timings; summed kernel,
queue, idle and barrier durations are not predicted unprofiled speedups.
"""
import bisect
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[4]
BATCH = REPO / 'logs/h100/phase6/ma4-g-pair-1x-50steps-20261025-095400-360259'
OUT = Path(__file__).resolve().parent


def stats(values):
    values = sorted(values)
    if not values:
        return {}
    return dict(n=len(values), minimum=values[0], median=statistics.median(values),
                mean=statistics.mean(values), maximum=values[-1], sum=sum(values),
                p10=values[int((len(values)-1)*.1)], p90=values[int((len(values)-1)*.9)])


def end(event):
    return event['ts'] + event.get('dur', 0)


def analyze(path):
    events = json.load(gzip.open(path, 'rt'))['traceEvents']
    cpus = [e for e in events if e.get('cat') == 'cpu_op']
    cpu_by_external = {e['args']['External id']: e for e in cpus if 'External id' in e.get('args', {})}
    calls = sorted((e for e in cpus if e['name'] == 'GemmRSFunction'), key=lambda e: e['ts'])
    runtimes = {e['args']['correlation']: e for e in events
                if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'correlation' in e.get('args', {})}
    scopes = collections.defaultdict(list)
    for event in events:
        if event.get('cat') == 'user_annotation' and event['name'] in ('scope.attn.candidate', 'scope.mlp.candidate'):
            scopes[event['tid']].append(event)
    for values in scopes.values():
        values.sort(key=lambda e: e['ts'])
    scope_starts = {tid: [e['ts'] for e in values] for tid, values in scopes.items()}
    device = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    bystream, byexternal = collections.defaultdict(list), collections.defaultdict(list)
    for event in device:
        bystream[event['args']['stream']].append(event)
        byexternal[event['args'].get('External id')].append(event)
    for values in bystream.values():
        values.sort(key=lambda e: e['ts'])
    positions = {id(e): (stream, i) for stream, values in bystream.items() for i, e in enumerate(values)}
    rows = []
    next_patterns = collections.Counter()

    def cpu_name(event):
        return cpu_by_external.get(event['args'].get('External id'), {}).get('name', '?')

    for index, call in enumerate(calls):
        scope_index = bisect.bisect_right(scope_starts[call['tid']], call['ts'])-1
        scope = scopes[call['tid']][scope_index]
        assert scope['ts'] <= call['ts'] and end(call) <= end(scope)+.01
        role = scope['name'].split('.')[1]
        nodes = byexternal[call['args']['External id']]
        gemms = [e for e in nodes if e['cat']=='kernel' and 'flux_bf16' in e['name'] and 'reducescatter' in e['name']]
        decoders = [e for e in nodes if e['cat']=='kernel' and 'taco_ma2c_decode_kernel' in e['name']]
        barriers = [e for e in nodes if e['cat']=='kernel' and 'CudaIpcBarrierAllKernel' in e['name']]
        assert len(gemms) == len(decoders) == len(barriers) == 1
        gemm, decoder, barrier = gemms[0], decoders[0], barriers[0]
        launch = runtimes[gemm['args']['correlation']]
        stream, position = positions[id(decoder)]
        following = bystream[stream][position+1:position+5]
        assert len(following) == 4
        bias, residual, norm, after_norm = following
        names = [cpu_name(e) for e in following]
        next_patterns[tuple(names)] += 1
        assert names[:3] == ['aten::add', 'aten::add', 'FusedLayerNormAffineFunction'], names
        assert all(end(a) <= b['ts']+.01 for a, b in zip([decoder, *following], following))
        bias_launch = runtimes[bias['args']['correlation']]
        residual_launch = runtimes[residual['args']['correlation']]
        norm_launch = runtimes[norm['args']['correlation']]
        bias_lateness = end(bias_launch)-end(decoder)
        row = dict(index=index, role=role, cpu_projection_us=call['dur'],
                   gemm_host_queue_us=gemm['ts']-end(launch),
                   gemm_host_prequeued=gemm['ts']-end(launch)>5,
                   decoder_us=decoder['dur'], bias_us=bias['dur'], residual_us=residual['dur'],
                   two_adds_device_us=bias['dur']+residual['dur'], norm_us=norm['dur'],
                   publication_barrier_us=barrier['dur'],
                   decode_to_bias_gap_us=max(0, bias['ts']-end(decoder)),
                   bias_to_residual_gap_us=max(0, residual['ts']-end(bias)),
                   residual_to_norm_gap_us=max(0, norm['ts']-end(residual)),
                   decode_end_to_residual_end_us=end(residual)-end(decoder),
                   decode_end_to_norm_end_us=end(norm)-end(decoder),
                   decode_start_to_norm_end_us=end(norm)-decoder['ts'],
                   after_norm_gap_us=max(0, after_norm['ts']-end(norm)),
                   bias_launch_lateness_us=bias_lateness,
                   bias_launch_prequeued=bias_lateness<0,
                   bias_launch_late_positive_us=max(0, bias_lateness),
                   residual_launch_lateness_us=end(residual_launch)-end(bias),
                   norm_launch_lateness_us=end(norm_launch)-end(residual),
                   next_mainstream_cpu=names[3])
        rows.append(row)
    assert len(rows) == 256 and collections.Counter(r['role'] for r in rows) == {'attn':128, 'mlp':128}
    groups = {}
    for role in ('attn', 'mlp'):
        selected = [r for r in rows if r['role']==role]
        groups[role] = {field: stats([r[field] for r in selected]) for field, value in selected[0].items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool) and field!='index'}
        groups[role]['fractions'] = {field: statistics.mean(r[field] for r in selected)
                                     for field in ('gemm_host_prequeued', 'bias_launch_prequeued')}
    cpu_gpu_totals = collections.defaultdict(list)
    for event in device:
        name = event['name']
        category = ('projection_gemm' if 'flux_bf16' in name and 'reducescatter' in name else
                    'projection_decoder' if 'taco_ma2c_decode_kernel' in name else
                    'publication_barrier' if 'CudaIpcBarrierAllKernel' in name else
                    'all_gather' if 'nccl' in name and 'AllGather' in name else
                    'reduce_scatter' if 'nccl' in name and 'ReduceScatter' in name else
                    'all_reduce' if 'nccl' in name and 'AllReduce' in name else
                    'optimizer_adam' if 'AdamFunctor' in name else cpu_name(event))
        cpu_gpu_totals[category].append(event['dur'])
    return dict(trace=str(path.relative_to(REPO)), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                gpu_span_us=max(end(e) for e in device)-min(e['ts'] for e in device),
                projection=groups, rows=rows,
                following_cpu_patterns={' -> '.join(key): value for key, value in next_patterns.items()},
                gpu_work_totals_us={key: stats(values) for key, values in cpu_gpu_totals.items()})


def main():
    result = dict(batch=str(BATCH.relative_to(REPO)), gpu_used=False,
        interpretation='Same-stream serial dependency/host queue evidence only; not the global eight-rank longest path. '
                       'All profiles contain instrumentation overhead. Summed kernels, idle gaps, queues and waits cannot be added '
                       'to predict end-to-end speedup. Two add kernel time is a diagnostic removable-work budget, not a guaranteed saving.',
        ranks={}, summary={})
    for arm in ('G', 'CR2'):
        for rank in range(8):
            result['ranks'][f'{arm}/rank{rank}'] = analyze(BATCH/f'fullstep/results/profile-{arm}/rank{rank}-profile.json.gz')
            print('Analyzed', arm, rank, flush=True)
    for arm in ('G', 'CR2'):
        ranks = [result['ranks'][f'{arm}/rank{rank}'] for rank in range(8)]
        result['summary'][arm] = dict(
            gpu_span_ms_rank_median=statistics.median(r['gpu_span_us']/1000 for r in ranks), projection={},
            two_add_device_sum_ms_rank_median=statistics.median(sum(row['two_adds_device_us'] for row in r['rows'])/1000 for r in ranks),
            decode_to_norm_end_sum_ms_rank_median=statistics.median(sum(row['decode_end_to_norm_end_us'] for row in r['rows'])/1000 for r in ranks),
            bias_launch_late_positive_sum_ms_rank_median=statistics.median(sum(row['bias_launch_late_positive_us'] for row in r['rows'])/1000 for r in ranks))
        for role in ('attn', 'mlp'):
            sample = ranks[0]['projection'][role]
            result['summary'][arm]['projection'][role] = {
                key: statistics.median(r['projection'][role][key]['median'] for r in ranks)
                for key in sample if key != 'fractions'}
            result['summary'][arm]['projection'][role]['fraction_rank_median'] = {
                key: statistics.median(r['projection'][role]['fractions'][key] for r in ranks)
                for key in sample['fractions']}
    (OUT/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
