#!/usr/bin/env python3
"""Attribute CUDA API calls and exposed stream gaps to projection scopes.

Trace-only diagnosis: instrumentation overhead and asynchronous overlap mean sums
are not predictions of unprofiled wall-time savings.
"""
import bisect
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BATCH = ROOT / 'logs/h100/phase6/ma2-cr2-2x-50steps-20261023-153237-090898'
OUT = Path(__file__).resolve().parent


def stats(xs):
    return {'n': len(xs), 'sum': sum(xs), 'mean': statistics.mean(xs),
            'median': statistics.median(xs), 'max': max(xs)} if xs else {}


def analyze(path):
    events = json.load(gzip.open(path, 'rt'))['traceEvents']
    calls = sorted((e for e in events if e.get('cat') == 'cpu_op'
                    and e['name'] == 'GemmRSFunction'), key=lambda e: e['ts'])
    starts = [e['ts'] for e in calls]
    api = collections.defaultdict(list)
    outside = collections.defaultdict(list)
    call_counts = [collections.Counter() for _ in calls]
    for e in events:
        if e.get('cat') != 'cuda_runtime':
            continue
        i = bisect.bisect_right(starts, e['ts']) - 1
        c = calls[i] if i >= 0 else None
        inside = c and e['tid'] == c['tid'] and e['ts'] + e['dur'] <= c['ts'] + c['dur'] + 0.01
        (api if inside else outside)[e['name']].append(e['dur'])
        if inside:
            call_counts[i][e['name']] += 1
    # Link kernels using their correlation ID to the exact host launch.
    launch = {e['args']['correlation']: e for e in events
              if e.get('cat') == 'cuda_runtime' and 'correlation' in e.get('args', {})}
    decoders = [e for e in events if e.get('cat') == 'kernel'
                and 'taco' in e['name'] and 'decode' in e['name']]
    bystream = collections.defaultdict(list)
    for e in events:
        if e.get('cat') in ('kernel', 'gpu_memset', 'gpu_memcpy'):
            bystream[e['args']['stream']].append(e)
    for es in bystream.values():
        es.sort(key=lambda e: e['ts'])
    positions = {id(e): (es, i) for es in bystream.values() for i, e in enumerate(es)}
    after = []
    for d in decoders:
        es, i = positions[id(d)]
        if i + 1 >= len(es):
            continue
        n = es[i + 1]
        l = launch.get(n['args'].get('correlation'))
        end = d['ts'] + d['dur']
        after.append({'gap_us': max(0, n['ts'] - end),
                      'host_late_us': max(0, l['ts'] + l['dur'] - end) if l else None,
                      'next_name': n['name']})
    def select_api(name):
        return stats([c[name] for c in call_counts])
    return {'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'projection_calls': len(calls),
            'api_inside_projection': {n: stats(v) for n, v in sorted(api.items())},
            'api_outside_projection': {n: stats(v) for n, v in sorted(outside.items())},
            'per_projection_api_counts': {n: select_api(n) for n in sorted(api)},
            'postdecode_gap_us': stats([a['gap_us'] for a in after]),
            'postdecode_gap_over_5us': sum(a['gap_us'] > 5 for a in after),
            'postdecode_host_late_us': stats([a['host_late_us'] for a in after if a['host_late_us'] is not None]),
            'next_kernels': dict(collections.Counter(a['next_name'] for a in after))}


def main():
    result = {'warning': 'Separate instrumented profiles; summed CPU API and GPU wait durations are not potential speedup.',
              'ranks': {}}
    for arm in ('CR2', 'CR', 'C', 'R'):
        for rank in range(8):
            result['ranks'][f'{arm}/{rank}'] = analyze(BATCH / f'fullstep/results/profile-{arm}/rank{rank}-profile.json.gz')
            print(arm, rank, flush=True)
    (OUT / 'profile-runtime.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
