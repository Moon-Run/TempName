"""CPU-only protocol for an explicitly experimental chunked GEMM/RS runtime."""
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

PROTOCOL = 'h100-phase7-chunk-prototype-v1'
VARIANTS = {
    'E0': dict(mode='barrier', ordering='identity'),
    'E1': dict(mode='barrier', ordering='shared'),
    'E2': dict(mode='pipeline', ordering='identity'),
    'E3': dict(mode='pipeline', ordering='shared'),
}
MODEL = dict(layers=32, hidden=4096, ffn=16384, heads=32, sequence=2048,
             micro_batch=1, global_batch=4, seed=1234, token_seed=4321,
             vocab=32768, tp=8)
MEGATRON_COMMIT = '3ea68ad6042cc1204386ae9364358f7c4de1bc37'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def schedule(blocks=4, seed=20261017):
    require(type(blocks) is int and blocks > 0 and blocks % 4 == 0,
            'blocks must be a positive multiple of four')
    order = list(VARIANTS)
    random.Random(seed).shuffle(order)
    rows = [order[b % 4:] + order[:b % 4] for b in range(blocks)]
    return [dict(round=r, block=b, variant=v,
                 directory=f'round{r}/block-{b:03d}-{v}')
            for r, rr in enumerate((rows, [x[::-1] for x in rows[::-1]]), 1)
            for b, row in enumerate(rr) for v in row]


def verify(root):
    from planner import validate_plan, default_shape
    root = Path(root).resolve()
    m = read(root / 'manifest.json')
    require(m['protocol'] == PROTOCOL and m['variants'] == VARIANTS, 'Unknown protocol/variants')
    require(m['kind'] in ('operator', 'model'), 'Unknown measurement kind')
    require(m['scope'] in ('attention', 'mlp', 'both'), 'Unknown scope')
    require(m['kind'] != 'operator' or m['scope'] != 'both', 'Operator campaign requires one role')
    require(m['schedule'] == schedule(m['blocks'], m['schedule_seed']), 'Schedule changed')
    require(m['model'] == MODEL and m['warmup'] > 0 and m['steps'] > 0, 'Model/timing changed')
    roles = ('attention', 'mlp') if m['scope'] == 'both' else (m['scope'],)
    required = {'scripts/' + name for name in ('worker.py', 'runtime.py', 'codec.py', 'adapter.py',
                                              'planner.py', 'protocol.py', 'phase7.py', 'reference.py')}
    required |= {f'plans/{role}-{order}.json' for role in roles for order in ('identity', 'shared')}
    required |= {f'provenance/{role}-phase6-plan.json' for role in roles}
    require(required <= set(m['files']), 'Manifest omits a required frozen input')
    for name, expected in m['files'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.is_file() and sha(path) == expected,
                'Frozen file missing or changed: ' + name)
    for role in roles:
        plans = {}
        for order in ('identity', 'shared'):
            p = read(root / 'plans' / f'{role}-{order}.json')
            validate_plan(p)
            require(p['shape'] == default_shape(role) and p['ordering'] == order, 'Wrong role/order/shape')
            require(p['chunk_tiles'] == m['chunk_tiles'], 'Chunk size changed')
            plans[order] = p
        require(plans['identity']['quantize_mask'] == plans['shared']['quantize_mask'],
                'The factorial experiment must use identical physical masks')
    return m


def validate_window(root, folder, expected):
    """Reject incomplete/tampered windows; a TP rank is never an independent repeat."""
    root, folder = Path(root), Path(folder)
    m = verify(root)
    require(folder.resolve() == (root / 'results' / expected['directory']).resolve(), 'Window directory mismatch')
    records = [read(folder / f'rank{rank}.json') for rank in range(8)]
    require(len({r['device']['uuid'] for r in records}) == 8, 'Ranks must use eight distinct GPUs')
    roles = ('attention', 'mlp') if m['scope'] == 'both' else (m['scope'],)
    order = VARIANTS[expected['variant']]['ordering']
    plans = {role: digest(read(root / 'plans' / f'{role}-{order}.json')) for role in roles}
    for rank, r in enumerate(records):
        require(r['protocol'] == PROTOCOL and r['rank'] == rank and r['world_size'] == 8,
                'Wrong rank/protocol')
        require(r['run_id'] == m['run_id'] and r['manifest_sha256'] == sha(root / 'manifest.json'),
                'Run identity changed')
        require(all(r[k] == expected[k] for k in ('variant', 'stage', 'round', 'block', 'directory')), 'Wrong job')
        require(r['variant_definition'] == VARIANTS[expected['variant']] and r['plans_sha256'] == plans,
                'Wrong plan/runtime mode')
        require(r['kind'] == m['kind'] and r['scope'] == m['scope'], 'Wrong measurement scope')
        require(r['model_config'] == m['model'] and r['implementation'] == m['implementation'],
                'Implementation/model identity changed')
        require(r['completed'] is True and r['passed'] is True, 'Failed/incomplete rank')
        require(r['device']['capability'] == [9, 0] and 'H100' in r['device']['name'] and r['device']['sm_count'] == 132,
                'Not the expected 132-SM H100 result')
        steps = 1 if expected['stage'] == 'profile' else m['steps']
        require(r['skips'] == 0 and r['steps'] == steps, 'Skipped or missing steps')
        require(r['execution'] == records[0]['execution'], 'Ranks used different runtime environments')
        require(math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] > 0, 'Invalid elapsed time')
        if m['kind'] == 'model':
            require(len(r['losses']) == len(r['grad_norms']) == steps, 'Missing training trajectory')
            require(all(math.isfinite(v) for v in r['losses'] + r['grad_norms']), 'Nonfinite trajectory')
            require(r['losses'] == records[0]['losses'], 'TP loss records disagree')
            calls = r.get('module_calls', {})
            suffixes = {'attention': '.self_attention.linear_proj', 'mlp': '.mlp.linear_fc2'}
            require(len(calls) == len(roles) * 32 and all(type(v) is int and v == steps * 4 for v in calls.values()),
                    'Incomplete model projection call counts')
            require(all(sum(name.endswith(suffixes[role]) for name in calls) == 32 for role in roles),
                    'Wrong model projection roles')
            for role in roles:
                suffix = suffixes[role]
                indices = {name[:-len(suffix)].rsplit('.', 1)[-1] for name in calls if name.endswith(suffix)}
                require(indices == {str(i) for i in range(32)}, 'Missing/duplicate transformer layer routes')
    return dict(ms=1000 * max(r['elapsed_seconds'] for r in records) / steps,
                fingerprints=[r['fingerprints'] for r in records],
                devices=[r['device'] for r in records],
                execution=records[0]['execution'],
                files={str((folder / f'rank{i}.json').relative_to(root)): sha(folder / f'rank{i}.json')
                       for i in range(8)})


def summarize(values, seed=20261017, resamples=10000):
    """Paired log ratios with whole-block bootstrap, including the interaction."""
    require(set(values) == set(VARIANTS), 'Missing variants')
    n = len(values['E0'])
    require(n > 0 and all(len(v) == n for v in values.values()), 'Incomplete paired blocks')
    require(all(math.isfinite(x) and x > 0 for v in values.values() for x in v), 'Invalid times')
    comparisons = {'arrival_barrier': ('E1', 'E0'), 'arrival_pipeline': ('E3', 'E2'),
                   'pipeline_identity': ('E2', 'E0'), 'pipeline_shared': ('E3', 'E1')}
    logs = {key: [math.log(a / b) for a, b in zip(values[x], values[y])]
            for key, (x, y) in comparisons.items()}
    # Negative interaction = arrival is more beneficial under the pipeline.
    logs['interaction'] = [a - b for a, b in zip(logs['arrival_pipeline'], logs['arrival_barrier'])]
    result = dict(blocks=n, mean_ms={k: statistics.mean(v) for k, v in values.items()}, comparisons={})
    rng = random.Random(seed)
    samples = {key: [] for key in logs}
    if n > 1:
        for _ in range(resamples):
            indices = [rng.randrange(n) for _ in range(n)]
            for key, series in logs.items():
                samples[key].append(100 * (1 - math.exp(statistics.mean(series[i] for i in indices))))
    for key, series in logs.items():
        sorted_samples = sorted(samples[key])
        result['comparisons'][key] = dict(
            percent=100 * (1 - math.exp(statistics.mean(series))),
            ci95=None if n < 2 else [sorted_samples[int(resamples * .025)], sorted_samples[int(resamples * .975)]],
            meaning='ratio of arrival ratios; positive favors pipeline interaction' if key == 'interaction'
                    else 'paired elapsed-time reduction; positive is faster')
    return result
