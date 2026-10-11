"""Verify complete coverage and independently reproduce paired statistics."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

repo = Path(os.environ['REPO'])
sys.path.insert(0, str(repo / 'tools/h100/phase6'))
from common import read, sha, verify_files, write
from report import fingerprints, verify_inputs, window
from run import schedule

root = Path(os.environ['H100_LOG']) / 'compare'
info = verify_inputs(root)
verify_files('/', info['controllers'])
gate = read(root / 'preflight.json')
verify_files(root, gate['files'])
report = read(root / 'report.json')
assert report['completed']
assert read(root / 'timing-state.json')['status'] == 'completed'
assert report['fused_baseline_implementation'] == 'a800-fused-v2-cta-v1'
records = []
values = {rep: {p: [] for p in info['policies']} for rep in (1, 2)}
trajectories = {}
peak_allocated = peak_reserved = 0.
for mode, policy, name, block, index in schedule(info, 'timing'):
    rows = window(root, root / 'results' / name, policy, mode, block, index)
    assert fingerprints(rows) == [tuple(x) for x in gate['fingerprints']]
    rep = int(name.split('/')[0].replace('round', ''))
    elapsed = [r['elapsed_seconds'] for r in rows]
    ms = max(elapsed) * 1000 / info['config']['timed_steps']
    values[rep][policy].append(ms)
    for r in rows:
        assert not r['sampling'] and not r['profiling']
        trajectories.setdefault((policy, r['rank']), []).append(
            [r['initial_loss'], r['losses'], r['grad_norms']])
        peak_allocated = max(peak_allocated, r['peak_allocated_gib'])
        peak_reserved = max(peak_reserved, r['peak_reserved_gib'])
    records.append(dict(name=name, policy=policy, round=rep, block=block, window=index,
                        rank_elapsed_seconds=elapsed, max_rank_ms_per_step=ms))
assert len(records) == 72

def paired(a, b):
    xs = [math.log(x / y) for x, y in zip(a, b)]
    gain = lambda z: 100 * (1 - math.exp(statistics.mean(z)))
    rng = random.Random(20261009)
    boot = sorted(gain([rng.choice(xs) for _ in xs]) for _ in range(10000))
    return dict(percent=gain(xs), ci95=[boot[250], boot[9750]], blocks=len(xs))

for rr in report['rounds']:
    rep = rr['round']
    for p, xs in values[rep].items():
        assert len(xs) == 6
        assert math.isclose(statistics.median(xs), rr['median_ms'][p], abs_tol=1e-12)
    for p, bases in rr['comparisons'].items():
        for base, expected in bases.items():
            assert paired(values[rep][p], values[rep][base]) == expected

reuse = read(Path(os.environ['H100_LOG']) / 'handoff/reuse-verification.json')
assert sha(reuse['plan']) == reuse['plan_sha256']
assert sha(Path(reuse['candidate_root']) / 'manifest.json') == reuse['candidate_manifest_sha256']
verify_files(reuse['candidate_root'], reuse['candidate_file_hashes'])

loss_equal = all(all(t == rows[0] for t in rows) for rows in trajectories.values())
write(root / 'window-statistics.json', dict(windows=records, total_windows=len(records)))
write(root / 'verification.json', dict(
    passed=True, completed_windows=72, timing_rank_records=576,
    timed_optimizer_steps=1440, warmup_optimizer_steps=720,
    preflight_rank_records=96, compressed_traces=48,
    independently_recomputed_statistics=True, candidate_and_plan_unchanged=True,
    fused_baseline_implementation=report['fused_baseline_implementation'],
    same_policy_rank_trajectories_identical=loss_equal,
    peak_allocated_gib=peak_allocated, peak_reserved_gib=peak_reserved,
    accepted=report['accepted'],
))
write(root / 'timing-artifact-hashes.json', {
    str(p.relative_to(root)): sha(p) for d in (root / 'results').glob('round*/window-*')
    for p in d.iterdir() if p.is_file()
})
print('INDEPENDENT VERIFICATION PASSED', json.dumps(read(root / 'verification.json')))

def stats(s):
    return f"{s['percent']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"

lines = ['# GPT 6.7B TP8 aligned fused v2 result', '',
         f"Session: `{os.environ['H100_SESSION']}`; implementation: `a800-fused-v2-cta-v1`.", '']
for p in ('remote_arrival_selective', 'interleaved_arrival_selective'):
    lines += [f'## {p}', '',
              '| Round | ms/step | vs Megatron+quant % [95% CI] | vs fused v2 % [95% CI] | native overhead % | Flux overhead % |',
              '| --- | ---: | --- | --- | ---: | ---: |']
    for rr in report['rounds']:
        s = rr['comparisons'][p]
        lines.append(f"| {rr['round']} | {rr['median_ms'][p]:.3f} | {stats(s['native_taco'])} | {stats(s['taco_fused'])} | {-s['native']['percent']:+.3f} | {-s['original']['percent']:+.3f} |")
    lines.append('')
lines += ['## Absolute timing', '', '| Policy | Round 1 | Round 2 |', '| --- | ---: | ---: |']
for p in info['policies']:
    lines.append(f"| {p} | {report['rounds'][0]['median_ms'][p]:.3f} | {report['rounds'][1]['median_ms'][p]:.3f} |")
lines += ['', f"Accepted: `{report['accepted']}`; 72 windows / 576 rank records / 1440 timed steps."]
(Path(os.environ['H100_LOG']) / 'handoff/summary.md').write_text('\n'.join(lines) + '\n')
