"""Independently reconstruct all-block statistics and verify local delivery inputs."""
import math
import os
from pathlib import Path
import random
import statistics
import sys

repo = Path(os.environ['REPO'])
out = Path(os.environ['H100_SCOPE_OUT'])
log = Path(os.environ['H100_SCOPE_LOG'])
root = log / 'compare'
sys.path.insert(0, str(repo / 'tools/h100/phase6'))
from common import read, sha, verify_files, write
from scope_calibrate import verify, calibration_records
from scope_plan import validate
from scope_protocol import POLICIES, SHAPES
from scope_report import COMPARISONS, PRIMARY, verify_inputs, window
from report import fingerprints

info = verify_inputs(root)
verify_files('/', info['controllers'])
gate = read(root / 'preflight.json')
verify_files(root, gate['files'])
result = read(root / 'report.json')
assert result['completed'] and not result['offline'] and result['timing_windows'] == 32
assert read(root / 'all-state.json')['status'] == 'completed'
verify_files(root, result['evidence']['timing_files'])
cal = log / 'calibration'
prepared = verify(cal, execute=True)
calibration_rows = calibration_records(cal, prepared)
assert len(calibration_rows) == 8 and prepared['roles'] == ['attn', 'mlp']
plans = read(root / 'scripts/selection-plan.json')['roles']
for role in SHAPES:
    validate(plans[role], role)
    assert plans[role]['origin']['source'] == str(cal)
    assert plans[role]['origin']['kind'] == 'scope_bf16_calibration'
    assert plans[role]['training_passes'] == [0, 1] and plans[role]['heldout_passes'] == [2]
    assert plans[role] == read(out / 'plans/measured' / (role + '.json'))

values = {rep: {key: [] for key in POLICIES} for rep in (1, 2)}
actual_samples = []
for job in info['schedule']:
    rows = window(root, root / 'results' / job['directory'], job['policy'], 'timing', job['block'], job['index'])
    assert fingerprints(rows) == [tuple(x) for x in gate['fingerprints']]
    elapsed = [r['elapsed_seconds'] for r in rows]
    ms = max(elapsed) * 1000 / 20
    values[job['round']][job['policy']].append(ms)
    actual_samples.append(dict(job, rank_elapsed_seconds=elapsed, ms_per_step=ms, tokens_per_second=8192000 / ms))
assert actual_samples == result['all_samples']

def paired(logs):
    gain = lambda xs: 100 * (1 - math.exp(statistics.mean(xs)))
    if len(logs) == 1:
        return dict(percent=100 * (1 - math.exp(logs[0])), ci95=None, blocks=1)
    rng = random.Random(20261009)
    boot = sorted(gain([rng.choice(logs) for _ in logs]) for _ in range(10000))
    return dict(percent=gain(logs), ci95=[boot[250], boot[9750]], blocks=len(logs))

def verify_summary(summary, measured, blocks):
    assert summary['blocks'] == len(blocks)
    if not blocks:
        assert not summary['comparisons'] and not summary['speedups']
        return
    for key, times in measured.items():
        assert summary['ms_per_step'][key] == times
        assert summary['median_ms'][key] == statistics.median(times)
        assert summary['median_tokens_per_second'][key] == statistics.median(8192000 / x for x in times)
    for a, b in COMPARISONS:
        logs = [math.log(x / y) for x, y in zip(measured[a], measured[b])]
        assert summary['comparisons'][a + '/' + b] == dict(paired(logs), log_ratios=logs)
    for a, b in PRIMARY:
        ratios = [y / x for x, y in zip(measured[a], measured[b])]
        best = max(range(len(ratios)), key=ratios.__getitem__)
        assert summary['speedups'][a] == dict(reference=b, median=statistics.median(ratios), ratios=ratios,
            best=dict(ratio=ratios[best], **blocks[best], reference_ms=measured[b][best], candidate_ms=measured[a][best]))

for rep in (1, 2):
    measured = values[rep]
    blocks = [dict(round=rep, block=i) for i in range(4)]
    verify_summary(result['rounds'][str(rep)], measured, blocks)
    medians = {key: statistics.median(times) for key, times in measured.items()}
    excluded = []
    kept = []
    for index, block in enumerate(blocks):
        triggers = {key: dict(ms=measured[key][index], threshold_ms=1.1 * medians[key])
                    for key in POLICIES if measured[key][index] > 1.1 * medians[key]}
        if triggers:
            excluded.append(dict(block, triggers=triggers))
        else:
            kept.append(index)
    filtered = result['sensitivity'][str(rep)]
    assert filtered['excluded'] == excluded and filtered['remaining_blocks'] == len(kept)
    verify_summary(filtered['summary'], {key: [times[i] for i in kept] for key, times in measured.items()},
                   [blocks[i] for i in kept])
combined = {key: values[1][key] + values[2][key] for key in POLICIES}
verify_summary(result['rounds']['combined'], combined, [dict(round=rep, block=i) for rep in (1, 2) for i in range(4)])
delivery = read(log / 'handoff/delivery.json')
assert delivery['status'] == 'complete' and delivery['actual'] == delivery['expected']
write(log / 'handoff/independent-verification.json', dict(
    passed=True, verified_on_disk_libraries=True, official_report_sha256=sha(root / 'report.json'),
    complete_timing_windows=32, timing_rank_records=256, preflight_rank_records=64,
    compressed_profile_traces=32, candidate_shape_mappings=3, new_calibration_rank_records=8,
    both_roles_newly_gpu_calibrated=True, train_passes=[0, 1], heldout_passes=[2],
    independently_recomputed_medians=True, independently_recomputed_bootstrap=True,
    independently_verified_whole_block_sensitivity=True, best_speedup_uses_same_paired_block=True,
    all_samples_preserved=True, no_historical_timing_used=True,
))
lines = ['# Verified H100 Attention/MLP result', '', f'Session: `{os.environ["H100_SCOPE_SESSION"]}`.', '',
         '| Scope | Round | Baseline ms/step | Candidate ms/step | Median speedup | Best paired speedup | Reduction % [95% CI] |',
         '| --- | --- | ---: | ---: | ---: | ---: | --- |']
for rep, summary in result['rounds'].items():
    for a, b in PRIMARY:
        speed = summary['speedups'][a]
        gain = summary['comparisons'][a + '/' + b]
        ci = gain['ci95']
        label = 'Attention only' if a == 'interleaved_full_attn' else 'Attention + MLP'
        lines.append(f'| {label} | {rep} | {summary["median_ms"][b]:.3f} | {summary["median_ms"][a]:.3f} | '
                     f'{speed["median"]:.6f} | {speed["best"]["ratio"]:.6f} | '
                     f'{gain["percent"]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |')
lines += ['', '| Net extension to Attention + MLP | Reduction % [95% CI], combined 8 blocks |',
          '| --- | --- |']
for a, b in COMPARISONS[2:]:
    gain = result['rounds']['combined']['comparisons'][a + '/' + b]
    ci = gain['ci95']
    lines.append(f'| {a}/{b} | {gain["percent"]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |')
lines += ['', 'All samples are primary. Best speedup is one paired block, not a stable effect. '
          'Negative reductions mean increased elapsed time. Separate slow-window sensitivity is in compare/report.md.',
          '', 'Delivery complete: 256 timing ranks, 64 preflight ranks, 32 traces, 3 comparison mappings; '
          '8 new calibration ranks contain both full shapes, with 2 calibration mappings.']
(log / 'handoff/results-summary.md').write_text('\n'.join(lines) + '\n')
print('INDEPENDENT DELIVERY AND STATISTICS VERIFICATION PASSED')
print('\n'.join(lines))
