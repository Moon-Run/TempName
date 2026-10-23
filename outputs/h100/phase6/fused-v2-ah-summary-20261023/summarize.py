"""Summarize all 48 unique taco_fused windows from the new-monitoring a-h runs."""
import hashlib
import json
import math
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SESSIONS = (
    ('ab', 'gpt67-tp8-fusedv2-20261012-211935-repeat1'),
    ('cd', 'gpt67-tp8-fusedv2-20261012-211935-repeat2'),
    ('ef', 'gpt67-tp8-fusedv2-20261013-001822-repeat1'),
    ('gh', 'gpt67-tp8-fusedv2-20261013-001822-repeat2'),
)


def read(path):
    return json.loads(path.read_text())


evidence = {}


def register(path):
    evidence[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()


windows, rounds = [], {}
for letters, session in SESSIONS:
    root = REPO/'logs/h100/phase6'/session/'compare'
    report, submission = read(root/'report.json'), read(root/'submission.json')
    register(root/'report.json')
    register(root/'submission.json')
    assert report['completed'] and report['fused_baseline_implementation'] == 'a800-fused-v2-cta-v1'
    assert submission['config']['warmup_steps'] == 10 and submission['config']['timed_steps'] == 20
    for number, letter in enumerate(letters, 1):
        selected = sorted((root/'results'/f'round{number}').glob('window-*-taco_fused'))
        assert len(selected) == 6
        samples = []
        for directory in selected:
            rows = []
            for rank in range(8):
                path = directory/f'rank{rank}.json'
                r = read(path)
                register(path)
                assert r['completed'] and r['passed'] and r['rank'] == rank
                assert r['policy'] == 'taco_fused' and r['mode'] == 'timing'
                assert r['run_id'] == submission['run_id']
                assert r['warmup_steps'] == 10 and r['timed_steps'] == 20
                assert r['tokens_per_optimizer_step'] == 8192 and r['skips'] == r['fallback_calls'] == 0
                assert not r['profiling'] and not r['sampling']
                assert len(r['losses']) == len(r['grad_norms']) == 20
                assert len(r['target_modules']) == 64
                quantized = [name for name,value in r['module_policies'].items() if value == 'taco_fused']
                attention = [name for name,value in r['module_policies'].items() if value == 'original']
                assert len(quantized) == len(attention) == 32
                assert all(name.endswith('.mlp.linear_fc2') and r['codec_placements'][name] == 'fused'
                           for name in quantized)
                assert all(name.endswith('.self_attention.linear_proj') and r['codec_placements'][name] == 'bf16'
                           for name in attention)
                assert all(math.isfinite(v) for v in r['losses']+r['grad_norms'])
                assert math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] > 0
                rows.append(r)
            assert all(r['losses'] == rows[0]['losses'] for r in rows)
            row = dict(round=letter, directory=str(directory.relative_to(REPO)),
                ms_per_step=max(r['elapsed_seconds'] for r in rows)*1000/20,
                mean_loss=statistics.mean(rows[0]['losses']), final_loss=rows[0]['losses'][-1])
            windows.append(row)
            samples.append(row['ms_per_step'])
        median = statistics.median(samples)
        expected = next(r for r in report['rounds'] if r['round'] == number)['median_ms']['taco_fused']
        assert math.isclose(median, expected, rel_tol=1e-12)
        rounds[letter] = dict(windows=6, mean_ms=statistics.mean(samples), median_ms=median,
                              maximum_ms=max(samples), report_median_matches=True)

assert len(windows) == len({r['directory'] for r in windows}) == 48
mean_ms = statistics.mean(r['ms_per_step'] for r in windows)
cr2 = read(REPO/'logs/h100/phase6/ma2-cr2-2x-50steps-20261023-153237-090898/fullstep/report.json')
cr2_mean = statistics.mean(r['ms_per_step'] for r in cr2['all_samples'] if r['arm'] == 'CR2')
result = dict(completed=True, gpu_used=False, label='Flux+改进量化', policy='taco_fused',
    implementation='a800-fused-v2-cta-v1', scope='32 MLP linear_fc2 forward projections; full optimizer-step timing',
    windows=48, timing_rank_records=384, warmup_steps_per_window=10, timed_steps_per_window=20,
    formal_optimizer_steps=960, rounds=rounds,
    mean_ms=mean_ms, median_ms=statistics.median(r['ms_per_step'] for r in windows),
    tokens_per_second=8192000/mean_ms, mean_loss=statistics.mean(r['mean_loss'] for r in windows),
    final_loss=statistics.mean(r['final_loss'] for r in windows),
    cr2_descriptive_drop_percent=100*(1-cr2_mean/mean_ms),
    all_windows_retained=True, evidence_sha256=evidence, all_samples=windows,
    limitation='Historical MLP-only 20-step windows. Cross-batch/scope/duration differences are descriptive, not controlled gains.')
(OUT/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('evidence_sha256','all_samples')},ensure_ascii=False,indent=2))
