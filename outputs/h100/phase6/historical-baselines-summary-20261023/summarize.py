"""Recompute full-window means for three historical baselines; no GPU run."""
import hashlib
import json
import math
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
AH = ['gpt67-tp8-fusedv2-20261012-211935-repeat1', 'gpt67-tp8-fusedv2-20261012-211935-repeat2',
      'gpt67-tp8-fusedv2-20261013-001822-repeat1', 'gpt67-tp8-fusedv2-20261013-001822-repeat2']
AD = [f'TTention-MLP-gpt67-tp8-20261015-012238-repeat{i}' for i in range(1,5)]
GROUPS = [('Megatron+朴素量化', 'native_taco_attn_mlp', AD, 4, 'tensor'),
          ('原生Megatron', 'native', AH, 6, 'native'),
          ('原始Flux', 'original', AH, 6, 'original')]


def read(path):
    return json.loads(path.read_text())


evidence = {}


def register(path):
    evidence[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()


cr2_path = REPO/'logs/h100/phase6/ma2-cr2-2x-50steps-20261023-153237-090898/fullstep/report.json'
cr2 = read(cr2_path)
register(cr2_path)
cr2_mean = statistics.mean(s['ms_per_step'] for s in cr2['all_samples'] if s['arm']=='CR2')
groups = {}
for label, policy, sessions, blocks, route in GROUPS:
    windows, rounds = [], []
    for session in sessions:
        root = REPO/'logs/h100/phase6'/session/'compare'
        report, submission = read(root/'report.json'), read(root/'submission.json')
        register(root/'report.json')
        register(root/'submission.json')
        assert report['completed']
        for number in (1,2):
            paths = sorted((root/'results'/f'round{number}').glob(f'window-*-{policy}'))
            assert len(paths) == blocks
            values = []
            for directory in paths:
                rows = []
                for rank in range(8):
                    path = directory/f'rank{rank}.json'
                    row = read(path)
                    register(path)
                    assert row['completed'] and row['passed'] and row['rank']==rank
                    assert row['run_id']==submission['run_id'] and row['policy']==policy and row['mode']=='timing'
                    assert row['warmup_steps']==10 and row['timed_steps']==20
                    assert row['tokens_per_optimizer_step']==8192 and row['skips']==row['fallback_calls']==0
                    assert not row['profiling'] and not row['sampling']
                    assert len(row['losses'])==len(row['grad_norms'])==20
                    assert all(math.isfinite(v) for v in row['losses']+row['grad_norms'])
                    assert math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>0
                    assert len(row['module_policies'])==64 and set(row['module_policies'].values())=={route}
                    if route != 'native':
                        assert len(row['codec_placements'])==64
                        assert set(row['codec_placements'].values())==({'tensor'} if route=='tensor' else {'bf16'})
                    rows.append(row)
                assert all(row['losses']==rows[0]['losses'] for row in rows)
                sample = dict(directory=str(directory.relative_to(REPO)),
                    ms_per_step=max(row['elapsed_seconds'] for row in rows)*1000/20,
                    mean_loss=statistics.mean(rows[0]['losses']), final_loss=rows[0]['losses'][-1])
                windows.append(sample)
                values.append(sample['ms_per_step'])
            if isinstance(report['rounds'],list):
                expected = next(r for r in report['rounds'] if r['round']==number)['median_ms'][policy]
            else:
                expected = statistics.median(report['rounds'][str(number)]['ms_per_step'][policy])
            assert math.isclose(statistics.median(values),expected,rel_tol=1e-12)
            rounds.append(dict(session=session,round=number,windows=blocks,
                               median_ms=expected,mean_ms=statistics.mean(values),report_median_matches=True))
    assert len(windows)==8*blocks==len({s['directory'] for s in windows})
    mean = statistics.mean(s['ms_per_step'] for s in windows)
    groups[label] = dict(policy=policy,windows=len(windows),timing_rank_records=len(windows)*8,
        formal_optimizer_steps=len(windows)*20,mean_ms=mean,
        median_ms=statistics.median(s['ms_per_step'] for s in windows),
        mean_of_round_medians_ms=statistics.mean(r['median_ms'] for r in rounds),
        mean_loss=statistics.mean(s['mean_loss'] for s in windows),
        final_loss=statistics.mean(s['final_loss'] for s in windows),tokens_per_second=8192000/mean,
        cr2_descriptive_drop_percent=100*(1-cr2_mean/mean),rounds=rounds,all_samples=windows)
    print(label,json.dumps({k:v for k,v in groups[label].items() if k not in ('rounds','all_samples')},ensure_ascii=False),flush=True)

result = dict(completed=True,gpu_used=False,all_windows_retained=True,
    warmup_steps_per_window=10,timed_steps_per_window=20,groups=groups,evidence_sha256=evidence,
    limitation='Historical full-window statistics, not a same-batch comparison with CR2 or 50-step runs.')
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
