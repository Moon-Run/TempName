#!/usr/bin/env python3
"""Offline, read-only audit of returned op4-v2 operator evidence."""
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

REPO = Path(__file__).resolve().parents[4]
ROOT = REPO/'logs/h100/phase6/arrival-op4-v2-2x-50steps-20261021-115709-698211/operator'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO/'tools/h100/phase6'))
import arrival_op4_v2_bench as bench

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = bench.verify(ROOT, offline=True)
report = bench.report(ROOT, save=False, offline=True)
profile = bench.profile_summary(ROOT, offline=True)
assert report == read(ROOT/'report.json')
assert profile == read(ROOT/'profile-summary.json')

def estimate(logs, seed=20261021):
    rng = random.Random(seed)
    boot = sorted(100*(1-math.exp(statistics.mean(rng.choices(logs,k=len(logs))))) for _ in range(10000))
    return dict(percent=100*(1-math.exp(statistics.mean(logs))),ci95=[boot[249],boot[9749]],blocks=len(logs))

samples=report['all_samples']
blocks={b:{r['arm']:r['rankmax_wall_us'] for r in samples if r['block']==b} for b in range(16)}
phases={}
interactions={}
for role in bench.ROLES:
    logs=[math.log(t['CM'][role]/t['SM'][role])-math.log(t['CP'][role]/t['SP'][role]) for t in blocks.values()]
    interactions[role]=estimate(logs)
    phases[role]={}
    for phase in range(8):
        b=[t for i,t in blocks.items() if i%8==phase]
        phases[role][phase]=dict(mean_us={a:statistics.mean(t[a][role] for t in b) for a in bench.ARMS},
            cm_vs_sm_percent=100*(1-math.exp(statistics.mean(math.log(t['CM'][role]/t['SM'][role]) for t in b))),blocks=2)

counts=collections.Counter()
metrics={a:collections.defaultdict(list) for a in bench.ARMS}
profile_metrics={a:collections.defaultdict(list) for a in bench.ARMS}
for path in sorted((ROOT/'results').glob('*/rank[0-7].json')):
    row=read(path); is_profile=row['profiling']; scope='profile' if is_profile else 'timing'
    counts[scope+'_rank_files']+=1
    for role,record in row['roles'].items():
        for check in record['checks']:
            counts[scope+'_codec_checks']+=1
            counts[scope+'_packet_checks']+=1
            assert check['correctness']['passed'] and check['packet_check']['passed']
            target=profile_metrics if is_profile else metrics
            for name,value in check['correctness'].items():
                if name in ['max_abs','relative_l2','protocol_relative_l2','max_tile_relative_l2','absolute_error_p99']:
                    target[row['job']['arm']][role+'/'+name].append(value)
            target[row['job']['arm']][role+'/packet_relative_l2'].append(check['packet_check']['relative_l2'])
        for check in record['retained_output_checks']:
            counts[scope+'_retained_output_checks']+=1
            assert check['correctness']['passed']
        assert record['workspace_count']==1 and not record['owns_output']
counts['timing_same_mask_packet_pairs']=counts['timing_packet_checks']//2
for path in (ROOT/'results').glob('*/rank*-*-profile.json.gz'):
    counts['profile_trace_files']+=1

numeric={a:{k:dict(min=min(v),max=max(v),mean=statistics.mean(v),count=len(v)) for k,v in ms.items()} for a,ms in metrics.items()}
profile_numeric={a:{k:dict(min=min(v),max=max(v),mean=statistics.mean(v),count=len(v)) for k,v in ms.items()} for a,ms in profile_metrics.items()}
profile_summary={}
for role in bench.ROLES:
    profile_summary[role]={}
    for arm in bench.ARMS:
        rows=[profile['profiles'][f'{arm}/rank{r}/{role}'] for r in range(8)]
        profile_summary[role][arm]={metric:dict(median=statistics.median(x[metric] for x in rows),mean=statistics.mean(x[metric] for x in rows),min=min(x[metric] for x in rows),max=max(x[metric] for x in rows)) for metric in ['gemm_median_us','decoder_median_us']}

source_checks={}
map_sources={'scripts/bench.py':'tools/h100/phase6/arrival_op4_v2_bench.py','scripts/closure_plan.py':'tools/h100/phase6/arrival_op4_v2_plan.py','scripts/closure_reference.py':'tools/h100/phase6/arrival_op4_v2_reference.py','scripts/reference.py':'tools/a800/phase4/reference.py'}
for frozen,local in map_sources.items():
    source_checks[local]=dict(expected=plan['source_files'][frozen],actual=sha(REPO/local),match=sha(REPO/local)==plan['source_files'][frozen])
for remote,digest in plan['build_identity']['builder_inputs'].items():
    local=remote.removeprefix('/workspace/TempName/')
    source_checks[local]=dict(expected=digest,actual=sha(REPO/local),match=sha(REPO/local)==digest)
assert all(x['match'] for x in source_checks.values())
result=dict(root=str(ROOT.relative_to(REPO)),report_recomputed_equal=True,profile_recomputed_equal=True,
    counts=dict(counts),source_checks=source_checks,roles=report['roles'],numeric=numeric,profile_numeric=profile_numeric,
    same_mask_max_sample_relative_l2=report['same_mask_max_sample_relative_l2'],
    interaction=interactions,interaction_formula='100*(1-exp(mean(log(CM/SM)-log(CP/SP))))',interaction_bootstrap_seed=20261021,
    phase_descriptive=phases,profile_aggregate=profile_summary,micro_savings_estimate=report['micro_savings_estimate'],
    limitations=['Recomputation uses offline=True: compiled binaries and generated build sources were not returned, so offline audit cannot rehash them.',
    'Profile covers phase 0 only, four calls per rank-role-arm; median across ranks of per-rank kernel medians is diagnostic, not timing estimator.',
    'Inverse-H128 counts are analytic protocol counts, not hardware counters.',
    'Per-phase figures use two blocks and are descriptive only.'])
(OUT/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
(OUT/'report.recomputed.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
(OUT/'profile-summary.recomputed.json').write_text(json.dumps(profile,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:result[k] for k in ['report_recomputed_equal','profile_recomputed_equal','counts','interaction','phase_descriptive','profile_aggregate']},indent=2))
