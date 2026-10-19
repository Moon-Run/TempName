import json, math, statistics, itertools
from pathlib import Path
path=Path('logs/h100/phase6/ablation6-then7-3x-50steps-20261019-162154-480841/ablation7/compare1/scripts/selection-plan.json')
x=json.loads(path.read_text())
out={'input':str(path),'roles':{}}
def stat(v):return dict(min=min(v),median=statistics.median(v),max=max(v))
for role,order in x['orders'].items():
 groups=order['entry']['arrival_order']['groups']; gr=[]
 for g in groups:
  ts=g['training']; n=len(g['slots']); eps=[t['max_uncertainty_us'] for t in ts]
  vals=[t['arrivals_us'] for t in ts]; base=[t['baseline_us'] for t in ts]; res=[t['residual_us'] for t in ts]; mx=[max(v) for v in vals]
  tops=[sorted(range(n),key=lambda i:-v[i])[:3] for v in vals]
  r=dict(source=g['source'],destination=g['destination'],uncertainty_us=eps,
    margin_6x_us=[6*e for e in eps],residual_range_us=[max(v)-min(v) for v in res],
    positive_residual_max_us=[max(v) for v in res],
    max_ties=[sum(abs(a-m)<1e-9 for a in v) for v,m in zip(vals,mx)],
    max_slot=[max(range(n),key=lambda i: vals[p][i]) for p in range(2)],
    near_max_count_6x=[sum(m-a<=6*e for a in v) for v,m,e in zip(vals,mx,eps)])
  r['same_max_slot']=r['max_slot'][0]==r['max_slot'][1]
  counters={k:0 for k in ('pairs','prefilter_pass0','prefilter_pass1','prefilter_both','prefilter_either_not_both','gain_positive_both','gain_positive_both_prefilter','gain_1x_both','gain_2x_both','gain_6x_both','gain_6x_both_prefilter')}
  best=[-math.inf,-math.inf]; best_both=-math.inf; best_pref=-math.inf; best_gain_margin=-math.inf
  for i in range(n):
   for j in range(i+1,n):
    eligible=[res[p][j]>eps[p] and res[p][j]-res[p][i]>2*eps[p] for p in range(2)]
    gains=[]
    for p in range(2):
     rest=next((vals[p][z] for z in tops[p] if z not in (i,j)),-math.inf)
     after=max(rest,base[p][i]+res[p][j],base[p][j]+res[p][i]); gains.append(mx[p]-after)
    counters['pairs']+=1
    for p in range(2):
     counters[f'prefilter_pass{p}']+=eligible[p];best[p]=max(best[p],gains[p])
    counters['prefilter_both']+=all(eligible);counters['prefilter_either_not_both']+=any(eligible) and not all(eligible)
    counters['gain_positive_both']+=all(v>1e-12 for v in gains)
    counters['gain_positive_both_prefilter']+=all(v>1e-12 for v in gains) and all(eligible)
    for m in (1,2,6):counters[f'gain_{m}x_both']+=all(v>m*e+1e-12 for v,e in zip(gains,eps))
    counters['gain_6x_both_prefilter']+=all(v>6*e+1e-12 for v,e in zip(gains,eps)) and all(eligible)
    best_both=max(best_both,min(gains));best_gain_margin=max(best_gain_margin,min(v-6*e for v,e in zip(gains,eps)))
    if all(eligible):best_pref=max(best_pref,min(gains))
  r['counters']=counters;r['best_individual_gain_us']=best;r['best_common_gain_us']=best_both;r['best_prefilter_common_gain_us']=best_pref if math.isfinite(best_pref) else None;r['best_common_net_6x_margin_us']=best_gain_margin
  # Unconstrained per-pass bottleneck assignment: anti-sort residual against nondecreasing baseline.
  opt=[mx[p]-max(a+b for a,b in zip(base[p],sorted(res[p],reverse=True))) for p in range(2)]
  r['per_pass_independent_optimum_gain_us']=opt
  # Shared candidate, not a valid speedup prediction: rank residual mean, no heldout fitting.
  pi=sorted(range(n),key=lambda i:(-(res[0][i]+res[1][i])/2,i))
  r['mean_residual_full_order_gain_us']=[mx[p]-max(base[p][slot]+res[p][tile] for slot,tile in enumerate(pi)) for p in range(2)]
  gr.append(r)
 totals={k:sum(r['counters'][k] for r in gr) for k in gr[0]['counters']}
 counts={k:sum(r['counters'][k]>0 for r in gr) for k in totals}
 summary=dict(groups=len(gr),pair_totals=totals,groups_with_nonzero_pair_count=counts,
   uncertainty_us=stat([v for r in gr for v in r['uncertainty_us']]),margin6_us=stat([v for r in gr for v in r['margin_6x_us']]),
   residual_range_us=stat([v for r in gr for v in r['residual_range_us']]),positive_residual_max_us=stat([v for r in gr for v in r['positive_residual_max_us']]),
   groups_exact_tied_tail_either_pass=sum(any(v>1 for v in r['max_ties']) for r in gr),groups_same_max_slot=sum(r['same_max_slot'] for r in gr),
   groups_both_individually_improvable=sum(all(v>1e-12 for v in r['best_individual_gain_us']) for r in gr),
   groups_both_individually_improvable_no_common_swap=sum(all(v>1e-12 for v in r['best_individual_gain_us']) and r['best_common_gain_us']<=1e-12 for r in gr),
   best_common_gain_us=stat([r['best_common_gain_us'] for r in gr]),
   independent_optimal_gain_us=stat([v for r in gr for v in r['per_pass_independent_optimum_gain_us']]),
   groups_independent_optima_exceed6x_both=sum(all(v>6*e+1e-12 for v,e in zip(r['per_pass_independent_optimum_gain_us'],r['uncertainty_us'])) for r in gr),
   max_window_min_pass_optimal_gain_over_uncertainty=max(min(v/e for v,e in zip(r['per_pass_independent_optimum_gain_us'],r['uncertainty_us'])) for r in gr),
   groups_mean_full_order_positive_both=sum(all(v>1e-12 for v in r['mean_residual_full_order_gain_us']) for r in gr),
   groups_mean_full_order_exceeds6x_both=sum(all(v>6*e+1e-12 for v,e in zip(r['mean_residual_full_order_gain_us'],r['uncertainty_us'])) for r in gr))
 out['roles'][role]=dict(summary=summary,groups=gr)
Path('/tmp/op1-residual-rejection-analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({r:v['summary'] for r,v in out['roles'].items()},indent=2))
