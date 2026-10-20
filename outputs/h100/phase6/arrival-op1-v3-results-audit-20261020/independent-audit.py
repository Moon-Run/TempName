from pathlib import Path
import json,csv,statistics,math,hashlib,collections
root=Path('logs/h100/phase6/ablation7-new-2x-50steps-20261020-110009-429282')
policies={'C':'interleaved_arrival_selective_b16_attn_mlp','A':'interleaved_all_attn_mlp','B':'interleaved_arrival_assignment_all_attn_mlp'}
reverse={v:k for k,v in policies.items()}
out={'root':str(root),'compares':{},'overall':{},'formal_rank_count':0,'smoke':{}}
all_windows=collections.defaultdict(list);all_loss=collections.defaultdict(list);all_final=collections.defaultdict(list);all_paths=collections.defaultdict(list);smoke_values=collections.defaultdict(list);plan_hashes=[]
for index in (1,2):
 c=root/f'compare{index}';r=json.loads((c/'report.json').read_text());p=c/'scripts/selection-plan.json';plan=json.loads(p.read_text());sha=hashlib.sha256(p.read_bytes()).hexdigest();plan_hashes.append(sha)
 ci={'run_id':r['run_id'],'completed':r['completed'],'correctness_passed':r['correctness_passed'],'plan_sha256':sha,'roles':{},'groups':{},'rounds':{},'formal_rank_count':0,'all_formal_completed':True,'all_finite':True,'all_50_steps':True,'all_rank_loss_identical':True,'skip_total':0,'fallback_total':0,'timing_matches_report':True}
 for role in ('attn','mlp'):
  def mapping(policy):return [{k:int(v) for k,v in row.items()} for row in csv.DictReader((c/f'results/mapping-{policy}-{role}.csv').open())]
  a,b=mapping(policies['A']),mapping(policies['B']);e=plan['orders'][role]['entry'];d=e['arrival_order'];physical=lambda row:row['m']*32+row['n']
  ci['roles'][role]={'method':d['method'],'uncertainty_multiplier':d['uncertainty_multiplier'],'max_displacement':d['max_displacement'],'accepted_windows':d['accepted_windows'],'status_counts':dict(collections.Counter(g['status'] for g in d['groups'])),'actual_mapping_changes':sum(physical(u)!=physical(v) for u,v in zip(a,b)),'actual_destination_changes':sum(u['destination']!=v['destination'] for u,v in zip(a,b)),'mapping_matches_frozen':all(physical(row)==e['coords'][row['rank']][row['index']] for row in b),'per_rank_bijection':all(sorted(physical(row) for row in b if row['rank']==rank)==list(range(512)) for rank in range(8)),'changed_per_rank':d['changed_per_rank'],'identity_fallback':d['identity_fallback'],'conservative_bound_positive_windows':d['accepted_windows_with_positive_conservative_bound']}
 windows=collections.defaultdict(list);rounds=collections.defaultdict(lambda:collections.defaultdict(list));losses=collections.defaultdict(list);finals=collections.defaultdict(list)
 for row in r['all_samples']:
  label=reverse[row['policy']]; records=[json.loads((c/'results'/row['directory']/f'rank{rank}.json').read_text()) for rank in range(8)];ci['formal_rank_count']+=len(records)
  ms=max(x['elapsed_seconds'] for x in records)*1000/50
  ci['timing_matches_report'] &= abs(ms-row['ms_per_step'])<1e-10
  for x in records:
   ci['all_formal_completed'] &= bool(x['completed'] and x['passed'])
   ci['all_finite'] &= all(math.isfinite(v) for field in ('losses','grad_norms') for v in x[field])
   ci['all_50_steps'] &= len(x['losses'])==50 and len(x['grad_norms'])==50 and x['timed_steps']==50
   ci['skip_total']+=x['skips'];ci['fallback_total']+=x['fallback_calls']
  ci['all_rank_loss_identical'] &= all(x['losses']==records[0]['losses'] for x in records)
  windows[label].append(ms);rounds[row['round']][label].append(ms);losses[label].append(statistics.mean(records[0]['losses']));finals[label].append(records[0]['losses'][-1]);all_windows[label].append(ms);all_loss[label].append(losses[label][-1]);all_final[label].append(finals[label][-1]);all_paths[label].append(records[0]['losses'])
 for label in policies:
  ci['groups'][label]={'windows':len(windows[label]),'mean_ms':statistics.mean(windows[label]),'median_ms':statistics.median(windows[label]),'throughput_tokens_s':8192000/statistics.mean(windows[label]),'mean_training_loss':statistics.mean(losses[label]),'mean_final_loss':statistics.mean(finals[label])}
 for round_id,groups in rounds.items():ci['rounds'][round_id]={label:statistics.mean(times) for label,times in groups.items()}
 for label,policy in policies.items():
  xs=[json.loads((c/f'results/smoke-{policy}/rank{rank}.json').read_text()) for rank in range(8)]
  for x in xs:
   assert x['passed'] and x['completed'] and x['fallback_calls']==0 and x['skips']==0
   assert len(x['forward_checks'])==64 and all(v['passed'] for v in x['forward_checks'].values())
   smoke_values[label].extend(x['forward_checks'].values())
 out['compares'][f'compare{index}']=ci;out['formal_rank_count']+=ci['formal_rank_count']
for label in policies:
 out['overall'][label]={'windows':len(all_windows[label]),'formal_optimizer_steps':50*len(all_windows[label]),'mean_ms':statistics.mean(all_windows[label]),'median_ms':statistics.median(all_windows[label]),'throughput_tokens_s':8192000/statistics.mean(all_windows[label]),'mean_training_loss':statistics.mean(all_loss[label]),'mean_final_loss':statistics.mean(all_final[label]),'all_window_loss_paths_identical':all(v==all_paths[label][0] for v in all_paths[label])}
 out['smoke'][label]={'rank_records':16,'projection_checks':len(smoke_values[label]),'max_relative_bf16_percent':100*max(v['relative_l2'] for v in smoke_values[label]),'max_relative_codec_percent':100*max(v['protocol_relative_l2'] for v in smoke_values[label]),'all_passed':True}
out['A_B_loss_paths_identical']=all_paths['A']==all_paths['B'];out['plan_hashes_equal']=len(set(plan_hashes))==1
for label in policies:out['overall'][label]['C_relative_drop_percent']=100*(1-out['overall']['C']['mean_ms']/out['overall'][label]['mean_ms'])
Path('/tmp/base-op1-v3-independent.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
