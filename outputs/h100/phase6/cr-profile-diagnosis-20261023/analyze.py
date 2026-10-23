#!/usr/bin/env python3
"""Read-only stage timing diagnosis from separately collected PyTorch traces."""
import bisect,collections,gzip,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
BATCH=ROOT/'logs/h100/phase6/ma2r-c-ablation-2x-50steps-20261023-104402-375258'
OUT=Path(__file__).resolve().parent

def stats(v):
    v=sorted(v);n=len(v)
    return dict(n=n,sum=sum(v),mean=statistics.mean(v),median=statistics.median(v),p10=v[int((n-1)*.1)],p90=v[int((n-1)*.9)],maximum=v[-1]) if v else {}

def analyze(path):
    events=json.load(gzip.open(path,'rt'))['traceEvents']
    cpu=[e for e in events if e.get('cat')=='cpu_op']; calls=sorted([e for e in cpu if e['name']=='GemmRSFunction'],key=lambda e:e['ts'])
    scopes=sorted([e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('scope.')],key=lambda e:e['ts']);starts=[e['ts'] for e in scopes]
    runtimes=[e for e in events if e.get('cat')=='cuda_runtime'];launches={e.get('args',{}).get('correlation'):e for e in runtimes if e['name'] in ('cudaLaunchKernel','cudaLaunchKernelExC','cudaMemcpyAsync','cudaMemsetAsync')}
    byexternal=collections.defaultdict(list)
    for e in events:
        if e.get('cat')=='kernel':byexternal[e.get('args',{}).get('External id')].append(e)
    streams=collections.defaultdict(list)
    for e in events:
        if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset'):streams[e['args']['stream']].append(e)
    for v in streams.values():v.sort(key=lambda e:e['ts'])
    indices={id(e):i for v in streams.values() for i,e in enumerate(v)}
    runtime_order=sorted(runtimes,key=lambda e:e['ts']); runtime_starts=[e['ts'] for e in runtime_order]
    cpu_order=sorted(cpu,key=lambda e:e['ts']); cpu_starts=[e['ts'] for e in cpu_order]
    groups=collections.defaultdict(lambda:collections.defaultdict(list)); detailed=[]
    for call in calls:
        scope=scopes[bisect.bisect_right(starts,call['ts'])-1];role=scope['name'].split('.')[1]
        assert call['ts']+call['dur']<=scope['ts']+scope['dur']+.01
        ext=call['args']['External id']; ks=sorted(byexternal[ext],key=lambda e:e['ts'])
        gemm=[e for e in ks if 'flux_bf16' in e['name'] and 'reducescatter' in e['name']][0]
        dec=[e for e in ks if 'taco' in e['name'] and 'decode' in e['name']][0]
        bars=[e for e in ks if 'CudaIpcBarrierAllKernel' in e['name']]
        rt=[e for e in runtime_order[bisect.bisect_left(runtime_starts,call['ts']):bisect.bisect_right(runtime_starts,call['ts']+call['dur'])] if call['ts']<=e['ts'] and e['ts']+e['dur']<=call['ts']+call['dur']+.01 and e['tid']==call['tid']]
        ops=[e for e in cpu_order[bisect.bisect_left(cpu_starts,call['ts']):bisect.bisect_right(cpu_starts,call['ts']+call['dur'])] if call['ts']<=e['ts'] and e['ts']+e['dur']<=call['ts']+call['dur']+.01 and e['tid']==call['tid'] and e is not call]
        # leaf CPU ops only: parent/view nested overhead must not be double counted.
        leaf=[e for e in ops if not any(e is not q and e['ts']<=q['ts'] and q['ts']+q['dur']<=e['ts']+e['dur']+.001 for q in ops)]
        dl=launches[dec['args']['correlation']];gl=launches[gemm['args']['correlation']]
        sv=streams[dec['args']['stream']]; ni=indices[id(dec)]+1; nex=sv[ni] if ni<len(sv) else None
        row=dict(role=role,cpu_us=call['dur'],scope_us=scope['dur'],gemm_us=gemm['dur'],decoder_us=dec['dur'],barrier_us=sum(b['dur'] for b in bars),
            pipeline_gpu_us=ks[-1]['ts']+ks[-1]['dur']-ks[0]['ts'],runtime_api_us=sum(e['dur'] for e in rt),leaf_tensor_op_us=sum(e['dur'] for e in leaf),
            start_to_gemm_launch_us=gl['ts']-call['ts'],gemm_to_decode_launch_us=dl['ts']-gl['ts'],decode_launch_to_return_us=call['ts']+call['dur']-dl['ts']-dl['dur'],
            gemm_launch_queue_us=gemm['ts']-gl['ts']-gl['dur'],decode_launch_queue_us=dec['ts']-dl['ts']-dl['dur'],
            gemm_to_barrier_gpu_gap_us=bars[0]['ts']-gemm['ts']-gemm['dur'],barrier_to_decode_gpu_gap_us=dec['ts']-bars[0]['ts']-bars[0]['dur'],
            next_stream_gap_us=(nex['ts']-dec['ts']-dec['dur']) if nex else 0,
            runtime_call_count=len(rt),kernel_launch_count=sum(e['name'] in ('cudaLaunchKernel','cudaLaunchKernelExC') for e in rt))
        if nex and nex['args'].get('correlation') in launches:
            nl=launches[nex['args']['correlation']]
            row['next_launch_lateness_us']=nl['ts']+nl['dur']-(dec['ts']+dec['dur'])
        for name in ('aten::empty','aten::slice','aten::view','aten::select','aten::reshape','aten::clone'):
            row[name+'_us']=sum(e['dur'] for e in ops if e['name']==name)
        for k,v in row.items():
            if k!='role':groups[role][k].append(v)
        detailed.append(row)
    gpu=[e for v in streams.values() for e in v]; span=max(e['ts']+e['dur'] for e in gpu)-min(e['ts'] for e in gpu)
    kinds=collections.defaultdict(list)
    for e in gpu:
        name=e['name']; kind='flux_gemm' if 'flux_bf16' in name and 'reducescatter' in name else 'taco_decoder' if 'taco' in name and 'decode' in name else 'ipc_barrier' if 'CudaIpcBarrier' in name else 'nccl' if 'nccl' in name else e['cat']
        kinds[kind].append(e['dur'])
    main=max(streams,key=lambda s:sum('taco' in e['name'] and 'decode' in e['name'] for e in streams[s])); sv=streams[main]
    gaps=[max(0,b['ts']-a['ts']-a['dur']) for a,b in zip(sv,sv[1:])]
    return dict(trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gpu_span_us=span,main_stream=main,main_stream_gap_us=stats(gaps),gpu_category_us={k:stats(v) for k,v in kinds.items()},
        projection={role:{k:stats(v) for k,v in cols.items()} for role,cols in groups.items()},
        projection_queued_fractions={role:{k:statistics.mean(v) for k,v in [('gemm_launch_queue_over_5us',[r['gemm_launch_queue_us']>5 for r in detailed if r['role']==role]),('next_launch_submitted_before_decoder_finishes',[r.get('next_launch_lateness_us',1)<0 for r in detailed if r['role']==role])]} for role in groups})

def main():
    result={'batch':str(BATCH.relative_to(ROOT)), 'warning':'Separate profiles include overhead. Summed kernel durations and queue measurements are not estimates of unprofiled critical path or predicted speedup.', 'ranks':{}}
    for arm in ('CR','C','R','A'):
        for rank in range(8):
            path=BATCH/'fullstep/results'/f'profile-{arm}'/f'rank{rank}-profile.json.gz'
            result['ranks'][f'{arm}/{rank}']=analyze(path)
            print(arm,rank,flush=True)
    summary={}
    for arm in ('CR','C','R','A'):
        rs=[result['ranks'][f'{arm}/{rank}'] for rank in range(8)]
        summary[arm]={'gpu_span_us_median_rank':statistics.median(r['gpu_span_us'] for r in rs),'projection':{}}
        for role in ('attn','mlp'):
            summary[arm]['projection'][role]={k:statistics.median(r['projection'][role][k]['median'] for r in rs) for k in rs[0]['projection'][role]}
            summary[arm]['projection'][role]['next_launch_prequeued_fraction']=statistics.median(r['projection_queued_fractions'][role]['next_launch_submitted_before_decoder_finishes'] for r in rs)
        summary[arm]['gpu_category_sum_us_median_rank']={k:statistics.median(r['gpu_category_us'][k]['sum'] for r in rs) for k in rs[0]['gpu_category_us']}
    result['summary']=summary
    result['rank_queue_table']={key:dict(gemm_launch_queue_median_us={role:row['projection'][role]['gemm_launch_queue_us']['median'] for role in ('attn','mlp')}, fractions=row['projection_queued_fractions'], ipc_barrier_total_us=row['gpu_category_us']['ipc_barrier']['sum'], nccl_total_us=row['gpu_category_us']['nccl']['sum']) for key,row in result['ranks'].items()}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
