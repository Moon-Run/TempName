"""Fit bounded arrival-priority permutations on passes 0/1; pass 2 is held out.

A tile's score for a source is its positive last-vs-second-last gap on the
receiver clock, beyond two polling cycles. This is an exploratory heuristic,
not a calibrated network-time model. It preserves every destination slot.
"""
import argparse,csv,hashlib,json,statistics
from pathlib import Path

def permute(rows,scores,window):
    assert window>0
    rows=sorted(rows,key=lambda r:r['index'])
    assert [r['index'] for r in rows]==list(range(len(rows)))
    mapping=list(range(len(rows)))
    for dest in sorted({r['destination'] for r in rows}):
        slots=[r['index'] for r in rows if r['destination']==dest]
        for start in range(0,len(slots),window):
            group=slots[start:start+window]
            ordered=sorted(group,key=lambda i:-scores.get(rows[i]['m']*rows[i]['tn']+rows[i]['n'],0))
            for i,j in zip(group,ordered):mapping[i]=j
    assert sorted(mapping)==list(range(len(rows)))
    assert all(rows[i]['destination']==rows[j]['destination'] for i,j in enumerate(mapping))
    return mapping

def observations(shape,rank):
    per=shape['partition_tiles'];result={}
    for s in shape['samples']:
        if s['mode']!='receiver_sparse':continue
        tile=rank*per+s['offset'];raw=[r for r in s['receiver'] if r['tile']==tile]
        if len(raw)!=32 or any(r['observed_ns']<0 for r in raw):continue
        arrivals=[]
        for src in range(4):
            fs=[r for r in raw if r['source']==src]
            assert {r['fragment'] for r in fs}==set(range(8))
            arrivals.append(max(r['observed_ns'] for r in fs)/1000)
        result[(s['repeat']//per,tile)]=(arrivals,2*s['max_poll_cycle_ns']/1000)
    return result

def main(source,out,window=16):
    cfg=json.loads((source/'calibration-complete.json').read_text());assert cfg['completed']
    result=dict(calibration=str(source.resolve()),window=window,training_passes=[0,1],validation_passes=[2],
       score='positive median(last - max(other sources)) beyond two polling cycles; bounded per-destination window',
       sampling_limitation='Exploratory sampled labels; no low-perturbation mechanism claim. Performance is tested independently.',policies={},inputs={})
    for policy in ['remote_first','interleaved']:
        reports=[json.loads((source/f'b0-{policy}-instrumented-rank{r}.json').read_text()) for r in range(4)]
        assert all(r['completed'] for r in reports)
        plans=[]
        for si,(m,n,k) in enumerate(cfg['config']['shapes']):
            tn=(n+127)//128;tiles=m//128*tn
            with (source/f'mapping-{policy}-{m}-{n}-{k}.csv').open() as f:
                rows=list(csv.DictReader(f))
            rows=[{k:int(v) for k,v in r.items()} for r in rows]
            scores=[{} for _ in range(4)];valid=0;heldout=[];over=[]
            for target,report in enumerate(reports):
                sh=report['shapes'][si];assert (sh['M'],sh['N'],sh['K_global'])==(m,n,k)
                obs=observations(sh,target)
                for tile in range(target*tiles//4,(target+1)*tiles//4):
                    if not all((p,tile) in obs for p in [0,1]):continue
                    valid+=1
                    train=[obs[(p,tile)] for p in [0,1]]
                    gaps=[statistics.median(a[r]-max(a[s] for s in range(4) if s!=r) for a,e in train) for r in range(4)]
                    uncertainty=max(e for a,e in train)
                    for r in range(4):scores[r][tile]=gaps[r] if gaps[r]>uncertainty else 0.
                    if (2,tile) in obs:
                        a,e=obs[(2,tile)];winner=max(range(4),key=lambda r:a[r])
                        if a[winner]-max(a[r] for r in range(4) if r!=winner)>e:
                            heldout.append(max(range(4),key=lambda r:gaps[r])==winner)
                off={s['repeat']:s['forward_us'] for s in sh['samples'] if s['mode']=='off'}
                over += [100*(s['forward_us']/off[s['repeat']]-1) for s in sh['samples'] if s['mode']=='receiver_sparse']
            maps=[];coords=[];changed=[]
            for rank in range(4):
                rr=sorted([dict(r,tn=tn) for r in rows if r['rank']==rank],key=lambda r:r['index'])
                mapping=permute(rr,scores[rank],window)
                maps.append(mapping);coords.append([rr[j]['m']*tn+rr[j]['n'] for j in mapping])
                changed.append(sum(i!=j for i,j in enumerate(mapping)))
            assert any(changed), (policy,m,n,k,'No supported nonidentity arrival ordering')
            plans.append(dict(M=m,N=n,K_global=k,K_local=k//4,tiles=tiles,maps=maps,coords=coords,
              changed_per_rank=changed,training_valid_tiles=valid,total_tiles=tiles,
              heldout_resolved_tiles=len(heldout),heldout_last_source_accuracy=statistics.mean(heldout) if heldout else None,
              sampled_rank_median_overhead_percent=statistics.median(over)))
        result['policies'][{'remote_first':'remote_arrival','interleaved':'interleaved_arrival'}[policy]]=dict(base=policy,shapes=plans)
    for p in [*source.glob('b0-*-rank*.json'),*source.glob('mapping-*.csv'),source/'calibration-complete.json']:
        result['inputs'][str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    out.write_text(json.dumps(result,indent=2)+'\n')
    for p,v in result['policies'].items():
        for s in v['shapes']:print(p,s['M'],s['N'],s['K_global'],s['changed_per_rank'],s['heldout_last_source_accuracy'])
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('source',type=Path);a.add_argument('out',type=Path);a.add_argument('--window',type=int,default=16);v=a.parse_args();main(v.source,v.out,v.window)
