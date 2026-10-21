"""Read-only audit of the returned standalone TACO baseline batch; no GPU."""
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import tempfile

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
sys.path.insert(0, str(REPO/'tools/h100/phase6'))
import baseline_report as reporter
import baseline_protocol as routing
from common import read, write, sha

BATCH = REPO/'logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672'
OLD = REPO/'logs/h100/phase6/ablation6-then7-3x-50steps-20261019-162154-480841/ablation6'
N, F, A, B, C = routing.POLICIES
names = {N:'Megatron', F:'Flux', A:'A', B:'B', C:'C'}


def remap(values):
    return {names[k]: v for k,v in values.items()}


def main():
    batch = read(BATCH/'batch-state.json')
    assert batch['status']=='completed'
    audit = dict(protocol='baseline-taco-readonly-results-audit-v1',batch=str(BATCH.relative_to(REPO)),
                 gpu_used=False,comparisons={},checks={},counts=defaultdict(int))
    values=defaultdict(list); losses=defaultdict(list); gradients=defaultdict(list)
    per_policy_checks=defaultdict(list); codec_checks=[]; all_reports=[]
    initial_hashes={}; input_identity=None
    for trial in ('compare1','compare2','compare3'):
        print('Audit',trial,flush=True)
        root=BATCH/trial; info=read(root/'submission.json'); original=read(root/'report.json')
        assert sha(root/'report.json')==batch['comparisons'][trial]['report_sha256']
        for file in ('submission.json','preflight.json','report.json','report.md','all-state.json'):
            initial_hashes[str((root/file).relative_to(BATCH))]=sha(root/file)
        controllers={p.name: sha(p)==sha(REPO/'tools/h100/phase6'/p.name)
                     for p in (root/'provenance/controllers').glob('*.py')}
        assert all(controllers.values()),'Current controller source differs from frozen source'
        # Raw inputs are symlinked. All functions which write output write into a
        # fresh temporary directory, never through a link to an original report.
        with tempfile.TemporaryDirectory(prefix='taco-baseline-readonly-') as tmp:
            scratch=Path(tmp)
            for p in root.iterdir():
                if p.name not in ('preflight.json','report.json','report.md'):
                    (scratch/p.name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
            print(trial,'recomputing preflight',flush=True)
            reporter.preflight(scratch,offline=True)
            gate=read(scratch/'preflight.json')
            assert gate==read(root/'preflight.json'),'Preflight recomputation differs'
            target=OUT/trial; target.mkdir(exist_ok=True)
            write(target/'preflight.recomputed.json',gate)
            print(trial,'preflight matched; recomputing report',flush=True)
            # JSON object equality has been checked above. Preserve the original
            # serialization for its evidence hash: file traversal order may differ
            # after transfer, while the file->SHA mapping has identical content.
            shutil.copy2(root/'preflight.json',scratch/'preflight.json')
            reporter.report(scratch,offline=True)
            report=read(scratch/'report.json')
            expected=dict(original,offline=True)
            write(target/'report.recomputed.json',report)
            if report!=expected:
                print('Changed top-level report fields:',[k for k in set(report)|set(expected)
                      if report.get(k)!=expected.get(k)],flush=True)
            assert report==expected,'Report recomputation differs beyond offline flag'
        audit['checks'][trial]=dict(preflight_equal=True,report_equal_except_offline=True,
            current_controller_hashes=controllers,raw_report_sha256=sha(root/'report.json'))
        if input_identity is None:input_identity=original['measurement_identity']
        else:assert original['measurement_identity']==input_identity
        summary=original['rounds']['combined']
        result=dict(run_id=info['run_id'],mean_ms=remap(summary['mean_ms']),
            mean_loss=remap(summary['mean_loss']),final_loss=remap(summary['mean_final_loss']),
            comparisons={names[k.split('/')[0]]+'/'+names[k.split('/')[1]]:
                {f:v[f] for f in ('percent','ci95')} for k,v in summary['comparisons'].items()},
            sensitivity={},slowest_ratio_by_round={})
        for rep in ('1','2'):
            s=original['rounds'][rep]
            result['slowest_ratio_by_round'][rep]=max(max(v)/statistics.median(v) for v in s['ms_per_step'].values())
            result['sensitivity'][rep]={k:original['sensitivity'][rep][k] for k in ('remaining_blocks','excluded')}
        for job in info['schedule']:
            folder=root/'results'/job['directory']; records=[read(folder/f'rank{rank}.json') for rank in range(8)]
            p=job['policy']; elapsed=max(r['elapsed_seconds'] for r in records)*1000/50
            values[p].append(elapsed); losses[p].append(records[0]['losses']); gradients[p].append(records[0]['grad_norms'])
            assert all(r['losses']==records[0]['losses'] for r in records)
            assert all(len(r['losses'])==r['timed_steps']==50 for r in records)
            audit['counts']['timing_windows']+=1; audit['counts']['timing_rank_records']+=len(records)
        for p in routing.POLICIES:
            for mode in ('smoke','profile'):
                folder=root/'results'/f'{mode}-{p}'
                rows=[read(folder/f'rank{rank}.json') for rank in range(8)]
                audit['counts'][mode+'_windows']+=1; audit['counts'][mode+'_rank_records']+=8
                for row in rows:
                    if mode=='smoke':
                        per_policy_checks[p].extend(row['forward_checks'].values())
                        if routing.is_tensor(p):codec_checks.extend(row['codec_validation']['cases'])
                if mode=='profile':
                    trace_count=len(list(folder.glob('rank*-profile.json.gz')))
                    assert trace_count==8
                    audit['counts']['profile_traces']+=trace_count
                    if routing.is_tensor(p):
                        audit['counts']['baseline_profile_traces']+=trace_count
                        audit['counts']['upstream_compress_kernel_calls_profile']+=trace_count*256
                        audit['counts']['upstream_decompress_kernel_calls_profile']+=trace_count*256
        old=read(OLD/trial/'submission.json')
        abc={p:info['variants'][p]==old['variants'][p] for p in (A,B,C)}
        assert all(abc.values())
        assert sha(root/'scripts/selection-plan.json')==sha(OLD/trial/'scripts/selection-plan.json')
        result['abc_variants_equal_old']=remap(abc)
        result['selection_plan_equal_old']=True
        audit['comparisons'][trial]=result;all_reports.append(original)
        print(trial,'strict recomputation matched',flush=True)
    combined={}
    for p in routing.POLICIES:
        v=values[p]; trajectories=losses[p]; checks=per_policy_checks[p]
        assert len(v)==30 and all(len(t)==50 for t in trajectories)
        assert all(t==trajectories[0] for t in trajectories)
        combined[names[p]]=dict(mean_ms=statistics.mean(v),median_ms=statistics.median(v),
            tokens_per_second=8192000/statistics.mean(v),mean_loss=statistics.mean(map(statistics.mean,trajectories)),
            final_loss=statistics.mean(t[-1] for t in trajectories),
            min_recorded_loss=min(trajectories[0]),min_loss_recorded_step=trajectories[0].index(min(trajectories[0]))+1,
            last_three_losses=trajectories[0][-3:],loss_trajectory=trajectories[0],
            all_30_loss_trajectories_identical=True,
            all_30_grad_trajectories_identical=all(t==gradients[p][0] for t in gradients[p]),
            forward_checks=len(checks),max_relative_l2=max(r['relative_l2'] for r in checks),
            max_protocol_relative_l2=max(r['protocol_relative_l2'] for r in checks))
    audit['combined']=combined
    audit['loss_equal_pairs']={'Flux/Megatron':losses[F][0]==losses[N][0],'A/B':losses[A][0]==losses[B][0]}
    audit['grad_equal_pairs']={'Flux/Megatron':gradients[F][0]==gradients[N][0],'A/B':gradients[A][0]==gradients[B][0]}
    audit['descriptive_reductions_percent']={names[p]+'/'+names[q]:100*(1-statistics.mean(values[p])/statistics.mean(values[q]))
                                           for p in routing.POLICIES for q in routing.POLICIES if p!=q}
    audit['codec_check_maxima']={k:max(r[k] for r in codec_checks) for k in ('relative_l2','protocol_relative_l2','packet_relative_l2')}
    audit['counts']['codec_preflight_cases']=len(codec_checks)
    audit['counts']['model_forward_checks']=sum(map(len,per_policy_checks.values()))
    audit['counts']['formal_optimizer_steps']=audit['counts']['timing_windows']*50
    audit['counts']=dict(audit['counts'])
    audit['codec_library']=all_reports[0]['codec_library']
    audit['returned_codec_binary_present']=(BATCH/'codec-build/libbaseline_taco.so').is_file()
    m=read(BATCH/'codec-build/manifest.json')
    audit['returned_build_sources_validated']={p:sha(BATCH/'codec-build'/p)==h for p,h in m['files'].items()
                                             if p!='libbaseline_taco.so'}
    assert all(audit['returned_build_sources_validated'].values())
    if audit['returned_codec_binary_present']:
        assert sha(BATCH/'codec-build/libbaseline_taco.so')==audit['codec_library']['sha256']
    audit['old_comparison']={}
    for p,oldp in ((F,'flux_tensor_taco_attn_mlp'),(N,'native_taco_attn_mlp'),(A,A),(B,B),(C,C)):
        prior=[read(OLD/t/'report.json')['rounds']['combined']['mean_ms'][oldp] for t in ('compare1','compare2','compare3')]
        old_mean=statistics.mean(prior);new_mean=combined[names[p]]['mean_ms']
        oldfolder=next((OLD/'compare1/results').glob('round1/window-*-'+oldp))
        oldrow=read(oldfolder/'rank0.json')
        audit['old_comparison'][names[p]]=dict(old_mean_ms=old_mean,new_mean_ms=new_mean,
            descriptive_drop_percent=100*(1-new_mean/old_mean),loss_trajectory_equal=oldrow['losses']==losses[p][0],
            old_final_loss=oldrow['losses'][-1])
    for path,digest in initial_hashes.items():assert sha(BATCH/path)==digest
    audit['original_metadata_unchanged']=True
    audit['evidence_sha256']=initial_hashes
    audit['limitations']=['No GPU execution during audit; missing binary was not rehashed or loaded.',
        'Old/new batch comparisons are descriptive and unpaired.',
        'Fixed synthetic minibatches and short trajectories do not establish convergence equivalence.',
        'Performance confidence intervals are from each independent comparison, not an invented pooled trial.']
    write(OUT/'audit.json',audit)
    print(json.dumps(dict(counts=audit['counts'],combined={k:{f:v[f] for f in ('mean_ms','mean_loss','final_loss','max_relative_l2','max_protocol_relative_l2')}
                     for k,v in combined.items()},codec_check_maxima=audit['codec_check_maxima']),indent=2),flush=True)


if __name__=='__main__':main()
