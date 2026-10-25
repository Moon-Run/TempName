"""MA6 schedule, frozen-worker, full-model evidence and GPU-trace parser tests.

Synthetic times below test validation only; they are not GPU performance data.
"""
import argparse
import ast
from collections import Counter
import contextlib
import copy
import io
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ma6_bench as bench
import ma6_fullstep as full
from test_ma2_cr2 import fixture_plan as cr2_plan, fixture_rows as cr2_rows
from test_ma4_f_harness import fingerprints


def plan_fixture(trials=1):
    plan=cr2_plan();plan.pop('target_percent')
    tuning=bench.validate_tuning(None)
    plan.update(protocol=full.PROTOCOL,trials=trials,blocks=12,
        trial_ids={str(i+1):f'trial{i+1}' for i in range(trials)},arms=list(bench.ARMS),
        schedule=full.expected_schedule(trials,12),target_ms=None,acceptance_rule=full.ACCEPTANCE_RULE,
        tuning=tuning,build_identity={'tuning':tuning},libraries={arm:{'files':{
            f'/synthetic/{arm}/{name}':arm+name for name in ('libflux_cuda.so','libflux_cuda_ths_op.so')}} for arm in bench.ARMS})
    return plan


def rows_fixture(plan,job,digest,means=None):
    rows=cr2_rows(plan,dict(job,arm='CR2'),digest);arm=job['arm']
    for row in rows:
        for key in list(row):
            if key.startswith('cr2_'):row['ma6_'+key[4:]]=row.pop(key)
        row.update(protocol=full.PROTOCOL,ma6_protocol=bench.SOURCE_PROTOCOL,ma6_arm=arm,
            ma6_native_tuning={name:bench.expected_tuning(plan,arm) for name in row['ma6_codec_state']},
            ma6_final_parameters_sha256='c'*64,ma6_consumer_state={},ma6_backward_state={})
        universe={name+suffix:dict(shape=[4096],dtype='torch.bfloat16') for name in row['target_modules'] for suffix in ('.weight','.bias')}
        universe.update({f'other.parameter{i}':dict(shape=[1],dtype='torch.bfloat16') for i in range(389-len(universe))})
        row['ma6_trainable_parameters']=universe
        row['ma6_parameter_trajectory']=fingerprints(universe) if job['mode']=='smoke' else []
        for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
            row[name]=dict(path=f'/synthetic/{arm}/{name}',sha256=arm+name)
        for i,(name,states) in enumerate(row['ma6_submission_state'].items()):
            calls=sum(s['successful_calls'] for s in states)
            row['ma6_consumer_state'][name]=dict(projection_calls=calls,bda_calls=calls,bias_parameter_id=100+i,
                native=[dict(s,bias_residual_calls=s['successful_calls']) for s in states])
            row['ma6_backward_state'][name]=dict(protocol=bench.ma6_adapter.PROTOCOL,arm=arm,
                forward_calls=calls,backward_calls=calls,baseline_backward_calls=0 if arm=='F4-gather' else calls,
                async_gather_calls=calls if arm=='F4-gather' else 0,wait_calls=calls if arm=='F4-gather' else 0,
                failed=False,active=False)
        row['elapsed_seconds']=(means or {'F4':501.,'F4-tile':499.,'F4-gather':500.})[arm]*len(row['losses'])/1000
    return rows


def profile_fixture(arm,overlap=True):
    tuning=bench.validate_tuning(None);events=[]
    def kernel(name,start,duration,external,corr,stream=7,grid=None,block=None):
        return dict(cat='kernel',name=name,ts=start,dur=duration,
            args={'External id':external,'correlation':corr,'stream':stream,
                  'grid':grid or [1,1,1],'block':block or [128,1,1]})
    def launch(start,corr,external):
        return dict(cat='cuda_runtime',name='cudaLaunchKernel',ts=start,dur=.1,tid=1,pid=1,
                    args={'correlation':corr,'External id':external})
    for i in range(256):
        role='attn' if i<128 else 'mlp';start=i*100;external=i;seq=5000+i
        events.extend([dict(cat='user_annotation',name=f'scope.{role}.candidate',ts=start,dur=30,tid=1,pid=1),
            dict(cat='cpu_op',name='ConsumerGemmRSFunction',ts=start+1,dur=20,tid=1,pid=1,
                 args={'External id':external,'Sequence number':seq}),
            dict(cat='cpu_op',name='FusedLayerNormAffineFunction',ts=start+25,dur=2,tid=1,pid=1,args={'External id':10000+i})])
        config=bench.expected_tuning({'tuning':tuning},arm)[role]
        nodes=[('flux_bf16_reducescatter',40,5,[512,1,1],[128,1,1]),
               ('CudaIpcBarrierAllKernel',45,2,[1,1,1],[128,1,1]),
               (bench.decoder_name(arm,role,tuning),47,5,[config['decoder_blocks'],1,1],[config['decoder_threads'],1,1])]
        for j,(name,offset,duration,grid,block) in enumerate(nodes):
            corr=i*10+j;events.extend([launch(start+2+j,corr,external),kernel(name,start+offset,duration,external,corr,grid=grid,block=block)])
        events.append(kernel('cuApplyLayerNorm',start+52,4,10000+i,90000+i))
        start=30000+i*100;external=20000+i;corr=40000+i*10
        events.append(dict(cat='cpu_op',name='ConsumerGemmRSFunctionBackward',ts=start,dur=25,tid=1,pid=1,
                           args={'External id':external,'Sequence number':seq}))
        gathered=arm=='F4-gather'
        starts=(55,52,74,80) if gathered and overlap else (52,62,84,90)
        entries=[('reduce_kernel',starts[0],8,7,4 if gathered else 2),
                 ('nccl_AllGather',starts[1],20,26,2 if gathered else 4),
                 ('dgrad_gemm',starts[2],5,7,8),('wgrad_gemm',starts[3],5,7,10)]
        for j,(name,offset,duration,stream,host) in enumerate(entries):
            events.extend([launch(start+host,corr+j,external),kernel(name,start+offset,duration,external,corr+j,stream)])
    return events


class ScheduleAndSources(unittest.TestCase):
    def test_twelve_blocks_have_all_six_orders_per_half(self):
        for trials in (1,2):
            jobs=full.expected_schedule(trials,12)
            self.assertEqual(len(jobs),36*trials)
            for trial in range(1,trials+1):
                selected=[j for j in jobs if j['trial']==trial]
                self.assertEqual(Counter(j['arm'] for j in selected),{a:12 for a in bench.ARMS})
                for half in (range(6),range(6,12)):
                    orders=[tuple(j['arm'] for j in selected if j['block']==b) for b in half]
                    self.assertEqual(set(orders),set(itertools.permutations(bench.ARMS)))
        for n in (0,8,13,True):
            with self.assertRaises(ValueError):bench.schedule(n)

    def test_generated_sources_keep_strong_checks_outside_timing(self):
        for source in (full.worker_source(),full.adapter_source(),full.taco_source()):ast.parse(source)
        self.assertIn("audit_parameters = MODE == 'smoke'",full.worker_source())
        self.assertIn("report['ma6_backward_state']",full.worker_source())
        self.assertIn("H100_MA6_ARM",full.adapter_source())
        self.assertNotIn('H100_MA4_F_ARM',full.adapter_source())

    def test_frozen_operator_has_no_unfrozen_import_dependency(self):
        with tempfile.TemporaryDirectory() as temp:
            scripts=Path(temp)/'operator/scripts';scripts.mkdir(parents=True)
            bench.freeze_helpers(scripts);shutil.copy2(bench.__file__,scripts/'bench.py')
            code='import sys;sys.path.insert(0,sys.argv[1]);import bench;assert bench.ARMS==("F4","F4-tile","F4-gather");assert len(bench.schedule())==36'
            process=subprocess.run([sys.executable,'-I','-c',code,str(scripts)],cwd=scripts,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)

    def test_fullstep_prepare_and_frozen_hash_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'fullstep';build=Path(temp)/'build';build.mkdir()
            bench.write(build/'selection-plan.json',cr2_plan()['selection'])
            args=argparse.Namespace(root=root,build=build,megatron=Path(temp)/'megatron',trials=1,blocks=12)
            libs={a:dict(root='/synthetic/'+a,files={}) for a in bench.ARMS}
            with patch.object(bench,'build_identity',return_value={'tuning':bench.validate_tuning(None)}),\
                 patch.object(bench,'libraries',return_value=libs),patch.object(full,'snapshot_megatron',return_value={}),\
                 contextlib.redirect_stdout(io.StringIO()):full.prepare(args)
            plan=full.verify(root,offline=True)
            self.assertEqual(len(plan['schedule']),36)
            self.assertIn('scripts/ma6_fullstep_f4.py',plan['source_files'])
            (root/'scripts/ma6_adapter.py').write_text('drift')
            with self.assertRaisesRegex(ValueError,'Frozen source changed'):full.verify(root,offline=True)

    def test_operator_prepare_freezes_three_arm_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'operator';build=Path(temp)/'build';build.mkdir()
            bench.write(build/'selection-plan.json',cr2_plan()['selection'])
            args=argparse.Namespace(root=root,build=build,blocks=12,warmup=1,iterations=2)
            libs={a:dict(root='/synthetic/'+a,files={}) for a in bench.ARMS}
            with patch.object(bench,'build_identity',return_value={'tuning':bench.validate_tuning(None)}),\
                 patch.object(bench,'libraries',return_value=libs),contextlib.redirect_stdout(io.StringIO()):
                bench.prepare(args)
            plan=bench.verify(root,offline=True)
            self.assertEqual(len(plan['schedule']),36)
            self.assertEqual(plan['arms'],list(bench.ARMS))
            self.assertIn('scripts/ma4_f_sources.py',plan['source_files'])
            (root/'scripts/ma4_f_adapter.py').write_text('drift')
            with self.assertRaisesRegex(ValueError,'Frozen consumer helper changed'):bench.verify(root,offline=True)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.plan=plan_fixture();bench.write(self.root/'plan.json',self.plan)
        self.digest=bench.sha(self.root/'plan.json')

    def test_all_arms_require_consumer_and_actual_backward_counts(self):
        for arm in bench.ARMS:
            job=dict(self.plan['schedule'][0],arm=arm,mode='timing')
            rows=rows_fixture(self.plan,job,self.digest)
            with patch.object(full,'read',side_effect=lambda p:rows[int(p.stem[4:])]):
                self.assertEqual(len(full.window(self.root,self.plan,job)),8)
                first=next(iter(rows[0]['ma6_backward_state'].values()));first['backward_calls']-=1
                with self.assertRaisesRegex(ValueError,'backward'):full.window(self.root,self.plan,job)

    def test_control_cannot_claim_gather_and_consumer_cannot_fallback(self):
        job=dict(self.plan['schedule'][0],arm='F4',mode='timing');rows=rows_fixture(self.plan,job,self.digest)
        next(iter(rows[0]['ma6_backward_state'].values()))['async_gather_calls']=240
        with patch.object(full,'read',side_effect=lambda p:rows[int(p.stem[4:])]):
            with self.assertRaises(ValueError):full.window(self.root,self.plan,job)
        rows=rows_fixture(self.plan,job,self.digest)
        next(iter(rows[0]['ma6_consumer_state'].values()))['native'][0]['bias_residual_calls']-=1
        with patch.object(full,'read',side_effect=lambda p:rows[int(p.stem[4:])]):
            with self.assertRaises(ValueError):full.window(self.root,self.plan,job)

    def test_every_smoke_parameter_is_compared(self):
        rows={a:rows_fixture(self.plan,dict(arm=a,mode='smoke',trial=0,block=-1,index=-1,directory='smoke-'+a),self.digest) for a in bench.ARMS}
        with patch.object(full,'verify',return_value=self.plan),patch.object(full,'window',side_effect=lambda root,plan,job:rows[job['arm']]),patch.object(full,'_preflight',return_value={'ok':True}):
            self.assertEqual(full.preflight(self.root),{'ok':True})
            next(iter(rows['F4-gather'][7]['ma6_parameter_trajectory'][1]['gradients'].values()))['sha256']='d'*64
            with self.assertRaisesRegex(ValueError,'gradients or updates'):full.preflight(self.root)

    def report(self,means,transform=None):
        def rows(root,plan,job):
            result=rows_fixture(plan,job,self.digest,means)
            if transform:transform(result,job)
            return result
        ref=rows_fixture(self.plan,dict(self.plan['schedule'][0],mode='timing'),self.digest,means)
        with patch.object(full,'verify',return_value=self.plan),patch.object(full,'window',side_effect=rows),\
             patch.object(full,'preflight',return_value=dict(files={},rank_identities=full.identity(ref))):
            return full.report(self.root,save=False)

    def test_each_candidate_is_accepted_independently_all_samples_kept(self):
        result=self.report({'F4':501.,'F4-tile':500.,'F4-gather':510.})
        self.assertEqual(result['formal_optimizer_steps'],1800)
        self.assertEqual(result['timing_rank_records'],288)
        self.assertEqual(result['increment_over_F4_validated'],{'F4-tile':True,'F4-gather':False})
        self.assertFalse(result['target_evaluated'])
        self.assertIn('F4-tile/F4-gather',full.report_markdown(result))

    def test_final_parameter_drift_is_rejected(self):
        def change(rows,job):
            if job['arm']=='F4-gather':rows[7]['ma6_final_parameters_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'model parameters differ'):
            self.report({'F4':501.,'F4-tile':500.,'F4-gather':499.},change)

    def test_each_independent_trial_must_pass(self):
        self.plan=plan_fixture(2)
        def change(rows,job):
            if job['trial']==2 and job['arm']=='F4-tile':
                for row in rows:row['elapsed_seconds']=510*50/1000
        result=self.report({'F4':501.,'F4-tile':500.,'F4-gather':499.},change)
        self.assertEqual(result['formal_optimizer_steps'],3600)
        self.assertEqual(result['increment_over_F4_validated'],{'F4-tile':False,'F4-gather':True})


class ProfileTests(unittest.TestCase):
    def test_all_native_paths_and_real_backward_intersections(self):
        for arm in bench.ARMS:
            result=full.trace_projection_records(profile_fixture(arm),arm,tuning=bench.validate_tuning(None))
            self.assertEqual(sum(r['projection_calls'] for r in result.values()),256)
            self.assertEqual(result['attn']['consumer_add_kernels'],0)
            backward=result['attn']['backward']
            self.assertEqual(backward['backward_calls'],128)
            self.assertEqual(backward['overlap_calls'],128 if arm=='F4-gather' else 0)
            self.assertEqual(backward['median_us']['local_span_us'],33 if arm=='F4-gather' else 43)

    def test_async_submission_without_overlap_is_a_valid_observation(self):
        result=full.trace_backward_records(profile_fixture('F4-gather',False),'F4-gather')
        self.assertEqual(result['attn']['overlap_calls'],0)
        self.assertTrue(result['attn']['gather_submitted_before_bias'])

    def test_wrong_kernel_or_block_cannot_pass(self):
        events=profile_fixture('F4-tile')
        decoder=next(e for e in events if e.get('cat')=='kernel' and 'ma6_attention' in e['name'])
        decoder['args']['block']=[128,1,1]
        with self.assertRaises(ValueError):full.trace_projection_records(events,'F4-tile',tuning=bench.validate_tuning(None))
        decoder['args']['block']=[256,1,1];decoder['name']='taco_ma4f_decode_consumer_kernel'
        with self.assertRaises(ValueError):full.trace_projection_records(events,'F4-tile',tuning=bench.validate_tuning(None))

    def test_async_counter_cannot_hide_original_launch_order(self):
        with self.assertRaisesRegex(ValueError,'not submitted before bias'):
            full.trace_backward_records(profile_fixture('F4'),'F4-gather')

    def test_missing_gather_and_early_dgrad_rejected(self):
        events=profile_fixture('F4-gather')
        events.remove(next(e for e in events if e.get('cat')=='kernel' and 'AllGather' in e['name']))
        with self.assertRaisesRegex(ValueError,'kernels changed'):full.trace_backward_records(events,'F4-gather')
        events=profile_fixture('F4-gather')
        next(e for e in events if e.get('name')=='dgrad_gemm')['ts']-=20
        with self.assertRaises(ValueError):full.trace_backward_records(events,'F4-gather')


class FrozenReadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.calls=[]
        def replay_report(root,offline=False,save=True):
            self.calls.append(('report',offline,save))
            if save:(Path(root)/'report.md').write_text('unexpected mutation')
            return {'completed':True}
        def replay_preflight(root,offline=False):
            self.calls.append(('preflight',offline))
            if not (Path(root)/'preflight.json').exists():
                (Path(root)/'preflight.json').write_text('{}')
            return {'completed':True}
        def replay_profile(root,offline=False):
            self.calls.append(('profile',offline))
            return {'completed':True,'profiles':{}}
        source=Path(full.__file__).read_text()
        branch=[node for node in ast.parse(source).body if isinstance(node,ast.If)
                and isinstance(node.test,ast.Name) and node.test.id=='_FROZEN'][-1]
        # Exercise the exact registered replay wrappers independently of raw GPU
        # evidence; isolated real frozen import/verify is covered separately.
        self.ns=dict(_FROZEN=True,_HERE=self.root/'scripts',Path=Path,argparse=argparse,json=json,
            require=bench.require,report=replay_report,preflight=replay_preflight,profile_summary=replay_profile)
        exec(compile(ast.get_source_segment(source,branch),str(full.__file__),'exec'),self.ns)

    def test_missing_gate_never_creates_evidence(self):
        for name in ('preflight','report','profile_summary'):
            with self.assertRaisesRegex(ValueError,'existing preflight'):
                self.ns[name](self.root)
        self.assertEqual(self.calls,[])
        self.assertEqual(list(self.root.iterdir()),[])

    def test_default_and_explicit_save_report_are_read_only(self):
        (self.root/'preflight.json').write_text('{}')
        (self.root/'report.md').write_text('original')
        before={p.name:p.read_bytes() for p in self.root.iterdir()}
        self.assertEqual(self.ns['report'](self.root),{'completed':True})
        self.assertEqual(self.ns['report'](self.root,save=True),{'completed':True})
        self.assertEqual(self.calls,[('report',True,False),('report',True,False)])
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})

    def test_cli_only_prints_offline_report_or_profile(self):
        (self.root/'preflight.json').write_text('{}')
        before={p.name:p.read_bytes() for p in self.root.iterdir()}
        for action in ('report','profile-report'):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.ns['main']([action,str(self.root),'--offline'])
            self.assertTrue(json.loads(output.getvalue())['completed'])
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})
        for argv in (['report',str(self.root)],['prepare',str(self.root)],['run',str(self.root)],
                     ['profile-report',str(self.root),'--offline','--save']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                self.ns['main'](argv)


if __name__=='__main__':unittest.main()
