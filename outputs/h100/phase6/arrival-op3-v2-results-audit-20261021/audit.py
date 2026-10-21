"""CPU audit of returned v2 evidence; never write to the source batch or use GPUs."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tools/h100/phase6'))
import arrival_op3_v2_bench as bench
import arrival_op3_v2_fullstep as full

ROOT = REPO / 'logs/h100/phase6/arrival-op3-v2-2x-50steps-20261020-235555-300968'
OUT = Path(__file__).resolve().parent
hashes = {}


def content(path):
    data = path.read_bytes()
    hashes[str(path.relative_to(REPO))] = hashlib.sha256(data).hexdigest()
    return data


def read(path):
    return json.loads(content(path))


def verify_files(root, files):
    for name, digest in files.items():
        assert hashlib.sha256(content(root / name)).hexdigest() == digest, name


def describe(values):
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                mean=statistics.mean(values), maximum=max(values)) if values else dict(count=0)


def offline_operator(root):
    plan = read(root / 'plan.json')
    assert plan['protocol'] == bench.PROTOCOL and plan['roles'] == bench.ROLES
    assert plan['arms'] == list(bench.ARMS) and plan['schedule'] == bench.schedule(6)
    verify_files(root, plan['source_files'])
    assert plan['build_identity'] == training_plan['build_identity']
    assert plan['libraries'] == training_plan['libraries']
    return plan


state = read(ROOT / 'batch-state.json')
operator_plan = read(ROOT / 'operator/plan.json')
training_plan = read(ROOT / 'fullstep/plan.json')
stored_operator = read(ROOT / 'operator/report.json')
stored_training = read(ROOT / 'fullstep/report.json')
stored_profile = read(ROOT / 'operator/profile-summary.json')
read(ROOT / 'fullstep/preflight.json')
controllers = read(ROOT / 'handoff/controllers.json')
for name, digest in controllers.items():
    path = REPO / name.removeprefix('/workspace/TempName/')
    assert hashlib.sha256(content(path)).hexdigest() == digest, name
verify_files(ROOT / 'fullstep', training_plan['source_files'])
verify_files(ROOT / 'fullstep', stored_training['evidence_files'])
verify_files(ROOT / 'operator', stored_operator['evidence_files'])

# Only absent remote build/library access is replaced. Raw record validation,
# source hashes, paired statistics and profile matching remain the strict reporters.
with patch.object(full, 'write', side_effect=AssertionError('Unexpected source write')):
    rebuilt_training = full.report(ROOT / 'fullstep', offline=True, save=False)
assert rebuilt_training == stored_training
with patch.object(bench, 'verify', side_effect=offline_operator), \
        patch.object(bench, 'write', side_effect=AssertionError('Unexpected source write')):
    rebuilt_operator = bench.report(ROOT / 'operator', save=False)
    rebuilt_profile = bench.profile_summary(ROOT / 'operator')
assert rebuilt_operator == stored_operator
for key, value in rebuilt_profile.items():
    if key != 'profiles':
        assert value == stored_profile[key], key
for key, value in rebuilt_profile['profiles'].items():
    old = dict(stored_profile['profiles'][key])
    old['trace'] = value['trace']  # The only difference allowed after relocation.
    assert value == old, key
assert state['status'] == 'completed'
for key in ('target_validated', 'arrival_benefit_validated', 'decoder_control_passed',
            'formal_optimizer_steps', 'acceptance_status', 'attribution_status'):
    assert state[key] == rebuilt_training[key], key
assert state['micro_savings_estimate'] == rebuilt_operator['micro_savings_estimate']
assert state['useful_early_consumption_validated'] == rebuilt_profile['useful_early_consumption_validated']
assert state['timing_windows'] == 96

training_rows, windows = [], []
for job in full.jobs(training_plan, 'timing'):
    rows = [read(ROOT / 'fullstep/results' / job['directory'] / f'rank{r}.json') for r in range(8)]
    training_rows.extend(rows)
    windows.append(dict(job, ms=max(r['elapsed_seconds'] for r in rows) * 1000 / 50))
assert len(training_rows) == 768 and len(windows) == 96
aggregate = {}
for arm in bench.ARMS:
    values = [w['ms'] for w in windows if w['arm'] == arm]
    rows = [r for r in training_rows if r['rank'] == 0 and r['op3_arm'] == arm]
    aggregate[arm] = dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
        tokens_per_second=8192000 / statistics.mean(values),
        mean_loss=statistics.mean(v for r in rows for v in r['losses']),
        mean_final_loss=statistics.mean(r['losses'][-1] for r in rows),
        max_window_over_round_median=max(
            w['ms'] / statistics.median([x['ms'] for x in windows
                if x['arm'] == arm and x['trial'] == w['trial'] and x['round'] == w['round']])
            for w in windows if w['arm'] == arm))
assert all(r['losses'] == training_rows[0]['losses'] for r in training_rows)
assert all(r['skips'] == r['fallback_calls'] == 0 for r in training_rows)
smoke = {}
for arm in bench.ARMS:
    rows = [read(ROOT / f'fullstep/results/smoke-{arm}/rank{r}.json') for r in range(8)]
    checks = [c for r in rows for c in r['forward_checks'].values()]
    smoke[arm] = dict(checks=len(checks), max_relative_l2=max(c['relative_l2'] for c in checks),
                     max_codec_relative_l2=max(c['protocol_relative_l2'] for c in checks))
operator_rows = [read(ROOT / 'operator/results' / job['directory'] / f'rank{r}.json')
                 for job in operator_plan['schedule'] for r in range(8)]
operator_checks = [c for r in operator_rows for role in r['roles'].values() for c in role['checks']]
assert len(operator_rows) == 192 and len(operator_checks) == 4608
assert all(c['correctness']['passed'] for c in operator_checks)

trace_calls = []
for arm in bench.ARMS:
    for rank in range(8):
        row = read(ROOT / f'operator/results/profile-{arm}/rank{rank}.json')
        for role in bench.ROLES:
            path = ROOT / f'operator/results/profile-{arm}/rank{rank}-{role}-profile.json.gz'
            events = json.loads(gzip.decompress(content(path)))['traceEvents']
            kernels = sorted([e for e in events if e.get('cat') == 'kernel'], key=lambda e: e['ts'])
            gemms = [e for e in kernels if 'flux_bf16' in e['name'] and 'reducescatter' in e['name']]
            tries = [e for e in kernels if 'taco_op3_v2_try_decode' in e['name']]
            stops = [e for e in kernels if 'taco_op3_v2_stop' in e['name']]
            decoder_name = {'A': 'taco_decode_ring_kernel(', 'D': 'taco_op3_v2_decode(',
                            'S': 'taco_op3_v2_tail_decode(', 'O': 'taco_op3_v2_tail_decode('}[arm]
            decoders = [e for e in kernels if decoder_name in e['name']]
            barriers = [e for e in kernels if 'CudaIpcBarrierAllKernel' in e['name']]
            assert len(gemms) == len(decoders) == 4
            assert len(tries) == (8 if arm == 'O' else 0)
            assert len(stops) == (4 if arm == 'O' else 0)
            assert len(barriers) == (8 if arm in ('A', 'D') else 4)
            diagnostics = row['roles'][role]['diagnostics']
            assert len(diagnostics) == (4 if arm in ('S', 'O') else 0)
            for i, g in enumerate(gemms):
                selected = tries[i*2:i*2+2]
                overlaps = [e for e in selected if e['args']['stream'] != g['args']['stream'] and
                            max(e['ts'], g['ts']) < min(e['ts']+e['dur'], g['ts']+g['dur'])]
                call = dict(arm=arm, rank=rank, role=role, invocation=i, gemm_us=g['dur'],
                            try_sum_us=sum(e['dur'] for e in selected), overlap_pairs=len(overlaps),
                            decoder_us=decoders[i]['dur'])
                if diagnostics:
                    d = diagnostics[i]
                    assert d['invocation'] == i and d['epoch'] == 125+i
                    useful = d['try_groups'] > 0 and 0 < d['first_decode_complete_ns'] < d['last_local_publish_ns']
                    assert useful == d['useful_overlap']
                    call.update(try_groups=d['try_groups'], groups_finished_after_stop=d['groups_finished_after_stop'],
                                useful=useful, same_call_useful_overlap=useful and bool(overlaps))
                    if useful:
                        call['first_complete_lead_us'] = (d['last_local_publish_ns']-d['first_decode_complete_ns']) / 1000
                    if selected:
                        end = max(e['ts']+e['dur'] for e in selected)
                        call.update(try_end_after_gemm_us=end-(g['ts']+g['dur']),
                                    try_end_after_stop_us=end-(stops[i]['ts']+stops[i]['dur']),
                                    tail_start_after_gemm_us=decoders[i]['ts']-(g['ts']+g['dur']))
                        assert decoders[i]['ts'] >= max(end, stops[i]['ts']+stops[i]['dur']) - .01
                trace_calls.append(call)
profiles = {}
for role in bench.ROLES:
    calls = [c for c in trace_calls if c['arm'] == 'O' and c['role'] == role]
    profiles[role] = dict(calls=len(calls), useful_calls=sum(c['same_call_useful_overlap'] for c in calls),
        overlap_calls=sum(c['overlap_pairs'] > 0 for c in calls), overlap_pairs=sum(c['overlap_pairs'] for c in calls),
        no_try_consumption=sum(c['try_groups'] == 0 for c in calls),
        reached_try_cap=sum(c['try_groups'] == 1024 for c in calls),
        try_groups=describe([c['try_groups'] for c in calls]),
        groups_finished_after_stop=describe([c['groups_finished_after_stop'] for c in calls]),
        useful_first_complete_lead_us=describe([c['first_complete_lead_us'] for c in calls if c['useful']]),
        positive_try_end_after_gemm_us=describe([c['try_end_after_gemm_us'] for c in calls if c['try_end_after_gemm_us'] > 0]),
        positive_try_end_after_stop_us=describe([c['try_end_after_stop_us'] for c in calls if c['try_end_after_stop_us'] > 0]),
        tail_start_after_gemm_us=describe([c['tail_start_after_gemm_us'] for c in calls]),
        profiled_kernel_medians={arm: {
            field: statistics.median(c[field] for c in trace_calls if c['role'] == role and c['arm'] == arm)
            for field in ('gemm_us', 'try_sum_us', 'decoder_us')} for arm in bench.ARMS})
assert {r: p['useful_calls'] for r, p in profiles.items()} == stored_profile['useful_and_overlapped_calls_by_role']
assert all(c.get('try_groups', 0) <= 1024 for c in trace_calls)
result = dict(completed=True, gpu_executed=False, source_batch=str(ROOT.relative_to(REPO)),
    cached_fullstep_report_exact_match=True, cached_operator_report_exact_match=True,
    cached_profile_summary_match_except_relocated_trace_paths=True, controllers_match=True,
    batch_state=state, fullstep=stored_training['comparisons'], aggregate=aggregate,
    descriptive_O_minus_A_ms=aggregate['O']['mean_ms']-aggregate['A']['mean_ms'],
    descriptive_O_over_A_increase_percent=100*(aggregate['O']['mean_ms']/aggregate['A']['mean_ms']-1),
    formal_optimizer_steps=4800, training_rank_records=768, smoke_rank_records=32, smoke=smoke,
    all_training_loss_trajectories_identical=True, micro_windows=24, micro_rank_records=192,
    micro_correctness_checks=4608, timed_distributed_operator_calls=24*2*200,
    profile_rank_records=32, trace_files=64, profile_calls=profiles, trace_calls=trace_calls,
    limitations=['Remote build, compiled libraries and Megatron snapshot were not returned; '
                 'their identities are checked against declarations, not the absent binary bytes.',
                 'Profiles are separate instrumented operator runs, not full-training stage measurements.',
                 'Source experiment evidence is read only. No GPU experiment was run by this audit.'],
    inputs_sha256=hashes)
(OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in
                 ('inputs_sha256', 'trace_calls', 'batch_state', 'fullstep')}, ensure_ascii=False, indent=2))
