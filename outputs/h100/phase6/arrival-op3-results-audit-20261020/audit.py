"""Read-only CPU audit of returned op3 evidence. Never load the remote CUDA libraries."""
import contextlib
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO/'tools/h100/phase6'))
import arrival_op3_bench as bench
import arrival_op3_fullstep as full

ROOT = REPO/'logs/h100/phase6/arrival-op3-2x-50steps-20261020-192646-582458'
OUT = Path(__file__).resolve().parent
hashes = {}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    hashes[str(path.relative_to(REPO))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def verify_files(root, files):
    for name, digest in files.items():
        path = root/name
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        assert actual == digest, str(path)
        hashes[str(path.relative_to(REPO))] = actual


def describe(values):
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                mean=statistics.mean(values), maximum=max(values)) if values else dict(count=0)


def offline_operator(root):
    plan = read(root/'plan.json')
    assert plan['protocol'] == bench.PROTOCOL and plan['roles'] == bench.ROLES
    assert plan['schedule'] == bench.schedule(6)
    verify_files(root, plan['source_files'])
    # The returned archive contains declarations, not the actual remote .so/build.
    # Check consistency across stages without claiming binary revalidation.
    assert plan['build_identity'] == training_plan['build_identity']
    assert plan['libraries'] == training_plan['libraries']
    return plan


state = read(ROOT/'batch-state.json')
operator_plan = read(ROOT/'operator/plan.json')
training_plan = read(ROOT/'fullstep/plan.json')
stored_operator = read(ROOT/'operator/report.json')
stored_training = read(ROOT/'fullstep/report.json')
stored_profile = read(ROOT/'operator/profile-summary.json')
stored_preflight = read(ROOT/'fullstep/preflight.json')
verify_files(ROOT/'fullstep', training_plan['source_files'])
verify_files(ROOT/'fullstep', stored_training['evidence_files'])
verify_files(ROOT/'operator', stored_operator['evidence_files'])
for name in ('arrival_op3_bench.py', 'arrival_op3_fullstep.py'):
    path = REPO/'tools/h100/phase6'/name
    hashes[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()

# The maintained strict reporters validate every returned rank, input identity,
# loss trajectory, mode, codec gate and rank-max timing. Capture their writes.
with contextlib.redirect_stdout(io.StringIO()), patch.object(full, 'write'):
    rebuilt_training = full.report(ROOT/'fullstep', offline=True)
assert rebuilt_training == stored_training
with contextlib.redirect_stdout(io.StringIO()), patch.object(bench, 'verify', side_effect=offline_operator), \
        patch.object(bench, 'write'):
    rebuilt_operator = bench.report(ROOT/'operator')
    rebuilt_profile = bench.profile_summary(ROOT/'operator')
assert rebuilt_operator == stored_operator
for key, value in rebuilt_profile.items():
    if key != 'profiles':
        assert value == stored_profile[key], key
for key, value in rebuilt_profile['profiles'].items():
    old = dict(stored_profile['profiles'][key])
    old['trace'] = value['trace']  # Only the local archive prefix changes.
    assert value == old, key

training_rows = []
windows = []
for job in full.jobs(training_plan, 'timing'):
    rows = [read(ROOT/'fullstep/results'/job['directory']/f'rank{rank}.json') for rank in range(8)]
    training_rows.extend(rows)
    windows.append(dict(job, ms=max(r['elapsed_seconds'] for r in rows)*1000/50))
assert len(training_rows) == 576 and len(windows) == 72
aggregate = {}
for arm in bench.ARMS:
    values = [w['ms'] for w in windows if w['arm'] == arm]
    rows = [r for r in training_rows if r['rank'] == 0 and r['op3_arm'] == arm]
    aggregate[arm] = dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
        tokens_per_second=8192000/statistics.mean(values),
        mean_loss=statistics.mean(v for r in rows for v in r['losses']),
        mean_final_loss=statistics.mean(r['losses'][-1] for r in rows),
        max_window_over_round_median=max(
            w['ms']/statistics.median([x['ms'] for x in windows
                if x['arm'] == arm and x['trial'] == w['trial'] and x['round'] == w['round']])
            for w in windows if w['arm'] == arm))
assert all(r['losses'] == training_rows[0]['losses'] for r in training_rows)
assert all(r['skips'] == r['fallback_calls'] == 0 for r in training_rows)

smoke = {}
for arm in bench.ARMS:
    rows = [read(ROOT/f'fullstep/results/smoke-{arm}/rank{rank}.json') for rank in range(8)]
    checks = [c for r in rows for c in r['forward_checks'].values()]
    smoke[arm] = dict(checks=len(checks),
        max_relative_l2=max(c['relative_l2'] for c in checks),
        max_codec_relative_l2=max(c['protocol_relative_l2'] for c in checks))

operator_rows = [read(ROOT/'operator/results'/job['directory']/f'rank{rank}.json')
                 for job in operator_plan['schedule'] for rank in range(8)]
operator_checks = [c for r in operator_rows for role in r['roles'].values() for c in role['checks']]
assert len(operator_rows) == 144 and len(operator_checks) == 3456
assert all(c['correctness']['passed'] for c in operator_checks)

# Independently associate each of the four trace GEMMs with its own diagnostic
# invocation. This is stronger than merely finding overlap somewhere in a trace.
trace_calls = []
for arm in bench.ARMS:
    for rank in range(8):
        row = read(ROOT/f'operator/results/profile-{arm}/rank{rank}.json')
        for role in bench.ROLES:
            path = ROOT/f'operator/results/profile-{arm}/rank{rank}-{role}-profile.json.gz'
            data = path.read_bytes()
            hashes[str(path.relative_to(REPO))] = hashlib.sha256(data).hexdigest()
            events = json.loads(gzip.decompress(data))['traceEvents']
            kernels = sorted([e for e in events if e.get('cat') == 'kernel'], key=lambda e: e['ts'])
            gemms = [e for e in kernels if 'flux_bf16' in e['name'] and 'reducescatter' in e['name']]
            assert len(gemms) == 4
            diagnostics = row['roles'][role]['diagnostics']
            assert len(diagnostics) == (0 if arm == 'A' else 4)
            for i, gemm in enumerate(gemms):
                stop = gemms[i+1]['ts'] if i < 3 else math.inf
                selected = [e for e in kernels if gemm['ts'] <= e['ts'] < stop]
                tries = [e for e in selected if 'taco_op3_try_decode' in e['name']]
                tails = [e for e in selected if 'taco_op3_tail_decode' in e['name']]
                baseline_decoders = [e for e in selected if 'taco_decode_ring_kernel' in e['name']]
                barriers = [e for e in selected if 'CudaIpcBarrierAllKernel' in e['name']]
                assert len(tries) == (2 if arm == 'O' else 0)
                assert len(tails) == (0 if arm == 'A' else 1)
                assert len(baseline_decoders) == (1 if arm == 'A' else 0)
                assert len(barriers) == (2 if arm == 'A' else 1)
                overlaps = [e for e in tries if e['args']['stream'] != gemm['args']['stream'] and
                            max(e['ts'], gemm['ts']) <
                            min(e['ts']+e['dur'], gemm['ts']+gemm['dur'])]
                call = dict(arm=arm, rank=rank, role=role, invocation=i, gemm_us=gemm['dur'],
                            try_sum_us=sum(e['dur'] for e in tries), overlap_pairs=len(overlaps),
                            decoder_us=(tails or baseline_decoders)[0]['dur'])
                if arm != 'A':
                    d = diagnostics[i]
                    assert d['invocation'] == i and d['epoch'] == 125+i
                    useful = d['try_fragments'] > 0 and 0 < d['first_decode_complete_ns'] < d['last_local_publish_ns']
                    assert useful == d['useful_overlap']
                    call.update(try_fragments=d['try_fragments'], useful=useful,
                                same_call_useful_overlap=useful and bool(overlaps))
                    if useful:
                        call['first_complete_lead_us'] = (d['last_local_publish_ns']-d['first_decode_complete_ns'])/1000
                    if tries:
                        call['try_end_after_gemm_us'] = max(e['ts']+e['dur'] for e in tries)-(gemm['ts']+gemm['dur'])
                trace_calls.append(call)
profiles = {}
for role in bench.ROLES:
    calls = [c for c in trace_calls if c['arm'] == 'O' and c['role'] == role]
    profiles[role] = dict(calls=len(calls), useful_calls=sum(c['useful'] for c in calls),
        same_call_useful_overlap=sum(c['same_call_useful_overlap'] for c in calls),
        overlap_pairs=sum(c['overlap_pairs'] for c in calls),
        no_try_consumption=sum(c['try_fragments'] == 0 for c in calls),
        all_fragments_in_try=sum(c['try_fragments'] == 512 for c in calls),
        fragment_counts=sorted({c['try_fragments'] for c in calls}),
        useful_first_complete_lead_us=describe([c['first_complete_lead_us'] for c in calls if c['useful']]),
        positive_try_end_after_gemm_us=describe([c['try_end_after_gemm_us'] for c in calls
                                               if c['try_end_after_gemm_us'] > 0]),
        profiled_kernel_medians={arm: {
            field: statistics.median(c[field] for c in trace_calls if c['role'] == role and c['arm'] == arm)
            for field in ('gemm_us', 'try_sum_us', 'decoder_us')}
            for arm in bench.ARMS})
assert sum(v['useful_calls'] for v in profiles.values()) == 25
assert sum(v['same_call_useful_overlap'] for v in profiles.values()) == 25

source_hashes = operator_plan['build_identity']['source_files']
arms = {arm: {p.split('/'+arm+'/', 1)[1]: h for p, h in source_hashes.items() if '/'+arm+'/' in p}
        for arm in ('A', 'O')}
changed = [p for p in arms['A'] if arms['A'][p] != arms['O'].get(p)]
added = sorted(set(arms['O'])-set(arms['A']))
result = dict(completed=True, gpu_executed=False, source_batch=str(ROOT.relative_to(REPO)),
    cached_fullstep_report_exact_match=True, cached_operator_report_exact_match=True,
    cached_profile_summary_match_except_relocated_trace_paths=True,
    batch_state=state, fullstep=stored_training['comparisons'], aggregate=aggregate,
    descriptive_O_minus_A_ms=aggregate['O']['mean_ms']-aggregate['A']['mean_ms'],
    descriptive_O_over_A_increase_percent=100*(aggregate['O']['mean_ms']/aggregate['A']['mean_ms']-1),
    formal_optimizer_steps=3600, training_rank_records=576, smoke_rank_records=24,
    smoke=smoke, all_training_loss_trajectories_identical=True,
    micro_windows=18, micro_rank_records=144, micro_correctness_checks=3456,
    timed_distributed_operator_calls=18*2*200, profile_rank_records=24, trace_files=48,
    profile_calls=profiles, trace_calls=trace_calls,
    registered_source_difference=dict(A_count=len(arms['A']), O_count=len(arms['O']), changed=changed, added=added),
    limitations=['Remote compiled build and libraries were not returned; source/library identities are checked against declarations, not loaded binary bytes.',
                 'Profiles are separate diagnostic operator runs, not full-training traces or uninstrumented stage timings.',
                 'No input evidence was rewritten; strict reporter writes were intercepted in memory.'],
    inputs_sha256=hashes)
(OUT/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('inputs_sha256','trace_calls','batch_state','fullstep')},
                 ensure_ascii=False, indent=2))
