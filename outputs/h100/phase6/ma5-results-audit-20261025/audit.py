"""Recompute returned MA5 reports and launch evidence without a GPU or raw-data edits."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tools/h100/phase6'))
import ma5_bench as bench
import ma5_fullstep as fullstep

BATCH = REPO / 'logs/h100/phase6/ma5-tile-block-1x-50steps-20261025-104210-136736'
CANDIDATE = 't256-b528-k32-s-1'
ROOT = BATCH / CANDIDATE


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


files = [BATCH / 'batch-state.json', BATCH / 'handoff/launch.json',
         BATCH / 'handoff/controllers.json', ROOT / 'fullstep/report.md'] + [
    ROOT / part / name for part in ('operator', 'fullstep')
    for name in ('plan.json', 'preflight.json', 'report.json', 'profile-summary.json')]
before = {str(p.relative_to(REPO)): sha(p) for p in files}
controllers = {}
for frozen, current in (
    ('operator/scripts/bench.py', 'ma5_bench.py'),
    ('fullstep/scripts/operator-controller.py', 'ma5_bench.py'),
    ('fullstep/scripts/fullstep-controller.py', 'ma5_fullstep.py'),
):
    digest = sha(ROOT / frozen)
    require(digest == sha(REPO / 'tools/h100/phase6' / current), 'Changed controller: ' + current)
    controllers[frozen] = digest
dependencies = read(BATCH / 'handoff/controllers.json')
for remote, digest in dependencies.items():
    require(remote.startswith('/workspace/TempName/'), 'Unexpected controller source root')
    local = REPO / remote.removeprefix('/workspace/TempName/')
    require(local.is_file() and sha(local) == digest, 'Changed dependency: ' + remote)

checks, recomputed = {}, {}
for name, callback in (
    ('operator/report.json', lambda: bench.report(ROOT / 'operator', save=False, offline=True)),
    ('fullstep/report.json', lambda: fullstep.report(ROOT / 'fullstep', save=False, offline=True)),
    ('operator/profile-summary.json', lambda: bench.profile_summary(ROOT / 'operator', offline=True)),
    ('fullstep/profile-summary.json', lambda: fullstep.profile_summary(ROOT / 'fullstep', offline=True)),
):
    result = callback()
    require(result == read(ROOT / name), 'Recomputation differs: ' + name)
    checks[name] = True
    recomputed[name] = result
    print('Recomputed identically:', name, flush=True)

state, launch = read(BATCH / 'batch-state.json'), read(BATCH / 'handoff/launch.json')
train, op = recomputed['fullstep/report.json'], recomputed['operator/report.json']
candidate = state['candidates'][CANDIDATE]
require(state['status'] == 'completed' and state['stage'] == 'done'
        and set(state['candidates']) == {CANDIDATE} and candidate['status'] == 'completed',
        'Incomplete batch or unexpected candidates')
require(state['trials'] == train['trials'] == launch['trials'] == 1
        and launch['sweep'] is False and not state['sweep_is_exploratory'], 'Unexpected sweep/trials')
tuning = dict(decoder_threads=256, decoder_blocks=528, gemm_sms=-1, gemm_tile_k=32)
require(train['tuning'] == op['tuning'] == candidate['tuning'] == tuning, 'Unexpected tuning')
require(state['formal_optimizer_steps'] == state['expected_formal_optimizer_steps']
        == train['formal_optimizer_steps'] == 800, 'Wrong formal step count')
require(state['expected_timing_windows'] == len(train['all_samples']) == 16
        and train['timing_rank_records'] == 128 and len(op['all_samples']) == 16, 'Missing windows')
require(len(recomputed['operator/profile-summary.json']['profiles']) == 32
        and len(recomputed['fullstep/profile-summary.json']['profiles']) == 16, 'Missing traces')
require(train['correctness_passed'] and train['identical_trajectories_within_pairs']
        and op['correctness_passed'] and op['bitwise_equal_within_pairs']
        and train['equivalence_pairs'] == op['equivalence_pairs'] == [['T', 'CR2']], 'Wrong numerical gate')
require(candidate['operator_correctness_passed'] and candidate['operator_execution_path_validated']
        and candidate['fullstep_execution_path_validated'], 'Incomplete execution gates')
require(candidate['report_sha256'] == sha(ROOT / 'fullstep/report.json')
        and candidate['comparisons'] == train['comparisons'], 'Batch/report mismatch')
for field, accepted in fullstep.aggregate_acceptance(train['comparisons']).items():
    require(candidate[field] == train[field] == accepted, 'Acceptance mismatch: ' + field)

medians = {}
for part in ('operator', 'fullstep'):
    profiles = recomputed[part + '/profile-summary.json']['profiles']
    medians[part] = {}
    for arm in bench.ARMS:
        medians[part][arm] = {}
        for role in bench.ROLES:
            rows = ([value[role] for key, value in profiles.items() if key.startswith(arm + '/')]
                    if part == 'fullstep' else [value for key, value in profiles.items()
                    if key.startswith(arm + '/') and key.endswith('/' + role)])
            require(len(rows) == 8, 'Missing rank profiles')
            medians[part][arm][role] = {key: statistics.median(row[key] for row in rows)
                                      for key in rows[0] if key.endswith('median_us')}

native, variability, profile_counts, actual_launches = {}, {}, {}, {}
profiles = recomputed['fullstep/profile-summary.json']['profiles']
for arm in bench.ARMS:
    samples = [s for s in train['all_samples'] if s['arm'] == arm]
    values = [s['ms_per_step'] for s in samples]
    variability[arm] = dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                           minimum_ms=min(values), maximum_ms=max(values))
    summary = dict(formal_rank_records=0, module_records=0, slot_records=0,
                   initializations=set(), successful_calls=set(), bound_k=set(), failed=set())
    for sample in samples:
        for rank in range(8):
            row = read(ROOT / 'fullstep/results' / sample['directory'] / f'rank{rank}.json')
            summary['formal_rank_records'] += 1
            for slots in row['ma5_submission_state'].values():
                summary['module_records'] += 1
                for slot in slots:
                    summary['slot_records'] += 1
                    for field in ('initializations', 'successful_calls', 'bound_k', 'failed'):
                        summary[field].add(slot[field])
    summary = {key: sorted(value) if isinstance(value, set) else value for key, value in summary.items()}
    require(summary == dict(formal_rank_records=64, module_records=4096, slot_records=8192,
            initializations=[1], successful_calls=[120], bound_k=[512, 2048], failed=[False]),
            'Unexpected native cache evidence: ' + arm)
    native[arm] = summary
    rows = [profiles[f'{arm}/rank{i}'][role] for i in range(8) for role in bench.ROLES]
    profile_counts[arm] = {field: sorted({row[field] for row in rows}) for field in
        ('projection_calls', 'barrier_calls', 'output_clones', 'output_slice_view_count', 'adapter_view_count')}
    require(profile_counts[arm] == dict(projection_calls=[128], barrier_calls=[128], output_clones=[0],
            output_slice_view_count=[0], adapter_view_count=[256]), 'Wrong projection execution counts')
    actual_launches[arm] = [json.loads(value) for value in sorted({json.dumps(item, sort_keys=True)
        for row in rows for item in row['launches']})]
    expected = dict(gemm_tile=[128, 128, 32], gemm_grid=[512, 1, 1], gemm_block=[128, 1, 1],
        configured_gemm_sms=-1, decoder_grid=[528 if arm == 'T' else 2048, 1, 1],
        decoder_block=[256 if arm == 'T' else 128, 1, 1])
    require(actual_launches[arm] == [expected], 'Actual launch differs from registered experiment')

wins = sum(next(s['ms_per_step'] for s in train['all_samples'] if s['block'] == block and s['arm'] == 'T')
           < next(s['ms_per_step'] for s in train['all_samples'] if s['block'] == block and s['arm'] == 'CR2')
           for block in range(8))
require(before == {str(p.relative_to(REPO)): sha(p) for p in files}, 'Original evidence changed')
audit = dict(completed=True, gpu_used=False, date='2026-10-25', batch=str(BATCH.relative_to(REPO)),
    candidate=CANDIDATE, tuning=tuning, recomputed_identically=checks,
    source_controllers_match=controllers, matched_local_dependencies=len(dependencies),
    original_reports_plans_gates_unchanged=True, evidence_sha256=before,
    formal_training=dict(windows=16, rank_records=128, optimizer_steps=800, candidate_faster_blocks=wins,
                         candidate_slower_blocks=8-wins),
    training_comparisons=train['comparisons'], numeric_summary=train['numeric_summary'],
    operator_comparisons=op['roles'], profile_rank_medians=medians,
    training_window_variability=variability, native_cache=native,
    training_profile_counts_per_rank_role=profile_counts, actual_training_launches=actual_launches,
    limitation='Offline raw records/trace verification only; server libraries are not reloaded or rehashed. '
               'Only the default decoder candidate was measured; no sweep, K64 or SM128 result is inferred.')
(Path(__file__).parent / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
print('Audit complete: original results unchanged; target_validated =', train['target_validated'], flush=True)
