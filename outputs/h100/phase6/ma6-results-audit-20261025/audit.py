"""Replay MA6-v1 using its frozen controllers, without GPU or evidence edits.

Default execution is read-only. --out may save a new audit to a fresh path;
the original 2026-10-25 audit.json/log remain historical evidence.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[4]
BATCH = REPO/'logs/h100/phase6/ma6-f4-tile-gather-1x-50steps-20261025-160359-496883'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, help='Optional fresh JSON output; never overwrite evidence')
args = parser.parse_args()
if args.out is not None and args.out.exists():
    parser.error('--out must be a fresh path')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


files = [BATCH/'batch-state.json', BATCH/'handoff/launch.json', BATCH/'handoff/controllers.json',
         BATCH/'fullstep/report.md'] + [BATCH/part/name for part in ('operator', 'fullstep')
    for name in ('plan.json', 'preflight.json', 'report.json', 'profile-summary.json')]
before = {str(p.relative_to(REPO)): sha(p) for p in files}
controllers = {}
dependencies = read(BATCH/'handoff/controllers.json')
for frozen, current in (('operator/scripts/bench.py', 'ma6_bench.py'),
                       ('fullstep/scripts/operator-controller.py', 'ma6_bench.py'),
                       ('fullstep/scripts/fullstep-controller.py', 'ma6_fullstep.py')):
    digest = sha(BATCH/frozen)
    remote = '/workspace/TempName/tools/h100/phase6/'+current
    require(digest == dependencies[remote], 'Changed frozen controller: '+frozen)
    controllers[frozen] = digest
frozen_files = {}
for part in ('operator', 'fullstep'):
    for name, digest in read(BATCH/part/'plan.json')['source_files'].items():
        path = BATCH/part/name
        require(path.resolve().is_relative_to((BATCH/part/'scripts').resolve()),
                'Unexpected frozen script path: '+name)
        require(path.is_file() and sha(path) == digest, 'Changed frozen dependency: '+str(path))
        frozen_files[part+'/'+name] = digest

# A fresh process imports exclusively from the returned v1 snapshot. Using the
# current checkout here would mix v2 schemas/defaults with v1 measurement data.
scripts = BATCH/'fullstep/scripts'
sys.path[:] = [str(scripts), *[p for p in sys.path
    if p and not Path(p).resolve().is_relative_to(REPO)]]
spec = importlib.util.spec_from_file_location('_ma6_v1_audit', scripts/'fullstep-controller.py')
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
bench = full.bench
require(Path(bench.__file__).resolve().parent == scripts.resolve(), 'Unfrozen benchmark import')
require(full._FROZEN and bench.SOURCE_PROTOCOL == 'h100-ma6-f4-v1', 'Wrong replay protocol')
print('Frozen v1 controllers and', len(frozen_files), 'snapshot files verified.', flush=True)

# Add a diagnostic while the original parser already holds each trace in
# memory. Return its original result unchanged; no profile is reparsed or saved.
gpu_spans = {arm: [] for arm in bench.ARMS}
original_parser = full.trace_projection_records


def capture_span(events, arm, tuning=None):
    result = original_parser(events, arm, tuning=tuning)
    device = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    require(device, 'Empty GPU trace')
    gpu_spans[arm].append((max(e['ts']+e.get('dur', 0) for e in device)
                          - min(e['ts'] for e in device))/1000)
    return result


checks, recomputed = {}, {}
full.trace_projection_records = capture_span
try:
    for name, callback in (
        ('operator/report.json', lambda: bench.report(BATCH/'operator', save=False, offline=True)),
        ('fullstep/report.json', lambda: full.report(BATCH/'fullstep', save=False, offline=True)),
        ('operator/profile-summary.json', lambda: bench.profile_summary(BATCH/'operator', offline=True)),
        ('fullstep/profile-summary.json', lambda: full.profile_summary(BATCH/'fullstep', offline=True)),
    ):
        result = callback()
        require(result == read(BATCH/name), 'Recomputation differs: '+name)
        checks[name] = True
        recomputed[name] = result
        print('Recomputed identically:', name, flush=True)
finally:
    full.trace_projection_records = original_parser

state, launch = read(BATCH/'batch-state.json'), read(BATCH/'handoff/launch.json')
train, op = recomputed['fullstep/report.json'], recomputed['operator/report.json']
profiles = recomputed['fullstep/profile-summary.json']['profiles']
require(state['status'] == 'completed' and state['stage'] == 'done'
        and state['arms'] == list(bench.ARMS), 'Incomplete batch')
require(state['trials'] == launch['trials'] == train['trials'] == 1, 'Wrong trial count')
require(state['formal_optimizer_steps'] == state['expected_formal_optimizer_steps']
        == train['formal_optimizer_steps'] == 1800, 'Missing formal steps')
require(state['timing_windows'] == state['expected_timing_windows'] == len(train['all_samples']) == 36
        and train['timing_rank_records'] == 288 and len(op['all_samples']) == 36, 'Missing windows')
require(state['tuning'] == train['tuning'] == bench.validate_tuning(None), 'Wrong measured tile')
require(len(profiles) == 24 and len(recomputed['operator/profile-summary.json']['profiles']) == 48,
        'Missing profile coverage')
require(train['correctness_passed'] and train['identical_trajectories_within_pairs']
        and train['consumer_parameter_gradients_validated'] and train['backward_schedule_validated']
        and op['correctness_passed'] and op['bitwise_equal_within_pairs'] and op['consumer_gradients_validated'],
        'Missing numerical gate')
require(state['operator_correctness_validated'] and state['operator_execution_path_validated']
        and state['fullstep_execution_path_validated'], 'Missing execution gate')
for field, accepted in full.aggregate_acceptance(train['comparisons']).items():
    require(state[field] == train[field] == accepted, 'Acceptance mismatch: '+field)
require(not state['target_evaluated'] and not train['target_evaluated'], 'C/R target unexpectedly evaluated')

blocks, wins = [], {}
for block in range(12):
    selected = [s for s in train['all_samples'] if s['block'] == block]
    require(len(selected) == 3 and {s['arm'] for s in selected} == set(bench.ARMS), 'Incomplete paired block')
    values = {s['arm']: s['ms_per_step'] for s in selected}
    blocks.append(dict(block=block, order=[s['arm'] for s in selected], ms_per_step=values))
for candidate, control in bench.COMPARISONS:
    outcomes = Counter('faster' if b['ms_per_step'][candidate] < b['ms_per_step'][control] else
                       'slower' if b['ms_per_step'][candidate] > b['ms_per_step'][control] else 'tied' for b in blocks)
    wins[candidate+'/'+control] = {key: outcomes[key] for key in ('faster', 'slower', 'tied')}

medians, overlaps, native, variability, launches = {}, {}, {}, {}, {}
for arm in bench.ARMS:
    medians[arm], overlaps[arm], launches[arm] = {}, {}, {}
    for role in bench.ROLES:
        rows = [profiles[f'{arm}/rank{rank}'][role] for rank in range(8)]
        medians[arm][role] = dict(forward={k: statistics.median(r[k] for r in rows)
            for k in rows[0] if k.endswith('_us')},
            backward={k: statistics.median(r['backward']['median_us'][k] for r in rows)
                      for k in rows[0]['backward']['median_us']})
        overlaps[arm][role] = dict(calls=sum(r['backward']['backward_calls'] for r in rows),
            overlap_calls=sum(r['backward']['overlap_calls'] for r in rows),
            per_rank_overlap_calls=[r['backward']['overlap_calls'] for r in rows])
        launches[arm][role] = rows[0]['launch']
        require(all(r['launch'] == rows[0]['launch'] and r['projection_calls'] == r['publication_barriers']
                    == r['layernorm_successors'] == r['backward']['backward_calls'] == 128
                    and r['consumer_add_kernels'] == 0 for r in rows), 'Wrong per-role path coverage')
    samples = [s for s in train['all_samples'] if s['arm'] == arm]
    values = [s['ms_per_step'] for s in samples]
    variability[arm] = dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                           minimum_ms=min(values), maximum_ms=max(values))
    summary = dict(rank_records=0, module_records=0, slot_records=0, initializations=set(),
        successful_calls=set(), bias_residual_calls=set(), bound_k=set(), failed=set(),
        trainable_parameters=set(), forward_calls=set(), backward_calls=set(),
        baseline_backward_calls=set(), async_gather_calls=set(), wait_calls=set())
    for sample in samples:
        for rank in range(8):
            row = read(BATCH/'fullstep/results'/sample['directory']/f'rank{rank}.json')
            summary['rank_records'] += 1
            summary['trainable_parameters'].add(len(row['ma6_trainable_parameters']))
            for name, state in row['ma6_consumer_state'].items():
                summary['module_records'] += 1
                for slot in state['native']:
                    summary['slot_records'] += 1
                    for key in ('initializations', 'successful_calls', 'bias_residual_calls', 'bound_k', 'failed'):
                        summary[key].add(slot[key])
                backward = row['ma6_backward_state'][name]
                for key in ('forward_calls', 'backward_calls', 'baseline_backward_calls', 'async_gather_calls', 'wait_calls'):
                    summary[key].add(backward[key])
    native[arm] = {k: sorted(v) if isinstance(v, set) else v for k, v in summary.items()}
    require(native[arm] == dict(rank_records=96, module_records=6144, slot_records=12288,
        initializations=[1], successful_calls=[120], bias_residual_calls=[120], bound_k=[512, 2048],
        failed=[False], trainable_parameters=[389], forward_calls=[240], backward_calls=[240],
        baseline_backward_calls=[0 if arm == 'F4-gather' else 240],
        async_gather_calls=[240 if arm == 'F4-gather' else 0], wait_calls=[240 if arm == 'F4-gather' else 0]),
        'Unexpected native/backward state: '+arm)
    require(len(gpu_spans[arm]) == 8, 'Missing GPU span diagnostics')

require(before == {str(p.relative_to(REPO)): sha(p) for p in files}, 'Original evidence changed')
audit = dict(completed=True, date=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat(),
    gpu_used=False, batch=str(BATCH.relative_to(REPO)), replay_mode='frozen-v1',
    recomputed_identically=checks, frozen_controllers_match=controllers,
    matched_frozen_snapshot_files=len(frozen_files), frozen_snapshot_sha256=frozen_files,
    original_reports_plans_gates_unchanged=True,
    evidence_sha256=before, formal_training=dict(windows=36, rank_records=288, optimizer_steps=1800),
    training_comparisons=train['comparisons'], numeric_summary=train['numeric_summary'],
    operator_comparisons=op['roles'], paired_blocks=blocks, paired_block_outcomes=wins,
    training_window_variability=variability, training_profile_rank_medians=medians,
    training_profile_per_rank=profiles, backward_overlap=overlaps, actual_training_launches=launches,
    native_and_backward_counts=native,
    profile_gpu_spans_ms={a:dict(values=v, median=statistics.median(v), minimum=min(v), maximum=max(v))
                         for a, v in gpu_spans.items()},
    increment_over_F4_validated=train['increment_over_F4_validated'], target_evaluated=False,
    limitation='Offline raw-record/trace verification; no GPU rerun or loading/rehashing of server binaries. '
               'One exploratory trial. Profile spans and overlap do not allocate unprofiled step time; '
               'no shared upstream-gradient-ready event, S6 control, combined candidate, or C/R comparison.')
if args.out is not None:
    with args.out.open('x') as handle:
        handle.write(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
print('Audit complete; original evidence unchanged; outcomes:', wins, flush=True)
