"""Read-only MA12 result replay, using only the returned frozen controllers.

No Torch import, GPU execution, runtime loading, or historical evidence writes.
--out must name a fresh JSON file outside the returned batch.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[4]
BATCH = REPO / 'logs/h100/phase6/ma12-v1-tile-publication-1x-50steps-20261027-133258-366517'
ARMS = ('F4-cache-zcount', 'F4-cache-zcount-tilepub')
CONTRAST = '/'.join(ARMS[::-1])


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def paired(candidate, baseline, seed):
    logs = [math.log(a / b) for a, b in zip(candidate, baseline)]
    rng = random.Random(seed)
    values = sorted(100 * (1 - math.exp(statistics.mean(rng.choices(logs, k=len(logs)))))
                    for _ in range(10000))
    return dict(percent=100 * (1 - math.exp(statistics.mean(logs))),
                ci95=[values[249], values[9749]], paired_blocks=len(logs))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
require(not args.out.exists() and not args.out.resolve().is_relative_to(BATCH), 'Use a fresh audit output outside the batch')
protected = [BATCH / 'batch-state.json', BATCH / 'handoff/launch.json',
             BATCH / 'handoff/controllers.json', BATCH / 'fullstep/report.md']
protected += [BATCH / part / name for part in ('operator', 'fullstep')
              for name in ('plan.json', 'preflight.json', 'report.json', 'profile-summary.json')]
before = {str(p.relative_to(REPO)): sha(p) for p in protected}
plans = {part: read(BATCH / part / 'plan.json') for part in ('operator', 'fullstep')}
frozen = {}
for part, plan in plans.items():
    require(tuple(plan['arms']) == ARMS, 'Wrong two-arm comparison')
    for relative, digest in plan['source_files'].items():
        path = BATCH / part / relative
        require(path.resolve().is_relative_to(BATCH / part / 'scripts'), 'Unexpected frozen path')
        require(sha(path) == digest, 'Frozen source changed: ' + str(path))
        frozen[part + '/' + relative] = digest
dependencies = read(BATCH / 'handoff/controllers.json')
current_matches = {}
for name, digest in dependencies.items():
    relative = name.removeprefix('/workspace/TempName/')
    require(not Path(relative).is_absolute(), 'Unknown controller root')
    local = REPO / relative
    current_matches[relative] = local.is_file() and sha(local) == digest
for source, original in (('operator/scripts/bench.py', 'ma12_bench.py'),
                         ('fullstep/scripts/operator-controller.py', 'ma12_bench.py'),
                         ('fullstep/scripts/fullstep-controller.py', 'ma12_fullstep.py')):
    require(sha(BATCH / source) == dependencies['/workspace/TempName/tools/h100/phase6/' + original],
            'Launch/frozen controller identity differs')
snapshot = plans['fullstep']['megatron_snapshot']
for relative, digest in snapshot['files'].items():
    require(sha(BATCH / 'fullstep/megatron' / relative) == digest, 'Changed returned Megatron source')
native = plans['fullstep']['build_identity']['source_files']
native_arms = {arm: {key.split('/build/' + arm + '/', 1)[1]: digest for key, digest in native.items()
                     if '/build/' + arm + '/' in key} for arm in ARMS}
native_added = sorted(set(native_arms[ARMS[1]]) - set(native_arms[ARMS[0]]))
native_changed = sorted(k for k in native_arms[ARMS[0]] if native_arms[ARMS[0]][k] != native_arms[ARMS[1]][k])
require(native_added == ['src/gemm_rs/ma12_tile_ready.cuh'] and
        native_changed == sorted(['src/cuda/version.ld', 'src/gemm_rs/taco_runtime.cu',
            'src/gemm_rs/taco_runtime.h', 'src/gemm_rs/epilogue_evt.hpp',
            'src/gemm_rs/ma4_f_decode.cuh', 'src/gemm_rs/ths_op/gemm_reduce_scatter.cc']), 'Unregistered native change')

scripts = BATCH / 'fullstep/scripts'
sys.path[:] = [str(scripts), *[p for p in sys.path if p and not Path(p).resolve().is_relative_to(REPO)]]
spec = importlib.util.spec_from_file_location('_ma12_frozen_audit', scripts / 'fullstep-controller.py')
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
bench = full.bench
require(full._FROZEN and bench.SOURCE_PROTOCOL == 'h100-ma12-tile-publication-v1' and
        Path(bench.__file__).parent == scripts, 'Wrong frozen replay controller')
print('Verified frozen source files:', len(frozen), 'Megatron files:', len(snapshot['files']), flush=True)
gpu_spans = {arm: [] for arm in ARMS}
original = full.trace_projection_records


projection_work = {arm: [] for arm in ARMS}


def capture_span(events, arm, tuning=None):
    metrics = []
    module = full.ma12_profile
    original_chain = module._chain
    def capture_chain(work, chain_arm, role, tuning=None):
        result = original_chain(work, chain_arm, role, tuning)
        metrics.append(dict(role=role, **result[-1]))
        return result
    module._chain = capture_chain
    try:
        result = original(events, arm, tuning=tuning)
    finally:
        module._chain = original_chain
    require(len(metrics) == 256, 'Incomplete projection work accounting')
    sums = {key.removesuffix('_us'): sum(row[key] for row in metrics) / 1000
            for key in metrics[0] if key != 'role'}
    projection_work[arm].append(sums)
    device = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    gpu_spans[arm].append((max(e['ts'] + e.get('dur', 0) for e in device) - min(e['ts'] for e in device)) / 1000)
    return result


recomputed = {}
full.trace_projection_records = capture_span
try:
    for name, callback in (
        ('operator/report.json', lambda: bench.report(BATCH / 'operator', save=False, offline=True)),
        ('fullstep/report.json', lambda: full.report(BATCH / 'fullstep', save=False, offline=True)),
        ('operator/profile-summary.json', lambda: bench.profile_summary(BATCH / 'operator', offline=True)),
        ('fullstep/profile-summary.json', lambda: full.profile_summary(BATCH / 'fullstep', offline=True)),
    ):
        result = callback()
        require(result == read(BATCH / name), 'Replay mismatch: ' + name)
        recomputed[name] = result
        print('Recomputed identically:', name, flush=True)
finally:
    full.trace_projection_records = original

train = recomputed['fullstep/report.json']
state = read(BATCH / 'batch-state.json')
require(state['status'] == 'completed' and state['stage'] == 'done' and train['trials'] == state['trials'] == 1,
        'Incomplete batch')
require(train['formal_optimizer_steps'] == 800 and train['timing_rank_records'] == 128 and
        len(train['all_samples']) == 16 and train['target_reference'] == ARMS[0], 'Wrong measured scope')
for key, value in full.aggregate_acceptance(train['comparisons']).items():
    require(value == train[key] == state[key], 'Final acceptance differs: ' + key)
values, memory = {a: {} for a in ARMS}, {a: [] for a in ARMS}
for sample in train['all_samples']:
    rows = [read(BATCH / 'fullstep/results' / sample['directory'] / f'rank{r}.json') for r in range(8)]
    ms = max(row['elapsed_seconds'] for row in rows) * 1000 / 50
    require(ms == sample['ms_per_step'], 'Raw window time differs')
    values[sample['arm']][sample['block']] = ms
    memory[sample['arm']].append({k: max(row[k] for row in rows) for k in ('peak_allocated_gib', 'peak_reserved_gib')})
    for row in rows:
        require(len(row['ma12_trainable_parameters']) == 389 and len(row['ma12_activation_setup']['module_roles']) == 64,
                'Parameter/module universe changed')
        require(row['ma12_activation_setup']['installed'] and row['ma12_activation_setup']['arm'] == 'F4-cache',
                'Missing common activation cache')
rounds = {}
for label, blocks in (('round1', range(4)), ('round2', range(4, 8)), ('combined', range(8))):
    selected = {arm: [values[arm][i] for i in blocks] for arm in ARMS}
    result = dict(mean_ms={a: statistics.mean(v) for a, v in selected.items()},
                  median_ms={a: statistics.median(v) for a, v in selected.items()},
                  comparisons={CONTRAST: paired(selected[ARMS[1]], selected[ARMS[0]], train['bootstrap']['seed'])})
    require(result == train['comparisons']['compare1']['rounds'][label], 'Independent paired bootstrap differs')
    rounds[label] = result
mem = {a: {k: dict(mean_window_rank_max=statistics.mean(x[k] for x in rows), max_window_rank_max=max(x[k] for x in rows))
           for k in ('peak_allocated_gib', 'peak_reserved_gib')} for a, rows in memory.items()}
require(mem == train['peak_cuda_memory_gib'], 'Independent memory summary differs')
blocks = [dict(block=i, control_ms=values[ARMS[0]][i], candidate_ms=values[ARMS[1]][i],
               saved_ms=values[ARMS[0]][i] - values[ARMS[1]][i]) for i in range(8)]
profiles = recomputed['fullstep/profile-summary.json']['profiles']
zero, cache = {}, {}
for arm in ARMS:
    rows = [profiles[f'{arm}/rank{i}'] for i in range(8)]
    zero[arm] = dict(ranks=[r['zero_count'] for r in rows],
                    median_ms={key: statistics.median(r['zero_count'][key] for r in rows) for key in
                               ('cpu_statistic_span_ms', 'gpu_compute_work_ms', 'gpu_allreduce_ms', 'gpu_statistic_span_ms')})
    cache[arm] = {key: sum(r['activation_cache'][key] for r in rows) for key in
                  ('column_calls', 'retained_forward_allgathers', 'backward_allgathers', 'retained_reduce_scatters')}
require(before == {str(p.relative_to(REPO)): sha(p) for p in protected}, 'Protected evidence modified')
require('torch' not in sys.modules, 'Audit unexpectedly imported Torch')
means = rounds['combined']['mean_ms']
result = dict(protocol='h100-ma12-results-audit-v1', batch=str(BATCH.relative_to(REPO)), gpu_rerun=False,
    original_evidence_unchanged=True, protected_sha256=before, reports_recomputed_identically=list(recomputed),
    frozen_source_files_verified=len(frozen), frozen_source_sha256=frozen, megatron_files_verified=len(snapshot['files']),
    controller_dependencies_match_current=current_matches,
    registered_native_files={a:len(v) for a,v in native_arms.items()}, native_added=native_added, native_changed=native_changed,
    returned_build_present=(BATCH/'build').exists(), binary_reload_performed=False,
    rounds=rounds, blocks=blocks, faster_blocks=sum(b['saved_ms'] > 0 for b in blocks),
    mean_saved_ms=means[ARMS[0]]-means[ARMS[1]], window_ranges_ms={a:[min(v.values()),max(v.values())] for a,v in values.items()},
    tokens_per_second={a:8192000/v for a,v in means.items()}, peak_cuda_memory_gib=mem,
    numeric_summary=train['numeric_summary'], zero_count_profiles=zero, cache_calls_across_eight_ranks=cache,
    projection_work_rank_sums_ms=projection_work,
    projection_work_median_rank_sum_ms={a:{k:statistics.median(row[k] for row in rows)
        for k in rows[0]} for a,rows in projection_work.items()},
    profile_gpu_spans_ms=gpu_spans, profile_gpu_span_median_ms={a:statistics.median(v) for a,v in gpu_spans.items()},
    limitation='One independent trial; profiler timings include waiting and instrumentation, not a decomposition of formal step savings. MA89 is the concurrent control.',
    audit_source_sha256=sha(Path(__file__)))
args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print('Raw timing, all paired bootstrap rounds and memory summaries match. Faster blocks:', result['faster_blocks'], flush=True)
print('Protected evidence unchanged. Audit saved:', args.out, flush=True)
