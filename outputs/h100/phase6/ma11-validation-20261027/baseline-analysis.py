#!/usr/bin/env python3
"""Read-only replay of all eight original MA89 full-step traces through MA11."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[4]
TOOLS = REPO / 'tools/h100/phase6'
sys.path.insert(0, str(TOOLS))
import ma11_fullstep
import ma11_profile

BATCH = 'ma89-v1-cache-zcount-1x-50steps-20261027-104719-859382'
ROOT = REPO / 'logs/h100/phase6' / BATCH / 'fullstep'
TUNING = json.loads((ROOT / 'plan.json').read_text())['tuning']
CONTROLLER_HASHES = {name: hashlib.sha256((TOOLS / name).read_bytes()).hexdigest()
                     for name in ('ma11_profile.py', 'ma11_fullstep.py')}
records = []
for rank in range(8):
    path = ROOT / 'results/profile-F4-cache-zcount' / f'rank{rank}-profile.json.gz'
    events = json.load(gzip.open(path))['traceEvents']
    # This is the complete public integration: original row forward parser,
    # row backward, MA8 cache, MA9 statistic, and MA11 attention gates.
    evidence = ma11_fullstep.trace_projection_records(events, ma11_profile.ARMS[0], TUNING)
    counts = collections.Counter(e['name'] for e in events if e.get('cat') == 'cpu_op')
    selected_counts = {key: counts[key] for key in (
        'aten::masked_fill_', 'aten::mul', 'MaskedFillBackward0',
        'AsStridedBackward0', 'torch::autograd::CopySlices')}
    records.append(dict(rank=rank, path=str(path.relative_to(REPO)),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        integration='ma11_fullstep.trace_projection_records',
                        evidence=evidence, global_cpu_counts=selected_counts))
    print(f'rank {rank}: complete MA11 reference profile integration passed', flush=True)
keys = records[0]['evidence']['attention_softmax']['gpu_work_ms']
report = dict(
    protocol='h100-ma11-design-profile-analysis-v1', baseline_batch=BATCH,
    baseline_arm=ma11_profile.ARMS[0], source_traces=8,
    integration='ma11_fullstep.trace_projection_records',
    controller_sha256=CONTROLLER_HASHES,
    records=records,
    median_gpu_work_ms={key: statistics.median(
        r['evidence']['attention_softmax']['gpu_work_ms'][key] for r in records) for key in keys},
    median_view_replay_cpu_ms={key: statistics.median(
        r['evidence']['attention_softmax']['view_replay_cpu_ms'][key] for r in records)
                              for key in ('as_strided', 'copy_slices')},
    median_view_replay_device={label: {key: statistics.median(
        r['evidence']['attention_softmax']['view_replay_device'][label].get(key, 0)
        for r in records) for key in records[0]['evidence']['attention_softmax']['view_replay_device'][label]}
                              for label in ('as_strided', 'copy_slices')},
    candidate_gpu_tested=False, performance_target_validated=False,
    limitation='Launch-correlated original MA89 profile events. CPU intervals overlap GPU work and '
               'profiler changes timing; no sum or difference is a prediction of unprofiled '
               'critical-path savings or proof of a 2% speedup.')
Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report['median_gpu_work_ms'], indent=2), flush=True)
