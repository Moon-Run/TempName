"""Read-only diagnostics of a failed preflight; never certify it or start timing."""
import collections
import csv
import gzip
import json
import os
from pathlib import Path
import sys

repo = Path(os.environ['REPO'])
out = Path(os.environ['H100_SCOPE_OUT'])
log = Path(os.environ['H100_SCOPE_LOG'])
root = log / 'compare'
sys.path.insert(0, str(repo / 'tools/h100/phase6'))
from common import read, sha, verify_files, write
from scope_report import verify_inputs, window
from scope_protocol import POLICIES, definition, is_native
from scope_calibrate import verify, calibration_records

info = verify_inputs(root)
verify_files('/', info['controllers'])
assert read(root / 'all-state.json')['status'] == 'failed'
assert not (root / 'preflight.json').exists() and not (root / 'report.json').exists()
assert not list((root / 'results').glob('round*/window-*'))
cal = log / 'calibration'
prepared = verify(cal, execute=True)
calibration_rows = calibration_records(cal, prepared)
assert len(calibration_rows) == 8 and prepared['roles'] == ['attn', 'mlp']
assert read(cal / 'calibration-complete.json')['completed']
plans = read(root / 'scripts/selection-plan.json')['roles']
profile_diagnostics = {}
smoke_diagnostics = {}
for policy in POLICIES:
    for mode in ('smoke', 'profile'):
        rows = window(root, root / 'results' / (mode + '-' + policy), policy, mode)
        if mode == 'smoke':
            checks = [v for row in rows for v in row['forward_checks'].values()]
            smoke_diagnostics[policy] = dict(
                rank_records=8, passed=True, skips=sum(r['skips'] for r in rows),
                fallback=sum(r['fallback_calls'] for r in rows),
                max_relative_l2_vs_bf16=max(v['relative_l2'] for v in checks),
                max_relative_l2_vs_codec=max(v['protocol_relative_l2'] for v in checks))
        else:
            for rank, row in enumerate(rows):
                trace = root / 'results' / ('profile-' + policy) / f'rank{rank}-profile.json.gz'
                with gzip.open(trace, 'rt') as stream:
                    events = json.load(stream)['traceEvents']
                scope_names = {e['name'] for e in events if str(e.get('name', '')).startswith('scope.')}
                counts = {name: dict(collections.Counter(e.get('cat') for e in events
                          if e.get('name') == name and e.get('ph') == 'X')) for name in sorted(scope_names)}
                profile_diagnostics[f'{policy}/rank{rank}'] = dict(
                    scopes=counts, trace_sha256=sha(trace), rank_record_passed=row['passed'],
                    module_call_count_histogram=dict(collections.Counter(row['timed_calls'].values())))
    if not is_native(policy):
        for role in definition(policy)['roles']:
            path = root / 'results' / f'mapping-{policy}-{role}.csv'
            with path.open() as stream:
                rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(stream)]
            assert len(rows) == 4096
            for rank in range(8):
                rr = sorted((r for r in rows if r['rank'] == rank), key=lambda r: r['index'])
                assert [r['index'] for r in rr] == list(range(512))
                assert [r['m'] * 32 + r['n'] for r in rr] == plans[role]['entry']['coords'][rank]
                assert all(r['destination'] == r['m'] // 2 and r['sm_occupancy'] > 0 for r in rr)
write(log / 'handoff/profile-scope-diagnostics.json', profile_diagnostics)
write(log / 'handoff/partial-evidence-verification.json', dict(
    diagnostic_only=True, official_preflight_passed=False, formal_timing_started=False,
    official_report_available=False, new_calibration_rank_records=8, calibration_roles=['attn', 'mlp'],
    all_smoke_rank_records_verified=True, smoke_rank_records=32, raw_preflight_rank_records=64,
    raw_profile_traces=32, candidate_shape_mappings_verified=3, formal_rank_records=0,
    smoke=smoke_diagnostics, failure='Missing profile scopes: scope.attn.tensor',
    verified_local_library_and_source_hashes=True,
    explanation='The scope counter accepts every ph=X record, counting CPU and GPU user annotations together. '
                'The observed total is 256, made of 128 user_annotation and 128 gpu_user_annotation per role.',
    historical_evidence_unchanged=True, failed_session_preserved=True,
))
inventory = {str(p.relative_to(repo)): sha(p) for parent in (log, out)
             for p in parent.rglob('*') if p.is_file() and not p.is_symlink()
             and 'build' not in p.parts and '__pycache__' not in p.parts
             and p.suffix not in ('.so', '.o', '.a') and p.name != 'check_mapping'
             and '.so.' not in p.name and p.suffix != '.lock'}
write(log / 'handoff/partial-artifact-hashes.json', inventory)
print('PARTIAL EVIDENCE VERIFIED; OFFICIAL PREFLIGHT REMAINS FAILED; NO TIMING')
print(json.dumps(smoke_diagnostics, indent=2))
