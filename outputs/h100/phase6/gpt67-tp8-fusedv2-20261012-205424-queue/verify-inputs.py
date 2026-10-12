"""Verify unchanged baseline and controllers before each independent repeat."""
import os
from pathlib import Path
import sys

repo = Path(os.environ['REPO'])
sys.path.insert(0, str(repo / 'tools/h100/phase6'))
from common import config, manifest, read, sha, verify_files, write
from report import verify_inputs

base = Path(os.environ['H100_BUILD']) / 'base'
data = manifest(base, 'baselines')
assert data['config'] == config(Path(os.environ['H100_CONFIG']))
assert data['variants']['taco_fused']['implementation'] == 'a800-fused-v2-cta-v1'
previous = repo / 'logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/compare'
info = verify_inputs(previous)
verify_files('/', info['controllers'])
assert read(previous / 'report.json')['completed']
assert read(previous / 'timing-state.json')['status'] == 'completed'
assert read(previous / 'verification.json')['passed']
drivers = [Path(os.environ['H100_BUILD']) / name for name in
           ('record-inputs.py', 'execute.py', 'verify-results.py')]
drivers += [Path(os.environ['H100_QUEUE_OUTPUT']) / name for name in
            ('queue.py', 'run-one.sh', 'verify-inputs.py')]
write(Path(os.environ['H100_LOG']) / 'handoff/repeat-inputs.json', dict(
    previous_compare=str(previous), base_build=str(base),
    base_manifest_sha256=sha(base / 'manifest.json'), base_files=data['files'],
    implementation='a800-fused-v2-cta-v1',
    drivers={str(p): sha(p) for p in drivers}, fresh_preflight=True,
    fresh_timing_windows=72, reuse_previous_timings=False,
))
print('BASE AND FROZEN CONTROLLERS VERIFIED; starting independent repeat', flush=True)
