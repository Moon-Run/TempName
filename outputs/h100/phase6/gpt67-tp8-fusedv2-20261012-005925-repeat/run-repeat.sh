#!/usr/bin/env bash
set -eo pipefail
source /workspace/TempName/outputs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/env.sh
mkdir -p "$H100_LOG/handoff"
exec 9>"$H100_LOG/launch.lock"
flock -n 9 || exit 1
test ! -e "$H100_LOG/compare/submission.json"
cd "$REPO"
python "$H100_BUILD/record-inputs.py"
python - <<'PY'
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
previous = repo / 'logs/h100/phase6' / os.environ['H100_REUSED_SESSION'] / 'compare'
info = verify_inputs(previous)
verify_files('/', info['controllers'])
assert read(previous / 'report.json')['completed']
assert read(previous / 'timing-state.json')['status'] == 'completed'
drivers = [Path(os.environ['H100_BUILD']) / name for name in
           ('record-inputs.py', 'execute.py', 'verify-results.py')]
drivers += [Path(os.environ['H100_OUTPUT']) / name for name in ('env.sh', 'run-repeat.sh')]
write(Path(os.environ['H100_LOG']) / 'handoff/repeat-inputs.json', dict(
    reused_session=os.environ['H100_REUSED_SESSION'], previous_compare=str(previous),
    base_build=str(base), base_manifest_sha256=sha(base / 'manifest.json'),
    base_files=data['files'], implementation='a800-fused-v2-cta-v1',
    drivers={str(p): sha(p) for p in drivers}, fresh_preflight=True,
    fresh_timing_windows=72, reuse_previous_timings=False,
))
print('BASE AND FROZEN CONTROLLERS VERIFIED; starting independent repeat', flush=True)
PY
python -u "$H100_BUILD/execute.py"
python -u "$H100_BUILD/verify-results.py"
