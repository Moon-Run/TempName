#!/usr/bin/env python3
"""Replay the complete MA12 control gates against eight original MA89 traces."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tools/h100/phase6'))
import ma12_profile

root = REPO / 'logs/h100/phase6/ma89-v1-cache-zcount-1x-50steps-20261027-104719-859382/fullstep'
tuning = json.loads((root / 'plan.json').read_text())['tuning']
results = []
for rank in range(8):
    path = root / 'results/profile-F4-cache-zcount' / f'rank{rank}-profile.json.gz'
    with gzip.open(path) as source:
        events = json.load(source)['traceEvents']
    record = ma12_profile.trace_projection_records(events, ma12_profile.ARMS[0], tuning)
    results.append(dict(rank=rank, evidence=record,
                        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    print(f'rank{rank}: complete MA12 reference parser passed', flush=True)
report = dict(protocol='ma12-original-profile-replay-v1',
              parser_sha256=hashlib.sha256(Path(ma12_profile.__file__).read_bytes()).hexdigest(),
              records=results, candidate_gpu_tested=False)
Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
