#!/usr/bin/env python3
"""Replay the MA10 trace gate on all eight immutable original CR3 traces."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / 'tools/h100/phase6'))
import ma10_profile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = REPO / 'logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/results/profile-F4'
    records = []
    for rank in range(8):
        trace = root / f'rank{rank}-profile.json.gz'
        events = json.load(gzip.open(trace, 'rt'))['traceEvents']
        record = ma10_profile.trace_gradient_batching_records(events, 'F4')
        record.update(rank=rank, trace=str(trace.relative_to(REPO)), trace_sha256=digest(trace))
        records.append(record)
    result = dict(protocol='ma10-original-cr3-profile-replay-v1', passed=True,
                  profile_sha256=digest(Path(ma10_profile.__file__)), records=records,
                  gpu_executed=False, candidate_gpu_validated=False,
                  limitation='Offline replay of historical F4 only; no candidate performance or GPU correctness claim.')
    Path(__file__).with_name('profile-replay.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, ranks=len(records),
                         gradient_kernel_calls=[r['gradient_kernel_calls'] for r in records],
                         profile_sha256=result['profile_sha256']), indent=2))


if __name__ == '__main__':
    main()
