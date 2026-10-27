"""Recompute MA13 parser validation of original MA11 traces without writes."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root / 'tools/h100/phase6'))
import ma13_profile
record = json.loads(Path(__file__).with_suffix('.json').read_text())
assert hashlib.sha256((root / 'tools/h100/phase6/ma13_profile.py').read_bytes()).hexdigest() == record['checker_sha256']
for expected in record['records']:
    path = root / expected['trace']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256']
    with gzip.open(path, 'rt') as handle:
        events = json.load(handle)['traceEvents']
    evidence = ma13_profile.trace_composed_records(events, record['arm'], expected['rank'])
    observed = dict(rank=expected['rank'], trace=expected['trace'], sha256=expected['sha256'],
        forward=evidence['attention_softmax']['forward_calls'],
        backward=evidence['attention_softmax']['backward_calls'],
        native_forward=evidence['attention_softmax']['forward_kernel'],
        cache_backward_allgathers=evidence['activation_cache']['backward_allgathers'])
    assert observed == expected
print(json.dumps(dict(passed=True, ranks=8, candidate_gpu_validated=False)))
