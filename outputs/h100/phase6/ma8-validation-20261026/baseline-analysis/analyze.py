"""Read existing F4 profile files and record MA8's redundant activation gather evidence.

Run from any directory: python3 path/to/analyze.py
Writes only analysis.json beside this script; never edits historical evidence.
"""
from pathlib import Path
import gzip
import hashlib
import json
import statistics
import sys

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[4]
sys.path.insert(0, str(REPO / 'tools/h100/phase6'))
import ma8_profile


def main():
    source = REPO / 'logs/h100/phase6/ma6-v3-receiver-1x-50steps-20261026-145811-248551/fullstep/results/profile-F4'
    ranks = []
    for rank in range(8):
        path = source / f'rank{rank}-profile.json.gz'
        raw = path.read_bytes()
        events = json.loads(gzip.decompress(raw))['traceEvents']
        ranks.append(dict(rank=rank, input=str(path.relative_to(REPO)),
                          sha256=hashlib.sha256(raw).hexdigest(),
                          columns=ma8_profile.historical_column_records(events),
                          unchanged_row_backward=ma8_profile.trace_row_backward_records(events, 'F4')))
    roles = {}
    for role in ('qkv', 'fc1'):
        keys = ranks[0]['columns']['roles'][role]['median_us']
        roles[role] = {key: statistics.median(r['columns']['roles'][role]['median_us'][key] for r in ranks)
                       for key in keys}
    output = dict(protocol='h100-ma8-historical-activation-gather-analysis-v1',
                  gpu_rerun=False, historical_inputs_read_only=True,
                  analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  profile_helper_sha256=hashlib.sha256(Path(ma8_profile.__file__).read_bytes()).hexdigest(),
                  caveat='Profile timings include rank arrival skew and instrumentation; they are not expected unprofiled savings.',
                  target_reference_ms=501.433, target_percent=2., target_net_ms=501.433 * .02,
                  avoided_calls_per_rank_per_step=256,
                  target_net_us_per_avoided_call=501.433 * .02 * 1000 / 256,
                  extra_retained_full_activation_bytes_per_microbatch=64 * 2048 * 4096 * 2,
                  avoided_logical_remote_input_bytes_per_rank_per_step=256 * 7 * 256 * 4096 * 2,
                  rank_median_then_cross_rank_median_us=roles, ranks=ranks)
    (OUT / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(rank_traces=8, column_calls=2048, row_calls=2048,
                         target_net_ms=output['target_net_ms'],
                         target_net_us_per_avoided_call=output['target_net_us_per_avoided_call']), indent=2))


if __name__ == '__main__':
    main()
