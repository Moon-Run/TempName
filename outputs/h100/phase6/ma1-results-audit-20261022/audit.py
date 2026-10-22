"""Offline audit of the completed MA1 batch; never modifies raw reports or uses GPUs."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tools/h100/phase6'))
import ma1_bench as bench
import ma1_fullstep as full

BATCH = REPO / 'logs/h100/phase6/ma1-prefetch-1x-50steps-20261021-233916-307862'
OUT = Path(__file__).parent


def main():
    state = bench.read(BATCH / 'batch-state.json')
    assert state['status'] == 'completed' and state['stage'] == 'done'
    checks, inputs = {}, {'batch-state.json': bench.sha(BATCH / 'batch-state.json')}
    reports = {}
    for kind, module, frozen in (
            ('operator', bench, 'scripts/bench.py'),
            ('fullstep', full, 'scripts/fullstep-controller.py')):
        root = BATCH / kind
        assert bench.sha(module.__file__) == bench.sha(root / frozen)
        # Both preflight files already exist, so even full.preflight does no write.
        assert (root / 'preflight.json').is_file()
        for name, recomputed in (
                ('preflight.json', module.preflight(root, offline=True, **({'save': False} if kind == 'operator' else {}))),
                ('report.json', module.report(root, offline=True, save=False)),
                ('profile-summary.json', module.profile_summary(root, offline=True))):
            assert recomputed == bench.read(root / name), (kind, name)
            checks[f'{kind}/{name}'] = True
            inputs[f'{kind}/{name}'] = bench.sha(root / name)
            for relative, digest in recomputed.get('evidence_files', recomputed.get('files', {})).items():
                assert bench.sha(root / relative) == digest, relative
            reports[(kind, name)] = recomputed
        inputs[f'{kind}/plan.json'] = bench.sha(root / 'plan.json')

    report = reports[('fullstep', 'report.json')]
    plan = bench.read(BATCH / 'fullstep/plan.json')
    assert plan['trials'] == 1 and plan['blocks'] == 12
    for key in ('target_validated', 'total_benefit_validated', 'prefetch_benefit_validated', 'warp_control_passed'):
        assert state[key] == report[key]
    assert state['formal_optimizer_steps'] == report['formal_optimizer_steps'] == 1800
    assert state['timing_windows'] == len(report['all_samples']) == 36
    assert state['operator_comparisons'] == {
        role: data['comparisons'] for role, data in reports[('operator', 'report.json')]['roles'].items()}
    assert state['operator_bitwise_equal'] and state['operator_execution_path_validated'] and state['fullstep_execution_path_validated']

    raw = {arm: [] for arm in bench.ARMS}
    rank_trajectories = {rank: set() for rank in range(8)}
    for sample in report['all_samples']:
        rows = [bench.read(BATCH / 'fullstep/results' / sample['directory'] / f'rank{rank}.json') for rank in range(8)]
        ms = max(row['elapsed_seconds'] for row in rows) * 1000 / 50
        assert ms == sample['ms_per_step']
        for rank, row in enumerate(rows):
            assert len(row['losses']) == len(row['grad_norms']) == 50
            rank_trajectories[rank].add(hashlib.sha256(json.dumps(
                [row['losses'], row['grad_norms']], separators=(',', ':')).encode()).hexdigest())
        raw[sample['arm']].append(dict(block=sample['block'], ms=ms,
            mean_loss=statistics.mean(rows[0]['losses']), final_loss=rows[0]['losses'][-1],
            initial_loss=rows[0]['initial_loss'], final_grad_norm=rows[0]['grad_norms'][-1]))
    stats = {}
    for arm, rows in raw.items():
        values = [row['ms'] for row in rows]
        stats[arm] = dict(windows=len(rows), formal_steps=len(rows)*50,
            mean_ms=statistics.mean(values), median_ms=statistics.median(values),
            min_ms=min(values), max_ms=max(values), max_over_median=max(values)/statistics.median(values),
            tokens_per_second=8192000/statistics.mean(values),
            mean_loss=statistics.mean(row['mean_loss'] for row in rows),
            final_loss_mean=statistics.mean(row['final_loss'] for row in rows),
            initial_loss_mean=statistics.mean(row['initial_loss'] for row in rows),
            final_grad_norm_mean=statistics.mean(row['final_grad_norm'] for row in rows))
        assert stats[arm]['mean_ms'] == report['comparisons']['compare1']['rounds']['combined']['mean_ms'][arm]

    numerics = {}
    for arm in bench.ARMS:
        items = [item for rank in range(8) for item in bench.read(
            BATCH / f'fullstep/results/smoke-{arm}/rank{rank}.json')['forward_checks'].values()]
        assert len(items) == 512
        numerics[arm] = dict(checks=len(items),
            max_relative_l2=max(x['relative_l2'] for x in items),
            max_protocol_relative_l2=max(x['protocol_relative_l2'] for x in items))

    profile_counts, decode = {}, {}
    for kind in ('operator', 'fullstep'):
        profiles = reports[(kind, 'profile-summary.json')]['profiles']
        profile_counts[kind] = len(profiles)
        decode[kind] = {}
        for arm in bench.ARMS:
            decode[kind][arm] = {}
            for role in bench.ROLES:
                key = 'decoder_median_us' if kind == 'operator' else 'decode_median_us'
                data = [profiles[f'{arm}/rank{rank}/{role}'] if kind == 'operator'
                        else profiles[f'{arm}/rank{rank}'][role] for rank in range(8)]
                values = [item[key] for item in data]
                decode[kind][arm][role] = dict(median_of_rank_medians_us=statistics.median(values),
                    min_rank_median_us=min(values), max_rank_median_us=max(values),
                    calls=sum(item['calls' if kind == 'operator' else 'projection_calls'] for item in data))

    provenance = dict(build_directory_returned=(BATCH/'build').exists(),
                      current_controllers_match_frozen=True, megatron_snapshot_hashes_verified=0)
    for name, digest in plan['megatron_snapshot']['files'].items():
        assert bench.sha(BATCH/'fullstep/megatron'/name) == digest, name
        provenance['megatron_snapshot_hashes_verified'] += 1
    assert bench.sha(BATCH/'fullstep/scripts/operator-controller.py') == bench.sha(bench.__file__)

    result = dict(completed=True, gpu_used=False, raw_logs_modified=False,
        batch=BATCH.name, report_recomputations=checks, input_sha256=inputs,
        source_sha256={str(Path(m.__file__).relative_to(REPO)): bench.sha(m.__file__) for m in (bench, full)},
        fullstep_run_id=plan['run_id'], comparison_run_id=plan['trial_ids']['1'],
        formal_rank_records=report['timing_rank_records'],
        fullstep_statistics=stats, rounds=report['comparisons']['compare1']['rounds'],
        unique_loss_gradient_trajectories_per_rank={str(rank):len(x) for rank,x in rank_trajectories.items()},
        forward_checks=numerics, profile_counts=profile_counts,
        decoder_profile=decode, provenance=provenance,
        boundaries=['One complete comparison only; no cross-batch replication.',
                    'No same-session C arm; do not compare against historical C timing.',
                    'Profile durations are diagnostics and cannot be summed into unprofiled step savings.',
                    'Missing build binaries cannot be independently reloaded or rehashed on this host.'])
    (OUT/'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
