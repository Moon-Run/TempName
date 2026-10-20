"""Read-only recomputation of the returned op2 calibration and full-step evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO/'tools/h100/phase6'))
import arrival_op2_report as reporter
import arrival_probe
import arrival_profile_audit as profiles

A = 'interleaved_all_attn_mlp'
S = 'interleaved_pair_order_all_attn_mlp'
POLICIES = (A, S)
HASHES = {}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    HASHES[str(path.relative_to(REPO))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def paired(a, s):
    logs = [math.log(y/x) for x, y in zip(a, s)]
    rng = random.Random(20261009)
    gain = lambda xs: 100*(1-math.exp(statistics.mean(xs)))
    bootstrap = sorted(gain([rng.choice(logs) for _ in logs]) for _ in range(10000))
    return dict(percent=gain(logs), ci95=[bootstrap[250], bootstrap[9750]], blocks=len(logs),
                log_ratios=logs)


def map_audit(root, plans):
    result = {}
    for role in ('attn', 'mlp'):
        tables = {}
        for policy in POLICIES:
            path = root/'results'/f'mapping-{policy}-{role}.csv'
            HASHES[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(path.open())]
            assert len(rows) == 4096
            by_rank = {}
            for rank in range(8):
                rr = sorted((row for row in rows if row['rank'] == rank), key=lambda r: r['index'])
                coords = [r['m']*32+r['n'] for r in rr]
                assert [r['index'] for r in rr] == list(range(512))
                assert sorted(coords) == list(range(512))
                assert coords == plans['variants'][policy]['roles'][role]['entry']['coords'][rank]
                assert all(r['destination'] == r['m']//2 for r in rr)
                by_rank[rank] = rr
            tables[policy] = by_rank
        changed = [sum((x['m'], x['n']) != (y['m'], y['n'])
                       for x, y in zip(tables[A][rank], tables[S][rank])) for rank in range(8)]
        destinations = sum(x['destination'] != y['destination']
                           for rank in range(8) for x, y in zip(tables[A][rank], tables[S][rank]))
        assert destinations == 0
        result[role] = dict(changed_per_rank=changed, changed_total=sum(changed),
                            changed_fraction=sum(changed)/4096, changed_destinations=destinations,
                            occupancies={p: sorted({row['sm_occupancy'] for rr in tables[p].values()
                                                   for row in rr}) for p in POLICIES})
    return result


def audit_compare(root):
    saved = read(root/'report.json')
    info = read(root/'submission.json')
    gate = read(root/'preflight.json')
    assert read(root/'all-state.json')['status'] == 'completed'
    # Replays all original validators in a temporary symlink tree, never source files.
    replay = arrival_probe.replay_report(dict(root=root, reporter=reporter))
    assert replay['completed'] and replay['correctness_passed']
    plans = read(root/'scripts/selection-plan.json')
    selected = read(root/'provenance/selection/selection.json')
    fused = read(root/'provenance/parent-fused-manifest.json')['variants']['interleaved_all']['source_sha256']
    improved = read(root/'provenance/improved-manifest.json')['variants']['paired_order']['source_sha256']
    changed = sorted(k for k in fused.keys() | improved.keys() if fused.get(k) != improved.get(k))
    assert changed == ['phase6_host.hpp', 'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp']
    masks = {}
    for role in ('attn', 'mlp'):
        a, s = (plans['variants'][p]['roles'][role]['entry']['mask'] for p in POLICIES)
        assert a == s and list(map(sum, a)) == [448]*8
        assert all(not a[rank][t] for rank in range(8) for t in range(rank*64, (rank+1)*64))
        masks[role] = list(map(sum, a))
    raw = []
    for job, archived in zip(info['schedule'], saved['all_samples']):
        folder = root/'results'/job['directory']
        rows = [read(folder/f'rank{rank}.json') for rank in range(8)]
        for rank, r in enumerate(rows):
            assert r['rank'] == rank and r['completed'] and r['passed']
            assert r['skips'] == r['fallback_calls'] == 0
            assert r['timed_steps'] == 50 and r['warmup_steps'] == 10
            assert len(r['losses']) == len(r['grad_norms']) == 50
            assert all(math.isfinite(v) for v in r['losses']+r['grad_norms'])
            assert r['losses'] == rows[0]['losses']
            assert r['runtime_spans'] == ({'attn': 32, 'mlp': 32} if job['policy'] == A else selected['selected_spans'])
            assert r['timed_target_calls_expected'] == 12800
        elapsed = [r['elapsed_seconds'] for r in rows]
        ms = max(elapsed)*1000/50
        assert elapsed == archived['rank_elapsed_seconds'] and ms == archived['ms_per_step']
        raw.append(dict(**job, ms_per_step=ms, losses=rows[0]['losses'],
                        mean_loss=statistics.mean(rows[0]['losses']), final_loss=rows[0]['losses'][-1]))
    summaries = {}
    for round_number in (1, 2, 'combined'):
        rows = [r for r in raw if round_number == 'combined' or r['round'] == round_number]
        data = {p: [r['ms_per_step'] for r in rows if r['policy'] == p] for p in POLICIES}
        # Both lists inherit the same (round,block) order from the balanced schedule.
        assert [[(r['round'], r['block']) for r in rows if r['policy'] == p] for p in POLICIES][0] == [
            (r['round'], r['block']) for r in rows if r['policy'] == S]
        stats = paired(data[A], data[S])
        assert same(stats, saved['rounds'][str(round_number)]['comparisons'][S+'/'+A])
        summaries[str(round_number)] = dict(
            means={p: statistics.mean(v) for p, v in data.items()},
            medians={p: statistics.median(v) for p, v in data.items()}, paired=stats)
    smoke = {}
    for policy in POLICIES:
        rows = [read(root/'results'/f'smoke-{policy}'/f'rank{rank}.json') for rank in range(8)]
        checks = [check for row in rows for check in row['forward_checks'].values()]
        assert len(checks) == 512 and all(c['passed'] for c in checks)
        smoke[policy] = dict(checks=len(checks),
                             max_relative_bf16_l2=max(c['relative_l2'] for c in checks),
                             max_relative_codec_l2=max(c['protocol_relative_l2'] for c in checks))
    trace_summaries = {}
    for policy in POLICIES:
        trace_summaries[policy] = {}
        for rank in (0, 4):
            path = root/'results'/f'profile-{policy}'/f'rank{rank}-profile.json.gz'
            HASHES[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
            trace_summaries[policy][str(rank)] = {
                role: profiles.summarize(records)
                for role, records in profiles.trace_projection_records(path).items()}
    return dict(run_id=saved['run_id'], strict_preflight_report_replay_passed=True,
                selection_sha256=info['selection_sha256'], measurements=raw, summaries=summaries,
                mappings=map_audit(root, plans), source_files=len(fused),
                changed_source_files=changed, non_scheduler_sources_identical=True, masks=masks,
                smoke=smoke, profiles=trace_summaries, acceptance=saved['acceptance'],
                sensitivity=saved['sensitivity'], measurement_identity=saved['measurement_identity'],
                role_plan_sha256=saved['role_plan_sha256'], selection=selected['selected_spans'])


def audit_probe(root):
    selected = read(root/'selection.json')
    plan = read(root/'scripts/config.json')
    values = {role: {label: [] for label, _ in plan['catalog']} for role in ('attn', 'mlp')}
    output_hashes, input_hashes = {}, {}
    for job in plan['schedule']:
        rows = [read(root/'results'/job['directory']/f'rank{rank}.json') for rank in range(8)]
        for rank, row in enumerate(rows):
            assert row['completed'] and row['rank'] == rank and row['job'] == job
            assert row['timing'] == {'warmup': 100, 'iterations': 200, 'batches_per_worker': 1}
            for role, sample in row['results'].items():
                key = (rank, role)
                assert sample['correctness']['passed'] and sample['repeated_bitwise_equal']
                assert sample['output_sha256'] == sample['final_output_sha256']
                assert output_hashes.setdefault(key, sample['output_sha256']) == sample['output_sha256']
                inputs = (sample['inputs_sha256'], sample['weights_sha256'])
                assert input_hashes.setdefault(key, inputs) == inputs
        for role in values:
            values[role][job['label']].append(max(r['results'][role]['wall_us'] for r in rows))
    assert values == selected['values']
    return dict(run_id=selected['run_id'], selected_spans=selected['selected_spans'],
                rank_records=len(plan['schedule'])*8, worker_windows=len(plan['schedule']),
                means_us={role: {label: statistics.mean(v) for label, v in rows.items()}
                          for role, rows in values.items()},
                decisions=selected['decisions'], bitwise_equal_to_A=True,
                timing=selected['timing'], software=selected['software'],
                hardware=selected['hardware'], selection_content_digest=selected['selection_sha256'],
                selection_file_sha256=HASHES[str((root/'selection.json').relative_to(REPO))])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--batch', type=Path, default=REPO/'logs/h100/phase6/arrival-op2-order-2x-50steps-20261020-161635-645726')
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('audit.json'))
    args = parser.parse_args()
    batch = args.batch.resolve()
    assert not args.out.exists()
    state = read(batch/'batch-state.json')
    assert state['status'] == 'completed' and not state['target_validated']
    result = dict(gpu_launched=False, source_evidence_modified=False, batch=str(batch.relative_to(REPO)),
                  state=state, probe=audit_probe(batch/'probe'), compares={})
    for name in ('compare1', 'compare2'):
        result['compares'][name] = audit_compare(batch/name)
        print(name, 'strict replay and independent raw checks passed', flush=True)
    c1, c2 = result['compares'].values()
    assert c1['measurement_identity'] == c2['measurement_identity']
    assert c1['role_plan_sha256'] == c2['role_plan_sha256']
    assert c1['selection_sha256'] == c2['selection_sha256'] == result['probe']['selection_file_sha256']
    assert len({c1['run_id'], c2['run_id'], result['probe']['run_id']}) == 3
    all_rows = c1['measurements']+c2['measurements']
    combined = {}
    for policy in POLICIES:
        rows = [r for r in all_rows if r['policy'] == policy]
        combined[policy] = dict(windows=len(rows), mean_ms=statistics.mean(r['ms_per_step'] for r in rows),
                                median_ms=statistics.median(r['ms_per_step'] for r in rows),
                                mean_loss=statistics.mean(r['mean_loss'] for r in rows),
                                mean_final_loss=statistics.mean(r['final_loss'] for r in rows))
        combined[policy]['tokens_per_second'] = 8192000/combined[policy]['mean_ms']
    assert all(row['losses'] == all_rows[0]['losses'] for row in all_rows)
    means = result['probe']['means_us']
    differences = {role: means[role]['A']-means[role]['g'+str(span)]
                   for role, span in result['probe']['selected_spans'].items()}
    baseline = combined[A]['mean_ms']
    result.update(combined_descriptive=combined,
                  all_window_loss_trajectories_identical=True,
                  combined_mean_time_reduction_percent=100*(1-combined[S]['mean_ms']/baseline),
                  candidate_minus_baseline_ms=combined[S]['mean_ms']-baseline,
                  target_budget=dict(step_ms=baseline*.015, average_projection_us=baseline*.015*1000/256),
                  illustrative_transfer=dict(operator_savings_us=differences,
                      assumed_step_savings_ms=128*sum(differences.values())/1000,
                      assumed_step_gain_percent=100*(128*sum(differences.values())/1000)/baseline,
                      caveat='Illustrative constant-savings transfer only; not measured step attribution or a bound.'),
                  input_sha256=HASHES)
    for relative, expected in HASHES.items():
        assert hashlib.sha256((REPO/relative).read_bytes()).hexdigest() == expected
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print('Wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
