"""CPU campaign regression tests; all result records are synthetic test fixtures.

Fixtures live only in TemporaryDirectory and never establish GPU validation.
Preparation uses the saved measured Phase6 plans, not invented calibration.
"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import phase7
import protocol

try:
    import torch
    import adapter
except ImportError:
    torch = None


def replace_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def fixture_window(root, job, elapsed=0.002):
    """Write explicit synthetic, non-executed rank records for validator tests."""
    m = protocol.verify(root)
    folder = Path(root) / 'results' / job['directory']
    folder.mkdir(parents=True)
    roles = ('attention', 'mlp') if m['scope'] == 'both' else (m['scope'],)
    ordering = protocol.VARIANTS[job['variant']]['ordering']
    plans = {role: protocol.digest(protocol.read(root / 'plans' / f'{role}-{ordering}.json')) for role in roles}
    count = 1 if job['stage'] == 'profile' else m['steps']
    for rank in range(8):
        record = dict(protocol=protocol.PROTOCOL, run_id=m['run_id'],
                      manifest_sha256=protocol.sha(root / 'manifest.json'),
                      rank=rank, world_size=8, **job, kind=m['kind'], scope=m['scope'],
                      variant_definition=protocol.VARIANTS[job['variant']], plans_sha256=plans,
                      completed=True, passed=True, skips=0, steps=count,
                      elapsed_seconds=elapsed * (1 + rank / 100),
                      fingerprints={'fixture_rank_input': str(rank)},
                      device=dict(name='NVIDIA H100 SYNTHETIC TEST RECORD', capability=[9, 0],
                                  sm_count=132, uuid=f'SYNTHETIC-NOT-A-DEVICE-{rank}'),
                      model_config=m['model'], implementation=m['implementation'],
                      execution=dict(python='synthetic-unit-test', torch='synthetic', cuda='synthetic', nccl='synthetic'),
                      forward_checks={role: dict(passed=True, relative_l2=0.0,
                                                protocol_relative_l2=0.0, max_abs=0.0) for role in roles},
                      backward_checked=True, losses=[], grad_norms=[],
                      synthetic_fixture=True, fixture_warning='NOT GPU EVIDENCE')
        if m['kind'] == 'model':
            modules = {f'decoder.layers.{layer}.' + ('self_attention.linear_proj' if role == 'attention' else 'mlp.linear_fc2'):
                       count * 4 for layer in range(32) for role in roles}
            record.update(losses=[2.0] * count, grad_norms=[1.0] * count, module_calls=modules,
                          forward_checks={name: dict(passed=True, relative_l2=0.0,
                                                     protocol_relative_l2=0.0, max_abs=0.0) for name in modules})
        replace_json(folder / f'rank{rank}.json', record)
    return folder


class ScheduleAndStatisticsTests(unittest.TestCase):
    def test_schedule_balances_positions_and_keeps_complete_paired_blocks(self):
        jobs = protocol.schedule(4)
        self.assertEqual(len(jobs), 32)
        self.assertEqual(len({x['directory'] for x in jobs}), 32)
        rounds = [[x for x in jobs if x['round'] == r] for r in (1, 2)]
        for round_jobs in rounds:
            rows = [[x['variant'] for x in round_jobs if x['block'] == b] for b in range(4)]
            self.assertTrue(all(set(row) == set(protocol.VARIANTS) for row in rows))
            for position in range(4):
                self.assertEqual({row[position] for row in rows}, set(protocol.VARIANTS))
        self.assertEqual([x['variant'] for x in rounds[0]][::-1], [x['variant'] for x in rounds[1]])
        for blocks in (0, 1, 3, -4, True):
            with self.assertRaises(ValueError):
                protocol.schedule(blocks)

    def test_paired_interaction_has_expected_direction_and_magnitude(self):
        summary = protocol.summarize(dict(E0=[100, 200], E1=[90, 180], E2=[80, 160], E3=[64, 128]), resamples=100)
        c = summary['comparisons']
        self.assertAlmostEqual(c['arrival_barrier']['percent'], 10)
        self.assertAlmostEqual(c['arrival_pipeline']['percent'], 20)
        self.assertAlmostEqual(c['interaction']['percent'], 100 * (1 - .8 / .9))
        self.assertEqual(summary['blocks'], 2)
        self.assertAlmostEqual(c['arrival_pipeline']['ci95'][0], c['arrival_pipeline']['ci95'][1])

    def test_single_block_has_no_interval_and_bad_times_are_rejected(self):
        values = {v: [1.0] for v in protocol.VARIANTS}
        self.assertIsNone(protocol.summarize(values)['comparisons']['interaction']['ci95'])
        for value in (0, -1, float('nan'), float('inf')):
            bad = copy.deepcopy(values)
            bad['E0'][0] = value
            with self.assertRaises(ValueError):
                protocol.summarize(bad)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='phase7-protocol-SYNTHETIC-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'campaign'
        phase7.prepare(self.root, warmup=1, steps=2)

    def smoke_job(self, variant='E0'):
        return dict(variant=variant, stage='smoke', round=0, block=-1, directory='smoke-' + variant)

    def first_job(self):
        return dict(protocol.read(self.root / 'manifest.json')['schedule'][0], stage='timing')

    def test_preparation_freezes_complete_input_and_equal_physical_masks(self):
        m = protocol.verify(self.root)
        self.assertFalse(m['gpu_validated'])
        self.assertEqual(m['implementation'], 'torch_chunked_gemm_nccl_alltoall_prototype')
        self.assertTrue({'scripts/worker.py', 'scripts/reference.py', 'scripts/runtime.py'} <= set(m['files']))
        identity = protocol.read(self.root / 'plans/attention-identity.json')
        shared = protocol.read(self.root / 'plans/attention-shared.json')
        self.assertEqual(identity['quantize_mask'], shared['quantize_mask'])
        self.assertFalse(identity['performance_validated'])
        self.assertEqual(identity['evidence_kind'], 'legacy_proxy')

    def test_prepare_refuses_existing_destination(self):
        before = protocol.sha(self.root / 'manifest.json')
        with self.assertRaises(FileExistsError):
            phase7.prepare(self.root)
        self.assertEqual(before, protocol.sha(self.root / 'manifest.json'))

    def test_frozen_worker_edit_and_omitted_hash_are_rejected(self):
        m = protocol.read(self.root / 'manifest.json')
        del m['files']['scripts/worker.py']
        replace_json(self.root / 'manifest.json', m)
        with self.assertRaisesRegex(ValueError, 'required frozen input'):
            protocol.verify(self.root)

    def test_frozen_plan_edit_is_rejected(self):
        path = self.root / 'plans/attention-shared.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Frozen file missing or changed'):
            protocol.verify(self.root)

    def test_dry_run_does_not_launch_processes_or_create_results(self):
        with mock.patch.object(subprocess, 'run') as spawn, contextlib.redirect_stdout(io.StringIO()):
            tasks = phase7.run(self.root, stage='all')
        spawn.assert_not_called()
        self.assertGreaterEqual(len(tasks), 36)
        self.assertEqual(list((self.root / 'results').iterdir()), [])
        self.assertFalse((self.root / 'all-started.json').exists())

    def test_worker_refuses_execution_before_importing_torch(self):
        env = dict(os.environ)
        env.pop('PHASE7_EXECUTE', None)
        result = subprocess.run([sys.executable, str(self.root / 'scripts/worker.py'), '--root', str(self.root)],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires the controller run --execute', result.stderr)
        self.assertNotIn("No module named 'torch'", result.stderr)

    def test_rank_max_is_one_window_and_missing_rank_is_rejected(self):
        job = self.first_job()
        folder = fixture_window(self.root, job)
        window = protocol.validate_window(self.root, folder, job)
        self.assertAlmostEqual(window['ms'], 1.07)
        self.assertEqual(len(window['fingerprints']), 8)
        (folder / 'rank7.json').unlink()
        with self.assertRaises(FileNotFoundError):
            protocol.validate_window(self.root, folder, job)

    def test_replaying_same_variant_in_another_block_is_rejected(self):
        job = self.first_job()
        folder = fixture_window(self.root, job)
        bad = protocol.read(folder / 'rank0.json')
        bad['block'] += 1
        replace_json(folder / 'rank0.json', bad)
        with self.assertRaisesRegex(ValueError, 'Wrong job'):
            protocol.validate_window(self.root, folder, job)

    def test_changed_runtime_plan_or_cross_rank_environment_is_rejected(self):
        job = self.first_job()
        folder = fixture_window(self.root, job)
        original = protocol.read(folder / 'rank0.json')
        bad = copy.deepcopy(original)
        bad['plans_sha256']['attention'] = '0' * 64
        replace_json(folder / 'rank0.json', bad)
        with self.assertRaisesRegex(ValueError, 'plan/runtime mode'):
            protocol.validate_window(self.root, folder, job)
        replace_json(folder / 'rank0.json', original)
        bad = protocol.read(folder / 'rank7.json')
        bad['execution']['torch'] = 'different-version'
        replace_json(folder / 'rank7.json', bad)
        with self.assertRaisesRegex(ValueError, 'runtime environments'):
            protocol.validate_window(self.root, folder, job)

    def test_device_identity_and_frozen_model_metadata_are_checked(self):
        job = self.first_job()
        folder = fixture_window(self.root, job)
        path = folder / 'rank7.json'
        original = protocol.read(path)
        mutations = (
            lambda r: r['device'].update(sm_count=114),
            lambda r: r['device'].update(uuid='SYNTHETIC-NOT-A-DEVICE-0'),
            lambda r: r['model_config'].update(layers=31),
            lambda r: r.update(implementation='unregistered_runtime'),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                bad = copy.deepcopy(original)
                mutate(bad)
                replace_json(path, bad)
                with self.assertRaises(ValueError):
                    protocol.validate_window(self.root, folder, job)
        replace_json(path, original)

    def test_model_rank_losses_must_agree_and_projection_calls_are_complete(self):
        root = Path(self.temp.name) / 'model-fixture'
        phase7.prepare(root, kind='model', scope='both', megatron='/synthetic-not-executed/Megatron-LM',
                       warmup=1, steps=2)
        job = dict(protocol.read(root / 'manifest.json')['schedule'][0], stage='timing')
        folder = fixture_window(root, job)
        protocol.validate_window(root, folder, job)
        path = folder / 'rank3.json'
        original = protocol.read(path)
        bad = copy.deepcopy(original)
        bad['losses'][0] += 1
        replace_json(path, bad)
        with self.assertRaisesRegex(ValueError, 'loss records disagree'):
            protocol.validate_window(root, folder, job)
        replace_json(path, original)
        bad = copy.deepcopy(original)
        del bad['module_calls'][next(iter(bad['module_calls']))]
        replace_json(path, bad)
        with self.assertRaises(ValueError):
            protocol.validate_window(root, folder, job)

    def test_preflight_requires_numerical_checks_on_every_rank(self):
        for variant in protocol.VARIANTS:
            fixture_window(self.root, self.smoke_job(variant))
        path = self.root / 'results/smoke-E2/rank5.json'
        bad = protocol.read(path)
        bad['forward_checks'] = {}
        replace_json(path, bad)
        with self.assertRaises(ValueError):
            phase7.preflight(self.root)
        self.assertFalse((self.root / 'preflight.json').exists())

    def test_preflight_rejects_missing_backward_or_exceeded_numerical_budget(self):
        for variant in protocol.VARIANTS:
            fixture_window(self.root, self.smoke_job(variant))
        path = self.root / 'results/smoke-E2/rank5.json'
        original = protocol.read(path)
        bad = copy.deepcopy(original)
        bad['backward_checked'] = False
        replace_json(path, bad)
        with self.assertRaisesRegex(ValueError, 'gradient check'):
            phase7.preflight(self.root)
        bad = copy.deepcopy(original)
        bad['forward_checks']['attention']['relative_l2'] = .051
        replace_json(path, bad)
        with self.assertRaisesRegex(ValueError, 'Numerical budget'):
            phase7.preflight(self.root)
        self.assertFalse((self.root / 'preflight.json').exists())

    def test_profile_is_one_step_and_excluded_from_timing_schedule(self):
        job = dict(variant='E3', stage='profile', round=0, block=-1, directory='profile-E3')
        folder = fixture_window(self.root, job)
        value = protocol.validate_window(self.root, folder, job)
        self.assertAlmostEqual(value['ms'], 2.14)
        m = protocol.read(self.root / 'manifest.json')
        self.assertTrue(all(j['stage'] == 'timing' for j in phase7.jobs(m, 'timing')))
        self.assertTrue(all(j['stage'] == 'profile' for j in phase7.jobs(m, 'profile')))
        bad = protocol.read(folder / 'rank0.json')
        bad['steps'] = m['steps']
        replace_json(folder / 'rank0.json', bad)
        with self.assertRaisesRegex(ValueError, 'missing steps'):
            protocol.validate_window(self.root, folder, job)

    def test_full_synthetic_report_uses_four_paired_blocks_and_rejects_rewrite(self):
        m = protocol.read(self.root / 'manifest.json')
        for variant in protocol.VARIANTS:
            fixture_window(self.root, self.smoke_job(variant))
        for item in m['schedule']:
            factor = dict(E0=1, E1=.9, E2=.8, E3=.64)[item['variant']]
            fixture_window(self.root, dict(item, stage='timing'), elapsed=.002 * factor * (item['block'] + 1))
        with mock.patch.object(phase7, 'summarize', side_effect=lambda v: protocol.summarize(v, resamples=100)):
            report = phase7.report(self.root)
            self.assertEqual(report['metric'], 'ms/operator')
            self.assertEqual(len(report['files']), 32 * 8)
            self.assertEqual([r['blocks'] for r in report['rounds'].values()], [4, 4])
            for r in report['rounds'].values():
                self.assertAlmostEqual(r['comparisons']['arrival_pipeline']['percent'], 20)
            self.assertEqual(report, phase7.report(self.root))
            job = m['schedule'][0]
            path = self.root / 'results' / job['directory'] / 'rank7.json'
            bad = protocol.read(path)
            bad['elapsed_seconds'] *= 10
            replace_json(path, bad)
            with self.assertRaisesRegex(ValueError, 'Report differs'):
                phase7.report(self.root)


@unittest.skipIf(torch is None, 'PyTorch unavailable; CPU autograd checks require torch')
class AdapterTests(unittest.TestCase):
    def test_bf16_ste_backward_gathers_sequence_shards_and_computes_both_gradients(self):
        generator = torch.Generator().manual_seed(11)
        x = torch.randn((16, 1, 4), generator=generator, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn((6, 4), generator=generator, dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn((2, 1, 6), generator=generator, dtype=torch.bfloat16)
        op = types.SimpleNamespace(group=object(), forward=lambda a, w: a.matmul(w.t())[:2])

        def gather(target, source, group):
            self.assertIs(group, op.group)
            target.copy_(source.repeat(8, 1, 1))

        with mock.patch.object(adapter.dist, 'get_world_size', return_value=8), \
                mock.patch.object(adapter.dist, 'all_gather_into_tensor', side_effect=gather) as exchange:
            out = adapter.ChunkedLinear.apply(x, weight, op)
            self.assertEqual(list(out.shape), [2, 1, 6])
            out.backward(grad)
        exchange.assert_called_once()
        full = grad.repeat(8, 1, 1)
        self.assertTrue(torch.equal(x.grad, full.matmul(weight.detach())))
        self.assertTrue(torch.equal(weight.grad, full.reshape(-1, 6).t().matmul(x.detach().reshape(-1, 4))))

    def fake_model(self):
        class Row:
            def __init__(self, k):
                self.input_is_parallel = self.sequence_parallel = True
                self.explicit_expert_comm = self.gradient_accumulation_fusion = False
                self.weight = torch.empty((4096, k), dtype=torch.bfloat16, device='meta')
                self.config = types.SimpleNamespace(_cpu_offloading_context=None)
                self.output_size, self.input_size_per_partition = 4096, k
                self.forward = lambda x: ('original', x)
        rows = []
        for layer in range(32):
            rows += [(f'decoder.layers.{layer}.self_attention.linear_proj', Row(512)),
                     (f'decoder.layers.{layer}.mlp.linear_fc2', Row(2048))]
        core = types.ModuleType('megatron.core')
        core.parallel_state = types.SimpleNamespace(get_tensor_model_parallel_group=lambda: object())
        layers = types.ModuleType('megatron.core.tensor_parallel.layers')
        layers.RowParallelLinear = Row
        modules = {'megatron.core': core, 'megatron.core.tensor_parallel.layers': layers}
        return rows, types.SimpleNamespace(named_modules=lambda: iter(rows)), modules

    def test_adapter_changes_only_requested_role_and_restore_is_exact(self):
        import planner
        rows, model, modules = self.fake_model()
        originals = {name: module.forward for name, module in rows}
        plans = {'attention': planner.make_plan(planner.default_shape('attention'), chunk_tiles=8)}
        with mock.patch.dict(sys.modules, modules):
            instance = adapter.Adapter(model, plans, mode='pipeline')
        self.assertEqual(len(instance.modules), 32)
        for name, module in rows:
            if name.endswith('.mlp.linear_fc2'):
                self.assertIs(module.forward, originals[name])
            else:
                self.assertIsNot(module.forward, originals[name])
        self.assertEqual(instance.ops, {})  # Lazy construction: validation does not launch collectives.
        instance.restore()
        for name, module in rows:
            self.assertIs(module.forward, originals[name])

    def test_late_invalid_module_does_not_partially_patch_model(self):
        import planner
        rows, model, modules = self.fake_model()
        originals = {name: module.forward for name, module in rows}
        rows[-2][1].sequence_parallel = False
        plans = {'attention': planner.make_plan(planner.default_shape('attention'), chunk_tiles=8)}
        with mock.patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, 'Unsupported Phase7 module'):
            adapter.Adapter(model, plans, mode='pipeline')
        for name, module in rows:
            self.assertIs(module.forward, originals[name])


if __name__ == '__main__':
    unittest.main()
