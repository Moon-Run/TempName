"""CPU routing/admission tests; temporary fixtures are never GPU evidence."""
import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import fused_support
import fused_protocol as routing
import fused_prepare as preparation
import fused_report as reporter
import fused_run as runner
import fused_build as builder
import fused_plan as planner
import scope_sources
from common import read, write, sha
from ablation_plan import mask_sha


def profile_fixture(policy):
    events = []
    for role_index, role in enumerate(routing.SHAPES):
        route = 'candidate' if role in routing.definition(policy)['roles'] else 'native_bf16'
        names = ['flux_bf16_sm80', 'taco_decode_ring_kernel'] if route == 'candidate' else ['native_gemm']
        for call in range(128):
            start = role_index * 10000 + call * 20
            events.append(dict(name=f'scope.{role}.{route}', cat='user_annotation', ph='X',
                               ts=start, dur=10, pid=1, tid=1))
            for offset, name in enumerate(names):
                correlation = role_index * 10000 + call * 10 + offset
                events.append(dict(name='cudaLaunchKernel', cat='cuda_runtime', ts=start + offset + 1,
                                   pid=1, tid=1, args=dict(correlation=correlation)))
                events.append(dict(name=name, cat='kernel', args=dict(correlation=correlation)))
    return events


class ProtocolTests(unittest.TestCase):
    def test_nine_groups_balance_and_primary_comparison(self):
        jobs = routing.schedule()
        self.assertEqual(len(jobs), 162)
        self.assertEqual(len(set(routing.POLICIES)), 9)
        rows = [[j['policy'] for j in jobs if j['round'] == rep and j['block'] == block]
                for rep in (1, 2) for block in range(9)]
        self.assertEqual(rows[9:], [row[::-1] for row in rows[:9][::-1]])
        for position in range(9):
            self.assertEqual({row[position] for row in rows[:9]}, set(routing.POLICIES))
        self.assertEqual(len(routing.PRIMARY), 3)
        self.assertEqual(len(routing.COMPARISONS), 18)
        for candidate, reference in routing.PRIMARY:
            self.assertEqual(routing.definition(candidate)['order'], 'improved')
            self.assertEqual(routing.definition(reference)['order'], 'legacy')
            self.assertEqual(routing.definition(candidate)['scope'], routing.definition(reference)['scope'])
        self.assertTrue(all(routing.definition(p)['quantization'] == 'targeted' for p in routing.POLICIES))
        for blocks in (0, 1, 3, 8, 10):
            with self.assertRaises(ValueError):
                routing.schedule(blocks)

    def test_each_role_has_one_identical_mask_in_every_order_and_scope(self):
        plans = planner.bundle(source_only=True)
        for role in routing.SHAPES:
            parent_mask = plans['parent']['roles'][role]['entry']['mask']
            for order in routing.ORDERS:
                single = routing.role_plans(plans, routing.policy_id(order, role))[role]
                both = routing.role_plans(plans, routing.policy_id(order, 'attn_mlp'))[role]
                self.assertEqual(single, both)
                self.assertEqual(single['entry']['mask'], parent_mask)
                self.assertEqual(single['mask_sha256'], mask_sha(parent_mask))
                routing.selection_entry(plans, routing.policy_id(order, role), role, *routing.SHAPES[role])
        with self.assertRaises(ValueError):
            routing.selection_entry(plans, routing.policy_id('improved', 'attn'), 'mlp', *routing.SHAPES['mlp'])

    def test_improved_regression_is_preserved_and_best_block_is_paired(self):
        values = {policy: [100.] for policy in routing.POLICIES}
        candidate, reference = routing.PRIMARY[0]
        values[candidate] = [120.]
        result = reporter.summarize(values, [dict(round=1, block=2)])
        comparison = result['comparisons'][candidate + '/' + reference]
        self.assertAlmostEqual(comparison['percent'], -20.)
        self.assertIsNone(comparison['ci95'])
        self.assertEqual(result['speedups'][candidate + '/' + reference]['best']['block'], 2)
        with self.assertRaises(ValueError):
            reporter.summarize(dict(values, **{candidate: []}), [dict(round=1, block=2)])

    def test_profile_requires_fused_v2_and_correct_scope(self):
        for policy in routing.POLICIES:
            result = reporter.profile_roles(profile_fixture(policy), policy)
            self.assertEqual({record['calls'] for record in result.values()}, {128})
        policy = routing.policy_id('improved', 'mlp')
        events = profile_fixture(policy)
        with self.assertRaises(ValueError):
            reporter.profile_roles(events, routing.policy_id('improved', 'attn'))
        for event in events:
            if event['name'] == 'flux_bf16_sm80':
                event['name'] = 'flux_bf16_sm90'
        with self.assertRaises(ValueError):
            reporter.profile_roles(events, policy)


class SourceTests(unittest.TestCase):
    def test_worker_gate_runs_before_torch_and_source_only_cannot_be_enabled(self):
        source = preparation.worker_source()
        self.assertLess(source.index('H100_PHASE7_FUSED_EXECUTE'), source.index('import torch'))
        self.assertNotIn('H100_SCOPE_EXECUTE', source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'worker.py'
            for source_only in (False, True):
                path.write_text(preparation.worker_source(source_only))
                env = dict(os.environ)
                env.pop('H100_PHASE7_FUSED_EXECUTE', None)
                if source_only:
                    env['H100_PHASE7_FUSED_EXECUTE'] = '1'
                result = subprocess.run([sys.executable, str(path)], env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Source-only' if source_only else 'phase7 fused --execute', result.stderr)
                self.assertNotIn('No module named', result.stderr)

    def test_wrapper_uses_original_double_buffer_and_exact_role_mask(self):
        class Codec:
            owns_configuration = True

            def __init__(self, group, m, n, k, **kwargs):
                self.kwargs = kwargs
                self.slots = [types.SimpleNamespace(m=m, n=n, k_local=k, rank=0,
                    selection_sha256=mask_sha(kwargs['selected']),
                    peers=[types.SimpleNamespace(data_ptr=lambda: 123)],
                    selected=types.SimpleNamespace(data_ptr=lambda: 456)) for _ in range(2)]

        torch = types.ModuleType('torch')
        torch.no_grad = lambda: lambda fn: fn
        dist = types.ModuleType('torch.distributed')
        dist.get_world_size = lambda _: 8
        torch.distributed = dist
        modules = {'torch': torch, 'torch.distributed': dist, 'routing': routing,
                   'reference': types.SimpleNamespace(roundtrip=lambda x: x),
                   'flux': types.ModuleType('flux'),
                   'flux.gemm_rs_taco': types.SimpleNamespace(GemmRSTacoDoubleBuffered=Codec)}
        plans = planner.bundle(source_only=True)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, modules):
            root = Path(directory)
            write(root / 'selection-plan.json', plans)
            namespace = {'__file__': str(root / 'taco_support.py')}
            exec(scope_sources.taco_source(), namespace)
            for policy in routing.POLICIES:
                for role in routing.definition(policy)['roles']:
                    m, n, k = routing.SHAPES[role]
                    op = namespace['TacoForward'](None, m, n, k // 8, policy, role)
                    record = routing.role_plans(plans, policy)[role]
                    self.assertEqual(op.codec.kwargs['selected'], record['entry']['mask'])
                    self.assertEqual(op.codec.kwargs['placement'], 'fused')
                    self.assertEqual(op.codec.kwargs['quantization_mode'], 'targeted')
                    self.assertEqual(op.plan_sha256, record['entry_sha256'])
                    self.assertEqual(len(op.workspace_info), 2)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='phase7-fused-CPU-FIXTURE-')
        cls.root = Path(cls.temp.name)
        cls.build = builder.prepare(cls.root / 'build', source_only=True)
        cls.compare = preparation.prepare(cls.root / 'compare', cls.build, Path('/absent'), source_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_prepare_dry_run_never_imports_torch_or_launches_gpu(self):
        with mock.patch.object(subprocess, 'run', side_effect=AssertionError('Unexpected process launch')), \
                mock.patch.object(subprocess, 'check_output', side_effect=AssertionError('Unexpected GPU query')):
            with contextlib.redirect_stdout(io.StringIO()):
                jobs = runner.run(self.compare)
            info = reporter.verify_inputs(self.compare)
            self.assertEqual(len(jobs), 180)
            self.assertEqual(len(list(runner.mapping_jobs(self.compare, info))), 12)
            with self.assertRaises(ValueError):
                runner.run(self.compare, execute=True)
            with self.assertRaises(ValueError):
                reporter.preflight(self.compare)
        self.assertEqual(sha(self.compare / 'scripts/runtime.py'), sha(fused_support.PHASE6 / 'runtime.py'))
        self.assertNotIn('ChunkedGemmRS', (self.compare / 'scripts/adapter.py').read_text())
        with self.assertRaises(ValueError):
            builder.verify(self.build, built=True)

    def test_rewritten_hashes_cannot_promote_rehearsal_or_change_library_order(self):
        path = self.compare / 'submission.json'
        original = read(path)
        try:
            promoted = dict(original, source_only=False, executable=True)
            write(path, promoted)
            with self.assertRaises(ValueError):
                reporter.verify_inputs(self.compare)
            changed = copy.deepcopy(original)
            changed['variants'][routing.policy_id('base', 'mlp')]['root'] = \
                changed['variants'][routing.policy_id('legacy', 'mlp')]['root']
            write(path, changed)
            with self.assertRaises(ValueError):
                reporter.verify_inputs(self.compare)
        finally:
            write(path, original)

    def test_script_and_schedule_mutation_is_detected(self):
        for name in ('scripts/taco_support.py', 'scripts/selection-plan.json', 'schedule.json'):
            path = self.compare / name
            original = path.read_bytes()
            try:
                path.write_bytes(original + b'\n')
                with self.assertRaises(ValueError):
                    reporter.verify_inputs(self.compare)
            finally:
                path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
