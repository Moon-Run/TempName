"""CPU checks for decoder attribution, frozen baselines and strict target reporting."""
import ast
import contextlib
import copy
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ablation3_plan
import ablation4_build
import arrival_op2 as cli
import arrival_op2_build as builder
import arrival_op2_plan as planner
import arrival_op2_prepare as preparation
import arrival_op2_protocol as routing
import arrival_op2_report as reporter
import arrival_op2_sources as sources
import scope_plan
import test_ablation7_report as report_fixtures
from common import read, sha, write


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = scope_plan.bundle(source_only=True)
        cls.plans = planner.bundle(cls.parent, source_only=True)

    def test_a_and_d_keep_exact_original_a_role_records(self):
        expected = ablation3_plan.derive(self.parent, routing.A)['roles']
        self.assertEqual(routing.role_plans(self.plans, routing.A), expected)
        self.assertEqual(routing.role_plans(self.plans, routing.D), expected)
        self.assertNotIn('scores', self.plans)
        for policy in routing.POLICIES:
            for record in routing.role_plans(self.plans, policy).values():
                self.assertEqual(record['entry']['selected_per_source'], [448]*8)
        self.assertEqual(routing.PRIMARY, [(routing.D, routing.A)])

    def test_explicit_fixed_v3_options_reject_missing_or_other_margins(self):
        self.assertEqual(planner.frozen_options(self.plans), {'uncertainty_multiplier': 1.0})
        for options in (None, {'uncertainty_multiplier': 4}, {'uncertainty_multiplier': 6}):
            bad = dict(self.plans, arrival_options=options)
            with self.assertRaises(ValueError):
                planner.validate(bad)

    def test_plan_identity_diagnostics_and_mask_drift_are_rejected(self):
        for field, value in (('method', 'receiver-tail-assignment-v2'),
                             ('accepted_windows', 1), ('identity_fallback', False)):
            bad = copy.deepcopy(self.plans)
            order = bad['orders']['attn']
            order['entry']['arrival_order'][field] = value
            order['entry_sha256'] = routing.digest(order['entry'])
            with self.subTest(field=field), self.assertRaises(ValueError):
                planner.validate(bad)
        bad = copy.deepcopy(self.plans)
        role = routing.role_plans(bad, routing.D)['mlp']
        role['entry']['mask'][0][64] = 0
        role['entry_sha256'] = routing.digest(role['entry'])
        with self.assertRaises(ValueError):
            planner.validate(bad)

    def test_all_fused_routes_are_single_buffered_and_cloned(self):
        for policy in routing.POLICIES:
            d = routing.definition(policy)
            self.assertTrue(routing.is_fused_v2(policy))
            self.assertFalse(routing.is_native(policy))
            self.assertEqual(d['workspace'], 'single')
            self.assertTrue(d['output_clone'] and d['publication_barrier'] and d['reuse_barrier'])
        self.assertEqual(routing.definition(routing.D)['order'], 'base')
        self.assertEqual(routing.definition(routing.R)['order'], 'arrival')

    def test_balanced_schedule_and_round_reversal(self):
        jobs = routing.schedule()
        self.assertEqual(len(jobs), 36)
        first, second = jobs[:18], jobs[18:]
        self.assertEqual([j['policy'] for j in second], [j['policy'] for j in first][::-1])
        for block in range(6):
            self.assertEqual({j['policy'] for j in first if j['block'] == block}, set(routing.POLICIES))
        for position in range(3):
            self.assertEqual({p: sum(j['policy'] == p and j['index'] % 3 == position for j in first)
                              for p in routing.POLICIES}, {p: 2 for p in routing.POLICIES})

    def test_acceptance_does_not_use_r_to_rescue_d(self):
        key = routing.D+'/'+routing.A
        def check(point, lower, upper=2.0):
            return reporter.acceptance(dict(comparisons={
                key: dict(percent=point, ci95=[lower, upper]),
                routing.R+'/'+routing.A: dict(percent=20, ci95=[19, 21])}))
        self.assertFalse(check(1.6, 1.49)['passed'])
        self.assertFalse(check(1.49, 1.5)['passed'])
        self.assertTrue(check(1.5, 1.5)['passed'])
        self.assertTrue(check(2.0, 1.7, 2.3)['passed'])
        self.assertFalse(reporter.acceptance(dict(comparisons={key: dict(percent=2, ci95=None)}))['passed'])

    def test_acceptance_rejects_invalid_statistics(self):
        key = routing.D+'/'+routing.A
        for value in (True, '2', float('nan'), float('inf')):
            with self.subTest(point=value), self.assertRaises(ValueError):
                reporter.acceptance(dict(comparisons={key: dict(percent=value, ci95=[1.6, 2.1])}))
        for interval in ([2, 1], [1], [1, 2, 3], [True, 2], ['1', 2],
                         [1, float('nan')], [1, float('inf')], '1,2'):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                reporter.acceptance(dict(comparisons={key: dict(percent=2, ci95=interval)}))

    def test_generated_worker_is_gated_before_torch_and_valid_python(self):
        for source in (sources.adapter_source(), sources.taco_source(),
                       sources.worker_source(), sources.worker_source(True)):
            ast.parse(source)
        with tempfile.TemporaryDirectory(prefix='op2-worker-CPU-') as temporary:
            worker = Path(temporary)/'worker.py'
            worker.write_text(sources.worker_source())
            env = dict(os.environ, H100_ABLATION4_EXECUTE='1', H100_ABLATION7_NEW_EXECUTE='1')
            env.pop('H100_ARRIVAL_OP2_EXECUTE', None)
            result = subprocess.run([sys.executable, str(worker)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('GPU worker requires arrival_op2 --execute', result.stderr)
            worker.write_text(sources.worker_source(True))
            result = subprocess.run([sys.executable, str(worker)],
                                    env=dict(env, H100_ARRIVAL_OP2_EXECUTE='1'),
                                    capture_output=True, text=True)
            self.assertIn('Source-only arrival_op2 worker', result.stderr)


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='arrival-op2-CPU-')
        cls.root = Path(cls.temp.name)
        cls.parent = ablation4_build.prepare(cls.root/'parent', source_only=True)
        cls.parent_hash = sha(cls.parent/'ablation4-manifest.json')
        cls.build = builder.prepare(cls.root/'build', cls.parent, source_only=True)
        cls.compare = preparation.prepare(cls.root/'compare', cls.build, '/unused', source_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_only_decoder_and_optional_order_sources_differ(self):
        record = builder.verify(self.build)
        self.assertEqual(sha(self.parent/'ablation4-manifest.json'), self.parent_hash)
        fused = read(self.build/'parent-fused-manifest.json')
        base = fused['variants']['interleaved_all']['source_sha256']
        d = record['improved_source_sha256']['decode_interleaved']
        r = record['improved_source_sha256']['decode_arrival']
        self.assertEqual({name for name in set(base) | set(d) if base.get(name) != d.get(name)},
                         {builder.HEADER, builder.RUNTIME})
        self.assertEqual(d, r)  # CPU fixture intentionally falls back to identity.
        self.assertEqual(builder.runtime_root(self.build, routing.A), self.parent/'fused/interleaved_all')
        self.assertFalse((builder.runtime_root(self.build, routing.D)/'libflux_cuda.so').exists())
        with self.assertRaises(ValueError):
            builder.verify(self.build, built=True)

    def test_prepare_and_dry_run_cannot_execute_cpu_fixture(self):
        info = reporter.verify_inputs(self.compare)
        self.assertEqual(info['timing_windows'], 36)
        self.assertEqual(info['budget_fractions'], {p: 1 for p in routing.POLICIES})
        self.assertFalse(read(self.compare/'scripts/config.json')['double_buffered'])
        with mock.patch.object(subprocess, 'run', side_effect=AssertionError('GPU invocation')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(len(cli.run(self.compare)), 42)
            self.assertEqual(len(list(cli.mapping_jobs(self.compare, info))), 6)
        for function in (reporter.report, reporter.preflight):
            with self.assertRaisesRegex(ValueError, 'CPU rehearsal'):
                function(self.compare, offline=True)

    def test_source_and_schedule_drift_rejected(self):
        path = self.compare/'submission.json'; saved = read(path)
        try:
            changed = copy.deepcopy(saved)
            changed['schedule'][0]['policy'] = routing.R
            # Ensure an actual change independently of initial schedule position.
            changed['timing_windows'] -= 1
            write(path, changed)
            with self.assertRaises(ValueError):
                reporter.verify_inputs(self.compare, offline=True)
        finally:
            write(path, saved)
        path = self.compare/'scripts/worker.py'; saved_source = path.read_text()
        try:
            path.write_text(saved_source+'\n')
            with self.assertRaises(ValueError):
                reporter.verify_inputs(self.compare, offline=True)
        finally:
            path.write_text(saved_source)

    def decoder_fixture(self, root):
        manifest = read(root/'provenance/build-manifest.json')
        digest = manifest['improved_source_sha256']['decode_interleaved'][builder.HEADER]
        result = dict(protocol=routing.PROTOCOL, completed=True, gpu_executed=True,
                      correctness_passed=True, measured=dict(bitwise_equal=True),
                      source_sha256={'/CPU-FIXTURE/arrival_op2/decoder.cuh': digest},
                      synthetic=True)
        write(root/'decoder-validation.json', result)
        return result

    def test_decoder_gate_rejects_compile_only_and_wrong_source(self):
        root = self.compare
        original = self.decoder_fixture(root)
        reporter.decoder_validation(root)
        try:
            for field in ('completed', 'gpu_executed', 'correctness_passed'):
                write(root/'decoder-validation.json', dict(original, **{field: False}))
                with self.assertRaises(ValueError):
                    reporter.decoder_validation(root)
            wrong = copy.deepcopy(original)
            wrong['source_sha256']['/CPU-FIXTURE/arrival_op2/decoder.cuh'] = '0'*64
            write(root/'decoder-validation.json', wrong)
            with self.assertRaisesRegex(ValueError, 'source'):
                reporter.decoder_validation(root)
        finally:
            (root/'decoder-validation.json').unlink()

    def test_complete_synthetic_report_preserves_attribution_and_target(self):
        root = preparation.prepare(self.root/'synthetic', self.build, '/unused', source_only=True)
        with mock.patch.object(report_fixtures, 'routing', routing):
            info = report_fixtures.synthetic_evidence(root)
        # The reusable fixture emits the legacy decoder symbol. New D/R must
        # demonstrate the actual optimized symbol through launch correlations.
        for policy in (routing.D, routing.R):
            for path in (root/'results'/f'profile-{policy}').glob('*-profile.json.gz'):
                with gzip.open(path, 'rt') as handle:
                    data = json.load(handle)
                for event in data['traceEvents']:
                    if event.get('cat') == 'kernel' and 'taco_decode_ring_kernel' in event.get('name', ''):
                        event['name'] = event['name'].replace('taco_decode_ring_kernel', 'taco_decode_ring_kernel_op2')
                with gzip.open(path, 'wt') as handle:
                    json.dump(data, handle)
        self.decoder_fixture(root)
        with mock.patch.object(reporter, 'verify_inputs', return_value=info):
            gate = reporter.preflight(root, offline=True)
            result = reporter.report(root, offline=True)
        self.assertIn('decoder-validation.json', gate['files'])
        self.assertEqual(result['protocol'], routing.PROTOCOL)
        self.assertEqual(result['timing_windows'], 36)
        self.assertEqual(result['formal_optimizer_steps'], 1800)
        self.assertEqual(result['rounds']['combined']['blocks'], 12)
        self.assertEqual(result['acceptance_threshold'], 1.5)
        self.assertEqual(result['acceptance']['candidate'], routing.D)
        self.assertTrue(result['arrival_diagnostic_only'])
        self.assertEqual(set(result['acceptance_by_round']), {'1', '2', 'combined'})
        self.assertIn('R/D只衡量到达表增量', (root/'report.md').read_text())
        self.assertTrue(result['arrival_schedule']['attn']['identical_to_D'])


if __name__ == '__main__':
    unittest.main()
