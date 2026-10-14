"""CPU-only validation of the preregistered remote-first single-round protocol."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import ablation_prepare as prep
import ablation_report as report
import ablation_run as runner
from common import read, write
from report import paired
from test_ablation import make_window

BUILDS = Path('/workspace/TempName/outputs/h100/phase6/ablation-gpt67-tp8-20261013-233551/candidates')


class SingleRoundTests(unittest.TestCase):
    def test_real_built_input_dry_run_has_100_windows_and_only_remote_round1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = prep.prepare(Path(tmp) / 'compare', BUILDS, Path('/workspace/Megatron-LM'),
                                bases=['remote_first'], rounds=1)
            info = report.verify_inputs(root)
            self.assertEqual(info['timing_windows'], 100)
            self.assertEqual(len(info['variants']), 10)
            self.assertEqual({(j['base'], j['round']) for j in info['schedule']}, {('remote_first', 1)})
            with mock.patch.object(runner.subprocess, 'run', side_effect=AssertionError('GPU launched')):
                with contextlib.redirect_stdout(io.StringIO()):
                    tasks = runner.run(root)
            self.assertEqual(len(tasks), 120)
            self.assertFalse((root / 'scripts/interleaved').exists())

    def test_strict_single_round_report_retains_samples_and_rejects_missing_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = prep.prepare(Path(tmp) / 'synthetic', BUILDS, Path('/absent'), blocks=2,
                                source_only=True, bases=['remote_first'], rounds=1)
            info = read(root / 'submission.json')
            info.update(source_only=False, executable=True)
            for v in info['variants'].values():
                if v['group'] != 'native':
                    v['libraries'] = {name: dict(path='/CPU-FIXTURE/' + name, sha256='f' * 64)
                                      for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
            write(root / 'submission.json', info)
            for job in runner.jobs(info, 'all'):
                make_window(root, info, job)
            entries = read(root / 'scripts/remote_first/selection-plan.json')['entries']
            for key, entry in entries.items():
                with (root / f'results/remote_first/mapping-{key}.csv').open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['rank', 'index', 'm', 'n', 'destination', 'sm_occupancy'])
                    writer.writeheader()
                    for rank, row in enumerate(entry['coords']):
                        for index, tile in enumerate(row):
                            writer.writerow(dict(rank=rank, index=index, m=tile // 32, n=tile % 32,
                                                 destination=tile // 64, sm_occupancy=2))
            report.preflight(root, offline=True)
            with mock.patch.object(report, 'paired', side_effect=lambda x: paired(x, resamples=100)):
                result = report.report(root, offline=True)
            self.assertTrue(result['completed'])
            self.assertEqual(len(result['all_samples']), 20)
            self.assertEqual(set(result['bases']['remote_first']), {'1', 'combined'})
            first = info['schedule'][0]
            (root / 'results' / first['directory'] / 'rank7.json').unlink()
            with self.assertRaises(FileNotFoundError):
                report.report(root, offline=True)
