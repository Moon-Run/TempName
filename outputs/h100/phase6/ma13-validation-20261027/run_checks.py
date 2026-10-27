"""Reproduce local MA13 validation with the pinned torch environment, no GPU required."""
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / 'tools/h100/phase6'
OUT = Path(__file__).resolve().parent
# -I -S avoids user site startup hooks; add this interpreter's own packages only.
sys.path[:0] = [str(HERE), str(Path(sys.executable).resolve().parents[1] / 'lib/python3.10/site-packages')]
os.environ.setdefault('NVCC', '/data/apps/cuda/12.8/bin/nvcc')
import torch

loader = unittest.TestLoader()
suite = unittest.TestSuite([loader.discover(str(HERE), pattern='test_ma13_*.py'),
                           loader.loadTestsFromName('test_ma11_adapter'),
                           loader.loadTestsFromName('test_ma11_harness')])
with (OUT / 'tests.log').open('w') as log:
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    record = dict(date='2026-10-27', python=sys.executable, torch=torch.__version__,
                  cuda_available=torch.cuda.is_available(), cuda_device_count=torch.cuda.device_count(),
                  tests=dict(total=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                             skipped=len(result.skipped), passed=result.wasSuccessful()),
                  command=[sys.executable, '-I', '-S', str(Path(__file__).resolve())],
                  scope='all test_ma13_*.py plus original MA11 adapter/harness regressions',
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(HERE.glob('*ma13*')) if p.is_file()},
                  gpu_kernel_executed=False, gpu_numerical_validated=False,
                  full_flux_linked=False, h100_eight_rank_performance_tested=False,
                  performance_target_percent=2.0, performance_target_validated=False)
    (OUT / 'validation.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record['tests'], indent=2), flush=True)
raise SystemExit(not result.wasSuccessful())
