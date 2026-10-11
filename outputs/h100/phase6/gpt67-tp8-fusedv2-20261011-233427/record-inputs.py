"""Record actual provenance and verify unchanged candidate/calibration reuse."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

REPO = Path(os.environ['REPO'])
sys.path.insert(0, str(REPO / 'tools/h100/phase6'))
from common import config, manifest, plan, read, sha, verify_files, write
from runtime import check_topology

log = Path(os.environ['H100_LOG'])
handoff = log / 'handoff'
handoff.mkdir(parents=True, exist_ok=True)

def capture(name, command):
    value = subprocess.check_output(command, text=True, cwd=REPO)
    (handoff / name).write_text(value)
    return value.strip()

commit = capture('commit.txt', ['git', 'rev-parse', 'HEAD'])
status = capture('worktree-before.txt', ['git', 'status', '--porcelain'])
capture('worktree-before.patch', ['git', 'diff', 'HEAD', '--binary'])
capture('submodules.txt', ['git', 'submodule', 'status'])
capture('nvidia-smi-before.txt', ['nvidia-smi'])
capture('topology.txt', ['nvidia-smi', 'topo', '-m'])
capture('pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze'])
capture('cuda128.txt', ['/usr/local/cuda-12.8/bin/nvcc', '--version'])
capture('cuda124.txt', ['/usr/local/cuda-12.4/bin/nvcc', '--version'])
capture('gcc.txt', ['/usr/bin/g++-11', '--version'])
capture('cmake.txt', ['cmake', '--version'])
megatron = capture('megatron-commit.txt', ['git', '-C', os.environ['MEGATRON_DIR'], 'rev-parse', 'HEAD'])
apex = capture('apex-commit.txt', ['git', '-C', os.environ['APEX_SRC'], 'rev-parse', 'HEAD'])
capture('apex-worktree.txt', ['git', '-C', os.environ['APEX_SRC'], 'status', '--porcelain'])
capture('apex.patch', ['git', '-C', os.environ['APEX_SRC'], 'diff', 'HEAD', '--binary'])

c = config(Path(os.environ['H100_CONFIG']))
candidate_path = Path(os.environ['H100_CANDIDATES'])
plan_path = Path(os.environ['H100_PLAN'])
candidate = manifest(candidate_path, 'candidates')
measured = plan(plan_path, c)
assert candidate['config'] == c
assert candidate['plan_sha256'] == sha(plan_path)
for name, variant in candidate['variants'].items():
    verify_files(candidate_path / name, variant['source_sha256'])

frozen = REPO / 'tools/h100/phase6/fused_v2'
provenance = read(frozen / 'provenance.json')
assert provenance['implementation'] == 'a800-fused-v2-cta-v1'
verify_files(frozen, provenance['files'])

os.environ.update(H100_TP=str(c['tp']), H100_SM_COUNT=str(c['sm_count']), LOCAL_RANK='0')
hardware = check_topology(c['tp'])
assert hardware['host'] == measured['hardware']['host']
assert hardware['devices'] == measured['hardware']['devices']
calibration = REPO / 'logs/h100/phase6/gpt67-tp8-20261010-112249/calibration-20261010-153517'
complete = read(calibration / 'calibration-complete.json')
assert complete['completed'] and complete['config'] == c
assert complete['hardware'] == hardware
calibration_base = manifest(Path(complete['build']), 'baselines')
assert sha(Path(complete['build']) / 'manifest.json') == complete['build_sha256']
for name, digest in calibration_base['nccl_files'].items():
    assert sha(name) == digest

old_compare = REPO / 'logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat'
software = dict(torch=torch.__version__, cuda=torch.version.cuda,
                nccl=list(torch.cuda.nccl.version()), python=sys.executable)
for rank in range(c['tp']):
    old = read(old_compare / f'results/smoke-native/rank{rank}.json')
    for key, value in software.items():
        assert old[key] == value, (key, old[key], value)
    assert old['gpu_uuid'] == hardware['devices'][rank]['uuid']

environment = {key: value for key, value in os.environ.items()
               if key.startswith('NCCL_') or key in (
                   'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_MAX_CONNECTIONS',
                   'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'OMP_NUM_THREADS',
                   'MKL_NUM_THREADS', 'CC', 'CXX', 'CUDA124', 'CUDA128', 'PYTHONNOUSERSITE')}
data = dict(
    recorded_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    commit=commit, worktree_status=status, megatron_commit=megatron, apex_commit=apex,
    implementation=provenance['implementation'], config=c, hardware=hardware, software=software,
    environment=environment, reuse_verified=True,
    candidate_root=str(candidate_path), candidate_manifest_sha256=sha(candidate_path / 'manifest.json'),
    plan=str(plan_path), plan_sha256=sha(plan_path), calibration=str(calibration),
    calibration_complete_sha256=sha(calibration / 'calibration-complete.json'),
    candidate_file_hashes=candidate['files'], plan_input_hashes=measured['inputs'],
    nccl_files=calibration_base['nccl_files'],
    reuse_reason='Exact config, source snapshots, library hashes, measured plan inputs, host, GPU order, peer access and runtime software matched; candidate/plan unchanged.',
)
write(handoff / 'reuse-verification.json', data)
print('REUSE VERIFIED; implementation:', provenance['implementation'])
print('session:', os.environ['H100_SESSION'])
