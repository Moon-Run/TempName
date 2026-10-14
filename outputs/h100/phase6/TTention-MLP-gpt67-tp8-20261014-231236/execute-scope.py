"""One strict Attention/MLP campaign with new calibration of both full shapes."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

repo = Path(os.environ['REPO'])
out = Path(os.environ['H100_SCOPE_OUT'])
log = Path(os.environ['H100_SCOPE_LOG'])
handoff = log / 'handoff'
compare = log / 'compare'
calibration = log / 'calibration'
entry = repo / 'tools/h100/phase6/mlp_attention.py'
sys.path.insert(0, str(entry.parent))
from common import read, sha, verify_files, write
from scope_protocol import POLICIES, SHAPES
from scope_plan import hardware_identity, model_config, validate

zone = datetime.timezone(datetime.timedelta(hours=8))
stage = 'environment'
child = None
commit = None
log.mkdir(parents=True, exist_ok=True)
handoff.mkdir(exist_ok=True)
suite_lock = open('/tmp/tempname-h100-phase6-repeat-queue.lock', 'a')
fcntl.flock(suite_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (log / 'execution.json').exists():
    raise RuntimeError('Campaign already started; preserve existing records')
controllers = {str(p): sha(p) for p in entry.parent.glob('*.py')}

def now():
    return datetime.datetime.now(zone).isoformat()

def state(status, detail):
    folders = list((compare / 'results').glob('round*/window-*'))
    temp = log / 'execution.json.tmp'
    write(temp, dict(status=status, stage=stage, detail=detail, updated_beijing=now(),
                    pid=os.getpid(), child_pid=child.pid if child else None,
                    session=os.environ['H100_SCOPE_SESSION'], compare=str(compare),
                    calibration=str(calibration), calibration_roles=['attn', 'mlp'],
                    policies=list(POLICIES), planned_windows=32, started_windows=len(folders),
                    completed_windows=sum(all((p / f'rank{r}.json').is_file() for r in range(8)) for p in folders),
                    gpu_monitoring='before_each_gpu_suite_only', auto_wait=False, auto_retry=False))
    temp.replace(log / 'execution.json')

def capture(name, args):
    result = subprocess.check_output([str(x) for x in args], cwd=repo, text=True)
    (handoff / name).write_text(result)
    return result.strip()

def run(args, name, env=None, cwd=repo):
    global child
    verify_files('/', controllers)
    print(now(), stage, 'START', name, flush=True)
    with (log / name).open('x') as stream:
        child = subprocess.Popen([str(x) for x in args], cwd=cwd, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
        write(handoff / (name + '-command.json'), dict(stage=stage, args=[str(x) for x in args],
              cwd=str(cwd), python=sys.executable, started_beijing=now()))
        while child.poll() is None:
            state('running', name)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        code = child.returncode
        child = None
    if code:
        raise RuntimeError(f'{name} exited {code}; see {log / name}')
    print(now(), stage, 'COMPLETE', name, flush=True)

def delivery(status, detail):
    results = compare / 'results'
    actual = dict(timing_rank=len(list(results.glob('round*/window-*/rank[0-7].json'))),
                  preflight_rank=len(list(results.glob('smoke-*/rank[0-7].json'))) +
                                 len(list(results.glob('profile-*/rank[0-7].json'))),
                  profile_traces=len(list(results.glob('profile-*/rank*-profile.json.gz'))),
                  candidate_mappings=len(list(results.glob('mapping-*.csv'))),
                  calibration_rank=len(list(calibration.glob('b0-interleaved-instrumented-rank[0-7].json'))),
                  calibration_mappings=len(list(calibration.glob('mapping-*.csv'))))
    expected = dict(timing_rank=256, preflight_rank=64, profile_traces=32, candidate_mappings=3,
                    calibration_rank=8, calibration_mappings=2)
    write(handoff / 'delivery.json', dict(status=status, detail=detail, actual=actual, expected=expected,
          code_commit=commit, session=os.environ['H100_SCOPE_SESSION'], updated_beijing=now(),
          strict_report=(compare / 'report.json').exists(), data_return_commit=None))
    (handoff / 'README.md').write_text(
        f'# H100 Attention/MLP scope experiment\n\nSession: `{os.environ["H100_SCOPE_SESSION"]}`. '
        f'Updated Beijing: {now()}.\n\nStatus: **{status}**. {detail}.\n\n'
        f'Maintenance source commit: `{commit}`. Initial environment, submodules, source hashes, worktree '
        'and patch are in this handoff directory; all exact commands are saved as command JSON. '
        'Measured compute source is frozen candidate v1 from ff38f5b1ff1c80cb831f58c64b95f42b0379879a.\n\n'
        'GPT 6.7B: 32 layers, H4096/FFN16384, 32 heads, S2048, micro1/global4, BF16, TP8. '
        'Attention shape [2048,4096,4096], local K512; MLP shape [2048,4096,16384], local K2048. '
        'Both roles receive NEW real H100 BF16 calibration in this session, training passes 0/1, heldout pass2. '
        'No old MLP-only library, source-only build or synthetic plan is used.\n\n'
        f'Four policies: `{list(POLICIES)}`. Attention-only groups keep native Megatron BF16 MLP. '
        'Both candidates share one newly built dual-shape library and exactly the same Attention plan. '
        'Four paired blocks per round, two rounds, 32 new windows, each 10 warmup and 20 full optimizer steps.\n\n'
        f'Actual evidence: `{actual}`; expected: `{expected}`. Calibration ranks contain both full shapes. '
        'The whole calibration directory, both measured plans, source snapshots, build manifests, scripts, '
        'schedule, profile traces, command/process logs and report are preserved. '
        'Strict report is generated on H100 without offline mode. All samples are primary; slow-window '
        'sensitivity uses a separate whole-block table and keeps negative results.\n\n'
        'GPU checks occur before calibration and before the comparison only. No Slurm, waiting queue, '
        'periodic runtime occupancy query, automatic retry or independent repeat. Any failed stage stops. '
        'Historical builds, baselines and data are unchanged. Libraries, object files and caches remain local '
        'and ignored by Git. Data-return commit SHA: not created.\n')

try:
    state('running', 'Recording current environment and frozen input identities')
    commit = capture('code-commit.txt', ['git', 'rev-parse', 'HEAD'])
    capture('worktree-before.txt', ['git', 'status', '--porcelain'])
    capture('worktree-before.patch', ['git', 'diff', 'HEAD', '--binary'])
    capture('submodules.txt', ['git', 'submodule', 'status'])
    capture('pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze'])
    for name, args in {
        'nvcc-build.txt': [Path(os.environ['H100_CUDA_BUILD']) / 'bin/nvcc', '--version'],
        'nvcc-run.txt': [Path(os.environ['H100_CUDA_RUN']) / 'bin/nvcc', '--version'],
        'gcc.txt': [os.environ['CXX'], '--version'], 'cmake.txt': ['cmake', '--version'],
        'megatron-commit.txt': ['git', '-C', os.environ['H100_MEGATRON'], 'rev-parse', 'HEAD'],
        'megatron-worktree.txt': ['git', '-C', os.environ['H100_MEGATRON'], 'status', '--porcelain', '--untracked-files=no'],
        'apex-commit.txt': ['git', '-C', os.environ['APEX_SRC'], 'rev-parse', 'HEAD'],
        'apex-worktree.txt': ['git', '-C', os.environ['APEX_SRC'], 'status', '--porcelain'],
        'apex.patch': ['git', '-C', os.environ['APEX_SRC'], 'diff', 'HEAD', '--binary'],
        'topology.txt': ['nvidia-smi', 'topo', '-m'],
    }.items():
        capture(name, args)
    from ablation_prepare import megatron_check
    from runtime import check_topology
    from scope_calibrate import startup
    megatron_check(Path(os.environ['H100_MEGATRON']))
    startup(handoff, os.environ['H100_SCOPE_SESSION'])
    import torch
    import apex_C, amp_C, fused_layer_norm_cuda
    from apex.optimizers import FusedAdam
    from apex.normalization import FusedLayerNorm
    os.environ.update(H100_TP='8', H100_SM_COUNT='132', LOCAL_RANK='0')
    hardware = check_topology(8)
    prior_hw = read(repo / 'logs/h100/phase6/ablation2-gpt67-tp8-interleaved-20261014-221952/handoff/environment.json')['hardware']
    assert hardware == prior_hw, 'Allocated host/GPU order differs from the previous assigned devices'
    software = dict(python=sys.executable, python_version=sys.version, torch=str(torch.__version__),
                    cuda=torch.version.cuda, nccl=list(torch.cuda.nccl.version()))
    selected_env = {k: v for k, v in os.environ.items() if k.startswith('NCCL_') or k in (
        'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_MAX_CONNECTIONS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'CC', 'CXX', 'PYTHONNOUSERSITE')}
    write(handoff / 'environment.json', dict(hardware=hardware, software=software, environment=selected_env,
          code_commit=commit, controller_hashes=controllers, model=model_config()))
    snapshot = out / 'controller-source'
    snapshot.mkdir(exist_ok=False)
    for p in entry.parent.glob('*.py'):
        shutil.copy2(p, snapshot / p.name)
    stage = 'cpu-tests'
    run([sys.executable, '-m', 'unittest', 'discover', '-s', entry.parent, '-p', 'test_*.py', '-v'],
        'cpu-tests.log', env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    stage = 'frozen-inputs'
    freeze = Path(os.environ['H100_FREEZE'])
    run(['sha256sum', '--quiet', '-c', 'SHA256SUMS.txt'], 'freeze-check.log', cwd=freeze)
    run(['sha256sum', '--quiet', '-c', freeze / 'REFERENCES.sha256.txt'], 'references-check.log')
    delivery('partial', 'Environment and CPU checks passed; new calibration build starting')
    stage = 'calibration-build'
    run([sys.executable, entry, 'calibration-build', out / 'calibration', '--roles', 'attn', 'mlp'], 'calibration-build.log')
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    stage = 'calibration-compile'
    run([sys.executable, entry, 'compile', out / 'calibration', '--cuda-home', os.environ['H100_CUDA_BUILD'],
         '--jobs', '2', '--nccl-root', os.environ['H100_NCCL']], 'calibration-compile.log', env=cpu_env)
    stage = 'calibration-prepare'
    run([sys.executable, entry, 'calibration-prepare', calibration, '--build', out / 'calibration'], 'calibration-prepare.log')
    stage = 'calibration-dry-run'
    run([sys.executable, entry, 'calibration-run', calibration, '--cuda-home', os.environ['H100_CUDA_RUN']], 'calibration-dry-run.log')
    stage = 'calibration-gpu'
    run([sys.executable, entry, 'calibration-run', calibration, '--cuda-home', os.environ['H100_CUDA_RUN'], '--execute'], 'calibration-run.log')
    done = read(calibration / 'calibration-complete.json')
    assert done['completed'] and done['roles'] == ['attn', 'mlp']
    assert hardware_identity(done['hardware']) == hardware_identity(hardware)
    stage = 'fit-measured-plans'
    plans = out / 'plans/measured'
    run([sys.executable, entry, 'fit', calibration, plans], 'fit.log')
    for role, shape in SHAPES.items():
        record = read(plans / (role + '.json'))
        validate(record, role)
        assert record['gpu_calibrated'] and not record['test_only'] and record['entry']['shape'] == shape
        assert record['origin']['kind'] == 'scope_bf16_calibration'
        assert record['origin']['source'] == str(calibration)
    stage = 'candidate-build'
    run([sys.executable, entry, 'build-prepare', out / 'candidates',
         '--attn-plan', plans / 'attn.json', '--mlp-plan', plans / 'mlp.json'], 'candidate-build.log')
    stage = 'candidate-compile'
    run([sys.executable, entry, 'compile', out / 'candidates', '--cuda-home', os.environ['H100_CUDA_BUILD'],
         '--jobs', '2', '--nccl-root', os.environ['H100_NCCL']], 'candidate-compile.log', env=cpu_env)
    stage = 'compare-prepare'
    run([sys.executable, entry, 'prepare', compare, '--build', out / 'candidates',
         '--megatron', os.environ['H100_MEGATRON'], '--blocks', '4'], 'prepare.log')
    info = read(compare / 'submission.json')
    assert info['executable'] and not info['source_only'] and info['timing_windows'] == 32
    assert set(info['variants']) == set(POLICIES)
    candidates = [v for v in info['variants'].values() if 'root' in v]
    assert len(candidates) == 2 and candidates[0]['root'] == candidates[1]['root']
    assert candidates[0]['libraries'] == candidates[1]['libraries']
    stage = 'compare-dry-run'
    run([sys.executable, entry, 'run', compare], 'dry-run.log')
    stage = 'preflight-and-timing'
    run([sys.executable, entry, 'run', compare, '--stage', 'all',
         '--cuda-home', os.environ['H100_CUDA_RUN'], '--execute'], 'run-all.log')
    stage = 'strict-report'
    run([sys.executable, entry, 'report', compare], 'report.log')
    report = read(compare / 'report.json')
    assert report['completed'] and len(report['all_samples']) == 32
    assert report['timing_rank_records'] == 256 and report['preflight_rank_records'] == 64
    assert report['profile_traces'] == 32
    delivery('complete', 'Both new calibrations and all four scopes passed strict report verification')
    inventory = read(handoff / 'delivery.json')
    assert inventory['actual'] == inventory['expected'], 'Missing delivery artifacts'
    state('completed', 'Strict report passed; all 32 fresh windows and delivery artifacts present')
    print(now(), 'COMPLETE', compare, flush=True)
except BaseException as error:
    state('failed', str(error))
    delivery('failed', str(error))
    raise
