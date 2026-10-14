"""One standard five-group suite; stop on busy GPUs or any failed gate."""
import collections
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

repo = Path(os.environ['REPO'])
out = Path(os.environ['H100_OUT'])
log = Path(os.environ['H100_LOG'])
compare = log / 'compare'
handoff = log / 'handoff'
entry = repo / 'tools/h100/phase6/ablation2.py'
sys.path.insert(0, str(entry.parent))
from common import read, sha, verify_files, write

zone = datetime.timezone(datetime.timedelta(hours=8))
stage = 'environment'
child = None
log.mkdir(parents=True, exist_ok=True)
handoff.mkdir(exist_ok=True)
lock = (log / 'session.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (log / 'execution.json').exists():
    raise RuntimeError('Session already started; preserve its records and use a new session')
controllers = {str(p): sha(p) for p in entry.parent.glob('*.py')}

def now():
    return datetime.datetime.now(zone).isoformat()

def state(status, detail):
    folders = list((compare / 'results').glob('*/round*/window-*'))
    value = dict(status=status, stage=stage, detail=detail, updated_beijing=now(),
                 pid=os.getpid(), child_pid=child.pid if child else None,
                 session=os.environ['H100_SESSION'], compare=str(compare),
                 policies=['native_taco', 'G1', 'G2', 'G3', 'G4'], bases=['interleaved'], rounds=2,
                 planned_windows=50, started_windows=len(folders),
                 completed_windows=sum(all((p / f'rank{r}.json').is_file() for r in range(8)) for p in folders),
                 gpu_monitoring='before_each_suite_only', auto_wait=False, auto_retry=False)
    temporary = log / 'execution.json.tmp'
    write(temporary, value)
    temporary.replace(log / 'execution.json')

def capture(name, args):
    text = subprocess.check_output([str(a) for a in args], cwd=repo, text=True)
    (handoff / name).write_text(text)
    return text.strip()

def run(args, name, env=None, cwd=repo):
    global child
    verify_files('/', controllers)
    print(now(), stage, 'START', name, flush=True)
    with (log / name).open('x') as stream:
        child = subprocess.Popen([str(a) for a in args], cwd=cwd, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
        state('running', name)
        while child.poll() is None:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            state('running', name)
        code = child.returncode
        child = None
    if code:
        raise RuntimeError(f'{name} exited {code}; see {log / name}')
    print(now(), stage, 'COMPLETE', name, flush=True)

def delivery(status, detail):
    results = compare / 'results'
    timing = list(results.glob('*/round*/window-*/rank[0-7].json'))
    preflight = list(results.glob('*/smoke-*/rank[0-7].json')) + list(results.glob('*/profile-*/rank[0-7].json'))
    traces = list(results.glob('*/profile-*/rank*-profile.json.gz'))
    mapping = list(results.glob('*/mapping-*.csv'))
    actual = dict(timing_rank=len(timing), preflight_rank=len(preflight), traces=len(traces), mappings=len(mapping))
    write(handoff / 'delivery.json', dict(status=status, detail=detail, actual=actual,
        expected=dict(timing_rank=400, preflight_rank=80, traces=40, mappings=4),
        code_commit=commit, session=os.environ['H100_SESSION'], updated_beijing=now(),
        strict_report=(compare / 'report.json').is_file(), data_return_commit=None))
    (handoff / 'README.md').write_text(
        f'# H100 interleaved five-group ablation\n\nSession: `{os.environ["H100_SESSION"]}`.\n\n'
        f'Status: **{status}**. {detail}. Updated Beijing: {now()}.\n\n'
        f'Code: `{commit}`; original worktree and patch are recorded in handoff. '
        'GPT 6.7B, 32 layers, H4096/FFN16384, 32 heads, S2048, mb1/global4, BF16, TP8. '
        'Five groups only: native_taco, G1, G2, G3, G4; interleaved; two rounds of five paired blocks. '
        '50 fresh windows, each 10 warmup + 20 full optimizer steps. No Attention extension.\n\n'
        f'Reused source build: `{os.environ["H100_REUSE"]}`; the four complete libraries stay on H100. '
        'The session candidates registry includes source references, hashes and all four plans. '
        'Compare preparation freezes baseline sources, controllers, worker, configuration and schedule.\n\n'
        f'Actual evidence counts: `{actual}`. Expected: 400 formal ranks, 80 preflight ranks, '
        '40 compressed traces and 4 mapping CSVs. Missing evidence is not filled with historical timing. '
        'Strict report and paired speedups/95% intervals exist only after successful GPU execution.\n\n'
        'Startup GPU availability is checked by ablation2.py run --stage all --execute. '
        'No automatic waiting, no repeated run, no other-task termination, no periodic runtime occupancy query. '
        'Failures are preserved. All-sample results are primary; whole-block slow-window sensitivity is separate. '
        'Libraries, object files, CMake/Ninja caches and Python caches remain ignored. '
        'Data return commit SHA: not created.\n')

commit = capture('code-commit.txt', ['git', 'rev-parse', 'HEAD'])
try:
    state('running', 'Recording local software and input identities')
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
    for p in entry.parent.glob('*.py'):
        dest = out / 'controller-source' / p.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(p, dest)
    stage = 'cpu-tests'
    run([sys.executable, '-m', 'unittest', 'discover', '-s', entry.parent, '-p', 'test_*.py', '-v'],
        'cpu-tests.log', env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    stage = 'frozen-inputs'
    freeze = Path(os.environ['H100_FREEZE'])
    run(['sha256sum', '--quiet', '-c', 'SHA256SUMS.txt'], 'freeze-check.log', cwd=freeze)
    run(['sha256sum', '--quiet', '-c', freeze / 'REFERENCES.sha256.txt'], 'references-check.log')
    stage = 'reuse-builds'
    run([sys.executable, entry, 'reuse', out / 'candidates', '--from-builds', os.environ['H100_REUSE']], 'reuse.log')
    data = read(out / 'candidates/ablation-manifest.json')
    assert data['bases'] == ['interleaved'] and data['seeds'] == [] and len(data['variants']) == 5
    from ablation_prepare import megatron_check
    from runtime import check_topology
    import torch
    megatron_check(Path(os.environ['H100_MEGATRON']))
    os.environ.update(H100_TP='8', H100_SM_COUNT='132', LOCAL_RANK='0')
    hardware = check_topology(8)
    assert hardware == data['calibration_hardware'], 'Parent calibration host/GPU/SM/peer identity mismatch'
    software = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                    nccl=list(torch.cuda.nccl.version()), python=sys.executable)
    previous = read(repo / 'logs/h100/phase6/ablation-gpt67-tp8-20261013-233551/compare/results/interleaved/smoke-G4/rank0.json')
    for k, v in software.items():
        assert previous[k] == v, (k, previous[k], v)
    library_evidence = {}
    for key in ('G1', 'G2', 'G3', 'G4'):
        v = data['variants']['interleaved-' + key]
        root = Path(v['build'])
        manifest = read(root / 'manifest.json')
        verify_files('/', manifest['nccl_files'])
        library_evidence[key] = dict(manifest_sha256=sha(root / 'manifest.json'),
                                     nccl_files=manifest['nccl_files'], files=manifest['files'])
    env = {k: v for k, v in os.environ.items() if k.startswith('NCCL_') or k in (
        'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_MAX_CONNECTIONS', 'PYTORCH_CUDA_ALLOC_CONF',
        'PYTORCH_ALLOC_CONF', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CC', 'CXX', 'PYTHONNOUSERSITE')}
    write(handoff / 'environment.json', dict(hardware=hardware, software=software, environment=env,
                                           code_commit=commit, controller_hashes=controllers,
                                           libraries=library_evidence))
    stage = 'prepare-compare'
    run([sys.executable, entry, 'prepare', compare, '--builds', out / 'candidates',
         '--megatron', os.environ['H100_MEGATRON'], '--blocks', '5', '--schedule-seed', '20261013'], 'prepare.log')
    info = read(compare / 'submission.json')
    assert info['timing_windows'] == 50 and len(info['schedule']) == 50
    assert info['bases'] == ['interleaved'] and {v['policy'] for v in info['variants'].values()} == {
        'native_taco', 'G1', 'G2', 'G3', 'G4'}
    assert collections.Counter((j['round'], j['policy']) for j in info['schedule']) == {
        (r, p): 5 for r in (1, 2) for p in ('native_taco', 'G1', 'G2', 'G3', 'G4')}
    stage = 'dry-run'
    run([sys.executable, entry, 'run', compare], 'dry-run.log')
    delivery('prepared', 'Validated five groups and 50 windows; GPU execution not yet passed')
    stage = 'preflight-and-timing'
    run([sys.executable, entry, 'run', compare, '--stage', 'all',
         '--cuda-home', os.environ['H100_CUDA_RUN'], '--execute'], 'run-all.log')
    stage = 'strict-report'
    run([sys.executable, entry, 'report', compare], 'report.log')
    result = read(compare / 'report.json')
    assert result['completed'] and result['timing_rank_records'] == 400 and len(result['all_samples']) == 50
    state('completed', 'Strict report passed; 50 fresh full optimizer-step windows')
    delivery('complete', 'All preflight, timing and strict evidence checks passed')
    print(now(), 'COMPLETE', compare, flush=True)
except BaseException as error:
    state('failed', str(error))
    delivery('failed', str(error))
    raise
