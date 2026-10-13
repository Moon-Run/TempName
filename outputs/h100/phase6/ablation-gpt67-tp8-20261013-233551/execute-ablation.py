"""One independent ablation campaign; failures stop without retries or GPU polling."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

repo = Path(os.environ['REPO'])
out = Path(os.environ['H100_OUT'])
logs = Path(os.environ['H100_LOG'])
handoff = logs / 'handoff'
compare = logs / 'compare'
builds = out / 'candidates'
tools = repo / 'tools/h100/phase6'
sys.path.insert(0, str(tools))
from common import read, sha, verify_files

zone = datetime.timezone(datetime.timedelta(hours=8))
logs.mkdir(parents=True, exist_ok=True)
handoff.mkdir(exist_ok=True)
lock = open('/tmp/tempname-h100-phase6-repeat-queue.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (logs / 'execution.json').exists():
    raise RuntimeError('Campaign already started; preserve existing state and results')
stage = 'cpu-tests'
child = None
controller_hashes = {str(p): sha(p) for p in tools.glob('*.py')}

def now():
    return datetime.datetime.now(zone).isoformat()

def state(status, detail, **extra):
    windows = list((compare / 'results').glob('*/round*/window-*'))
    manifests = list((builds / 'builds').glob('*/manifest.json'))
    built = sum(read(p)['status'] == 'built' for p in manifests)
    data = dict(status=status, stage=stage, detail=detail, updated_beijing=now(),
                pid=os.getpid(), child_pid=child.pid if child else None,
                session=os.environ['H100_SESSION'], compare=str(compare),
                built_candidates=built, planned_candidates=18,
                started_windows=len(windows),
                completed_windows=sum(all((p / f'rank{r}.json').is_file() for r in range(8)) for p in windows),
                planned_windows=400, gpu_monitoring='before_run_only', **extra)
    temp = logs / 'execution.json.tmp'
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(logs / 'execution.json')

def note(status, detail):
    (handoff / 'README.md').write_text(
        f'# H100 GPT 6.7B TP8 ablation\n\nSession: `{os.environ["H100_SESSION"]}`.\n\n'
        f'Status: **{status}**; {detail}. Updated Beijing: {now()}.\n\n'
        'One campaign only: remote_first/interleaved; native + G0-G6; G5 seeds 17/29/43; '
        '10 blocks, schedule seed 20261013; 10 warmup + 20 full optimizer steps; 400 formal windows.\n\n'
        f'Source commit: `{subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}`. '
        'Initial source/environment/patch evidence is in this handoff directory; actual source snapshots and '
        'derived plans are in the session candidates directory. Parent freeze: h100-phase6-candidate-v1-20261013.\n\n'
        'Original builds, fixed baselines and historical evidence are unchanged. Parent calibration reuse '
        'is allowed only after full local source/plan/library/environment/host/GPU-order verification. '
        'Each new candidate is built independently with CUDA 12.8 and executed using CUDA 12.4.\n\n'
        'GPU occupancy is checked before GPU execution only; no periodic resource queries during '
        'preflight/timing or on failure. Occupancy sampling during execution is disabled; '
        'exclusive GPU use throughout measurement is not asserted.\n\n'
        'No retries, slow-window deletion, independent repeat, or offline report substitution. '
        'Compiled libraries and caches remain on the H100 host. Data return commit SHA: pending; '
        'this background campaign does not publish Git commits. Final strict results: compare/report.json '
        'and compare/report.md when complete; otherwise retain all partial records and stage logs.\n')

def run(args, name, env=None, cwd=repo):
    global child
    verify_files('/', controller_hashes)
    print(now(), stage, 'START', name, flush=True)
    with (logs / name).open('x') as stream:
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
        raise RuntimeError(f'{name} exited {code}; see {logs / name}')
    print(now(), stage, 'COMPLETE', name, flush=True)

def idle_gate():
    expected = read(handoff / 'reuse-verification.json')['hardware']['devices']
    uuids = ['GPU-' + d['uuid'].removeprefix('GPU-') for d in expected]
    confirmations = 0
    while confirmations < 3:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True, timeout=30)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                        '--format=csv,noheader,nounits'], text=True, timeout=30)
        rows = [[v.strip() for v in line.split(',')] for line in gpu.strip().splitlines()]
        assert [r[1] for r in rows] == uuids, 'GPU identity or order changed'
        idle = not apps.strip() and all(int(r[2]) <= 1024 and int(r[3]) <= 5 for r in rows)
        with (logs / 'gpu-checks.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(checked_beijing=now(), gpu_csv=gpu, apps_csv=apps, idle=idle)) + '\n')
        confirmations = confirmations + 1 if idle else 0
        state('waiting', f'GPU idle confirmation {confirmations}/3')
        if confirmations < 3:
            deadline = time.monotonic() + (10 if idle else 600)
            while time.monotonic() < deadline:
                time.sleep(min(30, max(0, deadline - time.monotonic())))
                state('waiting', f'GPU idle confirmation {confirmations}/3')

try:
    state('running', 'Initializing campaign')
    note('partial', 'CPU and source checks starting; no GPU performance results yet')
    for name, args in {
        'code-commit.txt': ['git', 'rev-parse', 'HEAD'],
        'worktree-before.txt': ['git', 'status', '--porcelain'],
        'worktree-before.patch': ['git', 'diff', 'HEAD', '--binary'],
        'submodules.txt': ['git', 'submodule', 'status'],
    }.items():
        (handoff / name).write_text(subprocess.check_output(args, cwd=repo, text=True))
    (handoff / 'controller-hashes.json').write_text(json.dumps(controller_hashes, indent=2) + '\n')
    for p in tools.glob('*.py'):
        dest = out / 'controller-source' / p.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(p, dest)
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1')
    run([sys.executable, '-m', 'unittest', 'discover', '-s', tools, '-p', 'test_*.py', '-v'],
        'cpu-tests.log', env=cpu_env)
    stage = 'frozen-input-checks'
    freeze = Path(os.environ['H100_FREEZE'])
    run(['sha256sum', '--quiet', '-c', 'SHA256SUMS.txt'], 'freeze-check.log', cwd=freeze)
    run(['sha256sum', '--quiet', '-c', freeze / 'REFERENCES.sha256.txt'], 'references-check.log')
    stage = 'parent-calibration-verification'
    run([sys.executable, Path(os.environ['H100_BUILD']) / 'record-inputs.py'], 'reuse-verification.log')
    from ablation_plan import parent_plan
    from ablation_prepare import megatron_check
    from candidate import sources, verify
    parent, parent_path, evidence = parent_plan()
    reuse = read(handoff / 'reuse-verification.json')
    assert parent['hardware'] == reuse['hardware'], 'Parent calibration hardware mismatch'
    assert parent['config'] == reuse['config'], 'Parent model mismatch'
    assert sha(parent_path) == reuse['plan_sha256'], 'Parent plan or mapping mismatch'
    sources()
    megatron_check(Path(os.environ['H100_MEGATRON']))
    for name, digest in reuse['nccl_files'].items():
        assert sha(name) == digest
    stage = 'prepare-candidates'
    run([sys.executable, tools / 'candidate.py', 'prepare', builds,
         '--seeds', '17', '29', '43', '--bases', 'remote_first', 'interleaved'], 'candidate-prepare.log')
    stage = 'compile-candidates'
    note('partial', 'Preparing/compiling 18 independent libraries; GPU preflight and timing pending')
    run([sys.executable, tools / 'candidate.py', 'compile', builds, '--cuda-home', os.environ['H100_CUDA_BUILD'],
         '--jobs', '2', '--nccl-root', os.environ['H100_NCCL']], 'candidate-compile.log')
    data = verify(builds, built=True)
    assert sum(v['group'] != 'native' for v in data['variants'].values()) == 18
    stage = 'prepare-compare'
    run([sys.executable, tools / 'ablation_prepare.py', compare, '--builds', builds,
         '--megatron', os.environ['H100_MEGATRON'], '--blocks', '10', '--schedule-seed', '20261013'], 'prepare.log')
    submission = read(compare / 'submission.json')
    assert submission['executable'] and not submission['source_only']
    assert submission['timing_windows'] == 400 and submission['seeds'] == [17, 29, 43]
    stage = 'dry-run'
    run([sys.executable, tools / 'ablation_run.py', compare], 'dry-run.log')
    stage = 'waiting-for-gpu'
    idle_gate()
    stage = 'preflight'
    note('partial', 'GPU ablation preflight started; formal timing follows only after all gates pass')
    run([sys.executable, tools / 'ablation_run.py', compare, '--stage', 'preflight',
         '--cuda-home', os.environ['H100_CUDA_RUN'], '--execute'], 'preflight.log')
    assert read(compare / 'preflight.json')['completed']
    stage = 'timing'
    run([sys.executable, tools / 'ablation_run.py', compare, '--stage', 'timing',
         '--cuda-home', os.environ['H100_CUDA_RUN'], '--execute'], 'timing.log')
    stage = 'report'
    run([sys.executable, tools / 'ablation_report.py', compare], 'report.log')
    report = read(compare / 'report.json')
    assert report['completed'] and len(report['all_samples']) == 400
    state('completed', 'Strict report passed; all 400 formal windows preserved')
    note('complete', 'Strict report passed; 400 formal windows; data return commit pending')
    print(now(), 'COMPLETE', compare, flush=True)
except BaseException as error:
    state('failed', str(error))
    note('failed', str(error))
    raise
