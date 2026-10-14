"""Wait for the parent, then run exactly three strictly gated ablation campaigns."""
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
tag = 'ablation-gpt67-tp8-20261014-005018'
queue_out = repo / 'outputs/h100/phase6' / (tag + '-repeat-queue')
queue_log = repo / 'logs/h100/phase6' / (tag + '-repeat-queue')
parent_log = Path(os.environ['H100_LOG'])
builds = Path(os.environ['H100_OUT']) / 'candidates'
tools = repo / 'tools/h100/phase6'
sys.path.insert(0, str(tools))
from common import read, sha, verify_files
from candidate import verify
from ablation_plan import parent_plan
from ablation_prepare import megatron_check

zone = datetime.timezone(datetime.timedelta(hours=8))
queue_log.mkdir(parents=True, exist_ok=True)
queue_lock = (queue_log / 'queue.lock').open('a')
fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (queue_log / 'watchdog-state.json').exists():
    raise RuntimeError('Queue already started; preserve existing state')
state = dict(pid=os.getpid(), sessions=[tag + f'-repeat{i}' for i in (1, 2, 3)],
             parent_session=str(parent_log), status='waiting', stage='waiting-for-parent',
             completed_runs=0, total_runs=3, planned_windows=1200,
             gpu_monitoring='before_each_run_only', child_pid=None)
log = None
child = None
stage = 'waiting-for-parent'
controllers = read(parent_log / 'compare/submission.json')['controllers']

def now():
    return datetime.datetime.now(zone).isoformat()

def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def heartbeat(status, detail):
    state.update(status=status, stage=stage, detail=detail, updated_beijing=now(),
                 child_pid=child.pid if child else None)
    if log:
        folders = list((log / 'compare/results').glob('*/round*/window-*'))
        progress = dict(status=status, stage=stage, detail=detail, updated_beijing=now(),
                        pid=os.getpid(), child_pid=state['child_pid'], compare=str(log / 'compare'),
                        started_windows=len(folders), completed_windows=sum(
                            all((p / f'rank{r}.json').is_file() for r in range(8)) for p in folders),
                        planned_windows=400, gpu_monitoring='before_run_only')
        atomic(log / 'execution.json', progress)
        state['experiment'] = progress
    atomic(queue_log / 'watchdog-state.json', state)

def run(args, name, env):
    global child
    verify_files('/', controllers)
    print(now(), stage, 'START', name, flush=True)
    with (log / name).open('x') as stream:
        child = subprocess.Popen([str(x) for x in args], cwd=repo, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
        while child.poll() is None:
            heartbeat('running', name)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        code = child.returncode
        child = None
    if code:
        raise RuntimeError(f'{name} failed with {code}; see {log / name}')
    print(now(), stage, 'COMPLETE', name, flush=True)

def wait_gpu():
    uuids = ['GPU-' + d['uuid'].removeprefix('GPU-') for d in
             read(parent_log / 'handoff/reuse-verification.json')['hardware']['devices']]
    confirmations = 0
    while confirmations < 3:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True, timeout=30)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                        '--format=csv,noheader,nounits'], text=True, timeout=30)
        rows = [[v.strip() for v in line.split(',')] for line in gpu.strip().splitlines()]
        assert [r[1] for r in rows] == uuids, 'GPU identity/order changed'
        idle = not apps.strip() and all(int(r[2]) <= 1024 and int(r[3]) <= 5 for r in rows)
        with (log / 'gpu-checks.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time_beijing=now(), gpu_csv=gpu, apps_csv=apps, idle=idle)) + '\n')
        confirmations = confirmations + 1 if idle else 0
        heartbeat('waiting', f'GPU idle confirmation {confirmations}/3')
        if confirmations < 3:
            deadline = time.monotonic() + (10 if idle else 600)
            while time.monotonic() < deadline:
                time.sleep(min(30, max(0, deadline - time.monotonic())))
                heartbeat('waiting', f'GPU idle confirmation {confirmations}/3')

def handoff(status, detail):
    (log / 'handoff/README.md').write_text(
        f'# Independent H100 ablation repeat\n\nStatus: {status}; {detail}. Beijing: {now()}.\n\n'
        f'Reused build: `{builds}`; parent report: `{parent_log / "compare/report.json"}`.\n\n'
        'GPT 6.7B TP8; native + G0-G6; remote_first/interleaved; G5 seeds 17/29/43; '
        '10 blocks; schedule seed 20261013; 10 warmup + 20 full optimizer steps; 400 fresh windows. '
        'Each repeat creates a new compare and reruns all preflight and timing. '
        'No slow-window deletion or automatic retry. Runtime occupancy sampling disabled; '
        'exclusive use throughout execution is not asserted. Build sources and libraries remain '
        'in the referenced parent build; libraries/caches stay local. Data return commit SHA: pending.\n')

try:
    heartbeat('waiting', 'Waiting for the original campaign and strict report; no GPU queries')
    while True:
        prior = read(parent_log / 'execution.json')
        if prior['status'] == 'failed':
            raise RuntimeError('Parent campaign failed; no repeats will start')
        if prior['status'] == 'completed':
            break
        try:
            os.kill(prior['pid'], 0)
        except ProcessLookupError:
            raise RuntimeError('Parent controller exited before completion')
        time.sleep(30)
        heartbeat('waiting', 'Waiting for the original campaign and strict report; no GPU queries')
    result = read(parent_log / 'compare/report.json')
    assert result['completed'] and len(result['all_samples']) == 400
    shared_lock = open('/tmp/tempname-h100-phase6-repeat-queue.lock', 'a')
    while True:
        try:
            fcntl.flock(shared_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            heartbeat('waiting', 'Waiting for the experiment controller lock')
            time.sleep(30)
    verify_files('/', controllers)
    for index, session in enumerate(state['sessions'], 1):
        state.update(current_session=session, run_index=index)
        log = repo / 'logs/h100/phase6' / session
        out = repo / 'outputs/h100/phase6' / session
        log.mkdir(parents=True, exist_ok=False)
        (log / 'handoff').mkdir()
        out.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, H100_SESSION=session, H100_LOG=str(log), H100_OUT=str(out))
        handoff('partial', 'Waiting for GPU, input verification and fresh measurement')
        stage = 'waiting-for-gpu'
        wait_gpu()
        stage = 'verify-reuse'
        run([sys.executable, Path(os.environ['H100_BUILD']) / 'record-inputs.py'], 'reuse-verification.log', env)
        data = verify(builds, built=True)
        parent, path, _ = parent_plan()
        reuse = read(log / 'handoff/reuse-verification.json')
        assert reuse['hardware'] == data['calibration_hardware'] == parent['hardware']
        assert reuse['config'] == data['config'] == parent['config']
        assert reuse['plan_sha256'] == sha(path) == data['parent_plan']['sha256']
        megatron_check(Path(os.environ['MEGATRON_DIR']))
        verify_files('/', controllers)
        atomic(log / 'handoff/repeat-inputs.json', dict(
            reused_build=str(builds), manifest_sha256=sha(builds / 'ablation-manifest.json'),
            parent_report_sha256=sha(parent_log / 'compare/report.json'),
            controllers=controllers, fresh_windows=400, timing_reused=False,
            queue_source_sha256=sha(queue_out / 'queue.py')))
        snapshot = out / 'controller-source'
        snapshot.mkdir()
        for p in tools.glob('*.py'):
            shutil.copy2(p, snapshot / p.name)
        compare = log / 'compare'
        stage = 'prepare-compare'
        run([sys.executable, tools / 'ablation_prepare.py', compare, '--builds', builds,
             '--megatron', os.environ['MEGATRON_DIR'], '--blocks', '10', '--schedule-seed', '20261013'], 'prepare.log', env)
        submission = read(compare / 'submission.json')
        assert submission['executable'] and not submission['source_only']
        assert submission['timing_windows'] == 400 and submission['seeds'] == [17, 29, 43]
        stage = 'dry-run'
        run([sys.executable, tools / 'ablation_run.py', compare], 'dry-run.log', env)
        for stage in ('preflight', 'timing'):
            run([sys.executable, tools / 'ablation_run.py', compare, '--stage', stage,
                 '--cuda-home', os.environ['H100_CUDA_RUN'], '--execute'], stage + '.log', env)
            if stage == 'preflight':
                assert read(compare / 'preflight.json')['completed']
        stage = 'report'
        run([sys.executable, tools / 'ablation_report.py', compare], 'report.log', env)
        result = read(compare / 'report.json')
        assert result['completed'] and len(result['all_samples']) == 400
        heartbeat('completed', 'Strict report passed; 400 fresh windows')
        handoff('complete', 'Strict report passed; 400 fresh windows')
        state['completed_runs'] = index
        print(now(), 'COMPLETE', session, flush=True)
        log = None
    stage = 'completed'
    heartbeat('completed', 'All three independent campaigns completed; 1200 fresh windows')
except BaseException as error:
    heartbeat('failed', str(error))
    if log:
        handoff('failed', str(error))
    raise
