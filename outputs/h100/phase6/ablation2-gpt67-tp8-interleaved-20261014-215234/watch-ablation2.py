"""Poll before one suite only, then hand off to the strict five-group controller."""
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

repo = Path(os.environ['REPO'])
out = Path(os.environ['H100_OUT'])
log = Path(os.environ['H100_LOG'])
prior = Path(os.environ['H100_PREVIOUS_LOG'])
zone = datetime.timezone(datetime.timedelta(hours=8))
log.mkdir(parents=True, exist_ok=True)
(log / 'handoff').mkdir(exist_ok=True)
lock = open('/tmp/tempname-h100-phase6-repeat-queue.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (log / 'watchdog-state.json').exists() or (log / 'execution.json').exists():
    raise RuntimeError('This session has already started; preserve its records')
expected = json.loads((prior / 'handoff/environment.json').read_text())['hardware']
uuids = [d['uuid'].removeprefix('GPU-') for d in expected['devices']]
state = dict(pid=os.getpid(), session=os.environ['H100_SESSION'],
             total_runs=1, completed_runs=0, policies=['native_taco', 'G1', 'G2', 'G3', 'G4'],
             bases=['interleaved'], rounds=2, planned_windows=50,
             poll_seconds=600, idle_confirm_seconds=10, required_idle_confirmations=3,
             status='starting', stage='initializing', child_pid=None,
             gpu_monitoring='before_each_suite_only', previous_failed_session=str(prior),
             execution=str(log / 'execution.json'))
setup = dict(session=os.environ['H100_SESSION'], previous_failed_session=str(prior),
             policies=state['policies'], bases=state['bases'], rounds=2, timing_windows=50,
             timing_rank_records=400, preflight_rank_records=80, profile_traces=40, candidate_mappings=4,
             gpu_monitoring='before_each_suite_only', poll_seconds=600, required_idle_confirmations=3,
             controller_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
             code_files=[str(out / name) for name in ('env.sh', 'launch.sh', 'watch-ablation2.py', 'run-session.py')])
(log / 'handoff/watcher-setup.json').write_text(json.dumps(setup, indent=2) + '\n')
(log / 'handoff/watcher-setup.md').write_text(
    '# Five-group suite startup watcher\n\n'
    f'Session: `{os.environ["H100_SESSION"]}`. One suite only: interleaved, native_taco/G1-G4, '
    'two rounds, five paired blocks per round, 50 formal windows.\n\n'
    'The user authorized waiting after the previous startup resource rejection. '
    'Busy GPUs are checked every 600 seconds; three idle confirmations are spaced 10 seconds apart. '
    'A heartbeat is written every 30 seconds. No other task is stopped. '
    'All GPU polling stops when the suite is launched; the strict entry performs its final availability gate. '
    'Any failure stops the suite without an automatic retry. The earlier failed session is preserved.\n')

def now():
    return datetime.datetime.now(zone).isoformat()

def heartbeat(**updates):
    state.update(updates, updated_beijing=now())
    tmp = log / 'watchdog-state.json.tmp'
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(log / 'watchdog-state.json')

def gpu_check():
    query = lambda fields, kind: subprocess.check_output([
        'nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
        text=True, timeout=30)
    detail = dict(checked_beijing=now(), host=socket.gethostname())
    try:
        gpu = query('index,uuid,name,memory.used,utilization.gpu', 'gpu')
        apps = query('gpu_uuid,pid,process_name,used_memory', 'compute-apps')
    except subprocess.SubprocessError as error:
        detail.update(idle=False, error=str(error))
    else:
        rows = [[v.strip() for v in row] for row in csv.reader(gpu.splitlines())]
        order = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').split(',')
        rows = [next((r for r in rows if key.strip() in (r[0], r[1])), []) for key in order]
        same = socket.gethostname() == expected['host'] and len(rows) == 8 and all(rows)
        same = bool(same and [r[1].removeprefix('GPU-') for r in rows] == uuids)
        detail.update(gpu_csv=gpu, apps_csv=apps, same_hardware=same)
        if not same:
            raise RuntimeError('Host/GPU UUID/order changed; calibration must be revalidated')
        detail['idle'] = not apps.strip() and all(
            'H100' in r[2] and int(r[3]) <= 1024 and int(r[4]) <= 5 for r in rows)
    with (log / 'gpu-checks.jsonl').open('a') as stream:
        stream.write(json.dumps(detail) + '\n')
    heartbeat(last_gpu_check_beijing=detail['checked_beijing'], gpu_check=detail)
    print(now(), 'GPU CHECK', 'idle' if detail['idle'] else 'busy or unavailable', flush=True)
    return detail['idle']

def sleep_until_check(seconds):
    deadline = time.monotonic() + seconds
    heartbeat(next_gpu_check_beijing=(datetime.datetime.now(zone) + datetime.timedelta(seconds=seconds)).isoformat())
    while time.monotonic() < deadline:
        time.sleep(min(30, max(0, deadline - time.monotonic())))
        heartbeat()

heartbeat(status='waiting', stage='waiting-for-gpu', detail='One five-group, two-round suite pending')
try:
    confirmations = 0
    while confirmations < 3:
        idle = gpu_check()
        confirmations = confirmations + 1 if idle else 0
        heartbeat(status='waiting', stage='waiting-for-gpu',
                  detail=f'GPU idle confirmation {confirmations}/3')
        if confirmations < 3:
            sleep_until_check(10 if idle else 600)
    # The child performs its own final startup check; no GPU polling after this handoff.
    with (log / 'campaign.log').open('x') as stream:
        child = subprocess.Popen([sys.executable, '-u', str(out / 'run-session.py')],
                                 cwd=repo, stdout=stream, stderr=subprocess.STDOUT)
        heartbeat(status='running', stage='suite', child_pid=child.pid,
                  next_gpu_check_beijing=None, detail='Preparing and executing one strict suite')
        print(now(), 'SUITE START', child.pid, flush=True)
        while child.poll() is None:
            execution = log / 'execution.json'
            if execution.exists():
                try:
                    heartbeat(experiment=json.loads(execution.read_text()))
                except json.JSONDecodeError:
                    pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            heartbeat()
    if child.returncode:
        raise RuntimeError(f'Suite exited {child.returncode}; see {log / "campaign.log"}; no retry')
    result = json.loads((log / 'compare/report.json').read_text())
    assert result['completed'] and result['timing_windows'] == 50 and result['timing_rank_records'] == 400
    heartbeat(status='completed', stage='completed', child_pid=None, completed_runs=1,
              detail='Strict report passed; one suite of 50 fresh windows complete')
    print(now(), 'COMPLETE', log / 'compare', flush=True)
except BaseException as error:
    heartbeat(status='failed', stage='failed', detail=str(error))
    raise
