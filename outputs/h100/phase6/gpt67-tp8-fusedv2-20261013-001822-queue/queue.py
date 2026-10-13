"""Wait for eight idle H100s, then execute two independent 72-window repeats."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path('/workspace/TempName')
TAG = 'gpt67-tp8-fusedv2-20261013-001822'
OUTPUT = REPO / 'outputs/h100/phase6' / (TAG + '-queue')
LOG = REPO / 'logs/h100/phase6' / (TAG + '-queue')
SESSIONS = [TAG + '-repeat1', TAG + '-repeat2']
ZONE = datetime.timezone(datetime.timedelta(hours=8))
EXPECTED = json.loads((REPO / 'logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/handoff/reuse-verification.json').read_text())
UUIDS = ['GPU-' + d['uuid'].removeprefix('GPU-') for d in EXPECTED['hardware']['devices']]

def stamp():
    return datetime.datetime.now(ZONE).isoformat()

def write_state(**changes):
    state.update(changes, updated_beijing=stamp())
    temp = LOG / 'watchdog-state.json.tmp'
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(LOG / 'watchdog-state.json')

def check_gpu():
    try:
        rows = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=30)
        apps = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
            '--format=csv,noheader,nounits'], text=True, timeout=30)
        parsed = [[v.strip() for v in line.split(',')] for line in rows.strip().splitlines()]
        same_devices = [r[1] for r in parsed] == UUIDS
        idle = same_devices and not apps.strip() and all(
            int(r[2]) <= 1024 and int(r[3]) <= 5 for r in parsed)
        detail = dict(gpu_csv=rows, apps_csv=apps, same_devices=same_devices, idle=idle)
    except (subprocess.SubprocessError, ValueError, IndexError) as error:
        idle = False
        detail = dict(idle=False, error=str(error))
    checked = stamp()
    with (LOG / 'gpu-checks.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(checked_beijing=checked, **detail)) + '\n')
    write_state(last_gpu_check_beijing=checked, gpu_check=detail)
    print(checked, 'GPU CHECK', 'idle' if idle else 'busy or unavailable', flush=True)
    return idle

def wait_for_gpu():
    confirmations = 0
    while confirmations < 3:
        idle = check_gpu()
        confirmations = confirmations + 1 if idle else 0
        write_state(status='waiting', stage='waiting-for-gpu',
                    detail=f'GPU idle confirmation {confirmations}/3')
        if confirmations == 3:
            return
        deadline = time.monotonic() + (10 if idle else 600)
        due = datetime.datetime.now(ZONE) + datetime.timedelta(seconds=10 if idle else 600)
        write_state(next_gpu_check_beijing=due.isoformat())
        while time.monotonic() < deadline:
            time.sleep(min(30, max(0, deadline - time.monotonic())))
            write_state()

LOG.mkdir(parents=True, exist_ok=True)
lock = open('/tmp/tempname-h100-phase6-repeat-queue.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (LOG / 'watchdog-state.json').exists():
    raise RuntimeError('Queue has already been started; preserve existing state and results')
state = dict(pid=os.getpid(), sessions=SESSIONS, total_runs=2, completed_runs=0,
             poll_seconds=600, child_pid=None, current_session=None,
             gpu_monitoring='before_each_run_only',
             supersedes='gpt67-tp8-fusedv2-20261012-211935-queue')
write_state(status='starting', stage='initializing')
try:
    for index, session in enumerate(SESSIONS, 1):
        runlog = REPO / 'logs/h100/phase6' / session
        write_state(current_session=session, run_index=index, execution=str(runlog / 'execution.json'))
        wait_for_gpu()
        runlog.mkdir(parents=True, exist_ok=True)
        with (runlog / 'campaign.log').open('x') as stream:
            child = subprocess.Popen(['bash', str(OUTPUT / 'run-one.sh'), session],
                                     cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            write_state(status='running', stage='experiment', child_pid=child.pid,
                        next_gpu_check_beijing=None, detail='Fresh preflight and 72 timing windows')
            print(stamp(), 'START', session, child.pid, flush=True)
            while child.poll() is None:
                execution = runlog / 'execution.json'
                if execution.exists():
                    try:
                        write_state(experiment=json.loads(execution.read_text()))
                    except json.JSONDecodeError:
                        pass
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                write_state()
            if child.returncode:
                raise RuntimeError(f'{session} exited {child.returncode}; see {runlog / "campaign.log"}')
        verification = json.loads((runlog / 'compare/verification.json').read_text())
        assert verification['passed'] and verification['completed_windows'] == 72
        write_state(completed_runs=index, child_pid=None)
        print(stamp(), 'COMPLETE', session, flush=True)
    write_state(status='completed', stage='completed', detail='Both independent repeats verified; 144 new windows')
except BaseException as error:
    write_state(status='failed', stage='failed', detail=str(error))
    raise
