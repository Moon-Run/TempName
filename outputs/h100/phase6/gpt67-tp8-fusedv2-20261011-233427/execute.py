"""Execute the frozen fused-v2 comparison and record stage/resource evidence."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo = Path(os.environ['REPO'])
build = Path(os.environ['H100_BUILD'])
logroot = Path(os.environ['H100_LOG'])
compare = logroot / 'compare'
handoff = logroot / 'handoff'
sys.path.insert(0, str(repo / 'tools/h100/phase6'))
from common import read, sha, write

zone = datetime.timezone(datetime.timedelta(hours=8))
stage = 'prepare'
child = None

def now():
    return datetime.datetime.now(zone).isoformat()

def state(status, detail):
    dirs = list(compare.glob('results/round*/window-*'))
    complete = sum(all((d / f'rank{r}.json').is_file() for r in range(8)) for d in dirs)
    write(logroot / 'execution.json', dict(
        status=status, stage=stage, detail=detail, updated_beijing=now(),
        controller_pid=os.getpid(), child_pid=child.pid if child else None,
        compare=str(compare), started_windows=len(dirs), completed_windows=complete,
    ))

def snapshot():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    with (handoff / 'gpu-occupancy.jsonl').open('a') as f:
        f.write(json.dumps(dict(time_beijing=now(), stage=stage, gpu_csv=output, apps_csv=apps)) + '\n')
    return [line.split(',') for line in output.strip().splitlines()]

def run(command, name):
    global child
    print(now(), stage, 'START', name, flush=True)
    with (logroot / name).open('x') as f:
        child = subprocess.Popen(command, cwd=repo, stdout=f, stderr=subprocess.STDOUT)
        state('running', name)
        while child.poll() is None:
            snapshot()
            state('running', name)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        code = child.returncode
        child = None
        if code:
            raise RuntimeError(f'{name} exited with {code}; see {logroot / name}')
    print(now(), stage, 'COMPLETE', name, flush=True)

try:
    base_manifest = read(build / 'base/manifest.json')
    assert base_manifest['status'] == 'built'
    assert base_manifest['variants']['taco_fused']['implementation'] == 'a800-fused-v2-cta-v1'
    assert sha(os.environ['H100_PLAN']) == read(handoff / 'reuse-verification.json')['plan_sha256']
    run([sys.executable, 'tools/h100/phase6/prepare.py', str(compare),
         '--base-build', str(build / 'base'), '--candidate-build', os.environ['H100_CANDIDATES'],
         '--plan', os.environ['H100_PLAN'], '--megatron', os.environ['MEGATRON_DIR']], 'prepare.log')
    assert read(compare / 'submission.json')['variants']['taco_fused']['implementation'] == 'a800-fused-v2-cta-v1'
    stage = 'waiting-for-gpu'
    confirmations = 0
    while confirmations < 3:
        rows = snapshot()
        idle = len(rows) == 8 and all(int(r[2]) <= 1024 and int(r[3]) <= 5 for r in rows)
        confirmations = confirmations + 1 if idle else 0
        state('waiting', f'GPU idle confirmation {confirmations}/3')
        if confirmations < 3:
            time.sleep(10 if idle else 600)
    stage = 'preflight'
    run([sys.executable, 'tools/h100/phase6/run.py', str(compare), '--stage', 'preflight',
         '--python', sys.executable, '--cuda-home', os.environ['CUDA124'], '--execute'], 'preflight.log')
    gate = read(compare / 'preflight.json')
    assert gate['completed']
    cta = {key: [n for n in names if 'taco_decode_ring_kernel(' in n]
           for key, names in gate['kernel_names'].items() if key.startswith('taco_fused/')}
    assert len(cta) == 8 and all(cta.values())
    write(handoff / 'cta-trace-verification.json', dict(passed=True, ranks=cta))
    stage = 'timing'
    run([sys.executable, 'tools/h100/phase6/run.py', str(compare), '--stage', 'timing',
         '--python', sys.executable, '--cuda-home', os.environ['CUDA124'], '--execute'], 'timing.log')
    stage = 'report'
    run([sys.executable, 'tools/h100/phase6/report.py', str(compare)], 'report.log')
    result = read(compare / 'report.json')
    assert result['completed'] and result['fused_baseline_implementation'] == 'a800-fused-v2-cta-v1'
    state('completed', 'All preflight and 72 timing windows verified; report generated')
    print(now(), 'COMPLETE', compare, 'accepted', result['accepted'], flush=True)
except BaseException as error:
    state('failed', str(error))
    snapshot()
    raise
