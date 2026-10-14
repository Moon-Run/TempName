"""Seven fresh ablation2 suites, then seven fresh Attention/MLP suites."""
import argparse
import collections
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(os.environ['REPO'])
OUT = Path(os.environ['H100_BATCH_OUT'])
LOG = Path(os.environ['H100_BATCH_LOG'])
TOOLS = REPO / 'tools/h100/phase6'
sys.path.insert(0, str(TOOLS))
from common import read, require, sha, verify_files, write

ZONE = datetime.timezone(datetime.timedelta(hours=8))
STAMP = '20261015-012238'
PARENT = REPO / 'logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-003756'
ABLATION = REPO / 'logs/h100/phase6/ablation2-gpt67-tp8-interleaved-20261014-221952'
ABLATION_BUILD = REPO / 'outputs/h100/phase6/ablation2-gpt67-tp8-interleaved-20261014-221952/candidates'
SCOPE_BUILD = REPO / 'outputs/h100/phase6/TTention-MLP-gpt67-tp8-20261014-231236/candidates'
CALIBRATION = REPO / 'logs/h100/phase6/TTention-MLP-gpt67-tp8-20261014-231236/calibration'
ABLATION_POLICIES = ['native_taco', 'G1', 'G2', 'G3', 'G4']
SCOPE_POLICIES = ['native_taco_attn_mlp', 'native_taco_attn',
                  'interleaved_full_attn', 'interleaved_full_attn_mlp']
EXPECTED = {'ablation2': dict(timing_rank=400, preflight_rank=80, traces=40, mappings=4),
            'scope': dict(timing_rank=256, preflight_rank=64, traces=32, mappings=3)}


def now():
    return datetime.datetime.now(ZONE).isoformat()


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    write(temp, value)
    temp.replace(path)


def jobs():
    result = []
    for family, prefix, windows, blocks, policies in (
        ('ablation2', 'ablation2-gpt67-tp8-interleaved', 50, 5, ABLATION_POLICIES),
        ('scope', 'TTention-MLP-gpt67-tp8', 32, 4, SCOPE_POLICIES)):
        for repeat in range(1, 8):
            session = f'{prefix}-{STAMP}-repeat{repeat}'
            result.append(dict(index=len(result)+1, family=family, repeat=repeat, session=session,
                out=str(REPO / 'outputs/h100/phase6' / session),
                log=str(REPO / 'logs/h100/phase6' / session), windows=windows,
                blocks_per_round=blocks, rounds=2, policies=policies,
                warmup_steps=10, timed_steps=20, schedule_seed=20261013))
    return result


def progress(job):
    root = Path(job['log']) / 'compare/results'
    pattern = '*/round*/window-*' if job['family'] == 'ablation2' else 'round*/window-*'
    folders = list(root.glob(pattern))
    return dict(planned_windows=job['windows'], started_windows=len(folders),
                completed_windows=sum(all((p / f'rank{r}.json').is_file() for r in range(8)) for p in folders))


def inventory(job):
    root = Path(job['log']) / 'compare/results'
    prefix = '*/' if job['family'] == 'ablation2' else ''
    return dict(timing_rank=len(list(root.glob(prefix+'round*/window-*/rank[0-7].json'))),
                preflight_rank=sum(len(list(root.glob(prefix+mode+'-*/rank[0-7].json')))
                                   for mode in ('smoke', 'profile')),
                traces=len(list(root.glob(prefix+'profile-*/rank*-profile.json.gz'))),
                mappings=len(list(root.glob(prefix+'mapping-*.csv'))))


def verify_submission(info, job):
    require(info['executable'] and not info['source_only'], 'Fresh executable compare required')
    require(info['timing_windows'] == job['windows'] == len(info['schedule']), 'Wrong window count')
    require(info['blocks_per_round'] == job['blocks_per_round'] and
            info['schedule_seed'] == 20261013, 'Wrong paired schedule')
    policies = {v['policy'] for v in info['variants'].values()} if job['family'] == 'ablation2' else set(info['variants'])
    require(policies == set(job['policies']), 'Wrong experimental groups')
    expected = {(r, p): job['blocks_per_round'] for r in (1, 2) for p in job['policies']}
    require(collections.Counter((j['round'], j['policy']) for j in info['schedule']) == expected,
            'Each group needs the prescribed blocks in both rounds')
    require(info['config']['warmup_steps'] == 10 and info['config']['timed_steps'] == 20,
            'Full optimizer-step timing protocol changed')
    if job['family'] == 'ablation2':
        require(info['bases'] == ['interleaved'] and info['seeds'] == [], 'Wrong ablation2 scope')


def capture(root, name, args):
    with (root / name).open('x') as stream:
        subprocess.run(list(map(str, args)), cwd=REPO, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def source_evidence():
    from ablation_report import verify_inputs as verify_ablation
    from candidate import verify as verify_candidate
    from scope_report import verify_inputs as verify_scope
    from scope_build import verify as verify_build
    from scope_plan import validate, hardware_identity
    from ablation_plan import parent_plan
    from scope_calibrate import verify as verify_calibration, calibration_records
    pins = {str(p): sha(p) for p in TOOLS.glob('*.py')}
    old_ablation = verify_ablation(ABLATION / 'compare')
    old_scope = verify_scope(PARENT / 'compare')
    registry = verify_candidate(ABLATION_BUILD, built=True)
    parent_plan(ABLATION_BUILD / 'plans/parent.json')
    pins[str(ABLATION_BUILD / 'ablation-manifest.json')] = sha(ABLATION_BUILD / 'ablation-manifest.json')
    for relative, digest in registry['files'].items():
        pins[str(ABLATION_BUILD / relative)] = digest
    for identity, variant in registry['variants'].items():
        if variant['group'] == 'native_taco':
            continue
        root = ABLATION_BUILD / variant['build']
        manifest = read(root / 'manifest.json')
        verify_files('/', manifest['nccl_files'])
        pins.update(manifest['nccl_files'])
        pins[str(root / 'manifest.json')] = sha(root / 'manifest.json')
    candidate = verify_build(SCOPE_BUILD, built=True, kind='scope_candidate')
    sampler_root = SCOPE_BUILD.parent / 'calibration'
    sampler = verify_build(sampler_root, built=True, kind='scope_sampler')
    for root, manifest in ((SCOPE_BUILD, candidate), (sampler_root, sampler)):
        verify_files('/', manifest['inputs'])
        verify_files('/', manifest['nccl_files'])
        pins.update(manifest['inputs']); pins.update(manifest['nccl_files'])
        pins[str(root / 'manifest.json')] = sha(root / 'manifest.json')
        for relative, digest in manifest['files'].items():
            pins[str(root / relative)] = digest
    calibration_plan = verify_calibration(CALIBRATION)
    require(read(CALIBRATION / 'calibration-complete.json')['completed'] and
            calibration_plan['roles'] == ['attn', 'mlp'] and
            len(calibration_records(CALIBRATION, calibration_plan)) == 8, 'Both real BF16 calibrations required')
    plans = read(SCOPE_BUILD / 'plans.json')
    for role, record in plans['roles'].items():
        validate(record, role)
        require(record['origin']['kind'] == 'scope_bf16_calibration' and not record['test_only'],
                'Synthetic/relabelled calibration is forbidden')
    require(hardware_identity(old_ablation['calibration_hardware']) ==
            hardware_identity(old_scope['calibration_hardware']), 'Reference GPU identities disagree')
    for info in (old_ablation, old_scope):
        pins.update(info['build_files'])
        pins.update(info.get('source_manifests', {}))
        for variant in info['variants'].values():
            for library in variant.get('libraries', {}).values():
                pins[library['path']] = library['sha256']
    pins[str(ABLATION / 'compare/submission.json')] = sha(ABLATION / 'compare/submission.json')
    pins[str(PARENT / 'compare/submission.json')] = sha(PARENT / 'compare/submission.json')
    verify_files('/', pins)
    old_report = read(ABLATION / 'compare/report.json')
    require(old_report['completed'] and old_report['timing_rank_records'] == 400 and
            len(old_report['all_samples']) == 50, 'Reference ablation2 suite incomplete')
    return dict(passed=True, checked_beijing=now(), gpu_used=False, input_hashes=pins,
        ablation2_build=str(ABLATION_BUILD), scope_build=str(SCOPE_BUILD),
        scope_calibration=str(CALIBRATION), hardware=old_scope['calibration_hardware'],
        ablation2_reference=str(ABLATION), scope_reference=str(PARENT), timing_reused=False)


def initialize():
    require(not (OUT / 'batch-plan.json').exists(), 'Batch already prepared; preserve its plan')
    LOG.mkdir(parents=True, exist_ok=False)
    handoff = LOG / 'handoff'; handoff.mkdir()
    evidence = source_evidence()
    write(handoff / 'input-verification.json', evidence)
    capture(handoff, 'code-commit.txt', ['git', 'rev-parse', 'HEAD'])
    capture(handoff, 'worktree-before.txt', ['git', 'status', '--porcelain'])
    capture(handoff, 'worktree-before.patch', ['git', 'diff', 'HEAD', '--binary'])
    capture(handoff, 'submodules.txt', ['git', 'submodule', 'status'])
    capture(handoff, 'pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze'])
    for name, args in (
        ('nvcc-build.txt', [Path(os.environ['H100_CUDA_BUILD']) / 'bin/nvcc', '--version']),
        ('nvcc-run.txt', [Path(os.environ['H100_CUDA_RUN']) / 'bin/nvcc', '--version']),
        ('apex-commit.txt', ['git', '-C', os.environ['APEX_SRC'], 'rev-parse', 'HEAD']),
        ('apex.patch', ['git', '-C', os.environ['APEX_SRC'], 'diff', 'HEAD', '--binary']),
        ('megatron-commit.txt', ['git', '-C', os.environ['H100_MEGATRON'], 'rev-parse', 'HEAD'])):
        capture(handoff, name, args)
    snapshot = OUT / 'controller-source'; snapshot.mkdir()
    for path in TOOLS.glob('*.py'):
        shutil.copy2(path, snapshot / path.name)
    batch = dict(session=os.environ['H100_BATCH_SESSION'], created_beijing=now(),
        parent_session=str(PARENT), jobs=jobs(), total_runs=14, total_timing_windows=574,
        gpu_monitoring='before_each_suite_only', busy_gpu_action='stop', failure_action='stop',
        automatic_retry=False, extra_repeats=False, input_hashes=evidence['input_hashes'])
    require(len({j['session'] for j in batch['jobs']}) == 14, 'Duplicate session')
    for job in batch['jobs']:
        require(not Path(job['out']).exists() and not Path(job['log']).exists(), 'Session path already exists')
    write(OUT / 'batch-plan.json', batch)
    scripts = {str(p): sha(p) for p in (OUT / 'batch.py', OUT / 'env.sh', OUT / 'launch.sh')}
    write(OUT / 'batch-manifest.json', dict(plan_sha256=sha(OUT / 'batch-plan.json'), scripts=scripts,
        snapshot={str(p): sha(p) for p in snapshot.glob('*.py')}))
    (handoff / 'README.md').write_text(
        '# Serial H100 experiment batch\n\nSeven independent five-group ablation2 suites, '
        'then seven independent four-group Attention/MLP suites. Wait for the current scope '
        'suite and strict report before taking the shared experiment lock. Each ablation2 '
        'suite has 50 windows; each scope suite has 32; all have two rounds, 10 warmup and '
        '20 full optimizer steps. Total: 574 fresh windows.\n\n'
        'Verified existing builds and real calibration are reused read-only. Each suite '
        'creates its own compare, run_id, controller/source snapshots, schedule, full '
        'mapping/smoke/profile, timing and strict reports. Slow-window sensitivity stays '
        'separate; no old timing is substituted. Input or controller changes, busy GPUs '
        'or any failed gate stop the batch. No automatic retry or additional runs.\n\n'
        'Only parent/process/status heartbeats occur during execution. GPU occupancy is '
        'queried by the official entry before each suite only. No other tasks are stopped. '
        'Libraries, objects and caches remain local and ignored. Source/calibration '
        'references are retained in every handoff. Data-return commit: not created.\n\n'
        f'Plan: `{OUT / "batch-plan.json"}`. Status: `{LOG / "watchdog-state.json"}`.\n')
    print(json.dumps(dict(prepared=True, jobs=len(batch['jobs']), timing_windows=574), indent=2))


def load_plan():
    manifest = read(OUT / 'batch-manifest.json')
    require(sha(OUT / 'batch-plan.json') == manifest['plan_sha256'], 'Batch plan changed')
    verify_files('/', manifest['scripts']); verify_files('/', manifest['snapshot'])
    batch = read(OUT / 'batch-plan.json')
    require(batch['jobs'] == jobs() and batch['total_runs'] == 14 and
            batch['total_timing_windows'] == 574, 'Batch order/count changed')
    verify_files('/', batch['input_hashes'])
    return batch


def verify_environment(index):
    batch = load_plan(); job = batch['jobs'][index-1]
    from ablation_prepare import megatron_check
    from runtime import check_topology
    from scope_plan import hardware_identity
    import torch
    megatron_check(Path(os.environ['H100_MEGATRON']))
    os.environ.update(H100_TP='8', H100_SM_COUNT='132', LOCAL_RANK='0')
    hardware = check_topology(8)
    require(hardware_identity(hardware) == hardware_identity(
        read(LOG / 'handoff/input-verification.json')['hardware']), 'Recalibration required for changed GPUs')
    software = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                    nccl=list(torch.cuda.nccl.version()), python=sys.executable)
    reference = read(ABLATION / 'handoff/environment.json')
    require(software == reference['software'], 'Training environment changed')
    scope_software = read(PARENT / 'handoff/reuse-verification.json')['software']
    require(all(scope_software[k] == value for k, value in software.items()) and
            scope_software['python_version'] == sys.version, 'Scope calibration environment changed')
    keys = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_MAX_CONNECTIONS', 'PYTORCH_CUDA_ALLOC_CONF',
            'PYTORCH_ALLOC_CONF', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CC', 'CXX', 'PYTHONNOUSERSITE')
    environment = {k: v for k, v in os.environ.items() if k in keys or k.startswith('NCCL_')}
    require(environment == reference['environment'], 'Runtime environment/communication settings changed')
    write(Path(job['log']) / 'handoff/environment.json', dict(hardware=hardware, software=software,
        environment=environment, candidate_build=str(ABLATION_BUILD if job['family'] == 'ablation2' else SCOPE_BUILD),
        python_version=sys.version, build_cuda=os.environ['H100_CUDA_BUILD'], run_cuda=os.environ['H100_CUDA_RUN'],
        timing_reused=False, passed=True))


class Batch:
    def __init__(self):
        self.plan = load_plan()
        self.job = None; self.child = None; self.stage = 'waiting-for-parent'
        self.completed = []

    def heartbeat(self, status, detail):
        value = dict(status=status, stage=self.stage, detail=detail, updated_beijing=now(),
            pid=os.getpid(), child_pid=self.child.pid if self.child else None,
            session=self.plan['session'], parent_session=str(PARENT), total_runs=14,
            completed_runs=len(self.completed), completed_sessions=self.completed,
            planned_windows=574, gpu_monitoring='before_each_suite_only', busy_gpu_action='stop',
            auto_retry=False, jobs=self.plan['jobs'])
        if self.job:
            experiment = dict(session=self.job['session'], compare=str(Path(self.job['log']) / 'compare'),
                family=self.job['family'], rounds=2, **progress(self.job))
            value.update(current_session=self.job['session'], run_index=self.job['index'], experiment=experiment)
            atomic(Path(self.job['log']) / 'execution.json', dict(status=status, stage=self.stage,
                detail=detail, updated_beijing=now(), pid=os.getpid(), child_pid=value['child_pid'],
                gpu_monitoring=value['gpu_monitoring'], **experiment))
        atomic(LOG / 'watchdog-state.json', value)

    def command(self, args, name):
        load_plan()
        log = Path(self.job['log'])
        argv = list(map(str, args))
        write(log / 'handoff' / (name+'-command.json'), dict(args=argv, cwd=str(REPO)))
        env = dict(os.environ, H100_SESSION=self.job['session'], H100_OUT=self.job['out'], H100_LOG=self.job['log'],
                   H100_SCOPE_SESSION=self.job['session'], H100_SCOPE_OUT=self.job['out'], H100_SCOPE_LOG=self.job['log'])
        print(now(), self.job['session'], self.stage, 'START', name, flush=True)
        with (log / name).open('x') as stream:
            self.child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            while self.child.poll() is None:
                self.heartbeat('running', name)
                try:
                    self.child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            code = self.child.returncode; self.child = None
        require(code == 0, f'{name} exited {code}; see {log / name}')
        print(now(), self.job['session'], 'COMPLETE', name, flush=True)

    def deliver(self, status, detail):
        job = self.job; handoff = Path(job['log']) / 'handoff'
        report = Path(job['log']) / 'compare/report.json'
        actual = inventory(job)
        write(handoff / 'delivery.json', dict(status=status, detail=detail, actual=actual,
            expected=EXPECTED[job['family']], strict_report=report.is_file(), updated_beijing=now(),
            session=job['session'], batch=str(LOG), source_commit=(LOG / 'handoff/code-commit.txt').read_text().strip(),
            code_patch=str(LOG / 'handoff/worktree-before.patch'), data_return_commit=None))
        (handoff / 'README.md').write_text(
            f'# Independent H100 {job["family"]} repeat {job["repeat"]}\n\n'
            f'Session: `{job["session"]}`. Status: **{status}**. {detail}. Beijing: {now()}.\n\n'
            f'GPT 6.7B TP8; policies `{job["policies"]}`; two rounds; '
            f'{job["blocks_per_round"]} paired blocks per round; {job["windows"]} fresh windows, '
            '10 warmup + 20 complete optimizer steps. Formal timing and all preflight are fresh.\n\n'
            f'Reused builds: `{ABLATION_BUILD if job["family"] == "ablation2" else SCOPE_BUILD}`. '
            f'Reference compare: `{ABLATION if job["family"] == "ablation2" else PARENT}/compare`. '
            f'Scope real two-shape calibration: `{CALIBRATION}`. Reuse records and '
            'frozen plan/library/controller hashes accompany this compare. Original source '
            'and calibration evidence remain at their referenced locations.\n\n'
            f'Actual: `{actual}`; expected: `{EXPECTED[job["family"]]}`. '
            'Reports use all samples as primary and keep slow-window filtering separate. '
            'No missing records are filled with historical results. Busy GPUs, input changes '
            'or failure stop the entire remaining batch; no retry. GPU occupancy is checked '
            'before each suite only. No claim of measured runtime exclusivity. '
            'Libraries, objects, build and Python caches stay ignored. Data-return commit: not created.\n')
        return actual

    def execute(self):
        LOG.mkdir(exist_ok=True)
        with (LOG / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(not (LOG / 'watchdog-state.json').exists(), 'Queue already started; do not overwrite it')
            try:
                self.heartbeat('waiting', 'Waiting for the current scope suite and strict report; no GPU queries')
                while True:
                    parent = read(PARENT / 'execution.json')
                    require(parent['status'] != 'failed', 'Parent experiment failed; repeats stopped')
                    if parent['status'] == 'completed':
                        result = read(PARENT / 'compare/report.json')
                        require(result['completed'] and len(result['all_samples']) == 32 and
                                result['timing_rank_records'] == 256, 'Parent strict report incomplete')
                        break
                    os.kill(parent['pid'], 0)
                    time.sleep(30)
                    self.heartbeat('waiting', 'Waiting for current scope suite and strict report; no GPU queries')
                with open('/tmp/tempname-h100-phase6-repeat-queue.lock', 'a') as shared:
                    while True:
                        try:
                            fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            self.heartbeat('waiting', 'Waiting for current controller to release its experiment lock')
                            time.sleep(30)
                    self.run_jobs()
                self.stage = 'completed'; self.job = None
                write(LOG / 'batch-summary.json', dict(completed=True, completed_runs=14,
                    timing_windows=574, sessions=self.completed, completed_beijing=now(),
                    reports=[str(Path(j['log']) / 'compare/report.json') for j in self.plan['jobs']]))
                self.heartbeat('completed', 'All 14 suites, 574 fresh windows and strict reports passed')
            except BaseException as error:
                if self.child and self.child.poll() is None:
                    os.killpg(self.child.pid, signal.SIGTERM)
                    try:
                        self.child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.child.pid, signal.SIGKILL); self.child.wait(timeout=10)
                    self.child = None
                self.heartbeat('failed', str(error))
                if self.job:
                    self.deliver('failed', str(error))
                raise

    def run_jobs(self):
        run_ids = {read(ABLATION / 'compare/submission.json')['run_id'],
                   read(PARENT / 'compare/submission.json')['run_id']}
        for job in self.plan['jobs']:
            load_plan()
            out, log = Path(job['out']), Path(job['log'])
            out.mkdir(parents=True, exist_ok=False); log.mkdir(parents=True, exist_ok=False)
            (log / 'handoff').mkdir(); self.job = job
            self.stage = 'verify-reuse'
            self.heartbeat('running', 'Verifying unchanged software, devices and frozen builds')
            self.deliver('partial', 'Fresh session; measurement not completed')
            for name in ('code-commit.txt', 'submodules.txt', 'pip-freeze.txt', 'nvcc-build.txt',
                         'nvcc-run.txt', 'apex-commit.txt', 'apex.patch', 'megatron-commit.txt'):
                shutil.copy2(LOG / 'handoff' / name, log / 'handoff' / name)
            capture(log / 'handoff', 'worktree-before.txt', ['git', 'status', '--porcelain'])
            capture(log / 'handoff', 'worktree-before.patch', ['git', 'diff', 'HEAD', '--binary'])
            shutil.copytree(OUT / 'controller-source', out / 'controller-source')
            self.command([sys.executable, OUT / 'batch.py', '--verify-environment', job['index']], 'reuse-verification.log')
            entry = TOOLS / ('ablation2.py' if job['family'] == 'ablation2' else 'mlp_attention.py')
            if job['family'] == 'ablation2':
                self.command([sys.executable, entry, 'reuse', out / 'candidates', '--from-builds', ABLATION_BUILD], 'reuse.log')
                build_args = ['--builds', out / 'candidates', '--blocks', '5', '--schedule-seed', '20261013']
            else:
                shutil.copy2(SCOPE_BUILD / 'plans.json', out / 'plans.json')
                old = read(PARENT / 'compare/submission.json')
                write(out / 'reused-build.json', dict(build=str(SCOPE_BUILD),
                    manifest_sha256=sha(SCOPE_BUILD / 'manifest.json'),
                    libraries=old['variants']['interleaved_full_attn']['libraries'],
                    calibration=str(CALIBRATION), timing_reused=False))
                build_args = ['--build', SCOPE_BUILD, '--blocks', '4']
            compare = log / 'compare'; self.stage = 'prepare-compare'
            self.command([sys.executable, entry, 'prepare', compare, *build_args,
                          '--megatron', os.environ['H100_MEGATRON']], 'prepare.log')
            info = read(compare / 'submission.json'); verify_submission(info, job)
            require(info['run_id'] not in run_ids, 'Reused run_id'); run_ids.add(info['run_id'])
            write(log / 'handoff/repeat-inputs.json', dict(parent=str(ABLATION if job['family'] == 'ablation2' else PARENT),
                run_id=info['run_id'], library_hashes={k: v.get('libraries', {}) for k, v in info['variants'].items()},
                submission_sha256=sha(compare / 'submission.json'), timing_reused=False,
                scope_calibration=str(CALIBRATION), source_hashes=self.plan['input_hashes']))
            self.stage = 'dry-run'
            self.command([sys.executable, entry, 'run', compare], 'dry-run.log')
            self.stage = 'preflight-and-timing'
            self.command([sys.executable, entry, 'run', compare, '--stage', 'all',
                          '--cuda-home', os.environ['H100_CUDA_RUN'], '--execute'], 'run-all.log')
            self.stage = 'strict-report'
            self.command([sys.executable, entry, 'report', compare], 'report.log')
            result = read(compare / 'report.json')
            require(result['completed'] and len(result['all_samples']) == job['windows'] and
                    result['timing_rank_records'] == EXPECTED[job['family']]['timing_rank'] and
                    read(compare / 'all-state.json')['status'] == 'completed', 'Strict report incomplete')
            require(self.deliver('complete', 'All fresh preflight/timing and strict report passed') ==
                    EXPECTED[job['family']], 'Delivery evidence missing/extra')
            self.completed.append(job['session'])
            self.heartbeat('completed', 'This suite passed; moving to the next suite')
            print(now(), 'SUITE COMPLETE', job['session'], flush=True)


def self_test():
    from queue import Queue
    import torch
    require(callable(Queue) and not torch.cuda.is_initialized(),
            'The batch entry must preserve standard-library imports without initializing CUDA')
    spec = jobs()
    require(len(spec) == 14 and [j['family'] for j in spec] == ['ablation2']*7+['scope']*7,
            'Wrong batch order')
    require(sum(j['windows'] for j in spec) == 574 and all(j['rounds'] == 2 for j in spec),
            'Wrong batch timing protocol')
    from ablation2_plan import schedule as ablation_schedule, definitions
    from scope_protocol import schedule as scope_schedule
    for job in spec:
        schedule = (ablation_schedule('interleaved', definitions(), 5, 20261013)
                    if job['family'] == 'ablation2' else scope_schedule(4, 20261013))
        info = dict(executable=True, source_only=False, timing_windows=job['windows'], schedule=schedule,
            blocks_per_round=job['blocks_per_round'], schedule_seed=20261013,
            variants={p: {'policy': p} for p in job['policies']}, bases=['interleaved'], seeds=[],
            config=dict(warmup_steps=10, timed_steps=20))
        verify_submission(info, job)
        for bad in (dict(info, source_only=True), dict(info, schedule=schedule[:-1])):
            try:
                verify_submission(bad, job)
            except ValueError:
                continue
            raise AssertionError('Invalid/source-only submission was accepted')
    print('PASS: standard-library Queue/PyTorch imports, 14 suites in order, two rounds, 574 windows, source-only/incomplete schedules rejected; no GPU used')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--initialize', action='store_true')
    choice.add_argument('--execute', action='store_true')
    choice.add_argument('--self-test', action='store_true')
    choice.add_argument('--verify-environment', type=int, choices=range(1, 15))
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit('Stopped by SIGTERM'))
    if args.initialize:
        initialize()
    elif args.self_test:
        self_test()
    elif args.verify_environment:
        verify_environment(args.verify_environment)
    else:
        Batch().execute()
