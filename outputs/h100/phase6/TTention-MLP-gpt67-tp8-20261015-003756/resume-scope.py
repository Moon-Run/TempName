"""Reuse verified calibration/builds and freeze a fresh comparison after a report fix."""
import datetime
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

repo = Path(os.environ['REPO'])
out = Path(os.environ['H100_SCOPE_OUT'])
log = Path(os.environ['H100_SCOPE_LOG'])
parent_out = Path(os.environ['H100_SCOPE_PARENT_OUT'])
parent_log = Path(os.environ['H100_SCOPE_PARENT_LOG'])
calibration = parent_log / 'calibration'
builds = parent_out / 'candidates'
compare = log / 'compare'
handoff = log / 'handoff'
tools = repo / 'tools/h100/phase6'
sys.path.insert(0, str(tools))
from common import read, write, sha, verify_files
from scope_plan import hardware_identity, validate
from scope_protocol import POLICIES
from scope_report import verify_inputs
from scope_build import verify as verify_build
from scope_calibrate import verify as verify_calibration, calibration_records
from ablation_prepare import megatron_check
from runtime import check_topology

zone = datetime.timezone(datetime.timedelta(hours=8))
stage = 'verify-reuse'
child = None
log.mkdir(parents=True, exist_ok=True)
handoff.mkdir(exist_ok=True)
lock = open('/tmp/tempname-h100-phase6-repeat-queue.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (log / 'execution.json').exists():
    raise RuntimeError('Already started; preserve this session')
controllers = {str(p): sha(p) for p in tools.glob('*.py')}
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()

def now():
    return datetime.datetime.now(zone).isoformat()

def state(status, detail):
    folders = list((compare / 'results').glob('round*/window-*'))
    temp = log / 'execution.json.tmp'
    write(temp, dict(status=status, stage=stage, detail=detail, updated_beijing=now(),
        pid=os.getpid(), child_pid=child.pid if child else None, session=os.environ['H100_SCOPE_SESSION'],
        compare=str(compare), calibration=str(calibration), reused_build=str(builds),
        policies=list(POLICIES), planned_windows=32, started_windows=len(folders),
        completed_windows=sum(all((p / f'rank{r}.json').exists() for r in range(8)) for p in folders),
        gpu_monitoring='before_suite_only', auto_wait=False, auto_retry=False))
    temp.replace(log / 'execution.json')

def record(name, args):
    (handoff / name).write_text(subprocess.check_output([str(x) for x in args], cwd=repo, text=True))

def run(args, name):
    global child
    verify_files('/', controllers)
    print(now(), stage, 'START', name, flush=True)
    write(handoff / (name + '-command.json'), dict(args=[str(x) for x in args], cwd=str(repo)))
    with (log / name).open('x') as stream:
        child = subprocess.Popen([str(x) for x in args], cwd=repo, stdout=stream, stderr=subprocess.STDOUT)
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
        preflight_rank=len(list(results.glob('smoke-*/rank[0-7].json'))) + len(list(results.glob('profile-*/rank[0-7].json'))),
        profile_traces=len(list(results.glob('profile-*/rank*-profile.json.gz'))),
        candidate_mappings=len(list(results.glob('mapping-*.csv'))),
        calibration_rank=len(list(calibration.glob('b0-interleaved-instrumented-rank[0-7].json'))),
        calibration_mappings=len(list(calibration.glob('mapping-*.csv'))))
    expected = dict(timing_rank=256, preflight_rank=64, profile_traces=32, candidate_mappings=3,
        calibration_rank=8, calibration_mappings=2)
    write(handoff / 'delivery.json', dict(status=status, detail=detail, actual=actual, expected=expected,
        source_commit=commit, measured_scope_patch=str(handoff / 'worktree.patch'),
        prior_session=str(parent_log), reused_build=str(builds), reused_calibration=str(calibration),
        strict_report=(compare / 'report.json').exists(), data_return_commit=None))
    (handoff / 'README.md').write_text(
        f'# H100 Attention/MLP corrected profile experiment\n\nSession: `{os.environ["H100_SCOPE_SESSION"]}`. '
        f'Status: **{status}**. {detail}. Beijing: {now()}.\n\n'
        f'Code: `{commit}` plus the recorded scope_report.py/test_scope.py patch. The profile checker '
        'counts CPU user_annotation spans only, excluding mirrored gpu_user_annotation spans. '
        'It still requires exactly 128 calls per role and the original launch correlation/kernel checks. '
        'All 72 CPU tests passed before launch, including missing and duplicate CPU-call rejection.\n\n'
        f'Failed evidence preserved at `{parent_log}`. Reused candidate build: `{builds}`; '
        f'reused NEW real two-role calibration from that session: `{calibration}`. '
        'Actual source snapshots, plans, binaries and calibration hashes were verified before reuse. '
        'No calibration/build artifacts or historical submission are edited.\n\n'
        'GPT 6.7B TP8, 32 layers, H4096/FFN16384, 32 heads, S2048, micro1/global4. '
        'Four scope policies, four paired blocks per round, two rounds, 32 fresh windows; '
        '10 warmup + 20 complete optimizer steps. Attention-only MLP is native BF16. '
        'Both candidates share the same dual-shape library and Attention plan. All mappings, '
        'smoke/profile and timing are newly run in this compare.\n\n'
        f'Evidence actual: `{actual}`; expected: `{expected}`. Full original calibration/source '
        'is in the referenced parent session; new plans and provenance copies are included here. '
        'The report uses all samples as primary and keeps slow-window sensitivity separate. '
        'No Slurm, wait queue, periodic GPU query, skipped group, lowered budget or automatic repeat. '
        'Libraries/objects/caches stay local and ignored. Data-return commit SHA: not created.\n')

try:
    state('running', 'Verifying calibration, plans, binary hashes and current environment')
    record('code-commit.txt', ['git', 'rev-parse', 'HEAD'])
    record('worktree.txt', ['git', 'status', '--porcelain'])
    record('worktree.patch', ['git', 'diff', 'HEAD', '--binary'])
    record('submodules.txt', ['git', 'submodule', 'status'])
    record('pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze'])
    record('megatron-commit.txt', ['git', '-C', os.environ['H100_MEGATRON'], 'rev-parse', 'HEAD'])
    verify_files(repo, read(parent_log / 'handoff/partial-artifact-hashes.json'))
    prior_submission = verify_inputs(parent_log / 'compare')
    candidate = verify_build(builds, built=True, kind='scope_candidate')
    sampler = verify_build(parent_out / 'calibration', built=True, kind='scope_sampler')
    for manifest in (candidate, sampler):
        verify_files('/', manifest['inputs'])
        verify_files('/', manifest['nccl_files'])
    calibration_plan = verify_calibration(calibration)
    calibration_data = calibration_records(calibration, calibration_plan)
    assert read(calibration / 'calibration-complete.json')['completed']
    assert calibration_plan['roles'] == ['attn', 'mlp'] and len(calibration_data) == 8
    plans = read(builds / 'plans.json')
    for role, data in plans['roles'].items():
        validate(data, role)
        assert not data['test_only'] and data['origin']['kind'] == 'scope_bf16_calibration'
    import torch
    os.environ.update(H100_TP='8', H100_SM_COUNT='132', LOCAL_RANK='0')
    hardware = check_topology(8)
    assert hardware_identity(hardware) == hardware_identity(prior_submission['calibration_hardware'])
    old_software = read(parent_log / 'handoff/environment.json')['software']
    software = dict(python=sys.executable, python_version=sys.version, torch=str(torch.__version__),
                    cuda=torch.version.cuda, nccl=list(torch.cuda.nccl.version()))
    assert software == old_software
    megatron_check(Path(os.environ['H100_MEGATRON']))
    write(handoff / 'reuse-verification.json', dict(
        passed=True, parent_compare=str(parent_log / 'compare'), candidate_build=str(builds),
        candidate_manifest_sha256=sha(builds / 'manifest.json'), calibration=str(calibration),
        calibration_manifest_sha256=sha(calibration / 'plan.json'), hardware=hardware, software=software,
        calibration_unchanged=True, libraries_unchanged=True, timing_reused=False,
        controllers=controllers, plan_sha256={r: sha(parent_out / 'plans/measured' / (r + '.json')) for r in plans['roles']}))
    snapshot = out / 'controller-source'
    snapshot.mkdir()
    for p in tools.glob('*.py'):
        shutil.copy2(p, snapshot / p.name)
    (out / 'plans').mkdir()
    for role in plans['roles']:
        shutil.copy2(parent_out / 'plans/measured' / (role + '.json'), out / 'plans' / (role + '.json'))
    write(out / 'reused-build.json', dict(build=str(builds), manifest_sha256=sha(builds / 'manifest.json'),
        libraries=prior_submission['variants']['interleaved_full_attn']['libraries']))
    stage = 'prepare-compare'
    run([sys.executable, tools / 'mlp_attention.py', 'prepare', compare, '--build', builds,
         '--megatron', os.environ['H100_MEGATRON'], '--blocks', '4'], 'prepare.log')
    fresh = verify_inputs(compare)
    assert fresh['run_id'] != prior_submission['run_id'] and fresh['timing_windows'] == 32
    assert set(fresh['variants']) == set(POLICIES) and fresh['executable'] and not fresh['source_only']
    stage = 'dry-run'
    run([sys.executable, tools / 'mlp_attention.py', 'run', compare], 'dry-run.log')
    delivery('partial', 'New strict comparison prepared; waiting only for the explicit startup gate')
    stage = 'preflight-and-timing'
    run([sys.executable, tools / 'mlp_attention.py', 'run', compare, '--stage', 'all',
         '--cuda-home', os.environ['H100_CUDA_RUN'], '--execute'], 'run-all.log')
    stage = 'strict-report'
    run([sys.executable, tools / 'mlp_attention.py', 'report', compare], 'report.log')
    result = read(compare / 'report.json')
    assert result['completed'] and len(result['all_samples']) == 32
    assert result['timing_rank_records'] == 256 and result['preflight_rank_records'] == 64
    delivery('complete', 'New preflight, all 32 windows and strict report passed')
    inventory = read(handoff / 'delivery.json')
    assert inventory['actual'] == inventory['expected']
    state('completed', 'Strict report passed; all 32 fresh windows and delivery artifacts present')
    print(now(), 'COMPLETE', compare, flush=True)
except BaseException as error:
    state('failed', str(error))
    delivery('failed', str(error))
    raise
