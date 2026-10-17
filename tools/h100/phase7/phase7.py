"""Prepare, inspect and explicitly execute isolated H100 Phase7 experiments."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from protocol import (MODEL, MEGATRON_COMMIT, PROTOCOL, VARIANTS, digest, read, require,
                      schedule, sha, summarize, validate_window, verify, write_new)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_PLANS = REPO / 'outputs/h100/phase6/TTention-MLP-gpt67-tp8-20261014-231236/plans/measured'


def check_controller(root):
    for name in ('phase7.py', 'protocol.py', 'planner.py'):
        require(sha(HERE / name) == sha(Path(root) / 'scripts' / name),
                'Controller changed; use this session\'s frozen scripts/phase7.py: ' + name)


def prepare(root, parent=DEFAULT_PLANS, kind='operator', scope='attention', chunk_tiles=8,
            quantization='legacy', blocks=4, warmup=10, steps=20, megatron=None, seed=20261017):
    from planner import import_phase6_plan, with_mask_mode
    require(kind in ('operator', 'model') and scope in ('attention', 'mlp', 'both'), 'Invalid kind/scope')
    require(kind != 'operator' or scope != 'both', 'Prepare one operator role per campaign')
    require(kind != 'model' or megatron is not None, '--megatron is required for full-model campaigns')
    require(type(warmup) is int and warmup > 0 and type(steps) is int and steps > 0, 'Positive warmup/steps required')
    jobs = schedule(blocks, seed)
    roots, sources = {}, {}
    roles = ('attention', 'mlp') if scope == 'both' else (scope,)
    for role in roles:
        path = Path(parent) / ('attn.json' if role == 'attention' else 'mlp.json')
        old = read(path)
        require(old.get('gpu_calibrated') is True and old.get('test_only') is False, 'Measured Phase6 input required')
        require(digest(old['entry']) == old['entry_sha256'], 'Parent entry hash mismatch')
        for ordering in ('identity', 'shared'):
            p = import_phase6_plan(old['entry'], role, chunk_tiles=chunk_tiles, ordering=ordering)
            if quantization != 'legacy':
                p = with_mask_mode(p, quantization, seed=seed)
            roots[f'{role}-{ordering}'] = p
        sources[role] = dict(path=str(path.resolve()), sha256=sha(path), entry_sha256=old['entry_sha256'])
    # No existing directory or old artifacts are ever overwritten.
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ('scripts', 'plans', 'results', 'provenance'):
        (root / name).mkdir()
    for path in HERE.glob('*.py'):
        if not path.name.startswith('test_'):
            shutil.copy2(path, root / 'scripts' / path.name)
    # Independent tensor codec reference is frozen with every experiment.
    shutil.copy2(REPO / 'tools/a800/phase4/reference.py', root / 'scripts' / 'reference.py')
    for name, p in roots.items():
        write_new(root / 'plans' / (name + '.json'), p)
    for role, record in sources.items():
        shutil.copy2(record['path'], root / 'provenance' / (role + '-phase6-plan.json'))
    files = {str(p.relative_to(root)): sha(p) for folder in ('scripts', 'plans', 'provenance')
             for p in (root / folder).iterdir() if p.is_file()}
    m = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex, kind=kind, scope=scope,
             implementation='torch_chunked_gemm_nccl_alltoall_prototype',
             gpu_validated=False, model=MODEL, chunk_tiles=chunk_tiles, quantization=quantization,
             blocks=blocks, schedule_seed=seed, warmup=warmup, steps=steps,
             variants=VARIANTS, schedule=jobs, parent_plans=sources,
             megatron=str(Path(megatron).resolve()) if megatron else None,
             megatron_commit=MEGATRON_COMMIT, files=files,
             source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
             limitations=['Not the fused Flux V2/V3 kernel', 'Shared coarse order is a legacy proxy, not new calibration',
                          'No performance/quality claim before actual H100 execution'])
    write_new(root / 'manifest.json', m)
    verify(root)
    return root


def command(root, python):
    return [str(python), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
            str(Path(root).resolve() / 'scripts' / 'worker.py'), '--root', str(Path(root).resolve())]


def jobs(m, stage):
    require(stage in ('all', 'preflight', 'timing', 'profile'), 'Invalid stage')
    if stage == 'profile':
        for v in VARIANTS:
            yield dict(variant=v, stage='profile', round=0, block=-1, directory='profile-' + v)
    if stage in ('preflight', 'all'):
        for v in VARIANTS:
            yield dict(variant=v, stage='smoke', round=0, block=-1, directory='smoke-' + v)
    if stage in ('timing', 'all'):
        for item in m['schedule']:
            yield dict(item, stage='timing')


def preflight(root):
    root = Path(root)
    check_controller(root)
    m = verify(root)
    fingerprints = devices = execution = None
    hashes = {}
    for v in VARIANTS:
        w = validate_window(root, root / 'results' / ('smoke-' + v),
                            dict(variant=v, stage='smoke', round=0, block=-1, directory='smoke-' + v))
        if fingerprints is None:
            fingerprints, devices, execution = w['fingerprints'], w['devices'], w['execution']
        require(w['fingerprints'] == fingerprints and w['devices'] == devices and w['execution'] == execution,
                'Preflight identity differs across variants')
        for rank in range(8):
            r = read(root / 'results' / ('smoke-' + v) / f'rank{rank}.json')
            expected_checks = 1 if m['kind'] == 'operator' else 64 if m['scope'] == 'both' else 32
            require(len(r.get('forward_checks', {})) == expected_checks, 'Missing numerical checks')
            for c in r['forward_checks'].values():
                require(c['passed'] is True and 0 <= c['relative_l2'] <= .05 and 0 <= c['protocol_relative_l2'] <= .01,
                        'Numerical budget exceeded')
            if m['kind'] == 'operator':
                require(r.get('backward_checked') is True, 'Missing operator STE gradient check')
        hashes.update(w['files'])
    result = dict(protocol=PROTOCOL, completed=True, manifest_sha256=sha(root / 'manifest.json'),
                  fingerprints=fingerprints, devices=devices, execution=execution, files=hashes,
                  note='Only this prototype/configuration passed smoke; this is not a performance claim')
    path = root / 'preflight.json'
    if path.exists():
        require(read(path) == result, 'Preflight result changed')
    else:
        write_new(path, result)
    return result


def report(root):
    root = Path(root)
    check_controller(root)
    m = verify(root)
    gate = preflight(root)
    rounds, hashes = {}, {}
    for round_id in (1, 2):
        values = {v: [] for v in VARIANTS}
        for job in m['schedule']:
            if job['round'] != round_id:
                continue
            w = validate_window(root, root / 'results' / job['directory'], dict(job, stage='timing'))
            require(w['fingerprints'] == gate['fingerprints'] and w['devices'] == gate['devices'] and w['execution'] == gate['execution'],
                    'Timing model/input/device identity differs from preflight')
            values[job['variant']].append(w['ms'])
            hashes.update(w['files'])
        rounds[str(round_id)] = summarize(values)
    result = dict(protocol=PROTOCOL, completed=True, kind=m['kind'], scope=m['scope'],
                  metric='ms/operator' if m['kind'] == 'operator' else 'ms/full_optimizer_step',
                  manifest_sha256=sha(root / 'manifest.json'), rounds=rounds, files=hashes,
                  inference='Four matched prototype variants only; not speedup versus historical Flux/Megatron',
                  confidence='Whole-block bootstrap, no multiple-comparison adjustment')
    path = root / 'report.json'
    if path.exists():
        require(read(path) == result, 'Report differs from frozen result')
    else:
        write_new(path, result)
    return result


def run(root, stage='all', execute=False, python=sys.executable):
    root = Path(root).resolve()
    check_controller(root)
    m = verify(root)
    tasks = list(jobs(m, stage))
    cmd = command(root, python)
    if not execute:
        print('DRY RUN: no Torch import, GPU query, worker, or old-session mutation')
        print(cmd)
        for task in tasks:
            print(task)
        return tasks
    if m['kind'] == 'model':
        actual = subprocess.check_output(['git', '-C', m['megatron'], 'rev-parse', 'HEAD'], text=True).strip()
        require(actual == m['megatron_commit'], 'Wrong Megatron commit')
        require(not subprocess.check_output(['git', '-C', m['megatron'], 'status', '--porcelain', '--untracked-files=no'], text=True).strip(),
                'Megatron tracked source has local changes')
    with (root / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_new(root / (stage + '-started.json'), dict(status='running', tasks=len(tasks)))
        try:
            checked = False
            for job in tasks:
                verify(root)
                if job['stage'] in ('timing', 'profile') and not checked:
                    preflight(root)
                    checked = True
                folder = root / 'results' / job['directory']
                folder.mkdir(parents=True, exist_ok=False)
                env = dict(os.environ, PHASE7_EXECUTE='1', PHASE7_JOB=json_job(job), OMP_NUM_THREADS='1')
                if m['kind'] == 'model':
                    env['PYTHONPATH'] = m['megatron'] + os.pathsep + env.get('PYTHONPATH', '')
                write_new(folder / 'command.json', dict(command=cmd, job=job))
                print('START', job['directory'], flush=True)
                with (folder / 'process.log').open('x') as f:
                    subprocess.run(cmd, env=env, cwd=m['megatron'] or root, stdout=f, stderr=subprocess.STDOUT, check=True)
                validate_window(root, folder, job)
            result = dict(profile_windows=4) if stage == 'profile' else preflight(root) if stage == 'preflight' else report(root)
            write_new(root / (stage + '-finished.json'), dict(status='completed'))
            return result
        except BaseException:
            write_new(root / (stage + '-finished.json'), dict(status='failed'))
            raise


def json_job(job):
    import json
    return json.dumps(job, sort_keys=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare'); q.add_argument('root', type=Path)
    q.add_argument('--parent', type=Path, default=DEFAULT_PLANS)
    q.add_argument('--kind', choices=('operator', 'model'), default='operator')
    q.add_argument('--scope', choices=('attention', 'mlp', 'both'), default='attention')
    q.add_argument('--chunk-tiles', type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=8)
    q.add_argument('--quantization', choices=('legacy', 'bf16', 'all_remote', 'random_matched'), default='legacy')
    q.add_argument('--blocks', type=int, default=4); q.add_argument('--warmup', type=int, default=10)
    q.add_argument('--steps', type=int, default=20); q.add_argument('--megatron', type=Path)
    q = sub.add_parser('run'); q.add_argument('root', type=Path)
    q.add_argument('--stage', choices=('all', 'preflight', 'timing', 'profile'), default='all')
    q.add_argument('--execute', action='store_true'); q.add_argument('--python', default=sys.executable)
    for name in ('verify', 'report'):
        q = sub.add_parser(name); q.add_argument('root', type=Path)
    a = vars(p.parse_args()); action = a.pop('action')
    if action == 'prepare':
        print(prepare(**a))
    elif action == 'run':
        run(**a)
    elif action == 'verify':
        verify(a['root']); print('Frozen inputs verified')
    else:
        print(report(a['root']))


if __name__ == '__main__':
    main()
