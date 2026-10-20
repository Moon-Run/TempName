#!/usr/bin/env python3
"""Op2: old A versus warp decoder D, with arrival-order R as a separate diagnostic."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import arrival_op2_build as builder
import arrival_op2_prepare as preparation
import arrival_op2_protocol as routing
import arrival_op2_report as reporter
import ablation4_runtime as communication
import scope_run
from calibrate import environment
from common import HERE, REPO, read, require, sha, verify_files, write


def decoder_check(out, cuda_home, execute=False):
    """Compile without GPUs by default; explicit execution runs the exact-bit gate."""
    out, cuda_home = Path(out).resolve(), Path(cuda_home).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = HERE/'arrival_op2'
    pinned = HERE/'fused_v2/src'
    inputs = [source/'decoder.cuh', source/'decoder_bench.cu',
              pinned/'gemm_rs/taco_runtime.cu', pinned/'gemm_rs/taco_runtime.h',
              pinned/'gemm_rs/taco_codec.cuh']
    command = [str(cuda_home/'bin/nvcc'), '-std=c++17', '-O2',
               '-gencode=arch=compute_90,code=sm_90', '-Xptxas=-v',
               '-I'+str(source), '-I'+str(pinned), str(source/'decoder_bench.cu'),
               str(pinned/'gemm_rs/taco_runtime.cu'), '-o', str(out/'decoder-check')]
    result = dict(protocol=routing.PROTOCOL, completed=False, gpu_executed=False,
                  correctness_passed=False, source_sha256={str(p): sha(p) for p in inputs},
                  compile_command=command)
    try:
        with (out/'compile.log').open('x') as log:
            subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result.update(compiled=True, binary_sha256=sha(out/'decoder-check'))
        if execute:
            run = subprocess.run([str(out/'decoder-check')], capture_output=True, text=True, check=False)
            (out/'stdout.json').write_text(run.stdout)
            (out/'stderr.log').write_text(run.stderr)
            require(run.returncode == 0, 'Op2 decoder GPU correctness failed; see '+str(out/'stderr.log'))
            measured = json.loads(run.stdout)
            require(measured['correctness_passed'] is True and measured['bitwise_equal'] is True,
                    'Op2 decoder must match the original decoder bit for bit')
            result.update(gpu_executed=True, correctness_passed=True, measured=measured)
        result['completed'] = True
    finally:
        write(out/'validation.json', result)
    return result


def verify_decoder_check(path):
    path = Path(path).resolve()
    record = read(path)
    require(record['protocol'] == routing.PROTOCOL and record['completed'] and
            record['gpu_executed'] and record['correctness_passed'] and
            record['measured']['bitwise_equal'], 'GPU decoder bitwise check required before training')
    verify_files('/', record['source_sha256'])
    require(sha(path.parent/'decoder-check') == record['binary_sha256'], 'Decoder check binary changed')
    return record


def jobs(info, stage):
    return scope_run.jobs(info, stage, routing)


def mapping_jobs(root, info):
    return scope_run.mapping_jobs(root, info, routing)


def run(root, stage='all', execute=False, python=Path(sys.executable), cuda_home=None,
        decoder_validation=None):
    root = Path(root)
    if execute:
        require(decoder_validation is not None, 'Pass --decoder-validation from a GPU decoder-check --execute')
        checked = verify_decoder_check(decoder_validation)
        evidence = root/'decoder-validation.json'
        if evidence.exists():
            require(read(evidence) == checked, 'This comparison already has different decoder validation')
        else:
            with evidence.open('x') as handle:
                json.dump(checked, handle, indent=2)
                handle.write('\n')
    record = read(root/'submission.json')['communication']
    return scope_run.run(root, stage, execute, python, cuda_home, routing, reporter,
        'H100_ARRIVAL_OP2_EXECUTE',
        environment_fn=lambda *args: communication.apply(environment(*args), record))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('build-prepare')
    q.add_argument('out', type=Path)
    q.add_argument('--parent-build', type=Path, required=True)
    q.add_argument('--source-only', action='store_true')
    q = sub.add_parser('compile')
    q.add_argument('build', type=Path)
    q.add_argument('--cuda-home', type=Path, required=True)
    q.add_argument('--jobs', type=int, default=2)
    q.add_argument('--nccl-root', type=Path)
    q = sub.add_parser('compile-check')
    q.add_argument('build', type=Path)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--cuda-home', type=Path, required=True)
    q.add_argument('--cxx', default='g++')
    q = sub.add_parser('decoder-check')
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--cuda-home', type=Path, required=True)
    q.add_argument('--execute', action='store_true')
    q = sub.add_parser('prepare')
    q.add_argument('out', type=Path)
    q.add_argument('--build', type=Path, required=True)
    q.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    q.add_argument('--blocks', type=int, default=6)
    q.add_argument('--rounds', type=int, choices=(1, 2), default=2)
    q.add_argument('--nvls-enable', choices=('0', 'auto'), default='0')
    q.add_argument('--source-only', action='store_true')
    q = sub.add_parser('run')
    q.add_argument('root', type=Path)
    q.add_argument('--stage', choices=('preflight', 'timing', 'all'), default='all')
    q.add_argument('--python', type=Path, default=Path(sys.executable))
    q.add_argument('--cuda-home', type=Path)
    q.add_argument('--decoder-validation', type=Path)
    q.add_argument('--execute', action='store_true')
    q = sub.add_parser('report')
    q.add_argument('root', type=Path)
    q.add_argument('--preflight', action='store_true')
    q.add_argument('--offline', action='store_true')
    return p


def main():
    args = parser().parse_args()
    if args.action == 'build-prepare':
        print(builder.prepare(args.out, args.parent_build, args.source_only))
    elif args.action == 'compile':
        builder.compile_build(args.build, args.cuda_home, args.jobs, args.nccl_root)
    elif args.action == 'compile-check':
        builder.compile_check(args.build, args.out, args.cuda_home, args.cxx)
    elif args.action == 'decoder-check':
        decoder_check(args.out, args.cuda_home, args.execute)
    elif args.action == 'prepare':
        print(preparation.prepare(args.out, args.build, args.megatron, args.blocks,
                                  args.rounds, args.nvls_enable, args.source_only))
    elif args.action == 'run':
        run(args.root, args.stage, args.execute, args.python, args.cuda_home, args.decoder_validation)
    else:
        (reporter.preflight if args.preflight else reporter.report)(args.root, args.offline)


if __name__ == '__main__':
    main()
