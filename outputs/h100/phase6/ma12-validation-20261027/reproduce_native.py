#!/usr/bin/env python3
"""Generate exact MA89/MA12 sources and compile the native boundary without a GPU.

Run from any directory. Requires the repository's submodules, CUDA 12.8, g++,
and the pinned PyTorch 2.6 environment. No training or GPU kernel is executed.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tools/h100/phase6'))
import ma4_f_build
import ma9_sources
import ma12_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('/tmp') / ('ma12-native-reproduce-' + str(os.getpid())))
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'reproduced-native')
    parser.add_argument('--cuda-home', type=Path, default=Path('/data/apps/cuda/12.8'))
    parser.add_argument('--torch-python', type=Path,
        default=Path('/data/run01/scyb672/conda_envs/flux-megatron-a800/bin/python'))
    args = parser.parse_args()
    if args.work.exists() or args.out.exists():
        parser.error('Use fresh --work and --out directories; prior evidence is never overwritten')
    args.out.mkdir(parents=True)
    ma4_f_build.prepare(args.work / 'build')
    baseline = args.work / 'build/F4'
    for name, source in ma9_sources.sources(baseline).items():
        target = baseline / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    candidate = args.work / 'candidate'
    shutil.copytree(baseline, candidate, symlinks=True)
    changes = ma12_sources.sources(baseline)
    for name, source in changes.items():
        target = candidate / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    probe = args.work / 'producer.cu'
    probe.write_text((Path(__file__).parent / 'producer-probe.cu').read_text())
    nvcc = str(args.cuda_home / 'bin/nvcc')
    flags = ['-std=c++17', '-O2', '--use_fast_math', '--expt-relaxed-constexpr',
             '--expt-extended-lambda', '-arch=sm_90', '-Xptxas=-v,-warn-spills']
    includes = lambda root: ['-I' + str(root / p) for p in
        ('src', 'include', '3rdparty/cutlass/include', '3rdparty/cutlass/tools/util/include')]
    commands = []

    def run(name, command):
        commands.append(dict(name=name, argv=[str(x) for x in command]))
        with (args.out / (name + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    for name, root in [('baseline-producer', baseline), ('producer', candidate)]:
        run(name, [nvcc, *flags, '-DFLUX_TACO_BASELINE', *includes(root), '-c',
            str(probe), '-o', str(args.work / (name + '.o'))])
    run('runtime', [nvcc, *flags, *includes(candidate), '-c',
        str(candidate / ma12_sources.RUNTIME), '-o', str(args.work / 'runtime.o')])
    library = args.work / 'runtime.so'
    run('runtime-versioned', [nvcc, *flags, *includes(candidate), '-Xcompiler=-fPIC', '--shared',
        str(candidate / ma12_sources.RUNTIME), '-Xlinker',
        '--version-script=' + str(candidate / ma12_sources.VERSION_SCRIPT), '-o', str(library)])
    torch_info = json.loads(subprocess.check_output([str(args.torch_python), '-c',
        'import json,sysconfig;from torch.utils.cpp_extension import include_paths;'
        'print(json.dumps(include_paths()+[sysconfig.get_path("include")]))'], text=True).splitlines()[-1])
    run('wrapper', ['g++', '-std=c++17', '-O0', '-fPIC', '-DFLUX_TACO_BASELINE',
        '-D_GLIBCXX_USE_CXX11_ABI=0', *includes(candidate), '-I' + str(args.cuda_home / 'include'),
        *['-I' + p for p in torch_info], '-c', str(candidate / ma12_sources.WRAPPER),
        '-o', str(args.work / 'wrapper.o')])
    exports = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True)
    for name in (*ma12_sources.NATIVE_EXPORTS, *ma12_sources.ZERO_COUNT_EXPORTS):
        if not any(line.split()[-1] == name for line in exports.splitlines()):
            raise RuntimeError('Missing native export: ' + name)
    ctypes.CDLL(str(args.cuda_home / 'lib64/libcudart.so'), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(library))
    lib.taco_ma12_protocol_version.restype = ctypes.c_int
    assert lib.taco_ma12_protocol_version() == 1
    (args.out / 'native-exports.txt').write_text(exports)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    record = dict(completed=True, gpu_used=False, gpu_validated=False, full_flux_linked=False,
        commands=commands, source_sha256={n: sha(candidate / n) for n in changes},
        inputs={n: sha(REPO / 'tools/h100/phase6' / n) for n in ('ma12_sources.py', 'ma12_tile_ready.cuh')},
        limitations=['Compile/link/CPU ABI evidence only; no GPU execution or performance claim.'])
    (args.out / 'native-validation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(args.out / 'native-validation.json')


if __name__ == '__main__':
    main()
