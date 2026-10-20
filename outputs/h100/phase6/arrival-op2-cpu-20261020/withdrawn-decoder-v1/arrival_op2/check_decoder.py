#!/usr/bin/env python3
"""Compile and optionally run the standalone SM90 op-2 decoder gate.

No PyTorch, Flux build, network, or multi-GPU launcher is needed. The scalar
reference is linked directly from the pinned fused_v2 source, never a copy
rewritten by the experiment builder. A decoder microbenchmark is diagnostic;
only the paired end-to-end experiment can establish the user's 1.5% target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PINNED = HERE.parent / "fused_v2" / "src"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_command(nvcc: Path, binary: Path) -> list[str]:
    return [
        str(nvcc), "-std=c++17", "-O2",
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_90,code=compute_90",
        "-Xptxas=-v", "-lineinfo",
        "-I", str(PINNED), "-I", str(HERE),
        str(HERE / "decoder_bench.cu"),
        str(PINNED / "gemm_rs" / "taco_runtime.cu"),
        "-o", str(binary),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "artifacts" / "arrival_op2_decoder")
    parser.add_argument("--cuda-home", type=Path)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute", action="store_true",
                           help="run GPU correctness and timings after compilation")
    execution.add_argument("--compile-only", action="store_true",
                           help="compile without using a GPU (the default)")
    parser.add_argument("--check-only", action="store_true",
                        help="run correctness without timing")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--sanitizer", choices=("memcheck", "racecheck", "synccheck"),
                        help="run the correctness gate under Compute Sanitizer")
    args = parser.parse_args()
    if args.samples < 1 or args.iterations < 1:
        parser.error("--samples and --iterations must be positive")
    if not args.execute and (args.check_only or args.sanitizer):
        parser.error("--check-only and --sanitizer require --execute")
    nvcc = (args.cuda_home / "bin" / "nvcc" if args.cuda_home else
            Path(shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"))
    if not nvcc.is_file():
        parser.error("CUDA Toolkit not found; pass --cuda-home")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = output / "decoder_bench"
    sources = [
        HERE / "decoder.cuh", HERE / "decoder_bench.cu",
        PINNED / "gemm_rs" / "taco_runtime.cu",
        PINNED / "gemm_rs" / "taco_runtime.h",
        PINNED / "gemm_rs" / "taco_codec.cuh",
    ]
    command = compile_command(nvcc, binary)
    record: dict = {
        "schema": "arrival_op2.decoder_build.v1",
        "compile_command": command,
        "source_sha256": {str(path.relative_to(REPO)): digest(path) for path in sources},
        "gpu_validation": {"status": "not_run"},
    }
    version = subprocess.run([str(nvcc), "--version"], text=True, capture_output=True)
    record["nvcc_version"] = version.stdout.strip()
    built = subprocess.run(command, text=True, capture_output=True)
    (output / "compile.log").write_text(built.stdout + built.stderr)
    record["compile_returncode"] = built.returncode
    if built.returncode:
        record["status"] = "compile_failed"
        (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
        print(built.stderr, file=sys.stderr)
        return built.returncode
    record["binary_sha256"] = digest(binary)
    record["status"] = "compiled"
    result_code = 0
    if args.execute:
        command = [str(binary), "--samples", str(args.samples),
                   "--iterations", str(args.iterations)]
        if args.check_only or args.sanitizer:
            command.append("--check-only")
        if args.sanitizer:
            sanitizer = nvcc.parent / "compute-sanitizer"
            if not sanitizer.is_file():
                sanitizer = Path(shutil.which("compute-sanitizer") or "")
            if not sanitizer.is_file():
                parser.error("compute-sanitizer not found")
            # Keep tool diagnostics separate from the application's JSON.
            command = [str(sanitizer), "--tool", args.sanitizer,
                       "--error-exitcode", "1", "--log-file",
                       str(output / f"{args.sanitizer}.log"), *command]
        record["run_command"] = command
        env = dict(os.environ)
        run = subprocess.run(command, text=True, capture_output=True, env=env)
        (output / "run.stderr.log").write_text(run.stderr)
        (output / "decoder.json").write_text(run.stdout)
        record["run_returncode"] = run.returncode
        try:
            result = json.loads(run.stdout)
        except json.JSONDecodeError:
            result = {"status": "invalid_json", "stdout_file": str(output / "decoder.json")}
        record["gpu_validation"] = result
        result_code = run.returncode
        if result_code == 0 and result.get("status") != "passed":
            result_code = 1
        record["status"] = ("gpu_unavailable" if result_code == 77 else
                            "passed" if result_code == 0 else "failed")
    (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "build_json": str(output / "build.json"),
                      "gpu_validation": record["gpu_validation"].get("status")}))
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
