"""Resume an interrupted frozen H100 timing campaign without changing its inputs."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TOOLS = Path("/workspace/TempName/tools/h100/phase6")
sys.path.insert(0, str(TOOLS))

from calibrate import environment
from common import MEGATRON_COMMIT, read, require, shape, sha, verify_files, write
from report import preflight, report, verify_inputs, window
from run import arguments, schedule


def main(root, python, cuda_home):
    root = root.resolve()
    info = verify_inputs(root)
    config = info["config"]
    scripts = root / "scripts"
    verify_files("/", info["controllers"])
    megatron = Path(info["megatron"])
    revision = subprocess.check_output(
        ["git", "-C", str(megatron), "rev-parse", "HEAD"], text=True
    ).strip()
    require(revision == MEGATRON_COMMIT, "Megatron revision changed")
    dirty = subprocess.check_output(
        ["git", "-C", str(megatron), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    require(not dirty, "Megatron working tree changed")

    command = [
        str(python), "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={config['tp']}", str(scripts / "worker.py"),
        *arguments(config),
    ]
    lock = (root / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = root / "timing-state.json"
    write(state, {"status": "running", "stage": "timing", "resumed": True})
    try:
        gate = read(root / "preflight.json")
        require(
            gate["completed"] and gate["submission_sha256"] == sha(root / "submission.json"),
            "Preflight required before timing",
        )
        verify_files(root, gate["files"])
        for mode, policy, name, block, index in schedule(info, "timing"):
            verify_inputs(root)
            directory = root / "results" / name
            complete = directory.is_dir() and all(
                (directory / f"rank{rank}.json").is_file()
                for rank in range(config["tp"])
            )
            if complete:
                window(root, directory, policy, mode, block, index)
                print("SKIP", name, flush=True)
                continue
            if directory.exists():
                stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
                backup = directory.with_name(directory.name + f".failed-{stamp}")
                require(not backup.exists(), f"Failure backup already exists: {backup}")
                directory.rename(backup)
                print("ARCHIVE", backup, flush=True)
            directory.mkdir(parents=True)
            library = (
                Path(info["variants"][policy]["root"])
                if policy in info["variants"]
                else root / "no-flux-library"
            )
            env = environment(python, library, config, info["run_id"], scripts, cuda_home)
            env.update(
                E2E_POLICY=policy,
                E2E_MODE=mode,
                E2E_CASE="h100-phase6",
                E2E_BUILD=str(root / "build"),
                E2E_PROCESS_OUT=str(directory),
                E2E_BLOCK=str(block),
                E2E_WINDOW=str(index),
            )
            env["PYTHONPATH"] = f"{scripts}:{megatron}" + (
                f":{library / 'python'}" if policy in info["variants"] else ""
            )
            write(
                directory / "command.json",
                {"command": command, "policy": policy, "mode": mode, "block": block, "window": index},
            )
            print("START", name, flush=True)
            with (directory / "process.log").open("w") as log:
                subprocess.run(
                    command,
                    cwd=megatron,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            window(root, directory, policy, mode, block, index)
        report(root)
        write(state, {"status": "completed", "stage": "timing", "resumed": True})
    except BaseException:
        write(state, {"status": "failed", "stage": "timing", "resumed": True})
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.python, args.cuda_home)
