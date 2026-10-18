#!/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python
import os
from pathlib import Path
import sys

repo = Path('/workspace/TempName')
session = 'ablation4-gpt67-tp8-both-20261018-hostservice'
log_root = repo/'logs/h100/phase6'/session
script = log_root/'run-two.sh'
python = Path('/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python')
log_path = log_root/'handoff/container-supervisor.log'
pid_path = log_root/'handoff/container-supervisor.pid'

read_fd, write_fd = os.pipe()
first = os.fork()
if first:
    os.close(write_fd)
    with os.fdopen(read_fd, 'r') as pipe:
        detached_pid = int(pipe.readline())
    os.waitpid(first, 0)
    print(detached_pid)
    raise SystemExit(0)

os.close(read_fd)
os.setsid()
second = os.fork()
if second:
    os.write(write_fd, f'{second}\n'.encode())
    os.close(write_fd)
    os._exit(0)

os.close(write_fd)
os.chdir(repo)
os.umask(0o027)
with pid_path.open('w') as f:
    f.write(f'{os.getpid()}\n')
fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.close(fd)
os.execv('/bin/bash', ['/bin/bash', str(script), str(repo), str(python), str(log_root)])
