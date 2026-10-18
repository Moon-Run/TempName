# Host Start Command

Preparation is complete, but this container cannot access host systemd or Docker. On host `gxn75`, identify the running experiment container and start the prepared system service:

```bash
docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}'
CONTAINER='<the running container that mounts /workspace/TempName>'
bash /workspace/TempName/logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/handoff/start-host-service.sh "$CONTAINER"
```

The launcher uses service unit `h100-ablation4-two-runs-20261018-hostservice.service`. It runs `run-two.sh` inside the existing container, waits for compare1's complete strict report, then runs compare2. It uses `Restart=no` and `NCCL_NVLS_ENABLE=0`.

The launcher prints systemd state, `MainPID`, `ControlGroup`, exit result and journal tail, then `docker top`. Keep VSCode open until host-side verification confirms the system service is active and its MainPID is the foreground Docker exec client; confirm the container worker/control process is visible under that service's invocation. The command has not been run from this container.
