# VPS Skill

## Connection

```bash
ssh root@82.29.177.234
```

## Project Location

```
/root/autoresearch-matrixpfn/
```

## Pull Latest Changes

```bash
ssh root@82.29.177.234 "cd /root/autoresearch-matrixpfn && git pull"
```

## Run Orchestrator Code

The Python environment with the RunPod SDK is at `/root/autoresearch-env/`.

```bash
ssh root@82.29.177.234 'cd /root/autoresearch-matrixpfn && /root/autoresearch-env/bin/python3 -c "
import sys; sys.path.insert(0, \".\")
from orchestrator import PodManager
pm = PodManager()
for pod in pm.list_pods():
    print(f\"{pod.pod_id} | {pod.name} | {pod.status} | {pod.gpu_type}\")
"'
```

## Start Claude Code on VPS

```bash
ssh root@82.29.177.234
cd /root/autoresearch-matrixpfn
claude
```

Claude Code reads CLAUDE.md automatically and knows how to orchestrate experiments.

## Check VPS Status

```bash
# System resources
ssh root@82.29.177.234 "free -h && df -h / && uptime"

# Running Docker containers
ssh root@82.29.177.234 "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
```

## Environment Variables

```bash
# Already configured in /root/.bashrc:
# RUNPOD_API_KEY — RunPod API authentication
```

## Git Setup

- SSH key: `/root/.ssh/id_ed25519` (label: autoresearch-vps)
- GitHub account: Csed-dev
- Repo cloned at: `/root/autoresearch-matrixpfn/`

## Running Services on VPS

The VPS also runs Nextcloud, n8n, VS Code Server, and Caddy (reverse proxy). These are Docker containers and do not interfere with autoresearch.

## Workflow: Push Changes from Local, Run on VPS

```bash
# 1. Local: make changes, commit, push
cd /Users/jozef/Code/autoresearch-matrixpfn
git add -A && git commit -m "change" && git push

# 2. VPS: pull and run
ssh root@82.29.177.234 "cd /root/autoresearch-matrixpfn && git pull"
ssh root@82.29.177.234  # then start claude code
```
