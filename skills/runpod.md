# RunPod Orchestrator Skill

## Setup

```python
import sys
sys.path.insert(0, "/root/autoresearch-matrixpfn")
from orchestrator import PodManager, ExperimentRunner

pm = PodManager()
```

## Single Run

```python
pod_id = pm.create_pod("exp-001")
conn = pm.wait_until_ready(pod_id)
runner = ExperimentRunner(pm, conn)

try:
    runner.setup_pod(branch="main")
    result = runner.run_experiment()
    print(f"Score: {result.score}")
    print(f"SuiteSparse conv: {result.suitesparse_conv_pct}%")
finally:
    pm.terminate_pod(pod_id)
```

## Queue Mode (sequential experiments, one pod)

```python
pod_id = pm.create_pod("queue-run")
conn = pm.wait_until_ready(pod_id)
runner = ExperimentRunner(pm, conn)
runner.setup_pod(branch="autoresearch/experiment-batch")

try:
    best_score = float("inf")
    for i in range(N):
        # 1. Modify train.py locally, commit, push
        # 2. Sync to pod
        runner.sync_code(branch="autoresearch/experiment-batch")
        # 3. Run
        result = runner.run_experiment()
        # 4. Evaluate
        if result.score < best_score:
            best_score = result.score
            print(f"Run {i}: NEW BEST {result.score:.6f}")
        else:
            print(f"Run {i}: {result.score:.6f} (best: {best_score:.6f})")
finally:
    pm.terminate_pod(pod_id)
```

## Batch Mode (parallel experiments, multiple pods)

```python
experiments = [
    {"name": "batch-large", "branch": "autoresearch/large-model"},
    {"name": "batch-small", "branch": "autoresearch/small-model"},
]

pods = []
try:
    for exp in experiments:
        pod_id = pm.create_pod(exp["name"])
        conn = pm.wait_until_ready(pod_id)
        runner = ExperimentRunner(pm, conn)
        runner.setup_pod(branch=exp["branch"])
        pods.append({"pod_id": pod_id, "runner": runner, "exp": exp})

    for pod in pods:
        result = pod["runner"].run_experiment()
        print(f"{pod['exp']['name']}: score={result.score:.6f}")
finally:
    for pod in pods:
        pm.terminate_pod(pod["pod_id"])
```

## Pod Management

```python
# List all active pods
for pod in pm.list_pods():
    print(f"{pod.pod_id} | {pod.name} | {pod.status} | {pod.gpu_type} | ${pod.cost_per_hr}/hr")

# Stop pod (preserves state, no GPU cost, still billed for storage)
pm.stop_pod(pod_id)

# Terminate pod (destroys everything)
pm.terminate_pod(pod_id)
```

## Run arbitrary commands on pod

```python
output = pm.ssh_run(conn, "nvidia-smi")
print(output)

log = runner.retrieve_file("run.log")
```

## GPU Types

Common choices for autoresearch:

| GPU | VRAM | Use Case |
|-----|------|----------|
| NVIDIA L4 | 24GB | Default. Cheap, sufficient for MatrixPFN |
| NVIDIA RTX 4090 | 24GB | Faster than L4, similar VRAM |
| NVIDIA A100 80GB PCIe | 80GB | Large models, large batch sizes |

```python
pod_id = pm.create_pod("big-run", gpu_type="NVIDIA RTX 4090")
```

## Error Handling

ALWAYS terminate pods in a finally block. Orphaned pods burn money.

If pod creation fails with capacity errors, try a different GPU type:

```python
for gpu in ["NVIDIA L4", "NVIDIA RTX 4090", "NVIDIA RTX A5000"]:
    try:
        pod_id = pm.create_pod("exp", gpu_type=gpu)
        break
    except Exception as e:
        print(f"{gpu} unavailable: {e}")
        continue
```

## Cost Awareness

- L4: ~$0.39/hr
- RTX 4090: ~$0.44/hr
- A100 80GB: ~$1.64/hr
- A single experiment takes ~7 min = ~$0.05 on L4
- Always terminate when done
- Check for orphans: `pm.list_pods()`
