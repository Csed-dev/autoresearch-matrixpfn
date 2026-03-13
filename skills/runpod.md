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
| NVIDIA RTX A5000 | 24GB | Default. Cheapest 24GB option (~$0.16/hr) |
| NVIDIA GeForce RTX 3090 | 24GB | Good alternative (~$0.22/hr) |
| NVIDIA GeForce RTX 4090 | 24GB | Faster compute (~$0.34/hr) |
| NVIDIA RTX A6000 | 48GB | Large models (~$0.33/hr) |
| NVIDIA A100 80GB PCIe | 80GB | Maximum VRAM (~$1.19/hr) |

```python
pod_id = pm.create_pod("big-run", gpu_type="NVIDIA RTX 4090")
```

## Error Handling

ALWAYS terminate pods in a finally block. Orphaned pods burn money.

If pod creation times out (2 min), the pod is auto-terminated. Check available GPUs and pick one with stock:

```python
gpus = pm.get_available_gpus(min_memory_gb=20)
for g in gpus:
    gpu_id = g["id"]
    mem = g["memory_gb"]
    price = g["price_per_hr"]
    print(f"{gpu_id:45s} {mem:>4}GB  ${price:.2f}/hr")
```

Then retry with a different GPU type:

```python
pod_id = pm.create_pod("exp", gpu_type="NVIDIA RTX A5000")
```

Update `orchestrator/config.py` GPU_TYPE_DEFAULT if the default GPU is persistently unavailable.

## Cost Awareness

- RTX A5000: ~$0.16/hr (default)
- RTX 3090: ~$0.22/hr
- RTX A6000: ~$0.33/hr
- RTX 4090: ~$0.34/hr
- A100 80GB: ~$1.19/hr
- A single experiment takes ~7 min = ~$0.02 on RTX A5000
- Always terminate when done
- Check for orphans: `pm.list_pods()`
