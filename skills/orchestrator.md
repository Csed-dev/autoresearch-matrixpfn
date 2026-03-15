# Experiment Orchestrator Skill

## Setup

```python
import sys
sys.path.insert(0, "/root/autoresearch-matrixpfn")
from orchestrator import ExperimentManager

mgr = ExperimentManager("experiments.yaml", "scheduler_config.yaml")
```

## Core Workflow

```python
# 1. Validate configuration
print(mgr.validate())

# 2. Check what's ready
ready = mgr.get_ready()
print(f"Ready: {ready}")

# 3. Start an experiment (returns pod_id and SSH command)
pod_id, ssh_cmd = mgr.start("baseline")

# 4. Run the SSH command in background — you will be notified when it finishes
# Execute ssh_cmd via Bash tool with run_in_background=True

# 5. When process finishes, record the result
mgr.finish("baseline", exit_code=0, output=stdout_from_ssh)

# 6. Check what's ready next
print(mgr.get_ready())
```

## Status and Results

```python
# Full status overview
print(mgr.status())

# Detailed result for one experiment (config + metrics + log tail)
print(mgr.result("baseline"))

# Available templates
print(mgr.available_templates())

# Available GPUs with prices
print(mgr.available_gpus())
```

## Error Recovery

```python
# GPU unavailable — error includes alternatives
try:
    mgr.start("baseline", gpu_type="NVIDIA L4")
except GpuUnavailableError as e:
    print(e)  # shows cheapest available GPUs
    mgr.start("baseline", gpu_type="NVIDIA RTX A5000")

# Experiment failed — reset and retry
mgr.reset("baseline")
mgr.start("baseline")

# Orphaned pods — cleanup
print(mgr.cleanup())
```

## Pod Lifecycle

Each `mgr.start()` call:
1. Creates a new RunPod pod
2. Waits for SSH ready (2 min timeout, auto-terminates on failure)
3. Runs setup_command (git clone, deps — from scheduler_config.yaml)
4. Returns the SSH command to execute the experiment

Each `mgr.finish()` call:
1. Records result (log, metrics, status)
2. Terminates the pod
3. Propagates BLOCKED to dependents if failed

## Parallel Experiments

```python
for name in mgr.get_ready():
    pod_id, ssh_cmd = mgr.start(name)
    # Run ssh_cmd in background
    # Each will finish independently and wake you up
```

max_pods (from scheduler_config.yaml) limits concurrent experiments.

## Configuration Files

- `scheduler_config.yaml` — Infrastructure: GPU, image, timeouts, repo URL
- `experiments.yaml` — Experiment DAG: templates, experiments, dependencies

## Results Directory

```
results/
  baseline/
    config.yaml    # exact snapshot of what ran
    log.txt        # full stdout/stderr
    status.json    # timing, exit code, pod info
```
