# CLAUDE.md

## What This Is

Autonomous MatrixPFN research. An AI agent modifies `train.py`, runs 5-minute training experiments on RunPod GPUs, keeps improvements, discards regressions, and repeats indefinitely. Adapted from [autoresearch](https://github.com/karpathy/autoresearch) by @karpathy.

## Infrastructure

This runs on a VPS (82.29.177.234) that orchestrates RunPod GPU pods:

```
VPS (Claude Code)  →  RunPod API: create pod
                   →  SSH into pod: git clone, setup, run experiments
                   →  SSH: retrieve logs + results
                   →  RunPod API: terminate pod
```

The orchestrator module (`orchestrator/`) handles pod lifecycle and SSH. See `skills/runpod.md` for usage patterns.

## Commands

```bash
# On VPS — run orchestrator code
/root/autoresearch-env/bin/python3 -c "
from orchestrator import PodManager, ExperimentRunner
pm = PodManager()
pod_id = pm.create_pod('exp-001')
conn = pm.wait_until_ready(pod_id)
runner = ExperimentRunner(pm, conn)
runner.setup_pod()
result = runner.run_experiment()
print(f'score: {result.score}')
pm.terminate_pod(pod_id)
"

# On pod (via SSH) — manual commands
uv sync                          # install dependencies
uv run prepare.py                # one-time SuiteSparse download
uv run train.py                  # run experiment (~5 min train + ~2 min eval)
```

## Architecture

### Files

- **`train.py`** — THE ONLY FILE THE AGENT EDITS. Model config, domain selection, training loop.
- **`prepare.py`** — READ-ONLY. Evaluation harness, scoring function, constants.
- **`program.md`** — Research directions. THE HUMAN edits this.
- **`orchestrator/`** — RunPod pod management + SSH execution. Do not modify unless infrastructure changes.
- **`skills/runpod.md`** — Skill file: RunPod API patterns, usage examples, cost info.

### Experiment Workflow

1. Modify `train.py` (hyperparameters, domains, architecture)
2. `git commit && git push`
3. Create RunPod pod via orchestrator
4. Setup pod (git clone, uv sync, prepare.py) — only on first run or new pod
5. `sync_code()` to pull latest changes
6. `run_experiment()` — returns `ExperimentResult` with score and all metrics
7. Evaluate: if score improved → keep. If worse → revert train.py, commit, push.
8. Repeat from step 1.

### Three Modes

**Single Run** — One experiment, one pod. Create → setup → run → terminate.

**Queue Mode** — Sequential experiments on one pod. Create pod once, loop: modify → push → sync → run → evaluate. Pod stays alive between runs (no setup overhead). Terminate when done.

**Batch Mode** — Parallel experiments on multiple pods. Each pod gets a different branch with different train.py modifications. Run all, compare, keep best. Terminate all.

See `skills/runpod.md` for code examples of each mode.

### Iterative Research Loop

The agent works in an infinite loop:

1. Analyze previous results (score, convergence, per-matrix details)
2. Form a hypothesis (which domain to add, which hyperparameter to change)
3. Modify `train.py` accordingly
4. Commit and push
5. Run on GPU pod via orchestrator
6. Parse results, compare to best score
7. Keep or revert
8. Go to 1

Use queue mode for this — one pod stays alive, experiments run back-to-back.

## Key Metric

**`score`** — lower is better. Composite: 30% synthetic + 70% SuiteSparse. Each component: mean(iterations / max_iterations). Non-convergent = 1.0 (worst). Perfect = ~0.003 (best).

## MatrixPFN Background

MatrixPFN trains a GNN (ContextResGCN) to approximate A⁻¹r. The trained model serves as a nonlinear preconditioner inside FGMRES. Key difference from GNP (Chen 2025): MatrixPFN trains ONE model across many matrix domains using context pairs (x, Ax), enabling zero-shot preconditioning on unseen matrices.

### Available domains (matrixpfn.generator.base.MatrixDomain)

| Domain | SPD? | Symmetric? | Structure |
|--------|------|-----------|-----------|
| DIFFUSION | Yes | Yes | 5-point Laplace stencil |
| DIFFUSION_ADVECTION | No | No | Diffusion + transport |
| GRAPH_LAPLACIAN | Yes | Yes | Barabasi-Albert |
| ELASTICITY | Yes | Yes | 2D structural stiffness |
| STOKES | No | Yes | Saddle-point [A B; B^T C] |
| SBM | Yes | Yes | Stochastic Block Model |
| SPECTRAL_STRESS | Yes | Yes | Ill-conditioned diffusion |
| VARIABLE_DIFFUSION | Yes | Yes | Material discontinuities |
| VARIABLE_ADVECTION | No | No | Variable diff + advection |
| ENHANCED_DIFFUSION | Yes | Yes | Anisotropic + holes |
| ENHANCED_ADVECTION | No | No | Anisotropic + holes + advection |
| RANDOM_SPARSE | Yes-ish | Yes | Random diag-dominant |
| DIRECTED_POWER_LAW | No | No | Directed Barabasi-Albert |

### SuiteSparse eval matrices (all non-SPD)

sherman1 (n=1000), sherman3 (n=5005), sherman4 (n=1104), rdb1250 (n=1250), pde2961 (n=2961), epb0 (n=1794), thermal (n=3456), orsirr_1 (n=1030), orsreg_1 (n=2205), watt__1 (n=1856), saylr4 (n=3564).

Training on non-symmetric domains (DIFFUSION_ADVECTION, VARIABLE_ADVECTION, ENHANCED_ADVECTION, STOKES, DIRECTED_POWER_LAW) is critical for SuiteSparse performance since all eval matrices are non-SPD.

## Constraints

- Only modify `train.py`
- No new dependencies beyond `pyproject.toml`
- Single GPU (default: L4 24GB)
- Training runs exactly 5 minutes wall-clock
- If a run exceeds 15 minutes total, it is killed
- ALWAYS terminate RunPod pods when done — orphaned pods burn money
- Check `pm.list_pods()` before starting to clean up orphans
- If pod creation times out (2 min), the pod is auto-terminated. Check `pm.get_available_gpus()` for GPUs with stock, then update `orchestrator/config.py` GPU_TYPE_DEFAULT or pass a different `gpu_type` to `create_pod()`
