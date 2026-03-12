# CLAUDE.md

## What This Is

Autonomous MatrixPFN research. An AI agent modifies `train.py`, runs 5-minute training experiments on a single GPU, keeps improvements, discards regressions, and repeats indefinitely. Adapted from [autoresearch](https://github.com/karpathy/autoresearch) by @karpathy.

## Commands

```bash
uv sync                          # install dependencies (matrixpfn from PyPI)
uv run prepare.py                # one-time SuiteSparse matrix download
uv run train.py                  # run a training experiment (~5 min train + ~2 min eval)
uv run train.py > run.log 2>&1   # run with output capture (preferred)
grep "^score:\|^suitesparse_conv:" run.log  # extract key metrics
```

## Architecture

Three files matter:

- **`prepare.py`** — READ-ONLY. SuiteSparse download, evaluation function `evaluate_score()`, constants (`TIME_BUDGET=300`, `EVAL_MATRICES`, solver config). Scoring: `0.3 * synthetic + 0.7 * suitesparse`, where each is mean normalized iteration count (lower = better).
- **`train.py`** — THE ONLY FILE THE AGENT EDITS. Model config (ContextResGCN: layers, embed, hidden), domain selection and weights, training hyperparameters, training loop. Uses `matrixpfn` package.
- **`program.md`** — Agent instructions defining the autonomous experiment loop. THE HUMAN edits this.

## Key Metric

**`score`** — lower is better. Composite: 30% synthetic diffusion performance + 70% SuiteSparse real-world performance. Each component is mean(iterations / max_iterations) across test problems. Non-convergent cases score 1.0 (worst). Perfect convergence in 1 iteration scores ~0.003 (best).

## MatrixPFN Background

MatrixPFN trains a GNN (ContextResGCN) to approximate A⁻¹r. The trained model serves as a nonlinear preconditioner inside FGMRES. Key difference from GNP (Chen 2025): MatrixPFN trains ONE model across many matrix domains using context pairs (x, Ax), enabling zero-shot preconditioning on unseen matrices.

### Available domains (matrixpfn.generator.base.MatrixDomain)

| Domain | SPD? | Symmetric? | Structure |
|--------|------|-----------|-----------|
| DIFFUSION | Yes | Yes | 5-point Laplace stencil |
| DIFFUSION_ADVECTION | No | No | Diffusion + transport |
| GRAPH_LAPLACIAN | Yes | Yes | Barabási-Albert |
| ELASTICITY | Yes | Yes | 2D structural stiffness |
| STOKES | No | Yes | Saddle-point [A B; B^T C] |
| SBM | Yes | Yes | Stochastic Block Model |
| SPECTRAL_STRESS | Yes | Yes | Ill-conditioned diffusion |
| VARIABLE_DIFFUSION | Yes | Yes | Material discontinuities |
| VARIABLE_ADVECTION | No | No | Variable diff + advection |
| ENHANCED_DIFFUSION | Yes | Yes | Anisotropic + holes |
| ENHANCED_ADVECTION | No | No | Anisotropic + holes + advection |
| RANDOM_SPARSE | Yes-ish | Yes | Random diag-dominant |
| DIRECTED_POWER_LAW | No | No | Directed Barabási-Albert |

### SuiteSparse eval matrices (all non-SPD)

sherman1 (n=1000), sherman3 (n=5005), sherman4 (n=1104), rdb1250 (n=1250), pde2961 (n=2961), epb0 (n=1794), thermal (n=3456), orsirr_1 (n=1030), orsreg_1 (n=2205), watt__1 (n=1856), saylr4 (n=3564).

Training on non-symmetric domains (DIFFUSION_ADVECTION, VARIABLE_ADVECTION, ENHANCED_ADVECTION, STOKES, DIRECTED_POWER_LAW) is critical for SuiteSparse performance since all eval matrices are non-SPD.

## Constraints

- Only modify `train.py`
- No new dependencies beyond `pyproject.toml`
- Single GPU (tested on L4 24GB)
- Training runs exactly 5 minutes wall-clock
- If a run exceeds 15 minutes total, kill and treat as failure
