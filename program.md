# autoresearch-matrixpfn

Autonomous research for MatrixPFN: a GNN-based learned preconditioner for sparse linear systems.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the files**: Read these for full context:
   - `CLAUDE.md` — project context, architecture, what matters.
   - `prepare.py` — fixed evaluation, SuiteSparse download, scoring. Do not modify.
   - `train.py` — the file you modify. Model config, domains, training loop.
   - `results/v1-results.tsv` — previous experiment results. Do not repeat these.
4. **Verify data exists**: Check that `~/.cache/autoresearch-matrixpfn/suitesparse/` contains matrices. If not, run `uv run prepare.py`.
5. **Initialize results.tsv**: Create with just the header row.
6. **Confirm and go**.

## Experimentation

Each experiment trains MatrixPFN for a **fixed time budget of 5 minutes**, then evaluates on synthetic + SuiteSparse matrices. Launch: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — everything is fair game: model architecture, training loop, loss function, new model classes, anything.
- Define new model architectures directly in `train.py` if needed.

**What you CANNOT do:**
- Modify `prepare.py`. It contains the fixed evaluation and scoring.
- Install new packages beyond `pyproject.toml`.

**The goal: get the lowest `score`.** The score is `0.3 * synthetic_score + 0.7 * suitesparse_score`, where each sub-score is the mean normalized iteration count (iterations / max_iterations). Lower = better. A score of 0.0 means instant convergence on everything. A score of 1.0 means nothing converges.

## Output format

The script prints a summary:

```
---
score:             0.456789
synthetic_score:   0.123456
suitesparse_score: 0.600000
synthetic_conv:    100.0%
suitesparse_conv:  33.3%
...
```

Extract the key metric: `grep "^score:" run.log`

## Logging results

Log to `results.tsv` (tab-separated, untracked by git):

```
commit	score	ss_conv	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. score (e.g. 0.456789) — use 0.000000 for crashes
3. suitesparse_conv percentage (e.g. 33.3) — use 0.0 for crashes
4. peak memory in GB (peak_vram_mb / 1024, round to .1f) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short description

## The experiment loop

LOOP FOREVER:

1. Look at git state and previous results
2. Modify `train.py` with an experimental idea
3. git commit
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^score:\|^suitesparse_conv:\|^peak_vram_mb:" run.log`
6. If grep empty → crash. Run `tail -n 50 run.log` for traceback.
7. Record in results.tsv
8. If score improved (lower): keep the commit
9. If score equal or worse: `git reset --hard HEAD~1`

**Timeout**: If a run exceeds 15 minutes total, kill and treat as failure.

**NEVER STOP**: Run autonomously until manually interrupted. If stuck, think harder.

## v1 Results Summary (COMPLETED — do not repeat)

23 experiments were run on 2025-03-13 on an NVIDIA L4 GPU. Key findings:

- **Best score: 0.5832** (large model embed=256, hidden=512, 6 layers, context=8)
- **SuiteSparse convergence stuck at 27.3% (3/11)** across ALL 23 experiments
- Same 3 matrices always converge: thermal, sherman4, pde2961
- Same 8 matrices always fail: sherman1, sherman3, rdb1250, epb0, orsirr_1, orsreg_1, watt_1, saylr4
- Classical benchmark: ILU and AMG solve all 11 matrices at 100%

**What was tried and exhausted:**
- Domain selection: 8 domains, 9 (+SBM), 9 (+RANDOM_SPARSE), 12, boosted advection
- Model size: embed 64-256, hidden 128-512
- Layers: 4, 6, 8, 12
- Context pairs: 5, 8, 16
- Learning rate: 3e-4, 1e-3, 3e-3
- LR schedule: constant, cosine
- Grid sizes: (16,24,32), (16,24,32,48)
- Batch config: matrices_per_epoch 16, 32
- Combinations of the above

**Root cause identified:** The GCN's local receptive field (L hops) cannot capture global matrix structure. GCN message passing is structurally equivalent to weighted Jacobi iterations. ILU/AMG exploit global structure (factorization, multigrid hierarchy) that local message passing cannot reach.

## Research directions (v2)

The v1 hyperparameter search is exhausted. Further tuning of ContextResGCN will NOT break the 27.3% plateau. The architecture must change.

### Phase 5: Sparse M prediction (HIGH PRIORITY)
Instead of predicting x ≈ A⁻¹r (vector output, called every FGMRES step), predict a sparse preconditioner matrix M (called once, then z = Mr is cheap SpMV).

- Define a new model class in `train.py` that outputs sparse G with nnz(G) = nnz(A)
- Final preconditioner: M = D⁻¹(I + G), where G has same sparsity pattern as A
- The GCN predicts g_ij per edge (linear output head, unbounded)
- Application: z = M @ r (one SpMV per FGMRES iteration, no GNN forward pass)
- This is the SPAI (Sparse Approximate Inverse) approach with learned entries

### Phase 6: Node features
The current model only sees diag(A) as node feature. Add:
- Degree: number of non-zeros per row
- Diagonal dominance: |a_ii| / Σ_j |a_ij|
- Row/column norms
- These are cheap to compute and give the model structural information

### Phase 7: Knowledge Distillation from ILU
Instead of training on ||x_pred - x_true||, train on:
- ||M_pred @ A - I|| (how well does M precondition A?)
- Or: minimize FGMRES iterations directly (reinforcement-style)
- Or: approximate the ILU preconditioner entries (distillation)

### Phase 8: Multi-scale GCN
Add graph coarsening (pooling) to give the GCN global reach:
- Use TopKPool or learned pooling (NOT AMG — that would add classical dependency)
- Encoder: fine → coarse, Decoder: coarse → fine (U-Net style)
- Skip connections between encoder and decoder at same level

### Phase 9: Chebyshev spectral conditioning
Compute K Chebyshev moments of A (K matrix-vector products) to capture spectral properties.
Use FiLM conditioning (γ, β per layer) to inject this global info into every GCN layer.

## Classical preconditioner benchmark (reference)

All 11 SuiteSparse matrices tested with identical FGMRES settings (restart=30, max_iters=500, rtol=1e-6):

| Matrix | n | None | Jacobi | ILU | AMG |
|--------|---:|-----:|-------:|----:|----:|
| sherman1 | 1000 | 80%/0.889 | 100%/0.525 | 100%/0.004 | 100%/0.019 |
| sherman3 | 5005 | 0%/1.020 | 40%/0.944 | 100%/0.022 | 100%/0.013 |
| sherman4 | 1104 | 100%/0.330 | 100%/0.207 | 100%/0.004 | 100%/0.012 |
| rdb1250 | 1250 | 100%/0.464 | 0%/1.020 | 100%/0.064 | 100%/0.028 |
| pde2961 | 2961 | 100%/0.448 | 100%/0.387 | 100%/0.024 | 100%/0.015 |
| epb0 | 1794 | 0%/1.020 | 0%/1.020 | 100%/0.004 | 100%/0.350 |
| thermal | 3456 | 100%/0.042 | 100%/0.040 | 100%/0.004 | 100%/0.008 |
| orsirr_1 | 1030 | 0%/1.020 | 100%/0.360 | 100%/0.006 | 100%/0.008 |
| orsreg_1 | 2205 | 100%/0.387 | 100%/0.373 | 100%/0.006 | 100%/0.008 |
| watt_1 | 1856 | 100%/0.004 | 100%/0.002 | 100%/0.002 | 100%/0.002 |
| saylr4 | 3564 | 0%/1.020 | 0%/1.020 | 100%/0.008 | 100%/0.081 |

ILU solves everything. That's the target. The agent should focus on closing the gap between MatrixPFN and ILU, especially on the 8 matrices that currently fail.
