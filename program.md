# autoresearch-matrixpfn

Autonomous research for MatrixPFN: a GNN-based learned preconditioner for sparse linear systems.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar13`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the files**: Read these for full context:
   - `CLAUDE.md` — project context, architecture, what matters.
   - `prepare.py` — fixed evaluation, SuiteSparse download, scoring. Do not modify.
   - `train.py` — the file you modify. Model config, domains, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch-matrixpfn/suitesparse/` contains matrices. If not, run `uv run prepare.py`.
5. **Initialize results.tsv**: Create with just the header row.
6. **Confirm and go**.

## Experimentation

Each experiment trains MatrixPFN for a **fixed time budget of 5 minutes**, then evaluates on synthetic + SuiteSparse matrices. Launch: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — everything is fair game: model architecture (layers, embed, hidden), training domains and their weights, grid sizes, hyperparameters (LR, batch size, optimizer), training loop structure.

**What you CANNOT do:**
- Modify `prepare.py`. It contains the fixed evaluation and scoring.
- Install new packages.
- Modify the evaluation harness.

**The goal: get the lowest `score`.** The score is `0.3 * synthetic_score + 0.7 * suitesparse_score`, where each sub-score is the mean normalized iteration count (iterations / max_iterations). Lower = better. A score of 0.0 means instant convergence on everything. A score of 1.0 means nothing converges.

**The real challenge is SuiteSparse convergence.** The MVP baseline (Diffusion-only, 11K params) achieved 100% synthetic convergence but only 14% SuiteSparse convergence. Your job is to improve SuiteSparse convergence by:
1. Training on diverse domains (not just Diffusion)
2. Scaling the model (more params = more capacity)
3. Finding the right domain weights
4. Tuning hyperparameters

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

1. Look at git state
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

## Research directions

Prioritized list of things to try (start from top):

### Phase 1: Establish multi-domain baseline
- Run baseline as-is (8 domains, 128/256 model)
- If crashes on any domain, disable that domain and retry

### Phase 2: Domain tuning
- Try different domain weight distributions
- Remove domains that don't help (if removing a domain improves score, keep the removal)
- Add domains that might help (SBM, RANDOM_SPARSE, ENHANCED_DIFFUSION)
- DIFFUSION_ADVECTION and VARIABLE_ADVECTION are key for non-SPD SuiteSparse matrices

### Phase 3: Model scaling
- Try embed=64/hidden=128 (smaller, faster training, more epochs)
- Try embed=256/hidden=512 (larger, fewer epochs but more capacity)
- Try more/fewer layers
- Try more context pairs (8, 16)

### Phase 4: Training optimization
- Learning rate: try 3e-4, 1e-3, 3e-3
- Try cosine LR schedule (ramp down over training budget)
- Try larger matrices_per_epoch with gradient accumulation
- Try different grid sizes (add 48 to training for OOD robustness)

### Phase 5: Architecture changes
- Try ContextResMPNN instead of ContextResGCN (import from matrixpfn.nn.context_resgcn)
- Try BaselineResGCN (ablation without context)

## Key insight from MVP results

The MVP (Diffusion-only, 11K params) showed:
- 100% synthetic convergence (good within domain)
- thermal was the only SuiteSparse success (because thermal ≈ diffusion)
- Loss plateau at 9% (model too small)

This means: **domain diversity is the #1 priority**, model size is #2, hyperparameters are #3.
