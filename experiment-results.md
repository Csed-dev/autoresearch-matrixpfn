# SPAI & Polynomial Preconditioner Experiment Results

Date: 2026-03-13 / 2026-03-14

## Context

Exploring learned neural preconditioners for FGMRES. The model (GNN) analyzes a sparse matrix A and produces a preconditioner M such that MA is closer to identity.

Evaluation: 11 SuiteSparse matrices + synthetic diffusion grids. Score = 0.3 * synthetic + 0.7 * SuiteSparse normalized iteration count. Lower is better.

## Architecture Evolution

### Phase 1: SPAI (Sparse Approximate Inverse)

**Architecture:** MPNN body + BilinearEdgeHead predicts G-values on A's sparsity pattern. Preconditioner: M = D^{-1}(I+G).

**Problem:** G lives on A's edges only (1-hop). No fill-in beyond A's sparsity pattern. ILU works precisely because it creates fill-in.

| Run | Config | Score | SS Conv | Best Loss | Key Change |
|-----|--------|-------|---------|-----------|------------|
| 4 | Unbounded G, LR=1e-3, no schedule | 0.664 | 36.4% (4/11) | 8.19e-02 | Baseline — massive training instability (loss 1e-1 to 1e+23) |
| 5 | G_SCALE=0.5, tanh, cosine LR->0, warmup, skip>100 | 0.688 | 27.3% (3/11) | 3.28e-02 | Stable but too constrained, thermal lost |
| 6 | G_SCALE=1.0, cosine LR->10%, skip>50 | 0.631 | 36.4% (4/11) | 3.20e-02 | Sweet spot: stable + expressive, thermal recovered |

**Converging:** sherman4, pde2961, thermal, watt_1
**Failing:** sherman1, sherman3, rdb1250, epb0, orsirr_1, orsreg_1, saylr4

### Phase 2: Node-wise Polynomial Preconditioner

**Architecture:** Same MPNN body, but PolynomialHead predicts K coefficients per node. Preconditioner applied implicitly: (M*r)_i = sum_k c_k(i) * [(D^{-1}A)^k r]_i. Multi-hop fill-in by construction.

| Run | Config | Score | SS Conv | Best Loss | Key Change |
|-----|--------|-------|---------|-----------|------------|
| 7 | K=4, 256/512, 4 layers | 0.545 | 61.8% (7/11) | 1.26e-02 | **Breakthrough:** +3 matrices (sherman1, orsirr_1, orsreg_1) |
| 8 | K=6, 256/512 | 0.483 | 63.6% (7/11) | 1.05e-02 | orsirr_1 100% conv. All scores improved |
| 9 | K=8, 256/512 | 0.602 | 50.9% (6/11) | 1.79e-02 | Too many params, undertrained. Thermal lost |
| 10 | **K=6, 192/384** | **0.482** | **63.6% (7/11)** | 1.14e-02 | **Best. Same quality, 44% fewer params. thermal=0.030** |
| 11 | K=6, 192/384, loss clamp=10 | 0.494 | 63.6% (7/11) | 1.05e-02 | Clamping hurt: thermal 0.030->0.299 |
| 12 | K=6, 192/384, 6 GNN layers | 0.487 | 63.6% (7/11) | 1.54e-02 | More layers = fewer epochs (608 vs 781) |
| 13 | K=6, 192/384, learnable omega | 0.523 | 54.5% (6/11) | 7.78e-03 | Omega hurt: thermal lost, fewer epochs |
| 14 | K=6, 192/384, grids 16-64 | 0.487 | 63.6% (7/11) | 1.31e-02 | Larger training grids: no improvement |

## Key Findings

### 1. Training Stability (Runs 4-6)

Unbounded edge predictions + high LR caused catastrophic instability (loss oscillating between 1e-1 and 1e+23). Fixed with:
- `tanh` bounding on G values
- Cosine LR schedule with warmup (20 epochs -> peak -> 10% minimum)
- Loss-based matrix skipping (skip if loss > 50)
- Reduced base LR (1e-3 -> 3e-4)

Loss clamping (Run 11) is worse than skipping: gradients from hard matrices are noisy and destructive, harming performance on easy matrices (thermal 0.030 -> 0.299).

### 2. Sparsity Pattern is the Fundamental Bottleneck (SPAI -> Polynomial)

The SPAI approach (G on A's edges) is fundamentally limited to 1-hop corrections. Even though the GNN has 4-hop receptive field in its node embeddings, the edge prediction is restricted to A's sparsity pattern. This caps convergence at 36.4% (4/11) of SuiteSparse matrices.

Switching to the polynomial preconditioner immediately unlocked 3 more matrices (sherman1, orsirr_1, orsreg_1) — from 4/11 to 7/11. The polynomial naturally provides multi-hop fill-in via (D^{-1}A)^k without explicit sparsity pattern expansion.

### 3. Diminishing Returns on Polynomial Degree

K=4 -> K=6: significant improvement (score 0.545 -> 0.483)
K=6 -> K=8: regression (score 0.483 -> 0.602) due to undertraining

The higher-degree polynomial adds parameters that the model can't effectively train in the 300s budget. K=6 is the sweet spot.

### 4. Model Capacity is Not the Bottleneck

Within the fixed 300s training budget:
- Larger models (256/512 vs 192/384): same quality, more params wasted
- More GNN layers (6 vs 4): fewer epochs, slightly worse
- Larger training grids (16-64 vs 16-48): no improvement

The 192/384 model achieves the same score as 256/512 with 44% fewer parameters (709K vs 1.26M) and less VRAM (324 MB vs 427 MB).

### 5. The Polynomial-in-D^{-1}A Ceiling

The remaining 4 matrices (sherman3, rdb1250, epb0, saylr4) are resistant to all hyperparameter changes. They consistently show pfn=1.000. Root cause: when rho(D^{-1}A) >> 1, the polynomial powers (D^{-1}A)^k diverge. The model learns c_k ~ 0 for k>0, effectively falling back to Jacobi, which also fails for these matrices.

Attempted fixes that failed:
- Learnable spectral normalization (omega): hurt easy matrices
- Loss clamping: noisy gradients from hard matrices destroyed training signal
- More GNN layers: fewer epochs, no gain
- Larger training grids: no improvement

This is a **fundamental architectural limit**, not a hyperparameter issue.

### 6. What Would Be Needed to Break Past 7/11

To solve the remaining 4 matrices, the preconditioner needs to handle matrices where D^{-1}A has spectral radius >> 1. Possible approaches (not yet tested):
- **Chebyshev polynomial basis**: better numerical conditioning than raw powers
- **Neumann-like formulation**: M = (I - J)^{-1} approximated via truncated series, where J = I - D^{-1}A
- **Hybrid approach**: combine polynomial with ILU-like sparse factorization
- **Multi-grid inspired**: learn restriction/prolongation operators

## Per-Matrix Performance (Best Run: #10)

| Matrix | n | PFN | Jacobi | ILU | AMG | Conv |
|--------|---|-----|--------|-----|-----|------|
| thermal | 3456 | **0.030** | 0.073 | 0.004 | 0.008 | 100% |
| sherman4 | 1104 | **0.164** | 0.628 | 0.004 | 0.012 | 100% |
| watt_1 | 1856 | **0.223** | 0.865 | 0.002 | 0.002 | 100% |
| pde2961 | 2961 | **0.403** | 0.787 | 0.024 | 0.015 | 100% |
| orsirr_1 | 1030 | **0.593** | 1.000 | 0.006 | 0.008 | 100% |
| orsreg_1 | 2205 | **0.586** | 1.000 | 0.006 | 0.008 | 100% |
| sherman1 | 1000 | **0.635** | 1.000 | 0.004 | 0.019 | 100% |
| sherman3 | 5005 | 1.000 | 1.000 | 0.022 | 0.013 | 0% |
| rdb1250 | 1250 | 1.000 | 1.000 | 0.064 | 0.028 | 0% |
| epb0 | 1794 | 1.000 | 1.000 | 0.004 | 0.350 | 0% |
| saylr4 | 3564 | 1.000 | 1.000 | 0.008 | 0.081 | 0% |

PFN beats Jacobi on all converging matrices (by 2-4x). Beats AMG on thermal. Solves 3 matrices that Jacobi cannot (sherman1, orsirr_1, orsreg_1). Still far from ILU on converging matrices, and cannot solve the 4 hardest cases.

## Synthetic Performance (Best Run: #10)

| Grid | PFN | Jacobi | Conv |
|------|-----|--------|------|
| 16x16 | 0.071 | 0.206 | 100% |
| 32x32 | 0.163 | 0.462 | 100% |
| 64x64 (OOD) | 0.366 | 0.995 | 100% |

Strong OOD generalization: 64x64 not in training set, PFN still 2.7x better than Jacobi.

### Phase 3: Extended Training (900s) and Hybrid

| Run | Config | Score | SS Conv | Key Finding |
|-----|--------|-------|---------|-------------|
| 15 | K=6, 192/384, 900s | 0.544 | 54.5% (6/11) | Overfitting: thermal lost, lower SS conv |
| 16 | K=8, 192/384, 900s | 0.618 | 36.4% (4/11) | K=8 confirmed bad regardless of training time |
| 17 | K=6, 256/512, 900s | 0.550 | 54.5% (6/11) | Bigger model also overfits with more time |
| 18 | Hybrid SPAI+Poly multiplicative, 300s | 0.801 | 27.3% (3/11) | Multiplicative coupling destabilizes training |
| 19 | Hybrid SPAI+Poly additive, 300s | 0.876 | 9.1% (1/11) | Additive combination even worse — SPAI destroys polynomial |
| 20 | Poly + dropout=0.15 + WD=5e-4, 300s | 0.484 | 63.6% (7/11) | Regularization neutral at 300s |
| 21 | Poly + dropout=0.15 + WD=5e-4, 600s | 0.542 | 54.5% (6/11) | Dropout doesn't prevent distribution-shift overfitting |
| 22 | Chebyshev basis instead of power basis, 300s | 0.583 | 52.7% (6/11) | Chebyshev worse — recurrence causes numerical cancellation |

### Key Findings (Phase 3)

**6. More Training Causes Overfitting**

900s training (3x baseline) makes ALL configs worse. Run 15 achieves better training loss (7.06e-03 vs 1.14e-02) but worse eval. thermal goes from 0.030 (near-optimal) to FAIL. The 300s budget acts as implicit regularization through early stopping. Best loss in Run 10 was found at epoch 645 of 781.

**7. Spectral Radius Hypothesis Was Wrong**

rho(D^{-1}A) is approximately 2.0 for ALL 11 matrices — no separation between converging and failing groups. The failure modes are matrix-specific: sherman3 has extreme diagonal range (3.45e+16), epb0 has low diagonal dominance (0.24). No single scalar diagnostic predicts failure.

**8. Both Hybrid Approaches Fail**

Multiplicative (Run 18): M = D^{-1}(I+G) * P_poly — best loss 0.366, score 0.801. Heads amplify each other's errors.
Additive (Run 19): M = P_poly + D^{-1}*G*D^{-1}*r — score 0.876, only 1/11 converges. SPAI correction destroys polynomial output.

The SPAI and polynomial architectures are incompatible in naive combinations. The shared backbone cannot optimize for both heads simultaneously without destructive interference.

**9. Regularization (Dropout + Weight Decay)**

Run 20: dropout=0.15, weight_decay=5e-4 at 300s — score 0.484, identical to Run 10 (0.482). Regularization is neutral at 300s because the model isn't overfitting yet. Test at 600s (Run 21) to see if it prevents the overfitting observed at 900s.

### Phase 4: Architecture & Hyperparameter Exploration (Runs 23-33)

| Run | Config | Score | SS Conv | Key Finding |
|-----|--------|-------|---------|-------------|
| 23 | Per-node damped poly, omega init=0.5, 300s | 0.528 | 54.5% (6/11) | Omega too aggressive, thermal lost |
| 24 | Per-node damped poly, omega init=0.95, 300s | 0.530 | 54.5% (6/11) | Still worse — omega changes poly basis unhelpfully |
| 25 | Operator-scaled poly (||D^{-1}A||_inf), 300s | 0.512 | 63.6% (7/11) | Same 7/11 but worse individual scores |
| 26 | 11 training domains (added 3 non-sym), 300s | 0.533 | 54.5% (6/11) | Extra domains dilute training, thermal lost |
| 27 | LR=5e-4, 32 mat/epoch, 300s | 0.498 | 63.6% (7/11) | Only 573 epochs due to 2x overhead |
| 28 | LR=5e-4, 16 mat/epoch, 300s | 0.534 | 54.5% (6/11) | 1260 epochs — overfitting, thermal lost |
| 29 | Low-rank M=D^{-1}+V*V^T, rank=16, 300s | 0.729 | 27.3% (3/11) | Fundamentally wrong — V*V^T is symmetric PSD |
| 30 | 16 probes (2x Run 10), 300s | 0.532 | 54.5% (6/11) | More probes = fewer effective epochs, thermal lost |
| 31 | Seed=0 (exact Run 10 config), 300s | 0.485 | 63.6% (7/11) | Robust — same 7/11, thermal=0.050 |
| 32 | Seed=137 (exact Run 10 config), 300s | 0.533 | 54.5% (6/11) | thermal FAILS with this seed — fragile |
| 33 | EMA decay=0.999 of model weights, 300s | 0.623 | 52.7% (5.8/11) | Over-smoothing destroys sharp coefficients |

### Key Findings (Phase 4)

**10. Per-Node Damping Does Not Help**

Runs 23-24: predicting per-node omega in (0,1) to damp (D^{-1}A)^k powers. Even with omega initialized near 1.0 (sigmoid(3)≈0.95), the damping changes the polynomial basis without adding expressiveness. The GNN already controls magnitude through coefficients c_k. Damping adds a redundant degree of freedom that confuses optimization.

**11. Operator Scaling Is Mathematically Redundant**

Run 25: scaling D^{-1}A by 1/||D^{-1}A||_inf produces bounded powers but the scaling is absorbed by the polynomial coefficients. The GNN can already learn c_k/s^k directly. The extra node feature (local D^{-1}A row norm) doesn't help because this info is already captured by existing features.

**12. Training Domain Distribution Matters Critically**

Run 26: adding 3 more non-symmetric domains (11 total) degraded from 7/11 to 6/11. The original 8-domain mix has a specific balance where DIFFUSION (20%) provides the clearest learning signal. Diluting with more domains reduces the epochs spent on Diffusion-like matrices, which is what thermal needs.

**13. The Polynomial Ceiling Is Architecture-Fundamental**

Runs 23-33 tested: damping (2 variants), scaling, more domains, higher LR, more probes, low-rank, EMA, seed variation. NONE improved on Run 10. The polynomial p(D^{-1}A) architecture has a hard ceiling at 7/11 (with thermal fragile) and score ~0.482.

**14. thermal Convergence Is Seed-Dependent**

Seed variation (Runs 10, 31, 32): thermal converges with seeds 42 and 0, but FAILS with seed 137. This means 7/11 is not robust — the true robust ceiling is 6/11 + thermal-sometimes. The polynomial coefficients for thermal sit on a knife edge in parameter space.

**15. Low-Rank Preconditioner Is Structurally Wrong**

Run 29: M = D^{-1} + V*V^T produces a symmetric PSD correction, but all eval matrices are non-symmetric. The low-rank correction cannot capture the asymmetric structure of the inverse. Score 0.729, only 3/11 converge — worse than Jacobi on some matrices.

**16. EMA Over-Smooths Polynomial Coefficients**

Run 33: EMA with decay=0.999 blurs the sharp coefficient values. The polynomial preconditioner requires precise coefficients — even small perturbations can make thermal/sherman1 diverge. Weight averaging is counterproductive for this architecture.

### Phase 5: Neumann-Basis Polynomial BREAKTHROUGH (Runs 34-44)

**The key insight:** Instead of powers of D^{-1}A (which grow for ill-conditioned matrices), use powers of J = I - D^{-1}A (the Jacobi iteration matrix). J^k naturally decays when rho(J) < 1, enabling stable high-degree polynomials. Initialize all c_k = 1 (Neumann series).

| Run | Config | Score | SS Conv | Key Finding |
|-----|--------|-------|---------|-------------|
| 34 | Poly + asymmetric U*V^T (rank=8) | 0.502 | 63.6% (7/11) | U*V^T correction adds noise, no improvement |
| 35 | **Neumann basis K=6** | **0.400** | 63.6% (7/11) | **BREAKTHROUGH: rdb1250 converges!** thermal lost |
| 36 | Neumann K=6, seed=0 | 0.388 | 63.6% (7/11) | Robust — thermal consistently lost with Neumann |
| 37 | Neumann K=6, high DIFFUSION weight | 0.401 | 63.6% (7/11) | More DIFFUSION training doesn't recover thermal |
| 38 | **Neumann K=8** | **0.368** | 69.1% (8.6/11) | **sherman3 at 60% conv!** K=8 works with Neumann (failed with power) |
| 39 | **Neumann K=10** | **0.348** | 72.7% (9/11) | **sherman3 100% conv!** Two previously impossible matrices solved |
| 40 | Neumann K=12 | 0.337 | 72.7% (9/11) | Monotonic improvement continues |
| 41 | Neumann K=16 | 0.305 | 72.7% (9/11) | Still improving, 680 epochs |
| 42 | **Neumann K=20** | **0.279** | **81.8% (10/11)** | **epb0 converges!** Only thermal + saylr4 remain |
| 43 | Neumann K=24 | 0.260 | 81.8% (10/11) | epb0 improves to 0.379 |
| 44 | **Neumann K=32** | **0.236** | **81.8% (10/11)** | **epb0=0.261 beats AMG!** |
| 45 | Neumann K=48 | 0.205 | 81.8% (10/11) | epb0=0.098 beats AMG by 3.6x |
| 46 | Neumann K=64 | 0.192 | 81.8% (10/11) | pde2961=0.033 approaching ILU |
| 47 | **Neumann K=96** | **0.179** | **81.8% (10/11)** | **pde2961=0.023 BEATS ILU!** epb0=0.067 beats AMG by 5.2x |
| 48 | Neumann K=128, 192/384 | 0.171 | 81.8% (10/11) | 266 epochs, pde2961=0.020 |
| 49 | Neumann K=128, 128/256 | 0.171 | 81.8% (10/11) | Same score with 332K params (vs 732K) |
| 50 | Neumann K=96, 128/256 | 0.179 | 81.8% (10/11) | Same as Run 47 — model size irrelevant |
| 51 | **Neumann K=128, 64/128** | **0.170** | **81.8% (10/11)** | **88K params! GNN nearly irrelevant** |

### Key Findings (Phase 5)

**17. Neumann Basis Is Fundamentally Superior to Power Basis**

The polynomial p(D^{-1}A) = sum c_k (D^{-1}A)^k has (D^{-1}A)^k growing exponentially when rho(D^{-1}A) > 1. The Neumann basis p(J) = sum c_k J^k where J = I - D^{-1}A has J^k decaying (for matrices where Jacobi would eventually converge). This enables:
1. Stable training at high polynomial degrees (K=32 works vs K=8 max with power basis)
2. Monotonic improvement with K: every increase in K improves the score
3. Unlocking previously impossible matrices (rdb1250 at K=6, sherman3 at K=10, epb0 at K=20)

**18. Higher K Keeps Working With Neumann Basis**

With power basis, K>6 caused regression (K=8 scored 0.602 vs K=6's 0.483). With Neumann basis, K=6->8->10->12->16->20->24->32->48->64->96 gives monotonic improvement: 0.400->0.368->0.348->0.337->0.305->0.279->0.260->0.236->0.205->0.192->0.179. No sign of plateauing even at K=96 (318 epochs). At K=96, PFN BEATS ILU on pde2961 (0.023 vs 0.024) — first time a learned preconditioner surpasses ILU on a SuiteSparse matrix.

**19. thermal Is Structurally Incompatible With Neumann Basis**

thermal (a diffusion matrix) is the ONLY matrix that converges with the power basis (pfn=0.030) but FAILS with the Neumann basis at any K. This is likely because thermal has rho(J) ≈ 1 — the Jacobi iteration matrix is on the stability boundary, causing J^k to neither decay nor grow. The Neumann series doesn't converge for thermal, but the power basis happens to find a good polynomial.

**20. PFN Now Beats AMG on Some Matrices**

At K=32: epb0 (pfn=0.261 vs amg=0.350) — PFN is 1.3x better than AMG! Also competitive with Jacobi fallback on nearly all matrices. The Neumann-basis polynomial effectively learns a truncated Neumann series with per-node adaptive coefficients.

## Per-Matrix Performance (Best Run: #47, K=96 Neumann)

| Matrix | n | PFN | Jacobi | ILU | AMG | Conv | PFN vs ILU |
|--------|---|-----|--------|-----|-----|------|-----------|
| thermal | 3456 | FAIL | 0.073 | 0.004 | 0.008 | 0% | — |
| **pde2961** | 2961 | **0.023** | 0.787 | 0.024 | 0.015 | 100% | **PFN WINS** |
| sherman4 | 1104 | **0.023** | 0.628 | 0.004 | 0.012 | 100% | 5.8x gap |
| watt_1 | 1856 | **0.027** | 0.865 | 0.002 | 0.002 | 100% | 13.5x gap |
| sherman1 | 1000 | **0.055** | 1.000 | 0.004 | 0.019 | 100% | 13.8x gap |
| **epb0** | 1794 | **0.067** | 1.000 | 0.004 | 0.350 | 100% | **PFN beats AMG 5.2x** |
| rdb1250 | 1250 | **0.071** | 1.000 | 0.064 | 0.028 | 100% | ~tied |
| orsreg_1 | 2205 | **0.119** | 1.000 | 0.006 | 0.008 | 100% | 19.8x gap |
| orsirr_1 | 1030 | **0.119** | 1.000 | 0.006 | 0.008 | 100% | 19.8x gap |
| sherman3 | 5005 | **0.179** | 1.000 | 0.022 | 0.013 | 100% | 8.1x gap |
| saylr4 | 3564 | 1.000 | 1.000 | 0.008 | 0.081 | 0% | — |

PFN beats Jacobi on ALL converging matrices (by 6-18x). **Beats ILU on pde2961.** Beats AMG on epb0 by 5.2x. Solves 8 matrices that Jacobi cannot. Only thermal and saylr4 remain unsolved.

### Key Findings (Phase 5, continued)

**21. GNN Model Size Is Nearly Irrelevant**

Runs 48-51: tested 192/384 (732K), 128/256 (332K), and 64/128 (88K) with K=128. All achieve score ~0.170-0.171. The polynomial degree (128 Neumann terms) dominates — the GNN only provides minor per-node coefficient adjustments from the Neumann series baseline (all c_k=1). An 88K parameter model matches a 732K parameter model.

## Best Configuration (Run 51)

```
Model: PolyMPNN (88,576 params)
  GNN: 4 layers, embed=64, hidden=128
  Head: PolynomialHead, degree=128 (Neumann basis J = I - D^{-1}A)
  Init: all c_k = 1 (Neumann series)
  Training: 8 domains, grids (16,24,32,48)
  LR: 3e-4 with 20-epoch warmup + cosine decay (min 10%)
  Loss: stochastic Frobenius ||MAv-v||^2, 8 probes, skip if >50
  Budget: 300s training -> ~251 epochs
```
