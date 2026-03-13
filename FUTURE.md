# MatrixPFN v2 — Experiment Plan

## Goal

Break through the v1 SuiteSparse convergence ceiling (27.3%, 3/11 matrices) by transitioning from vector-output GCN to Sparse Approximate Inverse (SPAI) architecture. Target: beat ILU on specific PDE matrix classes.

Full architecture rationale: [ADR-11](../Thesis/docs/adr/11-v2-sparse-approximate-inverse-architecture.md)

## v1 Summary

23 experiments exhausted the ContextResGCN hyperparameter space. Best score: 0.5832. SuiteSparse convergence stuck at 27.3% regardless of model size, depth, domains, context pairs, or learning rate. Root cause: GCN local receptive field + vector output = learned weighted Jacobi.

## Experiment Sequence

Each run isolates one change. Compare against v1 baseline (score 0.5832, SS 27.3%).

### Run 1: Scaled v1 Baseline (Control)

**Status:** Ready

**Changes:** embed 128→256, hidden 256→512, context 5→8, batch 16→64, grids +(48)

**Hypothesis:** Score improves slightly from more data throughput, but SS convergence stays at ~27%. This confirms the ceiling is architectural, not resource-limited.

**Expected VRAM:** ~3-5 GB (vs 474 MB baseline)

### Run 2: Extended Features (v1 architecture + richer input)

**Status:** Planned

**Changes:** Add node features (diagonal dominance, row norm) and edge features (a_ij, |a_ij|/|a_ii|) to ContextResGCN input. Requires modifying the input MLP dimensions.

**Hypothesis:** Marginal improvement — the model gets more structural information but still has the local receptive field limitation. May help on matrices close to the convergence boundary.

### Run 3: Stochastic Frobenius Loss

**Status:** Planned

**Changes:** Replace L1 loss `||x_pred - x||_1` with stochastic Frobenius `E_v[||MAv - v||²]`. Requires changing the training loop: instead of predicting x from b, the model predicts M entries and the loss measures `||MAv - v||²` for random v.

**Hypothesis:** Better alignment between training objective and evaluation metric. The L1 loss optimizes for inverse approximation accuracy; the Frobenius loss optimizes for preconditioning quality directly.

**Note:** This requires the SPAI output head — cannot be tested with the current vector-output architecture. Combine with Run 4.

### Run 4: SPAI Head (Core Architecture Change)

**Status:** Planned — requires new model class

**Changes:**
- New model class `SpaiMPNN` in train.py
- MPNN body (MPNNConv layers with learned edge messages)
- Bilinear edge prediction head: `g_ij = h_i^T W h_j + w^T f_e`
- Output: sparse G with nnz(G) = nnz(A)
- Preconditioner: M = D⁻¹(I + G)
- Application: z = Mr (one SpMV, no GNN forward pass per FGMRES step)
- Loss: stochastic Frobenius `E_v[||MAv - v||²]`
- Node features: [a_ii, |a_ii|/Σ|a_ij|, ||a_i||]
- Edge features: [a_ij, |a_ij|/|a_ii|]

**Hypothesis:** First architecture that can potentially break the 27.3% barrier. The SPAI output captures off-diagonal corrections that Jacobi/weighted-Jacobi cannot.

**Risk:** MPNN OOD generalization failure (see ADR-04, ADR-08). Fallback: GCN body + bilinear edge head.

### Run 5: SPAI + FiLM Conditioning

**Status:** Planned

**Changes:** Add Chebyshev moment computation (K=20 mat-vecs) and FiLM conditioning (γ_l ⊙ h + β_l) to each MPNN layer.

**Hypothesis:** Chebyshev moments provide global spectral information that local MPNN cannot capture. FiLM conditioning injects this into every layer. Should help on spectrally diverse matrices (epb0, sherman3).

### Run 6: GCN Body Fallback (if Run 4 MPNN fails OOD)

**Status:** Contingency

**Changes:** Replace MPNN body with GCN body, keep bilinear edge head. Node embeddings from GCN, edge predictions from bilinear form on node pairs.

**Hypothesis:** Loses edge-level information in the body but avoids MPNN OOD catastrophe. May be competitive if edge features + bilinear head carry enough information.

## Future Phases (after v2.0 baseline established)

### v2.1: Multi-Scale (if v2.0 plateaus)

Add learned graph pooling (TopK/SAGPool) for encoder-decoder with skip connections. Test pooling strategies via autoresearch ablation.

No AMG-based coarsening — learned pooling only, to avoid classical dependency at inference.

### v2.2: Product Form (if single-sweep M hits sparsity ceiling)

M = D⁻¹(I + G₁)(I + G₂) with two bilinear heads. Implicit fill-in through product, application via two SpMVs. Requires symmetry-breaking mechanism between heads.

### v2.3: Defect Correction (if v2.2 quality insufficient)

Iterative refinement: compute per-node defect r_i = v_i - (M₀Av)_i, feed back into GNN to predict δG. M₁ = D⁻¹(I + G + δG).

## Known Limitations

1. **Sparsity ceiling:** nnz(G) = nnz(A) limits corrections to A's pattern. ILU has fill-in. Product form (v2.2) partially addresses this.
2. **Learned pooling risk:** No guarantee of preserving algebraically important structure. Must validate empirically.
3. **MPNN OOD:** Documented failure on graph_laplacian (ADR-04/08). Mitigated by Jacobi fallback + edge normalization.
4. **Single-matrix amortization:** GNN overhead only pays off with multiple solves or similar matrices.
5. **Training-eval gap:** Frobenius loss ≠ FGMRES convergence. Low Frobenius error doesn't guarantee fast convergence.

## Success Criteria

| Metric | v1 Best | Target (v2.0) | Stretch (v2.2) |
|--------|---------|---------------|----------------|
| Score | 0.5832 | < 0.45 | < 0.30 |
| SS Convergence | 27.3% (3/11) | > 50% (6/11) | > 80% (9/11) |
| SS matrices beating Jacobi | 3 | 6 | 9 |
| Peak VRAM | 0.8 GB | < 8 GB | < 16 GB |

## Classical Reference (target to beat)

| Matrix | Jacobi | MatrixPFN v1 | ILU | AMG |
|--------|--------|-------------|-----|-----|
| sherman1 | 100%/0.525 | FAIL | 100%/0.004 | 100%/0.019 |
| sherman3 | 40%/0.944 | FAIL | 100%/0.022 | 100%/0.013 |
| sherman4 | 100%/0.207 | OK/0.131 | 100%/0.004 | 100%/0.012 |
| rdb1250 | 0%/1.020 | FAIL | 100%/0.064 | 100%/0.028 |
| pde2961 | 100%/0.387 | OK/0.207 | 100%/0.024 | 100%/0.015 |
| epb0 | 0%/1.020 | FAIL | 100%/0.004 | 100%/0.350 |
| thermal | 100%/0.040 | OK/0.027 | 100%/0.004 | 100%/0.008 |
| orsirr_1 | 100%/0.360 | FAIL | 100%/0.006 | 100%/0.008 |
| orsreg_1 | 100%/0.373 | FAIL | 100%/0.006 | 100%/0.008 |
| watt_1 | 100%/0.002 | FAIL | 100%/0.002 | 100%/0.002 |
| saylr4 | 0%/1.020 | FAIL | 100%/0.008 | 100%/0.081 |
