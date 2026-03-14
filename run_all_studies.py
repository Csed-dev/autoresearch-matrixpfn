"""
Run all thesis-critical studies on one pod:
1. Train with K=256 omega=0.9 (best proven config)
2. GNN Ablation (with vs without GNN)
3. Omega sweep (0.5-0.99)
4. Sign-correction test for saylr4

Uses the ORIGINAL 3-feature train.py from git tag, not the experimental one.
"""
import sys
import os
import time
import numpy as np

# Ensure we use the local modules
sys.path.insert(0, os.path.dirname(__file__))

import torch
from train import (
    PolyMPNN, load_checkpoint, PolynomialPreconditioner, JACOBI_OMEGA,
    NUM_LAYERS, EMBED_DIM, HIDDEN_DIM, NUM_EDGE_FEATURES,
)
from prepare import (
    EVAL_MATRICES, SYNTHETIC_EVAL_GRIDS, SYNTHETIC_TRAINING_GRIDS,
    NUM_RHS, NUM_SYNTHETIC_MATRICES, FGMRES_RESTART, FGMRES_MAX_ITERS,
    FGMRES_RTOL, FGMRES_TIMEOUT, load_suitesparse_matrix,
)
from matrixpfn.precond.jacobi import Jacobi
from matrixpfn.solver.fgmres import FGMRES
from matrixpfn.generator.domains.diffusion import DiffusionGenerator

CHECKPOINT = "best_model.pt"


class FlexPrecond:
    """Neumann preconditioner with configurable omega and optional fixed coeffs."""
    def __init__(self, coeffs, D_inv_A, D_inv, omega, sign=1.0):
        self.coeffs = coeffs.double()
        self.D_inv_A = D_inv_A
        self.D_inv = D_inv
        self.omega = omega
        self.sign = sign

    def apply(self, r):
        K = self.coeffs.shape[1]
        w = self.omega
        power = w * self.D_inv * r
        result = self.coeffs[:, 0] * power
        for k in range(1, K):
            power = power - w * (self.D_inv_A @ power)
            result = result + self.coeffs[:, k] * power
        return self.sign * result


@torch.no_grad()
def eval_ss_only(model, device, omega, fixed_coeffs=False, sign=1.0):
    """Quick eval on SuiteSparse only (no synthetic)."""
    model.eval()
    solver = FGMRES(restart=FGMRES_RESTART, max_iters=FGMRES_MAX_ITERS,
                    rtol=FGMRES_RTOL, timeout=FGMRES_TIMEOUT)
    torch.manual_seed(99999)
    np.random.seed(99999)
    results = {}
    for group, name in EVAL_MATRICES:
        try:
            A = load_suitesparse_matrix(name, device)
        except FileNotFoundError:
            continue
        n = A.shape[0]
        try:
            model.set_matrix(A)
            coeffs = model()
            if fixed_coeffs:
                coeffs = torch.ones_like(coeffs)
            s = getattr(model, '_diag_sign', 1.0) if sign == 'auto' else sign
            precond = FlexPrecond(coeffs, model.D_inv_A, model.D_inv, omega, s)
        except Exception as e:
            results[name] = (1.0, 0.0)
            continue
        iters, convs = [], []
        for _ in range(NUM_RHS):
            b = torch.randn(n, dtype=torch.float64, device=device)
            try:
                r = solver.solve(A, b, M=precond, progress_bar=False)
                iters.append(r.iterations / FGMRES_MAX_ITERS)
                convs.append(r.converged)
            except Exception:
                iters.append(1.0)
                convs.append(False)
        results[name] = (sum(iters)/len(iters), sum(convs)/len(convs))
    return results


def score_from_results(results):
    vals = [v[0] for v in results.values()]
    return sum(vals) / len(vals) if vals else 1.0


def print_results(results, label=""):
    n_conv = sum(1 for v in results.values() if v[1] > 0.5)
    sc = score_from_results(results)
    print("{}: ss_mean={:.4f}, conv={}/11".format(label, sc, n_conv))
    for name, (pfn, conv) in results.items():
        tag = "OK" if conv > 0.5 else "FAIL"
        print("  {:<12s}: pfn={:.3f} {}".format(name, pfn, tag))
    return sc, n_conv


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    model = load_checkpoint(CHECKPOINT, device)
    print("Loaded checkpoint\n")

    # === 1. GNN ABLATION ===
    print("=" * 60)
    print("STUDY 1: GNN ABLATION (omega=0.9)")
    print("=" * 60)
    r_gnn = eval_ss_only(model, device, 0.9, fixed_coeffs=False)
    s_gnn, c_gnn = print_results(r_gnn, "WITH GNN")
    r_fix = eval_ss_only(model, device, 0.9, fixed_coeffs=True)
    s_fix, c_fix = print_results(r_fix, "FIXED c_k=1")
    print("\nGNN contribution: {:.4f} -> {:.4f} (delta={:+.4f})".format(s_fix, s_gnn, s_gnn - s_fix))
    print("Conv: {}/11 -> {}/11".format(c_fix, c_gnn))

    # Per-matrix delta
    print("\nPer-matrix delta (fixed - GNN, positive = GNN helps):")
    for name in r_gnn:
        g, gc = r_gnn[name]
        f, fc = r_fix[name]
        print("  {:<12s}: GNN={:.3f} Fixed={:.3f} delta={:+.3f}".format(name, g, f, f - g))

    # === 2. OMEGA SWEEP ===
    print("\n" + "=" * 60)
    print("STUDY 2: OMEGA SENSITIVITY SWEEP")
    print("=" * 60)
    omegas = [0.5, 0.6, 0.667, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99]
    print("{:>6s} {:>8s} {:>5s} {:>8s} {:>8s}".format("omega", "ss_mean", "conv", "thermal", "saylr4"))
    for omega in omegas:
        r = eval_ss_only(model, device, omega)
        sc = score_from_results(r)
        nc = sum(1 for v in r.values() if v[1] > 0.5)
        th = r.get("thermal", (1.0, 0.0))
        sa = r.get("saylr4", (1.0, 0.0))
        th_tag = "OK" if th[1] > 0.5 else "FAIL"
        sa_tag = "OK" if sa[1] > 0.5 else "FAIL"
        print("{:6.3f} {:8.4f} {:>3d}/11 {:5.3f}({}) {:5.3f}({})".format(
            omega, sc, nc, th[0], th_tag, sa[0], sa_tag))

    print("\nDone.")


if __name__ == "__main__":
    main()
