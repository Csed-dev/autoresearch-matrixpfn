"""
Sweep script: omega sensitivity + GNN ablation.
Runs eval-only (no training) using a trained checkpoint.
For ablation: replaces GNN coefficients with fixed c_k=1.

Usage: uv run sweep.py
"""

import sys
import numpy as np
import torch

from matrixpfn.precond.jacobi import Jacobi
from matrixpfn.solver.fgmres import FGMRES
from matrixpfn.generator.domains.diffusion import DiffusionGenerator

from prepare import (
    EVAL_MATRICES, SYNTHETIC_EVAL_GRIDS, SYNTHETIC_TRAINING_GRIDS,
    NUM_RHS, NUM_SYNTHETIC_MATRICES, FGMRES_RESTART, FGMRES_MAX_ITERS,
    FGMRES_RTOL, FGMRES_TIMEOUT, load_suitesparse_matrix,
)
from train import PolyMPNN, load_checkpoint

CHECKPOINT_PATH = "best_model.pt"


class NeumannPreconditioner:
    """Weighted-Jacobi Neumann polynomial with configurable omega and coefficients."""

    def __init__(self, coeffs, D_inv_A, D_inv, omega):
        self.coeffs = coeffs.double()
        self.D_inv_A = D_inv_A
        self.D_inv = D_inv
        self.omega = omega

    def apply(self, r):
        K = self.coeffs.shape[1]
        omega = self.omega
        d_inv_r = omega * self.D_inv * r
        power = d_inv_r
        result = self.coeffs[:, 0] * power
        for k in range(1, K):
            power = power - omega * (self.D_inv_A @ power)
            result = result + self.coeffs[:, k] * power
        return result


@torch.no_grad()
def eval_with_omega(model, device, omega, use_fixed_coeffs=False):
    """Evaluate model with a specific omega. If use_fixed_coeffs, ignore GNN and use c_k=1."""
    model.eval()
    solver = FGMRES(restart=FGMRES_RESTART, max_iters=FGMRES_MAX_ITERS,
                    rtol=FGMRES_RTOL, timeout=FGMRES_TIMEOUT)
    torch.manual_seed(99999)
    np.random.seed(99999)

    ss_scores = []
    ss_details = {}

    for group, name in EVAL_MATRICES:
        try:
            A = load_suitesparse_matrix(name, device)
        except FileNotFoundError:
            continue

        n = A.shape[0]
        try:
            model.set_matrix(A)
            coeffs = model()
            if use_fixed_coeffs:
                coeffs = torch.ones_like(coeffs)
            precond = NeumannPreconditioner(coeffs, model.D_inv_A, model.D_inv, omega)
        except Exception:
            ss_scores.extend([1.0] * NUM_RHS)
            ss_details[name] = {"pfn": 1.0, "conv": 0.0}
            continue

        mat_iters = []
        mat_conv = []
        for _ in range(NUM_RHS):
            b = torch.randn(n, dtype=torch.float64, device=device)
            try:
                result = solver.solve(A, b, M=precond, progress_bar=False)
                mat_iters.append(result.iterations / FGMRES_MAX_ITERS)
                mat_conv.append(result.converged)
            except Exception:
                mat_iters.append(1.0)
                mat_conv.append(False)

        pfn_mean = sum(mat_iters) / len(mat_iters)
        ss_scores.extend(mat_iters)
        ss_details[name] = {"pfn": pfn_mean, "conv": sum(mat_conv) / len(mat_conv)}

    # Synthetic eval
    synth_scores = []
    for gs in SYNTHETIC_EVAL_GRIDS:
        gen = DiffusionGenerator((gs,), device)
        for _ in range(NUM_SYNTHETIC_MATRICES):
            batch = gen.generate_batch(1, 5)
            A = torch.sparse_coo_tensor(
                batch.indices, batch.values[0], (batch.n, batch.n)
            ).coalesce().to_sparse_csc()
            b = torch.randn(batch.n, dtype=torch.float64, device=device)
            try:
                model.set_matrix(A)
                coeffs = model()
                if use_fixed_coeffs:
                    coeffs = torch.ones_like(coeffs)
                precond = NeumannPreconditioner(coeffs, model.D_inv_A, model.D_inv, omega)
                result = solver.solve(A, b, M=precond, progress_bar=False)
                synth_scores.append(result.iterations / FGMRES_MAX_ITERS)
            except Exception:
                synth_scores.append(1.0)

    ss_score = sum(ss_scores) / len(ss_scores) if ss_scores else 1.0
    synth_score = sum(synth_scores) / len(synth_scores) if synth_scores else 1.0
    combined = 0.3 * synth_score + 0.7 * ss_score

    return combined, ss_score, synth_score, ss_details


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # First: train the model (run train.py first!)
    # Load trained checkpoint
    model = load_checkpoint(CHECKPOINT_PATH, device)
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    # === Part 1: Omega Sensitivity Sweep ===
    print("\n" + "=" * 60)
    print("OMEGA SENSITIVITY SWEEP")
    print("=" * 60)

    omegas = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    print(f"Testing {len(omegas)} omega values...")

    for omega in omegas:
        score, ss, synth, details = eval_with_omega(model, device, omega)
        thermal_pfn = details.get("thermal", {}).get("pfn", -1)
        thermal_conv = details.get("thermal", {}).get("conv", 0)
        saylr4_pfn = details.get("saylr4", {}).get("pfn", -1)
        n_conv = sum(1 for d in details.values() if d["conv"] > 0.5)
        print(f"omega={omega:.2f} | score={score:.4f} | ss={ss:.4f} | synth={synth:.4f} | conv={n_conv}/11 | thermal={thermal_pfn:.3f}({'OK' if thermal_conv > 0.5 else 'FAIL'}) | saylr4={saylr4_pfn:.3f}")

    # === Part 2: GNN Ablation ===
    print("\n" + "=" * 60)
    print("GNN ABLATION: Fixed c_k=1 (no GNN) vs GNN-predicted")
    print("=" * 60)

    # With GNN (normal)
    score_gnn, ss_gnn, synth_gnn, details_gnn = eval_with_omega(model, device, 0.9, use_fixed_coeffs=False)
    print(f"WITH GNN:    score={score_gnn:.4f} | ss={ss_gnn:.4f} | synth={synth_gnn:.4f}")
    for name, d in details_gnn.items():
        status = "OK" if d["conv"] > 0.5 else "FAIL"
        print(f"  {name:<12s}: pfn={d['pfn']:.3f} {status}")

    # Without GNN (fixed coeffs)
    score_fix, ss_fix, synth_fix, details_fix = eval_with_omega(model, device, 0.9, use_fixed_coeffs=True)
    print(f"\nFIXED c_k=1: score={score_fix:.4f} | ss={ss_fix:.4f} | synth={synth_fix:.4f}")
    for name, d in details_fix.items():
        status = "OK" if d["conv"] > 0.5 else "FAIL"
        gnn_val = details_gnn.get(name, {}).get("pfn", -1)
        delta = d["pfn"] - gnn_val if gnn_val >= 0 else 0
        print(f"  {name:<12s}: pfn={d['pfn']:.3f} {status}  (delta={delta:+.3f})")

    print(f"\nGNN contribution: score {score_fix:.4f} -> {score_gnn:.4f} (delta={score_gnn - score_fix:+.4f})")


if __name__ == "__main__":
    main()
