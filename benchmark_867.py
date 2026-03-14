"""
Benchmark: Pure Neumann-series preconditioner on ALL 867 SuiteSparse matrices.
NO ML. NO TRAINING. NO GNN. Just classical numerics:
  M(r) = sum_{k=0}^{K-1} J_omega^k * (omega * D^{-1} * r)
  where J_omega = I - omega * D^{-1} * A, omega = 0.9
  Adaptive K: stop when ||p_k|| < tol * ||result|| (early termination)

Usage: uv run benchmark_867.py
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from scipy.io import mmread

from matrixpfn.precond.jacobi import Jacobi
from matrixpfn.solver.fgmres import FGMRES
from prepare import (
    download_matrix, SUITESPARSE_DIR,
    FGMRES_RESTART, FGMRES_MAX_ITERS, FGMRES_RTOL, FGMRES_TIMEOUT,
)

OMEGA = 0.9
K_MAX = 256
K_ADAPTIVE_TOL = 1e-10  # Stop Neumann recurrence when power norm drops below this
NUM_RHS = 2
MANIFEST_PATH = "suitesparse_manifest.json"


class AdaptiveNeumannPreconditioner:
    """Pure Neumann series with adaptive K. No GNN, no ML."""

    def __init__(self, A_csc, omega=0.9, k_max=256, tol=1e-10):
        if A_csc.layout == torch.sparse_csc:
            A_coo = A_csc.to_sparse_coo().coalesce()
        else:
            A_coo = A_csc.coalesce()

        indices = A_coo.indices()
        values = A_coo.values()
        n = A_csc.shape[0]
        rows, cols = indices

        diag = torch.zeros(n, dtype=values.dtype, device=values.device)
        diag_mask = rows == cols
        diag[rows[diag_mask]] = values[diag_mask]

        self.D_inv = 1.0 / diag
        d_inv_values = self.D_inv[rows] * values
        self.D_inv_A = torch.sparse_coo_tensor(
            indices, d_inv_values, (n, n)
        ).coalesce().to_sparse_csc()

        self.omega = omega
        self.k_max = k_max
        self.tol = tol
        self.last_k = 0  # Track how many terms were actually used

    def apply(self, r):
        omega = self.omega
        d_inv_r = omega * self.D_inv * r
        power = d_inv_r
        result = power.clone()
        result_norm = result.norm().item()

        for k in range(1, self.k_max):
            power = power - omega * (self.D_inv_A @ power)
            result = result + power
            # Adaptive stopping
            power_norm = power.norm().item()
            if result_norm > 0 and power_norm < self.tol * result_norm:
                self.last_k = k + 1
                return result
            result_norm = max(result_norm, result.norm().item())

        self.last_k = self.k_max
        return result


def load_matrix(group, name, device):
    """Download and load a SuiteSparse matrix."""
    mtx_file = SUITESPARSE_DIR / name / f"{name}.mtx"
    if not mtx_file.exists():
        download_matrix(group, name)

    A_scipy = mmread(str(mtx_file)).tocsc()
    n = A_scipy.shape[0]
    nnz = A_scipy.nnz

    # Check for zero/near-zero diagonal
    diag = np.array(A_scipy.diagonal()).flatten()
    n_zero = np.sum(np.abs(diag) < 1e-15)
    if n_zero > 0:
        return None, n, nnz, f"{n_zero} zero diag entries"

    # Check square
    if A_scipy.shape[0] != A_scipy.shape[1]:
        return None, n, nnz, "non-square"

    rows, cols = A_scipy.nonzero()
    values = np.array(A_scipy[rows, cols]).flatten().astype(np.float64)
    indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
    vals = torch.tensor(values, dtype=torch.float64, device=device)

    A = torch.sparse_coo_tensor(indices, vals, (n, n)).coalesce().to_sparse_csc()
    return A, n, nnz, None


@torch.no_grad()
def eval_matrix(A, n, name, device):
    """Evaluate Neumann preconditioner and Jacobi on one matrix."""
    solver = FGMRES(restart=FGMRES_RESTART, max_iters=FGMRES_MAX_ITERS,
                    rtol=FGMRES_RTOL, timeout=FGMRES_TIMEOUT)

    # Neumann preconditioner (adaptive K)
    try:
        precond = AdaptiveNeumannPreconditioner(A, omega=OMEGA, k_max=K_MAX, tol=K_ADAPTIVE_TOL)
    except Exception as e:
        return {"name": name, "n": n, "neumann": 1.0, "jacobi": 1.0,
                "neumann_conv": 0.0, "error": str(e)}

    neu_iters, neu_conv, k_used = [], [], []
    jac_iters, jac_conv = [], []

    for _ in range(NUM_RHS):
        b = torch.randn(n, dtype=torch.float64, device=device)

        # Neumann
        try:
            result = solver.solve(A, b, M=precond, progress_bar=False)
            neu_iters.append(result.iterations / FGMRES_MAX_ITERS)
            neu_conv.append(result.converged)
            k_used.append(precond.last_k)
        except Exception:
            neu_iters.append(1.0)
            neu_conv.append(False)
            k_used.append(K_MAX)

        # Jacobi
        try:
            jacobi = Jacobi(A)
            jac_result = solver.solve(A, b, M=jacobi, progress_bar=False)
            jac_iters.append(jac_result.iterations / FGMRES_MAX_ITERS)
            jac_conv.append(jac_result.converged)
        except Exception:
            jac_iters.append(1.0)
            jac_conv.append(False)

    return {
        "name": name,
        "n": n,
        "neumann": sum(neu_iters) / len(neu_iters),
        "jacobi": sum(jac_iters) / len(jac_iters),
        "neumann_conv": sum(neu_conv) / len(neu_conv),
        "jacobi_conv": sum(jac_conv) / len(jac_conv),
        "avg_k": sum(k_used) / len(k_used),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    print("Config: omega={}, K_max={}, adaptive_tol={}, num_rhs={}".format(
        OMEGA, K_MAX, K_ADAPTIVE_TOL, NUM_RHS))

    # Load manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    print("Manifest: {} matrices".format(len(manifest)))

    results = []
    skipped = 0
    errors = 0

    for i, mat in enumerate(manifest):
        group = mat["group"]
        name = mat["name"]
        n = mat["rows"]

        # Skip very large matrices (would be too slow)
        if n > 50000:
            skipped += 1
            continue

        t0 = time.time()
        try:
            A, actual_n, nnz, skip_reason = load_matrix(group, name, device)
            if skip_reason:
                print("{:>4d}/{} SKIP {:<20s} n={:>6d}: {}".format(
                    i+1, len(manifest), name, n, skip_reason))
                skipped += 1
                continue

            r = eval_matrix(A, actual_n, name, device)
            r["nnz"] = nnz
            r["group"] = group
            dt = time.time() - t0

            neu_tag = "OK" if r["neumann_conv"] > 0.5 else "FAIL"
            jac_tag = "OK" if r["jacobi_conv"] > 0.5 else "FAIL"
            beats_jac = r["neumann"] < r["jacobi"] - 0.01

            print("{:>4d}/{} {:<20s} n={:>6d} nnz={:>8d} | neu={:.3f}({:>4s}) jac={:.3f}({:>4s}) K_avg={:>3.0f} | {:.1f}s {}".format(
                i+1, len(manifest), name, actual_n, nnz,
                r["neumann"], neu_tag, r["jacobi"], jac_tag,
                r.get("avg_k", 0), dt,
                "BEATS_JAC" if beats_jac else ""))
            results.append(r)

        except Exception as e:
            print("{:>4d}/{} ERROR {:<20s}: {}".format(i+1, len(manifest), name, str(e)[:60]))
            errors += 1
            continue

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    n_total = len(results)
    n_neu_conv = sum(1 for r in results if r["neumann_conv"] > 0.5)
    n_jac_conv = sum(1 for r in results if r["jacobi_conv"] > 0.5)
    n_beats_jac = sum(1 for r in results if r["neumann"] < r["jacobi"] - 0.01)
    n_beats_jac_conv = sum(1 for r in results
                           if r["neumann_conv"] > 0.5 and r["jacobi_conv"] <= 0.5)

    print("Evaluated: {}/{} matrices (skipped {}, errors {})".format(
        n_total, len(manifest), skipped, errors))
    print("Neumann converged: {}/{} ({:.1f}%)".format(n_neu_conv, n_total, 100*n_neu_conv/max(n_total,1)))
    print("Jacobi converged: {}/{} ({:.1f}%)".format(n_jac_conv, n_total, 100*n_jac_conv/max(n_total,1)))
    print("Neumann beats Jacobi: {}/{} ({:.1f}%)".format(n_beats_jac, n_total, 100*n_beats_jac/max(n_total,1)))
    print("Neumann solves where Jacobi fails: {}".format(n_beats_jac_conv))

    # Adaptive K statistics
    k_values = [r.get("avg_k", K_MAX) for r in results if r["neumann_conv"] > 0.5]
    if k_values:
        print("\nAdaptive K (converging matrices):")
        print("  mean K: {:.1f}".format(sum(k_values)/len(k_values)))
        print("  min K: {:.0f}".format(min(k_values)))
        print("  max K: {:.0f}".format(max(k_values)))
        print("  K < 32: {}  K < 64: {}  K < 128: {}  K = 256: {}".format(
            sum(1 for k in k_values if k < 32),
            sum(1 for k in k_values if k < 64),
            sum(1 for k in k_values if k < 128),
            sum(1 for k in k_values if k >= 256)))

    # By problem kind
    kinds = {}
    for r in results:
        kind = "unknown"
        for m in manifest:
            if m["name"] == r["name"]:
                kind = m.get("kind", "unknown")
                break
        kinds.setdefault(kind, []).append(r)

    print("\nBy problem kind:")
    for kind, rs in sorted(kinds.items(), key=lambda x: -len(x[1])):
        n_k = len(rs)
        n_conv = sum(1 for r in rs if r["neumann_conv"] > 0.5)
        if n_k >= 3:
            print("  {:<45s}: {}/{} converge ({:.0f}%)".format(
                kind[:45], n_conv, n_k, 100*n_conv/n_k))


if __name__ == "__main__":
    main()
