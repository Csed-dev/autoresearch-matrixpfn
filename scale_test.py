"""
Scaling test: evaluate trained Neumann K=256 model on larger SuiteSparse matrices.
Tests how the preconditioner generalizes to n > 10K.

Usage: uv run train.py && uv run scale_test.py
"""

import time
import numpy as np
import torch

from matrixpfn.precond.jacobi import Jacobi
from matrixpfn.solver.fgmres import FGMRES
from prepare import (
    download_matrix, load_suitesparse_matrix, SUITESPARSE_DIR,
    FGMRES_RESTART, FGMRES_MAX_ITERS, FGMRES_RTOL, FGMRES_TIMEOUT,
)
from train import load_checkpoint, PolynomialPreconditioner, JACOBI_OMEGA

CHECKPOINT_PATH = "best_model.pt"
NUM_RHS = 3  # fewer RHS for speed on large matrices

# Diverse scaling test matrices (10K - 50K)
SCALE_MATRICES = [
    # 10K-15K range
    ("Goodwin", "Goodwin_030"),   # n=10142, CFD
    ("Bindel", "ted_A"),          # n=10605, thermal
    ("Mallya", "lhr10"),          # n=10672, chemical process
    ("Schenk_ISEI", "igbt3"),     # n=10938, semiconductor
    ("FIDAP", "ex19"),            # n=12005, CFD
    ("Bomhof", "circuit_3"),      # n=12127, circuit
    ("FEMLAB", "sme3Da"),         # n=12504, structural
    # 15K-25K range
    ("PARSEC", "Si5H12"),         # n=19896, quantum chemistry
    ("Oberwolfach", "t3dl_a"),    # n=20360, model reduction
    ("Sandia", "mult_dcop_02"),   # n=25187, circuit simulation
    # 30K-50K range
    ("ATandT", "onetone1"),       # n=36057, frequency-domain circuit
    ("Schenk_IBMNA", "c-62"),     # n=41731, optimization
]


@torch.no_grad()
def eval_matrix(model, A, name, n, solver, device):
    """Evaluate PFN and Jacobi on a single matrix."""
    try:
        model.set_matrix(A)
        coeffs = model()
        precond = PolynomialPreconditioner(coeffs, model.D_inv_A, model.D_inv)
    except Exception as e:
        return {"name": name, "n": n, "pfn": 1.0, "jacobi": 1.0,
                "pfn_conv": 0.0, "error": str(e)}

    pfn_iters = []
    pfn_conv = []
    jac_iters = []
    jac_conv = []

    for _ in range(NUM_RHS):
        b = torch.randn(n, dtype=torch.float64, device=device)

        try:
            result = solver.solve(A, b, M=precond, progress_bar=False)
            pfn_iters.append(result.iterations / FGMRES_MAX_ITERS)
            pfn_conv.append(result.converged)
        except Exception:
            pfn_iters.append(1.0)
            pfn_conv.append(False)

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
        "pfn": sum(pfn_iters) / len(pfn_iters),
        "jacobi": sum(jac_iters) / len(jac_iters),
        "pfn_conv": sum(pfn_conv) / len(pfn_conv),
        "jac_conv": sum(jac_conv) / len(jac_conv),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_checkpoint(CHECKPOINT_PATH, device)
    print(f"Loaded checkpoint")

    solver = FGMRES(restart=FGMRES_RESTART, max_iters=FGMRES_MAX_ITERS,
                    rtol=FGMRES_RTOL, timeout=FGMRES_TIMEOUT)

    torch.manual_seed(99999)
    np.random.seed(99999)

    print(f"\n{'='*70}")
    print(f"SCALING TEST: {len(SCALE_MATRICES)} matrices, n = 10K-50K")
    print(f"{'='*70}")

    results = []
    for group, name in SCALE_MATRICES:
        # Download if needed
        try:
            download_matrix(group, name)
        except Exception as e:
            print(f"  SKIP {name}: download failed ({e})")
            continue

        try:
            from scipy.io import mmread
            mtx_file = SUITESPARSE_DIR / name / f"{name}.mtx"
            A_scipy = mmread(str(mtx_file)).tocsc()
            n = A_scipy.shape[0]
            nnz = A_scipy.nnz

            # Check for zero diagonal (skip if so)
            diag = A_scipy.diagonal()
            zero_diag = np.sum(np.abs(diag) < 1e-15)
            if zero_diag > 0:
                print(f"  SKIP {name} (n={n}, nnz={nnz}): {zero_diag} zero diagonal entries")
                continue

            A = load_suitesparse_matrix(name, device)
        except Exception as e:
            print(f"  SKIP {name}: load failed ({e})")
            continue

        t0 = time.time()
        r = eval_matrix(model, A, name, n, solver, device)
        dt = time.time() - t0
        r["nnz"] = nnz
        r["eval_time"] = dt
        results.append(r)

        pfn_status = "OK" if r["pfn_conv"] > 0.5 else "FAIL"
        jac_status = "OK" if r.get("jac_conv", 0) > 0.5 else "FAIL"
        print(f"  {name:<20s} n={n:>6d} nnz={nnz:>8d} | pfn={r['pfn']:.3f}({pfn_status}) jac={r['jacobi']:.3f}({jac_status}) | {dt:.1f}s")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    n_tested = len(results)
    n_pfn_conv = sum(1 for r in results if r["pfn_conv"] > 0.5)
    n_jac_conv = sum(1 for r in results if r.get("jac_conv", 0) > 0.5)
    print(f"Tested: {n_tested} matrices")
    print(f"PFN converged: {n_pfn_conv}/{n_tested}")
    print(f"Jacobi converged: {n_jac_conv}/{n_tested}")

    # By size bucket
    for lo, hi in [(10000, 15000), (15000, 25000), (25000, 50000)]:
        bucket = [r for r in results if lo <= r["n"] < hi]
        if bucket:
            pfn_c = sum(1 for r in bucket if r["pfn_conv"] > 0.5)
            print(f"\n  n={lo//1000}K-{hi//1000}K ({len(bucket)} matrices): PFN converged {pfn_c}/{len(bucket)}")
            for r in bucket:
                pfn_status = "OK" if r["pfn_conv"] > 0.5 else "FAIL"
                print(f"    {r['name']:<20s} pfn={r['pfn']:.3f}({pfn_status}) jac={r['jacobi']:.3f}")


if __name__ == "__main__":
    main()
