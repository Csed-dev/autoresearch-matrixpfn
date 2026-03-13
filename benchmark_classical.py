"""
Benchmark classical preconditioners (Jacobi, ILU, AMG, BlockJacobi) on SuiteSparse matrices.
Uses the same matrices and FGMRES settings as prepare.py.

Usage: uv run benchmark_classical.py
"""

import numpy as np
import torch

from matrixpfn.solver.fgmres import FGMRES
from matrixpfn.precond.ilu import ILU
from matrixpfn.precond.amg import AMG
from matrixpfn.precond.jacobi import Jacobi
from matrixpfn.precond.block_jacobi import BlockJacobi

from prepare import (
    EVAL_MATRICES,
    FGMRES_RESTART,
    FGMRES_MAX_ITERS,
    FGMRES_RTOL,
    FGMRES_TIMEOUT,
    NUM_RHS,
    load_suitesparse_matrix,
    download_all_matrices,
)

PRECONDITIONERS = ["none", "jacobi", "ilu", "amg", "block_jacobi"]


def build_preconditioner(name: str, A_csc: torch.Tensor):
    if name == "none":
        return None
    if name == "jacobi":
        return Jacobi(A_csc)
    if name == "ilu":
        return ILU(A_csc)
    if name == "amg":
        return AMG(A_csc)
    if name == "block_jacobi":
        return BlockJacobi(A_csc)
    raise ValueError(f"Unknown preconditioner: {name}")


def benchmark_matrix(solver, A, n, precond, device):
    torch.manual_seed(12345)
    converged_count = 0
    total_iters = 0

    for _ in range(NUM_RHS):
        x_true = torch.randn(n, dtype=torch.float64, device=device)
        b = A @ x_true
        result = solver.solve(A, b, M=precond, progress_bar=False)
        if result.converged:
            converged_count += 1
        total_iters += result.iterations

    conv_rate = converged_count / NUM_RHS
    mean_norm_iter = total_iters / (NUM_RHS * FGMRES_MAX_ITERS)
    return conv_rate, mean_norm_iter


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    download_all_matrices()

    solver = FGMRES(
        restart=FGMRES_RESTART,
        max_iters=FGMRES_MAX_ITERS,
        rtol=FGMRES_RTOL,
        timeout=FGMRES_TIMEOUT,
    )

    header = f"{'matrix':<12s} {'n':>5s}"
    for p in PRECONDITIONERS:
        header += f" | {p:>14s}"
    print(header)
    print("-" * len(header))

    for group, name in EVAL_MATRICES:
        A = load_suitesparse_matrix(name, device)
        n = A.shape[0]
        row = f"{name:<12s} {n:>5d}"

        for pname in PRECONDITIONERS:
            precond = build_preconditioner(pname, A)
            conv_rate, mean_norm = benchmark_matrix(solver, A, n, precond, device)
            status = f"{conv_rate*100:.0f}%/{mean_norm:.3f}"
            row += f" | {status:>14s}"

        print(row)

    print()
    print("Format: conv% / normalized_iterations (lower = better)")


if __name__ == "__main__":
    main()
