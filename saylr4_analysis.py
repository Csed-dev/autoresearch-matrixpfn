"""Spectral analysis of saylr4 to find if any omega makes Neumann converge."""
import numpy as np
from scipy.io import mmread
from pathlib import Path
import time

cache = Path.home() / ".cache" / "autoresearch-matrixpfn" / "suitesparse" / "saylr4" / "saylr4.mtx"
A = mmread(str(cache)).tocsc()
n = A.shape[0]
diag = np.array(A.diagonal()).flatten()
print("saylr4: n={}, nnz={}".format(n, A.nnz))
print("Diag: min|d|={:.4e}, max|d|={:.4e}".format(np.min(np.abs(diag)), np.max(np.abs(diag))))
print("Neg diag: {}, Zero: {}".format(np.sum(diag < 0), np.sum(np.abs(diag) < 1e-15)))

A_dense = A.toarray()
row_norms = np.sum(np.abs(A_dense), axis=1)
off_diag = row_norms - np.abs(diag)
dd = np.abs(diag) / (off_diag + 1e-15)
print("Diag dominance: min={:.4f}, mean={:.4f}, pct_strict={:.1f}%".format(
    np.min(dd), np.mean(dd), 100 * np.mean(dd > 1)))
sym_err = np.max(np.abs((A - A.T).toarray()))
print("Symmetric: {} (max|A-AT|={:.2e})".format("YES" if sym_err < 1e-10 else "NO", sym_err))

D_inv_A = np.diag(1.0 / diag) @ A_dense

print("\nSpectral radius rho(J_omega) for J_omega = I - omega * D^-1 A:")
for omega in [0.3, 0.5, 0.667, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5]:
    t0 = time.time()
    J = np.eye(n) - omega * D_inv_A
    eigvals = np.linalg.eigvals(J)
    rho = np.max(np.abs(eigvals))
    dt = time.time() - t0
    tag = "CONV" if rho < 1 else "DIV"
    print("  omega={:.3f}: rho={:.6f} {} ({:.1f}s)".format(omega, rho, tag, dt))

print("\nCoarse search for optimal omega (0.05 to 2.0):")
best_omega, best_rho = 0, 999
for omega_100 in range(5, 201, 5):
    omega = omega_100 / 100.0
    J = np.eye(n) - omega * D_inv_A
    rho = np.max(np.abs(np.linalg.eigvals(J)))
    if rho < best_rho:
        best_omega, best_rho = omega, rho
    if omega_100 % 25 == 0 or rho < 1.01:
        tag = "CONV" if rho < 1 else "DIV"
        print("  omega={:.2f}: rho={:.6f} {}".format(omega, rho, tag))

tag = "CONV" if best_rho < 1 else "DIV"
print("\nBest: omega={:.2f}, rho={:.6f} {}".format(best_omega, best_rho, tag))
