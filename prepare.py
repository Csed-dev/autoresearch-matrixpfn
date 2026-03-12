"""
Fixed evaluation infrastructure for MatrixPFN autoresearch.
Downloads SuiteSparse evaluation matrices and defines the scoring function.

DO NOT MODIFY THIS FILE. The agent only edits train.py.

Usage:
    uv run prepare.py              # download eval matrices
    uv run prepare.py --verify     # verify all matrices are present
"""

import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from scipy.io import mmread
from scipy.sparse import issparse

CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "autoresearch-matrixpfn"
SUITESPARSE_DIR = CACHE_DIR / "suitesparse"

TIME_BUDGET = 300

FGMRES_RESTART = 30
FGMRES_MAX_ITERS = 300
FGMRES_RTOL = 1e-6
FGMRES_TIMEOUT = 60.0

NUM_RHS = 5
NUM_SYNTHETIC_MATRICES = 20

SYNTHETIC_EVAL_GRIDS = (16, 32, 64)
SYNTHETIC_TRAINING_GRIDS = (16, 24, 32)

EVAL_MATRICES = [
    ("HB", "sherman1"),
    ("HB", "sherman3"),
    ("HB", "sherman4"),
    ("Bai", "rdb1250"),
    ("Bai", "pde2961"),
    ("Averous", "epb0"),
    ("Brunetiere", "thermal"),
    ("HB", "orsirr_1"),
    ("HB", "orsreg_1"),
    ("HB", "watt__1"),
    ("HB", "saylr4"),
]

SUITESPARSE_BASE_URL = "https://suitesparse-collection-website.herokuapp.com/MM"


def download_matrix(group: str, name: str) -> Path:
    matrix_dir = SUITESPARSE_DIR / name
    mtx_file = matrix_dir / f"{name}.mtx"

    if mtx_file.exists():
        return mtx_file

    matrix_dir.mkdir(parents=True, exist_ok=True)
    url = f"{SUITESPARSE_BASE_URL}/{group}/{name}.tar.gz"
    tar_path = matrix_dir / f"{name}.tar.gz"

    print(f"  Downloading {group}/{name}...")
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            urllib.request.urlretrieve(url, tar_path)
            break
        except Exception as e:
            print(f"    Attempt {attempt}/{max_attempts} failed: {e}")
            if attempt == max_attempts:
                raise RuntimeError(f"Failed to download {group}/{name} after {max_attempts} attempts") from e
            time.sleep(2 ** attempt)

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".mtx"):
                member.name = Path(member.name).name
                tf.extract(member, matrix_dir)
                break

    tar_path.unlink()

    if not mtx_file.exists():
        raise RuntimeError(f"Matrix file not found after extraction: {mtx_file}")

    return mtx_file


def download_all_matrices():
    SUITESPARSE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(EVAL_MATRICES)} SuiteSparse evaluation matrices...")

    for group, name in EVAL_MATRICES:
        mtx_path = download_matrix(group, name)
        A = mmread(str(mtx_path))
        print(f"    {group}/{name}: n={A.shape[0]}, nnz={A.nnz}")

    print("Done.")


def load_suitesparse_matrix(name: str, device: torch.device) -> torch.Tensor:
    mtx_file = SUITESPARSE_DIR / name / f"{name}.mtx"
    if not mtx_file.exists():
        raise FileNotFoundError(f"Matrix not found: {mtx_file}. Run: uv run prepare.py")

    A_scipy = mmread(str(mtx_file)).tocsc()
    rows, cols = A_scipy.nonzero()
    values = np.array(A_scipy[rows, cols]).flatten().astype(np.float64)

    indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
    vals = torch.tensor(values, dtype=torch.float64, device=device)
    n = A_scipy.shape[0]

    return torch.sparse_coo_tensor(indices, vals, (n, n)).coalesce().to_sparse_csc()


def evaluate_score(model_path: str, device: torch.device) -> dict:
    from matrixpfn.precond.matrix_pfn import MatrixPFN
    from matrixpfn.precond.jacobi import Jacobi
    from matrixpfn.solver.fgmres import FGMRES
    from matrixpfn.generator.domains.diffusion import DiffusionGenerator

    pfn = MatrixPFN.from_pretrained(model_path, device=device)
    solver = FGMRES(
        restart=FGMRES_RESTART,
        max_iters=FGMRES_MAX_ITERS,
        rtol=FGMRES_RTOL,
        timeout=FGMRES_TIMEOUT,
    )
    jacobi = Jacobi()

    torch.manual_seed(99999)
    np.random.seed(99999)

    synthetic_scores = []
    synthetic_jacobi_scores = []
    synthetic_details = {}

    for gs in SYNTHETIC_EVAL_GRIDS:
        ood_tag = " (OOD)" if gs not in SYNTHETIC_TRAINING_GRIDS else ""
        gen = DiffusionGenerator((gs,), device)
        gs_pfn_iters = []
        gs_jacobi_iters = []

        for i in range(NUM_SYNTHETIC_MATRICES):
            batch = gen.generate_batch(1, 5)
            A = torch.sparse_coo_tensor(
                batch.indices, batch.values[0], (batch.n, batch.n)
            ).coalesce().to_sparse_csc()
            b = torch.randn(batch.n, dtype=torch.float64, device=device)

            try:
                result = pfn.solve(A, b, restart=FGMRES_RESTART,
                                   max_iters=FGMRES_MAX_ITERS, rtol=FGMRES_RTOL,
                                   timeout=FGMRES_TIMEOUT, progress_bar=False)
                normalized = result.iterations / FGMRES_MAX_ITERS if not result.converged else result.iterations / FGMRES_MAX_ITERS
                gs_pfn_iters.append(normalized)
            except Exception:
                gs_pfn_iters.append(1.0)

            try:
                diag_A = torch.zeros(batch.n, dtype=torch.float64, device=device)
                A_coo = A.to_sparse_coo().coalesce()
                idx = A_coo.indices()
                diag_mask = idx[0] == idx[1]
                diag_A[idx[0, diag_mask]] = A_coo.values()[diag_mask]
                jacobi.set_matrix(diag_A)
                jac_result = solver.solve(A, b, M=jacobi, progress_bar=False)
                gs_jacobi_iters.append(jac_result.iterations / FGMRES_MAX_ITERS)
            except Exception:
                gs_jacobi_iters.append(1.0)

        pfn_mean = sum(gs_pfn_iters) / len(gs_pfn_iters)
        jac_mean = sum(gs_jacobi_iters) / len(gs_jacobi_iters)
        synthetic_scores.extend(gs_pfn_iters)
        synthetic_jacobi_scores.extend(gs_jacobi_iters)

        pfn_conv = sum(1 for s in gs_pfn_iters if s < 1.0)
        synthetic_details[f"{gs}x{gs}{ood_tag}"] = {
            "pfn_mean_norm_iter": pfn_mean,
            "jacobi_mean_norm_iter": jac_mean,
            "pfn_conv_rate": pfn_conv / len(gs_pfn_iters),
        }

    ss_scores = []
    ss_jacobi_scores = []
    ss_details = {}

    for group, name in EVAL_MATRICES:
        try:
            A = load_suitesparse_matrix(name, device)
        except FileNotFoundError:
            print(f"  SKIP {name}: not downloaded")
            continue

        n = A.shape[0]
        mat_pfn_iters = []
        mat_jac_iters = []

        for rhs_idx in range(NUM_RHS):
            b = torch.randn(n, dtype=torch.float64, device=device)

            try:
                result = pfn.solve(A, b, restart=FGMRES_RESTART,
                                   max_iters=FGMRES_MAX_ITERS, rtol=FGMRES_RTOL,
                                   timeout=FGMRES_TIMEOUT, progress_bar=False)
                mat_pfn_iters.append(result.iterations / FGMRES_MAX_ITERS)
            except Exception:
                mat_pfn_iters.append(1.0)

            try:
                diag_A = torch.zeros(n, dtype=torch.float64, device=device)
                A_coo = A.to_sparse_coo().coalesce()
                idx = A_coo.indices()
                diag_mask = idx[0] == idx[1]
                diag_A[idx[0, diag_mask]] = A_coo.values()[diag_mask]
                jacobi.set_matrix(diag_A)
                jac_result = solver.solve(A, b, M=jacobi, progress_bar=False)
                mat_jac_iters.append(jac_result.iterations / FGMRES_MAX_ITERS)
            except Exception:
                mat_jac_iters.append(1.0)

        pfn_mean = sum(mat_pfn_iters) / len(mat_pfn_iters)
        jac_mean = sum(mat_jac_iters) / len(mat_jac_iters)
        pfn_conv = sum(1 for s in mat_pfn_iters if s < 1.0)

        ss_scores.extend(mat_pfn_iters)
        ss_jacobi_scores.extend(mat_jac_iters)
        ss_details[name] = {
            "pfn_mean_norm_iter": pfn_mean,
            "jacobi_mean_norm_iter": jac_mean,
            "pfn_conv_rate": pfn_conv / len(mat_pfn_iters),
            "n": n,
        }

    synth_score = sum(synthetic_scores) / len(synthetic_scores) if synthetic_scores else 1.0
    ss_score = sum(ss_scores) / len(ss_scores) if ss_scores else 1.0
    synth_jacobi = sum(synthetic_jacobi_scores) / len(synthetic_jacobi_scores) if synthetic_jacobi_scores else 1.0
    ss_jacobi = sum(ss_jacobi_scores) / len(ss_jacobi_scores) if ss_jacobi_scores else 1.0

    combined_score = 0.3 * synth_score + 0.7 * ss_score

    synth_conv = sum(1 for s in synthetic_scores if s < 1.0) / len(synthetic_scores) * 100 if synthetic_scores else 0.0
    ss_conv = sum(1 for s in ss_scores if s < 1.0) / len(ss_scores) * 100 if ss_scores else 0.0

    return {
        "score": combined_score,
        "synthetic_score": synth_score,
        "suitesparse_score": ss_score,
        "synthetic_jacobi": synth_jacobi,
        "suitesparse_jacobi": ss_jacobi,
        "synthetic_conv_pct": synth_conv,
        "suitesparse_conv_pct": ss_conv,
        "synthetic_details": synthetic_details,
        "suitesparse_details": ss_details,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {CACHE_DIR}")

    download_all_matrices()

    if args.verify:
        print("\nVerifying all matrices load correctly...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for group, name in EVAL_MATRICES:
            A = load_suitesparse_matrix(name, device)
            print(f"  {name}: n={A.shape[0]} OK")
        print("All matrices verified.")
