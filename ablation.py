"""Quick ablation: GNN vs fixed Neumann c_k=1. Uses saylr4-only eval for speed."""
import numpy as np
import torch
from matrixpfn.precond.jacobi import Jacobi
from matrixpfn.solver.fgmres import FGMRES
from prepare import EVAL_MATRICES, NUM_RHS, FGMRES_RESTART, FGMRES_MAX_ITERS, FGMRES_RTOL, FGMRES_TIMEOUT, load_suitesparse_matrix
from train import load_checkpoint, JACOBI_OMEGA

CHECKPOINT = "best_model.pt"


class NeumannPrecond:
    def __init__(self, coeffs, D_inv_A, D_inv, omega):
        self.coeffs = coeffs.double()
        self.D_inv_A = D_inv_A
        self.D_inv = D_inv
        self.omega = omega

    def apply(self, r):
        K = self.coeffs.shape[1]
        w = self.omega
        power = w * self.D_inv * r
        result = self.coeffs[:, 0] * power
        for k in range(1, K):
            power = power - w * (self.D_inv_A @ power)
            result = result + self.coeffs[:, k] * power
        return result


@torch.no_grad()
def eval_ss(model, device, omega, fixed=False):
    model.eval()
    solver = FGMRES(restart=FGMRES_RESTART, max_iters=FGMRES_MAX_ITERS, rtol=FGMRES_RTOL, timeout=FGMRES_TIMEOUT)
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
            if fixed:
                coeffs = torch.ones_like(coeffs)
            precond = NeumannPrecond(coeffs, model.D_inv_A, model.D_inv, omega)
        except Exception:
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(CHECKPOINT, device)
    print("=== GNN ABLATION: omega=0.9 ===")
    print("{:<12s} {:>8s} {:>8s} {:>8s}".format("Matrix", "GNN", "Fixed", "Delta"))
    print("-" * 40)
    gnn = eval_ss(model, device, 0.9, fixed=False)
    fix = eval_ss(model, device, 0.9, fixed=True)
    for name in gnn:
        g_pfn, g_conv = gnn[name]
        f_pfn, f_conv = fix[name]
        g_tag = "OK" if g_conv > 0.5 else "FAIL"
        f_tag = "OK" if f_conv > 0.5 else "FAIL"
        delta = f_pfn - g_pfn
        print("{:<12s} {:>5.3f}({}) {:>5.3f}({}) {:>+6.3f}".format(name, g_pfn, g_tag, f_pfn, f_tag, delta))
    g_scores = [v[0] for v in gnn.values()]
    f_scores = [v[0] for v in fix.values()]
    g_mean = sum(g_scores)/len(g_scores)
    f_mean = sum(f_scores)/len(f_scores)
    print("-" * 40)
    print("SS mean:     {:.4f}       {:.4f}       {:+.4f}".format(g_mean, f_mean, f_mean - g_mean))
    g_conv = sum(1 for v in gnn.values() if v[1] > 0.5)
    f_conv = sum(1 for v in fix.values() if v[1] > 0.5)
    print("Converged:   {}/11        {}/11".format(g_conv, f_conv))


if __name__ == "__main__":
    main()
