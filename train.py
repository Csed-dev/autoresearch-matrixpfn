"""
MatrixPFN autoresearch training script.
Run 35: Neumann-basis polynomial — uses J = I - D^{-1}A instead of D^{-1}A.
The Neumann series M = D^{-1} * sum J^k is the optimal polynomial preconditioner
when rho(J) < 1. Powers of J naturally decay (unlike D^{-1}A which grows).
Init all c_k = 1 (Neumann series). GNN learns corrections from there.

Usage: uv run train.py
"""

import math
import time
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from matrixpfn.precond.jacobi import Jacobi
from matrixpfn.solver.fgmres import FGMRES
from matrixpfn.generator.base import GeneratorConfig, MatrixDomain
from matrixpfn.generator.registry import build_training_registry, MatrixGeneratorRegistry
from matrixpfn.generator.online import OnlineMatrixDataset
from matrixpfn.generator.domains.diffusion import DiffusionGenerator

from prepare import (
    TIME_BUDGET, EVAL_MATRICES, SYNTHETIC_EVAL_GRIDS, SYNTHETIC_TRAINING_GRIDS,
    NUM_RHS, NUM_SYNTHETIC_MATRICES, FGMRES_RESTART, FGMRES_MAX_ITERS,
    FGMRES_RTOL, FGMRES_TIMEOUT, ILU_REFERENCE, AMG_REFERENCE,
    load_suitesparse_matrix,
)

SEED = 42
NUM_LAYERS = 4
EMBED_DIM = 192
HIDDEN_DIM = 384
POLY_DEGREE = 6
NUM_PROBES = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MATRICES_PER_EPOCH = 16
GRID_SIZES = (16, 24, 32, 48)
NUM_NODE_FEATURES = 3
NUM_EDGE_FEATURES = 2
LOSS_SKIP_THRESHOLD = 50.0
WARMUP_EPOCHS = 20
MIN_LR_RATIO = 0.1

DOMAIN_WEIGHTS = {
    MatrixDomain.DIFFUSION: 0.20,
    MatrixDomain.ELASTICITY: 0.15,
    MatrixDomain.STOKES: 0.10,
    MatrixDomain.DIFFUSION_ADVECTION: 0.15,
    MatrixDomain.VARIABLE_DIFFUSION: 0.10,
    MatrixDomain.SPECTRAL_STRESS: 0.10,
    MatrixDomain.GRAPH_LAPLACIAN: 0.10,
    MatrixDomain.ENHANCED_ADVECTION: 0.10,
}


class MPNNConv(nn.Module):

    def __init__(self, node_dim: int, out_dim: int, edge_feat_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.message_fn = nn.Sequential(
            nn.Linear(2 * node_dim + edge_feat_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_features: torch.Tensor, n: int) -> torch.Tensor:
        rows, cols = edge_index
        msg_input = torch.cat([h[rows], h[cols], edge_features], dim=-1)
        messages = self.message_fn(msg_input)
        out = torch.zeros(n, self.out_dim, dtype=h.dtype, device=h.device)
        out.scatter_add_(0, rows.unsqueeze(-1).expand_as(messages), messages)
        return out


class PolynomialHead(nn.Module):

    def __init__(self, node_dim: int, poly_degree: int):
        super().__init__()
        self.poly_degree = poly_degree
        self.coeff_net = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, poly_degree),
        )
        nn.init.zeros_(self.coeff_net[-1].weight)
        nn.init.zeros_(self.coeff_net[-1].bias)
        with torch.no_grad():
            # Neumann series: all c_k = 1
            self.coeff_net[-1].bias.fill_(1.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.coeff_net(h)


class PolyMPNN(nn.Module):

    def __init__(self, num_layers: int, embed: int, hidden: int,
                 edge_feat_dim: int, poly_degree: int):
        super().__init__()
        self.num_layers = num_layers
        self.embed = embed
        self.hidden = hidden
        self.edge_feat_dim = edge_feat_dim
        self.poly_degree = poly_degree

        self.node_encoder = nn.Sequential(
            nn.Linear(NUM_NODE_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed),
        )

        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(MPNNConv(embed, embed, edge_feat_dim))
            self.skips.append(nn.Linear(embed, embed))
            self.norms.append(nn.LayerNorm(embed))

        self.poly_head = PolynomialHead(embed, poly_degree)

        self.edge_index = None
        self.edge_features = None
        self.node_features = None
        self.D_inv = None
        self.D_inv_A = None
        self.n = None

    def set_matrix(self, A: torch.Tensor):
        if A.layout == torch.sparse_csc:
            A_coo = A.to_sparse_coo().coalesce()
        else:
            A_coo = A.coalesce()

        indices = A_coo.indices()
        values = A_coo.values()
        n = A.shape[0]
        rows, cols = indices

        diag = torch.zeros(n, dtype=values.dtype, device=values.device)
        diag_mask = rows == cols
        diag[rows[diag_mask]] = values[diag_mask]

        if (diag.abs() < 1e-15).any():
            raise ValueError(f"Matrix has {(diag.abs() < 1e-15).sum()} near-zero diagonal entries")

        row_norms = torch.zeros(n, dtype=values.dtype, device=values.device)
        row_norms.scatter_add_(0, rows, values.abs())
        row_norms = row_norms.clamp(min=1e-12)

        gamma = row_norms.max().item()

        self.node_features = torch.stack([
            diag / gamma,
            diag.abs() / row_norms,
            row_norms / gamma,
        ], dim=-1).float()

        diag_at_row = diag[rows].abs()
        self.edge_features = torch.stack([
            values / gamma,
            values.abs() / diag_at_row,
        ], dim=-1).float()

        self.edge_index = indices
        self.n = n
        self.D_inv = 1.0 / diag

        d_inv_values = self.D_inv[rows] * values
        self.D_inv_A = torch.sparse_coo_tensor(
            indices, d_inv_values, (n, n)
        ).coalesce().to_sparse_csc()

    def forward(self) -> torch.Tensor:
        h = self.node_encoder(self.node_features)

        for i in range(self.num_layers):
            h_new = self.convs[i](h, self.edge_index, self.edge_features, self.n)
            h_new = h_new + self.skips[i](h)
            h_new = self.norms[i](h_new)
            h_new = F.relu(h_new)
            h = h_new

        return self.poly_head(h)


class PolynomialPreconditioner:

    def __init__(self, coeffs: torch.Tensor, D_inv_A: torch.Tensor,
                 D_inv: torch.Tensor):
        self.coeffs = coeffs.double()
        self.D_inv_A = D_inv_A
        self.D_inv = D_inv

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        K = self.coeffs.shape[1]

        d_inv_r = self.D_inv * r
        power = d_inv_r  # J^0 * D^{-1} * r = D^{-1} * r
        result = self.coeffs[:, 0] * power

        for k in range(1, K):
            # J * power = power - D^{-1}A @ power
            power = power - self.D_inv_A @ power
            result = result + self.coeffs[:, k] * power

        return result


def poly_frobenius_loss(A: torch.Tensor, coeffs: torch.Tensor,
                        D_inv_A: torch.Tensor, D_inv: torch.Tensor,
                        num_probes: int) -> torch.Tensor:
    n = A.shape[0]
    device = A.device
    K = coeffs.shape[1]

    v = torch.randn(n, num_probes, dtype=torch.float64, device=device)
    Av = A @ v

    D_inv_unsq = D_inv.unsqueeze(-1)
    d_inv_Av = D_inv_unsq * Av

    power = d_inv_Av.float()
    coeffs_0 = coeffs[:, 0:1]
    MAv = coeffs_0 * power

    D_inv_A_f32 = D_inv_A.float()
    for k in range(1, K):
        # J * power = power - D^{-1}A @ power
        power = power - D_inv_A_f32 @ power
        coeffs_k = coeffs[:, k:k+1]
        MAv = MAv + coeffs_k * power

    v_f32 = v.float()
    residual = MAv - v_f32
    per_probe = (residual ** 2).sum(dim=0) / (v_f32 ** 2).sum(dim=0).clamp(min=1e-12)
    return per_probe.mean()


def save_checkpoint(model: PolyMPNN, path: str):
    torch.save({
        "model_type": "PolyMPNN",
        "config": {
            "num_layers": model.num_layers,
            "embed": model.embed,
            "hidden": model.hidden,
            "edge_feat_dim": model.edge_feat_dim,
            "poly_degree": model.poly_degree,
        },
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path: str, device: torch.device) -> PolyMPNN:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = PolyMPNN(
        num_layers=config["num_layers"],
        embed=config["embed"],
        hidden=config["hidden"],
        edge_feat_dim=config["edge_feat_dim"],
        poly_degree=config["poly_degree"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


@torch.no_grad()
def evaluate_poly(model: PolyMPNN, device: torch.device) -> dict:
    model.eval()
    solver = FGMRES(
        restart=FGMRES_RESTART,
        max_iters=FGMRES_MAX_ITERS,
        rtol=FGMRES_RTOL,
        timeout=FGMRES_TIMEOUT,
    )

    torch.manual_seed(99999)
    np.random.seed(99999)

    synthetic_scores = []
    synthetic_jacobi_scores = []
    synthetic_pfn_conv = []
    synthetic_jacobi_conv = []
    synthetic_details = {}

    for gs in SYNTHETIC_EVAL_GRIDS:
        ood_tag = " (OOD)" if gs not in SYNTHETIC_TRAINING_GRIDS else ""
        gen = DiffusionGenerator((gs,), device)
        gs_pfn_iters = []
        gs_jacobi_iters = []
        gs_pfn_conv = []
        gs_jacobi_conv = []

        for _ in range(NUM_SYNTHETIC_MATRICES):
            batch = gen.generate_batch(1, 5)
            A = torch.sparse_coo_tensor(
                batch.indices, batch.values[0], (batch.n, batch.n)
            ).coalesce().to_sparse_csc()
            b = torch.randn(batch.n, dtype=torch.float64, device=device)

            try:
                model.set_matrix(A)
                coeffs = model()
                precond = PolynomialPreconditioner(coeffs, model.D_inv_A, model.D_inv)
                result = solver.solve(A, b, M=precond, progress_bar=False)
                gs_pfn_iters.append(result.iterations / FGMRES_MAX_ITERS)
                gs_pfn_conv.append(result.converged)
            except Exception:
                gs_pfn_iters.append(1.0)
                gs_pfn_conv.append(False)

            try:
                jacobi = Jacobi(A)
                jac_result = solver.solve(A, b, M=jacobi, progress_bar=False)
                gs_jacobi_iters.append(jac_result.iterations / FGMRES_MAX_ITERS)
                gs_jacobi_conv.append(jac_result.converged)
            except Exception:
                gs_jacobi_iters.append(1.0)
                gs_jacobi_conv.append(False)

        pfn_mean = sum(gs_pfn_iters) / len(gs_pfn_iters)
        synthetic_scores.extend(gs_pfn_iters)
        synthetic_jacobi_scores.extend(gs_jacobi_iters)
        synthetic_pfn_conv.extend(gs_pfn_conv)
        synthetic_jacobi_conv.extend(gs_jacobi_conv)

        synthetic_details[f"{gs}x{gs}{ood_tag}"] = {
            "pfn_mean_norm_iter": pfn_mean,
            "jacobi_mean_norm_iter": sum(gs_jacobi_iters) / len(gs_jacobi_iters),
            "pfn_conv_rate": sum(gs_pfn_conv) / len(gs_pfn_iters),
        }

    ss_scores = []
    ss_jacobi_scores = []
    ss_pfn_conv = []
    ss_jacobi_conv = []
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
        mat_pfn_conv = []
        mat_jac_conv = []

        try:
            model.set_matrix(A)
            coeffs = model()
            precond = PolynomialPreconditioner(coeffs, model.D_inv_A, model.D_inv)
        except Exception:
            for _ in range(NUM_RHS):
                mat_pfn_iters.append(1.0)
                mat_pfn_conv.append(False)
                b = torch.randn(n, dtype=torch.float64, device=device)
                try:
                    jacobi = Jacobi(A)
                    jac_result = solver.solve(A, b, M=jacobi, progress_bar=False)
                    mat_jac_iters.append(jac_result.iterations / FGMRES_MAX_ITERS)
                    mat_jac_conv.append(jac_result.converged)
                except Exception:
                    mat_jac_iters.append(1.0)
                    mat_jac_conv.append(False)

            ss_scores.extend(mat_pfn_iters)
            ss_jacobi_scores.extend(mat_jac_iters)
            ss_pfn_conv.extend(mat_pfn_conv)
            ss_jacobi_conv.extend(mat_jac_conv)
            ss_details[name] = {
                "pfn_mean_norm_iter": 1.0,
                "jacobi_mean_norm_iter": sum(mat_jac_iters) / len(mat_jac_iters),
                "pfn_conv_rate": 0.0,
                "n": n,
            }
            continue

        for _ in range(NUM_RHS):
            b = torch.randn(n, dtype=torch.float64, device=device)

            try:
                result = solver.solve(A, b, M=precond, progress_bar=False)
                mat_pfn_iters.append(result.iterations / FGMRES_MAX_ITERS)
                mat_pfn_conv.append(result.converged)
            except Exception:
                mat_pfn_iters.append(1.0)
                mat_pfn_conv.append(False)

            try:
                jacobi = Jacobi(A)
                jac_result = solver.solve(A, b, M=jacobi, progress_bar=False)
                mat_jac_iters.append(jac_result.iterations / FGMRES_MAX_ITERS)
                mat_jac_conv.append(jac_result.converged)
            except Exception:
                mat_jac_iters.append(1.0)
                mat_jac_conv.append(False)

        pfn_mean = sum(mat_pfn_iters) / len(mat_pfn_iters)
        ss_scores.extend(mat_pfn_iters)
        ss_jacobi_scores.extend(mat_jac_iters)
        ss_pfn_conv.extend(mat_pfn_conv)
        ss_jacobi_conv.extend(mat_jac_conv)
        ss_details[name] = {
            "pfn_mean_norm_iter": pfn_mean,
            "jacobi_mean_norm_iter": sum(mat_jac_iters) / len(mat_jac_iters),
            "pfn_conv_rate": sum(mat_pfn_conv) / len(mat_pfn_iters),
            "n": n,
        }

    synth_score = sum(synthetic_scores) / len(synthetic_scores) if synthetic_scores else 1.0
    ss_score = sum(ss_scores) / len(ss_scores) if ss_scores else 1.0
    synth_jacobi = sum(synthetic_jacobi_scores) / len(synthetic_jacobi_scores) if synthetic_jacobi_scores else 1.0
    ss_jacobi = sum(ss_jacobi_scores) / len(ss_jacobi_scores) if ss_jacobi_scores else 1.0

    combined_score = 0.3 * synth_score + 0.7 * ss_score

    synth_conv = sum(synthetic_pfn_conv) / len(synthetic_pfn_conv) * 100 if synthetic_pfn_conv else 0.0
    ss_conv = sum(ss_pfn_conv) / len(ss_pfn_conv) * 100 if ss_pfn_conv else 0.0

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
        "ilu_reference": ILU_REFERENCE,
        "amg_reference": AMG_REFERENCE,
    }


t_start = time.time()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

config = GeneratorConfig(grid_sizes=GRID_SIZES)
full_registry = build_training_registry(config, device)

selected_generators = {
    domain: full_registry.generators[domain]
    for domain in DOMAIN_WEIGHTS
    if domain in full_registry.generators
}
registry = MatrixGeneratorRegistry(selected_generators)

print(f"Training domains ({len(selected_generators)}):")
for domain, weight in DOMAIN_WEIGHTS.items():
    if domain in selected_generators:
        print(f"  {domain.value}: {weight:.0%}")

model = PolyMPNN(
    num_layers=NUM_LAYERS,
    embed=EMBED_DIM,
    hidden=HIDDEN_DIM,
    edge_feat_dim=NUM_EDGE_FEATURES,
    poly_degree=POLY_DEGREE,
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: PolyMPNN ({num_params:,} params)")
print(f"  layers={NUM_LAYERS}, embed={EMBED_DIM}, hidden={HIDDEN_DIM}, poly_degree={POLY_DEGREE}")

dataset = OnlineMatrixDataset(registry, 1, domain_weights=DOMAIN_WEIGHTS)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

estimated_epochs = int(TIME_BUDGET / 0.45)


def lr_lambda(epoch: int) -> float:
    if epoch < WARMUP_EPOCHS:
        return epoch / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, estimated_epochs - WARMUP_EPOCHS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print(f"\nTime budget: {TIME_BUDGET}s")
print(f"Probes per matrix: {NUM_PROBES}, Matrices/epoch: {MATRICES_PER_EPOCH}")
print(f"Grid sizes: {GRID_SIZES}")
print(f"Loss: stochastic Frobenius ||MAv - v||^2 (polynomial)")
print(f"Poly degree: {POLY_DEGREE}, loss skip > {LOSS_SKIP_THRESHOLD}")
print(f"LR: {LEARNING_RATE} with {WARMUP_EPOCHS}-epoch warmup + cosine decay (min {MIN_LR_RATIO})")
print()

CHECKPOINT_PATH = "best_model.pt"
best_loss = float("inf")
total_training_time = 0.0
epoch = 0
data_iter = iter(dataset)
smooth_loss = 0.0
skipped_count = 0

while True:
    t0 = time.time()
    model.train()
    epoch_loss = 0.0
    valid_count = 0

    for _ in range(MATRICES_PER_EPOCH):
        data = next(data_iter)
        A = torch.sparse_coo_tensor(
            data.indices, data.values[0], (data.n, data.n)
        ).coalesce().to_sparse_csc()

        model.set_matrix(A)
        coeffs = model()

        loss = poly_frobenius_loss(A, coeffs, model.D_inv_A, model.D_inv, NUM_PROBES)
        loss_val = loss.item()

        if not math.isfinite(loss_val) or loss_val > LOSS_SKIP_THRESHOLD:
            skipped_count += 1
            continue

        epoch_loss += loss_val
        valid_count += 1

        scaled_loss = loss / MATRICES_PER_EPOCH
        scaled_loss.backward()

    if valid_count > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    avg_loss = epoch_loss / max(valid_count, 1)

    if avg_loss < best_loss and valid_count > 0:
        best_loss = avg_loss
        save_checkpoint(model, CHECKPOINT_PATH)

    t1 = time.time()
    dt = t1 - t0

    if epoch > 5:
        total_training_time += dt

    ema_beta = 0.95
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * avg_loss
    debiased = smooth_loss / (1 - ema_beta ** (epoch + 1))

    remaining = max(0, TIME_BUDGET - total_training_time)
    current_lr = scheduler.get_last_lr()[0]
    print(f"\repoch {epoch:04d} | loss: {debiased:.4e} | best: {best_loss:.4e} | lr: {current_lr:.1e} | skip: {skipped_count} | dt: {dt*1000:.0f}ms | {remaining:.0f}s    ", end="", flush=True)

    epoch += 1

    if epoch > 5 and total_training_time >= TIME_BUDGET:
        break

print()
print(f"\nTraining done: {epoch} epochs in {total_training_time:.1f}s")
print(f"Best loss: {best_loss:.4e}")

print("\nEvaluating...")
t_eval_start = time.time()
eval_model = load_checkpoint(CHECKPOINT_PATH, device)
results = evaluate_poly(eval_model, device)
t_eval_end = time.time()

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0

print()
print("---")
print(f"score:             {results['score']:.6f}")
print(f"synthetic_score:   {results['synthetic_score']:.6f}")
print(f"suitesparse_score: {results['suitesparse_score']:.6f}")
print(f"synthetic_conv:    {results['synthetic_conv_pct']:.1f}%")
print(f"suitesparse_conv:  {results['suitesparse_conv_pct']:.1f}%")
print(f"synth_vs_jacobi:   {results['synthetic_score']:.4f} vs {results['synthetic_jacobi']:.4f}")
print(f"ss_vs_jacobi:      {results['suitesparse_score']:.4f} vs {results['suitesparse_jacobi']:.4f}")
print(f"training_seconds:  {total_training_time:.1f}")
print(f"eval_seconds:      {t_eval_end - t_eval_start:.1f}")
print(f"total_seconds:     {t_end - t_start:.1f}")
print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
print(f"num_params:        {num_params}")
print(f"num_epochs:        {epoch}")
print(f"best_loss:         {best_loss:.6e}")
print(f"domains:           {len(selected_generators)}")

print("\nSuiteSparse details:")
ilu_ref = results.get("ilu_reference", {})
amg_ref = results.get("amg_reference", {})
for name, detail in results["suitesparse_details"].items():
    conv_pct = detail["pfn_conv_rate"] * 100
    pfn_iter = detail["pfn_mean_norm_iter"]
    jac_iter = detail["jacobi_mean_norm_iter"]
    ilu_iter = ilu_ref.get(name, {}).get("norm_iter", -1)
    amg_iter = amg_ref.get(name, {}).get("norm_iter", -1)
    n = detail["n"]
    status = "OK" if conv_pct > 50 else "FAIL"
    print(f"  {name:<12s} (n={n:>5d}): {status:<4s} pfn={pfn_iter:.3f} jac={jac_iter:.3f} ilu={ilu_iter:.3f} amg={amg_iter:.3f} conv={conv_pct:.0f}%")

print("\nSynthetic details:")
for grid, detail in results["synthetic_details"].items():
    conv_pct = detail["pfn_conv_rate"] * 100
    pfn_iter = detail["pfn_mean_norm_iter"]
    jac_iter = detail["jacobi_mean_norm_iter"]
    print(f"  {grid:<16s}: pfn={pfn_iter:.3f} jac={jac_iter:.3f} conv={conv_pct:.0f}%")
