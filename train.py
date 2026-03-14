"""
MatrixPFN autoresearch training script.
Run 19: Additive Hybrid — Polynomial + SPAI correction.
M*r = P_poly(r) + D^{-1}*G*(D^{-1}*r)
Polynomial provides multi-hop, SPAI adds independent local corrections.
No coupling — simpler optimization landscape.

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
BILINEAR_RANK = 64
G_SCALE = 1.0
NUM_PROBES = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MATRICES_PER_EPOCH = 16
GRID_SIZES = (16, 24, 32, 48)
TRAINING_TIME = 300
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

    def __init__(self, node_dim, out_dim, edge_feat_dim):
        super().__init__()
        self.out_dim = out_dim
        self.message_fn = nn.Sequential(
            nn.Linear(2 * node_dim + edge_feat_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, h, edge_index, edge_features, n):
        rows, cols = edge_index
        msg_input = torch.cat([h[rows], h[cols], edge_features], dim=-1)
        messages = self.message_fn(msg_input)
        out = torch.zeros(n, self.out_dim, dtype=h.dtype, device=h.device)
        out.scatter_add_(0, rows.unsqueeze(-1).expand_as(messages), messages)
        return out


class PolynomialHead(nn.Module):

    def __init__(self, node_dim, poly_degree):
        super().__init__()
        self.poly_degree = poly_degree
        self.net = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, poly_degree),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[0] = 1.0

    def forward(self, h):
        return self.net(h)


class BilinearEdgeHead(nn.Module):

    def __init__(self, node_dim, edge_feat_dim, rank):
        super().__init__()
        self.left = nn.Linear(node_dim, rank, bias=False)
        self.right = nn.Linear(node_dim, rank, bias=False)
        self.edge_linear = nn.Linear(edge_feat_dim, 1, bias=True)

        nn.init.xavier_uniform_(self.left.weight, gain=0.01)
        nn.init.xavier_uniform_(self.right.weight, gain=0.01)
        nn.init.zeros_(self.edge_linear.weight)
        nn.init.zeros_(self.edge_linear.bias)

    def forward(self, h, edge_index, edge_features):
        src, dst = edge_index
        bilinear = (self.left(h[src]) * self.right(h[dst])).sum(dim=-1)
        edge_bias = self.edge_linear(edge_features).squeeze(-1)
        return G_SCALE * torch.tanh(bilinear + edge_bias)


class HybridMPNN(nn.Module):

    def __init__(self, num_layers, embed, hidden, poly_degree, bilinear_rank):
        super().__init__()
        self.num_layers = num_layers
        self.embed = embed
        self.hidden = hidden
        self.poly_degree = poly_degree
        self.bilinear_rank = bilinear_rank

        self.node_encoder = nn.Sequential(
            nn.Linear(NUM_NODE_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed),
        )

        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(MPNNConv(embed, embed, NUM_EDGE_FEATURES))
            self.skips.append(nn.Linear(embed, embed))
            self.norms.append(nn.LayerNorm(embed))

        self.poly_head = PolynomialHead(embed, poly_degree)
        self.edge_head = BilinearEdgeHead(embed, NUM_EDGE_FEATURES, bilinear_rank)

        self.edge_index = None
        self.edge_features = None
        self.node_features = None
        self.D_inv = None
        self.D_inv_A = None
        self.n = None

    def set_matrix(self, A):
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
            diag / gamma, diag.abs() / row_norms, row_norms / gamma,
        ], dim=-1).float()

        diag_at_row = diag[rows].abs()
        self.edge_features = torch.stack([
            values / gamma, values.abs() / diag_at_row,
        ], dim=-1).float()

        self.edge_index = indices
        self.n = n
        self.D_inv = 1.0 / diag

        d_inv_values = self.D_inv[rows] * values
        self.D_inv_A = torch.sparse_coo_tensor(
            indices, d_inv_values, (n, n)
        ).coalesce().to_sparse_csc()

    def forward(self):
        h = self.node_encoder(self.node_features)

        for i in range(self.num_layers):
            h_new = self.convs[i](h, self.edge_index, self.edge_features, self.n)
            h_new = h_new + self.skips[i](h)
            h_new = self.norms[i](h_new)
            h_new = F.relu(h_new)
            h = h_new

        coeffs = self.poly_head(h)
        g_values = self.edge_head(h, self.edge_index, self.edge_features)
        return coeffs, g_values


class HybridPreconditioner:

    def __init__(self, coeffs, g_values, edge_index, D_inv, D_inv_A, n):
        self.coeffs = coeffs.double()
        self.D_inv = D_inv
        self.D_inv_A = D_inv_A

        device = D_inv.device
        diag_idx = torch.arange(n, device=device)
        rows, cols = edge_index
        all_rows = torch.cat([diag_idx, rows])
        all_cols = torch.cat([diag_idx, cols])
        all_vals = torch.cat([
            torch.ones(n, dtype=torch.float64, device=device),
            g_values.double(),
        ])
        self.IpG = torch.sparse_coo_tensor(
            torch.stack([all_rows, all_cols]), all_vals, (n, n)
        ).coalesce().to_sparse_csc()

    def apply(self, r):
        K = self.coeffs.shape[1]
        d_inv_r = self.D_inv * r

        power = d_inv_r
        poly_result = self.coeffs[:, 0] * power
        for k in range(1, K):
            power = self.D_inv_A @ power
            poly_result = poly_result + self.coeffs[:, k] * power

        spai_result = self.D_inv * (self.IpG @ d_inv_r) - d_inv_r

        return poly_result + spai_result


def hybrid_frobenius_loss(A, coeffs, g_values, edge_index,
                          D_inv_A, D_inv, num_probes):
    n = A.shape[0]
    device = A.device
    K = coeffs.shape[1]
    rows, cols = edge_index

    v = torch.randn(n, num_probes, dtype=torch.float64, device=device)
    Av = A @ v

    D_inv_unsq = D_inv.unsqueeze(-1)
    d_inv_Av = D_inv_unsq * Av

    power = d_inv_Av.float()
    z = coeffs[:, 0:1] * power

    D_inv_A_f32 = D_inv_A.float()
    for k in range(1, K):
        power = D_inv_A_f32 @ power
        z = z + coeffs[:, k:k+1] * power

    Av_at_cols = d_inv_Av.float()[cols]
    weighted = g_values.unsqueeze(-1) * Av_at_cols
    GAv_dinv = torch.zeros(n, num_probes, dtype=torch.float32, device=device)
    GAv_dinv.scatter_add_(0, rows.unsqueeze(-1).expand_as(weighted), weighted)

    D_inv_f32 = D_inv.float().unsqueeze(-1)
    spai_correction = D_inv_f32 * GAv_dinv

    MAv = z + spai_correction

    v_f32 = v.float()
    residual = MAv - v_f32
    per_probe = (residual ** 2).sum(dim=0) / (v_f32 ** 2).sum(dim=0).clamp(min=1e-12)
    return per_probe.mean()


def save_checkpoint(model, path):
    torch.save({
        "model_type": "HybridMPNN",
        "config": {
            "num_layers": model.num_layers,
            "embed": model.embed,
            "hidden": model.hidden,
            "poly_degree": model.poly_degree,
            "bilinear_rank": model.bilinear_rank,
        },
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = HybridMPNN(
        num_layers=config["num_layers"],
        embed=config["embed"],
        hidden=config["hidden"],
        poly_degree=config["poly_degree"],
        bilinear_rank=config["bilinear_rank"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


@torch.no_grad()
def evaluate_hybrid(model, device):
    model.eval()
    solver = FGMRES(restart=FGMRES_RESTART, max_iters=FGMRES_MAX_ITERS,
                    rtol=FGMRES_RTOL, timeout=FGMRES_TIMEOUT)

    torch.manual_seed(99999)
    np.random.seed(99999)

    synthetic_scores, synthetic_jacobi_scores = [], []
    synthetic_pfn_conv, synthetic_jacobi_conv = [], []
    synthetic_details = {}

    for gs in SYNTHETIC_EVAL_GRIDS:
        ood_tag = " (OOD)" if gs not in SYNTHETIC_TRAINING_GRIDS else ""
        gen = DiffusionGenerator((gs,), device)
        gs_pfn, gs_jac, gs_pc, gs_jc = [], [], [], []

        for _ in range(NUM_SYNTHETIC_MATRICES):
            batch = gen.generate_batch(1, 5)
            A = torch.sparse_coo_tensor(batch.indices, batch.values[0],
                                        (batch.n, batch.n)).coalesce().to_sparse_csc()
            b = torch.randn(batch.n, dtype=torch.float64, device=device)

            try:
                model.set_matrix(A)
                coeffs, g_values = model()
                precond = HybridPreconditioner(coeffs, g_values, model.edge_index,
                                              model.D_inv, model.D_inv_A, model.n)
                result = solver.solve(A, b, M=precond, progress_bar=False)
                gs_pfn.append(result.iterations / FGMRES_MAX_ITERS)
                gs_pc.append(result.converged)
            except Exception:
                gs_pfn.append(1.0)
                gs_pc.append(False)

            try:
                jacobi = Jacobi(A)
                jr = solver.solve(A, b, M=jacobi, progress_bar=False)
                gs_jac.append(jr.iterations / FGMRES_MAX_ITERS)
                gs_jc.append(jr.converged)
            except Exception:
                gs_jac.append(1.0)
                gs_jc.append(False)

        synthetic_scores.extend(gs_pfn)
        synthetic_jacobi_scores.extend(gs_jac)
        synthetic_pfn_conv.extend(gs_pc)
        synthetic_jacobi_conv.extend(gs_jc)
        synthetic_details[f"{gs}x{gs}{ood_tag}"] = {
            "pfn_mean_norm_iter": sum(gs_pfn) / len(gs_pfn),
            "jacobi_mean_norm_iter": sum(gs_jac) / len(gs_jac),
            "pfn_conv_rate": sum(gs_pc) / len(gs_pfn),
        }

    ss_scores, ss_jacobi_scores = [], []
    ss_pfn_conv, ss_jacobi_conv = [], []
    ss_details = {}

    for group, name in EVAL_MATRICES:
        try:
            A = load_suitesparse_matrix(name, device)
        except FileNotFoundError:
            print(f"  SKIP {name}: not downloaded")
            continue

        n = A.shape[0]
        mp, mj, mpc, mjc = [], [], [], []

        try:
            model.set_matrix(A)
            coeffs, g_values = model()
            precond = HybridPreconditioner(coeffs, g_values, model.edge_index,
                                          model.D_inv, model.D_inv_A, model.n)
        except Exception:
            for _ in range(NUM_RHS):
                mp.append(1.0); mpc.append(False)
                b = torch.randn(n, dtype=torch.float64, device=device)
                try:
                    jr = solver.solve(A, b, M=Jacobi(A), progress_bar=False)
                    mj.append(jr.iterations / FGMRES_MAX_ITERS); mjc.append(jr.converged)
                except Exception:
                    mj.append(1.0); mjc.append(False)
            ss_scores.extend(mp); ss_jacobi_scores.extend(mj)
            ss_pfn_conv.extend(mpc); ss_jacobi_conv.extend(mjc)
            ss_details[name] = {"pfn_mean_norm_iter": 1.0, "jacobi_mean_norm_iter": sum(mj)/len(mj),
                                "pfn_conv_rate": 0.0, "n": n}
            continue

        for _ in range(NUM_RHS):
            b = torch.randn(n, dtype=torch.float64, device=device)
            try:
                r = solver.solve(A, b, M=precond, progress_bar=False)
                mp.append(r.iterations / FGMRES_MAX_ITERS); mpc.append(r.converged)
            except Exception:
                mp.append(1.0); mpc.append(False)
            try:
                jr = solver.solve(A, b, M=Jacobi(A), progress_bar=False)
                mj.append(jr.iterations / FGMRES_MAX_ITERS); mjc.append(jr.converged)
            except Exception:
                mj.append(1.0); mjc.append(False)

        ss_scores.extend(mp); ss_jacobi_scores.extend(mj)
        ss_pfn_conv.extend(mpc); ss_jacobi_conv.extend(mjc)
        ss_details[name] = {"pfn_mean_norm_iter": sum(mp)/len(mp),
                            "jacobi_mean_norm_iter": sum(mj)/len(mj),
                            "pfn_conv_rate": sum(mpc)/len(mp), "n": n}

    synth_s = sum(synthetic_scores)/len(synthetic_scores) if synthetic_scores else 1.0
    ss_s = sum(ss_scores)/len(ss_scores) if ss_scores else 1.0
    synth_j = sum(synthetic_jacobi_scores)/len(synthetic_jacobi_scores) if synthetic_jacobi_scores else 1.0
    ss_j = sum(ss_jacobi_scores)/len(ss_jacobi_scores) if ss_jacobi_scores else 1.0

    return {
        "score": 0.3 * synth_s + 0.7 * ss_s,
        "synthetic_score": synth_s, "suitesparse_score": ss_s,
        "synthetic_jacobi": synth_j, "suitesparse_jacobi": ss_j,
        "synthetic_conv_pct": sum(synthetic_pfn_conv)/len(synthetic_pfn_conv)*100 if synthetic_pfn_conv else 0,
        "suitesparse_conv_pct": sum(ss_pfn_conv)/len(ss_pfn_conv)*100 if ss_pfn_conv else 0,
        "synthetic_details": synthetic_details, "suitesparse_details": ss_details,
        "ilu_reference": ILU_REFERENCE, "amg_reference": AMG_REFERENCE,
    }


t_start = time.time()
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

config = GeneratorConfig(grid_sizes=GRID_SIZES)
full_registry = build_training_registry(config, device)
selected_generators = {d: full_registry.generators[d] for d in DOMAIN_WEIGHTS if d in full_registry.generators}
registry = MatrixGeneratorRegistry(selected_generators)

print(f"Training domains ({len(selected_generators)}):")
for domain, weight in DOMAIN_WEIGHTS.items():
    if domain in selected_generators:
        print(f"  {domain.value}: {weight:.0%}")

model = HybridMPNN(NUM_LAYERS, EMBED_DIM, HIDDEN_DIM, POLY_DEGREE, BILINEAR_RANK).to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: HybridMPNN ({num_params:,} params)")
print(f"  layers={NUM_LAYERS}, embed={EMBED_DIM}, hidden={HIDDEN_DIM}")
print(f"  poly_degree={POLY_DEGREE}, bilinear_rank={BILINEAR_RANK}, G_SCALE={G_SCALE}")

dataset = OnlineMatrixDataset(registry, 1, domain_weights=DOMAIN_WEIGHTS)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
estimated_epochs = int(TRAINING_TIME / 0.45)

def lr_lambda(ep):
    if ep < WARMUP_EPOCHS: return ep / WARMUP_EPOCHS
    progress = (ep - WARMUP_EPOCHS) / max(1, estimated_epochs - WARMUP_EPOCHS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print(f"\nTime budget: {TRAINING_TIME}s")
print(f"Probes: {NUM_PROBES}, Matrices/epoch: {MATRICES_PER_EPOCH}, Grids: {GRID_SIZES}")
print(f"Loss: hybrid Frobenius ||D^-1(I+G)*P_poly*Av - v||^2")
print(f"Skip > {LOSS_SKIP_THRESHOLD}, LR: {LEARNING_RATE} warmup={WARMUP_EPOCHS} cosine(min={MIN_LR_RATIO})")
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
        A = torch.sparse_coo_tensor(data.indices, data.values[0],
                                    (data.n, data.n)).coalesce().to_sparse_csc()
        model.set_matrix(A)
        coeffs, g_values = model()

        loss = hybrid_frobenius_loss(A, coeffs, g_values, model.edge_index,
                                    model.D_inv_A, model.D_inv, NUM_PROBES)
        loss_val = loss.item()

        if not math.isfinite(loss_val) or loss_val > LOSS_SKIP_THRESHOLD:
            skipped_count += 1
            continue

        epoch_loss += loss_val
        valid_count += 1
        (loss / MATRICES_PER_EPOCH).backward()

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
    if epoch > 5: total_training_time += dt

    ema_beta = 0.95
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * avg_loss
    debiased = smooth_loss / (1 - ema_beta ** (epoch + 1))

    remaining = max(0, TRAINING_TIME - total_training_time)
    current_lr = scheduler.get_last_lr()[0]
    print(f"\repoch {epoch:04d} | loss: {debiased:.4e} | best: {best_loss:.4e} | lr: {current_lr:.1e} | skip: {skipped_count} | dt: {dt*1000:.0f}ms | {remaining:.0f}s    ", end="", flush=True)

    epoch += 1
    if epoch > 5 and total_training_time >= TRAINING_TIME: break

print()
print(f"\nTraining done: {epoch} epochs in {total_training_time:.1f}s")
print(f"Best loss: {best_loss:.4e}")

print("\nEvaluating...")
t_eval_start = time.time()
eval_model = load_checkpoint(CHECKPOINT_PATH, device)
results = evaluate_hybrid(eval_model, device)
t_eval_end = time.time()
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0

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
