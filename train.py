"""
MatrixPFN autoresearch training script.
This is THE ONLY FILE the agent edits.

Usage: uv run train.py
"""

import time
import random

import numpy as np
import torch

from matrixpfn.nn.context_resgcn import ContextResGCN
from matrixpfn.precond.matrix_pfn import MatrixPFN, TrainingConfig
from matrixpfn.generator.base import GeneratorConfig, MatrixDomain
from matrixpfn.generator.registry import build_training_registry, MatrixGeneratorRegistry
from matrixpfn.generator.online import OnlineMatrixDataset

from prepare import TIME_BUDGET, evaluate_score

SEED = 42
NUM_LAYERS = 8
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_CONTEXT_PAIRS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAINING_BATCH_SIZE = 16
MATRICES_PER_EPOCH = 16
GRID_SIZES = (16, 24, 32)

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

model = ContextResGCN(
    num_layers=NUM_LAYERS,
    embed=EMBED_DIM,
    hidden=HIDDEN_DIM,
    drop_rate=0.0,
    num_context_pairs=NUM_CONTEXT_PAIRS,
    scale_input=True,
    dtype=torch.float32,
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: ContextResGCN ({num_params:,} params)")
print(f"  layers={NUM_LAYERS}, embed={EMBED_DIM}, hidden={HIDDEN_DIM}, context={NUM_CONTEXT_PAIRS}")

pfn = MatrixPFN(model, model_device=device)
dataset = OnlineMatrixDataset(registry, NUM_CONTEXT_PAIRS, domain_weights=DOMAIN_WEIGHTS)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print(f"\nTime budget: {TIME_BUDGET}s")
print(f"Batch size: {TRAINING_BATCH_SIZE}, Matrices/epoch: {MATRICES_PER_EPOCH}")
print(f"Grid sizes: {GRID_SIZES}")
print()

CHECKPOINT_PATH = "best_model.pt"
best_loss = float("inf")
total_training_time = 0.0
epoch = 0
data_iter = iter(dataset)

t_start_training = time.time()
smooth_loss = 0.0

while True:
    t0 = time.time()
    model.train()
    epoch_loss = 0.0

    for _ in range(MATRICES_PER_EPOCH):
        data = next(data_iter)
        A = torch.sparse_coo_tensor(
            data.indices, data.values[0], (data.n, data.n)
        ).coalesce().to_sparse_csc()
        diag_A = data.diagonals[0]
        context_pairs = data.context_pairs[0]

        n = A.shape[0]
        x = torch.randn(n, TRAINING_BATCH_SIZE, dtype=torch.float64, device=device)
        b = A @ x

        gamma = model.set_matrix(A)
        b_input = b.to(device).to(torch.float32)
        diag_input = (diag_A / gamma).to(device).to(torch.float32)
        context_input = context_pairs.to(device).to(torch.float32)

        x_pred = model(b_input, diag_input, context_input)
        x_target = x.to(device).to(torch.float32)

        loss = torch.nn.functional.l1_loss(x_pred, x_target)
        epoch_loss += loss.item()

        scaled_loss = loss / MATRICES_PER_EPOCH
        scaled_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    avg_loss = epoch_loss / MATRICES_PER_EPOCH

    if avg_loss < best_loss:
        best_loss = avg_loss
        pfn.save_pretrained(CHECKPOINT_PATH)

    t1 = time.time()
    dt = t1 - t0

    if epoch > 5:
        total_training_time += dt

    ema_beta = 0.9
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * avg_loss
    debiased = smooth_loss / (1 - ema_beta ** (epoch + 1))

    remaining = max(0, TIME_BUDGET - total_training_time)
    print(f"\repoch {epoch:04d} | loss: {debiased:.4e} | best: {best_loss:.4e} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

    epoch += 1

    if epoch > 5 and total_training_time >= TIME_BUDGET:
        break

print()
print(f"\nTraining done: {epoch} epochs in {total_training_time:.1f}s")
print(f"Best loss: {best_loss:.4e}")

print("\nEvaluating...")
t_eval_start = time.time()
results = evaluate_score(CHECKPOINT_PATH, device)
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
for name, detail in results["suitesparse_details"].items():
    conv_pct = detail["pfn_conv_rate"] * 100
    pfn_iter = detail["pfn_mean_norm_iter"]
    jac_iter = detail["jacobi_mean_norm_iter"]
    n = detail["n"]
    status = "OK" if conv_pct > 50 else "FAIL"
    print(f"  {name:<12s} (n={n:>5d}): {status:<4s} pfn={pfn_iter:.3f} jac={jac_iter:.3f} conv={conv_pct:.0f}%")

print("\nSynthetic details:")
for grid, detail in results["synthetic_details"].items():
    conv_pct = detail["pfn_conv_rate"] * 100
    pfn_iter = detail["pfn_mean_norm_iter"]
    jac_iter = detail["jacobi_mean_norm_iter"]
    print(f"  {grid:<16s}: pfn={pfn_iter:.3f} jac={jac_iter:.3f} conv={conv_pct:.0f}%")
