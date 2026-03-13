#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"

RESULTS_FILE="results.tsv"
LOGS_DIR="logs"
mkdir -p "$LOGS_DIR"

if [ ! -f "$RESULTS_FILE" ]; then
    printf "commit\tscore\tss_conv\tmemory_gb\tstatus\tdescription\n" > "$RESULTS_FILE"
fi

best_score="999.0"
run_counter=0

already_tried() {
    local desc="$1"
    grep -qF "$desc" "$RESULTS_FILE" 2>/dev/null
}

run_experiment() {
    local desc="$1"
    run_counter=$((run_counter + 1))

    if already_tried "$desc"; then
        echo "SKIP: '$desc' already in results.tsv"
        return 0
    fi

    git add train.py
    local commit_hash
    commit_hash=$(git rev-parse --short HEAD 2>/dev/null || echo "uncommit")

    if git diff --cached --quiet; then
        echo "No changes to commit, running as-is..."
    else
        git commit -m "$desc"
        commit_hash=$(git rev-parse --short HEAD)
    fi

    local log_file="${LOGS_DIR}/run_$(printf '%03d' $run_counter)_${commit_hash}.log"

    echo "=== Run #${run_counter}: $desc (commit: $commit_hash) ==="
    echo "=== Log: $log_file ==="

    timeout 900 uv run train.py > "$log_file" 2>&1 || true

    cp "$log_file" run.log

    local score ss_conv peak_vram status
    score=$(grep "^score:" "$log_file" | awk '{print $2}' || echo "")
    ss_conv=$(grep "^suitesparse_conv:" "$log_file" | awk '{print $2}' | tr -d '%' || echo "")
    peak_vram=$(grep "^peak_vram_mb:" "$log_file" | awk '{print $2}' || echo "")

    if [ -z "$score" ]; then
        echo "CRASH — no score found"
        tail -30 "$log_file"
        printf "%s\t0.000000\t0.0\t0.0\tcrash\t%s\n" "$commit_hash" "$desc" >> "$RESULTS_FILE"
        git reset --hard HEAD~1 2>/dev/null || true
        return 1
    fi

    local mem_gb
    mem_gb=$(echo "$peak_vram" | awk '{printf "%.1f", $1/1024}')

    echo "Score: $score | SS conv: $ss_conv | VRAM: ${mem_gb}GB"

    local improved
    improved=$(echo "$score $best_score" | awk '{print ($1 < $2) ? "yes" : "no"}')

    if [ "$improved" = "yes" ]; then
        status="keep"
        best_score="$score"
        echo "IMPROVED! New best: $best_score"
        cp best_model.pt "best_model_${commit_hash}.pt" 2>/dev/null || true
    else
        status="discard"
        echo "No improvement ($score >= $best_score), reverting..."
        git reset --hard HEAD~1 2>/dev/null || true
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$commit_hash" "$score" "$ss_conv" "$mem_gb" "$status" "$desc" >> "$RESULTS_FILE"

    echo "---"
    cat "$RESULTS_FILE"
    echo "==="
}

apply_experiment() {
    local name="$1"
    local train_file="train.py"

    git checkout HEAD -- train.py 2>/dev/null || true

    case "$name" in
        phase2-add-sbm)
            sed -i 's/MatrixDomain.DIFFUSION: 0.20/MatrixDomain.DIFFUSION: 0.15/' "$train_file"
            sed -i '/ENHANCED_ADVECTION: 0.10,/a\    MatrixDomain.SBM: 0.08,' "$train_file"
            sed -i 's/ENHANCED_ADVECTION: 0.10/ENHANCED_ADVECTION: 0.07/' "$train_file"
            ;;
        phase2-add-random-sparse)
            sed -i 's/MatrixDomain.DIFFUSION: 0.20/MatrixDomain.DIFFUSION: 0.15/' "$train_file"
            sed -i '/ENHANCED_ADVECTION: 0.10,/a\    MatrixDomain.RANDOM_SPARSE: 0.08,' "$train_file"
            sed -i 's/ENHANCED_ADVECTION: 0.10/ENHANCED_ADVECTION: 0.07/' "$train_file"
            ;;
        phase2-boost-advection)
            sed -i 's/DIFFUSION_ADVECTION: 0.15/DIFFUSION_ADVECTION: 0.25/' "$train_file"
            sed -i 's/VARIABLE_DIFFUSION: 0.10/VARIABLE_DIFFUSION: 0.15/' "$train_file"
            sed -i 's/DIFFUSION: 0.20/DIFFUSION: 0.10/' "$train_file"
            sed -i 's/GRAPH_LAPLACIAN: 0.10/GRAPH_LAPLACIAN: 0.05/' "$train_file"
            ;;
        phase3-smaller-model)
            sed -i 's/EMBED_DIM = 128/EMBED_DIM = 64/' "$train_file"
            sed -i 's/HIDDEN_DIM = 256/HIDDEN_DIM = 128/' "$train_file"
            ;;
        phase3-larger-model)
            sed -i 's/EMBED_DIM = 128/EMBED_DIM = 256/' "$train_file"
            sed -i 's/HIDDEN_DIM = 256/HIDDEN_DIM = 512/' "$train_file"
            sed -i 's/NUM_LAYERS = 8/NUM_LAYERS = 6/' "$train_file"
            ;;
        phase3-more-context)
            sed -i 's/NUM_CONTEXT_PAIRS = 5/NUM_CONTEXT_PAIRS = 8/' "$train_file"
            ;;
        phase4-lr-3e4)
            sed -i 's/LEARNING_RATE = 1e-3/LEARNING_RATE = 3e-4/' "$train_file"
            ;;
        phase4-cosine-lr)
            sed -i '/optimizer = torch.optim.Adam/a\scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=700, eta_min=1e-5)' "$train_file"
            sed -i '/optimizer.zero_grad()/a\    scheduler.step()' "$train_file"
            ;;
        phase4-larger-grid)
            sed -i 's/GRID_SIZES = (16, 24, 32)/GRID_SIZES = (16, 24, 32, 48)/' "$train_file"
            ;;
        phase4-more-matrices)
            sed -i 's/MATRICES_PER_EPOCH = 16/MATRICES_PER_EPOCH = 32/' "$train_file"
            ;;
        combo-small-boost-advection)
            sed -i 's/EMBED_DIM = 128/EMBED_DIM = 64/' "$train_file"
            sed -i 's/HIDDEN_DIM = 256/HIDDEN_DIM = 128/' "$train_file"
            sed -i 's/DIFFUSION_ADVECTION: 0.15/DIFFUSION_ADVECTION: 0.25/' "$train_file"
            sed -i 's/VARIABLE_DIFFUSION: 0.10/VARIABLE_DIFFUSION: 0.15/' "$train_file"
            sed -i 's/DIFFUSION: 0.20/DIFFUSION: 0.10/' "$train_file"
            sed -i 's/GRAPH_LAPLACIAN: 0.10/GRAPH_LAPLACIAN: 0.05/' "$train_file"
            ;;
        combo-small-sbm)
            sed -i 's/EMBED_DIM = 128/EMBED_DIM = 64/' "$train_file"
            sed -i 's/HIDDEN_DIM = 256/HIDDEN_DIM = 128/' "$train_file"
            sed -i 's/MatrixDomain.DIFFUSION: 0.20/MatrixDomain.DIFFUSION: 0.15/' "$train_file"
            sed -i '/ENHANCED_ADVECTION: 0.10,/a\    MatrixDomain.SBM: 0.08,' "$train_file"
            sed -i 's/ENHANCED_ADVECTION: 0.10/ENHANCED_ADVECTION: 0.07/' "$train_file"
            ;;
        combo-small-cosine)
            sed -i 's/EMBED_DIM = 128/EMBED_DIM = 64/' "$train_file"
            sed -i 's/HIDDEN_DIM = 256/HIDDEN_DIM = 128/' "$train_file"
            sed -i '/optimizer = torch.optim.Adam/a\scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=700, eta_min=1e-5)' "$train_file"
            sed -i '/optimizer.zero_grad()/a\    scheduler.step()' "$train_file"
            ;;
        combo-small-context8)
            sed -i 's/EMBED_DIM = 128/EMBED_DIM = 64/' "$train_file"
            sed -i 's/HIDDEN_DIM = 256/HIDDEN_DIM = 128/' "$train_file"
            sed -i 's/NUM_CONTEXT_PAIRS = 5/NUM_CONTEXT_PAIRS = 8/' "$train_file"
            ;;
        combo-small-advection-cosine)
            sed -i 's/EMBED_DIM = 128/EMBED_DIM = 64/' "$train_file"
            sed -i 's/HIDDEN_DIM = 256/HIDDEN_DIM = 128/' "$train_file"
            sed -i 's/DIFFUSION_ADVECTION: 0.15/DIFFUSION_ADVECTION: 0.25/' "$train_file"
            sed -i 's/VARIABLE_DIFFUSION: 0.10/VARIABLE_DIFFUSION: 0.15/' "$train_file"
            sed -i 's/DIFFUSION: 0.20/DIFFUSION: 0.10/' "$train_file"
            sed -i 's/GRAPH_LAPLACIAN: 0.10/GRAPH_LAPLACIAN: 0.05/' "$train_file"
            sed -i '/optimizer = torch.optim.Adam/a\scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=700, eta_min=1e-5)' "$train_file"
            sed -i '/optimizer.zero_grad()/a\    scheduler.step()' "$train_file"
            ;;
        combo-large-context8)
            sed -i 's/EMBED_DIM = 128/EMBED_DIM = 256/' "$train_file"
            sed -i 's/HIDDEN_DIM = 256/HIDDEN_DIM = 512/' "$train_file"
            sed -i 's/NUM_LAYERS = 8/NUM_LAYERS = 6/' "$train_file"
            sed -i 's/NUM_CONTEXT_PAIRS = 5/NUM_CONTEXT_PAIRS = 8/' "$train_file"
            ;;
        combo-large-advection)
            sed -i 's/EMBED_DIM = 128/EMBED_DIM = 256/' "$train_file"
            sed -i 's/HIDDEN_DIM = 256/HIDDEN_DIM = 512/' "$train_file"
            sed -i 's/NUM_LAYERS = 8/NUM_LAYERS = 6/' "$train_file"
            sed -i 's/DIFFUSION_ADVECTION: 0.15/DIFFUSION_ADVECTION: 0.25/' "$train_file"
            sed -i 's/VARIABLE_DIFFUSION: 0.10/VARIABLE_DIFFUSION: 0.15/' "$train_file"
            sed -i 's/DIFFUSION: 0.20/DIFFUSION: 0.10/' "$train_file"
            sed -i 's/GRAPH_LAPLACIAN: 0.10/GRAPH_LAPLACIAN: 0.05/' "$train_file"
            ;;
        phase4-lr-3e3)
            sed -i 's/LEARNING_RATE = 1e-3/LEARNING_RATE = 3e-3/' "$train_file"
            ;;
        phase3-layers-12)
            sed -i 's/NUM_LAYERS = 8/NUM_LAYERS = 12/' "$train_file"
            ;;
        phase3-layers-4)
            sed -i 's/NUM_LAYERS = 8/NUM_LAYERS = 4/' "$train_file"
            ;;
        phase3-context-16)
            sed -i 's/NUM_CONTEXT_PAIRS = 5/NUM_CONTEXT_PAIRS = 16/' "$train_file"
            ;;
        phase2-all-domains)
            sed -i 's/DOMAIN_WEIGHTS = {/DOMAIN_WEIGHTS = {\n    MatrixDomain.SBM: 0.05,\n    MatrixDomain.RANDOM_SPARSE: 0.05,\n    MatrixDomain.ENHANCED_DIFFUSION: 0.05,\n    MatrixDomain.VARIABLE_ADVECTION: 0.05,/' "$train_file"
            sed -i 's/DIFFUSION: 0.20/DIFFUSION: 0.10/' "$train_file"
            sed -i 's/ELASTICITY: 0.15/ELASTICITY: 0.08/' "$train_file"
            sed -i 's/STOKES: 0.10/STOKES: 0.07/' "$train_file"
            sed -i 's/DIFFUSION_ADVECTION: 0.15/DIFFUSION_ADVECTION: 0.10/' "$train_file"
            sed -i 's/VARIABLE_DIFFUSION: 0.10/VARIABLE_DIFFUSION: 0.08/' "$train_file"
            sed -i 's/SPECTRAL_STRESS: 0.10/SPECTRAL_STRESS: 0.07/' "$train_file"
            sed -i 's/GRAPH_LAPLACIAN: 0.10/GRAPH_LAPLACIAN: 0.05/' "$train_file"
            sed -i 's/ENHANCED_ADVECTION: 0.10/ENHANCED_ADVECTION: 0.05/' "$train_file"
            ;;
    esac
}

echo "=========================================="
echo "autoresearch-matrixpfn loop starting"
echo "=========================================="

EXPERIMENTS=(
    "baseline|baseline: 8 domains, embed=128, hidden=256, 8 layers"
    "phase2-add-sbm|phase2: add SBM domain"
    "phase2-add-random-sparse|phase2: add RANDOM_SPARSE domain"
    "phase2-boost-advection|phase2: boost advection domains"
    "phase2-all-domains|phase2: all 12 domains"
    "phase3-smaller-model|phase3: smaller model embed=64 hidden=128"
    "phase3-larger-model|phase3: larger model embed=256 hidden=512 layers=6"
    "phase3-more-context|phase3: context pairs=8"
    "phase3-context-16|phase3: context pairs=16"
    "phase3-layers-12|phase3: 12 layers"
    "phase3-layers-4|phase3: 4 layers"
    "phase4-lr-3e4|phase4: lr=3e-4"
    "phase4-lr-3e3|phase4: lr=3e-3"
    "phase4-cosine-lr|phase4: cosine LR schedule"
    "phase4-larger-grid|phase4: add grid size 48"
    "phase4-more-matrices|phase4: matrices_per_epoch=32"
    "combo-small-boost-advection|combo: small model + boost advection"
    "combo-small-sbm|combo: small model + SBM"
    "combo-small-cosine|combo: small model + cosine LR"
    "combo-small-context8|combo: small model + context=8"
    "combo-small-advection-cosine|combo: small model + boost advection + cosine LR"
    "combo-large-context8|combo: large model + context=8"
    "combo-large-advection|combo: large model + boost advection"
)

for exp_line in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_name exp_desc <<< "$exp_line"
    echo ""
    echo ">>> Experiment: $exp_name"
    if [ "$exp_name" != "baseline" ]; then
        apply_experiment "$exp_name"
    fi
    run_experiment "$exp_desc" || true
done

echo ""
echo "=========================================="
echo "ALL PREDEFINED EXPERIMENTS COMPLETE"
echo "=========================================="
echo "Final results:"
cat "$RESULTS_FILE"
echo ""
echo "Best score: $best_score"
echo "Logs saved in: $LOGS_DIR/"
