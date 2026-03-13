#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"

RESULTS_FILE="results.tsv"

if [ ! -f "$RESULTS_FILE" ]; then
    printf "commit\tscore\tss_conv\tmemory_gb\tstatus\tdescription\n" > "$RESULTS_FILE"
fi

best_score="999.0"

run_experiment() {
    local desc="$1"

    git add train.py
    local commit_hash
    commit_hash=$(git rev-parse --short HEAD 2>/dev/null || echo "uncommit")

    if git diff --cached --quiet; then
        echo "No changes to commit, running as-is..."
    else
        git commit -m "$desc"
        commit_hash=$(git rev-parse --short HEAD)
    fi

    echo "=== Running: $desc (commit: $commit_hash) ==="

    timeout 900 uv run train.py > run.log 2>&1 || true

    local score ss_conv peak_vram status
    score=$(grep "^score:" run.log | awk '{print $2}' || echo "")
    ss_conv=$(grep "^suitesparse_conv:" run.log | awk '{print $2}' | tr -d '%' || echo "")
    peak_vram=$(grep "^peak_vram_mb:" run.log | awk '{print $2}' || echo "")

    if [ -z "$score" ]; then
        echo "CRASH — no score found"
        tail -30 run.log
        local mem_gb="0.0"
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

echo "=========================================="
echo "autoresearch-matrixpfn loop starting"
echo "=========================================="

run_experiment "baseline: 8 domains, embed=128, hidden=256, 8 layers"

EXPERIMENTS=(
    "phase2-add-sbm|MatrixDomain.SBM: 0.08|reduce DIFFUSION to 0.15, add SBM"
    "phase2-add-random-sparse|MatrixDomain.RANDOM_SPARSE: 0.08|reduce DIFFUSION to 0.15, add RANDOM_SPARSE"
    "phase2-boost-advection|boost DIFFUSION_ADVECTION to 0.25, VARIABLE_DIFFUSION to 0.15|boost advection domains"
    "phase3-smaller-model|embed=64,hidden=128|smaller model, more epochs"
    "phase3-larger-model|embed=256,hidden=512,layers=6|larger model, fewer layers"
    "phase3-more-context|context=8|more context pairs"
    "phase4-lr-3e4|lr=3e-4|lower learning rate"
    "phase4-cosine-lr|cosine schedule|cosine LR decay"
    "phase4-larger-grid|grids=(16,24,32,48)|add grid 48"
    "phase4-more-matrices|matrices_per_epoch=32|more diversity per step"
)

apply_experiment() {
    local name="$1"
    local train_file="train.py"

    git checkout train.py 2>/dev/null || true
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
    esac
}

for exp_line in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_name exp_detail exp_desc <<< "$exp_line"
    echo ""
    echo ">>> Applying experiment: $exp_name ($exp_desc)"
    apply_experiment "$exp_name"
    run_experiment "$exp_name: $exp_desc" || true
done

echo ""
echo "Phase 1-4 complete. Starting combinatorial experiments..."

best_commit=$(awk -F'\t' '$5=="keep" {print $1}' "$RESULTS_FILE" | tail -1)
if [ -n "$best_commit" ]; then
    echo "Best commit so far: $best_commit"
fi

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo "Final results:"
cat "$RESULTS_FILE"
