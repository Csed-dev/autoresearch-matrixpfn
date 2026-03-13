#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/Csed-dev/autoresearch-matrixpfn.git"
WORKSPACE="/workspace/autoresearch-matrixpfn"
BRANCH="${1:-main}"

echo "=== setup_pod.sh ==="
echo "Branch: $BRANCH"

if [ ! -d "$WORKSPACE/.git" ]; then
    echo "Cloning repository..."
    git clone "$REPO_URL" "$WORKSPACE"
fi

cd "$WORKSPACE"
git fetch origin
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"
echo "Repository ready: $(git log --oneline -1)"

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
fi

echo "Installing dependencies..."
uv sync

echo "Downloading SuiteSparse matrices..."
uv run prepare.py

echo "Verifying setup..."
uv run python3 -c "import matrixpfn; print(f'matrixpfn {matrixpfn.__version__}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== setup complete ==="
