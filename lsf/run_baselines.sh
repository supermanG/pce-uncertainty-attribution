#!/bin/bash
# Surrogate comparison: PCE vs MLP vs GP on BH + Sachs.
# CPU-only, ~30 min wall time.
#
# Usage:
#   cd $HOME/uq-gflownet
#   bash lsf/run_baselines.sh

set -e
PROJECT_DIR=$HOME/uq-gflownet
cd $PROJECT_DIR
mkdir -p logs results/baselines

echo "=========================================="
echo "  UQ-GFlowNet: Baseline Surrogate Comparison"
echo "=========================================="

bsub \
    -J baselines \
    -q normal \
    -n 8 \
    -W 1:00 \
    -M 32GB \
    -o "$PROJECT_DIR/logs/baselines_%J.out" \
    -e "$PROJECT_DIR/logs/baselines_%J.err" \
    "python3 $PROJECT_DIR/experiments/baselines/run_baselines.py \
        --task all \
        --bh_dir    $PROJECT_DIR/results/buchwald_hartwig \
        --sachs_dir $PROJECT_DIR/results/sachs \
        --n_train_bh    50  --n_test_bh    100 --pca_dim_bh    5 --pce_degree_bh    3 \
        --n_train_sachs 30  --n_test_sachs  50 --pca_dim_sachs 2 --pce_degree_sachs 3 \
        --n_sample 10000 \
        --results_dir $PROJECT_DIR/results/baselines"

echo ""
echo "  Monitor: bjobs -J baselines"
echo "  Log:     logs/baselines_*.out"
echo "  Results: results/baselines/comparison.json"
echo "=========================================="
