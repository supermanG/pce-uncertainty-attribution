#!/bin/bash
# LLM GFlowNet + uncertain PRM experiment (NMI).
# 80 members, CPU-only (no GPU queue available), then CPU analyze job.
#
# Usage:
#   cd $HOME/uq-gflownet
#   bash lsf/submit_llm.sh

set -e
PROJECT_DIR=$HOME/uq-gflownet
cd $PROJECT_DIR
mkdir -p logs results/llm_gfn/members

echo "=========================================="
echo "  UQ-GFlowNet: LLM GFlowNet (NMI)"
echo "=========================================="

# Train 80 GFlowNet members (CPU; falls back to GRU backbone if transformers absent)
LAST_JID=""
for i in $(seq 0 79); do
    JID=$(bsub \
        -J "llm_gfn_${i}" \
        -q normal \
        -n 4 \
        -W 2:00 \
        -M 16GB \
        -o "$PROJECT_DIR/logs/llm_gfn_${i}_%J.out" \
        -e "$PROJECT_DIR/logs/llm_gfn_${i}_%J.err" \
        "python3 $PROJECT_DIR/experiments/llm_gfn/train_single_member.py \
            --member_id $i \
            --output_dir $PROJECT_DIR/results/llm_gfn \
            --n_total 80 \
            --n_episodes 500 \
            --device cpu" \
    | grep -oP '(?<=Job <)\d+')
    LAST_JID=$JID
done
echo "  Train jobs submitted (80 members, last JID=$LAST_JID)"

# Analyze (waits for last train job)
bsub \
    -J llm_analyze \
    -q normal \
    -n 4 \
    -W 0:30 \
    -M 8GB \
    -w "done($LAST_JID)" \
    -o "$PROJECT_DIR/logs/llm_analyze_%J.out" \
    -e "$PROJECT_DIR/logs/llm_analyze_%J.err" \
    "python3 $PROJECT_DIR/experiments/llm_gfn/run_experiment.py \
        --mode analyze \
        --n_train 30 --n_test 50 \
        --pce_degree 5 --pca_dim 2 \
        --output_dir $PROJECT_DIR/results/llm_gfn"

echo ""
echo "  Monitor: bjobs -J 'llm_gfn*'"
echo "  Log:     logs/llm_gfn_*.out"
echo "  Results: results/llm_gfn/results.json"
echo "=========================================="
