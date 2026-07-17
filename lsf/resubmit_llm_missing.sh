#!/bin/bash
# Resubmit only the LLM GFlowNet members that did not produce output.
# Uses GRU backbone (fast, CPU-only) with 100 episodes.
#
# Usage:
#   cd $HOME/uq-gflownet
#   bash lsf/resubmit_llm_missing.sh

set -e
PROJECT_DIR=$HOME/uq-gflownet
cd $PROJECT_DIR
mkdir -p logs results/llm_gfn/members

echo "=========================================="
echo "  UQ-GFlowNet: LLM GFlowNet -- resubmit missing"
echo "=========================================="

LAST_JID=""
N_SUBMITTED=0
for i in $(seq 0 79); do
    NPZ="$PROJECT_DIR/results/llm_gfn/members/member_$(printf '%04d' $i).npz"
    if [ -f "$NPZ" ]; then
        continue   # already done
    fi
    JID=$(bsub \
        -J "llm_gfn_${i}" \
        -q normal \
        -n 2 \
        -W 0:30 \
        -M 8GB \
        -o "$PROJECT_DIR/logs/llm_gfn_${i}_r2_%J.out" \
        -e "$PROJECT_DIR/logs/llm_gfn_${i}_r2_%J.err" \
        "python3 $PROJECT_DIR/experiments/llm_gfn/train_single_member.py \
            --member_id $i \
            --output_dir $PROJECT_DIR/results/llm_gfn \
            --n_total 80 \
            --n_episodes 100 \
            --backbone gru \
            --device cpu" \
    | grep -oP '(?<=Job <)\d+')
    LAST_JID=$JID
    N_SUBMITTED=$((N_SUBMITTED + 1))
done

echo "  Submitted $N_SUBMITTED jobs (last JID=$LAST_JID)"

if [ -n "$LAST_JID" ]; then
    bsub \
        -J llm_analyze \
        -q normal \
        -n 4 \
        -W 0:30 \
        -M 8GB \
        -w "done($LAST_JID)" \
        -o "$PROJECT_DIR/logs/llm_analyze_r2_%J.out" \
        -e "$PROJECT_DIR/logs/llm_analyze_r2_%J.err" \
        "python3 $PROJECT_DIR/experiments/llm_gfn/run_experiment.py \
            --mode analyze \
            --n_train 30 --n_test 50 \
            --pce_degree 5 --pca_dim 2 \
            --output_dir $PROJECT_DIR/results/llm_gfn"
    echo "  Analyze job queued (waits for last member)"
fi

echo ""
echo "  Monitor: bjobs -J 'llm_gfn*'"
echo "  Results: results/llm_gfn/results.json"
echo "=========================================="
