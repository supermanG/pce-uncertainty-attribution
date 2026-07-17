#!/bin/bash
# Symbolic regression experiment: GRU GFlowNet + Wiener-noise reward.
# CPU-only, ~15 min per member (GRU training).
#
# Usage:
#   cd $HOME/uq-gflownet
#   bash lsf/submit_symreg.sh

set -e
PROJECT_DIR=$HOME/uq-gflownet
cd $PROJECT_DIR
mkdir -p logs results/symreg/members

echo "=========================================="
echo "  UQ-GFlowNet: Symbolic Regression"
echo "=========================================="

# 150 members (100 train + 50 test)
LAST_JID=""
for i in $(seq 0 149); do
    JID=$(bsub \
        -J "symreg_${i}" \
        -q normal \
        -n 2 \
        -W 0:45 \
        -M 8GB \
        -o "$PROJECT_DIR/logs/symreg_${i}_%J.out" \
        -e "$PROJECT_DIR/logs/symreg_${i}_%J.err" \
        "python3 $PROJECT_DIR/experiments/symreg/train_single_member.py \
            --member_id $i \
            --output_dir $PROJECT_DIR/results/symreg" \
    | grep -oP '(?<=Job <)\d+')
    LAST_JID=$JID
done
echo "  Train jobs submitted (150 members, last JID=$LAST_JID)"

bsub \
    -J symreg_analyze \
    -q normal \
    -n 4 \
    -W 0:20 \
    -M 8GB \
    -w "done($LAST_JID)" \
    -o "$PROJECT_DIR/logs/symreg_analyze_%J.out" \
    -e "$PROJECT_DIR/logs/symreg_analyze_%J.err" \
    "python3 $PROJECT_DIR/experiments/symreg/run_experiment.py \
        --run_mode analyze \
        --n_train 100 --n_test 50 \
        --pce_degree 5 \
        --output_dir $PROJECT_DIR/results/symreg"

echo ""
echo "  Monitor: bjobs -J 'symreg*'"
echo "  Results: results/symreg/results.json"
echo "=========================================="
