#!/bin/bash
# Overnight GPU runner: practical for single RTX 3000 Pro (12GB).
# Reduced member counts + episodes for local validation.
# Full-scale runs (150/80 members, 3000 episodes) reserved for CCC cluster.
#
# Local validation targets:
#   Sachs:     50 members (30 train + 20 test) x 500 ep = ~6 hours
#   MolDesign: 50 members x 500 ep (runs after Sachs if time permits)
#
# Usage:
#   nohup bash run_gpu_overnight.sh > results/gpu_overnight.log 2>&1 &

set -e
cd "$(dirname "$0")/.."
PY=python3
export PYTHONUNBUFFERED=1

GFN_EPISODES=500
SACHS_MEMBERS=50
MOLDESIGN_MEMBERS=50

echo "=========================================="
echo "  GPU Overnight Run - $(date)"
echo "  Episodes: $GFN_EPISODES"
echo "  Sachs: $SACHS_MEMBERS members"
echo "  MolDesign: $MOLDESIGN_MEMBERS members"
echo "=========================================="

# --- Sachs causal ---
echo ""
echo "=== Sachs Causal ($SACHS_MEMBERS members) ==="
echo "Started: $(date)"
mkdir -p results/sachs/members
SACHS_DONE=0
for i in $(seq 0 $((SACHS_MEMBERS - 1))); do
    NPZ="results/sachs/members/member_$(printf '%04d' $i).npz"
    if [ -f "$NPZ" ]; then
        echo "  Member $i: skipped (already done)"
        SACHS_DONE=$((SACHS_DONE + 1))
        continue
    fi
    echo "  Member $i: starting $(date)"
    $PY experiments/sachs_causal/train_single_member.py \
        --member_id $i \
        --output_dir results/sachs \
        --gfn_episodes $GFN_EPISODES
    echo "  Member $i: done $(date)"
    SACHS_DONE=$((SACHS_DONE + 1))
done
echo "Sachs training done: $SACHS_DONE/$SACHS_MEMBERS members $(date)"

echo "Running Sachs analysis..."
SACHS_DATA_ARG=""
if [ -f "data/sachs_real.csv" ]; then
    SACHS_DATA_ARG="--data_path data/sachs_real.csv"
fi
$PY experiments/sachs_causal/run_experiment.py \
    --mode analyze --n_train 30 --n_test 20 \
    --pce_degree 5 --pca_dim 2 \
    --output_dir results/sachs \
    $SACHS_DATA_ARG
echo "Sachs analysis done: $(date)"

# --- Molecular design ---
echo ""
echo "=== Molecular Design ($MOLDESIGN_MEMBERS members) ==="
echo "Started: $(date)"
mkdir -p results/molecular_design/members
if [ ! -f "results/molecular_design/metadata.json" ]; then
    $PY experiments/molecular_design/train_single_member.py \
        --setup --output_dir results/molecular_design
fi
for i in $(seq 0 $((MOLDESIGN_MEMBERS - 1))); do
    NPZ="results/molecular_design/members/member_$(printf '%04d' $i).npz"
    if [ -f "$NPZ" ]; then
        echo "  Member $i: skipped (already done)"
        continue
    fi
    echo "  Member $i: starting $(date)"
    $PY experiments/molecular_design/train_single_member.py \
        --member_id $i \
        --output_dir results/molecular_design \
        --gfn_episodes $GFN_EPISODES
    echo "  Member $i: done $(date)"
done
echo "MolDesign training done: $(date)"

echo "Running MolDesign analysis..."
$PY experiments/molecular_design/run_experiment.py \
    --mode analyze --n_train 30 --n_test 20 \
    --pce_degree 5 --pca_dim 2 \
    --output_dir results/molecular_design
echo "MolDesign analysis done: $(date)"

echo ""
echo "=========================================="
echo "  GPU EXPERIMENTS DONE - $(date)"
echo "=========================================="
