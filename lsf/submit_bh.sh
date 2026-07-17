#!/bin/bash
# Full Buchwald-Hartwig pipeline for IBM CCC (LSF, no array jobs).
# Submits 150 individual GPU jobs then a CPU analysis job.
#
# Usage:
#   cd $HOME/uq-gflownet
#   bash lsf/submit_bh.sh

set -e

PROJECT_DIR=$HOME/uq-gflownet
cd $PROJECT_DIR
mkdir -p logs results/buchwald_hartwig

echo "=========================================="
echo "  UQ-GFlowNet: Buchwald-Hartwig (LSF)"
echo "  Project: $PROJECT_DIR"
echo "=========================================="

# Step 1: Phase 1 setup (fast, interactive)
echo ""
echo "Step 1: Phase 1 setup..."
if [ -f results/buchwald_hartwig/metadata.json ]; then
    echo "  Metadata already exists -- skipping setup."
else
    python3 experiments/buchwald_hartwig/train_single_gfn.py \
        --setup \
        --csv_path $PROJECT_DIR/experiments/buchwald_hartwig/data/data_table.csv \
        --output_dir results/buchwald_hartwig
fi

# Step 2: Submit 150 individual GPU jobs (all named bh_gfn)
echo ""
echo "Step 2: Submitting 150 GPU jobs..."
N_MEMBERS=150
for i in $(seq 0 $((N_MEMBERS - 1))); do
    bsub -J bh_gfn \
         -q normal \
         -n 1 \
         -gpu "num=1:mode=exclusive_process" \
         -W 1:30 \
         -M 32GB \
         -o "$PROJECT_DIR/logs/bh_train_${i}_%J.out" \
         -e "$PROJECT_DIR/logs/bh_train_${i}_%J.err" \
         "python3 $PROJECT_DIR/experiments/buchwald_hartwig/train_single_gfn.py \
             --member_id $i \
             --output_dir $PROJECT_DIR/results/buchwald_hartwig \
             --csv_path $PROJECT_DIR/experiments/buchwald_hartwig/data/data_table.csv \
             --gfn_episodes 3000 \
             --proxy_epochs 200 \
             --temp 4.0" > /dev/null
done
echo "  Submitted $N_MEMBERS jobs (all named bh_gfn)."

# Step 3: Submit analysis job -- waits for ALL jobs named bh_gfn to finish
echo ""
echo "Step 3: Submitting analysis job (depends on all bh_gfn jobs)..."
ANALYZE_OUT=$(bsub \
    -J bh_analyze \
    -q normal \
    -n 8 \
    -W 2:00 \
    -M 32GB \
    -w "done(bh_gfn)" \
    -o "$PROJECT_DIR/logs/bh_analyze_%J.out" \
    -e "$PROJECT_DIR/logs/bh_analyze_%J.err" \
    "python3 $PROJECT_DIR/experiments/buchwald_hartwig/run_experiment.py \
        --mode analyze \
        --n_train 50 \
        --n_test 100 \
        --pce_degree 3 \
        --pca_dim 5 \
        --csv_path $PROJECT_DIR/experiments/buchwald_hartwig/data/data_table.csv \
        --output_dir $PROJECT_DIR/results/buchwald_hartwig && \
     python3 $PROJECT_DIR/decision_studies/common/analyze_ensemble.py \
        --members_dir $PROJECT_DIR/results/buchwald_hartwig \
        --n_train 50 --n_test 100 --degree 3 --pca_dims 5,7,10,15 \
        --out $PROJECT_DIR/results/outputs/bh")
echo "  $ANALYZE_OUT"

# Step 4 (optional): training-noise control ensemble.
# Fixes the reward realisation (proxy_seed=0) and varies only the GFlowNet seed,
# so the spread here is training noise alone and can be compared with the main
# ensemble's reward-driven spread. Enable by setting BH_NOISE_CONTROL=1.
if [ "${BH_NOISE_CONTROL:-0}" = "1" ]; then
    echo ""
    echo "Step 4: training-noise control ensemble (fixed reward, varying seed)..."
    CTRL_DIR=$PROJECT_DIR/results/buchwald_hartwig_noise_control
    mkdir -p $CTRL_DIR
    python3 experiments/buchwald_hartwig/train_single_gfn.py --setup \
        --csv_path $PROJECT_DIR/experiments/buchwald_hartwig/data/data_table.csv \
        --output_dir $CTRL_DIR
    N_CTRL=30
    for i in $(seq 0 $((N_CTRL - 1))); do
        bsub -J bh_gfn_ctrl -q normal -n 1 \
             -gpu "num=1:mode=exclusive_process" -W 1:30 -M 32GB \
             -o "$PROJECT_DIR/logs/bh_ctrl_${i}_%J.out" \
             -e "$PROJECT_DIR/logs/bh_ctrl_${i}_%J.err" \
             "python3 $PROJECT_DIR/experiments/buchwald_hartwig/train_single_gfn.py \
                 --member_id $i --output_dir $CTRL_DIR \
                 --csv_path $PROJECT_DIR/experiments/buchwald_hartwig/data/data_table.csv \
                 --gfn_episodes 3000 --proxy_epochs 200 --temp 4.0 \
                 --proxy_seed 0 --gfn_seed $i" > /dev/null
    done
    echo "  Submitted $N_CTRL control jobs (bh_gfn_ctrl, fixed proxy_seed=0)."
fi

echo ""
echo "=========================================="
echo "  Submitted!"
echo "  150 GPU jobs:  bjobs -J bh_gfn"
echo "  Analysis job:  bjobs -J bh_analyze"
echo "  Monitor all:   bjobs -u $USER"
echo "  Results:       results/buchwald_hartwig/results.json"
echo "  Logs:          logs/bh_train_*.out"
echo "=========================================="
