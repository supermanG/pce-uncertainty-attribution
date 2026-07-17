#!/bin/bash
# Molecular design pipeline for IBM CCC (LSF).
# 80 members (30 train + 50 test), one GPU each.
#
# Usage:
#   cd $HOME/uq-gflownet
#   bash lsf/submit_moldesign.sh

set -e

PROJECT_DIR=$HOME/uq-gflownet
cd $PROJECT_DIR
mkdir -p logs results/molecular_design

echo "=========================================="
echo "  UQ-GFlowNet: Molecular Design (LSF)"
echo "  Project: $PROJECT_DIR"
echo "=========================================="

# Step 1: Setup
echo ""
echo "Step 1: Setup..."
if [ -f results/molecular_design/metadata.json ]; then
    echo "  Metadata already exists -- skipping setup."
else
    python3 experiments/molecular_design/train_single_member.py \
        --setup \
        --output_dir results/molecular_design
fi

# Step 2: Submit 80 GPU jobs
echo ""
echo "Step 2: Submitting 80 GPU jobs..."
N_MEMBERS=80
for i in $(seq 0 $((N_MEMBERS - 1))); do
    bsub -J moldesign_gfn \
         -q normal \
         -n 1 \
         -gpu "num=1:mode=exclusive_process" \
         -W 1:00 \
         -M 16GB \
         -o "$PROJECT_DIR/logs/moldesign_train_${i}_%J.out" \
         -e "$PROJECT_DIR/logs/moldesign_train_${i}_%J.err" \
         "python3 $PROJECT_DIR/experiments/molecular_design/train_single_member.py \
             --member_id $i \
             --output_dir $PROJECT_DIR/results/molecular_design \
             --gfn_episodes 3000" > /dev/null
done
echo "  Submitted $N_MEMBERS jobs (all named moldesign_gfn)."

# Step 3: Submit analysis job
echo ""
echo "Step 3: Submitting analysis job (depends on all moldesign_gfn jobs)..."
ANALYZE_OUT=$(bsub \
    -J moldesign_analyze \
    -q normal \
    -n 8 \
    -W 2:00 \
    -M 16GB \
    -w "done(moldesign_gfn)" \
    -o "$PROJECT_DIR/logs/moldesign_analyze_%J.out" \
    -e "$PROJECT_DIR/logs/moldesign_analyze_%J.err" \
    "python3 $PROJECT_DIR/experiments/molecular_design/run_experiment.py \
        --mode analyze \
        --n_train 30 \
        --n_test 50 \
        --pce_degree 5 \
        --pca_dim 2 \
        --output_dir $PROJECT_DIR/results/molecular_design && \
     python3 $PROJECT_DIR/decision_studies/common/analyze_ensemble.py \
        --members_dir $PROJECT_DIR/results/molecular_design \
        --n_train 30 --n_test 50 --degree 3 --pca_dims 2,5,10 --n_boot 200 \
        --out $PROJECT_DIR/results/outputs/moldesign")
echo "  $ANALYZE_OUT"

echo ""
echo "=========================================="
echo "  Submitted!"
echo "  80 GPU jobs:  bjobs -J moldesign_gfn"
echo "  Analysis job: bjobs -J moldesign_analyze"
echo "  Monitor all:  bjobs -u $USER"
echo "  Results:      results/molecular_design/results.json"
echo "=========================================="
