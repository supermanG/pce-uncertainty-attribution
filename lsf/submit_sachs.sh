#!/bin/bash
# Sachs causal discovery pipeline for IBM CCC (LSF).
# 80 members (30 train + 50 test), one GPU each.
#
# Usage:
#   cd $HOME/uq-gflownet
#   bash lsf/submit_sachs.sh

set -e

PROJECT_DIR=$HOME/uq-gflownet
cd $PROJECT_DIR
mkdir -p logs results/sachs

echo "=========================================="
echo "  UQ-GFlowNet: Sachs Causal (LSF)"
echo "  Project: $PROJECT_DIR"
echo "=========================================="

# Step 1: Setup
echo ""
echo "Step 1: Setup..."
if [ -f results/sachs/metadata.json ]; then
    echo "  Metadata already exists -- skipping setup."
else
    python3 experiments/sachs_causal/train_single_member.py \
        --setup \
        --output_dir results/sachs
fi

# Step 2: Submit 80 GPU jobs
echo ""
echo "Step 2: Submitting 80 GPU jobs..."
N_MEMBERS=80
for i in $(seq 0 $((N_MEMBERS - 1))); do
    bsub -J sachs_gfn \
         -q normal \
         -n 1 \
         -gpu "num=1:mode=exclusive_process" \
         -W 1:00 \
         -M 16GB \
         -o "$PROJECT_DIR/logs/sachs_train_${i}_%J.out" \
         -e "$PROJECT_DIR/logs/sachs_train_${i}_%J.err" \
         "python3 $PROJECT_DIR/experiments/sachs_causal/train_single_member.py \
             --member_id $i \
             --output_dir $PROJECT_DIR/results/sachs \
             --gfn_episodes 3000" > /dev/null
done
echo "  Submitted $N_MEMBERS jobs (all named sachs_gfn)."

# Step 3: Submit analysis job
echo ""
echo "Step 3: Submitting analysis job (depends on all sachs_gfn jobs)..."

# Pass --data_path only when the real Sachs CSV is present so the pipeline
# falls back gracefully to synthetic data on machines where it has not been
# downloaded yet.
SACHS_DATA_ARG=""
if [ -f "$PROJECT_DIR/data/sachs_real.csv" ]; then
    SACHS_DATA_ARG="--data_path $PROJECT_DIR/data/sachs_real.csv"
    echo "  Real Sachs data found -- analysis will use $PROJECT_DIR/data/sachs_real.csv"
else
    echo "  sachs_real.csv not found -- analysis will use synthetic fallback data."
    echo "  Run: python3 $PROJECT_DIR/data/download_sachs.py to download it."
fi

ANALYZE_OUT=$(bsub \
    -J sachs_analyze \
    -q normal \
    -n 8 \
    -W 3:00 \
    -M 16GB \
    -w "done(sachs_gfn)" \
    -o "$PROJECT_DIR/logs/sachs_analyze_%J.out" \
    -e "$PROJECT_DIR/logs/sachs_analyze_%J.err" \
    "python3 $PROJECT_DIR/experiments/sachs_causal/run_experiment.py \
        --mode analyze \
        --n_train 30 \
        --n_test 50 \
        --pce_degree 5 \
        --pca_dim 2 \
        --output_dir $PROJECT_DIR/results/sachs \
        $SACHS_DATA_ARG && \
     python3 $PROJECT_DIR/decision_studies/common/analyze_ensemble.py \
        --members_dir $PROJECT_DIR/results/sachs \
        --n_train 30 --n_test 50 --degree 3 --pca_dims 2,5,10 --n_boot 100 \
        --out $PROJECT_DIR/results/outputs/sachs")
echo "  $ANALYZE_OUT"

echo ""
echo "=========================================="
echo "  Submitted!"
echo "  80 GPU jobs:  bjobs -J sachs_gfn"
echo "  Analysis job: bjobs -J sachs_analyze"
echo "  Monitor all:  bjobs -u $USER"
echo "  Results:      results/sachs/results.json"
echo "=========================================="
