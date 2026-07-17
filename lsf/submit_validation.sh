#!/bin/bash
# Submit all validation experiments to IBM CCC (LSF).
#
# Jobs:
#   1. pca_independence    -- CPU, ~5 min (gridworld fast; BH needs proxy training ~20 min)
#   2. negative_controls   -- CPU, ~15 min (gridworld ensemble training)
#   3. sobol_interactions   -- CPU, ~10 min (gridworld)
#   4. scaling_benchmark   -- CPU, ~10 min (gridworld)
#   5. bh_l60              -- GPU, ~2h (180 GFlowNets, sweep L=30,50,60,80)
#   6. bh_additive_valid   -- CPU, ~5 min (uses existing BH results if available)
#
# Usage:
#   cd $HOME/uq-gflownet
#   bash lsf/submit_validation.sh
#
# Or run just the CPU jobs (fast, no GPU):
#   bash lsf/submit_validation.sh --cpu-only

set -e

PROJECT_DIR=$HOME/uq-gflownet
cd $PROJECT_DIR
mkdir -p logs results/validation

CPU_ONLY=false
if [ "$1" == "--cpu-only" ]; then
    CPU_ONLY=true
fi

echo "=========================================="
echo "  UQ-GFlowNet: Validation Experiments"
echo "  Project: $PROJECT_DIR"
echo "  CPU-only: $CPU_ONLY"
echo "=========================================="

# ---- Job 1: PCA independence (CPU) ----
echo ""
echo "Job 1: PCA independence verification..."
bsub -J val_pca_indep \
     -q normal \
     -n 4 \
     -W 0:30 \
     -M 16GB \
     -o "$PROJECT_DIR/logs/val_pca_indep_%J.out" \
     -e "$PROJECT_DIR/logs/val_pca_indep_%J.err" \
     "cd $PROJECT_DIR && python3 experiments/validation/pca_independence.py"
echo "  Submitted."

# ---- Job 2: Negative controls (CPU) ----
echo ""
echo "Job 2: Negative controls..."
bsub -J val_neg_ctrl \
     -q normal \
     -n 4 \
     -W 0:30 \
     -M 16GB \
     -o "$PROJECT_DIR/logs/val_neg_ctrl_%J.out" \
     -e "$PROJECT_DIR/logs/val_neg_ctrl_%J.err" \
     "cd $PROJECT_DIR && python3 experiments/validation/negative_controls.py"
echo "  Submitted."

# ---- Job 3: Sobol interactions (CPU) ----
echo ""
echo "Job 3: Sobol interaction diagnostic..."
bsub -J val_sobol_int \
     -q normal \
     -n 4 \
     -W 0:20 \
     -M 8GB \
     -o "$PROJECT_DIR/logs/val_sobol_int_%J.out" \
     -e "$PROJECT_DIR/logs/val_sobol_int_%J.err" \
     "cd $PROJECT_DIR && python3 experiments/validation/sobol_interactions.py"
echo "  Submitted."

# ---- Job 4: Scaling benchmark (CPU) ----
echo ""
echo "Job 4: Computational scaling benchmark..."
bsub -J val_scaling \
     -q normal \
     -n 4 \
     -W 0:20 \
     -M 8GB \
     -o "$PROJECT_DIR/logs/val_scaling_%J.out" \
     -e "$PROJECT_DIR/logs/val_scaling_%J.err" \
     "cd $PROJECT_DIR && python3 experiments/validation/scaling_benchmark.py"
echo "  Submitted."

# ---- Job 5: BH L=60 comparison (GPU, ~2h) ----
if [ "$CPU_ONLY" == "false" ]; then
    echo ""
    echo "Job 5: BH ensemble size comparison (L=30,50,60,80) [GPU]..."
    bsub -J val_bh_l60 \
         -q normal \
         -n 4 \
         -gpu "num=1:mode=exclusive_process" \
         -W 3:00 \
         -M 32GB \
         -o "$PROJECT_DIR/logs/val_bh_l60_%J.out" \
         -e "$PROJECT_DIR/logs/val_bh_l60_%J.err" \
         "cd $PROJECT_DIR && python3 experiments/validation/bh_l60.py"
    echo "  Submitted."
else
    echo ""
    echo "Job 5: BH L=60 -- SKIPPED (--cpu-only)"
fi

# ---- Job 6: BH additive validation (CPU, uses existing results) ----
echo ""
echo "Job 6: BH additive coverage validation..."
bsub -J val_bh_additive \
     -q normal \
     -n 2 \
     -W 0:15 \
     -M 8GB \
     -o "$PROJECT_DIR/logs/val_bh_additive_%J.out" \
     -e "$PROJECT_DIR/logs/val_bh_additive_%J.err" \
     "cd $PROJECT_DIR && python3 experiments/validation/bh_additive_validation.py"
echo "  Submitted."

echo ""
echo "=========================================="
echo "  All validation jobs submitted!"
echo ""
echo "  Monitor:  bjobs -J 'val_*'"
echo "  Results:  results/validation/*.json"
echo "  Logs:     logs/val_*"
echo ""
echo "  CPU jobs (~15 min): val_pca_indep, val_neg_ctrl,"
echo "                      val_sobol_int, val_scaling, val_bh_additive"
echo "  GPU job  (~2h):     val_bh_l60"
echo "=========================================="
