#!/bin/bash
# Master launch script: submits ALL experiments to IBM CCC via LSF.
#
# Usage:
#   cd $HOME/uq-gflownet
#   bash lsf/launch_all.sh [--gpu-only] [--cpu-only] [--validation-only]
#
# Default: submits everything (GPU + CPU + validation).
# Total: ~312 GPU jobs, ~535 CPU jobs, plus dependent analysis jobs.

set -e

PROJECT_DIR=$HOME/uq-gflownet
cd $PROJECT_DIR

# Parse flags
RUN_GPU=true
RUN_CPU=true
RUN_VALIDATION=true

for arg in "$@"; do
    case $arg in
        --gpu-only)
            RUN_CPU=false
            RUN_VALIDATION=false
            ;;
        --cpu-only)
            RUN_GPU=false
            RUN_VALIDATION=false
            ;;
        --validation-only)
            RUN_GPU=false
            RUN_CPU=false
            ;;
    esac
done

echo "============================================================"
echo "  UQ-GFlowNet: Master Launch"
echo "  Project: $PROJECT_DIR"
echo "  GPU experiments:  $RUN_GPU"
echo "  CPU experiments:  $RUN_CPU"
echo "  Validation:       $RUN_VALIDATION"
echo "  Time:             $(date)"
echo "============================================================"

# Verify environment
echo ""
echo "Checking environment..."
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "  WARNING: torch not found -- GPU jobs will fail"
python3 -c "import numpy; print(f'  NumPy {numpy.__version__}')" 2>/dev/null || { echo "  ERROR: numpy not found"; exit 1; }
python3 -c "import scipy; print(f'  SciPy {scipy.__version__}')" 2>/dev/null || { echo "  ERROR: scipy not found"; exit 1; }
echo "  Environment OK."

mkdir -p logs results

# ===================== GPU EXPERIMENTS =====================
if [ "$RUN_GPU" == "true" ]; then
    echo ""
    echo "============================================================"
    echo "  Launching GPU experiments..."
    echo "============================================================"

    echo ""
    echo "--- Buchwald-Hartwig (150 GPU jobs + analysis) ---"
    bash lsf/submit_bh.sh

    echo ""
    echo "--- Sachs Causal (80 GPU jobs + analysis) ---"
    bash lsf/submit_sachs.sh

    echo ""
    echo "--- Molecular Design (80 GPU jobs + analysis) ---"
    bash lsf/submit_moldesign.sh
fi

# ===================== CPU EXPERIMENTS =====================
if [ "$RUN_CPU" == "true" ]; then
    echo ""
    echo "============================================================"
    echo "  Launching CPU experiments..."
    echo "============================================================"

    echo ""
    echo "--- Grid World (300 CPU jobs + 2 analysis) ---"
    bash lsf/submit_gridworld.sh

    echo ""
    echo "--- Symbolic Regression (150 CPU jobs + analysis) ---"
    bash lsf/submit_symreg.sh

    echo ""
    echo "--- LLM GFlowNet (80 CPU jobs + analysis) ---"
    bash lsf/submit_llm.sh
fi

# ===================== VALIDATION =====================
if [ "$RUN_VALIDATION" == "true" ]; then
    echo ""
    echo "============================================================"
    echo "  Launching validation experiments..."
    echo "============================================================"
    bash lsf/submit_validation.sh
fi

# ===================== SUMMARY =====================
echo ""
echo "============================================================"
echo "  ALL JOBS SUBMITTED at $(date)"
echo ""
echo "  Monitor:"
echo "    bjobs -u $USER                # all jobs"
echo "    bjobs -J 'bh_gfn'              # BH training (150)"
echo "    bjobs -J 'sachs_gfn'           # Sachs training (80)"
echo "    bjobs -J 'moldesign_gfn'       # MolDesign training (80)"
echo "    bjobs -J '*_analyze'           # analysis jobs (waiting)"
echo "    bjobs -J 'val_*'              # validation jobs"
echo ""
echo "  Results will appear in:"
echo "    results/buchwald_hartwig/results.json"
echo "    results/sachs/results.json"
echo "    results/molecular_design/results.json"
echo "    results/gridworld/results.json"
echo "    results/symreg/results.json"
echo "    results/llm_gfn/results.json"
echo "    results/validation/*.json"
echo "============================================================"
