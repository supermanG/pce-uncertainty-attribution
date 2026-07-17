#!/bin/bash
# LSF job: PCE fitting + Sobol analysis after all array members complete.
# Submitted with dependency via submit_bh.sh -- do not submit directly.

#BSUB -J "bh_pce_analyze"
#BSUB -o $HOME/uq-gflownet/logs/bh_analyze_%J.out
#BSUB -e $HOME/uq-gflownet/logs/bh_analyze_%J.err
#BSUB -q normal
#BSUB -n 8
#BSUB -W 0:30
#BSUB -M 32GB

PROJECT_DIR=$HOME/uq-gflownet
cd $PROJECT_DIR

echo "Running PCE analysis on: $(hostname)"

# Baseline (as-submitted) dense-ridge pipeline, at the manuscript config.
python3 experiments/buchwald_hartwig/run_experiment.py \
    --mode analyze \
    --n_train 50 \
    --n_test 100 \
    --pce_degree 3 \
    --pca_dim 5 \
    --csv_path $PROJECT_DIR/experiments/buchwald_hartwig/data/data_table.csv \
    --output_dir results/buchwald_hartwig

# Extended-analysis pipeline: sparse/aPC PCE, PCA-dim sweep, fragility, interactions,
# validation errors, bootstrap CIs, Source Data.
python3 decision_studies/common/analyze_ensemble.py \
    --members_dir results/buchwald_hartwig \
    --n_train 50 --n_test 100 --degree 3 --pca_dims 5,7,10,15 \
    --out results/outputs/bh

echo "Analysis complete at $(date)"
