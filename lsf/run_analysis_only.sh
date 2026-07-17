#!/bin/bash
# Re-run analysis only (all members already trained). Baseline then extended analysis, with
# ';' so a baseline hiccup does not block the extended analysis. CPU jobs.
set -e
P=$HOME/uq-gflownet
cd $P
BH_CSV=$P/experiments/buchwald_hartwig/data/data_table.csv
SACHS_CSV=$P/data/sachs_real.csv

bsub -J bh_analyze -q normal -n 8 -W 2:00 -M 32GB \
  -o $P/logs/bh_analyze_%J.out -e $P/logs/bh_analyze_%J.err \
  "python3 $P/experiments/buchwald_hartwig/run_experiment.py --mode analyze \
      --n_train 50 --n_test 100 --pce_degree 3 --pca_dim 5 --csv_path $BH_CSV \
      --output_dir $P/results/buchwald_hartwig ; \
   python3 $P/decision_studies/common/analyze_ensemble.py --members_dir $P/results/buchwald_hartwig \
      --n_train 50 --n_test 100 --degree 3 --pca_dims 5,7,10,15 --out $P/results/outputs/bh"

bsub -J sachs_analyze -q normal -n 8 -W 3:00 -M 16GB \
  -o $P/logs/sachs_analyze_%J.out -e $P/logs/sachs_analyze_%J.err \
  "python3 $P/experiments/sachs_causal/run_experiment.py --mode analyze \
      --n_train 30 --n_test 50 --pce_degree 5 --pca_dim 2 \
      --output_dir $P/results/sachs --data_path $SACHS_CSV ; \
   python3 $P/decision_studies/common/analyze_ensemble.py --members_dir $P/results/sachs \
      --n_train 30 --n_test 50 --degree 3 --pca_dims 2,5,10 --n_boot 100 --out $P/results/outputs/sachs"

bsub -J moldesign_analyze -q normal -n 8 -W 2:00 -M 16GB \
  -o $P/logs/moldesign_analyze_%J.out -e $P/logs/moldesign_analyze_%J.err \
  "python3 $P/experiments/molecular_design/run_experiment.py --mode analyze \
      --n_train 30 --n_test 50 --pce_degree 5 --pca_dim 2 \
      --output_dir $P/results/molecular_design ; \
   python3 $P/decision_studies/common/analyze_ensemble.py --members_dir $P/results/molecular_design \
      --n_train 30 --n_test 50 --degree 3 --pca_dims 2,5,10 --out $P/results/outputs/moldesign"

echo "analysis-only jobs submitted"
