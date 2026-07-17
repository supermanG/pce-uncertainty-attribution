#!/bin/bash
# Resubmit the handful of ensemble members that hit the wall-clock limit
# (LSF SIGUSR2), then fire each analysis job gated on those specific job IDs.
# The original analysis jobs died because -w "done(<name>)" is unsatisfiable once
# any same-named member EXITs; here we depend on the exact resubmitted IDs.
set -e
P=$HOME/uq-gflownet
cd $P
BH_CSV=$P/experiments/buchwald_hartwig/data/data_table.csv
SACHS_CSV=$P/data/sachs_real.csv

jid() { grep -oE 'Job <[0-9]+>' | grep -oE '[0-9]+'; }

# ---- Buchwald-Hartwig: members 46, 47 ----
BH_DEPS=""
for i in 46 47; do
  id=$(bsub -J bh_gfn_re -q normal -n 1 -gpu "num=1:mode=exclusive_process" \
        -W 2:00 -M 32GB -o $P/logs/bh_re_${i}_%J.out -e $P/logs/bh_re_${i}_%J.err \
        "python3 $P/experiments/buchwald_hartwig/train_single_gfn.py --member_id $i \
            --output_dir $P/results/buchwald_hartwig --csv_path $BH_CSV \
            --gfn_episodes 3000 --proxy_epochs 200 --temp 4.0" | jid)
  BH_DEPS="$BH_DEPS && done($id)"
done
BH_DEPS=${BH_DEPS# && }
bsub -J bh_analyze -q normal -n 8 -W 2:00 -M 32GB -w "$BH_DEPS" \
    -o $P/logs/bh_analyze_%J.out -e $P/logs/bh_analyze_%J.err \
    "python3 $P/experiments/buchwald_hartwig/run_experiment.py --mode analyze \
        --n_train 50 --n_test 100 --pce_degree 3 --pca_dim 5 --csv_path $BH_CSV \
        --output_dir $P/results/buchwald_hartwig && \
     python3 $P/decision_studies/common/analyze_ensemble.py --members_dir $P/results/buchwald_hartwig \
        --n_train 50 --n_test 100 --degree 3 --pca_dims 5,7,10,15 --out $P/results/outputs/bh"
echo "BH: resubmitted 46,47; analysis gated on [$BH_DEPS]"

# ---- Sachs: members 16, 68 ----
S_DEPS=""
for i in 16 68; do
  id=$(bsub -J sachs_gfn_re -q normal -n 1 -gpu "num=1:mode=exclusive_process" \
        -W 2:00 -M 16GB -o $P/logs/sachs_re_${i}_%J.out -e $P/logs/sachs_re_${i}_%J.err \
        "python3 $P/experiments/sachs_causal/train_single_member.py --member_id $i \
            --output_dir $P/results/sachs --gfn_episodes 3000 --data_path $SACHS_CSV" | jid)
  S_DEPS="$S_DEPS && done($id)"
done
S_DEPS=${S_DEPS# && }
bsub -J sachs_analyze -q normal -n 8 -W 3:00 -M 16GB -w "$S_DEPS" \
    -o $P/logs/sachs_analyze_%J.out -e $P/logs/sachs_analyze_%J.err \
    "python3 $P/experiments/sachs_causal/run_experiment.py --mode analyze \
        --n_train 30 --n_test 50 --pce_degree 5 --pca_dim 2 \
        --output_dir $P/results/sachs --data_path $SACHS_CSV && \
     python3 $P/decision_studies/common/analyze_ensemble.py --members_dir $P/results/sachs \
        --n_train 30 --n_test 50 --degree 3 --pca_dims 2,5,10 --n_boot 100 --out $P/results/outputs/sachs"
echo "Sachs: resubmitted 16,68; analysis gated on [$S_DEPS]"

# ---- Molecular design: members 23, 58 ----
M_DEPS=""
for i in 23 58; do
  id=$(bsub -J moldesign_gfn_re -q normal -n 1 -gpu "num=1:mode=exclusive_process" \
        -W 2:00 -M 16GB -o $P/logs/mol_re_${i}_%J.out -e $P/logs/mol_re_${i}_%J.err \
        "python3 $P/experiments/molecular_design/train_single_member.py --member_id $i \
            --output_dir $P/results/molecular_design --gfn_episodes 3000" | jid)
  M_DEPS="$M_DEPS && done($id)"
done
M_DEPS=${M_DEPS# && }
bsub -J moldesign_analyze -q normal -n 8 -W 2:00 -M 16GB -w "$M_DEPS" \
    -o $P/logs/moldesign_analyze_%J.out -e $P/logs/moldesign_analyze_%J.err \
    "python3 $P/experiments/molecular_design/run_experiment.py --mode analyze \
        --n_train 30 --n_test 50 --pce_degree 5 --pca_dim 2 \
        --output_dir $P/results/molecular_design && \
     python3 $P/decision_studies/common/analyze_ensemble.py --members_dir $P/results/molecular_design \
        --n_train 30 --n_test 50 --degree 3 --pca_dims 2,5,10 --out $P/results/outputs/moldesign"
echo "Moldesign: resubmitted 23,58; analysis gated on [$M_DEPS]"
echo "Monitor: bjobs -u $USER"
