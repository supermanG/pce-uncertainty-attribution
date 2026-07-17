#!/bin/bash
# LSF array job: train one GFlowNet ensemble member per task.
# Submit AFTER running setup: python3 experiments/buchwald_hartwig/train_single_gfn.py --setup
#
# Usage:
#   bsub < lsf/bh_train_array.sh
#
# 150 members (50 train + 100 test), one GPU each, ~40 min wall clock.
# Total: 150 GPU-hours, runs in parallel.

#BSUB -J bh_gfn[0-149]
#BSUB -o $HOME/uq-gflownet/logs/bh_train_%J_%I.out
#BSUB -e $HOME/uq-gflownet/logs/bh_train_%J_%I.err
#BSUB -q normal
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:30
#BSUB -M 32GB
#BSUB -m "cccxc701 cccxc702 cccxc704 cccxc710 cccxc711 cccxc712"

PROJECT_DIR=$HOME/uq-gflownet
cd $PROJECT_DIR
mkdir -p logs results/buchwald_hartwig

echo "Project dir: $PROJECT_DIR"
echo "Member ID:   $LSB_JOBINDEX"
echo "Host:        $(hostname)"
echo "GPU:         $CUDA_VISIBLE_DEVICES"

python3 experiments/buchwald_hartwig/train_single_gfn.py \
    --member_id $LSB_JOBINDEX \
    --output_dir results/buchwald_hartwig \
    --csv_path $PROJECT_DIR/experiments/buchwald_hartwig/data/data_table.csv \
    --gfn_episodes 3000 \
    --proxy_epochs 200 \
    --temp 4.0

echo "Member $LSB_JOBINDEX finished at $(date)"
