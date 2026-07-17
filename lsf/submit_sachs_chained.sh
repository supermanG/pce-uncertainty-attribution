#!/bin/bash
# Sachs pipeline, fully non-interactive (robust version).
#
# The Sachs setup trains a pilot GFlowNet to derive the reference trajectory.
# Running that on the login node inside an ssh session is slow and fragile, so
# here setup runs as its own GPU job and the member array + analysis are chained
# after it with LSF dependencies. Nothing heavy runs on the login node.
#
# Usage:  bash lsf/submit_sachs_chained.sh
set -e
PROJECT_DIR=$HOME/uq-gflownet
cd $PROJECT_DIR
mkdir -p logs results/sachs

DATA=$PROJECT_DIR/data/sachs_real.csv
DATA_ARG=""
[ -f "$DATA" ] && DATA_ARG="--data_path $DATA"

# Step 1: setup as a GPU job (pilot trajectory + real-data full_data + config).
SETUP_ID=$(bsub -J sachs_setup -q normal -n 1 \
    -gpu "num=1:mode=exclusive_process" -W 0:30 -M 16GB \
    -o "$PROJECT_DIR/logs/sachs_setup_%J.out" \
    -e "$PROJECT_DIR/logs/sachs_setup_%J.err" \
    "python3 $PROJECT_DIR/experiments/sachs_causal/train_single_member.py \
        --setup --output_dir $PROJECT_DIR/results/sachs $DATA_ARG" \
    | grep -oE 'Job <[0-9]+>' | grep -oE '[0-9]+')
echo "  setup job: $SETUP_ID"

# Step 2: 80 member jobs, each waiting for setup to finish.
N_MEMBERS=80
for i in $(seq 0 $((N_MEMBERS - 1))); do
    bsub -J sachs_gfn -q normal -n 1 \
         -gpu "num=1:mode=exclusive_process" -W 1:00 -M 16GB \
         -w "done($SETUP_ID)" \
         -o "$PROJECT_DIR/logs/sachs_train_${i}_%J.out" \
         -e "$PROJECT_DIR/logs/sachs_train_${i}_%J.err" \
         "python3 $PROJECT_DIR/experiments/sachs_causal/train_single_member.py \
             --member_id $i --output_dir $PROJECT_DIR/results/sachs \
             --gfn_episodes 3000 $DATA_ARG" > /dev/null
done
echo "  submitted $N_MEMBERS member jobs (sachs_gfn), waiting on setup."

# Step 3: analysis (baseline + extended), waiting for all members.
ANALYZE_OUT=$(bsub -J sachs_analyze -q normal -n 8 -W 3:00 -M 16GB \
    -w "done(sachs_gfn)" \
    -o "$PROJECT_DIR/logs/sachs_analyze_%J.out" \
    -e "$PROJECT_DIR/logs/sachs_analyze_%J.err" \
    "python3 $PROJECT_DIR/experiments/sachs_causal/run_experiment.py \
        --mode analyze --n_train 30 --n_test 50 --pce_degree 5 --pca_dim 2 \
        --output_dir $PROJECT_DIR/results/sachs $DATA_ARG && \
     python3 $PROJECT_DIR/decision_studies/common/analyze_ensemble.py \
        --members_dir $PROJECT_DIR/results/sachs \
        --n_train 30 --n_test 50 --degree 3 --pca_dims 2,5,10 --n_boot 100 \
        --out $PROJECT_DIR/results/outputs/sachs")
echo "  $ANALYZE_OUT"
echo "Done: bjobs -J sachs_setup ; bjobs -J sachs_gfn ; bjobs -J sachs_analyze"
