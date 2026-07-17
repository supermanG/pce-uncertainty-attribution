#!/bin/bash
# Grid-world experiment: discrete + continuous modes.
# CPU-only, ~10 min wall time per member.
#
# Usage:
#   cd $HOME/uq-gflownet
#   bash lsf/submit_gridworld.sh

set -e
PROJECT_DIR=$HOME/uq-gflownet
cd $PROJECT_DIR
mkdir -p logs results/gridworld/discrete/members results/gridworld/continuous/members

echo "=========================================="
echo "  UQ-GFlowNet: Grid-world (discrete + continuous)"
echo "=========================================="

# ---- Discrete: 150 members (50 train + 100 test) ----
# Submit individual jobs and collect last job id for dependency
DISC_LAST_JID=""
for i in $(seq 0 149); do
    JID=$(bsub \
        -J "gw_disc_${i}" \
        -q normal \
        -n 2 \
        -W 0:30 \
        -M 4GB \
        -o "$PROJECT_DIR/logs/gw_disc_${i}_%J.out" \
        -e "$PROJECT_DIR/logs/gw_disc_${i}_%J.err" \
        "python3 $PROJECT_DIR/experiments/gridworld/train_single_member.py \
            --mode discrete \
            --member_id $i \
            --output_dir $PROJECT_DIR/results/gridworld/discrete" \
    | grep -oP '(?<=Job <)\d+')
    DISC_LAST_JID=$JID
done
echo "  Discrete train jobs submitted (150 members, last JID=$DISC_LAST_JID)"

# Analyze waits for the last train job (others will be done by then)
bsub \
    -J gw_disc_analyze \
    -q normal \
    -n 4 \
    -W 0:20 \
    -M 8GB \
    -w "done($DISC_LAST_JID)" \
    -o "$PROJECT_DIR/logs/gw_disc_analyze_%J.out" \
    -e "$PROJECT_DIR/logs/gw_disc_analyze_%J.err" \
    "python3 $PROJECT_DIR/experiments/gridworld/run_experiment.py \
        --run_mode analyze --mode discrete \
        --n_train 50 --n_test 100 \
        --output_dir $PROJECT_DIR/results/gridworld/discrete"

# ---- Continuous: 150 members ----
CONT_LAST_JID=""
for i in $(seq 0 149); do
    JID=$(bsub \
        -J "gw_cont_${i}" \
        -q normal \
        -n 2 \
        -W 0:20 \
        -M 4GB \
        -o "$PROJECT_DIR/logs/gw_cont_${i}_%J.out" \
        -e "$PROJECT_DIR/logs/gw_cont_${i}_%J.err" \
        "python3 $PROJECT_DIR/experiments/gridworld/train_single_member.py \
            --mode continuous \
            --member_id $i \
            --output_dir $PROJECT_DIR/results/gridworld/continuous" \
    | grep -oP '(?<=Job <)\d+')
    CONT_LAST_JID=$JID
done
echo "  Continuous train jobs submitted (150 members, last JID=$CONT_LAST_JID)"

bsub \
    -J gw_cont_analyze \
    -q normal \
    -n 4 \
    -W 0:20 \
    -M 8GB \
    -w "done($CONT_LAST_JID)" \
    -o "$PROJECT_DIR/logs/gw_cont_analyze_%J.out" \
    -e "$PROJECT_DIR/logs/gw_cont_analyze_%J.err" \
    "python3 $PROJECT_DIR/experiments/gridworld/run_experiment.py \
        --run_mode analyze --mode continuous \
        --n_train 50 --n_test 100 \
        --output_dir $PROJECT_DIR/results/gridworld/continuous"

echo ""
echo "  Monitor: bjobs -J 'gw_*'"
echo "  Results: results/gridworld/{discrete,continuous}/results.json"
echo "=========================================="
