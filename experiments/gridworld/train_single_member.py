"""Standalone LSF array-job script: train one grid-world GFlowNet member.

Invocation (LSF):
    bsub -J "gw_discrete[0-149]" python train_single_member.py \
        --mode discrete \
        --member_id $LSB_JOBINDEX \
        --output_dir results/gridworld/discrete \
        --seed $LSB_JOBINDEX

The script:
1. Samples the reward configuration for the given member_id (seeded
   deterministically so that run_experiment.py::analyze_from_members
   can reconstruct the same mu via PCA).
2. Trains one GridGFlowNet.
3. Saves member_{member_id:04d}.npz to output_dir/members/.

CPU-only; expected wall-time <2 min.
"""
import os, sys, argparse, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.gridworld.run_experiment import (
    DISCRETE_ACTIONS, CONT_ACTIONS,
    DISCRETE_N_STEPS, CONT_N_STEPS,
    sample_reward_configs,
    train_single_member,
    GRID_SIZE,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train a single grid-world GFlowNet member for LSF array job."
    )
    parser.add_argument("--mode",       choices=["discrete", "continuous"],
                        required=True)
    parser.add_argument("--member_id",  type=int, required=True,
                        help="Zero-based member index (= LSB_JOBINDEX).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory; member saved to output_dir/members/.")
    parser.add_argument("--seed",       type=int, default=None,
                        help="RNG seed; defaults to member_id.")
    parser.add_argument("--n_episodes", type=int, default=400,
                        help="Number of GFlowNet training episodes.")
    parser.add_argument("--n_total",    type=int, default=150,
                        help="Total ensemble size (n_train + n_test); used to "
                             "draw a consistent reward config set.")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else args.member_id

    # Recreate the full reward config set with the same seed=0 used in
    # run_experiment.py so that member_id indexes into the same sequence.
    params, grids = sample_reward_configs(args.n_total, args.mode, seed=0)

    if args.member_id >= args.n_total:
        raise ValueError(
            f"member_id={args.member_id} >= n_total={args.n_total}; "
            "increase --n_total or lower --member_id."
        )

    reward_grid  = grids[args.member_id]
    reward_param = params[args.member_id]

    n_actions = DISCRETE_ACTIONS if args.mode == "discrete" else CONT_ACTIONS
    n_steps   = DISCRETE_N_STEPS if args.mode == "discrete" else CONT_N_STEPS

    print(f"[member {args.member_id:04d}] mode={args.mode}  seed={seed}  "
          f"n_actions={n_actions}  n_steps={n_steps}")

    policies, _ = train_single_member(
        reward_grid, mode=args.mode, seed=seed, n_episodes=args.n_episodes
    )
    # policies: (n_steps, n_actions)

    # Reference trajectory: at each step record the greedy action index
    ref_traj_actions = np.argmax(policies, axis=1)  # (n_steps,)

    members_dir = os.path.join(args.output_dir, "members")
    os.makedirs(members_dir, exist_ok=True)

    out_path = os.path.join(members_dir, f"member_{args.member_id:04d}.npz")
    np.savez_compressed(
        out_path,
        policies=policies,           # (n_steps, n_actions)
        ref_traj=ref_traj_actions,   # (n_steps,) greedy action indices
        reward_params=reward_param,  # (4,) or (N_BUMPS,)
        member_id=np.array(args.member_id),
        seed=np.array(seed),
    )
    print(f"  Saved {out_path}  (policies shape: {policies.shape})")

    # Write a per-member JSON sidecar for traceability
    sidecar = {
        "member_id":    args.member_id,
        "mode":         args.mode,
        "seed":         seed,
        "n_episodes":   args.n_episodes,
        "reward_param": reward_param.tolist(),
        "policies_shape": list(policies.shape),
    }
    sidecar_path = os.path.join(members_dir, f"member_{args.member_id:04d}.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)


if __name__ == "__main__":
    main()
