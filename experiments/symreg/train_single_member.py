"""Standalone LSF array-job script: train one symbolic regression GFlowNet member.

Invocation (LSF):
    bsub -J "symreg[0-149]" python train_single_member.py \
        --member_id $LSB_JOBINDEX \
        --output_dir results/symreg \
        --kl_coeff1 <mu_1> \
        --kl_coeff2 <mu_2>

KL coefficients are pre-sampled centrally and passed as arguments so that each
array job is fully reproducible.  Alternatively, --seed can be used to
re-derive (mu_1, mu_2) from the same numpy RandomState used in run_experiment.py.

Saves member_{member_id:04d}.npz to output_dir/members/.
"""
import os, sys, argparse, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.symreg.run_experiment import (
    noisy_target,
    train_symreg_gfn,
    collect_policies,
    N_STEPS, K_ACTIONS, TOKENS,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def main():
    parser = argparse.ArgumentParser(
        description="Train a single symreg GFlowNet member for LSF array job."
    )
    parser.add_argument("--member_id",   type=int, required=True,
                        help="Zero-based member index (= LSB_JOBINDEX).")
    parser.add_argument("--output_dir",  type=str, required=True,
                        help="Root output directory; member saved to output_dir/members/.")
    parser.add_argument("--kl_coeff1",   type=float, default=None,
                        help="First KL coefficient mu_1 ~ N(0,1).  "
                             "If omitted, derived from --seed.")
    parser.add_argument("--kl_coeff2",   type=float, default=None,
                        help="Second KL coefficient mu_2 ~ N(0,1).  "
                             "If omitted, derived from --seed.")
    parser.add_argument("--seed",        type=int, default=None,
                        help="RNG seed; used both to derive KL coefficients "
                             "(if not supplied explicitly) and to seed the "
                             "GFlowNet weight initialisation.")
    parser.add_argument("--n_episodes",  type=int, default=1000,
                        help="Number of GFlowNet training episodes.")
    parser.add_argument("--n_total",     type=int, default=150,
                        help="Total ensemble size used when deriving KL "
                             "coefficients via seed (must match run_experiment.py).")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else args.member_id

    # Derive KL coefficients if not supplied on the command line
    if args.kl_coeff1 is not None and args.kl_coeff2 is not None:
        mu1 = args.kl_coeff1
        mu2 = args.kl_coeff2
    else:
        # Replicate the central draw from run_symreg_experiment (seed=0)
        rng_main  = np.random.RandomState(0)
        kl_coeffs = rng_main.randn(args.n_total, 2)
        if args.member_id >= args.n_total:
            raise ValueError(
                f"member_id={args.member_id} >= n_total={args.n_total}; "
                "pass --kl_coeff1/2 directly or increase --n_total."
            )
        mu1, mu2 = float(kl_coeffs[args.member_id, 0]), float(kl_coeffs[args.member_id, 1])

    print(f"[member {args.member_id:04d}]  mu1={mu1:.4f}  mu2={mu2:.4f}  seed={seed}")

    f_noisy = noisy_target(mu1, mu2)   # (N_POINTS,)

    device = "cpu"
    if HAS_TORCH and torch.cuda.is_available():
        device = "cuda"

    gfn = train_symreg_gfn(
        f_noisy,
        n_episodes=args.n_episodes,
        lr=1e-3,
        device=device,
        seed=seed,
    )
    policies = collect_policies(gfn)   # (N_STEPS, K_ACTIONS)

    # Reference trajectory: greedy decode action indices
    ref_traj_actions = np.argmax(policies, axis=1)   # (N_STEPS,)

    members_dir = os.path.join(args.output_dir, "members")
    os.makedirs(members_dir, exist_ok=True)

    out_path = os.path.join(members_dir, f"member_{args.member_id:04d}.npz")
    np.savez_compressed(
        out_path,
        policies=policies,          # (N_STEPS, K_ACTIONS)
        kl_coeffs=np.array([mu1, mu2]),
        ref_traj=ref_traj_actions,  # (N_STEPS,) greedy action indices
        member_id=np.array(args.member_id),
        seed=np.array(seed),
    )
    print(f"  Saved {out_path}  (policies shape: {policies.shape})")

    # Per-member JSON sidecar
    sidecar = {
        "member_id":       args.member_id,
        "kl_coeff1":       mu1,
        "kl_coeff2":       mu2,
        "seed":            seed,
        "n_episodes":      args.n_episodes,
        "policies_shape":  list(policies.shape),
        "tokens":          TOKENS,
        "greedy_tokens":   [TOKENS[int(a)] for a in ref_traj_actions],
    }
    sidecar_path = os.path.join(members_dir, f"member_{args.member_id:04d}.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)


if __name__ == "__main__":
    main()
