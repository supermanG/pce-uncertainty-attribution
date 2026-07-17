"""Train a single ensemble member (proxy + GFlowNet) for LSF array dispatch.

Analogous to experiments/buchwald_hartwig/train_single_gfn.py.
Called by lsf/submit_moldesign.sh with --member_id $LSB_JOBINDEX.

Workflow:
  1. Run setup once to generate the shared reference dataset:
         python train_single_member.py --setup --output_dir results/molecular_design

  2. Submit the LSF array (one job per member):
         bsub < lsf/submit_moldesign.sh
     Each array element calls:
         python train_single_member.py --member_id $LSB_JOBINDEX \
                --output_dir results/molecular_design

  3. After all jobs complete, collect and analyse:
         python run_experiment.py --mode analyze \
                --output_dir results/molecular_design

Each job trains:
  - One RewardProxy MLP on train_fraction (30%) of the reference dataset,
    using member_id as the random seed (so each member sees a different
    training subset, producing the ensemble diversity that PCE captures).
  - One MolGFlowNet trained on that proxy via trajectory-balance loss.

Saves member_{member_id:04d}.npz to {output_dir}/members/.
"""
import argparse, json, os, sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    print("PyTorch not found."); sys.exit(1)

from experiments.molecular_design.run_experiment import (
    setup_phase1,
    train_and_save_member,
    FRAGMENT_NAMES, N_FRAGMENTS, N_STEPS,
)


# ---------------------------------------------------------------------------
# Design constants (override via CLI; must match analysis call in run_experiment)
# ---------------------------------------------------------------------------
GFN_EPISODES   = 3000
PROXY_EPOCHS   = 300
TRAIN_FRACTION = 0.3
TEMP           = 4.0
N_REF          = 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one molecular-design GFlowNet ensemble member."
    )
    parser.add_argument(
        "--member_id", type=int, default=None,
        help="LSB_JOBINDEX (0-indexed member ID). Required unless --setup is used.",
    )
    parser.add_argument("--output_dir",     default="results/molecular_design")
    parser.add_argument("--gfn_episodes",   type=int,   default=GFN_EPISODES)
    parser.add_argument("--proxy_epochs",   type=int,   default=PROXY_EPOCHS)
    parser.add_argument("--train_fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--temp",           type=float, default=TEMP)
    parser.add_argument("--n_ref",          type=int,   default=N_REF)
    parser.add_argument(
        "--setup", action="store_true",
        help="Run Phase 1 setup (generate reference dataset). "
             "Call once before submitting the LSF array.",
    )
    args = parser.parse_args()

    if args.setup:
        setup_phase1(args.output_dir, n_ref=args.n_ref)
        sys.exit(0)

    if args.member_id is None:
        parser.error("--member_id is required when not using --setup")

    # Load metadata and shared reference dataset written during setup
    meta_path = os.path.join(args.output_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(
            f"Metadata not found at {meta_path}.\n"
            f"Run: python train_single_member.py --setup --output_dir {args.output_dir}"
        )
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    ref_molecules = np.load(
        os.path.join(args.output_dir, "ref_molecules.npy"), allow_pickle=True
    ).tolist()
    ref_rewards = np.load(os.path.join(args.output_dir, "ref_rewards.npy"))

    t0 = time.time()
    train_and_save_member(
        member_id=args.member_id,
        ref_molecules=ref_molecules,
        ref_rewards=ref_rewards,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        gfn_episodes=args.gfn_episodes,
        proxy_epochs=args.proxy_epochs,
        temp=args.temp,
        device=device,
    )
    print(f"Member {args.member_id} total wall time: {time.time()-t0:.1f}s on {device}")
