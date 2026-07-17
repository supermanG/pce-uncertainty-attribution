"""Train a single ensemble member (proxy + GFlowNet) for SLURM array dispatch.

Called by slurm/bh_train_array.sh with --member_id $SLURM_ARRAY_TASK_ID.

Usage:
    python train_single_gfn.py --member_id 42 --output_dir results/buchwald_hartwig

Each job:
  1. Loads the shared reference trajectory from {output_dir}/ref_traj.json
     (created by setup_phase1.py before the array is submitted)
  2. Trains a yield proxy on train_fraction of the real dataset (seed=member_id)
  3. Trains a GFlowNet on that proxy
  4. Saves policy + proxy outputs to {output_dir}/members/member_{member_id:04d}.npz
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

from experiments.buchwald_hartwig.run_experiment import (
    load_dataset, train_and_save_member, ReactionGFlowNet
)


def _git_sha() -> str:
    """Current commit SHA for provenance, or 'unknown' outside a repo."""
    import subprocess
    try:
        root = str(Path(__file__).resolve().parent.parent.parent)
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def setup_phase1(
    output_dir: str,
    csv_path: str = None,
    train_fraction: float = 0.3,
    n_ref_reactions: int = 500,
    seed: int = 0,
    gfn_episodes: int = 3000,
    proxy_epochs: int = 200,
    temp: float = 4.0,
) -> None:
    """Pre-compute shared reference trajectory and save metadata + run config.

    Must be called ONCE before submitting the cluster array. Writes both
    metadata.json (shared reference state) and config.json (full provenance:
    git SHA, all hyperparameters, dataset hash) so every table/figure is
    regenerable from an archived run.
    """
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(meta_path):
        print(f"Metadata already exists at {meta_path}. Delete to re-run setup.")
        return

    ds = load_dataset(csv_path)
    rng = np.random.RandomState(seed)

    # Reference reactions for PCA (saved so all members use the same set)
    ref_idx = rng.choice(len(ds["reactions"]),
                          min(n_ref_reactions, len(ds["reactions"])), replace=False)
    ref_rxns = ds["reactions"][ref_idx]

    # Reference trajectory: argmax of true yields (or use proxy of member 0)
    ref_traj = [int(x) for x in ds["reactions"][np.argmax(ds["yields"])].tolist()]
    print(f"Reference trajectory: {ref_traj} (yield={ds['yields'].max():.1f}%)")

    metadata = {
        "csv_path": csv_path or "synthetic",
        "n_components": ds["n_components"],
        "component_names": ds["component_names"],
        "ref_traj": ref_traj,
        "ref_reaction_indices": ref_idx.tolist(),
        "dataset_source": ds["source"],
        "train_fraction": train_fraction,
        "n_ref_reactions": len(ref_idx),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    # Save reference reactions (used for PCA in Phase 2)
    np.save(os.path.join(output_dir, "ref_reactions.npy"), ref_rxns)

    # Full provenance for regenerating every downstream table/figure
    import hashlib
    yld = ds["yields"]
    config = {
        "experiment": "buchwald_hartwig",
        "git_sha": _git_sha(),
        "dataset_source": ds["source"],
        "dataset_n_reactions": int(len(yld)),
        "dataset_yield_sha1": hashlib.sha1(
            np.ascontiguousarray(yld).tobytes()).hexdigest(),
        "train_fraction": train_fraction,
        "n_ref_reactions": len(ref_idx),
        "gfn_episodes": gfn_episodes,
        "proxy_epochs": proxy_epochs,
        "temp": temp,
        "setup_seed": seed,
        "seed_convention": "main ensemble: proxy_seed = gfn_seed = member_id; "
                           "training-noise control: proxy_seed fixed, gfn_seed = member_id",
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Phase 1 setup complete. Metadata + config.json (git {config['git_sha'][:8]}) "
          f"saved to {output_dir}")
    print(f"Now submit: bash lsf/submit_bh.sh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--member_id", type=int, default=None,
                        help="SLURM_ARRAY_TASK_ID (0-indexed member); not needed with --setup")
    parser.add_argument("--output_dir", default="results/buchwald_hartwig")
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--gfn_episodes", type=int, default=3000)
    parser.add_argument("--proxy_epochs", type=int, default=200)
    parser.add_argument("--temp", type=float, default=4.0)
    parser.add_argument("--proxy_seed", type=int, default=None,
                        help="Reward-realisation seed (default: member_id). Fix this "
                             "across members for the training-noise control ensemble.")
    parser.add_argument("--gfn_seed", type=int, default=None,
                        help="GFlowNet training seed (default: member_id).")
    parser.add_argument("--setup", action="store_true",
                        help="Run Phase 1 setup (call once before array submission)")
    args = parser.parse_args()

    if args.setup:
        setup_phase1(args.output_dir, args.csv_path,
                     gfn_episodes=args.gfn_episodes, proxy_epochs=args.proxy_epochs,
                     temp=args.temp)
        sys.exit(0)

    if args.member_id is None:
        parser.error("--member_id is required when not using --setup")

    # Load metadata written during setup
    meta_path = os.path.join(args.output_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"Metadata not found at {meta_path}. Run: python train_single_gfn.py --setup first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    ds = load_dataset(meta.get("csv_path") if meta.get("dataset_source") == "real" else None)
    ref_traj = meta["ref_traj"]
    ref_rxns = np.load(os.path.join(args.output_dir, "ref_reactions.npy"))

    t0 = time.time()
    train_and_save_member(
        member_id=args.member_id,
        dataset=ds,
        ref_traj=ref_traj,
        ref_reactions=ref_rxns,
        output_dir=args.output_dir,
        train_fraction=meta["train_fraction"],
        gfn_episodes=args.gfn_episodes,
        device=device,
        proxy_epochs=args.proxy_epochs,
        temp=args.temp,
        proxy_seed=args.proxy_seed,
        gfn_seed=args.gfn_seed,
    )
    print(f"Member {args.member_id} total wall time: {time.time()-t0:.1f}s")
