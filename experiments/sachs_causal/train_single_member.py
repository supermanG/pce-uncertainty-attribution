"""Train a single Sachs ensemble member for LSF array dispatch.

Each member uses a different subsample of the synthetic Sachs data (seed = member_id + 100).
Saves per-step policy vectors along the ground-truth reference trajectory.

Usage:
    python3 train_single_member.py --member_id 0 --output_dir results/sachs
    python3 train_single_member.py --setup --output_dir results/sachs
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

from experiments.sachs_causal.run_experiment import (
    generate_sachs_data, load_sachs_data, DAGGFlowNet, train_dag_gfn,
    greedy_rollout, GT_EDGES, VARS
)

DEFAULT_REAL_CSV = str(Path(__file__).resolve().parent.parent.parent / "data" / "sachs_real.csv")


def _git_sha() -> str:
    import subprocess
    try:
        root = str(Path(__file__).resolve().parent.parent.parent)
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Design constants (must match lsf/submit_sachs.sh and run_experiment.py)
# ---------------------------------------------------------------------------
N_TOTAL   = 2000   # total synthetic observations drawn once
N_OBS     = 200    # observations per subsample
MAX_EDGES = 12     # reference trajectory length
GFN_EPISODES = 3000


def setup_sachs(output_dir: str, seed: int = 0, data_path: str = DEFAULT_REAL_CSV) -> None:
    """Pre-compute shared full dataset and reference trajectory metadata.

    Loads the real Sachs flow cytometry data (853 observations) when the CSV is
    available, matching the manuscript; falls back to synthetic only if absent.
    Trains a pilot GFlowNet on the first data subsample to derive the reference
    trajectory via greedy rollout, avoiding ground-truth leak.
    """
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(meta_path):
        print(f"Metadata already exists at {meta_path}. Delete to re-run setup.")
        return

    # Shared full dataset: real when available (manuscript consistency)
    full_data, is_real = load_sachs_data(data_path=data_path, n_total=N_TOTAL)
    np.save(os.path.join(output_dir, "full_data.npy"), full_data)

    # Train pilot GFlowNet on first subsample for reference trajectory
    rng = np.random.RandomState(100)  # member 0's seed
    pilot_data = full_data[rng.choice(len(full_data), N_OBS, replace=False)]
    print("  Training pilot GFlowNet for reference trajectory...")
    pilot_gfn = DAGGFlowNet(n_vars=11).to(device)
    pilot_gfn = train_dag_gfn(pilot_gfn, pilot_data, n_ep=GFN_EPISODES,
                               max_e=MAX_EDGES, device=device)
    ref_traj = greedy_rollout(pilot_gfn, max_edges=MAX_EDGES)
    if not ref_traj:
        print("  WARNING: pilot produced empty trajectory, using GT_EDGES fallback")
        ref_traj = [(int(i), int(j)) for (i, j) in GT_EDGES[:MAX_EDGES]]
    else:
        ref_traj = [(int(i), int(j)) for (i, j) in ref_traj]
    gt_overlap = len(set(ref_traj) & set(GT_EDGES))
    print(f"  Reference trajectory: {len(ref_traj)} edges ({gt_overlap} overlap with GT)")

    metadata = {
        "n_total": int(full_data.shape[0]),
        "n_obs": N_OBS,
        "max_edges": MAX_EDGES,
        "ref_traj": ref_traj,
        "variables": VARS,
        "seed": seed,
        "data_source": "real" if is_real else "synthetic",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    import hashlib
    config = {
        "experiment": "sachs_causal",
        "git_sha": _git_sha(),
        "data_source": "real" if is_real else "synthetic",
        "data_path": data_path if is_real else None,
        "n_total": int(full_data.shape[0]),
        "n_obs": N_OBS,
        "max_edges": MAX_EDGES,
        "gfn_episodes": GFN_EPISODES,
        "data_sha1": hashlib.sha1(np.ascontiguousarray(full_data).tobytes()).hexdigest(),
        "subsample_seed_convention": "subsample seed = member_id; gfn_seed = member_id",
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Setup complete. Full data: {full_data.shape} ({config['data_source']}), "
          f"ref_traj: {len(ref_traj)} edges, git {config['git_sha'][:8]}")
    print("Now submit: bash lsf/submit_sachs.sh")


def train_and_save_member(
    member_id: int,
    full_data: np.ndarray,
    ref_traj: list,
    output_dir: str,
    n_obs: int = N_OBS,
    gfn_episodes: int = GFN_EPISODES,
    device: str = "cpu",
    gfn_seed: int = None,
    subsample_seed: int = None,
) -> None:
    """Train one GFlowNet on a subsampled dataset and save its policies.

    The subsample (seed member_id + 100, unchanged so the baseline analysis stays
    coherent) fixes both the BGe-score reward and the covariance parameterisation.
    gfn_seed (default member_id) seeds training so runs are reproducible
    and reward vs training noise can be separated. proxy_outputs stores the
    flattened sample covariance, the PCA input, so the extended-analysis driver
    consumes Sachs members directly.
    """
    gfn_seed = member_id if gfn_seed is None else gfn_seed
    # subsample_seed fixes which 200 observations (hence the BGe reward); fix it
    # across members and vary only gfn_seed for the training-noise control.
    subsample_seed = (member_id + 100) if subsample_seed is None else subsample_seed
    os.makedirs(os.path.join(output_dir, "members"), exist_ok=True)
    out_path = os.path.join(output_dir, "members", f"member_{member_id:04d}.npz")
    if os.path.exists(out_path):
        print(f"Member {member_id} already exists, skipping.")
        return

    rng = np.random.RandomState(subsample_seed)
    idx = rng.choice(len(full_data), n_obs, replace=False)
    data_sub = full_data[idx]
    cov_flat = np.cov(data_sub.T).flatten().astype(np.float32)   # (121,) PCA input

    # Seed before constructing the network so init + training are reproducible.
    torch.manual_seed(gfn_seed)
    np.random.seed(gfn_seed)
    gfn = DAGGFlowNet(n_vars=11).to(device)
    t0 = time.time()
    gfn = train_dag_gfn(gfn, data_sub, n_ep=gfn_episodes, max_e=len(ref_traj), device=device)
    print(f"  Member {member_id} trained in {time.time()-t0:.1f}s on {device}")

    # Extract policy at each step along reference trajectory
    policies = []
    adj = np.zeros((11, 11))
    for step, (ei, ej) in enumerate(ref_traj):
        pol = gfn.get_policy(adj, step)   # (n_actions,)
        policies.append(pol)
        adj[ei, ej] = 1.0

    np.savez(out_path,
             member_id=member_id,
             policies=np.array(policies),   # (n_steps, n_actions)
             proxy_outputs=cov_flat,        # (121,) flattened sample covariance
             gfn_seed=np.array([gfn_seed]),
             subsample_seed=np.array([subsample_seed]),
             ref_traj=np.array(ref_traj))
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--member_id", type=int, default=None)
    parser.add_argument("--output_dir", default="results/sachs")
    parser.add_argument("--n_obs", type=int, default=N_OBS)
    parser.add_argument("--gfn_episodes", type=int, default=GFN_EPISODES)
    parser.add_argument("--gfn_seed", type=int, default=None,
                        help="GFlowNet training seed (default: member_id).")
    parser.add_argument("--data_path", default=DEFAULT_REAL_CSV,
                        help="Path to sachs_real.csv (real data). Absent -> synthetic.")
    parser.add_argument("--setup", action="store_true")
    args = parser.parse_args()

    if args.setup:
        setup_sachs(args.output_dir, data_path=args.data_path)
        sys.exit(0)

    if args.member_id is None:
        parser.error("--member_id is required when not using --setup")

    meta_path = os.path.join(args.output_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"Metadata not found. Run: python3 train_single_member.py --setup first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    full_data = np.load(os.path.join(args.output_dir, "full_data.npy"))
    ref_traj = [tuple(e) for e in meta["ref_traj"]]

    train_and_save_member(
        member_id=args.member_id,
        full_data=full_data,
        ref_traj=ref_traj,
        output_dir=args.output_dir,
        n_obs=args.n_obs,
        gfn_episodes=args.gfn_episodes,
        device=device,
        gfn_seed=args.gfn_seed,
    )
