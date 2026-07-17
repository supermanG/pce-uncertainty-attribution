"""Multi-reward training-noise control for molecular design.

Trains R blocks, each fixing a distinct reward realisation (proxy_seed = block) and
varying the GFlowNet seed over G members. The training-noise fraction is then the mean
over blocks of the within-block (fixed-reward) spread divided by the main ensemble's
total spread, averaged over steps, giving a reward-realisation-averaged estimate rather
than the single-reward estimate. Spread = trace of the per-step policy covariance.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def trace_cov(members_dir):
    from decision_studies.common.analyze_ensemble import load_ensemble
    pol, _, ns, n = load_ensemble(members_dir)
    return np.array([pol[s].var(axis=0).sum() for s in range(ns)]), n


def train_block(base_dir, block_dir, proxy_seed, n_members):
    from experiments.molecular_design.run_experiment import setup_phase1, train_and_save_member
    if not os.path.exists(os.path.join(block_dir, "metadata.json")):
        setup_phase1(block_dir, n_ref=500)   # deterministic seed=0 -> same ref dataset
    ref_m = np.load(os.path.join(block_dir, "ref_molecules.npy"), allow_pickle=True).tolist()
    ref_r = np.load(os.path.join(block_dir, "ref_rewards.npy"))
    for i in range(n_members):
        train_and_save_member(i, ref_m, ref_r, block_dir, proxy_seed=proxy_seed, gfn_seed=i)


def run(main_dir, root, reward_seeds, n_members, do_train):
    if do_train:
        for rs in reward_seeds:
            train_block(main_dir, os.path.join(root, f"reward_{rs}"), rs, n_members)
    total, nm = trace_cov(main_dir)
    within = []
    for rs in reward_seeds:
        bt, nb = trace_cov(os.path.join(root, f"reward_{rs}"))
        within.append(bt)
    within = np.array(within)                    # (R, n_steps)
    frac = within.mean(0) / (total + 1e-15)       # per-step training fraction
    print(f"MAIN total spread per step: {np.round(total,5).tolist()} (n={nm})")
    print(f"reward blocks: {reward_seeds}, {n_members} members each")
    print(f"per-step training-noise fraction: {np.round(frac,2).tolist()}")
    print(f"mean training-noise fraction (reward-averaged) = {float(frac.mean()):.2f}")
    print("VERDICT: " + ("training noise DOMINATES (reward->policy signal weak)"
                         if frac.mean() > 0.6 else "reward drives most spread"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_dir", default="results/cluster_run/results/molecular_design")
    ap.add_argument("--root", default="results/cluster_run/results/moldesign_multi_control")
    ap.add_argument("--reward_seeds", default="0,7,14")
    ap.add_argument("--n_members", type=int, default=6)
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    run(a.main_dir, a.root, [int(x) for x in a.reward_seeds.split(",")], a.n_members, a.train)
