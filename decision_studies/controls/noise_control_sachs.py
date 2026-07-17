"""Training-noise control for Sachs.

Fixes the reward (the 200-observation subsample, subsample_seed=100) and varies only
the GFlowNet training seed, then decomposes the ensemble spread. Because the Sachs
signal was recovered in the ILR dominant-vs-rest balance (relMSE 0.90), we report the
training fraction in BOTH the raw policy (trace-of-covariance) and that balance space;
the balance-space fraction is the decisive one. training fraction = Var(fixed-reward
control) / Var(main); small => the recovered signal is reward-driven (Sachs valid).
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from decision_studies.common.analyze_ensemble import load_ensemble


def balance_fixed(P, dom):
    """ILR dominant-vs-rest balance for a fixed dominant index."""
    P = np.clip(P, 1e-12, None)
    rest = np.delete(P, dom, axis=1)
    K = P.shape[1]
    g = np.exp(np.log(rest).mean(1))
    return np.sqrt((K - 1) / K) * np.log(P[:, dom] / g)


def train_control(control_dir, main_dir, n_control):
    from experiments.sachs_causal.train_single_member import train_and_save_member
    meta = json.load(open(os.path.join(main_dir, "metadata.json")))
    ref_traj = [tuple(e) for e in meta["ref_traj"]]
    full = np.load(os.path.join(main_dir, "full_data.npy"))
    os.makedirs(control_dir, exist_ok=True)
    for i in range(n_control):
        train_and_save_member(i, full, ref_traj, control_dir,
                              subsample_seed=100, gfn_seed=i)   # FIXED reward


def report(main_dir, control_dir):
    pm, _, ns, nm = load_ensemble(main_dir)
    pc, _, nsc, nc = load_ensemble(control_dir)
    ns = min(ns, nsc)
    print(f"\nMAIN: {nm} members (reward+training vary)  "
          f"CONTROL: {nc} members (reward FIXED, training varies)")
    print(f"{'step':>4} | {'raw frac':>9} | {'balance frac':>12}")
    print("-" * 32)
    raw, bal = [], []
    for s in range(ns):
        M, C = pm[s], pc[s]
        rf = (C.var(0).sum()) / (M.var(0).sum() + 1e-15)
        dom = int(np.argmax(M.mean(0)))
        bt = balance_fixed(M, dom).var()
        btr = balance_fixed(C, dom).var()
        bf = btr / (bt + 1e-15)
        raw.append(rf); bal.append(bf)
        print(f"{s:>4} | {rf:>9.2f} | {bf:>12.2f}")
    mr, mb = float(np.mean(raw)), float(np.mean(bal))
    print("-" * 32)
    print(f"mean training fraction: raw={mr:.2f}  balance={mb:.2f}")
    verdict = ("training noise DOMINATES the recovered signal -> Sachs also invalid"
               if mb > 0.6 else
               "recovered balance signal is REWARD-DRIVEN -> Sachs valid" if mb < 0.4 else
               "mixed: training noise is a substantial fraction of the balance signal")
    print(f"VERDICT (balance space): {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_dir", default="results/cluster_run/results/sachs")
    ap.add_argument("--control_dir", default="results/cluster_run/results/sachs_noise_control")
    ap.add_argument("--n_control", type=int, default=10)
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    if args.train:
        train_control(args.control_dir, args.main_dir, args.n_control)
    report(args.main_dir, args.control_dir)
