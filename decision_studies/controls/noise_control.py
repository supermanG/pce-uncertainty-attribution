"""Training-noise control for molecular design.

Trains a control ensemble with the reward FIXED (proxy_seed=0) and only the
GFlowNet training seed varying, then compares its policy spread to the main
ensemble (where both reward and training vary). Decomposition:

    Var_total (main)   = Var_reward + Var_training
    Var_training       = control ensemble spread (reward held fixed)
    training fraction  = Var_training / Var_total

Spread per step is the trace of the policy covariance across members
(sum of per-action variances). If the training fraction is large, the molecular-
design ensemble's variation is mostly training noise, the reward->policy signal is
weak, and the experiment should be reframed as qualitative rather than presented as
a reward-uncertainty decomposition.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def trace_cov_per_step(members_dir):
    from decision_studies.common.analyze_ensemble import load_ensemble
    pol, _, ns, n = load_ensemble(members_dir)
    return {s: float(pol[s].var(axis=0).sum()) for s in range(ns)}, n


def train_control(control_dir, n_control):
    from experiments.molecular_design.run_experiment import setup_phase1, train_and_save_member
    if not os.path.exists(os.path.join(control_dir, "metadata.json")):
        setup_phase1(control_dir, n_ref=500)   # deterministic seed=0 -> same ref dataset
    ref_m = np.load(os.path.join(control_dir, "ref_molecules.npy"), allow_pickle=True).tolist()
    ref_r = np.load(os.path.join(control_dir, "ref_rewards.npy"))
    for i in range(n_control):
        train_and_save_member(i, ref_m, ref_r, control_dir,
                              proxy_seed=0, gfn_seed=i)   # FIXED reward, varying training


def report(main_dir, control_dir):
    main, nm = trace_cov_per_step(main_dir)
    ctrl, nc = trace_cov_per_step(control_dir)
    ns = min(len(main), len(ctrl))
    print(f"\nMAIN ensemble: {nm} members (reward+training vary)")
    print(f"CONTROL ensemble: {nc} members (reward FIXED, training varies)")
    print(f"{'step':>4} | {'Var_total':>10} | {'Var_train':>10} | {'train frac':>10}")
    print("-" * 44)
    fracs = []
    for s in range(ns):
        vt, vtr = main[s], ctrl[s]
        f = vtr / vt if vt > 1e-12 else float("nan")
        fracs.append(f)
        print(f"{s:>4} | {vt:>10.3e} | {vtr:>10.3e} | {f:>10.2f}")
    mf = float(np.nanmean(fracs))
    print("-" * 44)
    print(f"mean training-noise fraction = {mf:.2f}")
    verdict = ("training noise DOMINATES -> weak reward->policy signal; reframe/drop"
               if mf > 0.6 else
               "reward drives most spread -> signal is real" if mf < 0.35 else
               "mixed: training noise is a substantial fraction")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_dir", default="results/cluster_run/results/molecular_design")
    ap.add_argument("--control_dir", default="results/cluster_run/results/moldesign_noise_control")
    ap.add_argument("--n_control", type=int, default=15)
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    if args.train:
        train_control(args.control_dir, args.n_control)
    report(args.main_dir, args.control_dir)
