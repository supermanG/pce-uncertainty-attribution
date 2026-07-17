"""d-scalability study: how PCE surrogate quality degrades with PCA dimension.

Sweeps PCA dimensionality d = 2, 3, 4, 5 on the gridworld experiment and
reports for each d:
  - PCA explained variance
  - Per-step Sobol MAE vs. a large-ensemble ground truth (d=2)
  - KS pass rate (fraction of actions where surrogate matches empirical)
  - Mean calibration coverage at the 90% level
  - PCE fitting time

This checks whether the framework's d=2 default
is critical, and how gracefully quality degrades in higher dimensions.

Usage:
    python experiments/gridworld/d_scalability.py
    python experiments/gridworld/d_scalability.py --n_train 80 --d_max 6
"""
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.decomposition import PCA
from core.pce_surrogate import (
    TrajectoryPCESurrogate,
    run_ks_battery,
    calibration_coverage,
)
from experiments.gridworld.run_experiment import (
    sample_reward_configs,
    train_single_member,
    DISCRETE_N_STEPS,
    GRID_SIZE,
)


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_d_scalability(
    d_values=None,
    n_train: int = 50,
    n_test: int = 100,
    pce_degree: int = 5,
    n_episodes: int = 400,
    mode: str = "discrete",
    seed: int = 0,
    output_dir: str = "results/gridworld/d_scalability",
):
    """Run the d-scalability sweep.

    Args:
        d_values:    list of PCA dimensions to test (default [2, 3, 4, 5]).
        n_train:     training ensemble size.
        n_test:      test ensemble size.
        pce_degree:  PCE polynomial degree.
        n_episodes:  GFlowNet training episodes per member.
        mode:        'discrete' or 'continuous'.
        seed:        master RNG seed.
        output_dir:  where to write results.

    Returns:
        results dict (also saved to JSON).
    """
    if d_values is None:
        d_values = [2, 3, 4, 5]

    os.makedirs(output_dir, exist_ok=True)
    n_total = n_train + n_test
    n_steps = DISCRETE_N_STEPS

    print("=" * 70)
    print("d-SCALABILITY STUDY")
    print("=" * 70)
    print(f"  PCA dimensions:  {d_values}")
    print(f"  Ensemble:        {n_train} train + {n_test} test")
    print(f"  PCE degree:      {pce_degree}")
    print(f"  Mode:            {mode}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Train the full ensemble once (shared across all d values)
    # ------------------------------------------------------------------
    print(f"Training {n_total} GFlowNet members ...")
    params, grids = sample_reward_configs(n_total, mode, seed=seed)
    flat_grids = grids.reshape(n_total, -1)

    all_policies = []
    for i in range(n_total):
        pol, _ = train_single_member(grids[i], mode=mode, seed=i + seed,
                                     n_episodes=n_episodes)
        all_policies.append(pol)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n_total}")

    tr_pol = {s: np.array([all_policies[i][s] for i in range(n_train)])
              for s in range(n_steps)}
    te_pol = {s: np.array([all_policies[n_train + i][s] for i in range(n_test)])
              for s in range(n_steps)}

    # ------------------------------------------------------------------
    # Step 2: Compute ground-truth Sobol at d=2 with full ensemble
    # ------------------------------------------------------------------
    print("\nComputing ground-truth Sobol (d=2, full ensemble) ...")
    pca_gt = PCA(n_components=2)
    mu_gt = pca_gt.fit_transform(flat_grids)
    mu_gt_all = mu_gt[:n_train]
    tsurr_gt = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(n_steps):
        tsurr_gt.fit_step(s, mu_gt_all, tr_pol[s])
    sobol_gt = tsurr_gt.sobol_all_steps()

    # ------------------------------------------------------------------
    # Step 3: Sweep over d values
    # ------------------------------------------------------------------
    sweep_results = []

    for d in d_values:
        print(f"\n--- d = {d} ---")

        pca = PCA(n_components=d)
        mu = pca.fit_transform(flat_grids)
        mu_tr = mu[:n_train]
        mu_te = mu[n_train:]
        exp_var = float(pca.explained_variance_ratio_.sum())
        print(f"  PCA explained variance: {exp_var:.3f}")

        # Fit PCE
        t0 = time.perf_counter()
        tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
        for s in range(n_steps):
            tsurr.fit_step(s, mu_tr, tr_pol[s])
        fit_time = time.perf_counter() - t0

        sobol = tsurr.sobol_all_steps()

        # KS battery
        ks = run_ks_battery(tsurr, te_pol)
        mean_ks_pass = float(np.mean([v["fraction_pass"] for v in ks.values()]))

        # Calibration at 90% level
        mc_samples = tsurr.sample_trajectory_policies(2000)
        coverages = []
        for s in range(n_steps):
            cov = calibration_coverage(mc_samples[s], te_pol[s])
            coverages.append(cov.get(0.9, 0.0))
        mean_cov_90 = float(np.mean(coverages)) if coverages else 0.0

        # Sobol MAE vs ground truth (compare first 2 dims only)
        sobol_mae_fo = []
        sobol_mae_to = []
        for s in range(n_steps):
            fo_gt = sobol_gt[s]["first_order"]   # (K-1, 2)
            to_gt = sobol_gt[s]["total_order"]   # (K-1, 2)
            fo_d = sobol[s]["first_order"][:, :2] if sobol[s]["first_order"].shape[1] >= 2 else sobol[s]["first_order"]
            to_d = sobol[s]["total_order"][:, :2] if sobol[s]["total_order"].shape[1] >= 2 else sobol[s]["total_order"]
            # Align shapes
            min_k = min(fo_gt.shape[0], fo_d.shape[0])
            min_dim = min(fo_gt.shape[1], fo_d.shape[1])
            sobol_mae_fo.append(float(np.mean(np.abs(
                fo_gt[:min_k, :min_dim] - fo_d[:min_k, :min_dim]))))
            sobol_mae_to.append(float(np.mean(np.abs(
                to_gt[:min_k, :min_dim] - to_d[:min_k, :min_dim]))))

        mean_mae_fo = float(np.mean(sobol_mae_fo))
        mean_mae_to = float(np.mean(sobol_mae_to))

        # Number of PCE terms (grows combinatorially)
        n_terms = tsurr.step_surrogates[0].coefficients[0].shape[0] if 0 in tsurr.step_surrogates else 0

        entry = {
            "d": d,
            "pca_explained_variance": exp_var,
            "fit_time_s": round(fit_time, 3),
            "n_pce_terms": n_terms,
            "ks_pass_rate": round(mean_ks_pass, 3),
            "calibration_90": round(mean_cov_90, 3),
            "sobol_mae_first_order": round(mean_mae_fo, 4),
            "sobol_mae_total_order": round(mean_mae_to, 4),
        }
        sweep_results.append(entry)

        print(f"  PCE terms:         {n_terms}")
        print(f"  Fit time:          {fit_time:.3f}s")
        print(f"  KS pass rate:      {mean_ks_pass:.3f}")
        print(f"  Calibration @90%:  {mean_cov_90:.3f}")
        print(f"  Sobol MAE (FO):    {mean_mae_fo:.4f}")
        print(f"  Sobol MAE (TO):    {mean_mae_to:.4f}")

    # ------------------------------------------------------------------
    # Step 4: Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("d-SCALABILITY SUMMARY")
    print("=" * 90)
    header = (f"{'d':>3s}  {'PCA var':>8s}  {'#terms':>6s}  {'Time(s)':>8s}  "
              f"{'KS pass':>8s}  {'Cal@90':>7s}  {'MAE(FO)':>8s}  {'MAE(TO)':>8s}")
    print(header)
    print("-" * len(header))
    for e in sweep_results:
        print(f"{e['d']:3d}  {e['pca_explained_variance']:8.3f}  "
              f"{e['n_pce_terms']:6d}  {e['fit_time_s']:8.3f}  "
              f"{e['ks_pass_rate']:8.3f}  {e['calibration_90']:7.3f}  "
              f"{e['sobol_mae_first_order']:8.4f}  {e['sobol_mae_total_order']:8.4f}")
    print()

    # Narrative
    best_d = sweep_results[0]
    worst_d = sweep_results[-1]
    degradation = worst_d["sobol_mae_total_order"] / max(best_d["sobol_mae_total_order"], 1e-10)
    time_ratio = worst_d["fit_time_s"] / max(best_d["fit_time_s"], 1e-10)
    print(f"From d={best_d['d']} to d={worst_d['d']}:")
    print(f"  - PCA explained variance: {best_d['pca_explained_variance']:.3f} -> "
          f"{worst_d['pca_explained_variance']:.3f}")
    print(f"  - PCE terms: {best_d['n_pce_terms']} -> {worst_d['n_pce_terms']} "
          f"({time_ratio:.1f}x slower)")
    print(f"  - Sobol MAE (TO): {best_d['sobol_mae_total_order']:.4f} -> "
          f"{worst_d['sobol_mae_total_order']:.4f}")
    print(f"  - KS pass rate: {best_d['ks_pass_rate']:.3f} -> "
          f"{worst_d['ks_pass_rate']:.3f}")

    results = {
        "config": {
            "d_values": d_values,
            "n_train": n_train,
            "n_test": n_test,
            "pce_degree": pce_degree,
            "n_episodes": n_episodes,
            "mode": mode,
            "seed": seed,
        },
        "sweep": sweep_results,
    }

    out_path = os.path.join(output_dir, "d_scalability.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NpEncoder)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="d-scalability study for PCE surrogate on gridworld"
    )
    parser.add_argument("--d_values", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--pce_degree", type=int, default=5)
    parser.add_argument("--n_episodes", type=int, default=400)
    parser.add_argument("--mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="results/gridworld/d_scalability")
    args = parser.parse_args()

    run_d_scalability(
        d_values=args.d_values,
        n_train=args.n_train,
        n_test=args.n_test,
        pce_degree=args.pce_degree,
        n_episodes=args.n_episodes,
        mode=args.mode,
        seed=args.seed,
        output_dir=args.output_dir,
    )
