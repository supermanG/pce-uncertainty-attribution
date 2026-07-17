#!/usr/bin/env python3
"""Total-order Sobol interaction diagnostic.

For each experiment, computes the sum of total-order Sobol indices
across all input dimensions. If sum(S_Ti) ~ 1.0, the model is
near-additive (no interactions). If sum(S_Ti) >> 1, interaction
effects dominate. Reports per-step and per-action breakdown.
"""
import sys, json, os, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.pce_surrogate import TrajectoryPCESurrogate, calibration_coverage


def sobol_interaction_summary(sobol_dict: dict) -> dict:
    """Summarise total-order Sobol sums for one trajectory."""
    step_summaries = []
    for step in sorted(sobol_dict.keys()):
        s = sobol_dict[step]
        to = s["total_order"]   # (K-1, d)
        fo = s["first_order"]
        var = s["variance"]

        # Sum of total-order per ALR component
        to_sums = to.sum(axis=1)     # (K-1,)
        fo_sums = fo.sum(axis=1)
        interaction_index = to_sums - fo_sums   # excess due to interactions

        step_summaries.append({
            "step": int(step),
            "n_actions": int(to.shape[0]),
            "n_dims": int(to.shape[1]),
            "mean_total_order_sum": round(float(to_sums.mean()), 4),
            "max_total_order_sum": round(float(to_sums.max()), 4),
            "mean_first_order_sum": round(float(fo_sums.mean()), 4),
            "mean_interaction_excess": round(float(interaction_index.mean()), 4),
            "max_interaction_excess": round(float(interaction_index.max()), 4),
            "mean_variance": round(float(var.mean()), 2),
        })

    overall_to_sums = [s["mean_total_order_sum"] for s in step_summaries]
    overall_interaction = [s["mean_interaction_excess"] for s in step_summaries]

    return {
        "per_step": step_summaries,
        "trajectory_mean_total_order_sum": round(float(np.mean(overall_to_sums)), 4),
        "trajectory_mean_interaction_excess": round(float(np.mean(overall_interaction)), 4),
        "near_additive": bool(np.mean(overall_to_sums) < 1.15),
    }


def run_gridworld_interactions(n_train=50, n_test=100, pce_degree=5, pca_dim=2):
    """Gridworld Sobol interaction analysis."""
    from experiments.gridworld.run_experiment import (
        sample_reward_configs, train_single_member,
    )
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("Gridworld (discrete)...")
    n_total = n_train + n_test
    params, grids = sample_reward_configs(n_total, mode="discrete", seed=0)
    flat_grids = grids.reshape(n_total, -1)

    all_policies = []
    for i in range(n_total):
        pol, _ = train_single_member(grids[i], mode="discrete", seed=i, n_episodes=400)
        all_policies.append(pol)

    n_steps = len(all_policies[0])
    tr_pol = {s: np.array([all_policies[i][s] for i in range(n_train)]) for s in range(n_steps)}

    pca = PCA(n_components=pca_dim)
    mu = pca.fit_transform(flat_grids)
    scaler = StandardScaler()
    mu = scaler.fit_transform(mu)
    mu_tr = mu[:n_train]

    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(n_steps):
        tsurr.fit_step(s, mu_tr, tr_pol[s])

    sobol = tsurr.sobol_all_steps()
    return sobol_interaction_summary(sobol)


def main():
    output_dir = ROOT / "results" / "validation"
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Gridworld (fast)
    try:
        results["gridworld_discrete"] = run_gridworld_interactions()
    except Exception as e:
        print(f"Gridworld skipped: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Total-Order Sobol Interaction Diagnostic")
    print("=" * 60)
    print(f"{'Task':30s} {'Mean sum(S_Ti)':>15s} {'Interaction':>12s} {'Additive?':>10s}")
    print("-" * 70)
    for task, r in results.items():
        ms = r["trajectory_mean_total_order_sum"]
        mi = r["trajectory_mean_interaction_excess"]
        add = "YES" if r["near_additive"] else "NO"
        print(f"{task:30s} {ms:15.4f} {mi:12.4f} {add:>10s}")

    print("\nPer-step detail:")
    for task, r in results.items():
        print(f"\n  {task}:")
        for s in r["per_step"]:
            print(f"    Step {s['step']}: sum(S_Ti)={s['mean_total_order_sum']:.4f}, "
                  f"interaction={s['mean_interaction_excess']:.4f}, "
                  f"D_mean={s['mean_variance']:.1f}")

    out_path = output_dir / "sobol_interactions.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
