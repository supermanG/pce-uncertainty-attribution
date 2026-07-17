#!/usr/bin/env python3
"""Computational scaling benchmark: PCE vs ensemble retraining.

Generates a log-log plot of wall-clock time vs number of policy samples
for PCE surrogate evaluation vs equivalent ensemble retraining.
Demonstrates the 4-5 orders of magnitude speedup claimed in the paper.
"""
import sys, json, os, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.pce_surrogate import PCESurrogate, TrajectoryPCESurrogate


def benchmark_pce_sampling(tsurr, n_samples_list, n_repeats=5):
    """Benchmark PCE surrogate sampling at various sample counts."""
    results = []
    for n in n_samples_list:
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = tsurr.sample_trajectory_policies(n)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        results.append({
            "n_samples": n,
            "mean_time_s": round(float(np.mean(times)), 6),
            "std_time_s": round(float(np.std(times)), 6),
        })
    return results


def estimate_ensemble_time(n_samples_list, time_per_gfn_s):
    """Estimate wall-clock time for training n GFlowNets."""
    return [{"n_samples": n, "estimated_time_s": round(n * time_per_gfn_s, 2)}
            for n in n_samples_list]


def run_gridworld_benchmark():
    """Benchmark on gridworld (fast, no GPU needed)."""
    from experiments.gridworld.run_experiment import (
        sample_reward_configs, train_single_member,
    )
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("=" * 60)
    print("Gridworld scaling benchmark")
    print("=" * 60)

    n_train, n_test = 50, 100
    n_total = n_train + n_test

    # Train ensemble
    print("Training ensemble...")
    params, grids = sample_reward_configs(n_total, mode="discrete", seed=0)
    flat_grids = grids.reshape(n_total, -1)

    t_train_start = time.perf_counter()
    all_policies = []
    for i in range(n_total):
        pol, _ = train_single_member(grids[i], mode="discrete", seed=i, n_episodes=400)
        all_policies.append(pol)
    t_train_total = time.perf_counter() - t_train_start
    time_per_member = t_train_total / n_total

    n_steps = len(all_policies[0])
    tr_pol = {s: np.array([all_policies[i][s] for i in range(n_train)]) for s in range(n_steps)}

    # PCA + PCE fit
    pca = PCA(n_components=2)
    mu = pca.fit_transform(flat_grids)
    scaler = StandardScaler()
    mu = scaler.fit_transform(mu)
    mu_tr = mu[:n_train]

    t_fit_start = time.perf_counter()
    tsurr = TrajectoryPCESurrogate(degree=5, basis="hermite")
    for s in range(n_steps):
        tsurr.fit_step(s, mu_tr, tr_pol[s])
    t_fit = time.perf_counter() - t_fit_start

    # Benchmark sampling
    n_samples_list = [10, 100, 1000, 10000, 100000]
    pce_results = benchmark_pce_sampling(tsurr, n_samples_list)
    ensemble_results = estimate_ensemble_time(n_samples_list, time_per_member)

    print(f"\nPCE fit time: {t_fit:.4f}s")
    print(f"GFlowNet time per member: {time_per_member:.2f}s")
    print(f"\n{'N samples':>10s}  {'PCE (s)':>12s}  {'Ensemble (s)':>12s}  {'Speedup':>10s}")
    print("-" * 50)
    for pce, ens in zip(pce_results, ensemble_results):
        pce_total = pce["mean_time_s"] + t_fit
        speedup = ens["estimated_time_s"] / pce_total if pce_total > 0 else float("inf")
        print(f"{pce['n_samples']:>10d}  {pce_total:>12.4f}  {ens['estimated_time_s']:>12.2f}  {speedup:>10.0f}x")

    return {
        "task": "gridworld_discrete",
        "pce_fit_time_s": round(t_fit, 4),
        "gfn_time_per_member_s": round(time_per_member, 2),
        "pce_sampling": pce_results,
        "ensemble_estimated": ensemble_results,
    }


def main():
    output_dir = ROOT / "results" / "validation"
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Gridworld benchmark
    try:
        results["gridworld"] = run_gridworld_benchmark()
    except Exception as e:
        print(f"Gridworld benchmark failed: {e}")
        import traceback; traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SCALING BENCHMARK SUMMARY")
    print("=" * 60)
    for task, r in results.items():
        fit = r["pce_fit_time_s"]
        member = r["gfn_time_per_member_s"]
        pce_10k = next((p["mean_time_s"] for p in r["pce_sampling"] if p["n_samples"] == 10000), None)
        if pce_10k:
            total_pce = fit + pce_10k
            total_ens = 10000 * member
            print(f"  {task}: PCE(10k)={total_pce:.3f}s vs Ensemble={total_ens:.0f}s "
                  f"({total_ens/total_pce:.0f}x speedup)")

    out_path = output_dir / "scaling_benchmark.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nResults saved to {out_path}")

    # Generate plot data for supplementary figure
    print("\nTo generate the supplementary figure, use:")
    print("  python figures/generate_figures.py --scaling-benchmark")


if __name__ == "__main__":
    main()
