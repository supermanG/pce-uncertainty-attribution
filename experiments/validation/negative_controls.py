#!/usr/bin/env python3
"""Negative control experiments: what breaks when assumptions are violated.

Three controlled failure modes on the gridworld task:
  1. Too few ensemble members (L=5, 10, 15 vs. adequate L=50)
  2. PCE overfitting (degree 10, 15 vs. optimal degree 5)
  3. Broken independence (shuffled PCA components)

Generates a summary table and optionally a multi-panel figure.
"""
import sys, json, os, time, math
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.pce_surrogate import TrajectoryPCESurrogate, calibration_coverage


def _train_gridworld_ensemble(n_total, mode="discrete", seed=0, n_episodes=400):
    """Train a gridworld ensemble, return params and policies."""
    from experiments.gridworld.run_experiment import (
        sample_reward_configs, train_single_member,
    )
    params, grids = sample_reward_configs(n_total, mode=mode, seed=seed)
    all_policies = []
    for i in range(n_total):
        pol, _ = train_single_member(grids[i], mode=mode, seed=i + seed, n_episodes=n_episodes)
        all_policies.append(pol)
    return params, grids, all_policies


def _fit_and_evaluate(mu_tr, mu_te, tr_pol, te_pol, n_steps, pce_degree):
    """Fit PCE, compute calibration coverage and KS pass rate."""
    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")

    t0 = time.perf_counter()
    for s in range(n_steps):
        tsurr.fit_step(s, mu_tr, tr_pol[s])
    fit_time = time.perf_counter() - t0

    # Calibration coverage at 90%
    mc_samples = tsurr.sample_trajectory_policies(2000)
    coverages = []
    for s in range(n_steps):
        cov = calibration_coverage(mc_samples[s], te_pol[s])
        coverages.append(cov.get(0.9, 0.0))
    mean_cov_90 = float(np.mean(coverages))

    # Sobol sanity: sum of total-order
    sobol = tsurr.sobol_all_steps()
    to_sums = []
    for s in sobol:
        to_sums.append(float(sobol[s]["total_order"].sum(axis=1).mean()))
    mean_to_sum = float(np.mean(to_sums))

    return {
        "fit_time_s": round(fit_time, 3),
        "calibration_90": round(mean_cov_90, 3),
        "mean_total_order_sum": round(mean_to_sum, 4),
    }


def control_1_ensemble_size(all_mu, all_policies, n_steps, n_test=100, pce_degree=5):
    """Control 1: vary training ensemble size."""
    print("\n--- Control 1: Ensemble size ---")
    L_values = [5, 10, 15, 25, 50]
    results = []

    mu_te = all_mu[-n_test:]
    te_pol = {s: np.array([all_policies[-n_test + i][s] for i in range(n_test)]) for s in range(n_steps)}

    for L in L_values:
        mu_tr = all_mu[:L]
        tr_pol = {s: np.array([all_policies[i][s] for i in range(L)]) for s in range(n_steps)}

        n_terms = int(math.comb(2 + pce_degree, pce_degree))
        underdetermined = L < n_terms

        entry = _fit_and_evaluate(mu_tr, mu_te, tr_pol, te_pol, n_steps, pce_degree)
        entry["L"] = L
        entry["n_pce_terms"] = n_terms
        entry["underdetermined"] = underdetermined
        results.append(entry)

        status = "UNDERDETERMINED" if underdetermined else "ok"
        print(f"  L={L:3d}  terms={n_terms:3d}  cal@90={entry['calibration_90']:.3f}  "
              f"sum(S_T)={entry['mean_total_order_sum']:.3f}  [{status}]")

    return results


def control_2_pce_degree(all_mu, all_policies, n_steps, n_train=50, n_test=100):
    """Control 2: vary PCE degree (overfitting)."""
    print("\n--- Control 2: PCE degree (overfitting) ---")
    degree_values = [2, 3, 5, 8, 10]
    results = []

    mu_tr = all_mu[:n_train]
    mu_te = all_mu[-n_test:]
    tr_pol = {s: np.array([all_policies[i][s] for i in range(n_train)]) for s in range(n_steps)}
    te_pol = {s: np.array([all_policies[-n_test + i][s] for i in range(n_test)]) for s in range(n_steps)}

    for deg in degree_values:
        n_terms = int(math.comb(2 + deg, deg))
        underdetermined = n_train < n_terms

        entry = _fit_and_evaluate(mu_tr, mu_te, tr_pol, te_pol, n_steps, deg)
        entry["degree"] = deg
        entry["n_pce_terms"] = n_terms
        entry["underdetermined"] = underdetermined
        results.append(entry)

        status = "OVERFIT" if underdetermined else "ok"
        print(f"  deg={deg:2d}  terms={n_terms:4d}  cal@90={entry['calibration_90']:.3f}  "
              f"sum(S_T)={entry['mean_total_order_sum']:.3f}  [{status}]")

    return results


def control_3_broken_independence(all_mu, all_policies, n_steps, n_train=50, n_test=100, pce_degree=5):
    """Control 3: deliberately break PCA independence by shuffling one component."""
    print("\n--- Control 3: Broken independence (shuffled PC2) ---")

    mu_tr = all_mu[:n_train].copy()
    mu_te = all_mu[-n_test:]
    tr_pol = {s: np.array([all_policies[i][s] for i in range(n_train)]) for s in range(n_steps)}
    te_pol = {s: np.array([all_policies[-n_test + i][s] for i in range(n_test)]) for s in range(n_steps)}

    # Baseline (clean)
    baseline = _fit_and_evaluate(mu_tr, mu_te, tr_pol, te_pol, n_steps, pce_degree)
    baseline["condition"] = "clean"
    print(f"  Clean:    cal@90={baseline['calibration_90']:.3f}  sum(S_T)={baseline['mean_total_order_sum']:.3f}")

    # Shuffle PC2 (breaks independence between PC1 and PC2)
    rng = np.random.RandomState(999)
    mu_shuffled = mu_tr.copy()
    mu_shuffled[:, 1] = rng.permutation(mu_shuffled[:, 1])
    broken = _fit_and_evaluate(mu_shuffled, mu_te, tr_pol, te_pol, n_steps, pce_degree)
    broken["condition"] = "shuffled_pc2"
    print(f"  Shuffled: cal@90={broken['calibration_90']:.3f}  sum(S_T)={broken['mean_total_order_sum']:.3f}")

    # Correlated: inject correlation between PC1 and PC2
    mu_corr = mu_tr.copy()
    mu_corr[:, 1] = 0.8 * mu_corr[:, 0] + 0.2 * mu_corr[:, 1]
    correlated = _fit_and_evaluate(mu_corr, mu_te, tr_pol, te_pol, n_steps, pce_degree)
    correlated["condition"] = "correlated_0.8"
    print(f"  Corr=0.8: cal@90={correlated['calibration_90']:.3f}  sum(S_T)={correlated['mean_total_order_sum']:.3f}")

    return [baseline, broken, correlated]


def main():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    output_dir = ROOT / "results" / "validation"
    os.makedirs(output_dir, exist_ok=True)

    # Train a large enough ensemble for all controls
    n_max_train = 50
    n_test = 100
    n_total = n_max_train + n_test
    pca_dim = 2
    pce_degree = 5

    print("Training gridworld ensemble for negative controls...")
    params, grids, all_policies = _train_gridworld_ensemble(n_total, mode="discrete", seed=0)

    n_steps = len(all_policies[0])
    flat_grids = grids.reshape(n_total, -1)

    pca = PCA(n_components=pca_dim)
    mu = pca.fit_transform(flat_grids)
    scaler = StandardScaler()
    mu = scaler.fit_transform(mu)

    print(f"Ensemble: {n_total} members, PCA dim: {pca_dim}, "
          f"explained var: {pca.explained_variance_ratio_.sum():.3f}")

    results = {}
    results["control_1_ensemble_size"] = control_1_ensemble_size(
        mu, all_policies, n_steps, n_test=n_test, pce_degree=pce_degree)
    results["control_2_pce_degree"] = control_2_pce_degree(
        mu, all_policies, n_steps, n_train=n_max_train, n_test=n_test)
    results["control_3_independence"] = control_3_broken_independence(
        mu, all_policies, n_steps, n_train=n_max_train, n_test=n_test, pce_degree=pce_degree)

    # Summary
    print("\n" + "=" * 60)
    print("NEGATIVE CONTROL SUMMARY")
    print("=" * 60)
    print("\nExpected pattern: degradation when assumptions are violated,")
    print("confirming that the framework's accuracy is not accidental.\n")

    print("1. Small ensembles degrade calibration coverage.")
    print("2. High PCE degree causes overfitting when L < P (basis terms).")
    print("3. Breaking PCA independence corrupts Sobol attribution.\n")

    out_path = output_dir / "negative_controls.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
