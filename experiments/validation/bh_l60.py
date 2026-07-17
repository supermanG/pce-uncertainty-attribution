#!/usr/bin/env python3
"""Run BH experiment with L=60 training members to address L=50 < L*=57 concern.

The paper notes that L=50 is marginally below the Theorem A bound L*=57
for the BH experiment. This script runs with L=60 and compares MAE,
calibration coverage, and Sobol indices against the L=50 baseline.
"""
import sys, json, os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run_bh_comparison(L_values=None, n_test=100, pce_degree=5, pca_dim=2, gfn_episodes=3000):
    """Run BH at multiple ensemble sizes and compare."""
    if L_values is None:
        L_values = [30, 50, 60, 80]

    from experiments.buchwald_hartwig.run_experiment import (
        load_dataset, train_proxy,
        ReactionGFlowNet, train_gflownet, extract_policy, select_reference_trajectory,
    )
    from core.pce_surrogate import TrajectoryPCESurrogate, calibration_coverage
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    L_max = max(L_values)
    n_total = L_max + n_test
    data = load_dataset()

    # Train all proxies upfront (shared across ensemble sizes)
    print(f"Training {n_total} yield proxies...")
    proxies = []
    for i in range(n_total):
        proxy = train_proxy(data, seed=i)
        proxies.append(proxy)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_total} proxies done")

    # PCA on proxy outputs (using all proxies for consistency)
    rng = np.random.RandomState(0)
    ref_idx = rng.choice(len(data["yields"]), size=500, replace=False)
    ref_rxns = data["reactions"][ref_idx]
    pout = np.array([p.predict_yield(ref_rxns) for p in proxies])
    pca = PCA(n_components=pca_dim)
    mu = pca.fit_transform(pout)
    scaler = StandardScaler()
    mu = scaler.fit_transform(mu)

    # Reference trajectory
    ref_traj = select_reference_trajectory(data, proxies[0])

    # Train all GFlowNets upfront
    print(f"Training {n_total} GFlowNets...")
    all_policies = {s: [] for s in range(4)}
    for i, proxy in enumerate(proxies):
        gfn = ReactionGFlowNet(data["n_components"])
        gfn = train_gflownet(gfn, proxy, n_ep=gfn_episodes)
        pols = extract_policy(gfn, ref_traj)
        for s in range(4):
            all_policies[s].append(pols[s])
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_total} GFlowNets done")

    # Test policies (always the last n_test members)
    te_pol = {s: np.array(all_policies[s][L_max:]) for s in range(4)}
    mu_te = mu[L_max:]

    component_names = data.get("component_names", ["catalyst", "base", "aryl_halide", "additive"])

    # Sweep over ensemble sizes
    results = []
    for L in L_values:
        print(f"\n--- L = {L} ---")
        mu_tr = mu[:L]
        tr_pol = {s: np.array(all_policies[s][:L]) for s in range(4)}

        tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
        for s in range(4):
            tsurr.fit_step(s, mu_tr, tr_pol[s])

        # Sobol
        sobol = tsurr.sobol_all_steps()
        sobol_summary = {}
        for s in range(4):
            name = component_names[s]
            fo = sobol[s]["first_order"].mean(axis=0)
            var = sobol[s]["variance"].mean()
            sobol_summary[name] = {
                "mean_first_order": fo.tolist(),
                "mean_variance": round(float(var), 2),
            }
            print(f"  {name:15s}: D={var:.1f}, S_PC1={fo[0]:.3f}")

        # Calibration coverage
        mc_pols = tsurr.sample_trajectory_policies(10000)
        cal = {}
        for s in range(4):
            cov = calibration_coverage(mc_pols[s], te_pol[s])
            cal[component_names[s]] = {str(k): round(v, 3) for k, v in cov.items()}
            print(f"  {component_names[s]:15s}: cal@90={cov.get(0.9, 0):.3f}, "
                  f"cal@95={cov.get(0.95, 0):.3f}")

        # MAE
        mae_list = []
        for s in range(4):
            pred = tsurr.step_surrogates[s].predict(mu_te)
            mae = float(np.abs(pred - te_pol[s]).mean())
            mae_list.append(mae)
        mean_mae = float(np.mean(mae_list))
        print(f"  Mean MAE: {mean_mae:.4f}")

        # Theorem A bound
        L_star = tsurr.required_ensemble_size(target_sobol_error=0.05, confidence=0.95)
        print(f"  Theorem A L*: {L_star}")

        results.append({
            "L": L,
            "L_star": L_star,
            "adequately_determined": L >= L_star,
            "mean_mae": round(mean_mae, 4),
            "calibration": cal,
            "sobol": sobol_summary,
        })

    return results


def main():
    output_dir = ROOT / "results" / "validation"
    os.makedirs(output_dir, exist_ok=True)

    print("BH experiment: ensemble size comparison (L=30, 50, 60, 80)")
    print("This may take 1-2 hours on GPU.\n")

    try:
        results = run_bh_comparison()
    except Exception as e:
        print(f"BH comparison failed: {e}")
        import traceback; traceback.print_exc()
        return

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: BH Ensemble Size Comparison")
    print("=" * 60)
    print(f"{'L':>5s} {'L*':>5s} {'Adequate':>10s} {'MAE':>8s} {'Cal@90':>8s} {'Cal@95':>8s}")
    print("-" * 50)
    for r in results:
        cal_90 = np.mean([float(v.get("0.9", 0)) for v in r["calibration"].values()])
        cal_95 = np.mean([float(v.get("0.95", 0)) for v in r["calibration"].values()])
        adeq = "YES" if r["adequately_determined"] else "no"
        print(f"{r['L']:>5d} {r['L_star']:>5d} {adeq:>10s} {r['mean_mae']:>8.4f} "
              f"{cal_90:>8.3f} {cal_95:>8.3f}")

    out_path = output_dir / "bh_l60_comparison.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
