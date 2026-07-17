#!/usr/bin/env python3
"""Validate that high-Sobol additives have sparse training coverage.

The manuscript claims that additives with high Sobol indices are those
whose recommendations depend on which reactions appeared in the training set.
This script directly tests that claim: we compute the number of observed
reaction combinations per additive across proxy training subsets, and
correlate it with the additive-level Sobol index.

If the Sobol decomposition is interpretable, we expect:
  - Additives with fewer observed combinations → higher Sobol index
  - Additives with dense coverage → lower Sobol index (stable recommendation)
"""
import sys, json, os
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def compute_additive_coverage(data, train_fraction=0.3, n_proxies=50, seed=0):
    """For each proxy, count how many reaction combinations each additive appears in."""
    rng = np.random.RandomState(seed)
    n_reactions = len(data["yields"])
    n_train = int(n_reactions * train_fraction)

    additive_names = data.get("additive_names", [f"add_{i}" for i in range(data["n_components"][3])])
    n_additives = data["n_components"][3]

    # For each proxy training subset, count reactions per additive
    coverage_counts = np.zeros((n_proxies, n_additives))
    for p in range(n_proxies):
        idx = rng.permutation(n_reactions)[:n_train]
        for i in idx:
            add_idx = data["reactions"][i][3]  # additive index (step 3)
            coverage_counts[p, add_idx] += 1

    # Mean and std of coverage across proxies
    mean_coverage = coverage_counts.mean(axis=0)
    std_coverage = coverage_counts.std(axis=0)
    cv_coverage = std_coverage / (mean_coverage + 1e-8)  # coefficient of variation

    return {
        "additive_names": additive_names,
        "mean_coverage": mean_coverage.tolist(),
        "std_coverage": std_coverage.tolist(),
        "cv_coverage": cv_coverage.tolist(),
    }


def compute_additive_sobol(data, n_train=50, n_test=100, pce_degree=5, pca_dim=2, seed=0):
    """Run BH experiment and extract per-additive Sobol indices at step 3."""
    from experiments.buchwald_hartwig.run_experiment import (
        load_dataset, train_proxy,
        ReactionGFlowNet, train_gflownet, extract_policy, select_reference_trajectory,
    )
    from core.pce_surrogate import TrajectoryPCESurrogate
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_total = n_train + n_test

    # Train proxies
    print(f"Training {n_total} yield proxies...")
    proxies = []
    for i in range(n_total):
        proxy = train_proxy(data, seed=seed + i)
        proxies.append(proxy)

    # PCA on proxy outputs
    rng = np.random.RandomState(0)
    ref_idx = rng.choice(len(data["yields"]), size=500, replace=False)
    ref_rxns = data["reactions"][ref_idx]
    pout = np.array([p.predict_yield(ref_rxns) for p in proxies])
    pca = PCA(n_components=pca_dim)
    mu = pca.fit_transform(pout)
    scaler = StandardScaler()
    mu = scaler.fit_transform(mu)
    mu_tr = mu[:n_train]

    # Train GFlowNets and extract policies
    print(f"Training {n_total} GFlowNets...")
    ref_traj = select_reference_trajectory(data, proxies[0])
    all_policies = {s: [] for s in range(4)}
    for i, proxy in enumerate(proxies):
        gfn = ReactionGFlowNet(data["n_components"])
        gfn = train_gflownet(gfn, proxy, n_ep=3000)
        pols = extract_policy(gfn, ref_traj)
        for s in range(4):
            all_policies[s].append(pols[s])

    tr_pol = {s: np.array(all_policies[s][:n_train]) for s in range(4)}

    # Fit PCE and get Sobol at step 3 (additive selection)
    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(4):
        tsurr.fit_step(s, mu_tr, tr_pol[s])

    sobol = tsurr.sobol_all_steps()
    # Step 3 = additive selection
    step3 = sobol[3]
    # Total variance per ALR component
    variance_per_action = step3["variance"]  # (K-1,)
    total_order = step3["total_order"]  # (K-1, d)

    return {
        "step3_variance": variance_per_action.tolist(),
        "step3_total_order": total_order.tolist(),
        "step3_mean_total_order": total_order.mean(axis=1).tolist(),
    }


def main():
    output_dir = ROOT / "results" / "validation"
    os.makedirs(output_dir, exist_ok=True)

    try:
        from experiments.buchwald_hartwig.run_experiment import load_dataset
        data = load_dataset()
    except Exception as e:
        print(f"Could not load BH data: {e}")
        print("This script requires the Doyle-Dreher dataset.")
        return

    # Compute coverage statistics (fast, no GPU)
    print("Computing additive coverage statistics...")
    coverage = compute_additive_coverage(data, n_proxies=50)

    n_additives = len(coverage["mean_coverage"])
    print(f"\nAdditive coverage (mean reactions per proxy across 50 subsets):")
    for i in range(n_additives):
        name = coverage["additive_names"][i] if i < len(coverage["additive_names"]) else f"add_{i}"
        print(f"  {name:20s}: mean={coverage['mean_coverage'][i]:.1f}, "
              f"CV={coverage['cv_coverage'][i]:.3f}")

    # If BH experiment results already exist, load Sobol; otherwise skip
    # (the full experiment takes ~1h on GPU)
    sobol_data = None
    results_path = ROOT / "results" / "buchwald_hartwig" / "results.json"
    if results_path.exists():
        print("\nLoading existing BH Sobol results...")
        with open(results_path) as f:
            bh_results = json.load(f)
        if "sobol" in bh_results and "3" in bh_results["sobol"]:
            sobol_data = bh_results["sobol"]["3"]

    if sobol_data is not None:
        variance = np.array(sobol_data.get("variance", []))
        mean_cov = np.array(coverage["mean_coverage"])

        if len(variance) == n_additives - 1:
            # ALR has K-1 components; we can still correlate with first K-1 additives
            corr_pearson, p_pearson = stats.pearsonr(mean_cov[:len(variance)], variance)
            corr_spearman, p_spearman = stats.spearmanr(mean_cov[:len(variance)], variance)

            print(f"\nCorrelation: coverage vs. Sobol variance (step 3, additive)")
            print(f"  Pearson:  r={corr_pearson:.3f}, p={p_pearson:.4f}")
            print(f"  Spearman: r={corr_spearman:.3f}, p={p_spearman:.4f}")

            if corr_pearson < -0.3:
                print("  -> CONFIRMED: sparse coverage correlates with higher Sobol variance")
            elif corr_pearson > 0.3:
                print("  -> UNEXPECTED: dense coverage correlates with higher variance")
            else:
                print("  -> WEAK: no strong linear relationship")
    else:
        print("\nNo existing BH results found. Run the BH experiment first:")
        print("  python run_all.py -e bh")

    # Save
    result = {"coverage": coverage}
    if sobol_data:
        result["sobol_step3"] = sobol_data
    out_path = output_dir / "bh_additive_validation.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
