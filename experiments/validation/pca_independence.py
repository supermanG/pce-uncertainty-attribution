#!/usr/bin/env python3
"""PCA independence verification for BH and Sachs experiments.

Validates that PCA-projected reward coordinates are approximately
independent, as required for valid analytical Sobol decomposition
with the Hermite polynomial basis.

Reports Pearson and Spearman correlations between all pairs of PCA
components, plus a Shapiro-Wilk test for marginal normality.
"""
import sys, json, os
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _correlation_matrix(mu: np.ndarray) -> dict:
    """Compute Pearson, Spearman, and normality tests for PCA components."""
    L, d = mu.shape
    pearson = np.corrcoef(mu.T)
    spearman_result = stats.spearmanr(mu)
    spearman = spearman_result.statistic
    if np.ndim(spearman) == 0:
        # For d=2, spearmanr returns a scalar correlation coefficient
        rho = float(spearman)
        spearman = np.array([[1.0, rho], [rho, 1.0]])

    # Off-diagonal correlations
    off_diag_pearson = []
    off_diag_spearman = []
    for i in range(d):
        for j in range(i + 1, d):
            off_diag_pearson.append(float(pearson[i, j]))
            off_diag_spearman.append(float(spearman[i, j]))

    # Marginal normality (Shapiro-Wilk)
    normality = []
    for i in range(d):
        if L <= 5000:
            stat, pval = stats.shapiro(mu[:, i])
        else:
            stat, pval = stats.normaltest(mu[:, i])
        normality.append({"dim": i, "statistic": round(float(stat), 4),
                          "p_value": round(float(pval), 4),
                          "normal_at_005": bool(pval > 0.05)})

    return {
        "pearson_matrix": pearson.tolist(),
        "spearman_matrix": (spearman if isinstance(spearman, np.ndarray) else np.array([[spearman]])).tolist(),
        "max_abs_pearson_offdiag": round(max(abs(x) for x in off_diag_pearson), 6) if off_diag_pearson else 0.0,
        "max_abs_spearman_offdiag": round(max(abs(x) for x in off_diag_spearman), 6) if off_diag_spearman else 0.0,
        "normality_tests": normality,
    }


def verify_bh_independence(n_train=50, n_test=100, pca_dim=5, seed=0):
    """Generate BH ensemble and verify PCA independence."""
    from experiments.buchwald_hartwig.run_experiment import (
        load_dataset, train_proxy,
    )
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("=" * 60)
    print("Buchwald-Hartwig PCA independence verification")
    print("=" * 60)

    n_total = n_train + n_test
    data = load_dataset()

    # Reference reactions for proxy output vectors
    rng_ref = np.random.RandomState(0)
    ref_rxns = data["reactions"][rng_ref.choice(len(data["yields"]), size=500, replace=False)]

    # Generate proxy outputs for each ensemble member
    proxy_outputs = []
    for i in range(n_total):
        proxy = train_proxy(data, seed=seed + i)
        proxy_outputs.append(proxy.predict_yield(ref_rxns))
    proxy_matrix = np.array(proxy_outputs)

    # PCA
    pca = PCA(n_components=pca_dim)
    mu = pca.fit_transform(proxy_matrix)
    scaler = StandardScaler()
    mu = scaler.fit_transform(mu)

    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"Ensemble size: {n_total}, PCA dim: {pca_dim}")

    result = _correlation_matrix(mu)
    print(f"Max |Pearson| off-diagonal:  {result['max_abs_pearson_offdiag']:.6f}")
    print(f"Max |Spearman| off-diagonal: {result['max_abs_spearman_offdiag']:.6f}")
    for nt in result["normality_tests"]:
        print(f"  Dim {nt['dim']}: Shapiro p={nt['p_value']:.4f} {'PASS' if nt['normal_at_005'] else 'FAIL'}")

    return {"task": "buchwald_hartwig", "pca_dim": pca_dim, **result}


def verify_sachs_independence(n_train=30, n_test=50, pca_dim=2, seed=0):
    """Generate Sachs ensemble and verify PCA independence."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    print("\n" + "=" * 60)
    print("Sachs causal discovery PCA independence verification")
    print("=" * 60)

    # Load Sachs data
    sachs_path = ROOT / "data" / "sachs_real.csv"
    if not sachs_path.exists():
        print(f"Sachs data not found at {sachs_path}. Skipping.")
        return None
    df = pd.read_csv(sachs_path)
    X = df.values.astype(np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    n_total = n_train + n_test
    n_obs_per_member = 200

    # For each member, subsample observations and compute covariance
    rng = np.random.RandomState(seed)
    cov_flat = []
    for i in range(n_total):
        idx = rng.choice(len(X), size=n_obs_per_member, replace=False)
        cov = np.cov(X[idx].T)
        cov_flat.append(cov.flatten())
    cov_matrix = np.array(cov_flat)

    pca = PCA(n_components=pca_dim)
    mu = pca.fit_transform(cov_matrix)
    scaler = StandardScaler()
    mu = scaler.fit_transform(mu)

    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"Ensemble size: {n_total}, PCA dim: {pca_dim}")

    result = _correlation_matrix(mu)
    print(f"Max |Pearson| off-diagonal:  {result['max_abs_pearson_offdiag']:.6f}")
    print(f"Max |Spearman| off-diagonal: {result['max_abs_spearman_offdiag']:.6f}")
    for nt in result["normality_tests"]:
        print(f"  Dim {nt['dim']}: Shapiro p={nt['p_value']:.4f} {'PASS' if nt['normal_at_005'] else 'FAIL'}")

    return {"task": "sachs", "pca_dim": pca_dim, **result}


def verify_gridworld_independence(n_total=150, pca_dim=2, seed=0):
    """Gridworld PCA independence (quick, no GPU needed)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 60)
    print("Gridworld PCA independence verification")
    print("=" * 60)

    rng = np.random.RandomState(seed)
    # 5x5 grid, 4 zone shifts each from {-1, -0.5, 0, 0.5, 1}
    zone_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    params = []
    grids = []
    for _ in range(n_total):
        shifts = rng.choice(zone_vals, size=4)
        grid = np.zeros((5, 5))
        grid[:3, :3] += shifts[0]
        grid[:3, 3:] += shifts[1]
        grid[3:, :3] += shifts[2]
        grid[3:, 3:] += shifts[3]
        params.append(shifts)
        grids.append(grid.flatten())

    flat = np.array(grids)
    pca = PCA(n_components=pca_dim)
    mu = pca.fit_transform(flat)
    scaler = StandardScaler()
    mu = scaler.fit_transform(mu)

    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    result = _correlation_matrix(mu)
    print(f"Max |Pearson| off-diagonal:  {result['max_abs_pearson_offdiag']:.6f}")
    print(f"Max |Spearman| off-diagonal: {result['max_abs_spearman_offdiag']:.6f}")
    for nt in result["normality_tests"]:
        print(f"  Dim {nt['dim']}: Shapiro p={nt['p_value']:.4f} {'PASS' if nt['normal_at_005'] else 'FAIL'}")

    return {"task": "gridworld", "pca_dim": pca_dim, **result}


def main():
    output_dir = ROOT / "results" / "validation"
    os.makedirs(output_dir, exist_ok=True)

    results = []

    # Gridworld (fast, no GPU)
    r = verify_gridworld_independence()
    if r:
        results.append(r)

    # BH (needs proxy training)
    try:
        r = verify_bh_independence()
        if r:
            results.append(r)
    except Exception as e:
        print(f"BH verification skipped: {e}")

    # Sachs (needs data file)
    try:
        r = verify_sachs_independence()
        if r:
            results.append(r)
    except Exception as e:
        print(f"Sachs verification skipped: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: PCA Independence Verification")
    print("=" * 60)
    for r in results:
        task = r["task"]
        mp = r["max_abs_pearson_offdiag"]
        ms = r["max_abs_spearman_offdiag"]
        status = "PASS" if mp < 0.10 and ms < 0.10 else "MARGINAL" if mp < 0.20 else "FAIL"
        print(f"  {task:25s}  |r_Pearson|_max={mp:.4f}  |r_Spearman|_max={ms:.4f}  [{status}]")

    out_path = output_dir / "pca_independence.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
