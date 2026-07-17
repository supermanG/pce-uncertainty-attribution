"""Cost comparison: PCE analytical vs Saltelli MC vs ensemble bootstrap for Sobol indices.

Benchmarks three approaches for computing Sobol sensitivity indices on a
synthetic gridworld GFlowNet ensemble:

  1. **PCE analytical** -- fit a PCE surrogate, read off Sobol indices from
     the orthonormal coefficients.  O(P) after fitting.
  2. **Saltelli MC estimator** -- the standard pick-freeze sampling design
     (Saltelli 2010) applied directly to the trained ensemble.
  3. **Ensemble bootstrap** -- resample training data with replacement,
     refit PCE each time, and use variance of resulting Sobol indices.

For a range of ensemble sizes L, measures wall-clock time and accuracy
(MAE vs. a large-ensemble ground truth).  Outputs a formatted table and
saves results to JSON.

Usage:
    python experiments/sobol_validation/cost_comparison.py
    python experiments/sobol_validation/cost_comparison.py --n_episodes 200 --seed 0
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.pce_surrogate import (
    PCESurrogate,
    build_design_matrix,
    fit_pce,
    alr_transform,
    alr_inverse,
)
from experiments.gridworld.run_experiment import (
    GridGFlowNet,
    sample_reward_configs,
    train_single_member,
    GRID_SIZE,
    DISCRETE_ACTIONS,
    DISCRETE_N_STEPS,
)
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# JSON encoder for numpy types
# ---------------------------------------------------------------------------

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Data generation: train a full ensemble up-front
# ---------------------------------------------------------------------------

def generate_ensemble(
    L: int,
    mode: str = "discrete",
    n_episodes: int = 400,
    seed: int = 0,
    pca_dim: int = 2,
    verbose: bool = True,
):
    """Train L GFlowNet members on random reward configs and return PCA-projected mu + policies.

    Returns:
        mu:       (L, pca_dim) -- PCA-projected reward parameters
        policies: dict  step -> (L, n_actions) policy arrays
        pca:      fitted PCA object
    """
    params, grids = sample_reward_configs(L, mode, seed=seed)
    flat_grids = grids.reshape(L, -1)
    pca = PCA(n_components=pca_dim)
    mu = pca.fit_transform(flat_grids)

    n_steps = DISCRETE_N_STEPS
    all_policies = []
    for i in range(L):
        if verbose and (i + 1) % 50 == 0:
            print(f"    Training member {i + 1}/{L} ...")
        pol, _ = train_single_member(grids[i], mode=mode, seed=i + seed, n_episodes=n_episodes)
        all_policies.append(pol)

    policies = {
        s: np.array([all_policies[i][s] for i in range(L)])
        for s in range(n_steps)
    }
    return mu, policies, pca


# ---------------------------------------------------------------------------
# Method 1: PCE analytical Sobol
# ---------------------------------------------------------------------------

def pce_analytical_sobol(mu_train, policies_step, degree=3, ridge_lambda=1e-4):
    """Fit PCE and return Sobol indices analytically from coefficients.

    Returns:
        sobol: dict with 'first_order' (K-1, d), 'total_order' (K-1, d), 'variance' (K-1,)
        elapsed: wall-clock seconds
    """
    t0 = time.perf_counter()
    surr = PCESurrogate(degree=degree, basis="hermite", ridge_lambda=ridge_lambda)
    surr.fit(mu_train, policies_step)
    sobol = surr.sobol_indices()
    elapsed = time.perf_counter() - t0
    return sobol, elapsed


# ---------------------------------------------------------------------------
# Method 2: Saltelli MC estimator (pick-freeze)
# ---------------------------------------------------------------------------

def saltelli_sobol(mu_train, policies_step, degree=3, ridge_lambda=1e-4, n_mc=4096, seed=42):
    """Compute Sobol indices using Saltelli (2010) pick-freeze design.

    We first fit a PCE surrogate (to have a fast-to-evaluate model), then
    apply the Saltelli estimator on that surrogate using MC sampling.

    The pick-freeze approach:
      For each input dimension i:
        1. Draw two independent sample matrices A, B of shape (N, d)
        2. Construct AB_i = B with column i replaced by A's column i
        3. Evaluate model at A, B, AB_i
        4. S_i  = mean(f(A) * (f(AB_i) - f(B))) / var(f(A))
        5. ST_i = mean((f(A) - f(AB_i))^2) / (2 * var(f(A)))

    Returns:
        sobol: dict with 'first_order', 'total_order', 'variance'
        elapsed: wall-clock seconds
    """
    t0 = time.perf_counter()

    # Fit surrogate (needed to evaluate at arbitrary mu)
    surr = PCESurrogate(degree=degree, basis="hermite", ridge_lambda=ridge_lambda)
    surr.fit(mu_train, policies_step)

    d = mu_train.shape[1]
    K = policies_step.shape[1]
    K_eff = K - 1  # ALR components

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_mc, d))
    B = rng.standard_normal((n_mc, d))

    # Evaluate ALR at A and B
    Phi_A, _ = build_design_matrix(A, degree, "hermite")
    Phi_B, _ = build_design_matrix(B, degree, "hermite")

    alr_A = np.zeros((n_mc, K_eff))
    alr_B = np.zeros((n_mc, K_eff))
    for k in range(K_eff):
        alr_A[:, k] = Phi_A @ surr.coefficients[k]
        alr_B[:, k] = Phi_B @ surr.coefficients[k]

    first_order = np.zeros((K_eff, d))
    total_order = np.zeros((K_eff, d))
    variance = np.zeros(K_eff)

    for k in range(K_eff):
        y_A = alr_A[:, k]
        y_B = alr_B[:, k]
        V = float(np.var(y_A))
        variance[k] = V
        if V < 1e-15:
            continue

        for i in range(d):
            # Construct AB_i: B with column i replaced by A's column i
            AB_i = B.copy()
            AB_i[:, i] = A[:, i]
            Phi_AB, _ = build_design_matrix(AB_i, degree, "hermite")
            y_AB = Phi_AB @ surr.coefficients[k]

            # First-order: S_i = mean(f(A) * (f(AB_i) - f(B))) / var(f(A))
            first_order[k, i] = float(np.mean(y_A * (y_AB - y_B)) / V)
            # Total-order: ST_i = mean((f(A) - f(AB_i))^2) / (2 * var(f(A)))
            total_order[k, i] = float(np.mean((y_A - y_AB) ** 2) / (2.0 * V))

    first_order = np.clip(first_order, 0.0, 1.0)
    total_order = np.clip(total_order, 0.0, 1.0)

    elapsed = time.perf_counter() - t0
    return {"first_order": first_order, "total_order": total_order, "variance": variance}, elapsed


# ---------------------------------------------------------------------------
# Method 3: Ensemble bootstrap
# ---------------------------------------------------------------------------

def bootstrap_sobol(mu_train, policies_step, degree=3, ridge_lambda=1e-4, n_bootstrap=200, seed=42):
    """Resample the ensemble with replacement, refit PCE, collect Sobol indices.

    The *mean* of bootstrap Sobol estimates is reported as the point estimate;
    the *std* gives an uncertainty measure.

    Returns:
        sobol: dict with 'first_order', 'total_order', 'variance' (mean over bootstraps)
        sobol_std: dict with 'first_order_std', 'total_order_std' (std over bootstraps)
        elapsed: wall-clock seconds
    """
    t0 = time.perf_counter()

    L = mu_train.shape[0]
    rng = np.random.default_rng(seed)

    fo_collection = []
    to_collection = []
    var_collection = []

    for b in range(n_bootstrap):
        idx = rng.choice(L, size=L, replace=True)
        mu_b = mu_train[idx]
        pol_b = policies_step[idx]

        surr = PCESurrogate(degree=degree, basis="hermite", ridge_lambda=ridge_lambda)
        surr.fit(mu_b, pol_b)
        sb = surr.sobol_indices()

        fo_collection.append(sb["first_order"])
        to_collection.append(sb["total_order"])
        var_collection.append(sb["variance"])

    fo_arr = np.array(fo_collection)   # (n_bootstrap, K-1, d)
    to_arr = np.array(to_collection)
    var_arr = np.array(var_collection)

    sobol = {
        "first_order": np.mean(fo_arr, axis=0),
        "total_order": np.mean(to_arr, axis=0),
        "variance": np.mean(var_arr, axis=0),
    }
    sobol_std = {
        "first_order_std": np.std(fo_arr, axis=0),
        "total_order_std": np.std(to_arr, axis=0),
    }

    elapsed = time.perf_counter() - t0
    return sobol, sobol_std, elapsed


# ---------------------------------------------------------------------------
# Ground truth from large ensemble
# ---------------------------------------------------------------------------

def compute_ground_truth(mu_all, policies_all, degree=3, ridge_lambda=1e-4):
    """PCE analytical Sobol from a very large ensemble (serves as ground truth)."""
    surr = PCESurrogate(degree=degree, basis="hermite", ridge_lambda=ridge_lambda)
    surr.fit(mu_all, policies_all)
    return surr.sobol_indices()


# ---------------------------------------------------------------------------
# Error metric
# ---------------------------------------------------------------------------

def sobol_mae(estimated, ground_truth):
    """Mean absolute error between estimated and ground-truth Sobol first-order and total-order."""
    mae_fo = float(np.mean(np.abs(estimated["first_order"] - ground_truth["first_order"])))
    mae_to = float(np.mean(np.abs(estimated["total_order"] - ground_truth["total_order"])))
    return {"first_order_mae": mae_fo, "total_order_mae": mae_to}


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_cost_comparison(
    L_grid=None,
    L_ground_truth=500,
    pce_degree=3,
    pca_dim=2,
    n_episodes=400,
    n_mc_saltelli=4096,
    n_bootstrap=200,
    step_to_analyze=0,
    seed=0,
    output_dir="results/sobol_validation",
):
    """Run the full cost-comparison benchmark.

    Args:
        L_grid:           list of ensemble sizes to benchmark
        L_ground_truth:   ensemble size for the ground-truth reference
        pce_degree:       PCE polynomial degree
        pca_dim:          PCA dimensionality for reward parameterisation
        n_episodes:       GFlowNet training episodes per member
        n_mc_saltelli:    number of MC samples for Saltelli estimator
        n_bootstrap:      number of bootstrap resamples
        step_to_analyze:  which trajectory step to benchmark (0-indexed)
        seed:             master RNG seed
        output_dir:       directory for JSON output

    Returns:
        results dict (also saved to JSON)
    """
    if L_grid is None:
        L_grid = [20, 30, 50, 80, 120, 200]

    # Ensure L_ground_truth covers all requested sizes
    L_max = max(max(L_grid), L_ground_truth)

    print("=" * 70)
    print("SOBOL COST COMPARISON BENCHMARK")
    print("=" * 70)
    print(f"  Ensemble sizes:     {L_grid}")
    print(f"  Ground truth L:     {L_ground_truth}")
    print(f"  PCE degree:         {pce_degree}")
    print(f"  PCA dim:            {pca_dim}")
    print(f"  Saltelli MC:        {n_mc_saltelli}")
    print(f"  Bootstrap resamples:{n_bootstrap}")
    print(f"  Step to analyze:    {step_to_analyze}")
    print(f"  Training episodes:  {n_episodes}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Generate the full ensemble (largest size needed)
    # ------------------------------------------------------------------
    print(f"Generating ensemble of {L_max} members (this takes a while) ...")
    mu_all, policies_all, pca = generate_ensemble(
        L=L_max, mode="discrete", n_episodes=n_episodes, seed=seed, pca_dim=pca_dim, verbose=True
    )
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print()

    # Extract the target step
    pol_all_step = policies_all[step_to_analyze]  # (L_max, K)

    # ------------------------------------------------------------------
    # Step 2: Compute ground truth from L_ground_truth members
    # ------------------------------------------------------------------
    print(f"Computing ground truth from L={L_ground_truth} ...")
    gt = compute_ground_truth(
        mu_all[:L_ground_truth], pol_all_step[:L_ground_truth],
        degree=pce_degree, ridge_lambda=1e-4,
    )
    print(f"  Ground-truth first-order Sobol: {gt['first_order']}")
    print(f"  Ground-truth total-order Sobol: {gt['total_order']}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Benchmark each method at each ensemble size
    # ------------------------------------------------------------------
    results = {
        "config": {
            "L_grid": L_grid,
            "L_ground_truth": L_ground_truth,
            "pce_degree": pce_degree,
            "pca_dim": pca_dim,
            "n_mc_saltelli": n_mc_saltelli,
            "n_bootstrap": n_bootstrap,
            "step_to_analyze": step_to_analyze,
            "n_episodes": n_episodes,
            "seed": seed,
        },
        "ground_truth": {
            "first_order": gt["first_order"].tolist(),
            "total_order": gt["total_order"].tolist(),
            "variance": gt["variance"].tolist(),
        },
        "benchmarks": [],
    }

    for L in L_grid:
        print(f"--- L = {L} ---")
        mu_sub = mu_all[:L]
        pol_sub = pol_all_step[:L]

        entry = {"L": L}

        # Method 1: PCE analytical
        sobol_pce, t_pce = pce_analytical_sobol(
            mu_sub, pol_sub, degree=pce_degree, ridge_lambda=1e-4,
        )
        err_pce = sobol_mae(sobol_pce, gt)
        entry["pce_analytical"] = {
            "time_s": round(t_pce, 4),
            "first_order": sobol_pce["first_order"].tolist(),
            "total_order": sobol_pce["total_order"].tolist(),
            **err_pce,
        }
        print(f"  PCE analytical:     {t_pce:8.4f}s  |  FO MAE={err_pce['first_order_mae']:.4f}  TO MAE={err_pce['total_order_mae']:.4f}")

        # Method 2: Saltelli MC
        sobol_salt, t_salt = saltelli_sobol(
            mu_sub, pol_sub, degree=pce_degree, ridge_lambda=1e-4,
            n_mc=n_mc_saltelli, seed=seed + L,
        )
        err_salt = sobol_mae(sobol_salt, gt)
        entry["saltelli_mc"] = {
            "time_s": round(t_salt, 4),
            "first_order": sobol_salt["first_order"].tolist(),
            "total_order": sobol_salt["total_order"].tolist(),
            **err_salt,
        }
        print(f"  Saltelli MC:        {t_salt:8.4f}s  |  FO MAE={err_salt['first_order_mae']:.4f}  TO MAE={err_salt['total_order_mae']:.4f}")

        # Method 3: Ensemble bootstrap
        sobol_boot, sobol_boot_std, t_boot = bootstrap_sobol(
            mu_sub, pol_sub, degree=pce_degree, ridge_lambda=1e-4,
            n_bootstrap=n_bootstrap, seed=seed + L,
        )
        err_boot = sobol_mae(sobol_boot, gt)
        entry["bootstrap"] = {
            "time_s": round(t_boot, 4),
            "first_order": sobol_boot["first_order"].tolist(),
            "total_order": sobol_boot["total_order"].tolist(),
            "first_order_std": sobol_boot_std["first_order_std"].tolist(),
            "total_order_std": sobol_boot_std["total_order_std"].tolist(),
            **err_boot,
        }
        print(f"  Ensemble bootstrap: {t_boot:8.4f}s  |  FO MAE={err_boot['first_order_mae']:.4f}  TO MAE={err_boot['total_order_mae']:.4f}")
        print()

        results["benchmarks"].append(entry)

    # ------------------------------------------------------------------
    # Step 4: Formatted comparison table
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("COMPARISON TABLE: Wall-clock time (seconds) and Sobol MAE vs ground truth")
    print("=" * 100)
    header = (
        f"{'L':>5s}  |  {'PCE Time':>10s}  {'PCE FO':>8s}  {'PCE TO':>8s}"
        f"  |  {'Salt Time':>10s}  {'Salt FO':>8s}  {'Salt TO':>8s}"
        f"  |  {'Boot Time':>10s}  {'Boot FO':>8s}  {'Boot TO':>8s}"
    )
    print(header)
    print("-" * len(header))

    for entry in results["benchmarks"]:
        L = entry["L"]
        p = entry["pce_analytical"]
        s = entry["saltelli_mc"]
        b = entry["bootstrap"]
        row = (
            f"{L:5d}  |  {p['time_s']:10.4f}  {p['first_order_mae']:8.4f}  {p['total_order_mae']:8.4f}"
            f"  |  {s['time_s']:10.4f}  {s['first_order_mae']:8.4f}  {s['total_order_mae']:8.4f}"
            f"  |  {b['time_s']:10.4f}  {b['first_order_mae']:8.4f}  {b['total_order_mae']:8.4f}"
        )
        print(row)

    print()

    # Speedup summary
    print("SPEEDUP SUMMARY (relative to Saltelli MC):")
    print(f"{'L':>5s}  |  {'PCE/Saltelli':>14s}  {'PCE/Bootstrap':>14s}")
    print("-" * 42)
    for entry in results["benchmarks"]:
        L = entry["L"]
        t_pce = entry["pce_analytical"]["time_s"]
        t_salt = entry["saltelli_mc"]["time_s"]
        t_boot = entry["bootstrap"]["time_s"]
        speedup_salt = t_salt / max(t_pce, 1e-9)
        speedup_boot = t_boot / max(t_pce, 1e-9)
        print(f"{L:5d}  |  {speedup_salt:13.1f}x  {speedup_boot:13.1f}x")
    print()

    # ------------------------------------------------------------------
    # Step 5: Save results
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "cost_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NpEncoder)
    print(f"Results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark PCE analytical vs Saltelli MC vs bootstrap for Sobol indices"
    )
    parser.add_argument(
        "--L_grid", type=int, nargs="+", default=[20, 30, 50, 80, 120, 200],
        help="Ensemble sizes to benchmark",
    )
    parser.add_argument("--L_ground_truth", type=int, default=500)
    parser.add_argument("--pce_degree", type=int, default=3)
    parser.add_argument("--pca_dim", type=int, default=2)
    parser.add_argument("--n_episodes", type=int, default=400)
    parser.add_argument("--n_mc_saltelli", type=int, default=4096)
    parser.add_argument("--n_bootstrap", type=int, default=200)
    parser.add_argument("--step", type=int, default=0, help="Trajectory step to analyze")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="results/sobol_validation")

    args = parser.parse_args()

    run_cost_comparison(
        L_grid=args.L_grid,
        L_ground_truth=args.L_ground_truth,
        pce_degree=args.pce_degree,
        pca_dim=args.pca_dim,
        n_episodes=args.n_episodes,
        n_mc_saltelli=args.n_mc_saltelli,
        n_bootstrap=args.n_bootstrap,
        step_to_analyze=args.step,
        seed=args.seed,
        output_dir=args.output_dir,
    )
