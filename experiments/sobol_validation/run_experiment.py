"""Ground-truth Sobol validation experiment.

Constructs a synthetic GFlowNet environment where the true first-order Sobol
indices are ANALYTICALLY KNOWN, then verifies that the PCE surrogate recovers
them as the ensemble size L increases.

Design:
  - 2D reward parameterisation: mu = (mu1, mu2), mu_i ~ N(0,1) independent
  - 2-step GFlowNet, 3 actions per step
  - Step 0 policy driven by mu1 ONLY  --> true Sobol: S_{mu1}=1.0, S_{mu2}=0.0
  - Step 1 policy driven by mu2 ONLY  --> true Sobol: S_{mu1}=0.0, S_{mu2}=1.0
  - Step 2 policy driven by mu1+mu2   --> true Sobol: S_{mu1}=0.5, S_{mu2}=0.5

These are exact (not approximate) ground-truth values given independent Gaussian mu_i.
The PCE should converge to these as L -> infinity.

This experiment validates:
  1. ALR transform correctness
  2. Sobol index computation
  3. Sample complexity bound (Theorem A)
  4. Convergence rate vs ensemble size

Output: results/sobol_validation/validation_results.json with:
  - Estimated vs. true Sobol indices for each L in L_grid
  - Sobol error vs L (should decay as 1/sqrt(L))
  - Required ensemble size from Theorem A bound vs. actual convergence
"""
import os, sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.pce_surrogate import PCESurrogate, TrajectoryPCESurrogate


# ---------------------------------------------------------------------------
# True Sobol indices (analytical)
# ---------------------------------------------------------------------------
# mu1, mu2 ~ N(0,1) independent.
# Step 0: logit_k = a0 + b0 * mu1   (only mu1 drives variance)
# Step 1: logit_k = a1 + b1 * mu2   (only mu2)
# Step 2: logit_k = a2 + b2 * mu1 + b2 * mu2  (equal contribution)
TRUE_SOBOL = {
    0: {"first_order": [[1.0, 0.0], [1.0, 0.0]],   # 2 ALR components, 2 mu dims
        "total_order": [[1.0, 0.0], [1.0, 0.0]]},
    1: {"first_order": [[0.0, 1.0], [0.0, 1.0]],
        "total_order": [[0.0, 1.0], [0.0, 1.0]]},
    2: {"first_order": [[0.5, 0.5], [0.5, 0.5]],
        "total_order": [[0.5, 0.5], [0.5, 0.5]]},
}


def synthetic_policy(mu: np.ndarray, step: int, seed_policy: int = 0) -> np.ndarray:
    """Compute ground-truth policy for given mu at a given step.

    Each policy is a deterministic function of mu (no GFlowNet approximation error),
    so the only error in the Sobol estimate comes from the PCE fit.

    Args:
        mu: (L, 2) array of reward parameterisations
        step: trajectory step (0, 1, or 2)
    Returns:
        policies: (L, 3) probability vectors (3 actions)
    """
    rng = np.random.RandomState(seed_policy)
    # Signal strength: controls variance explained
    b = 2.0
    # Action-specific offsets (fixed across mu draws; controls baseline preferences)
    offsets = rng.randn(2)   # K-1 = 2 ALR components

    L = mu.shape[0]
    alr = np.zeros((L, 2))   # K-1=2 ALR components

    if step == 0:
        # Only mu1 (index 0) drives variance
        for k in range(2):
            alr[:, k] = offsets[k] + b * mu[:, 0]
    elif step == 1:
        # Only mu2 (index 1)
        for k in range(2):
            alr[:, k] = offsets[k] + b * mu[:, 1]
    elif step == 2:
        # Equal contribution from mu1 and mu2
        for k in range(2):
            alr[:, k] = offsets[k] + b / np.sqrt(2) * mu[:, 0] + b / np.sqrt(2) * mu[:, 1]
    else:
        raise ValueError(f"Unknown step: {step}")

    # ALR inverse: softmax([alr_0, alr_1, 0])
    logits = np.concatenate([alr, np.zeros((L, 1))], axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


def run_validation(
    L_grid: list = None,
    pce_degree: int = 3,
    n_eval: int = 1000,
    output_dir: str = "results/sobol_validation",
    seed: int = 0,
) -> dict:
    """Run ground-truth Sobol validation across a range of ensemble sizes.

    Args:
        L_grid:     list of ensemble sizes to test
        pce_degree: PCE polynomial degree
        n_eval:     number of mu samples for Monte Carlo ground-truth check
        output_dir: where to save results

    Returns:
        results dict with estimated vs. true Sobol indices and error metrics
    """
    if L_grid is None:
        L_grid = [10, 20, 30, 50, 75, 100, 150, 200]

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    results = {"L_grid": L_grid, "pce_degree": pce_degree, "steps": {}}

    print("=" * 70)
    print("SOBOL GROUND-TRUTH VALIDATION")
    print(f"  PCE degree: {pce_degree}, L_grid: {L_grid}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # For each step, fit PCE at each ensemble size and compare to truth
    # ----------------------------------------------------------------
    for step in range(3):
        step_results = {"L": [], "estimated_fo": [], "true_fo": [], "error_fo": [],
                        "estimated_to": [], "true_to": [], "error_to": [],
                        "theorem_A_bound": []}
        true_fo = np.array(TRUE_SOBOL[step]["first_order"])  # (K-1, d)
        true_to = np.array(TRUE_SOBOL[step]["total_order"])

        for L in L_grid:
            # Draw mu training samples
            mu_train = rng.randn(L, 2)
            pol_train = synthetic_policy(mu_train, step, seed_policy=step * 10 + 1)

            # Fit PCE
            surr = PCESurrogate(degree=pce_degree, basis="hermite", ridge_lambda=1e-4)
            surr.fit(mu_train, pol_train)
            sobol = surr.sobol_indices()

            est_fo = sobol["first_order"]   # (K-1, d) = (2, 2)
            est_to = sobol["total_order"]

            error_fo = float(np.mean(np.abs(est_fo - true_fo)))
            error_to = float(np.mean(np.abs(est_to - true_to)))

            # Theorem A bound
            try:
                bound = surr.sample_complexity_bound(target_sobol_error=0.05, confidence=0.95)
            except Exception:
                bound = -1

            step_results["L"].append(L)
            step_results["estimated_fo"].append(est_fo.tolist())
            step_results["true_fo"].append(true_fo.tolist())
            step_results["error_fo"].append(error_fo)
            step_results["estimated_to"].append(est_to.tolist())
            step_results["true_to"].append(true_to.tolist())
            step_results["error_to"].append(error_to)
            step_results["theorem_A_bound"].append(bound)

            print(f"  Step {step}, L={L:4d}: "
                  f"FO error={error_fo:.4f}, TO error={error_to:.4f}, "
                  f"TheoremA bound={bound}")

        results["steps"][step] = step_results

    # ----------------------------------------------------------------
    # Monte Carlo ground truth verification (sanity check)
    # ----------------------------------------------------------------
    # Verify analytical Sobol by MC: draw many mu, compute policy variance
    print("\nMonte Carlo verification of analytical Sobol indices:")
    print("  (Uses Saltelli estimator: S_i = Var[E[Y|mu_i]] / Var[Y])")
    mu_mc = rng.randn(n_eval, 2)
    for step in range(3):
        pol_mc = synthetic_policy(mu_mc, step, seed_policy=step * 10 + 1)  # (n_eval, 3)
        # ALR transform
        eps = 1e-9
        p = np.clip(pol_mc, eps, 1.0)
        alr_mc = np.log(p[:, :-1] / p[:, -1:])   # (n_eval, 2)
        for dim in range(2):
            # Correct Saltelli estimator: S_i = Var_mu_i[ E[Y | mu_i] ] / Var[Y]
            # Condition on mu_i (dim), compute conditional mean by binning
            y = alr_mc[:, 0]   # representative ALR component
            D_total = float(np.var(y))
            if D_total < 1e-10:
                continue
            # Bin mu_i into n_bins quantile bins
            n_bins = 30
            bins = np.quantile(mu_mc[:, dim], np.linspace(0, 1, n_bins + 1))
            bin_idx = np.digitize(mu_mc[:, dim], bins[1:-1])
            # Conditional means E[Y | mu_i in bin b], weighted by bin count
            cond_means = []
            weights = []
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.sum() > 2:
                    cond_means.append(float(np.mean(y[mask])))
                    weights.append(int(mask.sum()))
            if not cond_means:
                continue
            w = np.array(weights, dtype=float); w /= w.sum()
            mu_cond = float(np.dot(w, cond_means))
            var_cond_means = float(np.dot(w, (np.array(cond_means) - mu_cond) ** 2))
            S_i_mc = var_cond_means / D_total
            print(f"  Step {step}, dim {dim}: MC S={S_i_mc:.3f} (true={TRUE_SOBOL[step]['first_order'][0][dim]:.1f})")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: Sobol error at each L (mean absolute, all steps/dims)")
    for L_idx, L in enumerate(L_grid):
        errors = [results["steps"][s]["error_fo"][L_idx] for s in range(3)]
        print(f"  L={L:4d}: mean Sobol error = {np.mean(errors):.4f}")

    # Save
    out_path = os.path.join(output_dir, "validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


def run_sample_complexity_experiment(
    target_errors: list = None,
    pce_degree: int = 3,
    output_dir: str = "results/sobol_validation",
    seed: int = 42,
) -> dict:
    """Verify Theorem A: compare bound to empirically required ensemble size.

    For each target Sobol error epsilon, find the smallest L such that the
    PCE achieves error < epsilon (empirically), and compare to the Theorem A bound.
    Uses the step-0 setting (step driven by mu1 only) with known Sobol S=(1,0).
    """
    if target_errors is None:
        target_errors = [0.20, 0.15, 0.10, 0.07, 0.05]

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    true_fo = np.array([[1.0, 0.0], [1.0, 0.0]])  # step 0

    print("\n" + "=" * 70)
    print("THEOREM A: Sample Complexity Bound vs. Empirical Convergence")
    print("=" * 70)

    L_max = 500
    L_dense = sorted(set(list(range(5, 50, 5)) + list(range(50, L_max + 1, 25))))
    mu_train_big = rng.randn(L_max, 2)
    pol_train_big = synthetic_policy(mu_train_big, step=0, seed_policy=1)

    results = {"target_errors": target_errors, "entries": []}

    for eps in target_errors:
        # Find empirical L* (smallest L achieving error < eps averaged over 10 repeats)
        L_empirical = None
        for L in L_dense:
            errors = []
            for rep in range(5):
                idx = rng.choice(L_max, L, replace=False)
                surr = PCESurrogate(degree=pce_degree, ridge_lambda=1e-4)
                surr.fit(mu_train_big[idx], pol_train_big[idx])
                sobol = surr.sobol_indices()
                errors.append(float(np.mean(np.abs(sobol["first_order"] - true_fo))))
            if np.mean(errors) < eps:
                L_empirical = L
                break

        # Theorem A bound (using sigma2 estimated at L=100)
        surr_ref = PCESurrogate(degree=pce_degree, ridge_lambda=1e-4)
        surr_ref.fit(mu_train_big[:100], pol_train_big[:100])
        L_bound = surr_ref.sample_complexity_bound(target_sobol_error=eps, confidence=0.95)

        ratio = L_bound / L_empirical if L_empirical else float("inf")
        print(f"  eps={eps:.2f}: empirical L*={L_empirical}, Theorem A bound={L_bound}, ratio={ratio:.2f}")
        results["entries"].append({
            "target_error": eps,
            "L_empirical": L_empirical,
            "L_theorem_A": L_bound,
            "ratio": ratio,
        })

    out_path = os.path.join(output_dir, "sample_complexity_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pce_degree", type=int, default=3)
    parser.add_argument("--output_dir", default="results/sobol_validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_theorem_a", action="store_true")
    args = parser.parse_args()

    run_validation(pce_degree=args.pce_degree, output_dir=args.output_dir, seed=args.seed)
    if not args.skip_theorem_a:
        run_sample_complexity_experiment(pce_degree=args.pce_degree,
                                         output_dir=args.output_dir, seed=args.seed + 1)
