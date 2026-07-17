"""Surrogate comparison: PCE vs MLP vs GP.

Loads saved ensemble member files for BH and/or Sachs, fits all three
surrogates on the training split, and evaluates on the test split.

Outputs:
  - Console: formatted comparison table (Policy MAE, fit time, sample time)
  - JSON:    results/baselines/comparison_{task}.json

Usage (on cluster login node or CPU job):
    python3 experiments/baselines/run_baselines.py --task bh \
        --output_dir results/buchwald_hartwig \
        --n_train 50 --n_test 100 --pca_dim 5

    python3 experiments/baselines/run_baselines.py --task sachs \
        --output_dir results/sachs \
        --n_train 30 --n_test 50 --pca_dim 2

    python3 experiments/baselines/run_baselines.py --task all \
        --bh_dir results/buchwald_hartwig \
        --sachs_dir results/sachs
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.pce_surrogate import TrajectoryPCESurrogate, PCESurrogate
from core.mlp_surrogate import MLPSurrogate
from core.gp_surrogate import GPSurrogate


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_bh_members(output_dir: str, n_train: int, n_test: int, pca_dim: int):
    """Load BH member files and compute PCA coordinates.

    BH member npz files store per-step policies as 'policy_step0',
    'policy_step1', ... and proxy outputs as 'proxy_outputs' (1-D).

    Returns:
        mu_tr, mu_te: (n_train/n_test, pca_dim) standardised reward coords
        tr_pol, te_pol: dict step -> (n_train/n_test, K) policies
        n_steps: int
        exp_var: float
    """
    member_files = sorted(glob.glob(os.path.join(output_dir, "members", "member_*.npz")))
    total = n_train + n_test
    if len(member_files) < total:
        raise RuntimeError(f"Expected {total} member files, found {len(member_files)}")
    member_files = member_files[:total]

    # Discover n_steps from first file
    d0 = np.load(member_files[0])
    step_keys = sorted([k for k in d0.keys() if k.startswith("policy_step")])
    n_steps = len(step_keys)
    if n_steps == 0:
        raise RuntimeError(f"No policy_stepN keys found. Keys: {list(d0.keys())}")

    all_proxy   = []
    # per_step_policies[s] = list of (K,) arrays, one per member
    per_step    = {s: [] for s in range(n_steps)}

    for fp in member_files:
        d = np.load(fp)
        all_proxy.append(d["proxy_outputs"])          # (n_ref,)
        for s, key in enumerate(step_keys):
            per_step[s].append(d[key])                # (K,)

    proxy_outputs = np.stack(all_proxy, axis=0)       # (total, n_ref)
    pca = PCA(n_components=pca_dim)
    mu_raw = pca.fit_transform(proxy_outputs)
    scaler = StandardScaler()
    mu = scaler.fit_transform(mu_raw)
    exp_var = float(pca.explained_variance_ratio_.sum())

    tr_pol = {s: np.array(per_step[s][:n_train])         for s in range(n_steps)}
    te_pol = {s: np.array(per_step[s][n_train:n_train+n_test]) for s in range(n_steps)}

    return mu[:n_train], mu[n_train:], tr_pol, te_pol, n_steps, exp_var


def load_sachs_members(output_dir: str, n_train: int, n_test: int, pca_dim: int):
    """Load Sachs member files and compute PCA coordinates.

    Returns:
        mu_tr, mu_te: (n_train/n_test, pca_dim) standardised reward coords
        tr_pol, te_pol: dict step -> (n_train/n_test, K) policies
        n_steps: int
    """
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    member_files = sorted(glob.glob(os.path.join(output_dir, "members", "member_*.npz")))
    total = n_train + n_test
    if len(member_files) < total:
        raise RuntimeError(f"Expected {total} member files, found {len(member_files)}")
    member_files = member_files[:total]

    all_policies = []
    for fp in member_files:
        d = np.load(fp)
        all_policies.append(d["policies"])    # (n_steps, K)

    full_data = np.load(os.path.join(output_dir, "full_data.npy"))
    covs = []
    for i in range(total):
        rng = np.random.RandomState(i + 100)
        idx = rng.choice(len(full_data), meta["n_obs"], replace=False)
        covs.append(np.cov(full_data[idx].T).flatten())
    covs = np.array(covs)
    pca = PCA(n_components=pca_dim)
    mu_raw = pca.fit_transform(covs)
    scaler = StandardScaler()
    mu = scaler.fit_transform(mu_raw)
    exp_var = float(pca.explained_variance_ratio_.sum())

    n_steps = all_policies[0].shape[0]
    tr_pol = {s: np.array([all_policies[i][s] for i in range(n_train)]) for s in range(n_steps)}
    te_pol = {s: np.array([all_policies[n_train + i][s] for i in range(n_test)]) for s in range(n_steps)}

    return mu[:n_train], mu[n_train:], tr_pol, te_pol, n_steps, exp_var


# ---------------------------------------------------------------------------
# Surrogate evaluation
# ---------------------------------------------------------------------------

def evaluate_surrogate(name, SurrogateClass, kwargs, mu_tr, mu_te, tr_pol, te_pol,
                        n_steps, pce_degree=None, n_sample=10000):
    """Fit a surrogate, compute MAE on test split, time fit and sampling.

    Returns dict with: mae_per_step, mean_mae, fit_time_s, sample_time_s,
                       has_sobol
    """
    print(f"    Fitting {name} ...", flush=True)
    step_maes = []
    fit_times = []
    sample_times = []
    has_sobol = True

    for step in range(n_steps):
        pol_tr = tr_pol[step]   # (n_train, K)
        pol_te = te_pol[step]   # (n_test, K)

        surr = SurrogateClass(**kwargs)

        t0 = time.perf_counter()
        surr.fit(mu_tr, pol_tr)
        fit_times.append(time.perf_counter() - t0)

        # Test MAE: predict at test mu values, compare to actual policies
        pred_te = surr.predict(mu_te)   # (n_test, K)
        mae = float(np.abs(pred_te - pol_te).mean())
        step_maes.append(mae)

        # Sample timing: draw n_sample policies
        t0 = time.perf_counter()
        _ = surr.sample_policies(n_sample)
        sample_times.append(time.perf_counter() - t0)

        if name == "PCE":
            sobol = surr.sobol_indices()
            has_sobol = sobol is not None
        elif name == "MLP":
            has_sobol = False

    return {
        "name": name,
        "mae_per_step": step_maes,
        "mean_mae": float(np.mean(step_maes)),
        "fit_time_s": float(np.sum(fit_times)),
        "fit_time_per_step_s": float(np.mean(fit_times)),
        "sample_time_s": float(np.sum(sample_times)),
        "sample_time_per_step_s": float(np.mean(sample_times)),
        "has_sobol": has_sobol,
    }


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(task: str, mu_tr, mu_te, tr_pol, te_pol, n_steps,
                   pce_degree: int = 3, n_sample: int = 10000) -> dict:
    """Run PCE, MLP, GP on a single task and return results."""
    print(f"\n  Running surrogates on {task.upper()} "
          f"(n_train={len(mu_tr)}, n_test={len(mu_te)}, "
          f"n_steps={n_steps}, d={mu_tr.shape[1]}) ...", flush=True)

    surrogates = [
        ("PCE", TrajectoryPCESurrogateWrapper,
         {"degree": pce_degree, "basis": "hermite"}),
        ("MLP", MLPSurrogate,
         {"hidden_layer_sizes": (64, 64), "max_iter": 1000}),
        ("GP",  GPSurrogate,
         {"n_restarts_optimizer": 1, "normalize_y": True}),
    ]

    results = []
    for name, cls, kwargs in surrogates:
        try:
            r = evaluate_surrogate(
                name, cls, kwargs,
                mu_tr, mu_te, tr_pol, te_pol,
                n_steps, pce_degree=pce_degree, n_sample=n_sample,
            )
            results.append(r)
            print(f"      {name}: MAE={r['mean_mae']:.4f}  "
                  f"fit={r['fit_time_s']:.1f}s  "
                  f"sample={r['sample_time_s']:.2f}s")
        except Exception as e:
            print(f"      {name}: FAILED ({e})")
            results.append({"name": name, "error": str(e)})

    return {"task": task, "surrogates": results}


# ---------------------------------------------------------------------------
# Thin wrapper so PCESurrogate matches the per-step interface expected by
# evaluate_surrogate (which calls .fit / .predict / .sample_policies per step)
# ---------------------------------------------------------------------------

class TrajectoryPCESurrogateWrapper(PCESurrogate):
    """Alias of PCESurrogate with __init__ signature matching run_comparison."""
    def __init__(self, degree=3, basis="hermite", **kw):
        super().__init__(degree=degree, basis=basis, **kw)


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(all_results: list):
    """Print a LaTeX-friendly comparison table to stdout."""
    print("\n" + "=" * 80)
    print("SURROGATE COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Surrogate':<10} {'Task':<8} {'MAE':>8}  {'Sobol?':>8}  {'Fit(s)':>8}  {'Sample10k(s)':>14}"
    print(header)
    print("-" * 80)
    for task_res in all_results:
        task = task_res["task"]
        for r in task_res["surrogates"]:
            if "error" in r:
                print(f"{r['name']:<10} {task:<8} {'ERROR':>8}")
                continue
            sobol_str = "yes" if r.get("has_sobol") else ("~MC" if r["name"] == "GP" else "no")
            print(f"{r['name']:<10} {task:<8} {r['mean_mae']:>8.4f}  "
                  f"{sobol_str:>8}  {r['fit_time_s']:>8.2f}  "
                  f"{r['sample_time_s']:>14.3f}")
    print("=" * 80)
    print("\nLaTeX table rows (MAE only):")
    for task_res in all_results:
        task = task_res["task"]
        maes = {r["name"]: r.get("mean_mae", float("nan"))
                for r in task_res["surrogates"]}
        print(f"  {task}: PCE={maes.get('PCE', 'nan'):.4f}  "
              f"MLP={maes.get('MLP', 'nan'):.4f}  GP={maes.get('GP', 'nan'):.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def collect_environment_info() -> dict:
    """Collect hardware and software environment details for reproducibility.

    Reports CPU model, RAM, GPU (if available), Python version, and library
    versions used in the surrogate comparison.  These are written into the
    output JSON so that absolute timings can be interpreted in context.

    Returns:
        dict with keys: cpu, ram_gb, gpu, python, numpy, scipy, sklearn, torch
    """
    import platform
    env = {
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }

    # RAM
    try:
        import psutil
        env["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        env["ram_gb"] = "unknown (psutil not installed)"

    # Library versions
    env["numpy"] = np.__version__
    try:
        import scipy; env["scipy"] = scipy.__version__
    except ImportError:
        env["scipy"] = "not installed"
    try:
        import sklearn; env["sklearn"] = sklearn.__version__
    except ImportError:
        env["sklearn"] = "not installed"

    # GPU
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
        else:
            env["gpu"] = "none (CPU only)"
    except ImportError:
        env["torch"] = "not installed"
        env["gpu"] = "none (torch not installed)"

    return env


def describe_surrogate_architectures() -> dict:
    """Return a machine-readable description of each surrogate's configuration.

    This documents what 'fit time' and 'sampling cost' refer to concretely:
      - fit time:  wall-clock time to train the surrogate on the training split
                   (per-step, summed across all trajectory steps)
      - sample time: wall-clock time to draw n_sample policy vectors from the
                     fitted surrogate (per-step, summed)
    """
    return {
        "PCE": {
            "method": "Polynomial Chaos Expansion with ridge regression",
            "basis": "Hermite (orthonormal w.r.t. N(0,1))",
            "ridge_lambda": 1e-4,
            "notes": "Closed-form fit via (Phi^T Phi + lambda I)^{-1} Phi^T y. "
                     "Sobol indices computed analytically from PCE coefficients.",
        },
        "MLP": {
            "method": "Multi-layer perceptron (scikit-learn MLPRegressor)",
            "hidden_layers": "(64, 64)",
            "activation": "relu",
            "solver": "adam",
            "l2_regularisation": 1e-3,
            "max_iter": 1000,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "notes": "Multi-output regression (all K-1 ALR components simultaneously). "
                     "No analytical Sobol indices; spread comes purely from input variation.",
        },
        "GP": {
            "method": "Gaussian Process (scikit-learn GaussianProcessRegressor)",
            "kernel": "ConstantKernel * Matern(nu=2.5) + WhiteKernel",
            "length_scale_bounds": "(0.01, 100)",
            "n_restarts_optimizer": 1,
            "normalize_y": True,
            "notes": "One GP per ALR component. Sobol indices approximated via "
                     "Saltelli MC estimator (O(n_mc * K * n_train), not analytical).",
        },
        "timing_definitions": {
            "fit_time_s": "Total wall-clock time (time.perf_counter) to fit the "
                          "surrogate on the training split, summed across all "
                          "trajectory steps.",
            "sample_time_s": "Total wall-clock time to draw n_sample=10000 policy "
                             "vectors from the fitted surrogate, summed across all "
                             "trajectory steps.",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["bh", "sachs", "all"], default="all")
    parser.add_argument("--bh_dir",    default="results/buchwald_hartwig")
    parser.add_argument("--sachs_dir", default="results/sachs")
    parser.add_argument("--output_dir", default=None,
                        help="For single-task mode; overrides --bh_dir/--sachs_dir")
    parser.add_argument("--n_train_bh",   type=int, default=50)
    parser.add_argument("--n_test_bh",    type=int, default=100)
    parser.add_argument("--pca_dim_bh",   type=int, default=5)
    parser.add_argument("--pce_degree_bh",type=int, default=3)
    parser.add_argument("--n_train_sachs",type=int, default=30)
    parser.add_argument("--n_test_sachs", type=int, default=50)
    parser.add_argument("--pca_dim_sachs",type=int, default=2)
    parser.add_argument("--pce_degree_sachs", type=int, default=3)
    parser.add_argument("--n_sample", type=int, default=10000)
    parser.add_argument("--results_dir", default="results/baselines")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Collect and display environment info
    env_info = collect_environment_info()
    arch_info = describe_surrogate_architectures()
    print("\n" + "=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    for k, v in env_info.items():
        print(f"  {k}: {v}")
    print()

    all_results = []

    # ---- BH ----------------------------------------------------------------
    if args.task in ("bh", "all"):
        bh_dir = args.output_dir if (args.task == "bh" and args.output_dir) else args.bh_dir
        print(f"\nLoading BH members from {bh_dir} ...", flush=True)
        try:
            mu_tr, mu_te, tr_pol, te_pol, n_steps, exp_var = load_bh_members(
                bh_dir, args.n_train_bh, args.n_test_bh, args.pca_dim_bh)
            print(f"  PCA explained variance: {exp_var:.3f}, steps: {n_steps}")
            res = run_comparison("bh", mu_tr, mu_te, tr_pol, te_pol, n_steps,
                                 pce_degree=args.pce_degree_bh, n_sample=args.n_sample)
            all_results.append(res)
        except Exception as e:
            print(f"  BH load failed: {e}")

    # ---- Sachs -------------------------------------------------------------
    if args.task in ("sachs", "all"):
        sachs_dir = args.output_dir if (args.task == "sachs" and args.output_dir) else args.sachs_dir
        print(f"\nLoading Sachs members from {sachs_dir} ...", flush=True)
        try:
            mu_tr, mu_te, tr_pol, te_pol, n_steps, exp_var = load_sachs_members(
                sachs_dir, args.n_train_sachs, args.n_test_sachs, args.pca_dim_sachs)
            print(f"  PCA explained variance: {exp_var:.3f}, steps: {n_steps}")
            res = run_comparison("sachs", mu_tr, mu_te, tr_pol, te_pol, n_steps,
                                 pce_degree=args.pce_degree_sachs, n_sample=args.n_sample)
            all_results.append(res)
        except Exception as e:
            print(f"  Sachs load failed: {e}")

    print_table(all_results)

    # Save JSON with environment and architecture metadata
    output = {
        "environment": env_info,
        "surrogate_architectures": arch_info,
        "results": all_results,
    }
    out_path = os.path.join(args.results_dir, "comparison.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
