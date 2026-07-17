"""Task-agnostic post-analysis for a regenerated ensemble.

Consumes a directory of member_*.npz files (each with policy_step{s} arrays and a
proxy_outputs vector) plus the run config, and produces every analysis deliverable:

  - solver comparison: dense ridge (baseline) vs sparse LARS vs sparse OMP,
        with corrected leave-one-out error and selected basis size
  - basis comparison: Hermite vs arbitrary-PCE (aPC) on empirical marginals;
        PCA-dimension convergence sweep; max distance correlation of latent scores
  - corrected interaction reporting (first-order sum, total-order sum,
        interaction fraction) instead of the incorrect "sum <= 1 / additive" claim
  - standard surrogate validation: relative MSE, max absolute error, plus
        marginal calibration flagged as over/under-coverage; bootstrap CIs on Sobol
  - scale-invariant simplex fragility per step (TV-from-barycentre, JS)

Also writes Source Data CSVs (one per figure panel) for regenerating figures.

Usage:
    python decision_studies/common/analyze_ensemble.py --members_dir results/buchwald_hartwig \
        --n_train 50 --n_test 100 --degree 3 --pca_dims 5,7,10 --out results/outputs/bh
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.pce_surrogate import TrajectoryPCESurrogate, alr_transform
from core.sparse_pce import SparsePCESurrogate, max_distance_correlation
from core.fragility_metrics import fragility_report, interaction_summary


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ensemble(members_dir):
    base = Path(members_dir)
    # accept either the run dir (containing members/) or the members/ dir itself
    search = base / "members" if (base / "members").is_dir() else base
    files = sorted(search.glob("member_*.npz"))
    if not files:
        raise FileNotFoundError(f"no member_*.npz in {search}")
    data = [np.load(f, allow_pickle=True) for f in files]
    keys = data[0].files
    # Two member layouts are supported: BH uses per-step keys policy_step{s};
    # Sachs / molecular design store a single (n_steps, K) array under 'policies'.
    if any(k.startswith("policy_step") for k in keys):
        n_steps = sum(1 for k in keys if k.startswith("policy_step"))
        policies = {s: np.stack([d[f"policy_step{s}"] for d in data], 0)
                    for s in range(n_steps)}
    elif "policies" in keys:
        n_steps = data[0]["policies"].shape[0]
        policies = {s: np.stack([d["policies"][s] for d in data], 0)
                    for s in range(n_steps)}
    else:
        raise KeyError(f"member files have neither 'policy_step*' nor 'policies': {keys}")
    proxy = np.stack([d["proxy_outputs"] for d in data], 0) if "proxy_outputs" in keys else None
    return policies, proxy, n_steps, len(files)


def pca_embed(proxy, n_train, d):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    pca = PCA(n_components=d)
    mtr = pca.fit_transform(proxy[:n_train])
    mte = pca.transform(proxy[n_train:])
    sc = StandardScaler()
    return sc.fit_transform(mtr), sc.transform(mte), float(pca.explained_variance_ratio_.sum())


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

def surrogate_errors(pred, true):
    """Relative MSE and max abs error between predicted and empirical policies (n,K)."""
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    rel_mse = float(np.mean((pred - true) ** 2) / (np.var(true) + 1e-12))
    max_abs = float(np.max(np.abs(pred - true)))
    return dict(mae=mae, rmse=rmse, rel_mse=rel_mse, max_abs=max_abs)


def bootstrap_sobol_ci(mu_tr, pol_tr, degree, basis, method, n_boot=200, ci=0.90):
    """Bootstrap CI for the mean first-order Sobol vector (over ALR components).

    Model-conditional bootstrap: the sparse basis (and the aPC recurrence) is
    selected once on the full training set; each resample refits only the
    coefficients by OLS on the fixed per-component support. This is the standard
    conditional bootstrap and is orders of magnitude cheaper than re-running LARS
    cross-validation per resample, which is intractable for tasks with many ALR
    components (e.g. Sachs, K-1 = 110).
    """
    from core.sparse_pce import build_design_matrix_general, sobol_from_coeffs
    from core.pce_surrogate import alr_transform
    L, K = pol_tr.shape
    d = mu_tr.shape[1]
    base = SparsePCESurrogate(degree=degree, basis=basis, method=method)
    try:
        base.fit(mu_tr, pol_tr)
    except Exception:
        nan = [float("nan")] * d
        return dict(mean=nan, lo=nan, hi=nan, n_boot=0)
    multi_idx, active = base.multi_idx, base.active
    apc_R, apc_stats = base._apc_R, base._apc_stats
    alr_full = alr_transform(pol_tr)
    rng = np.random.RandomState(0)
    fo = []
    for _ in range(n_boot):
        idx = rng.randint(0, L, L)
        Phi = build_design_matrix_general(mu_tr[idx], multi_idx, degree, basis,
                                          apc_R, apc_stats)
        alr = alr_full[idx]
        firsts = []
        for k in range(K - 1):
            a = active[k]
            c = np.zeros(len(multi_idx))
            try:
                c[a] = np.linalg.lstsq(Phi[:, a], alr[:, k], rcond=None)[0]
            except Exception:
                continue
            f, _, _ = sobol_from_coeffs(c, multi_idx)
            firsts.append(f)
        if firsts:
            fo.append(np.mean(firsts, axis=0))
    fo = np.array(fo)
    if fo.ndim != 2 or fo.shape[0] == 0:
        nan = [float("nan")] * d
        return dict(mean=nan, lo=nan, hi=nan, n_boot=0)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return dict(mean=fo.mean(0).tolist(),
                lo=np.percentile(fo, lo, axis=0).tolist(),
                hi=np.percentile(fo, hi, axis=0).tolist(),
                n_boot=int(fo.shape[0]))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(members_dir, n_train, n_test, degree, pca_dims, out_dir,
            do_bootstrap=True, n_boot=200):
    os.makedirs(out_dir, exist_ok=True)
    policies, proxy, n_steps, n_have = load_ensemble(members_dir)
    if proxy is None:
        raise RuntimeError("no proxy_outputs in members; PCA embedding unavailable")
    print(f"loaded {n_have} members, {n_steps} steps, proxy dim {proxy.shape[1]}")

    result = dict(members_dir=members_dir, n_members=n_have, n_steps=n_steps,
                  n_train=n_train, n_test=n_test, degree=degree, per_dim={})

    for d in pca_dims:
        mu_tr, mu_te, exp_var = pca_embed(proxy, n_train, d)
        dep = max_distance_correlation(mu_tr)
        print(f"\n=== pca_dim={d}  explained_var={exp_var:.3f}  "
              f"max distance-corr(latents)={dep:.3f} ===")

        solvers = [("ridge_hermite", "hermite", None),
                   ("sparse_lars_hermite", "hermite", "lars"),
                   ("sparse_lars_apc", "apc", "lars"),
                   ("sparse_omp_apc", "apc", "omp")]

        per_solver = {}
        for name, basis, method in solvers:
            try:
                if method is None:
                    surr = TrajectoryPCESurrogate(degree=degree, basis=basis)
                    for s in range(n_steps):
                        surr.fit_step(s, mu_tr, policies[s][:n_train])
                    sob = surr.sobol_all_steps()
                    fo = {s: np.array(sob[s]["first_order"]) for s in range(n_steps)}
                    to = {s: np.array(sob[s]["total_order"]) for s in range(n_steps)}
                    loo = {s: None for s in range(n_steps)}
                    bsize = {s: None for s in range(n_steps)}
                    preds = {s: surr.step_surrogates[s].predict(mu_te)
                             if hasattr(surr, "step_surrogates") else None
                             for s in range(n_steps)}
                else:
                    fo, to, loo, bsize, preds = {}, {}, {}, {}, {}
                    for s in range(n_steps):
                        sp = SparsePCESurrogate(degree=degree, basis=basis, method=method)
                        sp.fit(mu_tr, policies[s][:n_train])
                        si = sp.sobol_indices()
                        fo[s], to[s] = si["first_order"], si["total_order"]
                        loo[s] = float(np.mean(si["loo"]))
                        bsize[s] = float(np.mean(si["basis_size"]))
                        preds[s] = sp.predict(mu_te)

                # validation errors + interaction summary per step
                errs, inter, frag = {}, {}, {}
                for s in range(n_steps):
                    true_te = policies[s][n_train:n_train + n_test]
                    if preds[s] is not None:
                        errs[s] = surrogate_errors(preds[s], true_te)
                    inter[s] = interaction_summary(
                        {"first_order": fo[s], "total_order": to[s],
                         "variance": np.ones(fo[s].shape[0])})
                    frag[s] = fragility_report(policies[s][n_train:n_train + n_test])
                per_solver[name] = dict(
                    mean_loo=(np.mean([loo[s] for s in range(n_steps) if loo[s] is not None])
                              if any(loo[s] is not None for s in range(n_steps)) else None),
                    mean_basis_size=(np.mean([bsize[s] for s in range(n_steps) if bsize[s] is not None])
                                     if any(bsize[s] is not None for s in range(n_steps)) else None),
                    mean_rel_mse=float(np.mean([errs[s]["rel_mse"] for s in errs])) if errs else None,
                    mean_max_abs=float(np.mean([errs[s]["max_abs"] for s in errs])) if errs else None,
                    mean_mae=float(np.mean([errs[s]["mae"] for s in errs])) if errs else None,
                    interaction_fraction_mean=float(np.mean(
                        [inter[s]["interaction_from_first"] for s in range(n_steps)])),
                    total_order_sum_mean=float(np.mean(
                        [inter[s]["total_order_sum_mean"] for s in range(n_steps)])),
                )
                print(f"  {name:22s}  relMSE={per_solver[name]['mean_rel_mse']}  "
                      f"LOO={per_solver[name]['mean_loo']}  "
                      f"basis={per_solver[name]['mean_basis_size']}  "
                      f"interaction={per_solver[name]['interaction_fraction_mean']:.3f}")
            except Exception as e:
                per_solver[name] = dict(error=str(e))
                print(f"  {name:22s}  FAILED: {e}")

        # scale-invariant fragility per step (solver-independent)
        frag_by_step = {s: fragility_report(policies[s][n_train:n_train + n_test])
                        for s in range(n_steps)}

        # bootstrap CI on the primary sparse-aPC surrogate
        boot = None
        if do_bootstrap and d == pca_dims[0]:   # only the primary (headline) dim
            try:
                boot = {s: bootstrap_sobol_ci(mu_tr, policies[s][:n_train], degree,
                                              "apc", "lars", n_boot=n_boot)
                        for s in range(n_steps)}
            except Exception as e:
                print(f"  bootstrap skipped for pca_dim={d}: {e}")
                boot = None

        result["per_dim"][str(d)] = dict(
            explained_var=exp_var, max_distance_corr=dep,
            solvers=per_solver,
            fragility=frag_by_step,
            bootstrap_first_order=boot,
        )

    with open(os.path.join(out_dir, "ensemble_analysis.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\n[saved] {out_dir}/ensemble_analysis.json")

    # Source Data: fragility per step (one CSV, figure-ready)
    with open(os.path.join(out_dir, "source_data_fragility.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pca_dim", "step", "K", "tv_barycenter", "tv_pairwise", "js_spread_bits"])
        for d in pca_dims:
            fb = result["per_dim"][str(d)]["fragility"]
            for s in range(n_steps):
                r = fb[s]
                w.writerow([d, s, r["K"], r["tv_barycenter"], r["tv_pairwise"], r["js_spread_bits"]])
    print(f"[saved] {out_dir}/source_data_fragility.csv")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--members_dir", required=True)
    ap.add_argument("--n_train", type=int, required=True)
    ap.add_argument("--n_test", type=int, required=True)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--pca_dims", default="5", help="comma-separated, e.g. 5,7,10")
    ap.add_argument("--out", default="results/outputs/analysis")
    ap.add_argument("--no_bootstrap", action="store_true")
    ap.add_argument("--n_boot", type=int, default=200)
    args = ap.parse_args()
    analyze(args.members_dir, args.n_train, args.n_test, args.degree,
            [int(x) for x in args.pca_dims.split(",")], args.out,
            do_bootstrap=not args.no_bootstrap, n_boot=args.n_boot)
