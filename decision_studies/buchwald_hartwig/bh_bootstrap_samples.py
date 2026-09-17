"""Archive the bootstrap resamples behind main-text Fig. 2c.

Re-runs the model-conditional bootstrap of the per-step first-order Sobol' indices for the
Buchwald-Hartwig study (d = 5, degree 3, sparse LARS with an arbitrary-PCE basis; the
primary fit of ensemble_analysis.json) and stores the 200 resample vectors, so the figure
can show the bootstrap distribution (box plots) instead of bars with interval whiskers, as
the journal's data-presentation policy asks. The random stream is the one used by
analyze_ensemble.bootstrap_sobol_ci (RandomState(0)), so the archived mean / 5th / 95th
percentiles are reproduced exactly; the script checks this before writing.

    python decision_studies/buchwald_hartwig/bh_bootstrap_samples.py
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from decision_studies.common.analyze_ensemble import load_ensemble, pca_embed          # noqa: E402
from core.sparse_pce import (SparsePCESurrogate, build_design_matrix_general,  # noqa: E402
                             sobol_from_coeffs)
from core.pce_surrogate import alr_transform                            # noqa: E402

MEMBERS = os.path.join(ROOT, "results", "buchwald_hartwig")   # member_*.npz from run_all.py
ARCHIVE = os.path.join(ROOT, "results", "cluster_results", "bh_final", "ensemble_analysis.json")
OUT = os.path.join(ROOT, "results", "cluster_results", "bh_final", "bh_bootstrap_samples.json")
STEPS = ["catalyst", "base", "aryl_halide", "additive"]


def bootstrap_samples(mu_tr, pol_tr, degree, basis, method, n_boot=200):
    """Same algorithm as analyze_ensemble.bootstrap_sobol_ci, returning every resample."""
    L, K = pol_tr.shape
    base = SparsePCESurrogate(degree=degree, basis=basis, method=method)
    base.fit(mu_tr, pol_tr)
    multi_idx, active = base.multi_idx, base.active
    apc_R, apc_stats = base._apc_R, base._apc_stats
    alr_full = alr_transform(pol_tr)
    rng = np.random.RandomState(0)
    fo = []
    for _ in range(n_boot):
        idx = rng.randint(0, L, L)
        Phi = build_design_matrix_general(mu_tr[idx], multi_idx, degree, basis, apc_R, apc_stats)
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
    return np.array(fo)


def main():
    arch = json.load(open(ARCHIVE))
    n_train, degree = arch["n_train"], arch["degree"]
    d = 5
    policies, proxy, n_steps, n_have = load_ensemble(MEMBERS)
    print(f"loaded {n_have} members, {n_steps} steps")
    mu_tr, _, exp_var = pca_embed(proxy, n_train, d)
    out = dict(members_dir=arch["members_dir"], n_train=n_train, degree=degree, d=d,
               basis="apc", method="lars", n_boot=200, seed=0,
               note="Model-conditional bootstrap (basis selected once on the full training "
                    "set; coefficients refitted by least squares on each resample of the "
                    "n_train training members). Each row is the mean over the K-1 log-ratio "
                    "expansions of the per-expansion first-order Sobol' index.",
               steps={})
    ok = True
    for s in range(n_steps):
        fo = bootstrap_samples(mu_tr, policies[s][:n_train], degree, "apc", "lars")
        ref = arch["per_dim"][str(d)]["bootstrap_first_order"][str(s)]
        mean, lo, hi = fo.mean(0), np.percentile(fo, 5, axis=0), np.percentile(fo, 95, axis=0)
        match = (np.allclose(mean, ref["mean"], atol=1e-9) and np.allclose(lo, ref["lo"], atol=1e-9)
                 and np.allclose(hi, ref["hi"], atol=1e-9) and fo.shape[0] == ref["n_boot"])
        ok &= match
        print(f"step {s} ({STEPS[s]}): n_boot={fo.shape[0]}  mean={np.round(mean, 4).tolist()}  "
              f"matches archive: {match}")
        out["steps"][STEPS[s]] = dict(step=s, K=int(policies[s].shape[1]),
                                      samples=fo.tolist(),
                                      mean=mean.tolist(), p05=lo.tolist(), p95=hi.tolist(),
                                      median=np.median(fo, axis=0).tolist(),
                                      p25=np.percentile(fo, 25, axis=0).tolist(),
                                      p75=np.percentile(fo, 75, axis=0).tolist())
    if not ok:
        print("ERROR: resamples do not reproduce the archived summaries; nothing written")
        sys.exit(1)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
