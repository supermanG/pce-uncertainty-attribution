"""BH robustness for the second-round review: attribution-ranking stability across retained
dimension, calibration metrics beyond coverage (CRPS, PIT), and the distance-correlation
dependence numbers. Reuses the archived BH GFlowNet ensemble and the extended-analysis stack.

This checks: (1) that the ATTRIBUTION ranking (not just predictive error) is
stable in the retained dimension; (2) report actual distance-correlation values; (4) report
CRPS / PIT / coverage, not coverage alone.
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from decision_studies.common.analyze_ensemble import load_ensemble, pca_embed
from core.sparse_pce import SparsePCESurrogate, max_distance_correlation

COMPS = ["catalyst", "base", "aryl_halide", "additive"]


def crps_samples(pred, obs):
    """CRPS of an empirical predictive sample `pred` (1d) against scalar obs (sample estimator)."""
    pred = np.sort(np.asarray(pred, float))
    n = len(pred)
    e_abse = np.mean(np.abs(pred - obs))
    # E|X-X'| via sorted-sample formula: (2/n^2) sum_i (2i-n-1) x_(i)
    i = np.arange(1, n + 1)
    e_diff = (2.0 / (n * n)) * np.sum((2 * i - n - 1) * pred)
    return float(e_abse - 0.5 * e_diff)


def run(members_dir, out_dir, degree=3, n_train=50, n_test=100, dims=(5, 7, 10),
        basis="apc", method="lars", mc=3000, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    policies, proxy, n_steps, n = load_ensemble(members_dir)
    assert proxy is not None, "need proxy_outputs for PCA"
    rng = np.random.RandomState(seed)

    # ---- (1) attribution-ranking stability across retained dimension --------------
    stability = {}
    for d in dims:
        mu_tr, mu_te, ev = pca_embed(proxy, n_train, d)
        dep = float(max_distance_correlation(mu_tr))
        per_step_top = {}          # dominant uncertainty mode per decision step
        mode_mean = np.zeros(d)     # mean first-order Sobol per mode (over steps, ALR comps)
        for s in range(n_steps):
            surr = SparsePCESurrogate(degree=degree, basis=basis, method=method)
            surr.fit(mu_tr, policies[s][:n_train])
            fo = surr.sobol_indices()["first_order"]        # (K-1, d)
            m = fo.mean(0)                                   # mean over ALR components
            mode_mean += m
            per_step_top[COMPS[s] if s < len(COMPS) else f"step{s}"] = int(np.argmax(m)) + 1
        mode_mean /= n_steps
        rank = [int(x) + 1 for x in np.argsort(mode_mean)[::-1]]
        stability[str(d)] = dict(explained_var=ev, max_distance_corr=dep,
                                 mode_mean_sobol=np.round(mode_mean, 4).tolist(),
                                 mode_rank=rank, per_step_top_mode=per_step_top)

    # ---- (4) calibration: Gaussian predictive (surrogate mean + residual dispersion),
    #          non-circular: residual sigma from TRAIN, PIT/CRPS/coverage on TEST (d=5) ------
    from math import erf, sqrt, pi
    def _Phi(z): return 0.5 * (1 + erf(z / sqrt(2)))
    def _phi(z): return np.exp(-0.5 * z * z) / sqrt(2 * pi)
    def crps_gauss(mu, sig, y):
        z = (y - mu) / sig
        return float(sig * (z * (2 * _Phi(z) - 1) + 2 * _phi(z) - 1 / sqrt(pi)))
    d0 = dims[0]
    mu_tr, mu_te, _ = pca_embed(proxy, n_train, d0)
    crps_all, pit_all = [], []
    cover = {lv: [] for lv in (0.5, 0.8, 0.9, 0.95)}
    zc = {0.5: 0.674, 0.8: 1.282, 0.9: 1.645, 0.95: 1.960}
    for s in range(n_steps):
        surr = SparsePCESurrogate(degree=degree, basis=basis, method=method)
        surr.fit(mu_tr, policies[s][:n_train])
        resid_tr = policies[s][:n_train] - surr.predict(mu_tr)    # train residuals
        sigma = resid_tr.std(0) + 1e-6                            # per-action dispersion
        pred_te = surr.predict(mu_te)                            # (n_test, K) predictive means
        te = policies[s][n_train:n_train + n_test]
        K = te.shape[1]
        for a in range(K):
            for i in range(len(te)):
                mu_i, y = pred_te[i, a], te[i, a]
                crps_all.append(crps_gauss(mu_i, sigma[a], y))
                pit_all.append(_Phi((y - mu_i) / sigma[a]))
            for lv in cover:
                cover[lv].append(float(np.mean(np.abs(te[:, a] - pred_te[:, a]) <= zc[lv] * sigma[a])))
    pit = np.array(pit_all)
    # PIT uniformity: KS distance to Uniform(0,1)
    xs = np.sort(pit); Fn = np.arange(1, len(xs) + 1) / len(xs)
    ks = float(np.max(np.abs(Fn - xs)))
    calib = dict(mean_crps=float(np.mean(crps_all)),
                 pit_ks_to_uniform=ks,
                 pit_mean=float(pit.mean()), pit_std=float(pit.std()),
                 coverage={f"{int(100*lv)}": float(np.mean(cover[lv])) for lv in cover})

    res = dict(members_dir=members_dir, n=n, degree=degree, basis=basis,
               ranking_stability=stability, calibration=calib)
    json.dump(res, open(os.path.join(out_dir, "bh_robustness.json"), "w"), indent=2)

    print("=== attribution-ranking stability vs retained dimension ===")
    for d in dims:
        st = stability[str(d)]
        print(f"  d={d}: expl.var={st['explained_var']:.2f} maxDistCorr={st['max_distance_corr']:.2f} "
              f"mode-rank={st['mode_rank'][:4]} per-step-top={st['per_step_top_mode']}")
    print("\n=== calibration (d=%d, %d MC draws) ===" % (d0, mc))
    print(f"  mean CRPS={calib['mean_crps']:.4f}  PIT mean={calib['pit_mean']:.2f} (ideal 0.5) "
          f"KS-to-uniform={calib['pit_ks_to_uniform']:.3f}")
    print("  coverage: " + ", ".join(f"{k}%->{v:.2f}" for k, v in calib["coverage"].items()))
    print(f"[saved] {out_dir}/bh_robustness.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default=os.path.join(ROOT, "results", "cluster_run", "results", "buchwald_hartwig"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cluster_results", "bh_final"))
    a = ap.parse_args()
    run(a.members, a.out)
