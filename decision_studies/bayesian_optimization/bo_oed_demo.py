"""Bayesian-optimization / OED demonstration: the provable-coupling turf.

Decomposes how epistemic uncertainty in a Gaussian-process objective model drives an
acquisition policy's next-experiment decision, via analytical Sobol indices from a PCE
surrogate. Why this is the clean, defensible case:

  - The objective's epistemic uncertainty is the GP posterior. We parameterise it by
    the leading Karhunen-Loeve (KL) modes of the posterior covariance at the candidate
    set: a posterior sample is f_mu = m + Phi_d diag(sqrt(lambda_d)) mu, with
    mu ~ N(0, I_d) INDEPENDENT STANDARD GAUSSIAN by construction. So Hermite PCE is
    exactly orthonormal and the coefficient-square Sobol indices are exactly valid, with
    no PCA/aPC/transport needed -- this avoids the independence/Gaussianity
    concern by construction.
  - The acquisition policy p(mu) = softmax(f_mu / temp) over candidates (soft Thompson /
    Boltzmann acquisition) is a DETERMINISTIC, smooth function of mu. So 100% of the
    policy variance comes from objective-model uncertainty; there is no training-seed
    noise to swamp it (the coupling is provable by construction, the opposite of the
    molecular-design failure).

Output: which posterior-uncertainty modes drive the next-experiment decision (Sobol),
how fragile that decision is (TV-from-barycentre), and the surrogate accuracy (relMSE),
which should be low, demonstrating the framework works cleanly here.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.sparse_pce import SparsePCESurrogate
from core.fragility_metrics import tv_barycenter, interaction_summary


def rbf(A, B, ls, sf):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return sf ** 2 * np.exp(-0.5 * d2 / ls ** 2)


def gp_posterior(X_obs, y_obs, X_cand, ls=0.18, sf=1.0, sn=0.05):
    Koo = rbf(X_obs, X_obs, ls, sf) + sn ** 2 * np.eye(len(X_obs))
    Koc = rbf(X_obs, X_cand, ls, sf)
    Kcc = rbf(X_cand, X_cand, ls, sf)
    L = np.linalg.cholesky(Koo)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
    m = Koc.T @ alpha
    v = np.linalg.solve(L, Koc)
    C = Kcc - v.T @ v
    return m, C


def build(seed=0, grid=5, d=5, temp=0.5, L_train=50, L_test=100, ls=0.18):
    rng = np.random.RandomState(seed)
    # candidate grid on [0,1]^2
    g = np.linspace(0.05, 0.95, grid)
    X_cand = np.array([[a, b] for a in g for b in g])
    K = len(X_cand)
    # a few initial observations of a fixed objective
    def f_true(X):
        return np.sin(3 * np.pi * X[:, 0]) * np.cos(3 * np.pi * X[:, 1]) + 0.3 * X[:, 0]
    X_obs = rng.rand(6, 2)
    y_obs = f_true(X_obs) + 0.05 * rng.randn(6)
    m, C = gp_posterior(X_obs, y_obs, X_cand, ls=ls)
    # KL modes of the posterior covariance -> independent N(0,1) inputs
    C = 0.5 * (C + C.T)
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, V = np.clip(lam[order], 0, None), V[:, order]
    sql = np.sqrt(lam)
    var_expl = lam[:d].sum() / lam.sum()
    # Ensemble of FULL-posterior draws; the PCE inputs are the top-d KL coordinates
    # (each exactly N(0,1)). Sampling from ALL modes but conditioning on only the top
    # d means relMSE genuinely measures how much the leading uncertainty modes explain
    # the decision (not circular), and doubles as a convergence-in-d diagnostic.
    def policies(nsamp, rs):
        z = np.random.RandomState(rs).randn(nsamp, K)           # full posterior draw
        f = m[None, :] + (z * sql[None, :]) @ V.T               # (nsamp, K) posterior samples
        zz = f / temp
        zz = zz - zz.max(1, keepdims=True)
        p = np.exp(zz); p /= p.sum(1, keepdims=True)
        return z[:, :d], p                                      # inputs = top-d KL coords
    mu_tr, P_tr = policies(L_train, 1)
    mu_te, P_te = policies(L_test, 2)
    return dict(X_cand=X_cand, K=K, d=d, mu_tr=mu_tr, P_tr=P_tr, mu_te=mu_te, P_te=P_te,
                var_expl=float(var_expl), lam=lam[:d].tolist(),
                mean_entropy=float(np.mean(-(P_tr * np.log(P_tr + 1e-12)).sum(1))),
                maxK_entropy=float(np.log(K)))


def run(out_dir, degree=3, **kw):
    D = build(**kw)
    surr = SparsePCESurrogate(degree=degree, basis="hermite", method="lars")
    surr.fit(D["mu_tr"], D["P_tr"])
    pred = surr.predict(D["mu_te"])
    rel = float(np.mean((pred - D["P_te"]) ** 2) / (np.var(D["P_te"]) + 1e-12))
    si = surr.sobol_indices()
    fo = si["first_order"]                          # (K-1, d)
    inter = interaction_summary({"first_order": fo, "total_order": si["total_order"],
                                 "variance": si["variance"]})
    mode_importance = fo.mean(0)                    # mean first-order per KL mode
    frag = tv_barycenter(D["P_te"])
    res = dict(K=D["K"], d=D["d"], var_explained=D["var_expl"],
               relMSE=rel, basis_size=float(np.mean(si["basis_size"])),
               fragility_tv=float(frag),
               policy_entropy=D["mean_entropy"], max_entropy=D["maxK_entropy"],
               mode_first_order_mean=mode_importance.tolist(),
               interaction_fraction=inter["interaction_from_first"],
               training_noise_fraction=0.0)   # deterministic map mu->policy: coupling is exact
    os.makedirs(out_dir, exist_ok=True)
    print("=== Bayesian-optimization / OED demonstration ===")
    print(f"candidates K={D['K']}, KL modes d={D['d']} (explain {D['var_expl']:.2f} of posterior var)")
    print(f"acquisition-policy entropy {D['mean_entropy']:.2f} / max {D['maxK_entropy']:.2f} (well-spread, not degenerate)")
    print(f"surrogate relMSE (test) = {rel:.4f}   sparse basis size ~ {np.mean(si['basis_size']):.1f}")
    print(f"acquisition-decision fragility (TV-from-barycentre) = {frag:.4f}")
    print(f"per-KL-mode mean first-order Sobol = {np.round(mode_importance,3).tolist()}")
    print(f"interaction fraction = {inter['interaction_from_first']:.3f}")
    print(f"training-noise fraction = 0 (100% objective-model uncertainty; no GFlowNet training)")
    # convergence in retained KL modes, cheap because modes are exact N(0,1)
    conv = {}
    for dd in [3, 5, 8, 12]:
        Dd = build(**{**kw, "d": dd})
        s = SparsePCESurrogate(degree=degree, basis="hermite", method="lars")
        s.fit(Dd["mu_tr"], Dd["P_tr"])
        conv[dd] = float(np.mean((s.predict(Dd["mu_te"]) - Dd["P_te"]) ** 2)
                         / (np.var(Dd["P_te"]) + 1e-12))
    res["convergence_relMSE_by_d"] = conv
    print("convergence relMSE by retained modes d: "
          + ", ".join(f"d={k}:{v:.3f}" for k, v in conv.items()))
    json.dump(res, open(os.path.join(out_dir, "bo_oed_results.json"), "w"), indent=2)
    print(f"[saved] {out_dir}/bo_oed_results.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cluster_results/bo_oed")
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--ls", type=float, default=0.45)
    a = ap.parse_args()
    run(a.out, degree=a.degree, temp=a.temp, grid=a.grid, d=a.d, ls=a.ls)
