"""Real-data Bayesian optimization / OED on the Doyle-Dreher reaction space.

Same provable-coupling construction as bo_oed_demo.py, but the objective is now the
REAL measured Buchwald-Hartwig yield and the candidate space is the REAL reaction grid,
not a synthetic function. A Gaussian process is fit on a seed set of observed reactions
(one-hot component features -> measured yield); the next-experiment decision is a
soft-Thompson acquisition policy over an unobserved candidate pool. The GP posterior's
epistemic uncertainty is parameterised by the leading Karhunen-Loeve modes of the
posterior covariance at the pool, which are exactly independent standard Gaussians, so
the Hermite PCE is exactly orthonormal and the Sobol indices are exactly valid with no
distributional assumption. We attribute the next-experiment choice to those posterior
modes and report surrogate accuracy (relMSE), decision fragility, and convergence in the
number of retained modes.

This makes the OED demonstration a real-data study: which directions of posterior
uncertainty over real reaction yields most drive the choice of the next experiment.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from core.sparse_pce import SparsePCESurrogate
from core.fragility_metrics import tv_barycenter, interaction_summary
from decision_studies.bayesian_optimization.bo_oed_demo import rbf, gp_posterior
from experiments.buchwald_hartwig.run_experiment import load_dataset


def onehot(reactions, n_components):
    parts = []
    for i, n in enumerate(n_components):
        oh = np.zeros((len(reactions), n))
        oh[np.arange(len(reactions)), reactions[:, i].astype(int)] = 1.0
        parts.append(oh)
    return np.concatenate(parts, axis=1)


def build_real(seed=0, n_obs=30, n_cand=15, pool=600, d=8, temp=None, temp_mult=4.0,
               L_train=1500, L_test=400):
    ds = load_dataset()
    assert ds["source"] == "real"
    N = len(ds["yields"])
    X = onehot(ds["reactions"], ds["n_components"])            # (N, 45) real features
    y = ds["yields"].astype(float)
    ymu, ysd = y.mean(), y.std()
    yz = (y - ymu) / ysd                                       # standardised objective
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    obs_idx = perm[:n_obs]
    pool_idx = perm[n_obs:n_obs + pool]
    X_obs, y_obs = X[obs_idx], yz[obs_idx]
    X_pool = X[pool_idx]
    # median-heuristic lengthscale on the pool features
    dif = X_pool[:, None, :] - X_pool[None, :, :]
    d2 = (dif ** 2).sum(-1)
    ls = float(np.sqrt(np.median(d2[d2 > 0]) / 2.0))
    # shortlist the acquisition to the contention set: the top-n_cand candidates by
    # posterior mean (realistic OED -- one does not deliberate over thousands of
    # obviously-poor reactions -- and low-rank, so a few posterior modes explain the choice)
    m_pool, _ = gp_posterior(X_obs, y_obs, X_pool, ls=ls, sf=1.0, sn=0.1)
    top = np.argsort(m_pool)[::-1][:n_cand]
    X_cand = X_pool[top]
    K = len(X_cand)
    m, C = gp_posterior(X_obs, y_obs, X_cand, ls=ls, sf=1.0, sn=0.1)
    C = 0.5 * (C + C.T)
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, V = np.clip(lam[order], 0, None), V[:, order]
    sql = np.sqrt(lam)
    var_expl = lam[:d].sum() / (lam.sum() + 1e-12)
    # temperature: a multiple of the posterior-mean scale so the soft-Thompson
    # acquisition policy is well spread (not a near-discontinuous argmax)
    if temp is None:
        temp = temp_mult * (float(np.std(m)) or 1.0)

    def policies(nsamp, rs):
        z = np.random.RandomState(rs).randn(nsamp, K)          # full posterior draw
        f = m[None, :] + (z * sql[None, :]) @ V.T
        zz = f / temp
        zz = zz - zz.max(1, keepdims=True)
        p = np.exp(zz); p /= p.sum(1, keepdims=True)
        return z[:, :d], p                                     # inputs = top-d KL coords

    mu_tr, P_tr = policies(L_train, 1)
    mu_te, P_te = policies(L_test, 2)
    y_cand = yz[pool_idx][top]                                 # standardised true yields of candidates
    return dict(K=K, d=d, mu_tr=mu_tr, P_tr=P_tr, mu_te=mu_te, P_te=P_te,
                y_cand=y_cand, lam=lam[:d].tolist(),
                var_expl=float(var_expl), ls=ls, temp=float(temp),
                ymu=float(ymu), ysd=float(ysd),
                mean_entropy=float(np.mean(-(P_tr * np.log(P_tr + 1e-12)).sum(1))),
                maxK_entropy=float(np.log(K)))


def run(out_dir, degree=3, n_obs=30, n_cand=15, **kw):
    D = build_real(n_obs=n_obs, n_cand=n_cand, **kw)
    surr = SparsePCESurrogate(degree=degree, basis="hermite", method="lars")
    surr.fit(D["mu_tr"], D["P_tr"])
    pred = surr.predict(D["mu_te"])
    rel = float(np.mean((pred - D["P_te"]) ** 2) / (np.var(D["P_te"]) + 1e-12))
    si = surr.sobol_indices()
    fo = si["first_order"]
    inter = interaction_summary({"first_order": fo, "total_order": si["total_order"],
                                 "variance": si["variance"]})
    mode_importance = fo.mean(0)
    frag = tv_barycenter(D["P_te"])
    res = dict(dataset="doyle_dreher_real", K=D["K"], d=D["d"], n_obs=n_obs, n_cand=n_cand,
               var_explained=D["var_expl"], ls=D["ls"], temp=D["temp"],
               relMSE=rel, basis_size=float(np.mean(si["basis_size"])),
               fragility_tv=float(frag), policy_entropy=D["mean_entropy"],
               max_entropy=D["maxK_entropy"], mode_first_order_mean=mode_importance.tolist(),
               interaction_fraction=inter["interaction_from_first"],
               training_noise_fraction=0.0)
    os.makedirs(out_dir, exist_ok=True)
    print("=== Real-data Bayesian-optimization / OED (Doyle-Dreher yields) ===")
    print(f"observed reactions n_obs={n_obs}, candidate pool K={D['K']}, KL modes d={D['d']} "
          f"(explain {D['var_expl']:.2f} of posterior var), lengthscale={D['ls']:.2f}")
    print(f"acquisition-policy entropy {D['mean_entropy']:.2f} / max {D['maxK_entropy']:.2f}")
    print(f"surrogate relMSE (test) = {rel:.4f}   sparse basis size ~ {np.mean(si['basis_size']):.1f}")
    print(f"acquisition-decision fragility (TV-from-barycentre) = {frag:.4f}")
    print(f"per-KL-mode mean first-order Sobol = {np.round(mode_importance,3).tolist()}")
    print(f"interaction fraction = {inter['interaction_from_first']:.3f}")
    # convergence in retained KL modes
    conv = {}
    for dd in [3, 5, 8, 12]:
        Dd = build_real(n_obs=n_obs, n_cand=n_cand, **{**kw, "d": dd})
        s = SparsePCESurrogate(degree=degree, basis="hermite", method="lars")
        s.fit(Dd["mu_tr"], Dd["P_tr"])
        conv[dd] = float(np.mean((s.predict(Dd["mu_te"]) - Dd["P_te"]) ** 2)
                         / (np.var(Dd["P_te"]) + 1e-12))
    res["convergence_relMSE_by_d"] = conv
    print("convergence relMSE by retained modes d: "
          + ", ".join(f"d={k}:{v:.3f}" for k, v in conv.items()))
    json.dump(res, open(os.path.join(out_dir, "bo_oed_real_results.json"), "w"), indent=2)
    print(f"[saved] {out_dir}/bo_oed_real_results.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cluster_results", "bo_oed_real"))
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--n_obs", type=int, default=30)
    ap.add_argument("--n_cand", type=int, default=15)
    ap.add_argument("--d", type=int, default=8)
    a = ap.parse_args()
    run(a.out, degree=a.degree, n_obs=a.n_obs, n_cand=a.n_cand, d=a.d)
