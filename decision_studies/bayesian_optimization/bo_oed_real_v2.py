"""Real-data BO/OED, hardened: multi-seed confidence intervals + design-matrix mutual
coherence (an empirical stand-in for the restricted-isometry premise of Theorem A').

Over several random seeds (each a different seed set of observed reactions and posterior
draws), we report mean and standard deviation for relMSE, fragility, and the retained-mode
convergence, so the single-run numbers are backed by variability. We also report the mutual
coherence of the PCE design matrix (max off-diagonal of the column-normalised Gram of the
Hermite basis evaluated at the samples): low coherence supports the sparse-recovery premise
that the certified-recovery discussion (Theorem A') says must be checked from the realised
design matrix rather than assumed.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from core.sparse_pce import SparsePCESurrogate, build_design_matrix_general
from core.fragility_metrics import tv_barycenter
from decision_studies.bayesian_optimization.bo_oed_real import build_real


def mutual_coherence(surr, mu):
    """Max off-diagonal of the column-normalised Gram of the PCE design matrix at `mu`."""
    Phi = build_design_matrix_general(mu, surr.multi_idx, surr.degree, surr.basis,
                                      surr._apc_R, surr._apc_stats)
    Phi = np.asarray(Phi, dtype=float)
    norms = np.linalg.norm(Phi, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    Q = Phi / norms
    G = np.abs(Q.T @ Q)
    np.fill_diagonal(G, 0.0)
    return float(G.max())


def _mean_sd(xs):
    a = np.array([x for x in xs if x is not None and np.isfinite(x)])
    return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), float("nan"))


def run(out_dir, degree=3, n_obs=30, n_cand=15, d=8, seeds=5):
    os.makedirs(out_dir, exist_ok=True)
    rel_seed, frag_seed, coh_seed = [], [], []
    conv_seed = {dd: [] for dd in [3, 5, 8, 12]}
    modes_seed = []
    var_expl = None
    for s in range(seeds):
        D = build_real(seed=s, n_obs=n_obs, n_cand=n_cand, d=d)
        surr = SparsePCESurrogate(degree=degree, basis="hermite", method="lars")
        surr.fit(D["mu_tr"], D["P_tr"])
        pred = surr.predict(D["mu_te"])
        rel = float(np.mean((pred - D["P_te"]) ** 2) / (np.var(D["P_te"]) + 1e-12))
        rel_seed.append(rel); frag_seed.append(float(tv_barycenter(D["P_te"])))
        modes_seed.append(surr.sobol_indices()["first_order"].mean(0))
        coh_seed.append(mutual_coherence(surr, D["mu_tr"]))
        if var_expl is None:
            var_expl = D["var_expl"]
        for dd in conv_seed:
            Dd = build_real(seed=s, n_obs=n_obs, n_cand=n_cand, d=dd)
            sd = SparsePCESurrogate(degree=degree, basis="hermite", method="lars")
            sd.fit(Dd["mu_tr"], Dd["P_tr"])
            conv_seed[dd].append(float(np.mean((sd.predict(Dd["mu_te"]) - Dd["P_te"]) ** 2)
                                       / (np.var(Dd["P_te"]) + 1e-12)))
    ml = max(len(m) for m in modes_seed)
    modes = np.zeros(ml)
    for m in modes_seed:
        modes[:len(m)] += m
    modes /= len(modes_seed)
    res = dict(dataset="doyle_dreher_real", n_obs=n_obs, n_cand=n_cand, d=d, seeds=seeds,
               var_explained=var_expl,
               relMSE=dict(zip(["mean", "sd"], _mean_sd(rel_seed))),
               fragility=dict(zip(["mean", "sd"], _mean_sd(frag_seed))),
               mutual_coherence=dict(zip(["mean", "sd"], _mean_sd(coh_seed))),
               mode_first_order_mean=modes.tolist(),
               convergence_relMSE_by_d={str(dd): dict(zip(["mean", "sd"], _mean_sd(v)))
                                        for dd, v in conv_seed.items()},
               # per-seed values (seed s = a different random set of observed reactions and
               # posterior draws), kept so the figures can overlay the individual points
               relMSE_per_seed=[float(x) for x in rel_seed],
               fragility_per_seed=[float(x) for x in frag_seed],
               mutual_coherence_per_seed=[float(x) for x in coh_seed],
               convergence_relMSE_by_d_per_seed={str(dd): [float(x) for x in v]
                                                 for dd, v in conv_seed.items()},
               mode_first_order_per_seed=[m.tolist() for m in modes_seed])
    json.dump(res, open(os.path.join(out_dir, "bo_oed_real_v2_results.json"), "w"), indent=2)
    print("=== Real BO/OED (multi-seed) ===")
    print(f"seeds={seeds}, K={n_cand}, d={d}, var_expl={var_expl:.2f}")
    print(f"relMSE   = {res['relMSE']['mean']:.3f} +/- {res['relMSE']['sd']:.3f}")
    print(f"fragility= {res['fragility']['mean']:.3f} +/- {res['fragility']['sd']:.3f}")
    print(f"mutual coherence (design matrix) = {res['mutual_coherence']['mean']:.3f} "
          f"+/- {res['mutual_coherence']['sd']:.3f}")
    print("convergence relMSE by d: " + ", ".join(
        f"d={dd}:{v['mean']:.3f}+/-{v['sd']:.3f}" for dd, v in res["convergence_relMSE_by_d"].items()))
    print(f"[saved] {out_dir}/bo_oed_real_v2_results.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cluster_results", "bo_oed_real"))
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    run(a.out, seeds=a.seeds)
