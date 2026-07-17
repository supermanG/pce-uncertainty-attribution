"""Sequential Bayesian-optimization / OED: per-step acquisition sensitivity.

Extends the single-decision BO demo to a T-step design campaign, giving a per-step
Sobol decomposition that mirrors the Buchwald-Hartwig per-step story. The objective's
epistemic uncertainty is parameterised by the leading Karhunen-Loeve modes of the GP
posterior (exactly independent standard Gaussians, so Hermite PCE is exactly valid).
A reference design path is the greedy argmax on the posterior mean; at step t the
acquisition policy is a soft-Thompson choice among the REMAINING candidates. We report,
per step, which posterior-uncertainty modes drive the decision, the decision fragility,
and the surrogate accuracy.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from decision_studies.bayesian_optimization.bo_oed_demo import gp_posterior
from core.sparse_pce import SparsePCESurrogate
from core.fragility_metrics import tv_barycenter


def run(out_dir, grid=5, d=5, temp=1.0, ls=0.45, T=4, L_train=50, L_test=100, degree=3, seed=0):
    rng = np.random.RandomState(seed)
    g = np.linspace(0.05, 0.95, grid)
    X = np.array([[a, b] for a in g for b in g]); K = len(X)
    f_true = lambda Z: np.sin(3 * np.pi * Z[:, 0]) * np.cos(3 * np.pi * Z[:, 1]) + 0.3 * Z[:, 0]
    Xo = rng.rand(6, 2); yo = f_true(Xo) + 0.05 * rng.randn(6)
    m, C = gp_posterior(Xo, yo, X, ls=ls)
    C = 0.5 * (C + C.T); lam, V = np.linalg.eigh(C)
    o = np.argsort(lam)[::-1]; lam, V = np.clip(lam[o], 0, None), V[:, o]; sql = np.sqrt(lam)

    def sample(n, rs):
        z = np.random.RandomState(rs).randn(n, K)
        f = m[None, :] + (z * sql[None, :]) @ V.T
        return z[:, :d], f
    mu_tr, f_tr = sample(L_train, 1)
    mu_te, f_te = sample(L_test, 2)

    # greedy reference design path on the posterior mean
    ref, avail = [], list(range(K))
    mm = m.copy()
    for _ in range(T):
        j = avail[int(np.argmax(mm[avail]))]; ref.append(j); avail.remove(j)

    def step_policy(f, chosen):
        rem = [j for j in range(K) if j not in chosen]
        z = f[:, rem] / temp; z = z - z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        return p, rem

    res = {"K": K, "d": d, "T": T, "ref_path": ref, "steps": {}}
    print(f"=== Sequential BO/OED (K={K}, d={d} KL modes, T={T} steps) ===")
    print(f"{'step':>4} | {'nrem':>4} | {'relMSE':>7} | {'fragility':>9} | dom-mode-Sobol")
    print("-" * 56)
    for t in range(T):
        P_tr, rem = step_policy(f_tr, ref[:t])
        P_te, _ = step_policy(f_te, ref[:t])
        s = SparsePCESurrogate(degree=degree, basis="hermite", method="lars")
        s.fit(mu_tr, P_tr)
        rel = float(np.mean((s.predict(mu_te) - P_te) ** 2) / (np.var(P_te) + 1e-12))
        fo = s.sobol_indices()["first_order"].mean(0)
        frag = tv_barycenter(P_te)
        res["steps"][t] = {"n_remaining": len(rem), "relMSE": rel,
                           "fragility": float(frag), "mode_first_order": fo.tolist()}
        print(f"{t:>4} | {len(rem):>4} | {rel:>7.3f} | {frag:>9.3f} | max={fo.max():.3f} argmax={int(fo.argmax())}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "bo_oed_sequential.json"), "w"), indent=2)
    print(f"[saved] {out_dir}/bo_oed_sequential.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cluster_results/bo_oed_seq")
    ap.add_argument("--T", type=int, default=4)
    a = ap.parse_args()
    run(a.out, T=a.T)
