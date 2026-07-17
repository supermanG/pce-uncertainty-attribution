"""RLHF strong-coupling demonstration.

Decomposes how epistemic uncertainty in a learned reward model (RM) drives the
KL-regularised RLHF generation policy, via analytical Sobol indices from a PCE
surrogate, and shows the applicability diagnostic distinguishing a strong-coupling
(overoptimization) regime from a weak one.

Setup (uses the exact RLHF optimum, not a toy):
  - K candidate completions with a GOLD reward r_gold.
  - An ensemble of proxy RMs, r_m = r_gold + error. STRONG regime: low-rank,
    heavy-tailed structured disagreement (Student-t mode weights) -> the
    overoptimization regime documented by Gao et al. 2023 / Coste et al. 2024, where
    RM epistemic uncertainty (ensemble disagreement) strongly drives the policy.
    WEAK regime: small isotropic Gaussian error -> RM uncertainty barely moves the
    policy.
  - The KL-regularised RLHF-optimal policy is closed form:
        pi_m(y) proportional to pi_ref(y) * exp(r_m(y)/beta)
    i.e. softmax(log pi_ref + r_m/beta). It is a deterministic function of the RM,
    so 100% of the policy variance is reward-model uncertainty (no training-seed
    noise); beta is the KL penalty (small beta = hard optimization = Goodhart).
  - Reward-model uncertainty is parameterised by the leading modes of the RM
    ensemble (non-Gaussian under heavy tails, so we use arbitrary-PCE, aPC).

Outputs: which RM-uncertainty modes drive the generation decision (Sobol), which
completions are fragile to RM uncertainty (the reward-hacking-susceptible ones),
surrogate relMSE, and a beta sweep showing fragility and gold-regret rising as
optimization hardens (overoptimization).
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.sparse_pce import SparsePCESurrogate
from core.fragility_metrics import tv_barycenter, interaction_summary


def make_ensemble(K=12, M=80, d=5, regime="strong", seed=0):
    rng = np.random.RandomState(seed)
    # gold reward: a few clearly-good completions
    r_gold = rng.randn(K); r_gold[:3] += 2.0
    if regime == "strong":
        # heavy-tailed structured RM disagreement with a DECAYING spectrum (not exactly
        # rank-d), so a d-mode PCE truncation incurs a genuine (nonzero) relMSE.
        d_true = min(K, 10)
        B = rng.randn(K, d_true)
        B /= np.linalg.norm(B, axis=0, keepdims=True)
        scale = 1.0 / np.arange(1, d_true + 1) ** 0.7      # spectral decay
        a = rng.standard_t(df=2, size=(M, d_true)) * scale[None, :] * 1.4   # heavy-tailed
        err = a @ B.T
    else:  # weak
        err = 0.25 * rng.randn(M, K)              # light-tailed isotropic error
    r_m = r_gold[None, :] + err                    # (M, K) proxy RM scores
    return r_gold, r_m


def policy(r_m, beta, logpref=None):
    """KL-regularised RLHF optimum: softmax(log pi_ref + r/beta)."""
    K = r_m.shape[1]
    lp = np.zeros(K) if logpref is None else logpref
    z = lp[None, :] + r_m / beta
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p


def embed(r_m, n_train, d):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    pca = PCA(n_components=d)
    mtr = pca.fit_transform(r_m[:n_train]); mte = pca.transform(r_m[n_train:])
    sc = StandardScaler();
    return sc.fit_transform(mtr), sc.transform(mte), float(pca.explained_variance_ratio_.sum())


def analyze(r_m, r_gold, beta, n_train, d, degree=3):
    P = policy(r_m, beta)
    mu_tr, mu_te, ev = embed(r_m, n_train, d)
    surr = SparsePCESurrogate(degree=degree, basis="apc", method="lars")   # non-Gaussian -> aPC
    surr.fit(mu_tr, P[:n_train])
    pred = surr.predict(mu_te)
    rel = float(np.mean((pred - P[n_train:]) ** 2) / (np.var(P[n_train:]) + 1e-12))
    si = surr.sobol_indices()
    inter = interaction_summary({"first_order": si["first_order"],
                                 "total_order": si["total_order"], "variance": si["variance"]})
    frag = tv_barycenter(P)
    # per-completion fragility = std of its probability across the RM ensemble; the most
    # fragile completions are the ones most exposed to reward-model uncertainty
    # (reward-hacking-susceptible).
    comp_frag = P.std(0)
    return dict(relMSE=rel, var_explained=ev, fragility_tv=float(frag),
                mode_first_order=si["first_order"].mean(0).tolist(),
                interaction_fraction=inter["interaction_from_first"],
                top_fragile_completions=np.argsort(comp_frag)[::-1][:3].tolist())


def run(out_dir, K=12, M=80, d=5, beta=0.5, degree=3):
    n_train = M // 2
    os.makedirs(out_dir, exist_ok=True)
    res = {"K": K, "M": M, "d": d, "beta": beta, "regimes": {}}
    for regime in ["strong", "weak"]:
        r_gold, r_m = make_ensemble(K, M, d, regime)
        res["regimes"][regime] = analyze(r_m, r_gold, beta, n_train, d, degree)
    # beta sweep in the strong regime: as the KL penalty falls (harder proxy
    # optimization), the policy's sensitivity to reward-model uncertainty rises -- the
    # overoptimization / reward-hacking-exposure signature.
    r_gold, r_m = make_ensemble(K, M, d, "strong")
    sweep = {}
    for b in [2.0, 1.0, 0.5, 0.25, 0.1]:
        a = analyze(r_m, r_gold, b, n_train, d, degree)
        sweep[b] = {"fragility": a["fragility_tv"], "relMSE": a["relMSE"]}
    res["beta_sweep_strong"] = sweep

    print("=== RLHF reward-model-uncertainty demonstration ===")
    for regime in ["strong", "weak"]:
        a = res["regimes"][regime]
        print(f"\n[{regime} coupling]  relMSE={a['relMSE']:.3f}  fragility(TV)={a['fragility_tv']:.3f}  "
              f"var_expl={a['var_explained']:.2f}")
        print(f"   per-mode first-order Sobol = {np.round(a['mode_first_order'],3).tolist()}")
        print(f"   interaction={a['interaction_fraction']:.3f}  "
              f"fragile completions={a['top_fragile_completions']}")
    print("\nbeta sweep (strong regime): harder optimization (lower beta) -> policy more"
          "\nsensitive to reward-model uncertainty (overoptimization / reward-hacking exposure)")
    print(f"{'beta':>6} | {'fragility':>9} | {'relMSE':>7}")
    for b, v in sweep.items():
        print(f"{b:>6} | {v['fragility']:>9.3f} | {v['relMSE']:>7.3f}")
    json.dump(res, open(os.path.join(out_dir, "rlhf_results.json"), "w"), indent=2)
    print(f"\n[saved] {out_dir}/rlhf_results.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cluster_results/rlhf")
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--M", type=int, default=80)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--degree", type=int, default=3)
    a = ap.parse_args()
    run(a.out, K=a.K, M=a.M, d=a.d, beta=a.beta, degree=a.degree)
