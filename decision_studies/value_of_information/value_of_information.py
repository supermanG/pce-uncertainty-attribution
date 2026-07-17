"""Value-of-information spectra: variance-per-mode vs decision-relevance-per-mode.

The decision-relevant attribution is the Sobol decomposition of the scalar DECISION VALUE
V(xi) = E_{y ~ pi_xi}[r_gold(y)], not of the policy itself (the two differ: the map
model-uncertainty -> policy -> value has a kernel, e.g. softmax ignores a uniform shift).
For each learned-model -> decision pipeline we compute, per reward-model-uncertainty mode
(ordered by variance):

  var_frac[i]     the fraction of reward-model epistemic variance in mode i (PCA/KL spectrum)
  value_sobol[i]  the first-order Sobol index of the DECISION VALUE V on mode i
                  (= expected value of resolving that mode; the EVPPI-style decision relevance)

The message is the GAP between the two spectra: the highest-variance uncertainty direction is
often decision-irrelevant. In RLHF the top-variance mode is the shift-invariant direction the
softmax ignores, so its value Sobol collapses to ~0.

Value-of-information sensitivity / EVPPI is classical (Felli-Hazen 1999; Borgonovo;
Strong-Oakley-Brennan 2014); the contribution here is computing it, analytically from a PCE,
THROUGH non-differentiable trained generative/RL policies where nested-MC EVPPI and
differentiable decision-focused methods cannot operate.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from core.sparse_pce import (hyperbolic_multi_indices, build_design_matrix_general,
                             fit_apc_recurrence, fit_sparse)


# ---------------------------------------------------------------------------
# scalar sparse-PCE Sobol
# ---------------------------------------------------------------------------

def scalar_sobol(mu, y, degree=3, basis="hermite", q=0.7):
    """First- and total-order Sobol indices of scalar y(mu), plus fit relMSE (LOO-style split)."""
    mu = np.asarray(mu, float); y = np.asarray(y, float)
    d = mu.shape[1]
    midx = hyperbolic_multi_indices(d, degree, q)
    apc_R = apc_stats = None
    if basis == "apc":
        apc_R, apc_stats = fit_apc_recurrence(mu, degree)
    Phi = build_design_matrix_general(mu, midx, degree, basis, apc_R, apc_stats)
    c, active, loo = fit_sparse(Phi, y, method="lars")
    c = np.asarray(c, float)
    order = np.array([np.sum(midx[j]) for j in range(len(midx))])
    c2 = c ** 2
    D = float(c2[order > 0].sum()) + 1e-12
    fo = np.zeros(d); to = np.zeros(d)
    for i in range(d):
        mi = np.array([midx[j][i] for j in range(len(midx))])
        others = np.array([np.sum(np.delete(midx[j], i)) for j in range(len(midx))])
        fo[i] = c2[(mi > 0) & (others == 0)].sum() / D
        to[i] = c2[(mi > 0)].sum() / D
    # simple explained-variance check: relMSE = residual var / var(y)
    pred = Phi @ c
    rel = float(np.mean((pred - y) ** 2) / (np.var(y) + 1e-12))
    return fo, to, D, rel


def _pca(X, d):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = (S ** 2) / max(len(X) - 1, 1)
    mu = Xc @ Vt[:d].T                                        # scores, variance-ordered
    mu = (mu - mu.mean(0)) / (mu.std(0) + 1e-12)              # standardise for Hermite
    return mu, lam[:d], float(lam[:d].sum() / (lam.sum() + 1e-12))


# ---------------------------------------------------------------------------
# BH: reward model = yield predictor; gold = measured yield
# ---------------------------------------------------------------------------

def voi_bh(M=60, frac=0.10, epochs=150, d=6, n_ref=800, tau=4.0, degree=3, seed=0):
    from decision_studies.buchwald_hartwig.bh_closed_loop import load_dataset, train_proxy_on
    ds = load_dataset(); N = len(ds["yields"])
    rng = np.random.RandomState(123)
    ref = rng.choice(N, n_ref, replace=False)
    ref_rxn = ds["reactions"][ref]; ref_true = ds["yields"][ref].astype(float)
    outs = []
    for m in range(M):
        idx = np.random.RandomState(seed + m).permutation(N)[: int(N * frac)]
        p = train_proxy_on(ds, idx, seed=seed + m, epochs=epochs)
        outs.append(p.predict_yield(ref_rxn))
    outs = np.array(outs)                                     # (M, n_ref)
    mu, lam, ve = _pca(outs, d)
    # policy pi_m over the reference reactions; value = expected TRUE yield
    z = outs / tau; z -= z.max(1, keepdims=True)
    P = np.exp(z); P /= P.sum(1, keepdims=True)
    value = P @ ref_true                                     # (M,)
    fo, to, D, rel = scalar_sobol(mu, value, degree, "hermite")
    return dict(domain="BH", var_frac=(lam / lam.sum()).tolist(),
                value_sobol_first=fo.tolist(), value_sobol_total=to.tolist(),
                value_relMSE=rel, value_std=float(value.std()), var_explained=ve)


# ---------------------------------------------------------------------------
# BO: objective model = GP posterior; gold = measured yield
# ---------------------------------------------------------------------------

def voi_bo(d=8, degree=3, n_seeds=4):
    from decision_studies.bayesian_optimization.bo_oed_real import build_real
    fos, tos, vfs, rels = [], [], [], []
    for s in range(n_seeds):
        D = build_real(seed=s, d=d)
        mu = D["mu_te"]; P = D["P_te"]; ycand = np.asarray(D["y_cand"], float)
        value = P @ ycand
        fo, to, Dv, rel = scalar_sobol(mu, value, degree, "hermite")
        lam = np.asarray(D["lam"], float)
        fos.append(fo); tos.append(to); vfs.append(lam / lam.sum()); rels.append(rel)
    return dict(domain="BO", var_frac=np.mean(vfs, 0).tolist(),
                value_sobol_first=np.mean(fos, 0).tolist(),
                value_sobol_total=np.mean(tos, 0).tolist(),
                value_relMSE=float(np.mean(rels)))


# ---------------------------------------------------------------------------
# RLHF: reward model ensemble; gold = independent held-out reward model
# ---------------------------------------------------------------------------

def voi_rlhf(n_prompts=48, n=16, M=100, n_train_pairs=5000, beta=0.5, d=4, degree=3, seed=0):
    from datasets import load_dataset
    from decision_studies.rlhf.rlhf_real import (_dialogue_split, embed_texts, generate_candidates,
                                    train_rm_ensemble, best_of_n_policy)
    tr = load_dataset("Anthropic/hh-rlhf", split=f"train[:{n_train_pairs}]")
    pool = load_dataset("Anthropic/hh-rlhf", split=f"test[:{n_prompts}]")
    ch, rj = [], []
    for c, r in zip(tr["chosen"], tr["rejected"]):
        pc, rc = _dialogue_split(c); pr, rr = _dialogue_split(r)
        ch.append(pc + " " + rc); rj.append(pr + " " + rr)
    e_ch = embed_texts(ch, "chosen"); e_rj = embed_texts(rj, "rejected")
    half = n_train_pairs // 2
    heads = train_rm_ensemble(e_ch[:half], e_rj[:half], M, seed0=1000 * seed)
    gold = train_rm_ensemble(e_ch[half:], e_rj[half:], 8, seed0=9000)
    prompts = [_dialogue_split(x)[0] for x in pool["chosen"]]
    cands = generate_candidates(prompts, n=n, tag="gen16")
    flat = [prompts[p] + " " + cands[p][i] for p in range(len(prompts)) for i in range(n)]
    e_flat = embed_texts(flat, "cand16")
    S = np.stack([h.score(e_flat) for h in heads], 0).reshape(M, len(prompts), n)
    G = np.mean([h.score(e_flat) for h in gold], 0).reshape(len(prompts), n)   # gold reward
    fos, tos, vfs, rels = [], [], [], []
    for p in range(len(prompts)):
        mu, lam, ve = _pca(S[:, p, :], d)
        Pi = best_of_n_policy(S[:, p, :], beta)               # (M, n) best-of-n policy per member
        value = (Pi * G[p][None, :]).sum(1)                   # expected GOLD reward (M,)
        if value.std() < 1e-9:
            continue
        fo, to, Dv, rel = scalar_sobol(mu, value, degree, "apc")
        fos.append(fo); tos.append(to); vfs.append(lam / lam.sum()); rels.append(rel)
    return dict(domain="RLHF", var_frac=np.mean(vfs, 0).tolist(),
                value_sobol_first=np.mean(fos, 0).tolist(),
                value_sobol_total=np.mean(tos, 0).tolist(),
                value_relMSE=float(np.mean(rels)), n_prompts=len(fos))


def summary(r):
    vf = np.array(r["var_frac"]); vs = np.array(r["value_sobol_first"])
    order = np.argsort(vf)[::-1]
    rc = float(np.corrcoef(vf, vs)[0, 1]) if len(vf) > 1 else float("nan")
    top_var_mode = int(order[0])
    return (f"[{r['domain']:4}] var_frac={np.round(vf,3).tolist()}  "
            f"value_sobol(1st)={np.round(vs,3).tolist()}  "
            f"top-variance mode value-Sobol={vs[top_var_mode]:.3f}  corr(var,value-Sobol)={rc:+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="bh,bo,rlhf")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cluster_results", "voi"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    if "bh" in a.domains:
        res["BH"] = voi_bh(); print(summary(res["BH"]))
    if "bo" in a.domains:
        res["BO"] = voi_bo(); print(summary(res["BO"]))
    if "rlhf" in a.domains:
        res["RLHF"] = voi_rlhf(); print(summary(res["RLHF"]))
    json.dump(res, open(os.path.join(a.out, "voi_spectra.json"), "w"), indent=2)
    print(f"[saved] {a.out}/voi_spectra.json")
