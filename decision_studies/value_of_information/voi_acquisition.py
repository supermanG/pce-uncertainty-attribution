"""Retrospective acquisition curve: resolving uncertainty modes in value-of-information
order vs variance order vs random.

We simulate resolving reward/objective-model uncertainty modes one at a time and track the
residual DECISION value-variability (the value-variance still unexplained by the resolved
modes). Three orders are compared: by decision value-of-information (value-Sobol), by model
uncertainty variance (the default in reward-model active learning), and at random.

Two safeguards against circularity:
  - the value-Sobol ordering is estimated on a TRAIN split of ensemble members and the
    residual value-variance is measured on a held-out TEST split, so variance-order's
    shortfall is an out-of-sample result, not a definition;
  - the value-of-information order is greedy-optimal for this objective by construction, so
    the claim is not "VoI beats variance" but "variance-order can be arbitrarily suboptimal,
    and how much is pipeline-dependent" -- ~0 in the aligned regime (BH), large in the
    anti-aligned regime (RLHF), where variance-order spends its first and largest acquisition
    on the shift-invariant, decision-irrelevant mode.

residual_k = Var_test(V - E[V | resolved modes]) / Var_test(V), with E[V|S] the orthonormal
PCE restricted to basis terms supported on the resolved set S.
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


def _fit(mu_tr, y_tr, degree, basis, q=0.7):
    d = mu_tr.shape[1]
    midx = hyperbolic_multi_indices(d, degree, q)
    apc = (None, None)
    if basis == "apc":
        apc = fit_apc_recurrence(mu_tr, degree)
    Phi = build_design_matrix_general(mu_tr, midx, degree, basis, apc[0], apc[1])
    c, _, _ = fit_sparse(Phi, y_tr, "lars")
    return np.asarray(c, float), midx, apc


def _value_sobol_first(c, midx, d):
    c2 = c ** 2
    order = np.array([np.sum(midx[j]) for j in range(len(midx))])
    D = float(c2[order > 0].sum()) + 1e-12
    fo = np.zeros(d)
    for i in range(d):
        mi = np.array([midx[j][i] for j in range(len(midx))])
        others = np.array([np.sum(np.delete(midx[j], i)) for j in range(len(midx))])
        fo[i] = c2[(mi > 0) & (others == 0)].sum() / D
    return fo


def residual_curves(mu, value, var_frac, degree=3, basis="hermite", seed=0):
    """Return residual-value-variance curves (out-of-sample) for the three acquisition orders."""
    M, d = mu.shape
    ntr = max(d + 2, (M * 3) // 5)
    perm = np.random.RandomState(seed).permutation(M)
    tr, te = perm[:ntr], perm[ntr:]
    c, midx, apc = _fit(mu[tr], value[tr], degree, basis)
    Phi_te = build_design_matrix_general(mu[te], midx, degree, basis, apc[0], apc[1])
    y_te = value[te]; vy = float(np.var(y_te)) + 1e-12
    voi_order = list(np.argsort(_value_sobol_first(c, midx, d))[::-1])
    var_order = list(np.argsort(np.asarray(var_frac))[::-1])
    rnd_order = list(np.random.RandomState(seed + 7).permutation(d))
    supp = [set(np.nonzero(midx[j])[0]) for j in range(len(midx))]
    curves = {}
    for name, order in [("voi", voi_order), ("variance", var_order), ("random", rnd_order)]:
        cur = []
        for k in range(d + 1):
            S = set(order[:k])
            keep = np.array([supp[j].issubset(S) for j in range(len(midx))])
            pred = Phi_te[:, keep] @ c[keep]
            cur.append(float(np.mean((y_te - pred) ** 2) / vy))
        curves[name] = cur
    return curves


# ---- domain (mu, value, var_frac) providers -------------------------------------

def data_bh(M=60, frac=0.10, epochs=150, d=6, n_ref=800, tau=4.0):
    from decision_studies.buchwald_hartwig.bh_closed_loop import load_dataset, train_proxy_on
    from decision_studies.value_of_information.value_of_information import _pca
    ds = load_dataset(); N = len(ds["yields"])
    rng = np.random.RandomState(123); ref = rng.choice(N, n_ref, replace=False)
    ref_rxn = ds["reactions"][ref]; ref_true = ds["yields"][ref].astype(float)
    outs = np.array([train_proxy_on(ds, np.random.RandomState(m).permutation(N)[: int(N * frac)],
                                    seed=m, epochs=epochs).predict_yield(ref_rxn) for m in range(M)])
    mu, lam, _ = _pca(outs, d)
    z = outs / tau; z -= z.max(1, keepdims=True); P = np.exp(z); P /= P.sum(1, keepdims=True)
    value = P @ ref_true
    return [(mu, value, (lam / lam.sum()))], "hermite"


def data_rlhf(n_prompts=48, n=16, M=100, n_train_pairs=5000, beta=0.5, d=4):
    from datasets import load_dataset
    from decision_studies.rlhf.rlhf_real import (_dialogue_split, embed_texts, generate_candidates,
                                    train_rm_ensemble, best_of_n_policy)
    from decision_studies.value_of_information.value_of_information import _pca
    tr = load_dataset("Anthropic/hh-rlhf", split=f"train[:{n_train_pairs}]")
    pool = load_dataset("Anthropic/hh-rlhf", split=f"test[:{n_prompts}]")
    ch, rj = [], []
    for c, r in zip(tr["chosen"], tr["rejected"]):
        pc, rc = _dialogue_split(c); pr, rr = _dialogue_split(r)
        ch.append(pc + " " + rc); rj.append(pr + " " + rr)
    e_ch = embed_texts(ch, "chosen"); e_rj = embed_texts(rj, "rejected"); half = n_train_pairs // 2
    heads = train_rm_ensemble(e_ch[:half], e_rj[:half], M, seed0=0)
    gold = train_rm_ensemble(e_ch[half:], e_rj[half:], 8, seed0=9000)
    prompts = [_dialogue_split(x)[0] for x in pool["chosen"]]
    cands = generate_candidates(prompts, n=n, tag="gen16")
    flat = [prompts[p] + " " + cands[p][i] for p in range(len(prompts)) for i in range(n)]
    e_flat = embed_texts(flat, "cand16")
    S = np.stack([h.score(e_flat) for h in heads], 0).reshape(M, len(prompts), n)
    G = np.mean([h.score(e_flat) for h in gold], 0).reshape(len(prompts), n)
    items = []
    for p in range(len(prompts)):
        mu, lam, _ = _pca(S[:, p, :], d)
        value = (best_of_n_policy(S[:, p, :], beta) * G[p][None, :]).sum(1)
        if value.std() > 1e-9:
            items.append((mu, value, lam / lam.sum()))
    return items, "apc"


def aggregate(items, basis, degree=3):
    allc = {"voi": [], "variance": [], "random": []}
    for (mu, value, vf) in items:
        cur = residual_curves(mu, value, vf, degree, basis)
        for k in allc:
            allc[k].append(cur[k])
    return {k: np.mean(v, 0).tolist() for k, v in allc.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="bh,rlhf")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cluster_results", "voi"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    if "bh" in a.domains:
        items, basis = data_bh(); res["BH"] = aggregate(items, basis)
    if "rlhf" in a.domains:
        items, basis = data_rlhf(); res["RLHF"] = aggregate(items, basis)
    for dom, c in res.items():
        print(f"[{dom}] residual value-variance vs #modes resolved:")
        for k in ("voi", "variance", "random"):
            print(f"    {k:9s}: " + " ".join(f"{x:.3f}" for x in c[k]))
        # headline: area under the curve (lower = faster resolution); and the step-1 gap
        auc = {k: float(np.mean(c[k])) for k in c}
        print(f"    AUC voi={auc['voi']:.3f} variance={auc['variance']:.3f} random={auc['random']:.3f}"
              f"  | step-1 residual voi={c['voi'][1]:.3f} variance={c['variance'][1]:.3f}")
    json.dump(res, open(os.path.join(a.out, "voi_acquisition.json"), "w"), indent=2)
    print(f"[saved] {a.out}/voi_acquisition.json")
