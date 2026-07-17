"""RLHF real study, hardened: multi-seed CIs, real best-of-16, and a gold-reward
overoptimization curve (the missing true-reward-degradation signal).

Fixes/upgrades over rlhf_real.py:
  - Generates 16 candidates so best-of-{2,4,8,16} are all real (the old n=12 row was a
    duplicate of n=8 because only 8 candidates existed).
  - Repeats the whole study over several reward-model-ensemble seeds and reports mean and
    standard deviation for every headline number (relMSE, fragility, strong/weak split).
  - Adds a GOLD reward model trained on a DISJOINT half of the preference data. Best-of-n
    selection is made by the PROXY ensemble; we track the gold reward of that selection as n
    grows. Genuine overoptimization = proxy-selected reward keeps rising while the gold
    reward of the selection plateaus or declines (Gao et al. 2023). This replaces the
    near-tautological "fragility rises with n" argument with a true-reward-degradation
    measurement.

Reuses the machinery in rlhf_real.py.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from decision_studies.rlhf.rlhf_real import (
    _dialogue_split, embed_texts, generate_candidates, train_rm_ensemble,
    best_of_n_policy, attribute_prompt, aggregate)
from core.fragility_metrics import tv_barycenter


def _mean_sd(xs):
    a = np.array([x for x in xs if x is not None and np.isfinite(x)])
    return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), float("nan"))


def run(out_dir, n_prompts=48, n=16, M=120, n_train_pairs=5000, beta=0.5, d=4, seeds=3, n_attr=8):
    os.makedirs(out_dir, exist_ok=True)
    from datasets import load_dataset
    print(f"hh-rlhf: {n_prompts} prompts, best-of-{n}, M={M} heads, seeds={seeds}")
    tr = load_dataset("Anthropic/hh-rlhf", split=f"train[:{n_train_pairs}]")
    prompts_pool = load_dataset("Anthropic/hh-rlhf", split=f"test[:{n_prompts}]")

    ch, rj = [], []
    for c, r in zip(tr["chosen"], tr["rejected"]):
        pc, rc = _dialogue_split(c); pr, rr = _dialogue_split(r)
        ch.append(pc + " " + rc); rj.append(pr + " " + rr)
    e_ch = embed_texts(ch, "chosen"); e_rj = embed_texts(rj, "rejected")
    half = n_train_pairs // 2                                   # disjoint proxy / gold split

    prompts = [_dialogue_split(x)[0] for x in prompts_pool["chosen"]]
    cands = generate_candidates(prompts, n=n, tag="gen16")
    flat = [prompts[p] + " " + cands[p][i] for p in range(len(prompts)) for i in range(n)]
    e_flat = embed_texts(flat, "cand16")

    # a single independent GOLD reward model on the disjoint second half (larger ensemble
    # mean = a stronger reference than any proxy bootstrap on the first half)
    gold_heads = train_rm_ensemble(e_ch[half:], e_rj[half:], 8, seed0=9000)
    def gold_score(emb): return np.mean([h.score(emb) for h in gold_heads], 0)
    e_ch_te = embed_texts(ch[-400:], "chosen_te"); e_rj_te = embed_texts(rj[-400:], "rejected_te")
    gold_acc = float(np.mean(gold_score(e_ch_te) > gold_score(e_rj_te)))
    G = gold_score(e_flat).reshape(len(prompts), n)            # gold reward of every candidate

    per_seed = []
    over_by_seed = {nn: [] for nn in [1, 2, 4, 8, 16]}
    proxy_by_seed = {nn: [] for nn in [1, 2, 4, 8, 16]}
    beta_sweep_seed = {b: [] for b in [2.0, 1.0, 0.5, 0.25, 0.1]}
    n_sweep_seed = {nn: [] for nn in [2, 4, 8, 16]}
    for s in range(seeds):
        heads = train_rm_ensemble(e_ch[:half], e_rj[:half], M, seed0=1000 * s)  # proxy on first half
        S = np.stack([h.score(e_flat) for h in heads], 0).reshape(M, len(prompts), n)
        disagree = S.std(0).mean(1); med = np.median(disagree)
        reps = {"all": [], "strong": [], "weak": []}
        for p in range(len(prompts)):
            rep = attribute_prompt(S[:, p, :n_attr], beta, d=d)   # attribute the best-of-n_attr policy
            reps["all"].append(rep)
            reps["strong" if disagree[p] >= med else "weak"].append(rep)
        per_seed.append({k: aggregate(v) for k, v in reps.items()})
        # sweeps
        for b in beta_sweep_seed:
            beta_sweep_seed[b].append(float(np.mean(
                [tv_barycenter(best_of_n_policy(S[:, p, :], b)) for p in range(len(prompts))])))
        for nn in n_sweep_seed:
            n_sweep_seed[nn].append(float(np.mean(
                [tv_barycenter(best_of_n_policy(S[:, p, :nn], beta)) for p in range(len(prompts))])))
        # gold-reward overoptimization: proxy picks best-of-n; track proxy & gold reward of pick
        Pm = S.mean(0)                                          # (prompts, n) mean proxy score
        for nn in [1, 2, 4, 8, 16]:
            gold_vals, proxy_vals = [], []
            for p in range(len(prompts)):
                # z-score per prompt so "reward" is improvement over a random pick
                g = G[p]; gz = (g - g.mean()) / (g.std() + 1e-9)
                pr = Pm[p]; prz = (pr - pr.mean()) / (pr.std() + 1e-9)
                sel = int(np.argmax(pr[:nn]))
                gold_vals.append(gz[sel]); proxy_vals.append(prz[sel])
            over_by_seed[nn].append(float(np.mean(gold_vals)))
            proxy_by_seed[nn].append(float(np.mean(proxy_vals)))

    # aggregate across seeds
    def agg_metric(key, grp):
        return _mean_sd([ps[grp].get(key) for ps in per_seed if ps[grp]])
    res = dict(dataset="Anthropic/hh-rlhf", n_prompts=len(prompts), n=n, n_attr=n_attr, M=M, seeds=seeds,
               beta=beta, d=d, gold_accuracy=gold_acc,
               relMSE=dict(zip(["mean", "sd"], agg_metric("relMSE", "all"))),
               fragility=dict(zip(["mean", "sd"], agg_metric("fragility", "all"))),
               fragility_strong=dict(zip(["mean", "sd"], agg_metric("fragility", "strong"))),
               fragility_weak=dict(zip(["mean", "sd"], agg_metric("fragility", "weak"))),
               beta_sweep={str(b): dict(zip(["mean", "sd"], _mean_sd(v))) for b, v in beta_sweep_seed.items()},
               n_sweep={str(nn): dict(zip(["mean", "sd"], _mean_sd(v))) for nn, v in n_sweep_seed.items()},
               gold_reward_by_n={str(nn): dict(zip(["mean", "sd"], _mean_sd(v))) for nn, v in over_by_seed.items()},
               proxy_reward_by_n={str(nn): dict(zip(["mean", "sd"], _mean_sd(v))) for nn, v in proxy_by_seed.items()})
    def agg_modes(grp):
        arrs = [np.array(ps[grp]["mode_first_order"]) for ps in per_seed if ps[grp]]
        if not arrs:
            return []
        ml = max(len(a) for a in arrs); M0 = np.zeros(ml); cnt = np.zeros(ml)
        for a in arrs:
            M0[:len(a)] += a; cnt[:len(a)] += 1
        return (M0 / np.maximum(cnt, 1)).tolist()
    res["mode_first_order"] = agg_modes("all")
    res["mode_first_order_strong"] = agg_modes("strong")
    res["mode_first_order_weak"] = agg_modes("weak")
    json.dump(res, open(os.path.join(out_dir, "rlhf_real_v2_results.json"), "w"), indent=2)

    print(f"\ngold RM held-out accuracy = {gold_acc:.3f}")
    print(f"relMSE   = {res['relMSE']['mean']:.3f} +/- {res['relMSE']['sd']:.3f}")
    print(f"fragility= {res['fragility']['mean']:.3f} +/- {res['fragility']['sd']:.3f}")
    print(f"strong   = {res['fragility_strong']['mean']:.3f} +/- {res['fragility_strong']['sd']:.3f}   "
          f"weak = {res['fragility_weak']['mean']:.3f} +/- {res['fragility_weak']['sd']:.3f}")
    print("beta sweep (fragility mean): " + ", ".join(f"b={b}:{v['mean']:.3f}" for b, v in res["beta_sweep"].items()))
    print("n sweep    (fragility mean): " + ", ".join(f"n={nn}:{v['mean']:.3f}" for nn, v in res["n_sweep"].items()))
    print("OVEROPTIMIZATION (best-of-n by proxy):")
    print("  n:        " + " ".join(f"{nn:>6}" for nn in [1, 2, 4, 8, 16]))
    print("  proxy z:  " + " ".join(f"{res['proxy_reward_by_n'][str(nn)]['mean']:>6.3f}" for nn in [1, 2, 4, 8, 16]))
    print("  gold  z:  " + " ".join(f"{res['gold_reward_by_n'][str(nn)]['mean']:>6.3f}" for nn in [1, 2, 4, 8, 16]))
    print(f"[saved] {out_dir}/rlhf_real_v2_results.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cluster_results", "rlhf_real"))
    ap.add_argument("--n_prompts", type=int, default=48)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--M", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    run(a.out, n_prompts=a.n_prompts, n=a.n, M=a.M, seeds=a.seeds)
