"""Is the leading reward-model uncertainty mode the uniform-offset direction?

Theorem 6 (softmax shift-invariance) makes a mode that is a uniform offset of every
candidate's reward irrelevant to the best-of-n decision, whatever its variance. The RLHF
study reports that the top-variance PCA mode m1 of the reward-model ensemble has a near-zero
Sobol index. That is an expected consequence of Theorem 6 only if m1 is indeed the offset
direction; this script measures that directly, without any surrogate:

  offset_share   fraction of the reward-model ensemble's score variance over the
                 candidate pool that is a common offset of all candidates' rewards
                 (the kernel of the softmax); computed by projecting each member's
                 centered score vector onto the normalized all-ones vector
  cos_pc1        |cosine| between the first PCA loading vector of the M x n score
                 matrix and the normalized all-ones vector
  var_frac_pc1   share of the total score variance carried by PC1

Everything is computed per prompt and per reward-model-ensemble seed with the same
configuration as rlhf_real_v2.py (48 prompts, 16 candidates, M=120 heads, 5000
preference pairs, three seeds), then averaged. Encoder embeddings and GPT-2 generations are
produced with a fixed torch seed (rlhf_real_v2.py does not seed the sampler, so the
candidate pool differs from the archived run; the statistics reported here are properties of
the reward-model ensemble on any pool).

Usage:  python decision_studies/rlhf/rlhf_shift_mode.py --out results/cluster_results/rlhf_real
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from decision_studies.rlhf.rlhf_real import (_dialogue_split, embed_texts, generate_candidates,
                                             train_rm_ensemble)


def shift_stats(S):
    """S: (M, n) ensemble scores of n candidates. Returns (offset_share, cos_pc1, var_frac_pc1)."""
    M, n = S.shape
    Xc = S - S.mean(0, keepdims=True)                       # center over members
    u = np.ones(n) / np.sqrt(n)                             # normalized all-ones direction
    total = float((Xc ** 2).sum())
    offset = float(((Xc @ u) ** 2).sum())                   # variance along the offset direction
    U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = sv ** 2
    return offset / max(total, 1e-12), float(abs(Vt[0] @ u)), float(lam[0] / max(lam.sum(), 1e-12))


def run(out_dir, n_prompts=48, n=16, M=120, n_train_pairs=5000, seeds=3):
    import torch
    torch.manual_seed(0)
    from datasets import load_dataset
    tr = load_dataset("Anthropic/hh-rlhf", split=f"train[:{n_train_pairs}]")
    pool = load_dataset("Anthropic/hh-rlhf", split=f"test[:{n_prompts}]")
    ch, rj = [], []
    for c, r in zip(tr["chosen"], tr["rejected"]):
        pc, rc = _dialogue_split(c); pr, rr = _dialogue_split(r)
        ch.append(pc + " " + rc); rj.append(pr + " " + rr)
    e_ch = embed_texts(ch, "chosen"); e_rj = embed_texts(rj, "rejected")
    half = n_train_pairs // 2
    prompts = [_dialogue_split(x)[0] for x in pool["chosen"]]
    cands = generate_candidates(prompts, n=n, tag="gen16")
    flat = [prompts[p] + " " + cands[p][i] for p in range(len(prompts)) for i in range(n)]
    e_flat = embed_texts(flat, "cand16")

    per_seed = []
    for s in range(seeds):
        heads = train_rm_ensemble(e_ch[:half], e_rj[:half], M, seed0=1000 * s)
        S = np.stack([h.score(e_flat) for h in heads], 0).reshape(M, len(prompts), n)
        rows = np.array([shift_stats(S[:, p, :]) for p in range(len(prompts))])
        per_seed.append(dict(offset_share=rows[:, 0].mean(), cos_pc1=rows[:, 1].mean(),
                             var_frac_pc1=rows[:, 2].mean(),
                             cos_pc1_min=rows[:, 1].min(), offset_share_min=rows[:, 0].min()))
        print(f"seed {s}: offset share {rows[:,0].mean():.3f} (min {rows[:,0].min():.3f}), "
              f"|cos(PC1, 1)| {rows[:,1].mean():.3f} (min {rows[:,1].min():.3f}), "
              f"PC1 var frac {rows[:,2].mean():.3f}")

    def ms(k):
        a = np.array([d[k] for d in per_seed]); return dict(mean=float(a.mean()), sd=float(a.std()))
    res = dict(dataset="Anthropic/hh-rlhf", n_prompts=len(prompts), n=n, M=M, seeds=seeds,
               offset_share=ms("offset_share"), cos_pc1=ms("cos_pc1"), var_frac_pc1=ms("var_frac_pc1"),
               cos_pc1_min_over_prompts=ms("cos_pc1_min"),
               offset_share_min_over_prompts=ms("offset_share_min"),
               note="offset_share = share of the ensemble score variance along the all-ones "
                    "direction (the softmax kernel); cos_pc1 = |cosine| of the PC1 loading with "
                    "that direction; means over prompts, then mean/sd over seeds.")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "rlhf_shift_mode.json")
    json.dump(res, open(path, "w"), indent=2)
    print(f"[saved] {path}")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cluster_results", "rlhf_real"))
    a = ap.parse_args()
    run(a.out)
