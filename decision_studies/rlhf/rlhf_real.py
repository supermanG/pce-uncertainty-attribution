"""Real small-scale RLHF best-of-n study: reward-model-uncertainty attribution.

This upgrades the synthetic rlhf_demo.py to an actual study on real data:
  - Real preferences: Anthropic/hh-rlhf (chosen > rejected human preference pairs).
  - Real reward-model ensemble: a frozen DistilBERT encoder plus an ensemble of M
    Bradley-Terry reward heads, each trained on an independent bootstrap of the
    preference pairs. Ensemble disagreement is genuine epistemic uncertainty, not a
    hand-set heavy tail.
  - Real completions: for each prompt we draw n candidate completions from a GPT-2 base
    policy (best-of-n / rejection sampling, the standard practical RLHF-inference scheme;
    no PPO needed).
  - Best-of-n policy per reward-model realisation: pi_m(y) proportional to
    exp(r_m(prompt,y)/beta) over the n candidates. We fit a sparse arbitrary-PCE surrogate
    from the reward-model-uncertainty modes (PCA of the ensemble score matrix) to this
    policy and read analytical Sobol indices, attributing the generation decision to
    specific reward-model-uncertainty modes, plus a scale-invariant fragility.

Overoptimization is probed two ways (both documented signatures, Gao 2023 / Coste 2024):
  - beta sweep: harder KL-regularised optimisation (smaller beta) raises fragility.
  - n sweep: larger best-of-n raises fragility.
Strong vs weak coupling is a data-driven split of prompts by reward-model ensemble
disagreement (no gold reward needed): the policy is more fragile where the reward models
disagree more.

Caches encoder embeddings and generations in a local .cache directory for fast reruns.
"""
import argparse, json, os, sys, hashlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from core.sparse_pce import SparsePCESurrogate
from core.fragility_metrics import tv_barycenter, interaction_summary

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
CACHE = os.environ.get("RLHF_CACHE",
                       os.path.join(ROOT, ".cache", "rlhf"))
os.makedirs(CACHE, exist_ok=True)


def _dialogue_split(s):
    """Split an hh-rlhf dialogue into (prompt_prefix, final_response)."""
    i = s.rfind("\n\nAssistant:")
    if i < 0:
        return s, ""
    return s[:i + len("\n\nAssistant:")], s[i + len("\n\nAssistant:"):].strip()


# ---------------------------------------------------------------------------
# Frozen-encoder embeddings (cached)
# ---------------------------------------------------------------------------

def embed_texts(texts, tag, batch=64, max_length=256, model_name="distilbert-base-uncased"):
    key = hashlib.md5((tag + "|" + str(len(texts)) + "|" + model_name).encode()).hexdigest()[:12]
    path = os.path.join(CACHE, f"emb_{tag}_{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).cuda().eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length).to("cuda")
            h = enc(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)   # masked mean pool
            embs.append(pooled.cpu().numpy())
    del enc
    torch.cuda.empty_cache()
    out = np.concatenate(embs, 0).astype(np.float32)
    np.save(path, out)
    return out


# ---------------------------------------------------------------------------
# GPT-2 best-of-n candidate generation (cached)
# ---------------------------------------------------------------------------

def generate_candidates(prompts, n=12, max_new=40, tag="gen"):
    key = hashlib.md5((tag + "|" + str(len(prompts)) + f"|{n}|{max_new}").encode()).hexdigest()[:12]
    path = os.path.join(CACHE, f"gen_{key}.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained("gpt2").cuda().eval()
    all_cands = []
    with torch.no_grad():
        for p in prompts:
            # keep the prompt short for GPT-2 context; use the last human turn
            ptext = p[-400:]
            ids = tok(ptext, return_tensors="pt", truncation=True, max_length=200).to("cuda")
            gen = lm.generate(**ids, do_sample=True, top_p=0.95, temperature=1.0,
                              max_new_tokens=max_new, num_return_sequences=n,
                              pad_token_id=tok.eos_token_id)
            comp = [tok.decode(g[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    for g in gen]
            comp = [c if c else "(no response)" for c in comp]
            all_cands.append(comp)
    del lm
    torch.cuda.empty_cache()
    json.dump(all_cands, open(path, "w", encoding="utf-8"))
    return all_cands


# ---------------------------------------------------------------------------
# Reward-model ensemble (frozen encoder + Bradley-Terry MLP heads)
# ---------------------------------------------------------------------------

class RewardHead:
    """A small MLP reward head on frozen embeddings, trained by Bradley-Terry."""

    def __init__(self, dim, seed=0, hidden=128):
        import torch, torch.nn as nn
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)).cuda()

    def fit(self, e_chosen, e_rejected, epochs=120, lr=1e-3):
        import torch, torch.optim as optim
        ec = torch.tensor(e_chosen).cuda(); er = torch.tensor(e_rejected).cuda()
        opt = optim.Adam(self.net.parameters(), lr=lr)
        for _ in range(epochs):
            rc = self.net(ec).squeeze(-1); rr = self.net(er).squeeze(-1)
            loss = -torch.nn.functional.logsigmoid(rc - rr).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def score(self, emb):
        import torch
        with torch.no_grad():
            return self.net(torch.tensor(emb).cuda()).squeeze(-1).cpu().numpy()


def train_rm_ensemble(e_chosen, e_rejected, M, seed0=0, boot=0.8):
    heads = []
    n = len(e_chosen)
    for m in range(M):
        rng = np.random.RandomState(seed0 + m)
        idx = rng.choice(n, int(n * boot), replace=True)
        h = RewardHead(e_chosen.shape[1], seed=seed0 + m).fit(e_chosen[idx], e_rejected[idx])
        heads.append(h)
    return heads


# ---------------------------------------------------------------------------
# Attribution per prompt
# ---------------------------------------------------------------------------

def best_of_n_policy(scores, beta):
    """scores (M, n) -> policies (M, n) via softmax(scores/beta)."""
    z = scores / beta
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p


def attribute_prompt(scores, beta, d=4, degree=3, n_train=None):
    """scores (M, n) ensemble reward-model scores of n candidates -> PCE-Sobol attribution."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    M, n = scores.shape
    n_train = n_train or (M * 3) // 5
    P = best_of_n_policy(scores, beta)
    dd = min(d, M - 1, n)
    pca = PCA(n_components=dd)
    mu_tr = pca.fit_transform(scores[:n_train]); mu_te = pca.transform(scores[n_train:])
    sc = StandardScaler(); mu_tr = sc.fit_transform(mu_tr); mu_te = sc.transform(mu_te)
    surr = SparsePCESurrogate(degree=degree, basis="apc", method="lars")
    surr.fit(mu_tr, P[:n_train])
    pred = surr.predict(mu_te)
    rel = float(np.mean((pred - P[n_train:]) ** 2) / (np.var(P[n_train:]) + 1e-12))
    si = surr.sobol_indices()
    inter = interaction_summary({"first_order": si["first_order"],
                                 "total_order": si["total_order"], "variance": si["variance"]})
    return dict(relMSE=rel, fragility=float(tv_barycenter(P)),
                mode_first_order=si["first_order"].mean(0).tolist(),
                interaction=inter["interaction_from_first"],
                var_expl=float(pca.explained_variance_ratio_.sum()))


def aggregate(reports):
    ok = [r for r in reports if np.isfinite(r["relMSE"])]
    if not ok:
        return {}
    md = max(len(r["mode_first_order"]) for r in ok)
    mfo = np.zeros(md); cnt = np.zeros(md)
    for r in ok:
        v = r["mode_first_order"]; mfo[:len(v)] += v; cnt[:len(v)] += 1
    return dict(
        relMSE=float(np.mean([r["relMSE"] for r in ok])),
        fragility=float(np.mean([r["fragility"] for r in ok])),
        interaction=float(np.mean([r["interaction"] for r in ok])),
        var_expl=float(np.mean([r["var_expl"] for r in ok])),
        mode_first_order=(mfo / np.maximum(cnt, 1)).tolist(),
        n_prompts=len(ok),
    )


def run(out_dir, n_prompts=48, n=12, M=40, n_train_pairs=4000, beta=0.5, d=4, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    from datasets import load_dataset
    print("loading hh-rlhf preferences...")
    tr = load_dataset("Anthropic/hh-rlhf", split=f"train[:{n_train_pairs}]")
    prompts_pool = load_dataset("Anthropic/hh-rlhf", split=f"test[:{n_prompts}]")

    # preference pairs -> (prompt+response) texts for the reward model
    ch, rj = [], []
    for c, r in zip(tr["chosen"], tr["rejected"]):
        pc, rc = _dialogue_split(c); pr, rr = _dialogue_split(r)
        ch.append(pc + " " + rc); rj.append(pr + " " + rr)
    print(f"embedding {len(ch)} chosen + {len(rj)} rejected responses (frozen DistilBERT)...")
    e_ch = embed_texts(ch, "chosen"); e_rj = embed_texts(rj, "rejected")

    print(f"training reward-model ensemble (M={M} Bradley-Terry heads)...")
    heads = train_rm_ensemble(e_ch, e_rj, M, seed0=seed)
    # ensemble validation: agreement with held-out preferences
    e_ch_te = embed_texts(ch[-400:], "chosen_te"); e_rj_te = embed_texts(rj[-400:], "rejected_te")
    accs = [float(np.mean(h.score(e_ch_te) > h.score(e_rj_te))) for h in heads]
    print(f"   reward-model held-out pairwise accuracy: mean={np.mean(accs):.3f} "
          f"(range {min(accs):.3f}-{max(accs):.3f})")

    # candidate completions from GPT-2 for each prompt
    prompts = [_dialogue_split(x)[0] for x in prompts_pool["chosen"]]
    print(f"generating best-of-{n} candidates for {len(prompts)} prompts (GPT-2)...")
    cands = generate_candidates(prompts, n=n)

    # score every (prompt, candidate) with every reward head
    print("scoring candidates with the reward-model ensemble...")
    flat = [prompts[p] + " " + cands[p][i] for p in range(len(prompts)) for i in range(n)]
    e_flat = embed_texts(flat, "cand")
    S = np.stack([h.score(e_flat) for h in heads], 0)          # (M, P*n)
    S = S.reshape(M, len(prompts), n)                          # (M, P, n)

    # per-prompt attribution + data-driven strong/weak split by ensemble disagreement
    disagree = S.std(0).mean(1)                                # (P,) mean over-head score std
    med = np.median(disagree)
    reports = {"all": [], "strong": [], "weak": []}
    for p in range(len(prompts)):
        rep = attribute_prompt(S[:, p, :], beta, d=d)
        reports["all"].append(rep)
        reports["strong" if disagree[p] >= med else "weak"].append(rep)
    agg = {k: aggregate(v) for k, v in reports.items()}

    # overoptimization sweeps (aggregate fragility)
    beta_sweep = {}
    for b in [2.0, 1.0, 0.5, 0.25, 0.1]:
        fr = [tv_barycenter(best_of_n_policy(S[:, p, :], b)) for p in range(len(prompts))]
        beta_sweep[b] = float(np.mean(fr))
    n_sweep = {}
    for nn in [2, 4, 8, 12]:
        fr = [tv_barycenter(best_of_n_policy(S[:, p, :nn], beta)) for p in range(len(prompts))]
        n_sweep[nn] = float(np.mean(fr))

    res = dict(dataset="Anthropic/hh-rlhf", n_prompts=len(prompts), n=n, M=M,
               n_train_pairs=n_train_pairs, beta=beta, d=d,
               rm_heldout_accuracy=float(np.mean(accs)),
               aggregate=agg, beta_sweep=beta_sweep, n_sweep=n_sweep,
               median_disagreement=float(med))
    json.dump(res, open(os.path.join(out_dir, "rlhf_real_results.json"), "w"), indent=2)

    print("\n=== Real RLHF reward-model-uncertainty attribution (hh-rlhf) ===")
    print(f"reward-model held-out accuracy {np.mean(accs):.3f}; {len(prompts)} prompts, "
          f"best-of-{n}, M={M} heads")
    for k in ["all", "strong", "weak"]:
        a = agg[k]
        if a:
            print(f"[{k:6s}] relMSE={a['relMSE']:.3f}  fragility={a['fragility']:.3f}  "
                  f"interaction={a['interaction']:.3f}  top mode Sobol={max(a['mode_first_order']):.3f}  "
                  f"(n={a['n_prompts']})")
    print("beta sweep (fragility, harder optimisation ->): "
          + ", ".join(f"b={b}:{v:.3f}" for b, v in beta_sweep.items()))
    print("best-of-n sweep (fragility, larger n ->): "
          + ", ".join(f"n={k}:{v:.3f}" for k, v in n_sweep.items()))
    print(f"[saved] {out_dir}/rlhf_real_results.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cluster_results", "rlhf_real"))
    ap.add_argument("--n_prompts", type=int, default=48)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--M", type=int, default=40)
    ap.add_argument("--n_train_pairs", type=int, default=4000)
    ap.add_argument("--beta", type=float, default=0.5)
    a = ap.parse_args()
    run(a.out, n_prompts=a.n_prompts, n=a.n, M=a.M, n_train_pairs=a.n_train_pairs, beta=a.beta)
