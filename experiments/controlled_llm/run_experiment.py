"""Controlled LLM: GFlowNet finetuning with uncertain Process Reward Model.
Side experiment. See full version in conversation for complete documentation.
Uses simplified RNN-based reasoning GFlowNet as controlled proxy for full LLM.
"""
import os, sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    import torch, torch.nn as nn; HAS_TORCH = True
except ImportError: HAS_TORCH = False
from core.pce_surrogate import TrajectoryPCESurrogate
from core.distributional_analysis import (
    analyse_trajectory_bimodality, plot_bimodality_panel, plot_surrogate_comparison,
)
from sklearn.decomposition import PCA

class SimplePRM:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(768,128).astype(np.float32)*0.1
        self.b = rng.randn(128).astype(np.float32)*0.01
        self.v = rng.randn(128).astype(np.float32)*0.1
    def score(self, emb):
        return float(1.0/(1.0+np.exp(-(np.tanh(emb@self.W+self.b)@self.v))))

class ReasoningGFlowNet(nn.Module):
    def __init__(self, n_strat=10, max_steps=5, edim=64, hdim=128):
        super().__init__()
        self.n_strat = n_strat; self.max_steps = max_steps
        self.embed = nn.Embedding(n_strat, edim)
        self.rnn = nn.GRU(edim, hdim, batch_first=True)
        self.head = nn.Linear(hdim, n_strat+1)
        self.log_Z = nn.Parameter(torch.tensor(5.0))
        self.n_actions = n_strat+1

    def forward_policy(self, hist, step):
        if not hist:
            h0 = torch.zeros(1,1,self.rnn.hidden_size)
            _, hn = self.rnn(torch.zeros(1,1,self.embed.embedding_dim), h0)
        else:
            _, hn = self.rnn(self.embed(torch.tensor(hist).unsqueeze(0)))
        return torch.log_softmax(self.head(hn.squeeze(0).squeeze(0)), dim=-1)

    def get_policy(self, hist, step):
        with torch.no_grad(): return torch.exp(self.forward_policy(hist, step)).numpy()


def run_controlled_llm_experiment(n_train=15, n_test=25, n_strat=10, max_steps=5,
    pce_degree=5, pca_dim=2, gfn_episodes=1500, device="cpu", output_dir="results/llm"):
    os.makedirs(output_dir, exist_ok=True)
    print("="*70+"\nCONTROLLED LLM EXPERIMENT\n"+"="*70)
    rng = np.random.RandomState(42)
    prob_embs = rng.randn(30, 768).astype(np.float32)*0.5
    strat_embs = rng.randn(n_strat, 768).astype(np.float32)*0.3
    total = n_train + n_test
    prms = [SimplePRM(seed=100+i) for i in range(total)]

    # PCA on PRM outputs
    ref = [(p,s) for p in range(15) for s in range(n_strat)]
    pout = np.zeros((total, len(ref)))
    for i, prm in enumerate(prms):
        for j, (p,s) in enumerate(ref):
            pout[i,j] = prm.score(np.concatenate([prob_embs[p][:384], strat_embs[s][:384]]))
    pca = PCA(n_components=pca_dim)
    mu_tr = pca.fit_transform(pout[:n_train])
    mu_te = pca.transform(pout[n_train:])
    ref_traj = list(range(min(max_steps, n_strat)))

    print(f"  PCA var: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"  Training {total} GFlowNets...")

    def train_one(prm):
        gfn = ReasoningGFlowNet(n_strat=n_strat, max_steps=max_steps)
        opt = torch.optim.Adam(gfn.parameters(), lr=1e-3)
        for ep in range(gfn_episodes):
            loss = 0.0
            for _ in range(8):
                hist = []; lpf = 0.0; pi = rng.randint(len(prob_embs))
                for step in range(max_steps):
                    logp = gfn.forward_policy(hist, step)
                    dist = torch.distributions.Categorical(logits=logp)
                    a = dist.sample(); lpf += dist.log_prob(a)
                    if a.item() >= n_strat: break
                    hist.append(a.item())
                if not hist: hist = [0]
                r = np.prod([prm.score(np.concatenate([prob_embs[pi][:384],strat_embs[s][:384]]))
                             for s in hist])
                loss += (gfn.log_Z + lpf - np.log(r+1e-10)/2.0)**2
            loss /= 8; opt.zero_grad(); loss.backward(); opt.step()
        gfn.eval()
        pols = {}; h = []
        for s in range(len(ref_traj)): pols[s] = gfn.get_policy(h, s); h.append(ref_traj[s])
        return pols

    ns = len(ref_traj)
    tr_pol = {s:[] for s in range(ns)}; te_pol = {s:[] for s in range(ns)}
    for i in range(n_train):
        p = train_one(prms[i])
        for s in range(ns): tr_pol[s].append(p[s])
        if (i+1)%5==0: print(f"  Train: {i+1}/{n_train}")
    for i in range(n_test):
        p = train_one(prms[n_train+i])
        for s in range(ns): te_pol[s].append(p[s])
        if (i+1)%10==0: print(f"  Test: {i+1}/{n_test}")
    for s in range(ns): tr_pol[s]=np.array(tr_pol[s]); te_pol[s]=np.array(te_pol[s])

    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(ns): tsurr.fit_step(s, mu_tr, tr_pol[s])
    sobol = tsurr.sobol_all_steps()
    for s in range(ns):
        v = sobol[s]["variance"]; mi = np.argmax(v)
        t = sobol[s]["total_order"][mi] if v.max()>1e-10 else np.zeros(pca_dim)
        print(f"  Step {s}: S_PC1={t[0]:.3f}, S_PC2={t[1]:.3f}")

    # Accessible Sobol narrative
    print()
    print(tsurr.summarise_sobol(
        step_labels={s: f"strategy {s}" for s in range(ns)},
        dim_labels=["PC1 (PRM mean level)", "PC2 (PRM problem spread)"],
    ))

    # -----------------------------------------------------------------
    # Distributional analysis: the surrogate captures more than Sobol
    # -----------------------------------------------------------------
    n_mc = 5000
    surr_samples = tsurr.sample_trajectory_policies(n_mc)
    stop_action = n_strat  # last action = stop/terminate
    step_labels = {s: f"strategy {s}" for s in range(ns)}
    bimodality, narrative = analyse_trajectory_bimodality(
        surr_samples, action_idx=stop_action, step_labels=step_labels,
    )
    print(narrative)

    # Publication-quality figures
    fig = plot_bimodality_panel(
        surr_samples, action_idx=stop_action,
        step_labels=step_labels, action_name="stop",
        title="Controlled LLM: stop-action distributional structure",
        empirical_samples=te_pol,
        output_path=os.path.join(output_dir, "bimodality_panel.pdf"),
    )
    plot_surrogate_comparison(
        surr_samples, te_pol, action_idx=stop_action,
        step_labels=step_labels, action_name="stop",
        title="Controlled LLM: PCE surrogate vs empirical ensemble",
        output_path=os.path.join(output_dir, "surrogate_comparison.pdf"),
    )

    res = {
        "sobol": {s: {k: v.tolist() for k, v in sv.items()}
                  for s, sv in sobol.items()},
        "bimodality": {s: bm for s, bm in bimodality.items()},
    }
    with open(os.path.join(output_dir,"results.json"),"w") as f: json.dump(res,f,indent=2)
    return res

if __name__ == "__main__":
    if not HAS_TORCH: sys.exit(1)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train",      type=int,   default=30)
    parser.add_argument("--n_test",       type=int,   default=50)
    parser.add_argument("--n_strat",      type=int,   default=10)
    parser.add_argument("--max_steps",    type=int,   default=5)
    parser.add_argument("--pce_degree",   type=int,   default=5)
    parser.add_argument("--pca_dim",      type=int,   default=2)
    parser.add_argument("--gfn_episodes", type=int,   default=1500)
    parser.add_argument("--output_dir",   default="results/controlled_llm")
    parser.add_argument("--device",       default="cpu")
    args = parser.parse_args()
    run_controlled_llm_experiment(
        n_train=args.n_train, n_test=args.n_test,
        n_strat=args.n_strat, max_steps=args.max_steps,
        pce_degree=args.pce_degree, pca_dim=args.pca_dim,
        gfn_episodes=args.gfn_episodes,
        device=args.device, output_dir=args.output_dir,
    )
