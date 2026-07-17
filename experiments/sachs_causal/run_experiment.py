"""Sachs Protein Signaling Causal Discovery via GFlowNet + PCE UQ.
11 proteins, ~7500 observations, ground-truth from interventional experiments.
See full version in conversation for complete documentation.
Requires: torch, numpy, scipy, scikit-learn
"""
import os, sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.distributions import Categorical; HAS_TORCH = True
except ImportError: HAS_TORCH = False
from sklearn.decomposition import PCA
from core.pce_surrogate import TrajectoryPCESurrogate

VARS = ["Raf","Mek","Plcg","PIP2","PIP3","Erk","Akt","PKA","PKC","P38","Jnk"]
GT_EDGES = [(0,1),(1,5),(7,5),(7,9),(7,10),(7,6),(7,0),(7,1),(8,0),(8,1),(8,9),(8,10),(8,7),(2,3),(2,4),(2,8),(4,3)]

def load_sachs_data(data_path=None, n_total=7466):
    """Load Sachs flow cytometry data or fall back to synthetic generation.

    Parameters
    ----------
    data_path : str or None
        Path to ``sachs_real.csv``.  If None or the file does not exist the
        synthetic generator is used instead.
    n_total : int
        Number of synthetic observations to generate when falling back.

    Returns
    -------
    data : np.ndarray, shape (n_obs, 11)
        Standardised (zero-mean, unit-variance per column) data matrix.
    is_real : bool
        True when the CSV was loaded, False when synthetic data were used.
    """
    if data_path is not None and Path(data_path).is_file():
        import pandas as pd
        df = pd.read_csv(data_path)
        # Reorder / select columns to match VARS, then standardise
        df = df[VARS]
        data = df.values.astype(np.float64)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        print(f"  Loaded real Sachs data: {data.shape[0]} observations from {data_path}")
        return data, True
    if data_path is not None:
        print(f"  Warning: data_path '{data_path}' not found -- falling back to synthetic data.")
    data = generate_sachs_data(n_obs=n_total)
    print(f"  Using synthetic Sachs data: {data.shape[0]} observations.")
    return data, False


def generate_sachs_data(n_obs=500, seed=42):
    rng = np.random.RandomState(seed); n = 11
    topo = {2:0,4:1,3:2,8:3,7:4,0:5,1:6,5:7,6:8,9:9,10:10}
    inv = {v:k for k,v in topo.items()}
    adj = np.zeros((n,n))
    for (i,j) in GT_EDGES: adj[i,j] = rng.uniform(0.3,1.0)*rng.choice([-1,1])
    data = np.zeros((n_obs, n))
    for t in range(n):
        v = inv[t]; parents = [i for (i,j) in GT_EDGES if j==v]
        data[:,v] = sum(adj[p,v]*data[:,p] for p in parents) + rng.randn(n_obs)*0.7
    return data

def bge_score(data, edges):
    n_obs, n_vars = data.shape; parents = {j:[] for j in range(n_vars)}
    for (i,j) in edges: parents[j].append(i)
    score = 0.0
    for j in range(n_vars):
        fam = [j]+parents[j]; X = data[:,fam]; S = X.T@X/n_obs; l = len(fam)
        score += -0.5*n_obs*np.log(np.linalg.det(S+np.eye(l)*0.01)+1e-10)
    return score

class DAGGFlowNet(nn.Module):
    def __init__(self, n_vars=11, hidden=128):
        super().__init__(); self.n_vars = n_vars
        self.n_edge = n_vars*(n_vars-1); self.n_actions = self.n_edge+1
        self.net = nn.Sequential(nn.Linear(n_vars*n_vars+1,hidden),nn.ReLU(),
                                 nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,self.n_actions))
        self.log_Z = nn.Parameter(torch.tensor(10.0))

    def _edge(self, idx):
        i = idx//(self.n_vars-1); j_off = idx%(self.n_vars-1)
        return (i, j_off if j_off < i else j_off+1)

    def _acyclic(self, adj, i, j):
        vis = set(); stack = [j]
        while stack:
            node = stack.pop()
            if node == i: return False
            if node in vis: continue
            vis.add(node)
            for k in range(self.n_vars):
                if adj[node,k]>0: stack.append(k)
        return True

    def forward_policy(self, adj, step):
        device = next(self.parameters()).device
        state = torch.cat([torch.tensor(adj.flatten(),dtype=torch.float32),torch.tensor([step/20.0])]).to(device)
        logits = self.net(state)
        mask = torch.zeros(self.n_actions).to(device)
        for idx in range(self.n_edge):
            i,j = self._edge(idx)
            if adj[i,j]==0 and self._acyclic(adj,i,j): mask[idx]=1.0
        mask[-1]=1.0
        return torch.log_softmax(logits + (mask-1)*1e9, dim=-1)

    def get_policy(self, adj, step):
        with torch.no_grad(): return torch.exp(self.forward_policy(adj, step)).cpu().numpy()

def train_dag_gfn(gfn, data, n_ep=3000, max_e=18, batch=16, lr=1e-3, temp=1.0, device="cpu"):
    gfn = gfn.to(device); opt = optim.Adam(gfn.parameters(), lr=lr)
    loss_history = []
    for ep in range(n_ep):
        loss_sum = 0.0
        for _ in range(batch):
            adj = np.zeros((gfn.n_vars,gfn.n_vars)); edges = []; lpf = 0.0
            for step in range(max_e):
                logp = gfn.forward_policy(adj, step); dist = Categorical(logits=logp)
                a = dist.sample(); lpf += dist.log_prob(a)
                if a.item() >= gfn.n_edge: break
                i,j = gfn._edge(a.item()); adj[i,j]=1.0; edges.append((i,j))
            loss_sum += (gfn.log_Z + lpf - bge_score(data, edges)/temp)**2
        loss = loss_sum/batch; opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gfn.parameters(), 1.0); opt.step()
        loss_history.append(float(loss.item()))
        if (ep+1)%1000==0: print(f"  Ep {ep+1}/{n_ep}, L:{loss.item():.2f}")
    gfn.eval(); gfn.loss_history = loss_history; return gfn

def greedy_rollout(gfn, max_edges=12):
    """Extract reference trajectory via greedy rollout (no ground-truth leak)."""
    adj = np.zeros((11, 11))
    edges = []
    for step in range(max_edges):
        pol = gfn.get_policy(adj, step)
        a = int(np.argmax(pol))
        if a >= gfn.n_edge:  # EOS
            break
        i, j = gfn._edge(a)
        adj[i, j] = 1.0
        edges.append((i, j))
    return edges


def run_sachs_experiment(n_train=30, n_test=50, obs=200, n_total=2000, pce_degree=5,
    pca_dim=2, gfn_episodes=3000, max_edges=12, device="cpu", output_dir="results/sachs",
    data_path=None):
    os.makedirs(output_dir, exist_ok=True)
    print("="*70+"\nSACHS CAUSAL DISCOVERY\n"+"="*70)
    full, is_real = load_sachs_data(data_path=data_path, n_total=n_total)
    n_total_actual = len(full)
    dsets = [full[np.random.RandomState(i+100).choice(n_total_actual,obs,replace=False)] for i in range(n_train+n_test)]
    covs = np.array([np.cov(d.T).flatten() for d in dsets])
    pca = PCA(n_components=pca_dim)
    mu_tr = pca.fit_transform(covs[:n_train])
    mu_te = pca.transform(covs[n_train:])

    # Train pilot GFlowNet on first dataset to get reference trajectory
    # (avoids ground-truth leak from using GT_EDGES directly)
    pilot_gfn = train_dag_gfn(DAGGFlowNet(), dsets[0], n_ep=gfn_episodes,
                               max_e=max_edges, device=device)
    ref_edges = greedy_rollout(pilot_gfn, max_edges=max_edges)
    if not ref_edges:
        print("  WARNING: pilot GFlowNet produced empty trajectory, using GT_EDGES as fallback")
        ref_edges = GT_EDGES[:max_edges]
    print(f"  PCA var: {pca.explained_variance_ratio_.sum():.3f}, ref edges: {len(ref_edges)}")
    gt_overlap = len(set(ref_edges) & set(GT_EDGES))
    print(f"  Reference trajectory: {len(ref_edges)} edges ({gt_overlap} overlap with GT)")

    n_steps = len(ref_edges)
    tr_pol = {s:[] for s in range(n_steps)}; te_pol = {s:[] for s in range(n_steps)}
    for i in range(n_train):
        gfn = train_dag_gfn(DAGGFlowNet(), dsets[i], n_ep=gfn_episodes, max_e=max_edges, device=device)
        adj = np.zeros((11,11))
        for s,(ei,ej) in enumerate(ref_edges):
            tr_pol[s].append(gfn.get_policy(adj, s)); adj[ei,ej]=1.0
        if (i+1)%10==0: print(f"  Train: {i+1}/{n_train}")
    for i in range(n_test):
        gfn = train_dag_gfn(DAGGFlowNet(), dsets[n_train+i], n_ep=gfn_episodes, max_e=max_edges, device=device)
        adj = np.zeros((11,11))
        for s,(ei,ej) in enumerate(ref_edges):
            te_pol[s].append(gfn.get_policy(adj, s)); adj[ei,ej]=1.0
        if (i+1)%10==0: print(f"  Test: {i+1}/{n_test}")
    for s in range(n_steps): tr_pol[s]=np.array(tr_pol[s]); te_pol[s]=np.array(te_pol[s])

    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(n_steps): tsurr.fit_step(s, mu_tr, tr_pol[s])
    sobol = tsurr.sobol_all_steps()
    for s in range(min(n_steps,8)):
        e = ref_edges[s]; name = f"{VARS[e[0]]}->{VARS[e[1]]}"
        v = sobol[s]["variance"]; mi = np.argmax(v)
        t = sobol[s]["total_order"][mi] if v.max()>1e-10 else np.zeros(pca_dim)
        print(f"  Step {s} ({name}): S_PC1={t[0]:.3f}, S_PC2={t[1]:.3f}")

    # Accessible narrative
    edge_labels = {s: f"{VARS[ref_edges[s][0]]}->{VARS[ref_edges[s][1]]}"
                   for s in range(n_steps)}
    print()
    print(tsurr.summarise_sobol(
        step_labels=edge_labels,
        dim_labels=[f"PC{i+1} (data covariance)" for i in range(pca_dim)],
    ))

    res = {"sobol":{s:{k:v.tolist() for k,v in sv.items()} for s,sv in sobol.items()},
           "variables": VARS, "edges": [(str(e)) for e in ref_edges]}
    with open(os.path.join(output_dir,"results.json"),"w") as f: json.dump(res,f,indent=2)
    return res

def analyze_from_members(
    output_dir: str = "results/sachs",
    n_train: int = 30,
    n_test: int = 50,
    pce_degree: int = 5,
    pca_dim: int = 2,
    data_path: str = None,
) -> dict:
    """Load saved member npz files and fit PCE + compute Sobol indices.

    Called after LSF array training by lsf/sachs_analyze.sh.

    Parameters
    ----------
    data_path : str or None
        Optional path to ``sachs_real.csv``.  When provided and the file
        exists, real Sachs flow cytometry data replace ``full_data.npy`` for
        the PCA covariance step.  Falls back to the saved ``full_data.npy``
        (synthetic) when absent.
    """
    import glob
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    ref_traj = [tuple(e) for e in meta["ref_traj"]]
    n_steps = len(ref_traj)

    # Load all member files
    member_files = sorted(glob.glob(os.path.join(output_dir, "members", "member_*.npz")))
    if len(member_files) < n_train + n_test:
        raise RuntimeError(f"Expected {n_train+n_test} members, found {len(member_files)}")

    all_policies = []   # list of (n_steps, n_actions) arrays
    for fp in member_files[:n_train + n_test]:
        d = np.load(fp)
        all_policies.append(d["policies"])   # (n_steps, n_actions)

    # PCA on subsample covariance matrices to get mu (reward parameterisation).
    # Prefer real data when data_path resolves; otherwise use saved full_data.npy.
    real_data, is_real = load_sachs_data(data_path=data_path, n_total=meta.get("n_total", 2000))
    if is_real:
        full_data = real_data
    else:
        full_data = np.load(os.path.join(output_dir, "full_data.npy"))
    covs = []
    for i in range(n_train + n_test):
        rng = np.random.RandomState(i + 100)
        idx = rng.choice(len(full_data), meta["n_obs"], replace=False)
        covs.append(np.cov(full_data[idx].T).flatten())
    covs = np.array(covs)
    pca = PCA(n_components=pca_dim)
    mu_tr = pca.fit_transform(covs[:n_train])
    mu_te = pca.transform(covs[n_train:])
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Per-step policy arrays
    tr_pol = {s: np.array([all_policies[i][s] for i in range(n_train)]) for s in range(n_steps)}
    te_pol = {s: np.array([all_policies[n_train + i][s] for i in range(n_test)]) for s in range(n_steps)}

    # Fit PCE surrogate
    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(n_steps):
        tsurr.fit_step(s, mu_tr, tr_pol[s])
    sobol = tsurr.sobol_all_steps()

    print("\n" + "="*60)
    print("SACHS SOBOL RESULTS")
    print("="*60)
    for s in range(n_steps):
        e = ref_traj[s]
        name = f"{VARS[e[0]]}->{VARS[e[1]]}"
        v = sobol[s]["variance"]
        mi = np.argmax(v) if v.max() > 1e-10 else 0
        t = sobol[s]["total_order"][mi] if v.max() > 1e-10 else np.zeros(pca_dim)
        print(f"  Step {s:2d} ({name:12s}): var={v.max():.4f}  S_PC1={t[0]:.3f}  S_PC2={t[1]:.3f}")

    results = {
        "sobol": {s: {k: v.tolist() for k, v in sv.items()} for s, sv in sobol.items()},
        "variables": VARS,
        "edges": [list(e) for e in ref_traj],
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "n_train": n_train,
        "n_test": n_test,
        "pce_degree": pce_degree,
        "data_source": "real" if is_real else "synthetic",
    }
    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sequential", "analyze"], default="sequential")
    parser.add_argument("--output_dir", default="results/sachs")
    parser.add_argument("--n_train", type=int, default=30)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--pce_degree", type=int, default=5)
    parser.add_argument("--pca_dim", type=int, default=2)
    parser.add_argument("--data_path", default=None,
                        help="Path to sachs_real.csv. If omitted or file absent, "
                             "synthetic data are used.")
    args = parser.parse_args()

    if args.mode == "analyze":
        analyze_from_members(
            output_dir=args.output_dir,
            n_train=args.n_train,
            n_test=args.n_test,
            pce_degree=args.pce_degree,
            pca_dim=args.pca_dim,
            data_path=args.data_path,
        )
    else:
        if not HAS_TORCH:
            sys.exit(1)
        run_sachs_experiment(
            n_train=args.n_train,
            n_test=args.n_test,
            gfn_episodes=3000,
            device="cuda" if torch.cuda.is_available() else "cpu",
            output_dir=args.output_dir,
            data_path=args.data_path,
        )
