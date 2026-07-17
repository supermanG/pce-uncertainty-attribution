#!/usr/bin/env python3
"""beta-VAE vs PCA embedding on Ramon's discrete grid-world, through OUR PCE surrogate.

Uses Ramon's trained ensemble (50 matched reward-grid / GFlowNet-policy pairs) from
github.com/rnartallo/uq4gfn/discrete_grid. Embeds each member's reward grid two ways
(nonlinear conv beta-VAE, reimplemented in plain torch; and linear PCA), extracts the
along-path decision (probability of the taken action per ensemble member) and fits our
scalar PCE per step to read analytical Sobol indices. Compares:
  - reconstruction R^2 (variance captured by the embedding)
  - latent independence (max|rho|) and Gaussianity (Shapiro-Wilk)  -> analytical-Sobol validity
  - PCE surrogate fidelity (leave-one-out relMSE of the decision quantity)
  - Sobol attribution per decision step (which latent drives the decision)
"""
import sys, os, json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (nested -> repo root)
sys.path.insert(0, ROOT)
from core.pce_surrogate import build_design_matrix, fit_pce, select_lambda_gcv

# Ramon's trained grid-world ensemble (50 matched reward-grid / GFlowNet pairs). Not vendored;
# fetch into this dir from github.com/rnartallo/uq4gfn/discrete_grid/training_ensemble:
#   for n in $(seq 1 50); do
#     curl -fsSLO https://raw.githubusercontent.com/rnartallo/uq4gfn/main/discrete_grid/training_ensemble/gfn${n}.pth
#     curl -fsSLO https://raw.githubusercontent.com/rnartallo/uq4gfn/main/discrete_grid/training_ensemble/grid${n}.csv
#   done
RAMON = os.environ.get("GRIDWORLD_ENSEMBLE_DIR", os.path.join(os.path.dirname(__file__), "beta_vae_gridworld_data"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
LATENT_DIM = 2
BETA = 4.0
PCE_DEGREE = 2
grid_size, high_reward, mid_reward, low_reward = 10, 200, 40, 0.1
VALUES = [0.1, 40.0, 200.0]
V2I = {v: i for i, v in enumerate(VALUES)}

# ------------------------------------------------------------------ Ramon's env / model / helpers (pyro-free subset)
def create_reward_grid(hc):
    grid = np.full((grid_size, grid_size), low_reward)
    for cy, cx in hc:
        if 0 <= cy < grid_size and 0 <= cx < grid_size:
            grid[cy, cx] = high_reward
        for ny, nx in [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)]:
            if 0 <= ny < grid_size and 0 <= nx < grid_size and grid[ny, nx] != high_reward:
                grid[ny, nx] = mid_reward
    return grid

def shift_reward_centers(p, hc=[(2, 2), (2, 7), (7, 2), (7, 7)]):
    shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    out = []
    for c in hc:
        out.append(tuple(np.array(c) + np.array(random.choice(shifts))) if np.random.random() < p else c)
    return out

def one_hot(grids):
    n = len(grids)
    oh = np.zeros((n, grid_size, grid_size, 3), dtype=np.float32)
    for i, g in enumerate(grids):
        for r in range(grid_size):
            for c in range(grid_size):
                oh[i, r, c, V2I[round(float(g[r, c]), 1)]] = 1.0
    return torch.tensor(oh).permute(0, 3, 1, 2)  # (n,3,10,10)

def grid_indices(grids):
    idx = np.zeros((len(grids), grid_size * grid_size), dtype=np.int64)
    for i, g in enumerate(grids):
        idx[i] = np.array([V2I[round(float(v), 1)] for v in g.flatten()])
    return torch.tensor(idx)

class ConvEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1); self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.fc1 = nn.Linear(32*3*3, 128); self.fc_mean = nn.Linear(128, d); self.fc_lv = nn.Linear(128, d)
    def forward(self, x):
        h = F.relu(self.conv1(x)); h = F.relu(self.conv2(h)); h = F.relu(self.fc1(h.reshape(-1, 32*3*3)))
        return self.fc_mean(h), self.fc_lv(h)

class ConvDecoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 128); self.fc2 = nn.Linear(128, 32*3*3)
        self.dc1 = nn.ConvTranspose2d(32, 16, 3, 2, 1); self.dc2 = nn.ConvTranspose2d(16, 3, 3, 2, 1, output_padding=1)
    def forward(self, z):
        h = F.relu(self.fc1(z)); h = F.relu(self.fc2(h)).view(-1, 32, 3, 3)
        h = F.relu(self.dc1(h)); logits = self.dc2(h).permute(0, 2, 3, 1)  # (B,10,10,3)
        return logits.reshape(-1, 100, 3)

class GFlowNet(nn.Module):
    def __init__(self, state_dim, n_act):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_act))
        self.logZ = nn.Parameter(torch.tensor(3.0))
    def forward(self, s): return self.network(s)
    def to_input_tensor(self, state, gs):
        oh = torch.zeros(gs*gs); oh[state[0]*gs + state[1]] = 1.0
        return torch.cat([oh, torch.tensor([state[0]/(gs-1), state[1]/(gs-1)], dtype=torch.float32)])

class GridEnvironment:
    def __init__(self, reward_grid):
        self.grid_size = reward_grid.shape[0]; self.reward_grid = torch.tensor(reward_grid, dtype=torch.float32)
        self.action_space_size = 5
    def get_initial_state(self): return (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
    def step(self, s, a):
        y, x = s
        return (y-1, x) if a == 0 else (y+1, x) if a == 1 else (y, x-1) if a == 2 else (y, x+1) if a == 3 else (y, x)
    def get_valid_actions(self, s):
        y, x = s; va = []
        if y > 0: va.append(0)
        if y < self.grid_size-1: va.append(1)
        if x > 0: va.append(2)
        if x < self.grid_size-1: va.append(3)
        if self.reward_grid[y, x] > low_reward: va.append(4)
        return va

def meta_sampler(ens, env, n_samples, max_len, temp=1.0):
    trajs, acts = [], []
    with torch.no_grad():
        for _ in range(n_samples):
            path, actions = [], []; state = env.get_initial_state(); visited = {state}; path.append(state)
            for _ in range(max_len):
                st = ens[0].to_input_tensor(state, env.grid_size); va = env.get_valid_actions(state)
                unv = [a for a in va if a == 4 or env.step(state, a) not in visited]
                if not unv: break
                mask = torch.full((env.action_space_size,), -torch.inf); mask[unv] = 0
                logits = torch.zeros(env.action_space_size); M = len(ens)
                for n in range(M): logits += (1/M)*ens[n](st) + mask
                a = torch.distributions.Categorical(logits=logits/temp).sample().item()
                if a == 4: break
                state = env.step(state, a); visited.add(state); path.append(state); actions.append(a)
            trajs.append(path); acts.append(actions)
    return trajs, acts

def ensemble_flows(ens, env, traj, actions):
    flows, visited, M = [], [], len(ens)
    for t in range(len(traj)-1):
        visited.append(tuple(traj[t])); st = ens[0].to_input_tensor(traj[t], env.grid_size)
        va = env.get_valid_actions(traj[t]); unv = [a for a in va if a == 4 or env.step(traj[t], a) not in visited]
        mask = torch.full((env.action_space_size,), -torch.inf); mask[unv] = 0
        fe = np.zeros(M)
        for m in range(M):
            probs = torch.softmax(ens[m](st) + mask, dim=-1).detach().numpy(); fe[m] = probs[actions[t]]
        flows.append(fe)
    return np.array(flows)  # (T-1, M)

# ------------------------------------------------------------------ beta-VAE (plain torch)
def train_beta_vae(train_oh, train_idx, d, beta, epochs=400, bs=256, lr=1e-3):
    enc, dec = ConvEncoder(d).to(DEV), ConvDecoder(d).to(DEV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    X, I = train_oh.to(DEV), train_idx.to(DEV); n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for b in range(0, n, bs):
            xb, ib = X[perm[b:b+bs]], I[perm[b:b+bs]]
            mean, lv = enc(xb); z = mean + torch.randn_like(mean)*torch.exp(0.5*lv)
            logits = dec(z)
            recon = F.cross_entropy(logits.reshape(-1, 3), ib.reshape(-1), reduction="sum")/xb.shape[0]
            kl = -0.5*torch.sum(1 + lv - mean.pow(2) - lv.exp())/xb.shape[0]
            loss = recon + beta*kl
            opt.zero_grad(); loss.backward(); opt.step()
    return enc.eval(), dec.eval()

def vae_reconstruct_R2(enc, dec, oh, raw):
    with torch.no_grad():
        mean, _ = enc(oh.to(DEV)); probs = torch.softmax(dec(mean), dim=-1).cpu().numpy()  # (n,100,3)
    recon = probs @ np.array(VALUES)  # expected reward per cell (n,100)
    ss_res = np.sum((raw - recon)**2); ss_tot = np.sum((raw - raw.mean())**2)
    return 1 - ss_res/ss_tot, mean.detach().cpu().numpy()

# ------------------------------------------------------------------ scalar PCE + Sobol
def scalar_pce_sobol(Z, y, degree):
    Phi, midx = build_design_matrix(Z, degree, "hermite")
    lam = select_lambda_gcv(Phi, y); c = fit_pce(Phi, y, lam)
    A = Phi.T @ Phi + lam*np.eye(Phi.shape[1]); Ainv = np.linalg.inv(A)
    h = np.clip(np.einsum("ij,jk,ik->i", Phi, Ainv, Phi), None, 1-1e-9)
    loo = (y - Phi @ c)/(1 - h)
    relmse = float(np.sum(loo**2)/(np.sum((y - y.mean())**2) + 1e-12))
    d = Z.shape[1]; D = float(np.sum(c[1:]**2)) + 1e-12
    fo, to = np.zeros(d), np.zeros(d)
    for i in range(d):
        fo_mask = np.array([midx[j, i] > 0 and all(midx[j, k] == 0 for k in range(d) if k != i) for j in range(len(midx))])
        fo[i] = np.sum(c[fo_mask]**2)/D; to[i] = np.sum(c[midx[:, i] > 0]**2)/D
    return relmse, fo, to

def latent_diagnostics(Z):
    d = Z.shape[1]; C = np.corrcoef(Z.T); off = C[~np.eye(d, dtype=bool)]
    shp = [float(stats.shapiro(Z[:, i]).pvalue) for i in range(d)]
    return float(np.max(np.abs(off))), shp

# ------------------------------------------------------------------ main
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    N = 50
    grids = [np.loadtxt(os.path.join(RAMON, f"grid{i}.csv"), delimiter=",") for i in range(1, N+1)]
    ens = []
    for i in range(1, N+1):
        m = GFlowNet(grid_size*grid_size + 2, 5)
        m.load_state_dict(torch.load(os.path.join(RAMON, f"gfn{i}.pth"), map_location="cpu")); m.eval(); ens.append(m)
    raw = np.array([g.flatten() for g in grids])  # (50,100)
    print(f"loaded {N} grids + policies; reward-grid variation: {np.mean(np.std(raw,axis=0)>1e-6)*100:.0f}% of cells vary")

    # --- decision path (fixed, sampled from ensemble mean policy) ---
    env = GridEnvironment(create_reward_grid([(2, 2), (2, 7), (7, 2), (7, 7)]))
    trajs, acts = meta_sampler(ens, env, n_samples=120, max_len=20, temp=1.0)
    # pick a moderate-length path that is most informative (largest across-ensemble variation in the decision)
    best = None
    for t, a in zip(trajs, acts):
        if not (5 <= len(a) <= 10):
            continue
        fl = ensemble_flows(ens, env, t, a)
        score = float(np.mean(fl.std(axis=1)))
        if best is None or score > best[0]:
            best = (score, t, a, fl)
    _, traj, actions, flows = best
    T = flows.shape[0]
    print(f"decision path length {len(traj)} states, {T} attributed steps; actions={actions}")

    # --- embeddings ---
    # PCA on raw reward grids
    pca = PCA(n_components=LATENT_DIM).fit(raw)
    Zp = pca.transform(raw); recon_p = pca.inverse_transform(Zp)
    r2_pca = 1 - np.sum((raw - recon_p)**2)/np.sum((raw - raw.mean())**2)
    # beta-VAE on many generated grids
    gen = [create_reward_grid(shift_reward_centers(0.4)) for _ in range(3000)]
    enc, dec = train_beta_vae(one_hot(gen), grid_indices(gen), LATENT_DIM, BETA)
    r2_vae, Zv = vae_reconstruct_R2(enc, dec, one_hot(grids), raw)

    def standardize(Z): return (Z - Z.mean(0))/(Z.std(0) + 1e-9)
    Zp_s, Zv_s = standardize(Zp), standardize(Zv)

    results = {"config": {"N": N, "latent_dim": LATENT_DIM, "beta": BETA, "pce_degree": PCE_DEGREE,
                          "path_len": len(traj), "n_steps": T, "actions": actions}}
    for name, Z, r2 in [("pca", Zp_s, r2_pca), ("beta_vae", Zv_s, r2_vae)]:
        maxrho, shp = latent_diagnostics(Z)
        per_step = []
        for t in range(T):
            y = flows[t]
            if y.std() < 1e-6:
                per_step.append({"step": t, "relmse": None, "first": None, "total": None, "flat": True}); continue
            relmse, fo, to = scalar_pce_sobol(Z, y, PCE_DEGREE)
            per_step.append({"step": t, "relmse": relmse, "first": fo.tolist(), "total": to.tolist(),
                             "dominant_dim": int(np.argmax(to)), "flow_std": float(y.std())})
        valid = [s for s in per_step if not s.get("flat")]
        results[name] = {
            "reconstruction_R2": float(r2), "max_abs_latent_corr": maxrho,
            "shapiro_p": shp, "gaussian_all": bool(all(p > 0.05 for p in shp)),
            "mean_relMSE": float(np.mean([s["relmse"] for s in valid])) if valid else None,
            "per_step": per_step,
        }
        print(f"\n[{name}]  reconR2={r2:.3f}  max|rho|={maxrho:.3f}  gaussian_all={results[name]['gaussian_all']}  "
              f"meanRelMSE={results[name]['mean_relMSE']:.3f}")
        for s in valid:
            print(f"   step{s['step']}: relMSE={s['relmse']:.3f}  dom_dim={s['dominant_dim']}  "
                  f"total={[round(x,2) for x in s['total']]}")

    out = os.path.join(ROOT, "results", "beta_vae_gridworld_results.json")
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print("\nwrote", out)

if __name__ == "__main__":
    main()
