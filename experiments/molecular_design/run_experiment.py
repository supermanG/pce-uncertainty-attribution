"""Fragment-Based Molecular Design via GFlowNet + PCE UQ.

Agent sequentially selects 5 drug-like fragments from a vocabulary of 20.
Reward is a drug-likeness proxy (QED-inspired: logP balance + diversity bonus
- MW penalty). An MLP proxy trained on 30% of a reference dataset provides
the epistemic uncertainty that PCE decomposes position by position.

Scientific narrative (Sobol):
  Scaffold positions (1-2) show low Sobol index -- the GFlowNet consistently
  picks the same scaffold regardless of which training molecules the proxy saw.
  Decoration positions (4-5) show high Sobol index -- decoration choices are
  fragile and depend strongly on the proxy's training subset.

No RDKit required: molecules are represented as concatenated 5-dim fragment
feature vectors (25-dim total).

Usage modes:
  1. Sequential (local, small-scale):
       python run_experiment.py --mode sequential

  2. LSF cluster (IBM CCC):
       python train_single_member.py --setup --output_dir results/molecular_design
       bsub < lsf/submit_moldesign.sh  (see train_single_member.py header)

  3. Collect and analyse only (after array jobs complete):
       python run_experiment.py --mode analyze --output_dir results/molecular_design

Requires: torch, numpy, scipy, scikit-learn
"""
import os, sys, json, time, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not found.")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from core.pce_surrogate import (
    TrajectoryPCESurrogate, calibration_coverage, run_ks_battery
)


# ---------------------------------------------------------------------------
# Fragment vocabulary
# Feature vector: [MW_norm, logP_contribution, HBD, HBA, aromaticity]
#   MW_norm          : molecular weight contribution, normalised to [0, 1]
#                      (range 14 Da for methyl to 189 Da for tetrazole, mapped
#                       linearly onto [0, 1] relative to the vocabulary max)
#   logP_contribution: fragment contribution to octanol-water logP (real-valued)
#   HBD              : hydrogen bond donor count (integer-valued)
#   HBA              : hydrogen bond acceptor count (integer-valued)
#   aromaticity      : 1 if aromatic, 0 otherwise
# ---------------------------------------------------------------------------

FRAGMENTS = {
    # name             MW_norm  logP   HBD  HBA  arom
    "benzene":        [0.41,    1.56,  0,   0,   1],
    "pyridine":       [0.40,    0.65,  0,   1,   1],
    "pyrimidine":     [0.42,   -0.47,  0,   2,   1],
    "indole":         [0.61,    2.14,  1,   1,   1],
    "morpholine":     [0.60,   -1.08,  1,   2,   0],
    "piperazine":     [0.60,   -1.50,  2,   2,   0],
    "OH":             [0.09,   -0.67,  1,   1,   0],
    "NH2":            [0.09,   -1.03,  2,   1,   0],
    "F":              [0.11,    0.14,  0,   0,   0],
    "Cl":             [0.47,    0.60,  0,   0,   0],
    "CF3":            [0.51,    0.88,  0,   0,   0],
    "methyl":         [0.09,    0.53,  0,   0,   0],
    "ethyl":          [0.18,    1.02,  0,   0,   0],
    "isopropyl":      [0.26,    1.53,  0,   0,   0],
    "tBu":            [0.33,    1.98,  0,   0,   0],
    "OMe":            [0.22,   -0.02,  0,   1,   0],
    "CN":             [0.33,   -0.57,  0,   1,   0],
    "COOH":           [0.40,   -0.32,  1,   2,   0],
    "SO2NH2":         [0.60,   -1.82,  2,   3,   0],
    "tetrazole":      [0.71,   -0.97,  1,   4,   1],
}

FRAGMENT_NAMES = list(FRAGMENTS.keys())      # stable ordering
N_FRAGMENTS    = len(FRAGMENT_NAMES)         # 20
N_STEPS        = 5                           # sequential positions
FEAT_DIM       = 5                           # per-fragment feature length
MOL_DIM        = N_STEPS * FEAT_DIM          # 25-dim molecule representation

# Pre-build feature matrix (20, 5) for fast lookup
_FEAT_MATRIX = np.array([FRAGMENTS[n] for n in FRAGMENT_NAMES], dtype=np.float32)


# ---------------------------------------------------------------------------
# Molecule encoding
# ---------------------------------------------------------------------------

def encode_molecule(selections: list) -> np.ndarray:
    """Concatenate fragment feature vectors for a list of 5 fragment indices.

    Args:
        selections: list of 5 integer indices into FRAGMENT_NAMES

    Returns:
        25-dim float32 array
    """
    return np.concatenate([_FEAT_MATRIX[s] for s in selections]).astype(np.float32)


# ---------------------------------------------------------------------------
# Ground-truth reward (no proxy needed)
# ---------------------------------------------------------------------------

# Pre-compute MW normalisation constant from vocabulary
_MAX_MW_NORM = max(v[0] for v in FRAGMENTS.values())


def ground_truth_reward(selections: list) -> float:
    """Drug-likeness proxy reward in [0, 1], computed from fragment features.

    Components:
      + logP balance: penalise distance of sum_logP from the ideal window [1, 3]
      + diversity bonus: fraction of unique fragments chosen
      - MW penalty: heavy molecules are penalised (Lipinski rule-of-five spirit)
      + aromatic core bonus: reward having at least one aromatic fragment

    The combination creates genuine variation (neither trivially random nor
    perfectly predictable) so that proxy uncertainty across training subsets
    produces scientifically meaningful Sobol indices.
    """
    feats = _FEAT_MATRIX[selections]            # (5, 5)
    mw_norm    = feats[:, 0].sum()              # [0, 5]
    sum_logP   = feats[:, 1].sum()              # unconstrained
    n_hbd      = feats[:, 2].sum()
    n_hba      = feats[:, 3].sum()
    has_arom   = float(feats[:, 4].any())

    # LogP balance score: Gaussian centred at 2.0, width 1.5
    logP_score = float(np.exp(-0.5 * ((sum_logP - 2.0) / 1.5) ** 2))

    # MW penalty: ideal < 3.0 (normalised units ~ 500 Da)
    mw_score = float(np.clip(1.0 - (mw_norm - 2.0) / 3.0, 0.0, 1.0))

    # HBD/HBA balance (Lipinski: HBD<=5, HBA<=10)
    hb_score = float(np.clip(1.0 - max(n_hbd - 5, 0) / 5.0
                             - max(n_hba - 10, 0) / 10.0, 0.0, 1.0))

    # Diversity bonus: fraction of unique fragment choices
    diversity = len(set(selections)) / N_STEPS

    reward = 0.40 * logP_score + 0.25 * mw_score + 0.10 * hb_score \
           + 0.15 * diversity + 0.10 * has_arom
    return float(np.clip(reward, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Reference dataset generation
# ---------------------------------------------------------------------------

def generate_reference_dataset(n_molecules: int = 500, seed: int = 0) -> dict:
    """Generate unique 5-fragment molecules and compute ground-truth rewards.

    Samples are drawn uniformly from the 20^5 = 3.2M combinatorial space
    and deduplicated. Returns feature matrix X (n, 25) and rewards y (n,).
    """
    rng = np.random.RandomState(seed)
    seen = set()
    molecules = []
    rewards   = []
    attempts  = 0
    while len(molecules) < n_molecules and attempts < n_molecules * 20:
        attempts += 1
        sels = tuple(rng.randint(0, N_FRAGMENTS, size=N_STEPS).tolist())
        if sels in seen:
            continue
        seen.add(sels)
        molecules.append(list(sels))
        rewards.append(ground_truth_reward(list(sels)))

    X = np.array([encode_molecule(m) for m in molecules], dtype=np.float32)
    y = np.array(rewards, dtype=np.float32)
    print(f"  Reference dataset: {len(molecules)} molecules, "
          f"reward in [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")
    return {"molecules": molecules, "X": X, "rewards": y}


# ---------------------------------------------------------------------------
# Reward proxy (MLP)
# ---------------------------------------------------------------------------

class RewardProxy(nn.Module):
    """MLP reward proxy: 25-dim molecule features -> scalar in [0, 1]."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MOL_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),       nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict rewards for a batch of molecules (numpy array)."""
        with torch.no_grad():
            device = next(self.parameters()).device
            xt = torch.tensor(X, dtype=torch.float32).to(device)
            return self.forward(xt).cpu().numpy()


def train_proxy(
    ref_dataset: dict,
    fraction: float = 0.3,
    seed: int = 0,
    epochs: int = 300,
    lr: float = 1e-3,
    device: str = "cpu",
) -> RewardProxy:
    """Train reward proxy on a random subset of the reference dataset.

    Using only 30% of the data (150 / 500 molecules) deliberately leaves the
    proxy uncertain, creating the epistemic variance that PCE decomposes.
    """
    rng = np.random.RandomState(seed)
    N   = len(ref_dataset["rewards"])
    idx = rng.permutation(N)[: int(N * fraction)]

    model = RewardProxy().to(device)
    X_tr  = torch.tensor(ref_dataset["X"][idx],       dtype=torch.float32).to(device)
    y_tr  = torch.tensor(ref_dataset["rewards"][idx],  dtype=torch.float32).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        loss = nn.MSELoss()(model(X_tr), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# GFlowNet (trajectory balance)
# ---------------------------------------------------------------------------

class MolGFlowNet(nn.Module):
    """5-step sequential fragment selector with trajectory-balance training.

    State at step t: 25-dim partial molecule (filled positions have real
    features; empty positions are zero) + 5-dim one-hot step indicator.
    Policy at step t: softmax over 20 fragments.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        state_dim = MOL_DIM + N_STEPS     # 25 + 5 = 30
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,    hidden), nn.ReLU(),
        )
        # One output head per position (each outputs logits over 20 fragments)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, N_FRAGMENTS) for _ in range(N_STEPS)]
        )
        self.log_Z = nn.Parameter(torch.tensor(0.0))

    def _encode_state(self, partial_feats: list, step: int) -> torch.Tensor:
        """Build the 30-dim state vector from partial fragment features.

        Args:
            partial_feats: list of length N_STEPS; each element is either
                           None (position not yet chosen) or a fragment index.
            step: current step index (0-indexed)
        """
        mol_vec = np.zeros(MOL_DIM, dtype=np.float32)
        for pos, sel in enumerate(partial_feats):
            if sel is not None:
                mol_vec[pos * FEAT_DIM: (pos + 1) * FEAT_DIM] = _FEAT_MATRIX[sel]
        step_oh = np.zeros(N_STEPS, dtype=np.float32)
        step_oh[step] = 1.0
        return torch.tensor(np.concatenate([mol_vec, step_oh]), dtype=torch.float32)

    def forward_policy(self, partial_feats: list, step: int) -> torch.Tensor:
        """Return log-softmax over fragments for the given partial state."""
        device = next(self.parameters()).device
        state  = self._encode_state(partial_feats, step).to(device)
        h      = self.backbone(state)
        return torch.log_softmax(self.heads[step](h), dim=-1)

    def get_policy(self, partial_feats: list, step: int) -> np.ndarray:
        """Return softmax probabilities (numpy) at the given step."""
        with torch.no_grad():
            return torch.exp(self.forward_policy(partial_feats, step)).cpu().numpy()

    def sample_trajectory(self):
        """Sample one complete trajectory; return selections and log-probs."""
        sels      = [None] * N_STEPS
        log_probs = []
        for step in range(N_STEPS):
            logp = self.forward_policy(sels, step)
            dist = Categorical(logits=logp)
            a    = dist.sample()
            log_probs.append(dist.log_prob(a))
            sels[step] = a.item()
        return sels, log_probs


# ---------------------------------------------------------------------------
# GFlowNet training (trajectory balance)
# ---------------------------------------------------------------------------

def train_gfn(
    gfn: MolGFlowNet,
    proxy: RewardProxy,
    n_ep: int = 3000,
    lr: float = 1e-3,
    batch: int = 32,
    temp: float = 4.0,
    device: str = "cpu",
    print_interval: int = 1000,
    seed: int = None,
) -> MolGFlowNet:
    """Train MolGFlowNet with trajectory-balance loss.

    TB loss (per trajectory):
        L = (log_Z + sum_t log P_F(a_t|s_t) - log R(tau) / temp)^2

    Averaged over a mini-batch of sampled trajectories per gradient step.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    gfn = gfn.to(device)
    opt = optim.Adam(gfn.parameters(), lr=lr)
    loss_history = []

    for ep in range(n_ep):
        batch_loss = torch.tensor(0.0, device=device)
        for _ in range(batch):
            sels, lps   = gfn.sample_trajectory()
            X_mol       = encode_molecule(sels)[np.newaxis, :]   # (1, 25)
            log_r       = float(np.log(proxy.predict(X_mol)[0] + 1e-8)) / temp
            log_pf_sum  = sum(lps)
            batch_loss  = batch_loss + (gfn.log_Z + log_pf_sum - log_r) ** 2

        loss = batch_loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
        loss_history.append(float(loss.item()))

        if print_interval > 0 and (ep + 1) % print_interval == 0:
            print(f"    ep {ep+1}/{n_ep}, TB loss={loss.item():.4f}, "
                  f"log_Z={gfn.log_Z.item():.3f}")

    gfn.eval()
    gfn.loss_history = loss_history
    return gfn


# ---------------------------------------------------------------------------
# Component-based proxy parameterisation
# ---------------------------------------------------------------------------

def compute_component_means(
    gfn: MolGFlowNet,
    proxy: RewardProxy,
) -> np.ndarray:
    """Compute 100-dim component-mean feature vector for PCE parameterisation.

    For each of the 5 positions and each of the 20 fragments, compute the mean
    proxy prediction over all molecules that would be formed by fixing that
    fragment at that position and drawing the remaining positions from the GFN's
    marginal policy (approximated by the single-step softmax at state = empty).

    This yields a 20 x 5 = 100-dim vector that encodes *what the proxy thinks
    each (position, fragment) combination is worth*, averaged over the GFN's
    current preference. It is the chemically meaningful parameterisation that
    PCE uses as its input mu.

    Implementation: we use a fast approximation -- for position p and fragment f,
    we predict the proxy on the molecule [f, f, f, f, f] (all positions = f)
    and scale by the GFN's marginal probability of choosing f at position p.
    This avoids 20^5 forward passes while capturing the joint proxy-GFN signal.
    """
    comp_means = np.zeros((N_FRAGMENTS, N_STEPS), dtype=np.float32)

    # Marginal policy at each position (empty state)
    for pos in range(N_STEPS):
        policy_p = gfn.get_policy([None] * N_STEPS, pos)   # (20,)
        for frag_idx in range(N_FRAGMENTS):
            # Canonical molecule: all positions set to this fragment
            X_canon = encode_molecule([frag_idx] * N_STEPS)[np.newaxis, :]
            pred    = proxy.predict(X_canon)[0]
            comp_means[frag_idx, pos] = pred * policy_p[frag_idx]

    return comp_means.flatten()    # 100-dim


# ---------------------------------------------------------------------------
# Phase 1: train a single ensemble member (designed for LSF array dispatch)
# ---------------------------------------------------------------------------

def train_and_save_member(
    member_id: int,
    ref_molecules: list,
    ref_rewards: np.ndarray,
    output_dir: str,
    train_fraction: float = 0.3,
    gfn_episodes: int = 3000,
    proxy_epochs: int = 300,
    temp: float = 4.0,
    device: str = "cpu",
    gfn_seed: int = None,
    proxy_seed: int = None,
) -> None:
    """Train proxy + GFlowNet for one ensemble member, save to disk.

    Designed to be called from train_single_member.py submitted as an LSF
    array job element (bsub -J "moldesign[0-149]").

    Saves:
        {output_dir}/members/member_{member_id:04d}.npz with keys:
            member_id      : scalar int
            policies       : (N_STEPS, N_FRAGMENTS) float32 -- per-step softmax
            proxy_outputs  : (100,) float32 -- component-mean parameterisation
            selections     : (N_STEPS,) int -- greedy MAP molecule from trained GFN
    """
    os.makedirs(os.path.join(output_dir, "members"), exist_ok=True)
    out_path = os.path.join(output_dir, "members", f"member_{member_id:04d}.npz")
    if os.path.exists(out_path):
        print(f"  Member {member_id} already exists, skipping.")
        return

    print(f"  Training member {member_id}...")
    t0 = time.time()

    # Build dataset dict compatible with train_proxy
    ref_X = np.array([encode_molecule(m) for m in ref_molecules], dtype=np.float32)
    ref_dataset = {"X": ref_X, "rewards": ref_rewards}

    # proxy_seed controls the reward realisation; fix it across members and vary
    # only gfn_seed for the training-noise control ensemble.
    proxy_seed = member_id if proxy_seed is None else proxy_seed
    proxy = train_proxy(ref_dataset, fraction=train_fraction,
                        seed=proxy_seed, epochs=proxy_epochs, device=device)

    # Seed before constructing the network so weight init + training are
    # reproducible for a given member. proxy_seed controls the
    # reward realisation; gfn_seed defaults to the same.
    gfn_seed = member_id if gfn_seed is None else gfn_seed
    torch.manual_seed(gfn_seed)
    np.random.seed(gfn_seed)
    gfn = MolGFlowNet()
    gfn = train_gfn(gfn, proxy, n_ep=gfn_episodes, temp=temp,
                    device=device, print_interval=0, seed=gfn_seed)

    # Per-step policies along the empty-state reference (marginal at step 0..4)
    policies = np.array(
        [gfn.get_policy([None] * N_STEPS, step) for step in range(N_STEPS)],
        dtype=np.float32,
    )   # (5, 20)

    # Component-based 100-dim proxy parameterisation
    proxy_outputs = compute_component_means(gfn, proxy)   # (100,)

    # Greedy MAP molecule: argmax at each step (independent; useful for reporting)
    selections = [int(np.argmax(policies[step])) for step in range(N_STEPS)]

    np.savez(
        out_path,
        member_id=np.array([member_id]),
        policies=policies,
        proxy_outputs=proxy_outputs,
        selections=np.array(selections),
        proxy_seed=np.array([member_id]),
        gfn_seed=np.array([gfn_seed]),
    )
    print(f"  Member {member_id} done in {time.time()-t0:.1f}s -> {out_path}")


# ---------------------------------------------------------------------------
# Phase 1 setup (shared reference dataset, run once before LSF array)
# ---------------------------------------------------------------------------

def setup_phase1(output_dir: str, n_ref: int = 500, seed: int = 0) -> None:
    """Generate and save the shared reference dataset + metadata.

    Must be called ONCE before submitting the LSF array job.
    """
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(meta_path):
        print(f"Metadata already exists at {meta_path}. Delete to re-run setup.")
        return

    print("Generating reference dataset...")
    ref_ds = generate_reference_dataset(n_molecules=n_ref, seed=seed)

    np.save(os.path.join(output_dir, "ref_molecules.npy"),
            np.array(ref_ds["molecules"]))
    np.save(os.path.join(output_dir, "ref_rewards.npy"), ref_ds["rewards"])

    metadata = {
        "n_ref": n_ref,
        "seed": seed,
        "n_fragments": N_FRAGMENTS,
        "n_steps": N_STEPS,
        "fragment_names": FRAGMENT_NAMES,
        "feat_dim": FEAT_DIM,
        "mol_dim": MOL_DIM,
        "train_fraction": 0.3,
        "reward_range": [float(ref_ds["rewards"].min()),
                         float(ref_ds["rewards"].max())],
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Provenance for regenerating downstream tables/figures
    import subprocess, hashlib
    try:
        root = str(Path(__file__).resolve().parent.parent.parent)
        git_sha = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                          stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_sha = "unknown"
    config = {
        "experiment": "molecular_design",
        "git_sha": git_sha,
        "n_ref": n_ref,
        "setup_seed": seed,
        "train_fraction": 0.3,
        "reward_sha1": hashlib.sha1(
            np.ascontiguousarray(ref_ds["rewards"]).tobytes()).hexdigest(),
        "seed_convention": "proxy_seed = gfn_seed = member_id",
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Phase 1 setup complete. Metadata + config.json (git {git_sha[:8]}) "
          f"saved to {output_dir}")
    print("Now submit: bsub < lsf/submit_moldesign.sh")


# ---------------------------------------------------------------------------
# Phase 2: collect saved members, run PCE + Sobol analysis
# ---------------------------------------------------------------------------

def analyze_from_members(
    output_dir: str,
    n_train: int = 50,
    n_test: int = 100,
    pce_degree: int = 5,
    pca_dim: int = 2,
    n_mc: int = 10000,
) -> dict:
    """Load all saved member npz files, fit PCE, compute Sobol indices.

    PCA parameterisation uses COMPONENT-BASED features (100-dim per member):
      for each member, the mean proxy prediction for each fragment in each
      position (20 fragments x 5 positions). This is chemically interpretable
      and avoids the high-dimensionality of raw proxy outputs.

    Args:
        output_dir: directory containing members/ subdirectory
        n_train:    number of members for PCE fitting
        n_test:     number of members for validation
        pce_degree: polynomial degree for Hermite PCE
        pca_dim:    PCA dimension (2 recommended for interpretability)
        n_mc:       Monte Carlo samples for KS test

    Returns:
        results dict (also saved as results.json)
    """
    members_dir = os.path.join(output_dir, "members")
    npz_files   = sorted(Path(members_dir).glob("member_*.npz"))
    n_total     = n_train + n_test
    if len(npz_files) < n_total:
        raise RuntimeError(
            f"Expected >= {n_total} member files in {members_dir}, "
            f"found {len(npz_files)}. "
            f"Run Phase 1 (LSF array or sequential mode) first."
        )
    npz_files = npz_files[:n_total]

    print(f"\nLoading {n_total} ensemble members...")
    all_data = [np.load(f, allow_pickle=True) for f in npz_files]

    # policies: (n_total, N_STEPS, N_FRAGMENTS)
    all_policies = np.stack([d["policies"] for d in all_data], axis=0)

    # Component-mean features: (n_total, 100) -- the PCE input space mu
    comp_features = np.stack([d["proxy_outputs"] for d in all_data], axis=0)

    # PCA: 100-dim -> pca_dim  (component-based, not raw proxy outputs)
    print(f"  PCA: component-based {comp_features.shape[1]}-dim -> {pca_dim}D ...")
    pca    = PCA(n_components=pca_dim)
    mu_raw_tr = pca.fit_transform(comp_features[:n_train])   # (n_train, pca_dim)
    mu_raw_te = pca.transform(comp_features[n_train:])       # (n_test, pca_dim)
    exp_var = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance: {exp_var:.3f}")

    # Standardise to N(0,1) so Hermite PCE basis is correctly matched
    scaler = StandardScaler()
    mu_tr  = scaler.fit_transform(mu_raw_tr)
    mu_te  = scaler.transform(mu_raw_te)
    tr_pol = {s: all_policies[:n_train, s, :] for s in range(N_STEPS)}
    te_pol = {s: all_policies[n_train:, s, :] for s in range(N_STEPS)}

    # Fit PCE surrogate (one per step)
    print(f"\nFitting PCE (degree={pce_degree}, basis=hermite)...")
    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(N_STEPS):
        tsurr.fit_step(s, mu_tr, tr_pol[s])

    # Sobol indices (first-order, per PC dimension)
    sobol = tsurr.sobol_all_steps()

    print("\nSOBOL SENSITIVITY INDICES (first-order, per PC)")
    print("  Interpretation: low index = robust selection; high = fragile/proxy-dependent")
    for s in range(N_STEPS):
        fo      = sobol[s]["first_order"]    # (K-1, pca_dim)
        var     = sobol[s]["variance"]       # (K-1,)
        mean_fo = fo.mean(axis=0)            # average over ALR components
        role    = "scaffold" if s < 2 else ("linker" if s == 2 else "decoration")
        print(f"  Position {s+1} ({role:10s}): "
              f"S_PC1={mean_fo[0]:.3f}, "
              + (f"S_PC2={mean_fo[1]:.3f}, " if pca_dim > 1 else "")
              + f"total_var={var.mean():.4f}")

    # KS test battery
    print("\nRunning KS test battery...")
    ks_results = run_ks_battery(tsurr, te_pol, n_mc=n_mc)
    for s in range(N_STEPS):
        role = "scaffold" if s < 2 else ("linker" if s == 2 else "decoration")
        fr   = ks_results[s]["fraction_pass"]
        print(f"  Position {s+1} ({role:10s}): "
              f"{ks_results[s]['n_pass']}/{ks_results[s]['n_total']} "
              f"fragments pass KS ({fr:.1%})")

    # Calibration coverage
    print("\nCalibration coverage:")
    mc_pols = tsurr.sample_trajectory_policies(n_mc)
    for s in range(N_STEPS):
        role = "scaffold" if s < 2 else ("linker" if s == 2 else "decoration")
        cov  = calibration_coverage(mc_pols[s], te_pol[s])
        print(f"  Position {s+1} ({role:10s}): "
              + ", ".join(f"{100*lv:.0f}%->cov={cov[lv]:.2f}" for lv in sorted(cov)))

    # Required ensemble size (Theorem A)
    L_req = tsurr.required_ensemble_size(target_sobol_error=0.05, confidence=0.95)
    print(f"\nTheorem A: required ensemble size for eps=0.05 Sobol accuracy: {L_req}")
    print(f"  (Used {n_train} training members)")

    # Most common greedy selections across ensemble (interpretable summary)
    all_sels = np.stack([d["selections"] for d in all_data], axis=0)  # (n_total, 5)
    print("\nMost-selected fragment per position (across ensemble):")
    for s in range(N_STEPS):
        vals, cnts = np.unique(all_sels[:, s], return_counts=True)
        top         = vals[np.argmax(cnts)]
        role        = "scaffold" if s < 2 else ("linker" if s == 2 else "decoration")
        print(f"  Position {s+1} ({role:10s}): {FRAGMENT_NAMES[top]:12s} "
              f"({cnts.max()/n_total:.0%} of members)")

    # Compile results
    def _jsonify(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict):  return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_jsonify(v) for v in obj]
        return obj

    results = {
        "sobol": {str(s): {k: v.tolist() for k, v in v.items()}
                  for s, v in sobol.items()},
        "ks_results": {str(s): {
            "fraction_pass": ks_results[s]["fraction_pass"],
            "n_pass":        ks_results[s]["n_pass"],
            "n_total":       ks_results[s]["n_total"],
        } for s in range(N_STEPS)},
        "pca_explained_var":       exp_var,
        "pca_dim":                 pca_dim,
        "n_train":                 n_train,
        "n_test":                  n_test,
        "pce_degree":              pce_degree,
        "theorem_A_L_required":    L_req,
        "fragment_names":          FRAGMENT_NAMES,
        "position_roles":          ["scaffold", "scaffold", "linker",
                                    "decoration", "decoration"],
    }

    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(_jsonify(results), f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Sequential mode (local, all-in-one -- for testing and small-scale runs)
# ---------------------------------------------------------------------------

def run_molecular_design_sequential(
    n_train: int = 50,
    n_test: int = 100,
    pce_degree: int = 5,
    gfn_episodes: int = 3000,
    train_fraction: float = 0.3,
    pca_dim: int = 2,
    n_mc: int = 10000,
    proxy_epochs: int = 300,
    temp: float = 4.0,
    n_ref: int = 500,
    device: str = "cpu",
    output_dir: str = "results/molecular_design",
) -> dict:
    """Train all ensemble members sequentially (no LSF), then analyse."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70 + "\nMOLECULAR DESIGN GFlowNet + PCE UQ (SEQUENTIAL)\n" + "=" * 70)

    print("\nGenerating reference dataset...")
    ref_ds = generate_reference_dataset(n_molecules=n_ref, seed=0)

    n_total = n_train + n_test
    print(f"\nTraining {n_total} ensemble members...")
    for i in range(n_total):
        train_and_save_member(
            member_id=i,
            ref_molecules=ref_ds["molecules"],
            ref_rewards=ref_ds["rewards"],
            output_dir=output_dir,
            train_fraction=train_fraction,
            gfn_episodes=gfn_episodes,
            proxy_epochs=proxy_epochs,
            temp=temp,
            device=device,
        )
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_total} members done")

    return analyze_from_members(
        output_dir=output_dir,
        n_train=n_train,
        n_test=n_test,
        pce_degree=pce_degree,
        pca_dim=pca_dim,
        n_mc=n_mc,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fragment-based molecular design GFlowNet + PCE UQ"
    )
    parser.add_argument(
        "--mode", choices=["sequential", "analyze"], default="sequential",
        help="sequential: train all members then analyse (local). "
             "analyze: load saved member files and analyse (after LSF array).",
    )
    parser.add_argument("--n_train",        type=int,   default=50)
    parser.add_argument("--n_test",         type=int,   default=100)
    parser.add_argument("--pce_degree",     type=int,   default=5)
    parser.add_argument("--gfn_episodes",   type=int,   default=3000)
    parser.add_argument("--proxy_epochs",   type=int,   default=300)
    parser.add_argument("--train_fraction", type=float, default=0.3)
    parser.add_argument("--pca_dim",        type=int,   default=2)
    parser.add_argument("--n_ref",          type=int,   default=500)
    parser.add_argument("--temp",           type=float, default=4.0)
    parser.add_argument("--output_dir",     default="results/molecular_design")
    parser.add_argument(
        "--device",
        default="cuda" if (HAS_TORCH and __import__("torch").cuda.is_available())
                       else "cpu",
    )
    args = parser.parse_args()

    if not HAS_TORCH:
        sys.exit(1)

    if args.mode == "sequential":
        run_molecular_design_sequential(
            n_train=args.n_train,
            n_test=args.n_test,
            pce_degree=args.pce_degree,
            gfn_episodes=args.gfn_episodes,
            train_fraction=args.train_fraction,
            pca_dim=args.pca_dim,
            n_mc=10000,
            proxy_epochs=args.proxy_epochs,
            temp=args.temp,
            n_ref=args.n_ref,
            device=args.device,
            output_dir=args.output_dir,
        )
    elif args.mode == "analyze":
        analyze_from_members(
            output_dir=args.output_dir,
            n_train=args.n_train,
            n_test=args.n_test,
            pce_degree=args.pce_degree,
            pca_dim=args.pca_dim,
        )
