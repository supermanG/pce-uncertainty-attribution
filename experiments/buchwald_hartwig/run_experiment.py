"""Buchwald-Hartwig Reaction Condition Optimization via GFlowNet + PCE UQ.

Dataset: Doyle-Dreher (Ahneman et al., Science 2018) -- real Pd-catalysed C-N coupling.
GFlowNet: sequentially selects (catalyst, base, aryl_halide, additive) in 4 steps.

Usage modes:
  1. Sequential (local, small-scale):
       python run_experiment.py --mode sequential

  2. SLURM cluster (IBM CCC):
       See ../../../slurm/submit_bh.sh -- submits array job for Phase 1,
       then this script for Phase 2 analysis.

  3. Collect and analyse only (after array jobs complete):
       python run_experiment.py --mode analyze --output_dir results/buchwald_hartwig

Data: place the Doyle-Dreher CSV at data/data_table.csv
  (download from https://github.com/doylelab/rxnpredict -- file: data_table.csv)
  Falls back to synthetic data with a warning if CSV not found.

Requires: torch, numpy, scipy, scikit-learn, pandas
"""
import os, sys, json, time, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not found.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from core.pce_surrogate import (
    TrajectoryPCESurrogate, calibration_coverage, run_ks_battery
)
from core.distributional_analysis import (
    analyse_trajectory_bimodality, plot_bimodality_panel, plot_surrogate_comparison,
)


# ---------------------------------------------------------------------------
# Real Doyle-Dreher data loading
# ---------------------------------------------------------------------------

# Column name aliases used across different versions of the public CSV
_LIGAND_COLS   = ["Ligand", "ligand", "Catalyst", "catalyst", "L"]
_REAGENT_COLS  = ["Reagent", "reagent", "Base", "base", "B"]
_ARYL_COLS     = ["Aryl halide", "Aryl_halide", "aryl_halide", "Aryl", "ArylHalide", "AH"]
_ADDITIVE_COLS = ["Additive", "additive", "Ad", "A"]
_YIELD_COLS    = ["Output", "output", "Yield", "yield", "Yield_M", "%Yield", "y"]

# Known dataset sizes for validation
_EXPECTED_REACTIONS = (3955, 4600)   # after removing failures vs. full grid (data_table.csv has 4599 incl. replicates)


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_real_doyle_dreher(csv_path: str) -> dict:
    """Load the Doyle-Dreher dataset from a CSV file.

    Handles multiple known column-name variants. Categorical columns are
    label-encoded to integer indices in the order they first appear.

    Args:
        csv_path: path to the CSV file

    Returns:
        dataset dict compatible with the rest of this experiment, or raises
        FileNotFoundError / ValueError with a descriptive message.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required to load real data: pip install pandas")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Doyle-Dreher CSV not found at: {csv_path}\n"
            f"Download from: https://github.com/doylelab/rxnpredict\n"
            f"Expected filename: data_table.csv  (repo root)"
        )

    df = pd.read_csv(csv_path)
    print(f"  Loaded CSV with {len(df)} rows, columns: {list(df.columns)}")

    # Locate columns
    lig_col = _find_col(df, _LIGAND_COLS)
    rea_col = _find_col(df, _REAGENT_COLS)
    ary_col = _find_col(df, _ARYL_COLS)
    add_col = _find_col(df, _ADDITIVE_COLS)
    yld_col = _find_col(df, _YIELD_COLS)

    missing = [name for name, col in [
        ("Ligand/Catalyst", lig_col), ("Reagent/Base", rea_col),
        ("Aryl halide", ary_col), ("Additive", add_col), ("Yield", yld_col)
    ] if col is None]
    if missing:
        raise ValueError(
            f"Could not find columns for: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Please open the CSV and rename columns to match one of the expected names."
        )

    # Drop rows with NaN yield
    df = df.dropna(subset=[yld_col])
    yields = df[yld_col].values.astype(float)

    # Label-encode each categorical
    def encode(series):
        cats = list(dict.fromkeys(series.values))   # unique, order-preserving
        mapping = {c: i for i, c in enumerate(cats)}
        return np.array([mapping[v] for v in series.values]), cats

    cat_enc, cat_names = zip(*[encode(df[c]) for c in [lig_col, rea_col, ary_col, add_col]])
    reactions = np.stack(cat_enc, axis=1)   # (N, 4)
    n_components = [len(cn) for cn in cat_names]

    # Validate
    N = len(yields)
    if N < _EXPECTED_REACTIONS[0] or N > _EXPECTED_REACTIONS[1]:
        print(f"  WARNING: expected {_EXPECTED_REACTIONS[0]}-{_EXPECTED_REACTIONS[1]} "
              f"reactions, got {N}. Proceed with caution.")

    print(f"  Dataset: {N} reactions, components: {n_components}")
    print(f"  Yield range: [{yields.min():.1f}, {yields.max():.1f}]%, "
          f"mean={yields.mean():.1f}%")

    return {
        "reactions": reactions,
        "yields": yields,
        "n_components": n_components,
        "component_names": ["catalyst", "base", "aryl_halide", "additive"],
        "component_labels": [list(cn) for cn in cat_names],
        "source": "real",
        "csv_path": csv_path,
    }


def generate_doyle_dreher_synthetic(seed: int = 42) -> dict:
    """Synthetic Doyle-Dreher dataset (fallback when real CSV not available).

    Uses the same additive model structure as in the original paper exploration,
    but does NOT represent real experimental data.
    """
    print("  WARNING: Using synthetic Doyle-Dreher data. "
          "Results will NOT be publishable.\n"
          "  Download real data from: https://github.com/doylelab/rxnpredict\n"
          "  Place at: experiments/buchwald_hartwig/data/Dreher_and_Doyle_input_data.csv")
    rng = np.random.RandomState(seed)
    n_aryl, n_cat, n_base, n_add = 15, 4, 3, 23
    aryl_eff = rng.randn(n_aryl) * 15 + 50
    cat_eff  = rng.randn(n_cat) * 20
    base_eff = rng.randn(n_base) * 10
    add_eff  = rng.randn(n_add) * 12
    cat_add  = rng.randn(n_cat, n_add) * 8
    reactions, yields = [], []
    for a in range(n_aryl):
        for c in range(n_cat):
            for b in range(n_base):
                for ad in range(n_add):
                    y = float(np.clip(
                        aryl_eff[a] + cat_eff[c] + base_eff[b] + add_eff[ad]
                        + cat_add[c, ad] + rng.randn() * 5, 0, 100
                    ))
                    reactions.append([a, c, b, ad])
                    yields.append(y)
    return {
        "reactions": np.array(reactions),
        "yields": np.array(yields),
        "n_components": [n_aryl, n_cat, n_base, n_add],
        "component_names": ["aryl_halide", "catalyst", "base", "additive"],
        "component_labels": None,
        "source": "synthetic",
    }


def load_dataset(csv_path: str = None) -> dict:
    """Load real or synthetic dataset with graceful fallback."""
    if csv_path is None:
        # Try default location relative to this script
        script_dir = Path(__file__).parent
        # Try data_table.csv (preferred), then legacy filename
        default = script_dir / "data" / "data_table.csv"
        if not default.exists():
            default = script_dir / "data" / "Dreher_and_Doyle_input_data.csv"
        csv_path = str(default)
    try:
        return load_real_doyle_dreher(csv_path)
    except (FileNotFoundError, ImportError) as e:
        print(f"  Could not load real data: {e}")
        print("  Falling back to SYNTHETIC data (not for publication).")
        return generate_doyle_dreher_synthetic()


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class YieldProxy(nn.Module):
    def __init__(self, n_components: list, hidden: int = 128):
        super().__init__()
        self.n_components = n_components
        self.net = nn.Sequential(
            nn.Linear(sum(n_components), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def encode(self, reactions: np.ndarray) -> torch.Tensor:
        parts = []
        for i, n in enumerate(self.n_components):
            oh = np.zeros((len(reactions), n))
            oh[np.arange(len(reactions)), reactions[:, i].astype(int)] = 1.0
            parts.append(oh)
        return torch.tensor(np.concatenate(parts, axis=1), dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_yield(self, reactions: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            device = next(self.parameters()).device
            return self.forward(self.encode(reactions).to(device)).cpu().numpy()


class ReactionGFlowNet(nn.Module):
    def __init__(self, n_components: list, hidden: int = 64):
        super().__init__()
        self.n_components = n_components
        self.n_steps = len(n_components)
        total = sum(n_components) + self.n_steps
        self.backbone = nn.Sequential(
            nn.Linear(total, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in n_components])
        self.log_Z = nn.Parameter(torch.tensor(5.0))

    def _encode(self, sels: list, step: int) -> torch.Tensor:
        parts = []
        for i, n in enumerate(self.n_components):
            oh = torch.zeros(n)
            if sels[i] is not None:
                oh[sels[i]] = 1.0
            parts.append(oh)
        s = torch.zeros(self.n_steps)
        s[step] = 1.0
        parts.append(s)
        return torch.cat(parts)

    def forward_policy(self, sels: list, step: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.log_softmax(
            self.heads[step](self.backbone(self._encode(sels, step).to(device))), dim=-1
        )

    def get_policy(self, sels: list, step: int) -> np.ndarray:
        with torch.no_grad():
            return torch.exp(self.forward_policy(sels, step)).cpu().numpy()

    def sample_trajectory(self):
        sels = [None] * self.n_steps
        log_probs = []
        for step in range(self.n_steps):
            logp = self.forward_policy(sels, step)
            dist = Categorical(logits=logp)
            a = dist.sample()
            log_probs.append(dist.log_prob(a))
            sels[step] = a.item()
        return sels, log_probs


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_proxy(
    dataset: dict,
    fraction: float = 0.3,
    seed: int = 0,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
) -> YieldProxy:
    rng = np.random.RandomState(seed)
    N = len(dataset["yields"])
    idx = rng.permutation(N)[: int(N * fraction)]
    model = YieldProxy(dataset["n_components"]).to(device)
    x = model.encode(dataset["reactions"][idx]).to(device)
    y = torch.tensor(dataset["yields"][idx], dtype=torch.float32).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        loss = nn.MSELoss()(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def train_gflownet(
    gfn: ReactionGFlowNet,
    proxy: YieldProxy,
    n_ep: int = 3000,
    lr: float = 1e-3,
    batch: int = 32,
    device: str = "cpu",
    temp: float = 4.0,
    print_interval: int = 1000,
    seed: int = None,
) -> ReactionGFlowNet:
    # Seed the GFlowNet training stochasticity for reproducibility. When set, the
    # ensemble spread from the reward (proxy) can be isolated from training noise
    # by fixing the proxy seed and varying only this one.
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    gfn = gfn.to(device)
    opt = optim.Adam(gfn.parameters(), lr=lr)
    loss_history = []
    for ep in range(n_ep):
        loss_sum = 0.0
        for _ in range(batch):
            sels, lps = gfn.sample_trajectory()
            r = proxy.predict_yield(np.array([sels]))[0] / temp
            loss_sum += (gfn.log_Z + sum(lps) - r) ** 2
        loss = loss_sum / batch
        opt.zero_grad(); loss.backward(); opt.step()
        loss_history.append(float(loss.item()))
        if print_interval > 0 and (ep + 1) % print_interval == 0:
            print(f"    ep {ep+1}/{n_ep}, loss={loss.item():.4f}")
    gfn.eval()
    gfn.loss_history = loss_history
    return gfn


def extract_policy(gfn: ReactionGFlowNet, ref_traj: list) -> dict:
    """Extract policy at each step along a reference trajectory."""
    sels = [None] * len(ref_traj)
    policies = {}
    for step in range(len(ref_traj)):
        policies[step] = gfn.get_policy(sels, step)
        sels[step] = ref_traj[step]
    return policies


def select_reference_trajectory(dataset: dict, proxy: YieldProxy = None) -> list:
    """Select reference trajectory as the highest-yield condition.

    If a proxy is provided, use proxy predictions (proxy-optimal trajectory).
    Otherwise use true yields.
    """
    if proxy is not None:
        predicted = proxy.predict_yield(dataset["reactions"])
        best_idx = int(np.argmax(predicted))
    else:
        best_idx = int(np.argmax(dataset["yields"]))
    ref = dataset["reactions"][best_idx].tolist()
    print(f"  Reference trajectory: {ref} "
          f"(yield={dataset['yields'][best_idx]:.1f}%)")
    return ref


# ---------------------------------------------------------------------------
# Phase 1: train a single ensemble member (designed for SLURM array jobs)
# ---------------------------------------------------------------------------

def train_and_save_member(
    member_id: int,
    dataset: dict,
    ref_traj: list,
    ref_reactions: np.ndarray,
    output_dir: str,
    train_fraction: float = 0.3,
    gfn_episodes: int = 3000,
    device: str = "cpu",
    proxy_epochs: int = 200,
    temp: float = 4.0,
    proxy_seed: int = None,
    gfn_seed: int = None,
) -> None:
    """Train proxy + GFlowNet for one ensemble member, save policy to disk.

    This function is designed to be called from train_single_gfn.py, which is
    submitted as a cluster array job element.

    Seeds (reproducibility):
        proxy_seed controls the reward realisation (which data subset the proxy
        sees); gfn_seed controls GFlowNet training stochasticity. Both default to
        member_id (the main ensemble: each member is a distinct reward + training
        draw). For the training-noise control ensemble, fix proxy_seed across
        members and vary only gfn_seed to isolate training variance.

    Saves {output_dir}/members/member_{member_id:04d}.npz with per-step policies,
    proxy_outputs on the reference reactions, member_id, both seeds, final_loss.
    """
    proxy_seed = member_id if proxy_seed is None else proxy_seed
    gfn_seed = member_id if gfn_seed is None else gfn_seed
    os.makedirs(os.path.join(output_dir, "members"), exist_ok=True)
    out_path = os.path.join(output_dir, "members", f"member_{member_id:04d}.npz")
    if os.path.exists(out_path):
        print(f"  Member {member_id} already exists, skipping.")
        return

    print(f"  Training member {member_id} (proxy_seed={proxy_seed}, gfn_seed={gfn_seed})...")
    t0 = time.time()

    proxy = train_proxy(dataset, fraction=train_fraction, seed=proxy_seed,
                        epochs=proxy_epochs, device=device)
    proxy_outputs = proxy.predict_yield(ref_reactions)  # (n_ref,)

    # Seed before constructing the network so weight initialisation (not just the
    # sampling RNG during training) is reproducible for a given gfn_seed.
    torch.manual_seed(gfn_seed)
    np.random.seed(gfn_seed)
    gfn = ReactionGFlowNet(dataset["n_components"])
    gfn = train_gflownet(gfn, proxy, n_ep=gfn_episodes, device=device,
                         temp=temp, print_interval=0, seed=gfn_seed)

    policies = extract_policy(gfn, ref_traj)
    loss_hist = getattr(gfn, "loss_history", [])
    final_loss = loss_hist[-1] if loss_hist else float("nan")

    np.savez(out_path,
             **{f"policy_step{s}": v for s, v in policies.items()},
             proxy_outputs=proxy_outputs,
             member_id=np.array([member_id]),
             proxy_seed=np.array([proxy_seed]),
             gfn_seed=np.array([gfn_seed]),
             final_loss=np.array([final_loss]))

    print(f"  Member {member_id} done in {time.time()-t0:.1f}s, "
          f"final_loss={final_loss:.4f}")


# ---------------------------------------------------------------------------
# Experimental design recommendation from Sobol indices
# ---------------------------------------------------------------------------

def recommend_experimental_design(sobol, component_names, pca_dim):
    """Generate an actionable experimental design recommendation from Sobol indices.

    After computing Sobol sensitivity indices for each trajectory step, this
    function ranks the experimental dimensions (catalyst, base, aryl_halide,
    additive) by their mean total-order Sobol index across ALR components.
    Steps with high indices are those where reward uncertainty most strongly
    drives policy uncertainty -- targeted data collection there would most
    reduce epistemic uncertainty.

    Args:
        sobol: dict {step_int -> {"first_order": (K-1, d), "total_order": (K-1, d),
               "variance": (K-1,)}} as returned by TrajectoryPCESurrogate.sobol_all_steps().
        component_names: list of str, human-readable name for each step
               (e.g. ["catalyst", "base", "aryl_halide", "additive"]).
        pca_dim: int, number of PCA dimensions (d) used in the reward parameterisation.

    Returns:
        dict with keys:
            "ranking":        list of dicts sorted by descending mean S_total,
                              each with "step", "name", "mean_S_total", "action".
            "text":           multi-line plain-English recommendation string.
    """
    n_steps = len(sobol)

    # 1. Average total-order Sobol indices across ALR components for each step.
    #    total_order has shape (K-1, d) where K-1 = ALR components, d = PCA dims.
    #    We average over both ALR components and PCA dimensions to get a single
    #    scalar per step.
    mean_totals = {}
    for s in range(n_steps):
        to = sobol[s]["total_order"]  # (K-1, d)
        # Average over ALR components (axis 0) then over PCA dims (axis 1)
        mean_totals[s] = float(np.mean(to))

    # 2. Sort steps by descending mean total-order index.
    ranked = sorted(mean_totals.items(), key=lambda x: x[1], reverse=True)

    # 3. Assign action labels.  Use the median as a threshold: top half EXPLORE,
    #    bottom half HOLD FIXED.
    n_explore = max(1, n_steps // 2)  # at least 1 explore
    ranking = []
    for rank_idx, (step, mean_st) in enumerate(ranked):
        name = component_names[step] if step < len(component_names) else f"step_{step}"
        if rank_idx < n_explore:
            if rank_idx == 0:
                action = "EXPLORE: policy most sensitive here"
            else:
                action = "EXPLORE: moderate sensitivity"
        else:
            action = "HOLD FIXED: low sensitivity"
        ranking.append({
            "step": int(step),
            "name": name,
            "mean_S_total": round(mean_st, 4),
            "action": action,
        })

    # 4. Build plain-English text.
    lines = []
    lines.append("")
    lines.append("EXPERIMENTAL DESIGN RECOMMENDATION")
    lines.append("=" * 39)
    lines.append("Priority ranking (highest Sobol = most uncertain, explore first):")
    max_name_len = max(len(r["name"]) for r in ranking)
    for i, r in enumerate(ranking):
        lines.append(
            f"  {i+1}. {r['name']:<{max_name_len}} (mean S_total = {r['mean_S_total']:.2f})"
            f" -- {r['action']}"
        )

    explore_names = [r["name"] for r in ranking[:n_explore]]
    hold_names = [r["name"] for r in ranking[n_explore:]]
    lines.append("")
    if len(explore_names) == 1:
        explore_str = f"varying the {explore_names[0]} dimension"
    else:
        explore_str = (
            "varying the "
            + " and ".join([", ".join(explore_names[:-1]), explore_names[-1]])
            + " dimensions"
        )
    if hold_names:
        if len(hold_names) == 1:
            hold_str = (
                f"The {hold_names[0]} choice has minimal impact on policy "
                f"uncertainty and can be held at its current best value."
            )
        else:
            hold_str = (
                f"The {' and '.join([', '.join(hold_names[:-1]), hold_names[-1]])} "
                f"choices have minimal impact on policy uncertainty and can be "
                f"held at their current best values."
            )
    else:
        hold_str = ""

    lines.append(
        f"Recommendation: Focus experimental resources on {explore_str}."
    )
    if hold_str:
        lines.append(hold_str)

    text = "\n".join(lines)

    return {"ranking": ranking, "text": text}


# ---------------------------------------------------------------------------
# Phase 2: collect all saved members, run PCE + Sobol analysis
# ---------------------------------------------------------------------------

def collect_and_analyze(
    output_dir: str,
    n_train: int = 50,
    n_test: int = 100,
    pce_degree: int = 5,
    pca_dim: int = 2,
    n_mc: int = 10000,
    dataset: dict = None,
) -> dict:
    """Load all saved member files, fit PCE, compute Sobol indices.

    Args:
        output_dir: directory containing members/ subdirectory
        n_train:    number of members used for PCE fitting
        n_test:     number of members used for validation
        pce_degree: PCE polynomial degree
        pca_dim:    PCA dimension for reward parameterisation
        n_mc:       Monte Carlo samples for KS test
        dataset:    original dataset (for labels/component names)

    Returns:
        results dict
    """
    members_dir = os.path.join(output_dir, "members")
    npz_files = sorted(Path(members_dir).glob("member_*.npz"))
    n_total = n_train + n_test
    if len(npz_files) < n_total:
        raise RuntimeError(
            f"Expected >= {n_total} member files in {members_dir}, "
            f"found {len(npz_files)}. "
            f"Run Phase 1 (SLURM array or sequential mode) first."
        )
    npz_files = npz_files[:n_total]

    # Load all members
    print(f"\nLoading {n_total} ensemble members...")
    all_data = [np.load(f, allow_pickle=True) for f in npz_files]

    # Discover trajectory steps
    sample = all_data[0]
    n_steps = sum(1 for k in sample.files if k.startswith("policy_step"))

    # Assemble policy matrices: {step -> (n_total, K)}
    all_policies = {}
    for step in range(n_steps):
        all_policies[step] = np.stack(
            [d[f"policy_step{step}"] for d in all_data], axis=0
        )  # (n_total, K)

    # Proxy outputs for PCA: (n_total, n_ref)
    proxy_outputs = np.stack([d["proxy_outputs"] for d in all_data], axis=0)

    # PCA on proxy outputs -> reward parameterisation mu
    # IMPORTANT: standardize to N(0,1) after PCA so Hermite PCE basis is correct
    print(f"  PCA: {proxy_outputs.shape[1]} proxy dims -> {pca_dim}D...")
    pca = PCA(n_components=pca_dim)
    mu_raw_tr = pca.fit_transform(proxy_outputs[:n_train])
    mu_raw_te = pca.transform(proxy_outputs[n_train:])
    exp_var = float(pca.explained_variance_ratio_.sum())
    print(f"  Explained variance: {exp_var:.3f}")
    scaler = StandardScaler()
    mu_tr = scaler.fit_transform(mu_raw_tr)
    mu_te = scaler.transform(mu_raw_te)
    tr_pol = {s: all_policies[s][:n_train] for s in range(n_steps)}
    te_pol = {s: all_policies[s][n_train:] for s in range(n_steps)}

    # Convergence summary (if final_loss was saved)
    final_losses = []
    for d in all_data:
        if "final_loss" in d.files:
            final_losses.append(float(d["final_loss"][0]))
    if final_losses:
        print(f"  GFlowNet final TB loss: mean={np.mean(final_losses):.4f}, "
              f"max={np.max(final_losses):.4f}, "
              f"median={np.median(final_losses):.4f}")

    # Fit PCE
    print(f"\nFitting PCE (degree={pce_degree}, basis=hermite)...")
    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(n_steps):
        tsurr.fit_step(s, mu_tr, tr_pol[s])

    # Sobol indices
    sobol = tsurr.sobol_all_steps()
    component_names = (dataset or {}).get("component_names",
                                          [f"step_{s}" for s in range(n_steps)])
    print("\nSOBOL SENSITIVITY INDICES (first-order, per PC)")
    for s in range(n_steps):
        fo = sobol[s]["first_order"]   # (K-1, pca_dim)
        var = sobol[s]["variance"]     # (K-1,)
        name = component_names[s] if s < len(component_names) else f"step_{s}"
        mean_fo = fo.mean(axis=0)      # average over ALR components
        print(f"  Step {s} ({name:12s}): "
              f"S_PC1={mean_fo[0]:.3f}, S_PC2={mean_fo[1]:.3f}, "
              f"total_var={var.mean():.4f}")

    # Accessible narrative interpretation
    print()
    print(tsurr.summarise_sobol(
        step_labels={s: component_names[s] if s < len(component_names)
                     else f"step_{s}" for s in range(n_steps)},
    ))

    # Experimental design recommendation
    recommendation = recommend_experimental_design(sobol, component_names, pca_dim)
    print(recommendation["text"])

    # KS test battery
    print("\nRunning KS test battery...")
    ks_results = run_ks_battery(tsurr, te_pol, n_mc=n_mc)
    for s in range(n_steps):
        name = component_names[s] if s < len(component_names) else f"step_{s}"
        fr = ks_results[s]["fraction_pass"]
        print(f"  Step {s} ({name:12s}): {ks_results[s]['n_pass']}/{ks_results[s]['n_total']} "
              f"actions pass KS ({fr:.1%})")

    # Calibration (marginal and joint)
    from core.pce_surrogate import calibration_coverage_joint
    print("\nCalibration coverage:")
    mc_pols = tsurr.sample_trajectory_policies(n_mc)
    cal_marginal = {}
    cal_joint = {}
    for s in range(n_steps):
        cal_marginal[s] = calibration_coverage(mc_pols[s], te_pol[s])
        cal_joint[s] = calibration_coverage_joint(mc_pols[s], te_pol[s])
        name = component_names[s] if s < len(component_names) else f"step_{s}"
        print(f"  Step {s} ({name:12s}) marginal: "
              + ", ".join(f"{100*lv:.0f}%->cov={cal_marginal[s][lv]:.2f}" for lv in sorted(cal_marginal[s])))
        print(f"  {'':15s}    joint: "
              + ", ".join(f"{100*lv:.0f}%->cov={cal_joint[s][lv]:.2f}" for lv in sorted(cal_joint[s])))

    # Distributional analysis: bimodality and surrogate vs empirical figures
    step_labels = {s: component_names[s] if s < len(component_names)
                   else f"step_{s}" for s in range(n_steps)}
    stop_action = all_policies[0].shape[1] - 1  # last action = stop/EOS
    bimodality, narrative = analyse_trajectory_bimodality(
        mc_pols, action_idx=stop_action, step_labels=step_labels,
    )
    print(narrative)
    plot_bimodality_panel(
        mc_pols, action_idx=stop_action,
        step_labels=step_labels, action_name="stop",
        title="Buchwald-Hartwig: stop-action distributional structure",
        empirical_samples=te_pol,
        output_path=os.path.join(output_dir, "bimodality_panel.pdf"),
    )
    plot_surrogate_comparison(
        mc_pols, te_pol, action_idx=0,
        step_labels=step_labels, action_name="first action",
        title="Buchwald-Hartwig: PCE surrogate vs empirical ensemble",
        output_path=os.path.join(output_dir, "surrogate_comparison.pdf"),
    )

    # Required ensemble size (Theorem A)
    L_required = tsurr.required_ensemble_size(target_sobol_error=0.05, confidence=0.95)
    print(f"\nTheorem A: required ensemble size for eps=0.05 Sobol accuracy: {L_required}")
    print(f"  (Used {n_train} training members)")

    # Compile results
    results = {
        "sobol": {str(s): {k: v.tolist() for k, v in v.items()} for s, v in sobol.items()},
        "ks_results": {str(s): {
            "fraction_pass": ks_results[s]["fraction_pass"],
            "n_pass": ks_results[s]["n_pass"],
            "n_total": ks_results[s]["n_total"],
        } for s in range(n_steps)},
        "pca_explained_var": exp_var,
        "pca_variance_per_component": pca.explained_variance_ratio_.tolist(),
        "calibration_marginal": {str(s): {str(k): v for k, v in cal_marginal[s].items()}
                                 for s in range(n_steps)},
        "calibration_joint": {str(s): {str(k): v for k, v in cal_joint[s].items()}
                              for s in range(n_steps)},
        "pca_dim": pca_dim,
        "n_train": n_train,
        "n_test": n_test,
        "pce_degree": pce_degree,
        "theorem_A_L_required": L_required,
        "component_names": component_names,
        "dataset_source": (dataset or {}).get("source", "unknown"),
        "recommendation": recommendation,
    }

    def _jsonify(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_jsonify(v) for v in obj]
        return obj

    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(_jsonify(results), f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Sequential mode (local, all-in-one)
# ---------------------------------------------------------------------------

def run_buchwald_hartwig_sequential(
    n_train: int = 50,
    n_test: int = 100,
    pce_degree: int = 5,
    gfn_episodes: int = 3000,
    train_fraction: float = 0.3,
    pca_dim: int = 2,
    n_mc: int = 10000,
    device: str = "cpu",
    output_dir: str = "results/buchwald_hartwig",
    csv_path: str = None,
) -> dict:
    """Sequential (non-parallel) experiment runner for local testing."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70 + "\nBUCHWALD-HARTWIG (SEQUENTIAL)\n" + "=" * 70)

    ds = load_dataset(csv_path)
    # Use 500 random reference reactions for proxy PCA (stratified sample)
    rng = np.random.RandomState(0)
    ref_idx = rng.choice(len(ds["reactions"]), min(500, len(ds["reactions"])), replace=False)
    ref_rxns = ds["reactions"][ref_idx]

    # Train all proxies first to get PCA parameterisation
    print(f"\nTraining {n_train + n_test} proxies for PCA...")
    proxies = []
    for i in range(n_train + n_test):
        p = train_proxy(ds, fraction=train_fraction, seed=i, device=device)
        proxies.append(p)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_train+n_test} proxies done")

    # PCA on proxy outputs (fit on train only to avoid data leakage)
    pout = np.array([p.predict_yield(ref_rxns) for p in proxies])
    pca = PCA(n_components=pca_dim)
    mu_raw_tr = pca.fit_transform(pout[:n_train])
    mu_raw_te = pca.transform(pout[n_train:])
    mu = np.concatenate([mu_raw_tr, mu_raw_te], axis=0)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Select reference trajectory using proxy 0 (representative)
    ref_traj = select_reference_trajectory(ds, proxies[0])

    # Train GFlowNets and extract policies
    print(f"\nTraining {n_train + n_test} GFlowNets...")
    all_policies = {s: [] for s in range(4)}
    for i, proxy in enumerate(proxies):
        gfn = ReactionGFlowNet(ds["n_components"])
        gfn = train_gflownet(gfn, proxy, n_ep=gfn_episodes, device=device,
                              temp=4.0, print_interval=0)
        pols = extract_policy(gfn, ref_traj)
        for s in range(4):
            all_policies[s].append(pols[s])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_train+n_test} GFlowNets done")

    for s in range(4):
        all_policies[s] = np.array(all_policies[s])

    mu_tr, mu_te = mu[:n_train], mu[n_train:]
    tr_pol = {s: all_policies[s][:n_train] for s in range(4)}
    te_pol = {s: all_policies[s][n_train:] for s in range(4)}

    # PCE
    print(f"\nFitting PCE (degree={pce_degree})...")
    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(4):
        tsurr.fit_step(s, mu_tr, tr_pol[s])

    sobol = tsurr.sobol_all_steps()
    print("\nSOBOL SENSITIVITY INDICES")
    for s in range(4):
        name = ds["component_names"][s]
        fo = sobol[s]["first_order"].mean(axis=0)
        print(f"  Step {s} ({name:12s}): S_PC1={fo[0]:.3f}, S_PC2={fo[1]:.3f}")

    # Experimental design recommendation
    recommendation = recommend_experimental_design(sobol, ds["component_names"], pca_dim)
    print(recommendation["text"])

    ks = run_ks_battery(tsurr, te_pol, n_mc=n_mc)

    results = {
        "sobol": {str(s): {k: v.tolist() for k, v in v.items()} for s, v in sobol.items()},
        "pca_explained_var": float(pca.explained_variance_ratio_.sum()),
        "ks_fraction_pass": {str(s): ks[s]["fraction_pass"] for s in range(4)},
        "ref_traj": ref_traj,
        "component_names": ds["component_names"],
        "dataset_source": ds["source"],
        "theorem_A_L_required": tsurr.required_ensemble_size(0.05),
        "recommendation": recommendation,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}/results.json")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buchwald-Hartwig GFlowNet + PCE UQ")
    parser.add_argument("--mode", choices=["sequential", "analyze"],
                        default="sequential",
                        help="sequential: train+analyze in one process. "
                             "analyze: collect from saved member files (after SLURM array).")
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--pce_degree", type=int, default=5)
    parser.add_argument("--gfn_episodes", type=int, default=3000)
    parser.add_argument("--train_fraction", type=float, default=0.3)
    parser.add_argument("--pca_dim", type=int, default=2)
    parser.add_argument("--output_dir", default="results/buchwald_hartwig")
    parser.add_argument("--csv_path", default=None,
                        help="Path to Dreher_and_Doyle_input_data.csv")
    parser.add_argument("--device", default="cuda" if (HAS_TORCH and
                        __import__("torch").cuda.is_available()) else "cpu")
    args = parser.parse_args()

    if not HAS_TORCH:
        sys.exit(1)

    if args.mode == "sequential":
        run_buchwald_hartwig_sequential(
            n_train=args.n_train, n_test=args.n_test,
            pce_degree=args.pce_degree, gfn_episodes=args.gfn_episodes,
            train_fraction=args.train_fraction, pca_dim=args.pca_dim,
            device=args.device, output_dir=args.output_dir,
            csv_path=args.csv_path,
        )
    elif args.mode == "analyze":
        ds = load_dataset(args.csv_path)
        collect_and_analyze(
            output_dir=args.output_dir, n_train=args.n_train, n_test=args.n_test,
            pce_degree=args.pce_degree, pca_dim=args.pca_dim, dataset=ds,
        )
