#!/usr/bin/env python3
"""PCA vs Kernel PCA vs β-VAE vs Normalizing Flow embedding ablation.

Compares four dimensionality reduction / Gaussianization strategies for
the PCE surrogate on the Buchwald-Hartwig task:
  1. Linear PCA (baseline) — orthogonal, Gaussian-compatible, analytical Sobol
  2. Kernel PCA (RBF) — nonlinear, uncorrelated by construction, Sobol if Gaussian
  3. β-VAE — nonlinear, approximately independent at high β, Sobol approximate
  4. PCA + Normalizing Flow (RealNVP) — PCA for dimensionality reduction, then
     learned invertible map to exact N(0, I). Hermite basis is optimal, analytical
     Sobol is exactly valid by construction.

For each embedding, fits PCE surrogates and reports:
  - Reconstruction error / variance explained
  - Latent independence (max |Pearson ρ|)
  - Gaussianity of latents (Shapiro-Wilk)
  - PCE calibration coverage (marginal)
  - Sobol index sum (ΣS_T ≈ 1 indicates near-additive structure)
  - Analytical Sobol validity assessment
"""
import sys, json, os, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

from core.pce_surrogate import (
    TrajectoryPCESurrogate, calibration_coverage
)


# ---------------------------------------------------------------------------
# β-VAE
# ---------------------------------------------------------------------------

class BetaVAE(nn.Module):
    """Simple β-VAE for low-dimensional embedding of proxy output vectors."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def train_beta_vae(data: np.ndarray, latent_dim: int, beta: float = 4.0,
                   n_epochs: int = 2000, lr: float = 1e-3, seed: int = 0):
    """Train a β-VAE and return encoder mappings for train/test splits."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = data.shape[1]
    model = BetaVAE(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x = torch.tensor(data, dtype=torch.float32)

    losses = []
    for epoch in range(n_epochs):
        model.train()
        recon, mu, logvar = model(x)
        recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        mu_out, _ = model.encode(x)
    return mu_out.numpy(), model, losses


# ---------------------------------------------------------------------------
# Normalizing Flow (RealNVP-style)
# ---------------------------------------------------------------------------

class AffineCouplingLayer(nn.Module):
    """Single affine coupling layer for RealNVP.

    Splits input into two halves. First half is unchanged; second half is
    affine-transformed conditioned on the first half.
    """

    def __init__(self, dim: int, hidden_dim: int = 64, mask_even: bool = True):
        super().__init__()
        self.dim = dim
        # Mask: which dims are "unchanged" vs "transformed"
        mask = torch.zeros(dim)
        if mask_even:
            mask[::2] = 1.0
        else:
            mask[1::2] = 1.0
        self.register_buffer("mask", mask)

        # Scale and translate networks
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        """Forward: data -> latent. Returns (z, log_det_J)."""
        x_masked = x * self.mask
        s = self.scale_net(x_masked) * (1 - self.mask)
        t = self.translate_net(x_masked) * (1 - self.mask)
        # Clamp scale for stability
        s = torch.clamp(s, -5.0, 5.0)
        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return z, log_det

    def inverse(self, z):
        """Inverse: latent -> data."""
        z_masked = z * self.mask
        s = self.scale_net(z_masked) * (1 - self.mask)
        t = self.translate_net(z_masked) * (1 - self.mask)
        s = torch.clamp(s, -5.0, 5.0)
        x = z_masked + (1 - self.mask) * (z - t) * torch.exp(-s)
        return x


class RealNVPFlow(nn.Module):
    """RealNVP normalizing flow: learns bijection from data to N(0, I)."""

    def __init__(self, dim: int, n_layers: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        layers = []
        for i in range(n_layers):
            layers.append(AffineCouplingLayer(dim, hidden_dim, mask_even=(i % 2 == 0)))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """Transform data x to latent z ~ N(0, I). Returns (z, log_prob)."""
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det
        # log p(x) = log p_z(z) + log |det dz/dx|
        log_pz = -0.5 * (z.pow(2).sum(dim=-1) + self.dim * np.log(2 * np.pi))
        log_px = log_pz + log_det_total
        return z, log_px

    def inverse(self, z):
        """Transform latent z back to data x."""
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x


def train_normalizing_flow(data: np.ndarray, n_layers: int = 8,
                            hidden_dim: int = 64, n_epochs: int = 3000,
                            lr: float = 5e-4, seed: int = 0):
    """Train a RealNVP flow to Gaussianize d-dimensional input data.

    Args:
        data: (n, d) array of PCA-transformed, standardized samples.
        n_layers: number of coupling layers.
        hidden_dim: hidden units per coupling network.
        n_epochs: training epochs.
        lr: learning rate.
        seed: random seed.

    Returns:
        z_out: (n, d) flow-transformed samples (should be ~ N(0, I)).
        model: trained RealNVPFlow.
        losses: list of negative log-likelihood per epoch.
    """
    torch.manual_seed(seed)
    dim = data.shape[1]
    model = RealNVPFlow(dim, n_layers=n_layers, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x = torch.tensor(data, dtype=torch.float32)

    losses = []
    for epoch in range(n_epochs):
        model.train()
        _, log_px = model(x)
        loss = -log_px.mean()  # negative log-likelihood
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        z_out, _ = model(x)
    return z_out.numpy(), model, losses


def flow_transform(model: RealNVPFlow, data: np.ndarray) -> np.ndarray:
    """Apply trained flow to new data (e.g., test set)."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32)
        z, _ = model(x)
    return z.numpy()


# ---------------------------------------------------------------------------
# Independence & Gaussianity diagnostics
# ---------------------------------------------------------------------------

def check_independence(mu: np.ndarray) -> dict:
    """Check pairwise independence of latent dimensions."""
    d = mu.shape[1]
    max_abs_rho = 0.0
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            rho, _ = stats.pearsonr(mu[:, i], mu[:, j])
            pairs.append((i, j, float(rho)))
            max_abs_rho = max(max_abs_rho, abs(rho))
    return {
        "max_abs_pearson": round(max_abs_rho, 4),
        "pairs": pairs,
        "independent": max_abs_rho < 0.15,  # conservative threshold
    }


def check_gaussianity(mu: np.ndarray) -> dict:
    """Shapiro-Wilk test on each latent dimension."""
    d = mu.shape[1]
    results = []
    for i in range(d):
        stat, p = stats.shapiro(mu[:, i])
        results.append({
            "dim": i,
            "shapiro_stat": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "gaussian": p > 0.05,
        })
    return {
        "per_dim": results,
        "all_gaussian": all(r["gaussian"] for r in results),
    }


def reconstruction_error(original: np.ndarray, mu: np.ndarray,
                         method: str, **kwargs) -> float:
    """Compute variance explained by the embedding (R² in proxy space).

    For PCA/KernelPCA: inverse_transform and compare.
    For VAE: decode and compare.
    """
    if method == "pca":
        pca_obj = kwargs["pca_obj"]
        scaler_obj = kwargs["scaler_obj"]
        recon = scaler_obj.inverse_transform(
            pca_obj.inverse_transform(
                scaler_obj.transform(mu) if hasattr(scaler_obj, 'transform') else mu
            )
        ) if scaler_obj else pca_obj.inverse_transform(mu)
        # Actually simpler: use explained_variance_ratio
        return float(pca_obj.explained_variance_ratio_.sum())

    elif method == "kernel_pca":
        # KernelPCA with inverse_transform
        kpca_obj = kwargs["kpca_obj"]
        try:
            recon = kpca_obj.inverse_transform(mu)
            ss_res = np.sum((original - recon) ** 2)
            ss_tot = np.sum((original - original.mean(axis=0)) ** 2)
            return round(float(1.0 - ss_res / ss_tot), 4)
        except Exception:
            return float("nan")

    elif method == "vae":
        model = kwargs["model"]
        model.eval()
        with torch.no_grad():
            z = torch.tensor(mu, dtype=torch.float32)
            recon = model.decode(z).numpy()
        ss_res = np.sum((original - recon) ** 2)
        ss_tot = np.sum((original - original.mean(axis=0)) ** 2)
        return round(float(1.0 - ss_res / ss_tot), 4)

    return float("nan")


# ---------------------------------------------------------------------------
# Main ablation
# ---------------------------------------------------------------------------

def run_ablation(
    n_train: int = 50,
    n_test: int = 100,
    pca_dim: int = 5,
    pce_degree: int = 3,
    beta: float = 4.0,
    gfn_episodes: int = 3000,
    seed: int = 0,
):
    """Run PCA vs KernelPCA vs β-VAE ablation on Buchwald-Hartwig."""
    from experiments.buchwald_hartwig.run_experiment import (
        load_dataset, train_proxy, ReactionGFlowNet, train_gflownet,
        extract_policy, select_reference_trajectory,
    )

    output_dir = ROOT / "results" / "validation"
    os.makedirs(output_dir, exist_ok=True)

    n_total = n_train + n_test
    device = "cpu"

    # --- Load data and train proxies ---
    ds = load_dataset()
    rng = np.random.RandomState(seed)
    ref_idx = rng.choice(len(ds["reactions"]), min(500, len(ds["reactions"])), replace=False)
    ref_rxns = ds["reactions"][ref_idx]

    print("Training proxies...")
    proxies = []
    for i in range(n_total):
        p = train_proxy(ds, fraction=0.3, seed=i, device=device)
        proxies.append(p)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_total} proxies")

    pout = np.array([p.predict_yield(ref_rxns) for p in proxies])
    print(f"  Proxy output shape: {pout.shape}")

    # --- Train GFlowNets ---
    print("Training GFlowNets...")
    ref_traj = select_reference_trajectory(ds, proxies[0])
    all_policies = {s: [] for s in range(4)}
    for i, proxy in enumerate(proxies):
        gfn = ReactionGFlowNet(ds["n_components"])
        gfn = train_gflownet(gfn, proxy, n_ep=gfn_episodes, device=device, temp=4.0, print_interval=0)
        pols = extract_policy(gfn, ref_traj)
        for s in range(4):
            all_policies[s].append(pols[s])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_total} GFlowNets")

    for s in range(4):
        all_policies[s] = np.array(all_policies[s])

    tr_pol = {s: all_policies[s][:n_train] for s in range(4)}
    te_pol = {s: all_policies[s][n_train:] for s in range(4)}

    # =====================================================================
    # Embedding 1: Linear PCA
    # =====================================================================
    print("\n" + "=" * 60)
    print("EMBEDDING 1: Linear PCA")
    print("=" * 60)

    pca = PCA(n_components=pca_dim)
    mu_pca_tr = pca.fit_transform(pout[:n_train])
    mu_pca_te = pca.transform(pout[n_train:])
    scaler_pca = StandardScaler()
    mu_pca_tr = scaler_pca.fit_transform(mu_pca_tr)
    mu_pca_te = scaler_pca.transform(mu_pca_te)

    pca_var = float(pca.explained_variance_ratio_.sum())
    pca_indep = check_independence(mu_pca_tr)
    pca_gauss = check_gaussianity(mu_pca_tr)

    print(f"  Variance explained: {pca_var:.3f}")
    print(f"  Max |ρ|: {pca_indep['max_abs_pearson']:.4f} ({'PASS' if pca_indep['independent'] else 'FAIL'})")
    print(f"  Gaussian: {pca_gauss['all_gaussian']}")

    # Fit PCE
    tsurr_pca = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(4):
        tsurr_pca.fit_step(s, mu_pca_tr, tr_pol[s])
    cal_pca = {s: calibration_coverage(tsurr_pca, s, mu_pca_te, te_pol[s])
               for s in range(4)}
    sobol_pca = tsurr_pca.sobol_all_steps()
    sobol_ci_pca = {s: tsurr_pca.sobol_indices_with_ci(s, mu_pca_tr, tr_pol[s])
                    for s in range(4)}

    print(f"  Calibration @95%: {[round(cal_pca[s].get(0.95, cal_pca[s].get('0.95', 0)), 3) for s in range(4)]}")

    # =====================================================================
    # Embedding 2: Kernel PCA (RBF)
    # =====================================================================
    print("\n" + "=" * 60)
    print("EMBEDDING 2: Kernel PCA (RBF)")
    print("=" * 60)

    kpca = KernelPCA(n_components=pca_dim, kernel="rbf", gamma=None,
                     fit_inverse_transform=True, random_state=seed)
    mu_kpca_tr = kpca.fit_transform(pout[:n_train])
    mu_kpca_te = kpca.transform(pout[n_train:])
    scaler_kpca = StandardScaler()
    mu_kpca_tr = scaler_kpca.fit_transform(mu_kpca_tr)
    mu_kpca_te = scaler_kpca.transform(mu_kpca_te)

    kpca_r2 = reconstruction_error(pout[:n_train], kpca.inverse_transform(
        scaler_kpca.inverse_transform(mu_kpca_tr)), "kernel_pca", kpca_obj=kpca)
    kpca_indep = check_independence(mu_kpca_tr)
    kpca_gauss = check_gaussianity(mu_kpca_tr)

    print(f"  Reconstruction R²: {kpca_r2}")
    print(f"  Max |ρ|: {kpca_indep['max_abs_pearson']:.4f} ({'PASS' if kpca_indep['independent'] else 'FAIL'})")
    print(f"  Gaussian: {kpca_gauss['all_gaussian']}")

    # Fit PCE — use Hermite if Gaussian, note if not
    basis_kpca = "hermite" if kpca_gauss["all_gaussian"] else "hermite"
    tsurr_kpca = TrajectoryPCESurrogate(degree=pce_degree, basis=basis_kpca)
    for s in range(4):
        tsurr_kpca.fit_step(s, mu_kpca_tr, tr_pol[s])
    cal_kpca = {s: calibration_coverage(tsurr_kpca, s, mu_kpca_te, te_pol[s])
                for s in range(4)}
    sobol_kpca = tsurr_kpca.sobol_all_steps()

    print(f"  Calibration @95%: {[round(cal_kpca[s].get(0.95, cal_kpca[s].get('0.95', 0)), 3) for s in range(4)]}")

    # =====================================================================
    # Embedding 3: β-VAE
    # =====================================================================
    print("\n" + "=" * 60)
    print(f"EMBEDDING 3: β-VAE (β={beta})")
    print("=" * 60)

    # Standardize proxy outputs before VAE training
    pout_scaler = StandardScaler()
    pout_std = pout_scaler.fit_transform(pout)

    mu_vae_all, vae_model, vae_losses = train_beta_vae(
        pout_std[:n_train], latent_dim=pca_dim, beta=beta,
        n_epochs=3000, seed=seed,
    )
    # Encode test set
    vae_model.eval()
    with torch.no_grad():
        mu_vae_te_raw, _ = vae_model.encode(
            torch.tensor(pout_std[n_train:], dtype=torch.float32)
        )
        mu_vae_te_raw = mu_vae_te_raw.numpy()

    # Standardize VAE latents for PCE
    scaler_vae = StandardScaler()
    mu_vae_tr = scaler_vae.fit_transform(mu_vae_all)
    mu_vae_te = scaler_vae.transform(mu_vae_te_raw)

    vae_r2 = reconstruction_error(pout_std[:n_train], mu_vae_all, "vae", model=vae_model)
    vae_indep = check_independence(mu_vae_tr)
    vae_gauss = check_gaussianity(mu_vae_tr)

    print(f"  Reconstruction R²: {vae_r2}")
    print(f"  Max |ρ|: {vae_indep['max_abs_pearson']:.4f} ({'PASS' if vae_indep['independent'] else 'FAIL'})")
    print(f"  Gaussian: {vae_gauss['all_gaussian']}")
    print(f"  Final VAE loss: {vae_losses[-1]:.1f}")

    # Fit PCE
    tsurr_vae = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(4):
        tsurr_vae.fit_step(s, mu_vae_tr, tr_pol[s])
    cal_vae = {s: calibration_coverage(tsurr_vae, s, mu_vae_te, te_pol[s])
               for s in range(4)}
    sobol_vae = tsurr_vae.sobol_all_steps()

    print(f"  Calibration @95%: {[round(cal_vae[s].get(0.95, cal_vae[s].get('0.95', 0)), 3) for s in range(4)]}")

    # =====================================================================
    # Embedding 4: PCA + Normalizing Flow (Gaussianization)
    # =====================================================================
    print("\n" + "=" * 60)
    print("EMBEDDING 4: PCA + Normalizing Flow (RealNVP)")
    print("=" * 60)

    # Start from raw (unstandardized) PCA components
    pca_nf = PCA(n_components=pca_dim)
    mu_pca_nf_tr_raw = pca_nf.fit_transform(pout[:n_train])
    mu_pca_nf_te_raw = pca_nf.transform(pout[n_train:])
    # Standardize before flow (helps training stability)
    scaler_nf_pre = StandardScaler()
    mu_pca_nf_tr_pre = scaler_nf_pre.fit_transform(mu_pca_nf_tr_raw)
    mu_pca_nf_te_pre = scaler_nf_pre.transform(mu_pca_nf_te_raw)

    print("  Training normalizing flow to Gaussianize PCA components...")
    mu_nf_tr, nf_model, nf_losses = train_normalizing_flow(
        mu_pca_nf_tr_pre, n_layers=8, hidden_dim=64, n_epochs=3000, seed=seed,
    )
    mu_nf_te = flow_transform(nf_model, mu_pca_nf_te_pre)

    nf_var = float(pca_nf.explained_variance_ratio_.sum())  # same as PCA
    nf_indep = check_independence(mu_nf_tr)
    nf_gauss = check_gaussianity(mu_nf_tr)

    print(f"  PCA variance explained: {nf_var:.3f} (same linear PCA first stage)")
    print(f"  Max |ρ| after flow: {nf_indep['max_abs_pearson']:.4f} ({'PASS' if nf_indep['independent'] else 'FAIL'})")
    print(f"  Gaussian after flow: {nf_gauss['all_gaussian']}")
    for r in nf_gauss["per_dim"]:
        print(f"    Dim {r['dim']}: Shapiro p={r['p_value']:.4f} ({'Gaussian' if r['gaussian'] else 'NON-Gaussian'})")
    print(f"  Final NF NLL: {nf_losses[-1]:.3f}")

    # Fit PCE on flow-transformed variables
    tsurr_nf = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(4):
        tsurr_nf.fit_step(s, mu_nf_tr, tr_pol[s])
    cal_nf = {s: calibration_coverage(tsurr_nf, s, mu_nf_te, te_pol[s])
              for s in range(4)}
    sobol_nf = tsurr_nf.sobol_all_steps()

    print(f"  Calibration @95%: {[round(cal_nf[s].get(0.95, cal_nf[s].get('0.95', 0)), 3) for s in range(4)]}")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY: PCA vs Kernel PCA vs β-VAE vs PCA+Flow")
    print("=" * 60)

    def mean_cal95(cal_dict):
        vals = []
        for s in range(4):
            c = cal_dict[s]
            vals.append(c.get(0.95, c.get("0.95", 0)))
        return round(float(np.mean(vals)), 3)

    def mean_sobol_sum(sobol_dict):
        sums = []
        for s in range(4):
            to = sobol_dict[s]["total_order"]
            sums.append(float(to.sum(axis=1).mean()))
        return round(float(np.mean(sums)), 3)

    header = f"{'Method':15s} {'Var/R²':>8s} {'Max|ρ|':>8s} {'Gauss?':>8s} {'Indep?':>8s} {'Cal@95':>8s} {'ΣS_T':>8s}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Linear PCA", pca_var, pca_indep["max_abs_pearson"],
         pca_gauss["all_gaussian"], pca_indep["independent"],
         mean_cal95(cal_pca), mean_sobol_sum(sobol_pca)),
        ("Kernel PCA", kpca_r2 if isinstance(kpca_r2, float) else 0.0,
         kpca_indep["max_abs_pearson"],
         kpca_gauss["all_gaussian"], kpca_indep["independent"],
         mean_cal95(cal_kpca), mean_sobol_sum(sobol_kpca)),
        (f"β-VAE (β={beta})", vae_r2,
         vae_indep["max_abs_pearson"],
         vae_gauss["all_gaussian"], vae_indep["independent"],
         mean_cal95(cal_vae), mean_sobol_sum(sobol_vae)),
        ("PCA+Flow", nf_var, nf_indep["max_abs_pearson"],
         nf_gauss["all_gaussian"], nf_indep["independent"],
         mean_cal95(cal_nf), mean_sobol_sum(sobol_nf)),
    ]

    for name, var, rho, gauss, indep, cal, st in rows:
        g = "YES" if gauss else "NO"
        i = "YES" if indep else "NO"
        print(f"{name:15s} {var:8.3f} {rho:8.4f} {g:>8s} {i:>8s} {cal:8.3f} {st:8.3f}")

    # Sobol analytical validity assessment
    print("\nSOBOL VALIDITY ASSESSMENT:")
    for name, indep_result, gauss_result in [
        ("Linear PCA", pca_indep, pca_gauss),
        ("Kernel PCA", kpca_indep, kpca_gauss),
        (f"β-VAE (β={beta})", vae_indep, vae_gauss),
        ("PCA+Flow", nf_indep, nf_gauss),
    ]:
        if indep_result["independent"] and gauss_result["all_gaussian"]:
            verdict = "FULLY VALID — independent + Gaussian → analytical Sobol exact"
        elif indep_result["independent"]:
            verdict = "APPROXIMATELY VALID — independent but non-Gaussian → consider aPC basis"
        elif gauss_result["all_gaussian"]:
            verdict = "CAUTION — Gaussian but correlated → Sobol decomposition leaks across dims"
        else:
            verdict = "INVALID — correlated + non-Gaussian → use MC Sobol only"
        print(f"  {name:15s}: {verdict}")

    # Save results
    results = {
        "pca_dim": pca_dim,
        "pce_degree": pce_degree,
        "n_train": n_train,
        "n_test": n_test,
        "beta": beta,
        "linear_pca": {
            "variance_explained": pca_var,
            "max_abs_pearson": pca_indep["max_abs_pearson"],
            "all_gaussian": pca_gauss["all_gaussian"],
            "independent": pca_indep["independent"],
            "mean_cal_95": mean_cal95(cal_pca),
            "mean_sobol_sum": mean_sobol_sum(sobol_pca),
        },
        "kernel_pca": {
            "reconstruction_r2": kpca_r2 if isinstance(kpca_r2, float) else None,
            "max_abs_pearson": kpca_indep["max_abs_pearson"],
            "all_gaussian": kpca_gauss["all_gaussian"],
            "independent": kpca_indep["independent"],
            "mean_cal_95": mean_cal95(cal_kpca),
            "mean_sobol_sum": mean_sobol_sum(sobol_kpca),
        },
        "beta_vae": {
            "reconstruction_r2": vae_r2,
            "beta": beta,
            "max_abs_pearson": vae_indep["max_abs_pearson"],
            "all_gaussian": vae_gauss["all_gaussian"],
            "independent": vae_indep["independent"],
            "mean_cal_95": mean_cal95(cal_vae),
            "mean_sobol_sum": mean_sobol_sum(sobol_vae),
            "final_loss": round(vae_losses[-1], 1),
        },
        "pca_plus_flow": {
            "pca_variance_explained": nf_var,
            "max_abs_pearson": nf_indep["max_abs_pearson"],
            "all_gaussian": nf_gauss["all_gaussian"],
            "independent": nf_indep["independent"],
            "mean_cal_95": mean_cal95(cal_nf),
            "mean_sobol_sum": mean_sobol_sum(sobol_nf),
            "final_nll": round(nf_losses[-1], 3),
            "gaussianity_per_dim": nf_gauss["per_dim"],
        },
    }

    out_path = output_dir / "pca_vs_vae_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_ablation()
