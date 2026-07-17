"""Embedding utilities for the PCE surrogate framework.

Provides drop-in dimensionality-reduction embeddings that can replace PCA
when fitting PCE surrogates. Each embedding implements fit / transform /
inverse_transform with a common interface.

Available embeddings:
  - BetaVAEEmbedding: nonlinear encoder producing approximately independent
    and Gaussian latents (exact Sobol with Hermite PCE).
  - RealNVPEmbedding: PCA + normalizing flow for exact Gaussianisation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# β-VAE
# ---------------------------------------------------------------------------

class BetaVAE(nn.Module):
    """Simple β-VAE for low-dimensional embedding of proxy output vectors."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class BetaVAEEmbedding:
    """Drop-in embedding using a β-VAE.

    Produces latent means that are approximately independent and Gaussian
    when β is sufficiently large (β >= 4 recommended), enabling exact
    analytical Sobol indices with a Hermite PCE basis.

    Usage::

        emb = BetaVAEEmbedding(latent_dim=5, beta=4.0)
        mu_train = emb.fit_transform(proxy_output_train)
        mu_test  = emb.transform(proxy_output_test)
        # Then pass mu_train / mu_test to TrajectoryPCESurrogate as usual.
    """

    def __init__(self, latent_dim: int = 5, beta: float = 4.0,
                 hidden_dim: int = 128, n_epochs: int = 3000,
                 lr: float = 1e-3, seed: int = 0):
        self.latent_dim = latent_dim
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.seed = seed
        self.model_ = None
        self.input_scaler_ = StandardScaler()
        self.latent_scaler_ = StandardScaler()
        self.losses_ = []

    def fit(self, X: np.ndarray) -> "BetaVAEEmbedding":
        """Fit the β-VAE on input data X (n_samples, input_dim)."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_std = self.input_scaler_.fit_transform(X)
        x_t = torch.tensor(X_std, dtype=torch.float32)

        model = BetaVAE(X.shape[1], self.latent_dim, self.hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        for _ in range(self.n_epochs):
            model.train()
            recon, mu, logvar = model(x_t)
            recon_loss = nn.functional.mse_loss(recon, x_t, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + self.beta * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.losses_.append(float(loss.item()))

        model.eval()
        self.model_ = model

        with torch.no_grad():
            mu_out, _ = model.encode(x_t)
        self.latent_scaler_.fit(mu_out.numpy())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode X to latent space (standardised)."""
        X_std = self.input_scaler_.transform(X)
        self.model_.eval()
        with torch.no_grad():
            mu, _ = self.model_.encode(
                torch.tensor(X_std, dtype=torch.float32)
            )
        return self.latent_scaler_.transform(mu.numpy())

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and return latent coordinates."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Decode from latent space back to input space."""
        z_raw = self.latent_scaler_.inverse_transform(Z)
        self.model_.eval()
        with torch.no_grad():
            recon_std = self.model_.decode(
                torch.tensor(z_raw, dtype=torch.float32)
            ).numpy()
        return self.input_scaler_.inverse_transform(recon_std)

    def reconstruction_r2(self, X: np.ndarray) -> float:
        """Compute R² between original and reconstructed data."""
        recon = self.inverse_transform(self.transform(X))
        ss_res = np.sum((X - recon) ** 2)
        ss_tot = np.sum((X - X.mean(axis=0)) ** 2)
        return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Diagnostics (shared by all embeddings)
# ---------------------------------------------------------------------------

def check_independence(mu: np.ndarray, threshold: float = 0.15) -> dict:
    """Check pairwise independence via Pearson correlation."""
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
        "independent": max_abs_rho < threshold,
    }


def check_gaussianity(mu: np.ndarray, alpha: float = 0.05) -> dict:
    """Shapiro-Wilk test on each latent dimension."""
    results = []
    for i in range(mu.shape[1]):
        stat, p = stats.shapiro(mu[:, i])
        results.append({
            "dim": i,
            "shapiro_stat": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "gaussian": p > alpha,
        })
    return {
        "per_dim": results,
        "all_gaussian": all(r["gaussian"] for r in results),
    }
