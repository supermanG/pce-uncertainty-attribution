"""GP surrogate baseline for GFlowNet policy UQ comparison table.

Fits one GaussianProcessRegressor per ALR component per trajectory step.
Provides posterior mean + variance predictions and posterior samples.
Sobol indices are approximated via MC integration over GP posterior samples
(expensive; not analytical).

Interface mirrors PCESurrogate so run_baselines.py can use all three
surrogates interchangeably.
"""
import numpy as np
from typing import Dict, List, Optional

from core.pce_surrogate import alr_transform, alr_inverse


class GPSurrogate:
    """GP surrogate: one GP per ALR component.

    Kernel: Matern(nu=2.5) + WhiteKernel for noise.
    All K-1 GPs share the same kernel hyperparameters initialisation;
    each is optimised independently via marginal likelihood.

    For large K (e.g. Sachs with K=120), fitting K-1=119 GPs is
    tractable with small n_train (<=80) and low d (<=5).
    """

    def __init__(
        self,
        kernel=None,
        n_restarts_optimizer: int = 2,
        normalize_y: bool = True,
        random_state: int = 42,
        n_sobol_mc: int = 2000,
    ):
        self.kernel = kernel            # set in fit() if None
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.n_sobol_mc = n_sobol_mc

        self._gps: List = []            # one GPR per ALR component
        self.n_actions: Optional[int] = None
        self._d: Optional[int] = None
        self._mu_train: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _make_kernel(self):
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
        )

    # ------------------------------------------------------------------
    def fit(self, mu_train: np.ndarray, policies_train: np.ndarray) -> None:
        """Fit one GP per ALR component.

        Args:
            mu_train:       (L, d) PCA-standardised reward coordinates
            policies_train: (L, K) policy simplex rows
        """
        from sklearn.gaussian_process import GaussianProcessRegressor

        L, K = policies_train.shape
        self.n_actions = K
        self._d = mu_train.shape[1]
        self._mu_train = mu_train.copy()

        alr = alr_transform(policies_train)   # (L, K-1)
        self._gps = []

        kernel = self.kernel or self._make_kernel()
        for k in range(K - 1):
            gp = GaussianProcessRegressor(
                kernel=kernel.clone_with_theta(kernel.theta),
                n_restarts_optimizer=self.n_restarts_optimizer,
                normalize_y=self.normalize_y,
                random_state=self.random_state + k,
            )
            gp.fit(mu_train, alr[:, k])
            self._gps.append(gp)

    # ------------------------------------------------------------------
    def predict(self, mu_new: np.ndarray) -> np.ndarray:
        """Predict policy distributions using GP posterior mean.

        Returns:
            policies: (n_new, K) probability vectors
        """
        n = mu_new.shape[0]
        K_eff = len(self._gps)
        alr_pred = np.zeros((n, K_eff))
        for k, gp in enumerate(self._gps):
            alr_pred[:, k] = gp.predict(mu_new)
        return alr_inverse(alr_pred)

    # ------------------------------------------------------------------
    def predict_with_std(self, mu_new: np.ndarray):
        """Predict posterior mean and standard deviation per ALR component.

        Returns:
            mean_policies: (n, K)
            alr_std:       (n, K-1)  -- posterior std in ALR space
        """
        n = mu_new.shape[0]
        K_eff = len(self._gps)
        alr_mean = np.zeros((n, K_eff))
        alr_std  = np.zeros((n, K_eff))
        for k, gp in enumerate(self._gps):
            m, s = gp.predict(mu_new, return_std=True)
            alr_mean[:, k] = m
            alr_std[:, k]  = s
        return alr_inverse(alr_mean), alr_std

    # ------------------------------------------------------------------
    def sample_policies(
        self,
        n_samples: int,
        mu_distribution: str = "gaussian",
        mu_params: Optional[dict] = None,
    ) -> np.ndarray:
        """Draw n_samples policies from GP joint posterior.

        Samples ALR components independently from their marginal posteriors
        at randomly drawn mu values.  (Ignores inter-component covariance
        for tractability; sufficient for L1-norm MAE and coverage checks.)
        """
        mu_params = mu_params or {}
        d = self._d
        if mu_distribution == "gaussian":
            mean = mu_params.get("mean", np.zeros(d))
            std  = mu_params.get("std",  np.ones(d))
            mu_s = np.random.randn(n_samples, d) * std + mean
        else:
            low  = mu_params.get("low",  -np.ones(d))
            high = mu_params.get("high",  np.ones(d))
            mu_s = np.random.uniform(low, high, (n_samples, d))

        K_eff = len(self._gps)
        alr_samples = np.zeros((n_samples, K_eff))
        for k, gp in enumerate(self._gps):
            m, s = gp.predict(mu_s, return_std=True)
            alr_samples[:, k] = m + s * np.random.randn(n_samples)
        return alr_inverse(alr_samples)

    # ------------------------------------------------------------------
    def sobol_indices(self, n_mc: Optional[int] = None) -> Dict:
        """Approximate Sobol indices via MC integration over GP posterior.

        Draws n_mc mu samples from N(0,I), evaluates GP posterior mean,
        and computes sample-based ANOVA decomposition of the ALR outputs.

        This is O(n_mc * K * n_train) and is NOT analytical.
        Returns same dict format as PCESurrogate.sobol_indices().
        """
        if not self._gps:
            return None
        n_mc = n_mc or self.n_sobol_mc
        d = self._d
        K_eff = len(self._gps)

        # Draw mu from N(0,I) (standardised input space)
        rng = np.random.default_rng(0)
        mu_mc = rng.standard_normal((n_mc, d))

        # Predict GP posterior means at MC points
        alr_mc = np.zeros((n_mc, K_eff))
        for k, gp in enumerate(self._gps):
            alr_mc[:, k] = gp.predict(mu_mc)

        # Sample-based ANOVA: for each input dimension i,
        # variance of E[g|mu_i] estimated by conditioning MC samples on mu_i
        # using the Saltelli pick-freeze estimator.
        mu_mc2 = rng.standard_normal((n_mc, d))
        first_order = np.zeros((K_eff, d))
        total_order  = np.zeros((K_eff, d))
        variance = np.zeros(K_eff)

        alr_mc2 = np.zeros((n_mc, K_eff))
        for k, gp in enumerate(self._gps):
            alr_mc2[:, k] = gp.predict(mu_mc2)

        for k in range(K_eff):
            y1 = alr_mc[:, k]
            y2 = alr_mc2[:, k]
            V  = y1.var()
            variance[k] = V
            if V < 1e-15:
                continue
            for i in range(d):
                # Saltelli (2010) estimator using AB matrix
                mu_ab = mu_mc2.copy()
                mu_ab[:, i] = mu_mc[:, i]
                alr_ab = np.zeros(n_mc)
                alr_ab = self._gps[k].predict(mu_ab)
                # First-order: S_i = (1/n) sum y1*(y_ab - y2) / V
                first_order[k, i] = float(np.mean(y1 * (alr_ab - y2)) / V)
                # Total-order: ST_i = (1/2n) sum (y1 - y_ab)^2 / V
                total_order[k, i]  = float(np.mean((y1 - alr_ab) ** 2) / (2.0 * V))

        # Clip to [0, 1] (MC estimator can give small negatives)
        first_order = np.clip(first_order, 0.0, 1.0)
        total_order  = np.clip(total_order, 0.0, 1.0)

        return {"first_order": first_order, "total_order": total_order, "variance": variance}

    # ------------------------------------------------------------------
    def policy_mae(self, mu_test: np.ndarray, policies_test: np.ndarray) -> float:
        """Mean L1 distance between GP posterior mean predictions and test policies."""
        pred = self.predict(mu_test)
        return float(np.abs(pred - policies_test).mean())
