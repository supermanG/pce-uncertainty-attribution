"""MLP surrogate baseline for GFlowNet policy UQ comparison table.

Fits a multi-layer perceptron from mu (PCA reward coordinates) to
ALR-transformed policies.  No analytical Sobol indices; sampling is
deterministic (mean prediction at queried mu values).

Interface mirrors PCESurrogate so run_baselines.py can use all three
surrogates interchangeably.
"""
import time
import numpy as np
from typing import Dict, Optional

from core.pce_surrogate import alr_transform, alr_inverse


class MLPSurrogate:
    """MLP surrogate: mu -> ALR policy via multi-output MLP regression.

    Uses scikit-learn MLPRegressor.  A single MLPRegressor handles all
    K-1 ALR outputs simultaneously (multi-output regression).
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 64),
        max_iter: int = 1000,
        random_state: int = 42,
        alpha: float = 1e-3,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self._model = None
        self.n_actions: Optional[int] = None
        self._d: Optional[int] = None
        self._mu_std: Optional[np.ndarray] = None
        self._mu_mean: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, mu_train: np.ndarray, policies_train: np.ndarray) -> None:
        """Fit MLP on training ensemble.

        Args:
            mu_train:       (L, d) PCA-standardised reward coordinates
            policies_train: (L, K) policy simplex rows
        """
        from sklearn.neural_network import MLPRegressor

        L, K = policies_train.shape
        self.n_actions = K
        self._d = mu_train.shape[1]
        # Store normalisation statistics (mu is already standardised by caller
        # but we keep this as a no-op for API consistency)
        self._mu_mean = mu_train.mean(axis=0)
        self._mu_std  = mu_train.std(axis=0).clip(min=1e-8)

        alr = alr_transform(policies_train)   # (L, K-1)

        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=min(0.15, 3.0 / L),
            n_iter_no_change=30,
            tol=1e-5,
        )
        self._model.fit(mu_train, alr)

    # ------------------------------------------------------------------
    def predict(self, mu_new: np.ndarray) -> np.ndarray:
        """Predict policy distributions (mean MLP output).

        Returns:
            policies: (n_new, K) probability vectors
        """
        alr_pred = self._model.predict(mu_new)   # (n, K-1)
        if alr_pred.ndim == 1:
            alr_pred = alr_pred.reshape(1, -1)
        return alr_inverse(alr_pred)

    # ------------------------------------------------------------------
    def sample_policies(
        self,
        n_samples: int,
        mu_distribution: str = "gaussian",
        mu_params: Optional[dict] = None,
    ) -> np.ndarray:
        """Draw n_samples policies by evaluating MLP at random mu values.

        Note: MLP gives a deterministic prediction -- no epistemic uncertainty
        is captured per sample.  The spread comes purely from variation in mu.
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
        return self.predict(mu_s)

    # ------------------------------------------------------------------
    def sobol_indices(self) -> None:
        """MLP has no analytical Sobol indices (returns None)."""
        return None

    # ------------------------------------------------------------------
    def policy_mae(self, mu_test: np.ndarray, policies_test: np.ndarray) -> float:
        """Mean L1 distance between MLP predictions and test ensemble policies."""
        pred = self.predict(mu_test)
        return float(np.abs(pred - policies_test).mean())
