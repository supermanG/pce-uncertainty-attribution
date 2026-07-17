"""PCE Surrogate for GFlowNet Policy UQ -- Core engine.

Overview (for readers unfamiliar with UQ / Sobol analysis)
----------------------------------------------------------
A GFlowNet learns a policy -- a distribution over actions at each step of a
sequential decision.  When the *reward function* is uncertain (e.g. because
a reward model was trained on limited data), the policy inherits that
uncertainty: different plausible rewards lead to different policies.

This module answers the question:

    "Which aspects of reward uncertainty matter most for which decisions?"

It does so in three steps:

  1. **Dimensionality reduction (PCA, not VAE).**  The space of plausible
     reward functions is high-dimensional, so we project it onto a small
     number of principal components (mu_1, mu_2, ..., mu_d).  PCA is used
     *by design*: it produces uncorrelated components that, under the
     Gaussian assumption matched by the Hermite polynomial basis, are also
     statistically independent.  This independence is a prerequisite for
     step 3.  A VAE would give a nonlinear, potentially correlated latent
     space, invalidating the analytical Sobol decomposition below.

  2. **Polynomial chaos expansion (PCE).**  For each trajectory step and
     each action, we fit a polynomial that maps the reward parameters
     (mu_1, ..., mu_d) to the log-odds of choosing that action.  Because
     the polynomial basis is *orthonormal* with respect to the input
     distribution, the squared coefficients directly partition the output
     variance by input dimension -- no sampling required.

  3. **Sobol sensitivity indices.**  A Sobol index S_i for input dimension
     i tells you: "What fraction of the total policy uncertainty at this
     step is caused by uncertainty in mu_i alone?"  An index near 1 means
     that component of reward uncertainty is the dominant driver of the
     policy decision; near 0 means it hardly matters.  These are read off
     analytically from the PCE coefficients, which is why step 1 must use
     PCA (independent inputs) rather than a VAE.

Mathematical convention (consistent with paper Methods, Eq. 2-4):
  - Additive log-ratio (ALR) transform with reference action K-1 (last):
        y_k = log(p_k / p_{K-1}),   k = 0, ..., K-2
  - One PCE fitted per ALR component; reference action has identically-zero logit
  - Probabilities recovered via softmax([y_0, ..., y_{K-2}, 0])
  - Sobol indices computed from PCE coefficients for each ALR component

Note on PCA vs. VAE (historical context)
-----------------------------------------
An earlier version of this work used a VAE for dimensionality reduction.
The shift to PCA is deliberate: the analytical Sobol formula

    S_i = sum_{j: only dim i active} c_j^2  /  sum_{j != 0} c_j^2

holds exactly only when the input dimensions are *independent*.  PCA
components are uncorrelated by construction, and under a Gaussian input
model they are also independent -- matching the Hermite basis assumption.
A VAE's latent space offers neither guarantee: the approximate posterior
q(z|x) retains residual correlations, and the nonlinear decoder entangles
the dimensions.  With correlated inputs one would need the Kucherenko
(2012) extension, which requires Monte Carlo estimation and loses the
computational advantage of the coefficient-based formula.
"""
import math
import numpy as np
from itertools import product as cartesian_product
from typing import Optional, Tuple, Dict, List


# ---------------------------------------------------------------------------
# Univariate basis functions (orthonormal w.r.t. target measure)
# ---------------------------------------------------------------------------

def hermite_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Probabilist's Hermite polynomials He_n, normalised to unit L^2(N(0,1)) norm.

    He_0=1, He_1=x, He_n = x*He_{n-1} - (n-1)*He_{n-2}.
    Orthonormal version: psi_n = He_n / sqrt(n!).
    """
    He = [np.ones_like(x), x]
    for n in range(2, degree + 1):
        He.append(x * He[-1] - (n - 1) * He[-2])
    norms = [np.sqrt(float(math.factorial(n))) for n in range(degree + 1)]
    return np.array([He[n] / norms[n] for n in range(degree + 1)])


def legendre_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Legendre polynomials P_n on [-1,1], normalised to unit L^2(Uniform) norm.

    Orthonormal version: psi_n = P_n * sqrt(2n+1).
    """
    P = [np.ones_like(x), x]
    for n in range(2, degree + 1):
        P.append(((2 * n - 1) * x * P[-1] - (n - 1) * P[-2]) / n)
    norms = [1.0 / np.sqrt(2.0 * n + 1.0) for n in range(degree + 1)]
    return np.array([P[n] / norms[n] for n in range(degree + 1)])


# ---------------------------------------------------------------------------
# Multi-index bookkeeping
# ---------------------------------------------------------------------------

def _multi_indices(d: int, p: int) -> np.ndarray:
    """All multi-indices j in N^d with |j|_1 = sum(j) <= p, sorted by total degree.

    The constant term [0,...,0] is always at position 0.
    """
    if d == 1:
        return np.arange(p + 1).reshape(-1, 1)
    indices = []
    for total in range(p + 1):
        for combo in cartesian_product(range(total + 1), repeat=d):
            if sum(combo) == total:
                indices.append(combo)
    return np.array(indices)


def build_design_matrix(
    mu_samples: np.ndarray, degree: int, basis: str = "hermite"
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the PCE design matrix Phi of shape (L, P) and multi-index array (P, d).

    Phi[l, j] = prod_{dim} psi_{j[dim]}(mu_samples[l, dim]).
    """
    L, d = mu_samples.shape
    multi_idx = _multi_indices(d, degree)
    P = len(multi_idx)
    basis_fn = hermite_basis if basis == "hermite" else legendre_basis
    # Precompute all univariate basis evaluations: shape (d, degree+1, L)
    uni_basis = np.zeros((d, degree + 1, L))
    for dim in range(d):
        uni_basis[dim] = basis_fn(mu_samples[:, dim], degree)
    # Tensor product
    Phi = np.ones((L, P))
    for j_idx, j in enumerate(multi_idx):
        for dim in range(d):
            Phi[:, j_idx] *= uni_basis[dim, j[dim]]
    return Phi, multi_idx


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def fit_pce(Phi: np.ndarray, y: np.ndarray, lam: float = 1e-4) -> np.ndarray:
    """Ridge regression: c = (Phi^T Phi + lam*I)^{-1} Phi^T y."""
    P = Phi.shape[1]
    A = Phi.T @ Phi + lam * np.eye(P)
    cond = np.linalg.cond(A)
    if cond > 1e12:
        import warnings
        warnings.warn(f"Ill-conditioned design matrix (cond={cond:.1e}). "
                      f"Consider increasing ridge_lambda or reducing degree.")
    return np.linalg.solve(A, Phi.T @ y)


def _gcv_score(Phi: np.ndarray, y: np.ndarray, lam: float) -> float:
    """Generalised cross-validation score for ridge parameter selection.

    GCV(lam) = (1/L) * ||y - Phi c_lam||^2 / (1 - tr(H_lam)/L)^2
    where H_lam = Phi (Phi^T Phi + lam I)^{-1} Phi^T is the hat matrix.
    """
    L, P = Phi.shape
    A = Phi.T @ Phi + lam * np.eye(P)
    c = np.linalg.solve(A, Phi.T @ y)
    residuals = y - Phi @ c
    # tr(H) = tr(Phi A^{-1} Phi^T) = tr(Phi^T Phi A^{-1}) via cyclic property
    A_inv_PhiTPhi = np.linalg.solve(A, Phi.T @ Phi)
    tr_H = np.trace(A_inv_PhiTPhi)
    rss = float(np.sum(residuals ** 2))
    denom = (1.0 - tr_H / L) ** 2
    if denom < 1e-15:
        return float('inf')
    return rss / (L * denom)


def select_lambda_gcv(Phi: np.ndarray, y: np.ndarray,
                      lambdas: Optional[np.ndarray] = None) -> float:
    """Select optimal ridge lambda via generalised cross-validation."""
    if lambdas is None:
        lambdas = np.logspace(-8, 0, 17)  # 1e-8 to 1e0
    best_lam, best_score = lambdas[0], float('inf')
    for lam in lambdas:
        score = _gcv_score(Phi, y, lam)
        if score < best_score:
            best_score = score
            best_lam = lam
    return float(best_lam)


# ---------------------------------------------------------------------------
# ALR transform and inverse (paper Eq. 2-3)
# ---------------------------------------------------------------------------

def alr_transform(policies: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Additive log-ratio transform: y_k = log(p_k / p_{K-1}), k = 0..K-2.

    Args:
        policies: (L, K) probability simplex rows
    Returns:
        alr: (L, K-1) log-ratios relative to the reference (last) action
    """
    # Clip ratios, not components, to avoid breaking simplex normalization
    ratios = policies[:, :-1] / np.clip(policies[:, -1:], eps, None)
    return np.log(np.clip(ratios, eps, 1.0 / eps))


def alr_inverse(y: np.ndarray) -> np.ndarray:
    """Inverse ALR: softmax([y_0, ..., y_{K-2}, 0]).

    Args:
        y: (..., K-1) array of log-ratios
    Returns:
        p: (..., K) probability vectors (sum to 1, all positive)
    """
    zero = np.zeros((*y.shape[:-1], 1))
    logits = np.concatenate([y, zero], axis=-1)
    return _softmax(logits)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / exp_l.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# PCE surrogate: single step
# ---------------------------------------------------------------------------

class PCESurrogate:
    """PCE surrogate for a single trajectory step.

    Fits K-1 PCEs in ALR-space (reference = last action, index K-1).
    Sobol indices are computed per ALR component and per reward dimension.
    """

    def __init__(self, degree: int = 5, basis: str = "hermite",
                 ridge_lambda: float = 1e-4, auto_lambda: bool = True):
        self.degree = degree
        self.basis = basis
        self.ridge_lambda = ridge_lambda
        self.auto_lambda = auto_lambda
        # Keyed by k in 0..K-2 (ALR components)
        self.coefficients: Dict[int, np.ndarray] = {}
        self.lambdas: Dict[int, float] = {}  # per-ALR-component lambda
        self.multi_idx: Optional[np.ndarray] = None
        self.n_actions: Optional[int] = None
        # Cached training data for bootstrap and sigma^2 estimation
        self._Phi_train: Optional[np.ndarray] = None
        self._mu_train: Optional[np.ndarray] = None
        self._alr_train: Optional[np.ndarray] = None

    def fit(self, mu_train: np.ndarray, policies_train: np.ndarray) -> None:
        """Fit PCE surrogates using ALR transform.

        Args:
            mu_train:      (L, d) reward parameterisations in R^d
            policies_train: (L, K) policy probability vectors (rows sum to 1)
        """
        L, K = policies_train.shape
        self.n_actions = K
        Phi, self.multi_idx = build_design_matrix(mu_train, self.degree, self.basis)
        self._Phi_train = Phi
        self._mu_train = mu_train
        # Normality check (warn if Hermite basis used on non-Gaussian inputs)
        if self.basis == "hermite" and L >= 20:
            from scipy.stats import shapiro
            for dim in range(mu_train.shape[1]):
                _, p_val = shapiro(mu_train[:, dim])
                if p_val < 0.01:
                    import warnings
                    warnings.warn(
                        f"Shapiro-Wilk p={p_val:.4f} for dim {dim}: inputs may "
                        f"not be Gaussian. Hermite basis assumes N(0,1) inputs.")
                    break  # warn once
        # ALR transform: K-1 components, reference = last action
        alr = alr_transform(policies_train)   # (L, K-1)
        self._alr_train = alr
        for k in range(K - 1):
            if self.auto_lambda and L >= 10:
                lam = select_lambda_gcv(Phi, alr[:, k])
                self.lambdas[k] = lam
            else:
                lam = self.ridge_lambda
                self.lambdas[k] = lam
            self.coefficients[k] = fit_pce(Phi, alr[:, k], lam)

    def predict(self, mu_new: np.ndarray) -> np.ndarray:
        """Predict policy distributions at new reward parameterisations.

        Returns:
            policies: (n_new, K) probability vectors (sum to 1, all > 0)
        """
        Phi_new, _ = build_design_matrix(mu_new, self.degree, self.basis)
        alr_pred = np.zeros((mu_new.shape[0], self.n_actions - 1))
        for k in range(self.n_actions - 1):
            alr_pred[:, k] = Phi_new @ self.coefficients[k]
        return alr_inverse(alr_pred)

    def sample_policies(
        self,
        n_samples: int,
        mu_distribution: str = "gaussian",
        mu_params: Optional[dict] = None,
    ) -> np.ndarray:
        mu_params = mu_params or {}
        d = mu_params.get("d", self.multi_idx.shape[1])
        if mu_distribution == "gaussian":
            mean = mu_params.get("mean", np.zeros(d))
            std = mu_params.get("std", np.ones(d))
            mu_samples = np.random.randn(n_samples, d) * std + mean
        else:
            low = mu_params.get("low", -np.ones(d))
            high = mu_params.get("high", np.ones(d))
            mu_samples = np.random.uniform(low, high, (n_samples, d))
        return self.predict(mu_samples)

    def sobol_indices(self) -> Dict[str, np.ndarray]:
        """Compute first-order and total-order Sobol indices from PCE coefficients.

        Plain-language summary
        ~~~~~~~~~~~~~~~~~~~~~~
        Each Sobol index answers a simple question:

          "If I could eliminate uncertainty in reward component i,
           how much would total policy uncertainty at this step decrease?"

        A first-order index S_i close to 1 means that component alone is
        essentially the whole story -- fix it and the policy becomes
        deterministic.  S_i close to 0 means it barely matters.

        Total-order indices T_i additionally include interaction effects
        (uncertainty in component i *combined with* other components).
        When T_i >> S_i, the component's influence comes mainly through
        interactions rather than in isolation.

        Mathematical detail
        ~~~~~~~~~~~~~~~~~~~
        Because the PCE basis functions are orthonormal, the total output
        variance decomposes exactly into contributions from each basis
        term.  Grouping terms by which input dimensions they involve gives
        the Sobol indices analytically -- no Monte Carlo sampling needed:

            S_i^(k) = sum_{j: j_i>0, j_{-i}=0} c_j^2 / sum_{j!=0} c_j^2

        This formula is valid because PCA ensures the inputs are
        independent (see module docstring for the PCA-vs-VAE rationale).

        Returns dict with:
            'first_order': (K-1, d) -- first-order index per ALR component
                           per reward dimension
            'total_order': (K-1, d) -- total-order index (includes interactions)
            'variance':    (K-1,)   -- total ALR-space variance per component
        """
        K_eff = self.n_actions - 1
        d = self.multi_idx.shape[1]

        # Precompute index masks once (vectorised over multi_idx)
        mi = self.multi_idx   # (P, d)
        fo_masks = []  # first-order: only dimension i active
        to_masks = []  # total-order: dimension i appears anywhere
        for i in range(d):
            # First order: j_i > 0 AND sum(j) == j_i  (all others are 0)
            fo = (mi[:, i] > 0) & (mi.sum(axis=1) == mi[:, i])
            fo[0] = False   # explicitly exclude the constant term (index 0)
            fo_masks.append(fo)
            # Total order: j_i > 0 (constant term automatically excluded)
            to_masks.append(mi[:, i] > 0)

        first_order = np.zeros((K_eff, d))
        total_order = np.zeros((K_eff, d))
        variance = np.zeros(K_eff)

        for k in range(K_eff):
            c = self.coefficients[k]
            D_total = float(np.sum(c[1:] ** 2))   # exclude constant term c[0]
            variance[k] = D_total
            if D_total < 1e-15:
                continue
            for i in range(d):
                first_order[k, i] = float(np.sum(c[fo_masks[i]] ** 2)) / D_total
                total_order[k, i] = float(np.sum(c[to_masks[i]] ** 2)) / D_total

        return {"first_order": first_order, "total_order": total_order, "variance": variance}

    def sobol_indices_with_ci(
        self,
        n_bootstrap: int = 200,
        ci_level: float = 0.90,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """Sobol indices with bootstrap confidence intervals.

        Resamples training data with replacement, refits PCE, recomputes
        Sobol for each resample to produce empirical CIs.

        Returns dict with:
            'first_order':    (K-1, d) point estimates
            'total_order':    (K-1, d) point estimates
            'variance':       (K-1,)
            'first_order_ci': (K-1, d, 2) lower/upper bounds
            'total_order_ci': (K-1, d, 2) lower/upper bounds
        """
        if self._Phi_train is None or self._alr_train is None or self._mu_train is None:
            raise RuntimeError("Call fit() first.")
        rng = np.random.RandomState(seed)
        L = self._mu_train.shape[0]
        K_eff = self.n_actions - 1
        d = self.multi_idx.shape[1]

        # Collect bootstrap Sobol samples
        fo_samples = np.zeros((n_bootstrap, K_eff, d))
        to_samples = np.zeros((n_bootstrap, K_eff, d))

        for b in range(n_bootstrap):
            idx = rng.choice(L, size=L, replace=True)
            Phi_b = self._Phi_train[idx]
            alr_b = self._alr_train[idx]
            mi = self.multi_idx
            for k in range(K_eff):
                lam = self.lambdas.get(k, self.ridge_lambda)
                c = fit_pce(Phi_b, alr_b[:, k], lam)
                D_total = float(np.sum(c[1:] ** 2))
                if D_total < 1e-15:
                    continue
                for i in range(d):
                    fo = (mi[:, i] > 0) & (mi.sum(axis=1) == mi[:, i])
                    fo[0] = False
                    fo_samples[b, k, i] = float(np.sum(c[fo] ** 2)) / D_total
                    to_mask = mi[:, i] > 0
                    to_samples[b, k, i] = float(np.sum(c[to_mask] ** 2)) / D_total

        # Point estimates
        point = self.sobol_indices()

        # CI bounds
        alpha = (1.0 - ci_level) / 2.0
        fo_ci = np.stack([
            np.quantile(fo_samples, alpha, axis=0),
            np.quantile(fo_samples, 1.0 - alpha, axis=0),
        ], axis=-1)
        to_ci = np.stack([
            np.quantile(to_samples, alpha, axis=0),
            np.quantile(to_samples, 1.0 - alpha, axis=0),
        ], axis=-1)

        return {
            **point,
            "first_order_ci": fo_ci,
            "total_order_ci": to_ci,
            "n_bootstrap": n_bootstrap,
            "ci_level": ci_level,
        }

    def residual_sigma2(self) -> float:
        """Estimate noise variance sigma^2 from training residuals.

        Uses the unbiased estimator: sigma^2 = ||y - Phi c||^2 / (L - P).
        Averaged over all K-1 ALR components.
        """
        if self._Phi_train is None or self._alr_train is None:
            return 1.0
        Phi, alr = self._Phi_train, self._alr_train
        L, P = Phi.shape
        df = max(L - P, 1)
        sigma2_list = []
        for k in range(self.n_actions - 1):
            resid = alr[:, k] - Phi @ self.coefficients[k]
            sigma2_list.append(float(np.sum(resid ** 2) / df))
        return float(np.mean(sigma2_list))

    def sample_complexity_bound(
        self,
        target_sobol_error: float = 0.05,
        confidence: float = 0.95,
        sigma2_override: Optional[float] = None,
    ) -> int:
        """Required ensemble size for target Sobol accuracy (Theorem A, paper Sec 2.6).

        Bound derivation:
          - Ridge estimator error: E[||c_hat - c*||^2] <= P * sigma^2 / (L - P)
          - Sobol index error (delta method): |S_hat_i - S_i| <= 2*sqrt(P*sigma^2/(L-P)) / D^{1/2}
          - Solving for L: L >= P + 4*P*sigma^2*log(K*d/delta) / (D_min * eps^2)

        Args:
            target_sobol_error: desired |S_hat - S_true| bound (epsilon)
            confidence:         probability level (1 - delta)
            sigma2_override:    noise variance; if None, estimated from residuals

        Returns:
            Minimum L for the given accuracy/confidence.
        """
        if not self.coefficients:
            raise RuntimeError("Call fit() first.")
        P = len(self.multi_idx)
        K_eff = self.n_actions - 1
        d = self.multi_idx.shape[1]
        delta = 1.0 - confidence
        sigma2 = sigma2_override if sigma2_override is not None else self.residual_sigma2()
        sobol = self.sobol_indices()
        D_min = max(float(sobol["variance"].min()), 1e-6)
        log_term = np.log(max(K_eff * d / delta, 2.0))
        L_base = int(np.ceil(4.0 * P * sigma2 * log_term / (D_min * target_sobol_error ** 2)))
        return P + L_base   # must have at least P samples (else underdetermined)


# ---------------------------------------------------------------------------
# PCE surrogate: full trajectory
# ---------------------------------------------------------------------------

class TrajectoryPCESurrogate:
    """Collection of per-step PCE surrogates for a full trajectory."""

    def __init__(self, degree: int = 5, basis: str = "hermite",
                 ridge_lambda: float = 1e-4, auto_lambda: bool = True):
        self.degree = degree
        self.basis = basis
        self.ridge_lambda = ridge_lambda
        self.auto_lambda = auto_lambda
        self.step_surrogates: Dict[int, PCESurrogate] = {}

    def fit_step(self, step: int, mu_train: np.ndarray, policies_train: np.ndarray) -> None:
        s = PCESurrogate(self.degree, self.basis, self.ridge_lambda, self.auto_lambda)
        s.fit(mu_train, policies_train)
        self.step_surrogates[step] = s

    def sobol_all_steps(self) -> Dict[int, Dict[str, np.ndarray]]:
        return {step: s.sobol_indices() for step, s in self.step_surrogates.items()}

    def sobol_all_steps_with_ci(self, n_bootstrap: int = 200, ci_level: float = 0.90
                                 ) -> Dict[int, Dict[str, np.ndarray]]:
        return {step: s.sobol_indices_with_ci(n_bootstrap, ci_level)
                for step, s in self.step_surrogates.items()}

    def summarise_sobol(
        self,
        step_labels: Optional[Dict[int, str]] = None,
        dim_labels: Optional[List[str]] = None,
    ) -> str:
        """Return a human-readable narrative summary of Sobol indices.

        This is intended for experiment scripts that want to print an
        accessible interpretation alongside the raw numbers.

        Args:
            step_labels: optional mapping from step index to a short name
                         (e.g. {0: "catalyst", 1: "base"}).
            dim_labels:  optional names for reward-parameter dimensions
                         (e.g. ["PC1 (mean level)", "PC2 (spread)"]).
                         Defaults to "PC1", "PC2", ...

        Returns:
            Multi-line string with per-step narrative.
        """
        sobol = self.sobol_all_steps()
        d = list(self.step_surrogates.values())[0].multi_idx.shape[1]
        if dim_labels is None:
            dim_labels = [f"PC{i+1}" for i in range(d)]
        lines = []
        lines.append("Sobol sensitivity summary")
        lines.append("=" * 60)
        lines.append(
            "Each index S_i answers: 'What fraction of policy uncertainty at "
            "this step is attributable to reward-parameter component i?'"
        )
        lines.append(
            "Values near 1 = dominant driver; near 0 = negligible influence."
        )
        lines.append("")
        for step in sorted(sobol.keys()):
            sv = sobol[step]
            var_arr = sv["variance"]
            total_var = float(var_arr.sum())
            label = step_labels[step] if step_labels and step in step_labels else f"Step {step}"
            if total_var < 1e-12:
                lines.append(f"  {label}: negligible policy variance (effectively deterministic)")
                continue
            # Pick the ALR component with highest variance for the headline
            mi = int(np.argmax(var_arr))
            tot = sv["total_order"][mi]
            fo = sv["first_order"][mi]
            parts = []
            for i in range(d):
                parts.append(f"{dim_labels[i]}={tot[i]:.2f}")
            headline = ", ".join(parts)
            # Identify dominant driver
            dom_idx = int(np.argmax(tot))
            dom_share = tot[dom_idx]
            if d >= 2:
                runner_up = sorted(range(d), key=lambda i: tot[i], reverse=True)[1]
                ratio = tot[dom_idx] / max(tot[runner_up], 1e-10)
                if ratio > 3.0:
                    interp = f"{dim_labels[dom_idx]} dominates"
                elif ratio > 1.5:
                    interp = f"{dim_labels[dom_idx]} leads, {dim_labels[runner_up]} also contributes"
                else:
                    interp = f"{dim_labels[dom_idx]} and {dim_labels[runner_up]} contribute comparably"
            else:
                interp = f"{dim_labels[dom_idx]}: {dom_share:.0%} of variance"
            lines.append(f"  {label:20s}  {headline}  -> {interp}")
        lines.append("")
        return "\n".join(lines)

    def sample_trajectory_policies(
        self, n_samples: int, mu_distribution: str = "gaussian", mu_params: Optional[dict] = None
    ) -> Dict[int, np.ndarray]:
        return {
            step: s.sample_policies(n_samples, mu_distribution, mu_params)
            for step, s in self.step_surrogates.items()
        }

    def required_ensemble_size(
        self, target_sobol_error: float = 0.05, confidence: float = 0.95
    ) -> int:
        """Maximum required ensemble size across all trajectory steps (Theorem A)."""
        return max(
            s.sample_complexity_bound(target_sobol_error, confidence)
            for s in self.step_surrogates.values()
        )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def ks_test_2sample(s1: np.ndarray, s2: np.ndarray):
    from scipy.stats import ks_2samp
    return ks_2samp(s1, s2)


def calibration_coverage(
    surr_samples: np.ndarray,
    test_policies: np.ndarray,
    levels: List[float] = [0.5, 0.8, 0.9, 0.95],
) -> Dict[float, float]:
    """Empirical MARGINAL coverage of PCE-derived credible intervals vs. test ensemble.

    Reports the fraction of (test_point, action) pairs falling within the
    marginal credible interval.  This is the standard calibration metric:
    at level alpha, ideally alpha fraction of marginals are covered.

    For joint coverage (all K actions simultaneously within interval),
    use calibration_coverage_joint().
    """
    coverages = {}
    for alpha in levels:
        lo, hi = (1 - alpha) / 2, 1 - (1 - alpha) / 2
        lower = np.quantile(surr_samples, lo, axis=0)  # (K,)
        upper = np.quantile(surr_samples, hi, axis=0)  # (K,)
        # Marginal: average over both test points AND actions
        in_iv = (test_policies >= lower) & (test_policies <= upper)  # (n_test, K)
        coverages[alpha] = float(np.mean(in_iv))
    return coverages


def calibration_coverage_joint(
    surr_samples: np.ndarray,
    test_policies: np.ndarray,
    levels: List[float] = [0.5, 0.8, 0.9, 0.95],
) -> Dict[float, float]:
    """Joint coverage: fraction of test points where ALL actions fall within interval."""
    coverages = {}
    for alpha in levels:
        lo, hi = (1 - alpha) / 2, 1 - (1 - alpha) / 2
        lower = np.quantile(surr_samples, lo, axis=0)
        upper = np.quantile(surr_samples, hi, axis=0)
        in_iv = np.all((test_policies >= lower) & (test_policies <= upper), axis=-1)
        coverages[alpha] = float(np.mean(in_iv))
    return coverages


def run_ks_battery(
    tsurr: TrajectoryPCESurrogate,
    te_pol: Dict[int, np.ndarray],
    n_mc: int = 10000,
    alpha: float = 0.05,
) -> Dict[int, Dict]:
    """Run KS test battery across all steps, return pass fraction."""
    results = {}
    d = list(tsurr.step_surrogates.values())[0].multi_idx.shape[1]
    mu_params = {"d": d}
    for step, surr in tsurr.step_surrogates.items():
        mc_pols = surr.sample_policies(n_mc, mu_params=mu_params)  # (n_mc, K)
        test_pol = te_pol[step]                                      # (n_test, K)
        K = mc_pols.shape[1]
        ks_results = []
        for k in range(K):
            stat, p = ks_test_2sample(mc_pols[:, k], test_pol[:, k])
            ks_results.append({"action": k, "statistic": float(stat), "p_value": float(p),
                                "passes": p > alpha})
        n_pass = sum(r["passes"] for r in ks_results)
        results[step] = {"per_action": ks_results, "fraction_pass": n_pass / K,
                         "n_pass": n_pass, "n_total": K}
    return results


def check_convergence(
    loss_history: np.ndarray,
    window: int = 100,
    rtol: float = 0.05,
) -> Dict:
    """Check whether a GFlowNet training loss has converged.

    Compares the mean loss in the last *window* episodes to the previous
    *window* episodes.  If the relative change is below *rtol*, training
    is considered converged.

    Args:
        loss_history: 1-D array of per-episode loss values.
        window:       number of episodes in the comparison windows.
        rtol:         relative tolerance for declaring convergence.

    Returns:
        dict with keys: converged (bool), rel_change (float),
        final_mean (float), prev_mean (float).
    """
    loss_history = np.asarray(loss_history, dtype=np.float64)
    n = len(loss_history)
    if n < 2 * window:
        window = max(n // 4, 1)

    prev_mean = float(np.mean(loss_history[-2 * window: -window]))
    final_mean = float(np.mean(loss_history[-window:]))
    if abs(prev_mean) < 1e-15:
        rel_change = 0.0 if abs(final_mean) < 1e-15 else float("inf")
    else:
        rel_change = abs(final_mean - prev_mean) / abs(prev_mean)

    return {
        "converged": rel_change < rtol,
        "rel_change": round(rel_change, 6),
        "final_mean": round(final_mean, 6),
        "prev_mean": round(prev_mean, 6),
        "window": window,
    }
