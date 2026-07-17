"""Sparse and arbitrary polynomial chaos expansions.

Two methodological components:

  1. Replace dense ridge PCE with modern sparse PCE. We provide LARS/LASSO
        (`LassoLarsCV`) and orthogonal matching pursuit (`OrthogonalMatchingPursuitCV`)
        solvers over a large candidate basis, with hyperbolic (q-norm) truncation
        so degree/dimension can grow without materialising the full total-degree
        basis. Model quality is reported via the corrected leave-one-out error
        (Blatman & Sudret 2011), the standard sparse-PCE selection metric.

  2. Provide a basis orthonormal w.r.t. the actual input distribution rather than
        assuming Gaussian inputs. `apc_univariate_basis` builds data-driven
        orthonormal polynomials from the empirical marginal moments (arbitrary PCE,
        Oladyshkin & Nowak 2012) via a Gram/Cholesky construction, so the
        coefficient-square Sobol formula stays exact under input independence
        (independence itself is checked separately with distance correlation).

The Sobol indices are computed from the (sparse) coefficients exactly as before,
because the selected basis is a subset of an orthonormal basis.

Dependencies: numpy, scipy, scikit-learn.
"""
from __future__ import annotations

import itertools
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.pce_surrogate import hermite_basis, legendre_basis, alr_transform


# ---------------------------------------------------------------------------
# Candidate basis with hyperbolic (q-norm) truncation
# ---------------------------------------------------------------------------

def _bounded_degree_tuples(d: int, p: int):
    """All d-tuples of nonnegative ints with total degree sum <= p.

    Enumerated recursively so the count is C(d+p, p), not the (p+1)^d full grid;
    this keeps generation feasible at the higher PCA dimensions the extended-analysis
    sweeps (e.g. d=15, p=5 gives 15504 tuples, not 6^15).
    """
    def rec(pos, remaining, cur):
        if pos == d:
            yield tuple(cur)
            return
        for v in range(remaining + 1):
            cur.append(v)
            yield from rec(pos + 1, remaining - v, cur)
            cur.pop()
    return rec(0, p, [])


def hyperbolic_multi_indices(d: int, p: int, q: float = 1.0,
                             max_interaction: Optional[int] = None) -> np.ndarray:
    """Multi-indices j in N^d with q-norm (sum j^q)^(1/q) <= p.

    q = 1 recovers the total-degree basis; q < 1 favours low-order interactions,
    the standard sparse-PCE truncation that tames the curse of dimensionality.
    `max_interaction` optionally caps the number of nonzero components (rank).
    For q <= 1 every admissible index has total degree <= p, so we enumerate the
    total-degree set (small) and filter, rather than the full grid.
    """
    out = []
    for combo in _bounded_degree_tuples(d, p):
        if max_interaction is not None and sum(1 for c in combo if c > 0) > max_interaction:
            continue
        if q >= 1.0 - 1e-12:
            out.append(combo)
        else:
            qnorm = float(sum(c ** q for c in combo)) ** (1.0 / q) if any(combo) else 0.0
            if qnorm <= p + 1e-9:
                out.append(combo)
    out.sort(key=lambda a: (sum(a), a))   # constant term first, then by degree
    return np.array(out, dtype=int)


# ---------------------------------------------------------------------------
# Arbitrary PCE: data-driven orthonormal univariate polynomials
# ---------------------------------------------------------------------------

def apc_univariate_basis(x: np.ndarray, degree: int,
                         jitter: float = 1e-10) -> np.ndarray:
    """Orthonormal polynomials up to `degree` w.r.t. the empirical measure of `x`.

    Construction: with the monomial Vandermonde V[l,k] = x_l^k (k=0..p), the Gram
    matrix G = (1/L) V^T V holds the empirical moments E[x^{i+j}]. If G = R^T R
    (Cholesky), then Psi = V @ inv(R) satisfies (1/L) Psi^T Psi = I, i.e. its
    columns are orthonormal polynomials w.r.t. the empirical marginal.

    Returns array of shape (degree+1, L): row n is psi_n evaluated at the samples,
    matching the layout of `hermite_basis` / `legendre_basis`.
    """
    x = np.asarray(x, dtype=float)
    L = x.shape[0]
    # standardise for conditioning; orthonormality is affine-invariant
    mu, sd = x.mean(), x.std()
    xs = (x - mu) / (sd if sd > 1e-12 else 1.0)
    V = np.vander(xs, N=degree + 1, increasing=True)      # (L, p+1)
    G = (V.T @ V) / L
    G = G + jitter * np.eye(degree + 1)
    try:
        R = np.linalg.cholesky(G).T                       # upper triangular, G = R^T R
    except np.linalg.LinAlgError:
        # near-singular empirical moments (small L / low-degree): add more jitter
        R = np.linalg.cholesky(G + 1e-6 * np.eye(degree + 1)).T
    Psi = V @ np.linalg.inv(R)                            # (L, p+1), columns orthonormal
    return Psi.T                                          # (p+1, L)


def _univariate_evals(mu: np.ndarray, degree: int, basis: str,
                      apc_R: Optional[List[np.ndarray]] = None,
                      apc_stats: Optional[List[Tuple[float, float]]] = None
                      ) -> np.ndarray:
    """Univariate basis evaluations, shape (d, degree+1, L).

    For 'apc' the caller may pass precomputed (R, mean, std) per dimension so that
    train-fitted polynomials can be applied to test points without refitting.
    """
    L, d = mu.shape
    out = np.zeros((d, degree + 1, L))
    if basis == "apc":
        for dim in range(d):
            if apc_R is not None:
                R = apc_R[dim]; m, s = apc_stats[dim]
                xs = (mu[:, dim] - m) / (s if s > 1e-12 else 1.0)
                V = np.vander(xs, N=degree + 1, increasing=True)
                out[dim] = (V @ np.linalg.inv(R)).T
            else:
                out[dim] = apc_univariate_basis(mu[:, dim], degree)
    else:
        fn = hermite_basis if basis == "hermite" else legendre_basis
        for dim in range(d):
            out[dim] = fn(mu[:, dim], degree)
    return out


def fit_apc_recurrence(mu: np.ndarray, degree: int, jitter: float = 1e-10):
    """Return per-dimension (R, mean, std) so aPC polynomials fit on train transfer to test."""
    L, d = mu.shape
    Rs, stats = [], []
    for dim in range(d):
        x = mu[:, dim]
        m, s = x.mean(), x.std()
        xs = (x - m) / (s if s > 1e-12 else 1.0)
        V = np.vander(xs, N=degree + 1, increasing=True)
        G = (V.T @ V) / L + jitter * np.eye(degree + 1)
        try:
            R = np.linalg.cholesky(G).T
        except np.linalg.LinAlgError:
            R = np.linalg.cholesky(G + 1e-6 * np.eye(degree + 1)).T
        Rs.append(R); stats.append((float(m), float(s)))
    return Rs, stats


def build_design_matrix_general(mu: np.ndarray, multi_idx: np.ndarray, degree: int,
                                basis: str = "hermite",
                                apc_R=None, apc_stats=None) -> np.ndarray:
    """Design matrix Phi (L, P) for an arbitrary multi-index set and basis."""
    L, d = mu.shape
    P = len(multi_idx)
    uni = _univariate_evals(mu, degree, basis, apc_R, apc_stats)
    Phi = np.ones((L, P))
    for jx, j in enumerate(multi_idx):
        for dim in range(d):
            Phi[:, jx] *= uni[dim, j[dim]]
    return Phi


# ---------------------------------------------------------------------------
# Corrected leave-one-out error (Blatman & Sudret 2011)
# ---------------------------------------------------------------------------

def corrected_loo_error(Phi_active: np.ndarray, y: np.ndarray) -> float:
    """Relative corrected LOO error for OLS on the given (active) basis.

    err_LOO = mean_i [ (y_i - yhat_i) / (1 - h_ii) ]^2 * T / Var(y),
    with hat matrix H = Phi (Phi^T Phi)^{-1} Phi^T and the standard correction
    factor T = (L / (L - P)) (1 + tr((Phi^T Phi)^{-1} C_emp)) approximated here by
    the common (L/(L-P)) multiplier. Returned value is normalised by Var(y).
    """
    L, P = Phi_active.shape
    if L <= P + 1:
        return float("inf")
    G = Phi_active.T @ Phi_active
    try:
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return float("inf")
    H = Phi_active @ Ginv @ Phi_active.T
    h = np.clip(np.diag(H), None, 1 - 1e-8)
    c = Ginv @ Phi_active.T @ y
    resid = y - Phi_active @ c
    press = np.mean((resid / (1.0 - h)) ** 2)
    var_y = np.var(y)
    if var_y < 1e-15:
        return 0.0
    corr = (L / (L - P)) * (1.0 + np.trace(Ginv))
    return float(press * corr / var_y)


# ---------------------------------------------------------------------------
# Sparse solvers over the candidate basis
# ---------------------------------------------------------------------------

def fit_sparse(Phi: np.ndarray, y: np.ndarray, method: str = "lars",
               ols_refit: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit sparse PCE coefficients.

    Returns (coeffs (P,), active_mask (P,), corrected_loo_error).
    The constant column (index 0) is always kept active.
    method: 'lars' (LassoLarsCV), 'omp' (OrthogonalMatchingPursuitCV).
    """
    from sklearn.linear_model import LassoLarsCV, OrthogonalMatchingPursuitCV

    L, P = Phi.shape
    # center y; the constant term is handled by intercept then folded back
    y_mean = y.mean()
    yc = y - y_mean
    # Degenerate (near-constant) response: the CV solvers throw "argmin of empty
    # sequence" (common for ALR components of a policy that concentrates on a few
    # actions, e.g. Sachs with K=111). Return a constant fit (zero non-constant
    # variance, hence zero Sobol contribution), which is the correct limit.
    if np.std(yc) < 1e-10:
        coeffs = np.zeros(P); coeffs[0] = y_mean
        active = np.zeros(P, dtype=bool); active[0] = True
        return coeffs, active, 0.0
    X = Phi[:, 1:]
    cv = min(5, L)
    coeffs = np.zeros(P)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "omp":
                n_max = max(1, min(P - 1, L - 2))
                model = OrthogonalMatchingPursuitCV(cv=cv, max_iter=n_max, fit_intercept=True)
            else:
                model = LassoLarsCV(cv=cv, fit_intercept=True, max_iter=2000)
            model.fit(X, yc)
        coeffs[0] = y_mean + getattr(model, "intercept_", 0.0)
        coeffs[1:] = model.coef_
    except Exception:
        # CV solver failed on this component; fall back to the constant predictor.
        coeffs = np.zeros(P); coeffs[0] = y_mean
        active = np.zeros(P, dtype=bool); active[0] = True
        return coeffs, active, float("inf")
    active = np.abs(coeffs) > 1e-12
    active[0] = True
    # OLS refit on active set (debiases shrunk coefficients) + corrected LOO
    if ols_refit and active.sum() < L - 1:
        Phi_a = Phi[:, active]
        c_a, *_ = np.linalg.lstsq(Phi_a, y, rcond=None)
        coeffs = np.zeros(P)
        coeffs[active] = c_a
        loo = corrected_loo_error(Phi_a, y)
    else:
        loo = corrected_loo_error(Phi[:, active], y)
    return coeffs, active, loo


# ---------------------------------------------------------------------------
# Sobol indices from (sparse) coefficients on an orthonormal basis
# ---------------------------------------------------------------------------

def sobol_from_coeffs(coeffs: np.ndarray, multi_idx: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, float]:
    """First-order and total-order Sobol indices from orthonormal-basis coefficients.

    Returns (first_order (d,), total_order (d,), total_variance).
    """
    d = multi_idx.shape[1]
    nz = np.arange(len(multi_idx)) != 0  # exclude constant
    c2 = coeffs ** 2
    D = float(c2[nz].sum())
    first = np.zeros(d); total = np.zeros(d)
    if D <= 0:
        return first, total, 0.0
    for i in range(d):
        only_i = (multi_idx[:, i] > 0) & (np.delete(multi_idx, i, axis=1).sum(axis=1) == 0)
        any_i = multi_idx[:, i] > 0
        first[i] = c2[only_i].sum() / D
        total[i] = c2[any_i].sum() / D
    return first, total, D


# ---------------------------------------------------------------------------
# Sparse PCE surrogate (per trajectory step; K-1 ALR components)
# ---------------------------------------------------------------------------

class SparsePCESurrogate:
    """Sparse / arbitrary PCE surrogate for one trajectory step.

    basis: 'hermite' | 'legendre' | 'apc'.  method: 'lars' | 'omp'.
    """

    def __init__(self, degree: int = 5, basis: str = "apc", method: str = "lars",
                 q_norm: float = 1.0, max_interaction: Optional[int] = None):
        self.degree = degree
        self.basis = basis
        self.method = method
        self.q_norm = q_norm
        self.max_interaction = max_interaction
        self.multi_idx: Optional[np.ndarray] = None
        self.coefficients: Dict[int, np.ndarray] = {}
        self.active: Dict[int, np.ndarray] = {}
        self.loo: Dict[int, float] = {}
        self.n_actions: Optional[int] = None
        self._apc_R = None
        self._apc_stats = None

    def fit(self, mu_train: np.ndarray, policies_train: np.ndarray) -> None:
        L, K = policies_train.shape
        d = mu_train.shape[1]
        self.n_actions = K
        self.multi_idx = hyperbolic_multi_indices(d, self.degree, self.q_norm,
                                                   self.max_interaction)
        if self.basis == "apc":
            self._apc_R, self._apc_stats = fit_apc_recurrence(mu_train, self.degree)
        Phi = build_design_matrix_general(mu_train, self.multi_idx, self.degree,
                                           self.basis, self._apc_R, self._apc_stats)
        alr = alr_transform(policies_train)  # (L, K-1)
        for k in range(K - 1):
            c, a, loo = fit_sparse(Phi, alr[:, k], self.method)
            self.coefficients[k] = c
            self.active[k] = a
            self.loo[k] = loo

    def predict(self, mu_new: np.ndarray) -> np.ndarray:
        """Predict policy distributions (n, K) at new reward parameterisations."""
        from core.pce_surrogate import alr_inverse
        Phi = build_design_matrix_general(mu_new, self.multi_idx, self.degree,
                                           self.basis, self._apc_R, self._apc_stats)
        K = self.n_actions
        alr = np.column_stack([Phi @ self.coefficients[k] for k in range(K - 1)])
        return alr_inverse(alr)

    def sobol_indices(self) -> Dict[str, np.ndarray]:
        K, d = self.n_actions, self.multi_idx.shape[1]
        fo = np.zeros((K - 1, d)); to = np.zeros((K - 1, d)); var = np.zeros(K - 1)
        for k in range(K - 1):
            f, t, D = sobol_from_coeffs(self.coefficients[k], self.multi_idx)
            fo[k], to[k], var[k] = f, t, D
        return {"first_order": fo, "total_order": to, "variance": var,
                "loo": np.array([self.loo[k] for k in range(K - 1)]),
                "basis_size": np.array([int(self.active[k].sum()) for k in range(K - 1)]),
                "candidate_size": len(self.multi_idx)}


# ---------------------------------------------------------------------------
# Dependence diagnostics beyond Pearson
# ---------------------------------------------------------------------------

def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Distance correlation in [0,1]; 0 iff independent (Szekely et al. 2007)."""
    x = np.asarray(x, float).reshape(-1, 1)
    y = np.asarray(y, float).reshape(-1, 1)
    n = x.shape[0]
    a = np.abs(x - x.T); b = np.abs(y - y.T)
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov2 = (A * B).sum() / n ** 2
    dvarx = (A * A).sum() / n ** 2
    dvary = (B * B).sum() / n ** 2
    denom = math.sqrt(dvarx * dvary)
    return float(math.sqrt(max(dcov2, 0.0) / denom)) if denom > 1e-15 else 0.0


def max_distance_correlation(mu: np.ndarray) -> float:
    d = mu.shape[1]
    m = 0.0
    for i in range(d):
        for j in range(i + 1, d):
            m = max(m, distance_correlation(mu[:, i], mu[:, j]))
    return m
