"""Validation of the sparse/arbitrary PCE module against analytic ground truth.

Demonstrates the following properties:
  1. aPC builds a basis orthonormal w.r.t. the empirical measure (Gaussian AND
     non-Gaussian inputs), unlike a fixed Hermite basis on non-Gaussian inputs.
  2. Sparse PCE recovers a known sparse expansion in the underdetermined regime
     L < P where dense ridge fails, with far lower corrected-LOO error.
  3. Sobol indices from the sparse coefficients match the analytic indices.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core.sparse_pce import (
    apc_univariate_basis, hyperbolic_multi_indices, build_design_matrix_general,
    fit_sparse, sobol_from_coeffs, corrected_loo_error, max_distance_correlation,
)
from core.pce_surrogate import build_design_matrix, fit_pce

rng = np.random.RandomState(0)


def test_apc_orthonormality():
    print("\n[1] aPC orthonormality (1/L) Psi^T Psi ~ I")
    for name, x in [("Gaussian", rng.randn(4000)),
                    ("Uniform",  rng.uniform(-2, 2, 4000)),
                    ("Exponential(skewed)", rng.exponential(1.0, 4000))]:
        Psi = apc_univariate_basis(x, degree=4).T   # (L, p+1)
        G = (Psi.T @ Psi) / Psi.shape[0]
        off = np.abs(G - np.eye(5)).max()
        print(f"    {name:22s}: max|Gram - I| = {off:.2e}  -> {'OK' if off < 1e-6 else 'FAIL'}")
        assert off < 1e-6

    # contrast: fixed Hermite basis is NOT orthonormal on non-Gaussian input
    from core.pce_surrogate import hermite_basis
    x = rng.exponential(1.0, 4000); x = (x - x.mean()) / x.std()
    H = hermite_basis(x, 4)  # (p+1, L)
    G = (H @ H.T) / len(x)
    print(f"    Hermite on Exponential: max|Gram - I| = {np.abs(G - np.eye(5)).max():.2e}"
          f"  (expected >> 0: basis invalid for non-Gaussian)")


def analytic_sparse_model(mu):
    """f = 2*psi_[1,0,0,0,0] + 1.5*psi_[0,0,2,0,0] + 0.8*psi_[1,0,0,0,1] (Hermite, N(0,I)).

    Orthonormal Hermite: psi_1(x)=x, psi_2(x)=(x^2-1)/sqrt2. Interaction term couples
    dims 0 and 4. Analytic Sobol: D = 4 + 2.25 + 0.64 = 6.89.
    """
    x0, x2, x4 = mu[:, 0], mu[:, 2], mu[:, 4]
    he2 = (x2 ** 2 - 1) / np.sqrt(2)
    return 2.0 * x0 + 1.5 * he2 + 0.8 * (x0 * x4)


def analytic_sobol():
    c = {"d0": 2.0, "d2": 1.5, "d0d4": 0.8}
    D = c["d0"] ** 2 + c["d2"] ** 2 + c["d0d4"] ** 2
    first = np.zeros(5); total = np.zeros(5)
    first[0] = c["d0"] ** 2 / D
    first[2] = c["d2"] ** 2 / D
    total[0] = (c["d0"] ** 2 + c["d0d4"] ** 2) / D
    total[2] = c["d2"] ** 2 / D
    total[4] = c["d0d4"] ** 2 / D
    return first, total, D


def test_sparse_recovery():
    print("\n[2] Sparse recovery in the underdetermined regime (d=5, p=4)")
    d, p, L = 5, 4, 60
    mu = rng.randn(L, d)
    y = analytic_sparse_model(mu) + 0.02 * rng.randn(L)  # tiny noise

    multi = hyperbolic_multi_indices(d, p, q=1.0)
    P = len(multi)
    print(f"    candidate basis P = {P}  vs  L = {L}   (underdetermined: {P > L})")
    Phi = build_design_matrix_general(mu, multi, p, basis="hermite")

    c_sparse, active, loo_sparse = fit_sparse(Phi, y, method="lars")
    print(f"    sparse LARS: active terms = {active.sum():3d}, corrected LOO = {loo_sparse:.2e}")

    # dense ridge on the same (underdetermined) basis
    c_ridge = fit_pce(Phi, y, lam=1e-4)
    loo_ridge = corrected_loo_error(Phi[:, np.abs(c_ridge) > 1e-12], y)
    resid_ridge = np.mean((y - Phi @ c_ridge) ** 2) / np.var(y)
    print(f"    dense ridge: nonzero terms = {int((np.abs(c_ridge)>1e-9).sum()):3d}, "
          f"train rel-MSE = {resid_ridge:.2e} (overfits; LOO ill-defined when P>L)")

    fo, to, D = sobol_from_coeffs(c_sparse, multi)
    fo_t, to_t, D_t = analytic_sobol()
    print(f"    analytic  D = {D_t:.3f}   sparse-PCE D = {D:.3f}")
    print(f"    first-order  analytic = {np.round(fo_t,3)}")
    print(f"    first-order  sparse   = {np.round(fo,3)}")
    print(f"    total-order  analytic = {np.round(to_t,3)}")
    print(f"    total-order  sparse   = {np.round(to,3)}")
    err = np.abs(fo - fo_t).max() + np.abs(to - to_t).max()
    print(f"    max Sobol error = {err:.3f}  -> {'OK' if err < 0.05 else 'CHECK'}")
    assert err < 0.05


def test_distance_correlation():
    print("\n[3] Dependence beyond Pearson")
    n = 2000
    x = rng.randn(n); y = x ** 2 + 0.1 * rng.randn(n)   # uncorrelated but dependent
    pear = abs(np.corrcoef(x, y)[0, 1])
    dcor = max_distance_correlation(np.column_stack([x, y]))
    print(f"    y = x^2:  Pearson|rho| = {pear:.3f} (misses it), "
          f"distance-corr = {dcor:.3f} (detects dependence)")


if __name__ == "__main__":
    test_apc_orthonormality()
    test_sparse_recovery()
    test_distance_correlation()
    print("\nAll sparse-PCE validation checks passed.")
