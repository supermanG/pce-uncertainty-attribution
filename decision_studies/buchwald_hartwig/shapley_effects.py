"""Given-data Shapley-effect estimator + analytic validation.

Shapley effects are the recommended dependent-input attribution (non-negative, in [0,1],
sum to one under dependence), computed on the fitted surrogate. Here we implement the
exhaustive given-data estimator (Broto-Bachoc-Depecker 2020; Song-Nelson-Staum 2016 game)
and VALIDATE it against a linear-Gaussian case with closed-form Shapley effects, at
several sample sizes, to determine whether it is reliable at L=50 before we trust it on
Buchwald-Hartwig. c(S) = Var(E[Y|X_S]); Shapley by the exact subset formula (d small).
"""
import itertools, math
import numpy as np


def _cond_var_given_data(X, Y, S, k):
    """Given-data estimate of Var(E[Y|X_S]): mean Y over kNN in standardized X_S space,
    then variance of those neighbourhood means."""
    n = len(Y)
    if len(S) == 0:
        return 0.0
    Xs = X[:, list(S)]
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
    m = np.empty(n)
    for i in range(n):
        d2 = ((Xs - Xs[i]) ** 2).sum(1)
        idx = np.argpartition(d2, min(k, n - 1))[:k]
        m[i] = Y[idx].mean()
    return float(np.var(m))


def shapley_given_data(X, Y, k=None):
    n, d = X.shape
    k = k or max(3, int(round(np.sqrt(n))))
    varY = float(np.var(Y))
    # cache c(S) for all subsets
    c = {}
    for r in range(d + 1):
        for S in itertools.combinations(range(d), r):
            if len(S) == 0:
                c[S] = 0.0
            elif len(S) == d:
                c[S] = varY
            else:
                c[S] = _cond_var_given_data(X, Y, S, k)
    sh = np.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for r in range(d):
            w = math.factorial(r) * math.factorial(d - r - 1) / math.factorial(d)
            for S in itertools.combinations(others, r):
                sh[i] += w * (c[tuple(sorted(S + (i,)))] - c[tuple(sorted(S))])
    return sh / (varY + 1e-12)   # normalised Shapley effects (sum to ~1)


def shapley_on_surrogate(surr, mu):
    """Per-mode Shapley effects computed on the SURROGATE prediction (denoised), the
    metamodel-based route (Iooss-Prieur 2019). Averages per-ALR-component Shapley weighted
    by the component's variance, so near-constant/noisy components do not dilute the signal.
    Computing Shapley on the raw ALR instead is noise-washed and gives near-uniform indices.
    """
    from core.sparse_pce import build_design_matrix_general
    Phi = build_design_matrix_general(mu, surr.multi_idx, surr.degree, surr.basis,
                                      surr._apc_R, surr._apc_stats)
    K = surr.n_actions
    alr = np.column_stack([Phi @ surr.coefficients[k] for k in range(K - 1)])
    w = alr.var(0); w = w / (w.sum() + 1e-12)
    sh = np.zeros(mu.shape[1])
    for k in range(K - 1):
        if w[k] > 1e-6:
            sh += w[k] * shapley_given_data(mu, alr[:, k])
    return sh


def analytic_linear_gaussian_shapley(beta, Sigma):
    """Exact Shapley effects for Y = beta^T X, X ~ N(0, Sigma).
    c(S) = Var(E[Y|X_S]) with E[Y|X_S] linear via the Gaussian conditional."""
    d = len(beta)
    def cS(S):
        if len(S) == 0:
            return 0.0
        S = list(S); Sc = [j for j in range(d) if j not in S]
        Ss = Sigma[np.ix_(S, S)]
        bS = beta[S]
        if Sc:
            bSc = beta[Sc]
            A = Sigma[np.ix_(S, Sc)] @ np.linalg.solve(Sigma[np.ix_(Sc, Sc)] * 0 + Ss, np.zeros((len(S), 0))) if False else None
            # E[X_Sc | X_S] = Sigma[Sc,S] Ss^-1 X_S ; coefficient on X_S is (bS + Sigma[S,Sc]... )
            coef = bS + np.linalg.solve(Ss, Sigma[np.ix_(S, Sc)] @ bSc)
        else:
            coef = bS
        return float(coef @ Ss @ coef)
    varY = float(beta @ Sigma @ beta)
    sh = np.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for r in range(d):
            w = math.factorial(r) * math.factorial(d - r - 1) / math.factorial(d)
            for S in itertools.combinations(others, r):
                sh[i] += w * (cS(sorted(S + (i,))) - cS(sorted(S)))
    return sh / varY


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    d = 3
    beta = np.array([1.0, 0.6, 0.3])
    rho = 0.6
    Sigma = np.array([[1, rho, 0.2], [rho, 1, 0.0], [0.2, 0.0, 1]], float)
    true = analytic_linear_gaussian_shapley(beta, Sigma)
    print(f"analytic Shapley effects: {np.round(true,3).tolist()}  (sum {true.sum():.3f})")
    L = np.linalg.cholesky(Sigma)
    print(f"{'n':>6} | {'estimate':>26} | max abs err")
    for n in [50, 200, 1000, 5000]:
        errs = []
        for rep in range(5):
            X = rng.randn(n, d) @ L.T
            Y = X @ beta + 1e-6 * rng.randn(n)
            sh = shapley_given_data(X, Y)
            errs.append(np.abs(sh - true).max())
        # one representative estimate
        X = rng.randn(n, d) @ L.T; Y = X @ beta
        sh = shapley_given_data(X, Y)
        print(f"{n:>6} | {np.round(sh,3).tolist()!s:>26} | {np.mean(errs):.3f} (mean of 5)")
