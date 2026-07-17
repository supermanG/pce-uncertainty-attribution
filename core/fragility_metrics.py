"""Scale-invariant policy-fragility metrics and corrected interaction reporting.

Motivation: the manuscript's "total policy variance" D is a
sum over K-1 ALR-component variances, so it grows with the action-space cardinality
K and is not comparable across trajectory steps with different K. We define fragility
directly on the probability simplex, where every step is measured on the same [0,1]
scale regardless of K.

Given an ensemble of per-member action distributions P = {p^(1), ..., p^(L)},
each p^(l) in the K-simplex, we report:

  tv_barycenter    mean_l TV(p^(l), pbar)          pbar = mean member distribution
                   TV(p,q) = 0.5 * sum_k |p_k - q_k|  in [0,1]
  tv_pairwise      mean_{l<m} TV(p^(l), p^(m))     in [0,1]
  js_spread        mean_l JSD(p^(l) || pbar)        generalised Jensen-Shannon, in [0, log2 K] -> report in bits
                   (also normalised by log2(K) to a [0,1] comparability score)

These are invariant to relabelling and comparable across K, unlike raw ALR variance D.

Corrected interaction reporting: for each scalar output (one ALR
component's PCE), the Sobol indices of all orders sum to 1, so with d inputs the
interaction fraction is  1 - sum_i S_i^(first-order)  (equivalently, for d=2,
(sum_i S_Ti) - 1). This replaces the incorrect "total-order indices sum to <= 1"
and the "near-additive / negligible interactions" claim.

Everything here operates on cached/regenerated per-member policy arrays; no PCE
refit is required, so the same code applies to any embedding or solver we switch to.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Simplex fragility metrics
# ---------------------------------------------------------------------------

def _normalise(P, eps=1e-12):
    P = np.asarray(P, dtype=float)
    P = np.clip(P, eps, None)
    return P / P.sum(axis=1, keepdims=True)


def tv_barycenter(P):
    """Mean total-variation distance from members to the ensemble barycentre. [0,1]."""
    P = _normalise(P)
    pbar = P.mean(axis=0, keepdims=True)
    return float(np.mean(0.5 * np.abs(P - pbar).sum(axis=1)))


def tv_pairwise(P):
    """Mean pairwise total-variation distance across members. [0,1]."""
    P = _normalise(P)
    L = P.shape[0]
    if L < 2:
        return 0.0
    tot, n = 0.0, 0
    for i in range(L):
        d = 0.5 * np.abs(P[i + 1:] - P[i]).sum(axis=1)
        tot += d.sum(); n += d.shape[0]
    return float(tot / n)


def _entropy(p):
    p = np.clip(p, 1e-12, None)
    return float(-(p * np.log2(p)).sum())


def js_spread(P, normalise=True):
    """Generalised Jensen-Shannon spread: H(pbar) - mean_l H(p^(l)), in bits.

    Equals the mean KL(p^(l) || pbar) up to the standard JSD identity; bounded by
    log2(K). If normalise, also divide by log2(K) for a K-comparable [0,1] score.
    """
    P = _normalise(P)
    pbar = P.mean(axis=0)
    jsd = _entropy(pbar) - float(np.mean([_entropy(p) for p in P]))
    jsd = max(jsd, 0.0)
    if normalise:
        K = P.shape[1]
        return jsd / np.log2(K) if K > 1 else 0.0
    return jsd


def fragility_report(P):
    """All scale-invariant fragility scalars for one ensemble (L, K)."""
    return dict(
        L=int(P.shape[0]), K=int(P.shape[1]),
        tv_barycenter=tv_barycenter(P),
        tv_pairwise=tv_pairwise(P),
        js_spread_bits=js_spread(P, normalise=False),
        js_spread_norm=js_spread(P, normalise=True),
    )


# ---------------------------------------------------------------------------
# Corrected Sobol interaction reporting
# ---------------------------------------------------------------------------

def interaction_fraction(first_order):
    """Interaction fraction per output = 1 - sum_i S_i (first-order), clipped to [0,1].

    first_order: (n_outputs, d). Returns (n_outputs,) interaction fractions.
    """
    fo = np.asarray(first_order, dtype=float)
    return np.clip(1.0 - fo.sum(axis=1), 0.0, 1.0)


def interaction_summary(sobol_step):
    """sobol_step: {'first_order':(K-1,d), 'total_order':(K-1,d), 'variance':(K-1,)}.

    Returns dict with mean first-order sum, mean total-order sum, and the two
    equivalent interaction estimates (should agree for d=2).
    """
    fo = np.asarray(sobol_step["first_order"], dtype=float)
    to = np.asarray(sobol_step["total_order"], dtype=float)
    fo_sum = fo.sum(axis=1)
    to_sum = to.sum(axis=1)
    return dict(
        first_order_sum_mean=float(fo_sum.mean()),
        total_order_sum_mean=float(to_sum.mean()),
        interaction_from_first=float(np.mean(1.0 - fo_sum)),
        interaction_from_total=float(np.mean(to_sum - 1.0)),
    )


# ---------------------------------------------------------------------------
# Ensemble loader (works on regenerated member npz files)
# ---------------------------------------------------------------------------

def load_member_policies(members_dir, n_steps=None):
    """Load per-step policy arrays from a directory of member_*.npz files.

    Returns {step -> (L, K)} stacked across members.
    """
    from pathlib import Path
    files = sorted(Path(members_dir).glob("member_*.npz"))
    if not files:
        raise FileNotFoundError(f"no member_*.npz in {members_dir}")
    data = [np.load(f, allow_pickle=True) for f in files]
    if n_steps is None:
        n_steps = sum(1 for k in data[0].files if k.startswith("policy_step"))
    return {s: np.stack([d[f"policy_step{s}"] for d in data], axis=0)
            for s in range(n_steps)}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(0)

    # A: near-deterministic ensemble (robust step) -> low fragility, any K
    A = _normalise(np.tile([10, 1, 1, 1], (40, 1)) + rng.rand(40, 4) * 0.05)
    # B: highly dispersed ensemble (fragile step), same K
    B = _normalise(rng.rand(40, 4) ** 3)
    # C: dispersed but with LARGER K (to show scale-invariance vs raw variance)
    C = _normalise(rng.rand(40, 24) ** 3)

    print("robust  K=4 :", {k: round(v, 3) if isinstance(v, float) else v
                             for k, v in fragility_report(A).items()})
    print("fragile K=4 :", {k: round(v, 3) if isinstance(v, float) else v
                             for k, v in fragility_report(B).items()})
    print("fragile K=24:", {k: round(v, 3) if isinstance(v, float) else v
                             for k, v in fragility_report(C).items()})
    print("\nTV-barycentre orders robust<fragile regardless of K:",
          tv_barycenter(A) < tv_barycenter(B), "|",
          f"K=4 fragile {tv_barycenter(B):.3f} vs K=24 fragile {tv_barycenter(C):.3f} "
          "(comparable scale)")

    # interaction reporting sanity: fabricate d=2 indices with known interaction
    fo = np.array([[0.3, 0.3], [0.5, 0.1]])           # sums 0.6, 0.6 -> interaction 0.4
    to = np.array([[0.7, 0.7], [0.9, 0.5]])           # sums 1.4, 1.4 -> interaction 0.4
    s = interaction_summary({"first_order": fo, "total_order": to, "variance": [1, 1]})
    print("\ninteraction check (expect ~0.4 both ways):",
          {k: round(v, 3) for k, v in s.items()})
