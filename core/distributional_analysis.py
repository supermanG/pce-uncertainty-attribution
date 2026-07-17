"""Distributional analysis utilities for GFlowNet policy UQ.

Beyond scalar Sobol indices, the PCE surrogate is a full generative model
of the policy distribution.  This module provides tools to analyse the
*shape* of that distribution -- detecting bimodality, characterising
behavioural regimes, and generating publication-quality figures.

Why this matters
~~~~~~~~~~~~~~~~
A Sobol index tells you *how much* of the policy variance comes from a
given reward component, but it says nothing about the *structure* of that
variance.  A unimodal distribution with high variance and a bimodal
distribution with the same variance are qualitatively different:

  - Unimodal: the policy shifts smoothly as the reward parameter changes.
  - Bimodal: there is a *bifurcation* -- some reward configurations lead
    to one behavioural regime (e.g. confident termination) while others
    lead to a qualitatively different regime (e.g. continued exploration).

The bimodal case is directly actionable: it identifies the parameter region
where additional data collection would most reduce structural ambiguity.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Bimodality detection
# ---------------------------------------------------------------------------

def detect_bimodality(
    samples: np.ndarray,
    n_bins: int = 50,
    prominence_ratio: float = 0.15,
) -> dict:
    """Detect bimodality in a 1-D sample via histogram peak finding.

    A distribution is considered bimodal if two peaks are found with the
    valley between them at least `prominence_ratio` below the lower peak.

    Args:
        samples:          (N,) array of scalar samples.
        n_bins:           histogram resolution.
        prominence_ratio: minimum dip depth to declare bimodality.

    Returns:
        dict with:
            is_bimodal:     bool
            n_modes:        int
            dip_score:      float in [0,1] (higher = deeper valley)
            mode_locations: list of float (peak centres)
            density:        (n_bins,) smoothed density values
            bin_centres:    (n_bins,) bin centre coordinates
    """
    from scipy.signal import find_peaks

    counts, edges = np.histogram(samples, bins=n_bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.array([0.25, 0.5, 0.25])
    smoothed = np.convolve(counts, kernel, mode="same")

    peaks, _ = find_peaks(smoothed, prominence=0)
    if len(peaks) < 2:
        return {
            "is_bimodal": False,
            "n_modes": max(len(peaks), 1),
            "dip_score": 0.0,
            "mode_locations": [float(centres[p]) for p in peaks],
            "density": smoothed,
            "bin_centres": centres,
        }

    sorted_peaks = sorted(peaks, key=lambda p: smoothed[p], reverse=True)[:2]
    p1, p2 = sorted(sorted_peaks)
    valley = smoothed[p1 : p2 + 1].min()
    lower_peak = min(smoothed[p1], smoothed[p2])
    dip = 1.0 - valley / lower_peak if lower_peak > 0 else 0.0

    return {
        "is_bimodal": dip > prominence_ratio and len(peaks) >= 2,
        "n_modes": len(peaks),
        "dip_score": float(dip),
        "mode_locations": [float(centres[p]) for p in sorted_peaks],
        "density": smoothed,
        "bin_centres": centres,
    }


def analyse_trajectory_bimodality(
    surr_samples: Dict[int, np.ndarray],
    action_idx: int,
    step_labels: Optional[Dict[int, str]] = None,
) -> Tuple[Dict[int, dict], str]:
    """Analyse bimodality of a specific action across all trajectory steps.

    Args:
        surr_samples: dict step -> (n_mc, K) policy samples from the surrogate.
        action_idx:   which action column to analyse (e.g. EOS index).
        step_labels:  optional step -> label mapping for output text.

    Returns:
        bimodality: dict step -> detect_bimodality result.
        narrative:  multi-line human-readable summary.
    """
    bimodality = {}
    lines = []
    lines.append("DISTRIBUTIONAL ANALYSIS (beyond Sobol)")
    lines.append("=" * 60)
    lines.append(
        "  The PCE surrogate is a full generative model of policy"
    )
    lines.append(
        "  uncertainty.  Below: bimodality test on the target action"
    )
    lines.append(
        "  marginal at each step.  Bimodality = distinct behavioural regimes."
    )
    lines.append("")

    found_bimodal = False
    for s in sorted(surr_samples.keys()):
        # The target action index may not exist at every step (steps can have
        # different action-space sizes, e.g. Buchwald-Hartwig 4/3/16/24); skip
        # steps where it is out of range rather than indexing past the end.
        if action_idx >= surr_samples[s].shape[1]:
            continue
        p_action = surr_samples[s][:, action_idx]
        bm = detect_bimodality(p_action)
        bimodality[s] = bm
        tag = "BIMODAL" if bm["is_bimodal"] else "unimodal"
        if bm["is_bimodal"]:
            found_bimodal = True
        label = step_labels[s] if step_labels and s in step_labels else f"Step {s}"
        lines.append(
            f"  {label:20s}: {tag}  "
            f"(dip={bm['dip_score']:.3f}, modes={bm['n_modes']}, "
            f"range=[{p_action.min():.3f}, {p_action.max():.3f}])"
        )

    lines.append("")
    if found_bimodal:
        lines.append(
            "  => Bifurcation detected: epistemic uncertainty in the reward"
        )
        lines.append(
            "     model induces qualitatively different behavioural regimes."
        )
        lines.append(
            "     This structural insight is invisible to scalar Sobol indices"
        )
        lines.append(
            "     and directly identifies where additional data would most"
        )
        lines.append("     reduce ambiguity.")
    else:
        lines.append("  => No strong bimodality detected at these settings.")

    return bimodality, "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_bimodality_panel(
    surr_samples: Dict[int, np.ndarray],
    action_idx: int,
    step_labels: Optional[Dict[int, str]] = None,
    action_name: str = "stop/EOS",
    title: str = "Policy distributional structure across trajectory",
    empirical_samples: Optional[Dict[int, np.ndarray]] = None,
    output_path: Optional[str] = None,
    max_cols: int = 4,
) -> "matplotlib.figure.Figure":
    """Generate a publication-quality panel of histograms showing the
    target-action marginal distribution at each trajectory step.

    Bimodal steps are highlighted with a red border.  Optionally overlays
    the empirical ensemble distribution for comparison.

    Args:
        surr_samples:      dict step -> (n_mc, K) PCE surrogate samples.
        action_idx:        which action to plot.
        step_labels:       optional step -> label mapping.
        action_name:       name for the action (used in axis labels).
        title:             overall figure title.
        empirical_samples: optional dict step -> (n_test, K) real ensemble
                           policies, overlaid as a second histogram.
        output_path:       if given, save figure as PDF here.
        max_cols:          max columns in the panel grid.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    # Only panel steps where the target action index exists (steps can have
    # different action-space sizes, e.g. Buchwald-Hartwig 4/3/16/24).
    steps = [s for s in sorted(surr_samples.keys())
             if action_idx < surr_samples[s].shape[1]]
    n_steps = len(steps)
    n_cols = min(n_steps, max_cols)
    n_rows = int(np.ceil(n_steps / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )

    for idx, s in enumerate(steps):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        p_action = surr_samples[s][:, action_idx]
        bm = detect_bimodality(p_action)

        # Surrogate histogram
        ax.hist(
            p_action, bins=40, density=True, alpha=0.7,
            color="#4878CF", edgecolor="white", linewidth=0.3,
            label="PCE surrogate",
        )

        # Empirical overlay
        if empirical_samples is not None and s in empirical_samples:
            p_emp = empirical_samples[s][:, action_idx]
            ax.hist(
                p_emp, bins=15, density=True, alpha=0.5,
                color="#E8A838", edgecolor="white", linewidth=0.3,
                label="Empirical ensemble",
            )

        # Smoothed density line
        ax.plot(
            bm["bin_centres"], bm["density"],
            color="#2E3440", linewidth=1.2, alpha=0.8,
        )

        # Mark modes
        for ml in bm["mode_locations"]:
            ax.axvline(ml, color="#BF616A", linestyle="--", linewidth=0.8, alpha=0.6)

        # Highlight bimodal panels
        if bm["is_bimodal"]:
            for spine in ax.spines.values():
                spine.set_edgecolor("#BF616A")
                spine.set_linewidth(2.0)

        label = step_labels[s] if step_labels and s in step_labels else f"Step {s}"
        tag = " [BIMODAL]" if bm["is_bimodal"] else ""
        ax.set_title(f"{label}{tag}", fontsize=9, fontweight="bold" if bm["is_bimodal"] else "normal")
        ax.set_xlabel(f"P({action_name})", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(n_steps, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    # Legend on first panel
    if empirical_samples is not None:
        axes[0][0].legend(fontsize=7, loc="upper right")

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"  Figure saved to {output_path}")

    return fig


def plot_surrogate_comparison(
    surr_samples: Dict[int, np.ndarray],
    empirical_samples: Dict[int, np.ndarray],
    action_idx: int,
    step_labels: Optional[Dict[int, str]] = None,
    action_name: str = "action",
    title: str = "PCE surrogate vs empirical ensemble",
    output_path: Optional[str] = None,
    max_cols: int = 4,
) -> "matplotlib.figure.Figure":
    """Side-by-side comparison of surrogate and empirical distributions.

    For each step, plots overlapping histograms of the surrogate and
    empirical action-probability distributions, plus a KS statistic.

    Args:
        surr_samples:      dict step -> (n_mc, K) PCE surrogate samples.
        empirical_samples: dict step -> (n_test, K) real ensemble policies.
        action_idx:        which action to compare.
        step_labels:       optional step -> label mapping.
        action_name:       name for the action.
        title:             figure title.
        output_path:       if given, save as PDF.
        max_cols:          max columns.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp

    steps = sorted(set(surr_samples.keys()) & set(empirical_samples.keys()))
    n_steps = len(steps)
    n_cols = min(n_steps, max_cols)
    n_rows = int(np.ceil(n_steps / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )

    for idx, s in enumerate(steps):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        p_surr = surr_samples[s][:, action_idx]
        p_emp = empirical_samples[s][:, action_idx]

        # Shared bin edges
        lo = min(p_surr.min(), p_emp.min())
        hi = max(p_surr.max(), p_emp.max())
        bins = np.linspace(lo, hi, 35)

        ax.hist(p_surr, bins=bins, density=True, alpha=0.6,
                color="#4878CF", edgecolor="white", linewidth=0.3,
                label="PCE surrogate")
        ax.hist(p_emp, bins=bins, density=True, alpha=0.5,
                color="#E8A838", edgecolor="white", linewidth=0.3,
                label="Empirical")

        stat, pval = ks_2samp(p_surr, p_emp)
        label = step_labels[s] if step_labels and s in step_labels else f"Step {s}"
        ax.set_title(f"{label}\nKS={stat:.3f}, p={pval:.3f}", fontsize=8)
        ax.set_xlabel(f"P({action_name})", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(n_steps, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    axes[0][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"  Figure saved to {output_path}")

    return fig
