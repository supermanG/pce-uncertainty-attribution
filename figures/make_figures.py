"""Generate the six main-text figures from the committed result JSONs.

Colorblind-safe (Okabe-Ito) palette, publication styling. Outputs PDFs to figures/.
Second-round revision (September 2026): fonts embedded as TrueType (Type 42) so the PDFs
pass journal preflight, larger tick/legend/annotation fonts, panel titles that no longer
collide, spelled-out labels ("reward-model" instead of "RM", a proper beta), American
spelling in axis labels, and a framework schematic that states the input and output of
every stage. Final-submission pass (September 2026, editorial checklist): Fig. 2c shows the
bootstrap distribution of the Sobol' indices as box plots (n = 200 resamples, archived in
bh_bootstrap_samples.json) instead of bars with interval whiskers; Fig. 4a-c overlay the
five individual seeds on the seed means. No number changes: every value is read from the
archived result files.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CR = os.path.join(ROOT, "results", "cluster_results")
FIG = os.path.join(ROOT, "figures")
OK = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,        # embed TrueType, not Type 3
    "mathtext.default": "regular",
})
ANN = 7.5   # annotation font size


def load(p):
    return json.load(open(os.path.join(CR, p)))


def title(ax, txt, **kw):
    ax.set_title(txt, loc="left", fontweight="bold", **kw)


def fig_bh():
    bh = load("bh_final/ensemble_analysis.json"); p5 = bh["per_dim"]["5"]
    sh = load("bh_final/bh_shapley.json")
    steps = ["catalyst", "base", "aryl halide", "additive"]
    frag = [p5["fragility"][str(s)]["tv_barycenter"] for s in range(4)]
    fig, ax = plt.subplots(2, 2, figsize=(8.0, 6.3))
    ax[0, 0].bar(steps, frag, color=OK[0]); ax[0, 0].set_ylabel("fragility (TV from barycenter)")
    title(ax[0, 0], "a  Per-step decision fragility")
    ax[0, 0].tick_params(axis="x", rotation=20)
    solv = {"dense ridge": p5["solvers"]["ridge_hermite"]["mean_rel_mse"],
            "sparse LARS + aPC": p5["solvers"]["sparse_lars_apc"]["mean_rel_mse"],
            "sparse OMP + aPC": p5["solvers"]["sparse_omp_apc"]["mean_rel_mse"]}
    ax[0, 1].bar(list(solv), list(solv.values()), color=[OK[3], OK[2], OK[2]])
    ax[0, 1].axhline(1, ls=":", c="gray", lw=0.8); ax[0, 1].set_ylabel("test relative MSE")
    title(ax[0, 1], "b  Sparse versus dense surrogate")
    ax[0, 1].tick_params(axis="x", rotation=20)
    modes = [f"m{i+1}" for i in range(5)]
    # additive step: the 200 model-conditional bootstrap resamples of the first-order Sobol'
    # vector (bh_bootstrap_samples.py reproduces the archived mean / 5th / 95th percentiles)
    bs = load("bh_final/bh_bootstrap_samples.json")["steps"]["additive"]
    S = np.array(bs["samples"])                               # (n_boot, 5)
    shp = np.array(sh["additive"]["shapley"])[:5]
    x = np.arange(5); w = 0.4
    ax[1, 0].boxplot([S[:, i] for i in range(5)], positions=x - w/2, widths=w * 0.85,
                     whis=(5, 95), showmeans=True, patch_artist=True,
                     boxprops=dict(facecolor=OK[0], edgecolor=OK[0], alpha=0.85),
                     medianprops=dict(color="black", lw=1.0),
                     meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black",
                                    markersize=3.2, markeredgewidth=0.7),
                     whiskerprops=dict(color=OK[0], lw=0.9), capprops=dict(color=OK[0], lw=0.9),
                     flierprops=dict(marker="o", markersize=2.2, markerfacecolor=OK[0],
                                     markeredgecolor="none", alpha=0.7))
    ax[1, 0].bar(x + w/2, shp, w, color=OK[1])
    handles = [Patch(facecolor=OK[0], alpha=0.85, label=f"first-order Sobol', bootstrap (n = {S.shape[0]})"),
               Patch(facecolor=OK[1], label="Shapley effect (surrogate)")]
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(modes)
    ax[1, 0].legend(handles=handles, frameon=False, loc="upper left")
    ax[1, 0].set_ylim(0, max(S.max(), shp.max()) * 1.30)
    ax[1, 0].set_ylabel("share of decision variance")
    ax[1, 0].set_xlabel("uncertainty mode")
    title(ax[1, 0], "c  Additive step: Sobol' versus Shapley")
    dims = ["5", "7", "10"]
    rel = [bh["per_dim"][d]["solvers"]["sparse_lars_apc"]["mean_rel_mse"] for d in dims]
    ax[1, 1].plot([int(d) for d in dims], rel, "o-", color=OK[2])
    ax[1, 1].set_xlabel("retained PCA dimension $d$"); ax[1, 1].set_ylabel("test relative MSE")
    ax[1, 1].set_xticks([5, 6, 7, 8, 9, 10])
    ax[1, 1].set_ylim(0, max(rel) * 1.3); title(ax[1, 1], "d  Accuracy versus retained modes")
    fig.tight_layout(h_pad=2.0, w_pad=1.6); fig.savefig(os.path.join(FIG, "fig_bh_revised.pdf")); plt.close(fig)


def fig_bo():
    # real-data BO on the Doyle-Dreher reaction space (multi-seed); synthetic GP reference
    bo = load("bo_oed_real/bo_oed_real_v2_results.json")
    syn = load("bo_oed/bo_oed_results.json")
    fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.1))
    fo = bo["mode_first_order_mean"]; per_mode = np.array(bo["mode_first_order_per_seed"])   # (seeds, d)
    xm = np.arange(len(fo))
    ax[0].bar(xm, fo, color=OK[0], label=f"mean over seeds (n = {per_mode.shape[0]})")
    for j, xj in enumerate(xm):                               # individual seeds as open circles
        ax[0].scatter(xj + np.linspace(-0.2, 0.2, per_mode.shape[0]), per_mode[:, j], s=8,
                      facecolors="white", edgecolors="black", linewidths=0.6, zorder=3)
    ax[0].scatter([], [], s=8, facecolors="white", edgecolors="black", linewidths=0.6, label="individual seeds")
    ax[0].set_xticks(xm); ax[0].set_xticklabels([f"m{i+1}" for i in range(len(fo))])
    ax[0].set_ylim(0, per_mode.max() * 1.45); ax[0].legend(frameon=False, loc="upper right")
    ax[0].set_ylabel("first-order Sobol'"); ax[0].set_xlabel("posterior mode (KL, variance-ranked)")
    title(ax[0], "a  Acquisition attribution\n(real Doyle–Dreher)")
    conv = bo["convergence_relMSE_by_d"]; dd = sorted(int(k) for k in conv)
    ax[1].errorbar(dd, [conv[str(k)]["mean"] for k in dd], yerr=[conv[str(k)]["sd"] for k in dd],
                   fmt="o-", color=OK[2], capsize=2, label="real reaction space (mean ± s.d.)")
    per_d = bo["convergence_relMSE_by_d_per_seed"]
    for k in dd:                                              # individual seeds as open circles
        v = per_d[str(k)]
        ax[1].scatter(k + np.linspace(-0.25, 0.25, len(v)), v, s=8, facecolors="white",
                      edgecolors=OK[2], linewidths=0.6, zorder=3)
    ax[1].scatter([], [], s=8, facecolors="white", edgecolors=OK[2], linewidths=0.6, label="individual seeds")
    convs = syn["convergence_relMSE_by_d"]; dds = sorted(int(k) for k in convs)
    ax[1].plot(dds, [convs[str(k)] for k in dds], "s--", color=OK[1], label="synthetic GP (exact)")
    ax[1].axhline(1, ls=":", c="gray", lw=0.8)
    ax[1].set_xlabel("retained KL modes $d$"); ax[1].set_ylabel("test relative MSE")
    ax[1].set_ylim(0, 1.12); ax[1].legend(frameon=False, loc="upper right")
    title(ax[1], "b  Convergence in retained modes")
    labels = ["real\nDoyle–Dreher", "synthetic\nGP (exact)"]
    vals = [bo["relMSE"]["mean"], syn["relMSE"]]
    errs = [bo["relMSE"]["sd"], 0]
    ax[2].bar(labels, vals, yerr=errs, capsize=3, color=[OK[2], OK[1]])
    rs = bo["relMSE_per_seed"]                                # individual seeds as open circles
    ax[2].scatter(np.linspace(-0.15, 0.15, len(rs)), rs, s=10, facecolors="white",
                  edgecolors="black", linewidths=0.6, zorder=3, label="individual seeds")
    ax[2].legend(frameon=False, loc="upper right")
    ax[2].axhline(1, ls=":", c="gray", lw=0.8); ax[2].set_ylabel("test relative MSE")
    ax[2].set_ylim(0, 1.12)
    title(ax[2], "c  Surrogate accuracy")
    fig.tight_layout(w_pad=1.6); fig.savefig(os.path.join(FIG, "fig_bo_oed.pdf")); plt.close(fig)


def fig_rlhf():
    # real best-of-n study on hh-rlhf preferences (multi-seed, with gold reward)
    rl = load("rlhf_real/rlhf_real_v2_results.json")
    fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.1))
    # a: per-mode Sobol, strong (high-disagreement) vs weak (low-disagreement) prompts
    st = rl["mode_first_order_strong"]; wk = rl["mode_first_order_weak"]
    x = np.arange(len(st)); w = 0.4
    ax[0].bar(x - w/2, st, w, label="high reward-model disagreement", color=OK[3])
    ax[0].bar(x + w/2, wk, w, label="low reward-model disagreement", color=OK[5])
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"m{i+1}" for i in range(len(st))])
    ax[0].set_ylabel("first-order Sobol'"); ax[0].set_xlabel("reward-model uncertainty mode")
    ax[0].set_ylim(0, max(max(st), max(wk)) * 1.65)          # headroom so the legend clears the bars
    ax[0].legend(frameon=False, loc="upper right")
    ax[0].annotate("m1: uniform-offset direction,\nsoftmax-invariant (T6)", xy=(0 - w/2, st[0]),
                   xytext=(-0.45, max(st) * 1.08), fontsize=ANN, ha="left", va="bottom",
                   arrowprops=dict(arrowstyle="->", lw=0.6))
    title(ax[0], "a  Generation-decision attribution\n(real hh-rlhf)")
    # b: beta sweep (overoptimization) with error bars
    sw = rl["beta_sweep"]; betas = sorted((float(k) for k in sw), reverse=True)
    ax[1].errorbar([str(b) for b in betas], [sw[str(b)]["mean"] for b in betas],
                   yerr=[sw[str(b)]["sd"] for b in betas], fmt="o-", color=OK[3], capsize=2)
    ax[1].set_xlabel(r"KL penalty $\beta$ (optimization harder to the right)")
    ax[1].set_ylabel("fragility to reward-model uncertainty")
    title(ax[1], r"b  Overoptimization versus $\beta$")
    # c: proxy vs gold reward by n -- the true overoptimization signal
    pr = rl["proxy_reward_by_n"]; gr = rl["gold_reward_by_n"]
    ns = sorted(int(k) for k in pr)
    ax[2].plot(ns, [pr[str(k)]["mean"] for k in ns], "o-", color=OK[0], label="proxy reward (selected on)")
    ax[2].plot(ns, [gr[str(k)]["mean"] for k in ns], "s--", color=OK[3], label="gold reward (true)")
    ax[2].set_xscale("log", base=2); ax[2].set_xticks(ns); ax[2].set_xticklabels(ns)
    ax[2].set_xlabel("best-of-$n$ pool size $n$"); ax[2].set_ylabel("reward (standardized units)")
    ax[2].legend(frameon=False, loc="upper left")
    title(ax[2], "c  Reward overoptimization")
    fig.tight_layout(w_pad=1.6); fig.savefig(os.path.join(FIG, "fig_rlhf.pdf")); plt.close(fig)


def fig_voi():
    """Composite: (a-c) model-uncertainty variance vs decision value-of-information spectra
    across three pipelines; (d,e) retrospective acquisition curves; (f) the alignment gradient
    (Pearson correlation of the two spectra) from chemistry to RLHF."""
    sp = load("voi/voi_spectra.json"); aq = load("voi/voi_acquisition.json")
    fig, ax = plt.subplots(2, 3, figsize=(10.2, 6.0))
    specs = [("BH", "a  Chemistry (Buchwald–Hartwig)"), ("BO", "b  Experimental design (BO)"),
             ("RLHF", "c  Language-model alignment (RLHF)")]
    corrs = {}
    for k, (dom, ttl) in enumerate(specs):
        r = sp[dom]; vf = np.array(r["var_frac"]); vs = np.array(r["value_sobol_first"])
        x = np.arange(len(vf)); w = 0.42
        ax[0, k].bar(x - w / 2, vf, w, color=OK[5], label="model-uncertainty variance")
        ax[0, k].bar(x + w / 2, vs, w, color=OK[3], label="decision value-of-information")
        rc = float(np.corrcoef(vf, vs)[0, 1]); corrs[dom] = rc
        title(ax[0, k], ttl)
        ax[0, k].set_xlabel("uncertainty mode (variance-ranked)")
        ax[0, k].set_xticks(x); ax[0, k].set_xticklabels([f"{i+1}" for i in x])
        ax[0, k].set_ylim(0, max(max(vf), max(vs)) * 1.32)     # headroom for legend / label
        ax[0, k].text(0.03, 0.92, f"corr = {rc:+.2f}", transform=ax[0, k].transAxes, ha="left",
                      fontsize=8.5, fontweight="bold")
        if k == 0:
            ax[0, k].set_ylabel("share")
    ax[0, 2].annotate("mode 1: uniform-offset direction\n93% of the variance,\n8% of the value-of-information",
                      xy=(0 - w / 2, sp["RLHF"]["var_frac"][0]),
                      xytext=(0.9, 0.62), fontsize=ANN, arrowprops=dict(arrowstyle="->", lw=0.6))
    handles, labels = ax[0, 0].get_legend_handles_labels()      # one figure-level legend, clear of every panel
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.015), fontsize=8)
    # acquisition panels d, e
    sty = {"voi": ("value-of-information order", OK[3], "o-"),
           "variance": ("variance order", OK[0], "s--"), "random": ("random order", "gray", "^:")}
    for k, (dom, ttl) in enumerate([("BH", "d  Acquisition (chemistry)"),
                                    ("RLHF", "e  Acquisition (RLHF)")]):
        c = aq[dom]
        for name in ("voi", "variance", "random"):
            lab, col, ls = sty[name]
            ax[1, k].plot(np.arange(len(c[name])), c[name], ls, color=col, label=lab, markersize=3.5)
        title(ax[1, k], ttl)
        ax[1, k].set_xlabel("uncertainty modes resolved"); ax[1, k].set_ylim(0, 1.18)
        if k == 0:
            ax[1, k].set_ylabel("residual decision value-variance\n(held-out members)")
            ax[1, k].legend(frameon=False, loc="upper right")
    ax[1, 1].annotate("variance order:\nfirst acquisition\nresolves nothing",
                      xy=(1, aq["RLHF"]["variance"][1]), xytext=(1.5, 0.42), fontsize=ANN,
                      arrowprops=dict(arrowstyle="->", lw=0.6))
    # panel f: alignment gradient
    order = ["BH", "BO", "RLHF"]; labs = ["chemistry", "experimental\ndesign", "RLHF"]
    cols = [OK[2] if corrs[d] > 0.5 else (OK[1] if corrs[d] > -0.5 else OK[3]) for d in order]
    ax[1, 2].bar(labs, [corrs[d] for d in order], color=cols)
    ax[1, 2].axhline(0, color="black", lw=0.7); ax[1, 2].set_ylim(-1, 1)
    ax[1, 2].set_ylabel("corr(variance, value-of-information)")
    title(ax[1, 2], "f  Alignment gradient")
    fig.tight_layout(h_pad=2.0, w_pad=1.6, rect=(0, 0, 1, 0.965))
    fig.savefig(os.path.join(FIG, "fig_voi.pdf")); plt.close(fig)


def fig_bh_closedloop():
    cl = load("bh_closed_loop/bh_closed_loop.json")
    comps = ["catalyst", "base", "aryl_halide", "additive"]
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1))
    # a: per-step fragility from the reward-model ensemble (reproduces the ranking)
    frag = [cl["fragility"][c] for c in comps]
    ax[0].bar([c.replace("_", "\n") for c in comps], frag, color=OK[0])
    ax[0].set_ylabel("fragility (TV from barycenter)")
    title(ax[0], "a  Reward-model-ensemble fragility\n(measured yields)")
    # b: learning curve -- fragility vs fraction of reactions observed, per component
    lc = cl["learning_curve"]; fracs = sorted(float(k) for k in lc)
    palette = {"additive": OK[3], "aryl_halide": OK[1], "catalyst": OK[0], "base": OK[2]}
    for c in comps:
        ax[1].plot([100 * f for f in fracs], [lc[f"{f}" if f"{f}" in lc else str(f)][c] for f in fracs],
                   "o-", color=palette[c], label=c.replace("_", " "), markersize=3.5)
    ax[1].set_xlabel("reactions observed (% of dataset)")
    ax[1].set_ylabel("decision fragility")
    ax[1].legend(frameon=False)
    title(ax[1], "b  Fragility versus reactions observed")
    fig.tight_layout(w_pad=1.6); fig.savefig(os.path.join(FIG, "fig_bh_closedloop.pdf")); plt.close(fig)


def fig_framework():
    """Panel a of Fig. 1: the pipeline with the input and output of every stage. Panel b (the
    per-study instantiation table) is typeset in LaTeX beneath this panel."""
    fig, ax = plt.subplots(figsize=(11.0, 3.0)); ax.axis("off"); ax.set_xlim(0, 4); ax.set_ylim(0, 1)
    stages = [
        ("Objective-model\nuncertainty",
         "in:  $L$ realizations of the objective\n       model $R$, each evaluated on one\n       fixed reference set",
         "out: $\\mu \\in \\mathbb{R}^d$ (PCA scores of the\n       $L$ vectors, or exact KL modes)"),
        ("One policy\nper realization",
         "in:  realization $l$ (a trained GFlowNet,\n       an acquisition rule on a posterior\n       draw, or a best-of-$n$ selection)",
         "out: action distribution $\\pi_l(\\cdot\\,|\\,s_t)$\n       at each decision $t$ on the path"),
        ("Sparse / arbitrary PCE\nper decision",
         "in:  $(\\mu_l,\\ \\log p_k/p_K)$ over the training\n       members, one expansion per\n       log-ratio $k = 1,\\ldots,K-1$",
         "out: coefficients $c_{\\mathbf{j}}$; held-out\n       error on the test members"),
        ("Read-outs",
         "Sobol' indices $S_i$, $T_i$ (bootstrap CI),\nmean over the $K-1$ log-ratios;\nfragility $\\mathcal{F}_t$; value-of-information\nspectrum from the value surrogate",
         "diagnostic: held-out error,\ntraining-noise fraction $\\phi$"),
    ]
    for i, (head, inp, out) in enumerate(stages):
        x0 = i + 0.03
        ax.add_patch(FancyBboxPatch((x0, 0.04), 0.94, 0.90, boxstyle="round,pad=0.01",
                                    fc=OK[i % len(OK)], ec="black", alpha=0.16, lw=1))
        ax.add_patch(FancyBboxPatch((x0, 0.74), 0.94, 0.20, boxstyle="round,pad=0.01",
                                    fc=OK[i % len(OK)], ec="none", alpha=0.30, lw=0))
        ax.text(x0 + 0.47, 0.84, head, ha="center", va="center", fontsize=8.6, fontweight="bold",
                linespacing=1.1)
        ax.text(x0 + 0.04, 0.68, inp, ha="left", va="top", fontsize=7.1, linespacing=1.25)
        ax.text(x0 + 0.04, 0.31, out, ha="left", va="top", fontsize=7.1, linespacing=1.25)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x0 + 0.945, 0.5), (x0 + 1.0, 0.5), arrowstyle="-|>",
                                         mutation_scale=14, lw=1.3, color="black"))
    ax.text(-0.02, 1.0, "a", fontsize=10, fontweight="bold", ha="left", va="bottom")
    fig.savefig(os.path.join(FIG, "fig_framework.pdf")); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_framework(); fig_bh(); fig_bh_closedloop(); fig_bo(); fig_rlhf(); fig_voi()
    print("wrote:", ["fig_framework", "fig_bh_revised", "fig_bh_closedloop",
                     "fig_bo_oed", "fig_rlhf", "fig_voi"])
