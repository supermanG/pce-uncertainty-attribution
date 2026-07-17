"""Generate the four manuscript figures from the committed result JSONs.
Colourblind-safe (Okabe-Ito) palette, clean publication styling. Outputs PDFs to figures/.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CR = os.path.join(ROOT, "results", "cluster_results")
FIG = os.path.join(ROOT, "figures")
OK = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(p):
    return json.load(open(os.path.join(CR, p)))


def fig_bh():
    bh = load("bh_final/ensemble_analysis.json"); p5 = bh["per_dim"]["5"]
    sh = load("bh_final/bh_shapley.json")
    steps = ["catalyst", "base", "aryl halide", "additive"]
    frag = [p5["fragility"][str(s)]["tv_barycenter"] for s in range(4)]
    fig, ax = plt.subplots(2, 2, figsize=(7.6, 6.2))
    ax[0, 0].bar(steps, frag, color=OK[0]); ax[0, 0].set_ylabel("fragility (TV from barycentre)")
    ax[0, 0].set_title("a  Per-step decision fragility", loc="left", fontweight="bold")
    ax[0, 0].tick_params(axis="x", rotation=20)
    solv = {"dense ridge": p5["solvers"]["ridge_hermite"]["mean_rel_mse"],
            "sparse LARS+aPC": p5["solvers"]["sparse_lars_apc"]["mean_rel_mse"],
            "sparse OMP+aPC": p5["solvers"]["sparse_omp_apc"]["mean_rel_mse"]}
    ax[0, 1].bar(list(solv), list(solv.values()), color=[OK[3], OK[2], OK[2]])
    ax[0, 1].axhline(1, ls=":", c="gray", lw=0.8); ax[0, 1].set_ylabel("test relative MSE")
    ax[0, 1].set_title("b  Sparse vs dense surrogate", loc="left", fontweight="bold")
    ax[0, 1].tick_params(axis="x", rotation=20)
    modes = [f"m{i+1}" for i in range(5)]
    bfa = p5["bootstrap_first_order"]["3"]                  # additive step: first-order Sobol + 90% CI
    so = np.array(bfa["mean"]); lo = np.array(bfa["lo"]); hi = np.array(bfa["hi"])
    shp = np.array(sh["additive"]["shapley"])[:5]
    x = np.arange(5); w = 0.4
    ax[1, 0].bar(x - w/2, so, w, yerr=[so - lo, hi - so], capsize=2,
                 label="Sobol 1st-order (90% CI)", color=OK[0])
    ax[1, 0].bar(x + w/2, shp, w, label="Shapley (surrogate)", color=OK[1])
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(modes); ax[1, 0].legend(frameon=False, fontsize=6.5)
    ax[1, 0].set_ylabel("attribution")
    ax[1, 0].set_title("c  Additive step: Sobol (90% CI) vs Shapley", loc="left", fontweight="bold")
    dims = ["5", "7", "10"]
    rel = [bh["per_dim"][d]["solvers"]["sparse_lars_apc"]["mean_rel_mse"] for d in dims]
    ax[1, 1].plot([int(d) for d in dims], rel, "o-", color=OK[2])
    ax[1, 1].set_xlabel("retained PCA dimension"); ax[1, 1].set_ylabel("test relative MSE")
    ax[1, 1].set_ylim(0, max(rel) * 1.3); ax[1, 1].set_title("d  Accuracy vs retained modes", loc="left", fontweight="bold")
    for a in ax.flat:
        a.title.set_fontsize(8.5)
    fig.tight_layout(h_pad=1.8, w_pad=1.4); fig.savefig(os.path.join(FIG, "fig_bh_revised.pdf")); plt.close(fig)


def fig_bo():
    # real-data BO on the Doyle-Dreher reaction space (multi-seed); synthetic GP reference
    bo = load("bo_oed_real/bo_oed_real_v2_results.json")
    syn = load("bo_oed/bo_oed_results.json")
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
    fo = bo["mode_first_order_mean"]
    ax[0].bar([f"m{i+1}" for i in range(len(fo))], fo, color=OK[0])
    ax[0].set_ylabel("first-order Sobol")
    ax[0].set_title("a  Acquisition-decision attribution\n(real Doyle-Dreher)", loc="left", fontweight="bold")
    ax[0].tick_params(axis="x", labelsize=6)
    conv = bo["convergence_relMSE_by_d"]; dd = sorted(int(k) for k in conv)
    ax[1].errorbar(dd, [conv[str(k)]["mean"] for k in dd], yerr=[conv[str(k)]["sd"] for k in dd],
                   fmt="o-", color=OK[2], capsize=2, label="real reaction space")
    convs = syn["convergence_relMSE_by_d"]; dds = sorted(int(k) for k in convs)
    ax[1].plot(dds, [convs[str(k)] for k in dds], "s--", color=OK[1], label="synthetic GP (exact)")
    ax[1].axhline(1, ls=":", c="gray", lw=0.8)
    ax[1].set_xlabel("retained KL modes"); ax[1].set_ylabel("test relative MSE")
    ax[1].legend(frameon=False, fontsize=6.5)
    ax[1].set_title("b  Convergence in modes", loc="left", fontweight="bold")
    labels = ["real\nDoyle-Dreher", "synthetic\nGP (exact)"]
    vals = [bo["relMSE"]["mean"], syn["relMSE"]]
    errs = [bo["relMSE"]["sd"], 0]
    ax[2].bar(labels, vals, yerr=errs, capsize=3, color=[OK[2], OK[1]])
    ax[2].axhline(1, ls=":", c="gray", lw=0.8); ax[2].set_ylabel("test relative MSE")
    ax[2].tick_params(axis="x", labelsize=6.5)
    ax[2].set_title("c  Surrogate accuracy", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_bo_oed.pdf")); plt.close(fig)


def fig_rlhf():
    # real best-of-n study on hh-rlhf preferences (multi-seed, with gold reward)
    rl = load("rlhf_real/rlhf_real_v2_results.json")
    fig, ax = plt.subplots(1, 3, figsize=(9.8, 2.9))
    # a: per-mode Sobol, strong (high-disagreement) vs weak (low-disagreement) prompts
    st = rl["mode_first_order_strong"]; wk = rl["mode_first_order_weak"]
    x = np.arange(len(st)); w = 0.4
    ax[0].bar(x - w/2, st, w, label="high RM disagreement", color=OK[3])
    ax[0].bar(x + w/2, wk, w, label="low RM disagreement", color=OK[5])
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"m{i+1}" for i in range(len(st))])
    ax[0].set_ylabel("first-order Sobol")
    ax[0].set_ylim(0, max(max(st), max(wk)) * 1.5)          # headroom so the legend clears the bars
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper center", ncol=1)
    ax[0].annotate("shift-invariant\nmode (ignored)", xy=(0 - w/2, st[0]), xytext=(-0.35, max(st) * 1.06),
                   fontsize=6, ha="left", va="bottom", arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[0].set_title("a  Generation-decision attribution\n(real hh-rlhf)", loc="left", fontweight="bold")
    # b: beta sweep (overoptimization) with error bars
    sw = rl["beta_sweep"]; betas = sorted((float(k) for k in sw), reverse=True)
    ax[1].errorbar([str(b) for b in betas], [sw[str(b)]["mean"] for b in betas],
                   yerr=[sw[str(b)]["sd"] for b in betas], fmt="o-", color=OK[3], capsize=2)
    ax[1].set_xlabel("KL penalty beta (harder to the right)"); ax[1].set_ylabel("fragility to RM uncertainty")
    ax[1].set_title("b  Overoptimization vs beta", loc="left", fontweight="bold")
    # c: proxy vs gold reward by n -- the true overoptimization signal
    pr = rl["proxy_reward_by_n"]; gr = rl["gold_reward_by_n"]
    ns = sorted(int(k) for k in pr)
    ax[2].plot(ns, [pr[str(k)]["mean"] for k in ns], "o-", color=OK[0], label="proxy reward (selected on)")
    ax[2].plot(ns, [gr[str(k)]["mean"] for k in ns], "s--", color=OK[3], label="gold reward (true)")
    ax[2].set_xscale("log", base=2); ax[2].set_xticks(ns); ax[2].set_xticklabels(ns)
    ax[2].set_xlabel("best-of-n pool size"); ax[2].set_ylabel("reward (standardised)")
    ax[2].legend(frameon=False, fontsize=6.5)
    ax[2].set_title("c  Reward overoptimization", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_rlhf.pdf")); plt.close(fig)


def fig_voi_spectra():
    """Centerpiece: model-uncertainty variance spectrum vs decision value-of-information
    spectrum, across three pipelines. The gap between them (aligned in chemistry, anti-aligned
    in RLHF) is the message."""
    voi = load("voi/voi_spectra.json")
    doms = [("BH", "a  Chemistry (Buchwald-Hartwig)"),
            ("BO", "b  Experimental design (BO)"),
            ("RLHF", "c  Alignment (RLHF)")]
    fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.0))
    for k, (dom, title) in enumerate(doms):
        r = voi[dom]
        vf = np.array(r["var_frac"]); vs = np.array(r["value_sobol_first"])
        x = np.arange(len(vf)); w = 0.42
        ax[k].bar(x - w / 2, vf, w, color=OK[5], label="model-uncertainty variance")
        ax[k].bar(x + w / 2, vs, w, color=OK[3], label="decision value-of-information")
        rc = np.corrcoef(vf, vs)[0, 1] if len(vf) > 1 else float("nan")
        ax[k].set_title(title, loc="left", fontweight="bold", fontsize=8.5)
        ax[k].set_xlabel("uncertainty mode (variance-ranked)")
        ax[k].set_xticks(x); ax[k].set_xticklabels([f"{i+1}" for i in x], fontsize=7)
        ax[k].text(0.97, 0.9, f"corr = {rc:+.2f}", transform=ax[k].transAxes,
                   ha="right", fontsize=8, fontweight="bold")
        if k == 0:
            ax[k].set_ylabel("fraction")
        if dom == "RLHF":
            ax[k].annotate("93% of the variance,\n8% of the relevance", xy=(0 + w / 2, vs[0]),
                           xytext=(1.2, 0.55), fontsize=6.5,
                           arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_voi_spectra.pdf")); plt.close(fig)


def fig_voi():
    """Composite centerpiece: (a-c) model-uncertainty variance vs decision value-of-information
    spectra across three pipelines; (d,e) retrospective acquisition curves; (f) the
    alignment gradient (correlation of the two spectra) from chemistry to RLHF."""
    sp = load("voi/voi_spectra.json"); aq = load("voi/voi_acquisition.json")
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.7))
    specs = [("BH", "a  Chemistry"), ("BO", "b  Experimental design"), ("RLHF", "c  Alignment (RLHF)")]
    corrs = {}
    for k, (dom, title) in enumerate(specs):
        r = sp[dom]; vf = np.array(r["var_frac"]); vs = np.array(r["value_sobol_first"])
        x = np.arange(len(vf)); w = 0.42
        ax[0, k].bar(x - w / 2, vf, w, color=OK[5], label="model-uncertainty variance")
        ax[0, k].bar(x + w / 2, vs, w, color=OK[3], label="decision value-of-information")
        rc = float(np.corrcoef(vf, vs)[0, 1]); corrs[dom] = rc
        ax[0, k].set_title(title, loc="left", fontweight="bold", fontsize=9)
        ax[0, k].set_xlabel("uncertainty mode (variance-ranked)", fontsize=7.5)
        ax[0, k].set_xticks(x); ax[0, k].set_xticklabels([f"{i+1}" for i in x], fontsize=6.5)
        ax[0, k].set_ylim(0, max(max(vf), max(vs)) * 1.30)     # headroom for legend / label
        ax[0, k].text(0.03, 0.92, f"corr={rc:+.2f}", transform=ax[0, k].transAxes, ha="left",
                      fontsize=8, fontweight="bold")
        if k == 0:
            ax[0, k].set_ylabel("fraction")
    ax[0, 2].annotate("mode 1 = shift-invariant direction\n(93% of variance, 8% of relevance)",
                      xy=(0 - w / 2, sp["RLHF"]["var_frac"][0]),
                      xytext=(0.72, 0.58), fontsize=5.6, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[0, 0].legend(frameon=False, fontsize=6, loc="upper right")
    # acquisition panels d, e
    sty = {"voi": ("value-of-information order", OK[3], "o-"),
           "variance": ("variance order", OK[0], "s--"), "random": ("random", "gray", "^:")}
    for k, (dom, title) in enumerate([("BH", "d  Acquisition (chemistry)"),
                                      ("RLHF", "e  Acquisition (RLHF)")]):
        c = aq[dom]
        for name in ("voi", "variance", "random"):
            lab, col, ls = sty[name]
            ax[1, k].plot(np.arange(len(c[name])), c[name], ls, color=col, label=lab, markersize=3.5)
        ax[1, k].set_title(title, loc="left", fontweight="bold", fontsize=9)
        ax[1, k].set_xlabel("uncertainty modes resolved", fontsize=7.5); ax[1, k].set_ylim(0, 1.15)
        if k == 0:
            ax[1, k].set_ylabel("residual decision\nvalue-variance"); ax[1, k].legend(frameon=False, fontsize=6)
    ax[1, 1].annotate("variance order:\nfirst acquisition\nresolves nothing",
                      xy=(1, aq["RLHF"]["variance"][1]), xytext=(1.4, 0.45), fontsize=6,
                      arrowprops=dict(arrowstyle="->", lw=0.6))
    # panel f: alignment gradient
    order = ["BH", "BO", "RLHF"]; labs = ["chemistry", "exp. design", "RLHF"]
    cols = [OK[2] if corrs[d] > 0.5 else (OK[1] if corrs[d] > -0.5 else OK[3]) for d in order]
    ax[1, 2].bar(labs, [corrs[d] for d in order], color=cols)
    ax[1, 2].axhline(0, color="black", lw=0.7); ax[1, 2].set_ylim(-1, 1)
    ax[1, 2].set_ylabel("corr(variance, value-of-info)"); ax[1, 2].tick_params(axis="x", labelsize=7)
    ax[1, 2].set_title("f  Alignment gradient", loc="left", fontweight="bold", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_voi.pdf")); plt.close(fig)


def fig_voi_acquisition():
    """Retrospective acquisition: residual decision value-variance vs modes resolved, in
    value-of-information order vs variance order vs random. BH aligned (curves coincide);
    RLHF anti-aligned (variance order wastes its first, largest acquisition)."""
    aq = load("voi/voi_acquisition.json")
    doms = [("BH", "a  Chemistry (aligned)"), ("RLHF", "b  Alignment / RLHF (anti-aligned)")]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
    sty = {"voi": ("value-of-information order", OK[3], "o-"),
           "variance": ("variance order", OK[0], "s--"),
           "random": ("random order", "gray", "^:")}
    for k, (dom, title) in enumerate(doms):
        c = aq[dom]
        for name in ("voi", "variance", "random"):
            lab, col, ls = sty[name]
            x = np.arange(len(c[name]))
            ax[k].plot(x, c[name], ls, color=col, label=lab, markersize=4)
        ax[k].set_title(title, loc="left", fontweight="bold", fontsize=9)
        ax[k].set_xlabel("uncertainty modes resolved")
        if k == 0:
            ax[k].set_ylabel("residual decision value-variance")
        ax[k].set_ylim(0, 1.15)
    ax[1].annotate("variance order:\nfirst (largest) acquisition\nresolves nothing",
                   xy=(1, aq["RLHF"]["variance"][1]), xytext=(1.3, 0.45), fontsize=6.5,
                   arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_voi_acquisition.pdf")); plt.close(fig)


def fig_bh_closedloop():
    cl = load("bh_closed_loop/bh_closed_loop.json")
    comps = ["catalyst", "base", "aryl_halide", "additive"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    # a: per-step fragility from the reward-model ensemble (reproduces the ranking)
    frag = [cl["fragility"][c] for c in comps]
    ax[0].bar([c.replace("_", "\n") for c in comps], frag, color=OK[0])
    ax[0].set_ylabel("fragility (TV from barycentre)")
    ax[0].tick_params(axis="x", labelsize=7)
    ax[0].set_title("a  Reward-model-ensemble fragility\n(reproduces ranking on real yields)",
                    loc="left", fontweight="bold")
    # b: learning curve -- fragility vs fraction of reactions observed, per component
    lc = cl["learning_curve"]; fracs = sorted(float(k) for k in lc)
    palette = {"additive": OK[3], "aryl_halide": OK[1], "catalyst": OK[0], "base": OK[2]}
    for c in comps:
        ax[1].plot([100 * f for f in fracs], [lc[f"{f}" if f"{f}" in lc else str(f)][c] for f in fracs],
                   "o-", color=palette[c], label=c.replace("_", " "), markersize=3)
    ax[1].set_xlabel("reactions observed (% of dataset)")
    ax[1].set_ylabel("decision fragility")
    ax[1].legend(frameon=False, fontsize=6.5)
    ax[1].set_title("b  Value of information", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_bh_closedloop.pdf")); plt.close(fig)


def fig_framework():
    fig, ax = plt.subplots(figsize=(9.5, 1.9)); ax.axis("off"); ax.set_xlim(0, 4); ax.set_ylim(0, 1)
    boxes = ["Objective-model\nuncertainty\n(PCA / KL modes)", "Ensemble of L\ntrained policies",
             "Sparse / arbitrary\nPCE per decision", "Sobol attribution +\nfragility + diagnostic"]
    for i, txt in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((i + 0.05, 0.25), 0.8, 0.5, boxstyle="round,pad=0.02",
                                    fc=OK[i % len(OK)], ec="black", alpha=0.18, lw=1))
        ax.text(i + 0.45, 0.5, txt, ha="center", va="center", fontsize=8.5)
        if i < 3:
            ax.add_patch(FancyArrowPatch((i + 0.86, 0.5), (i + 1.04, 0.5), arrowstyle="-|>",
                                         mutation_scale=12, lw=1.2, color="black"))
    fig.savefig(os.path.join(FIG, "fig_framework.pdf")); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_framework(); fig_bh(); fig_bh_closedloop(); fig_bo(); fig_rlhf(); fig_voi()
    print("wrote:", [f for f in ("fig_framework", "fig_bh_revised", "fig_bh_closedloop",
                                 "fig_bo_oed", "fig_rlhf", "fig_voi (composite centerpiece)")])
