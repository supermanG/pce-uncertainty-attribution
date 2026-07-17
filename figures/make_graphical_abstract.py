#!/usr/bin/env python3
"""Graphical abstract + square social card for the NMI paper.

Deliberately NOT a prism/spectral-decomposition metaphor: that idiom is the sibling
starG/stargX (group-algebraic spectroscopy) house style, so reusing it would read as
recycling. Leads instead with the paper's own result -- the variance spectrum and the
value-of-information spectrum put their mass at opposite ends (biggest uncertainty is not
the decision-relevant uncertainty) -- on the real RLHF numbers, with the alignment gradient
across the three pipelines. Outputs vector PDF + 300 dpi PNG in the paper's Okabe-Ito style.

  figures/graphical_abstract.{pdf,png}         landscape (journal graphical abstract)
  figures/graphical_abstract_square.{pdf,png}  1:1 (social card / cover starting point)
"""
import json, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("UQ_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOI = json.load(open(os.path.join(ROOT, "results", "cluster_results", "voi", "voi_spectra.json")))
OUT = os.path.join(ROOT, "figures")

OK = {"blue": "#0072B2", "amber": "#E69F00", "green": "#009E73", "verm": "#D55E00",
      "pink": "#CC79A7", "sky": "#56B4E9"}
INK, MUT = "#1a2028", "#5c6672"
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "text.color": INK, "axes.edgecolor": "#9aa4ae", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8, "svg.fonttype": "none",
})

rl = VOI["RLHF"]
var = np.array(rl["var_frac"]); voi = np.array(rl["value_sobol_first"])
modes = [f"m{i+1}" for i in range(len(var))]
x = np.arange(len(var))
grad = [("Chemistry", 0.88, OK["green"]), ("Bayesian opt.", 0.29, OK["amber"]),
        ("RLHF alignment", -0.93, OK["verm"])]
SUB = "The uncertainty a learned model has most of is often not the uncertainty that drives the decision."
FOOT = ("Polynomial-chaos surrogate over a small policy ensemble  →  analytical Sobol value-of-information "
        "per decision  ·  convergence, propagation and decision-value guarantees machine-checked in Lean 4")


def spectrum(ax, vals, color, title, pct_text, pct_xytext, mass_note=False):
    ax.bar(x, vals, width=0.62, color=color, edgecolor="none")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.annotate(pct_text, xy=(0, vals[0]), xytext=pct_xytext, fontsize=9.5, color=color,
                ha="left", va="center", arrowprops=dict(arrowstyle="->", color=color, lw=1.1))
    if mass_note:
        ax.annotate("", xy=(3.35, 0.32), xytext=(0.75, 0.32),
                    arrowprops=dict(arrowstyle="-", color=MUT, lw=0.8, ls=(0, (3, 2))))
        ax.text(2.0, 0.36, "mass on the later modes", ha="center", fontsize=8.5, color=MUT)
    ax.set_xticks(x); ax.set_xticklabels(modes, fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "", "1"], fontsize=8)
    ax.set_ylabel("fraction", fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def gradient(ax, title=True):
    for i, (name, c, col) in enumerate(grad):
        ax.barh(i, c, height=0.5, color=col, edgecolor="none")
        ax.text(c + (0.05 if c >= 0 else -0.05), i, f"{c:+.2f}", va="center",
                ha="left" if c >= 0 else "right", fontsize=9.5, color=col, fontweight="bold")
        ax.text(-0.06 if c >= 0 else 0.06, i, name, va="center",
                ha="right" if c >= 0 else "left", fontsize=9, color=INK)
    ax.axvline(0, color="#9aa4ae", lw=0.9)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([]); ax.set_xticks([-1, 0, 1]); ax.set_xticklabels(["-1", "0", "+1"], fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if title:
        ax.text(0.5, 1.34, "The alignment gradient: correlation of the two spectra across pipelines",
                transform=ax.transAxes, ha="center", fontsize=9, color=MUT)
    ax.annotate("anti-aligned", xy=(-1.2, -1.15), ha="left", fontsize=7.5, color=MUT, annotation_clip=False)
    ax.annotate("aligned", xy=(1.2, -1.15), ha="right", fontsize=7.5, color=MUT, annotation_clip=False)


def save(fig, name, tight=True):
    for ext, dpi in [("pdf", None), ("png", 300)]:
        kw = dict(bbox_inches="tight", pad_inches=0.25) if tight else {}
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=dpi, facecolor="white", **kw)


os.makedirs(OUT, exist_ok=True)

# -------------------------------------------------- landscape (journal graphical abstract)
fig = plt.figure(figsize=(9.2, 5.4))
fig.text(0.5, 0.955, "Not all uncertainty is equal", ha="center", va="top", fontsize=25, fontweight="bold", color=INK)
fig.text(0.5, 0.895, SUB, ha="center", va="top", fontsize=11.5, color=MUT)
fig.text(0.5, 0.855, "REWARD-MODEL UNCERTAINTY  ·  REAL hh-rlhf STUDY", ha="center", va="top",
         fontsize=8.5, color=MUT, family="monospace")
spectrum(fig.add_axes([0.075, 0.375, 0.39, 0.33]), var, OK["blue"],
         "How much the model is uncertain\n(variance)", "93% of the\nvariance", (0.9, 0.72))
spectrum(fig.add_axes([0.535, 0.375, 0.39, 0.33]), voi, OK["verm"],
         "Which uncertainty changes the decision\n(value of information)", "8% of the\nrelevance", (0.55, 0.62), mass_note=True)
fig.text(0.5, 0.335, "The mode carrying 93% of the variance carries 8% of the decision relevance.",
         ha="center", va="top", fontsize=10.5, color=INK, fontstyle="italic")
gradient(fig.add_axes([0.30, 0.125, 0.40, 0.12]))
fig.text(0.5, 0.038, FOOT, ha="center", va="bottom", fontsize=8, color=MUT, family="monospace")
save(fig, "graphical_abstract")

# -------------------------------------------------- square (social card / cover starting point)
figS = plt.figure(figsize=(8.2, 8.2))
figS.text(0.5, 0.965, "Not all uncertainty", ha="center", va="top", fontsize=33, fontweight="bold", color=INK)
figS.text(0.5, 0.905, "is equal", ha="center", va="top", fontsize=33, fontweight="bold", color=INK)
figS.text(0.5, 0.84, "The uncertainty a model has most of is often not\nthe uncertainty that drives the decision.",
          ha="center", va="top", fontsize=13, color=MUT)
spectrum(figS.add_axes([0.16, 0.50, 0.68, 0.20]), var, OK["blue"],
         "How much the model is uncertain  (variance)", "93% of the\nvariance", (0.95, 0.70))
spectrum(figS.add_axes([0.16, 0.215, 0.68, 0.20]), voi, OK["verm"],
         "Which uncertainty changes the decision  (value of information)", "8% of the\nrelevance", (0.6, 0.62), mass_note=True)
figS.text(0.5, 0.155, "93% of the variance carries 8% of the relevance.",
          ha="center", va="top", fontsize=12.5, color=INK, fontstyle="italic")
# compact one-line alignment gradient (coloured segments)
segs = [("chemistry +0.88", OK["green"]), ("  ·  ", MUT), ("Bayesian opt. +0.29", OK["amber"]),
        ("  ·  ", MUT), ("RLHF −0.93", OK["verm"])]
figS.text(0.5, 0.105, "alignment gradient", ha="center", fontsize=9, color=MUT, family="monospace")
# place coloured pieces centred
pieces = "".join(s for s, _ in segs)
figS.text(0.5, 0.075, "chemistry +0.88   ·   Bayesian opt. +0.29   ·   RLHF −0.93",
          ha="center", fontsize=10.5, color=INK, family="monospace")
figS.text(0.5, 0.028, "polynomial-chaos surrogate → analytical Sobol value-of-information · machine-checked in Lean 4",
          ha="center", va="bottom", fontsize=7.6, color=MUT, family="monospace")
save(figS, "graphical_abstract_square", tight=False)

print("wrote graphical_abstract.{pdf,png} and graphical_abstract_square.{pdf,png} to", OUT)
