"""Re-analysis: scale-dependence of D and first- vs total-order Sobol sums.

Re-examines two issues using the CACHED analysis outputs (no retraining):

  Issue 1   Total policy variance D is a sum over K-1 ALR components, so raw
                 cross-step / cross-edge D comparisons may be confounded by action
                 cardinality K.  We report D, K-1, and the per-dimension D/(K-1).

  Issue 2   The manuscript claims total-order Sobol indices "sum to at most 1"
                 and "approximately 1 -> negligible interactions".  Total-order
                 indices sum to >= 1; the excess is the interaction effect.  We
                 report, per task, the first-order sum (must be <= 1 for d=2) and
                 the total-order sum (diagnoses interactions).

Data sources:
  results/molecular_design/results.json   full first_order/total_order/variance
  results/sachs/results.json               full first_order/total_order/variance
  Buchwald-Hartwig                         D-values hardcoded in figures/generate_figures.py
                                           (from the 150-member cluster run); per-member
                                           policies not cached locally, so BH here is the
                                           analytic K-1 argument only.

Run:  python decision_studies/reanalysis_sobol_scale.py
"""
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def summarise_task(name, sobol, step_labels=None):
    """sobol: dict {step_str -> {first_order, total_order, variance}}.

    first_order / total_order: (K-1, d) lists.  variance: (K-1,) list.
    """
    steps = sorted(sobol.keys(), key=lambda s: int(s))
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    print(f"{'step':>18} | {'K-1':>4} | {'D=sum var':>10} | {'D/(K-1)':>9} | "
          f"{'foSum':>6} | {'toSum':>6} | {'interact':>8}")
    print("-" * 78)

    per_step = []
    all_fo_sums, all_to_sums = [], []
    for s in steps:
        fo = np.array(sobol[s]["first_order"])      # (K-1, d)
        to = np.array(sobol[s]["total_order"])       # (K-1, d)
        var = np.array(sobol[s]["variance"])          # (K-1,)
        Km1 = var.shape[0]
        D = float(var.sum())
        Dnorm = D / Km1

        # per-action sums over the d input dimensions
        fo_sum = fo.sum(axis=1)   # (K-1,) first-order index sum per action
        to_sum = to.sum(axis=1)   # (K-1,) total-order index sum per action
        # interaction proxy: total-order sum minus first-order sum (>=0), averaged
        interact = float(np.mean(to_sum - fo_sum))

        all_fo_sums.append(fo_sum)
        all_to_sums.append(to_sum)

        lbl = (step_labels[int(s)] if step_labels and int(s) < len(step_labels)
               else f"step {s}")
        print(f"{lbl:>18} | {Km1:>4d} | {D:>10.2f} | {Dnorm:>9.3f} | "
              f"{fo_sum.mean():>6.3f} | {to_sum.mean():>6.3f} | {interact:>8.3f}")
        per_step.append(dict(step=s, label=lbl, Km1=Km1, D=D, Dnorm=Dnorm,
                             fo_sum_mean=float(fo_sum.mean()),
                             to_sum_mean=float(to_sum.mean()),
                             interaction_mean=interact))

    fo_all = np.concatenate(all_fo_sums)
    to_all = np.concatenate(all_to_sums)
    print("-" * 78)
    print(f"first-order sum  (should be <= 1 for d=2): "
          f"mean={fo_all.mean():.3f}  max={fo_all.max():.3f}  "
          f"frac>1={np.mean(fo_all > 1.0):.2%}")
    print(f"total-order sum  (>= 1 signals interactions): "
          f"mean={to_all.mean():.3f}  max={to_all.max():.3f}  "
          f"frac>1={np.mean(to_all > 1.0):.2%}")
    print(f"manuscript claim 'total-order ~ 1, negligible interactions' is "
          f"{'CONTRADICTED' if to_all.mean() > 1.15 else 'PLAUSIBLE'} "
          f"(mean total-order sum = {to_all.mean():.3f})")

    # D ranking vs D/(K-1) ranking
    order_D = [per_step[i]['label'] for i in np.argsort([-p['D'] for p in per_step])]
    order_Dn = [per_step[i]['label'] for i in np.argsort([-p['Dnorm'] for p in per_step])]
    print(f"\nfragility rank by raw D      : {' > '.join(order_D)}")
    print(f"fragility rank by D/(K-1)    : {' > '.join(order_Dn)}")
    print(f"ranking {'CHANGES' if order_D != order_Dn else 'is STABLE'} under normalisation")
    return dict(per_step=per_step,
                fo_sum_mean=float(fo_all.mean()), fo_sum_max=float(fo_all.max()),
                to_sum_mean=float(to_all.mean()), to_sum_max=float(to_all.max()),
                to_frac_gt1=float(np.mean(to_all > 1.0)),
                rank_D=order_D, rank_Dnorm=order_Dn,
                rank_changes=order_D != order_Dn)


def bh_analytic():
    """BH: analytic K-1 argument from hardcoded D-values (per-member policies not cached)."""
    print(f"\n{'='*78}\nBUCHWALD-HARTWIG (analytic K-1 argument; D from 150-member run)\n{'='*78}")
    labels = ["catalyst", "base", "aryl_halide", "additive"]
    K = [4, 3, 16, 24]
    D = [71.3, 69.0, 105.6, 179.2]
    Km1 = [k - 1 for k in K]
    Dn = [d / m for d, m in zip(D, Km1)]
    print(f"{'step':>12} | {'K':>3} | {'K-1':>4} | {'D':>7} | {'D/(K-1)':>8}")
    print("-" * 50)
    for l, k, m, d, dn in zip(labels, K, Km1, D, Dn):
        print(f"{l:>12} | {k:>3d} | {m:>4d} | {d:>7.1f} | {dn:>8.2f}")
    order_D = [labels[i] for i in np.argsort([-d for d in D])]
    order_Dn = [labels[i] for i in np.argsort([-d for d in Dn])]
    print(f"\nheadline rank by raw D    : {' > '.join(order_D)}")
    print(f"rank by per-dim D/(K-1)   : {' > '.join(order_Dn)}")
    print("Pearson corr(D, K-1) = %.3f  (near 1 => D tracks cardinality)"
          % np.corrcoef(D, Km1)[0, 1])
    print("=> The 'additive most fragile, catalyst robust' headline REVERSES per-dimension.")
    return dict(labels=labels, K=K, D=D, Dnorm=Dn,
                rank_D=order_D, rank_Dnorm=order_Dn,
                corr_D_Km1=float(np.corrcoef(D, Km1)[0, 1]))


if __name__ == "__main__":
    out = {}
    md = load("results/molecular_design/results.json")
    out["molecular_design"] = summarise_task(
        "MOLECULAR DESIGN (K=20 at every position => K-1 constant)",
        md["sobol"], md.get("position_roles"))

    sa = load("results/sachs/results.json")
    out["sachs"] = summarise_task("SACHS causal discovery", sa["sobol"])

    out["buchwald_hartwig"] = bh_analytic()

    os.makedirs(os.path.join(ROOT, "results", "outputs"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "outputs", "sobol_scale_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] results/outputs/sobol_scale_summary.json")
