"""BH fragility: cardinality-confound control and honest reframing.

The per-step fragility ranking (additive > aryl halide
> catalyst > base) is the exact order of the action-space cardinality K (24 > 16 > 4 > 3).
This script tests whether that is a spurious dimension-counting artefact or a genuine,
decision-relevant property, and reframes the claim accordingly.

Three diagnostics, all on the real Doyle-Dreher yields with the reward-model ensemble:

1. Multi-seed fragility with confidence intervals (was single-seed).
2. Per-component reward-model uncertainty magnitude (ensemble std of predicted yield at the
   reference context). If this is ~equal across components while fragility varies 20x, then
   fragility is NOT tracking how uncertain the predictor is; it is tracking decision
   geometry.
3. Contestedness: how many options per component are within a small yield margin of the best
   (from the ensemble-mean prediction). This is the decision-geometry driver.
4. Equal-noise null: replace the ensemble by (real mean scores + i.i.d. Gaussian noise of a
   SINGLE shared magnitude across all components). If the fragility ranking reproduces under
   equal noise, the ranking is driven by the score landscape (near-ties), not by
   component-specific predictor uncertainty. Reported as the null fragility per component.

Conclusion we expect and will state honestly: the predictor's epistemic-uncertainty
magnitude is roughly uniform across components; additive selection is the most fragile
DECISION because its top options are the most contested (near-tied), so the same uncertainty
most readily changes the recommendation. Fragility is decision robustness, not a claim that
the predictor is more uncertain about additives.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from decision_studies.buchwald_hartwig.bh_closed_loop import (
    load_dataset, train_proxy_on, component_decision_dists, COMPONENTS, TAU)
from core.fragility_metrics import tv_barycenter

K_BY_COMP = {"catalyst": 4, "base": 3, "aryl_halide": 16, "additive": 24}


def build_ensemble(ds, N, M, frac, epochs, base_seed):
    """Return per-component list of (decision dist, score vector) across M proxies."""
    dists = {c: [] for c in COMPONENTS}
    scores = {c: [] for c in COMPONENTS}
    ref = ds["reactions"][int(np.argmax(ds["yields"]))].astype(int)
    for m in range(M):
        rng = np.random.RandomState(base_seed + m)
        idx = rng.permutation(N)[: int(N * frac)]
        proxy = train_proxy_on(ds, idx, seed=base_seed + m, epochs=epochs)
        dd, sc = component_decision_dists(proxy, ds, ref)
        for c in COMPONENTS:
            dists[c].append(dd[c]); scores[c].append(sc[c])
    return {c: (np.array(dists[c]), np.array(scores[c])) for c in COMPONENTS}, ref


def run(out_dir, M=18, epochs=180, frac=0.30, seeds=3, margin=10.0):
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(); N = len(ds["yields"])
    assert ds["source"] == "real"
    print(f"real Doyle-Dreher N={N}; multi-seed fragility, S={seeds} seeds x M={M} proxies")

    # ---- 1. multi-seed fragility (CI) + uncertainty magnitude ---------------------
    frag_by_seed = {c: [] for c in COMPONENTS}
    unc_by_seed = {c: [] for c in COMPONENTS}
    mean_scores = None
    for s in range(seeds):
        ens, ref = build_ensemble(ds, N, M, frac, epochs, base_seed=1000 * s)
        for c in COMPONENTS:
            dd, sc = ens[c]
            frag_by_seed[c].append(tv_barycenter(dd))
            unc_by_seed[c].append(float(np.mean(np.std(sc, axis=0))))  # ens std of yield %
        if s == 0:
            mean_scores = {c: ens[c][1].mean(0) for c in COMPONENTS}   # for null + contested
    frag = {c: (float(np.mean(frag_by_seed[c])), float(np.std(frag_by_seed[c]))) for c in COMPONENTS}
    unc = {c: float(np.mean(unc_by_seed[c])) for c in COMPONENTS}
    sigma = float(np.mean(list(unc.values())))

    # ---- 2. contestedness: options within `margin` %yield of the best -------------
    contested = {}
    for c in COMPONENTS:
        sc = mean_scores[c]
        contested[c] = int(np.sum(sc >= sc.max() - margin))

    # ---- 3. equal-noise null: real landscape + single shared sigma ----------------
    rng = np.random.RandomState(31337)
    null_frag = {}
    for c in COMPONENTS:
        base = mean_scores[c]
        pseudo = base[None, :] + rng.randn(400, len(base)) * sigma        # 400 pseudo-members
        z = pseudo / TAU; z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        null_frag[c] = tv_barycenter(p)

    # ---- report -------------------------------------------------------------------
    print(f"\ncomp        K   frag(mean+/-sd)   RM-unc(pct)   contested(<={margin:.0f}pct)   null-frag(equal noise)")
    for c in ["additive", "aryl_halide", "catalyst", "base"]:
        m, sd = frag[c]
        print(f"  {c:11s} {K_BY_COMP[c]:2d}   {m:.3f}+/-{sd:.3f}     {unc[c]:.2f}        {contested[c]:2d}/{K_BY_COMP[c]:<2d}            {null_frag[c]:.3f}")
    # correlations
    ks = np.array([K_BY_COMP[c] for c in COMPONENTS])
    fr = np.array([frag[c][0] for c in COMPONENTS])
    un = np.array([unc[c] for c in COMPONENTS])
    co = np.array([contested[c] for c in COMPONENTS])
    def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
    print(f"\ncorr(fragility, K)           = {corr(fr, ks):+.3f}")
    print(f"corr(fragility, RM-uncert)   = {corr(fr, un):+.3f}   (near-zero/negative => not driven by uncertainty magnitude)")
    print(f"corr(fragility, contested)   = {corr(fr, co):+.3f}   (high => driven by decision contestedness)")
    print(f"corr(real frag, null frag)   = {corr(fr, np.array([null_frag[c] for c in COMPONENTS])):+.3f}   (high => landscape/geometry, not differential noise)")

    res = dict(N=N, M=M, seeds=seeds, frac=frac, sigma_shared=sigma, margin=margin,
               fragility_mean={c: frag[c][0] for c in COMPONENTS},
               fragility_std={c: frag[c][1] for c in COMPONENTS},
               rm_uncertainty=unc, contestedness=contested, null_fragility=null_frag,
               K=K_BY_COMP,
               corr_frag_K=corr(fr, ks), corr_frag_unc=corr(fr, un),
               corr_frag_contested=corr(fr, co),
               corr_frag_null=corr(fr, np.array([null_frag[c] for c in COMPONENTS])))
    json.dump(res, open(os.path.join(out_dir, "bh_confound_control.json"), "w"), indent=2)
    print(f"\n[saved] {out_dir}/bh_confound_control.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cluster_results", "bh_closed_loop"))
    ap.add_argument("--M", type=int, default=18)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    run(a.out, M=a.M, epochs=a.epochs, seeds=a.seeds)
