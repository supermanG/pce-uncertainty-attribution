"""BH closed-loop validation: does the fragility ranking predict the value of data?

The manuscript's actionable claim (Buchwald-Hartwig): additive selection is the most
fragile decision to reward-model (yield-predictor) uncertainty and base selection the
most robust, so "additive recommendations warrant the most additional data". This
script tests that claim directly on the real Doyle-Dreher yields, using the reward-model
ensemble alone (no GFlowNet), which isolates the reward-driven signal from training noise.

Two results:
  1. Cross-check. From an ensemble of yield-predictor proxies (each trained on an
     independent random subset), compute a per-component decision fragility (TV-from-
     barycentre of the softmax-over-marginal-yield decision distribution). This should
     reproduce the manuscript ranking additive > aryl_halide > catalyst > base, showing
     the fragility ranking is a property of the reward model's epistemic uncertainty.

  2. Value of information (the closed loop). Starting from a small labelled seed set,
     acquire a fixed budget of new reactions by coverage-greedy targeting of a chosen
     component (additive / base / catalyst) or at random, retrain the proxy ensemble, and
     measure how far each acquisition drives down (a) that component's decision fragility
     and (b) held-out yield RMSE. The claim predicts additive-targeted acquisition yields
     the largest fragility reduction (there is the most to resolve), whereas base-targeted
     acquisition changes little (already robust). This is the falsifiable payoff:
     fragility ranks where data has value.

Uses the same YieldProxy MLP and reward temperature (tau=4.0) as the main experiment.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "buchwald_hartwig"))

import torch, torch.nn as nn, torch.optim as optim
from experiments.buchwald_hartwig.run_experiment import load_dataset, YieldProxy
from core.fragility_metrics import tv_barycenter

TAU = 4.0
COMPONENTS = ["catalyst", "base", "aryl_halide", "additive"]


def train_proxy_on(dataset, idx, seed=0, epochs=200, lr=1e-3):
    """Train a YieldProxy on an explicit set of reaction indices (deterministic in seed)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = YieldProxy(dataset["n_components"])
    x = model.encode(dataset["reactions"][idx])
    y = torch.tensor(dataset["yields"][idx], dtype=torch.float32)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        loss = nn.MSELoss()(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def component_decision_dists(proxy, dataset, ref):
    """For each component c, the decision distribution at a fixed reference context.

    Following the manuscript's per-step policy construction: fix the other three
    components at the reference trajectory `ref` and vary component c over its choices,
    score each candidate reaction with the proxy, and take decision_c = softmax(score/tau).
    Evaluating at a single context (not averaged over the dataset) exposes the reward
    model's epistemic disagreement about that decision, exactly as the GFlowNet policy does.
    Returns {c_name -> prob vector over that component's choices}.
    """
    dists, scores = {}, {}
    for ci, name in enumerate(COMPONENTS):
        n_c = dataset["n_components"][ci]
        cand = np.tile(np.asarray(ref, dtype=int), (n_c, 1))
        cand[:, ci] = np.arange(n_c)
        score = proxy.predict_yield(cand)
        z = score / TAU; z -= z.max()
        p = np.exp(z); p /= p.sum()
        dists[name] = p
        scores[name] = score
    return dists, scores


def ensemble_fragility(dataset, train_idx_fn, M, epochs, ref, base_seed=0):
    """Fragility per component across an M-proxy ensemble; also mean held-out RMSE.

    train_idx_fn(seed) -> (train_idx, test_idx). Each proxy uses its own draw.
    Decisions are evaluated at the shared reference context `ref`.
    """
    per_comp = {c: [] for c in COMPONENTS}
    per_score = {c: [] for c in COMPONENTS}
    rmses = []
    for m in range(M):
        tr, te = train_idx_fn(base_seed + m)
        proxy = train_proxy_on(dataset, tr, seed=base_seed + m, epochs=epochs)
        dists, scores = component_decision_dists(proxy, dataset, ref)
        for c in COMPONENTS:
            per_comp[c].append(dists[c])
            per_score[c].append(scores[c])
        if len(te):
            pr = proxy.predict_yield(dataset["reactions"][te])
            rmses.append(float(np.sqrt(np.mean((pr - dataset["yields"][te]) ** 2))))
    frag = {c: tv_barycenter(np.array(per_comp[c])) for c in COMPONENTS}
    # reward-model epistemic uncertainty about component c: mean over its choices of the
    # ensemble std of the predicted yield at the reference context (in % yield units).
    unc = {c: float(np.mean(np.std(np.array(per_score[c]), axis=0))) for c in COMPONENTS}
    return frag, unc, float(np.mean(rmses)) if rmses else float("nan")


def coverage_greedy_acquire(dataset, seed_idx, target_component, budget, rng):
    """Pick `budget` new reaction indices (outside seed_idx) that most improve coverage
    of `target_component` (round-robin over its least-represented levels). If
    target_component is None, pick uniformly at random."""
    N = len(dataset["yields"])
    pool = np.setdiff1d(np.arange(N), seed_idx)
    rng.shuffle(pool)
    if target_component is None:
        return pool[:budget]
    ci = COMPONENTS.index(target_component)
    rxn = dataset["reactions"]
    counts = {j: int(np.sum(rxn[seed_idx, ci] == j)) for j in range(dataset["n_components"][ci])}
    picked, used = [], set()
    while len(picked) < budget:
        # target the currently least-represented level, then next, round-robin
        order = sorted(counts, key=lambda j: counts[j])
        progressed = False
        for j in order:
            cand = [p for p in pool if rxn[p, ci] == j and p not in used]
            if cand:
                picked.append(cand[0]); used.add(cand[0]); counts[j] += 1; progressed = True
                if len(picked) >= budget:
                    break
        if not progressed:
            break
    # top up randomly if the targeted levels ran out
    if len(picked) < budget:
        extra = [p for p in pool if p not in used][: budget - len(picked)]
        picked += extra
    return np.array(picked[:budget])


def run(out_dir, M=25, epochs=150, frac=0.30, seed_frac=0.08, budget_frac=0.06,
        voi_reps=4, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset()
    assert ds["source"] == "real", "need real Doyle-Dreher data"
    N = len(ds["yields"])
    print(f"real Doyle-Dreher: N={N}, components={dict(zip(COMPONENTS, ds['n_components']))}")
    # shared reference context = highest-yield reaction (manuscript convention)
    ref = ds["reactions"][int(np.argmax(ds["yields"]))].astype(int)
    print(f"reference trajectory (best reaction): {ref.tolist()} "
          f"(yield={ds['yields'].max():.1f}%)")

    # ---- Result 1: fragility ranking at a fixed training fraction ------------------
    def rand_split(s):
        rng = np.random.RandomState(s)
        idx = rng.permutation(N)
        k = int(N * frac)
        return idx[:k], idx[k:k + 800]        # train subset, held-out panel
    print(f"\n[1] per-component fragility (ensemble of {M} proxies, frac={frac})...")
    frag, unc, rmse = ensemble_fragility(ds, rand_split, M, epochs, ref, base_seed=seed)
    ranking = sorted(frag, key=lambda c: frag[c], reverse=True)
    unc_ranking = sorted(unc, key=lambda c: unc[c], reverse=True)
    print("    TV fragility:", {c: round(frag[c], 3) for c in COMPONENTS})
    print("    RM uncertainty (yield-std %):", {c: round(unc[c], 2) for c in COMPONENTS})
    print("    ranking (most->least fragile):", ranking, f"  held-out RMSE={rmse:.2f}")
    manuscript_order = ["additive", "aryl_halide", "catalyst", "base"]
    print(f"    manuscript order: {manuscript_order}  ->  TV match: {ranking == manuscript_order}"
          f"  | unc match: {unc_ranking == manuscript_order}")

    # ---- Result 2a: fragility learning curve ---------------------------------------
    # As more reactions are observed (reward-model uncertainty shrinks), each decision's
    # fragility falls. The fragile decision (additive) starts highest and has the most to
    # shed; the robust decision (base) is low throughout. This is the "warrant the most
    # additional data" claim as a value-of-information curve.
    M_curve = max(12, M // 2)
    print(f"\n[2a] fragility learning curve (ensemble of {M_curve} proxies per point)...")
    curve = {}
    for f in [0.02, 0.05, 0.10, 0.20, 0.40]:
        def split_f(s, _f=f):
            rng = np.random.RandomState(s)
            idx = rng.permutation(N); k = max(24, int(N * _f))
            return idx[:k], idx[k:k + 600]
        fr, _, rm = ensemble_fragility(ds, split_f, M_curve, epochs, ref, base_seed=200)
        curve[f] = {c: float(fr[c]) for c in COMPONENTS}
        print(f"    frac={f:.2f} (n={int(N*f)}): "
              + ", ".join(f"{c[:4]}={fr[c]:.3f}" for c in COMPONENTS) + f"   RMSE={rm:.1f}")

    # ---- Result 2b: targeted acquisition reduces total decision fragility -----------
    # From a small seed set, acquire a fixed budget targeting the fragile axis (additive)
    # vs the robust axis (base) vs random; measure the reduction in TOTAL decision
    # fragility. Targeting where the fragility lives should reduce it most per sample.
    seed_k = int(N * seed_frac); budget = int(N * budget_frac)
    M_voi = max(10, M // 2)
    print(f"\n[2b] targeted acquisition: seed={seed_k}, budget={budget}, reps={voi_reps}, "
          f"{M_voi} proxies...")
    voi = {}
    for target in ["additive", "base", None]:
        tname = target or "random"
        tot_red, tgt_red, drmse = [], [], []
        for rep in range(voi_reps):
            rng = np.random.RandomState(1000 + rep)
            perm = rng.permutation(N)
            seed_idx = perm[:seed_k]; held = perm[seed_k:seed_k + 600]
            f0, _, r0 = ensemble_fragility(ds, (lambda s, _si=seed_idx, _h=held: (_si, _h)),
                                           M_voi, epochs, ref, base_seed=5000 + rep * 50)
            acq = coverage_greedy_acquire(ds, seed_idx, target, budget, rng)
            aug = np.concatenate([seed_idx, acq])
            f1, _, r1 = ensemble_fragility(ds, (lambda s, _a=aug, _h=held: (_a, _h)),
                                           M_voi, epochs, ref, base_seed=6000 + rep * 50)
            tot_red.append(sum(f0[c] - f1[c] for c in COMPONENTS))
            tgt_red.append((f0[target] - f1[target]) if target else
                           np.mean([f0[c] - f1[c] for c in COMPONENTS]))
            drmse.append(r0 - r1)
        voi[tname] = dict(
            total_frag_reduction=float(np.mean(tot_red)),
            total_frag_reduction_std=float(np.std(tot_red)),
            target_frag_reduction=float(np.mean(tgt_red)),
            rmse_reduction=float(np.mean(drmse)),
        )
        print(f"    target={tname:9s}: total dFragility={voi[tname]['total_frag_reduction']:+.3f}"
              f" +/-{voi[tname]['total_frag_reduction_std']:.3f}"
              f"   target dFragility={voi[tname]['target_frag_reduction']:+.3f}"
              f"   dRMSE={voi[tname]['rmse_reduction']:+.2f}")

    res = dict(
        N=N, M=M, frac=frac, epochs=epochs, tau=TAU, reference=ref.tolist(),
        fragility=frag, rm_uncertainty=unc, fragility_ranking=ranking,
        uncertainty_ranking=unc_ranking, heldout_rmse=rmse,
        manuscript_order=manuscript_order,
        ranking_matches_manuscript=(ranking == manuscript_order),
        uncertainty_matches_manuscript=(unc_ranking == manuscript_order),
        learning_curve={str(k): v for k, v in curve.items()},
        voi=voi, seed_frac=seed_frac, budget_frac=budget_frac, voi_reps=voi_reps,
    )
    json.dump(res, open(os.path.join(out_dir, "bh_closed_loop.json"), "w"), indent=2)
    print(f"\n[saved] {out_dir}/bh_closed_loop.json")
    # headline
    print(f"\nHEADLINE (ranking): reward-model ensemble reproduces manuscript fragility order "
          f"{ranking} -> {ranking == manuscript_order}")
    if voi:
        add = voi.get("additive", {}).get("total_frag_reduction")
        bas = voi.get("base", {}).get("total_frag_reduction")
        print(f"HEADLINE (VOI): total decision-fragility reduction, additive-targeted={add:+.3f} "
              f"vs base-targeted={bas:+.3f} vs random={voi.get('random',{}).get('total_frag_reduction'):+.3f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cluster_results", "bh_closed_loop"))
    ap.add_argument("--M", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--voi_reps", type=int, default=4)
    a = ap.parse_args()
    run(a.out, M=a.M, epochs=a.epochs, voi_reps=a.voi_reps)
