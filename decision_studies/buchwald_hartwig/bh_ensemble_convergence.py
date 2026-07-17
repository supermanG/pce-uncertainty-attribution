"""Convergence of the fragility attribution in the ensemble size L.

Here we assess sensitivity of the results to the ensemble size, on the real
Doyle-Dreher yields, we vary the reward-model ensemble size L and report the per-component
decision fragility, showing the attribution stabilises by L ~ 15-20 members.
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from decision_studies.buchwald_hartwig.bh_closed_loop import load_dataset, COMPONENTS
from decision_studies.buchwald_hartwig.bh_confound_control import build_ensemble
from core.fragility_metrics import tv_barycenter


def run(out_dir, epochs=180, frac=0.30, Ls=(6, 10, 15, 20, 25, 30)):
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(); N = len(ds["yields"])
    curve = {}
    print("L    " + "  ".join(f"{c[:4]:>6}" for c in COMPONENTS))
    for L in Ls:
        ens, _ = build_ensemble(ds, N, L, frac, epochs, base_seed=0)
        frag = {c: float(tv_barycenter(ens[c][0])) for c in COMPONENTS}
        curve[str(L)] = frag
        print(f"{L:<4} " + "  ".join(f"{frag[c]:6.3f}" for c in COMPONENTS))
    json.dump({"frac": frac, "epochs": epochs, "curve": curve},
              open(os.path.join(out_dir, "bh_ensemble_convergence.json"), "w"), indent=2)
    print(f"[saved] {out_dir}/bh_ensemble_convergence.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cluster_results", "bh_closed_loop"))
    ap.add_argument("--epochs", type=int, default=180)
    a = ap.parse_args()
    run(a.out, epochs=a.epochs)
