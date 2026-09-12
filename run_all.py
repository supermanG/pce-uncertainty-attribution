#!/usr/bin/env python3
"""Master runner. Usage: python run_all.py [--experiment bh|sachs|llm|all] [--quick]"""
import argparse, os, sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", choices=["bh","sachs","llm","all"], default="all")
    parser.add_argument("--quick", "-q", action="store_true")
    parser.add_argument("--device", "-d", default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Quick: {args.quick}")
    os.makedirs(ROOT/"results", exist_ok=True)
    timings = {}

    if args.experiment in ("bh","all"):
        # Generates the ensemble members; the paper's PCE analysis of them is
        # decision_studies/common/analyze_ensemble.py (degree 3, d = 5, 7, 10), see REPRODUCE.md.
        from experiments.buchwald_hartwig.run_experiment import run_buchwald_hartwig_sequential
        t0 = time.time()
        run_buchwald_hartwig_sequential(
            n_train=10 if args.quick else 50, n_test=20 if args.quick else 100,
            gfn_episodes=500 if args.quick else 3000, device=device)
        timings["buchwald_hartwig"] = time.time()-t0

    if args.experiment in ("sachs","all"):
        from experiments.sachs_causal.run_experiment import run_sachs_experiment
        t0 = time.time()
        run_sachs_experiment(
            n_train=10 if args.quick else 30, n_test=15 if args.quick else 50,
            gfn_episodes=500 if args.quick else 3000, device=device)
        timings["sachs"] = time.time()-t0

    if args.experiment in ("llm","all"):
        from experiments.controlled_llm.run_experiment import run_controlled_llm_experiment
        t0 = time.time()
        run_controlled_llm_experiment(
            n_train=5 if args.quick else 15, n_test=10 if args.quick else 25,
            gfn_episodes=300 if args.quick else 1500, device=device)
        timings["llm"] = time.time()-t0

    print("\nTimings:", {k:f"{v:.0f}s" for k,v in timings.items()})
    json.dump(timings, open(ROOT/"results"/"timings.json","w"), indent=2)

if __name__ == "__main__": main()
