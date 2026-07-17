#!/usr/bin/env python3
"""
Local runner for UQ-GFlowNet experiments.
Runs ensemble members sequentially (or with limited parallelism for CPU jobs)
on a single machine instead of via LSF.

Usage:
    python run_local.py --experiment gridworld_discrete
    python run_local.py --experiment all_cpu
    python run_local.py --experiment bh --device cuda
"""
import argparse
import subprocess
import sys
import os
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_DIR = Path(__file__).resolve().parent.parent


def member_done(output_dir, member_id):
    """Check if a member's output already exists (for resume)."""
    npz = Path(output_dir) / "members" / f"member_{member_id:04d}.npz"
    return npz.exists()


def run_member(cmd, member_id, experiment_name):
    """Run a single ensemble member and return success/failure."""
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,  # 2h max
            cwd=str(PROJECT_DIR),
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  [{experiment_name}] member {member_id} FAILED ({elapsed:.0f}s)", flush=True)
            print(f"    stderr: {result.stderr[-500:]}" if result.stderr else "", flush=True)
            return False
        print(f"  [{experiment_name}] member {member_id} done ({elapsed:.0f}s)", flush=True)
        return True
    except subprocess.TimeoutExpired:
        print(f"  [{experiment_name}] member {member_id} TIMEOUT", flush=True)
        return False
    except Exception as e:
        print(f"  [{experiment_name}] member {member_id} ERROR: {e}", flush=True)
        return False


def run_experiment(name, train_cmd_fn, analyze_cmd, n_members,
                   max_workers=1, device="cpu", output_dir=None):
    """Run all members then analysis for one experiment."""
    # Check for already-completed members (resume support)
    todo = list(range(n_members))
    skipped = 0
    if output_dir:
        todo = [i for i in todo if not member_done(output_dir, i)]
        skipped = n_members - len(todo)

    print(f"\n{'='*60}", flush=True)
    print(f"  {name}: {n_members} members ({skipped} already done, {len(todo)} to run)", flush=True)
    print(f"  workers={max_workers}, device={device}", flush=True)
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    successes = skipped
    failures = 0

    if max_workers == 1:
        # Sequential
        for i in todo:
            cmd = train_cmd_fn(i)
            ok = run_member(cmd, i, name)
            if ok:
                successes += 1
            else:
                failures += 1
    else:
        # Parallel with limited workers
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for i in todo:
                cmd = train_cmd_fn(i)
                f = pool.submit(run_member, cmd, i, name)
                futures[f] = i
            for f in as_completed(futures):
                if f.result():
                    successes += 1
                else:
                    failures += 1

    train_elapsed = time.time() - t0
    print(f"\n  Training done: {successes}/{n_members} succeeded, "
          f"{failures} failed ({train_elapsed:.0f}s)", flush=True)

    if successes == 0:
        print(f"  SKIPPING analysis -- no successful members", flush=True)
        return False

    # Run analysis
    print(f"  Running analysis...", flush=True)
    t1 = time.time()
    result = subprocess.run(
        analyze_cmd, capture_output=True, text=True, timeout=3600,
        cwd=str(PROJECT_DIR),
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    analyze_elapsed = time.time() - t1
    total_elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  Analysis FAILED ({analyze_elapsed:.0f}s)", flush=True)
        print(f"    stdout: {result.stdout[-500:]}" if result.stdout else "", flush=True)
        print(f"    stderr: {result.stderr[-1000:]}" if result.stderr else "", flush=True)
        return False

    print(f"  Analysis done ({analyze_elapsed:.0f}s)", flush=True)
    print(f"  Total: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)
    return True


def get_experiments(device="cpu"):
    """Define all experiments."""
    py = sys.executable
    pd = str(PROJECT_DIR)

    experiments = {}

    # --- Gridworld discrete (CPU, ~10 min/member) ---
    gw_disc_dir = f"{pd}/results/gridworld/discrete"
    os.makedirs(f"{gw_disc_dir}/members", exist_ok=True)
    experiments["gridworld_discrete"] = dict(
        train_cmd_fn=lambda i: [
            py, f"{pd}/experiments/gridworld/train_single_member.py",
            "--mode", "discrete", "--member_id", str(i),
            "--output_dir", gw_disc_dir
        ],
        analyze_cmd=[
            py, f"{pd}/experiments/gridworld/run_experiment.py",
            "--run_mode", "analyze", "--mode", "discrete",
            "--n_train", "50", "--n_test", "100",
            "--output_dir", gw_disc_dir
        ],
        n_members=150,
        max_workers=4,  # CPU parallel
        device="cpu",
        output_dir=gw_disc_dir,
    )

    # --- Gridworld continuous (CPU) ---
    gw_cont_dir = f"{pd}/results/gridworld/continuous"
    os.makedirs(f"{gw_cont_dir}/members", exist_ok=True)
    experiments["gridworld_continuous"] = dict(
        train_cmd_fn=lambda i: [
            py, f"{pd}/experiments/gridworld/train_single_member.py",
            "--mode", "continuous", "--member_id", str(i),
            "--output_dir", gw_cont_dir
        ],
        analyze_cmd=[
            py, f"{pd}/experiments/gridworld/run_experiment.py",
            "--run_mode", "analyze", "--mode", "continuous",
            "--n_train", "50", "--n_test", "100",
            "--output_dir", gw_cont_dir
        ],
        n_members=150,
        max_workers=4,
        device="cpu",
        output_dir=gw_cont_dir,
    )

    # --- Symbolic regression (CPU, ~15 min/member) ---
    symreg_dir = f"{pd}/results/symreg"
    os.makedirs(f"{symreg_dir}/members", exist_ok=True)
    experiments["symreg"] = dict(
        train_cmd_fn=lambda i: [
            py, f"{pd}/experiments/symreg/train_single_member.py",
            "--member_id", str(i),
            "--output_dir", symreg_dir
        ],
        analyze_cmd=[
            py, f"{pd}/experiments/symreg/run_experiment.py",
            "--run_mode", "analyze",
            "--n_train", "100", "--n_test", "50",
            "--pce_degree", "5",
            "--output_dir", symreg_dir
        ],
        n_members=150,
        max_workers=4,
        device="cpu",
        output_dir=symreg_dir,
    )

    # --- LLM GFlowNet (CPU, ~30 min/member) ---
    llm_dir = f"{pd}/results/llm_gfn"
    os.makedirs(f"{llm_dir}/members", exist_ok=True)
    experiments["llm_gfn"] = dict(
        train_cmd_fn=lambda i: [
            py, f"{pd}/experiments/llm_gfn/train_single_member.py",
            "--member_id", str(i),
            "--n_total", "80",
            "--device", "cpu",
            "--output_dir", llm_dir
        ],
        analyze_cmd=[
            py, f"{pd}/experiments/llm_gfn/run_experiment.py",
            "--mode", "analyze",
            "--n_train", "30", "--n_test", "50",
            "--pce_degree", "5",
            "--output_dir", llm_dir
        ],
        n_members=80,
        max_workers=4,
        device="cpu",
        output_dir=llm_dir,
    )

    # --- Buchwald-Hartwig (GPU, ~40 min/member) ---
    bh_dir = f"{pd}/results/buchwald_hartwig"
    bh_csv = f"{pd}/experiments/buchwald_hartwig/data/data_table.csv"
    os.makedirs(f"{bh_dir}/members", exist_ok=True)
    experiments["bh"] = dict(
        train_cmd_fn=lambda i: [
            py, f"{pd}/experiments/buchwald_hartwig/train_single_gfn.py",
            "--member_id", str(i),
            "--output_dir", bh_dir,
            "--csv_path", bh_csv,
            "--gfn_episodes", "3000",
            "--proxy_epochs", "200",
            "--temp", "4.0"
        ],
        analyze_cmd=[
            py, f"{pd}/experiments/buchwald_hartwig/run_experiment.py",
            "--mode", "analyze",
            "--n_train", "50", "--n_test", "100",
            "--pce_degree", "3", "--pca_dim", "5",
            "--csv_path", bh_csv,
            "--output_dir", bh_dir
        ],
        n_members=150,
        max_workers=1,  # GPU sequential
        device=device,
        output_dir=bh_dir,
    )

    # --- Sachs causal (GPU) ---
    sachs_dir = f"{pd}/results/sachs"
    os.makedirs(f"{sachs_dir}/members", exist_ok=True)
    sachs_data = f"{pd}/data/sachs_real.csv"
    sachs_analyze_extra = ["--data_path", sachs_data] if os.path.exists(sachs_data) else []
    experiments["sachs"] = dict(
        train_cmd_fn=lambda i: [
            py, f"{pd}/experiments/sachs_causal/train_single_member.py",
            "--member_id", str(i),
            "--output_dir", sachs_dir,
            "--gfn_episodes", "3000"
        ],
        analyze_cmd=[
            py, f"{pd}/experiments/sachs_causal/run_experiment.py",
            "--mode", "analyze",
            "--n_train", "30", "--n_test", "50",
            "--pce_degree", "5", "--pca_dim", "2",
            "--output_dir", sachs_dir
        ] + sachs_analyze_extra,
        n_members=80,
        max_workers=1,
        device=device,
        output_dir=sachs_dir,
    )

    # --- Molecular design (GPU) ---
    mol_dir = f"{pd}/results/molecular_design"
    os.makedirs(f"{mol_dir}/members", exist_ok=True)
    experiments["moldesign"] = dict(
        train_cmd_fn=lambda i: [
            py, f"{pd}/experiments/molecular_design/train_single_member.py",
            "--member_id", str(i),
            "--output_dir", mol_dir,
            "--gfn_episodes", "3000"
        ],
        analyze_cmd=[
            py, f"{pd}/experiments/molecular_design/run_experiment.py",
            "--mode", "analyze",
            "--n_train", "30", "--n_test", "50",
            "--pce_degree", "5", "--pca_dim", "2",
            "--output_dir", mol_dir
        ],
        n_members=80,
        max_workers=1,
        device=device,
        output_dir=mol_dir,
    )

    return experiments


# Experiment groups
GROUPS = {
    "all_cpu": ["gridworld_discrete", "gridworld_continuous", "symreg", "llm_gfn"],
    "all_gpu": ["bh", "sachs", "moldesign"],
    "all": ["gridworld_discrete", "gridworld_continuous", "symreg", "llm_gfn",
            "bh", "sachs", "moldesign"],
    "quick_test": ["gridworld_discrete"],  # fast sanity check
}


def main():
    parser = argparse.ArgumentParser(description="Local UQ-GFlowNet experiment runner")
    parser.add_argument("--experiment", "-e", required=True,
                        help="Experiment name or group (all_cpu, all_gpu, all, quick_test)")
    parser.add_argument("--device", default="cuda",
                        help="Device for GPU experiments (default: cuda)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Override max parallel workers for CPU experiments")
    parser.add_argument("--setup-only", action="store_true",
                        help="Run setup phases only, don't train")
    args = parser.parse_args()

    experiments = get_experiments(device=args.device)

    # Resolve group or single experiment
    if args.experiment in GROUPS:
        exp_names = GROUPS[args.experiment]
    elif args.experiment in experiments:
        exp_names = [args.experiment]
    else:
        print(f"Unknown experiment: {args.experiment}")
        print(f"Available: {', '.join(list(experiments.keys()) + list(GROUPS.keys()))}")
        sys.exit(1)

    # BH needs setup phase first
    if "bh" in exp_names:
        bh_dir = str(PROJECT_DIR / "results" / "buchwald_hartwig")
        bh_csv = str(PROJECT_DIR / "experiments" / "buchwald_hartwig" / "data" / "data_table.csv")
        meta = Path(bh_dir) / "metadata.json"
        if not meta.exists():
            print("Running BH setup phase...")
            subprocess.run([
                sys.executable,
                str(PROJECT_DIR / "experiments" / "buchwald_hartwig" / "train_single_gfn.py"),
                "--setup",
                "--csv_path", bh_csv,
                "--output_dir", bh_dir
            ], cwd=str(PROJECT_DIR), check=True)
            print("BH setup done.")

    if args.setup_only:
        print("Setup complete. Exiting.")
        return

    # Run experiments
    results = {}
    total_t0 = time.time()
    for name in exp_names:
        exp = experiments[name]
        if args.max_workers is not None and exp["device"] == "cpu":
            exp["max_workers"] = args.max_workers
        ok = run_experiment(name, **exp)
        results[name] = "SUCCESS" if ok else "FAILED"

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  ALL DONE ({total_elapsed/3600:.1f} hours)")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name}: {status}")

    # Save summary
    summary_path = PROJECT_DIR / "results" / "local_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "results": results,
            "total_seconds": total_elapsed,
            "device": args.device,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
