"""Train a single LLM GFlowNet member for LSF array job.

Usage:
    python3 experiments/llm_gfn/train_single_member.py \
        --member_id 0 --output_dir results/llm_gfn \
        --n_episodes 500 --device cuda
"""
import argparse, json, os, sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.llm_gfn.run_experiment import (
    GPT2Backbone, GRUBackbone, ProcessRewardModel,
    make_arithmetic_problems, train_prm, train_llm_gfn,
    extract_policies, compute_prm_outputs,
    HAS_TRANSFORMERS, HAS_TORCH, VOCAB_SIZE, N_PROBLEMS, N_PRM_TRAIN, MAX_STEPS,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--member_id",   type=int, required=True)
    parser.add_argument("--output_dir",  default="results/llm_gfn")
    parser.add_argument("--n_total",     type=int, default=80,
                        help="Total members (train+test) for PCA fit")
    parser.add_argument("--n_episodes",  type=int, default=500)
    parser.add_argument("--n_prm_label", type=int, default=N_PRM_TRAIN)
    parser.add_argument("--pca_dim",     type=int, default=2)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--backbone",    default="auto", choices=["auto", "gpt2", "gru"],
                        help="'auto' uses GPT-2 if transformers available, else GRU")
    args = parser.parse_args()

    if not HAS_TORCH:
        raise RuntimeError("torch required.")

    os.makedirs(os.path.join(args.output_dir, "members"), exist_ok=True)

    # Shared objects
    use_gpt2 = (args.backbone == "gpt2") or (args.backbone == "auto" and HAS_TRANSFORMERS)
    if use_gpt2 and HAS_TRANSFORMERS:
        backbone = GPT2Backbone(freeze=True)
        backbone.eval()
    else:
        backbone = GRUBackbone(vocab_size=VOCAB_SIZE, hidden_size=128)
        backbone.eval()
        use_gpt2 = False

    problems  = make_arithmetic_problems(n=N_PROBLEMS, seed=0)
    ref_tokens = [i for i, c in enumerate("123") if c in "0123456789"]  # fallback
    from experiments.llm_gfn.run_experiment import TOKEN2IDX, ARITH_VOCAB
    ref_tokens = [TOKEN2IDX.get(c, 0) for c in problems[0]["prompt"][:MAX_STEPS]
                  if c in TOKEN2IDX]

    # Build PRM for this member
    prm = train_prm(backbone, problems,
                    seed=100 + args.member_id,
                    n_label=args.n_prm_label)

    # Fit PCA across all members (need all PRM outputs for consistent embedding)
    # Each member computes its own PRM output; we store it and recompute PCA in
    # analyze_from_members.  Here we just store mu placeholder = prm output vector.
    pout_i = compute_prm_outputs(backbone, [prm], problems)[0]  # (n_ref_evals,)

    # Train GFlowNet
    gfn = train_llm_gfn(backbone, prm, problems,
                         n_episodes=args.n_episodes,
                         device=args.device,
                         seed=args.member_id)
    pol = extract_policies(gfn, ref_tokens)  # (n_steps, vocab_size)

    np.savez(
        os.path.join(args.output_dir, "members", f"member_{args.member_id:04d}.npz"),
        member_id=args.member_id,
        policies=pol,
        ref_traj=np.array(ref_tokens),
        prm_output=pout_i,   # stored for post-hoc PCA
    )
    print(f"Saved member {args.member_id:04d}  policies={pol.shape}")

    # Save metadata on member 0
    if args.member_id == 0:
        meta = {
            "n_problems": N_PROBLEMS,
            "max_steps":  MAX_STEPS,
            "vocab_size": VOCAB_SIZE,
            "n_episodes": args.n_episodes,
            "ref_problem": problems[0]["prompt"],
            "backbone":   "gpt2" if use_gpt2 else "gru",
        }
        json.dump(meta, open(os.path.join(args.output_dir, "metadata.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
