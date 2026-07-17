"""GFlowNet fine-tuning of GPT-2 with uncertain Process Reward Model (PRM).

This experiment provides the NMI-level LLM demonstration referenced in the paper.
A pre-trained GPT-2 model is fine-tuned via trajectory balance GFlowNet to generate
arithmetic chain-of-thought reasoning sequences.  The PRM -- a linear probe on GPT-2
hidden states trained on limited labelled examples -- carries substantial epistemic
uncertainty.  Different PRM training subsets yield different reward landscapes, and
the PCE surrogate decomposes the resulting policy uncertainty into contributions from
each principal component of PRM variation.

Task
----
Generate step-by-step solutions to 2-operation arithmetic problems of the form:
    "Q: (a op1 b) op2 c = ?  A: step1, step2, answer"
The GFlowNet policy selects tokens from GPT-2's vocabulary (restricted to arithmetic
tokens: digits, operators, space, EOS).  Steps: up to MAX_STEPS = 8 tokens.

PRM
---
A 2-layer MLP that takes GPT-2 last-hidden-state at each step and predicts a score
in [0,1] indicating solution quality.  Trained on N_PRM_TRAIN labelled (step, score)
pairs sampled from a reference pool.  Different seeds → different labelled subsets →
different PRMs → different reward functions.

PCE surrogate
-------------
PCA on PRM output vectors (evaluated on a reference problem set) → d=2 latent.
PCE degree=5 fitted to n_train=30 GFlowNet members.

Requirements
------------
    pip install transformers torch accelerate
    (GPT-2 weights downloaded automatically on first run)

Usage (single machine, GPU)
----------------------------
    python3 experiments/llm_gfn/run_experiment.py --n_train 30 --n_test 50

Usage (LSF cluster, via train_single_member.py)
-----------------------------------------------
    bash lsf/submit_llm.sh
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from core.pce_surrogate import TrajectoryPCESurrogate, calibration_coverage
from core.distributional_analysis import (
    analyse_trajectory_bimodality, plot_bimodality_panel, plot_surrogate_comparison,
)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not available; falling back to synthetic policies.")

try:
    from transformers import GPT2Tokenizer, GPT2Model
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers not available; using GRU backbone instead.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_STEPS   = 8      # maximum tokens in a generated sequence
N_PROBLEMS  = 40     # reference arithmetic problems for PRM evaluation
N_PRM_TRAIN = 20     # labelled steps per PRM (epistemic uncertainty source)
HIDDEN_DIM  = 768    # GPT-2 small hidden size

VARS = ["Raf", "Mek", "Plcg", "PIP2", "PIP3",
        "Erk", "Akt", "PKA",  "PKC",  "P38",  "Jnk"]  # unused here; kept for compat


# ---------------------------------------------------------------------------
# Arithmetic problem generator
# ---------------------------------------------------------------------------

def make_arithmetic_problems(n: int = N_PROBLEMS, seed: int = 0) -> List[dict]:
    """Generate simple 2-operation arithmetic problems with step-by-step solutions."""
    rng = np.random.RandomState(seed)
    ops = ["+", "-", "*"]
    problems = []
    for _ in range(n):
        a, b, c = rng.randint(1, 10, size=3)
        op1, op2 = rng.choice(ops, size=2)
        step1 = int(eval(f"{a}{op1}{b}"))
        answer = int(eval(f"{step1}{op2}{c}"))
        problems.append({
            "prompt": f"{a}{op1}{b}{op2}{c}",
            "step1":  step1,
            "answer": answer,
        })
    return problems


# ---------------------------------------------------------------------------
# Backbone: GPT-2 or GRU fallback
# ---------------------------------------------------------------------------

if not HAS_TORCH:
    # Minimal stubs so the module can be imported for analyze-only usage
    import types as _types
    nn = _types.SimpleNamespace(Module=object)

class GPT2Backbone(nn.Module):
    """Wraps a frozen GPT-2 as a feature extractor for the GFlowNet policy."""

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained("gpt2")
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.hidden_size = self.model.config.n_embd  # 768

    def encode(self, token_ids: List[int]) -> torch.Tensor:
        """Return last-hidden-state vector for the given token sequence."""
        if not token_ids:
            return torch.zeros(self.hidden_size)
        inp = torch.tensor([token_ids]).long()
        with torch.no_grad():
            out = self.model(inp)
        return out.last_hidden_state[0, -1]  # (768,)


class GRUBackbone(nn.Module):
    """GRU fallback when transformers is not available."""

    def __init__(self, vocab_size: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size + 1, 32, padding_idx=0)
        self.gru   = nn.GRU(32, hidden_size, batch_first=True)

    def encode(self, token_ids: List[int]) -> torch.Tensor:
        if not token_ids:
            return torch.zeros(self.hidden_size)
        inp = torch.tensor([[t + 1 for t in token_ids]]).long()
        with torch.no_grad():
            _, hn = self.gru(self.embed(inp))
        return hn.squeeze()


# ---------------------------------------------------------------------------
# Vocabulary: arithmetic tokens
# ---------------------------------------------------------------------------

ARITH_VOCAB = list("0123456789+-*/= ") + ["EOS"]
VOCAB_SIZE   = len(ARITH_VOCAB)
TOKEN2IDX    = {t: i for i, t in enumerate(ARITH_VOCAB)}
EOS_IDX      = TOKEN2IDX["EOS"]


# ---------------------------------------------------------------------------
# Process Reward Model (PRM)
# ---------------------------------------------------------------------------

class ProcessRewardModel(nn.Module):
    """2-layer MLP: backbone_hidden → [0,1] step quality score."""

    def __init__(self, input_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 1),         nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


def make_prm_dataset(
    backbone,
    problems: List[dict],
    n_label: int = N_PRM_TRAIN,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a labelled (hidden_state, quality_score) dataset for PRM training.

    Quality = 1.0 if the generated partial token is a digit from the correct
    step-1 answer; 0.5 for correct operator; 0.0 otherwise.  This gives a
    naturally uncertain scoring problem -- different labelled subsets weight
    different aspects of quality.
    """
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    for prob in problems:
        correct_digits = set(str(abs(prob["step1"])) + str(abs(prob["answer"])))
        for partial_len in range(1, MAX_STEPS):
            token_seq = list(prob["prompt"][:partial_len])
            h = backbone.encode([TOKEN2IDX.get(t, 0) for t in token_seq
                                  if t in TOKEN2IDX])
            last_char = prob["prompt"][partial_len - 1] if partial_len <= len(prob["prompt"]) else "?"
            if last_char in correct_digits:
                score = 1.0
            elif last_char in "+-*/=":
                score = 0.5
            else:
                score = 0.0
            X_list.append(h.detach().numpy())
            y_list.append(score)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Sub-sample to n_label examples
    idx = rng.choice(len(X), min(n_label, len(X)), replace=False)
    return torch.tensor(X[idx]), torch.tensor(y[idx])


def train_prm(backbone, problems: List[dict], seed: int = 0,
              n_label: int = N_PRM_TRAIN, n_epochs: int = 200) -> ProcessRewardModel:
    """Train a PRM on n_label labelled examples drawn with the given seed."""
    input_dim = backbone.hidden_size
    prm = ProcessRewardModel(input_dim=input_dim)
    X, y = make_prm_dataset(backbone, problems, n_label=n_label, seed=seed)
    opt = torch.optim.Adam(prm.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    for _ in range(n_epochs):
        opt.zero_grad()
        criterion(prm(X), y).backward()
        opt.step()
    prm.eval()
    return prm


# ---------------------------------------------------------------------------
# GFlowNet policy with LoRA-style adapter
# ---------------------------------------------------------------------------

class LLMGFlowNet(nn.Module):
    """GFlowNet policy: backbone encoding + lightweight adapter → token logits."""

    def __init__(self, backbone, vocab_size: int = VOCAB_SIZE,
                 adapter_dim: int = 64):
        super().__init__()
        self.backbone  = backbone
        self.adapter   = nn.Sequential(
            nn.Linear(backbone.hidden_size, adapter_dim), nn.ReLU(),
            nn.Linear(adapter_dim, vocab_size),
        )
        self.log_Z = nn.Parameter(torch.tensor(0.0))
        self.vocab_size = vocab_size

    def forward_policy(self, token_ids: List[int]) -> torch.Tensor:
        h = self.backbone.encode(token_ids)
        logits = self.adapter(h.float())
        return torch.log_softmax(logits, dim=-1)

    def get_policy(self, token_ids: List[int]) -> np.ndarray:
        with torch.no_grad():
            return torch.exp(self.forward_policy(token_ids)).numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_llm_gfn(
    backbone,
    prm: ProcessRewardModel,
    problems: List[dict],
    n_episodes: int = 500,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
) -> LLMGFlowNet:
    """Train one GFlowNet member with the given PRM reward."""
    gfn = LLMGFlowNet(backbone).to(device)
    opt = torch.optim.Adam(
        list(gfn.adapter.parameters()) + [gfn.log_Z], lr=lr
    )
    rng = np.random.RandomState(seed)

    for ep in range(n_episodes):
        prob = problems[rng.randint(len(problems))]
        tokens: List[int] = []
        log_pf = torch.tensor(0.0)

        for step in range(MAX_STEPS):
            logp = gfn.forward_policy(tokens)
            dist = torch.distributions.Categorical(logits=logp)
            a = dist.sample()
            log_pf = log_pf + dist.log_prob(a)
            if a.item() == EOS_IDX:
                break
            tokens.append(a.item())

        # Reward: PRM score at each step, aggregated as product
        if tokens:
            token_seq = [TOKEN2IDX.get(ARITH_VOCAB[t], 0) for t in tokens
                         if t < len(ARITH_VOCAB)]
            h = backbone.encode(token_seq)
            reward = float(torch.sigmoid(prm(h.float().unsqueeze(0))).item()) + 1e-6
        else:
            reward = 1e-6

        loss = (gfn.log_Z + log_pf - np.log(reward)) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()

    gfn.eval()
    return gfn


# ---------------------------------------------------------------------------
# Policy extraction along reference trajectory
# ---------------------------------------------------------------------------

def extract_policies(gfn: LLMGFlowNet, ref_tokens: List[int]) -> np.ndarray:
    """Extract policy at each step along the reference token sequence.

    Returns:
        policies: (n_steps, vocab_size) array of probability vectors
    """
    policies = []
    context: List[int] = []
    for tok in ref_tokens:
        policies.append(gfn.get_policy(context))
        context.append(tok)
    return np.array(policies)  # (n_steps, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# PRM output PCA (reward parameterisation)
# ---------------------------------------------------------------------------

def compute_prm_outputs(
    backbone, prms: List[ProcessRewardModel], problems: List[dict]
) -> np.ndarray:
    """Evaluate each PRM on the reference problem set.

    Returns:
        pout: (n_prm, n_reference_evals) matrix for PCA
    """
    # Reference: encode each problem prompt character-by-character
    ref_encodings = []
    for prob in problems:
        for length in range(1, min(len(prob["prompt"]) + 1, MAX_STEPS)):
            token_ids = [TOKEN2IDX.get(c, 0) for c in prob["prompt"][:length]
                         if c in TOKEN2IDX]
            ref_encodings.append(backbone.encode(token_ids).detach().numpy())

    ref_tensor = torch.tensor(np.array(ref_encodings, dtype=np.float32))

    pout = np.zeros((len(prms), len(ref_encodings)))
    for i, prm in enumerate(prms):
        with torch.no_grad():
            pout[i] = prm(ref_tensor).numpy()
    return pout


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_llm_experiment(
    n_train:     int = 30,
    n_test:      int = 50,
    pce_degree:  int = 5,
    pca_dim:     int = 2,
    n_episodes:  int = 500,
    n_prm_label: int = N_PRM_TRAIN,
    device:      str = "cpu",
    output_dir:  str = "results/llm_gfn",
) -> dict:
    """Run the full LLM GFlowNet + uncertain PRM experiment."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "members"), exist_ok=True)

    if not HAS_TORCH:
        raise RuntimeError("torch required for LLM experiment.")

    print("=" * 70)
    print("LLM GFlowNet + Uncertain PRM  (NMI experiment)")
    print("=" * 70)

    # Backbone (shared, frozen)
    if HAS_TRANSFORMERS:
        print("  Loading GPT-2 backbone...")
        backbone = GPT2Backbone(freeze=True).to(device)
        backbone.eval()
    else:
        print("  transformers not available; using GRU backbone.")
        backbone = GRUBackbone(vocab_size=VOCAB_SIZE, hidden_size=128).to(device)
        backbone.eval()

    problems = make_arithmetic_problems(n=N_PROBLEMS, seed=0)

    # Reference token sequence: first 8 tokens of problem 0's prompt
    ref_tokens = [TOKEN2IDX.get(c, 0) for c in problems[0]["prompt"][:MAX_STEPS]
                  if c in TOKEN2IDX]
    n_steps = len(ref_tokens)
    print(f"  Reference sequence: '{problems[0]['prompt'][:MAX_STEPS]}' ({n_steps} steps)")

    # Train PRMs (one per ensemble member)
    total = n_train + n_test
    print(f"  Training {total} PRMs (seed 100..{100+total-1})...")
    prms = [train_prm(backbone, problems, seed=100 + i, n_label=n_prm_label)
            for i in range(total)]

    # PCA on PRM outputs → reward parameterisation mu
    print("  Computing PRM output PCA...")
    pout = compute_prm_outputs(backbone, prms, problems)
    pca     = PCA(n_components=pca_dim)
    mu_raw_tr = pca.fit_transform(pout[:n_train])
    mu_raw_te = pca.transform(pout[n_train:])
    scaler  = StandardScaler()
    mu_tr   = scaler.fit_transform(mu_raw_tr)
    mu_te   = scaler.transform(mu_raw_te)
    mu_all  = np.concatenate([mu_tr, mu_te], axis=0)   # for per-member saving
    exp_var = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance: {exp_var:.3f}")

    # Train GFlowNet ensemble
    print(f"  Training {total} GFlowNet members ({n_episodes} episodes each)...")
    all_policies = []
    for i in range(total):
        gfn = train_llm_gfn(backbone, prms[i], problems,
                             n_episodes=n_episodes, device=device, seed=i)
        pol = extract_policies(gfn, ref_tokens)  # (n_steps, vocab_size)
        all_policies.append(pol)

        # Save member
        np.savez(
            os.path.join(output_dir, "members", f"member_{i:04d}.npz"),
            member_id=i,
            policies=pol,
            ref_traj=np.array(ref_tokens),
            mu=mu_all[i],
        )
        if (i + 1) % 10 == 0:
            split = "Train" if i < n_train else "Test"
            print(f"    {split}: {min(i+1, n_train)}/{n_train} | {max(0,i+1-n_train)}/{n_test}")

    # Fit PCE surrogate
    print("  Fitting PCE surrogate...")
    tr_pol = {s: np.array([all_policies[i][s] for i in range(n_train)])
              for s in range(n_steps)}
    te_pol = {s: np.array([all_policies[n_train + i][s] for i in range(n_test)])
              for s in range(n_steps)}

    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(n_steps):
        tsurr.fit_step(s, mu_tr, tr_pol[s])

    sobol = tsurr.sobol_all_steps()

    # Calibration
    surr_samples = tsurr.sample_trajectory_policies(n_samples=2000)
    calib = {}
    for s in range(n_steps):
        calib[s] = calibration_coverage(surr_samples[s], te_pol[s])

    print("\n" + "=" * 60)
    print("LLM GFLOWNET SOBOL RESULTS")
    print("=" * 60)
    for s in range(n_steps):
        sv  = sobol[s]
        var_arr = sv["variance"]
        mi  = int(np.argmax(var_arr)) if var_arr.max() > 1e-10 else 0
        tot = sv["total_order"][mi]
        tok = ARITH_VOCAB[ref_tokens[s]] if ref_tokens[s] < len(ARITH_VOCAB) else "?"
        print(f"  Step {s:2d} (token='{tok}'): "
              f"var={var_arr[mi]:.4f}  S_PC1={tot[0]:.3f}  S_PC2={tot[1]:.3f}")

    # Accessible Sobol narrative
    tok_labels = {}
    for s in range(n_steps):
        tok = ARITH_VOCAB[ref_tokens[s]] if ref_tokens[s] < len(ARITH_VOCAB) else "?"
        tok_labels[s] = f"token {s} ('{tok}')"
    print()
    print(tsurr.summarise_sobol(
        step_labels=tok_labels,
        dim_labels=["PC1 (PRM mean quality)", "PC2 (PRM problem spread)"],
    ))

    # -----------------------------------------------------------------
    # Distributional analysis: the surrogate captures more than Sobol
    # -----------------------------------------------------------------
    bimodality, narrative = analyse_trajectory_bimodality(
        surr_samples, action_idx=EOS_IDX, step_labels=tok_labels,
    )
    print(narrative)

    # Publication-quality figures
    fig = plot_bimodality_panel(
        surr_samples, action_idx=EOS_IDX,
        step_labels=tok_labels, action_name="EOS",
        title="LLM GFlowNet: EOS-action distributional structure",
        empirical_samples=te_pol,
        output_path=os.path.join(output_dir, "bimodality_panel.pdf"),
    )
    plot_surrogate_comparison(
        surr_samples, te_pol, action_idx=EOS_IDX,
        step_labels=tok_labels, action_name="EOS",
        title="LLM GFlowNet: PCE surrogate vs empirical ensemble",
        output_path=os.path.join(output_dir, "surrogate_comparison.pdf"),
    )

    results = {
        "sobol":               {s: {k: v.tolist() for k, v in sv.items()}
                                 for s, sv in sobol.items()},
        "vocabulary":          ARITH_VOCAB,
        "ref_tokens":          ref_tokens,
        "pca_explained_variance": exp_var,
        "n_train":             n_train,
        "n_test":              n_test,
        "pce_degree":          pce_degree,
        "backbone":            "gpt2" if HAS_TRANSFORMERS else "gru",
        "n_prm_label":         n_prm_label,
        "calibration":         {s: {str(k): v for k, v in cv.items()}
                                for s, cv in calib.items()},
        "bimodality":          {s: bm for s, bm in bimodality.items()},
    }

    meta = {
        "n_problems":  N_PROBLEMS,
        "max_steps":   MAX_STEPS,
        "vocab_size":  VOCAB_SIZE,
        "n_episodes":  n_episodes,
        "ref_problem": problems[0]["prompt"],
    }
    json.dump(meta,    open(os.path.join(output_dir, "metadata.json"), "w"), indent=2)
    json.dump(results, open(os.path.join(output_dir, "results.json"),  "w"), indent=2)
    print(f"\nResults saved to {output_dir}/results.json")
    return results


# ---------------------------------------------------------------------------
# Cluster workflow: analyze from saved member files
# ---------------------------------------------------------------------------

def analyze_from_members(
    output_dir:  str = "results/llm_gfn",
    n_train:     int = 30,
    n_test:      int = 50,
    pce_degree:  int = 5,
    pca_dim:     int = 2,
) -> dict:
    """Load saved member npz files and fit PCE + Sobol.  Called by LSF analyze job."""
    meta = json.load(open(os.path.join(output_dir, "metadata.json")))
    member_files = sorted(glob.glob(os.path.join(output_dir, "members", "member_*.npz")))
    if len(member_files) < n_train + n_test:
        raise RuntimeError(f"Expected {n_train+n_test} members, found {len(member_files)}")

    all_policies  = []
    all_prm_out   = []
    for fp in member_files[:n_train + n_test]:
        d = np.load(fp)
        all_policies.append(d["policies"])    # (n_steps, vocab_size)
        # Support both old ('mu') and new ('prm_output') key names
        if "mu" in d:
            all_prm_out.append(d["mu"])
        else:
            all_prm_out.append(d["prm_output"])   # (n_ref_evals,)

    # Derive 2-D mu via PCA on PRM outputs then standardise
    from sklearn.decomposition import PCA as _PCA
    pca_raw = _PCA(n_components=pca_dim)
    prm_arr = np.array(all_prm_out)
    mu_raw_tr = pca_raw.fit_transform(prm_arr[:n_train])
    mu_raw_te = pca_raw.transform(prm_arr[n_train:])
    scaler = StandardScaler()
    mu_tr = scaler.fit_transform(mu_raw_tr)
    mu_te = scaler.transform(mu_raw_te)
    print(f"  PRM PCA explained variance: {pca_raw.explained_variance_ratio_.sum():.3f}")
    n_steps = all_policies[0].shape[0]

    tr_pol = {s: np.array([all_policies[i][s] for i in range(n_train)])
              for s in range(n_steps)}
    te_pol = {s: np.array([all_policies[n_train + i][s] for i in range(n_test)])
              for s in range(n_steps)}

    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(n_steps):
        tsurr.fit_step(s, mu_tr, tr_pol[s])

    sobol = tsurr.sobol_all_steps()

    surr_samples = tsurr.sample_trajectory_policies(n_samples=2000)
    calib = {s: calibration_coverage(surr_samples[s], te_pol[s])
             for s in range(n_steps)}

    results = {
        "sobol":    {s: {k: v.tolist() for k, v in sv.items()}
                     for s, sv in sobol.items()},
        "n_train":  n_train,
        "n_test":   n_test,
        "pce_degree": pce_degree,
        "calibration": {s: {str(k): v for k, v in cv.items()}
                        for s, cv in calib.items()},
    }
    json.dump(results,
              open(os.path.join(output_dir, "results.json"), "w"), indent=2)
    print(f"Results saved to {output_dir}/results.json")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM GFlowNet + uncertain PRM experiment"
    )
    parser.add_argument("--mode",        choices=["train", "analyze"], default="train")
    parser.add_argument("--n_train",     type=int, default=30)
    parser.add_argument("--n_test",      type=int, default=50)
    parser.add_argument("--pce_degree",  type=int, default=5)
    parser.add_argument("--pca_dim",     type=int, default=2)
    parser.add_argument("--n_episodes",  type=int, default=500)
    parser.add_argument("--n_prm_label", type=int, default=N_PRM_TRAIN)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--output_dir",  default="results/llm_gfn")
    args = parser.parse_args()

    if args.mode == "train":
        run_llm_experiment(
            n_train=args.n_train,
            n_test=args.n_test,
            pce_degree=args.pce_degree,
            pca_dim=args.pca_dim,
            n_episodes=args.n_episodes,
            n_prm_label=args.n_prm_label,
            device=args.device,
            output_dir=args.output_dir,
        )
    else:
        analyze_from_members(
            output_dir=args.output_dir,
            n_train=args.n_train,
            n_test=args.n_test,
            pce_degree=args.pce_degree,
            pca_dim=args.pca_dim,
        )
