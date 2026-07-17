"""Symbolic regression GFlowNet UQ experiment.

Target function: f(x) = sin(x) + 2 - x, evaluated at 50 points x in [0, 2*pi].
Reward noise: additive Wiener process W(x) scaled by noise_level=0.3, expressed
via a Karhunen-Loeve (KL) truncation of standard Brownian motion on [0, 2*pi].

KL parameterisation (2 modes)
------------------------------
  phi_1(x) = sqrt(2/(2*pi)) * sin(pi*x/(2*pi))    eigenvalue lambda_1 = (2*pi/(pi))^2
  phi_2(x) = sqrt(2/(2*pi)) * sin(3*pi*x/(2*pi))  eigenvalue lambda_2 = (2*pi/(3*pi))^2
  W(x; mu) ~ mu_1 * sqrt(lambda_1) * phi_1(x) + mu_2 * sqrt(lambda_2) * phi_2(x)
  mu_1, mu_2 ~ N(0, 1)

Token vocabulary (K=7)
-----------------------
  0: sin   1: cos   2: +   3: -   4: x   5: 2   6: EOS

GFlowNet builds token sequences of up to 9 tokens (L=9 steps).
Reward = exp(-MSE(eval(expr), f_noisy(x))).

PCE: degree=5, d=2 (2 KL modes).
n_train=100, n_test=50.

Requires: numpy, scipy, sklearn.  torch optional (falls back to numpy GRU).
"""
import os, sys, json, numpy as np

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as _np
        if isinstance(obj, _np.integer): return int(obj)
        if isinstance(obj, _np.floating): return float(obj)
        if isinstance(obj, _np.ndarray): return obj.tolist()
        return super().default(obj)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sklearn.decomposition import PCA
from core.pce_surrogate import TrajectoryPCESurrogate, run_ks_battery, calibration_coverage

try:
    import torch, torch.nn as nn, torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_POINTS  = 50                              # evaluation grid size
X_EVAL    = np.linspace(0, 2 * np.pi, N_POINTS)
F_TARGET  = np.sin(X_EVAL) + 2 - X_EVAL   # target function values (N_POINTS,)

TOKENS    = ["sin", "cos", "+", "-", "x", "2", "EOS"]
K_ACTIONS = len(TOKENS)   # 7
EOS_IDX   = TOKENS.index("EOS")
N_STEPS   = 9             # maximum token sequence length

KL_L      = 2 * np.pi     # domain length
NOISE_LVL = 0.3

# Karhunen-Loeve eigenvalues for modes 1 and 2 (Brownian motion on [0, L])
_KL_LAMBDA_1 = (KL_L / np.pi)     ** 2   # = (2*pi / pi)^2 = 4
_KL_LAMBDA_2 = (KL_L / (3*np.pi)) ** 2   # = (2*pi / 3*pi)^2 = 4/9

# ---------------------------------------------------------------------------
# KL basis functions and noisy reward construction
# ---------------------------------------------------------------------------

def kl_phi_1(x: np.ndarray) -> np.ndarray:
    """First KL eigenfunction of standard Brownian motion on [0, L]."""
    return np.sqrt(2.0 / KL_L) * np.sin(np.pi * x / KL_L)


def kl_phi_2(x: np.ndarray) -> np.ndarray:
    """Second KL eigenfunction of standard Brownian motion on [0, L]."""
    return np.sqrt(2.0 / KL_L) * np.sin(3.0 * np.pi * x / KL_L)


def wiener_kl(mu1: float, mu2: float, x: np.ndarray) -> np.ndarray:
    """Truncated KL approximation of Wiener process at points x.

    Args:
        mu1, mu2: independent N(0,1) random coefficients.
        x:        evaluation points, shape (N,).

    Returns:
        W: (N,) Wiener process realisation.
    """
    w  = mu1 * np.sqrt(_KL_LAMBDA_1) * kl_phi_1(x)
    w += mu2 * np.sqrt(_KL_LAMBDA_2) * kl_phi_2(x)
    return w


def noisy_target(mu1: float, mu2: float) -> np.ndarray:
    """Compute the noisy target at X_EVAL for given KL coefficients."""
    return F_TARGET + NOISE_LVL * wiener_kl(mu1, mu2, X_EVAL)


# ---------------------------------------------------------------------------
# Expression evaluator
# ---------------------------------------------------------------------------

def eval_token_sequence(tokens: list, x: np.ndarray) -> np.ndarray:
    """Evaluate a postfix-like token sequence at points x.

    Grammar: the sequence is interpreted as a stack machine.
      - 'x'  : push x
      - '2'  : push 2.0
      - 'sin': pop a; push sin(a)
      - 'cos': pop a; push cos(a)
      - '+'  : pop b, a; push a+b
      - '-'  : pop b, a; push a-b
      - 'EOS': stop processing

    Returns the top of stack after processing, or zeros on invalid expression.
    """
    stack = []
    for tok in tokens:
        if tok == "EOS":
            break
        elif tok == "x":
            stack.append(x.copy())
        elif tok == "2":
            stack.append(np.full_like(x, 2.0))
        elif tok == "sin":
            if len(stack) < 1:
                return np.zeros_like(x)
            a = stack.pop(); stack.append(np.sin(a))
        elif tok == "cos":
            if len(stack) < 1:
                return np.zeros_like(x)
            a = stack.pop(); stack.append(np.cos(a))
        elif tok == "+":
            if len(stack) < 2:
                return np.zeros_like(x)
            b, a = stack.pop(), stack.pop(); stack.append(a + b)
        elif tok == "-":
            if len(stack) < 2:
                return np.zeros_like(x)
            b, a = stack.pop(), stack.pop(); stack.append(a - b)
    return stack[-1] if stack else np.zeros_like(x)


def mse_reward(tokens: list, f_noisy: np.ndarray) -> float:
    """Compute exp(-MSE) reward for a token sequence vs. noisy target."""
    pred = eval_token_sequence(tokens, X_EVAL)
    if not np.all(np.isfinite(pred)):
        return 1e-8
    mse = float(np.mean((pred - f_noisy) ** 2))
    return float(np.exp(-mse))


# ---------------------------------------------------------------------------
# GFlowNet: torch GRU implementation
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class SymRegGFlowNet(nn.Module):
        """GRU-based GFlowNet for symbolic regression token generation.

        At each step t the GRU receives the previous token embedding and
        hidden state, then outputs a softmax over K=7 tokens.  The network
        is trained with trajectory balance loss.
        """

        def __init__(self, hidden: int = 64, n_tokens: int = K_ACTIONS):
            super().__init__()
            self.hidden   = hidden
            self.n_tokens = n_tokens
            self.embed    = nn.Embedding(n_tokens, hidden)
            self.gru      = nn.GRUCell(hidden, hidden)
            self.head     = nn.Linear(hidden, n_tokens)
            self.log_Z    = nn.Parameter(torch.tensor(0.0))

        def forward_step(
            self, token_idx: torch.Tensor, h: torch.Tensor
        ) -> tuple:
            """One GRU step.

            Args:
                token_idx: (batch,) long tensor of previous token indices.
                h:         (batch, hidden) GRU hidden state.

            Returns:
                log_probs: (batch, n_tokens) log-softmax over actions.
                h_new:     (batch, hidden) updated hidden state.
            """
            e     = self.embed(token_idx)
            h_new = self.gru(e, h)
            return torch.log_softmax(self.head(h_new), dim=-1), h_new

        def get_policy(self, token_idx: int, h_np: np.ndarray) -> tuple:
            """Numpy-friendly inference for a single position.

            Returns:
                probs:  (n_tokens,) numpy probability array.
                h_new:  (hidden,) numpy updated hidden state.
            """
            with torch.no_grad():
                t = torch.tensor([token_idx], dtype=torch.long)
                h = torch.tensor(h_np[None], dtype=torch.float32)
                lp, h_new = self.forward_step(t, h)
                return torch.exp(lp[0]).cpu().numpy(), h_new[0].cpu().numpy()

        def initial_hidden(self) -> np.ndarray:
            return np.zeros(self.hidden, dtype=np.float32)

else:
    class SymRegGFlowNet:  # type: ignore[no-redef]
        """Fallback numpy GRU for environments without torch."""

        def __init__(self, hidden: int = 64, n_tokens: int = K_ACTIONS):
            rng = np.random.RandomState(0)
            self.hidden   = hidden
            self.n_tokens = n_tokens
            # GRU parameters (reset + update + new gate, compact form)
            s = np.sqrt(1.0 / hidden)
            self.Wz = rng.uniform(-s, s, (n_tokens + hidden, hidden))
            self.Wr = rng.uniform(-s, s, (n_tokens + hidden, hidden))
            self.Wn = rng.uniform(-s, s, (n_tokens + hidden, hidden))
            self.embed_W = rng.uniform(-s, s, (n_tokens, n_tokens))
            self.head_W  = rng.uniform(-s, s, (hidden, n_tokens))
            self.head_b  = np.zeros(n_tokens)
            self.log_Z   = 0.0

        def _embed(self, idx: int) -> np.ndarray:
            e = np.zeros(self.n_tokens); e[idx] = 1.0
            return e @ self.embed_W

        def _gru_step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
            xh = np.concatenate([x, h])
            z  = 1 / (1 + np.exp(-(xh @ self.Wz)))
            r  = 1 / (1 + np.exp(-(xh @ self.Wr)))
            n  = np.tanh(np.concatenate([x, r * h]) @ self.Wn)
            return (1 - z) * n + z * h

        def get_policy(self, token_idx: int, h: np.ndarray) -> tuple:
            e   = self._embed(token_idx)
            h_n = self._gru_step(e, h)
            logits = h_n @ self.head_W + self.head_b
            logits -= logits.max()
            exp_l  = np.exp(logits)
            probs  = exp_l / exp_l.sum()
            return probs, h_n

        def initial_hidden(self) -> np.ndarray:
            return np.zeros(self.hidden, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training (torch path)
# ---------------------------------------------------------------------------

def train_symreg_gfn_torch(
    gfn: "SymRegGFlowNet",
    f_noisy: np.ndarray,
    n_episodes: int = 1000,
    lr: float = 1e-3,
    temp: float = 1.0,
    device: str = "cpu",
) -> "SymRegGFlowNet":
    """Train SymRegGFlowNet with trajectory balance loss.

    Args:
        gfn:        SymRegGFlowNet (torch).
        f_noisy:    (N_POINTS,) noisy target values.
        n_episodes: number of trajectory episodes.
        lr:         Adam learning rate.
        temp:       reward temperature.
        device:     torch device string.

    Returns:
        Trained gfn (on CPU, in eval mode).
    """
    gfn = gfn.to(device)
    opt = optim.Adam(gfn.parameters(), lr=lr)
    rng = np.random.RandomState(0)
    loss_history = []

    for ep in range(n_episodes):
        gfn.train()
        h = torch.zeros(1, gfn.hidden, device=device)
        prev_tok = torch.zeros(1, dtype=torch.long, device=device)  # BOS = token 0
        log_pf   = torch.tensor(0.0, device=device)
        tokens   = []

        for step in range(N_STEPS):
            lp, h = gfn.forward_step(prev_tok, h)
            dist  = torch.distributions.Categorical(logits=lp[0])
            a     = dist.sample()
            log_pf = log_pf + dist.log_prob(a)
            tok = a.item()
            tokens.append(TOKENS[tok])
            prev_tok = a.unsqueeze(0)
            if tok == EOS_IDX:
                break

        r     = mse_reward(tokens, f_noisy)
        log_r = torch.tensor(np.log(r + 1e-15) / temp, device=device)
        loss  = (gfn.log_Z + log_pf - log_r) ** 2
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(gfn.parameters(), 1.0)
        opt.step()
        loss_history.append(float(loss.item()))

        if (ep + 1) % 500 == 0:
            print(f"    ep {ep+1}/{n_episodes}  loss={loss.item():.4f}")

    gfn.eval()
    gfn.loss_history = loss_history
    return gfn.cpu()


def train_symreg_gfn_numpy(
    gfn: "SymRegGFlowNet",
    f_noisy: np.ndarray,
    n_episodes: int = 1000,
    lr: float = 5e-3,
    temp: float = 1.0,
) -> "SymRegGFlowNet":
    """Train numpy SymRegGFlowNet with approximate gradient updates.

    Uses REINFORCE-style update on policy parameters.

    Args:
        gfn:        numpy SymRegGFlowNet.
        f_noisy:    (N_POINTS,) noisy target values.
        n_episodes: number of episodes.
        lr:         learning rate.
        temp:       reward temperature.

    Returns:
        Trained gfn.
    """
    rng = np.random.RandomState(0)
    for ep in range(n_episodes):
        h      = gfn.initial_hidden()
        prev   = EOS_IDX   # start token (use EOS as BOS placeholder)
        tokens = []
        lp_sum = 0.0
        # Store per-step data for proper gradient computation
        step_data = []  # (hidden_state, action, probs) per step

        for step in range(N_STEPS):
            probs, h = gfn.get_policy(prev, h)
            a        = rng.choice(K_ACTIONS, p=probs)
            lp_sum  += np.log(probs[a] + 1e-15)
            step_data.append((h.copy(), a, probs.copy()))
            tokens.append(TOKENS[a])
            prev = a
            if a == EOS_IDX:
                break

        r     = mse_reward(tokens, f_noisy)
        log_r = np.log(r + 1e-15) / temp
        tb_g  = 2.0 * (gfn.log_Z + lp_sum - log_r)

        # Proper gradient: d log p(a|h) / d head_W = h^T (e_a - probs)
        # Accumulate across trajectory steps
        grad_W = np.zeros_like(gfn.head_W)
        grad_b = np.zeros_like(gfn.head_b)
        for h_t, a_t, p_t in step_data:
            e_a = np.zeros(gfn.n_tokens)
            e_a[a_t] = 1.0
            grad_W += np.outer(h_t, e_a - p_t)
            grad_b += e_a - p_t
        gfn.head_W -= lr * tb_g * grad_W
        gfn.head_b -= lr * tb_g * grad_b
        gfn.log_Z  -= lr * tb_g

        if (ep + 1) % 500 == 0:
            print(f"    ep {ep+1}/{n_episodes}  reward={r:.4f}")
    return gfn


def train_symreg_gfn(
    f_noisy: np.ndarray,
    n_episodes: int = 1000,
    lr: float = 1e-3,
    temp: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
) -> "SymRegGFlowNet":
    """Unified training entry point.  Uses torch if available, else numpy.

    Args:
        f_noisy:    (N_POINTS,) noisy target values.
        n_episodes: training episodes.
        lr:         learning rate.
        temp:       reward temperature.
        device:     torch device (ignored when HAS_TORCH is False).
        seed:       RNG seed for weight initialisation.

    Returns:
        Trained SymRegGFlowNet.
    """
    if HAS_TORCH:
        torch.manual_seed(seed)
        gfn = SymRegGFlowNet(hidden=64, n_tokens=K_ACTIONS)
        return train_symreg_gfn_torch(gfn, f_noisy, n_episodes, lr, temp, device)
    else:
        np.random.seed(seed)
        gfn = SymRegGFlowNet(hidden=64, n_tokens=K_ACTIONS)
        return train_symreg_gfn_numpy(gfn, f_noisy, n_episodes, lr, temp)


# ---------------------------------------------------------------------------
# Policy collection
# ---------------------------------------------------------------------------

def greedy_trajectory(gfn: "SymRegGFlowNet") -> list:
    """Extract greedy (argmax) token sequence from a trained GFlowNet.

    Returns:
        List of token indices chosen at each step (length <= N_STEPS).
    """
    h = gfn.initial_hidden()
    prev = EOS_IDX
    traj = []
    for step in range(N_STEPS):
        probs, h = gfn.get_policy(prev, h)
        a = int(np.argmax(probs))
        traj.append(a)
        prev = a
        if a == EOS_IDX:
            break
    return traj


def collect_policies(gfn: "SymRegGFlowNet", ref_traj: list = None) -> np.ndarray:
    """Collect step-wise policy distributions along a reference trajectory.

    Args:
        gfn:       Trained GFlowNet.
        ref_traj:  Shared reference trajectory (token indices). If None,
                   uses this GFlowNet's own greedy decode (legacy behavior).

    Returns:
        policies: (N_STEPS, K_ACTIONS) probability array.
    """
    if ref_traj is None:
        ref_traj = greedy_trajectory(gfn)

    h = gfn.initial_hidden()
    prev = EOS_IDX  # BOS token
    policies = []
    for step in range(N_STEPS):
        probs, h = gfn.get_policy(prev, h)
        policies.append(probs.copy())
        if step < len(ref_traj):
            prev = ref_traj[step]
        else:
            prev = EOS_IDX
        if prev == EOS_IDX and step < N_STEPS - 1:
            # After EOS in reference trajectory, remaining policies are
            # evaluated at the EOS-continuation state (post-termination)
            for s2 in range(step + 1, N_STEPS):
                probs2, h = gfn.get_policy(EOS_IDX, h)
                policies.append(probs2.copy())
            break
    return np.array(policies[:N_STEPS])  # (N_STEPS, K_ACTIONS)


# ---------------------------------------------------------------------------
# Sequential experiment
# ---------------------------------------------------------------------------

def run_symreg_experiment(
    n_train: int = 100,
    n_test: int = 50,
    pce_degree: int = 5,
    pca_dim: int = 2,
    n_episodes: int = 1000,
    device: str = "cpu",
    output_dir: str = "results/symreg",
) -> dict:
    """Run full symbolic regression UQ experiment sequentially.

    Args:
        n_train:     number of train GFlowNets.
        n_test:      number of test GFlowNets.
        pce_degree:  PCE degree.
        pca_dim:     latent dimension (must equal 2 for 2 KL modes).
        n_episodes:  training episodes per GFlowNet.
        device:      torch device string (ignored if torch unavailable).
        output_dir:  directory for output files.

    Returns:
        Dictionary of results (also written to results.json).
    """
    os.makedirs(output_dir, exist_ok=True)
    members_dir = os.path.join(output_dir, "members")
    os.makedirs(members_dir, exist_ok=True)

    print("=" * 70)
    print("SYMBOLIC REGRESSION GFlowNet UQ EXPERIMENT")
    print("=" * 70)

    n_total = n_train + n_test

    # Sample KL coefficients (mu_1, mu_2) ~ N(0,1) for each member
    rng_main = np.random.RandomState(0)
    kl_coeffs = rng_main.randn(n_total, 2)   # (n_total, 2)

    # Compute noisy targets for each member
    f_noisy_all = np.array([
        noisy_target(kl_coeffs[i, 0], kl_coeffs[i, 1]) for i in range(n_total)
    ])   # (n_total, N_POINTS)

    # mu is the KL coefficient vector itself (already 2-dim, no PCA needed when
    # pca_dim=2 and the parameterisation is directly Gaussian); we still run PCA
    # on the noisy-function evaluations to stay consistent with the paper's
    # reward-parameterisation pipeline.
    pca = PCA(n_components=pca_dim)
    mu_tr = pca.fit_transform(f_noisy_all[:n_train])   # (n_train, pca_dim)
    mu_te = pca.transform(f_noisy_all[n_train:])       # (n_test, pca_dim)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Save metadata
    meta = {
        "n_train":    n_train,
        "n_test":     n_test,
        "pce_degree": pce_degree,
        "pca_dim":    pca_dim,
        "n_steps":    N_STEPS,
        "n_actions":  K_ACTIONS,
        "tokens":     TOKENS,
        "noise_level": NOISE_LVL,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    np.save(os.path.join(output_dir, "kl_coeffs.npy"), kl_coeffs)
    np.save(os.path.join(output_dir, "f_noisy_all.npy"), f_noisy_all)

    # Train GFlowNets (use member 0's greedy decode as shared reference trajectory)
    all_gfns = []
    for i in range(n_total):
        print(f"  Training member {i+1}/{n_total} ...")
        gfn = train_symreg_gfn(
            f_noisy_all[i], n_episodes=n_episodes, lr=1e-3,
            device=device, seed=i,
        )
        all_gfns.append(gfn)

    # Shared reference trajectory from member 0
    ref_traj = greedy_trajectory(all_gfns[0])
    print(f"  Reference trajectory (member 0): {ref_traj[:8]}{'...' if len(ref_traj)>8 else ''}")

    all_policies = []
    for i, gfn in enumerate(all_gfns):
        pol = collect_policies(gfn, ref_traj=ref_traj)  # (N_STEPS, K_ACTIONS)
        all_policies.append(pol)
        np.savez_compressed(
            os.path.join(members_dir, f"member_{i:04d}.npz"),
            policies=pol,
            kl_coeffs=kl_coeffs[i],
            ref_traj=np.array(ref_traj),
        )

    # Per-step policy arrays
    tr_pol = {s: np.array([all_policies[i][s] for i in range(n_train)])
              for s in range(N_STEPS)}
    te_pol = {s: np.array([all_policies[n_train + i][s] for i in range(n_test)])
              for s in range(N_STEPS)}

    # Fit PCE surrogate
    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(N_STEPS):
        tsurr.fit_step(s, mu_tr, tr_pol[s])
    sobol = tsurr.sobol_all_steps()

    # Print Sobol summary -- both raw numbers and accessible narrative.
    #
    # The two PCA components here correspond to the two Karhunen-Loeve
    # modes of the Wiener-process reward noise:
    #   PC1 ~ KL mode 1 (slow, large-amplitude global shift, eigenvalue 4)
    #   PC2 ~ KL mode 2 (faster spatial oscillation, eigenvalue 4/9)
    #
    # The narrative helper produces a plain-language interpretation
    # of which component drives which token decision, letting the data
    # speak for itself rather than hard-coding an interpretation.
    print("\n" + "=" * 60)
    print("SYMREG SOBOL RESULTS")
    print("=" * 60)
    for s in range(N_STEPS):
        v  = sobol[s]["variance"]
        mi = int(np.argmax(v)) if v.max() > 1e-10 else 0
        t  = sobol[s]["total_order"][mi] if v.max() > 1e-10 else np.zeros(pca_dim)
        tok_label = f"step {s}"
        print(f"  {tok_label:8s}: var={v.max():.4f}  "
              f"S_PC1={t[0]:.3f}  S_PC2={t[1]:.3f}")

    # Human-readable narrative summary
    print()
    print(tsurr.summarise_sobol(
        step_labels={s: f"token {s} ({TOKENS[s] if s < len(TOKENS) else '?'})"
                     for s in range(N_STEPS)},
        dim_labels=["PC1 (global shift)", "PC2 (fine structure)"],
    ))

    # KS battery
    ks_results = run_ks_battery(tsurr, te_pol)

    # Calibration coverage
    mc_samples = tsurr.sample_trajectory_policies(2000)
    coverage_per_step = {}
    for s in range(N_STEPS):
        coverage_per_step[s] = calibration_coverage(mc_samples[s], te_pol[s])

    results = {
        "sobol": {s: {k: v.tolist() for k, v in sv.items()}
                  for s, sv in sobol.items()},
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()) if hasattr(pca, "explained_variance_ratio_") else 1.0,
        "n_train": n_train,
        "n_test":  n_test,
        "pce_degree": pce_degree,
        "ks_results": {s: {"fraction_pass": v["fraction_pass"],
                            "n_pass": v["n_pass"],
                            "n_total": v["n_total"]}
                       for s, v in ks_results.items()},
        "calibration": {s: {str(a): c for a, c in cov.items()}
                        for s, cov in coverage_per_step.items()},
        "tokens": TOKENS,
    }
    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NpEncoder)
    print(f"\nResults saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Cluster analysis path (matching sachs pattern)
# ---------------------------------------------------------------------------

def analyze_from_members(
    output_dir: str = "results/symreg",
    n_train: int = 100,
    n_test: int = 50,
    pce_degree: int = 5,
    pca_dim: int = 2,
) -> dict:
    """Load saved member npz files, fit PCE, and compute Sobol indices.

    Called after an LSF array job has populated output_dir/members/.
    Mirrors the sachs_causal::analyze_from_members pattern.

    Args:
        output_dir:  directory containing metadata.json, f_noisy_all.npy,
                     and members/.
        n_train:     number of train ensemble members.
        n_test:      number of test ensemble members.
        pce_degree:  PCE degree.
        pca_dim:     latent dimension for PCA on noisy function evaluations.

    Returns:
        Dictionary of results (also written to results.json).
    """
    import glob

    meta_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        n_steps = meta["n_steps"]
    else:
        _probe = np.load(sorted(
            __import__("glob").glob(os.path.join(output_dir, "members", "member_*.npz"))
        )[0])
        n_steps = int(_probe["policies"].shape[0])
        print(f"  metadata.json not found; inferred n_steps={n_steps}")

    # Load all member files
    member_files = sorted(glob.glob(os.path.join(output_dir, "members", "member_*.npz")))
    if len(member_files) < n_train + n_test:
        raise RuntimeError(
            f"Expected {n_train + n_test} members, found {len(member_files)}"
        )

    all_policies = []
    all_kl       = []
    for fp in member_files[: n_train + n_test]:
        d = np.load(fp)
        all_policies.append(d["policies"])   # (n_steps, n_actions)
        all_kl.append(d["kl_coeffs"])        # (2,)

    # Reconstruct mu: use kl_coeffs directly (analytically known parameterisation)
    # if f_noisy_all.npy not present; otherwise fall back to PCA for consistency.
    pca = None
    f_noisy_path = os.path.join(output_dir, "f_noisy_all.npy")
    if os.path.exists(f_noisy_path):
        f_noisy_all = np.load(f_noisy_path)
        pca = PCA(n_components=pca_dim)
        f_noisy_subset = f_noisy_all[: n_train + n_test]
        mu_tr = pca.fit_transform(f_noisy_subset[:n_train])
        mu_te = pca.transform(f_noisy_subset[n_train:])
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        # kl_coeffs ARE the natural 2D parameterisation -- standardise and use directly
        from sklearn.preprocessing import StandardScaler as _SS
        _scaler = _SS()
        kl_arr = np.array(all_kl)
        mu_tr = _scaler.fit_transform(kl_arr[:n_train])
        mu_te = _scaler.transform(kl_arr[n_train:])
        print("  f_noisy_all.npy not found; using kl_coeffs as mu directly.")

    # Per-step policy arrays
    tr_pol = {s: np.array([all_policies[i][s] for i in range(n_train)])
              for s in range(n_steps)}
    te_pol = {s: np.array([all_policies[n_train + i][s] for i in range(n_test)])
              for s in range(n_steps)}

    # Fit PCE surrogate
    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(n_steps):
        tsurr.fit_step(s, mu_tr, tr_pol[s])
    sobol = tsurr.sobol_all_steps()

    print("\n" + "=" * 60)
    print("SYMREG SOBOL RESULTS")
    print("=" * 60)
    for s in range(n_steps):
        v  = sobol[s]["variance"]
        mi = int(np.argmax(v)) if v.max() > 1e-10 else 0
        t  = sobol[s]["total_order"][mi] if v.max() > 1e-10 else np.zeros(pca_dim)
        print(f"  Step {s:2d}: var={v.max():.4f}  S_PC1={t[0]:.3f}  S_PC2={t[1]:.3f}")

    print()
    print(tsurr.summarise_sobol(
        step_labels={s: f"token {s}" for s in range(n_steps)},
        dim_labels=["PC1 (global shift)", "PC2 (fine structure)"],
    ))

    # KS battery
    ks_results = run_ks_battery(tsurr, te_pol)

    # Calibration coverage
    mc_samples = tsurr.sample_trajectory_policies(2000)
    coverage_per_step = {}
    for s in range(n_steps):
        coverage_per_step[s] = calibration_coverage(mc_samples[s], te_pol[s])

    results = {
        "sobol": {s: {k: v.tolist() for k, v in sv.items()}
                  for s, sv in sobol.items()},
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()) if hasattr(pca, "explained_variance_ratio_") else 1.0,
        "n_train": n_train,
        "n_test":  n_test,
        "pce_degree": pce_degree,
        "ks_results": {s: {"fraction_pass": v["fraction_pass"],
                            "n_pass": v["n_pass"],
                            "n_total": v["n_total"]}
                       for s, v in ks_results.items()},
        "calibration": {s: {str(a): c for a, c in cov.items()}
                        for s, cov in coverage_per_step.items()},
        "tokens": TOKENS,
    }
    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NpEncoder)
    print(f"\nResults saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Symbolic regression GFlowNet UQ experiment"
    )
    parser.add_argument("--run_mode",   choices=["sequential", "analyze"],
                        default="sequential",
                        help="'sequential' trains all members here; "
                             "'analyze' loads pre-trained members from output_dir/members/")
    parser.add_argument("--output_dir", default="results/symreg")
    parser.add_argument("--n_train",    type=int, default=100)
    parser.add_argument("--n_test",     type=int, default=50)
    parser.add_argument("--pce_degree", type=int, default=5)
    parser.add_argument("--pca_dim",    type=int, default=2)
    parser.add_argument("--n_episodes", type=int, default=1000)
    args = parser.parse_args()

    if args.run_mode == "analyze":
        analyze_from_members(
            output_dir=args.output_dir,
            n_train=args.n_train,
            n_test=args.n_test,
            pce_degree=args.pce_degree,
            pca_dim=args.pca_dim,
        )
    else:
        device = "cpu"
        if HAS_TORCH:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        run_symreg_experiment(
            n_train=args.n_train,
            n_test=args.n_test,
            pce_degree=args.pce_degree,
            pca_dim=args.pca_dim,
            n_episodes=args.n_episodes,
            device=device,
            output_dir=args.output_dir,
        )

