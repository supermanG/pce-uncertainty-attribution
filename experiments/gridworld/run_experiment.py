"""Grid-world GFlowNet UQ experiment -- discrete and continuous reward modes.

Discrete mode
-------------
5x5 grid, 4 directional actions + stop (K=5), up to 5 trajectory steps.
Reward is the sum of grid-cell values, parameterised by 4 zone-level shifts
(each drawn from {-1, -0.5, 0, 0.5, 1}).  PCE degree=5.

Continuous mode
---------------
2D continuous reward consisting of 3 Gaussian bumps whose amplitudes are
drawn from N(0,1).  Agent moves on a discretised 5x5 grid with 5 movement
directions per step, 4 steps.  PCE degree=5.

Both modes fit a TrajectoryPCESurrogate, compute Sobol indices, run KS tests,
and write results/gridworld/{discrete,continuous}/results.json.

CPU-only; typical wall-time <2 min per member.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 5          # 5x5 grid
N_CELLS   = GRID_SIZE * GRID_SIZE   # 25 cells

# Discrete mode
DISCRETE_ACTIONS  = 5  # up, down, left, right, stop
DISCRETE_N_STEPS  = 5
ZONE_VALUES       = [-1.0, -0.5, 0.0, 0.5, 1.0]  # 5 choices per zone

# Continuous mode
CONT_ACTIONS = 5       # up, down, left, right, stay
CONT_N_STEPS = 4
N_BUMPS      = 3       # 3 Gaussian bumps, each with uncertain amplitude

# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------

def _zone_for_cell(row: int, col: int) -> int:
    """Assign each cell to one of 4 quadrant zones (0..3)."""
    top  = row < GRID_SIZE // 2 + GRID_SIZE % 2
    left = col < GRID_SIZE // 2 + GRID_SIZE % 2
    return (0 if top else 2) + (0 if left else 1)


def build_zone_map() -> np.ndarray:
    """Return (GRID_SIZE, GRID_SIZE) integer array of zone indices."""
    zmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            zmap[r, c] = _zone_for_cell(r, c)
    return zmap

ZONE_MAP = build_zone_map()   # cached at import time


def compute_reward_grid(zone_shifts: np.ndarray) -> np.ndarray:
    """Compute (GRID_SIZE, GRID_SIZE) reward grid from 4 zone shift values.

    Args:
        zone_shifts: (4,) array, each element in ZONE_VALUES.

    Returns:
        grid: (GRID_SIZE, GRID_SIZE) float array.
    """
    base = np.zeros((GRID_SIZE, GRID_SIZE))
    for z, v in enumerate(zone_shifts):
        base += (ZONE_MAP == z).astype(float) * v
    # Add small spatial gradient to break degeneracy
    rows, cols = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
    base += (rows + cols) * 0.1
    return base


def sample_zone_shifts(rng: np.random.RandomState) -> np.ndarray:
    """Sample one set of 4 zone-shift values uniformly from ZONE_VALUES."""
    return np.array([rng.choice(ZONE_VALUES) for _ in range(4)])


# ---------------------------------------------------------------------------
# Continuous reward helpers
# ---------------------------------------------------------------------------

BUMP_CENTRES = np.array([[1.0, 1.0], [3.0, 3.0], [1.0, 3.0]])  # fixed centres


def compute_cont_reward_grid(amplitudes: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Compute (GRID_SIZE, GRID_SIZE) reward grid from 3 Gaussian bump amplitudes.

    Args:
        amplitudes: (N_BUMPS,) array of amplitude draws from N(0,1).
        sigma:      width of each Gaussian.

    Returns:
        grid: (GRID_SIZE, GRID_SIZE) float array.
    """
    xs = np.arange(GRID_SIZE, dtype=float)
    rows, cols = np.meshgrid(xs, xs, indexing="ij")  # (5,5)
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for a, (cr, cc) in zip(amplitudes, BUMP_CENTRES):
        grid += a * np.exp(-((rows - cr) ** 2 + (cols - cc) ** 2) / (2 * sigma ** 2))
    return grid


# ---------------------------------------------------------------------------
# GFlowNet (discrete) -- pure numpy, CPU-only
# ---------------------------------------------------------------------------

class GridGFlowNet:
    """Tabular-ish MLP policy for the 5x5 discrete grid-world.

    State representation: flattened one-hot position (25) + step fraction (1) = 26 inputs.
    Policy: two-layer network with softmax output over K=5 actions.
    Trajectory balance loss trained with SGD (numpy).
    """

    def __init__(self, n_actions: int = DISCRETE_ACTIONS, hidden: int = 32, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.n_actions = n_actions
        fan_in = N_CELLS + 1
        self.W1 = rng.randn(fan_in, hidden) * np.sqrt(2.0 / fan_in)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, n_actions) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(n_actions)
        self.log_Z = 0.0   # learned partition function (scalar)

    def _state_vec(self, pos: int, step: int, max_steps: int) -> np.ndarray:
        x = np.zeros(N_CELLS + 1)
        x[pos] = 1.0
        x[-1] = step / max(max_steps, 1)
        return x

    def _forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits -= logits.max()
        e = np.exp(logits)
        return e / e.sum()

    def get_policy(self, pos: int, step: int, max_steps: int) -> np.ndarray:
        """Return probability vector over actions given current state."""
        x = self._state_vec(pos, step, max_steps)
        return self._forward(x)

    def _trajectory_balance_loss(
        self, reward_grid: np.ndarray, max_steps: int, rng: np.random.RandomState, temp: float
    ) -> float:
        """Sample one trajectory, compute TB loss, return (loss, grad_dict)."""
        pos   = rng.randint(N_CELLS)  # random start
        row, col = divmod(pos, GRID_SIZE)
        log_pf = 0.0
        log_r  = 0.0
        grads  = {"W1": np.zeros_like(self.W1), "b1": np.zeros_like(self.b1),
                  "W2": np.zeros_like(self.W2), "b2": np.zeros_like(self.b2)}

        # Direction offsets: up(-row), down(+row), left(-col), right(+col)
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for step in range(max_steps):
            x   = self._state_vec(pos, step, max_steps)
            h   = np.tanh(x @ self.W1 + self.b1)
            raw = h @ self.W2 + self.b2
            raw -= raw.max()
            e   = np.exp(raw)
            p   = e / e.sum()

            # Build valid action mask
            mask = np.ones(self.n_actions)
            for a, (dr, dc) in enumerate(DIRS):
                nr, nc = row + dr, col + dc
                if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                    mask[a] = 0.0
            mask[-1] = 1.0  # stop always valid
            p_masked = p * mask
            p_masked_sum = p_masked.sum()
            if p_masked_sum < 1e-15:
                p_masked = mask / mask.sum()
            else:
                p_masked = p_masked / p_masked_sum

            a = rng.choice(self.n_actions, p=p_masked)
            log_pf += np.log(p_masked[a] + 1e-15)
            log_r  += reward_grid[row, col] / temp

            # Backprop through this step: dL/d(raw_a) = d/d(raw_a) log(p_masked[a])
            # Since p_masked = softmax then mask-renorm, gradient is approximate but
            # sufficient for learning the policy direction.
            delta_out = -p.copy()
            delta_out[a] += 1.0
            dW2 = np.outer(h, delta_out)
            db2 = delta_out
            dh  = delta_out @ self.W2.T
            dtanh = (1 - h ** 2) * dh
            dW1 = np.outer(x, dtanh)
            db1 = dtanh
            grads["W1"] += dW1; grads["b1"] += db1
            grads["W2"] += dW2; grads["b2"] += db2

            if a == self.n_actions - 1:  # stop
                break
            dr, dc = DIRS[a]
            row += dr; col += dc
            pos = row * GRID_SIZE + col

        loss = (self.log_Z + log_pf - log_r) ** 2
        tb_grad = 2.0 * (self.log_Z + log_pf - log_r)
        # Scale parameter grads by TB gradient factor
        for k in grads:
            grads[k] *= tb_grad
        return loss, grads

    def train(
        self,
        reward_grid: np.ndarray,
        n_episodes: int = 500,
        lr: float = 5e-3,
        max_steps: int = DISCRETE_N_STEPS,
        temp: float = 1.0,
        seed: int = 0,
    ) -> "GridGFlowNet":
        """Train the GFlowNet with trajectory balance loss (numpy SGD).

        Args:
            reward_grid: (GRID_SIZE, GRID_SIZE) float reward values.
            n_episodes:  number of training episodes.
            lr:          learning rate.
            max_steps:   maximum trajectory length.
            temp:        reward temperature.
            seed:        RNG seed for trajectory sampling.

        Returns:
            self (for chaining).  Loss history is stored in self.loss_history.
        """
        rng  = np.random.RandomState(seed)
        # Adam accumulators
        m = {k: np.zeros_like(v) for k, v in
             [("W1", self.W1), ("b1", self.b1), ("W2", self.W2), ("b2", self.b2)]}
        v = {k: np.zeros_like(val) for k, val in
             [("W1", self.W1), ("b1", self.b1), ("W2", self.W2), ("b2", self.b2)]}
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        t = 0
        self.loss_history = []

        for ep in range(n_episodes):
            t += 1
            loss, grads = self._trajectory_balance_loss(reward_grid, max_steps, rng, temp)
            self.loss_history.append(float(loss))
            # log_Z gradient
            self.log_Z -= lr * 2.0 * (self.log_Z - 0.0) * 0.0  # no separate reg
            for k, param in [("W1", self.W1), ("b1", self.b1),
                              ("W2", self.W2), ("b2", self.b2)]:
                g = grads[k]
                m[k] = beta1 * m[k] + (1 - beta1) * g
                v[k] = beta2 * v[k] + (1 - beta2) * g ** 2
                m_hat = m[k] / (1 - beta1 ** t)
                v_hat = v[k] / (1 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

            if (ep + 1) % 200 == 0:
                print(f"    ep {ep+1}/{n_episodes}  loss={loss:.4f}")
        return self

    def collect_policies(self, max_steps: int) -> np.ndarray:
        """Collect policy at each step along a fixed reference trajectory.

        Returns:
            policies: (max_steps, n_actions) array of probability vectors.
        """
        pos = 0  # always start top-left for the reference trajectory
        row, col = 0, 0
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        policies = []
        for step in range(max_steps):
            p = self.get_policy(pos, step, max_steps)
            policies.append(p)
            # Follow deterministic reference: move right then down
            nc = col + 1
            if nc < GRID_SIZE:
                col = nc
            elif row + 1 < GRID_SIZE:
                row += 1
            pos = row * GRID_SIZE + col
        return np.array(policies)  # (max_steps, n_actions)


# ---------------------------------------------------------------------------
# Reward parameter sampling
# ---------------------------------------------------------------------------

def sample_reward_configs(n: int, mode: str, seed: int = 0) -> tuple:
    """Sample n reward configurations and return (params_array, grids_array).

    Discrete: params_array shape (n, 4) with zone shifts.
    Continuous: params_array shape (n, N_BUMPS) with Gaussian amplitudes.
    Grids: (n, GRID_SIZE, GRID_SIZE).
    """
    rng = np.random.RandomState(seed)
    if mode == "discrete":
        params = np.array([sample_zone_shifts(rng) for _ in range(n)])
        grids  = np.array([compute_reward_grid(p) for p in params])
    else:
        params = rng.randn(n, N_BUMPS)
        grids  = np.array([compute_cont_reward_grid(p) for p in params])
    return params, grids


# ---------------------------------------------------------------------------
# Single member training (used both by sequential and cluster workflows)
# ---------------------------------------------------------------------------

def train_single_member(
    reward_grid: np.ndarray,
    mode: str,
    seed: int = 0,
    n_episodes: int = 400,
) -> np.ndarray:
    """Train one GFlowNet on the given reward grid and return policy array.

    Args:
        reward_grid: (GRID_SIZE, GRID_SIZE) float array.
        mode:        'discrete' or 'continuous'.
        seed:        training RNG seed.
        n_episodes:  number of training episodes.

    Returns:
        policies: (n_steps, n_actions) probability vectors along the reference
                  trajectory.
    """
    n_actions = DISCRETE_ACTIONS if mode == "discrete" else CONT_ACTIONS
    n_steps   = DISCRETE_N_STEPS if mode == "discrete" else CONT_N_STEPS
    gfn = GridGFlowNet(n_actions=n_actions, hidden=32, seed=seed)
    gfn.train(reward_grid, n_episodes=n_episodes, lr=5e-3,
              max_steps=n_steps, temp=1.0, seed=seed)
    return gfn.collect_policies(n_steps), getattr(gfn, "loss_history", [])


# ---------------------------------------------------------------------------
# Sequential experiment
# ---------------------------------------------------------------------------

def run_gridworld_experiment(
    mode: str = "discrete",
    n_train: int = 50,
    n_test: int = 100,
    pce_degree: int = 5,
    pca_dim: int = 2,
    n_episodes: int = 400,
    output_dir: str = "results/gridworld/discrete",
) -> dict:
    """Run the full grid-world UQ experiment sequentially (single machine).

    Args:
        mode:        'discrete' or 'continuous'.
        n_train:     number of train GFlowNets.
        n_test:      number of test GFlowNets.
        pce_degree:  degree of PCE surrogate.
        pca_dim:     PCA latent dimension for reward parameterisation.
        n_episodes:  training episodes per GFlowNet.
        output_dir:  where to write results.json and members/.

    Returns:
        Dictionary of results (also written to results.json).
    """
    os.makedirs(output_dir, exist_ok=True)
    members_dir = os.path.join(output_dir, "members")
    os.makedirs(members_dir, exist_ok=True)

    print("=" * 70)
    print(f"GRID-WORLD ({mode.upper()}) UQ EXPERIMENT")
    print("=" * 70)

    n_total   = n_train + n_test
    n_actions = DISCRETE_ACTIONS if mode == "discrete" else CONT_ACTIONS
    n_steps   = DISCRETE_N_STEPS if mode == "discrete" else CONT_N_STEPS

    # Sample reward configurations
    params, grids = sample_reward_configs(n_total, mode, seed=0)
    # grids: (n_total, GRID_SIZE, GRID_SIZE)

    # PCA on flattened reward grids to get d-dim reward parameterisation mu
    # Fit PCA on training data only to avoid test-set leakage
    flat_grids = grids.reshape(n_total, -1)  # (n_total, 25)
    flat_tr, flat_te = flat_grids[:n_train], flat_grids[n_train:]
    pca = PCA(n_components=pca_dim)
    pca.fit(flat_tr)
    mu_tr = pca.transform(flat_tr)
    mu_te = pca.transform(flat_te)
    pca_explained_var = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance: {pca_explained_var:.3f}")

    # Save metadata so analyze_from_members can reconstruct mu without retraining
    ref_traj = [(s, 0) for s in range(n_steps)]  # placeholder (step, action_idx)
    meta = {
        "mode": mode,
        "n_train": n_train,
        "n_test": n_test,
        "pce_degree": pce_degree,
        "pca_dim": pca_dim,
        "n_actions": n_actions,
        "n_steps": n_steps,
        "ref_traj": ref_traj,
        "grid_size": GRID_SIZE,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    np.save(os.path.join(output_dir, "reward_params.npy"), params)
    np.save(os.path.join(output_dir, "reward_grids.npy"), grids)

    # Train GFlowNets and collect policies
    all_policies = []
    all_loss_histories = []
    for i in range(n_total):
        print(f"  Training member {i+1}/{n_total} ...")
        pol, loss_hist = train_single_member(grids[i], mode=mode, seed=i, n_episodes=n_episodes)
        all_policies.append(pol)
        all_loss_histories.append(loss_hist)
        np.savez_compressed(
            os.path.join(members_dir, f"member_{i:04d}.npz"),
            policies=pol,
            ref_traj=np.array(ref_traj),
            reward_params=params[i],
        )

    # Convergence summary
    from core.pce_surrogate import check_convergence
    n_converged = 0
    for i, lh in enumerate(all_loss_histories):
        if lh:
            conv = check_convergence(np.array(lh))
            if conv["converged"]:
                n_converged += 1
            elif i < 5:  # warn for first few non-converged
                print(f"  WARNING: member {i} may not have converged "
                      f"(rel_change={conv['rel_change']:.4f})")
    print(f"  Convergence: {n_converged}/{n_total} members converged (rtol=5%)")

    # Organise per-step arrays
    tr_pol = {s: np.array([all_policies[i][s] for i in range(n_train)])
              for s in range(n_steps)}
    te_pol = {s: np.array([all_policies[n_train + i][s] for i in range(n_test)])
              for s in range(n_steps)}

    # Fit PCE surrogate
    tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
    for s in range(n_steps):
        tsurr.fit_step(s, mu_tr, tr_pol[s])
    sobol = tsurr.sobol_all_steps()

    # Print Sobol summary
    print("\n" + "=" * 60)
    print(f"GRID-WORLD ({mode.upper()}) SOBOL RESULTS")
    print("=" * 60)
    for s in range(n_steps):
        v = sobol[s]["variance"]
        mi = int(np.argmax(v)) if v.max() > 1e-10 else 0
        t  = sobol[s]["total_order"][mi] if v.max() > 1e-10 else np.zeros(pca_dim)
        labels = " ".join(f"S_PC{d+1}={t[d]:.3f}" for d in range(pca_dim))
        print(f"  Step {s:2d}: var={v.max():.4f}  {labels}")

    # Accessible narrative
    reward_desc = ("zone-level shifts" if mode == "discrete"
                   else "Gaussian bump amplitudes")
    print()
    print(tsurr.summarise_sobol(
        step_labels={s: f"move {s}" for s in range(n_steps)},
        dim_labels=[f"PC{i+1} ({reward_desc})" for i in range(pca_dim)],
    ))

    # KS battery
    ks_results = run_ks_battery(tsurr, te_pol)

    # Calibration coverage (marginal and joint)
    from core.pce_surrogate import calibration_coverage_joint
    mc_samples = tsurr.sample_trajectory_policies(2000)
    coverage_marginal = {}
    coverage_joint = {}
    for s in range(n_steps):
        coverage_marginal[s] = calibration_coverage(mc_samples[s], te_pol[s])
        coverage_joint[s] = calibration_coverage_joint(mc_samples[s], te_pol[s])

    results = {
        "sobol": {s: {k: v.tolist() for k, v in sv.items()}
                  for s, sv in sobol.items()},
        "pca_explained_variance": pca_explained_var,
        "pca_variance_per_component": pca.explained_variance_ratio_.tolist(),
        "n_train": n_train,
        "n_test": n_test,
        "pce_degree": pce_degree,
        "ks_results": {s: {"fraction_pass": v["fraction_pass"],
                            "n_pass": v["n_pass"],
                            "n_total": v["n_total"]}
                       for s, v in ks_results.items()},
        "calibration_marginal": {s: {str(a): c for a, c in cov.items()}
                                 for s, cov in coverage_marginal.items()},
        "calibration_joint": {s: {str(a): c for a, c in cov.items()}
                              for s, cov in coverage_joint.items()},
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
    output_dir: str = "results/gridworld/discrete",
    n_train: int = 50,
    n_test: int = 100,
    pce_degree: int = 5,
    pca_dim: int = 2,
) -> dict:
    """Load saved member npz files and fit PCE + compute Sobol indices.

    Called after an LSF array job has populated output_dir/members/.
    Mirrors the sachs_causal pattern exactly.

    Args:
        output_dir:  directory containing metadata.json and members/.
        n_train:     number of train ensemble members.
        n_test:      number of test ensemble members.
        pce_degree:  degree of PCE surrogate.
        pca_dim:     latent dimension for reward PCA.

    Returns:
        Dictionary of results (also written to results.json).
    """
    import glob

    meta_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        n_steps = meta["n_steps"]
        mode    = meta.get("mode", "discrete")
    else:
        # Reconstruct from first member file if metadata.json not present
        _probe = np.load(sorted(
            __import__("glob").glob(os.path.join(output_dir, "members", "member_*.npz"))
        )[0])
        n_steps = int(_probe["policies"].shape[0])
        mode    = "discrete"  # default; overridden by CLI --mode flag
        print(f"  metadata.json not found; inferred n_steps={n_steps}, mode={mode}")

    # Load all member files
    member_files = sorted(glob.glob(os.path.join(output_dir, "members", "member_*.npz")))
    if len(member_files) < n_train + n_test:
        raise RuntimeError(
            f"Expected {n_train + n_test} members, found {len(member_files)}"
        )

    all_policies    = []
    all_reward_params = []
    for fp in member_files[: n_train + n_test]:
        d = np.load(fp)
        all_policies.append(d["policies"])       # (n_steps, n_actions)
        all_reward_params.append(d["reward_params"])  # (4,) zone shifts

    # Reconstruct mu: PCA on reward_grids.npy if available, else use reward_params directly
    # Fit on training data only to avoid test-set leakage
    pca = None
    pca_explained_var = 1.0
    grids_path = os.path.join(output_dir, "reward_grids.npy")
    if os.path.exists(grids_path):
        grids = np.load(grids_path)
        flat_grids = grids[: n_train + n_test].reshape(n_train + n_test, -1)
        flat_tr, flat_te = flat_grids[:n_train], flat_grids[n_train:]
        pca = PCA(n_components=pca_dim)
        pca.fit(flat_tr)
        mu_tr = pca.transform(flat_tr)
        mu_te = pca.transform(flat_te)
        pca_explained_var = float(pca.explained_variance_ratio_.sum())
        print(f"  PCA explained variance: {pca_explained_var:.3f}")
    else:
        # reward_params (4 zone shifts) are the natural parameterisation -- standardise
        from sklearn.preprocessing import StandardScaler as _SS
        all_rp = np.array(all_reward_params)
        scaler = _SS()
        scaler.fit(all_rp[:n_train])
        mu_all = scaler.transform(all_rp)
        pca_dim = min(pca_dim, mu_all.shape[1])
        mu_all = mu_all[:, :pca_dim]
        mu_tr, mu_te = mu_all[:n_train], mu_all[n_train:]
        print("  reward_grids.npy not found; using reward_params as mu directly.")

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
    print(f"GRID-WORLD ({mode.upper()}) SOBOL RESULTS")
    print("=" * 60)
    for s in range(n_steps):
        v  = sobol[s]["variance"]
        mi = int(np.argmax(v)) if v.max() > 1e-10 else 0
        t  = sobol[s]["total_order"][mi] if v.max() > 1e-10 else np.zeros(pca_dim)
        labels = " ".join(f"S_PC{d+1}={t[d]:.3f}" for d in range(pca_dim))
        print(f"  Step {s:2d}: var={v.max():.4f}  {labels}")

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
        "pca_explained_variance": pca_explained_var,
        "n_train": n_train,
        "n_test": n_test,
        "pce_degree": pce_degree,
        "ks_results": {s: {"fraction_pass": v["fraction_pass"],
                            "n_pass": v["n_pass"],
                            "n_total": v["n_total"]}
                       for s, v in ks_results.items()},
        "calibration": {s: {str(a): c for a, c in cov.items()}
                        for s, cov in coverage_per_step.items()},
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
        description="Grid-world GFlowNet UQ experiment"
    )
    parser.add_argument("--mode",       choices=["discrete", "continuous"],
                        default="discrete")
    parser.add_argument("--run_mode",   choices=["sequential", "analyze"],
                        default="sequential",
                        help="'sequential' trains all members here; "
                             "'analyze' loads pre-trained members from output_dir/members/")
    parser.add_argument("--output_dir", default=None,
                        help="Override default results directory.")
    parser.add_argument("--n_train",    type=int, default=50)
    parser.add_argument("--n_test",     type=int, default=100)
    parser.add_argument("--pce_degree", type=int, default=5)
    parser.add_argument("--pca_dim",    type=int, default=2)
    parser.add_argument("--n_episodes", type=int, default=400)
    args = parser.parse_args()

    default_dir = f"results/gridworld/{args.mode}"
    out_dir = args.output_dir if args.output_dir else default_dir

    if args.run_mode == "analyze":
        analyze_from_members(
            output_dir=out_dir,
            n_train=args.n_train,
            n_test=args.n_test,
            pce_degree=args.pce_degree,
            pca_dim=args.pca_dim,
        )
    else:
        run_gridworld_experiment(
            mode=args.mode,
            n_train=args.n_train,
            n_test=args.n_test,
            pce_degree=args.pce_degree,
            pca_dim=args.pca_dim,
            n_episodes=args.n_episodes,
            output_dir=out_dir,
        )

