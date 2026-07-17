# Reproducibility Manifest

Instructions to reproduce all tables and figures in the manuscript.

## Environment

```bash
# Tested on Python 3.10+ (PyTorch requires <=3.12 for some CUDA builds)
conda create -n uqgfn python=3.10 -y && conda activate uqgfn
pip install -r requirements.txt
```

Hardware used for reported numbers: single NVIDIA A100 GPU (GFlowNet training), single CPU core (PCE fitting and sampling).

## Quick smoke test

Runs all three main experiments with small ensembles (~5 min on GPU):

```bash
python run_all.py --quick
```

## Full reproduction

### Table 1 and Figures 2, S4 (Buchwald-Hartwig)

```bash
python -c "
from experiments.buchwald_hartwig.run_experiment import run_buchwald_hartwig_experiment
run_buchwald_hartwig_experiment(n_train=50, n_test=100, gfn_episodes=3000, pca_dim=5, pce_degree=3)
"
```

**Expected outputs:**
- Total policy variance: D_catalyst ~ 71, D_additive ~ 179 (2.5x ratio)
- Calibration coverage @95%: catalyst=1.00, base=1.00, aryl_halide=0.97, additive=0.77
- PCE MAE: 0.153 (with L=50 < L*=57; improves with L=60)
- Sobol indices: catalyst S_PC1 ~ 0.02 (robust), additive S_PC1 >> catalyst (fragile)

### Table 1 and Figure 3 (Sachs causal discovery)

```bash
python -c "
from experiments.sachs_causal.run_experiment import run_sachs_experiment
run_sachs_experiment(n_train=30, n_test=50, gfn_episodes=3000, pca_dim=2, pce_degree=5)
"
```

**Expected outputs:**
- PCA explained variance: 38.4%
- MAPK-pathway edges (Raf->Mek, Mek->Erk, PKA->P38): S_PC2 > 0.95
- PKA/PKC hub edges: S_PC1 > 0.93
- PCE MAE: 0.016
- PCE fit time: ~0.026s; GP fit time: ~36.6s (ratio ~1400x)

### Figure 4 (Grid-world validation)

```bash
python -c "
from experiments.gridworld.run_experiment import run as run_gridworld
run_gridworld()
"
```

### Figure 5 (Symbolic regression)

```bash
python -c "
from experiments.symreg.run_experiment import run as run_symreg
run_symreg()
"
```

### Figure 6 (LLM GFlowNet)

```bash
python -c "
from experiments.llm_gfn.run_experiment import run as run_llm
run_llm()
"
```

### Supplementary Figure S8 (Controlled LLM)

```bash
python -c "
from experiments.controlled_llm.run_experiment import run_controlled_llm_experiment
run_controlled_llm_experiment(n_train=15, n_test=25, gfn_episodes=1500)
"
```

### Figure 7 (Fragment-based molecular design)

```bash
python -c "
from experiments.molecular_design.run_experiment import run as run_moldesign
run_moldesign()
"
```

### Supplementary Figure S7 (Ablation / d-scalability)

```bash
python experiments/gridworld/d_scalability.py
```

### Table S3 (Cost comparison: PCE analytical vs Saltelli MC vs bootstrap)

```bash
python experiments/sobol_validation/cost_comparison.py
```

## Validation experiments

### PCA independence verification

```bash
python experiments/validation/pca_independence.py
```

Computes Pearson and Spearman correlations between PCA components for BH and Sachs tasks. Expected: all |r| < 0.05 (confirming near-independence required for valid Sobol decomposition).

### Negative control experiments

```bash
python experiments/validation/negative_controls.py
```

Demonstrates degradation when assumptions are violated: too few ensemble members (L=10), PCE overfitting (degree >> optimal), and VAE instead of PCA.

### Total-order Sobol interaction diagnostic

```bash
python experiments/validation/sobol_interactions.py
```

Reports sum of total-order Sobol indices across all experiments. Values near 1.0 indicate near-additive structure; values >> 1 indicate strong interaction effects.

## Lean 4 formal verification

```bash
cd lean
lake build
```

Requires Lean 4 (v4.x) and Mathlib. The `lakefile.lean` handles dependency management. Four of five theorems compile without `sorry`; T2 and T5 have partial proofs documented in the source.

## Random seeds

GFlowNet training uses PyTorch seeding. Each ensemble member l uses seed `base_seed + l` where base_seed is set per experiment. PCE fitting is deterministic given the same ensemble outputs. PCA is deterministic (SVD-based). Minor numerical variation across hardware/BLAS implementations is expected but does not affect qualitative conclusions.

## Output directory

All experiments write results to `results/`. Figures are saved as PDF in experiment-specific subdirectories. The `figures/` directory contains the versions used in the manuscript.
