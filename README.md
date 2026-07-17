# uq-gflownet

**Interpretable epistemic uncertainty attribution for decision-making under learned-model uncertainty**

Nartallo-Kaluarachchi, R., Ubaru, S., Zimon, M.J., Huh, D.,
Manson-Sawko, R., Horesh, L., Bengio, Y. (2026)
*Nature Machine Intelligence* (under review)

---

## Overview

This repository provides the full implementation of a surrogate modelling framework that propagates reward uncertainty through GFlowNet policies and decomposes it into interpretable components via analytical Sobol sensitivity indices.

**Key idea:** train a small ensemble of GFlowNets under varied reward conditions, fit a polynomial chaos expansion (PCE) over the low-dimensional reward-parameter space, and read off closed-form Sobol indices — revealing *which components of reward uncertainty drive which generative decisions* — at a fraction of the cost of exhaustive retraining.

---

## Repository structure

```
core/
  pce_surrogate.py            TrajectoryPCESurrogate: per-step PCE fitting,
                              GCV ridge selection, bootstrap CIs, marginal
                              and joint calibration coverage, analytical
                              Sobol indices, ensemble-size bound (Theorem A).
  mlp_surrogate.py            MLP baseline surrogate.
  gp_surrogate.py             GP baseline surrogate.
  distributional_analysis.py  KS-test battery, bimodality detection,
                              surrogate-vs-empirical comparison plots.
  embeddings.py               PCA / kernel PCA / beta-VAE / normalizing-flow
                              latent constructions for the embedding ablation.
  fragility_metrics.py        Scale-invariant simplex fragility and corrected
                              interaction reporting (shared by the studies).

experiments/
  buchwald_hartwig/           Pd-catalysed C-N coupling (Doyle-Dreher dataset);
                              4-step GFlowNet, 50/100 ensemble, d=5 PCA.
  sachs_causal/               Bayesian causal discovery on real Sachs flow
                              cytometry (853 obs, 11 proteins); BGe reward.
  molecular_design/           Fragment-based drug-likeness GFlowNet (20-frag
                              vocab, 5 positions, MLP proxy reward).
  gridworld/                  Discrete and continuous grid-world validation
                              (ground-truth reachable).
  symreg/                     Symbolic regression GRU with Wiener-process
                              reward noise; KL-mode parameterisation.
  llm_gfn/                    GRU-based reasoning GFlowNet with uncertain
                              process reward model (PRM).
  controlled_llm/             Strategy-selection reasoning GFlowNet
                              (10 strategies, 5 steps, uncertain PRM).
  baselines/                  PCE vs MLP vs GP head-to-head comparison.
  sobol_validation/           Sobol convergence and Theorem A sample-
                              complexity validation.
  validation/                 Negative controls, scaling benchmarks, PCA
                              independence checks, BH additive validation.

lean/
  PCESurrogate.lean           Lean 4 formal verification: all five theorems
                              (T1-T5) machine-checked, no `sorry`.
  lakefile.lean               Lake project descriptor.
  lake-manifest.json          Pinned Mathlib4 dependency manifest.
  lean-toolchain              Pinned Lean toolchain version.

decision_studies/               Decision-making studies under learned-model uncertainty.
  common/
    analyze_ensemble.py       Task-agnostic post-analysis: sparse/aPC PCE,
                              PCA-dimension sweep, validation metrics,
                              bootstrap Sobol CIs, Source Data CSVs.
  bayesian_optimization/      BO / optimal experimental design on the real
                              reaction space; exactly-valid KL-mode indices.
  rlhf/                       RLHF reward-model uncertainty: best-of-n study
                              on real human preferences (hh-rlhf).
  value_of_information/       Value-of-information spectrum (decision-relevant
                              vs total uncertainty) and VoI-guided acquisition.
  buchwald_hartwig/           Closed-loop fragility validation on measured
                              yields, cardinality-confound control, ensemble-
                              size convergence, Shapley effects.
  controls/                   Training-noise controls (applicability diagnostic).
  embedding_check/            beta-VAE grid-world embedding-agnostic check.
  reanalysis_sobol_scale.py   Scale-dependence re-analysis on cached outputs.

figures/
  generate_figures.py         Regenerates all publication figures from results.
  make_figures.py             Builds the manuscript composite figures.
  make_graphical_abstract.py  Builds the graphical abstract.
  *.pdf, *.png                Source files for every main and supplementary figure.

results/                      Committed result JSONs (cluster_results/, outputs/)
                              backing every figure and table.

data/
  download_sachs.py           Downloads real Sachs flow cytometry data.

tests/
  test_sparse_pce.py          Sparse / arbitrary-PCE recovery and Sobol checks.

notebooks/
  demo_uq_pipeline.ipynb      End-to-end walkthrough on a toy example.

lsf/                          IBM CCC cluster submission scripts (LSF/bsub)
                              for every experiment plus the BH array job.

scripts/
  run_local.py                Local-machine sequential runner.
  run_gpu_overnight.sh        Single-GPU end-to-end run.

run_all.py                    Master experiment runner (CLI).
requirements.txt              Pip dependency pin.
REPRODUCE.md                  Step-by-step reproducibility manifest.
```

---

## Quick start

```bash
git clone https://github.com/supermanG/uq-gflownet-release
cd uq-gflownet-release
pip install -r requirements.txt

# Smoke test (reduced ensemble, fast)
python run_all.py --quick

# Individual experiments
python run_all.py --experiment bh        # Buchwald-Hartwig (~30 min, CPU)
python run_all.py --experiment sachs     # Sachs causal discovery (~20 min, CPU)
python run_all.py --experiment moldesign # Fragment molecular design (~15 min, CPU)
python run_all.py --experiment llm       # LLM GFlowNet (~15 min, CPU)

# Reproduce all figures from cached results
python figures/generate_figures.py
```

See [`REPRODUCE.md`](REPRODUCE.md) for the full per-figure / per-table reproduction recipe.

---

## Requirements

```
Python >= 3.10
torch >= 2.0
numpy >= 1.24
scipy >= 1.10
scikit-learn >= 1.2
matplotlib >= 3.7
pandas >= 1.5            # Doyle-Dreher CSV loader
networkx >= 3.0          # Sachs DAG figure
transformers >= 4.30     # GRU/GPT-2 stacks for the LLM GFlowNet (optional)
```

Install all:
```bash
pip install -r requirements.txt
```

For the Lean 4 proofs: install [Lean 4](https://leanprover.github.io/) and run `lake build` inside `lean/`. The Mathlib4 dependency is pinned in `lake-manifest.json` (currently v4.30).

---

## Reproducing paper results

### Real datasets

The Doyle-Dreher Buchwald-Hartwig dataset is available at
https://github.com/doylelab/rxnpredict
(place `data_table.csv` at `experiments/buchwald_hartwig/data/`).

The Sachs flow cytometry dataset (853 observations, 11 proteins) is available at
https://www.bnlearn.com/bnrepository/

To download automatically:
```bash
python data/download_sachs.py
```

### Cluster (IBM CCC / LSF)

LSF submission scripts for every experiment are in `lsf/`. The BH experiment uses an array job for the 50-member ensemble:

```bash
bash lsf/submit_bh.sh           # Phase 1: array of single-member trainers
bash lsf/bh_analyze.sh          # Phase 2: collect + PCE fit + Sobol
```

### Figures

After experiments complete, regenerate all publication figures:
```bash
python figures/generate_figures.py
```
Vector PDFs are written to `figures/`.

---

## Formal verification

All five theorems are machine-checked in Lean 4 with no `sorry`; each depends only on the standard axioms (`propext`, `Classical.choice`, `Quot.sound`):

| ID  | Statement                                  | Status   |
|-----|--------------------------------------------|----------|
| T1  | PCE truncation error converges to zero     | Verified |
| T2  | Sobol index convergence from l^2 closeness | Verified |
| T3a | Softmax strict positivity                  | Verified |
| T3b | Softmax normalisation identity             | Verified |
| T4  | Lipschitz uncertainty propagation          | Verified |
| T5  | Softmax Lipschitz bound (constant 2)       | Verified |

Source: `lean/PCESurrogate.lean`. Build with `cd lean && lake build`; `#print axioms` confirms the axiom dependencies.

---

## Citation

```bibtex
@article{nartallo2026uqgfn,
  title   = {Interpretable epistemic uncertainty attribution for
             decision-making under learned-model uncertainty},
  author  = {Nartallo-Kaluarachchi, Ram\'on and Ubaru, Shashanka and
             Zimon, Ma{\l}gorzata J. and Huh, Dongsung and
             Manson-Sawko, Robert and Horesh, Lior and Bengio, Yoshua},
  journal = {Nature Machine Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
