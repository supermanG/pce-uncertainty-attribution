# pce-uncertainty-attribution

**Interpretable attribution of decision-relevant uncertainty in AI**

Nartallo-Kaluarachchi, R., Ubaru, S., Zimon, M.J., Huh, D.,
Manson-Sawko, R., Horesh, L., Bengio, Y. (2026)
*Nature Machine Intelligence* (accepted in principle, September 2026)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22904515.svg)](https://doi.org/10.5281/zenodo.22904515)

Version v1.0.0 is the code used for the paper, archived at Zenodo under DOI
[10.5281/zenodo.22904515](https://doi.org/10.5281/zenodo.22904515)
(concept DOI for all versions: [10.5281/zenodo.22904514](https://doi.org/10.5281/zenodo.22904514)).

---

## Overview

This repository provides the implementation and the reproduction material for the paper.
The framework parameterises a learned objective model's epistemic uncertainty in a
low-dimensional basis (PCA modes of an ensemble, or the Karhunen-Loeve modes of a Gaussian
process posterior), trains a small ensemble of decision policies under that uncertainty, fits
a sparse or arbitrary polynomial chaos expansion (PCE) per decision, and reads off
closed-form Sobol indices, a scale-invariant decision fragility, and an applicability
diagnostic.

**Key idea:** the decision-relevant part of a learned model's uncertainty is a distinct,
separately computable object from its largest part. The value-of-information spectrum (the
Sobol decomposition of the decision *value*) can be aligned with the variance spectrum, as in
chemistry, or opposed to it, as under RLHF alignment, where the highest-variance
reward-model direction is the shift-invariant one the softmax ignores exactly (Theorem 6).

**Scope, and what is a negative control.** The positive demonstrations are
Buchwald-Hartwig cross-coupling (real Doyle-Dreher yields), Bayesian optimization and
optimal experimental design on the same real reaction space, and a best-of-n RLHF study on
real human preferences (hh-rlhf). The **Sachs causal-discovery and fragment-based
molecular-design settings are negative controls**: the training-noise diagnostic shows their
policy ensembles are dominated by training-seed noise rather than reward uncertainty, and the
surrogate cannot predict the policy from the reward parameterisation, so no reward-uncertainty
attribution is reported for them. Their code is retained here because the abstention is a
reported result, not because a quantitative claim is made from them.

---

## Repository structure

```
core/
  pce_surrogate.py            TrajectoryPCESurrogate: per-step PCE fitting,
                              GCV ridge selection, bootstrap CIs, marginal
                              and joint calibration coverage, analytical
                              Sobol indices, ensemble-size bound.
  sparse_pce.py               Sparse PCE (LARS / OMP) and arbitrary PCE (aPC)
                              on empirical marginals; design-matrix coherence.
  mlp_surrogate.py            MLP baseline surrogate.
  gp_surrogate.py             GP baseline surrogate.
  distributional_analysis.py  KS-test battery, bimodality detection,
                              surrogate-vs-empirical comparison plots.
  embeddings.py               PCA / kernel PCA / beta-VAE / normalizing-flow
                              latent constructions for the embedding ablation.
  fragility_metrics.py        Scale-invariant simplex fragility and corrected
                              interaction reporting (shared by the studies).

decision_studies/             Decision-making studies under learned-model uncertainty.
                              These produce the six main-text figures.
  common/
    analyze_ensemble.py       Task-agnostic post-analysis: sparse/aPC PCE,
                              PCA-dimension sweep, validation metrics,
                              bootstrap Sobol CIs, Source Data CSVs.
  buchwald_hartwig/           Closed-loop fragility validation on measured
                              yields, cardinality-confound control, ensemble-
                              size convergence, Shapley effects, robustness.
  bayesian_optimization/      BO / optimal experimental design on the real
                              reaction space; exactly-valid KL-mode indices.
  rlhf/                       RLHF reward-model uncertainty: best-of-n study
                              on real human preferences (hh-rlhf), and the
                              check that the top-variance mode is the
                              uniform-offset direction (rlhf_shift_mode.py).
  value_of_information/       Value-of-information spectrum (decision-relevant
                              vs total uncertainty) and VoI-guided acquisition.
  controls/                   Training-noise controls (applicability diagnostic)
                              for the molecular-design and Sachs negative controls.
  embedding_check/            beta-VAE grid-world embedding-agnostic check.
  reanalysis_sobol_scale.py   Scale-dependence re-analysis on cached outputs.

experiments/
  buchwald_hartwig/           Pd-catalysed C-N coupling (Doyle-Dreher dataset);
                              4-step GFlowNet, 150 members (50 train / 100 test),
                              d=5 PCA, PCE degree 3.
  gridworld/                  Discrete and continuous grid-world validation
                              (ground-truth reachable).
  symreg/                     Symbolic regression GRU with Wiener-process
                              reward noise; KL-mode parameterisation.
  llm_gfn/                    GRU-based reasoning GFlowNet with uncertain
                              process reward model (PRM).
  controlled_llm/             Strategy-selection reasoning GFlowNet
                              (10 strategies, 5 steps, uncertain PRM).
  sachs_causal/               NEGATIVE CONTROL. Bayesian causal discovery on
                              real Sachs flow cytometry; training-seed dominated,
                              no attribution reported.
  molecular_design/           NEGATIVE CONTROL. Fragment-based drug-likeness
                              GFlowNet; training-seed dominated, no attribution
                              reported.
  baselines/                  PCE vs MLP vs GP head-to-head comparison.
  sobol_validation/           Sobol convergence, sample-complexity validation,
                              and the analytical-vs-Monte-Carlo cost comparison.
  validation/                 Assumption-violation controls, scaling benchmarks,
                              PCA independence checks, embedding ablation,
                              BH additive validation, L=60 check.

lean/
  PCESurrogate.lean           Lean 4 formal verification: eight lemmas covering
                              the seven numbered results T1-T7, all `sorry`-free.
  lakefile.lean               Lake project descriptor.
  lake-manifest.json          Dependency manifest.
  lean-toolchain              Pinned Lean toolchain version.

figures/
  make_figures.py             Builds the six main-text figures from the
                              committed result JSONs in results/.
  generate_figures.py         Builds the supplementary figure set (S1-S7) from
                              the per-experiment results/ directories, plus
                              earlier-version panels that the current manuscript
                              does not use (flagged in the script and listed in
                              REPRODUCE.md).
  make_graphical_abstract.py  Builds the graphical abstract.
  *.pdf, *.png                fig_framework, fig_bh_revised, fig_bh_closedloop,
                              fig_bo_oed, fig_rlhf, fig_voi are main-text
                              Figs 1-6. Supplementary Figs S1-S7 are fig_s1 to
                              fig_s6 plus fig_s8_controlled_llm.pdf, whose file
                              name is kept from an earlier numbering and which is
                              Supplementary Fig S7 in the current manuscript. The
                              remaining PDFs, fig_s7_ablation.pdf included, are
                              superseded drafts from earlier versions of the study
                              and are not cited by the manuscript.

results/                      Committed result JSONs (cluster_results/, outputs/)
                              backing the main-text figures.

data/
  download_sachs.py           Downloads real Sachs flow cytometry data.
  sachs_real.csv              Cached copy of that dataset.

tests/
  test_sparse_pce.py          Sparse / arbitrary-PCE recovery and Sobol checks.

notebooks/
  demo_uq_pipeline.ipynb      End-to-end walkthrough on a toy example.

lsf/                          IBM CCC cluster submission scripts (LSF/bsub)
                              for the experiment ensembles.

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
git clone https://github.com/supermanG/pce-uncertainty-attribution
cd pce-uncertainty-attribution
pip install -r requirements.txt

# Rebuild the six main-text figures from the committed result JSONs (seconds, CPU)
python figures/make_figures.py

# Smoke test of the GFlowNet training pipeline (reduced ensembles, fast)
python run_all.py --quick

# Individual training pipelines accepted by run_all.py
python run_all.py --experiment bh     # Buchwald-Hartwig (~30 min, CPU)
python run_all.py --experiment sachs  # Sachs (negative control, ~20 min, CPU)
python run_all.py --experiment llm    # Controlled LLM GFlowNet (~15 min, CPU)
```

The molecular-design negative control is not exposed through `run_all.py`; run it directly:

```bash
python experiments/molecular_design/run_experiment.py --mode sequential
```

See [`REPRODUCE.md`](REPRODUCE.md) for the full per-figure reproduction recipe.

---

## Code Ocean capsule

`run` at the repository root is the entry point of the Code Ocean compute capsule (the
repository imported into `/code`) and also works locally:

```bash
bash run
```

It runs the unit tests of the PCE core and rebuilds the six main-text figures from the
committed result files, writing them with a SHA-256 list to `/results` on Code Ocean or
to `results/capsule/` elsewhere. `codeocean/environment/Dockerfile` is the capsule
environment (Python 3.14, numpy, scipy, matplotlib, pytest; CPU only) and
`codeocean/metadata/metadata.yml` its record; see `codeocean/README.md`. The heavy
experiments are not part of the capsule; their recipes are in [REPRODUCE.md](REPRODUCE.md).

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
transformers >= 4.30     # DistilBERT / GPT-2 stacks for the RLHF study (optional)
datasets                 # hh-rlhf preference data for the RLHF study (optional)
```

Install:
```bash
pip install -r requirements.txt
```

Only `numpy` and `matplotlib` are needed to rebuild the main-text figures from the committed
result JSONs.

---

## Reproducing paper results

### Real datasets

The Doyle-Dreher Buchwald-Hartwig dataset is committed at
`experiments/buchwald_hartwig/data/data_table.csv`; upstream source
https://github.com/doylelab/rxnpredict.

The Sachs flow cytometry dataset (853 observations, 11 proteins) is available at
https://www.bnlearn.com/bnrepository/ and cached at `data/sachs_real.csv`. To re-download:
```bash
python data/download_sachs.py
```

The RLHF study uses `Anthropic/hh-rlhf`
(https://huggingface.co/datasets/Anthropic/hh-rlhf), downloaded at run time.

### Figures

```bash
python figures/make_figures.py     # main-text Figs 1-6, from committed JSONs
python figures/generate_figures.py # supplementary figure set, needs results/
```

Vector PDFs are written to `figures/`.

### Cluster (IBM CCC / LSF)

LSF submission scripts for the training ensembles are in `lsf/`. The BH experiment submits
one job per ensemble member and a dependent analysis job:

```bash
bash lsf/submit_bh.sh           # per-member trainers, then the analysis job
```

---

## Formal verification

> **Prerequisite, please read before building.** `lean/lakefile.lean` requires Mathlib4 from a
> **local path**, `../../mathlib4` relative to `lean/`, that is a directory named `mathlib4`
> sitting **beside this repository**. A bare clone will therefore not build until that
> checkout exists. Full setup, with expected timings, is in
> [`REPRODUCE.md`](REPRODUCE.md#lean-4-formal-verification); the short version is:
>
> ```bash
> # 1. Lean toolchain manager (once)
> curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
>
> # 2. Mathlib4 beside this repository, at the toolchain pinned in lean/lean-toolchain
> cd ..                                       # parent of pce-uncertainty-attribution
> git clone https://github.com/leanprover-community/mathlib4.git
> cd mathlib4
> git checkout v4.30.0-rc1                    # must match lean/lean-toolchain
> cat lean-toolchain                          # expect: leanprover/lean4:v4.30.0-rc1
> lake exe cache get                          # prebuilt Mathlib artifacts
>
> # 3. Build the development, then audit its axioms
> cd ../pce-uncertainty-attribution/lean
> lake build
> lake env lean AxiomCheck.lean
> ```
>
> If the tag above is absent or its `lean-toolchain` does not match, check out instead any
> Mathlib commit whose `lean-toolchain` is exactly `leanprover/lean4:v4.30.0-rc1`. Step 2
> dominates the wall-clock time: `lake exe cache get` downloads several GB of prebuilt
> artifacts, and without that cache Lake compiles Mathlib from source, which takes hours.
> Step 3 is a single file and is quick once Mathlib is in place.

`lean/PCESurrogate.lean` machine-checks eight lemmas covering the seven numbered results
T1-T7 of the manuscript. Every proof is `sorry`-free, `lake build` completes with no errors,
and `#print axioms` reports that each theorem depends only on the three standard axioms
`propext`, `Classical.choice`, `Quot.sound`. `lean/AxiomCheck.lean` reruns that axiom audit
for all eight lemmas in one command (`lake env lean AxiomCheck.lean`), so the claim is
checkable without reading the proofs. Results proved outright are distinguished from
those proved conditional on a stated hypothesis, exactly as in Supplementary Table S1:

| ID  | Lean lemma                   | Statement                                       | Status      |
|-----|------------------------------|-------------------------------------------------|-------------|
| T1  | `pce_error_tendsto_zero`     | PCE truncation error converges to zero           | conditional (Sobolev rate assumed) |
| T2  | `sobol_convergence_from_L2`  | Sobol index convergence from l^2 closeness       | conditional (shared basis, positive variance, first-order) |
| T3a | `softmax_positive`           | Softmax strict positivity                        | outright    |
| T3b | `softmax_sums_to_one`        | Softmax normalisation identity                   | outright    |
| T4  | `uncertainty_propagation`    | Lipschitz uncertainty propagation                | conditional (Lipschitz constant given) |
| T5  | `softmax_lipschitz`          | Softmax l-infinity to l-1 bound (constant 2)     | outright    |
| T6  | `softmax_shift_invariant`    | Softmax invariance to a uniform logit shift      | outright    |
| T7  | `value_diff_le_l1`           | Decision value-variability bounded by fragility  | outright    |

An auxiliary lemma, `simplex_l1_le_two` (the l-1 diameter of the probability simplex is 2),
supports the T5 proof and is audited alongside the eight.

---

## Citation

```bibtex
@article{nartallo2026uqgfn,
  title   = {Interpretable attribution of decision-relevant uncertainty in AI},
  author  = {Nartallo-Kaluarachchi, Ram\'on and Ubaru, Shashanka and
             Zimon, Ma{\l}gorzata J. and Huh, Dongsung and
             Manson-Sawko, Robert and Horesh, Lior and Bengio, Yoshua},
  journal = {Nature Machine Intelligence},
  year    = {2026},
  note    = {Accepted in principle, September 2026}
}
```

For the code itself (the version used for the paper):

```bibtex
@software{nartallo2026pceuq_code,
  title     = {pce-uncertainty-attribution: code for ``Interpretable attribution of
               decision-relevant uncertainty in AI''},
  author    = {Nartallo-Kaluarachchi, Ram\'on and Ubaru, Shashanka and
               Zimon, Ma{\l}gorzata J. and Huh, Dongsung and
               Manson-Sawko, Robert and Horesh, Lior and Bengio, Yoshua},
  year      = {2026},
  version   = {v1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22904515},
  url       = {https://github.com/supermanG/pce-uncertainty-attribution}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
