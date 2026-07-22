# Reproducibility Manifest

Instructions to reproduce the figures and analyses of the manuscript.

The main text has six figures:

| Fig | File                     | Content                                          |
|-----|--------------------------|--------------------------------------------------|
| 1   | `fig_framework.pdf`      | Attribution-framework schematic                  |
| 2   | `fig_bh_revised.pdf`     | Buchwald-Hartwig attribution                     |
| 3   | `fig_bh_closedloop.pdf`  | Buchwald-Hartwig closed-loop validation          |
| 4   | `fig_bo_oed.pdf`         | Bayesian optimization / optimal experimental design |
| 5   | `fig_rlhf.pdf`           | RLHF reward-model uncertainty                    |
| 6   | `fig_voi.pdf`            | Value of information vs variance                 |

All six are built by `figures/make_figures.py`. Eight supplementary figures (S1-S8) are built
by `figures/generate_figures.py`. Both are covered below.

## Environment

```bash
conda create -n uqgfn python=3.10 -y && conda activate uqgfn
pip install -r requirements.txt
```

Hardware used for the reported numbers: single NVIDIA A100 GPU (GFlowNet training, reward-model
embedding and best-of-n generation), single CPU core (PCE fitting, Sobol indices, figure
building). Only `numpy` and `matplotlib` are required to rebuild the main-text figures from the
committed result JSONs.

Run every command below from the repository root.

## Fast path: rebuild every main-text figure from the committed results

```bash
python figures/make_figures.py
```

This reads only the JSONs committed under `results/cluster_results/` and writes
`fig_framework.pdf`, `fig_bh_revised.pdf`, `fig_bh_closedloop.pdf`, `fig_bo_oed.pdf`,
`fig_rlhf.pdf` and `fig_voi.pdf` into `figures/`. It needs no GPU, no dataset download and no
retraining, and takes a few seconds.

## Per-figure recipes

Each recipe lists the driver that produces the underlying result JSON, the JSON it writes (the
copy committed here), and the `figures/make_figures.py` entry point that consumes it.

### Figure 1: framework schematic (`fig_framework.pdf`)

Schematic; no data input.

* Entry point: `figures/make_figures.py` -> `fig_framework()`

### Figure 2: Buchwald-Hartwig attribution (`fig_bh_revised.pdf`)

Panels: **a** per-step decision fragility, **b** sparse versus dense surrogate accuracy,
**c** additive-step first-order Sobol indices with bootstrap 90% intervals against Shapley
effects, **d** accuracy versus retained PCA dimension.

1. Train the GFlowNet ensemble on the real Doyle-Dreher yields (150 members; the dataset is
   committed at `experiments/buchwald_hartwig/data/data_table.csv`):

   ```bash
   python experiments/buchwald_hartwig/train_single_gfn.py --setup \
       --csv_path experiments/buchwald_hartwig/data/data_table.csv \
       --output_dir results/buchwald_hartwig
   # then, per member i = 0 ... 149 (see lsf/submit_bh.sh for the cluster version)
   python experiments/buchwald_hartwig/train_single_gfn.py --member_id $i \
       --output_dir results/buchwald_hartwig \
       --csv_path experiments/buchwald_hartwig/data/data_table.csv \
       --gfn_episodes 3000 --proxy_epochs 200 --temp 4.0
   ```

2. Post-analysis (sparse and aPC PCE, PCA-dimension sweep, fragility, bootstrap Sobol CIs,
   Source Data CSVs):

   ```bash
   python decision_studies/common/analyze_ensemble.py \
       --members_dir results/buchwald_hartwig \
       --n_train 50 --n_test 100 --degree 3 --pca_dims 5,7,10 \
       --out results/cluster_results/bh_final
   ```

   Writes `results/cluster_results/bh_final/ensemble_analysis.json` (committed) and
   `source_data_fragility.csv` (committed).

3. Panel **c** additionally uses `results/cluster_results/bh_final/bh_shapley.json`
   (committed), the given-data Shapley effects of the fitted surrogate. The estimator lives in
   `decision_studies/buchwald_hartwig/shapley_effects.py`; running that file directly executes
   its analytic linear-Gaussian validation (closed-form Shapley effects at several sample
   sizes), which is what establishes that the estimator is trustworthy at L = 50.

* Entry point: `figures/make_figures.py` -> `fig_bh()`

### Figure 3: Buchwald-Hartwig closed-loop validation (`fig_bh_closedloop.pdf`)

Reward-model ensemble only (no GFlowNet), on measured yields, so the reward-driven signal is
isolated from training noise.

```bash
python decision_studies/buchwald_hartwig/bh_closed_loop.py \
    --out results/cluster_results/bh_closed_loop
```

Writes `results/cluster_results/bh_closed_loop/bh_closed_loop.json` (committed): per-component
fragility and the learning curve of fragility against the fraction of reactions observed.

* Entry point: `figures/make_figures.py` -> `fig_bh_closedloop()`

### Figure 4: Bayesian optimization / OED (`fig_bo_oed.pdf`)

Two inputs, the real reaction space and the synthetic Gaussian-process reference:

```bash
# real Doyle-Dreher reaction space, multi-seed
python decision_studies/bayesian_optimization/bo_oed_real_v2.py \
    --out results/cluster_results/bo_oed_real
# synthetic GP objective (exactly independent KL modes)
python decision_studies/bayesian_optimization/bo_oed_demo.py \
    --out results/cluster_results/bo_oed
```

Write `results/cluster_results/bo_oed_real/bo_oed_real_v2_results.json` and
`results/cluster_results/bo_oed/bo_oed_results.json` (both committed).

* Entry point: `figures/make_figures.py` -> `fig_bo()`

### Figure 5: RLHF reward-model uncertainty (`fig_rlhf.pdf`)

Real `Anthropic/hh-rlhf` preferences, DistilBERT features with an ensemble of Bradley-Terry
reward heads, GPT-2 best-of-n candidates, and a gold reward model trained on a disjoint half of
the preference data. Requires `transformers`, `datasets`, network access, and a CUDA GPU (the
encoder and generator call `.cuda()`).

```bash
python decision_studies/rlhf/rlhf_real_v2.py \
    --out results/cluster_results/rlhf_real
```

Writes `results/cluster_results/rlhf_real/rlhf_real_v2_results.json` (committed): per-mode
Sobol indices split by reward-model disagreement, the KL-penalty sweep, and the proxy versus
gold reward curves against the best-of-n pool size.

* Entry point: `figures/make_figures.py` -> `fig_rlhf()`

### Figure 6: value of information (`fig_voi.pdf`)

Panels **a-c** and **f** come from the spectra, panels **d,e** from the acquisition simulation.

```bash
python decision_studies/value_of_information/value_of_information.py \
    --domains bh,bo,rlhf --out results/cluster_results/voi
python decision_studies/value_of_information/voi_acquisition.py \
    --domains bh,rlhf --out results/cluster_results/voi
```

Write `results/cluster_results/voi/voi_spectra.json` and
`results/cluster_results/voi/voi_acquisition.json` (both committed). The `bh` domain reuses the
reward-model ensemble of `bh_closed_loop.py`, the `bo` domain reuses `bo_oed_real.py`, and the
`rlhf` domain reuses the hh-rlhf stack of `rlhf_real.py`, so the RLHF prerequisites above apply.

* Entry point: `figures/make_figures.py` -> `fig_voi()`

## Committed result files and their drivers

| Result file (under `results/`)                          | Driver                                                        |
|---------------------------------------------------------|---------------------------------------------------------------|
| `cluster_results/bh_final/ensemble_analysis.json`        | `decision_studies/common/analyze_ensemble.py`                  |
| `cluster_results/bh_final/source_data_fragility.csv`     | `decision_studies/common/analyze_ensemble.py`                  |
| `cluster_results/bh_final/bh_robustness.json`            | `decision_studies/buchwald_hartwig/bh_robustness.py`           |
| `cluster_results/bh_final/bh_shapley.json`               | estimator in `decision_studies/buchwald_hartwig/shapley_effects.py` |
| `cluster_results/bh_closed_loop/bh_closed_loop.json`     | `decision_studies/buchwald_hartwig/bh_closed_loop.py`          |
| `cluster_results/bh_closed_loop/bh_confound_control.json`| `decision_studies/buchwald_hartwig/bh_confound_control.py`     |
| `cluster_results/bh_closed_loop/bh_ensemble_convergence.json` | `decision_studies/buchwald_hartwig/bh_ensemble_convergence.py` |
| `cluster_results/bo_oed_real/bo_oed_real_v2_results.json`| `decision_studies/bayesian_optimization/bo_oed_real_v2.py`     |
| `cluster_results/bo_oed_real/bo_oed_real_results.json`   | `decision_studies/bayesian_optimization/bo_oed_real.py`        |
| `cluster_results/bo_oed/bo_oed_results.json`             | `decision_studies/bayesian_optimization/bo_oed_demo.py`        |
| `cluster_results/bo_oed_seq/bo_oed_sequential.json`      | `decision_studies/bayesian_optimization/bo_oed_sequential.py`  |
| `cluster_results/rlhf_real/rlhf_real_v2_results.json`    | `decision_studies/rlhf/rlhf_real_v2.py`                        |
| `cluster_results/rlhf_real/rlhf_real_results.json`       | `decision_studies/rlhf/rlhf_real.py`                           |
| `cluster_results/rlhf/rlhf_results.json`                 | `decision_studies/rlhf/rlhf_demo.py`                           |
| `cluster_results/voi/voi_spectra.json`                   | `decision_studies/value_of_information/value_of_information.py`|
| `cluster_results/voi/voi_acquisition.json`               | `decision_studies/value_of_information/voi_acquisition.py`     |
| `outputs/sobol_scale_summary.json`                       | `decision_studies/reanalysis_sobol_scale.py`                   |
| `beta_vae_gridworld_results.json`                        | `decision_studies/embedding_check/beta_vae_gridworld.py`       |

`results/` is listed in `.gitignore` (runs can be large); the files above were added
deliberately so the figures rebuild without retraining.

## Supplementary analyses

### Attribution stability, dependence and calibration (Buchwald-Hartwig)

```bash
python decision_studies/buchwald_hartwig/bh_robustness.py \
    --members results/buchwald_hartwig --out results/cluster_results/bh_final
```

Ranking stability across retained dimension d = 5, 7, 10; maximum distance correlation between
retained modes; coverage, PIT and CRPS.

### Cardinality-confound control and ensemble-size convergence

```bash
python decision_studies/buchwald_hartwig/bh_confound_control.py \
    --out results/cluster_results/bh_closed_loop
python decision_studies/buchwald_hartwig/bh_ensemble_convergence.py \
    --out results/cluster_results/bh_closed_loop
```

The first tests whether the per-component fragility ranking is a spurious action-space
cardinality artefact; the second varies the reward-model ensemble size L.

### Sequential BO

```bash
python decision_studies/bayesian_optimization/bo_oed_sequential.py \
    --out results/cluster_results/bo_oed_seq
```

### Scale-dependence re-analysis

```bash
python decision_studies/reanalysis_sobol_scale.py
```

Re-examines total policy variance and first- versus total-order Sobol sums with no retraining,
reading the cached `results/molecular_design/results.json` and `results/sachs/results.json`, so
those two runs must exist first. Writes `results/outputs/sobol_scale_summary.json` (committed).

### Embedding-agnostic check

```bash
python decision_studies/embedding_check/beta_vae_gridworld.py
```

beta-VAE against PCA on the grid-world ensemble, through the same PCE surrogate. The script
header lists the `curl` commands that fetch the 50 reward grids and trained policies it
consumes; point `GRIDWORLD_ENSEMBLE_DIR` at the download directory. Writes
`results/beta_vae_gridworld_results.json` (committed).

### Embedding ablation (PCA vs kernel PCA vs beta-VAE vs PCA + normalizing flow)

```bash
python experiments/validation/pca_vs_vae_ablation.py
```

### Analytical Sobol versus Monte Carlo cost

```bash
python experiments/sobol_validation/cost_comparison.py
```

### PCA independence check

```bash
python experiments/validation/pca_independence.py
```

Pearson and Spearman correlations between retained PCA components (the linear part of the
independence requirement; the nonlinear part is the distance correlation reported by
`bh_robustness.py`).

### Assumption-violation controls

```bash
python experiments/validation/negative_controls.py
```

Degradation under too few ensemble members, PCE overfitting, and a broken-independence
embedding.

### Interaction diagnostic

```bash
python experiments/validation/sobol_interactions.py
```

Reports first-order and total-order Sobol sums; the interaction fraction is the corrected
statistic used in the manuscript.

### Sparse / arbitrary PCE unit checks

```bash
python tests/test_sparse_pce.py
```

aPC orthonormality on non-Gaussian inputs, sparse recovery in the underdetermined regime
L < P, and agreement of the sparse-coefficient Sobol indices with analytic values.

## Negative controls (Sachs and molecular design)

These two settings are reported as negative controls: the training-noise diagnostic shows the
policy ensembles are dominated by training-seed noise rather than reward uncertainty, and the
surrogate cannot predict the policy from the reward parameterisation, so **no reward-uncertainty
attribution is claimed for either**. To reproduce the abstention, train the main ensemble and a
control ensemble in which the reward realisation is fixed and only the training seed varies,
then run the decomposition:

```bash
# molecular design
python experiments/molecular_design/run_experiment.py --mode sequential
python decision_studies/controls/noise_control.py         # single fixed reward
python decision_studies/controls/noise_control_multi.py   # averaged over reward realisations

# Sachs
python experiments/sachs_causal/run_experiment.py
python decision_studies/controls/noise_control_sachs.py   # raw and ILR balance space
```

Each control script takes `--main_dir` and `--control_dir` (see `--help`) pointing at the two
ensembles; the Buchwald-Hartwig version of the same control is the optional `BH_NOISE_CONTROL=1`
stage of `lsf/submit_bh.sh`.

## Supplementary figures (S1-S8)

```bash
python figures/generate_figures.py
```

Writes the supplementary PDFs into `figures/`. This script reads the per-experiment result
directories under `results/`, so run the relevant experiment first; when a directory is absent
the panel falls back to placeholder data and says so on the console, so check the console output
before using a regenerated panel. The script also rebuilds a number of earlier-version panels
that the current manuscript does not cite.

| Fig | File                        | Produced from                                                      |
|-----|-----------------------------|---------------------------------------------------------------------|
| S1  | `fig_s1_discrete_grid.pdf`  | `results/gridworld/discrete/` (`experiments/gridworld/run_experiment.py --mode discrete`) |
| S2  | `fig_s2_continuous_grid.pdf`| `results/gridworld/continuous/` (`--mode continuous`)               |
| S3  | `fig_s3_sobol_all.pdf`      | `results/{gridworld/discrete,gridworld/continuous,symreg,llm_gfn}/results.json` |
| S4  | `fig_s4_calibration.pdf`    | coverage values recorded in the Buchwald-Hartwig and Sachs runs, embedded in the plotting script (no results file is read) |
| S5  | `fig_s5_pce_vs_mlp.pdf`     | `results/baselines/comparison.json` (`experiments/baselines/run_baselines.py`) |
| S6  | `fig_s6_pce_vs_gp.pdf`      | `results/baselines/comparison.json`                                 |
| S7  | `fig_s7_ablation.pdf`       | ablation reference curves embedded in the plotting script (no results file is read) |
| S8  | `fig_s8_controlled_llm.pdf` | `results/controlled_llm/results.json` (`experiments/controlled_llm/run_experiment.py`) |

Commands for the underlying experiments:

```bash
python experiments/gridworld/run_experiment.py --mode discrete
python experiments/gridworld/run_experiment.py --mode continuous
python experiments/symreg/run_experiment.py
python experiments/llm_gfn/run_experiment.py --mode train
python experiments/controlled_llm/run_experiment.py
python experiments/baselines/run_baselines.py
python experiments/gridworld/d_scalability.py     # d-scalability sweep
```

## Lean 4 formal verification

```bash
cd lean && lake build
```

`lean/PCESurrogate.lean` machine-checks **eight lemmas covering the seven numbered results
T1-T7** of the manuscript (T3 is split into T3a positivity and T3b normalisation), plus the
auxiliary lemma `simplex_l1_le_two` used by the T5 proof. **Every proof is `sorry`-free**, the
build completes with no errors, and `#print axioms` reports that each theorem depends only on
the three standard axioms `propext`, `Classical.choice`, `Quot.sound`. Supplementary Table S1
records, for each result, whether it is proved outright (T3a, T3b, T5, T6, T7) or conditional on
a stated hypothesis (T1 assumes the Sobolev decay rate; T2 assumes a shared polynomial basis,
l^2 coefficient convergence and positive total variance, and is for the first-order indices; T4
assumes a Lipschitz constant, supplied for the softmax link by T5).

The Lean version is pinned in `lean/lean-toolchain`. The project requires a Mathlib4 checkout as
a sibling directory of this repository, as declared by the `require` line of
`lean/lakefile.lean`; use a Mathlib revision matching the pinned toolchain.

## Random seeds and determinism

GFlowNet training uses PyTorch seeding; ensemble member l uses seed `base_seed + l`, with
`base_seed` set per experiment. The reward-model ensembles of the closed-loop, BO and RLHF
studies seed each member explicitly (`seed + m`) and the multi-seed studies report mean and
standard deviation over whole-study repetitions. PCE fitting is deterministic given the
ensemble outputs, and PCA is deterministic (SVD-based). Minor numerical variation across
hardware and BLAS implementations is expected and does not affect the qualitative conclusions.
Figures rebuilt from the committed JSONs are deterministic.

## Output layout

```
results/cluster_results/    committed result JSONs consumed by figures/make_figures.py
results/outputs/            analysis outputs (Source Data CSVs, re-analysis summaries)
results/<experiment>/       per-experiment training outputs (member_*.npz, results.json)
figures/                    all figure PDFs
```
