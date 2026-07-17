#!/usr/bin/env python3
"""Phase 2: Run 4-way embedding comparison from cached proxy/GFlowNet data."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from core.pce_surrogate import TrajectoryPCESurrogate, calibration_coverage
from experiments.validation.pca_vs_vae_ablation import (
    train_beta_vae, train_normalizing_flow, flow_transform,
    check_independence, check_gaussianity,
)
import torch

ROOT = Path(__file__).resolve().parents[2]
n_train, n_test = 30, 50
pca_dim, pce_degree, beta = 5, 3, 4.0

# Load cached data
cache = np.load(ROOT / 'results' / 'validation' / '_ablation_cache.npz')
pout = cache['pout']
all_policies = {s: cache[f'pol_{s}'] for s in range(4)}
tr_pol = {s: all_policies[s][:n_train] for s in range(4)}
te_pol = {s: all_policies[s][n_train:] for s in range(4)}
print(f"Loaded cache: pout={pout.shape}, pol_0={all_policies[0].shape}")


def mean_cal95(cal_dict):
    vals = []
    for s in range(4):
        c = cal_dict[s]
        vals.append(c.get(0.95, c.get("0.95", 0)))
    return round(float(np.mean(vals)), 3)


def mean_sobol_sum(sobol_dict):
    sums = []
    for s in range(4):
        to = sobol_dict[s]["total_order"]
        sums.append(float(to.sum(axis=1).mean()))
    return round(float(np.mean(sums)), 3)


results_all = {}

# ===== 1. Linear PCA =====
print("\n" + "=" * 60)
print("EMBEDDING 1: Linear PCA")
pca = PCA(n_components=pca_dim)
mu_tr = pca.fit_transform(pout[:n_train])
mu_te = pca.transform(pout[n_train:])
sc = StandardScaler()
mu_tr = sc.fit_transform(mu_tr)
mu_te = sc.transform(mu_te)
pca_var = float(pca.explained_variance_ratio_.sum())
pca_indep = check_independence(mu_tr)
pca_gauss = check_gaussianity(mu_tr)
print(f"  Var explained: {pca_var:.3f}")
print(f"  Max|rho|: {pca_indep['max_abs_pearson']:.4f} ({'PASS' if pca_indep['independent'] else 'FAIL'})")
print(f"  Gaussian: {pca_gauss['all_gaussian']}")

tsurr = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
for s in range(4):
    tsurr.fit_step(s, mu_tr, tr_pol[s])
surr_samp = tsurr.sample_trajectory_policies(5000)
cal = {s: calibration_coverage(surr_samp[s], te_pol[s]) for s in range(4)}
sobol = tsurr.sobol_all_steps()
print(f"  Cal@95: {mean_cal95(cal)}, Sum(S_T): {mean_sobol_sum(sobol)}")
results_all["linear_pca"] = dict(var=pca_var, rho=pca_indep["max_abs_pearson"],
    gaussian=pca_gauss["all_gaussian"], independent=pca_indep["independent"],
    cal95=mean_cal95(cal), sobol_sum=mean_sobol_sum(sobol))

# ===== 2. Kernel PCA =====
print("\n" + "=" * 60)
print("EMBEDDING 2: Kernel PCA (RBF)")
kpca = KernelPCA(n_components=pca_dim, kernel="rbf", gamma=None,
                 fit_inverse_transform=True, random_state=0)
mu_tr_k = kpca.fit_transform(pout[:n_train])
mu_te_k = kpca.transform(pout[n_train:])
sc_k = StandardScaler()
mu_tr_k = sc_k.fit_transform(mu_tr_k)
mu_te_k = sc_k.transform(mu_te_k)
kpca_indep = check_independence(mu_tr_k)
kpca_gauss = check_gaussianity(mu_tr_k)
print(f"  Max|rho|: {kpca_indep['max_abs_pearson']:.4f} ({'PASS' if kpca_indep['independent'] else 'FAIL'})")
print(f"  Gaussian: {kpca_gauss['all_gaussian']}")

tsurr_k = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
for s in range(4):
    tsurr_k.fit_step(s, mu_tr_k, tr_pol[s])
surr_samp_k = tsurr_k.sample_trajectory_policies(5000)
cal_k = {s: calibration_coverage(surr_samp_k[s], te_pol[s]) for s in range(4)}
sobol_k = tsurr_k.sobol_all_steps()
print(f"  Cal@95: {mean_cal95(cal_k)}, Sum(S_T): {mean_sobol_sum(sobol_k)}")
results_all["kernel_pca"] = dict(rho=kpca_indep["max_abs_pearson"],
    gaussian=kpca_gauss["all_gaussian"], independent=kpca_indep["independent"],
    cal95=mean_cal95(cal_k), sobol_sum=mean_sobol_sum(sobol_k))

# ===== 3. beta-VAE =====
print("\n" + "=" * 60)
print(f"EMBEDDING 3: beta-VAE (beta={beta})")
pout_sc = StandardScaler()
pout_std = pout_sc.fit_transform(pout)
mu_vae_tr, vae_model, vae_losses = train_beta_vae(
    pout_std[:n_train], latent_dim=pca_dim, beta=beta, n_epochs=3000, seed=0
)
vae_model.eval()
with torch.no_grad():
    mu_vae_te_raw, _ = vae_model.encode(
        torch.tensor(pout_std[n_train:], dtype=torch.float32)
    )
    mu_vae_te_raw = mu_vae_te_raw.numpy()

sc_v = StandardScaler()
mu_vae_tr = sc_v.fit_transform(mu_vae_tr)
mu_vae_te = sc_v.transform(mu_vae_te_raw)

vae_indep = check_independence(mu_vae_tr)
vae_gauss = check_gaussianity(mu_vae_tr)

# Reconstruction R2
vae_model.eval()
with torch.no_grad():
    z_t = torch.tensor(sc_v.inverse_transform(mu_vae_tr), dtype=torch.float32)
    recon = vae_model.decode(z_t).numpy()
ss_res = np.sum((pout_std[:n_train] - recon) ** 2)
ss_tot = np.sum((pout_std[:n_train] - pout_std[:n_train].mean(0)) ** 2)
vae_r2 = round(1.0 - ss_res / ss_tot, 4)

print(f"  R2: {vae_r2}")
print(f"  Max|rho|: {vae_indep['max_abs_pearson']:.4f} ({'PASS' if vae_indep['independent'] else 'FAIL'})")
print(f"  Gaussian: {vae_gauss['all_gaussian']}")

tsurr_v = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
for s in range(4):
    tsurr_v.fit_step(s, mu_vae_tr, tr_pol[s])
surr_samp_v = tsurr_v.sample_trajectory_policies(5000)
cal_v = {s: calibration_coverage(surr_samp_v[s], te_pol[s]) for s in range(4)}
sobol_v = tsurr_v.sobol_all_steps()
print(f"  Cal@95: {mean_cal95(cal_v)}, Sum(S_T): {mean_sobol_sum(sobol_v)}")
results_all["beta_vae"] = dict(r2=vae_r2, rho=vae_indep["max_abs_pearson"],
    gaussian=vae_gauss["all_gaussian"], independent=vae_indep["independent"],
    cal95=mean_cal95(cal_v), sobol_sum=mean_sobol_sum(sobol_v))

# ===== 4. PCA + Normalizing Flow =====
print("\n" + "=" * 60)
print("EMBEDDING 4: PCA + Normalizing Flow (RealNVP)")
pca_nf = PCA(n_components=pca_dim)
mu_pre_tr = pca_nf.fit_transform(pout[:n_train])
mu_pre_te = pca_nf.transform(pout[n_train:])
sc_pre = StandardScaler()
mu_pre_tr = sc_pre.fit_transform(mu_pre_tr)
mu_pre_te = sc_pre.transform(mu_pre_te)

mu_nf_tr, nf_model, nf_losses = train_normalizing_flow(
    mu_pre_tr, n_layers=8, hidden_dim=64, n_epochs=3000, seed=0
)
mu_nf_te = flow_transform(nf_model, mu_pre_te)

nf_var = float(pca_nf.explained_variance_ratio_.sum())
nf_indep = check_independence(mu_nf_tr)
nf_gauss = check_gaussianity(mu_nf_tr)
print(f"  PCA var: {nf_var:.3f}")
print(f"  Max|rho| after flow: {nf_indep['max_abs_pearson']:.4f} ({'PASS' if nf_indep['independent'] else 'FAIL'})")
print(f"  Gaussian after flow: {nf_gauss['all_gaussian']}")
for r in nf_gauss["per_dim"]:
    tag = "Gaussian" if r["gaussian"] else "NON-Gaussian"
    print(f"    Dim {r['dim']}: Shapiro p={r['p_value']:.4f} ({tag})")

tsurr_nf = TrajectoryPCESurrogate(degree=pce_degree, basis="hermite")
for s in range(4):
    tsurr_nf.fit_step(s, mu_nf_tr, tr_pol[s])
surr_samp_nf = tsurr_nf.sample_trajectory_policies(5000)
cal_nf = {s: calibration_coverage(surr_samp_nf[s], te_pol[s]) for s in range(4)}
sobol_nf = tsurr_nf.sobol_all_steps()
print(f"  Cal@95: {mean_cal95(cal_nf)}, Sum(S_T): {mean_sobol_sum(sobol_nf)}")
results_all["pca_plus_flow"] = dict(var=nf_var, rho=nf_indep["max_abs_pearson"],
    gaussian=nf_gauss["all_gaussian"], independent=nf_indep["independent"],
    cal95=mean_cal95(cal_nf), sobol_sum=mean_sobol_sum(sobol_nf),
    gaussianity_per_dim=[r for r in nf_gauss["per_dim"]])

# ===== Summary =====
print("\n" + "=" * 60)
print("ABLATION SUMMARY: PCA vs Kernel PCA vs beta-VAE vs PCA+Flow")
print("=" * 60)
hdr = f"{'Method':18s} {'Var/R2':>8s} {'Max|rho|':>8s} {'Gauss':>7s} {'Indep':>7s} {'Cal@95':>8s} {'SumST':>8s}"
print(hdr)
print("-" * len(hdr))
for name, r in [("Linear PCA", results_all["linear_pca"]),
                ("Kernel PCA", results_all["kernel_pca"]),
                ("beta-VAE", results_all["beta_vae"]),
                ("PCA+Flow", results_all["pca_plus_flow"])]:
    v = r.get("var", r.get("r2", 0))
    g = "YES" if r["gaussian"] else "NO"
    i = "YES" if r["independent"] else "NO"
    print(f"{name:18s} {v:8.3f} {r['rho']:8.4f} {g:>7s} {i:>7s} {r['cal95']:8.3f} {r['sobol_sum']:8.3f}")

print("\nSOBOL VALIDITY:")
for name, r in [("Linear PCA", results_all["linear_pca"]),
                ("Kernel PCA", results_all["kernel_pca"]),
                ("beta-VAE", results_all["beta_vae"]),
                ("PCA+Flow", results_all["pca_plus_flow"])]:
    if r["independent"] and r["gaussian"]:
        v = "EXACT -- independent + Gaussian"
    elif r["independent"]:
        v = "APPROX -- independent, non-Gaussian (consider aPC)"
    elif r["gaussian"]:
        v = "CAUTION -- Gaussian but correlated"
    else:
        v = "INVALID -- correlated + non-Gaussian"
    print(f"  {name:18s}: {v}")

out = ROOT / "results" / "validation" / "pca_vs_vae_ablation.json"
with open(out, "w") as f:
    json.dump(results_all, f, indent=2, default=str)
print(f"\nSaved to {out}")
