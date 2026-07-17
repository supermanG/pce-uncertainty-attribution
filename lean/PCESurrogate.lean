/-
  Formal verification of PCE-surrogate properties (Lean 4 + Mathlib).

  Seven properties underpin the surrogate analysis in the manuscript.  All seven
  are proved here with no `sorry`; each theorem below depends only on the
  standard axioms (propext, Classical.choice, Quot.sound):

    T1   the convergence-rate bound forces the truncation error to vanish
    T2   Sobol indices converge once the PCE coefficients converge in ℓ²
    T3a  softmax outputs are strictly positive
    T3b  softmax outputs sum to one
    T4   an L-Lipschitz reward-to-policy map maps an ε perturbation to one of
         size at most L·ε
    T5   the softmax ℓ∞ → ℓ¹ Lipschitz bound ‖σ(v) - σ(w)‖₁ ≤ 2‖v - w‖_∞
    T6   softmax is invariant to a uniform shift of the logits (so a uniform-shift
         uncertainty mode has zero policy effect and zero decision value-of-information)
    T7   the decision value E_p[r] is 1-Lipschitz (times ‖r‖_∞) in the ℓ¹ policy
         distance, so fragility bounds the decision value-variability
-/

import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus


open BigOperators Real Finset


-- -------------------------------------------------------------------------
-- Structures
-- -------------------------------------------------------------------------

/-- A polynomial chaos expansion over `d` input dimensions.
    The `support` is the finite set of multi-indices with nonzero coefficients. -/
structure PCExpansion (d : ℕ) where
  degree : ℕ
  support : Finset (Fin d → ℕ)
  coefficients : (Fin d → ℕ) → ℝ
  support_covers : ∀ j, j ∉ support → coefficients j = 0

/-- Softmax of a real-valued input vector. -/
noncomputable def softmax' {n : ℕ} (v : Fin n → ℝ) (k : Fin n) : ℝ :=
  exp (v k) / ∑ j, exp (v j)

/-- First-order Sobol sensitivity index for PCE dimension `i`.
    `D`  = total variance = Σ_{j ∈ support, |j|>0} c_j²
    `Di` = variance from dimension `i` alone = Σ_{j: j_i>0, j_{-i}=0} c_j² -/
noncomputable def sobol_first_order {d : ℕ} (pce : PCExpansion d) (i : Fin d) : ℝ :=
  let D  := ∑ j ∈ pce.support, if (∑ k, j k) > 0 then pce.coefficients j ^ 2 else 0
  let Di := ∑ j ∈ pce.support, if j i > 0 ∧ (∀ k, k ≠ i → j k = 0)
            then pce.coefficients j ^ 2 else 0
  if D > 0 then Di / D else 0


-- -------------------------------------------------------------------------
-- Theorem 1: convergence from the PCE rate bound
-- -------------------------------------------------------------------------
-- If `g ∈ H^s`, the degree-`p` PCE satisfies ‖g - g_p‖_{L²} ≤ C·p^{-s}; the
-- constant `C` and exponent `s` come from the Sobolev embedding (Supplementary
-- Information).  We take that bound as the hypothesis `h_rate` and verify its
-- operative consequence: a nonnegative error dominated by C·p^{-s} (s ≥ 1)
-- converges to zero, by squeezing between `0` and the rate.

theorem pce_error_tendsto_zero
    (err : ℕ → ℝ) (C : ℝ) (s : ℕ) (hs : 1 ≤ s)
    (h_nonneg : ∀ p, 0 ≤ err p)
    (h_rate : ∀ p, err p ≤ C * (p : ℝ)⁻¹ ^ s) :
    Filter.Tendsto err Filter.atTop (nhds 0) := by
  have hinv : Filter.Tendsto (fun p : ℕ => (p : ℝ)⁻¹) Filter.atTop (nhds 0) :=
    tendsto_inv_atTop_zero.comp tendsto_natCast_atTop_atTop
  have hrate : Filter.Tendsto (fun p : ℕ => C * (p : ℝ)⁻¹ ^ s) Filter.atTop (nhds 0) := by
    have hpow : Filter.Tendsto (fun p : ℕ => (p : ℝ)⁻¹ ^ s) Filter.atTop (nhds 0) := by
      simpa [zero_pow (show s ≠ 0 by omega)] using hinv.pow s
    simpa using hpow.const_mul C
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hrate h_nonneg h_rate


-- -------------------------------------------------------------------------
-- Theorem 2: Sobol index convergence from ℓ² coefficient convergence
-- -------------------------------------------------------------------------
-- The approximants `pce_seq p` and the limit share the multi-index basis `S`
-- (hypothesis `h_supp_seq`); without that constraint the claim is false, since an
-- approximant could place arbitrary mass on indices outside `S` while matching
-- `c_true` on `S`.  Given it, `D` and `Di` are continuous in the coefficient
-- vector and `Si = Di / D` is continuous away from `D = 0`.
--
-- Proof sketch.  With u = c_p - c and the indicator restricting the sum,
--   |D_p - D| = |∑_j (c_p(j)² - c(j)²)| ≤ ∑_j |c_p(j) - c(j)|·|c_p(j) + c(j)|
--             ≤ ‖c_p - c‖₂ · ‖c_p + c‖₂                         (Cauchy-Schwarz)
-- and likewise for `Di`.  The first factor tends to 0 by `h_l2`, the second is
-- bounded, so `D_p → D > 0`, eventually `D_p > D/2`, and
--   |Di_p/D_p - Di/D| ≤ |Di_p - Di|/D_p + Di·|D_p - D|/(D·D_p) → 0.
-- The open step is Cauchy-Schwarz under `Finset.sum` (`inner_mul_le_norm_mul_norm`)
-- together with the pointwise identity |a² - b²| = |a - b|·|a + b|.

-- Moving a coefficient-weighted sum from an approximant's own support onto the
-- shared basis `S`: terms outside `S` vanish by `h_supp_pce`, terms in `S` but
-- outside the support vanish by `support_covers`, so both equal the sum over the
-- intersection.
private lemma sum_support_eq_sum_basis {d : ℕ}
    (pce : PCExpansion d) (S : Finset (Fin d → ℕ))
    (h_supp_pce : ∀ j ∉ S, pce.coefficients j = 0)
    (g : (Fin d → ℕ) → ℝ) (hg : ∀ j, pce.coefficients j = 0 → g j = 0) :
    (∑ j ∈ pce.support, g j) = ∑ j ∈ S, g j := by
  have h1 : ∑ j ∈ pce.support ∩ S, g j = ∑ j ∈ pce.support, g j := by
    refine Finset.sum_subset Finset.inter_subset_left (fun x hx hxni => ?_)
    exact hg x (h_supp_pce x (fun hxs => hxni (Finset.mem_inter.mpr ⟨hx, hxs⟩)))
  have h2 : ∑ j ∈ pce.support ∩ S, g j = ∑ j ∈ S, g j := by
    refine Finset.sum_subset Finset.inter_subset_right (fun x hx hxni => ?_)
    exact hg x (pce.support_covers x (fun hxsupp => hxni (Finset.mem_inter.mpr ⟨hxsupp, hx⟩)))
  rw [← h1, h2]

-- An indicator-restricted sum of squared coefficients is continuous in the
-- coefficient vector: coordinatewise convergence on `S` gives convergence of the
-- finite sum.
private lemma indic_sum_tendsto {d : ℕ}
    (pce_seq : ℕ → PCExpansion d) (S : Finset (Fin d → ℕ)) (c_true : (Fin d → ℕ) → ℝ)
    (hcoef : ∀ j ∈ S,
      Filter.Tendsto (fun p => (pce_seq p).coefficients j) Filter.atTop (nhds (c_true j)))
    (cond : (Fin d → ℕ) → Prop) [DecidablePred cond] :
    Filter.Tendsto (fun p => ∑ j ∈ S, if cond j then (pce_seq p).coefficients j ^ 2 else 0)
      Filter.atTop (nhds (∑ j ∈ S, if cond j then c_true j ^ 2 else 0)) := by
  refine tendsto_finset_sum _ (fun j hj => ?_)
  by_cases hc : cond j
  · simp only [if_pos hc]; exact (hcoef j hj).pow 2
  · simp only [if_neg hc]; exact tendsto_const_nhds

theorem sobol_convergence_from_L2
    {d : ℕ}
    (pce_seq : ℕ → PCExpansion d)         -- sequence of PCE approximations
    (S : Finset (Fin d → ℕ))              -- common finite index set (basis)
    (c_true  : (Fin d → ℕ) → ℝ)           -- true PCE coefficients
    (h_supp_seq : ∀ p, ∀ j ∉ S, (pce_seq p).coefficients j = 0)
                                          -- approximants live on the basis `S`
    (h_l2    : ∀ ε > 0, ∃ N, ∀ p ≥ N,    -- ℓ² coefficient convergence (from L²)
               ∑ j ∈ S, ((pce_seq p).coefficients j - c_true j) ^ 2 < ε)
    (i       : Fin d)
    (h_supp  : ∀ j, j ∉ S → c_true j = 0) -- limit coefficients vanish outside `S`
    (hD      : (∑ j ∈ S, if (∑ k, j k) > 0 then c_true j ^ 2 else 0) > 0) :
    ∀ ε > 0, ∃ N, ∀ p ≥ N,
      |sobol_first_order (pce_seq p) i -
       sobol_first_order ⟨0, S, c_true, h_supp⟩ i| < ε := by
  intro ε hε
  -- ℓ² coefficient error tends to 0
  have hsum0 : Filter.Tendsto
      (fun p => ∑ j ∈ S, ((pce_seq p).coefficients j - c_true j) ^ 2)
      Filter.atTop (nhds 0) := by
    rw [Metric.tendsto_atTop]
    intro δ hδ
    obtain ⟨N, hN⟩ := h_l2 δ hδ
    refine ⟨N, fun p hp => ?_⟩
    have hnn : 0 ≤ ∑ j ∈ S, ((pce_seq p).coefficients j - c_true j) ^ 2 :=
      Finset.sum_nonneg (fun _ _ => sq_nonneg _)
    rw [Real.dist_eq, sub_zero, abs_of_nonneg hnn]
    exact hN p hp
  -- coordinatewise convergence on the basis `S`
  have hcoef : ∀ j ∈ S, Filter.Tendsto
      (fun p => (pce_seq p).coefficients j) Filter.atTop (nhds (c_true j)) := by
    intro j hj
    have hterm : Filter.Tendsto
        (fun p => ((pce_seq p).coefficients j - c_true j) ^ 2) Filter.atTop (nhds 0) := by
      refine squeeze_zero (fun _ => sq_nonneg _) (fun p => ?_) hsum0
      exact Finset.single_le_sum
        (f := fun j => ((pce_seq p).coefficients j - c_true j) ^ 2)
        (fun _ _ => sq_nonneg _) hj
    have habs : Filter.Tendsto
        (fun p => |(pce_seq p).coefficients j - c_true j|) Filter.atTop (nhds 0) := by
      have h := hterm.sqrt
      rw [Real.sqrt_zero] at h
      simpa [Real.sqrt_sq_eq_abs] using h
    have hdiff : Filter.Tendsto
        (fun p => (pce_seq p).coefficients j - c_true j) Filter.atTop (nhds 0) := by
      rw [tendsto_zero_iff_norm_tendsto_zero]
      simpa [Real.norm_eq_abs] using habs
    have heq : (fun p => (pce_seq p).coefficients j)
             = (fun p => ((pce_seq p).coefficients j - c_true j) + c_true j) := by
      funext p; ring
    rw [heq]
    simpa using hdiff.add_const (c_true j)
  -- the two variance sums converge to their true values
  have hDtend := indic_sum_tendsto pce_seq S c_true hcoef (fun j => (∑ k, j k) > 0)
  have hDitend := indic_sum_tendsto pce_seq S c_true hcoef
    (fun j => j i > 0 ∧ ∀ k, k ≠ i → j k = 0)
  -- the Sobol index of `pce_seq p`, rewritten onto the basis `S`
  have hsob_eq : ∀ p, sobol_first_order (pce_seq p) i =
      if 0 < ∑ j ∈ S, if (∑ k, j k) > 0 then (pce_seq p).coefficients j ^ 2 else 0
      then (∑ j ∈ S, if (j i > 0 ∧ ∀ k, k ≠ i → j k = 0)
            then (pce_seq p).coefficients j ^ 2 else 0)
         / (∑ j ∈ S, if (∑ k, j k) > 0 then (pce_seq p).coefficients j ^ 2 else 0)
      else 0 := by
    intro p
    have e1 := sum_support_eq_sum_basis (pce_seq p) S (h_supp_seq p)
      (fun j => if (∑ k, j k) > 0 then (pce_seq p).coefficients j ^ 2 else 0)
      (fun j hj => by simp [hj])
    have e2 := sum_support_eq_sum_basis (pce_seq p) S (h_supp_seq p)
      (fun j => if (j i > 0 ∧ ∀ k, k ≠ i → j k = 0)
                then (pce_seq p).coefficients j ^ 2 else 0)
      (fun j hj => by simp [hj])
    simp only [sobol_first_order, e1, e2]
  -- the limiting Sobol index (denominator positive by `hD`)
  have hsob_lim : sobol_first_order (⟨0, S, c_true, h_supp⟩ : PCExpansion d) i =
      (∑ j ∈ S, if (j i > 0 ∧ ∀ k, k ≠ i → j k = 0) then c_true j ^ 2 else 0)
         / (∑ j ∈ S, if (∑ k, j k) > 0 then c_true j ^ 2 else 0) := by
    simp only [sobol_first_order]
    rw [if_pos hD]
  -- the denominator is eventually positive, so the index equals the ratio there
  have hev := hDtend.eventually (eventually_gt_nhds hD)
  have hratio := hDitend.div hDtend (ne_of_gt hD)
  have main : Filter.Tendsto (fun p => sobol_first_order (pce_seq p) i) Filter.atTop
      (nhds ((∑ j ∈ S, if (j i > 0 ∧ ∀ k, k ≠ i → j k = 0) then c_true j ^ 2 else 0)
           / (∑ j ∈ S, if (∑ k, j k) > 0 then c_true j ^ 2 else 0))) := by
    refine hratio.congr' ?_
    filter_upwards [hev] with p hp
    simp only [Pi.div_apply]
    rw [hsob_eq p, if_pos hp]
  rw [hsob_lim]
  obtain ⟨N, hN⟩ := (Metric.tendsto_atTop.mp main) ε hε
  refine ⟨N, fun p hp => ?_⟩
  have hd := hN p hp
  rwa [Real.dist_eq] at hd


-- -------------------------------------------------------------------------
-- Theorem 3a: softmax strict positivity
-- -------------------------------------------------------------------------

theorem softmax_positive {n : ℕ} (hn : 0 < n) (v : Fin n → ℝ) (k : Fin n) :
    0 < softmax' v k := by
  unfold softmax'
  apply div_pos (exp_pos _)
  apply Finset.sum_pos (fun j _ => exp_pos (v j))
  exact ⟨⟨0, hn⟩, mem_univ _⟩


-- -------------------------------------------------------------------------
-- Theorem 3b: softmax sums to one
-- -------------------------------------------------------------------------

theorem softmax_sums_to_one {n : ℕ} (hn : 0 < n) (v : Fin n → ℝ) :
    ∑ k : Fin n, softmax' v k = 1 := by
  have hD : ∑ j : Fin n, exp (v j) ≠ 0 :=
    ne_of_gt (Finset.sum_pos (fun j _ => exp_pos (v j)) ⟨⟨0, hn⟩, mem_univ _⟩)
  simp only [softmax', ← Finset.sum_div, div_self hD]


-- -------------------------------------------------------------------------
-- Theorem 4: Lipschitz uncertainty propagation
-- -------------------------------------------------------------------------
-- If the reward-to-policy map `f` is `L`-Lipschitz and the reward
-- parameterisation is perturbed by at most `ε`, the induced perturbation in `f`
-- is at most `L·ε`.

theorem uncertainty_propagation
    {f : ℝ → ℝ} {L ε : ℝ} (hL : 0 ≤ L)
    (h_lip : ∀ a b, |f a - f b| ≤ L * |a - b|)
    {x y : ℝ} (h_pert : |x - y| ≤ ε) :
    |f x - f y| ≤ L * ε :=
  calc |f x - f y| ≤ L * |x - y| := h_lip x y
    _ ≤ L * ε := mul_le_mul_of_nonneg_left h_pert hL


-- -------------------------------------------------------------------------
-- Theorem 5: softmax ℓ∞ → ℓ¹ Lipschitz continuity
-- -------------------------------------------------------------------------
-- Auxiliary: |a - b| ≤ a + b when 0 ≤ a and 0 ≤ b.
private lemma abs_sub_le_add {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    |a - b| ≤ a + b := by
  rw [abs_le]
  constructor <;> linarith

-- Auxiliary: any two simplex points are within ℓ¹ distance 2.
lemma simplex_l1_le_two {n : ℕ} (hn : 0 < n) (v w : Fin n → ℝ) :
    ∑ k : Fin n, |softmax' v k - softmax' w k| ≤ 2 := by
  have hv := softmax_sums_to_one hn v
  have hw := softmax_sums_to_one hn w
  have hpos_v : ∀ k, 0 ≤ softmax' v k := fun k => le_of_lt (softmax_positive hn v k)
  have hpos_w : ∀ k, 0 ≤ softmax' w k := fun k => le_of_lt (softmax_positive hn w k)
  calc ∑ k : Fin n, |softmax' v k - softmax' w k|
      ≤ ∑ k : Fin n, (softmax' v k + softmax' w k) := by
          apply Finset.sum_le_sum; intro k _
          exact abs_sub_le_add (hpos_v k) (hpos_w k)
    _ = (∑ k : Fin n, softmax' v k) + (∑ k : Fin n, softmax' w k) :=
          Finset.sum_add_distrib
    _ = 1 + 1 := by rw [hv, hw]
    _ = 2 := by norm_num

-- Softmax is Lipschitz from ℓ∞ to ℓ¹ with constant 2:
--   ∑_k |σ_k(v) - σ_k(w)| ≤ 2 ‖v - w‖_∞.
-- For ‖v - w‖_∞ ≥ 1 the simplex-diameter bound gives 2 ≤ 2‖v - w‖_∞ directly.
-- For ‖v - w‖_∞ < 1 we take φ(t) = softmax(v + t(w - v)) along the segment.  Each
-- coordinate is differentiable with dφ_k/dt = σ_k(ξ)(u_k - ⟨σ(ξ), u⟩), u = w - v,
-- so ∑_k |dφ_k/dt| ≤ 2 ∑_k σ_k(ξ)|u_k| ≤ 2‖u‖_∞ pointwise; the fundamental
-- theorem of calculus and the swap of sum and integral then give the bound.

theorem softmax_lipschitz {n : ℕ} (hn : 0 < n) (v w : Fin n → ℝ) :
    ∑ k : Fin n, |softmax' v k - softmax' w k| ≤
    2 * ⨆ i : Fin n, |v i - w i| := by
  by_cases h : 1 ≤ ⨆ i : Fin n, |v i - w i|
  · -- ‖v - w‖_∞ ≥ 1: the simplex diameter ≤ 2 ≤ 2‖v - w‖_∞.
    calc ∑ k : Fin n, |softmax' v k - softmax' w k|
        ≤ 2 := simplex_l1_le_two hn v w
      _ = 2 * 1 := (mul_one 2).symm
      _ ≤ 2 * ⨆ i : Fin n, |v i - w i| := by linarith
  · -- ‖v - w‖_∞ < 1: mean-value argument along the segment v + t(w - v).  The
    -- proof gives the constant 2 (a valid, non-tight bound; the sharp softmax
    -- l-inf-to-l1 Lipschitz constant is 1) for every δ; the case split only routes the
    -- large-perturbation regime through the simpler simplex bound above.
    set δ := ⨆ i : Fin n, |v i - w i| with hδ
    set L : ℝ → Fin n → ℝ := fun t i => v i + t * (w i - v i) with hL
    set F : Fin n → ℝ → ℝ :=
      fun k t => softmax' (L t) k * ((w k - v k) - ∑ j, softmax' (L t) j * (w j - v j))
      with hF
    -- endpoints of the path
    have hL0 : softmax' (L 0) = softmax' v := by
      have h0 : L 0 = v := by funext i; simp [hL]
      rw [h0]
    have hL1 : softmax' (L 1) = softmax' w := by
      have h1 : L 1 = w := by funext i; simp [hL]
      rw [h1]
    -- every coordinate gap is bounded by δ
    have hbdd : BddAbove (Set.range fun i => |v i - w i|) := Finite.bddAbove_range _
    have hδbound : ∀ k, |w k - v k| ≤ δ := by
      intro k; rw [hδ, abs_sub_comm]; exact le_ciSup hbdd k
    -- the interpolated normaliser is positive
    have hZpos : ∀ t, 0 < ∑ j, Real.exp (L t j) :=
      fun t => Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨⟨0, hn⟩, Finset.mem_univ _⟩
    -- derivative of each interpolated coordinate input
    have hLderiv : ∀ (t : ℝ) (k : Fin n), HasDerivAt (fun s => L s k) (w k - v k) t := by
      intro t k
      have h0 : HasDerivAt (fun s => v k + s * (w k - v k)) (w k - v k) t := by
        simpa using ((hasDerivAt_id t).mul_const (w k - v k)).const_add (v k)
      simpa only [hL] using h0
    -- derivative of each softmax coordinate along the path
    have hderiv : ∀ (t : ℝ) (k : Fin n),
        HasDerivAt (fun s => softmax' (L s) k) (F k t) t := by
      intro t k
      have hnum : HasDerivAt (fun s => Real.exp (L s k))
          (Real.exp (L t k) * (w k - v k)) t := (hLderiv t k).exp
      have hden : HasDerivAt (fun s => ∑ j, Real.exp (L s j))
          (∑ j, Real.exp (L t j) * (w j - v j)) t :=
        HasDerivAt.fun_sum (fun j _ => (hLderiv t j).exp)
      have hZ : (∑ j, Real.exp (L t j)) ≠ 0 := ne_of_gt (hZpos t)
      have hdiv : HasDerivAt (fun s => softmax' (L s) k)
          ((Real.exp (L t k) * (w k - v k) * (∑ j, Real.exp (L t j))
            - Real.exp (L t k) * (∑ j, Real.exp (L t j) * (w j - v j)))
            / (∑ j, Real.exp (L t j)) ^ 2) t := hnum.div hden hZ
      have hFeq : F k t =
          (Real.exp (L t k) * (w k - v k) * (∑ j, Real.exp (L t j))
            - Real.exp (L t k) * (∑ j, Real.exp (L t j) * (w j - v j)))
            / (∑ j, Real.exp (L t j)) ^ 2 := by
        simp only [hF]
        have hsum_eq : (∑ j, softmax' (L t) j * (w j - v j))
            = (∑ j, Real.exp (L t j) * (w j - v j)) / (∑ i, Real.exp (L t i)) := by
          rw [Finset.sum_div]
          refine Finset.sum_congr rfl (fun j _ => ?_)
          simp only [softmax']; ring
        rw [hsum_eq]; simp only [softmax']; field_simp
      rw [hFeq]; exact hdiv
    -- continuity of each derivative coordinate (for integrability)
    have hcont : ∀ k, Continuous (F k) := by
      intro k
      have hsmc : ∀ m, Continuous (fun t => softmax' (L t) m) := by
        intro m
        simp only [softmax', hL]
        refine Continuous.div (Real.continuous_exp.comp (by fun_prop))
          (continuous_finset_sum _ (fun j _ => Real.continuous_exp.comp (by fun_prop)))
          (fun t => ?_)
        exact ne_of_gt (Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨⟨0, hn⟩, Finset.mem_univ _⟩)
      simp only [hF]
      exact (hsmc k).mul
        (continuous_const.sub (continuous_finset_sum _ (fun j _ => (hsmc j).mul continuous_const)))
    -- pointwise ℓ¹ bound on the derivative: ∑_k |F k t| ≤ 2 δ
    have hbound : ∀ t, ∑ k, |F k t| ≤ 2 * δ := by
      intro t
      simp only [hF]
      have hpos : ∀ k, 0 ≤ softmax' (L t) k := fun k => le_of_lt (softmax_positive hn (L t) k)
      have hs1 : ∑ k, softmax' (L t) k = 1 := softmax_sums_to_one hn (L t)
      have hinner : |∑ j, softmax' (L t) j * (w j - v j)| ≤ δ := by
        calc |∑ j, softmax' (L t) j * (w j - v j)|
            ≤ ∑ j, |softmax' (L t) j * (w j - v j)| := abs_sum_le_sum_abs _ _
          _ = ∑ j, softmax' (L t) j * |w j - v j| := by
              refine Finset.sum_congr rfl (fun j _ => ?_)
              rw [abs_mul, abs_of_nonneg (hpos j)]
          _ ≤ ∑ j, softmax' (L t) j * δ :=
              Finset.sum_le_sum (fun j _ => mul_le_mul_of_nonneg_left (hδbound j) (hpos j))
          _ = δ := by rw [← Finset.sum_mul, hs1, one_mul]
      calc ∑ k, |softmax' (L t) k * ((w k - v k) - ∑ j, softmax' (L t) j * (w j - v j))|
          = ∑ k, softmax' (L t) k *
              |(w k - v k) - ∑ j, softmax' (L t) j * (w j - v j)| := by
            refine Finset.sum_congr rfl (fun k _ => ?_)
            rw [abs_mul, abs_of_nonneg (hpos k)]
        _ ≤ ∑ k, softmax' (L t) k * (2 * δ) := by
            refine Finset.sum_le_sum (fun k _ => mul_le_mul_of_nonneg_left ?_ (hpos k))
            calc |(w k - v k) - ∑ j, softmax' (L t) j * (w j - v j)|
                ≤ |w k - v k| + |∑ j, softmax' (L t) j * (w j - v j)| := by
                  rw [sub_eq_add_neg]; refine (abs_add_le _ _).trans ?_; rw [abs_neg]
              _ ≤ δ + δ := add_le_add (hδbound k) hinner
              _ = 2 * δ := by ring
        _ = 2 * δ := by rw [← Finset.sum_mul, hs1, one_mul]
    -- assemble via the fundamental theorem of calculus
    calc ∑ k : Fin n, |softmax' v k - softmax' w k|
        = ∑ k : Fin n, |softmax' (L 0) k - softmax' (L 1) k| := by rw [hL0, hL1]
      _ = ∑ k : Fin n, |∫ t in (0:ℝ)..1, F k t| := by
          refine Finset.sum_congr rfl (fun k _ => ?_)
          have hftc : ∫ t in (0:ℝ)..1, F k t = softmax' (L 1) k - softmax' (L 0) k :=
            intervalIntegral.integral_eq_sub_of_hasDerivAt (fun t _ => hderiv t k)
              ((hcont k).intervalIntegrable 0 1)
          rw [hftc, abs_sub_comm]
      _ ≤ ∑ k : Fin n, ∫ t in (0:ℝ)..1, |F k t| :=
          Finset.sum_le_sum (fun k _ =>
            intervalIntegral.abs_integral_le_integral_abs (by norm_num))
      _ = ∫ t in (0:ℝ)..1, ∑ k : Fin n, |F k t| :=
          (intervalIntegral.integral_finset_sum (fun k _ => (hcont k).abs.intervalIntegrable 0 1)).symm
      _ ≤ ∫ _t in (0:ℝ)..1, 2 * δ := by
          refine intervalIntegral.integral_mono_on (by norm_num) ?_
            (continuous_const.intervalIntegrable 0 1) (fun t _ => hbound t)
          exact (continuous_finset_sum _ (fun k _ => (hcont k).abs)).intervalIntegrable 0 1
      _ = 2 * δ := by rw [intervalIntegral.integral_const]; simp


-- -------------------------------------------------------------------------
-- Theorem 6: softmax shift-invariance
-- -------------------------------------------------------------------------
-- Adding the same constant to every logit leaves the softmax unchanged.  A
-- reward-model uncertainty mode that shifts all action scores equally therefore
-- has NO effect on the policy, hence zero decision value-of-information -- even
-- when it carries most of the reward-model's uncertainty variance (the RLHF
-- top-variance mode).  This turns the empirical observation into a theorem.

theorem softmax_shift_invariant {n : ℕ} (v : Fin n → ℝ) (c : ℝ) (k : Fin n) :
    softmax' (fun j => v j + c) k = softmax' v k := by
  have hc : Real.exp c ≠ 0 := ne_of_gt (Real.exp_pos c)
  have hden : (∑ j, Real.exp (v j)) ≠ 0 :=
    ne_of_gt (Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨k, Finset.mem_univ k⟩)
  unfold softmax'
  simp only [Real.exp_add]
  rw [← Finset.sum_mul]
  field_simp


-- -------------------------------------------------------------------------
-- Theorem 7: decision value-variability bound
-- -------------------------------------------------------------------------
-- For two policies p, q over n actions and a bounded reward r (|r k| ≤ R), the
-- decision VALUES E_p[r], E_q[r] differ by at most R times the l1 policy
-- distance, i.e. 2R times the total-variation (the fragility metric).  Composed
-- with the softmax Lipschitz bound (T5) this certifies that a small logit
-- perturbation yields a small change in the true decision value.

theorem value_diff_le_l1 {n : ℕ} (p q r : Fin n → ℝ) (R : ℝ)
    (hr : ∀ k, |r k| ≤ R) :
    |∑ k, (p k - q k) * r k| ≤ R * ∑ k, |p k - q k| := by
  calc |∑ k, (p k - q k) * r k|
      ≤ ∑ k, |(p k - q k) * r k| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ k, |p k - q k| * |r k| := by
        refine Finset.sum_congr rfl (fun k _ => ?_); rw [abs_mul]
    _ ≤ ∑ k, |p k - q k| * R := by
        refine Finset.sum_le_sum (fun k _ => ?_)
        exact mul_le_mul_of_nonneg_left (hr k) (abs_nonneg _)
    _ = (∑ k, |p k - q k|) * R := by rw [← Finset.sum_mul]
    _ = R * ∑ k, |p k - q k| := by rw [mul_comm]
